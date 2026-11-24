"""Test 6: Interface Binding Alanine Scan Ranking (`Chain A + Partner B -> Rank X->Ala Binding Hotspots`).

Evaluates whether frontier LLMs (`gpt-6.2-astra`, `opus-5-5`, `gpt-6.2`, `gemini-3.1-pro`)
can **rank** a balanced `15–20` residue candidate panel of `X -> Ala` mutations on Protein A from
**#1 (most critical binding hotspot — largest loss of binding affinity to Partner B)** down to
**#K (most neutral non-interface site)** directly from the two anonymized sequences.

Includes matched **same-amino-acid neutral decoys** (e.g., interface `Tyr` vs. non-interface `Tyr`),
so static 1D lookup tables (`Bogan & Thorn` `Trp/Tyr/Arg`) score **exact `0.000`** on
`spearman_rho_aa_demeaned` and **exact `0.500`** (coin-flip) on `same_aa_win_rate`!
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import re
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
from scipy.stats import pearsonr, spearmanr

from protein_without_tools.alanine_scan import (
    binary_auroc,
    compute_aa_demeaned_spearman,
    compute_ndcg_at_k,
    compute_same_aa_win_rate,
    compute_topk_hotspot_precision,
    parse_ranking_items_to_scores,
)
from protein_without_tools.pdb_utils import CANONICAL_AA_SET

BOGAN_THORN_BINDING_HOTSPOT_SCALE: Dict[str, float] = {
    "W": 2.40,
    "Y": 2.10,
    "R": 1.65,
    "F": 1.45,
    "I": 1.35,
    "L": 1.20,
    "D": 1.05,
    "H": 1.00,
    "E": 0.90,
    "M": 0.85,
    "V": 0.80,
    "K": 0.75,
    "N": 0.65,
    "Q": 0.55,
    "T": 0.45,
    "P": 0.40,
    "S": 0.30,
    "C": 0.30,
    "G": 0.20,
    "A": 0.00,
}


@dataclass
class BindingAlanineTarget:
    """Single protein–protein complex with a balanced candidate panel of `X -> Ala` binding mutations to rank."""

    target_id: str
    category: str
    assay_type: str
    scanned_sequence: str
    partner_sequence: str
    positions_1idx: List[int]
    wt_aas: List[str]
    exp_binding_loss: List[float]
    is_interface_flags: Optional[List[int]] = None
    panel_roles: Optional[List[str]] = None
    memorization_stratum: str = "known_pdb_pre2026"  # "known_pdb_pre2026" | "unseen_or_post2026"


@dataclass
class BindingAlanineParsed:
    pred_binding_loss: np.ndarray
    n_matched: int
    wt_match_rate: float
    parse_ok: bool


@dataclass
class BindingAlanineGrade:
    target_id: str
    category: str
    memorization_stratum: str
    scanned_length: int
    partner_length: int
    n_assayed: int
    parse_ok: bool
    coverage: float
    spearman_rho: float
    pearson_r: float
    ndcg_at_5: float
    top_k_precision: float
    same_aa_win_rate: float
    spearman_rho_aa_demeaned: float
    binding_hotspot_auroc: float
    interface_3d_auroc: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def build_binding_alanine_prompt(target: BindingAlanineTarget) -> str:
    """Build a zero-leakage two-chain `X -> Ala` binding-importance ranking prompt."""
    sa = target.scanned_sequence.strip().upper()
    sb = target.partner_sequence.strip().upper()
    for s in (sa, sb):
        if any(c not in CANONICAL_AA_SET for c in s):
            raise ValueError(f"Non-canonical residue in {target.target_id}")

    n_mut = len(target.positions_1idx)
    mut_tokens = [f"{p}{sa[p - 1]}->A" for p in target.positions_1idx]

    prompt = (
        f"You are performing an in-silico Interface Binding Alanine Scan Ranking (`X -> A`) on a protein–protein complex.\n\n"
        f"Scanned Protein A (1-indexed, length {len(sa)} AA — candidate Alanine mutations are on THIS chain):\n"
        f"{sa}\n\n"
        f"Binding Partner Protein B (length {len(sb)} AA — wild-type binding partner):\n"
        f"{sb}\n\n"
        f"Candidate Alanine Mutations on Protein A ({n_mut} total, listed in ascending sequence order):\n"
        f"{json.dumps(mut_tokens)}\n\n"
        f"Task:\n"
        f"Rank ALL {n_mut} candidate `X -> A` mutations on Protein A from **#1 (MOST critical binding hotspot — "
        f"largest loss of direct binding affinity to Partner Protein B when mutated to Alanine)** down to "
        f"**#{n_mut} (LEAST important for binding — neutral non-interface surface site or affinity-enhancing)**.\n"
        f"Notes:\n"
        f"  1. Monomer-unfolding core mutations have been excluded so every candidate folds stably as a monomer; "
        f"rank strictly by direct binding-interface importance to Partner B.\n"
        f"  2. The panel mixes true interface hotspots with non-interface decoys of the same amino-acid types.\n\n"
        f"Return ONLY a single compact JSON object with exact schema:\n"
        f'{{"ranking": ["{mut_tokens[0]}", "{mut_tokens[1]}", ...]}}\n'
        f"ordered strictly from #1 (most critical for binding) to #{n_mut} (least important for binding), "
        f"containing all {n_mut} mutations with no commentary outside JSON."
    )
    assert target.target_id not in prompt
    return prompt


def parse_binding_alanine_response(raw_text: str, target: BindingAlanineTarget) -> BindingAlanineParsed:
    """Parse EITHER ranked list `{"ranking": ["102Y->A", ...]}` OR continuous `{"binding_scan": [[pos, "WT", score], ...]}`."""
    n = len(target.positions_1idx)
    preds = np.zeros(n, dtype=np.float64)
    pos_map = {p: idx for idx, p in enumerate(target.positions_1idx)}

    if not raw_text or not raw_text.strip():
        return BindingAlanineParsed(preds, 0, 0.0, False)

    cleaned = raw_text.strip().replace("\u201c", '"').replace("\u201d", '"').replace("\u2018", "'").replace("\u2019", "'")
    m_f = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", cleaned, flags=re.DOTALL)
    if m_f:
        cleaned = m_f.group(1)
    else:
        i0, i1 = cleaned.find("{"), cleaned.rfind("}")
        if i0 != -1 and i1 > i0:
            cleaned = cleaned[i0 : i1 + 1]

    obj: Any = None
    try:
        obj = json.loads(cleaned)
    except Exception:
        obj = None

    # 1. Primary path: Ranked list `{"ranking": [...]}`
    if isinstance(obj, dict) and any(k in obj for k in ("ranking", "ranked_mutations", "ranked_positions", "order")):
        r_list = obj.get("ranking") or obj.get("ranked_mutations") or obj.get("ranked_positions") or obj.get("order") or []
        sc, n_m, wt_r = parse_ranking_items_to_scores(r_list, target.positions_1idx, target.wt_aas)
        return BindingAlanineParsed(
            pred_binding_loss=sc,
            n_matched=n_m,
            wt_match_rate=round(wt_r, 4),
            parse_ok=(n_m >= max(4, int(0.5 * n))),
        )

    # 2. Secondary path: Continuous specialist / legacy tuples `{"binding_scan": [[pos, "WT", score], ...]}`
    matched = 0
    wt_ok = 0
    seen: set[int] = set()

    if isinstance(obj, dict):
        rows = obj.get("binding_scan") or obj.get("scan") or obj.get("predictions") or []
        for item in rows:
            if isinstance(item, (list, tuple)) and len(item) >= 3:
                p = int(item[0])
                wt = str(item[1]).strip().upper()[:1]
                val = float(item[2])
            elif isinstance(item, dict):
                p = int(item.get("pos", item.get("position", -1)))
                wt = str(item.get("wt", "")).strip().upper()[:1]
                val = float(item.get("ddg_bind", item.get("ddg", item.get("score", 0.0))))
            else:
                continue
            if p in pos_map and p not in seen:
                seen.add(p)
                idx = pos_map[p]
                preds[idx] = val
                matched += 1
                if wt == target.wt_aas[idx]:
                    wt_ok += 1
    elif isinstance(obj, list) and obj and isinstance(obj[0], (str, int)):
        sc, n_m, wt_r = parse_ranking_items_to_scores(obj, target.positions_1idx, target.wt_aas)
        return BindingAlanineParsed(sc, n_m, round(wt_r, 4), n_m >= max(4, int(0.5 * n)))
    else:
        tok_matches = re.findall(r"\b(\d+[A-Z]\s*(?:->|→)\s*A)\b", raw_text.upper())
        if len(tok_matches) >= max(4, int(0.5 * n)):
            sc, n_m, wt_r = parse_ranking_items_to_scores(tok_matches, target.positions_1idx, target.wt_aas)
            return BindingAlanineParsed(sc, n_m, round(wt_r, 4), n_m >= max(4, int(0.5 * n)))

        for m in re.finditer(r"\[\s*(\d+)\s*,\s*\"?([A-Za-z])\"?\s*,\s*([-+]?\d*\.?\d+)", raw_text):
            p = int(m.group(1))
            wt = m.group(2).upper()
            val = float(m.group(3))
            if p in pos_map and p not in seen:
                seen.add(p)
                idx = pos_map[p]
                preds[idx] = val
                matched += 1
                if wt == target.wt_aas[idx]:
                    wt_ok += 1

    cov = matched / float(max(1, n))
    return BindingAlanineParsed(
        pred_binding_loss=preds,
        n_matched=matched,
        wt_match_rate=round(wt_ok / float(max(1, matched)), 4),
        parse_ok=(cov >= 0.50),
    )


def _safe_spearman(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 4 or float(np.std(x)) < 1e-12 or float(np.std(y)) < 1e-12:
        return 0.0
    r = float(spearmanr(x, y).statistic)
    return 0.0 if np.isnan(r) else round(r, 4)


def _safe_pearson(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 4 or float(np.std(x)) < 1e-12 or float(np.std(y)) < 1e-12:
        return 0.0
    r = float(pearsonr(x, y).statistic)
    return 0.0 if np.isnan(r) else round(r, 4)


def grade_binding_alanine_target(
    target: BindingAlanineTarget,
    parsed: BindingAlanineParsed,
) -> BindingAlanineGrade:
    y_true = np.asarray(target.exp_binding_loss, dtype=np.float64)
    y_pred = np.asarray(parsed.pred_binding_loss, dtype=np.float64)
    n = len(y_true)
    cov = parsed.n_matched / float(max(1, n))

    rho = _safe_spearman(y_pred, y_true) if parsed.parse_ok else 0.0
    pr = _safe_pearson(y_pred, y_true) if parsed.parse_ok else 0.0
    ndcg5 = compute_ndcg_at_k(y_true, y_pred, k=5) if parsed.parse_ok else 0.0
    topk_prec = compute_topk_hotspot_precision(y_true, y_pred) if parsed.parse_ok else 0.0
    same_aa_win = compute_same_aa_win_rate(target.wt_aas, y_true, y_pred) if parsed.parse_ok else 0.5
    rho_dm = compute_aa_demeaned_spearman(target.wt_aas, y_true, y_pred) if parsed.parse_ok else 0.0

    if target.assay_type == "spr_itc_ddg_kcal":
        hs_labels = [1 if v >= 1.0 else 0 for v in y_true]
        if sum(hs_labels) == 0 or sum(hs_labels) == n:
            thresh = float(np.quantile(y_true, 0.72))
            hs_labels = [1 if v >= thresh else 0 for v in y_true]
    else:
        thresh = float(np.quantile(y_true, 0.72))
        hs_labels = [1 if v >= thresh else 0 for v in y_true]

    hs_auc = (binary_auroc(hs_labels, y_pred) or 0.5) if parsed.parse_ok else 0.5

    iface_auc: Optional[float] = None
    if target.is_interface_flags and len(target.is_interface_flags) == n:
        if 0 < sum(target.is_interface_flags) < n:
            iface_auc = (binary_auroc(target.is_interface_flags, y_pred) or 0.5) if parsed.parse_ok else 0.5

    return BindingAlanineGrade(
        target_id=target.target_id,
        category=target.category,
        memorization_stratum=target.memorization_stratum,
        scanned_length=len(target.scanned_sequence),
        partner_length=len(target.partner_sequence),
        n_assayed=n,
        parse_ok=parsed.parse_ok,
        coverage=round(cov, 4),
        spearman_rho=rho,
        pearson_r=pr,
        ndcg_at_5=ndcg5,
        top_k_precision=topk_prec,
        same_aa_win_rate=same_aa_win,
        spearman_rho_aa_demeaned=rho_dm,
        binding_hotspot_auroc=round(float(hs_auc), 4),
        interface_3d_auroc=(round(float(iface_auc), 4) if iface_auc is not None else None),
    )
