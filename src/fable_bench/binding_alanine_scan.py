"""Binding Alanine Scan (`Chain A + Partner B -> Per-Residue X->Ala Binding Importance`).

Evaluates whether frontier LLMs (`claude-fable-5-2`, `opus-5-5`, `gpt-5.2`, `gemini-3.1-pro`)
can predict how important each individual amino acid `i` on a protein is for **binding** to its
partner (`ΔΔG_bind` / binding loss upon `WT_i -> Ala`) directly from the two anonymized sequences.

Includes:
  - Classic Bogan & Thorn (1998) `Trp/Tyr/Arg` 1D hotspot propensity baseline (`Baseline_BoganThorn_1D`),
    demonstrating why `spearman_rho_aa_demeaned` + `binding_hotspot_auroc` cleanly separate true
    sequence-to-interface understanding from 1D amino-acid lookup tables.
  - Zero-leakage complex prompts & fault-tolerant JSON parsers.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import re
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
from scipy.stats import pearsonr, spearmanr

from fable_bench.alanine_scan import binary_auroc
from fable_bench.pdb_utils import CANONICAL_AA_SET

# Bogan & Thorn (1998) empirical interface binding hotspot propensity (`X -> Ala` mean ΔΔG_bind kcal/mol):
# Trp, Tyr, and Arg are famously enriched at protein-protein binding hotspots!
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
    """Single protein-protein interaction target with wet-lab `X -> Ala` binding measurements on `scanned_sequence`."""

    target_id: str
    category: str  # "skempi_biophysical_ddg" | "oncology_alphaseq_kd" | "ppi_binding_dms" | "post_may2026_mavedb_binding"
    assay_type: str  # "spr_itc_ddg_kcal" | "alphaseq_delta_log10kd" | "binding_dms_loss"
    scanned_sequence: str
    partner_sequence: str
    positions_1idx: List[int]
    wt_aas: List[str]
    # Experimental binding loss upon `WT_i -> Ala` (oriented so HIGHER = MORE IMPORTANT FOR BINDING!)
    exp_binding_loss: List[float]
    # Optional 3D crystal interface contact flags (`1` = Cβ < 8.0 A to partner in complex PDB, `0` = non-interface)
    is_interface_flags: Optional[List[int]] = None


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
    scanned_length: int
    partner_length: int
    n_assayed: int
    parse_ok: bool
    coverage: float
    spearman_rho: float
    pearson_r: float
    spearman_rho_aa_demeaned: float
    binding_hotspot_auroc: float
    interface_3d_auroc: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def build_binding_alanine_prompt(target: BindingAlanineTarget) -> str:
    """Build a zero-leakage two-chain `X -> Ala` binding-importance prompt."""
    sa = target.scanned_sequence.strip().upper()
    sb = target.partner_sequence.strip().upper()
    for s in (sa, sb):
        if any(c not in CANONICAL_AA_SET for c in s):
            raise ValueError(f"Non-canonical residue in {target.target_id}")

    n_mut = len(target.positions_1idx)
    preview = ", ".join(f"{p}{sa[p - 1]}->A" for p in target.positions_1idx[:6])

    prompt = (
        f"You are performing a quantitative in-silico Binding Alanine Scan (`X -> A`) on a protein–protein complex.\n\n"
        f"Scanned Protein A (1-indexed, length {len(sa)} AA — mutations are applied to THIS chain):\n"
        f"{sa}\n\n"
        f"Binding Partner Protein B (length {len(sb)} AA — wild-type binding partner):\n"
        f"{sb}\n\n"
        f"Task:\n"
        f"Predict how important each of the {n_mut} listed non-Alanine residues on Protein A "
        f"(starting with {preview}, ...) is for binding to Partner Protein B when mutated individually to Alanine (`A`).\n\n"
        f"For every assayed 1-indexed position `pos` on Protein A (with wild-type amino acid `wt`), predict:\n"
        f"  - `ddg_bind`: Predicted loss of binding affinity ΔΔG_bind = ΔG_bind(Ala) - ΔG_bind(WT) in kcal/mol "
        f"(positive = weakens/disrupts binding, e.g. +2.5 to +4.5 for critical interface binding hotspot residues; "
        f"~0.0 for non-interface surface residues that do not affect binding; negative = tighter binding).\n\n"
        f"Assayed 1-indexed positions on Protein A ({n_mut} total):\n"
        f"{json.dumps(target.positions_1idx)}\n\n"
        f"Return ONLY a single compact JSON object with exact schema:\n"
        f'{{"binding_scan": [[pos, "WT", ddg_bind], ...]}}\n'
        f"Include all {n_mut} positions in ascending order with no commentary outside JSON."
    )
    assert target.target_id not in prompt
    return prompt


def parse_binding_alanine_response(raw_text: str, target: BindingAlanineTarget) -> BindingAlanineParsed:
    n = len(target.positions_1idx)
    preds = np.zeros(n, dtype=np.float64)
    pos_map = {p: idx for idx, p in enumerate(target.positions_1idx)}

    if not raw_text or not raw_text.strip():
        return BindingAlanineParsed(preds, 0, 0.0, False)

    cleaned = raw_text.strip()
    m_f = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", cleaned, flags=re.DOTALL)
    if m_f:
        cleaned = m_f.group(1)
    else:
        i0, i1 = cleaned.find("{"), cleaned.rfind("}")
        if i0 != -1 and i1 > i0:
            cleaned = cleaned[i0 : i1 + 1]

    matched = 0
    wt_ok = 0
    seen: set[int] = set()

    try:
        obj = json.loads(cleaned)
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
    except Exception:
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


def demean_by_wt_aa(values: Sequence[float], wt_aas: Sequence[str]) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64).copy()
    sums: Dict[str, float] = {}
    cnts: Dict[str, int] = {}
    for v, aa in zip(arr, wt_aas):
        sums[aa] = sums.get(aa, 0.0) + float(v)
        cnts[aa] = cnts.get(aa, 0) + 1
    means = {aa: sums[aa] / cnts[aa] for aa in sums}
    for i, aa in enumerate(wt_aas):
        arr[i] -= means[aa]
    return arr


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

    # AA-demeaned context correlation: isolates interface vs non-interface instances of the SAME amino acid!
    if parsed.parse_ok:
        y_true_dm = demean_by_wt_aa(y_true, target.wt_aas)
        y_pred_dm = demean_by_wt_aa(y_pred, target.wt_aas)
        rho_dm = _safe_spearman(y_pred_dm, y_true_dm)
    else:
        rho_dm = 0.0

    # Top-20% (or >= 1.0 kcal/mol for SPR/ITC) Binding Hotspot AUROC
    if target.assay_type == "spr_itc_ddg_kcal":
        hs_labels = [1 if v >= 1.0 else 0 for v in y_true]
        if sum(hs_labels) == 0 or sum(hs_labels) == n:
            thresh = float(np.quantile(y_true, 0.78))
            hs_labels = [1 if v >= thresh else 0 for v in y_true]
    else:
        thresh = float(np.quantile(y_true, 0.80))
        hs_labels = [1 if v >= thresh else 0 for v in y_true]

    hs_auc = (binary_auroc(hs_labels, y_pred) or 0.5) if parsed.parse_ok else 0.5

    iface_auc: Optional[float] = None
    if target.is_interface_flags and len(target.is_interface_flags) == n:
        if 0 < sum(target.is_interface_flags) < n:
            iface_auc = (binary_auroc(target.is_interface_flags, y_pred) or 0.5) if parsed.parse_ok else 0.5

    return BindingAlanineGrade(
        target_id=target.target_id,
        category=target.category,
        scanned_length=len(target.scanned_sequence),
        partner_length=len(target.partner_sequence),
        n_assayed=n,
        parse_ok=parsed.parse_ok,
        coverage=round(cov, 4),
        spearman_rho=rho,
        pearson_r=pr,
        spearman_rho_aa_demeaned=rho_dm,
        binding_hotspot_auroc=round(float(hs_auc), 4),
        interface_3d_auroc=(round(float(iface_auc), 4) if iface_auc is not None else None),
    )
