"""Test 5: Monomer In-Silico Alanine Scan Ranking (`X_i -> A`) Prompts, Parsers, and Ranking Grader.

Evaluates whether frontier LLMs (`gpt-6.2-astra`, `opus-5-5`, `gpt-6.2`, `gemini-3.1-pro`)
can **rank** a balanced `15–20` residue candidate panel of `X -> Ala` mutations on an anonymized
monomeric protein from **#1 (most destabilizing / deleterious core hotspot)** to **#K (most neutral /
solvent-tolerated)**—including matched **same-amino-acid neutral decoys** (e.g., core `Leu` vs.
surface `Leu`).

Also accepts continuous specialist baseline scores (`ESM-2`, `ThermoMPNN`, `HydrophobicityLookup`)
so LLMs and GPU bio-tools are evaluated on the exact same ranking metrics:
  - `spearman_rho`: Full-panel rank correlation
  - `ndcg_at_5`: Normalized Discounted Cumulative Gain at Top-5 ranked mutations
  - `top_k_precision`: Precision on the top ~30% experimental hotspots (`Top-5` on an `18`-site panel)
  - `spearman_rho_aa_demeaned`: Within-AA context rank correlation (`0.000` for 1D lookup!)
  - `same_aa_win_rate`: Pairwise win rate on matched same-AA pairs (`0.500` coin-flip for 1D lookup!)
  - `hotspot_auroc_top20pct` & `core_burial_auroc`
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import math
import re
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy.stats import pearsonr, rankdata, spearmanr

from protein_without_tools.pdb_utils import CANONICAL_AA_SET

AA_TO_ALA_LOOKUP_DDG: Dict[str, float] = {
    "W": 2.65,
    "F": 2.45,
    "I": 2.35,
    "Y": 2.15,
    "L": 2.10,
    "V": 1.85,
    "M": 1.65,
    "C": 1.20,
    "T": 0.75,
    "P": 0.70,
    "H": 0.65,
    "N": 0.45,
    "Q": 0.40,
    "S": 0.35,
    "R": 0.35,
    "D": 0.30,
    "E": 0.25,
    "K": 0.20,
    "G": 0.15,
    "A": 0.00,
}

AA_CHEM_CLASS: Dict[str, str] = {
    "W": "arom", "Y": "arom", "F": "arom",
    "L": "aliph", "I": "aliph", "V": "aliph", "M": "aliph", "C": "aliph",
    "R": "pos", "K": "pos", "H": "pos",
    "D": "neg", "E": "neg",
    "S": "polar", "T": "polar", "N": "polar", "Q": "polar",
    "P": "turn", "G": "turn", "A": "ala",
}


def select_balanced_alanine_panel(
    positions_1idx: Sequence[int],
    wt_aas: Sequence[str],
    exp_scores: Sequence[float],
    target_panel_size: int = 18,
) -> Tuple[List[int], List[str]]:
    """Deterministically select a balanced `15–20` candidate ranking panel (`hotspot` + `same_aa_decoy` + `moderate`).

    Returns `(selected_indices_into_input, panel_roles)` sorted in **ascending sequence-position order**
    so sequence order leaks zero ranking information.
    """
    m = len(positions_1idx)
    if m <= 20:
        scores_arr = np.asarray(exp_scores, dtype=np.float64)
        q70 = float(np.quantile(scores_arr, 0.70))
        q35 = float(np.quantile(scores_arr, 0.35))
        roles = [
            "hotspot" if float(v) >= q70 else ("same_aa_decoy" if float(v) <= q35 else "moderate")
            for v in exp_scores
        ]
        return list(range(m)), roles

    k_hot = max(5, target_panel_size // 3)       # 6 hotspots
    k_decoy = max(5, target_panel_size // 3)     # 6 same-AA neutral decoys
    k_mod = target_panel_size - k_hot - k_decoy  # 6 moderate gradient sites

    scores = np.asarray(exp_scores, dtype=np.float64)
    desc_order = np.argsort(-scores).tolist()  # highest loss first
    asc_order = np.argsort(scores).tolist()    # most neutral / tolerated first

    chosen: Dict[int, str] = {}

    # 1. Pick top `k_hot` hotspots (preferring >=2 residue spacing so we sample distinct structural sites)
    for min_sep in (2, 0):
        for idx in desc_order:
            if len(chosen) >= k_hot:
                break
            if idx in chosen:
                continue
            p_cand = positions_1idx[idx]
            if min_sep > 0 and any(abs(p_cand - positions_1idx[c]) < min_sep for c in chosen):
                continue
            chosen[idx] = "hotspot"
        if len(chosen) >= k_hot:
            break

    hot_indices = [idx for idx, role in chosen.items() if role == "hotspot"]

    # 2. For each hotspot, pick a neutral decoy from the bottom 40% with the EXACT SAME wild-type AA
    #    (or same chemical class if that AA never occurs in the neutral pool).
    neutral_cutoff_rank = max(k_decoy + 4, int(0.42 * m))
    neutral_pool = [idx for idx in asc_order[:neutral_cutoff_rank] if idx not in chosen]

    for h_idx in hot_indices:
        if sum(1 for r in chosen.values() if r == "same_aa_decoy") >= k_decoy:
            break
        h_aa = wt_aas[h_idx]
        h_cls = AA_CHEM_CLASS.get(h_aa, "other")
        match_idx = next((idx for idx in neutral_pool if idx not in chosen and wt_aas[idx] == h_aa), None)
        if match_idx is None:
            match_idx = next(
                (idx for idx in neutral_pool if idx not in chosen and AA_CHEM_CLASS.get(wt_aas[idx]) == h_cls),
                None,
            )
        if match_idx is None:
            match_idx = next((idx for idx in asc_order if idx not in chosen), None)
        if match_idx is not None:
            chosen[match_idx] = "same_aa_decoy"

    while sum(1 for r in chosen.values() if r == "same_aa_decoy") < k_decoy:
        nxt = next((idx for idx in asc_order if idx not in chosen), None)
        if nxt is None:
            break
        chosen[nxt] = "same_aa_decoy"

    # 3. Fill remaining `k_mod` slots evenly from the middle quantiles (`30%..75%`)
    mid_pool = [
        idx for idx in desc_order[int(0.22 * m) : int(0.75 * m)]
        if idx not in chosen
    ]
    if len(mid_pool) >= k_mod and k_mod > 0:
        lin_picks = np.linspace(0, len(mid_pool) - 1, k_mod, dtype=int)
        for lp in lin_picks:
            chosen[mid_pool[int(lp)]] = "moderate"
    for idx in desc_order:
        if len(chosen) >= target_panel_size:
            break
        if idx not in chosen:
            chosen[idx] = "moderate"

    sorted_sel = sorted(chosen.keys(), key=lambda i: positions_1idx[i])
    return sorted_sel, [chosen[i] for i in sorted_sel]


@dataclass
class AlanineScanTarget:
    """Single protein target with a balanced candidate panel of `X -> A` mutations to rank."""

    target_id: str
    category: str
    assay_type: str
    sequence: str
    length: int
    positions_1idx: List[int]
    wt_aas: List[str]
    exp_destab_scores: List[float]
    exp_func_loss_scores: Optional[List[float]] = None
    is_core_flags: Optional[List[int]] = None
    panel_roles: Optional[List[str]] = None


@dataclass
class AlanineScanParsed:
    """Parsed per-position ranking importance scores aligned to `target.positions_1idx` (higher = ranked more critical)."""

    pred_ddg: np.ndarray
    pred_func_loss: np.ndarray
    n_matched_positions: int
    wt_aa_match_rate: float
    parse_ok: bool


@dataclass
class AlanineScanGrade:
    """Quantitative ranking metrics for a protein's in-silico alanine scan."""

    target_id: str
    category: str
    length: int
    n_assayed: int
    parse_ok: bool
    coverage: float
    spearman_rho: float
    pearson_r: float
    ndcg_at_5: float
    top_k_precision: float
    same_aa_win_rate: float
    spearman_rho_aa_demeaned: float
    hotspot_auroc_top20pct: float
    core_burial_auroc: Optional[float] = None
    func_only_hotspot_auroc: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def non_alanine_positions(sequence: str) -> List[int]:
    return [idx + 1 for idx, aa in enumerate(sequence) if aa != "A"]


def build_alanine_scan_prompt(target: AlanineScanTarget) -> str:
    """Build a zero-leakage ranking prompt over the candidate `X -> A` mutations."""
    seq = target.sequence.strip().upper()
    L = len(seq)
    if L != target.length:
        raise ValueError(f"Sequence length {L} != target.length {target.length}")
    for ch in seq:
        if ch not in CANONICAL_AA_SET:
            raise ValueError(f"Invalid amino acid {ch!r} in {target.target_id}")

    n_mut = len(target.positions_1idx)
    mut_tokens = [f"{p}{seq[p - 1]}->A" for p in target.positions_1idx]

    prompt = (
        f"You are performing an in-silico Alanine Scan Ranking (`X -> A`) on an anonymized "
        f"protein of length L = {L} residues.\n\n"
        f"Sequence (1-indexed, length {L}):\n"
        f"{seq}\n\n"
        f"Candidate Alanine Mutations ({n_mut} total, listed in ascending sequence order):\n"
        f"{json.dumps(mut_tokens)}\n\n"
        f"Task:\n"
        f"Rank ALL {n_mut} candidate `X -> A` mutations from **#1 (MOST disruptive / critical hotspot — "
        f"largest loss of 3D folding stability or biological activity when mutated to Alanine)** down to "
        f"**#{n_mut} (LEAST disruptive — most neutral, surface-tolerated, or stabilizing)**.\n"
        f"Note: the panel contains both buried/critical residues and neutral decoys of the same amino-acid types, "
        f"so rank by 3D structural context rather than amino-acid identity alone.\n\n"
        f"Return ONLY a single compact JSON object with exact schema:\n"
        f'{{"ranking": ["{mut_tokens[0]}", "{mut_tokens[1]}", ...]}}\n'
        f"ordered strictly from #1 (most disruptive) to #{n_mut} (most neutral), containing all {n_mut} mutations "
        f"with no commentary outside JSON."
    )
    assert target.target_id not in prompt
    return prompt


def parse_ranking_items_to_scores(
    rank_list: Sequence[Any],
    positions_1idx: Sequence[int],
    wt_aas: Sequence[str],
) -> Tuple[np.ndarray, int, float]:
    """Convert an ordered list `["7W->A", "11L->A", ...]` (#1 most critical first) into descending importance scores."""
    n = len(positions_1idx)
    scores = np.zeros(n, dtype=np.float64)
    pos_to_idx = {p: idx for idx, p in enumerate(positions_1idx)}
    seen: set[int] = set()
    wt_matches = 0

    for item in rank_list:
        p_i: Optional[int] = None
        wt_c: Optional[str] = None
        if isinstance(item, int):
            p_i = item
        elif isinstance(item, str):
            s = item.strip().upper()
            # Match `7W->A`, `7W`, `W7A`, `W7->A`, `POS 7 (W)`
            m1 = re.match(r"^(\d+)\s*([A-Z])(?:\s*(?:->|→|>)\s*A)?$", s)
            m2 = re.match(r"^([A-Z])\s*(\d+)(?:\s*(?:->|→|>)?\s*A)?$", s)
            m3 = re.search(r"(\d+)", s)
            if m1:
                p_i, wt_c = int(m1.group(1)), m1.group(2)
            elif m2:
                wt_c, p_i = m2.group(1), int(m2.group(2))
            elif m3:
                p_i = int(m3.group(1))
        elif isinstance(item, (list, tuple)) and len(item) >= 2:
            # Handle `[rank, "7W->A"]` or `[rank, 7, "W"]` or `[7, "W"]`
            if isinstance(item[1], str) and re.search(r"\d", item[1]):
                sub_s, sub_m, sub_w = parse_ranking_items_to_scores([item[1]], positions_1idx, wt_aas)
                if sub_m == 1:
                    idx_hit = int(np.argmax(sub_s))
                    p_i = positions_1idx[idx_hit]
                    wt_c = wt_aas[idx_hit] if sub_w == 1.0 else None
            elif len(item) >= 3 and isinstance(item[2], str) and len(item[2]) == 1:
                try:
                    p_i = int(item[1])
                    wt_c = item[2].upper()
                except (TypeError, ValueError):
                    pass
            else:
                try:
                    p_i = int(item[0])
                    wt_c = str(item[1]).strip().upper()[:1]
                except (TypeError, ValueError):
                    pass
        elif isinstance(item, dict):
            raw_mut = item.get("mutation") or item.get("mut") or item.get("id")
            if raw_mut is not None:
                sub_s, sub_m, sub_w = parse_ranking_items_to_scores([raw_mut], positions_1idx, wt_aas)
                if sub_m == 1:
                    idx_hit = int(np.argmax(sub_s))
                    p_i = positions_1idx[idx_hit]
                    wt_c = wt_aas[idx_hit] if sub_w == 1.0 else None
            elif "pos" in item or "position" in item:
                try:
                    p_i = int(item.get("pos") or item.get("position"))
                    wt_c = str(item.get("wt") or "").strip().upper()[:1] or None
                except (TypeError, ValueError):
                    pass

        if p_i in pos_to_idx and p_i not in seen:
            idx = pos_to_idx[p_i]
            rank_0 = len(seen)
            seen.add(p_i)
            # Strict descending score: #1 gets `n`, #2 gets `n-1`, ..., #n gets `1`
            scores[idx] = float(n - rank_0)
            if wt_c is None or wt_c == wt_aas[idx]:
                wt_matches += 1

    n_matched = len(seen)
    return scores, n_matched, (wt_matches / float(max(1, n_matched)))


def parse_alanine_scan_response(raw_text: str, target: AlanineScanTarget) -> AlanineScanParsed:
    """Parse EITHER ranked list `{"ranking": ["7W->A", ...]}` OR continuous `{"scan": [[pos, "WT", score], ...]}`."""
    n = len(target.positions_1idx)
    pred_ddg = np.zeros(n, dtype=np.float64)
    pred_func = np.zeros(n, dtype=np.float64)
    pos_to_idx = {p: idx for idx, p in enumerate(target.positions_1idx)}

    if not raw_text or not raw_text.strip():
        return AlanineScanParsed(pred_ddg, pred_func, 0, 0.0, False)

    cleaned = raw_text.strip().replace("\u201c", '"').replace("\u201d", '"').replace("\u2018", "'").replace("\u2019", "'")
    fence_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", cleaned, flags=re.DOTALL)
    if fence_match:
        cleaned = fence_match.group(1)
    else:
        b0, b1 = cleaned.find("{"), cleaned.rfind("}")
        if b0 != -1 and b1 > b0:
            cleaned = cleaned[b0 : b1 + 1]

    obj: Any = None
    try:
        obj = json.loads(cleaned)
    except json.JSONDecodeError:
        obj = None

    # 1. Primary path: Ranked list `{"ranking": [...]}`
    if isinstance(obj, dict) and any(k in obj for k in ("ranking", "ranked_mutations", "ranked_positions", "order")):
        r_list = obj.get("ranking") or obj.get("ranked_mutations") or obj.get("ranked_positions") or obj.get("order") or []
        sc, n_m, wt_r = parse_ranking_items_to_scores(r_list, target.positions_1idx, target.wt_aas)
        fn_sc = sc.copy()
        if isinstance(obj.get("func_ranking"), list):
            fn_sc, _, _ = parse_ranking_items_to_scores(obj["func_ranking"], target.positions_1idx, target.wt_aas)
        return AlanineScanParsed(
            pred_ddg=sc,
            pred_func_loss=fn_sc,
            n_matched_positions=n_m,
            wt_aa_match_rate=round(wt_r, 4),
            parse_ok=(n_m >= max(4, int(0.5 * n))),
        )

    # 2. Secondary path: Continuous specialist / legacy tuples `{"scan": [[pos, "WT", ddg, func], ...]}`
    rows: Optional[List[Any]] = None
    if isinstance(obj, dict):
        rows = obj.get("scan") or obj.get("predictions") or obj.get("mutations")
    elif isinstance(obj, list):
        # Could be a bare ranked list `["7W->A", ...]` or tuple list
        if obj and isinstance(obj[0], (str, int)):
            sc, n_m, wt_r = parse_ranking_items_to_scores(obj, target.positions_1idx, target.wt_aas)
            return AlanineScanParsed(sc, sc.copy(), n_m, round(wt_r, 4), n_m >= max(4, int(0.5 * n)))
        rows = obj

    if rows is None:
        # Regex fallback for `"7W->A"` tokens if `ranking` JSON truncated
        tok_matches = re.findall(r"\b(\d+[A-Z]\s*(?:->|→)\s*A)\b", raw_text.upper())
        if len(tok_matches) >= max(4, int(0.5 * n)):
            sc, n_m, wt_r = parse_ranking_items_to_scores(tok_matches, target.positions_1idx, target.wt_aas)
            return AlanineScanParsed(sc, sc.copy(), n_m, round(wt_r, 4), n_m >= max(4, int(0.5 * n)))

        rows = []
        pat = re.compile(
            r"\[\s*(\d+)\s*,\s*\"?([A-Z])\"?\s*,\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)"
            r"(?:\s*,\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?))?\s*\]"
        )
        for m in pat.finditer(raw_text):
            p_i = int(m.group(1))
            wt_c = m.group(2)
            v_ddg = float(m.group(3))
            v_fn = float(m.group(4)) if m.group(4) is not None else v_ddg
            rows.append([p_i, wt_c, v_ddg, v_fn])

    if not rows:
        return AlanineScanParsed(pred_ddg, pred_func, 0, 0.0, False)

    seen = set()
    wt_matches = 0
    for item in rows:
        p_i: Optional[int] = None
        wt_c: Optional[str] = None
        v_ddg: Optional[float] = None
        v_fn: Optional[float] = None
        if isinstance(item, (list, tuple)) and len(item) >= 3:
            try:
                if isinstance(item[1], str) and len(item[1]) == 1:
                    p_i = int(item[0])
                    wt_c = item[1].upper()
                    v_ddg = float(item[2])
                    v_fn = float(item[3]) if len(item) >= 4 else v_ddg
                else:
                    p_i = int(item[0])
                    v_ddg = float(item[1])
                    v_fn = float(item[2])
            except (TypeError, ValueError):
                continue
        elif isinstance(item, dict):
            try:
                p_i = int(item.get("pos") or item.get("position"))
                wt_c = str(item.get("wt") or item.get("aa") or "").upper()[:1] or None
                v_ddg = float(item.get("ddg", 0.0))
                v_fn = float(item.get("func_loss", item.get("fitness_loss", v_ddg)))
            except (TypeError, ValueError):
                continue

        if p_i in pos_to_idx and p_i not in seen and v_ddg is not None:
            k = pos_to_idx[p_i]
            seen.add(p_i)
            pred_ddg[k] = v_ddg
            pred_func[k] = v_fn if v_fn is not None else v_ddg
            if wt_c is None or wt_c == target.wt_aas[k]:
                wt_matches += 1

    n_matched = len(seen)
    wt_rate = wt_matches / float(max(1, n_matched))
    return AlanineScanParsed(
        pred_ddg=pred_ddg,
        pred_func_loss=pred_func,
        n_matched_positions=n_matched,
        wt_aa_match_rate=round(wt_rate, 4),
        parse_ok=(n_matched >= max(4, int(0.5 * n))),
    )


def binary_auroc(labels_01: Sequence[int], scores: Sequence[float]) -> Optional[float]:
    """Exact Mann-Whitney U AUROC without sklearn dependency."""
    y = np.asarray(labels_01, dtype=np.int32)
    s = np.asarray(scores, dtype=np.float64)
    n_pos = int(np.sum(y == 1))
    n_neg = int(np.sum(y == 0))
    if n_pos == 0 or n_neg == 0:
        return None
    order = np.argsort(s)
    ranks = np.empty(len(s), dtype=np.float64)
    i = 0
    while i < len(s):
        j = i
        while j + 1 < len(s) and s[order[j + 1]] == s[order[i]]:
            j += 1
        avg_rank = 0.5 * (i + j) + 1.0
        ranks[order[i : j + 1]] = avg_rank
        i = j + 1
    sum_pos_ranks = float(np.sum(ranks[y == 1]))
    auc = (sum_pos_ranks - n_pos * (n_pos + 1) / 2.0) / float(n_pos * n_neg)
    return round(float(auc), 4)


def compute_ndcg_at_k(exp_vals: np.ndarray, pred_scores: np.ndarray, k: int = 5) -> float:
    """Normalized Discounted Cumulative Gain at Top-`k` (tie-averaged when `pred_scores` has ties)."""
    n = len(exp_vals)
    if n == 0:
        return 0.0
    k_eff = min(k, n)
    # Shift experimental relevance above median so neutral/stabilizing decoys have 0 gain and hotspots dominate
    base = float(np.percentile(exp_vals, 45.0))
    gains = np.maximum(0.0, exp_vals - base)
    if float(np.max(gains)) <= 1e-12:
        gains = np.maximum(0.0, exp_vals - float(np.min(exp_vals)))
    if float(np.max(gains)) <= 1e-12:
        return 0.0

    discounts = np.array([1.0 / math.log2(r + 2.0) for r in range(n)], dtype=np.float64)
    discounts[k_eff:] = 0.0

    ideal_order = np.argsort(-gains)
    idcg = float(np.sum(gains[ideal_order] * discounts))
    if idcg <= 1e-12:
        return 0.0

    # Tie-aware expected DCG (so constant/1D-tied baselines get exact tie-averaged expectation, zero order bias!)
    desc = np.argsort(-pred_scores, kind="mergesort")
    dcg = 0.0
    i = 0
    while i < n:
        j = i
        while j + 1 < n and abs(pred_scores[desc[j + 1]] - pred_scores[desc[i]]) <= 1e-12:
            j += 1
        avg_disc = float(np.mean(discounts[i : j + 1]))
        dcg += float(np.sum(gains[desc[i : j + 1]])) * avg_disc
        i = j + 1

    return round(float(dcg / idcg), 4)


def compute_topk_hotspot_precision(exp_vals: np.ndarray, pred_scores: np.ndarray) -> float:
    """Tie-aware Precision@K on the top ~30% experimental hotspots (`K = max(3, round(0.30 * N))`)."""
    n = len(exp_vals)
    if n < 4:
        return 0.0
    k_top = max(3, min(n - 1, int(round(0.30 * n))))
    true_top_idx = set(np.argsort(-exp_vals)[:k_top].tolist())
    is_hot = np.array([1.0 if idx in true_top_idx else 0.0 for idx in range(n)], dtype=np.float64)

    weights = np.zeros(n, dtype=np.float64)
    weights[:k_top] = 1.0

    desc = np.argsort(-pred_scores, kind="mergesort")
    hits = 0.0
    i = 0
    while i < n:
        j = i
        while j + 1 < n and abs(pred_scores[desc[j + 1]] - pred_scores[desc[i]]) <= 1e-12:
            j += 1
        avg_w = float(np.mean(weights[i : j + 1]))
        hits += float(np.sum(is_hot[desc[i : j + 1]])) * avg_w
        i = j + 1

    return round(float(hits / k_top), 4)


def compute_same_aa_win_rate(
    wt_aas: Sequence[str],
    exp_vals: np.ndarray,
    pred_scores: np.ndarray,
) -> float:
    """Pairwise win rate across pairs `(i, j)` sharing the **exact same wild-type amino acid** (`wt_i == wt_j`).

    Any 1D amino-acid lookup table assigns identical score to `wt_i` and `wt_j`, scoring **exact `0.500`**!
    """
    sd_exp = float(np.std(exp_vals))
    min_gap = max(1e-6, 0.25 * sd_exp)
    wins = 0.0
    total = 0
    n = len(wt_aas)
    for i in range(n):
        for j in range(i + 1, n):
            if wt_aas[i] != wt_aas[j]:
                continue
            diff_e = float(exp_vals[i] - exp_vals[j])
            if abs(diff_e) < min_gap:
                continue
            total += 1
            diff_p = float(pred_scores[i] - pred_scores[j])
            if abs(diff_p) <= 1e-12:
                wins += 0.5
            elif diff_e * diff_p > 0:
                wins += 1.0
    if total == 0:
        return 0.5
    return round(float(wins / total), 4)


def compute_aa_demeaned_spearman(
    wt_aas: Sequence[str],
    exp_vals: np.ndarray,
    pred_vals: np.ndarray,
) -> float:
    """Rank-demean BOTH `exp_vals` and `pred_vals` within each wild-type amino acid:
    - Any static 1D amino-acid lookup table assigns equal rank to all instances of AA `X` -> scores `0.000`!
    - An exact rank-order prediction (`{"ranking": [...]}`) scores `1.000` with zero unit/scale distortion!
    """
    exp_res = rankdata(exp_vals, method="average").astype(np.float64)
    pred_res = rankdata(pred_vals, method="average").astype(np.float64)
    by_aa: Dict[str, List[int]] = {}
    for idx, aa in enumerate(wt_aas):
        by_aa.setdefault(aa, []).append(idx)

    valid_idx: List[int] = []
    for _aa, idxs in by_aa.items():
        if len(idxs) >= 2:
            arr_i = np.array(idxs, dtype=np.int32)
            exp_res[arr_i] -= float(np.mean(exp_res[arr_i]))
            pred_res[arr_i] -= float(np.mean(pred_res[arr_i]))
            valid_idx.extend(idxs)

    if len(valid_idx) < 4:
        return 0.0
    vi = np.array(sorted(valid_idx), dtype=np.int32)
    if np.std(exp_res[vi]) < 1e-9 or np.std(pred_res[vi]) < 1e-9:
        return 0.0
    rho, _ = spearmanr(exp_res[vi], pred_res[vi])
    return 0.0 if np.isnan(rho) else round(float(rho), 4)


def grade_alanine_scan(target: AlanineScanTarget, parsed: AlanineScanParsed) -> AlanineScanGrade:
    n = len(target.positions_1idx)
    exp = np.asarray(target.exp_destab_scores, dtype=np.float64)
    pred = parsed.pred_func_loss if target.assay_type == "dms_activity" else parsed.pred_ddg

    if not parsed.parse_ok or np.std(pred) < 1e-9 or np.std(exp) < 1e-9:
        rho_val = 0.0
        pr_val = 0.0
    else:
        r_s, _ = spearmanr(exp, pred)
        r_p, _ = pearsonr(exp, pred)
        rho_val = 0.0 if np.isnan(r_s) else float(r_s)
        pr_val = 0.0 if np.isnan(r_p) else float(r_p)

    ndcg5 = compute_ndcg_at_k(exp, pred, k=5) if parsed.parse_ok else 0.0
    topk_prec = compute_topk_hotspot_precision(exp, pred) if parsed.parse_ok else 0.0
    same_aa_win = compute_same_aa_win_rate(target.wt_aas, exp, pred) if parsed.parse_ok else 0.5
    rho_ctx = compute_aa_demeaned_spearman(target.wt_aas, exp, pred) if parsed.parse_ok else 0.0

    thresh_75 = float(np.percentile(exp, 72.0))
    hot_labels = [1 if v >= thresh_75 else 0 for v in exp]
    hot_auc = (binary_auroc(hot_labels, pred) or 0.5) if parsed.parse_ok else 0.5

    core_auc: Optional[float] = None
    if target.is_core_flags is not None and len(target.is_core_flags) == n:
        if 0 < sum(target.is_core_flags) < n:
            core_auc = binary_auroc(target.is_core_flags, parsed.pred_ddg) if parsed.parse_ok else 0.5

    func_only_auc: Optional[float] = None
    if target.exp_func_loss_scores is not None and len(target.exp_func_loss_scores) == n:
        exp_fn = np.asarray(target.exp_func_loss_scores, dtype=np.float64)
        med_ddg = float(np.median(exp))
        fn_cut = float(np.percentile(exp_fn, 70.0))
        fn_only_labels = [1 if (e_d <= med_ddg and e_f >= fn_cut) else 0 for e_d, e_f in zip(exp, exp_fn)]
        if 0 < sum(fn_only_labels) < n:
            z_fn = (parsed.pred_func_loss - np.mean(parsed.pred_func_loss)) / (np.std(parsed.pred_func_loss) + 1e-8)
            z_ddg = (parsed.pred_ddg - np.mean(parsed.pred_ddg)) / (np.std(parsed.pred_ddg) + 1e-8)
            func_only_auc = binary_auroc(fn_only_labels, z_fn - 0.5 * z_ddg)

    return AlanineScanGrade(
        target_id=target.target_id,
        category=target.category,
        length=target.length,
        n_assayed=n,
        parse_ok=parsed.parse_ok,
        coverage=round(parsed.n_matched_positions / float(max(1, n)), 4),
        spearman_rho=round(rho_val, 4),
        pearson_r=round(pr_val, 4),
        ndcg_at_5=ndcg5,
        top_k_precision=topk_prec,
        same_aa_win_rate=same_aa_win,
        spearman_rho_aa_demeaned=rho_ctx,
        hotspot_auroc_top20pct=hot_auc,
        core_burial_auroc=core_auc,
        func_only_hotspot_auroc=func_only_auc,
    )
