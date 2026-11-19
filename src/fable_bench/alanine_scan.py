"""Stage 3A: In-Silico Alanine Scan (`X_i -> A`) Prompts, Parsers, and Biophysical Grader.

Evaluates whether frontier LLMs (e.g., `claude-fable-5-2`) can predict the experimental
energetic (`ΔΔG_fold`) and functional (`DMS`) consequences of mutating every non-Alanine
residue `i` to Alanine (`X_i -> A`) directly from an anonymized 1D sequence.

Crucially, includes an **Amino-Acid-Demeaned Context Correlation (`rho_context`)** metric
alongside raw Spearman `rho`, testing whether a model distinguishes *buried vs. exposed*
instances of the **same** amino acid (e.g. core Leu vs. surface Leu) rather than relying
on a static 20-letter hydrophobicity lookup table.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import re
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy.stats import pearsonr, spearmanr

from fable_bench.pdb_utils import CANONICAL_AA_SET

# Empirical mean ΔΔG (kcal/mol) of X -> Ala mutations across globular proteins
# (derived from classic Kellis/Matthews/Tsuboyama side-chain truncation scales),
# used ONLY as our zero-context 1D lookup baseline (`Baseline_AA_HydrophobicityLookup`).
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


@dataclass
class AlanineScanTarget:
    """Single protein target with experimental X -> A ground-truth measurements."""

    target_id: str
    category: str  # "de_novo_ddg" | "dual_stability_function" | "post_cutoff_mavedb"
    assay_type: str  # "ddg_kcal" | "dms_activity" | "dual"
    sequence: str
    length: int
    # 1-indexed non-Alanine positions that have experimental X -> A measurements
    positions_1idx: List[int]
    wt_aas: List[str]
    # Experimental destabilization / loss score (oriented so HIGHER = MORE DESTABILIZING / DELETERIOUS!)
    # For ΔΔG_fold targets: exact ΔΔG = ΔG_mut(Ala) - ΔG_WT in kcal/mol (+ = destabilizing).
    # For DMS fitness targets: -Δfitness so positive = loss of function.
    exp_destab_scores: List[float]
    # Optional secondary functional-loss assay (for Category B dual-assay targets, higher = worse function)
    exp_func_loss_scores: Optional[List[float]] = None
    # Optional 3D burial flag (`1` = buried core residue in 3D structure, `0` = surface)
    is_core_flags: Optional[List[int]] = None


@dataclass
class AlanineScanParsed:
    """Parsed per-position LLM predictions aligned to `target.positions_1idx`."""

    pred_ddg: np.ndarray
    pred_func_loss: np.ndarray
    n_matched_positions: int
    wt_aa_match_rate: float
    parse_ok: bool


@dataclass
class AlanineScanGrade:
    """Quantitative metrics for a single protein's in-silico alanine scan."""

    target_id: str
    category: str
    length: int
    n_assayed: int
    parse_ok: bool
    coverage: float
    spearman_rho: float
    pearson_r: float
    spearman_rho_aa_demeaned: float
    hotspot_auroc_top20pct: float
    core_burial_auroc: Optional[float] = None
    func_only_hotspot_auroc: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def non_alanine_positions(sequence: str) -> List[int]:
    """Return 1-indexed positions where `sequence[i] != 'A'`."""
    return [idx + 1 for idx, aa in enumerate(sequence) if aa != "A"]


def build_alanine_scan_prompt(target: AlanineScanTarget) -> str:
    """Build a compact, zero-leakage full-protein `X_i -> A` alanine scan prompt."""
    seq = target.sequence.strip().upper()
    L = len(seq)
    if L != target.length:
        raise ValueError(f"Sequence length {L} != target.length {target.length}")
    for ch in seq:
        if ch not in CANONICAL_AA_SET:
            raise ValueError(f"Invalid amino acid {ch!r} in {target.target_id}")

    pos_preview = ", ".join(f"{p}{seq[p - 1]}->A" for p in target.positions_1idx[:6])
    n_mut = len(target.positions_1idx)

    prompt = (
        f"You are performing a quantitative in-silico Alanine Scan (`X -> A`) on an anonymized "
        f"single-domain protein of length L = {L} residues.\n\n"
        f"Sequence (1-indexed, length {L}):\n"
        f"{seq}\n\n"
        f"Task:\n"
        f"Predict the biophysical effect of mutating EACH of the {n_mut} non-Alanine positions "
        f"(starting with {pos_preview}, ... across all {n_mut} assayed positions) to Alanine (`A`).\n\n"
        f"For every assayed 1-indexed position `pos` (with wild-type residue `wt`), output two numbers:\n"
        f"  1. `ddg`: Predicted folding stability change ΔΔG_fold = ΔG(Ala) - ΔG(WT) in kcal/mol "
        f"(positive = destabilizing to the 3D fold, e.g. +2.5 for a buried core hydrophobic cavity; "
        f"~0.0 for a solvent-exposed neutral site; negative = stabilizing).\n"
        f"  2. `func_loss`: Predicted overall loss of biological activity / fitness in [0.0, 1.0] "
        f"(0.0 = wild-type/neutral, 1.0 = severe loss of folding stability, catalysis, or binding).\n\n"
        f"Assayed 1-indexed positions ({n_mut} total):\n"
        f"{json.dumps(target.positions_1idx)}\n\n"
        f"Return ONLY a single compact JSON object with exact schema:\n"
        f'{{"scan": [[pos, "WT", ddg, func_loss], ...]}}\n'
        f"Include all {n_mut} listed positions in ascending order with no markdown commentary outside JSON."
    )

    # Zero-leakage check: target_id is anonymized (e.g. `s3a_t001`); ensure no real PDB/gene leakage
    assert target.target_id not in prompt
    return prompt


def parse_alanine_scan_response(raw_text: str, target: AlanineScanTarget) -> AlanineScanParsed:
    """Fault-tolerantly parse `{"scan": [[pos, "WT", ddg, func_loss], ...]}`."""
    n = len(target.positions_1idx)
    pred_ddg = np.zeros(n, dtype=np.float64)
    pred_func = np.zeros(n, dtype=np.float64)
    pos_to_idx = {p: idx for idx, p in enumerate(target.positions_1idx)}

    if not raw_text or not raw_text.strip():
        return AlanineScanParsed(pred_ddg, pred_func, 0, 0.0, False)

    # Strip optional markdown code fences
    cleaned = raw_text.strip()
    fence_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", cleaned, flags=re.DOTALL)
    if fence_match:
        cleaned = fence_match.group(1)
    else:
        brace_start = cleaned.find("{")
        brace_end = cleaned.rfind("}")
        if brace_start != -1 and brace_end > brace_start:
            cleaned = cleaned[brace_start : brace_end + 1]

    rows: Optional[List[Any]] = None
    try:
        obj = json.loads(cleaned)
        if isinstance(obj, dict):
            rows = obj.get("scan") or obj.get("predictions") or obj.get("mutations")
        elif isinstance(obj, list):
            rows = obj
    except json.JSONDecodeError:
        # Fallback regex recovery of `[pos, "X", ddg, func]` tuples even if output truncated!
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
        parse_ok=(n_matched >= max(5, int(0.5 * n))),
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
    # Average ties
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


def compute_aa_demeaned_spearman(
    wt_aas: Sequence[str],
    exp_vals: np.ndarray,
    pred_vals: np.ndarray,
) -> float:
    """Residualize BOTH `exp_vals` and `pred_vals` by wild-type amino acid identity within-protein
    (or across repeated amino acids) so static 20-AA hydrophobicity tables score ~0.00!
    """
    exp_res = exp_vals.copy()
    pred_res = pred_vals.copy()
    by_aa: Dict[str, List[int]] = {}
    for idx, aa in enumerate(wt_aas):
        by_aa.setdefault(aa, []).append(idx)

    valid_idx: List[int] = []
    for _aa, idxs in by_aa.items():
        if len(idxs) >= 2:
            arr_i = np.array(idxs, dtype=np.int32)
            exp_res[arr_i] -= float(np.mean(exp_vals[arr_i]))
            pred_res[arr_i] -= float(np.mean(pred_vals[arr_i]))
            valid_idx.extend(idxs)

    if len(valid_idx) < 6:
        return 0.0
    vi = np.array(sorted(valid_idx), dtype=np.int32)
    if np.std(exp_res[vi]) < 1e-9 or np.std(pred_res[vi]) < 1e-9:
        return 0.0
    rho, _ = spearmanr(exp_res[vi], pred_res[vi])
    return 0.0 if np.isnan(rho) else round(float(rho), 4)


def grade_alanine_scan(target: AlanineScanTarget, parsed: AlanineScanParsed) -> AlanineScanGrade:
    """Compute Spearman ρ, AA-demeaned context ρ, hotspot AUROC, 3D core AUROC, and func-vs-ddg AUROC."""
    n = len(target.positions_1idx)
    exp = np.asarray(target.exp_destab_scores, dtype=np.float64)
    # Choose primary prediction channel matching target's primary experimental assay
    pred = parsed.pred_func_loss if target.assay_type == "dms_activity" else parsed.pred_ddg

    if np.std(pred) < 1e-9 or np.std(exp) < 1e-9:
        rho_val = 0.0
        pr_val = 0.0
    else:
        r_s, _ = spearmanr(exp, pred)
        r_p, _ = pearsonr(exp, pred)
        rho_val = 0.0 if np.isnan(r_s) else float(r_s)
        pr_val = 0.0 if np.isnan(r_p) else float(r_p)

    rho_ctx = compute_aa_demeaned_spearman(target.wt_aas, exp, pred)

    # Top-20% most deleterious / destabilizing experimental positions AUROC
    thresh_80 = float(np.percentile(exp, 80.0))
    hot_labels = [1 if v >= thresh_80 else 0 for v in exp]
    hot_auc = binary_auroc(hot_labels, pred) or 0.5

    core_auc: Optional[float] = None
    if target.is_core_flags is not None and len(target.is_core_flags) == n:
        core_auc = binary_auroc(target.is_core_flags, parsed.pred_ddg)

    func_only_auc: Optional[float] = None
    if target.exp_func_loss_scores is not None and len(target.exp_func_loss_scores) == n:
        exp_fn = np.asarray(target.exp_func_loss_scores, dtype=np.float64)
        # Functional-only hotspots: stable on folding (`exp <= median`), top-25% functional loss!
        med_ddg = float(np.median(exp))
        fn_cut = float(np.percentile(exp_fn, 75.0))
        fn_only_labels = [1 if (e_d <= med_ddg and e_f >= fn_cut) else 0 for e_d, e_f in zip(exp, exp_fn)]
        # Score by standardized `pred_func_loss - pred_ddg` excess functional sensitivity
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
        spearman_rho_aa_demeaned=rho_ctx,
        hotspot_auroc_top20pct=hot_auc,
        core_burial_auroc=core_auc,
        func_only_hotspot_auroc=func_only_auc,
    )
