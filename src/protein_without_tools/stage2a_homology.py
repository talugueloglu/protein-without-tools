"""Stage 2A: Remote Homology Detection Ladder (`BLOSUM62` -> `Foldseek 3Di` -> `PickyBinders/tea`).

Implements:
  - Fast vectorized Smith-Waterman local alignment for both `BLOSUM62` (standard amino acids)
    and **`PickyBinders/tea`'s `MATCHA` substitution matrix** (TEA 20-letter representation alphabet),
    complete with Karlin-Altschul bit-scores & E-values.
  - Zero-leakage pairwise remote-homology prompts & fault-tolerant JSON parsers.
  - Stratified AUROC, AUPRC, and Rung-wise Recall @ 10% False Positive Rate (calibrated on
    length- and SCOPe-class-matched hard negative fold decoys!).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import math
import re
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from protein_without_tools.alanine_scan import binary_auroc
from protein_without_tools.pdb_utils import CANONICAL_AA_SET

AA_ORDER = "ARNDCQEGHILKMFPSTWYV"
AA_TO_IDX = {c: i for i, c in enumerate(AA_ORDER)}

# Standard NCBI BLOSUM62 20x20 matrix in `ARNDCQEGHILKMFPSTWYV` order (ungapped lambda=0.267, K=0.041)
BLOSUM62_MAT = np.array(
    [
        [ 4, -1, -2, -2,  0, -1, -1,  0, -2, -1, -1, -1, -1, -2, -1,  1,  0, -3, -2,  0],
        [-1,  5,  0, -2, -3,  1,  0, -2,  0, -3, -2,  2, -1, -3, -2, -1, -1, -3, -2, -3],
        [-2,  0,  6,  1, -3,  0,  0,  0,  1, -3, -3,  0, -2, -3, -2,  1,  0, -4, -2, -3],
        [-2, -2,  1,  6, -3,  0,  2, -1, -1, -3, -4, -1, -3, -3, -1,  0, -1, -4, -3, -3],
        [ 0, -3, -3, -3,  9, -3, -4, -3, -3, -1, -1, -3, -1, -2, -3, -1, -1, -2, -2, -1],
        [-1,  1,  0,  0, -3,  5,  2, -2,  0, -3, -2,  1,  0, -3, -1,  0, -1, -2, -1, -2],
        [-1,  0,  0,  2, -4,  2,  5, -2,  0, -3, -3,  1, -2, -3, -1,  0, -1, -3, -2, -2],
        [ 0, -2,  0, -1, -3, -2, -2,  6, -2, -4, -4, -2, -3, -3, -2,  0, -2, -2, -3, -3],
        [-2,  0,  1, -1, -3,  0,  0, -2,  8, -3, -3, -1, -2, -1, -2, -1, -2, -2,  2, -3],
        [-1, -3, -3, -3, -1, -3, -3, -4, -3,  4,  2, -3,  1,  0, -3, -2, -1, -3, -1,  3],
        [-1, -2, -3, -4, -1, -2, -3, -4, -3,  2,  4, -2,  2,  0, -3, -2, -1, -2, -1,  1],
        [-1,  2,  0, -1, -3,  1,  1, -2, -1, -3, -2,  5, -1, -3, -1,  0, -1, -3, -2, -2],
        [-1, -1, -2, -3, -1,  0, -2, -3, -2,  1,  2, -1,  5,  0, -2, -1, -1, -1, -1,  1],
        [-2, -3, -3, -3, -2, -3, -3, -3, -1,  0,  0, -3,  0,  6, -4, -2, -2,  1,  3, -1],
        [-1, -2, -2, -1, -3, -1, -1, -2, -2, -3, -3, -1, -2, -4,  7, -1, -1, -4, -3, -2],
        [ 1, -1,  1,  0, -1,  0,  0,  0, -1, -2, -2,  0, -1, -2, -1,  4,  1, -3, -2, -2],
        [ 0, -1,  0, -1, -1, -1, -1, -2, -2, -1, -1, -1, -1, -2, -1,  1,  5, -2, -2,  0],
        [-3, -3, -4, -4, -2, -2, -3, -2, -2, -3, -2, -3, -1,  1, -4, -3, -2, 11,  2, -3],
        [-2, -2, -2, -3, -2, -1, -2, -3,  2, -1, -1, -2, -1,  3, -3, -2, -2,  2,  7, -1],
        [ 0, -3, -3, -3, -1, -2, -2, -3, -3,  3,  1, -2,  1, -1, -2, -2,  0, -3, -1,  4],
    ],
    dtype=np.float64,
)


@dataclass
class HomologyPair:
    """Single SCOPe pair in the Stage 2A Remote Homology Ladder."""

    pair_id: str
    rung: str  # "1_seq_easy" | "2_foldseek_3di" | "3_deep_tea_zone" | "4_hard_negative_decoy"
    label_homolog: int  # 1 = same SCOPe superfamily, 0 = different SCOPe fold (same class!)
    seq_a: str
    seq_b: str
    scop_a: str
    scop_b: str
    domain_a: str
    domain_b: str
    seq_identity_pct: float
    blosum_bitscore: float
    blosum_evalue: float


@dataclass
class HomologyPredParsed:
    """Parsed LLM remote-homology probability & fold-relationship judgment."""

    prob_homolog: float
    predicted_same_superfamily: bool
    parse_ok: bool


def parse_substitution_matrix_file(text: str) -> Tuple[str, np.ndarray, float]:
    """Parse NCBI/MMseqs2-formatted substitution matrix (such as `PickyBinders/tea`'s `matcha.out`)."""
    lam = 0.3217
    alphabet = ""
    rows: List[List[float]] = []
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if line.startswith("#"):
            m_lam = re.search(r"Lambda.*?:\s*([0-9.]+)", line, flags=re.IGNORECASE)
            if m_lam:
                lam = float(m_lam.group(1))
            continue
        toks = line.split()
        if not alphabet and all(len(t) == 1 and t.isalpha() for t in toks):
            alphabet = "".join(t.upper() for t in toks)
            continue
        if len(toks) >= len(alphabet) + 1 and len(toks[0]) == 1:
            vals = [float(x) for x in toks[1 : len(alphabet) + 1]]
            rows.append(vals)
    mat = np.array(rows, dtype=np.float64)
    return alphabet, mat, lam


def smith_waterman_affine(
    seq_a: str,
    seq_b: str,
    alphabet: str = AA_ORDER,
    sub_matrix: np.ndarray = BLOSUM62_MAT,
    gap_open: float = 11.0,
    gap_extend: float = 1.0,
    karlin_lambda: float = 0.267,
    karlin_k: float = 0.041,
    db_residues: float = 3_000_000.0,
) -> Tuple[float, float, float]:
    """Exact Gotoh affine-gap Smith-Waterman alignment returning `(bitscore, evalue, seqid_pct)`."""
    a_up = seq_a.upper()
    b_up = seq_b.upper()
    la, lb = len(a_up), len(b_up)
    if la == 0 or lb == 0:
        return 0.0, 999.0, 0.0

    c2i = {c: idx for idx, c in enumerate(alphabet)}
    unk_idx = c2i.get("X", 0)
    ia = np.array([c2i.get(c, unk_idx) for c in a_up], dtype=np.int32)
    ib = np.array([c2i.get(c, unk_idx) for c in b_up], dtype=np.int32)

    # Row-by-row Gotoh DP (vectorized inner preparations; exact local score + aligned identity estimate)
    h_prev = np.zeros(lb + 1, dtype=np.float64)
    f_prev = np.full(lb + 1, -1e9, dtype=np.float64)
    best_raw = 0.0

    for i in range(1, la + 1):
        scores_row = sub_matrix[ia[i - 1], ib]  # (lb,)
        h_cur = np.zeros(lb + 1, dtype=np.float64)
        f_cur = np.maximum(h_prev[1:] - gap_open, f_prev[1:] - gap_extend)
        e_val = -1e9
        diag_vec = h_prev[:-1] + scores_row
        for j in range(1, lb + 1):
            e_val = max(h_cur[j - 1] - gap_open, e_val - gap_extend)
            v = max(0.0, float(diag_vec[j - 1]), float(f_cur[j - 1]), e_val)
            h_cur[j] = v
            if v > best_raw:
                best_raw = v
        h_prev = h_cur
        f_prev[1:] = f_cur

    bitscore = (karlin_lambda * best_raw - math.log(karlin_k)) / math.log(2.0)
    bitscore = max(0.0, bitscore)
    evalue = float(la * db_residues * (2.0 ** (-bitscore)))

    # Quick ungapped/semi-global identity % for diagnostic reporting
    # (exact matches along best diagonal offset as fast lower-overhead companion)
    best_id = 0.0
    for shift in range(-lb + 10, la - 10):
        a_s = max(0, shift)
        b_s = max(0, -shift)
        ov = min(la - a_s, lb - b_s)
        if ov >= 12:
            matches = int(np.sum(ia[a_s : a_s + ov] == ib[b_s : b_s + ov]))
            pct = 100.0 * matches / float(min(la, lb))
            if pct > best_id:
                best_id = pct

    return round(bitscore, 2), round(min(999.0, evalue), 6), round(best_id, 1)


def build_remote_homology_prompt(pair: HomologyPair) -> str:
    """Build a zero-leakage pairwise remote homology prompt."""
    sa = pair.seq_a.strip().upper()
    sb = pair.seq_b.strip().upper()
    for s in (sa, sb):
        if any(c not in CANONICAL_AA_SET for c in s):
            raise ValueError(f"Non-canonical residue in {pair.pair_id}")

    prompt = (
        "You are evaluating protein remote homology and 3D fold superfamily membership from primary sequence alone.\n\n"
        f"Protein A (length {len(sa)}):\n{sa}\n\n"
        f"Protein B (length {len(sb)}):\n{sb}\n\n"
        "Task:\n"
        "Determine whether Protein A and Protein B belong to the SAME evolutionary superfamily "
        "(sharing a homologous core 3D fold topology in SCOPe/ECOD, even if sequence identity is in the <15% twilight zone), "
        "or whether they adopt DIFFERENT, non-homologous 3D folds.\n\n"
        "Return ONLY a single compact JSON object with exact schema:\n"
        '{"same_superfamily": true_or_false, "probability": 0.0_to_1.0, "predicted_fold_class": "brief fold guess"}\n'
        "where `probability` is your calibrated confidence in [0.0, 1.0] that A and B share the same evolutionary superfamily."
    )
    assert pair.domain_a not in prompt and pair.domain_b not in prompt and pair.scop_a not in prompt
    return prompt


def parse_remote_homology_response(raw_text: str) -> HomologyPredParsed:
    if not raw_text or not raw_text.strip():
        return HomologyPredParsed(0.5, False, False)

    cleaned = raw_text.strip()
    m_f = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", cleaned, flags=re.DOTALL)
    if m_f:
        cleaned = m_f.group(1)
    else:
        i0, i1 = cleaned.find("{"), cleaned.rfind("}")
        if i0 != -1 and i1 > i0:
            cleaned = cleaned[i0 : i1 + 1]

    try:
        obj = json.loads(cleaned)
        prob = float(obj.get("probability", obj.get("prob", obj.get("score", 0.5))))
        prob = float(np.clip(prob, 0.0, 1.0))
        same = obj.get("same_superfamily")
        if isinstance(same, bool):
            pred_b = same
        else:
            pred_b = prob >= 0.5
        return HomologyPredParsed(round(prob, 4), pred_b, True)
    except Exception:
        m_p = re.search(r'"probability"\s*:\s*([0-9]*\.?[0-9]+)', raw_text)
        if m_p:
            p = float(np.clip(float(m_p.group(1)), 0.0, 1.0))
            return HomologyPredParsed(round(p, 4), p >= 0.5, True)
        return HomologyPredParsed(0.5, False, False)


def compute_auprc(labels_01: Sequence[int], scores: Sequence[float]) -> float:
    y = np.asarray(labels_01, dtype=np.int32)
    s = np.asarray(scores, dtype=np.float64)
    n_pos = int(np.sum(y == 1))
    if n_pos == 0:
        return 0.0
    order = np.argsort(-s, kind="mergesort")
    y_sorted = y[order]
    tp = np.cumsum(y_sorted == 1)
    fp = np.cumsum(y_sorted == 0)
    prec = tp / np.maximum(1, tp + fp)
    rec = tp / float(n_pos)
    # Stepwise average precision
    ap = float(np.sum(prec[y_sorted == 1]) / float(n_pos))
    _ = rec
    return round(ap, 4)


def evaluate_homology_predictions(
    pairs: Sequence[HomologyPair],
    scores: Sequence[float],
) -> Dict[str, Any]:
    """Compute overall AUROC, AUPRC, and per-rung AUROC & Recall @ 10% FPR against hard fold decoys."""
    y_all = [p.label_homolog for p in pairs]
    s_all = [float(x) for x in scores]

    overall_auc = binary_auroc(y_all, s_all) or 0.5
    overall_ap = compute_auprc(y_all, s_all)

    neg_scores = sorted([s for p, s in zip(pairs, s_all) if p.label_homolog == 0], reverse=True)
    # 10% False Positive Rate threshold on hard negative decoys
    if neg_scores:
        k_fpr10 = max(0, int(math.floor(0.10 * len(neg_scores))) - 1)
        thresh_fpr10 = neg_scores[k_fpr10]
    else:
        thresh_fpr10 = 0.5

    by_rung: Dict[str, Any] = {}
    for rung_name in ("1_seq_easy", "2_foldseek_3di", "3_deep_tea_zone"):
        pos_idx = [i for i, p in enumerate(pairs) if p.rung == rung_name and p.label_homolog == 1]
        neg_idx = [i for i, p in enumerate(pairs) if p.label_homolog == 0]
        if not pos_idx:
            continue
        sub_y = [1] * len(pos_idx) + [0] * len(neg_idx)
        sub_s = [s_all[i] for i in pos_idx] + [s_all[i] for i in neg_idx]
        r_auc = binary_auroc(sub_y, sub_s) or 0.5
        rec_10 = float(np.mean([1.0 if s_all[i] > thresh_fpr10 else 0.0 for i in pos_idx]))
        by_rung[rung_name] = {
            "n_pos": len(pos_idx),
            "n_neg": len(neg_idx),
            "auroc_vs_hard_decoys": r_auc,
            "recall_at_fpr10": round(rec_10, 4),
        }

    return {
        "n_pairs": len(pairs),
        "overall_auroc": overall_auc,
        "overall_auprc": overall_ap,
        "fpr10_threshold": round(float(thresh_fpr10), 4),
        "by_rung": by_rung,
    }
