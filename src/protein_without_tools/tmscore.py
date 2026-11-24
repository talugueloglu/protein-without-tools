"""Exact Zhang & Skolnick (2004) TM-score, Kabsch Cα-RMSD, and Cα-lDDT metrics."""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass
class StructuralAlignmentResult:
    """Metrics comparing a predicted Cα trace against ground-truth Cα coordinates."""

    tm_score: float
    tm_score_mirror_best: float
    rmsd_global: float
    lddt_ca: float
    aligned_pred_ca: np.ndarray


def tm_d0(length: int) -> float:
    """Official Zhang & Skolnick length-dependent normalization scale d0(L)."""
    if length <= 21:
        return 0.5
    val = 1.24 * ((length - 15.0) ** (1.0 / 3.0)) - 1.8
    return max(0.5, float(val))


def kabsch_superpose(mobile: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    """Find optimal proper rotation R in SO(3) and translation t minimizing ||mobile @ R + t - target||."""
    mu_m = mobile.mean(axis=0)
    mu_t = target.mean(axis=0)
    xm = mobile - mu_m
    xt = target - mu_t
    cov = xm.T @ xt
    u, _s, vt = np.linalg.svd(cov)
    d = np.linalg.det(u @ vt)
    if d < 0.0:
        u[:, -1] *= -1.0
    rot = u @ vt
    trans = mu_t - mu_m @ rot
    diff = (mobile @ rot + trans) - target
    rmsd = float(np.sqrt(np.mean(np.sum(diff * diff, axis=1))))
    return rot, trans, rmsd


def compute_tm_score_so3(pred_ca: np.ndarray, true_ca: np.ndarray) -> tuple[float, np.ndarray]:
    """Fast iterative fragment-seeded Zhang-Skolnick TM-score maximization over SO(3)."""
    L = true_ca.shape[0]
    if pred_ca.shape != true_ca.shape:
        raise ValueError(f"Shape mismatch: {pred_ca.shape} vs {true_ca.shape}")
    if L < 4:
        _, _, rmsd = kabsch_superpose(pred_ca, true_ca)
        return float(1.0 / (1.0 + (rmsd / 0.5) ** 2)), pred_ca.copy()

    d0 = tm_d0(L)
    d0_sq = d0 * d0
    search_cutoffs_sq = [(4.0 * d0) ** 2, (2.0 * d0) ** 2, (1.0 * d0) ** 2]

    best_tm = -1.0
    best_aligned = pred_ca.copy()

    frag_lengths = sorted({L, max(4, L // 2), max(4, L // 4), min(L, 6)}, reverse=True)
    for l_init in frag_lengths:
        stride = max(2, l_init // 2) if l_init < L else 1
        starts = list(range(0, L - l_init + 1, stride))
        if (L - l_init) not in starts:
            starts.append(L - l_init)

        for s_idx in starts:
            init_sub = np.arange(s_idx, s_idx + l_init)
            for d_cut_sq in search_cutoffs_sq:
                cur_sub = init_sub
                for _iter in range(8):
                    if cur_sub.size < 3:
                        break
                    rot, trans, _ = kabsch_superpose(pred_ca[cur_sub], true_ca[cur_sub])
                    aligned = pred_ca @ rot + trans
                    dists_sq = np.sum((aligned - true_ca) ** 2, axis=1)
                    tm_val = float(np.mean(1.0 / (1.0 + dists_sq / d0_sq)))
                    if tm_val > best_tm:
                        best_tm = tm_val
                        best_aligned = aligned
                    new_sub = np.flatnonzero(dists_sq < d_cut_sq)
                    if new_sub.size < 3:
                        new_sub = np.argsort(dists_sq)[:4]
                    if np.array_equal(new_sub, cur_sub):
                        break
                    cur_sub = new_sub

    return best_tm, best_aligned


def compute_lddt_ca(pred_ca: np.ndarray, true_ca: np.ndarray, inclusion_radius: float = 15.0) -> float:
    """Superposition-free Cα-lDDT score (Mariani et al., 2013) in [0, 1]."""
    L = true_ca.shape[0]
    d_true = np.linalg.norm(true_ca[:, None, :] - true_ca[None, :, :], axis=-1)
    d_pred = np.linalg.norm(pred_ca[:, None, :] - pred_ca[None, :, :], axis=-1)

    idx_i, idx_j = np.triu_indices(L, k=1)
    mask = d_true[idx_i, idx_j] < inclusion_radius
    if not np.any(mask):
        return 0.0
    diffs = np.abs(d_true[idx_i[mask], idx_j[mask]] - d_pred[idx_i[mask], idx_j[mask]])
    return sum(float(np.mean(diffs < thr)) for thr in (0.5, 1.0, 2.0, 4.0)) / 4.0


def evaluate_3d_superposition(pred_ca: np.ndarray, true_ca: np.ndarray) -> StructuralAlignmentResult:
    """Compute proper SO(3) TM-score, mirror-checked TM-score, global RMSD, and Cα-lDDT."""
    _, _, rmsd_global = kabsch_superpose(pred_ca, true_ca)
    tm_proper, best_aligned = compute_tm_score_so3(pred_ca, true_ca)
    reflected = pred_ca * np.array([-1.0, 1.0, 1.0])
    tm_reflected, _ = compute_tm_score_so3(reflected, true_ca)
    lddt = compute_lddt_ca(pred_ca, true_ca)

    return StructuralAlignmentResult(
        tm_score=tm_proper,
        tm_score_mirror_best=max(tm_proper, tm_reflected),
        rmsd_global=rmsd_global,
        lddt_ca=lddt,
        aligned_pred_ca=best_aligned,
    )


def _nw_align_score_matrix(score_mat: np.ndarray, gap_open: float = -0.6) -> list[tuple[int, int]]:
    """Standard TM-align Needleman-Wunsch DP over `score_mat` (L1, L2) with linear/open gap penalty."""
    n, m = score_mat.shape
    dp = np.zeros((n + 1, m + 1), dtype=np.float64)
    ptr = np.zeros((n + 1, m + 1), dtype=np.uint8)  # 1: diag, 2: up, 3: left
    for i in range(1, n + 1):
        si = score_mat[i - 1]
        dpi = dp[i]
        dp_prev = dp[i - 1]
        ptri = ptr[i]
        for j in range(1, m + 1):
            v_diag = dp_prev[j - 1] + si[j - 1]
            v_up = dp_prev[j] + gap_open
            v_left = dpi[j - 1] + gap_open
            if v_diag >= v_up and v_diag >= v_left:
                dpi[j] = v_diag
                ptri[j] = 1
            elif v_up >= v_left:
                dpi[j] = v_up
                ptri[j] = 2
            else:
                dpi[j] = v_left
                ptri[j] = 3
    pairs: list[tuple[int, int]] = []
    i, j = n, m
    while i > 0 and j > 0:
        p = ptr[i, j]
        if p == 1:
            pairs.append((i - 1, j - 1))
            i -= 1
            j -= 1
        elif p == 2:
            i -= 1
        else:
            j -= 1
    pairs.reverse()
    return pairs


def tm_align_ca(
    pred_ca: np.ndarray,
    true_ca: np.ndarray,
    norm_length: int | None = None,
    pred_q3: str = "",
    true_q3: str = "",
) -> tuple[float, float, int]:
    """Zhang-Skolnick iterative DP + SO(3) `TM-align` for equal or unequal chain lengths.

    Returns `(best_tm_score, aligned_rmsd, n_aligned)`.
    """
    p = np.asarray(pred_ca, dtype=np.float64)
    q = np.asarray(true_ca, dtype=np.float64)
    l1, l2 = len(p), len(q)
    if l1 < 4 or l2 < 4:
        return 0.0, 99.9, 0
    l_norm = int(norm_length) if norm_length is not None else min(l1, l2)
    d0 = tm_d0(l_norm)
    d0_sq = d0 * d0
    cutoffs_sq = [(4.0 * d0) ** 2, (2.0 * d0) ** 2, (1.0 * d0) ** 2]

    seeds: list[list[tuple[int, int]]] = []
    # 1. Gapless sliding seeds
    w = min(l1, l2)
    step = max(1, abs(l1 - l2) // 4 + 1)
    if l1 <= l2:
        for off in range(0, l2 - l1 + 1, step):
            seeds.append([(k, k + off) for k in range(l1)])
    else:
        for off in range(0, l1 - l2 + 1, step):
            seeds.append([(k + off, k) for k in range(l2)])
    # 2. Secondary-structure DP seed
    if len(pred_q3) == l1 and len(true_q3) == l2:
        ss_mat = np.array(
            [[1.0 if a == b else -0.2 for b in true_q3] for a in pred_q3],
            dtype=np.float64,
        )
        seeds.append(_nw_align_score_matrix(ss_mat, gap_open=-0.5))
    # 3. Fragment seeds (N-term, mid, C-term)
    frag = min(16, w)
    for p_off in (0, max(0, (l1 - frag) // 2), max(0, l1 - frag)):
        for q_off in (0, max(0, (l2 - frag) // 2), max(0, l2 - frag)):
            seeds.append([(p_off + k, q_off + k) for k in range(frag)])

    best_tm = 0.0
    best_rmsd = 99.9
    best_naln = 0

    for init_pairs in seeds:
        if len(init_pairs) < 4:
            continue
        pairs = init_pairs
        for _dp_iter in range(4):
            idx_p = np.array([a for a, _ in pairs], dtype=np.int64)
            idx_q = np.array([b for _, b in pairs], dtype=np.int64)
            if len(idx_p) < 4:
                break
            qp = p[idx_p]
            tp = q[idx_q]
            best_rot, best_trans, rmsd_cur = kabsch_superpose(qp, tp)
            best_iter_tm = -1.0
            for cut_sq in cutoffs_sq:
                cur_sub = np.arange(len(idx_p))
                for _ in range(5):
                    if len(cur_sub) < 4:
                        break
                    rot, trans, _ = kabsch_superpose(qp[cur_sub], tp[cur_sub])
                    diff = (qp @ rot + trans) - tp
                    dsq = np.sum(diff * diff, axis=1)
                    tm_v = float(np.sum(1.0 / (1.0 + dsq / d0_sq)) / float(l_norm))
                    if tm_v > best_iter_tm:
                        best_iter_tm = tm_v
                        best_rot, best_trans = rot, trans
                    nxt = np.flatnonzero(dsq < cut_sq)
                    if len(nxt) < 4:
                        nxt = np.argsort(dsq)[:4]
                    if np.array_equal(nxt, cur_sub):
                        break
                    cur_sub = nxt

            if best_iter_tm > best_tm:
                best_tm = best_iter_tm
                best_rmsd = rmsd_cur
                best_naln = len(idx_p)

            p_rot = p @ best_rot + best_trans
            d2_full = np.sum((p_rot[:, None, :] - q[None, :, :]) ** 2, axis=-1)
            score_mat = 1.0 / (1.0 + d2_full / d0_sq)
            new_pairs = _nw_align_score_matrix(score_mat, gap_open=-0.6)
            if new_pairs == pairs or len(new_pairs) < 4:
                break
            pairs = new_pairs

    return round(best_tm, 4), round(float(best_rmsd), 3), best_naln
