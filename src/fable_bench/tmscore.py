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
