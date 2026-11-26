"""Task 11: Conformational Dynamics, Apo-Holo Mobility & Fold-Switching Benchmark.

Tests whether frontier reasoning models (without tools) and conformational ensemble
specialists (BioEmu) can predict which internal residues undergo large structural
fluctuations (MD RMSF) or ligand-induced conformational rearrangements (apo vs holo),
while explicitly neutralizing 1D disorder/termini shortcuts:

  1. N- and C-terminal tails (first/last 6 residues) are excluded.
  2. Every mobile / cryptic-motion residue is paired 1-to-1 with a rigid decoy
     matched on 1D disorder propensity (TOP-IDP scale) AND same-AA where possible,
     holding 1D disorder heuristics (IUPred-like / hydropathy / terminus distance)
     near 50% win rate.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
import re
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy.stats import spearmanr

from protein_without_tools.alanine_scan import binary_auroc, compute_ndcg_at_k
from protein_without_tools.task10_ligand_site import parse_site_ranking_response

# Campen et al. (2008) TOP-IDP amino-acid disorder propensity scale (higher = more disorder-promoting)
TOP_IDP: Dict[str, float] = {
    "W": -0.884, "F": -0.697, "Y": -0.510, "I": -0.486, "M": -0.397,
    "L": -0.326, "V": -0.121, "N": 0.007, "C": 0.020, "T": 0.059,
    "A": 0.060, "G": 0.166, "R": 0.180, "D": 0.192, "H": 0.303,
    "Q": 0.318, "K": 0.586, "S": 0.341, "E": 0.736, "P": 0.987,
}


@dataclass(frozen=True)
class DynamicsResidue:
    res_id: str         # canonical "G45" (1-indexed)
    aa: str
    pos: int            # 1-indexed
    mobility: float     # RMSF (A) or apo-holo local Ca displacement (A)
    local_idp: float    # 7-residue window mean TOP-IDP disorder score
    is_mobile: bool = False


def compute_local_idp(sequence: str, window: int = 7) -> List[float]:
    L = len(sequence)
    half = window // 2
    vals = [TOP_IDP.get(aa, 0.0) for aa in sequence]
    out: List[float] = []
    for i in range(L):
        lo, hi = max(0, i - half), min(L, i + half + 1)
        out.append(float(np.mean(vals[lo:hi])))
    return out


def select_disorder_matched_dynamics_panel(
    sequence: str,
    mobility_per_res: Sequence[float],
    n_total: int = 16,
    terminal_trim: int = 6,
    min_sep: int = 4,
) -> Optional[List[DynamicsResidue]]:
    """Select `n_total // 2` high-mobility internal residues paired 1-to-1 with rigid
    decoys matched on 1D TOP-IDP disorder propensity and residue identity (when available).
    """
    L = len(sequence)
    if L != len(mobility_per_res) or L < 45:
        return None
    half = n_total // 2
    local_idp = compute_local_idp(sequence, window=7)

    internal = [
        DynamicsResidue(
            res_id=f"{sequence[i]}{i + 1}",
            aa=sequence[i],
            pos=i + 1,
            mobility=round(float(mobility_per_res[i]), 3),
            local_idp=round(local_idp[i], 3),
        )
        for i in range(terminal_trim, L - terminal_trim)
        if not math.isnan(mobility_per_res[i])
    ]
    if len(internal) < n_total * 2:
        return None

    mobs = np.array([r.mobility for r in internal], dtype=float)
    hi_cut = float(np.percentile(mobs, 75.0))
    lo_cut = float(np.percentile(mobs, 35.0))
    if hi_cut - lo_cut < 0.45:  # require genuine dynamic contrast (>= 0.45 A)
        return None

    mobile_pool = sorted([r for r in internal if r.mobility >= hi_cut], key=lambda r: -r.mobility)
    rigid_pool = [r for r in internal if r.mobility <= lo_cut]
    if len(mobile_pool) < half or len(rigid_pool) < half:
        return None

    # Greedily pick well-spaced mobile residues (>= min_sep apart so we don't pick 8 consecutive loop residues)
    chosen_mob: List[DynamicsResidue] = []
    for cand in mobile_pool:
        if len(chosen_mob) >= half:
            break
        if all(abs(cand.pos - p.pos) >= min_sep for p in chosen_mob):
            chosen_mob.append(cand)
    if len(chosen_mob) < half:
        for cand in mobile_pool:
            if len(chosen_mob) >= half:
                break
            if cand.res_id not in {x.res_id for x in chosen_mob}:
                chosen_mob.append(cand)
    if len(chosen_mob) < half:
        return None

    # Match each mobile residue to a rigid decoy minimizing 1D disorder propensity diff + terminus-dist diff
    chosen_rig: List[DynamicsResidue] = []
    avail = list(rigid_pool)
    for m in chosen_mob:
        m_term = min(m.pos - 1, L - m.pos)
        best_i, best_cost = None, float("inf")
        for idx, r in enumerate(avail):
            if any(abs(r.pos - prev.pos) < 3 for prev in chosen_rig):
                continue
            r_term = min(r.pos - 1, L - r.pos)
            idp_cost = abs(m.local_idp - r.local_idp) * 3.5
            aa_bonus = -0.35 if r.aa == m.aa else 0.0
            term_cost = abs(m_term - r_term) / max(12.0, L * 0.25)
            cost = idp_cost + term_cost + aa_bonus
            if cost < best_cost:
                best_cost = cost
                best_i = idx
        if best_i is None and avail:
            best_i = 0
        if best_i is None:
            return None
        chosen_rig.append(avail.pop(best_i))

    panel = [
        DynamicsResidue(r.res_id, r.aa, r.pos, r.mobility, r.local_idp, is_mobile=True)
        for r in chosen_mob
    ] + [
        DynamicsResidue(r.res_id, r.aa, r.pos, r.mobility, r.local_idp, is_mobile=False)
        for r in chosen_rig
    ]
    panel.sort(key=lambda r: r.pos)
    return panel


def build_dynamics_prompt(
    sequence: str,
    stratum: str,
    panel: Sequence[DynamicsResidue],
) -> str:
    """Build a zero-leakage prompt for conformational flexibility / apo-holo motion ranking."""
    L = len(sequence)
    n = len(panel)
    toks = [f'"{r.res_id}"' for r in panel]
    if stratum == "apo_holo_motion":
        phenom = (
            "large conformational rearrangement (backbone C-alpha displacement between unbound apo "
            "and ligand/partner-bound holo crystal states)"
        )
    else:
        phenom = (
            "high equilibrium backbone conformational flexibility (all-atom molecular dynamics "
            "root-mean-square fluctuation, RMSF)"
        )
    ex1 = panel[0].res_id if panel else "G24"
    ex2 = panel[1].res_id if len(panel) > 1 else "L58"
    return (
        f"You are predicting 3D conformational dynamics of an anonymized monomeric protein "
        f"from its primary amino-acid sequence alone.\n\n"
        f"Protein Sequence (1-indexed, length L = {L}):\n{sequence}\n\n"
        f"Candidate Internal Residues ({n} total, N/C-terminal tails excluded, ascending order):\n"
        f"[{', '.join(toks)}]\n\n"
        f"Task:\n"
        f"Rank ALL {n} candidate residues from #1 (HIGHEST conformational mobility — {phenom}) "
        f"down to #{n} (MOST structurally rigid core/scaffold position).\n"
        f"Note: mobile residues and rigid decoys are matched on local 1D disorder propensity and "
        f"distance from the chain termini, so rank by 3D fold packing, hinge/loop architecture, "
        f"and domain-motion physics rather than 1D disorder heuristics.\n\n"
        f"Return ONLY a single compact JSON object with exact schema:\n"
        f'{{"ranking": ["{ex1}", "{ex2}", ...]}}\n'
        f"ordered strictly from #1 (most mobile) to #{n} (most rigid), containing all {n} residues "
        f"with no text outside JSON."
    )


def grade_dynamics_panel(
    panel: Sequence[DynamicsResidue],
    predicted_ranking: Sequence[str],
    predicted_scores: Optional[Dict[str, float]] = None,
) -> Dict[str, Any]:
    by_id = {r.res_id: r for r in panel}
    n = len(panel)
    if predicted_scores is not None:
        scores = {rid: float(predicted_scores.get(rid, 0.0)) for rid in by_id}
    else:
        scores = {}
        seen = set()
        for idx, rid in enumerate(predicted_ranking):
            if rid in by_id and rid not in seen:
                scores[rid] = float(n - idx)
                seen.add(rid)
        for rid in by_id:
            if rid not in seen:
                scores[rid] = 0.0

    rids = list(by_id.keys())
    y_mob = np.array([by_id[r].mobility for r in rids], dtype=float)
    y_pred = np.array([scores[r] for r in rids], dtype=float)
    y_bin = [1 if by_id[r].is_mobile else 0 for r in rids]

    auroc = float(binary_auroc(y_bin, [scores[r] for r in rids]))
    ndcg5 = float(compute_ndcg_at_k(y_mob, y_pred, k=5))
    if np.std(y_pred) < 1e-12 or np.std(y_mob) < 1e-12:
        rho = 0.0
    else:
        rv = spearmanr(y_pred, y_mob).statistic
        rho = 0.0 if (rv is None or math.isnan(rv)) else float(rv)

    return {
        "n_candidates": n,
        "coverage": len([r for r in predicted_ranking if r in by_id]) / float(max(1, n)),
        "auroc": round(auroc, 4),
        "spearman_rho": round(rho, 4),
        "ndcg_at_5": round(ndcg5, 4),
    }


__all__ = [
    "TOP_IDP",
    "DynamicsResidue",
    "build_dynamics_prompt",
    "compute_local_idp",
    "grade_dynamics_panel",
    "parse_site_ranking_response",
    "select_disorder_matched_dynamics_panel",
]
