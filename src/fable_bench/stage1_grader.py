"""Official Stage 1A (Forward Folding) and Stage 1B (Inverse Folding) grading metrics."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from fable_bench.pdb_utils import ProteinStructure, write_ca_trace_pdb
from fable_bench.reconstruct_3d import reconstruct_ca_from_contacts
from fable_bench.stage1_prompts_and_parsers import (
    ForwardPredictionParsed,
    InversePredictionParsed,
)
from fable_bench.tmscore import evaluate_3d_superposition

PHYSICOCHEMICAL_GROUP: Dict[str, str] = {
    "A": "hydrophobic",
    "V": "hydrophobic",
    "I": "hydrophobic",
    "L": "hydrophobic",
    "M": "hydrophobic",
    "F": "aromatic",
    "Y": "aromatic",
    "W": "aromatic",
    "S": "polar",
    "T": "polar",
    "N": "polar",
    "Q": "polar",
    "C": "polar",
    "K": "positive",
    "R": "positive",
    "H": "positive",
    "D": "negative",
    "E": "negative",
    "G": "special_G",
    "P": "special_P",
}


@dataclass
class ForwardGradeResult:
    """Complete CASP/CAMEO metrics for Stage 1A Forward Folding on one target."""

    target_id: str
    length: int
    parse_ok: bool
    n_predicted_contacts: int
    q3_accuracy: float
    # Medium + Long range (|i - j| >= 6, matching @sokrypton's prompt)
    p_at_l_sep6: float
    p_at_l2_sep6: float
    p_at_l5_sep6: float
    # Medium + Long range (|i - j| >= 12)
    p_at_l_sep12: float
    p_at_l5_sep12: float
    # Strict CASP Long-Range Tertiary Contacts (|i - j| >= 24)
    p_at_l_sep24: float
    p_at_l2_sep24: float
    p_at_l5_sep24: float
    # Deterministic 3D Distance-Geometry Reconstruction Accuracy
    tm_score: float
    tm_score_mirror_best: float
    rmsd_ca: float
    lddt_ca: float

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class InverseGradeResult:
    """Complete ProteinMPNN/CATH metrics for Stage 1B Inverse Folding on one target."""

    target_id: str
    length: int
    parse_ok: bool
    length_exact_match: bool
    nsr_overall: float          # Native Sequence Recovery [0, 1]
    nsr_core: float             # Buried core (nbr10A >= 14) NSR [0, 1]
    nsr_surface: float          # Exposed surface (nbr10A < 14) NSR [0, 1]
    biochem_group_recovery: float
    seq_shannon_entropy_bits: float  # Catches degenerate homopolymer collapse (e.g. 'AAAA...')
    designed_sequence: str
    # Populated when Modal ESMFold / AlphaFold2 refolding oracle runs:
    sc_tm_score: Optional[float] = None
    sc_rmsd_ca: Optional[float] = None
    esmfold_plddt: Optional[float] = None

    def to_dict(self) -> dict:
        return asdict(self)


def _contact_precision_at_k(
    contacts: Sequence[Tuple[int, int, float]],
    cb_dist_matrix: np.ndarray,
    min_sep: int,
    k: int,
    cutoff_angstrom: float = 8.0,
) -> float:
    """Compute top-k contact precision for predicted pairs with |i - j| >= min_sep."""
    if k <= 0:
        return 0.0
    filtered = [(i, j) for (i, j, _c) in contacts if (j - i) >= min_sep]
    if not filtered:
        return 0.0
    top_k = filtered[:k]
    hits = sum(1 for (i, j) in top_k if cb_dist_matrix[i - 1, j - 1] < cutoff_angstrom)
    # Divide by min(k, total true contacts at min_sep) if structure has fewer than k true contacts,
    # else standard CASP denominator k.
    L = cb_dist_matrix.shape[0]
    ii, jj = np.triu_indices(L, k=min_sep)
    n_true = int(np.sum(cb_dist_matrix[ii, jj] < cutoff_angstrom))
    if n_true == 0:
        return 0.0
    denom = min(k, n_true)
    return float(hits) / float(denom)


def grade_forward_prediction(
    struct: ProteinStructure,
    pred: ForwardPredictionParsed,
    out_pdb_path: Optional[str | Path] = None,
    n_restarts: int = 4,
    seed: int = 42,
) -> ForwardGradeResult:
    """Grade Stage 1A contact map, Q3 accuracy, and reconstructed 3D Cα fold."""
    L = struct.length
    k_l = max(1, L)
    k_l2 = max(1, L // 2)
    k_l5 = max(1, L // 5)

    q3_matches = sum(1 for a, b in zip(pred.q3, struct.q3) if a == b)
    q3_acc = float(q3_matches) / float(L)

    p_l_6 = _contact_precision_at_k(pred.contacts, struct.cb_dist_matrix, 6, k_l)
    p_l2_6 = _contact_precision_at_k(pred.contacts, struct.cb_dist_matrix, 6, k_l2)
    p_l5_6 = _contact_precision_at_k(pred.contacts, struct.cb_dist_matrix, 6, k_l5)

    p_l_12 = _contact_precision_at_k(pred.contacts, struct.cb_dist_matrix, 12, k_l)
    p_l5_12 = _contact_precision_at_k(pred.contacts, struct.cb_dist_matrix, 12, k_l5)

    p_l_24 = _contact_precision_at_k(pred.contacts, struct.cb_dist_matrix, 24, k_l)
    p_l2_24 = _contact_precision_at_k(pred.contacts, struct.cb_dist_matrix, 24, k_l2)
    p_l5_24 = _contact_precision_at_k(pred.contacts, struct.cb_dist_matrix, 24, k_l5)

    pred_ca = reconstruct_ca_from_contacts(
        length=L,
        contacts=pred.contacts,
        q3=pred.q3,
        sequence=struct.sequence,
        n_restarts=n_restarts,
        seed=seed,
    )
    sup = evaluate_3d_superposition(pred_ca, struct.coords_ca)

    if out_pdb_path is not None:
        write_ca_trace_pdb(struct.sequence, sup.aligned_pred_ca, out_pdb_path)

    return ForwardGradeResult(
        target_id=struct.target_id,
        length=L,
        parse_ok=pred.parse_ok,
        n_predicted_contacts=len(pred.contacts),
        q3_accuracy=q3_acc,
        p_at_l_sep6=p_l_6,
        p_at_l2_sep6=p_l2_6,
        p_at_l5_sep6=p_l5_6,
        p_at_l_sep12=p_l_12,
        p_at_l5_sep12=p_l5_12,
        p_at_l_sep24=p_l_24,
        p_at_l2_sep24=p_l2_24,
        p_at_l5_sep24=p_l5_24,
        tm_score=sup.tm_score,
        tm_score_mirror_best=sup.tm_score_mirror_best,
        rmsd_ca=sup.rmsd_global,
        lddt_ca=sup.lddt_ca,
    )


def _sequence_entropy_bits(seq: str) -> float:
    """Compute 1-mer Shannon entropy in bits (max ~4.32 bits across 20 AAs)."""
    if not seq:
        return 0.0
    counts: Dict[str, int] = {}
    for ch in seq:
        counts[ch] = counts.get(ch, 0) + 1
    n = float(len(seq))
    return -sum((c / n) * np.log2(c / n) for c in counts.values())


def grade_inverse_prediction(
    struct: ProteinStructure,
    pred: InversePredictionParsed,
) -> InverseGradeResult:
    """Grade Stage 1B designed sequence on Native Sequence Recovery (overall, core, surface)."""
    L = struct.length
    true_seq = struct.sequence
    des_seq = pred.sequence

    exact_arr = np.array([a == b for a, b in zip(des_seq, true_seq)], dtype=bool)
    nsr_all = float(np.mean(exact_arr))

    core_mask = struct.is_core_mask
    surf_mask = ~core_mask
    nsr_core = float(np.mean(exact_arr[core_mask])) if np.any(core_mask) else nsr_all
    nsr_surf = float(np.mean(exact_arr[surf_mask])) if np.any(surf_mask) else nsr_all

    grp_matches = sum(
        1
        for a, b in zip(des_seq, true_seq)
        if PHYSICOCHEMICAL_GROUP.get(a, "x") == PHYSICOCHEMICAL_GROUP.get(b, "y")
    )
    grp_rec = float(grp_matches) / float(L)

    return InverseGradeResult(
        target_id=struct.target_id,
        length=L,
        parse_ok=pred.parse_ok,
        length_exact_match=pred.length_exact_match,
        nsr_overall=nsr_all,
        nsr_core=nsr_core,
        nsr_surface=nsr_surf,
        biochem_group_recovery=grp_rec,
        seq_shannon_entropy_bits=float(_sequence_entropy_bits(des_seq)),
        designed_sequence=des_seq,
    )
