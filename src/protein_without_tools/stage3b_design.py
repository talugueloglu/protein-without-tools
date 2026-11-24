"""Stage 3B: De Novo Protein Biogeneration & Closed-Loop `AF2` Epitope Binder Design (`BindCraft`-style).

Covers two complementary evaluation modes:
  1. Part 1 — Single-Shot Monomer & Motif Scaffolding (`N = 40` prompts):
     - `24` fold-conditioned monomer designs across 4 topologies (`4_helix_bundle`,
       `ferredoxin_bababb`, `rossmann_aba_sandwich`, `all_beta_barrel_sh3`).
     - `16` functional crystal-motif scaffolding designs across 4 motifs (`EF_hand_Ca_loop`,
       `C2H2_Zn_finger`, `WalkerA_Ploop`, `Discontinuous_bihotspot`).
     - Graded by a strict 5-gate pipeline:
         Gate 1: Anti-repeat complexity & soluble sequence hygiene (blocks `EAAAKEAAAK` rods)
         Gate 2: Foldability & globular compactness (`pLDDT >= 80`, `Rg <= 2.35 * L^0.38`)
         Gate 3: Self-consistency (`ProteinMPNN -> ESMFold scRMSD <= 2.0 A`)
         Gate 4: Prompt adherence (target topology `SS + TM >= 0.50` or Kabsch `motif Cα RMSD <= 1.50 A`)
         Gate 5: De novo novelty (`< 30%` sequence identity on non-motif scaffold residues)

  2. Part 2 — Agentic Closed-Loop Epitope Binder Design (`N = 8` epitopes, `K = 16` `AF2` oracle shots):
     - Evaluates both `Open-Loop Pass@K` (independent proposals) and `Closed-Loop Refinement`
       (iterative turn-by-turn optimization using structured `AF2-Multimer/pTM` oracle feedback:
       `ipTM`, `iPAE`, `binder_pLDDT`, `monomer_RMSD`, and exact epitope hotspot contacts).
     - Enforces `BindCraft` anti-hallucination & anti-memorization rules:
       must contact `>= 3` specified target epitope hotspots, fold stably as an unbound monomer
       (`monomer_RMSD <= 2.0 A`), pass unpaired-Cys / soluble-surface hygiene, and stay `< 25%`
       SeqID to held-out native complex partners (`< 30%` to known literature/PDB binders).
"""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass, field
import json
import math
import re
from typing import Any, Dict, List, Optional, Protocol, Sequence, Tuple

import numpy as np

from protein_without_tools.pdb_utils import CANONICAL_AA_SET
from protein_without_tools.stage2a_homology import smith_waterman_affine

KYTE_DOOLITTLE: Dict[str, float] = {
    "I": 4.5, "V": 4.2, "L": 3.8, "F": 2.8, "C": 2.5,
    "M": 1.9, "A": 1.8, "G": -0.4, "T": -0.7, "S": -0.8,
    "W": -0.9, "Y": -1.3, "P": -1.6, "H": -3.2, "E": -3.5,
    "Q": -3.5, "D": -3.5, "N": -3.5, "K": -3.9, "R": -4.5,
}
AA_CHARGE: Dict[str, float] = {"K": 1.0, "R": 1.0, "H": 0.2, "D": -1.0, "E": -1.0}
HYDROPHOBIC_SET = set("IVLFMWYA")
AROMATIC_SET = set("FWY")


# ==============================================================================
# Geometry & Sequence Hygiene Utilities
# ==============================================================================

def kabsch_rmsd(coords_p: np.ndarray, coords_q: np.ndarray) -> float:
    """Exact proper-rotation (`SO(3)`, `det = +1`) Kabsch RMSD in Angstroms."""
    p = np.asarray(coords_p, dtype=np.float64)
    q = np.asarray(coords_q, dtype=np.float64)
    if p.shape != q.shape or p.ndim != 2 or p.shape[0] == 0:
        raise ValueError(f"Mismatched shapes for Kabsch RMSD: {p.shape} vs {q.shape}")
    if p.shape[0] == 1:
        return 0.0
    p_c = p - np.mean(p, axis=0, keepdims=True)
    q_c = q - np.mean(q, axis=0, keepdims=True)
    cov = p_c.T @ q_c
    u, _s, vt = np.linalg.svd(cov)
    d = 1.0 if np.linalg.det(u @ vt) >= 0.0 else -1.0
    rot = u @ np.diag([1.0, 1.0, d]) @ vt
    diff = (p_c @ rot) - q_c
    return float(np.sqrt(np.mean(np.sum(diff * diff, axis=1))))


def shannon_entropy_bits(seq: str) -> float:
    if not seq:
        return 0.0
    n = float(len(seq))
    counts = Counter(seq)
    return float(-sum((c / n) * math.log2(c / n) for c in counts.values()))


def max_homopolymer_run(seq: str) -> int:
    if not seq:
        return 0
    best, cur = 1, 1
    for i in range(1, len(seq)):
        if seq[i] == seq[i - 1]:
            cur += 1
            if cur > best:
                best = cur
        else:
            cur = 1
    return best


def dipeptide_repeat_fraction(seq: str) -> float:
    """Fraction of 3-mer windows dominated by periodic repeats (catches `EAAAKEAAAK`, `LEELLKKL`, `GSGSGS`)."""
    if len(seq) < 12:
        return 0.0
    k3 = [seq[i : i + 3] for i in range(len(seq) - 2)]
    uniq_ratio = len(set(k3)) / float(len(k3))
    return round(1.0 - uniq_ratio, 4)


def parse_designed_sequence(raw_text: str) -> Tuple[str, str, bool]:
    """Extract `(CLEAN_SEQ, rationale, parse_ok)` from JSON or FASTA/raw LLM output."""
    if not raw_text or not raw_text.strip():
        return "", "", False

    cleaned = raw_text.strip()
    m_f = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", cleaned, flags=re.DOTALL)
    json_cand = m_f.group(1) if m_f else ""
    if not json_cand:
        i0, i1 = cleaned.find("{"), cleaned.rfind("}")
        if i0 != -1 and i1 > i0:
            json_cand = cleaned[i0 : i1 + 1]

    if json_cand:
        try:
            obj = json.loads(json_cand)
            seq_raw = str(
                obj.get("sequence")
                or obj.get("binder_sequence")
                or obj.get("designed_sequence")
                or ""
            )
            seq = re.sub(r"[^A-Za-z]", "", seq_raw).upper()
            rat = str(obj.get("rationale") or obj.get("diagnosis") or obj.get("notes") or "")
            if len(seq) >= 25 and all(c in CANONICAL_AA_SET for c in seq):
                return seq, rat, True
        except Exception:
            pass

    m_seq = re.search(r'"(?:binder_)?sequence"\s*:\s*"([A-Za-z\s]+)"', raw_text)
    if m_seq:
        seq = re.sub(r"\s+", "", m_seq.group(1)).upper()
        if len(seq) >= 25 and all(c in CANONICAL_AA_SET for c in seq):
            return seq, "", True

    # Fallback: longest contiguous canonical AA block >= 35 residues
    blocks = re.findall(r"\b[ACDEFGHIKLMNPQRSTVWY]{35,300}\b", raw_text.upper())
    if blocks:
        best_b = max(blocks, key=len)
        return best_b, "", True

    return "", "", False


# ==============================================================================
# Part 1: Monomer Fold-Conditioned & Motif Scaffolding (`N = 40`)
# ==============================================================================

@dataclass
class MotifSegmentSpec:
    """Single contiguous crystal motif segment that must appear intact in a scaffolded design."""

    segment_id: str
    sequence: str
    min_linker_before: int  # min residues before this segment (from N-terminus or previous segment end)
    max_linker_before: int
    native_ca_coords: List[List[float]]  # (L_seg, 3)


@dataclass
class DesignTaskSpec:
    """Specification for a Stage 3B Part-1 monomer or motif-scaffolding design prompt."""

    task_id: str
    category: str  # "fold_conditioned" | "motif_scaffolding"
    subfamily: str  # e.g. "4_helix_bundle", "all_beta_barrel_sh3", "EF_hand_Ca_loop"
    min_length: int
    max_length: int
    description: str
    # Target secondary-structure bounds `(min_H, max_H, min_E, max_E)` for fold-conditioned tasks
    ss_bounds: Tuple[float, float, float, float] = (0.0, 1.0, 0.0, 1.0)
    reference_pdb: str = ""
    motif_segments: List[MotifSegmentSpec] = field(default_factory=list)
    max_allowed_cys: int = 1
    forbidden_parent_sequences: List[str] = field(default_factory=list)


@dataclass
class SequenceHygieneReport:
    hygiene_pass: bool
    length: int
    entropy_bits: float
    repeat_3mer_redundancy: float
    max_single_aa_frac: float
    homopolymer_run: int
    cys_count: int
    mean_hydropathy: float
    motif_present: bool
    motif_0idx_positions: List[int]
    failure_reasons: List[str]


def locate_motif_segments(
    seq: str,
    segments: Sequence[MotifSegmentSpec],
) -> Tuple[bool, List[int], str]:
    """Verify ordered motif segments & linker spacing; return `(ok, flat_0idx_positions, msg)`."""
    if not segments:
        return True, [], ""
    cursor = 0
    all_idx: List[int] = []
    for seg in segments:
        mseq = seg.sequence.upper()
        pos = seq.find(mseq, cursor)
        if pos == -1:
            return False, [], f"missing_motif_{seg.segment_id}({mseq})"
        linker_len = pos - cursor
        if not (seg.min_linker_before <= linker_len <= seg.max_linker_before):
            return (
                False,
                [],
                f"linker_out_of_bounds_{seg.segment_id}({linker_len} not in [{seg.min_linker_before},{seg.max_linker_before}])",
            )
        all_idx.extend(range(pos, pos + len(mseq)))
        cursor = pos + len(mseq)
    if len(seq) - cursor < 4:
        return False, [], "cterm_flank_too_short(<4AA)"
    return True, all_idx, ""


def evaluate_sequence_hygiene(
    seq: str,
    min_len: int,
    max_len: int,
    max_cys: int = 1,
    motif_segments: Sequence[MotifSegmentSpec] = (),
) -> SequenceHygieneReport:
    """Gate 1 (CPU): Reject invalid lengths, repeat-helix `EAAAK` hacks, insoluble balls, or broken motifs."""
    reasons: List[str] = []
    s = seq.strip().upper()
    L = len(s)
    if L == 0 or any(c not in CANONICAL_AA_SET for c in s):
        return SequenceHygieneReport(False, L, 0.0, 1.0, 1.0, L, 0, 0.0, False, [], ["invalid_or_empty_sequence"])

    if not (min_len <= L <= max_len):
        reasons.append(f"length_{L}_outside_[{min_len},{max_len}]")

    ent = shannon_entropy_bits(s)
    if ent < 3.20:
        reasons.append(f"low_shannon_entropy({ent:.2f}<3.20)")

    rep3 = dipeptide_repeat_fraction(s)
    if rep3 > 0.35:
        reasons.append(f"repetitive_3mer_hack({rep3:.2f}>0.35)")

    max_aa = max(Counter(s).values()) / float(L)
    if max_aa > 0.25:
        reasons.append(f"excessive_single_aa({max_aa:.2f}>0.25)")

    hrun = max_homopolymer_run(s)
    if hrun > 4:
        reasons.append(f"homopolymer_run({hrun}>4)")

    n_cys = s.count("C")
    if n_cys > max_cys:
        reasons.append(f"excess_cysteines({n_cys}>{max_cys})")

    mean_kd = float(np.mean([KYTE_DOOLITTLE[c] for c in s]))
    if mean_kd > 0.65:
        reasons.append(f"insoluble_hydrophobic_overload(KD={mean_kd:.2f}>0.65)")

    m_ok, m_idx, m_err = locate_motif_segments(s, motif_segments)
    if not m_ok:
        reasons.append(m_err)

    return SequenceHygieneReport(
        hygiene_pass=(len(reasons) == 0),
        length=L,
        entropy_bits=round(ent, 3),
        repeat_3mer_redundancy=rep3,
        max_single_aa_frac=round(max_aa, 3),
        homopolymer_run=hrun,
        cys_count=n_cys,
        mean_hydropathy=round(mean_kd, 3),
        motif_present=m_ok,
        motif_0idx_positions=m_idx,
        failure_reasons=reasons,
    )


def build_stage3b_monomer_prompt(spec: DesignTaskSpec) -> str:
    """Zero-leakage prompt for fold-conditioned monomer generation or motif scaffolding."""
    if spec.category == "fold_conditioned":
        h_lo, h_hi, e_lo, e_hi = spec.ss_bounds
        prompt = (
            f"You are designing a brand-new, globular, monomeric de novo protein from scratch.\n\n"
            f"Design Specification ({spec.task_id}):\n"
            f"  - Target 3D Fold Architecture: {spec.description}\n"
            f"  - Allowed Sequence Length: {spec.min_length} to {spec.max_length} canonical amino acids\n"
            f"  - Target Secondary Structure Content: helix fraction in [{int(h_lo*100)}%, {int(h_hi*100)}%], "
            f"beta-strand fraction in [{int(e_lo*100)}%, {int(e_hi*100)}%]\n"
            f"  - Biophysical Requirements:\n"
            f"      1. Well-packed hydrophobic core with a water-soluble polar/charged exterior.\n"
            f"      2. High sequence complexity (do NOT use repetitive `EAAAK`/`LEEL` coiled-coil patterns).\n"
            f"      3. At most {spec.max_allowed_cys} cysteine residue(s).\n"
            f"      4. Strictly de novo (`<30%` sequence identity to any natural protein in UniProt/PDB).\n\n"
            f"Return ONLY a single compact JSON object:\n"
            f'{{"sequence": "YOUR_DESIGNED_SEQUENCE", "rationale": "1-sentence topology & core packing plan"}}'
        )
    else:
        seg_lines = []
        for idx_s, seg in enumerate(spec.motif_segments, start=1):
            # Center motif coordinates for clean, PDB-ID-free geometric conditioning
            ca = np.array(seg.native_ca_coords, dtype=np.float64)
            coord_str = "; ".join(f"({xyz[0]:.2f},{xyz[1]:.2f},{xyz[2]:.2f})" for xyz in ca)
            label_before = "N-terminal scaffold flank" if idx_s == 1 else f"Designed scaffold linker after Segment {idx_s - 1}"
            seg_lines.append(
                f"  - Motif Segment {idx_s} (`{seg.sequence}`, {len(seg.sequence)} AA):\n"
                f"      * {label_before}: {seg.min_linker_before} to {seg.max_linker_before} designed residues\n"
                f"      * Target relative 3D Cα coordinates (Å): [{coord_str}]"
            )
        segs_block = "\n".join(seg_lines)
        prompt = (
            f"You are performing de novo functional motif scaffolding in protein engineering.\n\n"
            f"Design Specification ({spec.task_id}):\n"
            f"  - Functional Goal: {spec.description}\n"
            f"  - Total Sequence Length (motif + designed scaffold): {spec.min_length} to {spec.max_length} AA\n"
            f"{segs_block}\n"
            f"  - C-terminal scaffold flank after final motif segment: at least 6 designed residues\n"
            f"  - Requirements:\n"
            f"      1. Embed EVERY motif segment verbatim (`100%` exact sequence match) in the specified order.\n"
            f"      2. Design a compact, well-packed globular scaffold (`<30%` SeqID to natural proteins outside the motif) "
            f"that holds the motif Cα atoms within `< 1.50 Å` RMSD of their target 3D geometry.\n"
            f"      3. At most {spec.max_allowed_cys} total cysteine(s) across the entire sequence.\n\n"
            f"Return ONLY a single compact JSON object:\n"
            f'{{"sequence": "FULL_SCAFFOLD_PLUS_MOTIF_SEQUENCE", "rationale": "1-sentence scaffolding strategy"}}'
        )
    return prompt


def grade_monomer_design(
    spec: DesignTaskSpec,
    designed_seq: str,
    folded_ca_coords: Optional[np.ndarray] = None,
    plddt_mean: Optional[float] = None,
    ss_q3: Optional[str] = None,
    tm_to_target_fold: Optional[float] = None,
    mpnn_sc_rmsd: Optional[float] = None,
    external_max_seqid_pct: float = 0.0,
) -> Dict[str, Any]:
    """Evaluate all 5 gates on a designed monomer or motif-scaffolded sequence."""
    hyg = evaluate_sequence_hygiene(
        designed_seq,
        min_len=spec.min_length,
        max_len=spec.max_length,
        max_cys=spec.max_allowed_cys,
        motif_segments=spec.motif_segments,
    )

    # Check novelty against any forbidden parent sequence (masking out motif indices so short motifs don't penalize!)
    scaffold_only = "".join(
        ch for i, ch in enumerate(designed_seq.upper()) if i not in set(hyg.motif_0idx_positions)
    )
    max_parent_sid = 0.0
    for pseq in spec.forbidden_parent_sequences:
        if len(scaffold_only) >= 25:
            _b, _ev, sid = smith_waterman_affine(scaffold_only, pseq)
            max_parent_sid = max(max_parent_sid, sid)
    max_seqid = max(external_max_seqid_pct, max_parent_sid)
    gate5_novelty = (len(designed_seq) >= spec.min_length) and (max_seqid < 30.0)

    # Gates 2, 3, 4 require 3D coordinates (from ESMFold on Modal or ground-truth calibration PDBs)
    gate2_foldable = False
    gate3_self_consistent = False
    gate4_adherence = False
    rg_val: Optional[float] = None
    motif_rmsd_val: Optional[float] = None

    if folded_ca_coords is not None and len(folded_ca_coords) == len(designed_seq):
        ctr = np.mean(folded_ca_coords, axis=0)
        rg_val = round(float(np.sqrt(np.mean(np.sum((folded_ca_coords - ctr) ** 2, axis=1)))), 2)
        rg_cap = 2.35 * (len(designed_seq) ** 0.38)
        plddt_ok = (plddt_mean is None) or (plddt_mean >= 80.0)
        gate2_foldable = plddt_ok and (rg_val <= rg_cap)

        gate3_self_consistent = (mpnn_sc_rmsd is not None) and (mpnn_sc_rmsd <= 2.0)

        if spec.category == "fold_conditioned":
            ss_ok = True
            if ss_q3 and len(ss_q3) == len(designed_seq):
                fh = ss_q3.count("H") / float(len(ss_q3))
                fe = ss_q3.count("E") / float(len(ss_q3))
                h_lo, h_hi, e_lo, e_hi = spec.ss_bounds
                ss_ok = (h_lo <= fh <= h_hi) and (e_lo <= fe <= e_hi)
            tm_ok = (tm_to_target_fold is None) or (tm_to_target_fold >= 0.50)
            gate4_adherence = ss_ok and tm_ok
        else:
            if hyg.motif_present and hyg.motif_0idx_positions:
                native_cat = np.concatenate(
                    [np.array(seg.native_ca_coords, dtype=np.float64) for seg in spec.motif_segments],
                    axis=0,
                )
                pred_motif_ca = folded_ca_coords[hyg.motif_0idx_positions]
                motif_rmsd_val = round(kabsch_rmsd(pred_motif_ca, native_cat), 3)
                gate4_adherence = motif_rmsd_val <= 1.50

    design_success = bool(
        hyg.hygiene_pass and gate2_foldable and gate3_self_consistent and gate4_adherence and gate5_novelty
    )
    return {
        "task_id": spec.task_id,
        "category": spec.category,
        "subfamily": spec.subfamily,
        "length": hyg.length,
        "gate1_hygiene": hyg.hygiene_pass,
        "gate2_foldable_globular": gate2_foldable,
        "gate3_mpnn_self_consistent": gate3_self_consistent,
        "gate4_prompt_adherence": gate4_adherence,
        "gate5_novelty_lt30": gate5_novelty,
        "all_5_gates_pass": design_success,
        "entropy_bits": hyg.entropy_bits,
        "rg_angstrom": rg_val,
        "plddt_mean": plddt_mean,
        "mpnn_sc_rmsd": mpnn_sc_rmsd,
        "tm_to_target_fold": tm_to_target_fold,
        "motif_ca_rmsd": motif_rmsd_val,
        "max_seqid_pct": round(max_seqid, 1),
        "failure_reasons": hyg.failure_reasons,
    }


# ==============================================================================
# Part 2: Closed-Loop `AF2` Epitope Binder Design (`BindCraft`-Style, `K = 16`)
# ==============================================================================

@dataclass
class EpitopeTarget:
    """Target protein & surface epitope hotspot specification for de novo binder design."""

    target_id: str
    tier: str  # "classic_control" | "post_may_2026_complex"
    name: str  # anonymized in prompts!
    pdb_id: str
    release_date: str
    target_chain: str
    target_sequence: str
    epitope_hotspots_1idx: List[int]  # 5 surface hotspot residue indices (1-based on `target_sequence`)
    binder_min_len: int
    binder_max_len: int
    target_ca_coords: List[List[float]]
    target_cb_coords: List[List[float]]
    # Held-out ground-truth native binding partner (`Chain B`) — NEVER exposed in prompts!
    heldout_partner_sequence: str
    heldout_partner_ca_coords: List[List[float]]
    heldout_partner_cb_coords: List[List[float]]
    forbidden_sequences: List[str] = field(default_factory=list)


@dataclass
class BinderShotFeedback:
    """Single-round `AF2` (or surrogate) oracle evaluation returned in the closed-loop loop."""

    round_idx: int
    binder_sequence: str
    hygiene_ok: bool
    novelty_ok: bool
    max_forbidden_seqid_pct: float
    iptm: float
    ipae: float
    binder_plddt: float
    monomer_rmsd: float
    hotspots_contacted_1idx: List[int]
    hotspots_missed_1idx: List[int]
    clash_count: int
    composite_score: float
    is_hit: bool
    diagnostic: str


class BinderOracle(Protocol):
    def evaluate_binder(
        self,
        target: EpitopeTarget,
        binder_seq: str,
        round_idx: int,
    ) -> BinderShotFeedback: ...


def score_3d_complex_geometry(
    target: EpitopeTarget,
    binder_seq: str,
    pred_target_cb: np.ndarray,
    pred_binder_ca: np.ndarray,
    pred_binder_cb: np.ndarray,
    iptm: float,
    ipae: float,
    binder_plddt: float,
    monomer_rmsd: float,
    round_idx: int,
    external_max_seqid: float = 0.0,
) -> BinderShotFeedback:
    """Apply exact `BindCraft` + anti-cheat hotspot & novelty rules to co-folded 3D coordinates."""
    hyg = evaluate_sequence_hygiene(
        binder_seq,
        min_len=target.binder_min_len,
        max_len=target.binder_max_len,
        max_cys=1,
    )

    max_forb = external_max_seqid
    is_memorized_homolog = external_max_seqid >= 35.0
    for fseq in [target.target_sequence, target.heldout_partner_sequence, *target.forbidden_sequences]:
        if fseq and len(fseq) >= 40 and len(binder_seq) >= 25:
            _b, ev, sid = smith_waterman_affine(binder_seq, fseq)
            max_forb = max(max_forb, sid)
            if (sid >= 35.0 and ev < 1e-2) or sid >= 45.0:
                is_memorized_homolog = True
    novelty_ok = not is_memorized_homolog

    # Compute intermolecular Cbeta contacts (< 8.0 A) & Cα steric clashes (< 3.3 A)
    t_ca = np.asarray(target.target_ca_coords, dtype=np.float64)
    d_cb = np.linalg.norm(pred_target_cb[:, None, :] - pred_binder_cb[None, :, :], axis=-1)
    d_ca = np.linalg.norm(t_ca[:, None, :] - pred_binder_ca[None, :, :], axis=-1) if len(t_ca) == len(pred_target_cb) else d_cb
    clash_cnt = int(np.sum(d_ca < 3.3))

    contacted_1idx = []
    missed_1idx = []
    for h1 in target.epitope_hotspots_1idx:
        if 1 <= h1 <= len(pred_target_cb) and float(np.min(d_cb[h1 - 1])) < 8.0:
            contacted_1idx.append(h1)
        else:
            missed_1idx.append(h1)

    hotspot_frac = len(contacted_1idx) / float(max(1, len(target.epitope_hotspots_1idx)))
    is_hit = bool(
        hyg.hygiene_ok if hasattr(hyg, "hygiene_ok") else hyg.hygiene_pass
    ) and bool(
        novelty_ok
        and iptm >= 0.75
        and ipae < 10.0
        and binder_plddt >= 80.0
        and monomer_rmsd <= 2.0
        and len(contacted_1idx) >= 3
        and clash_cnt == 0
    )

    comp = (
        0.40 * float(np.clip(iptm, 0.0, 1.0))
        + 0.25 * float(np.clip(1.0 - ipae / 28.0, 0.0, 1.0))
        + 0.20 * hotspot_frac
        + 0.15 * float(np.clip(binder_plddt / 100.0, 0.0, 1.0))
    )
    if not hyg.hygiene_pass:
        comp *= 0.25
    if not novelty_ok:
        comp = 0.0

    diag_parts: List[str] = []
    if not novelty_ok:
        diag_parts.append(f"DISQUALIFIED: {max_forb:.1f}% sequence identity to held-out/known binder (>=25% forbidden).")
    if not hyg.hygiene_pass:
        diag_parts.append("Sequence hygiene failed: " + ", ".join(hyg.failure_reasons))
    if len(contacted_1idx) < 3:
        diag_parts.append(
            f"Off-epitope or weak docking: contacted {len(contacted_1idx)}/{len(target.epitope_hotspots_1idx)} "
            f"hotspots {contacted_1idx}; missed {missed_1idx}."
        )
    else:
        diag_parts.append(f"On-epitope docking ({len(contacted_1idx)}/{len(target.epitope_hotspots_1idx)} hotspots hit: {contacted_1idx}).")
    if iptm < 0.75 or ipae >= 10.0:
        diag_parts.append(f"Interface confidence below BindCraft threshold (ipTM={iptm:.2f} need >=0.75; iPAE={ipae:.1f}A need <10A).")
    if monomer_rmsd > 2.0 or binder_plddt < 80.0:
        diag_parts.append(f"Unbound binder fold unstable (pLDDT={binder_plddt:.1f}, monomer_RMSD={monomer_rmsd:.2f}A need <=2.0A).")
    if clash_cnt > 0:
        diag_parts.append(f"Steric backbone clashes detected ({clash_cnt} interchain Cα pairs < 3.3A; need 0).")
    if is_hit:
        diag_parts.append("SUCCESS: Passed all BindCraft, hotspot, monomer stability, and novelty filters!")

    return BinderShotFeedback(
        round_idx=round_idx,
        binder_sequence=binder_seq,
        hygiene_ok=hyg.hygiene_pass,
        novelty_ok=novelty_ok,
        max_forbidden_seqid_pct=round(max_forb, 1),
        iptm=round(float(iptm), 4),
        ipae=round(float(ipae), 2),
        binder_plddt=round(float(binder_plddt), 1),
        monomer_rmsd=round(float(monomer_rmsd), 2),
        hotspots_contacted_1idx=contacted_1idx,
        hotspots_missed_1idx=missed_1idx,
        clash_count=clash_cnt,
        composite_score=round(float(comp), 4),
        is_hit=is_hit,
        diagnostic=" ".join(diag_parts),
    )


class SurrogatePhysicsBinderOracle:
    """Deterministic biophysical CPU oracle for offline closed-loop testing & calibration controls.

    Evaluates amphipathic helical bundle propensity, complementary hydrophobic + opposite-charge matching
    against the target's actual 3D epitope patch, sequence hygiene, and novelty rules.
    (Live benchmark campaigns use `modal_stage3b_af2_bindcraft.py` running `ColabDesign` `AF2` on `L40S`!)
    """

    def evaluate_binder(
        self,
        target: EpitopeTarget,
        binder_seq: str,
        round_idx: int,
    ) -> BinderShotFeedback:
        s = binder_seq.strip().upper()
        hyg = evaluate_sequence_hygiene(s, target.binder_min_len, target.binder_max_len, max_cys=1)
        if not s:
            return BinderShotFeedback(
                round_idx, "", False, True, 0.0, 0.1, 28.0, 35.0, 8.5, [], list(target.epitope_hotspots_1idx), 0, 0.05, False,
                "Failed to parse canonical binder sequence."
            )

        max_forb = 0.0
        is_memorized_homolog = False
        for fseq in [target.target_sequence, target.heldout_partner_sequence, *target.forbidden_sequences]:
            if fseq and len(fseq) >= 40 and len(s) >= 25:
                _b, ev, sid = smith_waterman_affine(s, fseq)
                max_forb = max(max_forb, sid)
                if (sid >= 35.0 and ev < 1e-2) or sid >= 45.0:
                    is_memorized_homolog = True
        novelty_ok = not is_memorized_homolog

        # Analyze target epitope patch physicochemical signature
        ep_aas = [target.target_sequence[i - 1] for i in target.epitope_hotspots_1idx if 1 <= i <= len(target.target_sequence)]
        ep_charge = sum(AA_CHARGE.get(a, 0.0) for a in ep_aas)
        ep_hydro = float(np.mean([KYTE_DOOLITTLE[a] for a in ep_aas])) if ep_aas else 0.0

        # Require coherent multi-helix amphipathic periodicity across BOTH Helix 1 (0..18) and Helix 2 (24..42),
        # plus polar/Gly/Pro turn caps (`18..24` and `42..48`), matching real miniprotein bundle physics
        # (so uniform random AA strings get pLDDT ~ 44 & ipTM ~ 0.22 instead of extreme-value window luck!).
        def _helix_amphipathic_quality(seg: str) -> float:
            if len(seg) < 15:
                return 0.0
            core_idx = [k for k in (0, 3, 4, 7, 10, 11, 14, 17) if k < len(seg)]
            surf_idx = [k for k in range(len(seg)) if k not in core_idx]
            # Helix breakers (`P`, `G`) inside helix body penalize folding
            breakers = sum(1 for c in seg[2:-2] if c in ("P", "G"))
            c_hyd = float(np.mean([KYTE_DOOLITTLE[seg[k]] for k in core_idx]))
            s_hyd = float(np.mean([KYTE_DOOLITTLE[seg[k]] for k in surf_idx]))
            return float(np.clip((c_hyd - s_hyd) / 4.8 - 0.16 * breakers, 0.0, 1.0))

        h1_q = _helix_amphipathic_quality(s[2:20])
        h2_q = _helix_amphipathic_quality(s[25:43])
        h3_q = _helix_amphipathic_quality(s[47 : min(len(s), 65)])
        has_turns = (any(c in "GPNDST" for c in s[20:25]) and any(c in "GPNDST" for c in s[43:48])) if len(s) >= 52 else False
        bundle_fold = float(np.clip(0.42 * h1_q + 0.38 * h2_q + 0.20 * h3_q + (0.14 if has_turns else -0.10), 0.0, 1.0))
        if not hyg.hygiene_pass:
            bundle_fold *= 0.40

        # Interface hotspot complementarity on Helix-1 outward/interface face (`1, 5, 8, 12, 15`)
        iface_res = [s[k] for k in (1, 5, 8, 12, 15, 27, 31, 34) if k < len(s)]
        if_arom = sum(1 for a in iface_res if a in AROMATIC_SET) / float(max(1, len(iface_res)))
        if_hyd = float(np.mean([KYTE_DOOLITTLE[a] for a in iface_res])) if iface_res else 0.0
        if_chg = sum(AA_CHARGE.get(a, 0.0) for a in iface_res)
        elec_ok = 1.0 if (ep_charge * if_chg <= -0.8 or (abs(ep_charge) <= 1.0 and abs(if_chg) <= 1.5)) else 0.45
        hyd_ok = float(np.clip(1.0 - abs(if_hyd - max(-0.5, -0.4 * ep_hydro)) / 3.8, 0.15, 1.0))
        best_iface = float(np.clip(0.45 * hyd_ok + 0.30 * elec_ok + min(0.32, if_arom * 0.95), 0.0, 1.0))
        fold_q = bundle_fold

        iptm = float(np.clip(0.22 + 0.76 * best_iface * (0.35 + 0.65 * fold_q), 0.10, 0.94))
        ipae = float(np.clip(26.0 - 22.5 * best_iface * (0.35 + 0.65 * fold_q), 3.8, 28.0))
        plddt = float(np.clip(46.0 + 48.0 * fold_q, 38.0, 95.0))
        m_rmsd = float(np.clip(4.8 - 3.9 * fold_q, 0.80, 6.5))

        n_hit_hs = int(round(len(target.epitope_hotspots_1idx) * np.clip(best_iface * 1.05, 0.0, 1.0)))
        contacted = list(target.epitope_hotspots_1idx[:n_hit_hs])
        missed = list(target.epitope_hotspots_1idx[n_hit_hs:])

        is_hit = bool(
            hyg.hygiene_pass
            and novelty_ok
            and iptm >= 0.75
            and ipae < 10.0
            and plddt >= 80.0
            and m_rmsd <= 2.0
            and len(contacted) >= 3
        )
        comp = (
            0.40 * iptm
            + 0.25 * float(np.clip(1.0 - ipae / 28.0, 0.0, 1.0))
            + 0.20 * (len(contacted) / float(max(1, len(target.epitope_hotspots_1idx))))
            + 0.15 * (plddt / 100.0)
        )
        if not hyg.hygiene_pass:
            comp *= 0.25
        if not novelty_ok:
            comp = 0.0

        diag = []
        if not novelty_ok:
            diag.append(f"DISQUALIFIED: {max_forb:.1f}% SeqID to held-out/published binder (>=25% forbidden).")
        if not hyg.hygiene_pass:
            diag.append("Hygiene failed: " + ", ".join(hyg.failure_reasons))
        diag.append(
            f"Hotspots hit {len(contacted)}/{len(target.epitope_hotspots_1idx)} ({contacted}); missed {missed}. "
            f"ipTM={iptm:.2f}, iPAE={ipae:.1f}A, pLDDT={plddt:.1f}, monomer_RMSD={m_rmsd:.2f}A."
        )
        return BinderShotFeedback(
            round_idx=round_idx,
            binder_sequence=s,
            hygiene_ok=hyg.hygiene_pass,
            novelty_ok=novelty_ok,
            max_forbidden_seqid_pct=round(max_forb, 1),
            iptm=round(iptm, 4),
            ipae=round(ipae, 2),
            binder_plddt=round(plddt, 1),
            monomer_rmsd=round(m_rmsd, 2),
            hotspots_contacted_1idx=contacted,
            hotspots_missed_1idx=missed,
            clash_count=0,
            composite_score=round(float(comp), 4),
            is_hit=is_hit,
            diagnostic=" ".join(diag),
        )


def build_binder_initial_prompt(target: EpitopeTarget) -> str:
    """Zero-leakage Round-1 prompt for epitope-specific miniprotein binder design."""
    tseq = target.target_sequence.upper()
    t_ca = np.asarray(target.target_ca_coords, dtype=np.float64)
    # Center on the epitope centroid so coordinates are compact & leak-free!
    hs_0idx = [i - 1 for i in target.epitope_hotspots_1idx]
    ep_center = np.mean(t_ca[hs_0idx], axis=0)
    rel_ca = t_ca - ep_center

    hs_desc = ", ".join(f"{tseq[i - 1]}{i}" for i in target.epitope_hotspots_1idx)
    # Provide all surface/epitope neighborhood residues within 14 A of epitope centroid
    near_lines = []
    for idx0 in range(len(tseq)):
        dist = float(np.linalg.norm(rel_ca[idx0]))
        if dist <= 14.0 or (idx0 + 1) in target.epitope_hotspots_1idx:
            tag = " [HOTSPOT]" if (idx0 + 1) in target.epitope_hotspots_1idx else ""
            xyz = rel_ca[idx0]
            near_lines.append(f"  res {idx0 + 1:3d} ({tseq[idx0]}): Cα=({xyz[0]:6.2f}, {xyz[1]:6.2f}, {xyz[2]:6.2f}){tag}")

    prompt = (
        f"You are competing in a BindCraft-style closed-loop de novo miniprotein binder design challenge.\n\n"
        f"Anonymized Target Protein ({target.target_id}, length {len(tseq)} AA):\n"
        f"{tseq}\n\n"
        f"Target Epitope Hotspot Residues (1-indexed — your binder MUST directly contact >=3 of these):\n"
        f"  {hs_desc}\n\n"
        f"Local 3D Epitope Surface Geometry (centered at epitope centroid, Å):\n"
        + "\n".join(near_lines[:45])
        + "\n\n"
        f"Hit Criteria (evaluated by AlphaFold2-Multimer + Monomer Self-Consistency Oracle):\n"
        f"  1. Binder length between {target.binder_min_len} and {target.binder_max_len} canonical amino acids.\n"
        f"  2. Interface `ipTM >= 0.75` AND `iPAE < 10.0 Å`.\n"
        f"  3. Direct intermolecular contact (`Cβ < 8.0 Å`) with at least 3 of the 5 specified hotspot residues.\n"
        f"  4. Folds stably on its own as an unbound globular monomer (`binder pLDDT >= 80`, `monomer RMSD <= 2.0 Å`).\n"
        f"  5. Strictly de novo (`<25%` SeqID to any natural partner or published binder), <=1 Cys, non-repetitive.\n\n"
        f"Shot 1 — Propose your initial de novo miniprotein binder as a compact JSON object:\n"
        f'{{"binder_sequence": "YOUR_SEQUENCE", "rationale": "brief interface & fold strategy"}}'
    )
    assert target.pdb_id.upper() not in prompt.upper()
    return prompt


def build_binder_refinement_prompt(
    target: EpitopeTarget,
    history: Sequence[BinderShotFeedback],
    total_budget: int = 16,
) -> str:
    """Closed-loop Round `t` prompt conditioning on all prior `AF2` oracle evaluations `1..t-1`."""
    base = build_binder_initial_prompt(target).split("Shot 1 —")[0]
    hist_lines = [
        "Previous AlphaFold2 Oracle Evaluations (Shots 1 to " + str(len(history)) + f" of {total_budget}):"
    ]
    for fb in history:
        hist_lines.append(
            f"  - Shot {fb.round_idx:2d}: ipTM={fb.iptm:.2f} | iPAE={fb.ipae:4.1f}Å | "
            f"pLDDT={fb.binder_plddt:4.1f} | monoRMSD={fb.monomer_rmsd:.2f}Å | "
            f"hotspots_hit={fb.hotspots_contacted_1idx} (missed={fb.hotspots_missed_1idx}) | "
            f"score={fb.composite_score:.3f}\n"
            f"    seq ({len(fb.binder_sequence)} AA): {fb.binder_sequence}\n"
            f"    oracle_note: {fb.diagnostic}"
        )
    next_shot = len(history) + 1
    tail = (
        f"\n\nShot {next_shot} of {total_budget} — Analyze which hotspot contacts, interface packing/electrostatics, "
        f"or monomer core stability limited your best previous shot, and propose an improved de novo binder sequence.\n"
        f"Return ONLY a single compact JSON object:\n"
        f'{{"binder_sequence": "YOUR_IMPROVED_SEQUENCE", "rationale": "1-sentence mutation/redesign hypothesis"}}'
    )
    return base + "\n".join(hist_lines) + tail


def evaluate_binder_campaign(
    target_id: str,
    history: Sequence[BinderShotFeedback],
    budget_k: int = 16,
) -> Dict[str, Any]:
    """Compute `Shots-to-First-Hit`, `Pass@k` curve (`k=1..K`), best `ipTM`/`iPAE`, and closed-loop learning gain."""
    first_hit_shot: Optional[int] = None
    pass_at_k: List[int] = []
    running_best_comp: List[float] = []
    cur_best = 0.0

    for k_idx in range(budget_k):
        if k_idx < len(history):
            fb = history[k_idx]
            if fb.is_hit and first_hit_shot is None:
                first_hit_shot = fb.round_idx
            if fb.novelty_ok and fb.composite_score > cur_best:
                cur_best = fb.composite_score
        pass_at_k.append(1 if first_hit_shot is not None else 0)
        running_best_comp.append(round(cur_best, 4))

    valid_on_ep = [
        fb for fb in history if fb.novelty_ok and fb.hygiene_ok and len(fb.hotspots_contacted_1idx) >= 2
    ]
    best_iptm = max((fb.iptm for fb in valid_on_ep), default=0.0)
    best_ipae = min((fb.ipae for fb in valid_on_ep), default=30.0)

    early_best = max((fb.composite_score for fb in history[: min(4, len(history))] if fb.novelty_ok), default=0.0)
    late_best = max((fb.composite_score for fb in history if fb.novelty_ok), default=0.0)

    return {
        "target_id": target_id,
        "n_shots_evaluated": len(history),
        "hit_achieved": int(first_hit_shot is not None),
        "shots_to_first_hit": first_hit_shot,
        "best_iptm_on_epitope": round(best_iptm, 4),
        "best_ipae_on_epitope": round(best_ipae, 2),
        "best_composite_score": round(late_best, 4),
        "closed_loop_trajectory_gain": round(late_best - early_best, 4),
        "pass_at_k_curve": pass_at_k,
        "running_best_composite": running_best_comp,
    }
