"""End-to-end offline tests for Stage 1 geometry, reconstruction, anti-leakage, and grading."""

from __future__ import annotations

import json
import math
from pathlib import Path
import numpy as np
import pytest

from fable_bench.pdb_utils import (
    load_structure,
    write_backbone_only_pdb,
)
from fable_bench.reconstruct_3d import reconstruct_ca_from_contacts
from fable_bench.stage1_grader import (
    grade_forward_prediction,
    grade_inverse_prediction,
)
from fable_bench.stage1_prompts_and_parsers import (
    ForwardPredictionParsed,
    build_forward_folding_prompt,
    build_inverse_folding_prompt,
    parse_forward_folding_response,
    parse_inverse_folding_response,
)
from fable_bench.tmscore import evaluate_3d_superposition


def _make_synthetic_two_helix_hairpin_pdb(tmp_path: Path) -> Path:
    """Create an idealized 56-residue antiparallel 2-helix bundle PDB with real N, CA, C, O."""
    seq = "GSWAEFKQRLAAIKTRLQALGGSEAELAAFEKEIAAFESELQAYKGKGNPEVEALR"
    L = len(seq)
    aa1_to_3 = {
        "A": "ALA", "E": "GLU", "F": "PHE", "G": "GLY", "I": "ILE",
        "K": "LYS", "L": "LEU", "N": "ASN", "P": "PRO", "Q": "GLN",
        "R": "ARG", "S": "SER", "T": "THR", "V": "VAL", "W": "TRP", "Y": "TYR",
    }

    # Build smooth right-handed helix 1 (res 1..24), 3.8Å-spaced turn (25..28), antiparallel helix 2 (29..56)
    ca_list = []
    for i in range(24):
        theta = i * math.radians(100.0)
        ca_list.append([2.3 * math.cos(theta), 2.3 * math.sin(theta), 1.5 * i])
    p_end = np.array(ca_list[-1])
    # 4-residue semi-circular turn stepping ~3.8 Å per residue to x_shift = 9.2 Å
    p_next_start = np.array([9.2 + 2.3, 0.0, 1.5 * 23])
    for t_idx in range(1, 5):
        frac = t_idx / 5.0
        arc = p_end * (1.0 - frac) + p_next_start * frac + np.array([0.0, 2.6 * math.sin(math.pi * frac), 2.2 * math.sin(math.pi * frac)])
        ca_list.append(arc.tolist())
    for j in range(28):
        theta = j * math.radians(100.0)
        ca_list.append([9.2 + 2.3 * math.cos(-theta), 2.3 * math.sin(-theta), 1.5 * (23 - j)])
        # Wait: right-handed helix descending along -z has x = R cos(+theta), y = R sin(-theta), z = -1.5*j!
    ca_list = ca_list[:28]
    for j in range(28):
        theta = j * math.radians(100.0)
        ca_list.append([9.2 + 2.3 * math.cos(theta), -2.3 * math.sin(theta), 1.5 * (23 - j)])

    ca = np.array(ca_list, dtype=np.float64)
    lines = []
    serial = 1
    for i in range(L):
        fwd = ca[min(L - 1, i + 1)] - ca[max(0, i - 1)]
        fwd = fwd / (np.linalg.norm(fwd) + 1e-8)
        if 0 < i < L - 1:
            u_p = (ca[i] - ca[i - 1]) / (np.linalg.norm(ca[i] - ca[i - 1]) + 1e-8)
            u_n = (ca[i + 1] - ca[i]) / (np.linalg.norm(ca[i + 1] - ca[i]) + 1e-8)
            b_h = (u_p - u_n) / (np.linalg.norm(u_p - u_n) + 1e-8)
            c_h = np.cross(u_p, u_n) / (np.linalg.norm(np.cross(u_p, u_n)) + 1e-8)
            cb_xyz = ca[i] + 1.25 * b_h - 0.79 * c_h
            perp = b_h
        else:
            perp = np.cross(fwd, np.array([0.0, 0.0, 1.0]))
            perp = perp / (np.linalg.norm(perp) + 1e-8)
            cb_xyz = ca[i] + 1.50 * perp
        n_xyz = ca[i] - 0.72 * fwd - 0.45 * perp
        c_xyz = ca[i] + 0.75 * fwd - 0.45 * perp
        o_xyz = c_xyz + 1.23 * fwd
        r3 = aa1_to_3[seq[i]]
        atoms_to_write = [("N", n_xyz, "N"), ("CA", ca[i], "C"), ("C", c_xyz, "C"), ("O", o_xyz, "O")]
        if seq[i] != "G":
            atoms_to_write.append(("CB", cb_xyz, "C"))
        for aname, xyz, el in atoms_to_write:
            lines.append(
                f"ATOM  {serial:5d}  {aname:<3s} {r3} A{i + 1:4d}    "
                f"{xyz[0]:8.3f}{xyz[1]:8.3f}{xyz[2]:8.3f}  1.00 20.00           {el:>2s}"
            )
            serial += 1
    lines.append("TER\nEND\n")
    pdb_file = tmp_path / "synth_9XYZ.pdb"
    pdb_file.write_text("\n".join(lines), encoding="utf-8")
    return pdb_file


def test_tmscore_chirality_and_identity(tmp_path: Path) -> None:
    pdb_file = _make_synthetic_two_helix_hairpin_pdb(tmp_path)
    struct = load_structure(pdb_file, pdb_id="9XYZ", target_id="s1_test")
    assert struct.length == 56

    # Identical rotated + translated structure must score TM = 1.000, RMSD = 0.000
    angle = 0.9
    rot = np.array(
        [
            [math.cos(angle), -math.sin(angle), 0.0],
            [math.sin(angle), math.cos(angle), 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    transformed = struct.coords_ca @ rot + np.array([15.0, -7.5, 42.0])
    sup = evaluate_3d_superposition(transformed, struct.coords_ca)
    assert sup.tm_score == pytest.approx(1.0, abs=1e-4)
    assert sup.rmsd_global == pytest.approx(0.0, abs=1e-4)
    assert sup.lddt_ca == pytest.approx(1.0, abs=1e-4)

    # Mirror-reflected enantiomer MUST be penalized under proper SO(3) TM-score
    mirrored = struct.coords_ca * np.array([-1.0, 1.0, 1.0])
    sup_m = evaluate_3d_superposition(mirrored, struct.coords_ca)
    assert sup_m.tm_score < 0.85
    assert sup_m.tm_score_mirror_best == pytest.approx(1.0, abs=1e-4)


def test_3d_contact_reconstruction_oracle_vs_random(tmp_path: Path) -> None:
    pdb_file = _make_synthetic_two_helix_hairpin_pdb(tmp_path)
    struct = load_structure(pdb_file, pdb_id="9XYZ", target_id="s1_test")
    L = struct.length

    ii, jj = np.triu_indices(L, k=6)
    true_contacts = [
        (int(i + 1), int(j + 1), 0.95)
        for i, j in zip(ii, jj)
        if struct.cb_dist_matrix[i, j] < 8.0
    ]
    assert len(true_contacts) >= 25

    oracle_pred = ForwardPredictionParsed(
        q3="H" * 24 + "C" * 4 + "H" * 28,
        contacts=true_contacts,
        parse_ok=True,
        raw_q3_len=L,
    )
    oracle_grade = grade_forward_prediction(struct, oracle_pred, n_restarts=4, seed=7)

    rng = np.random.default_rng(123)
    all_pairs = [(int(i + 1), int(j + 1)) for i, j in zip(ii, jj)]
    sel = rng.choice(len(all_pairs), size=60, replace=False)
    rand_pred = ForwardPredictionParsed(
        q3="C" * L,
        contacts=[(all_pairs[k][0], all_pairs[k][1], 0.5) for k in sel],
        parse_ok=True,
        raw_q3_len=L,
    )
    rand_grade = grade_forward_prediction(struct, rand_pred, n_restarts=2, seed=7)

    assert oracle_grade.p_at_l_sep6 == pytest.approx(1.0, abs=1e-5)
    assert oracle_grade.tm_score > 0.72
    assert rand_grade.tm_score < 0.42
    assert oracle_grade.tm_score - rand_grade.tm_score > 0.35


def test_zero_leakage_prompts_and_parsers(tmp_path: Path) -> None:
    pdb_file = _make_synthetic_two_helix_hairpin_pdb(tmp_path)
    struct = load_structure(pdb_file, pdb_id="9XYZ", target_id="s1_t001")

    fwd_prompt = build_forward_folding_prompt(struct)
    inv_prompt = build_inverse_folding_prompt(struct)

    assert "9XYZ" not in fwd_prompt
    assert "9XYZ" not in inv_prompt
    assert struct.sequence not in inv_prompt

    # Stripped backbone file must contain zero native side-chain identities
    stripped = tmp_path / "stripped.pdb"
    write_backbone_only_pdb(struct, stripped)
    stripped_txt = stripped.read_text(encoding="utf-8")
    assert "TRP" not in stripped_txt and "PHE" not in stripped_txt

    # Test forward & inverse parsers with markdown code fences
    mock_fwd = "Here is my analysis:\n```json\n" + json.dumps(
        {
            "fold_summary": "Antiparallel 2-helix hairpin",
            "q3": "H" * struct.length,
            "contacts": [[5, 50, 0.92], [8, 47, 0.88], [1, 2, 0.99]],  # [1,2] < 6 sep must be filtered!
        }
    ) + "\n```"
    pf = parse_forward_folding_response(mock_fwd, struct.length)
    assert pf.parse_ok
    assert len(pf.contacts) == 2

    mock_inv = "```json\n" + json.dumps({"sequence": struct.sequence, "design_notes": "Exact native"}) + "\n```"
    pi = parse_inverse_folding_response(mock_inv, struct.length)
    assert pi.parse_ok and pi.length_exact_match
    gi = grade_inverse_prediction(struct, pi)
    assert gi.nsr_overall == pytest.approx(1.0)
