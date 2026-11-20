#!/usr/bin/env python3
"""Offline Builder, Zero-Leakage Prompt Exporter & Calibration Suite for Stage 3B.

Builds `data/stage3b_design/` (100% offline / sandbox-safe from cached `.cif` files):
  1. Part 1 (`monomer_motif_manifest.json`, `N = 40` prompts):
     - `24` Fold-Conditioned Monomer prompts (`6` specs x `4` architectures: `4_helix_bundle`,
       `ferredoxin_bababb`, `rossmann_aba_sandwich`, `all_beta_barrel_sh3`).
     - `16` Functional Crystal-Motif Scaffolding prompts (`4` specs x `4` motifs: `EF_hand_Ca_loop`,
       `C2H2_Zn_finger`, `WalkerA_Ploop`, `Discontinuous_bihotspot`).
     - Exports `outputs/prompts/stage3b_denovo_prompts.jsonl`.

  2. Part 2 (`epitope_binder_manifest.json`, `N = 8` Target Epitopes, `Budget K = 16` `AF2` Shots):
     - `2` Classic `BindCraft` / Baker Controls (`PD-L1` IgV `5O45`, `TrkA-d5` `1WWW`) with strict
       anti-memorization checks against native & published Cao-2022 minibinders.
     - `6` Post-May-2026 (`2026-08` to `2026-11`) Newly Solved Heterodimeric Complexes (`9S83`, `12TB`,
       `10CY`, `9XGP`, `44HP`, `11GH`), holding out `Chain B` so every target epitope is crystallographically
       proven bindable while having zero pre-cutoff binders!
     - Writes clean Chain-A `.pdb` templates (`data/stage3b_design/target_pdbs/`) for `ColabDesign` `AF2`.
     - Exports Round-1 prompts `outputs/prompts/stage3b_binder_round1_prompts.jsonl`.

  3. Runs Offline Calibration Controls & Closed-Loop vs Open-Loop `Shots-to-Hit` Benchmarks, saving
     `outputs/results/stage3b_evaluation_report.json` and `outputs/figures/stage3b_overview.png`.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import random
import sys
from typing import Any, Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path("/Users/taluguloglu/.gemini/antigravity/scratch/fable-protein-bench")
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from fable_bench.pdb_utils import AA3_TO_1, load_structure
from fable_bench.stage3b_design import (
    BinderShotFeedback,
    DesignTaskSpec,
    EpitopeTarget,
    MotifSegmentSpec,
    SurrogatePhysicsBinderOracle,
    build_binder_initial_prompt,
    build_stage3b_monomer_prompt,
    evaluate_binder_campaign,
    grade_monomer_design,
    parse_designed_sequence,
)
from scripts.download_stage3b_raw import parse_cif_all_protein_chains

AA1_TO_3 = {v: k for k, v in AA3_TO_1.items() if k != "MSE"}


def write_backbone_pdb(seq: str, n: np.ndarray, ca: np.ndarray, c: np.ndarray, o: np.ndarray, out_path: Path) -> None:
    lines = []
    ser = 1
    for i, aa in enumerate(seq):
        r3 = AA1_TO_3.get(aa, "ALA")
        for an, xyz, el in (("N", n[i], "N"), ("CA", ca[i], "C"), ("C", c[i], "C"), ("O", o[i], "O")):
            lines.append(
                f"ATOM  {ser:5d}  {an:<3s} {r3} A{i + 1:4d}    "
                f"{xyz[0]:8.3f}{xyz[1]:8.3f}{xyz[2]:8.3f}  1.00 20.00           {el:>2s}"
            )
            ser += 1
    lines.append("TER\nEND\n")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines), encoding="utf-8")


def build_part1_monomer_and_motif(
    raw_cif_dir: Path,
    s1_gt_dir: Path,
    out_dir: Path,
    prompts_dir: Path,
) -> List[DesignTaskSpec]:
    ref_pdb_dir = out_dir / "reference_pdbs"
    ref_pdb_dir.mkdir(parents=True, exist_ok=True)

    # Copy / extract 4 canonical fold templates (`4_helix_bundle`, `ferredoxin`, `rossmann`, `all_beta_sh3`)
    chs_1cll = parse_cif_all_protein_chains((raw_cif_dir / "1CLL.cif").read_text(encoding="utf-8"))["A"]
    chs_1aay = parse_cif_all_protein_chains((raw_cif_dir / "1AAY.cif").read_text(encoding="utf-8"))["A"]
    chs_5p21 = parse_cif_all_protein_chains((raw_cif_dir / "5P21.cif").read_text(encoding="utf-8"))["A"]
    chs_1shg = parse_cif_all_protein_chains((raw_cif_dir / "1SHG.cif").read_text(encoding="utf-8"))["A"]
    chs_9s83 = parse_cif_all_protein_chains((raw_cif_dir / "9S83.cif").read_text(encoding="utf-8"))["A"]

    write_backbone_pdb(chs_5p21["sequence"], chs_5p21["n"], chs_5p21["ca"], chs_5p21["c"], chs_5p21["o"], ref_pdb_dir / "rossmann_5P21.pdb")
    write_backbone_pdb(chs_1shg["sequence"], chs_1shg["n"], chs_1shg["ca"], chs_1shg["c"], chs_1shg["o"], ref_pdb_dir / "all_beta_1SHG.pdb")

    m_s1 = json.loads((s1_gt_dir.parent / "manifest.json").read_text(encoding="utf-8"))
    tid_6mrr = next(t["target_id"] for t in m_s1["targets"] if t["pdb_id"] == "6MRR")
    tid_6czh = next(t["target_id"] for t in m_s1["targets"] if t["pdb_id"] == "6CZH")
    p_6mrr = next((REPO_ROOT / "data" / "raw_pdb_cache").rglob("6MRR.cif"), next(s1_gt_dir.glob(f"{tid_6mrr}*")))
    p_6czh = next((REPO_ROOT / "data" / "raw_pdb_cache").rglob("6CZH.cif"), next(s1_gt_dir.glob(f"{tid_6czh}*")))
    st_4h = load_structure(p_6mrr, pdb_id="6MRR")
    st_fe = load_structure(p_6czh, pdb_id="6CZH")
    write_backbone_pdb(st_4h.sequence, st_4h.coords_n, st_4h.coords_ca, st_4h.coords_c, st_4h.coords_o, ref_pdb_dir / "4helix_6MRR.pdb")
    write_backbone_pdb(st_fe.sequence, st_fe.coords_n, st_fe.coords_ca, st_fe.coords_c, st_fe.coords_o, ref_pdb_dir / "ferredoxin_6CZH.pdb")

    specs: List[DesignTaskSpec] = []
    idx = 1

    # 1. 24 Fold-Conditioned Monomer Prompts (6 length bands x 4 architectures)
    fold_defs = [
        (
            "4_helix_bundle",
            "Up-down-up-down antiparallel 4-helix bundle (all-alpha globular core)",
            (0.55, 0.90, 0.00, 0.08),
            [(76, 86), (80, 90), (84, 94), (88, 98), (92, 102), (96, 108)],
            "reference_pdbs/4helix_6MRR.pdb",
            [st_4h.sequence],
        ),
        (
            "ferredoxin_bababb",
            "Ferredoxin-like alpha+beta sandwich with beta-alpha-beta-beta-alpha-beta topology",
            (0.22, 0.48, 0.24, 0.52),
            [(78, 88), (82, 92), (86, 96), (90, 100), (94, 104), (98, 110)],
            "reference_pdbs/ferredoxin_6CZH.pdb",
            [st_fe.sequence],
        ),
        (
            "rossmann_aba_sandwich",
            "Rossmann-like 3-layer alpha/beta/alpha parallel beta-sheet sandwich",
            (0.28, 0.54, 0.18, 0.42),
            [(104, 116), (108, 120), (112, 124), (116, 128), (120, 132), (124, 136)],
            "reference_pdbs/rossmann_5P21.pdb",
            [chs_5p21["sequence"]],
        ),
        (
            "all_beta_barrel_sh3",
            "SH3-like antiparallel 5-stranded all-beta barrel (strictly minimal/zero alpha-helix)",
            (0.00, 0.12, 0.36, 0.72),
            [(64, 74), (68, 78), (72, 82), (76, 86), (80, 90), (84, 96)],
            "reference_pdbs/all_beta_1SHG.pdb",
            [chs_1shg["sequence"]],
        ),
    ]

    for subfam, desc, ss_b, len_pairs, ref_p, forb in fold_defs:
        for l_min, l_max in len_pairs:
            specs.append(
                DesignTaskSpec(
                    task_id=f"s3b_d{idx:03d}",
                    category="fold_conditioned",
                    subfamily=subfam,
                    min_length=l_min,
                    max_length=l_max,
                    description=desc,
                    ss_bounds=ss_b,
                    reference_pdb=ref_p,
                    max_allowed_cys=1,
                    forbidden_parent_sequences=forb,
                )
            )
            idx += 1

    # 2. 16 Functional Motif Scaffolding Prompts (4 length/placement specs x 4 crystal motifs)
    # Motif A: EF-hand Ca2+ binding loop `DKDGDGTITTKE` in 1CLL
    ef_pos = chs_1cll["sequence"].index("DKDGDGTITTKE")
    ef_ca = chs_1cll["ca"][ef_pos : ef_pos + 12] - np.mean(chs_1cll["ca"][ef_pos : ef_pos + 12], axis=0)

    # Motif B: C2H2 Zn-finger discontinuous chelating core in 1AAY (`YACPVESCDRR` + linker + `LTRHIRIHT`)
    zn1_pos = chs_1aay["sequence"].index("YACPVESCDRR")
    zn2_pos = chs_1aay["sequence"].index("LTRHIRIHT")
    zn_center = np.mean(
        np.concatenate([chs_1aay["ca"][zn1_pos : zn1_pos + 11], chs_1aay["ca"][zn2_pos : zn2_pos + 9]], axis=0),
        axis=0,
    )
    zn1_ca = chs_1aay["ca"][zn1_pos : zn1_pos + 11] - zn_center
    zn2_ca = chs_1aay["ca"][zn2_pos : zn2_pos + 9] - zn_center

    # Motif C: Walker-A P-loop nucleotide cradle `LVVVGAGGVGKS` in 5P21
    pl_pos = chs_5p21["sequence"].index("LVVVGAGGVGKS")
    pl_ca = chs_5p21["ca"][pl_pos : pl_pos + 12] - np.mean(chs_5p21["ca"][pl_pos : pl_pos + 12], axis=0)

    # Motif D: Discontinuous two-segment functional epitope (`AKELAEKALA` + `[14..32 AA]` + `WSRENMYRAA` from 2026 `9S83`)
    bi1_seq = chs_9s83["sequence"][5:15]   # `AKELAEKALA`
    bi2_seq = chs_9s83["sequence"][42:52]  # `WSRENMYRAA`
    bi_ctr = np.mean(np.concatenate([chs_9s83["ca"][5:15], chs_9s83["ca"][42:52]], axis=0), axis=0)
    bi1_ca = chs_9s83["ca"][5:15] - bi_ctr
    bi2_ca = chs_9s83["ca"][42:52] - bi_ctr

    motif_defs = [
        (
            "EF_hand_Ca_loop",
            "Scaffold a 12-residue Ca2+-chelating EF-hand loop in its native pentagonal-bipyramidal geometry",
            1,
            [chs_1cll["sequence"]],
            [
                (75, 95, [MotifSegmentSpec("ef_loop", "DKDGDGTITTKE", 16, 45, np.round(ef_ca, 3).tolist())]),
                (85, 105, [MotifSegmentSpec("ef_loop", "DKDGDGTITTKE", 22, 55, np.round(ef_ca, 3).tolist())]),
                (90, 112, [MotifSegmentSpec("ef_loop", "DKDGDGTITTKE", 28, 60, np.round(ef_ca, 3).tolist())]),
                (95, 120, [MotifSegmentSpec("ef_loop", "DKDGDGTITTKE", 32, 68, np.round(ef_ca, 3).tolist())]),
            ],
        ),
        (
            "C2H2_Zn_finger",
            "Scaffold a discontinuous C2H2 Zn2+-chelating beta-hairpin + alpha-helix motif in native 3D orientation",
            2,
            [chs_1aay["sequence"]],
            [
                (
                    70,
                    92,
                    [
                        MotifSegmentSpec("zn_c2_hairpin", "YACPVESCDRR", 10, 35, np.round(zn1_ca, 3).tolist()),
                        MotifSegmentSpec("zn_h2_helix", "LTRHIRIHT", 3, 10, np.round(zn2_ca, 3).tolist()),
                    ],
                ),
                (
                    80,
                    102,
                    [
                        MotifSegmentSpec("zn_c2_hairpin", "YACPVESCDRR", 14, 42, np.round(zn1_ca, 3).tolist()),
                        MotifSegmentSpec("zn_h2_helix", "LTRHIRIHT", 3, 12, np.round(zn2_ca, 3).tolist()),
                    ],
                ),
                (
                    88,
                    110,
                    [
                        MotifSegmentSpec("zn_c2_hairpin", "YACPVESCDRR", 18, 48, np.round(zn1_ca, 3).tolist()),
                        MotifSegmentSpec("zn_h2_helix", "LTRHIRIHT", 3, 12, np.round(zn2_ca, 3).tolist()),
                    ],
                ),
                (
                    95,
                    118,
                    [
                        MotifSegmentSpec("zn_c2_hairpin", "YACPVESCDRR", 22, 55, np.round(zn1_ca, 3).tolist()),
                        MotifSegmentSpec("zn_h2_helix", "LTRHIRIHT", 3, 14, np.round(zn2_ca, 3).tolist()),
                    ],
                ),
            ],
        ),
        (
            "WalkerA_Ploop",
            "Scaffold a 12-residue Walker-A P-loop phosphate-binding cradle (`GxxxxGKS`) inside a de novo globular core",
            1,
            [chs_5p21["sequence"]],
            [
                (82, 104, [MotifSegmentSpec("ploop", "LVVVGAGGVGKS", 8, 36, np.round(pl_ca, 3).tolist())]),
                (90, 112, [MotifSegmentSpec("ploop", "LVVVGAGGVGKS", 14, 44, np.round(pl_ca, 3).tolist())]),
                (98, 120, [MotifSegmentSpec("ploop", "LVVVGAGGVGKS", 20, 52, np.round(pl_ca, 3).tolist())]),
                (105, 128, [MotifSegmentSpec("ploop", "LVVVGAGGVGKS", 25, 60, np.round(pl_ca, 3).tolist())]),
            ],
        ),
        (
            "Discontinuous_bihotspot",
            "Scaffold two discontinuous 10-residue binding segments at their exact relative 3D translation & rotation",
            1,
            [chs_9s83["sequence"]],
            [
                (
                    72,
                    92,
                    [
                        MotifSegmentSpec("seg1", bi1_seq, 6, 26, np.round(bi1_ca, 3).tolist()),
                        MotifSegmentSpec("seg2", bi2_seq, 14, 34, np.round(bi2_ca, 3).tolist()),
                    ],
                ),
                (
                    80,
                    102,
                    [
                        MotifSegmentSpec("seg1", bi1_seq, 10, 32, np.round(bi1_ca, 3).tolist()),
                        MotifSegmentSpec("seg2", bi2_seq, 16, 38, np.round(bi2_ca, 3).tolist()),
                    ],
                ),
                (
                    88,
                    110,
                    [
                        MotifSegmentSpec("seg1", bi1_seq, 14, 38, np.round(bi1_ca, 3).tolist()),
                        MotifSegmentSpec("seg2", bi2_seq, 18, 42, np.round(bi2_ca, 3).tolist()),
                    ],
                ),
                (
                    96,
                    118,
                    [
                        MotifSegmentSpec("seg1", bi1_seq, 18, 44, np.round(bi1_ca, 3).tolist()),
                        MotifSegmentSpec("seg2", bi2_seq, 20, 46, np.round(bi2_ca, 3).tolist()),
                    ],
                ),
            ],
        ),
    ]

    for subfam, desc, max_c, forb, variants in motif_defs:
        for l_min, l_max, segs in variants:
            specs.append(
                DesignTaskSpec(
                    task_id=f"s3b_d{idx:03d}",
                    category="motif_scaffolding",
                    subfamily=subfam,
                    min_length=l_min,
                    max_length=l_max,
                    description=desc,
                    motif_segments=segs,
                    max_allowed_cys=max_c,
                    forbidden_parent_sequences=forb,
                )
            )
            idx += 1

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "monomer_motif_manifest.json").write_text(
        json.dumps(
            {
                "benchmark": "fable-protein-bench-stage3b-part1-monomer-motif",
                "n_tasks": len(specs),
                "tasks": [asdict(s) for s in specs],
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    prompts_dir.mkdir(parents=True, exist_ok=True)
    p_lines = []
    for s in specs:
        pr = build_stage3b_monomer_prompt(s)
        p_lines.append(
            json.dumps(
                {
                    "target_id": s.task_id,
                    "task": "stage3b_monomer_motif",
                    "category": s.category,
                    "subfamily": s.subfamily,
                    "length": (s.min_length + s.max_length) // 2,
                    "prompt": pr,
                }
            )
        )
    (prompts_dir / "stage3b_denovo_prompts.jsonl").write_text("\n".join(p_lines) + "\n", encoding="utf-8")

    spot_dir = prompts_dir / "web_chat_spotcheck"
    spot_dir.mkdir(parents=True, exist_ok=True)
    (spot_dir / "s3b_d019_all_beta_barrel.txt").write_text(build_stage3b_monomer_prompt(specs[18]), encoding="utf-8")
    (spot_dir / "s3b_d025_ef_hand_scaffold.txt").write_text(build_stage3b_monomer_prompt(specs[24]), encoding="utf-8")

    return specs


def select_surface_hydrophobic_patch_hotspots(seq: str, ca: np.ndarray, cb: np.ndarray) -> List[int]:
    """Select 5 coherent, solvent-exposed hydrophobic/aromatic patch hotspot residues (1-indexed) for apo targets (`BoltzProt-1`)."""
    from fable_bench.stage3b_design import AROMATIC_SET, KYTE_DOOLITTLE

    dmat = np.linalg.norm(cb[:, None, :] - cb[None, :, :], axis=-1)
    # Burial count (`Cbeta < 10.0 A`, excluding self & +/-1 neighbors)
    nbr_cnt = np.sum(dmat < 10.0, axis=1) - 1
    median_nbr = float(np.median(nbr_cnt))
    # Surface-exposed mask (`nbr_cnt <= median_nbr`)
    surf_mask = nbr_cnt <= median_nbr

    res_prop = np.array(
        [
            (1.6 if aa in AROMATIC_SET else 0.0) + max(0.0, KYTE_DOOLITTLE[aa] * 0.45) + (0.35 if aa in "HYQNRK" else 0.0)
            for aa in seq
        ],
        dtype=np.float64,
    )
    best_patch_score = -1e9
    best_anchor = 0
    for i in range(len(seq)):
        if not surf_mask[i]:
            continue
        local_surf = [j for j in range(len(seq)) if surf_mask[j] and dmat[i, j] <= 12.5]
        if len(local_surf) < 6:
            continue
        sc = float(np.sum(res_prop[local_surf]))
        if sc > best_patch_score:
            best_patch_score = sc
            best_anchor = i

    patch_cands = sorted(
        [j for j in range(len(seq)) if surf_mask[j] and dmat[best_anchor, j] <= 13.5],
        key=lambda j: (-res_prop[j], dmat[best_anchor, j]),
    )
    chosen_1idx: List[int] = []
    for j in patch_cands:
        if all(abs((j + 1) - h) >= 2 for h in chosen_1idx):
            chosen_1idx.append(int(j + 1))
        if len(chosen_1idx) == 5:
            break
    return sorted(chosen_1idx)


def build_part2_epitope_binders(
    raw_cif_dir: Path,
    out_dir: Path,
    prompts_dir: Path,
) -> List[EpitopeTarget]:
    """Build `N = 12` Proven-De-Novo-Bindable Epitope Targets across 3 Tiers:
      - Tier 1 (`rfd4_proteina_classic`, n=4): `5O45` (`PD-L1`), `1WWW` (`TrkA`), `3DI3` (`IL-7Ra`), `3MJG` (`PDGFRb`)
      - Tier 2 (`boltzprot1_hard_novel`, n=5): `3QKG` (`AMBP`), `1G13` (`GM2A`), `1JQD` (`HNMT`), `7AAH` (`MZB1`), `1NB0` (`RFK`)
      - Tier 3 (`post_may2026_denovo_crystal`, n=3): `9S83` (`LEDGF PWWP + JUBO4`), `9RVK` (`Aurora-A + DBL5`), `9TLV` (`MCL-1 + Sc-aMCL1`)
    """
    target_pdb_dir = out_dir / "target_pdbs"
    if target_pdb_dir.exists():
        for old_p in target_pdb_dir.glob("*.pdb"):
            old_p.unlink()
    target_pdb_dir.mkdir(parents=True, exist_ok=True)

    raw_specs = [
        # Tier 1: RFD4-Proteina / Proteina-Complexa / ProtDBench Canonical Targets (n = 4)
        (
            "s3b_ep01",
            "rfd4_proteina_classic",
            "PDL1_IgV_ProtDBench",
            "5O45",
            "2017-07-26",
            "A",
            None,
            "pdl1_canonical",
            ["SDLGKKLLEAARAGQDDEVRILMANGADVNA", "DDFKKLLERAKKGDDEVRKLMANGADVNA"],
        ),
        (
            "s3b_ep02",
            "rfd4_proteina_classic",
            "TrkA_d5_ProtDBench",
            "1WWW",
            "2001-09-19",
            "X",
            "V",
            None,
            [],
        ),
        (
            "s3b_ep03",
            "rfd4_proteina_classic",
            "IL7Ra_ECD_ProtDBench",
            "3DI3",
            "2009-04-21",
            "B",
            "A",
            None,
            [],
        ),
        (
            "s3b_ep04",
            "rfd4_proteina_classic",
            "PDGFRb_D2D3_ProteinaComplexa",
            "3MJG",
            "2010-05-19",
            "X",
            "B",
            None,
            [],
        ),
        # Tier 2: BoltzProt-1 / BoltzGen (June 2026) Hard Novel Targets — Zero prior PDB binders, wet-lab nM proven (n = 5)
        (
            "s3b_ep05",
            "boltzprot1_hard_novel",
            "AMBP_BoltzProt1_Novel",
            "3QKG",
            "2011-05-04",
            "A",
            None,
            "surface_patch",
            [],
        ),
        (
            "s3b_ep06",
            "boltzprot1_hard_novel",
            "GM2A_BoltzProt1_Novel",
            "1G13",
            "2001-01-10",
            "A",
            None,
            "surface_patch",
            [],
        ),
        (
            "s3b_ep07",
            "boltzprot1_hard_novel",
            "HNMT_BoltzProt1_Novel",
            "1JQD",
            "2001-10-17",
            "A",
            None,
            "surface_patch",
            [],
        ),
        (
            "s3b_ep08",
            "boltzprot1_hard_novel",
            "MZB1_BoltzProt1_Novel",
            "7AAH",
            "2021-03-31",
            "A",
            None,
            "surface_patch",
            [],
        ),
        (
            "s3b_ep09",
            "boltzprot1_hard_novel",
            "RFK_BoltzProt1_Novel",
            "1NB0",
            "2003-02-04",
            "A",
            None,
            "surface_patch",
            [],
        ),
        # Tier 3: Post-May-2026 Wet-Lab De Novo Minibinder Crystal Structures (n = 3, 0% pre-cutoff binder SeqID!)
        (
            "s3b_ep10",
            "post_may2026_denovo_crystal",
            "LEDGF_PWWP_JUBO4_Aug2026",
            "9S83",
            "2026-08-05",
            "B",  # 89 AA LEDGF PWWP target; Chain A = 74 AA de novo minibinder JUBO4
            "A",
            None,
            [],
        ),
        (
            "s3b_ep11",
            "post_may2026_denovo_crystal",
            "AuroraA_DBL5_Jul2026",
            "9RVK",
            "2026-07-22",
            "B",  # 264 AA Aurora-A target; Chain A = 60 AA de novo minibinder DBL5
            "A",
            None,
            [],
        ),
        (
            "s3b_ep12",
            "post_may2026_denovo_crystal",
            "MCL1_ScHairpin_Aug2026",
            "9TLV",
            "2026-08-19",
            "B",  # 149 AA MCL-1 target; Chain A = 69 AA de novo coiled-coil minibinder
            "A",
            None,
            [],
        ),
    ]

    epitope_targets: List[EpitopeTarget] = []

    for tid, tier, name, pid, rdate, tch, pch, hs_mode, extra_forb in raw_specs:
        chs = parse_cif_all_protein_chains((raw_cif_dir / f"{pid}.cif").read_text(encoding="utf-8"))
        ta = chs[tch]
        write_backbone_pdb(ta["sequence"], ta["n"], ta["ca"], ta["c"], ta["o"], target_pdb_dir / f"{tid}.pdb")

        if pch and pch in chs:
            tb = chs[pch]
            dmat = np.linalg.norm(ta["cb"][:, None, :] - tb["cb"][None, :, :], axis=-1)
            min_d = np.min(dmat, axis=1)
            cand_0idx = list(np.argsort(min_d))
            hotspots_1idx: List[int] = []
            for idx0 in cand_0idx:
                if min_d[idx0] < 9.0 and all(abs((idx0 + 1) - h) >= 2 for h in hotspots_1idx):
                    hotspots_1idx.append(int(idx0 + 1))
                if len(hotspots_1idx) == 5:
                    break
            hotspots_1idx.sort()
            p_seq = tb["sequence"]
            p_ca = np.round(tb["ca"], 3).tolist()
            p_cb = np.round(tb["cb"], 3).tolist()
        elif hs_mode == "pdl1_canonical":
            s_pdl1 = ta["sequence"]
            i_liv = s_pdl1.index("LIVYWEM")
            i_cmi = s_pdl1.index("CMISYGGADY")
            hotspots_1idx = sorted([i_liv + 2, i_liv + 4, i_cmi + 2, i_cmi + 8, i_cmi + 10])
            p_seq = "MDEEVKRKLEALVKKYNPDAVLVIKELVKKFEEKLAQMGDPNTVRKLIEALKKQGWYPEELK"
            p_ca = []
            p_cb = []
        else:
            hotspots_1idx = select_surface_hydrophobic_patch_hotspots(ta["sequence"], ta["ca"], ta["cb"])
            p_seq = ""
            p_ca = []
            p_cb = []

        ep = EpitopeTarget(
            target_id=tid,
            tier=tier,
            name=name,
            pdb_id=pid,
            release_date=rdate,
            target_chain=tch,
            target_sequence=ta["sequence"],
            epitope_hotspots_1idx=hotspots_1idx,
            binder_min_len=55,
            binder_max_len=85,
            target_ca_coords=np.round(ta["ca"], 3).tolist(),
            target_cb_coords=np.round(ta["cb"], 3).tolist(),
            heldout_partner_sequence=p_seq,
            heldout_partner_ca_coords=p_ca,
            heldout_partner_cb_coords=p_cb,
            forbidden_sequences=extra_forb,
        )
        epitope_targets.append(ep)

    (out_dir / "epitope_binder_manifest.json").write_text(
        json.dumps(
            {
                "benchmark": "fable-protein-bench-stage3b-part2-closed-loop-binders",
                "version": "2.0",
                "budget_shots_per_target": 16,
                "n_targets": len(epitope_targets),
                "tier_counts": {
                    "rfd4_proteina_classic": sum(1 for e in epitope_targets if e.tier == "rfd4_proteina_classic"),
                    "boltzprot1_hard_novel": sum(1 for e in epitope_targets if e.tier == "boltzprot1_hard_novel"),
                    "post_may2026_denovo_crystal": sum(1 for e in epitope_targets if e.tier == "post_may2026_denovo_crystal"),
                },
                "targets": [asdict(e) for e in epitope_targets],
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    p1_lines = []
    for ep in epitope_targets:
        pr = build_binder_initial_prompt(ep)
        p1_lines.append(
            json.dumps(
                {
                    "target_id": ep.target_id,
                    "task": "stage3b_binder_round1",
                    "tier": ep.tier,
                    "length": len(ep.target_sequence),
                    "prompt": pr,
                }
            )
        )
    (prompts_dir / "stage3b_binder_round1_prompts.jsonl").write_text("\n".join(p1_lines) + "\n", encoding="utf-8")
    (prompts_dir / "web_chat_spotcheck" / "s3b_ep03_closed_loop_binder_shot1.txt").write_text(
        build_binder_initial_prompt(epitope_targets[9]), encoding="utf-8"
    )

    return epitope_targets


def run_stage3b_calibration_and_plot(
    part1_specs: List[DesignTaskSpec],
    epitope_targets: List[EpitopeTarget],
    pred_dir: Path,
    results_dir: Path,
    figures_dir: Path,
) -> Dict[str, Any]:
    results_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    # ==========================================================================
    # Part 1 Calibration Controls (`N = 40` Monomer + Motif Tasks)
    # ==========================================================================
    rng = random.Random(20261120)
    p1_summary: Dict[str, Any] = {}

    def _make_diverse_scaffold(length: int, seed_offset: int) -> str:
        # Realistic high-entropy soluble globular composition with zero periodic k-mer repeats
        alphabet_soluble = list("ADEFGHIKLMNPQRSTVWY" * 8)
        local_rng = random.Random(1000 + seed_offset)
        local_rng.shuffle(alphabet_soluble)
        out_chars: List[str] = []
        for ch in alphabet_soluble:
            if len(out_chars) >= 2 and out_chars[-1] == ch and out_chars[-2] == ch:
                continue
            out_chars.append(ch)
            if len(out_chars) >= length:
                break
        return "".join(out_chars)

    controls_p1 = [
        "PositiveControl_ValidDeNovo",
        "Cheat_MemorizedParent",
        "Cheat_EAAAK_RepeatRod",
        "Negative_WrongTopologySwap",
    ]

    for cname in controls_p1:
        grades = []
        for idx_s, spec in enumerate(part1_specs):
            mid_l = (spec.min_length + spec.max_length) // 2
            if cname == "PositiveControl_ValidDeNovo":
                if spec.category == "fold_conditioned":
                    seq = _make_diverse_scaffold(mid_l, idx_s)
                    h_lo, h_hi, e_lo, e_hi = spec.ss_bounds
                    n_h = int(round(mid_l * ((h_lo + h_hi) / 2.0)))
                    n_e = int(round(mid_l * ((e_lo + e_hi) / 2.0)))
                    ss = ("H" * n_h + "E" * n_e + "C" * mid_l)[:mid_l]
                    ca = np.array([[4.2 * np.cos(0.6 * k), 4.2 * np.sin(0.6 * k), 0.14 * k] for k in range(mid_l)])
                    g = grade_monomer_design(spec, seq, ca, plddt_mean=87.0, ss_q3=ss, tm_to_target_fold=0.74, mpnn_sc_rmsd=1.28, external_max_seqid_pct=18.5)
                else:
                    # Build exact motif-scaffolded sequence with exact 3D motif geometry
                    parts = []
                    pos_cursor = 0
                    m_positions: List[int] = []
                    for seg_i, seg in enumerate(spec.motif_segments):
                        lnk = (seg.min_linker_before + min(seg.max_linker_before, seg.min_linker_before + 10)) // 2
                        flank = _make_diverse_scaffold(lnk, idx_s * 10 + seg_i)
                        # Remove any accidental Cys in flank
                        flank = flank.replace("C", "A")
                        parts.append(flank)
                        pos_cursor += len(flank)
                        m_positions.extend(range(pos_cursor, pos_cursor + len(seg.sequence)))
                        parts.append(seg.sequence)
                        pos_cursor += len(seg.sequence)
                    rem = max(8, mid_l - pos_cursor)
                    parts.append(_make_diverse_scaffold(rem, idx_s * 10 + 9).replace("C", "A"))
                    seq = "".join(parts)[: spec.max_length]
                    ca = np.array([[4.2 * np.cos(0.6 * k), 4.2 * np.sin(0.6 * k), 0.14 * k] for k in range(len(seq))])
                    native_cat = np.concatenate([np.array(s.native_ca_coords) for s in spec.motif_segments], axis=0)
                    ca[m_positions] = native_cat + np.random.default_rng(idx_s).normal(0.0, 0.35, size=native_cat.shape)
                    g = grade_monomer_design(spec, seq, ca, plddt_mean=85.5, mpnn_sc_rmsd=1.35, external_max_seqid_pct=19.0)
            elif cname == "Cheat_MemorizedParent":
                pseq = spec.forbidden_parent_sequences[0]
                if len(pseq) >= spec.min_length:
                    seq = pseq[: min(len(pseq), spec.max_length)]
                else:
                    seq = (pseq + _make_diverse_scaffold(spec.min_length, idx_s))[: spec.min_length]
                ca = np.array([[4.0 * np.cos(0.6 * k), 4.0 * np.sin(0.6 * k), 0.14 * k] for k in range(len(seq))])
                g = grade_monomer_design(spec, seq, ca, plddt_mean=93.0, tm_to_target_fold=0.92, mpnn_sc_rmsd=0.75, external_max_seqid_pct=96.0)
            elif cname == "Cheat_EAAAK_RepeatRod":
                seq = ("AEAAAKEAAAKEAAAK" * 10)[:mid_l]
                # Extended single-helix rod (`Rg` huge!)
                ca = np.array([[1.5 * k, 2.2 * np.cos(1.7 * k), 2.2 * np.sin(1.7 * k)] for k in range(mid_l)])
                g = grade_monomer_design(spec, seq, ca, plddt_mean=91.0, tm_to_target_fold=0.35, mpnn_sc_rmsd=1.10, external_max_seqid_pct=22.0)
            else:  # Negative_WrongTopologySwap
                seq = _make_diverse_scaffold(mid_l, idx_s + 500)
                ca = np.array([[4.2 * np.cos(0.6 * k), 4.2 * np.sin(0.6 * k), 0.14 * k] for k in range(mid_l)])
                # Wrong SS (`100% H` when beta requested, `100% E` when helix requested) + poor motif RMSD
                wrong_ss = "E" * mid_l if spec.subfamily == "4_helix_bundle" else "H" * mid_l
                g = grade_monomer_design(spec, seq, ca, plddt_mean=84.0, ss_q3=wrong_ss, tm_to_target_fold=0.32, mpnn_sc_rmsd=1.60, external_max_seqid_pct=17.0)
            grades.append(g)

        p1_summary[cname] = {
            "n_tasks": len(grades),
            "gate1_hygiene_rate": round(float(np.mean([x["gate1_hygiene"] for x in grades])), 4),
            "gate2_foldable_rate": round(float(np.mean([x["gate2_foldable_globular"] for x in grades])), 4),
            "gate3_mpnn_sc_rate": round(float(np.mean([x["gate3_mpnn_self_consistent"] for x in grades])), 4),
            "gate4_adherence_rate": round(float(np.mean([x["gate4_prompt_adherence"] for x in grades])), 4),
            "gate5_novelty_rate": round(float(np.mean([x["gate5_novelty_lt30"] for x in grades])), 4),
            "design_success_rate": round(float(np.mean([x["all_5_gates_pass"] for x in grades])), 4),
        }

    # Also grade any live LLM / ESM3 JSONLs (`*_stage3b.jsonl`) on Gate 1 + Gate 5 offline (or full 3D if Modal JSON exists)
    if pred_dir.exists():
        for pf in sorted(pred_dir.glob("*_stage3b.jsonl")):
            tag = pf.stem.replace("_stage3b", "")
            raw_by_id = {}
            for ln in pf.read_text(encoding="utf-8").splitlines():
                if ln.strip():
                    o = json.loads(ln)
                    raw_by_id[str(o.get("target_id"))] = o
            grades_live = []
            for s in part1_specs:
                rec = raw_by_id.get(s.task_id, {})
                seq_p, _rat, _ok = parse_designed_sequence(str(rec.get("text") or ""))
                grades_live.append(grade_monomer_design(s, seq_p))
            p1_summary[tag] = {
                "n_tasks": len(grades_live),
                "gate1_hygiene_rate": round(float(np.mean([x["gate1_hygiene"] for x in grades_live])), 4),
                "gate5_novelty_rate": round(float(np.mean([x["gate5_novelty_lt30"] for x in grades_live])), 4),
            }

    # ==========================================================================
    # Part 2 Closed-Loop vs Open-Loop `Shots-to-Hit` Calibration (`N = 8`, `K = 16`)
    # ==========================================================================
    oracle = SurrogatePhysicsBinderOracle()
    budget_k = 16
    aa_pool = list("ADEFGHIKLMNPQRSTVWY")  # Cys-free soluble design alphabet

    def _random_seed_binder(seed_rng: random.Random, length: int = 68) -> str:
        return "".join(seed_rng.choice(aa_pool) for _ in range(length))

    p2_methods: Dict[str, Any] = {}

    def _semi_structured_proposal(seed_rng: random.Random) -> str:
        # Partial amphipathic 3-helix miniprotein proposal (realistic Shot-1 prior before AF2 feedback)
        chars: List[str] = []
        for pos in range(66):
            if pos in (21, 22, 44, 45):
                chars.append(seed_rng.choice("GPNDS"))
            elif pos in (2, 5, 6, 9, 12, 13, 16, 19, 25, 28, 29, 32, 35, 36, 39, 42, 47, 50, 51, 54, 57, 58, 61, 64) and seed_rng.random() < 0.36:
                chars.append(seed_rng.choice("LIVFMYA"))
            else:
                chars.append(seed_rng.choice("DEKRQNSTAVYHFW"))
        return "".join(chars)

    # Method 1: Closed-Loop MCMC Mutational Hill-Climber (uses oracle feedback every round!)
    mcmc_campaigns = []
    core_set = {2, 5, 6, 9, 12, 13, 16, 19, 25, 28, 29, 32, 35, 36, 39, 42, 47, 50, 51, 54, 57, 58, 61, 64}
    iface_set = {1, 8, 15, 27, 31, 34}
    for idx_e, ep in enumerate(epitope_targets):
        mrng = random.Random(7000 + idx_e)
        best_seq = _semi_structured_proposal(mrng)
        best_sc = -1.0
        best_fb: BinderShotFeedback | None = None
        hist: List[BinderShotFeedback] = []
        for r in range(1, budget_k + 1):
            if r == 1:
                cand = best_seq
            else:
                seq_list = list(best_seq)
                # Diagnostic-conditioned mutation: if monomer unstable -> pack core/turns; if hotspots weak -> tune interface!
                focus_core = (best_fb is not None) and (best_fb.binder_plddt < 81.0 or best_fb.monomer_rmsd > 2.0)
                n_mut = mrng.randint(3, 5)
                for _ in range(n_mut):
                    if focus_core and mrng.random() < 0.65:
                        pos = mrng.choice(sorted(core_set))
                        seq_list[pos] = mrng.choice("LIVFMV")
                    elif mrng.random() < 0.45:
                        pos = mrng.choice(sorted(iface_set))
                        seq_list[pos] = mrng.choice("FWYDEKRQN")
                    else:
                        pos = mrng.randrange(len(seq_list))
                        if pos in (21, 22, 44, 45):
                            seq_list[pos] = mrng.choice("NDST")
                        elif pos not in core_set and pos not in iface_set:
                            seq_list[pos] = mrng.choice("DEKRQNSTAH")
                cand = "".join(seq_list)
            fb = oracle.evaluate_binder(ep, cand, round_idx=r)
            hist.append(fb)
            if fb.novelty_ok and fb.composite_score >= best_sc:
                best_sc = fb.composite_score
                best_seq = cand
                best_fb = fb
        mcmc_campaigns.append(evaluate_binder_campaign(ep.target_id, hist, budget_k=budget_k))

    # Method 2: Open-Loop Blind Multi-Shot (`Pass@16` from the EXACT SAME proposal prior, zero feedback!)
    open_campaigns = []
    for idx_e, ep in enumerate(epitope_targets):
        orng = random.Random(9000 + idx_e)
        hist_o = [oracle.evaluate_binder(ep, _semi_structured_proposal(orng), round_idx=r) for r in range(1, budget_k + 1)]
        open_campaigns.append(evaluate_binder_campaign(ep.target_id, hist_o, budget_k=budget_k))

    # Method 3: Held-Out Partner / Target Copy-Paste Cheat (tests `<25%` SeqID anti-memorization block!)
    cheat_campaigns = []
    for ep in epitope_targets:
        pseq = (ep.heldout_partner_sequence or ep.target_sequence)[:80]
        if len(pseq) < 55:
            pseq = (pseq + "GSDEELKRKLEALVKKYNPDAVLVIK")[:60]
        hist_c = [oracle.evaluate_binder(ep, pseq, round_idx=r) for r in range(1, budget_k + 1)]
        cheat_campaigns.append(evaluate_binder_campaign(ep.target_id, hist_c, budget_k=budget_k))

    for m_label, camps in (
        ("ClosedLoop_OracleHillClimber_16shot", mcmc_campaigns),
        ("OpenLoop_BlindSampling_PassAt16", open_campaigns),
        ("Cheat_HeldoutPartnerCopyPaste", cheat_campaigns),
    ):
        hits = [c["hit_achieved"] for c in camps]
        shots_hit = [c["shots_to_first_hit"] for c in camps if c["shots_to_first_hit"] is not None]
        pass_curve = np.mean([c["pass_at_k_curve"] for c in camps], axis=0).tolist()
        traj_mean = np.mean([c["running_best_composite"] for c in camps], axis=0).tolist()
        traj_sem = (np.std([c["running_best_composite"] for c in camps], axis=0) / np.sqrt(len(camps))).tolist()
        p2_methods[m_label] = {
            "n_targets": len(camps),
            "hit_rate": round(float(np.mean(hits)), 4),
            "mean_shots_to_hit_among_hits": round(float(np.mean(shots_hit)), 2) if shots_hit else None,
            "mean_best_iptm": round(float(np.mean([c["best_iptm_on_epitope"] for c in camps])), 4),
            "mean_closed_loop_gain": round(float(np.mean([c["closed_loop_trajectory_gain"] for c in camps])), 4),
            "pass_at_k_curve": [round(x, 4) for x in pass_curve],
            "trajectory_mean": [round(x, 4) for x in traj_mean],
            "trajectory_sem": [round(x, 4) for x in traj_sem],
            "per_target": camps,
        }

    # Load any live LLM / BindCraft campaign files (`outputs/predictions/stage3b_binder_campaign_*.json`)
    if pred_dir.exists():
        for cf in sorted(pred_dir.glob("stage3b_binder_campaign_*.json")):
            obj = json.loads(cf.read_text(encoding="utf-8"))
            p2_methods[obj["model_tag"]] = obj["summary"]

    report = {
        "stage3b_part1_monomer_motif": p1_summary,
        "stage3b_part2_closed_loop_binders": p2_methods,
    }
    (results_dir / "stage3b_evaluation_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    plot_stage3b_overview(
        p1_summary=p1_summary,
        epitope_targets=epitope_targets,
        p2_methods=p2_methods,
        out_png=figures_dir / "stage3b_overview.png",
    )
    return report


def plot_stage3b_overview(
    p1_summary: Dict[str, Any],
    epitope_targets: List[EpitopeTarget],
    p2_methods: Dict[str, Any],
    out_png: Path,
) -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.size": 10,
            "axes.titlesize": 11.5,
            "axes.titleweight": "bold",
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )
    fig, axes = plt.subplots(2, 2, figsize=(13.8, 9.8), dpi=200)
    fig.patch.set_facecolor("#FAFAFC")

    # Panel A: Part 1 (`N = 40` Monomer + Motif Design) — 5-Gate Anti-Cheat Filter Verification
    ax = axes[0, 0]
    ax.set_facecolor("#FFFFFF")
    gate_keys = [
        "gate1_hygiene_rate",
        "gate2_foldable_rate",
        "gate3_mpnn_sc_rate",
        "gate4_adherence_rate",
        "gate5_novelty_rate",
        "design_success_rate",
    ]
    gate_labels = ["Gate 1\nComplexity", "Gate 2\npLDDT & Rg", "Gate 3\nMPNN scRMSD", "Gate 4\nFold / Motif", "Gate 5\n<30% Novelty", "ALL 5\nSUCCESS"]
    ctrl_meta = [
        ("PositiveControl_ValidDeNovo", "Valid De Novo Control", "#0D9488"),
        ("Cheat_MemorizedParent", "Memorized Natural Cheat", "#DC2626"),
        ("Cheat_EAAAK_RepeatRod", "EAAAK Repeat-Helix Rod", "#F59E0B"),
        ("Negative_WrongTopologySwap", "Wrong Fold / Motif Swap", "#6366F1"),
    ]
    x = np.arange(len(gate_keys))
    w = 0.78 / len(ctrl_meta)
    for i_c, (ckey, clbl, ccol) in enumerate(ctrl_meta):
        vals = [p1_summary[ckey][gk] for gk in gate_keys]
        off = (i_c - (len(ctrl_meta) - 1) / 2.0) * w
        ax.bar(x + off, vals, width=w * 0.92, label=clbl, color=ccol, alpha=0.88)
    ax.axvline(4.5, color="#334155", linestyle="--", linewidth=1.0)
    ax.set_xticks(x)
    ax.set_xticklabels(gate_labels, fontsize=8.6)
    ax.set_ylim(0, 1.34)
    ax.set_ylabel("Pass Fraction Across N = 40 Tasks")
    ax.set_title("A. Stage 3B Part 1: 5-Gate Anti-Cheat Verification (N = 40 Prompts)")
    ax.legend(loc="upper center", ncol=2, fontsize=7.6, frameon=True)
    ax.grid(True, axis="y", alpha=0.22)

    # Panel B: Published 2026 Specialist Pass Rates (`ProtDBench` AF2_Easy_SR %) on Tier-1 Targets + 3-Tier Overview
    ax = axes[0, 1]
    ax.set_facecolor("#FFFFFF")
    t1_targets = ["PD-L1\n(5O45)", "PDGFRβ\n(3MJG)", "TrkA\n(1WWW)", "IL-7Rα\n(3DI3)"]
    # Exact published values from `congliuUvA/ProtDBench` (`ALL_sr__10targets.csv` & `ALL_sr__cao5targets.csv`)
    protd_specialists = [
        ("BindCraft (AF2 Hallucination)", [67.30, 54.19, 51.45, 39.56], "#0D9488"),
        ("PXDesign (2026)", [45.33, 29.80, 23.55, 29.80], "#2563EB"),
        ("BoltzDesign-1", [31.91, 45.73, 34.77, 7.61], "#7C3AED"),
        ("BoltzGen", [14.26, 34.15, 28.65, 12.41], "#F59E0B"),
        ("RFdiffusion-3", [16.81, 26.94, 14.44, 6.47], "#64748B"),
    ]
    xb = np.arange(len(t1_targets))
    wb = 0.78 / len(protd_specialists)
    for idx_sp, (sp_lbl, sp_vals, sp_col) in enumerate(protd_specialists):
        off = (idx_sp - (len(protd_specialists) - 1) / 2.0) * wb
        ax.bar(xb + off, sp_vals, width=wb * 0.90, label=sp_lbl, color=sp_col, alpha=0.90)
    ax.set_xticks(xb)
    ax.set_xticklabels(t1_targets, fontsize=9.0)
    ax.set_ylim(0, 82.0)
    ax.set_ylabel("Published Single-Shot AF2 Pass Rate (%)")
    ax.set_title("B. Published ProtDBench Specialist Pass Rates (Tier-1 Targets)")
    ax.legend(loc="upper right", fontsize=7.6, frameon=True)
    ax.grid(True, axis="y", alpha=0.22)

    # Panel C: Closed-Loop vs Open-Loop `Pass@k` (`Shots-to-Hit` Curve over 1..16 Shots)
    ax = axes[1, 0]
    ax.set_facecolor("#FFFFFF")
    shots_x = np.arange(1, 17)
    curve_styles = [
        ("ClosedLoop_OracleHillClimber_16shot", "Closed-Loop Oracle Refinement (16 turns)", "#0D9488", "-", "o"),
        ("OpenLoop_BlindSampling_PassAt16", "Open-Loop Blind Sampling (Pass@16, no feedback)", "#2563EB", "--", "s"),
        ("Cheat_HeldoutPartnerCopyPaste", "Copy-Paste Partner Cheat (>=25% SeqID blocked)", "#DC2626", ":", "x"),
    ]
    for mkey, mlbl, mcol, ls, mk in curve_styles:
        if mkey not in p2_methods:
            continue
        ys = p2_methods[mkey]["pass_at_k_curve"]
        ax.plot(shots_x, ys, label=f"{mlbl} (Hit@16={ys[-1]:.2f})", color=mcol, linestyle=ls, marker=mk, markersize=4.8, linewidth=2.1)
    # Also overlay any live LLM/BindCraft runs if present
    for extra_k, extra_v in p2_methods.items():
        if extra_k not in {c[0] for c in curve_styles}:
            ys = extra_v["pass_at_k_curve"]
            ax.plot(shots_x, ys, label=f"{extra_k} (Hit@16={ys[-1]:.2f})", linewidth=2.3, marker="D", markersize=5)
    ax.set_xticks(shots_x)
    ax.set_ylim(-0.04, 1.08)
    ax.set_xlabel("Oracle Evaluation Budget (Shot k = 1 .. 16)")
    ax.set_ylabel("Cumulative Epitope Hit Rate (Pass@k)")
    ax.set_title("C. Shots-to-Hit: Closed-Loop Feedback vs. Blind Multi-Shot")
    ax.legend(loc="upper left", fontsize=8.0, frameon=True)
    ax.grid(True, alpha=0.22)

    # Panel D: Running-Best Interface Composite Score Trajectory (Rounds 1..16)
    ax = axes[1, 1]
    ax.set_facecolor("#FFFFFF")
    for mkey, mlbl, mcol, ls, _mk in curve_styles[:2]:
        mu = np.array(p2_methods[mkey]["trajectory_mean"])
        se = np.array(p2_methods[mkey]["trajectory_sem"])
        gain = p2_methods[mkey]["mean_closed_loop_gain"]
        ax.plot(shots_x, mu, label=f"{mlbl} (Δgain=+{gain:.2f})", color=mcol, linestyle=ls, linewidth=2.2)
        ax.fill_between(shots_x, mu - se, mu + se, color=mcol, alpha=0.16)
    ax.axhline(0.76, color="#64748B", linestyle="--", linewidth=1.0, label="Approximate BindCraft hit regime")
    ax.set_xticks(shots_x)
    ax.set_ylim(0.30, 0.95)
    ax.set_xlabel("Closed-Loop Turn / Oracle Shot (1 .. 16)")
    ax.set_ylabel("Running-Best Interface Score (ipTM + iPAE + Hotspots)")
    ax.set_title("D. Iterative In-Context Structural Optimization Trajectory")
    ax.legend(loc="lower right", fontsize=8.0, frameon=True)
    ax.grid(True, alpha=0.22)

    fig.suptitle(
        "Stage 3B Calibration: De Novo Monomer/Motif Generation (Part 1) & Closed-Loop AF2 Binder Design (Part 2)",
        fontsize=13.2,
        fontweight="bold",
        y=0.985,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(out_png, dpi=200)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build & evaluate Stage 3B (De Novo Design & Closed-Loop Binders).")
    parser.add_argument("--eval-only", action="store_true")
    args = parser.parse_args()

    raw_cif_dir = REPO_ROOT / "data" / "raw_stage3b_cache" / "cifs"
    s1_gt_dir = REPO_ROOT / "data" / "stage1_benchmark" / "ground_truth"
    out_dir = REPO_ROOT / "data" / "stage3b_design"
    prompts_dir = REPO_ROOT / "outputs" / "prompts"
    pred_dir = REPO_ROOT / "outputs" / "predictions"
    results_dir = REPO_ROOT / "outputs" / "results"
    figures_dir = REPO_ROOT / "outputs" / "figures"

    if not args.eval_only:
        print("Building Stage 3B Part 1 (N = 40 Monomer Fold-Conditioned & Motif Scaffolding tasks)...", flush=True)
        p1_specs = build_part1_monomer_and_motif(raw_cif_dir, s1_gt_dir, out_dir, prompts_dir)
        print(f"  Built {len(p1_specs)} Part-1 design tasks.", flush=True)

        print("Building Stage 3B Part 2 (N = 8 Closed-Loop Epitope Binder Targets)...", flush=True)
        ep_targets = build_part2_epitope_binders(raw_cif_dir, out_dir, prompts_dir)
        for ep in ep_targets:
            print(
                f"  [{ep.target_id}] {ep.pdb_id} ({ep.release_date}, {ep.tier}): "
                f"target={len(ep.target_sequence)} AA, hotspots={ep.epitope_hotspots_1idx}, "
                f"heldout_partner={len(ep.heldout_partner_sequence)} AA",
                flush=True,
            )
    else:
        m1 = json.loads((out_dir / "monomer_motif_manifest.json").read_text(encoding="utf-8"))
        p1_specs = []
        for t in m1["tasks"]:
            segs = [MotifSegmentSpec(**sg) for sg in t.pop("motif_segments", [])]
            t["ss_bounds"] = tuple(t["ss_bounds"])
            p1_specs.append(DesignTaskSpec(**t, motif_segments=segs))
        m2 = json.loads((out_dir / "epitope_binder_manifest.json").read_text(encoding="utf-8"))
        ep_targets = [EpitopeTarget(**e) for e in m2["targets"]]

    report = run_stage3b_calibration_and_plot(p1_specs, ep_targets, pred_dir, results_dir, figures_dir)
    print("\n=== Stage 3B Part 1 (Monomer & Motif 5-Gate Calibration) ===")
    for k, v in report["stage3b_part1_monomer_motif"].items():
        print(f"  {k:<32s} | Success={v.get('design_success_rate', 'N/A')} | G1={v['gate1_hygiene_rate']:.2f}, G5_Novelty={v['gate5_novelty_rate']:.2f}")

    print("\n=== Stage 3B Part 2 (Closed-Loop vs Open-Loop Epitope Binders, K=16) ===")
    for k, v in report["stage3b_part2_closed_loop_binders"].items():
        print(
            f"  {k:<38s} | Hit@16={v['hit_rate']:.3f} | MeanShots={v['mean_shots_to_hit_among_hits']} | "
            f"Best_ipTM={v['mean_best_iptm']:.3f} | ClosedLoopGain=+{v['mean_closed_loop_gain']:.3f}"
        )


if __name__ == "__main__":
    main()
