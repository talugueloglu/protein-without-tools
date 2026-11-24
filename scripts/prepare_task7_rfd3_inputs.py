#!/usr/bin/env python3
"""Prepare clean 1-indexed PDB inputs and RFdiffusion3 contigs for all 40 Task 7 design tasks."""

from __future__ import annotations

import json
from pathlib import Path
import sys
from typing import Any, Dict, List

REPO = Path(__file__).resolve().parents[1]
if str(REPO / "src") not in sys.path:
    sys.path.insert(0, str(REPO / "src"))
if str(REPO / "scripts") not in sys.path:
    sys.path.insert(0, str(REPO / "scripts"))

from build_and_evaluate_stage3b import parse_cif_all_protein_chains, write_backbone_pdb
from protein_without_tools.pdb_utils import load_structure


def main() -> None:
    s3b_dir = REPO / "data" / "stage3b_design"
    raw_cifs = REPO / "data" / "raw_stage3b_cache" / "cifs"
    out_pdb_dir = REPO / "outputs" / "japanfold_cache" / "t7_motif_pdbs"
    out_pdb_dir.mkdir(parents=True, exist_ok=True)

    # Export clean 1-indexed Chain A N/CA/C/O PDBs for the 4 motif crystal parents
    motif_sources: Dict[str, Dict[str, Any]] = {}
    for cid in ("1CLL", "1AAY", "5P21", "9S83"):
        ch = parse_cif_all_protein_chains((raw_cifs / f"{cid}.cif").read_text(encoding="utf-8"))["A"]
        pdb_p = out_pdb_dir / f"motif_src_{cid}.pdb"
        write_backbone_pdb(ch["sequence"], ch["n"], ch["ca"], ch["c"], ch["o"], pdb_p)
        motif_sources[cid] = {"seq": ch["sequence"], "pdb_path": str(pdb_p), "length": len(ch["sequence"])}
        print(f"Prepared {pdb_p.name}: L={len(ch['sequence'])}", flush=True)

    subfam_to_cid = {
        "EF_hand_Ca_loop": "1CLL",
        "C2H2_Zn_finger": "1AAY",
        "WalkerA_Ploop": "5P21",
    }

    manifest = json.loads((s3b_dir / "monomer_motif_manifest.json").read_text(encoding="utf-8"))["tasks"]
    specs_out: List[Dict[str, Any]] = []

    for t in manifest:
        tid = t["task_id"]
        cat = t["category"]
        subfam = t["subfamily"]
        l_min, l_max = int(t["min_length"]), int(t["max_length"])
        l_mid = (l_min + l_max) // 2

        if cat == "fold_conditioned":
            ref_p = s3b_dir / t["reference_pdb"]
            st = load_structure(ref_p, t["reference_pdb"])
            l_ref = st.length
            # Write clean 1..L_ref PDB so contig residue numbers are strictly 1..L_ref
            clean_p = out_pdb_dir / f"fold_src_{ref_p.stem}.pdb"
            if not clean_p.exists():
                write_backbone_pdb(st.sequence, st.coords_n, st.coords_ca, st.coords_c, st.coords_o, clean_p)

            if l_mid >= l_ref + 6:
                kept = l_ref - 4
                rem = l_mid - kept
                n_pre = max(3, rem // 2)
                n_post = rem - n_pre
                contig = f"{n_pre}, A3-{2 + kept}, {n_post}"
                total_len = n_pre + kept + n_post
            elif l_mid <= l_ref - 6:
                n_pre = 4
                n_post = 6
                kept = l_mid - n_pre - n_post
                contig = f"{n_pre}, A3-{2 + kept}, {n_post}"
                total_len = n_pre + kept + n_post
            else:
                kept = l_mid - 10
                contig = f"5, A4-{3 + kept}, 5"
                total_len = kept + 10

            assert l_min <= total_len <= l_max, f"{tid}: {total_len} outside [{l_min}, {l_max}]"
            specs_out.append(
                {
                    "task_id": tid,
                    "category": cat,
                    "subfamily": subfam,
                    "pdb_path": str(clean_p),
                    "contig": contig,
                    "expected_length": total_len,
                    "fixed_0idx": [],
                    "fixed_motif_seqs": [],
                    "max_allowed_cys": int(t.get("max_allowed_cys", 1)),
                    "forbidden_parent_sequences": t.get("forbidden_parent_sequences", []),
                }
            )
        else:
            cid = subfam_to_cid.get(subfam, "9S83")
            msrc = motif_sources[cid]
            pseq = msrc["seq"]
            segs = t["motif_segments"]
            if len(segs) == 1:
                s1 = segs[0]
                m1 = s1["sequence"]
                i1 = pseq.index(m1) + 1  # 1-indexed in clean PDB
                j1 = i1 + len(m1) - 1
                n_pre = min(s1["max_linker_before"], max(s1["min_linker_before"], (l_mid - len(m1)) // 2))
                n_post = l_mid - n_pre - len(m1)
                if n_post < 8:
                    shift = 8 - n_post
                    n_pre = max(s1["min_linker_before"], n_pre - shift)
                    n_post = l_mid - n_pre - len(m1)
                contig = f"{n_pre}, A{i1}-{j1}, {n_post}"
                fixed_0idx = list(range(n_pre, n_pre + len(m1)))
                total_len = n_pre + len(m1) + n_post
            else:
                s1, s2 = segs[0], segs[1]
                m1, m2 = s1["sequence"], s2["sequence"]
                i1 = pseq.index(m1) + 1
                j1 = i1 + len(m1) - 1
                i2 = pseq.index(m2, i1) + 1
                j2 = i2 + len(m2) - 1
                n_link = (s2["min_linker_before"] + s2["max_linker_before"]) // 2
                rem = l_mid - len(m1) - len(m2) - n_link
                n_pre = min(s1["max_linker_before"], max(s1["min_linker_before"], rem // 2))
                n_post = l_mid - n_pre - len(m1) - n_link - len(m2)
                if n_post < 8:
                    shift = 8 - n_post
                    n_pre = max(s1["min_linker_before"], n_pre - shift)
                    n_post = l_mid - n_pre - len(m1) - n_link - len(m2)
                contig = f"{n_pre}, A{i1}-{j1}, {n_link}, A{i2}-{j2}, {n_post}"
                fixed_0idx = list(range(n_pre, n_pre + len(m1))) + list(
                    range(n_pre + len(m1) + n_link, n_pre + len(m1) + n_link + len(m2))
                )
                total_len = n_pre + len(m1) + n_link + len(m2) + n_post

            assert l_min <= total_len <= l_max, f"{tid}: {total_len} outside [{l_min}, {l_max}]"
            specs_out.append(
                {
                    "task_id": tid,
                    "category": cat,
                    "subfamily": subfam,
                    "pdb_path": msrc["pdb_path"],
                    "contig": contig,
                    "expected_length": total_len,
                    "fixed_0idx": fixed_0idx,
                    "fixed_motif_seqs": [s["sequence"] for s in segs],
                    "max_allowed_cys": int(t.get("max_allowed_cys", 1)),
                    "forbidden_parent_sequences": t.get("forbidden_parent_sequences", []),
                }
            )

    out_json = REPO / "outputs" / "japanfold_cache" / "t7_rfd3_specs.json"
    out_json.write_text(json.dumps(specs_out, indent=2), encoding="utf-8")
    print(f"Prepared {len(specs_out)} Task 7 RFdiffusion3 specs -> {out_json.name}", flush=True)
    for sp in specs_out[:3] + specs_out[24:28]:
        print(f"  {sp['task_id']} ({sp['subfamily']}): L={sp['expected_length']} contig={sp['contig']}", flush=True)


if __name__ == "__main__":
    main()
