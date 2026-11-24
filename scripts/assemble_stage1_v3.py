#!/usr/bin/env python3
"""Assemble Stage 1 v3 (`N = 100` targets) prioritizing genuinely structurally novel
(`TM-score < 0.50` / `< 0.55`) post-May-2026 structures while retaining clean controls.

Combines:
  1. Existing TM-verified targets from Stage 1 v2 (`data/stage1_benchmark/manifest.json`)
  2. Newly mined & TM-verified targets (`data/raw_pdb_cache/mined_novel_candidates_verified.json`)

Produces a clean, non-redundant `N = 100` benchmark stratified by TRUE 3D TM-score & SeqID:
  - `pre_cutoff_control` (15): Pre-2025 classic controls (`6MRR`, `2A3D`, `6CZH`, `1UBQ`, ...)
  - `post_cutoff_struct_novel` (all available `PDB TM < 0.52`, sorted lowest TM first!)
  - `post_cutoff_de_novo` (`<30%` PDB & AFDB SeqID, sorted lowest PDB TM upward!)
  - `post_cutoff_pdb_novel` (`<30%` PDB SeqID, `>=30%` AFDB SeqID, dropping `>=30%` PDB re-crysts!)
  - `post_cutoff_homolog_gradient` (12): Clean `35-85%` PDB SeqID controls (incl. `@WUFang` `37NT`)
"""

from __future__ import annotations

from collections import Counter
import json
from pathlib import Path
import shutil
from typing import Any, Dict, List, Tuple

import numpy as np

from protein_without_tools.pdb_utils import (
    ProteinStructure,
    write_backbone_only_pdb,
    write_ca_trace_pdb,
)


def rehydrate_npz(npz_bytes_or_path: Path | Dict[str, Any], tid: str, pdb_id: str, chain_id: str) -> ProteinStructure:
    if isinstance(npz_bytes_or_path, Path):
        raw = dict(np.load(npz_bytes_or_path, allow_pickle=False))
    else:
        raw = npz_bytes_or_path
    return ProteinStructure(
        target_id=tid,
        pdb_id=pdb_id,
        chain_id=chain_id,
        sequence=str(raw["sequence"]),
        res_numbers=raw["res_numbers"].tolist(),
        coords_n=raw["coords_n"],
        coords_ca=raw["coords_ca"],
        coords_c=raw["coords_c"],
        coords_o=raw["coords_o"],
        coords_cb=raw["coords_cb"],
        phi_deg=raw["phi_deg"],
        psi_deg=raw["psi_deg"],
        q3=str(raw["q3"]),
        cb_dist_matrix=raw["cb_dist_matrix"],
        ca_coord_num=raw["ca_coord_num"],
    )


def main() -> None:
    repo = Path(__file__).resolve().parents[1]
    data_dir = repo / "data" / "stage1_benchmark"
    mined_path = repo / "data" / "raw_pdb_cache" / "mined_novel_candidates_verified.json"

    v2_manifest = json.loads((data_dir / "manifest.json").read_text(encoding="utf-8"))
    mined_list = json.loads(mined_path.read_text(encoding="utf-8")) if mined_path.exists() else []

    # Load all existing v2 structures into memory first so re-numbering never clobbers files!
    pool: List[Tuple[Dict[str, Any], Dict[str, Any]]] = []
    for rec in v2_manifest["targets"]:
        tid = rec["target_id"]
        npz_dict = {k: v.copy() for k, v in np.load(data_dir / "ground_truth" / f"{tid}.npz", allow_pickle=False).items()}
        r_copy = dict(rec)
        r_copy.pop("target_id", None)
        pool.append((r_copy, npz_dict))

    for mrec in mined_list:
        npz_p = Path(mrec["npz_path"])
        if not npz_p.exists():
            continue
        npz_dict = {k: v.copy() for k, v in np.load(npz_p, allow_pickle=False).items()}
        r_copy = dict(mrec)
        r_copy.pop("npz_path", None)
        r_copy.pop("entity_id", None)
        r_copy.pop("is_synth", None)
        r_copy["temporal_split"] = "post_cutoff"
        r_copy["size_bin"] = "compact_50_100" if r_copy["length"] <= 105 else "medium_101_260"
        pool.append((r_copy, npz_dict))

    # Partition into clean biological buckets using TRUE TM-score & combined PDB SeqID
    pre_controls = []
    struct_novel = []  # Post-cutoff with PDB TM < 0.52 AND PDB SeqID < 30%
    seq_novel_denovo = []  # Post-cutoff with PDB SeqID < 30%, AFDB SeqID < 30%, PDB TM >= 0.52
    seq_novel_afdb = []  # Post-cutoff with PDB SeqID < 30%, AFDB SeqID >= 30%, PDB TM >= 0.52
    homolog_controls = []  # Post-cutoff with 30% <= PDB SeqID <= 88% (plus 37NT D4)

    for rec, npz_d in pool:
        if rec["temporal_split"] == "pre_cutoff":
            rec["stratum"] = "pre_cutoff_control"
            pre_controls.append((rec, npz_d))
            continue

        ps = float(rec.get("pdb_precutoff_seqid_combined", rec.get("pdb_precutoff_seqid_max", 0.0)))
        afs = max(float(rec.get("afdb_max_seqid", 0.0)), float(rec.get("afdb_tm_seqid", 0.0)))
        ptm = float(rec.get("pdb_precutoff_tm_query", 1.0))

        if ps >= 30.0:
            # Drop accidental >90% re-crystallizations (e.g. Lysozyme 38BH, RAD52 32DI) EXCEPT @WUFang's 37NT anchor!
            if ps <= 88.0 or rec["pdb_id"] == "37NT":
                rec["stratum"] = "post_cutoff_homolog_gradient"
                homolog_controls.append((rec, npz_d))
            continue

        if ptm < 0.52:
            rec["stratum"] = "post_cutoff_struct_novel"
            struct_novel.append((rec, npz_d))
        elif afs < 30.0:
            rec["stratum"] = "post_cutoff_de_novo"
            seq_novel_denovo.append((rec, npz_d))
        else:
            rec["stratum"] = "post_cutoff_pdb_novel"
            seq_novel_afdb.append((rec, npz_d))

    # Sort every post-cutoff bucket from LOWEST PDB TM-score upward so we keep the most novel structures!
    struct_novel.sort(key=lambda x: (x[0]["pdb_precutoff_tm_query"], x[0]["afdb_tm_query"]))
    seq_novel_denovo.sort(key=lambda x: x[0]["pdb_precutoff_tm_query"])
    seq_novel_afdb.sort(key=lambda x: x[0]["pdb_precutoff_tm_query"])
    homolog_controls.sort(key=lambda x: (0 if x[0]["pdb_id"] == "37NT" else 1, x[0]["pdb_precutoff_seqid_combined"]))

    target_total = 100
    chosen: List[Tuple[Dict[str, Any], Dict[str, Any]]] = []
    chosen.extend(pre_controls[:15])
    chosen.extend(struct_novel)  # Keep 100% of genuinely structurally novel (`TM < 0.52`) structures!
    chosen.extend(homolog_controls[:12])

    rem = max(0, target_total - len(chosen))
    n_afdb = min(len(seq_novel_afdb), max(15, rem // 3))
    n_denovo = min(len(seq_novel_denovo), rem - n_afdb)
    chosen.extend(seq_novel_denovo[:n_denovo])
    chosen.extend(seq_novel_afdb[:n_afdb])

    # Top up to 100 if any bucket had spare capacity
    used_ids = {id(x[0]) for x in chosen}
    for extra_pool in (seq_novel_denovo, seq_novel_afdb, homolog_controls):
        for item in extra_pool:
            if len(chosen) >= target_total:
                break
            if id(item[0]) not in used_ids:
                chosen.append(item)
                used_ids.add(id(item[0]))

    # Write to staging dirs first so errors never clobber existing data!
    gt_tmp = data_dir / "ground_truth_tmp"
    bb_tmp = data_dir / "backbones_stripped_tmp"
    shutil.rmtree(gt_tmp, ignore_errors=True)
    shutil.rmtree(bb_tmp, ignore_errors=True)
    gt_tmp.mkdir(parents=True, exist_ok=True)
    bb_tmp.mkdir(parents=True, exist_ok=True)

    final_targets: List[Dict[str, Any]] = []
    for idx, (rec, npz_d) in enumerate(chosen, start=1):
        tid = f"s1_t{idx:03d}"
        rec["target_id"] = tid
        rec["size_bin"] = "compact_50_100" if rec["length"] <= 105 else "medium_101_260"
        np.savez_compressed(gt_tmp / f"{tid}.npz", **npz_d)
        st = rehydrate_npz(npz_d, tid, rec["pdb_id"], rec["chain_id"])
        write_ca_trace_pdb(st.sequence, st.coords_ca, gt_tmp / f"{tid}_ca.pdb")
        write_backbone_only_pdb(st, bb_tmp / f"{tid}.pdb")
        final_targets.append(rec)

    gt_dir = data_dir / "ground_truth"
    bb_dir = data_dir / "backbones_stripped"
    shutil.rmtree(gt_dir, ignore_errors=True)
    shutil.rmtree(bb_dir, ignore_errors=True)
    gt_tmp.rename(gt_dir)
    bb_tmp.rename(bb_dir)

    strata_counts = dict(Counter(r["stratum"] for r in final_targets))
    out_manifest = {
        "benchmark": "protein-without-tools-stage1-v3",
        "cutoff_date": "2026-05-01",
        "total_targets": len(final_targets),
        "strata_counts": strata_counts,
        "n_pre_cutoff": sum(1 for r in final_targets if r["temporal_split"] == "pre_cutoff"),
        "n_post_cutoff": sum(1 for r in final_targets if r["temporal_split"] == "post_cutoff"),
        "n_pdb_tm_below_050": sum(1 for r in final_targets if r["pdb_precutoff_tm_query"] < 0.50),
        "n_pdb_tm_below_055": sum(1 for r in final_targets if r["pdb_precutoff_tm_query"] < 0.55),
        "n_both_tm_below_050": sum(
            1 for r in final_targets if r["pdb_precutoff_tm_query"] < 0.50 and r["afdb_tm_query"] < 0.50
        ),
        "targets": final_targets,
    }
    (data_dir / "manifest.json").write_text(json.dumps(out_manifest, indent=2), encoding="utf-8")
    print(f"Assembled Stage 1 v3 (`N = {len(final_targets)}`):")
    print("  Strata:", strata_counts)
    print(
        f"  PDB TM < 0.50: {out_manifest['n_pdb_tm_below_050']} | "
        f"PDB TM < 0.55: {out_manifest['n_pdb_tm_below_055']} | "
        f"Both PDB & AFDB TM < 0.50: {out_manifest['n_both_tm_below_050']}"
    )


if __name__ == "__main__":
    main()
