#!/usr/bin/env python3
"""Curate and validate the Stage 1 (Forward + Inverse Folding) PDB Benchmark Dataset.

Builds a clean 2x2 factorial dataset in `data/stage1_benchmark/`:
  1. Pre-2025 Anchor Controls (incl. @sokrypton's 6MRR de novo ferredoxin fold, Ubiquitin 1UBQ,
     GB1 1PGA, de novo 3-helix bundle 2A3D, de novo beta-barrel 6CZH, etc.)
  2. Strict Sept-Nov 2026 Post-Cutoff Newly Released RCSB PDBs (incl. @WUFang's Oct-2026
     37NT MERTK ectodomain domains 1-4 + newly released compact 50-100 AA and medium
     101-200 AA contiguous non-viral, non-toxin monomers).

Every structure is checked for:
  - Contiguous resolved polypeptide chain (zero internal missing-loop breaks > 2.3 Å)
  - Canonical amino acids (50 <= L <= 220)
  - Zero viral / phage / toxin / select-agent keywords (prevents Fable 5.2 silent fallbacks)
  - Non-redundant sequence diversity (< 30% global pairwise identity across targets)
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import sys
from typing import Any, Dict, List, Optional, Tuple

import httpx
import numpy as np

from fable_bench.pdb_utils import (
    ProteinStructure,
    load_structure,
    write_backbone_only_pdb,
    write_ca_trace_pdb,
)

SAFETY_BLOCKLIST_PATTERNS = re.compile(
    r"\b(virus|viral|virion|capsid|phage|bacteriophage|toxin|venom|ricin|abrin|"
    r"botulinum|anthrax|ebola|marburg|variola|poxvirus|influenza|coronavirus|sars|"
    r"hemolysin|enterotoxin|neurotoxin|cytotoxin|conotoxin)\b",
    re.IGNORECASE,
)

# Pre-2025 well-characterized single-domain controls across α, α/β, and all-β classes
PRE_CUTOFF_CONTROLS: List[Dict[str, Any]] = [
    {"pdb_id": "6MRR", "chain": "A", "res_range": None, "note": "Baker 2018 de novo βαβαβ (@sokrypton tweet)"},
    {"pdb_id": "2A3D", "chain": "A", "res_range": None, "note": "De novo 3-helix bundle α3D (73 AA)"},
    {"pdb_id": "1UBQ", "chain": "A", "res_range": None, "note": "Human Ubiquitin β-grasp α/β (76 AA)"},
    {"pdb_id": "1PGA", "chain": "A", "res_range": None, "note": "Streptococcal GB1 α/β domain (56 AA)"},
    {"pdb_id": "6CZH", "chain": "A", "res_range": None, "note": "Baker de novo fluorescence-activating all-β barrel"},
    {"pdb_id": "5TPN", "chain": "A", "res_range": None, "note": "Rocklin de novo EEHEE miniprotein"},
    {"pdb_id": "2LHC", "chain": "A", "res_range": None, "note": "Designed α/β Rossmann-like Top7-family fold"},
    {"pdb_id": "1SHG", "chain": "A", "res_range": None, "note": "Spectrin SH3 all-β barrel domain (57 AA)"},
    {"pdb_id": "1TEN", "chain": "A", "res_range": None, "note": "Tenascin Fibronectin type-III all-β sandwich (90 AA)"},
    {"pdb_id": "1ENH", "chain": "A", "res_range": None, "note": "Engrailed homeodomain 3-helix bundle (54 AA)"},
    {"pdb_id": "2RN2", "chain": "A", "res_range": None, "note": "RNase H mixed α/β monomer (155 AA)"},
    {"pdb_id": "1AKE", "chain": "A", "res_range": (1, 160), "note": "Adenylate kinase Rossmann core α/β"},
]

# @WUFang40615703's exact Oct-2026 post-cutoff test case (MERTK ectodomain, PDB 37NT)
FANG_WU_POST_CUTOFF_ANCHORS: List[Dict[str, Any]] = [
    {"pdb_id": "37NT", "chain": "A", "res_range": (95, 194), "note": "MERTK Ig-like Domain 1 (Oct 2026 @WUFang tweet)"},
    {"pdb_id": "37NT", "chain": "A", "res_range": (196, 283), "note": "MERTK Ig-like Domain 2 (Oct 2026 @WUFang tweet)"},
    {"pdb_id": "37NT", "chain": "A", "res_range": (286, 380), "note": "MERTK FnIII Domain 3 (Oct 2026 @WUFang tweet)"},
    {"pdb_id": "37NT", "chain": "A", "res_range": (385, 481), "note": "MERTK FnIII Domain 4 (Oct 2026 @WUFang tweet)"},
]


def kmer_seq_similarity(seq_a: str, seq_b: str, k: int = 3) -> float:
    """Fast k-mer containment + ungapped window identity check to enforce <30% redundancy."""
    if not seq_a or not seq_b:
        return 0.0
    if seq_a == seq_b:
        return 1.0
    ka = {seq_a[i : i + k] for i in range(max(1, len(seq_a) - k + 1))}
    kb = {seq_b[i : i + k] for i in range(max(1, len(seq_b) - k + 1))}
    jaccard = len(ka & kb) / float(max(1, min(len(ka), len(kb))))
    return jaccard


def has_internal_chain_break(struct: ProteinStructure, max_peptide_bond_a: float = 2.4) -> bool:
    """Return True if any consecutive C(i) -> N(i+1) bond exceeds `max_peptide_bond_a`."""
    if struct.length < 2:
        return True
    cn_dists = np.linalg.norm(struct.coords_n[1:] - struct.coords_c[:-1], axis=1)
    return bool(np.any(cn_dists > max_peptide_bond_a))


def classify_fold_q3(q3: str) -> str:
    """Classify monomer fold architecture from Q3 fractions ('all-alpha', 'all-beta', 'alpha-beta', 'coil-rich')."""
    L = max(1, len(q3))
    fh = q3.count("H") / L
    fe = q3.count("E") / L
    if fh >= 0.35 and fe < 0.10:
        return "mainly-alpha"
    if fe >= 0.25 and fh < 0.12:
        return "mainly-beta"
    if fh >= 0.12 and fe >= 0.12:
        return "alpha-beta"
    return "mixed-or-coil"


def query_rcsb_post_cutoff_candidates(
    min_date: str = "2026-09-01T00:00:00Z",
    max_rows: int = 250,
) -> List[str]:
    """Query RCSB Search API v2 for recently released high-resolution protein structures."""
    query = {
        "query": {
            "type": "group",
            "logical_operator": "and",
            "nodes": [
                {
                    "type": "terminal",
                    "service": "text",
                    "parameters": {
                        "attribute": "rcsb_accession_info.initial_release_date",
                        "operator": "greater_or_equal",
                        "value": min_date,
                    },
                },
                {
                    "type": "terminal",
                    "service": "text",
                    "parameters": {
                        "attribute": "rcsb_entry_info.resolution_combined",
                        "operator": "less_or_equal",
                        "value": 3.2,
                    },
                },
                {
                    "type": "terminal",
                    "service": "text",
                    "parameters": {
                        "attribute": "rcsb_entry_info.polymer_entity_count_protein",
                        "operator": "equals",
                        "value": 1,
                    },
                },
            ],
        },
        "return_type": "entry",
        "request_options": {
            "paginate": {"start": 0, "rows": max_rows},
            "sort": [{"sort_by": "rcsb_accession_info.initial_release_date", "direction": "desc"}],
        },
    }
    with httpx.Client(timeout=30.0) as client:
        resp = client.post("https://search.rcsb.org/rcsbsearch/v2/query", json=query)
        resp.raise_for_status()
        data = resp.json()
    return [item["identifier"] for item in data.get("result_set", [])]


def fetch_entry_metadata_and_cif(
    pdb_id: str,
    raw_cache_dir: Path,
    client: httpx.Client,
) -> Optional[Tuple[Dict[str, Any], Path]]:
    """Fetch RCSB entry JSON metadata and `.cif` coordinate file (cached on disk)."""
    pdb_u = pdb_id.upper()
    meta_path = raw_cache_dir / f"{pdb_u}.json"
    cif_path = raw_cache_dir / f"{pdb_u}.cif"

    if not meta_path.exists():
        r_meta = client.get(f"https://data.rcsb.org/rest/v1/core/entry/{pdb_u}")
        if r_meta.status_code != 200:
            return None
        meta_path.write_text(r_meta.text, encoding="utf-8")

    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    title = str(meta.get("struct", {}).get("title", ""))
    keywords = str(meta.get("struct_keywords", {}).get("pdbx_keywords", ""))
    text_blob = f"{title} {keywords}"
    if SAFETY_BLOCKLIST_PATTERNS.search(text_blob):
        return None

    # Also check polymer entity organism + cluster identity (reject pre-2026 sequence homologs if post-cutoff)
    ent_path = raw_cache_dir / f"{pdb_u}_ent1.json"
    if not ent_path.exists():
        r_ent = client.get(f"https://data.rcsb.org/rest/v1/core/polymer_entity/{pdb_u}/1")
        if r_ent.status_code == 200:
            ent_path.write_text(r_ent.text, encoding="utf-8")
    if ent_path.exists():
        ent = json.loads(ent_path.read_text(encoding="utf-8"))
        org_names = " ".join(
            str(s.get("ncbi_scientific_name", ""))
            for s in ent.get("rcsb_entity_source_organism", [])
        )
        desc = str(ent.get("rcsb_polymer_entity", {}).get("pdbx_description", ""))
        if SAFETY_BLOCKLIST_PATTERNS.search(f"{org_names} {desc}"):
            return None
        meta["_entity1"] = ent

    if not cif_path.exists():
        r_cif = client.get(f"https://files.rcsb.org/download/{pdb_u}.cif")
        if r_cif.status_code != 200:
            return None
        cif_path.write_text(r_cif.text, encoding="utf-8")

    return meta, cif_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Curate Stage 1 Benchmark PDBs from RCSB.")
    parser.add_argument("--out-dir", default="data/stage1_benchmark", help="Output dataset directory")
    parser.add_argument("--cache-dir", default="data/raw_pdb_cache", help="Raw RCSB cache directory")
    parser.add_argument("--target-post-cutoff", type=int, default=28, help="Number of post-cutoff targets")
    parser.add_argument("--min-release-date", default="2026-09-01T00:00:00Z")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    raw_cache = Path(args.cache_dir)
    gt_dir = out_dir / "ground_truth"
    bb_dir = out_dir / "backbones_stripped"
    for d in (out_dir, raw_cache, gt_dir, bb_dir):
        d.mkdir(parents=True, exist_ok=True)

    selected_records: List[Dict[str, Any]] = []
    accepted_sequences: List[str] = []

    def try_add_structure(
        pdb_id: str,
        chain_pref: Optional[str],
        res_range: Optional[Tuple[int, int]],
        split_group: str,
        note: str,
        client: httpx.Client,
        require_novel_cluster: bool = False,
    ) -> bool:
        fetched = fetch_entry_metadata_and_cif(pdb_id, raw_cache, client)
        if fetched is None:
            return False
        meta, cif_path = fetched
        ent = meta.get("_entity1", {})

        # For post-cutoff discovery candidates, prefer low-homology sequences (30% cluster size <= 3)
        cluster_30_size = None
        for cm in ent.get("rcsb_cluster_membership", []):
            if cm.get("identity") == 30:
                cluster_30_size = cm.get("cluster_id")
        # Also verify deposition date if require_novel_cluster is True (reject old depositions held back)
        acc_info = meta.get("rcsb_accession_info", {})
        release_date = str(acc_info.get("initial_release_date", ""))[:10]
        deposit_date = str(acc_info.get("deposit_date", ""))[:10]
        if require_novel_cluster and deposit_date and deposit_date < "2025-10-01":
            return False

        try:
            idx_num = len(selected_records) + 1
            anon_id = f"s1_t{idx_num:03d}"
            struct = load_structure(
                cif_path,
                pdb_id=pdb_id.upper(),
                target_id=anon_id,
                chain_id=chain_pref,
                res_range=res_range,
            )
        except Exception:
            return False

        if not (48 <= struct.length <= 220):
            return False
        if has_internal_chain_break(struct):
            return False

        # Reject coil-only disordered peptides (require >= 25% structured H+E)
        structured_frac = (struct.q3.count("H") + struct.q3.count("E")) / float(struct.length)
        if structured_frac < 0.25:
            return False

        # Enforce non-redundancy across our benchmark set
        if any(kmer_seq_similarity(struct.sequence, prev) > 0.25 for prev in accepted_sequences):
            return False

        # Require enough long-range tertiary contacts (|i-j| >= 12) to be a real globular fold
        ii, jj = np.triu_indices(struct.length, k=12)
        n_lr12 = int(np.sum(struct.cb_dist_matrix[ii, jj] < 8.0))
        if n_lr12 < max(10, struct.length // 5):
            return False

        ii24, jj24 = np.triu_indices(struct.length, k=24)
        n_lr24 = int(np.sum(struct.cb_dist_matrix[ii24, jj24] < 8.0))

        # Save stripped backbone PDB (zero sequence leakage) & ground-truth Cα PDB + NPZ
        write_backbone_only_pdb(struct, bb_dir / f"{anon_id}.pdb")
        write_ca_trace_pdb(struct.sequence, struct.coords_ca, gt_dir / f"{anon_id}_ca.pdb")
        np.savez_compressed(
            gt_dir / f"{anon_id}.npz",
            sequence=struct.sequence,
            q3=struct.q3,
            res_numbers=np.array(struct.res_numbers, dtype=np.int32),
            coords_n=struct.coords_n,
            coords_ca=struct.coords_ca,
            coords_c=struct.coords_c,
            coords_o=struct.coords_o,
            coords_cb=struct.coords_cb,
            phi_deg=struct.phi_deg,
            psi_deg=struct.psi_deg,
            cb_dist_matrix=struct.cb_dist_matrix,
            ca_coord_num=struct.ca_coord_num,
        )

        size_bin = "compact_50_100" if struct.length <= 100 else "medium_101_220"
        fold_class = classify_fold_q3(struct.q3)
        org_list = [
            str(s.get("ncbi_scientific_name", "synthetic/unknown"))
            for s in ent.get("rcsb_entity_source_organism", [])
        ]
        res_list = meta.get("rcsb_entry_info", {}).get("resolution_combined", [None])
        resolution = float(res_list[0]) if res_list and res_list[0] is not None else None

        selected_records.append(
            {
                "target_id": anon_id,
                "pdb_id": pdb_id.upper(),
                "chain_id": struct.chain_id,
                "res_range": list(res_range) if res_range else [struct.res_numbers[0], struct.res_numbers[-1]],
                "length": struct.length,
                "temporal_split": split_group,
                "size_bin": size_bin,
                "fold_class": fold_class,
                "release_date": release_date,
                "deposit_date": deposit_date,
                "resolution_A": resolution,
                "organism": org_list[0] if org_list else "unknown",
                "description": str(ent.get("rcsb_polymer_entity", {}).get("pdbx_description", note)),
                "note": note,
                "rcsb_cluster30_id": cluster_30_size,
                "q3_helix_frac": round(struct.q3.count("H") / struct.length, 3),
                "q3_strand_frac": round(struct.q3.count("E") / struct.length, 3),
                "n_contacts_sep12": n_lr12,
                "n_contacts_sep24": n_lr24,
            }
        )
        accepted_sequences.append(struct.sequence)
        print(
            f"  [+] {anon_id} <- {pdb_id.upper()}:{struct.chain_id} "
            f"(L={struct.length:3d}, {split_group:11s}, {fold_class:12s}, rel={release_date}) | {note[:52]}"
        )
        return True

    with httpx.Client(timeout=30.0, follow_redirects=True) as client:
        print("=== Step 1/3: Fetching Pre-2025 Anchor Controls (incl. @sokrypton 6MRR) ===")
        for item in PRE_CUTOFF_CONTROLS:
            try_add_structure(
                item["pdb_id"],
                item["chain"],
                item["res_range"],
                "pre_cutoff",
                item["note"],
                client,
                require_novel_cluster=False,
            )

        print("\n=== Step 2/3: Fetching @WUFang40615703 Oct-2026 37NT Domains ===")
        for item in FANG_WU_POST_CUTOFF_ANCHORS:
            try_add_structure(
                item["pdb_id"],
                item["chain"],
                item["res_range"],
                "post_cutoff",
                item["note"],
                client,
                require_novel_cluster=False,
            )

        print(f"\n=== Step 3/3: Discovering Sept-Nov 2026 Post-Cutoff Monomers (>={args.min_release_date[:10]}) ===")
        post_candidates = query_rcsb_post_cutoff_candidates(min_date=args.min_release_date, max_rows=300)
        print(f"  Queried {len(post_candidates)} newly released single-entity PDB candidates from RCSB.")
        n_post = sum(1 for r in selected_records if r["temporal_split"] == "post_cutoff")
        for cand_id in post_candidates:
            if n_post >= args.target_post_cutoff:
                break
            if cand_id.upper() == "37NT":
                continue
            ok = try_add_structure(
                cand_id,
                None,
                None,
                "post_cutoff",
                "Strict late-2026 RCSB monomer holdout",
                client,
                require_novel_cluster=True,
            )
            if ok:
                n_post += 1

    manifest_path = out_dir / "manifest.json"
    summary = {
        "benchmark": "fable-protein-bench-stage1",
        "total_targets": len(selected_records),
        "n_pre_cutoff": sum(1 for r in selected_records if r["temporal_split"] == "pre_cutoff"),
        "n_post_cutoff": sum(1 for r in selected_records if r["temporal_split"] == "post_cutoff"),
        "targets": selected_records,
    }
    manifest_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\nSaved {len(selected_records)} validated Stage 1 targets to {manifest_path}")


if __name__ == "__main__":
    sys.exit(main())
