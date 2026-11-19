#!/usr/bin/env python3
"""Offline Sandbox Builder for Stage 3A In-Silico Alanine Scan (`X_i -> A`) Benchmark.

Reads cached raw files from `data/raw_alanine_cache/` and constructs `data/alanine_scan_benchmark/`:
  1. `de_novo_ddg` (~22 targets): Synthetic *de novo* Rocklin (`HEEH`, `EEHEE`, `EHEE`, `HHH`) &
     TrRosetta-hallucinated miniproteins with exact wet-lab `X -> A` ΔΔG_fold (kcal/mol) + 3D core flags.
  2. `natural_ddg` (~14 targets): Diverse natural domains from Tsuboyama `test`/`val` splits with
     exact wet-lab `X -> A` ΔΔG_fold (kcal/mol) + 3D core flags.
  3. `dual_stability_function` & `post_cutoff_mavedb` (~14 targets): Strictly post-May-2026 MaveDB
     experimental DMS scans (including Oct/Nov-2026 paired abundance + function assays on `TYR`,
     `KCNQ1`, `OPRM1`, plus `SGCA`, `GBA1`, `RAF1`/`BRAF`/`KSR1`, and Double-deepPCA domains).

Also exports zero-leakage LLM prompts (`outputs/prompts/stage3a_alanine_prompts.jsonl` + web-chat `.txt`).
"""

from __future__ import annotations

from collections import Counter, defaultdict
import csv
import json
from pathlib import Path
import re
import shutil
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pyarrow.parquet as pq

from fable_bench.alanine_scan import AlanineScanTarget, build_alanine_scan_prompt
from fable_bench.pdb_utils import AA3_TO_1, CANONICAL_AA_SET, load_structure

MUT_ALA_RE = re.compile(r"^([ACDEFGHIKLMNPQRSTVWY])(\d+)A$")
HGVS_PRO_RE = re.compile(r"^(?:[^:]+:)?p\.\(?([A-Z][a-z]{2})(\d+)([A-Z][a-z]{2}|Ter|=)\)?$")


def kmer_jaccard(seq_a: str, seq_b: str, k: int = 3) -> float:
    if seq_a == seq_b:
        return 1.0
    ka = {seq_a[i : i + k] for i in range(max(1, len(seq_a) - k + 1))}
    kb = {seq_b[i : i + k] for i in range(max(1, len(seq_b) - k + 1))}
    return len(ka & kb) / float(max(1, min(len(ka), len(kb))))


def load_megascale_pdb_map(pdb_pq_path: Path) -> Dict[str, str]:
    """Load `{WT_name: pdb_string}` from `megascale_3d_pdbs.parquet` if available."""
    if not pdb_pq_path.exists():
        return {}
    tbl = pq.read_table(pdb_pq_path)
    cols = tbl.column_names
    # Identify name column and pdb content column
    name_col = next((c for c in cols if "name" in c.lower() or "id" in c.lower() or "file" in c.lower()), cols[0])
    pdb_col = next((c for c in cols if c != name_col and ("pdb" in c.lower() or "coord" in c.lower() or "text" in c.lower() or "content" in c.lower())), cols[-1])
    out: Dict[str, str] = {}
    for n_v, p_v in zip(tbl[name_col].to_pylist(), tbl[pdb_col].to_pylist()):
        if n_v and p_v:
            key = str(n_v).split("/")[-1]
            if not key.endswith(".pdb"):
                key += ".pdb"
            txt = p_v.decode("utf-8", errors="ignore") if isinstance(p_v, bytes) else str(p_v)
            if "ATOM" in txt:
                out[key] = txt
    return out


def extract_megascale_targets(cache_dir: Path, pdb_map: Dict[str, str], tmp_dir: Path) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Parse `test`, `val`, and `train-00000` MegaScale splits for `X -> A` scans."""
    wt_seq_by_name: Dict[str, str] = {}
    ala_ddg_by_wt: Dict[str, Dict[int, List[float]]] = defaultdict(lambda: defaultdict(list))
    split_by_wt: Dict[str, str] = {}

    for split_label, pq_name in (
        ("test", "test-00000-of-00001.parquet"),
        ("val", "val-00000-of-00001.parquet"),
        ("train0", "train-00000-of-00003.parquet"),
    ):
        pq_file = cache_dir / pq_name
        if not pq_file.exists():
            continue
        tbl = pq.read_table(pq_file, columns=["WT_name", "mut_type", "aa_seq", "ddG_ML"])
        for wtn_raw, mt_raw, seq_raw, ddg_raw in zip(
            tbl["WT_name"].to_pylist(),
            tbl["mut_type"].to_pylist(),
            tbl["aa_seq"].to_pylist(),
            tbl["ddG_ML"].to_pylist(),
        ):
            if not wtn_raw or not seq_raw or not mt_raw:
                continue
            wtn = str(wtn_raw)
            if wtn not in split_by_wt:
                split_by_wt[wtn] = split_label
            seq_s = str(seq_raw).strip().upper()
            mt_s = str(mt_raw).strip()
            if mt_s == "wt":
                wt_seq_by_name[wtn] = seq_s
                continue
            m = MUT_ALA_RE.match(mt_s)
            if not m:
                continue
            wt_aa, pos1 = m.group(1), int(m.group(2))
            if 1 <= pos1 <= len(seq_s) and seq_s[pos1 - 1] == "A":
                if wtn not in wt_seq_by_name:
                    chars = list(seq_s)
                    chars[pos1 - 1] = wt_aa
                    wt_seq_by_name[wtn] = "".join(chars)
                try:
                    # Flip sign: in Tsuboyama MegaScale, ddG_ML < 0 means destabilizing!
                    destab_kcal = -float(ddg_raw)
                    if not np.isnan(destab_kcal):
                        ala_ddg_by_wt[wtn][pos1].append(destab_kcal)
                except (TypeError, ValueError):
                    continue

    de_novo_out: List[Dict[str, Any]] = []
    natural_out: List[Dict[str, Any]] = []
    seen_seqs: List[str] = []

    # Prioritize test & val splits before train0
    ordered_wts = sorted(
        ala_ddg_by_wt.keys(),
        key=lambda w: ({"test": 0, "val": 1, "train0": 2}.get(split_by_wt[w], 3), w),
    )

    for wtn in ordered_wts:
        wt_seq = wt_seq_by_name.get(wtn, "")
        if not (42 <= len(wt_seq) <= 150) or any(c not in CANONICAL_AA_SET for c in wt_seq):
            continue
        non_ala = [i + 1 for i, c in enumerate(wt_seq) if c != "A"]
        pos_map = ala_ddg_by_wt[wtn]
        assayed_pos = sorted(p for p in non_ala if p in pos_map)
        if len(assayed_pos) < max(30, int(0.85 * len(non_ala))):
            continue
        if any(kmer_jaccard(wt_seq, prev) > 0.28 for prev in seen_seqs):
            continue

        is_pdb4 = bool(re.match(r"^[0-9][A-Za-z0-9]{3}(\.pdb|_)", wtn))
        cat = "natural_ddg" if is_pdb4 else "de_novo_ddg"
        if cat == "de_novo_ddg" and len(de_novo_out) >= 24:
            continue
        if cat == "natural_ddg" and len(natural_out) >= 14:
            continue

        # Extract 3D core flags if PDB is in `pdb_map`
        core_flags: Optional[List[int]] = None
        pdb_txt = pdb_map.get(wtn) or pdb_map.get(wtn.replace("|", "_"))
        if pdb_txt:
            tp = tmp_dir / f"tmp_{len(seen_seqs)}.pdb"
            tp.write_text(pdb_txt, encoding="utf-8")
            try:
                st = load_structure(tp, pdb_id=wtn[:4].upper())
                if st.sequence == wt_seq:
                    cmask = st.is_core_mask
                    core_flags = [1 if bool(cmask[p - 1]) else 0 for p in assayed_pos]
            except Exception:
                pdb_txt = None

        wt_aas = [wt_seq[p - 1] for p in assayed_pos]
        exp_ddg = [round(float(np.mean(pos_map[p])), 4) for p in assayed_pos]
        rec = {
            "source_name": wtn,
            "source_db": f"MegaScale_Tsuboyama2023_{split_by_wt[wtn]}",
            "category": cat,
            "assay_type": "ddg_kcal",
            "sequence": wt_seq,
            "length": len(wt_seq),
            "positions_1idx": assayed_pos,
            "wt_aas": wt_aas,
            "exp_destab_scores": exp_ddg,
            "exp_func_loss_scores": None,
            "is_core_flags": core_flags,
            "pdb_text": pdb_txt,
        }
        seen_seqs.append(wt_seq)
        if cat == "de_novo_ddg":
            de_novo_out.append(rec)
        else:
            natural_out.append(rec)

    return de_novo_out, natural_out


def parse_single_mavedb_csv(csv_path: Path, full_seq: str) -> Optional[Tuple[Dict[int, float], float]]:
    """Extract 1-indexed `X_i -> A` loss scores oriented so HIGHER = MORE DELETERIOUS."""
    ala_by_pos: Dict[int, List[float]] = defaultdict(list)
    syn_scores: List[float] = []
    ter_scores: List[float] = []

    with csv_path.open("r", encoding="utf-8", errors="ignore") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            hgvs = str(row.get("hgvs_pro") or "").strip()
            sc_s = str(row.get("score") or "").strip()
            if not hgvs or sc_s in ("", "NA", "nan", "None"):
                continue
            try:
                sc = float(sc_s)
                if np.isnan(sc):
                    continue
            except ValueError:
                continue
            m = HGVS_PRO_RE.match(hgvs)
            if not m:
                continue
            wt3, pos1, mut3 = m.group(1).upper(), int(m.group(2)), m.group(3)
            wt1 = AA3_TO_1.get(wt3)
            if not wt1 or not (1 <= pos1 <= len(full_seq)) or full_seq[pos1 - 1] != wt1:
                continue
            if mut3 == "=" or AA3_TO_1.get(mut3.upper()) == wt1:
                syn_scores.append(sc)
            elif mut3 == "Ter":
                ter_scores.append(sc)
            elif mut3.upper() == "ALA" and wt1 != "A":
                ala_by_pos[pos1].append(sc)

    if len(ala_by_pos) < 25:
        return None

    # Orient sign so higher = loss of function/abundance!
    # Standard MaveDB convention: WT/synonymous ~ 0 (or 1), nonsense (Ter) < WT -> flip sign (-sc)!
    flip = -1.0
    if len(ter_scores) >= 5 and len(syn_scores) >= 5:
        if float(np.mean(ter_scores)) > float(np.mean(syn_scores)):
            flip = +1.0

    oriented = {p: round(flip * float(np.mean(vals)), 4) for p, vals in ala_by_pos.items()}
    return oriented, flip


def extract_mavedb_targets(mave_dir: Path) -> List[Dict[str, Any]]:
    """Build paired stability+function (`TYR`, `OPRM1`, `KCNQ1`, Double-deepPCA) & 2026 post-cutoff MaveDB scans."""
    by_gene: Dict[str, List[Tuple[Dict[str, Any], Dict[int, float], str]]] = defaultdict(list)

    for meta_p in sorted(mave_dir.glob("urn_mavedb_*.json")):
        csv_p = meta_p.with_suffix(".csv")
        if not csv_p.exists():
            continue
        jm = json.loads(meta_p.read_text(encoding="utf-8"))
        tgenes = jm.get("targetGenes") or []
        if not tgenes:
            continue
        seq = str(((tgenes[0].get("targetSequence") or {}).get("sequence")) or "").strip().upper()
        if not seq or any(c not in CANONICAL_AA_SET for c in seq):
            continue
        parsed = parse_single_mavedb_csv(csv_p, seq)
        if parsed is None:
            continue
        ala_map, _flip = parsed
        gname = str(tgenes[0].get("name") or meta_p.stem).strip()
        by_gene[f"{gname}::{len(seq)}"].append((jm, ala_map, seq))

    out_records: List[Dict[str, Any]] = []
    for gkey, entries in sorted(by_gene.items()):
        gname = gkey.split("::")[0]
        seq = entries[0][2]

        # Check if this gene has BOTH an abundance/expression scan AND a function/signaling/binding scan!
        abund_entry = None
        func_entry = None
        for jm, amap, _s in entries:
            txt = f"{jm.get('title', '')} {jm.get('shortDescription', '')}".lower()
            if any(k in txt for k in ("abundance", "expression", "vamp-seq", "stability", "surface")):
                if abund_entry is None:
                    abund_entry = (jm, amap)
            else:
                if func_entry is None:
                    func_entry = (jm, amap)

        if abund_entry and func_entry:
            jm_a, map_a = abund_entry
            jm_f, map_f = func_entry
            shared_pos = sorted(set(map_a.keys()) & set(map_f.keys()))
            if len(shared_pos) >= 30:
                # Slice to compact mutagenized window (<= 320 AA) if protein is large (e.g. TYR, KCNQ1, OPRM1)
                p_min, p_max = min(shared_pos), max(shared_pos)
                w_start = max(1, p_min - 4) if len(seq) > 320 else 1
                w_end = min(len(seq), w_start + 319) if len(seq) > 320 else len(seq)
                sub_pos = [p - w_start + 1 for p in shared_pos if w_start <= p <= w_end]
                orig_pos = [p for p in shared_pos if w_start <= p <= w_end]
                sub_seq = seq[w_start - 1 : w_end]
                if len(sub_pos) >= 30:
                    pdt = str(jm_a.get("publishedDate") or jm_f.get("publishedDate") or "")[:10]
                    out_records.append(
                        {
                            "source_name": f"{gname} ({jm_a['urn']} + {jm_f['urn']})",
                            "source_db": f"MaveDB_Paired_{pdt}",
                            "published_date": pdt,
                            "category": "dual_stability_function",
                            "assay_type": "ddg_kcal",
                            "sequence": sub_seq,
                            "length": len(sub_seq),
                            "positions_1idx": sub_pos,
                            "wt_aas": [sub_seq[p - 1] for p in sub_pos],
                            "exp_destab_scores": [map_a[p] for p in orig_pos],
                            "exp_func_loss_scores": [map_f[p] for p in orig_pos],
                            "is_core_flags": None,
                            "pdb_text": None,
                        }
                    )
                    continue

        # Single-assay post-cutoff MaveDB target
        jm0, map0, _ = entries[0]
        all_pos = sorted(map0.keys())
        p_min, p_max = min(all_pos), max(all_pos)
        w_start = max(1, p_min - 4) if len(seq) > 320 else 1
        w_end = min(len(seq), w_start + 319) if len(seq) > 320 else len(seq)
        orig_pos = [p for p in all_pos if w_start <= p <= w_end]
        sub_pos = [p - w_start + 1 for p in orig_pos]
        sub_seq = seq[w_start - 1 : w_end]
        if len(sub_pos) < 30:
            continue
        pdt = str(jm0.get("publishedDate") or "")[:10]
        out_records.append(
            {
                "source_name": f"{gname} ({jm0['urn']})",
                "source_db": f"MaveDB_{pdt}",
                "published_date": pdt,
                "category": "post_cutoff_mavedb",
                "assay_type": "dms_activity",
                "sequence": sub_seq,
                "length": len(sub_seq),
                "positions_1idx": sub_pos,
                "wt_aas": [sub_seq[p - 1] for p in sub_pos],
                "exp_destab_scores": [map0[p] for p in orig_pos],
                "exp_func_loss_scores": None,
                "is_core_flags": None,
                "pdb_text": None,
            }
        )
    return out_records


def main() -> None:
    repo = Path("/Users/taluguloglu/.gemini/antigravity/scratch/fable-protein-bench")
    cache_dir = repo / "data" / "raw_alanine_cache"
    out_dir = repo / "data" / "alanine_scan_benchmark"
    targets_dir = out_dir / "targets"
    structs_dir = out_dir / "structures"
    prompts_dir = repo / "outputs" / "prompts"
    spot_dir = prompts_dir / "web_chat_spotcheck"

    shutil.rmtree(out_dir, ignore_errors=True)
    for d in (targets_dir, structs_dir, prompts_dir, spot_dir):
        d.mkdir(parents=True, exist_ok=True)

    tmp_dir = cache_dir / "_tmp_pdb"
    tmp_dir.mkdir(parents=True, exist_ok=True)

    pdb_map = load_megascale_pdb_map(cache_dir / "megascale_3d_pdbs.parquet")
    print(f"Loaded {len(pdb_map)} 3D PDB models from megascale_3d_pdbs.parquet")

    de_novo_recs, nat_recs = extract_megascale_targets(cache_dir, pdb_map, tmp_dir)
    mave_recs = extract_mavedb_targets(cache_dir / "mavedb")
    shutil.rmtree(tmp_dir, ignore_errors=True)

    combined = de_novo_recs + nat_recs + mave_recs
    manifest_targets = []
    prompt_lines = []

    for idx, rec in enumerate(combined, start=1):
        tid = f"s3a_t{idx:03d}"
        pdb_txt = rec.pop("pdb_text", None)
        if pdb_txt:
            (structs_dir / f"{tid}.pdb").write_text(pdb_txt, encoding="utf-8")

        t_payload = {
            "target_id": tid,
            "category": rec["category"],
            "assay_type": rec["assay_type"],
            "sequence": rec["sequence"],
            "length": rec["length"],
            "positions_1idx": rec["positions_1idx"],
            "wt_aas": rec["wt_aas"],
            "exp_destab_scores": rec["exp_destab_scores"],
            "exp_func_loss_scores": rec["exp_func_loss_scores"],
            "is_core_flags": rec["is_core_flags"],
        }
        (targets_dir / f"{tid}.json").write_text(json.dumps(t_payload, indent=2), encoding="utf-8")

        t_obj = AlanineScanTarget(**t_payload)
        prompt_str = build_alanine_scan_prompt(t_obj)
        prompt_lines.append(
            json.dumps(
                {
                    "custom_id": f"{tid}_alanine",
                    "target_id": tid,
                    "task": "stage3a_alanine_scan",
                    "length": t_obj.length,
                    "n_mutations": len(t_obj.positions_1idx),
                    "prompt": prompt_str,
                }
            )
        )
        if idx <= 3 or rec["category"] in ("dual_stability_function", "post_cutoff_mavedb") and idx <= 45:
            if len(list(spot_dir.glob("s3a_*.txt"))) < 6:
                (spot_dir / f"{tid}_alanine.txt").write_text(prompt_str + "\n", encoding="utf-8")

        manifest_targets.append(
            {
                "target_id": tid,
                "source_name": rec["source_name"],
                "source_db": rec["source_db"],
                "published_date": rec.get("published_date", "2023-07-19"),
                "category": rec["category"],
                "assay_type": rec["assay_type"],
                "length": rec["length"],
                "n_assayed_positions": len(rec["positions_1idx"]),
                "has_3d_core_flags": rec["is_core_flags"] is not None,
                "has_dual_func_assay": rec["exp_func_loss_scores"] is not None,
            }
        )

    cat_counts = dict(Counter(m["category"] for m in manifest_targets))
    total_muts = sum(m["n_assayed_positions"] for m in manifest_targets)
    manifest = {
        "benchmark": "fable-protein-bench-stage3a-alanine-scan",
        "total_targets": len(manifest_targets),
        "total_alanine_mutations": total_muts,
        "category_counts": cat_counts,
        "targets": manifest_targets,
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    p_out = prompts_dir / "stage3a_alanine_prompts.jsonl"
    p_out.write_text("\n".join(prompt_lines) + "\n", encoding="utf-8")

    print(f"Built Stage 3A Alanine Scan Benchmark: {len(manifest_targets)} proteins, {total_muts} X->A mutations!")
    print("  Categories:", cat_counts)
    print(f"  Exported prompts -> {p_out}")


if __name__ == "__main__":
    main()
