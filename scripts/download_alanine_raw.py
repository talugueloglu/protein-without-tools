#!/usr/bin/env python3
"""Download raw public experimental datasets for Stage 3A In-Silico Alanine Scan (`X -> A`):
  1. `RosettaCommons/MegaScale`:
     - `dataset3_single/data/{test,val,train-00000}.parquet` (25+ synthetic de novo designs + natural domains)
     - `AlphaFold_model_PDBs/data/train-00000-of-00001.parquet` (3D PDB coordinates for 3D core burial & ThermoMPNN)
  2. `MaveDB` (`api.mavedb.org`):
     - Distinct late-2026 (`>= 2026-05-01`) experimental DMS score-sets (including paired abundance + function
       scans on `TYR`, `OPRM1`, `KCNQ1`, plus `SGCA`, `GBA1`, `GPR161`, `RAF1`, `BRAF`, `ARAF`, `KSR1`, etc.)
     - Classic Faure/Lehner Double-deepPCA paired abundance + binding domains (`GRB2-SH3`, `PSD95-PDZ3`).
"""

from __future__ import annotations

import json
from pathlib import Path
import re
from typing import Any, Dict, List
import httpx
import pyarrow.parquet as pq

SAFETY_BLOCKLIST = re.compile(
    r"\b(virus|viral|virion|capsid|phage|bacteriophage|toxin|venom|ricin|abrin|"
    r"botulinum|anthrax|ebola|marburg|variola|poxvirus|influenza|coronavirus|sars|hiv)\b",
    re.IGNORECASE,
)


def main() -> None:
    repo = Path(__file__).resolve().parents[1]
    cache_dir = repo / "data" / "raw_alanine_cache"
    mave_cache = cache_dir / "mavedb"
    for d in (cache_dir, mave_cache):
        d.mkdir(parents=True, exist_ok=True)

    with httpx.Client(timeout=90.0, follow_redirects=True) as client:
        # 1. MegaScale single-mutant ΔΔG Parquet splits + 3D PDB Parquet
        pq_targets = [
            ("test-00000-of-00001.parquet", "dataset3_single/data/test-00000-of-00001.parquet"),
            ("val-00000-of-00001.parquet", "dataset3_single/data/val-00000-of-00001.parquet"),
            ("train-00000-of-00003.parquet", "dataset3_single/data/train-00000-of-00003.parquet"),
            ("megascale_3d_pdbs.parquet", "AlphaFold_model_PDBs/data/train-00000-of-00001.parquet"),
        ]
        for local_name, remote_subpath in pq_targets:
            out_pq = cache_dir / local_name
            if not out_pq.exists():
                url = f"https://huggingface.co/datasets/RosettaCommons/MegaScale/resolve/main/{remote_subpath}"
                print(f"Downloading {local_name} ...", flush=True)
                r = client.get(url)
                r.raise_for_status()
                out_pq.write_bytes(r.content)
            tbl = pq.read_table(out_pq)
            print(f"  Cached {local_name}: {tbl.num_rows} rows, cols={tbl.column_names[:6]}", flush=True)

        # 2. Fetch distinct post-May-2026 MaveDB experiments (1 primary score-set per experimentSet!) + paired sets
        exps: List[Dict[str, Any]] = json.loads((mave_cache / "experiments.json").read_text(encoding="utf-8"))

        # Explicit high-value Oct/Nov 2026 paired abundance+function & post-cutoff URNs first
        priority_urns: List[str] = [
            "urn:mavedb:00001328-a-1",  # 2026-11-12 SGCA abundance
            "urn:mavedb:00001327-a-1",  # 2026-11-11 GBA1 function
            "urn:mavedb:00001326-a-1",  # 2026-11-09 OPRM1 surface expression (stability)
            "urn:mavedb:00001326-b-1",  # 2026-11-09 OPRM1 DAMGO signaling (function)
            "urn:mavedb:00001324-a-1",  # 2026-11-04 TYR VAMP-seq abundance (stability)
            "urn:mavedb:00001325-a-1",  # 2026-11-04 TYR pigmentation rescue (function)
            "urn:mavedb:00001321-a-1",  # 2026-10-26 KCNQ1 surface abundance (stability)
            "urn:mavedb:00001320-a-1",  # 2026-10-26 KCNQ1 channel function (function)
            "urn:mavedb:00001323-c-1",  # 2026-10-30 GPR161 cAMP signaling
            "urn:mavedb:00001318-d-1",  # 2026-10-16 KSR2 N-term construct
            "urn:mavedb:00001317-d-1",  # 2026-10-16 KSR1 N-term construct
            "urn:mavedb:00001316-f-1",  # 2026-10-16 RAF1 N-term construct
            "urn:mavedb:00001315-h-1",  # 2026-10-16 BRAF N-term construct
            "urn:mavedb:00001314-f-1",  # 2026-10-16 ARAF N-term construct
        ]

        seen_exp_sets = set()
        for e in reversed(exps):
            title = str(e.get("title") or "")
            abstract = str(e.get("abstractText") or "")
            if SAFETY_BLOCKLIST.search(f"{title} {abstract}"):
                continue
            pdate = str(e.get("publishedDate") or "")[:10]
            es = e.get("experimentSetUrn") or e.get("urn")
            is_dual = any(kw in f"{title} {abstract}" for kw in ("Double-deepPCA", "GRB2-SH3", "PSD95-PDZ3"))
            if is_dual:
                for u in (e.get("scoreSetUrns") or [])[:2]:
                    if u not in priority_urns:
                        priority_urns.append(u)
            elif pdate >= "2026-05-01" and es not in seen_exp_sets:
                seen_exp_sets.add(es)
                surns = e.get("scoreSetUrns") or []
                if surns and surns[0] not in priority_urns:
                    priority_urns.append(surns[0])

        saved = 0
        for surn in priority_urns[:65]:
            if saved >= 30:
                break
            safe_u = surn.replace(":", "_")
            meta_p = mave_cache / f"{safe_u}.json"
            csv_p = mave_cache / f"{safe_u}.csv"
            if meta_p.exists() and csv_p.exists():
                jm = json.loads(meta_p.read_text(encoding="utf-8"))
                gname = str(((jm.get("targetGenes") or [{}])[0]).get("name") or "")
                pdt = str(jm.get("publishedDate") or "")[:10]
                print(f"    [CachedMave {saved + 1:2d}] {surn} ({pdt}, {gname})", flush=True)
                saved += 1
                continue

            rm = client.get(f"https://api.mavedb.org/api/v1/score-sets/{surn}")
            if rm.status_code != 200:
                continue
            jm = rm.json()
            tgenes = jm.get("targetGenes") or []
            if not tgenes:
                continue
            tseq_obj = (tgenes[0].get("targetSequence") or {})
            if tseq_obj.get("sequenceType") != "protein":
                continue
            seq_str = str(tseq_obj.get("sequence") or "").strip()
            if not (40 <= len(seq_str) <= 600):
                continue
            gname = str(tgenes[0].get("name") or "")
            if SAFETY_BLOCKLIST.search(f"{gname} {jm.get('title', '')}"):
                continue

            rc = client.get(f"https://api.mavedb.org/api/v1/score-sets/{surn}/scores")
            if rc.status_code != 200 or "Ala" not in rc.text:
                continue
            meta_p.write_text(json.dumps(jm, indent=2), encoding="utf-8")
            csv_p.write_text(rc.text, encoding="utf-8")
            saved += 1
            pdt = str(jm.get("publishedDate") or "")[:10]
            print(f"    [MaveDB {saved:2d}] {surn} ({pdt}, {gname}, L={len(seq_str)}) | {str(jm.get('title', ''))[:50]}", flush=True)

    print("Completed Stage 3A raw cache!", flush=True)


if __name__ == "__main__":
    main()
