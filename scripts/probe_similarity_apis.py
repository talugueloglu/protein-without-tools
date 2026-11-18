#!/usr/bin/env python3
"""Verify fast standalone RCSB MMseqs2 Search + Foldseek Web API (pdb100 + afdb-swissprot + afdb50)."""

import json
from pathlib import Path
import time
import httpx
import numpy as np


def rcsb_fast_seq_search(client: httpx.Client, seq: str) -> list[dict]:
    """Standalone RCSB sequence search (< 0.25s per query)."""
    q = {
        "query": {
            "type": "terminal",
            "service": "sequence",
            "parameters": {
                "evalue_cutoff": 1.0,
                "identity_cutoff": 0.0,
                "sequence_type": "protein",
                "value": seq,
            },
        },
        "return_type": "polymer_entity",
        "request_options": {
            "results_content_type": ["experimental"],
            "scoring_strategy": "sequence",
            "return_all_hits": False,
            "paginate": {"start": 0, "rows": 15},
        },
    }
    r = client.post("https://search.rcsb.org/rcsbsearch/v2/query", json=q)
    if r.status_code == 204:
        return []
    r.raise_for_status()
    return r.json().get("result_set", [])


def foldseek_search(client: httpx.Client, pdb_path: Path) -> dict:
    with pdb_path.open("rb") as fh:
        r = client.post(
            "https://search.foldseek.com/api/ticket",
            data={"mode": "3diaa", "database[]": ["pdb100", "afdb-swissprot", "afdb50"]},
            files={"q": (pdb_path.name, fh, "chemical/x-pdb")},
        )
    r.raise_for_status()
    tid = r.json()["id"]
    for _ in range(25):
        time.sleep(1.2)
        st = client.get(f"https://search.foldseek.com/api/ticket/{tid}").json()
        if st.get("status") == "COMPLETE":
            res = client.get(f"https://search.foldseek.com/api/result/{tid}/0").json()
            out = {}
            for db_entry in res.get("results", []):
                db = db_entry.get("db")
                alns = db_entry.get("alignments") or []
                flat = alns[0] if (alns and isinstance(alns[0], list)) else alns
                out[db] = flat[:3]
            return out
    return {"error": "timeout"}


def main() -> None:
    repo = Path("/Users/taluguloglu/.gemini/antigravity/scratch/fable-protein-bench")
    with httpx.Client(timeout=20.0, follow_redirects=True) as client:
        for tid in ("s1_t025", "s1_t022"):  # 47WY (de novo) vs 36ZR (KRas!)
            npz = np.load(repo / f"data/stage1_benchmark/ground_truth/{tid}.npz")
            seq = str(npz["sequence"])
            t0 = time.perf_counter()
            seq_hits = rcsb_fast_seq_search(client, seq)
            print(f"\n=== {tid} (L={len(seq)}) [RCSB seq {time.perf_counter()-t0:.2f}s] ===")
            print("  RCSB Seq Hits:", json.dumps(seq_hits[:3])[:350])
            t1 = time.perf_counter()
            fs_hits = foldseek_search(client, repo / f"data/stage1_benchmark/ground_truth/{tid}_ca.pdb")
            print(f"  [Foldseek {time.perf_counter()-t1:.2f}s]")
            for db, hits in fs_hits.items():
                print(f"  Foldseek [{db}]:", json.dumps(hits[:1])[:360])


if __name__ == "__main__":
    main()
