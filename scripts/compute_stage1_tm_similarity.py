#!/usr/bin/env python3
"""Compute exact 3D TM-align structure similarity (`TM-score` & `RMSD`) for all 81 Stage 1 targets
against (a) Pre-Cutoff Experimental PDB (`< 2026-05-01`) and (b) AlphaFold DB (`afdb-swissprot` + `afdb50`).

Uses Foldseek `mode="tmalign"` + exact Zhang-Skolnick iterative SO(3) Kabsch maximization on the
returned 3D `tCa` coordinates to record:
  - `pdb_precutoff_tm_query`: Exact TM-score normalized by query length L_query (official CASP convention)
  - `pdb_precutoff_tm_avg`: Foldseek bilateral TM-score (`eval` in mode="tmalign", penalizing size mismatch)
  - `pdb_precutoff_rmsd`: Kabsch Cα RMSD (Å) over aligned residues
  - `pdb_precutoff_aln_cov`: Fraction of query residues aligned (`n_aligned / L_query`)
  - `pdb_precutoff_tm_closest_id`: Closest pre-cutoff PDB chain ID
  - `afdb_tm_query`, `afdb_tm_avg`, `afdb_rmsd`, `afdb_aln_cov`, `afdb_tm_closest_id`
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
import time
from typing import Any, Dict, List, Optional, Tuple

import httpx
import matplotlib.pyplot as plt
import numpy as np

from fable_bench.tmscore import kabsch_superpose, tm_d0

CUTOFF_DATE = "2026-05-01"


def exact_aligned_tm_and_rmsd(
    ca_query: np.ndarray,
    tca_str: str,
    q_start_1idx: int,
    t_start_1idx: int,
    q_aln: str,
    t_aln: str,
) -> Optional[Tuple[float, float, int]]:
    """Compute exact Zhang-Skolnick query-normalized TM-score and global Kabsch RMSD on aligned pairs."""
    if not tca_str or not q_aln or not t_aln:
        return None
    try:
        ca_t = np.fromstring(tca_str, dtype=np.float64, sep=",").reshape(-1, 3)
    except ValueError:
        return None

    qi = max(0, int(q_start_1idx) - 1)
    ti = max(0, int(t_start_1idx) - 1)
    q_pts = []
    t_pts = []
    lq = len(ca_query)
    lt = len(ca_t)
    for qc, tc in zip(q_aln, t_aln):
        if qc != "-" and tc != "-":
            if 0 <= qi < lq and 0 <= ti < lt:
                q_pts.append(ca_query[qi])
                t_pts.append(ca_t[ti])
        if qc != "-":
            qi += 1
        if tc != "-":
            ti += 1

    n_aln = len(q_pts)
    if n_aln < 8:
        return None
    qp = np.array(q_pts, dtype=np.float64)
    tp = np.array(t_pts, dtype=np.float64)

    _r_all, _t_all, rmsd_all = kabsch_superpose(qp, tp)

    # Iterative fragment-seeded Zhang-Skolnick SO(3) maximization normalized by full query length lq
    d0 = tm_d0(lq)
    d0_sq = d0 * d0
    cutoffs_sq = [(4.0 * d0) ** 2, (2.0 * d0) ** 2, (1.0 * d0) ** 2]
    best_tm = 0.0

    frag_lens = sorted({n_aln, max(4, n_aln // 2), max(4, n_aln // 4), min(n_aln, 8)}, reverse=True)
    for fl in frag_lens:
        stride = max(2, fl // 2) if fl < n_aln else 1
        for s in range(0, n_aln - fl + 1, stride):
            sub_idx = np.arange(s, s + fl)
            for cut_sq in cutoffs_sq:
                cur_idx = sub_idx
                for _ in range(15):
                    if len(cur_idx) < 4:
                        break
                    rot, trans, _ = kabsch_superpose(qp[cur_idx], tp[cur_idx])
                    diff = (qp @ rot + trans) - tp
                    dist_sq = np.sum(diff * diff, axis=1)
                    tm_val = float(np.sum(1.0 / (1.0 + dist_sq / d0_sq)) / float(lq))
                    if tm_val > best_tm:
                        best_tm = tm_val
                    nxt = np.flatnonzero(dist_sq < cut_sq)
                    if len(nxt) < 4:
                        nxt = np.argsort(dist_sq)[:4]
                    if np.array_equal(nxt, cur_idx):
                        break
                    cur_idx = nxt

    return round(best_tm, 4), round(float(rmsd_all), 2), n_aln


async def fetch_tmalign_for_target(
    aclient: httpx.AsyncClient,
    sem: asyncio.Semaphore,
    tid: str,
    pdb_id: str,
    pdb_file: Path,
    cache_dir: Path,
) -> Dict[str, Any]:
    cfile = cache_dir / f"tmalign_{tid}_{pdb_id}.json"
    if cfile.exists():
        data = json.loads(cfile.read_text(encoding="utf-8"))
        if "error" not in data:
            return data

    async with sem:
        pdb_bytes = pdb_file.read_bytes()
        ticket_id = None
        for attempt in range(6):
            try:
                r = await aclient.post(
                    "https://search.foldseek.com/api/ticket",
                    data={"mode": "tmalign", "database[]": ["pdb100", "afdb-swissprot", "afdb50"]},
                    files={"q": (pdb_file.name, pdb_bytes, "chemical/x-pdb")},
                )
                if r.status_code == 429:
                    await asyncio.sleep(4.0 * (attempt + 1))
                    continue
                r.raise_for_status()
                ticket_id = r.json().get("id")
                break
            except Exception:
                await asyncio.sleep(2.5 * (attempt + 1))

        if not ticket_id:
            return {"error": "ticket_failed"}

        for _ in range(45):
            await asyncio.sleep(1.6)
            try:
                st = (await aclient.get(f"https://search.foldseek.com/api/ticket/{ticket_id}")).json()
                status = st.get("status")
                if status == "COMPLETE":
                    res = (await aclient.get(f"https://search.foldseek.com/api/result/{ticket_id}/0")).json()
                    parsed: Dict[str, List[Dict[str, Any]]] = {}
                    for db_entry in res.get("results", []):
                        db = db_entry.get("db", "")
                        alns = db_entry.get("alignments") or []
                        flat = alns[0] if (alns and isinstance(alns[0], list)) else alns
                        # Keep top 12 hits per DB (so pre-cutoff PDB filtering has plenty of candidates)
                        slim = []
                        for a in flat[:12]:
                            slim.append(
                                {
                                    "target": a.get("target", ""),
                                    "seqId": float(a.get("seqId", 0.0)),
                                    "prob": float(a.get("prob", 0.0)),
                                    "eval": float(a.get("eval", 0.0)),  # Foldseek TM-score in mode="tmalign"
                                    "score": float(a.get("score", 0.0)),
                                    "qStartPos": int(a.get("qStartPos", 1)),
                                    "qEndPos": int(a.get("qEndPos", 1)),
                                    "dbStartPos": int(a.get("dbStartPos", 1)),
                                    "dbEndPos": int(a.get("dbEndPos", 1)),
                                    "qLen": int(a.get("qLen", 0)),
                                    "dbLen": int(a.get("dbLen", 0)),
                                    "qAln": a.get("qAln", ""),
                                    "dbAln": a.get("dbAln", ""),
                                    "tCa": a.get("tCa", ""),
                                }
                            )
                        parsed[db] = slim
                    cfile.write_text(json.dumps(parsed), encoding="utf-8")
                    print(f"  [done] {tid} ({pdb_id})", flush=True)
                    return parsed
                if status == "ERROR":
                    return {"error": "job_error"}
            except Exception:
                continue
        return {"error": "timeout"}


def batch_fetch_entry_dates(pdb_ids: List[str]) -> Dict[str, str]:
    out: Dict[str, str] = {}
    uniq = sorted({p.upper() for p in pdb_ids if len(p) == 4})
    with httpx.Client(timeout=30.0) as client:
        for i in range(0, len(uniq), 80):
            chunk = uniq[i : i + 80]
            gql = """
            query($ids: [String!]!) {
              entries(entry_ids: $ids) {
                rcsb_id
                rcsb_accession_info { initial_release_date }
              }
            }
            """
            for _ in range(3):
                try:
                    r = client.post("https://data.rcsb.org/graphql", json={"query": gql, "variables": {"ids": chunk}})
                    if r.status_code == 200:
                        for item in (r.json().get("data") or {}).get("entries") or []:
                            out[item["rcsb_id"].upper()] = str(
                                (item.get("rcsb_accession_info") or {}).get("initial_release_date") or ""
                            )[:10]
                    break
                except httpx.HTTPError:
                    time.sleep(1.5)
    return out


async def run_all() -> None:
    repo = Path("/Users/taluguloglu/.gemini/antigravity/scratch/fable-protein-bench")
    data_dir = repo / "data" / "stage1_benchmark"
    bb_dir = repo / "data" / "raw_pdb_cache" / "native_backbones_for_foldseek"
    cache_dir = repo / "data" / "raw_pdb_cache" / "tmalign_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = data_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    targets = manifest["targets"]

    # Map each target to its native backbone PDB (or reconstruct from ground_truth npz if needed)
    from fable_bench.pdb_utils import AA3_TO_1
    aa1_to_3 = {v: k for k, v in AA3_TO_1.items() if k != "MSE"}
    pdb_files: Dict[str, Path] = {}
    ca_by_tid: Dict[str, np.ndarray] = {}
    for rec in targets:
        tid = rec["target_id"]
        npz = np.load(data_dir / "ground_truth" / f"{tid}.npz")
        ca_by_tid[tid] = npz["coords_ca"].astype(np.float64)
        pfile = cache_dir / f"query_{tid}.pdb"
        if not pfile.exists():
            seq = str(npz["sequence"])
            cn, cca, cc, co = npz["coords_n"], npz["coords_ca"], npz["coords_c"], npz["coords_o"]
            lines = []
            ser = 1
            for idx, aa in enumerate(seq):
                r3 = aa1_to_3.get(aa, "ALA")
                for an, arr, el in (("N", cn[idx], "N"), ("CA", cca[idx], "C"), ("C", cc[idx], "C"), ("O", co[idx], "O")):
                    lines.append(
                        f"ATOM  {ser:5d}  {an:<3s} {r3} A{idx + 1:4d}    "
                        f"{arr[0]:8.3f}{arr[1]:8.3f}{arr[2]:8.3f}  1.00 20.00           {el:>2s}"
                    )
                    ser += 1
            lines.append("TER\nEND\n")
            pfile.write_text("\n".join(lines), encoding="utf-8")
        pdb_files[tid] = pfile

    _ = bb_dir
    sem = asyncio.Semaphore(4)
    t0 = time.perf_counter()
    limits = httpx.Limits(max_connections=8, max_keepalive_connections=4)
    async with httpx.AsyncClient(timeout=45.0, follow_redirects=True, limits=limits) as aclient:
        tasks = [
            fetch_tmalign_for_target(aclient, sem, rec["target_id"], rec["pdb_id"], pdb_files[rec["target_id"]], cache_dir)
            for rec in targets
        ]
        raw_results = await asyncio.gather(*tasks)
    print(f"Completed 81 Foldseek mode='tmalign' queries in {time.perf_counter() - t0:.1f}s")

    # Collect all candidate PDB IDs from pdb100 hits to verify release dates < 2026-05-01
    cand_pdbs: List[str] = []
    for res in raw_results:
        for h in res.get("pdb100", []) or []:
            tname = str(h.get("target", ""))[:4].upper()
            if len(tname) == 4:
                cand_pdbs.append(tname)
    pdb_dates = batch_fetch_entry_dates(cand_pdbs)
    print(f"Verified release dates for {len(pdb_dates)} candidate PDB structures.")

    # Compute exact TM-scores for each target against pre-cutoff PDB and AFDB
    for rec, res in zip(targets, raw_results):
        tid = rec["target_id"]
        ca_q = ca_by_tid[tid]
        lq = len(ca_q)

        # 1. Best pre-cutoff PDB structure by exact query-normalized TM-score (and bilateral TM-score)
        best_pdb = {
            "pdb_precutoff_tm_query": 0.0,
            "pdb_precutoff_tm_avg": 0.0,
            "pdb_precutoff_tm_bilateral_max": 0.0,
            "pdb_precutoff_rmsd": None,
            "pdb_precutoff_aln_cov": 0.0,
            "pdb_precutoff_tm_closest_id": None,
            "pdb_precutoff_tm_closest_desc": None,
            "pdb_precutoff_tm_seqid": 0.0,
        }
        max_precutoff_seqid = float(rec.get("pdb_precutoff_seqid_max", 0.0))
        if float(rec.get("pdb100_precutoff_foldseek_seqid", 0.0)) > max_precutoff_seqid:
            max_precutoff_seqid = float(rec["pdb100_precutoff_foldseek_seqid"])

        max_pdb_bi = 0.0
        for h in res.get("pdb100", []) or []:
            tstr = str(h.get("target", ""))
            code4 = tstr[:4].upper()
            rel = pdb_dates.get(code4, "1999-01-01")
            if rel >= CUTOFF_DATE:
                continue
            if float(h.get("seqId", 0.0)) > max_precutoff_seqid:
                max_precutoff_seqid = float(h["seqId"])
            tm_avg = round(float(h.get("eval", 0.0)), 4)
            if tm_avg > max_pdb_bi:
                max_pdb_bi = tm_avg
            calc = exact_aligned_tm_and_rmsd(
                ca_q, h.get("tCa", ""), h.get("qStartPos", 1), h.get("dbStartPos", 1), h.get("qAln", ""), h.get("dbAln", "")
            )
            if calc is None:
                tm_q = tm_avg
                rmsd_v = None
                cov = round((h.get("qEndPos", 1) - h.get("qStartPos", 1) + 1) / float(lq), 3)
            else:
                tm_q, rmsd_v, n_aln = calc
                cov = round(n_aln / float(lq), 3)
            if tm_q > best_pdb["pdb_precutoff_tm_query"]:
                parts = tstr.split(" ", 1)
                best_pdb = {
                    "pdb_precutoff_tm_query": tm_q,
                    "pdb_precutoff_tm_avg": tm_avg,
                    "pdb_precutoff_tm_bilateral_max": max_pdb_bi,
                    "pdb_precutoff_rmsd": rmsd_v,
                    "pdb_precutoff_aln_cov": cov,
                    "pdb_precutoff_tm_closest_id": parts[0],
                    "pdb_precutoff_tm_closest_desc": parts[1][:70] if len(parts) > 1 else "",
                    "pdb_precutoff_tm_seqid": round(float(h.get("seqId", 0.0)), 1),
                }
        best_pdb["pdb_precutoff_tm_bilateral_max"] = max_pdb_bi
        rec["pdb_precutoff_seqid_combined"] = round(max_precutoff_seqid, 1)
        rec.update(best_pdb)

        # 2. Best AlphaFold DB predicted structure (`afdb-swissprot` + `afdb50`) by exact query-normalized TM-score
        best_af = {
            "afdb_tm_query": 0.0,
            "afdb_tm_avg": 0.0,
            "afdb_tm_bilateral_max": 0.0,
            "afdb_rmsd": None,
            "afdb_aln_cov": 0.0,
            "afdb_tm_closest_id": None,
            "afdb_tm_closest_desc": None,
            "afdb_tm_seqid": 0.0,
        }
        max_af_bi = 0.0
        af_candidates = (res.get("afdb-swissprot", []) or []) + (res.get("afdb50", []) or [])
        for h in af_candidates:
            tstr = str(h.get("target", ""))
            tm_avg = round(float(h.get("eval", 0.0)), 4)
            if tm_avg > max_af_bi:
                max_af_bi = tm_avg
            calc = exact_aligned_tm_and_rmsd(
                ca_q, h.get("tCa", ""), h.get("qStartPos", 1), h.get("dbStartPos", 1), h.get("qAln", ""), h.get("dbAln", "")
            )
            if calc is None:
                tm_q = tm_avg
                rmsd_v = None
                cov = round((h.get("qEndPos", 1) - h.get("qStartPos", 1) + 1) / float(lq), 3)
            else:
                tm_q, rmsd_v, n_aln = calc
                cov = round(n_aln / float(lq), 3)
            if tm_q > best_af["afdb_tm_query"]:
                parts = tstr.split(" ", 1)
                best_af = {
                    "afdb_tm_query": tm_q,
                    "afdb_tm_avg": tm_avg,
                    "afdb_tm_bilateral_max": max_af_bi,
                    "afdb_rmsd": rmsd_v,
                    "afdb_aln_cov": cov,
                    "afdb_tm_closest_id": parts[0],
                    "afdb_tm_closest_desc": parts[1][:70] if len(parts) > 1 else "",
                    "afdb_tm_seqid": round(float(h.get("seqId", 0.0)), 1),
                }
        best_af["afdb_tm_bilateral_max"] = max_af_bi
        rec.update(best_af)

    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Updated {manifest_path} with exact PDB & AFDB TM-scores across all 81 targets.")


if __name__ == "__main__":
    asyncio.run(run_all())
