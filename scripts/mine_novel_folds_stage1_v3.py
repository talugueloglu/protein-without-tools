#!/usr/bin/env python3
"""Mine genuinely structurally novel (`TM-score < 0.50` / `< 0.55`) post-May-2026 PDB structures
and upgrade Stage 1 to v3 (`N = 100` targets) with exact Foldseek `TM-align` TM-scores & Cα RMSD.

Why v2 only had 9 `TM < 0.50` targets:
  1. v2 searched `return_type="entry"` and only parsed Chain 1—skipping every de novo binder/design
     crystallized as Chain B/C/D against a natural receptor target!
  2. v2 stopped after the first 30 synthetic hits (`Nov -> July 2026`), never scanning `May -> July 2026`
     or `DUF / uncharacterized / orphan` natural proteins, and included tiny `55-72 AA` 3-helix bundles
     that trivially match existing helical fragments (`TM > 0.75`).

Three-Stage Funnel in this script:
  Stage 1: Query RCSB `return_type="polymer_entity"` (`>= 2026-05-01`, `80 <= L <= 260`, `res <= 3.5 Å`)
           across (a) ALL `synthetic construct` / `de novo` / `designed` entities (even in complexes!)
           and (b) ALL `uncharacterized` / `DUF` / `hypothetical` / orphan natural entities + general monomers.
           Filter strictly by `SAFETY_BLOCKLIST`, non-redundancy (`kmer_jaccard <= 0.22`), and
           `< 25%` sequence identity to pre-May-2026 PDB (`MMseqs2` `rows=100` + GraphQL release dates).
  Stage 2: Fast `Foldseek mode="3diaa"` screen (`concurrency=5`) on all new sequence-novel chains to
           rank candidates by pre-cutoff `pdb100` 3Di structural score (`score <= 75` strongly enriches `TM < 0.55`).
  Stage 3: Exact `Foldseek mode="tmalign"` + iterative Kabsch SO(3) `TM-score` & `RMSD` on the top 55
           most structurally novel candidates against both pre-cutoff `PDB` and `AlphaFold DB`, then
           build the clean `N = 100` Stage 1 v3 benchmark & re-export zero-leakage prompts.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
from pathlib import Path
import re
import time
from typing import Any, Dict, List, Optional, Tuple

import httpx
import numpy as np

from fable_bench.pdb_utils import (
    AA3_TO_1,
    ProteinStructure,
    load_structure,
    write_backbone_only_pdb,
    write_ca_trace_pdb,
)
from fable_bench.tmscore import kabsch_superpose, tm_d0

CUTOFF_DATE = "2026-05-01"
CUTOFF_ISO = "2026-05-01T00:00:00Z"

SAFETY_BLOCKLIST = re.compile(
    r"\b(virus|viral|virion|capsid|phage|bacteriophage|toxin|venom|ricin|abrin|"
    r"botulinum|anthrax|ebola|marburg|variola|poxvirus|influenza|coronavirus|sars|"
    r"hemolysin|enterotoxin|neurotoxin|cytotoxin|conotoxin)\b",
    re.IGNORECASE,
)


def kmer_jaccard(seq_a: str, seq_b: str, k: int = 3) -> float:
    if seq_a == seq_b:
        return 1.0
    ka = {seq_a[i : i + k] for i in range(max(1, len(seq_a) - k + 1))}
    kb = {seq_b[i : i + k] for i in range(max(1, len(seq_b) - k + 1))}
    return len(ka & kb) / float(max(1, min(len(ka), len(kb))))


def nw_global_seqid(seq_a: str, seq_b: str) -> float:
    if not seq_a or not seq_b:
        return 0.0
    la, lb = len(seq_a), len(seq_b)
    prev_score = np.zeros(lb + 1, dtype=np.int32)
    prev_matches = np.zeros(lb + 1, dtype=np.int32)
    for i in range(1, la + 1):
        cur_score = np.zeros(lb + 1, dtype=np.int32)
        cur_matches = np.zeros(lb + 1, dtype=np.int32)
        ca = seq_a[i - 1]
        for j in range(1, lb + 1):
            m = 2 if ca == seq_b[j - 1] else -1
            d_s = prev_score[j - 1] + m
            u_s = prev_score[j] - 1
            l_s = cur_score[j - 1] - 1
            if d_s >= u_s and d_s >= l_s:
                cur_score[j] = d_s
                cur_matches[j] = prev_matches[j - 1] + (1 if ca == seq_b[j - 1] else 0)
            elif u_s >= l_s:
                cur_score[j] = u_s
                cur_matches[j] = prev_matches[j]
            else:
                cur_score[j] = l_s
                cur_matches[j] = cur_matches[j - 1]
        prev_score = cur_score
        prev_matches = cur_matches
    best_j = int(np.argmax(prev_score))
    best_m = max(int(prev_matches[-1]), int(prev_matches[best_j]))
    return round(100.0 * best_m / float(max(1, min(la, lb))), 1)


def classify_fold_q3(q3: str) -> str:
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


def write_native_pdb(struct: ProteinStructure, out_path: Path) -> None:
    aa1_to_3 = {v: k for k, v in AA3_TO_1.items() if k != "MSE"}
    lines = []
    ser = 1
    for idx, aa in enumerate(struct.sequence):
        r3 = aa1_to_3.get(aa, "ALA")
        for an, arr, el in (
            ("N", struct.coords_n[idx], "N"),
            ("CA", struct.coords_ca[idx], "C"),
            ("C", struct.coords_c[idx], "C"),
            ("O", struct.coords_o[idx], "O"),
        ):
            lines.append(
                f"ATOM  {ser:5d}  {an:<3s} {r3} A{idx + 1:4d}    "
                f"{arr[0]:8.3f}{arr[1]:8.3f}{arr[2]:8.3f}  1.00 20.00           {el:>2s}"
            )
            ser += 1
    lines.append("TER\nEND\n")
    out_path.write_text("\n".join(lines), encoding="utf-8")


def exact_aligned_tm_and_rmsd(
    ca_query: np.ndarray,
    tca_str: str,
    q_start_1idx: int,
    t_start_1idx: int,
    q_aln: str,
    t_aln: str,
) -> Optional[Tuple[float, float, int]]:
    if not tca_str or not q_aln or not t_aln:
        return None
    try:
        ca_t = np.fromstring(tca_str, dtype=np.float64, sep=",").reshape(-1, 3)
    except ValueError:
        return None

    qi = max(0, int(q_start_1idx) - 1)
    ti = max(0, int(t_start_1idx) - 1)
    q_pts, t_pts = [], []
    lq, lt = len(ca_query), len(ca_t)
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
    _, _, rmsd_all = kabsch_superpose(qp, tp)

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
                for _ in range(12):
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


def extract_longest_globular_segment(cif_p: Path, pdb_u: str, ch: str) -> Optional[ProteinStructure]:
    """Load chain or its longest contiguous break-free globular segment (`90 <= L <= 320`)."""
    st_full = load_structure(cif_p, pdb_id=pdb_u, chain_id=ch, res_range=None)
    if st_full.length < 78:
        return None
    cn = np.linalg.norm(st_full.coords_n[1:] - st_full.coords_c[:-1], axis=1)
    ca_d = np.linalg.norm(np.diff(st_full.coords_ca, axis=0), axis=1)
    breaks = np.flatnonzero((cn > 2.4) | (ca_d > 4.35))

    if len(breaks) == 0:
        cand = st_full if st_full.length <= 320 else load_structure(
            cif_p, pdb_id=pdb_u, chain_id=ch, res_range=(st_full.res_numbers[0], st_full.res_numbers[319])
        )
    else:
        # Find longest contiguous break-free span of residue indices
        bounds = [-1] + breaks.tolist() + [st_full.length - 1]
        best_s, best_e = 0, 0
        for k in range(len(bounds) - 1):
            s_idx = bounds[k] + 1
            e_idx = bounds[k + 1]
            if (e_idx - s_idx + 1) > (best_e - best_s + 1):
                best_s, best_e = s_idx, e_idx
        span_len = best_e - best_s + 1
        if span_len < 88:
            return None
        if span_len > 320:
            best_e = best_s + 319
        r_start, r_end = st_full.res_numbers[best_s], st_full.res_numbers[best_e]
        cand = load_structure(cif_p, pdb_id=pdb_u, chain_id=ch, res_range=(r_start, r_end))
        cn2 = np.linalg.norm(cand.coords_n[1:] - cand.coords_c[:-1], axis=1)
        ca2 = np.linalg.norm(np.diff(cand.coords_ca, axis=0), axis=1)
        if np.any(cn2 > 2.4) or np.any(ca2 > 4.35):
            return None

    if not (78 <= cand.length <= 320):
        return None
    ss_frac = (cand.q3.count("H") + cand.q3.count("E")) / float(cand.length)
    if ss_frac < 0.25:
        return None
    ii, jj = np.triu_indices(cand.length, k=12)
    if int(np.sum(cand.cb_dist_matrix[ii, jj] < 8.0)) < max(12, cand.length // 5):
        return None
    # Strict globular compactness gate (rejects unfolded cryo-EM ribosomal tentacles!)
    rg = float(np.sqrt(np.mean(np.sum((cand.coords_ca - cand.coords_ca.mean(axis=0)) ** 2, axis=1))))
    if rg > 2.60 * (cand.length ** 0.38):
        return None
    return cand


def search_rcsb_polymer_entities(client: httpx.Client, mode: str, rows: int = 800, start: int = 0) -> List[str]:
    base_nodes: List[Dict[str, Any]] = [
        {
            "type": "terminal",
            "service": "text",
            "parameters": {
                "attribute": "rcsb_accession_info.initial_release_date",
                "operator": "greater_or_equal",
                "value": CUTOFF_ISO,
            },
        },
        {
            "type": "terminal",
            "service": "text",
            "parameters": {
                "attribute": "entity_poly.rcsb_sample_sequence_length",
                "operator": "range",
                "value": {"from": 85, "include_lower": True, "to": 330, "include_upper": True},
            },
        },
        {
            "type": "terminal",
            "service": "text",
            "parameters": {
                "attribute": "entity_poly.rcsb_entity_polymer_type",
                "operator": "exact_match",
                "value": "Protein",
            },
        },
    ]

    if mode not in ("synthetic", "designed_text"):
        # For natural proteins, restrict to clean globular monomers/heterodimers (<=2 protein entities, res <= 3.5A)
        base_nodes.append(
            {
                "type": "terminal",
                "service": "text",
                "parameters": {
                    "attribute": "rcsb_entry_info.polymer_entity_count_protein",
                    "operator": "less_or_equal",
                    "value": 2,
                },
            }
        )
        base_nodes.append(
            {
                "type": "terminal",
                "service": "text",
                "parameters": {
                    "attribute": "rcsb_entry_info.resolution_combined",
                    "operator": "less_or_equal",
                    "value": 3.5,
                },
            }
        )

    if mode == "synthetic":
        base_nodes.append(
            {
                "type": "terminal",
                "service": "text",
                "parameters": {
                    "attribute": "rcsb_entity_source_organism.ncbi_scientific_name",
                    "operator": "exact_match",
                    "value": "synthetic construct",
                },
            }
        )
    elif mode == "designed_text":
        base_nodes.append(
            {
                "type": "terminal",
                "service": "text",
                "parameters": {
                    "attribute": "struct.title",
                    "operator": "contains_words",
                    "value": "novo designed design binder scaffold cage enzyme barrel hallucinated",
                },
            }
        )
    elif mode == "orphan_duf":
        base_nodes.append(
            {
                "type": "terminal",
                "service": "text",
                "parameters": {
                    "attribute": "rcsb_polymer_entity.pdbx_description",
                    "operator": "contains_words",
                    "value": "DUF uncharacterized hypothetical unknown orphan effector antitoxin",
                },
            }
        )
    # mode == "general" uses base_nodes directly!

    q = {
        "query": {"type": "group", "logical_operator": "and", "nodes": base_nodes},
        "return_type": "polymer_entity",
        "request_options": {
            "paginate": {"start": start, "rows": rows},
            "sort": [{"sort_by": "rcsb_accession_info.initial_release_date", "direction": "desc"}],
        },
    }
    r = client.post("https://search.rcsb.org/rcsbsearch/v2/query", json=q, timeout=35.0)
    if r.status_code == 204:
        return []
    r.raise_for_status()
    return [x["identifier"] for x in r.json().get("result_set", [])]


def batch_fetch_entity_metadata(client: httpx.Client, entity_ids: List[str]) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    uniq = sorted(set(entity_ids))
    gql = """
    query($ids: [String!]!) {
      polymer_entities(entity_ids: $ids) {
        rcsb_id
        entity_poly {
          pdbx_seq_one_letter_code_can
          pdbx_strand_id
        }
        rcsb_polymer_entity_container_identifiers {
          auth_asym_ids
        }
        rcsb_polymer_entity {
          pdbx_description
        }
        rcsb_entity_source_organism {
          ncbi_scientific_name
        }
        entry {
          rcsb_id
          struct { title }
          struct_keywords { pdbx_keywords }
          rcsb_accession_info { initial_release_date deposit_date }
          rcsb_entry_info { resolution_combined }
        }
      }
    }
    """
    for i in range(0, len(uniq), 60):
        chunk = uniq[i : i + 60]
        for _ in range(3):
            try:
                r = client.post("https://data.rcsb.org/graphql", json={"query": gql, "variables": {"ids": chunk}}, timeout=35.0)
                if r.status_code == 200:
                    body = r.json()
                    if body.get("errors") and i == 0:
                        print("GraphQL warning:", body["errors"][0], flush=True)
                    for it in (body.get("data") or {}).get("polymer_entities") or []:
                        out[it["rcsb_id"]] = it
                elif i == 0:
                    print(f"GraphQL HTTP {r.status_code}: {r.text[:250]}", flush=True)
                break
            except httpx.HTTPError:
                time.sleep(1.5)
    return out


def batch_fetch_dates_and_seqs(client: httpx.Client, entity_ids: List[str], cache: Dict[str, Tuple[str, str]]) -> None:
    missing = [eid for eid in set(entity_ids) if eid not in cache]
    gql = """
    query($ids: [String!]!) {
      polymer_entities(entity_ids: $ids) {
        rcsb_id
        entity_poly { pdbx_seq_one_letter_code_can }
        entry { rcsb_accession_info { initial_release_date } }
      }
    }
    """
    for i in range(0, len(missing), 70):
        chunk = missing[i : i + 70]
        for _ in range(3):
            try:
                r = client.post("https://data.rcsb.org/graphql", json={"query": gql, "variables": {"ids": chunk}}, timeout=35.0)
                if r.status_code == 200:
                    for it in (r.json().get("data") or {}).get("polymer_entities") or []:
                        rel = str(((it.get("entry") or {}).get("rcsb_accession_info") or {}).get("initial_release_date") or "")[:10]
                        seq = str((it.get("entity_poly") or {}).get("pdbx_seq_one_letter_code_can") or "").replace("\n", "")
                        cache[it["rcsb_id"]] = (rel, seq)
                break
            except httpx.HTTPError:
                time.sleep(1.5)


def check_precutoff_seqid(
    client: httpx.Client,
    seq: str,
    tag: str,
    sim_cache: Path,
    date_seq_cache: Dict[str, Tuple[str, str]],
) -> float:
    h8 = hashlib.md5(seq.encode("utf-8")).hexdigest()[:8]
    cfile = sim_cache / f"rcsb_seq100_{tag}_{len(seq)}_{h8}.json"
    if cfile.exists():
        return float(json.loads(cfile.read_text(encoding="utf-8"))["max_seqid"])

    q = {
        "query": {
            "type": "terminal",
            "service": "sequence",
            "parameters": {
                "evalue_cutoff": 0.1,
                "identity_cutoff": 0.20,
                "sequence_type": "protein",
                "value": seq,
            },
        },
        "return_type": "polymer_entity",
        "request_options": {
            "results_content_type": ["experimental"],
            "scoring_strategy": "sequence",
            "return_all_hits": False,
            "paginate": {"start": 0, "rows": 100},
        },
    }
    hits = []
    for _ in range(2):
        try:
            r = client.post("https://search.rcsb.org/rcsbsearch/v2/query", json=q, timeout=35.0)
            hits = [] if r.status_code == 204 else r.json().get("result_set", [])
            break
        except httpx.HTTPError:
            time.sleep(1.0)

    eids = [h["identifier"] for h in hits]
    batch_fetch_dates_and_seqs(client, eids, date_seq_cache)
    best = 0.0
    for eid in eids:
        rel, hseq = date_seq_cache.get(eid, ("", ""))
        if not rel or rel >= CUTOFF_DATE:
            continue
        sid = nw_global_seqid(seq, hseq)
        if sid > best:
            best = sid
    cfile.write_text(json.dumps({"max_seqid": best}), encoding="utf-8")
    return best


async def run_foldseek_batch(
    items: List[Tuple[str, Path]],
    mode: str,
    cache_dir: Path,
    concurrency: int = 5,
) -> List[Dict[str, Any]]:
    sem = asyncio.Semaphore(concurrency)
    limits = httpx.Limits(max_connections=10, max_keepalive_connections=5)

    async def _one(aclient: httpx.AsyncClient, key: str, pfile: Path) -> Dict[str, Any]:
        cfile = cache_dir / f"{mode}_{key}.json"
        if cfile.exists():
            d = json.loads(cfile.read_text(encoding="utf-8"))
            if "error" not in d:
                return d
        async with sem:
            pbytes = pfile.read_bytes()
            tid = None
            for att in range(6):
                try:
                    r = await aclient.post(
                        "https://search.foldseek.com/api/ticket",
                        data={"mode": mode, "database[]": ["pdb100", "afdb-swissprot", "afdb50"]},
                        files={"q": (pfile.name, pbytes, "chemical/x-pdb")},
                    )
                    if r.status_code == 429:
                        await asyncio.sleep(3.5 * (att + 1))
                        continue
                    r.raise_for_status()
                    tid = r.json().get("id")
                    break
                except Exception:
                    await asyncio.sleep(2.5 * (att + 1))
            if not tid:
                return {"error": "ticket_failed"}
            max_polls = 45 if mode == "tmalign" else 28
            for _ in range(max_polls):
                await asyncio.sleep(1.5)
                try:
                    st = (await aclient.get(f"https://search.foldseek.com/api/ticket/{tid}")).json()
                    if st.get("status") == "COMPLETE":
                        res = (await aclient.get(f"https://search.foldseek.com/api/result/{tid}/0")).json()
                        parsed: Dict[str, List[Dict[str, Any]]] = {}
                        for dbe in res.get("results", []):
                            db = dbe.get("db", "")
                            alns = dbe.get("alignments") or []
                            flat = alns[0] if (alns and isinstance(alns[0], list)) else alns
                            slim = []
                            for a in flat[:12]:
                                entry = {
                                    "target": a.get("target", ""),
                                    "seqId": float(a.get("seqId", 0.0)),
                                    "prob": float(a.get("prob", 0.0)),
                                    "eval": float(a.get("eval", 0.0)),
                                    "score": float(a.get("score", 0.0)),
                                    "qStartPos": int(a.get("qStartPos", 1)),
                                    "qEndPos": int(a.get("qEndPos", 1)),
                                    "dbStartPos": int(a.get("dbStartPos", 1)),
                                    "dbEndPos": int(a.get("dbEndPos", 1)),
                                    "qLen": int(a.get("qLen", 0)),
                                    "dbLen": int(a.get("dbLen", 0)),
                                }
                                if mode == "tmalign":
                                    entry["qAln"] = a.get("qAln", "")
                                    entry["dbAln"] = a.get("dbAln", "")
                                    entry["tCa"] = a.get("tCa", "")
                                slim.append(entry)
                            parsed[db] = slim
                        cfile.write_text(json.dumps(parsed), encoding="utf-8")
                        print(f"  [{mode} done] {key}", flush=True)
                        return parsed
                    if st.get("status") == "ERROR":
                        return {"error": "job_error"}
                except Exception:
                    continue
            return {"error": "timeout"}

    async with httpx.AsyncClient(timeout=45.0, follow_redirects=True, limits=limits) as aclient:
        return await asyncio.gather(*[_one(aclient, k, p) for k, p in items])


def batch_fetch_entry_dates(pdb_ids: List[str]) -> Dict[str, str]:
    out: Dict[str, str] = {}
    uniq = sorted({p.upper() for p in pdb_ids if len(p) == 4})
    gql = """
    query($ids: [String!]!) {
      entries(entry_ids: $ids) {
        rcsb_id
        rcsb_accession_info { initial_release_date }
      }
    }
    """
    with httpx.Client(timeout=30.0) as client:
        for i in range(0, len(uniq), 80):
            chunk = uniq[i : i + 80]
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


def main() -> None:
    repo = Path("/Users/taluguloglu/.gemini/antigravity/scratch/fable-protein-bench")
    data_dir = repo / "data" / "stage1_benchmark"
    raw_cache = repo / "data" / "raw_pdb_cache"
    sim_cache = raw_cache / "similarity_cache"
    tm_cache = raw_cache / "tmalign_cache"
    mine_pdb_dir = raw_cache / "mined_native_pdbs"
    for d in (sim_cache, tm_cache, mine_pdb_dir):
        d.mkdir(parents=True, exist_ok=True)

    cur_manifest = json.loads((data_dir / "manifest.json").read_text(encoding="utf-8"))
    existing_targets = cur_manifest["targets"]
    existing_pdbs = {t["pdb_id"].upper() for t in existing_targets}
    existing_seqs: List[str] = []
    for t in existing_targets:
        npz = np.load(data_dir / "ground_truth" / f"{t['target_id']}.npz")
        existing_seqs.append(str(npz["sequence"]))

    date_seq_cache: Dict[str, Tuple[str, str]] = {}
    new_candidates: List[Tuple[ProteinStructure, Dict[str, Any], Path]] = []

    with httpx.Client(timeout=35.0, follow_redirects=True) as client:
        print("=== Stage 1: Searching RCSB Polymer Entities (>= 2026-05-01, L=85..330) ===", flush=True)
        synth_eids = search_rcsb_polymer_entities(client, "synthetic", rows=1500)
        des_eids = search_rcsb_polymer_entities(client, "designed_text", rows=1500)
        duf_eids = search_rcsb_polymer_entities(client, "orphan_duf", rows=1500)
        gen_eids: List[str] = []
        for pg in (0, 1500, 3000):
            gen_eids.extend(search_rcsb_polymer_entities(client, "general", rows=1500, start=pg))

        # Prioritize synthetic + designed + orphan DUF entities, then general globular monomers across all 7 months
        ordered_eids: List[str] = []
        seen_e = set()
        for eid in synth_eids + des_eids + duf_eids + gen_eids:
            pdb4 = eid.split("_")[0].upper()
            if eid not in seen_e and pdb4 not in existing_pdbs:
                seen_e.add(eid)
                ordered_eids.append(eid)
        print(
            f"  Found {len(ordered_eids)} new candidate entities "
            f"(synthetic={len(synth_eids)}, designed_text={len(des_eids)}, duf_orphan={len(duf_eids)}, general_monomers={len(gen_eids)})",
            flush=True,
        )

        meta_map = batch_fetch_entity_metadata(client, ordered_eids[:3500])
        print(f"  Fetched GraphQL metadata for {len(meta_map)} entities.", flush=True)

        # Step 1A: Locally deduplicate & safety-filter all 3,500 entities in memory (<1s)
        dedup_items: List[Tuple[str, str, Dict[str, Any], List[str]]] = []
        local_pool_seqs = list(existing_seqs)
        for eid in ordered_eids[:3500]:
            it = meta_map.get(eid)
            if not it:
                continue
            epoly = it.get("entity_poly") or {}
            can_seq = str(epoly.get("pdbx_seq_one_letter_code_can") or "").replace("\n", "").strip()
            cid = it.get("rcsb_polymer_entity_container_identifiers") or {}
            strands = [str(s).strip() for s in (cid.get("auth_asym_ids") or []) if str(s).strip()]
            if not strands:
                strands = [s.strip() for s in str(epoly.get("pdbx_strand_id") or "").split(",") if s.strip()]
            if not (82 <= len(can_seq) <= 340) or not strands or "X" in can_seq:
                continue
            entry = it.get("entry") or {}
            title = str((entry.get("struct") or {}).get("title") or "")
            kw = str((entry.get("struct_keywords") or {}).get("pdbx_keywords") or "")
            desc = str((it.get("rcsb_polymer_entity") or {}).get("pdbx_description") or "")
            orgs = [str(o.get("ncbi_scientific_name") or "") for o in (it.get("rcsb_entity_source_organism") or [])]
            if SAFETY_BLOCKLIST.search(f"{title} {kw} {desc} {' '.join(orgs)}"):
                continue
            if any(kmer_jaccard(can_seq, prev) > 0.22 for prev in local_pool_seqs):
                continue
            local_pool_seqs.append(can_seq)
            dedup_items.append((eid, can_seq, it, strands))

        print(f"  Deduplicated to {len(dedup_items)} unique non-redundant post-cutoff entities. Running parallel MMseqs2 screen...", flush=True)

        async def _parallel_mmseqs() -> Dict[str, List[str]]:
            sem_m = asyncio.Semaphore(14)
            lim_m = httpx.Limits(max_connections=20, max_keepalive_connections=14)
            out_hits: Dict[str, List[str]] = {}

            async def _q_one(ac: httpx.AsyncClient, eid_k: str, seq_k: str) -> None:
                h8 = hashlib.md5(seq_k.encode("utf-8")).hexdigest()[:8]
                cf = sim_cache / f"rcsb_hits100_{eid_k}_{len(seq_k)}_{h8}.json"
                if cf.exists():
                    out_hits[eid_k] = json.loads(cf.read_text(encoding="utf-8"))
                    return
                q_obj = {
                    "query": {
                        "type": "terminal",
                        "service": "sequence",
                        "parameters": {"evalue_cutoff": 0.1, "identity_cutoff": 0.20, "sequence_type": "protein", "value": seq_k},
                    },
                    "return_type": "polymer_entity",
                    "request_options": {
                        "results_content_type": ["experimental"],
                        "scoring_strategy": "sequence",
                        "return_all_hits": False,
                        "paginate": {"start": 0, "rows": 100},
                    },
                }
                async with sem_m:
                    for _ in range(2):
                        try:
                            rr = await ac.post("https://search.rcsb.org/rcsbsearch/v2/query", json=q_obj)
                            hl = [] if rr.status_code == 204 else [x["identifier"] for x in rr.json().get("result_set", [])]
                            cf.write_text(json.dumps(hl), encoding="utf-8")
                            out_hits[eid_k] = hl
                            return
                        except Exception:
                            await asyncio.sleep(0.8)
                    out_hits[eid_k] = []

            async with httpx.AsyncClient(timeout=30.0, limits=lim_m) as ac:
                await asyncio.gather(*[_q_one(ac, e_k, s_k) for (e_k, s_k, _i, _st) in dedup_items])
            return out_hits

        mmseqs_hits_map = asyncio.run(_parallel_mmseqs())
        all_hit_eids = [h for hl in mmseqs_hits_map.values() for h in hl]
        batch_fetch_dates_and_seqs(client, all_hit_eids, date_seq_cache)
        print(f"  Completed parallel MMseqs2 & date check across {len(dedup_items)} unique entities.", flush=True)

        accepted_pool_seqs = list(existing_seqs)
        for eid, can_seq, it, strands in dedup_items:
            if len(new_candidates) >= 140:
                break
            max_sid = 0.0
            for heid in mmseqs_hits_map.get(eid, []):
                rel, hseq = date_seq_cache.get(heid, ("", ""))
                if not rel or rel >= CUTOFF_DATE:
                    continue
                sid = nw_global_seqid(can_seq, hseq)
                if sid > max_sid:
                    max_sid = sid
            if max_sid >= 25.0:
                continue

            entry = it.get("entry") or {}
            pdb_u = str(entry.get("rcsb_id") or eid.split("_")[0]).upper()
            title = str((entry.get("struct") or {}).get("title") or "")
            desc = str((it.get("rcsb_polymer_entity") or {}).get("pdbx_description") or "")
            orgs = [str(o.get("ncbi_scientific_name") or "") for o in (it.get("rcsb_entity_source_organism") or [])]

            # Download .cif & parse exact chain / longest break-free globular segment matching this entity
            cif_p = raw_cache / f"{pdb_u}.cif"
            if not cif_p.exists():
                try:
                    rc = client.get(f"https://files.rcsb.org/download/{pdb_u}.cif", timeout=35.0)
                    if rc.status_code != 200:
                        continue
                    cif_p.write_text(rc.text, encoding="utf-8")
                except httpx.HTTPError:
                    continue

            parsed_st: Optional[ProteinStructure] = None
            for ch in strands[:3]:
                try:
                    cand_st = extract_longest_globular_segment(cif_p, pdb_u, ch)
                    if cand_st is not None:
                        parsed_st = cand_st
                        break
                except Exception:
                    continue

            if parsed_st is None:
                continue
            if any(kmer_jaccard(parsed_st.sequence, prev) > 0.22 for prev in accepted_pool_seqs):
                continue

            acc = entry.get("rcsb_accession_info") or {}
            res_arr = (entry.get("rcsb_entry_info") or {}).get("resolution_combined") or [None]
            org_str = orgs[0] if orgs else "unknown"
            is_synth = (
                "synthetic" in org_str.lower()
                or "de novo" in f"{title} {desc}".lower()
                or "designed" in f"{title} {desc}".lower()
            )
            info = {
                "entity_id": eid,
                "pdb_id": pdb_u,
                "chain_id": parsed_st.chain_id,
                "res_range": [parsed_st.res_numbers[0], parsed_st.res_numbers[-1]],
                "length": parsed_st.length,
                "release_date": str(acc.get("initial_release_date") or "")[:10],
                "deposit_date": str(acc.get("deposit_date") or "")[:10],
                "resolution_A": float(res_arr[0]) if res_arr and res_arr[0] is not None else None,
                "organism": org_str,
                "description": (desc or title)[:80],
                "is_synth": is_synth,
                "pdb_precutoff_seqid_max": max_sid,
            }
            npdb = mine_pdb_dir / f"{eid}_{parsed_st.chain_id}.pdb"
            write_native_pdb(parsed_st, npdb)
            new_candidates.append((parsed_st, info, npdb))
            accepted_pool_seqs.append(parsed_st.sequence)
            print(
                f"  [SeqNovel {len(new_candidates):3d}] {eid} ({pdb_u}:{parsed_st.chain_id}, L={parsed_st.length:3d}, "
                f"prePDB_Seq={max_sid:.1f}%, synth={is_synth}) | {info['description'][:44]}",
                flush=True,
            )

    print(f"\n=== Stage 2: Fast Foldseek 3diaa Screen on {len(new_candidates)} New Sequence-Novel Chains ===", flush=True)
    fs3di_inputs = [(f"{inf['entity_id']}_{inf['chain_id']}", p) for (_s, inf, p) in new_candidates]
    fs3di_res = asyncio.run(run_foldseek_batch(fs3di_inputs, mode="3diaa", cache_dir=sim_cache, concurrency=5))

    # Collect PDB IDs from pdb100 hits to exclude post-cutoff entries
    all_hit_pdbs: List[str] = []
    for r in fs3di_res:
        for h in r.get("pdb100", []) or []:
            c4 = str(h.get("target", ""))[:4].upper()
            if len(c4) == 4:
                all_hit_pdbs.append(c4)
    pdb_dates = batch_fetch_entry_dates(all_hit_pdbs)

    scored_cands: List[Tuple[float, ProteinStructure, Dict[str, Any], Path]] = []
    for (st, inf, pfile), r3 in zip(new_candidates, fs3di_res):
        if "error" in r3:
            continue
        best_p_score, best_p_prob, best_p_eval, best_p_seq, best_p_t = 0.0, 0.0, 999.0, 0.0, None
        max_p_seq = inf["pdb_precutoff_seqid_max"]
        for h in r3.get("pdb100", []) or []:
            tstr = str(h.get("target", ""))
            c4 = tstr[:4].upper()
            if pdb_dates.get(c4, "1999-01-01") >= CUTOFF_DATE:
                continue
            if h["seqId"] > max_p_seq:
                max_p_seq = h["seqId"]
            if h["score"] > best_p_score:
                best_p_score = h["score"]
                best_p_prob = h["prob"]
                best_p_eval = h["eval"]
                best_p_seq = h["seqId"]
                best_p_t = tstr.split(" ")[0]
        if max_p_seq >= 30.0:
            continue  # Catch hidden homologs missed by MMseqs2!

        af_hits = (r3.get("afdb-swissprot", []) or []) + (r3.get("afdb50", []) or [])
        best_a_score, best_a_prob, best_a_eval, max_a_seq, best_a_t = 0.0, 0.0, 999.0, 0.0, None
        for h in af_hits:
            if h["seqId"] > max_a_seq:
                max_a_seq = h["seqId"]
            if h["score"] > best_a_score:
                best_a_score = h["score"]
                best_a_prob = h["prob"]
                best_a_eval = h["eval"]
                best_a_t = str(h.get("target", "")).split(" ")[0]

        inf.update(
            {
                "pdb_precutoff_seqid_combined": round(max_p_seq, 1),
                "pdb100_precutoff_foldseek_prob": best_p_prob,
                "pdb100_precutoff_foldseek_eval": best_p_eval,
                "pdb100_precutoff_foldseek_score": best_p_score,
                "pdb100_precutoff_foldseek_seqid": best_p_seq,
                "pdb100_precutoff_closest_struct": best_p_t,
                "afdb_max_seqid": round(max_a_seq, 1),
                "afdb_foldseek_prob": best_a_prob,
                "afdb_foldseek_eval": best_a_eval,
                "afdb_foldseek_score": best_a_score,
                "afdb_closest_predicted_struct": best_a_t,
            }
        )
        scored_cands.append((best_p_score, st, inf, pfile))

    scored_cands.sort(key=lambda t: t[0])  # Lowest pre-cutoff PDB 3Di score = most structurally novel!
    print(f"  {len(scored_cands)} sequence-novel candidates survived (<30% PDB SeqID).", flush=True)
    for sc, _st, inf, _pf in scored_cands[:25]:
        print(
            f"    [Top3DiNovel] {inf['pdb_id']}:{inf['chain_id']} L={inf['length']:3d} "
            f"3di_score={sc:5.1f} 3di_prob={inf['pdb100_precutoff_foldseek_prob']:.2f} "
            f"AF_seq={inf['afdb_max_seqid']:4.1f}% | {inf['description'][:42]}",
            flush=True,
        )

    # Stage 3: Run exact Foldseek mode="tmalign" on top 55 lowest-3Di-score candidates!
    top_for_tm = scored_cands[:55]
    print(f"\n=== Stage 3: Exact Foldseek TM-align Verification on Top {len(top_for_tm)} Novel Candidates ===", flush=True)
    tm_inputs = [(f"{inf['entity_id']}_{inf['chain_id']}", pf) for (_sc, _st, inf, pf) in top_for_tm]
    tm_res = asyncio.run(run_foldseek_batch(tm_inputs, mode="tmalign", cache_dir=tm_cache, concurrency=5))

    tm_hit_pdbs = [
        str(h.get("target", ""))[:4].upper()
        for r in tm_res
        for h in (r.get("pdb100", []) or [])
        if len(str(h.get("target", ""))[:4]) == 4
    ]
    pdb_dates.update(batch_fetch_entry_dates(tm_hit_pdbs))

    verified_new: List[Tuple[ProteinStructure, Dict[str, Any]]] = []
    for (_sc, st, inf, _pf), rtm in zip(top_for_tm, tm_res):
        if "error" in rtm:
            continue
        ca_q = st.coords_ca
        lq = st.length

        # Pre-cutoff PDB exact TM-score
        best_pdb_tm_q, best_pdb_tm_avg, max_pdb_bi = 0.0, 0.0, 0.0
        best_pdb_rmsd, best_pdb_cov, best_pdb_id, best_pdb_desc, best_pdb_tseq = None, 0.0, None, "", 0.0
        max_p_seq = inf["pdb_precutoff_seqid_combined"]
        for h in rtm.get("pdb100", []) or []:
            tstr = str(h.get("target", ""))
            c4 = tstr[:4].upper()
            if pdb_dates.get(c4, "1999-01-01") >= CUTOFF_DATE:
                continue
            if h["seqId"] > max_p_seq:
                max_p_seq = h["seqId"]
            tm_avg = round(float(h.get("eval", 0.0)), 4)
            if tm_avg > max_pdb_bi:
                max_pdb_bi = tm_avg
            calc = exact_aligned_tm_and_rmsd(
                ca_q, h.get("tCa", ""), h.get("qStartPos", 1), h.get("dbStartPos", 1), h.get("qAln", ""), h.get("dbAln", "")
            )
            tm_q, rmsd_v, cov = (
                (calc[0], calc[1], round(calc[2] / float(lq), 3))
                if calc
                else (tm_avg, None, round((h.get("qEndPos", 1) - h.get("qStartPos", 1) + 1) / float(lq), 3))
            )
            if tm_q > best_pdb_tm_q:
                parts = tstr.split(" ", 1)
                best_pdb_tm_q, best_pdb_tm_avg, best_pdb_rmsd, best_pdb_cov = tm_q, tm_avg, rmsd_v, cov
                best_pdb_id = parts[0]
                best_pdb_desc = parts[1][:70] if len(parts) > 1 else ""
                best_pdb_tseq = round(float(h.get("seqId", 0.0)), 1)

        # AFDB exact TM-score
        best_af_tm_q, best_af_tm_avg, max_af_bi = 0.0, 0.0, 0.0
        best_af_rmsd, best_af_cov, best_af_id, best_af_desc, best_af_tseq = None, 0.0, None, "", 0.0
        for h in (rtm.get("afdb-swissprot", []) or []) + (rtm.get("afdb50", []) or []):
            tstr = str(h.get("target", ""))
            tm_avg = round(float(h.get("eval", 0.0)), 4)
            if tm_avg > max_af_bi:
                max_af_bi = tm_avg
            calc = exact_aligned_tm_and_rmsd(
                ca_q, h.get("tCa", ""), h.get("qStartPos", 1), h.get("dbStartPos", 1), h.get("qAln", ""), h.get("dbAln", "")
            )
            tm_q, rmsd_v, cov = (
                (calc[0], calc[1], round(calc[2] / float(lq), 3))
                if calc
                else (tm_avg, None, round((h.get("qEndPos", 1) - h.get("qStartPos", 1) + 1) / float(lq), 3))
            )
            if tm_q > best_af_tm_q:
                parts = tstr.split(" ", 1)
                best_af_tm_q, best_af_tm_avg, best_af_rmsd, best_af_cov = tm_q, tm_avg, rmsd_v, cov
                best_af_id = parts[0]
                best_af_desc = parts[1][:70] if len(parts) > 1 else ""
                best_af_tseq = round(float(h.get("seqId", 0.0)), 1)

        if best_pdb_tm_q == 0.0:
            continue
        inf.update(
            {
                "pdb_precutoff_seqid_combined": round(max_p_seq, 1),
                "pdb_precutoff_tm_query": best_pdb_tm_q,
                "pdb_precutoff_tm_avg": best_pdb_tm_avg,
                "pdb_precutoff_tm_bilateral_max": max_pdb_bi,
                "pdb_precutoff_rmsd": best_pdb_rmsd,
                "pdb_precutoff_aln_cov": best_pdb_cov,
                "pdb_precutoff_tm_closest_id": best_pdb_id,
                "pdb_precutoff_tm_closest_desc": best_pdb_desc,
                "pdb_precutoff_tm_seqid": best_pdb_tseq,
                "afdb_tm_query": best_af_tm_q,
                "afdb_tm_avg": best_af_tm_avg,
                "afdb_tm_bilateral_max": max_af_bi,
                "afdb_rmsd": best_af_rmsd,
                "afdb_aln_cov": best_af_cov,
                "afdb_tm_closest_id": best_af_id,
                "afdb_tm_closest_desc": best_af_desc,
                "afdb_tm_seqid": best_af_tseq,
            }
        )
        verified_new.append((st, inf))
        print(
            f"  [TM-Verified] {inf['pdb_id']}:{inf['chain_id']} L={lq:3d} -> "
            f"PDB_TM={best_pdb_tm_q:.3f} (bi={max_pdb_bi:.3f}), AFDB_TM={best_af_tm_q:.3f} (bi={max_af_bi:.3f}) | {inf['description'][:38]}",
            flush=True,
        )

    # Save mined pool summary to disk so we can inspect & build clean N=100 manifest deterministically!
    mined_summary = []
    for st, inf in verified_new:
        npz_out = raw_cache / "mined_npz" / f"{inf['entity_id']}_{inf['chain_id']}.npz"
        npz_out.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            npz_out,
            sequence=st.sequence,
            res_numbers=np.array(st.res_numbers, dtype=np.int32),
            coords_n=st.coords_n,
            coords_ca=st.coords_ca,
            coords_c=st.coords_c,
            coords_o=st.coords_o,
            coords_cb=st.coords_cb,
            phi_deg=st.phi_deg,
            psi_deg=st.psi_deg,
            q3=st.q3,
            cb_dist_matrix=st.cb_dist_matrix,
            ca_coord_num=st.ca_coord_num,
        )
        inf["npz_path"] = str(npz_out)
        inf["q3_helix_frac"] = round(st.q3.count("H") / float(st.length), 3)
        inf["q3_strand_frac"] = round(st.q3.count("E") / float(st.length), 3)
        ii12, jj12 = np.triu_indices(st.length, k=12)
        ii24, jj24 = np.triu_indices(st.length, k=24)
        inf["n_contacts_sep12"] = int(np.sum(st.cb_dist_matrix[ii12, jj12] < 8.0))
        inf["n_contacts_sep24"] = int(np.sum(st.cb_dist_matrix[ii24, jj24] < 8.0))
        inf["fold_class"] = classify_fold_q3(st.q3)
        mined_summary.append(inf)

    out_json = raw_cache / "mined_novel_candidates_verified.json"
    out_json.write_text(json.dumps(mined_summary, indent=2), encoding="utf-8")
    n_lt50 = sum(1 for x in mined_summary if x["pdb_precutoff_tm_query"] < 0.50)
    n_lt55 = sum(1 for x in mined_summary if x["pdb_precutoff_tm_query"] < 0.55)
    print(
        f"\nSaved {len(mined_summary)} newly TM-verified structures -> {out_json} "
        f"(PDB TM < 0.50: {n_lt50}, PDB TM < 0.55: {n_lt55})",
        flush=True,
    )


if __name__ == "__main__":
    main()
