#!/usr/bin/env python3
"""Curate N=100 Stratified Stage 1 Structures + Full PDB & AFDB Sequence/Structure Similarity.

Constructs a 100-target benchmark (`data/stage1_benchmark/manifest.json`) across 4 strata:
  1. `pre_cutoff_control` ( ~15 targets): Classic pre-2025 folds (incl. @sokrypton's 6MRR)
  2. `post_cutoff_de_novo` (~25 targets): Mid/Late-2026 `synthetic construct` / de novo designs
     with <30% sequence identity to pre-cutoff PDB (e.g., 47WY, 36BS)
  3. `post_cutoff_pdb_novel` (~45 targets): Mid/Late-2026 natural structures with <30% sequence
     identity to ANY pre-May-2026 experimental PDB (incl. @WUFang's 37NT domains, 36DO, 21DO)
  4. `post_cutoff_homolog_gradient` (~15 targets): Mid/Late-2026 structures with 30%-95% SeqID
     to pre-2026 PDB, retained in an explicit stratum to plot continuous performance vs. similarity!

For EVERY target (all 100), computes and caches continuous similarity metrics against BOTH:
  (a) Pre-Cutoff Experimental PDB (`< 2026-05-01`):
      - `pdb_precutoff_seqid_max` & `pdb_precutoff_seq_eval` (RCSB MMseqs2/BLAST sequence search)
      - `pdb_precutoff_foldseek_prob`, `pdb_precutoff_foldseek_eval`, `pdb_precutoff_foldseek_score`
        (Foldseek 3Di+AA structure search against `pdb100`, excluding post-cutoff entries!)
  (b) AlphaFold / Predicted Structure Databases (`afdb-swissprot` + `afdb50`):
      - `afdb_seqid_max` (max sequence identity across AFDB hits)
      - `afdb_foldseek_prob`, `afdb_foldseek_eval`, `afdb_foldseek_score`, `afdb_closest_target`
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
from pathlib import Path
import re
import sys
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

CUTOFF_DATE = "2026-05-01"
CUTOFF_ISO = "2026-05-01T00:00:00Z"

SAFETY_BLOCKLIST = re.compile(
    r"\b(virus|viral|virion|capsid|phage|bacteriophage|toxin|venom|ricin|abrin|"
    r"botulinum|anthrax|ebola|marburg|variola|poxvirus|influenza|coronavirus|sars|"
    r"hemolysin|enterotoxin|neurotoxin|cytotoxin|conotoxin)\b",
    re.IGNORECASE,
)

PRE_CUTOFF_SEEDS: List[Dict[str, Any]] = [
    {"pdb_id": "6MRR", "chain": "A", "res_range": None, "note": "Baker 2018 de novo βαβαβ (@sokrypton tweet)"},
    {"pdb_id": "2A3D", "chain": "A", "res_range": None, "note": "De novo 3-helix bundle α3D"},
    {"pdb_id": "1UBQ", "chain": "A", "res_range": None, "note": "Human Ubiquitin β-grasp α/β"},
    {"pdb_id": "1PGA", "chain": "A", "res_range": None, "note": "Streptococcal GB1 α/β domain"},
    {"pdb_id": "6CZH", "chain": "A", "res_range": None, "note": "Baker de novo mFAP0 all-β barrel"},
    {"pdb_id": "1SHG", "chain": "A", "res_range": None, "note": "Spectrin SH3 all-β barrel"},
    {"pdb_id": "1TEN", "chain": "A", "res_range": None, "note": "Tenascin FnIII all-β sandwich"},
    {"pdb_id": "1ENH", "chain": "A", "res_range": None, "note": "Engrailed homeodomain 3-helix bundle"},
    {"pdb_id": "2RN2", "chain": "A", "res_range": None, "note": "E. coli RNase H mixed α/β"},
    {"pdb_id": "1AKE", "chain": "A", "res_range": (1, 160), "note": "Adenylate kinase Rossmann core"},
    {"pdb_id": "1CSP", "chain": "A", "res_range": None, "note": "Cold-shock protein OB-fold β-barrel"},
    {"pdb_id": "1CTF", "chain": "A", "res_range": None, "note": "Ribosomal L7/L12 C-terminal α/β domain"},
    {"pdb_id": "2LZM", "chain": "A", "res_range": None, "note": "Lysozyme classic α+β fold"},
    {"pdb_id": "1VII", "chain": "A", "res_range": None, "note": "Villin headpiece subdomain"},
    {"pdb_id": "7JJK", "chain": "A", "res_range": None, "note": "De novo designed miniprotein binder"},
    {"pdb_id": "6NUK", "chain": "A", "res_range": None, "note": "De novo designed 4-helix bundle"},
    {"pdb_id": "1RIS", "chain": "A", "res_range": None, "note": "Ribosomal protein S6 ferredoxin-like α/β"},
]

FANG_WU_SEEDS: List[Dict[str, Any]] = [
    {"pdb_id": "37NT", "chain": "A", "res_range": (95, 194), "note": "MERTK Ig Domain 1 (Oct 2026 @WUFang)"},
    {"pdb_id": "37NT", "chain": "A", "res_range": (196, 283), "note": "MERTK Ig Domain 2 (Oct 2026 @WUFang)"},
    {"pdb_id": "37NT", "chain": "A", "res_range": (286, 380), "note": "MERTK FnIII Domain 3 (Oct 2026 @WUFang)"},
    {"pdb_id": "37NT", "chain": "A", "res_range": (385, 481), "note": "MERTK FnIII Domain 4 (Oct 2026 @WUFang)"},
]


def write_native_backbone_pdb(struct: ProteinStructure, out_path: Path) -> None:
    """Write native-residue `(N, CA, C, O)` PDB strictly for Foldseek 3Di+AA indexing (not prompts)."""
    aa1_to_3 = {v: k for k, v in AA3_TO_1.items() if k != "MSE"}
    lines = []
    serial = 1
    center = struct.coords_ca.mean(axis=0)
    for idx, aa in enumerate(struct.sequence):
        r3 = aa1_to_3.get(aa, "ALA")
        for aname, arr, el in (
            ("N", struct.coords_n[idx] - center, "N"),
            ("CA", struct.coords_ca[idx] - center, "C"),
            ("C", struct.coords_c[idx] - center, "C"),
            ("O", struct.coords_o[idx] - center, "O"),
        ):
            lines.append(
                f"ATOM  {serial:5d}  {aname:<3s} {r3} A{idx + 1:4d}    "
                f"{arr[0]:8.3f}{arr[1]:8.3f}{arr[2]:8.3f}  1.00 20.00           {el:>2s}"
            )
            serial += 1
    lines.append("TER\nEND\n")
    out_path.write_text("\n".join(lines), encoding="utf-8")


def kmer_jaccard(seq_a: str, seq_b: str, k: int = 3) -> float:
    if seq_a == seq_b:
        return 1.0
    ka = {seq_a[i : i + k] for i in range(max(1, len(seq_a) - k + 1))}
    kb = {seq_b[i : i + k] for i in range(max(1, len(seq_b) - k + 1))}
    return len(ka & kb) / float(max(1, min(len(ka), len(kb))))


def nw_global_seqid(seq_a: str, seq_b: str) -> float:
    """Exact Needleman-Wunsch / banded sequence identity (%) between query and hit sequence."""
    if not seq_a or not seq_b:
        return 0.0
    la, lb = len(seq_a), len(seq_b)
    if abs(la - lb) > max(la, lb) * 0.75:
        # Use k-mer upper bound only when lengths differ drastically, else exact DP
        pass
    # Space-efficient 2-row global/semi-global alignment maximizing matches (gap = -1)
    prev_score = np.zeros(lb + 1, dtype=np.int32)
    prev_matches = np.zeros(lb + 1, dtype=np.int32)
    for i in range(1, la + 1):
        cur_score = np.zeros(lb + 1, dtype=np.int32)
        cur_matches = np.zeros(lb + 1, dtype=np.int32)
        ca = seq_a[i - 1]
        for j in range(1, lb + 1):
            m = 2 if ca == seq_b[j - 1] else -1
            diag_s = prev_score[j - 1] + m
            up_s = prev_score[j] - 1
            left_s = cur_score[j - 1] - 1
            if diag_s >= up_s and diag_s >= left_s:
                cur_score[j] = diag_s
                cur_matches[j] = prev_matches[j - 1] + (1 if ca == seq_b[j - 1] else 0)
            elif up_s >= left_s:
                cur_score[j] = up_s
                cur_matches[j] = prev_matches[j]
            else:
                cur_score[j] = left_s
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


def search_rcsb_post_cutoff_ids(client: httpx.Client, synthetic_only: bool, rows: int = 500) -> List[str]:
    nodes: List[Dict[str, Any]] = [
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
                "operator": "less_or_equal",
                "value": 2 if synthetic_only else 1,
            },
        },
        {
            "type": "terminal",
            "service": "text",
            "parameters": {
                "attribute": "entity_poly.rcsb_sample_sequence_length",
                "operator": "range",
                "value": {"from": 48, "include_lower": True, "to": 220, "include_upper": True},
            },
        },
    ]
    if synthetic_only:
        nodes.append(
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
    query = {
        "query": {"type": "group", "logical_operator": "and", "nodes": nodes},
        "return_type": "entry",
        "request_options": {
            "paginate": {"start": 0, "rows": rows},
            "sort": [{"sort_by": "rcsb_accession_info.initial_release_date", "direction": "desc"}],
        },
    }
    r = client.post("https://search.rcsb.org/rcsbsearch/v2/query", json=query)
    if r.status_code == 204:
        return []
    r.raise_for_status()
    return [x["identifier"] for x in r.json().get("result_set", [])]


def batch_fetch_pdb_release_dates_and_seqs(
    client: httpx.Client,
    entity_ids: List[str],
    date_cache: Dict[str, Tuple[str, str]],
) -> None:
    """Batch-query RCSB GraphQL for release dates and canonical sequences of polymer_entities."""
    missing = [eid for eid in set(entity_ids) if eid not in date_cache]
    for i in range(0, len(missing), 60):
        chunk = missing[i : i + 60]
        gql = """
        query($ids: [String!]!) {
          polymer_entities(entity_ids: $ids) {
            rcsb_id
            entity_poly { pdbx_seq_one_letter_code_can }
            entry { rcsb_accession_info { initial_release_date } }
          }
        }
        """
        for _att in range(3):
            try:
                r = client.post(
                    "https://data.rcsb.org/graphql",
                    json={"query": gql, "variables": {"ids": chunk}},
                    timeout=35.0,
                )
                if r.status_code == 200:
                    for item in (r.json().get("data") or {}).get("polymer_entities") or []:
                        rid = item["rcsb_id"]
                        rel = str(((item.get("entry") or {}).get("rcsb_accession_info") or {}).get("initial_release_date") or "")[:10]
                        seq = str((item.get("entity_poly") or {}).get("pdbx_seq_one_letter_code_can") or "").replace("\n", "")
                        date_cache[rid] = (rel, seq)
                break
            except httpx.HTTPError:
                time.sleep(1.5)


def compute_precutoff_pdb_seq_similarity(
    client: httpx.Client,
    seq: str,
    self_pdb_id: str,
    sim_cache_dir: Path,
    entity_date_cache: Dict[str, Tuple[str, str]],
) -> Dict[str, Any]:
    """Query standalone RCSB MMseqs2 sequence search & compute exact max SeqID to pre-cutoff PDB."""
    h8 = hashlib.md5(seq.encode("utf-8")).hexdigest()[:8]
    cfile = sim_cache_dir / f"rcsb_seq_{self_pdb_id}_{len(seq)}_{h8}.json"
    if cfile.exists():
        return json.loads(cfile.read_text(encoding="utf-8"))

    hits = []
    for id_cut, ev_cut in ((0.20, 0.1), (0.28, 0.01)):
        q = {
            "query": {
                "type": "terminal",
                "service": "sequence",
                "parameters": {
                    "evalue_cutoff": ev_cut,
                    "identity_cutoff": id_cut,
                    "sequence_type": "protein",
                    "value": seq,
                },
            },
            "return_type": "polymer_entity",
            "request_options": {
                "results_content_type": ["experimental"],
                "scoring_strategy": "sequence",
                "return_all_hits": False,
                "paginate": {"start": 0, "rows": 35},
            },
        }
        try:
            r = client.post("https://search.rcsb.org/rcsbsearch/v2/query", json=q, timeout=35.0)
            hits = [] if r.status_code == 204 else r.json().get("result_set", [])
            break
        except httpx.HTTPError:
            continue
    eids = [h["identifier"] for h in hits]
    batch_fetch_pdb_release_dates_and_seqs(client, eids, entity_date_cache)

    best_seqid = 0.0
    best_hit = None
    best_rel = None
    for h in hits:
        eid = h["identifier"]
        pdb4 = eid.split("_")[0].upper()
        rel, hseq = entity_date_cache.get(eid, ("", ""))
        if not rel or rel >= CUTOFF_DATE:
            continue
        sid = nw_global_seqid(seq, hseq)
        if sid > best_seqid:
            best_seqid = sid
            best_hit = eid
            best_rel = rel

    res = {
        "pdb_precutoff_seqid_max": best_seqid,
        "pdb_precutoff_closest_entity": best_hit,
        "pdb_precutoff_closest_release_date": best_rel,
        "total_pdb_seq_hits_any_date": len(hits),
    }
    cfile.write_text(json.dumps(res, indent=2), encoding="utf-8")
    return res


async def query_foldseek_with_cache(
    aclient: httpx.AsyncClient,
    sem: asyncio.Semaphore,
    target_key: str,
    native_bb_pdb: Path,
    sim_cache_dir: Path,
) -> Dict[str, Any]:
    """Query Foldseek (`pdb100` + `afdb-swissprot` + `afdb50`) with disk cache & polite pacing."""
    cfile = sim_cache_dir / f"foldseek_{target_key}.json"
    if cfile.exists():
        return json.loads(cfile.read_text(encoding="utf-8"))

    async with sem:
        pdb_bytes = native_bb_pdb.read_bytes()
        ticket_id = None
        for attempt in range(6):
            try:
                r = await aclient.post(
                    "https://search.foldseek.com/api/ticket",
                    data={"mode": "3diaa", "database[]": ["pdb100", "afdb-swissprot", "afdb50"]},
                    files={"q": (native_bb_pdb.name, pdb_bytes, "chemical/x-pdb")},
                )
                if r.status_code == 429:
                    await asyncio.sleep(4.0 * (attempt + 1))
                    continue
                r.raise_for_status()
                ticket_id = r.json().get("id")
                break
            except Exception:
                await asyncio.sleep(3.0 * (attempt + 1))

        if not ticket_id:
            return {"error": "ticket_failed"}

        for _ in range(35):
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
                        slim = []
                        for a in flat[:8]:
                            slim.append(
                                {
                                    "target": a.get("target", ""),
                                    "seqId": float(a.get("seqId", 0.0)),
                                    "prob": float(a.get("prob", 0.0)),
                                    "eval": float(a.get("eval", 999.0)),
                                    "score": float(a.get("score", 0.0)),
                                    "alnLength": int(a.get("alnLength", 0)),
                                    "qLen": int(a.get("qLen", 0)),
                                }
                            )
                        parsed[db] = slim
                    cfile.write_text(json.dumps(parsed, indent=2), encoding="utf-8")
                    return parsed
                if status == "ERROR":
                    return {"error": "foldseek_job_error"}
            except Exception:
                continue
        return {"error": "timeout"}


def extract_foldseek_summary(
    fs_raw: Dict[str, Any],
    pdb_date_lookup: Dict[str, str],
) -> Dict[str, Any]:
    """Summarize Foldseek hits against (a) pre-cutoff pdb100 and (b) AFDB (afdb-swissprot + afdb50)."""
    # 1. Pre-cutoff PDB100 structural similarity
    pdb_hits = fs_raw.get("pdb100", []) or []
    best_pdb_prob = 0.0
    best_pdb_eval = 999.0
    best_pdb_score = 0.0
    best_pdb_seqid = 0.0
    best_pdb_target = None

    for h in pdb_hits:
        t_str = str(h.get("target", ""))
        pdb_code = t_str[:4].upper()
        rel = pdb_date_lookup.get(pdb_code, "1999-01-01")
        if rel >= CUTOFF_DATE:
            continue
        if h["score"] > best_pdb_score or h["prob"] > best_pdb_prob:
            if h["prob"] >= best_pdb_prob:
                best_pdb_prob = h["prob"]
                best_pdb_eval = h["eval"]
                best_pdb_score = h["score"]
                best_pdb_seqid = h["seqId"]
                best_pdb_target = t_str.split(" ")[0]

    # 2. AlphaFold DB (Predicted Structures: afdb-swissprot + afdb50)
    af_hits = (fs_raw.get("afdb-swissprot", []) or []) + (fs_raw.get("afdb50", []) or [])
    best_af_prob = 0.0
    best_af_eval = 999.0
    best_af_score = 0.0
    best_af_seqid = 0.0
    best_af_target = None

    for h in af_hits:
        if h["seqId"] > best_af_seqid:
            best_af_seqid = h["seqId"]
        if h["prob"] > best_af_prob or (h["prob"] == best_af_prob and h["score"] > best_af_score):
            best_af_prob = h["prob"]
            best_af_eval = h["eval"]
            best_af_score = h["score"]
            best_af_target = str(h.get("target", "")).split(" ")[0]

    return {
        "pdb100_precutoff_foldseek_prob": round(best_pdb_prob, 3),
        "pdb100_precutoff_foldseek_eval": best_pdb_eval,
        "pdb100_precutoff_foldseek_score": best_pdb_score,
        "pdb100_precutoff_foldseek_seqid": round(best_pdb_seqid, 1),
        "pdb100_precutoff_closest_struct": best_pdb_target,
        "afdb_max_seqid": round(best_af_seqid, 1),
        "afdb_foldseek_prob": round(best_af_prob, 3),
        "afdb_foldseek_eval": best_af_eval,
        "afdb_foldseek_score": best_af_score,
        "afdb_closest_predicted_struct": best_af_target,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Build N=100 Stage 1 benchmark with PDB & AFDB similarity.")
    parser.add_argument("--out-dir", default="data/stage1_benchmark")
    parser.add_argument("--cache-dir", default="data/raw_pdb_cache")
    parser.add_argument("--target-total", type=int, default=100)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    raw_cache = Path(args.cache_dir)
    sim_cache = raw_cache / "similarity_cache"
    native_bb_dir = raw_cache / "native_backbones_for_foldseek"
    gt_dir = out_dir / "ground_truth"
    bb_dir = out_dir / "backbones_stripped"

    for d in (out_dir, raw_cache, sim_cache, native_bb_dir, gt_dir, bb_dir):
        d.mkdir(parents=True, exist_ok=True)
    # Clean previous numbering in gt_dir and bb_dir so 1..100 numbering is clean
    for old_f in list(gt_dir.glob("s1_t*")) + list(bb_dir.glob("s1_t*")):
        old_f.unlink()

    entity_date_cache: Dict[str, Tuple[str, str]] = {}
    accepted_candidates: List[Tuple[ProteinStructure, Dict[str, Any]]] = []
    accepted_seqs: List[str] = []

    Quotas = {
        "pre_cutoff_control": 15,
        "post_cutoff_de_novo": 30,
        "post_cutoff_pdb_novel": 40,
        "post_cutoff_homolog_gradient": 15,
    }
    counts = {k: 0 for k in Quotas}

    def fetch_and_parse(
        client: httpx.Client,
        pdb_id: str,
        chain_pref: Optional[str],
        res_range: Optional[Tuple[int, int]],
    ) -> Optional[Tuple[ProteinStructure, Dict[str, Any]]]:
        pdb_u = pdb_id.upper()
        meta_p = raw_cache / f"{pdb_u}.json"
        ent_p = raw_cache / f"{pdb_u}_ent1.json"
        cif_p = raw_cache / f"{pdb_u}.cif"

        try:
            if not meta_p.exists():
                rm = client.get(f"https://data.rcsb.org/rest/v1/core/entry/{pdb_u}", timeout=25.0)
                if rm.status_code != 200:
                    return None
                meta_p.write_text(rm.text, encoding="utf-8")
            meta = json.loads(meta_p.read_text(encoding="utf-8"))

            if not ent_p.exists():
                re_resp = client.get(f"https://data.rcsb.org/rest/v1/core/polymer_entity/{pdb_u}/1", timeout=25.0)
                if re_resp.status_code == 200:
                    ent_p.write_text(re_resp.text, encoding="utf-8")
            ent = json.loads(ent_p.read_text(encoding="utf-8")) if ent_p.exists() else {}

            title = str((meta.get("struct") or {}).get("title", ""))
            kw = str((meta.get("struct_keywords") or {}).get("pdbx_keywords", ""))
            orgs = [str(s.get("ncbi_scientific_name", "")) for s in ent.get("rcsb_entity_source_organism") or []]
            desc = str((ent.get("rcsb_polymer_entity") or {}).get("pdbx_description", ""))
            if SAFETY_BLOCKLIST.search(f"{title} {kw} {' '.join(orgs)} {desc}"):
                return None

            if not cif_p.exists():
                rc = client.get(f"https://files.rcsb.org/download/{pdb_u}.cif", timeout=35.0)
                if rc.status_code != 200:
                    return None
                cif_p.write_text(rc.text, encoding="utf-8")
        except httpx.HTTPError:
            return None

        try:
            st = load_structure(cif_p, pdb_id=pdb_u, chain_id=chain_pref, res_range=res_range)
        except Exception:
            return None

        if not (48 <= st.length <= 220):
            return None
        cn_dists = np.linalg.norm(st.coords_n[1:] - st.coords_c[:-1], axis=1)
        if bool(np.any(cn_dists > 2.4)):
            return None
        if (st.q3.count("H") + st.q3.count("E")) / float(st.length) < 0.25:
            return None
        ii, jj = np.triu_indices(st.length, k=12)
        if int(np.sum(st.cb_dist_matrix[ii, jj] < 8.0)) < max(10, st.length // 5):
            return None

        # Non-redundant within our 100-target benchmark (allow 37NT's distinct domains)
        if any(kmer_jaccard(st.sequence, prev) > 0.25 for prev in accepted_seqs):
            return None

        acc = meta.get("rcsb_accession_info") or {}
        res_arr = (meta.get("rcsb_entry_info") or {}).get("resolution_combined") or [None]
        info = {
            "pdb_id": pdb_u,
            "chain_id": st.chain_id,
            "res_range": list(res_range) if res_range else [st.res_numbers[0], st.res_numbers[-1]],
            "release_date": str(acc.get("initial_release_date", ""))[:10],
            "deposit_date": str(acc.get("deposit_date", ""))[:10],
            "resolution_A": float(res_arr[0]) if res_arr and res_arr[0] is not None else None,
            "organism": orgs[0] if orgs else "unknown",
            "description": desc or title[:80],
        }
        return st, info

    t_start = time.perf_counter()
    with httpx.Client(timeout=25.0, follow_redirects=True) as client:
        print("=== Phase 1: Curating Pre-2025 Anchor Controls (Quota = 15) ===")
        for seed in PRE_CUTOFF_SEEDS:
            if counts["pre_cutoff_control"] >= Quotas["pre_cutoff_control"]:
                break
            out = fetch_and_parse(client, seed["pdb_id"], seed["chain"], seed["res_range"])
            if out is None:
                continue
            st, info = out
            seq_sim = compute_precutoff_pdb_seq_similarity(client, st.sequence, info["pdb_id"], sim_cache, entity_date_cache)
            info.update(seq_sim)
            info["stratum"] = "pre_cutoff_control"
            info["temporal_split"] = "pre_cutoff"
            info["note"] = seed["note"]
            accepted_candidates.append((st, info))
            accepted_seqs.append(st.sequence)
            counts["pre_cutoff_control"] += 1
            print(f"  [Control {counts['pre_cutoff_control']:2d}/15] {info['pdb_id']} (L={st.length:3d}) | {seed['note']}")

        print("\n=== Phase 2: Curating @WUFang Oct-2026 37NT Domains ===")
        for seed in FANG_WU_SEEDS:
            out = fetch_and_parse(client, seed["pdb_id"], seed["chain"], seed["res_range"])
            if out is None:
                continue
            st, info = out
            seq_sim = compute_precutoff_pdb_seq_similarity(client, st.sequence, f"{info['pdb_id']}_{info['res_range'][0]}", sim_cache, entity_date_cache)
            info.update(seq_sim)
            stratum = "post_cutoff_pdb_novel" if seq_sim["pdb_precutoff_seqid_max"] < 30.0 else "post_cutoff_homolog_gradient"
            info["stratum"] = stratum
            info["temporal_split"] = "post_cutoff"
            info["note"] = seed["note"]
            accepted_candidates.append((st, info))
            accepted_seqs.append(st.sequence)
            counts[stratum] += 1
            print(
                f"  [37NT] {info['pdb_id']} {info['res_range']} (L={st.length:3d}) "
                f"-> pre-cutoff PDB max SeqID = {seq_sim['pdb_precutoff_seqid_max']:.1f}% [{stratum}]"
            )

        print("\n=== Phase 3: Discovering 2026 Synthetic De Novo Designs (Quota = 25) ===")
        synth_ids = search_rcsb_post_cutoff_ids(client, synthetic_only=True, rows=400)
        print(f"  Found {len(synth_ids)} post-May-2026 synthetic construct PDB candidates.")
        for pid in synth_ids:
            if counts["post_cutoff_de_novo"] >= Quotas["post_cutoff_de_novo"]:
                break
            out = fetch_and_parse(client, pid, None, None)
            if out is None:
                continue
            st, info = out
            seq_sim = compute_precutoff_pdb_seq_similarity(client, st.sequence, pid, sim_cache, entity_date_cache)
            if seq_sim["pdb_precutoff_seqid_max"] >= 30.0:
                continue  # Reject trivial point mutants of old synthetic proteins!
            info.update(seq_sim)
            info["stratum"] = "post_cutoff_de_novo"
            info["temporal_split"] = "post_cutoff"
            info["note"] = f"2026 synthetic/de novo ({info['release_date']})"
            accepted_candidates.append((st, info))
            accepted_seqs.append(st.sequence)
            counts["post_cutoff_de_novo"] += 1
            print(
                f"  [DeNovo {counts['post_cutoff_de_novo']:2d}/{Quotas['post_cutoff_de_novo']}] "
                f"{pid} (L={st.length:3d}, rel={info['release_date']}, prePDB_SeqID={seq_sim['pdb_precutoff_seqid_max']:.1f}%) | {info['description'][:42]}"
            )

        print("\n=== Phase 4: Discovering 2026 PDB-Novel Natural Structures (<30% SeqID) & Homolog Gradient ===")
        nat_ids = search_rcsb_post_cutoff_ids(client, synthetic_only=False, rows=900)
        print(f"  Found {len(nat_ids)} post-May-2026 single-entity PDB candidates.")
        for pid in nat_ids:
            need_novel = counts["post_cutoff_pdb_novel"] < Quotas["post_cutoff_pdb_novel"] or (
                len(accepted_candidates) < args.target_total
            )
            need_hom = counts["post_cutoff_homolog_gradient"] < Quotas["post_cutoff_homolog_gradient"]
            if len(accepted_candidates) >= args.target_total and not (need_novel or need_hom):
                break
            if pid.upper() == "37NT":
                continue
            out = fetch_and_parse(client, pid, None, None)
            if out is None:
                continue
            st, info = out
            seq_sim = compute_precutoff_pdb_seq_similarity(client, st.sequence, pid, sim_cache, entity_date_cache)
            max_sid = seq_sim["pdb_precutoff_seqid_max"]

            is_synth = "synthetic" in info["organism"].lower() or "de novo" in info["description"].lower()
            if max_sid < 30.0:
                if is_synth and counts["post_cutoff_de_novo"] < Quotas["post_cutoff_de_novo"]:
                    chosen_stratum = "post_cutoff_de_novo"
                elif len(accepted_candidates) - counts["post_cutoff_homolog_gradient"] < (
                    args.target_total - Quotas["post_cutoff_homolog_gradient"]
                ):
                    chosen_stratum = "post_cutoff_pdb_novel"
                else:
                    continue
            elif 30.0 <= max_sid <= 96.0 and counts["post_cutoff_homolog_gradient"] < Quotas["post_cutoff_homolog_gradient"]:
                chosen_stratum = "post_cutoff_homolog_gradient"
            else:
                continue

            info.update(seq_sim)
            info["stratum"] = chosen_stratum
            info["temporal_split"] = "post_cutoff"
            info["note"] = f"2026 {chosen_stratum.replace('post_cutoff_', '')} ({info['release_date']})"
            accepted_candidates.append((st, info))
            accepted_seqs.append(st.sequence)
            counts[chosen_stratum] += 1
            print(
                f"  [{len(accepted_candidates):3d}/{args.target_total}] {pid} (L={st.length:3d}, rel={info['release_date']}) "
                f"prePDB_SeqID={max_sid:5.1f}% -> {chosen_stratum}"
            )
            if len(accepted_candidates) >= args.target_total:
                break

    print(f"\n=== Phase 5: Running Foldseek Structure+Sequence Search (`pdb100` + `afdb-swissprot` + `afdb50`) on {len(accepted_candidates)} Targets ===")
    native_paths: List[Tuple[str, Path]] = []
    for idx, (st, info) in enumerate(accepted_candidates):
        tkey = f"{info['pdb_id']}_{info['res_range'][0]}_{info['res_range'][1]}"
        np_p = native_bb_dir / f"{tkey}.pdb"
        write_native_backbone_pdb(st, np_p)
        native_paths.append((tkey, np_p))

    async def _run_all_foldseek() -> List[Dict[str, Any]]:
        sem = asyncio.Semaphore(3)
        async with httpx.AsyncClient(timeout=45.0, follow_redirects=True) as aclient:
            tasks = [
                query_foldseek_with_cache(aclient, sem, tkey, np_p, sim_cache)
                for (tkey, np_p) in native_paths
            ]
            return await asyncio.gather(*tasks)

    fs_results = asyncio.run(_run_all_foldseek())

    # Batch-lookup release dates of all top pdb100 hits so post-May-2026 hits in pdb100 are excluded!
    pdb100_codes = set()
    for fsr in fs_results:
        for h in fsr.get("pdb100", []) or []:
            code = str(h.get("target", ""))[:4].upper()
            if len(code) == 4:
                pdb100_codes.add(f"{code}_1")
    with httpx.Client(timeout=25.0) as client:
        batch_fetch_pdb_release_dates_and_seqs(client, sorted(pdb100_codes), entity_date_cache)
    pdb_date_lookup = {k.split("_")[0].upper(): v[0] for k, v in entity_date_cache.items() if v[0]}

    # Finalize 1..N anonymized benchmark files & manifest
    final_records: List[Dict[str, Any]] = []
    for idx, ((st, info), fsr) in enumerate(zip(accepted_candidates, fs_results), start=1):
        tid = f"s1_t{idx:03d}"
        st.target_id = tid
        write_backbone_only_pdb(st, bb_dir / f"{tid}.pdb")
        write_ca_trace_pdb(st.sequence, st.coords_ca, gt_dir / f"{tid}_ca.pdb")
        np.savez_compressed(
            gt_dir / f"{tid}.npz",
            sequence=st.sequence,
            q3=st.q3,
            res_numbers=np.array(st.res_numbers, dtype=np.int32),
            coords_n=st.coords_n,
            coords_ca=st.coords_ca,
            coords_c=st.coords_c,
            coords_o=st.coords_o,
            coords_cb=st.coords_cb,
            phi_deg=st.phi_deg,
            psi_deg=st.psi_deg,
            cb_dist_matrix=st.cb_dist_matrix,
            ca_coord_num=st.ca_coord_num,
        )

        fs_summary = extract_foldseek_summary(fsr, pdb_date_lookup)
        ii12, jj12 = np.triu_indices(st.length, k=12)
        ii24, jj24 = np.triu_indices(st.length, k=24)

        rec = {
            "target_id": tid,
            "pdb_id": info["pdb_id"],
            "chain_id": info["chain_id"],
            "res_range": info["res_range"],
            "length": st.length,
            "temporal_split": info["temporal_split"],
            "stratum": info["stratum"],
            "size_bin": "compact_50_100" if st.length <= 100 else "medium_101_220",
            "fold_class": classify_fold_q3(st.q3),
            "release_date": info["release_date"],
            "deposit_date": info["deposit_date"],
            "resolution_A": info["resolution_A"],
            "organism": info["organism"],
            "description": info["description"],
            "note": info["note"],
            "q3_helix_frac": round(st.q3.count("H") / st.length, 3),
            "q3_strand_frac": round(st.q3.count("E") / st.length, 3),
            "n_contacts_sep12": int(np.sum(st.cb_dist_matrix[ii12, jj12] < 8.0)),
            "n_contacts_sep24": int(np.sum(st.cb_dist_matrix[ii24, jj24] < 8.0)),
            # (a) Sequence & Structure Similarity to Pre-Cutoff Experimental PDB
            "pdb_precutoff_seqid_max": info["pdb_precutoff_seqid_max"],
            "pdb_precutoff_closest_entity": info["pdb_precutoff_closest_entity"],
            "pdb_precutoff_closest_release_date": info["pdb_precutoff_closest_release_date"],
            **fs_summary,
        }
        final_records.append(rec)

    manifest = {
        "benchmark": "fable-protein-bench-stage1-v2",
        "cutoff_date": CUTOFF_DATE,
        "total_targets": len(final_records),
        "strata_counts": {
            s: sum(1 for r in final_records if r["stratum"] == s)
            for s in Quotas
        },
        "n_pre_cutoff": sum(1 for r in final_records if r["temporal_split"] == "pre_cutoff"),
        "n_post_cutoff": sum(1 for r in final_records if r["temporal_split"] == "post_cutoff"),
        "targets": final_records,
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(
        f"\nCompleted N={len(final_records)} curation + PDB/AFDB similarity profiling in "
        f"{time.perf_counter() - t_start:.1f}s -> {out_dir / 'manifest.json'}"
    )
    print("Strata breakdown:", json.dumps(manifest["strata_counts"]))


if __name__ == "__main__":
    sys.exit(main())
