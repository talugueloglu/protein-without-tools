#!/usr/bin/env python3
"""Single-pass Raw Public Data Downloader for Stage 2A (TEA + SCOPe + Foldseek) & Stage 2B (CAFA 2026 GOA).

Downloads & caches into `data/raw_stage2_cache/`:
  1. Stage 2A (`tea_scope/`):
     - `PickyBinders/tea` repository assets: `matcha.out`, `model.py`, `convert.py`,
       `fold_split.json`, `lookup_file.tsv`, and `scope40_2_08_from_alignments.fasta`.
     - ASTRAL `SCOPe 2.08` 3D domain PDBs for ~55 query domains across diverse held-out superfamilies
       + `Foldseek mode="3diaa"` (`pdb100`) search results so the offline builder can cleanly separate
       Rung 2 (`Foldseek-caught`) from Rung 3 (`Foldseek-missed / Deep TEA Zone`).
  2. Stage 2B (`cafa_goa/`):
     - `go-basic.obo` (`current.geneontology.org`)
     - Candidate 2026 UniProt proteins with experimental GO annotations + exact `EBI QuickGO`
       timestamped experimental annotation histories (`ECO:0000269` descendants) to isolate true
       post-cutoff (`2026`) CAFA No-Knowledge (`NK`) and Limited-Knowledge (`LK`) proteins.
     - Pre-2026 Swiss-Prot experimental GO reference bank (`~2,000` proteins across major clades)
       for official `CAFA Naive` + `Sequence Homology Transfer` + Modal `ESM-2 k-NN` baselines.
"""

from __future__ import annotations

import asyncio
from collections import defaultdict
import json
from pathlib import Path
import random
import re
from typing import Any, Dict, List, Set, Tuple

import httpx

from fable_bench.pdb_utils import AA3_TO_1, CANONICAL_AA_SET, load_structure
from fable_bench.stage2a_homology import smith_waterman_affine

SAFETY_BLOCKLIST = re.compile(
    r"\b(virus|viral|virion|capsid|phage|bacteriophage|toxin|venom|ricin|abrin|"
    r"botulinum|anthrax|ebola|marburg|variola|poxvirus|influenza|coronavirus|sars|hiv)\b",
    re.IGNORECASE,
)


def parse_fasta(text: str) -> Dict[str, Tuple[str, str]]:
    """Return `{id: (header, UPPER_SEQ)}`."""
    out: Dict[str, Tuple[str, str]] = {}
    cur_id = ""
    cur_hdr = ""
    buf: List[str] = []
    for line in text.splitlines():
        s = line.strip()
        if not s:
            continue
        if s.startswith(">"):
            if cur_id and buf:
                seq = "".join(buf).upper()
                if all(c in CANONICAL_AA_SET for c in seq):
                    out[cur_id] = (cur_hdr, seq)
            cur_hdr = s[1:]
            cur_id = cur_hdr.split()[0]
            buf = []
        else:
            buf.append(s)
    if cur_id and buf:
        seq = "".join(buf).upper()
        if all(c in CANONICAL_AA_SET for c in seq):
            out[cur_id] = (cur_hdr, seq)
    return out


async def download_scope_pdbs_and_foldseek(
    query_domains: List[str],
    pdb_dir: Path,
    fs_dir: Path,
) -> None:
    sem = asyncio.Semaphore(5)
    limits = httpx.Limits(max_connections=10, max_keepalive_connections=5)

    aa1_to_3 = {v: k for k, v in AA3_TO_1.items() if k != "MSE"}

    async def _one_domain(ac: httpx.AsyncClient, dom_id: str) -> None:
        pdb_p = pdb_dir / f"{dom_id}.pdb"
        if not pdb_p.exists():
            pdb4 = dom_id[1:5].upper()
            ch_raw = dom_id[5]
            ch_id = ch_raw.upper() if ch_raw.isalnum() else None
            cif_tmp = pdb_dir / f"_raw_{pdb4}.cif"
            async with sem:
                if not cif_tmp.exists():
                    try:
                        rc = await ac.get(f"https://files.rcsb.org/download/{pdb4}.cif")
                        if rc.status_code == 200:
                            cif_tmp.write_text(rc.text, encoding="utf-8")
                    except Exception:
                        pass
            if cif_tmp.exists():
                try:
                    st = load_structure(cif_tmp, pdb_id=pdb4, chain_id=ch_id)
                    lines = []
                    ser = 1
                    for idx, aa in enumerate(st.sequence[:260]):
                        r3 = aa1_to_3.get(aa, "ALA")
                        for an, arr, el in (
                            ("N", st.coords_n[idx], "N"),
                            ("CA", st.coords_ca[idx], "C"),
                            ("C", st.coords_c[idx], "C"),
                            ("O", st.coords_o[idx], "O"),
                        ):
                            lines.append(
                                f"ATOM  {ser:5d}  {an:<3s} {r3} A{idx + 1:4d}    "
                                f"{arr[0]:8.3f}{arr[1]:8.3f}{arr[2]:8.3f}  1.00 20.00           {el:>2s}"
                            )
                            ser += 1
                    lines.append("TER\nEND\n")
                    if ser > 80:
                        pdb_p.write_text("\n".join(lines), encoding="utf-8")
                except Exception:
                    pass

        if not pdb_p.exists():
            return

        fs_p = fs_dir / f"fs3di_{dom_id}.json"
        if fs_p.exists():
            return

        async with sem:
            pbytes = pdb_p.read_bytes()
            tid = None
            for att in range(5):
                try:
                    rt = await ac.post(
                        "https://search.foldseek.com/api/ticket",
                        data={"mode": "3diaa", "database[]": ["pdb100"]},
                        files={"q": (pdb_p.name, pbytes, "chemical/x-pdb")},
                    )
                    if rt.status_code == 429:
                        await asyncio.sleep(3.0 * (att + 1))
                        continue
                    rt.raise_for_status()
                    tid = rt.json().get("id")
                    break
                except Exception:
                    await asyncio.sleep(2.0 * (att + 1))
            if not tid:
                return
            for _ in range(28):
                await asyncio.sleep(1.4)
                try:
                    st = (await ac.get(f"https://search.foldseek.com/api/ticket/{tid}")).json()
                    if st.get("status") == "COMPLETE":
                        res = (await ac.get(f"https://search.foldseek.com/api/result/{tid}/0")).json()
                        hits = []
                        for dbe in res.get("results", []):
                            if dbe.get("db") == "pdb100":
                                alns = dbe.get("alignments") or []
                                flat = alns[0] if (alns and isinstance(alns[0], list)) else alns
                                for a in flat[:80]:
                                    hits.append(
                                        {
                                            "target": str(a.get("target", "")).split(" ")[0],
                                            "eval": float(a.get("eval", 999.0)),
                                            "prob": float(a.get("prob", 0.0)),
                                            "score": float(a.get("score", 0.0)),
                                            "seqId": float(a.get("seqId", 0.0)),
                                        }
                                    )
                        fs_p.write_text(json.dumps(hits), encoding="utf-8")
                        print(f"    [SCOPe Foldseek] {dom_id} -> {len(hits)} pdb100 hits", flush=True)
                        return
                    if st.get("status") == "ERROR":
                        return
                except Exception:
                    continue

    async with httpx.AsyncClient(timeout=40.0, follow_redirects=True, limits=limits) as ac:
        await asyncio.gather(*[_one_domain(ac, d) for d in query_domains])


def main() -> None:
    repo = Path("/Users/taluguloglu/.gemini/antigravity/scratch/fable-protein-bench")
    raw_root = repo / "data" / "raw_stage2_cache"
    tea_dir = raw_root / "tea_scope"
    scope_pdb_dir = tea_dir / "pdbs"
    scope_fs_dir = tea_dir / "foldseek_pdb100"
    cafa_dir = raw_root / "cafa_goa"
    for d in (tea_dir, scope_pdb_dir, scope_fs_dir, cafa_dir):
        d.mkdir(parents=True, exist_ok=True)

    with httpx.Client(timeout=90.0, follow_redirects=True) as client:
        print("=== Part 1: Downloading PickyBinders/tea Assets & SCOPe 2.08 Split ===", flush=True)
        tea_files = {
            "matcha.out": "tea/matcha.out",
            "tea_model.py": "tea/model.py",
            "tea_convert.py": "tea/convert.py",
            "fold_split.json": "tea_train/data/scope40_2_08/fold_split.json",
            "lookup_file.tsv": "tea_train/data/scope40_2_08/lookup_file.tsv",
            "scope40_2_08.fasta": "tea_train/data/scope40_2_08/scope40_2_08_from_alignments.fasta",
        }
        for local_fn, gh_path in tea_files.items():
            dest = tea_dir / local_fn
            if not dest.exists():
                r = client.get(f"https://raw.githubusercontent.com/PickyBinders/tea/HEAD/{gh_path}")
                r.raise_for_status()
                dest.write_bytes(r.content)
            print(f"  Cached {local_fn} ({dest.stat().st_size / 1024:.1f} KB)", flush=True)

        # Select ~55 diverse SCOPe domains from multi-family superfamilies (classes a, b, c, d, 75 <= L <= 240)
        seqs_map = parse_fasta((tea_dir / "scope40_2_08.fasta").read_text(encoding="utf-8"))
        fold_split = json.loads((tea_dir / "fold_split.json").read_text(encoding="utf-8"))
        dom_to_sccs: Dict[str, str] = {}
        for line in (tea_dir / "lookup_file.tsv").read_text(encoding="utf-8").splitlines():
            parts = line.strip().split("\t")
            if len(parts) >= 2 and parts[0] in seqs_map:
                sccs = parts[1].strip()
                if sccs[0] in ("a", "b", "c", "d"):
                    seq_len = len(seqs_map[parts[0]][1])
                    if 70 <= seq_len <= 240:
                        dom_to_sccs[parts[0]] = sccs

        # Group by superfamily `c.f.sf` -> family `c.f.sf.fa` -> list of domains
        sf_to_fams: Dict[str, Dict[str, List[str]]] = defaultdict(lambda: defaultdict(list))
        for dom, sccs in dom_to_sccs.items():
            toks = sccs.split(".")
            if len(toks) >= 4:
                sf = ".".join(toks[:3])
                sf_to_fams[sf][sccs].append(dom)

        # Prioritize superfamilies in fold_split == 0 (held-out fold split) first, then others with >= 3 families!
        rich_sfs = [sf for sf, fams in sf_to_fams.items() if len(fams) >= 3]
        rich_sfs.sort(
            key=lambda sf: (
                0 if fold_split.get(".".join(sf.split(".")[:2]), 1) == 0 else 1,
                -len(sf_to_fams[sf]),
                sf,
            )
        )
        rng = random.Random(20261120)
        query_doms: List[str] = []
        for sf in rich_sfs[:52]:
            fams = sorted(sf_to_fams[sf].keys())
            d0 = sf_to_fams[sf][fams[0]][0]
            # Verify there is at least one cross-family partner with BLOSUM62 E > 1.5!
            s0 = seqs_map[d0][1]
            has_remote = False
            for f_other in fams[1:5]:
                d_other = sf_to_fams[sf][f_other][0]
                _b, ev, _sid = smith_waterman_affine(s0, seqs_map[d_other][1])
                if ev > 1.5:
                    has_remote = True
                    break
            if has_remote:
                query_doms.append(d0)
        rng.keep = query_doms[:55]
        print(f"  Fetching 3D ASTRAL PDBs & Foldseek pdb100 hits for {len(rng.keep)} remote-superfamily query domains...", flush=True)

    asyncio.run(download_scope_pdbs_and_foldseek(rng.keep, scope_pdb_dir, scope_fs_dir))

    with httpx.Client(timeout=120.0, follow_redirects=True) as client:
        print("\n=== Part 2: Downloading Gene Ontology DAG (`go-basic.obo`) & 2026 Experimental GOA Holdouts ===", flush=True)
        obo_file = cafa_dir / "go-basic.obo"
        if not obo_file.exists():
            ro = client.get("https://current.geneontology.org/ontology/go-basic.obo")
            ro.raise_for_status()
            obo_file.write_bytes(ro.content)
        print(f"  Cached go-basic.obo ({obo_file.stat().st_size / (1024 * 1024):.1f} MB)", flush=True)

        # Query UniProt across Swiss-Prot & TrEMBL for proteins with 2026 experimental GO papers / recent creation
        cand_uniprot: Dict[str, Dict[str, Any]] = {}
        uniprot_queries = [
            "(date_created:[2025-06-01 TO *]) AND (go_exp:*) AND (length:[60 TO 520])",
            "(lit_pubdate:2026) AND (go_exp:*) AND (taxonomy_id:2) AND (length:[60 TO 520])",
            "(lit_pubdate:2026) AND (go_exp:*) AND (taxonomy_id:4751) AND (length:[60 TO 520])",
            "(lit_pubdate:2026) AND (go_exp:*) AND (taxonomy_id:33090) AND (length:[60 TO 520])",
            "(lit_pubdate:2026) AND (go_exp:*) AND (taxonomy_id:33208) AND (length:[60 TO 520])",
            "(reviewed:true) AND (lit_pubdate:2025) AND (go_exp:*) AND (taxonomy_id:2) AND (length:[60 TO 520])",
        ]
        for uq in uniprot_queries:
            ru = client.get(
                "https://rest.uniprot.org/uniprotkb/search",
                params={
                    "query": uq,
                    "fields": "accession,id,protein_name,organism_name,length,sequence,date_created",
                    "size": 300,
                },
            )
            if ru.status_code == 200:
                for entry in ru.json().get("results", []):
                    acc = entry["primaryAccession"]
                    seq = str((entry.get("sequence") or {}).get("value") or "").strip().upper()
                    org = str((entry.get("organism") or {}).get("scientificName") or "")
                    pname = json.dumps(entry.get("proteinDescription") or {})
                    if not (60 <= len(seq) <= 520) or any(c not in CANONICAL_AA_SET for c in seq):
                        continue
                    if SAFETY_BLOCKLIST.search(f"{org} {pname}"):
                        continue
                    cand_uniprot[acc] = {
                        "accession": acc,
                        "entry_id": entry.get("uniProtkbId", ""),
                        "organism": org,
                        "sequence": seq,
                        "length": len(seq),
                        "date_created": (entry.get("entryAudit") or {}).get("firstPublicDate", ""),
                    }
        print(f"  Found {len(cand_uniprot)} candidate 2025/2026 experimental-GO UniProt proteins.", flush=True)
        (cafa_dir / "candidate_uniprot_entries.json").write_text(
            json.dumps(list(cand_uniprot.values()), indent=2), encoding="utf-8"
        )

        # Fetch exact timestamped experimental QuickGO annotations for all candidate accessions!
        qgo_file = cafa_dir / "quickgo_exp_annotations.json"
        accs = sorted(cand_uniprot.keys())
        anns_by_acc: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        for i in range(0, len(accs), 25):
            batch = accs[i : i + 25]
            gp_param = ",".join(f"UniProtKB:{a}" for a in batch)
            for pg_num in (1, 2):
                rq = client.get(
                    "https://www.ebi.ac.uk/QuickGO/services/annotation/search",
                    params={
                        "geneProductId": gp_param,
                        "evidenceCode": "ECO:0000269",
                        "evidenceCodeUsage": "descendants",
                        "limit": 200,
                        "page": pg_num,
                    },
                    headers={"Accept": "application/json"},
                )
                if rq.status_code != 200:
                    break
                jbody = rq.json()
                results = jbody.get("results") or []
                for row in results:
                    qual = str(row.get("qualifier") or "")
                    if "NOT" in qual.upper():
                        continue
                    gpid = str(row.get("geneProductId") or "").replace("UniProtKB:", "")
                    anns_by_acc[gpid].append(
                        {
                            "goId": row.get("goId"),
                            "goAspect": row.get("goAspect"),
                            "goEvidence": row.get("goEvidence"),
                            "date": str(row.get("date") or ""),
                            "reference": row.get("reference"),
                        }
                    )
                if pg_num >= int((jbody.get("pageInfo") or {}).get("total", 1)):
                    break
        qgo_file.write_text(json.dumps(anns_by_acc, indent=2), encoding="utf-8")
        print(f"  Cached timestamped QuickGO experimental annotations for {len(anns_by_acc)} proteins.", flush=True)

        # Download pre-2025 Swiss-Prot experimental GO reference bank (~1,800 proteins across diverse taxa)
        ref_file = cafa_dir / "pre2026_swissprot_go_reference.json"
        ref_entries: Dict[str, Dict[str, Any]] = {}
        taxon_queries = [
            "(reviewed:true) AND (date_created:[* TO 2024-12-31]) AND (go_exp:*) AND (organism_id:9606) AND (length:[60 TO 520])",
            "(reviewed:true) AND (date_created:[* TO 2024-12-31]) AND (go_exp:*) AND (organism_id:559292) AND (length:[60 TO 520])",
            "(reviewed:true) AND (date_created:[* TO 2024-12-31]) AND (go_exp:*) AND (organism_id:83333) AND (length:[60 TO 520])",
            "(reviewed:true) AND (date_created:[* TO 2024-12-31]) AND (go_exp:*) AND (organism_id:3702) AND (length:[60 TO 520])",
            "(reviewed:true) AND (date_created:[* TO 2024-12-31]) AND (go_exp:*) AND (taxonomy_id:2) AND (length:[60 TO 520])",
        ]
        exp_ev_codes = {"EXP", "IDA", "IPI", "IMP", "IGI", "IEP", "HTP", "HDA", "HMP", "HGI", "HEP", "TAS", "IC"}
        for tq in taxon_queries:
            rr = client.get(
                "https://rest.uniprot.org/uniprotkb/search",
                params={
                    "query": tq,
                    "fields": "accession,sequence,go",
                    "size": 400,
                },
            )
            if rr.status_code != 200:
                continue
            for ent in rr.json().get("results", []):
                acc = ent["primaryAccession"]
                seq = str((ent.get("sequence") or {}).get("value") or "").strip().upper()
                if not (60 <= len(seq) <= 520) or any(c not in CANONICAL_AA_SET for c in seq):
                    continue
                go_terms: Set[str] = set()
                for xref in ent.get("uniProtKBCrossReferences") or []:
                    if xref.get("database") == "GO":
                        gid = str(xref.get("id") or "")
                        props = {p.get("key"): str(p.get("value") or "") for p in (xref.get("properties") or [])}
                        ev = props.get("GoEvidenceType", "").split(":")[0].strip()
                        if gid.startswith("GO:") and (not ev or ev in exp_ev_codes):
                            go_terms.add(gid)
                if go_terms:
                    ref_entries[acc] = {"accession": acc, "sequence": seq, "go_terms": sorted(go_terms)}
        ref_file.write_text(json.dumps(list(ref_entries.values())), encoding="utf-8")
        print(f"  Cached {len(ref_entries)} pre-2025 Swiss-Prot experimental GO reference proteins.", flush=True)

    print("Completed all Stage 2 raw downloads!", flush=True)


if __name__ == "__main__":
    main()
