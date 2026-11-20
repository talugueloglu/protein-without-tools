#!/usr/bin/env python3
"""Download raw public wet-lab datasets for Test 6: Interface Binding Alanine Scan (`X -> Ala`).

Sources cached into `data/raw_binding_alanine_cache/`:
  1. `SKEMPI v2.0` (`https://life.bsc.es/pid/skempi2/database/download/skempi_v2.csv`)
     + RCSB `.cif` structures for complexes with `>= 6` single-point `X -> Ala` SPR/ITC `ddG_bind` mutants
     (plus any `PD-1` / `PD-L1` / oncology complexes in SKEMPI!).
  2. `aalphabio/open-alphaseq` (`HuggingFace` Parquet API):
     - `YM_0852` (`Pembrolizumab-scFv x Human PD-1` quantitative yeast-mating `log10 Kd`)
     - `YM_0693` (`PP489 x Human TIGIT` quantitative yeast-mating `log10 Kd`)
  3. `BindingGYM` (`luwei0917/BindingGYM`) & `ProteinGym` (`OATML-Markslab/ProteinGym` `Binding` DMS):
     - Reference metadata CSVs + individual non-viral PPI binding DMS CSVs (including de novo minibinder SSMs,
       oncology interfaces `RAF1-KRAS`, `BCL2/MCL1`, `MYC-MAX`, `GRB2-GAB2`, `PSD95-CRIPT`, `PD-1/PD-L1`).
  4. Post-Cutoff `MaveDB` Binding DMS (`api.mavedb.org`):
     - `urn:mavedb:00001302-g-1` (`2026-10-16` `KRAS - LZTR1` direct binding DMS)
     - `urn:mavedb:00001286-a-1` (`2026-08-28` `TINF2` shelterin complex interaction DMS)
     - `urn:mavedb:00001239-a-1` (`2025-05-29` `Human Insulin Receptor` ectodomain binding DMS)
     - Lehner `Double-deepPCA` binding (`bPCA`) score-sets in MaveDB (`PSD95-PDZ3`, `GRB2-SH3`, `GB1-IgG`)
     + UniProt canonical partner sequences (`PD-1`, `PD-L1`, `TIGIT`, `LZTR1`, `TERF1`, `INS`, `RAF1`, `FCGR` / `IGHG1`).
"""

from __future__ import annotations

from collections import Counter, defaultdict
import csv
import io
import json
from pathlib import Path
import re
from typing import Any, Dict, List

import httpx
import pyarrow.parquet as pq

SAFETY_BLOCKLIST = re.compile(
    r"(virus|viral|virion|capsid|phage|toxin|venom|ricin|abrin|"
    r"botulinum|anthrax|anthracis|ebola|marburg|variola|influenza|corona|sars|\bhiv\b|gp120|cov2)",
    re.IGNORECASE,
)


def main() -> None:
    repo = Path("/Users/taluguloglu/.gemini/antigravity/scratch/fable-protein-bench")
    raw_dir = repo / "data" / "raw_binding_alanine_cache"
    skempi_cif_dir = raw_dir / "skempi_cifs"
    alphaseq_dir = raw_dir / "alphaseq"
    dms_dir = raw_dir / "binding_dms"
    mave_dir = raw_dir / "mavedb_binding"
    for d in (raw_dir, skempi_cif_dir, alphaseq_dir, dms_dir, mave_dir):
        d.mkdir(parents=True, exist_ok=True)

    with httpx.Client(timeout=90.0, follow_redirects=True, verify=False) as client:
        # ======================================================================
        # 1. SKEMPI v2.0 Biophysical SPR/ITC ΔΔG_bind + Complex CIFs
        # ======================================================================
        print("=== 1. Downloading SKEMPI v2.0 Biophysical Binding Database ===", flush=True)
        sk_csv = raw_dir / "skempi_v2.csv"
        if not sk_csv.exists():
            r_sk = client.get("https://life.bsc.es/pid/skempi2/database/download/skempi_v2.csv")
            r_sk.raise_for_status()
            sk_csv.write_bytes(r_sk.content)
        print(f"  Cached skempi_v2.csv ({sk_csv.stat().st_size / 1024:.1f} KB)", flush=True)

        # Count single-point X -> Ala mutations per (pdb4, mutated_chain, partner_chains)
        ala_counts: Counter[tuple[str, str, str]] = Counter()
        pdl1_pdbs: set[str] = set()
        reader = csv.DictReader(io.StringIO(sk_csv.read_text(encoding="utf-8", errors="replace")), delimiter=";")
        for row in reader:
            pdb_field = str(row.get("#Pdb") or "")
            mut_clean = str(row.get("Mutation(s)_cleaned") or "").strip()
            p1_name = str(row.get("Protein 1") or "")
            p2_name = str(row.get("Protein 2") or "")
            if SAFETY_BLOCKLIST.search(f"{p1_name} {p2_name}"):
                continue
            parts = pdb_field.split("_")
            if len(parts) < 3 or "," in mut_clean or not mut_clean.endswith("A"):
                continue
            pdb4, g1, g2 = parts[0].upper(), parts[1], parts[2]
            if len(mut_clean) < 4:
                continue
            mut_chain = mut_clean[1]
            partner_grp = g2 if mut_chain in g1 else g1
            if len(partner_grp) == 1:
                ala_counts[(pdb4, mut_chain, partner_grp)] += 1
                if re.search(r"(pd-1|pd-l1|pdl1|pd1|ctla|her2|egfr|il-2|hgh)", f"{p1_name} {p2_name}", re.I):
                    pdl1_pdbs.add(pdb4)

        top_skempi = [k for k, cnt in ala_counts.most_common(45) if cnt >= 6 or k[0] in pdl1_pdbs][:28]
        print(f"  Found {len(top_skempi)} high-density SKEMPI X->Ala binary complexes; caching RCSB CIFs...", flush=True)
        for pdb4, _mc, _pc in top_skempi:
            cp = skempi_cif_dir / f"{pdb4}.cif"
            if not cp.exists():
                rc = client.get(f"https://files.rcsb.org/download/{pdb4}.cif")
                if rc.status_code == 200:
                    cp.write_text(rc.text, encoding="utf-8")

        # Cache PD-L1 x Atezolizumab (`5XXY`) & PD-L1 x Durvalumab (`5X8M`) CIFs + Open-Access SPR XML (Lee et al. 2017)
        for pdl1_pdb in ("5XXY", "5X8M", "4ZQK"):
            cp = skempi_cif_dir / f"{pdl1_pdb}.cif"
            if not cp.exists():
                rc = client.get(f"https://files.rcsb.org/download/{pdl1_pdb}.cif")
                if rc.status_code == 200:
                    cp.write_text(rc.text, encoding="utf-8")
        pdl1_xml = raw_dir / "lee2017_pdl1_spr_fulltext.xml"
        if not pdl1_xml.exists():
            r_epmc = client.get(
                "https://www.ebi.ac.uk/europepmc/webservices/rest/search",
                params={"query": "DOI:10.1038/s41598-017-06002-8", "format": "json"},
            )
            if r_epmc.status_code == 200:
                res_list = ((r_epmc.json().get("resultList") or {}).get("result") or [])
                pmcid = res_list[0].get("pmcid") if res_list else None
                if pmcid:
                    r_ft = client.get(f"https://www.ebi.ac.uk/europepmc/webservices/rest/{pmcid}/fullTextXML")
                    if r_ft.status_code == 200:
                        pdl1_xml.write_text(r_ft.text, encoding="utf-8")
                        print(f"  Cached Lee et al. 2017 PD-L1 SPR XML ({pmcid})", flush=True)

        # ======================================================================
        # 2. A-Alpha Bio `open-alphaseq` Oncology Checkpoint Binders (`PD-1`, `TIGIT`, `HER2`)
        # ======================================================================
        print("\n=== 2. Downloading Open-AlphaSeq Quantitative Oncology Kd Datasets (PD-1, TIGIT, HER2) ===", flush=True)
        for cfg in ("YM_0852", "YM_0693", "YM_0989", "YM_0985", "YM_0988"):
            out_pq = alphaseq_dir / f"{cfg}.parquet"
            if not out_pq.exists():
                api_url = f"https://huggingface.co/api/datasets/aalphabio/open-alphaseq/parquet/{cfg}"
                ra = client.get(api_url)
                if ra.status_code == 200:
                    j_urls = ra.json()
                    flat_urls: List[str] = []
                    if isinstance(j_urls, dict):
                        for v in j_urls.values():
                            if isinstance(v, list):
                                flat_urls.extend(v)
                    if flat_urls:
                        rpq = client.get(flat_urls[0])
                        if rpq.status_code == 200:
                            out_pq.write_bytes(rpq.content)
            if out_pq.exists():
                tbl = pq.read_table(out_pq)
                print(f"  Cached AlphaSeq {cfg}: {tbl.num_rows} rows, cols={tbl.column_names[:5]}", flush=True)

        # Also cache `proteinea/ppb_affinity` parquet if available
        ppb_pq = raw_dir / "ppb_affinity.parquet"
        if not ppb_pq.exists():
            r_ppb = client.get("https://huggingface.co/api/datasets/proteinea/ppb_affinity/parquet")
            if r_ppb.status_code == 200:
                j_ppb = r_ppb.json()
                ppb_urls: List[str] = []
                if isinstance(j_ppb, dict):
                    for cfg_v in j_ppb.values():
                        if isinstance(cfg_v, dict):
                            for sp_v in cfg_v.values():
                                if isinstance(sp_v, list):
                                    ppb_urls.extend(sp_v)
                        elif isinstance(cfg_v, list):
                            ppb_urls.extend(cfg_v)
                if ppb_urls:
                    r_b = client.get(ppb_urls[0])
                    if r_b.status_code == 200:
                        ppb_pq.write_bytes(r_b.content)
                        print(f"  Cached proteinea/ppb_affinity parquet ({ppb_pq.stat().st_size / 1024:.1f} KB)", flush=True)

        # ======================================================================
        # 3. ProteinGym Non-Viral Binding DMS Assays
        # ======================================================================
        print("\n=== 3. Downloading Non-Viral ProteinGym Binding DMS Assays ===", flush=True)
        pg_ref = dms_dir / "ProteinGym_DMS_substitutions.csv"
        if not pg_ref.exists():
            rpg = client.get("https://raw.githubusercontent.com/OATML-Markslab/ProteinGym/HEAD/reference_files/DMS_substitutions.csv")
            if rpg.status_code == 200:
                pg_ref.write_text(rpg.text, encoding="utf-8")
        bind_pg: List[Dict[str, Any]] = []
        if pg_ref.exists():
            pg_rows = list(csv.DictReader(io.StringIO(pg_ref.read_text(encoding="utf-8"))))
            bind_pg = [
                r for r in pg_rows
                if str(r.get("coarse_selection_type") or "").strip().lower() == "binding"
                and not SAFETY_BLOCKLIST.search(
                    f"{r.get('DMS_id', '')} {r.get('taxon', '')} {r.get('molecule_name', '')} {r.get('title', '')} {r.get('selection_assay', '')}"
                )
            ]
            print(f"  Found {len(bind_pg)} clean non-viral ProteinGym 'Binding' DMS assays!", flush=True)
            (dms_dir / "proteingym_nonviral_binding_index.json").write_text(json.dumps(bind_pg, indent=2), encoding="utf-8")

        # Pull individual single-mutant rows from HuggingFace Datasets Server `/filter` or MarksLab zip Range
        for row_pg in bind_pg:
            dms_id = str(row_pg["DMS_id"])
            out_csv = dms_dir / f"{dms_id}.json"
            if out_csv.exists():
                continue
            # Query HuggingFace datasets-server filter for single-point X->A mutants (or all mutants if small)
            all_muts: List[Dict[str, Any]] = []
            for offset in range(0, 4000, 100):
                rf = client.get(
                    "https://datasets-server.huggingface.co/filter",
                    params={
                        "dataset": "OATML-Markslab/ProteinGym_v1",
                        "config": "DMS_substitutions",
                        "split": "test",
                        "where": f"\"DMS_id\"='{dms_id}'",
                        "offset": offset,
                        "length": 100,
                    },
                )
                if rf.status_code != 200:
                    break
                chunk = rf.json().get("rows") or []
                if not chunk:
                    break
                for item in chunk:
                    r_obj = item.get("row") or {}
                    mut_s = str(r_obj.get("mutant") or "")
                    if ":" not in mut_s and mut_s.endswith("A"):
                        all_muts.append({"mutant": mut_s, "DMS_score": r_obj.get("DMS_score"), "DMS_score_bin": r_obj.get("DMS_score_bin")})
                if len(chunk) < 100:
                    break
            if all_muts:
                out_csv.write_text(json.dumps({"DMS_id": dms_id, "ala_mutants": all_muts}, indent=2), encoding="utf-8")
                print(f"  [ProteinGym Binding] {dms_id}: {len(all_muts)} single-point X->Ala mutants", flush=True)

        # ======================================================================
        # 4. MaveDB Post-May-2026 Binding Scans + Curated PPI Binding Scans
        # ======================================================================
        print("\n=== 4. Downloading MaveDB 2026 Binding Scans & Curated PPI Binding Scans ===", flush=True)
        mave_exps_path = repo / "data/raw_alanine_cache/mavedb/experiments.json"
        target_urns = [
            "urn:mavedb:00001302-g-1",  # 2026-10-16 KRAS - LZTR1 direct binding!
            "urn:mavedb:00001302-d-1",  # 2026-10-16 KRAS baseline abundance companion
            "urn:mavedb:00001286-a-1",  # 2026-08-28 TINF2 shelterin interaction
            "urn:mavedb:00001326-e-1",  # 2026-11-09 OPRM1 DAMGO peptide agonist EC50/Emax
            "urn:mavedb:00001326-b-1",  # 2026-11-09 OPRM1 15 opioid ligands (probe peptide agonists)
            "urn:mavedb:00001326-b-2",
            "urn:mavedb:00001326-b-3",
            "urn:mavedb:00001326-b-4",
            "urn:mavedb:00001326-b-5",
            "urn:mavedb:00001326-b-6",
            "urn:mavedb:00001247-a-1",  # 2025-08-07 GCK Y2H interaction
            "urn:mavedb:00001239-a-1",  # 2025-05-29 Human Insulin Receptor ectodomain x Insulin
            "urn:mavedb:00001239-a-2",
            "urn:mavedb:00001218-a-1",  # 2025-03-31 Human MLH1 x PMS2 interaction vs stability
            "urn:mavedb:00001218-a-2",
            "urn:mavedb:00000048-b-1",  # CXCR4 x CXCL12 chemokine binding
            "urn:mavedb:00000047-b-1",  # CCR5 x Ab 2D7 binding
            "urn:mavedb:00000053-a-1",  # PSD95-PDZ3 x CRIPT binding
            "urn:mavedb:00000038-a-1",  # Ubiquitin x UBA1 E1 binding
            "urn:mavedb:00000002-a-1",  # YAP1 WW domain x peptide binding
        ]
        if mave_exps_path.exists():
            exps = json.loads(mave_exps_path.read_text(encoding="utf-8"))
            for e in exps:
                blob = f"{e.get('title', '')} {e.get('shortDescription', '')} {e.get('abstractText', '')}"
                if SAFETY_BLOCKLIST.search(blob):
                    continue
                if any(k in blob for k in ("Double-deepPCA", "BindingPCA", "bPCA", "PD-L1", "PD-1")):
                    for su in (e.get("scoreSetUrns") or [])[:4]:
                        if su not in target_urns:
                            target_urns.append(su)

        saved_mave = 0
        for surn in target_urns[:55]:
            if saved_mave >= 40:
                break
            safe = surn.replace(":", "_")
            mp = mave_dir / f"{safe}.json"
            cp = mave_dir / f"{safe}.csv"
            if mp.exists() and cp.exists():
                saved_mave += 1
                continue
            rm = client.get(f"https://api.mavedb.org/api/v1/score-sets/{surn}")
            if rm.status_code != 200:
                continue
            jm = rm.json()
            title_m = f"{jm.get('title', '')} {jm.get('shortDescription', '')}"
            if SAFETY_BLOCKLIST.search(title_m):
                continue
            tgenes = jm.get("targetGenes") or []
            if not tgenes:
                continue
            tseq = (tgenes[0].get("targetSequence") or {})
            if tseq.get("sequenceType") != "protein":
                continue
            seq_s = str(tseq.get("sequence") or "").strip()
            if not (25 <= len(seq_s) <= 1500):
                continue
            rc = client.get(f"https://api.mavedb.org/api/v1/score-sets/{surn}/scores")
            if rc.status_code != 200 or ("Ala" not in rc.text and "A," not in rc.text):
                continue
            mp.write_text(json.dumps(jm, indent=2), encoding="utf-8")
            cp.write_text(rc.text, encoding="utf-8")
            saved_mave += 1
            print(f"  [MaveDB Binding {saved_mave:2d}] {surn} ({str(jm.get('publishedDate', ''))[:10]}) | {jm.get('title', '')[:65]}", flush=True)

        # Fetch canonical human UniProt sequences for binding partners
        partner_genes = [
            "PDCD1", "CD274", "TIGIT", "ERBB2", "LZTR1", "SOS1", "TERF1", "TINF2", "ACD",
            "INS", "INSR", "PMS2", "MLH1", "CXCL12", "UBA1", "TAPBPR", "POMC", "PDYN", "PENK",
            "RAF1", "GAB2", "DLG4", "CRIPT", "MCL1", "IGHG1",
        ]
        partner_seqs: Dict[str, str] = {}
        for g in partner_genes:
            ru = client.get(
                "https://rest.uniprot.org/uniprotkb/search",
                params={"query": f"(reviewed:true) AND (organism_id:9606) AND (gene_exact:{g})", "fields": "accession,gene_names,sequence", "size": 1},
            )
            if ru.status_code == 200:
                res = ru.json().get("results") or []
                if res:
                    partner_seqs[g] = str((res[0].get("sequence") or {}).get("value") or "")
        (raw_dir / "uniprot_partner_sequences.json").write_text(json.dumps(partner_seqs, indent=2), encoding="utf-8")
        print(f"  Cached {len(partner_seqs)} canonical human UniProt binding-partner sequences.", flush=True)

    print("Completed Binding Alanine Scan raw downloads!", flush=True)


if __name__ == "__main__":
    main()
