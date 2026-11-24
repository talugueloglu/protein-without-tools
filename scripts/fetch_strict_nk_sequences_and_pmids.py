#!/usr/bin/env python3
"""Fetch UniProt sequences/metadata and NCBI PubMed publication dates for all non-IntAct
strict-NK (>= 2026-05-01) proteins from the 13 EBI GOA species GAFs."""

from __future__ import annotations

from collections import Counter
import json
from pathlib import Path
import re
from typing import Any, Dict, List, Set

import httpx

REPO = Path(__file__).resolve().parents[1]
OUT_DIR = REPO / "data" / "raw_stage2_cache" / "cafa_goa_post_may2026"
CANONICAL_AA_SET = set("ACDEFGHIKLMNPQRSTVWY")

SAFETY_BLOCKLIST = re.compile(
    r"(toxin|venom|ricin|abrin|botulinum|anthrax|ebola|marburg|variola|smallpox|lassa|"
    r"nipah|hendra|yersinia pestis|francisella tularensis|burkholderia mallei|"
    r"burkholderia pseudomallei|clostridium botulinum|bacillus anthracis|"
    r"staphylococcus enterotoxin|shiga|diphtheria|pertussis|tetanus|saxitoxin|"
    r"conotoxin|bungarotoxin|cobratoxin|sars-cov|mers-cov|influenza|hiv-1|"
    r"virulence|hemolysin|cytolysin|pore-forming)",
    re.IGNORECASE,
)

MONTH_MAP = {
    "jan": "01", "feb": "02", "mar": "03", "apr": "04", "may": "05", "jun": "06",
    "jul": "07", "aug": "08", "sep": "09", "oct": "10", "nov": "11", "dec": "12",
}


def parse_pubmed_date(pubdate: str) -> str:
    """Convert NCBI esummary pubdate like '2026 Jul 14' or '2026 Jun' -> 'YYYY-MM-DD'."""
    s = (pubdate or "").strip()
    m = re.match(r"^(\d{4})(?:\s+([A-Za-z]{3})(?:\s+(\d{1,2}))?)?", s)
    if not m:
        return ""
    yr = m.group(1)
    mo = MONTH_MAP.get((m.group(2) or "").lower(), "01")
    dy = f"{int(m.group(3)):02d}" if m.group(3) else "01"
    return f"{yr}-{mo}-{dy}"


def main() -> None:
    raw = json.loads((OUT_DIR / "ebi_goa_strict_nk.json").read_text(encoding="utf-8"))

    # Filter out IntAct bulk re-stamps and GO:0005515-only proteins
    curated: Dict[str, Dict[str, Any]] = {}
    all_pmids: Set[str] = set()
    for acc, rec in raw.items():
        clean_anns = [
            a for a in rec["annotations"]
            if a["assignedBy"] != "IntAct" and a["goId"] != "GO:0005515"
        ]
        if not clean_anns:
            continue
        dates = [a["date"] for a in clean_anns if a["date"]]
        if not dates or min(dates) < "20260501":
            continue
        pmids = sorted({
            a["ref"].split(":", 1)[1]
            for a in clean_anns
            if a.get("ref", "").startswith("PMID:") and a["ref"].split(":", 1)[1].isdigit()
        })
        all_pmids.update(pmids)
        curated[acc] = {
            "accession": acc,
            "species_gaf": rec["species_gaf"],
            "min_go_date": min(dates),
            "max_go_date": max(dates),
            "annotations": clean_anns,
            "pmids": pmids,
        }

    print(f"Curated non-IntAct strict-NK proteins (>=2026-05-01): {len(curated)}", flush=True)
    print(f"Unique PMIDs to check publication dates: {len(all_pmids)}", flush=True)

    with httpx.Client(timeout=120.0, follow_redirects=True) as client:
        # 1. Fetch PubMed pubdates via NCBI esummary
        pmid_dates: Dict[str, str] = {}
        pmid_list = sorted(all_pmids)
        for i in range(0, len(pmid_list), 150):
            batch = pmid_list[i : i + 150]
            r = client.get(
                "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi",
                params={"db": "pubmed", "id": ",".join(batch), "retmode": "json"},
            )
            if r.status_code == 200:
                res = r.json().get("result") or {}
                for pid in batch:
                    info = res.get(pid) or {}
                    pdt = parse_pubmed_date(str(info.get("pubdate") or ""))
                    epdt = parse_pubmed_date(str(info.get("epubdate") or ""))
                    # Use earlier of pubdate/epubdate if both exist (conservative anti-leakage!)
                    chosen = min(x for x in (pdt, epdt) if x) if (pdt or epdt) else ""
                    pmid_dates[pid] = chosen

        # 2. Fetch UniProt sequences & metadata in batches of 50 accessions
        accs = sorted(curated.keys())
        uniprot_info: Dict[str, Dict[str, Any]] = {}
        for i in range(0, len(accs), 50):
            batch = accs[i : i + 50]
            q = " OR ".join(f"accession:{a}" for a in batch)
            r = client.get(
                "https://rest.uniprot.org/uniprotkb/search",
                params={
                    "query": q,
                    "fields": "accession,id,protein_name,organism_name,length,sequence,date_created,reviewed",
                    "size": 100,
                },
            )
            if r.status_code == 200:
                for ent in r.json().get("results", []):
                    acc = str(ent.get("primaryAccession") or "")
                    seq = str((ent.get("sequence") or {}).get("value") or "").strip().upper()
                    org = str((ent.get("organism") or {}).get("scientificName") or "")
                    pname = json.dumps(ent.get("proteinDescription") or {})
                    audit = ent.get("entryAudit") or {}
                    uniprot_info[acc] = {
                        "accession": acc,
                        "entry_id": str(ent.get("uniProtkbId") or ""),
                        "organism": org,
                        "protein_name": pname[:240],
                        "reviewed": str(ent.get("entryType") or "").lower().startswith("uniprotkb reviewed"),
                        "sequence": seq,
                        "length": len(seq),
                        "date_created": str(audit.get("firstPublicDate") or ""),
                        "safe": not bool(SAFETY_BLOCKLIST.search(f"{org} {pname}")),
                        "canonical_aa": all(c in CANONICAL_AA_SET for c in seq) if seq else False,
                    }

        # Combine and summarize!
        enriched: List[Dict[str, Any]] = []
        for acc, rec in curated.items():
            u = uniprot_info.get(acc)
            if not u:
                continue
            pdts = [pmid_dates[p] for p in rec["pmids"] if pmid_dates.get(p)]
            rec_full = dict(rec)
            rec_full.update(u)
            rec_full["min_pmid_date"] = min(pdts) if pdts else ""
            rec_full["max_pmid_date"] = max(pdts) if pdts else ""
            enriched.append(rec_full)

        clean_len = [
            x for x in enriched
            if x["safe"] and x["canonical_aa"] and 60 <= x["length"] <= 520
        ]
        clean_len_1000 = [
            x for x in enriched
            if x["safe"] and x["canonical_aa"] and 50 <= x["length"] <= 800
        ]
        post_may_paper = [x for x in clean_len if x["min_pmid_date"] >= "2026-05-01"]
        post_2026_paper = [x for x in clean_len if x["min_pmid_date"] >= "2026-01-01"]
        post_may_paper_800 = [x for x in clean_len_1000 if x["min_pmid_date"] >= "2026-05-01"]
        post_2026_paper_800 = [x for x in clean_len_1000 if x["min_pmid_date"] >= "2026-01-01"]

        print("\n=== ENRICHED STRICT-NK SUMMARY ===", flush=True)
        print(f"  Matched in UniProt                      : {len(enriched)}", flush=True)
        print(f"  Clean & length 60..520 AA               : {len(clean_len)}", flush=True)
        print(f"  Clean & length 50..800 AA               : {len(clean_len_1000)}", flush=True)
        print(f"  Length 60..520 + PMID pubdate >=2026-05 : {len(post_may_paper)}", flush=True)
        print(f"  Length 60..520 + PMID pubdate >=2026-01 : {len(post_2026_paper)}", flush=True)
        print(f"  Length 50..800 + PMID pubdate >=2026-05 : {len(post_may_paper_800)}", flush=True)
        print(f"  Length 50..800 + PMID pubdate >=2026-01 : {len(post_2026_paper_800)}", flush=True)
        print("  Clean 60..520 by species:", Counter(x["species_gaf"] for x in clean_len), flush=True)
        print("  Clean 60..520 by PMID year:", Counter((x["min_pmid_date"][:4] or "none") for x in clean_len).most_common(12), flush=True)
        print("  Post-May-2026 paper (60..520) by species:", Counter(x["species_gaf"] for x in post_may_paper), flush=True)
        print("  Post-May-2026 paper (50..800) by species:", Counter(x["species_gaf"] for x in post_may_paper_800), flush=True)

        (OUT_DIR / "enriched_strict_nk.json").write_text(json.dumps(enriched, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
