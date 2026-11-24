#!/usr/bin/env python3
"""Query UniProtKB + EBI QuickGO directly for true post-May-2026 experimental GO holdouts
(CAFA No-Knowledge criterion: ZERO experimental GO annotations prior to 2026-05-01,
i.e., quickgo_min_exp_date >= 20260501).

Saves cached raw results into `data/raw_stage2_cache/cafa_goa_post_may2026/` so
`scripts/build_stage2b_post_may2026.py` can run completely offline inside the sandbox.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import re
import time
from typing import Any, Dict, List, Optional, Set, Tuple

import httpx

REPO = Path(__file__).resolve().parents[1]
OUT_DIR = REPO / "data" / "raw_stage2_cache" / "cafa_goa_post_may2026"
CANONICAL_AA_SET = set("ACDEFGHIKLMNPQRSTVWY")
CUTOFF_ISO = "2026-05-01"
CUTOFF_QGO = "20260501"

SAFETY_BLOCKLIST = re.compile(
    r"(toxin|venom|ricin|abrin|botulinum|anthrax|ebola|marburg|variola|smallpox|lassa|"
    r"nipah|hendra|yersinia pestis|francisella tularensis|burkholderia mallei|"
    r"burkholderia pseudomallei|clostridium botulinum|bacillus anthracis|"
    r"staphylococcus enterotoxin|shiga|diphtheria|pertussis|tetanus|saxitoxin|"
    r"conotoxin|bungarotoxin|cobratoxin|sars-cov|mers-cov|influenza|hiv-1|"
    r"virulence|hemolysin|cytolysin|pore-forming)",
    re.IGNORECASE,
)


def _next_link(resp: httpx.Response) -> Optional[str]:
    link = resp.headers.get("link") or resp.headers.get("Link") or ""
    m = re.search(r'<([^>]+)>;\s*rel="next"', link)
    return m.group(1) if m else None


def fetch_uniprot_paginated(
    client: httpx.Client,
    query: str,
    max_entries: int = 1000,
    page_size: int = 500,
) -> Tuple[int, List[Dict[str, Any]]]:
    url: Optional[str] = "https://rest.uniprot.org/uniprotkb/search"
    params: Optional[Dict[str, Any]] = {
        "query": query,
        "fields": "accession,id,protein_name,organism_name,length,sequence,date_created,reviewed",
        "size": min(page_size, max_entries),
    }
    total = 0
    rows: List[Dict[str, Any]] = []
    while url and len(rows) < max_entries:
        r = client.get(url, params=params)
        if r.status_code != 200:
            print(f"  [WARN] UniProt HTTP {r.status_code} for {query!r}: {r.text[:200]}", flush=True)
            break
        if total == 0:
            try:
                total = int(r.headers.get("x-total-results") or r.headers.get("X-Total-Results") or 0)
            except ValueError:
                total = 0
        batch = r.json().get("results", [])
        if not batch:
            break
        rows.extend(batch)
        url = _next_link(r)
        params = None
    return total, rows[:max_entries]


def parse_uniprot_entry(entry: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    acc = str(entry.get("primaryAccession") or "").strip()
    seq = str((entry.get("sequence") or {}).get("value") or "").strip().upper()
    org = str((entry.get("organism") or {}).get("scientificName") or "").strip()
    pname = json.dumps(entry.get("proteinDescription") or {})
    if not acc or not (60 <= len(seq) <= 520) or any(c not in CANONICAL_AA_SET for c in seq):
        return None
    if SAFETY_BLOCKLIST.search(f"{org} {pname}"):
        return None
    audit = entry.get("entryAudit") or {}
    return {
        "accession": acc,
        "entry_id": str(entry.get("uniProtkbId") or ""),
        "organism": org,
        "reviewed": str(entry.get("entryType") or "").lower().startswith("uniprotkb reviewed"),
        "sequence": seq,
        "length": len(seq),
        "date_created": str(audit.get("firstPublicDate") or ""),
        "date_modified": str(audit.get("lastAnnotationUpdateDate") or ""),
    }


def fetch_all_quickgo_exp(
    client: httpx.Client,
    accessions: List[str],
    batch_size: int = 25,
) -> Dict[str, List[Dict[str, Any]]]:
    """Fetch ALL experimental GO annotations (ECO:0000269 descendants) for every accession,
    paginating through ALL pages so we NEVER miss an earlier pre-cutoff annotation."""
    anns_by_acc: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for i in range(0, len(accessions), batch_size):
        batch = accessions[i : i + batch_size]
        gp_param = ",".join(f"UniProtKB:{a}" for a in batch)
        page = 1
        while True:
            rq = client.get(
                "https://www.ebi.ac.uk/QuickGO/services/annotation/search",
                params={
                    "geneProductId": gp_param,
                    "evidenceCode": "ECO:0000269",
                    "limit": 200,
                    "page": page,
                },
                headers={"Accept": "application/json"},
            )
            if rq.status_code != 200:
                print(f"  [WARN] QuickGO HTTP {rq.status_code} batch {i} p{page}", flush=True)
                break
            js = rq.json()
            res = js.get("results", [])
            for r in res:
                gp = str(r.get("geneProductId") or "").replace("UniProtKB:", "")
                qual = str(r.get("qualifier") or "")
                if "NOT" in qual.upper():
                    continue
                anns_by_acc[gp].append(
                    {
                        "goId": r.get("goId"),
                        "goAspect": r.get("goAspect"),
                        "goEvidence": r.get("goEvidence"),
                        "date": str(r.get("date") or ""),
                        "reference": r.get("reference"),
                        "assignedBy": r.get("assignedBy"),
                    }
                )
            pinfo = js.get("pageInfo") or {}
            total_pages = int(pinfo.get("total") or 1)
            if page >= total_pages or page >= 25 or not res:
                break
            page += 1
        if (i // batch_size) % 10 == 0:
            print(f"  [QuickGO] Queried {min(i + batch_size, len(accessions))}/{len(accessions)} accessions...", flush=True)
    return dict(anns_by_acc)


def probe_quickgo_recent_annotations(client: httpx.Client) -> Dict[str, Any]:
    """Also check QuickGO's own total counts of recent experimental GO annotations across UniProtKB."""
    stats: Dict[str, Any] = {}
    for asp in ("molecular_function", "biological_process", "cellular_component"):
        r = client.get(
            "https://www.ebi.ac.uk/QuickGO/services/annotation/search",
            params={
                "geneProductType": "protein",
                "evidenceCode": "ECO:0000269",
                "aspect": asp,
                "limit": 1,
                "page": 1,
            },
            headers={"Accept": "application/json"},
        )
        if r.status_code == 200:
            stats[f"quickgo_total_exp_{asp}"] = r.json().get("numberOfHits")
    return stats


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-per-query", type=int, default=600)
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    with httpx.Client(timeout=120.0, follow_redirects=True) as client:
        print("=== Step 1: Querying UniProtKB for post-May-2026 & 2026 experimental-GO proteins ===", flush=True)
        queries = [
            # 1. ALL proteins created on or after 2026-05-01 with experimental GO annotations!
            ("new_since_may2026_all", "(date_created:[2026-05-01 TO *]) AND (go_exp:*) AND (length:[60 TO 520])", 2000),
            # 2. Proteins created in 2026 with experimental GO (both Swiss-Prot & TrEMBL)
            ("new_since_2026_reviewed", "(reviewed:true) AND (date_created:[2026-01-01 TO *]) AND (go_exp:*) AND (length:[60 TO 520])", 1000),
            ("new_since_2026_unreviewed", "(reviewed:false) AND (date_created:[2026-01-01 TO *]) AND (go_exp:*) AND (length:[60 TO 520])", 1000),
            # 3. Proteins with 2026 literature & experimental GO (across diverse taxa, recently updated after May 2026)
            ("lit2026_mod_postmay_bac", "(lit_pubdate:2026) AND (date_modified:[2026-05-01 TO *]) AND (go_exp:*) AND (taxonomy_id:2) AND (length:[60 TO 520])", args.max_per_query),
            ("lit2026_mod_postmay_fungi", "(lit_pubdate:2026) AND (date_modified:[2026-05-01 TO *]) AND (go_exp:*) AND (taxonomy_id:4751) AND (length:[60 TO 520])", args.max_per_query),
            ("lit2026_mod_postmay_plants", "(lit_pubdate:2026) AND (date_modified:[2026-05-01 TO *]) AND (go_exp:*) AND (taxonomy_id:33090) AND (length:[60 TO 520])", args.max_per_query),
            ("lit2026_mod_postmay_metazoa", "(lit_pubdate:2026) AND (date_modified:[2026-05-01 TO *]) AND (go_exp:*) AND (taxonomy_id:33208) AND (length:[60 TO 520])", args.max_per_query),
            # 4. Newly annotated TrEMBL/Swiss-Prot entries modified post-May-2026 (low annotation score = freshly characterized NK candidates!)
            ("mod_postmay_score123_sp", "(reviewed:true) AND (date_modified:[2026-05-01 TO *]) AND (go_exp:*) AND (length:[60 TO 520]) AND (annotation_score:1 OR annotation_score:2 OR annotation_score:3)", args.max_per_query),
            ("mod_postmay_trembl", "(reviewed:false) AND (date_modified:[2026-05-01 TO *]) AND (go_exp:*) AND (length:[60 TO 520])", args.max_per_query),
        ]

        query_totals: Dict[str, int] = {}
        cands: Dict[str, Dict[str, Any]] = {}
        for label, uq, lim in queries:
            tot, rows = fetch_uniprot_paginated(client, uq, max_entries=lim)
            query_totals[label] = tot
            added = 0
            for ent in rows:
                parsed = parse_uniprot_entry(ent)
                if parsed and parsed["accession"] not in cands:
                    cands[parsed["accession"]] = parsed
                    added += 1
            print(f"  [{label}] UniProt total hits={tot:,} | fetched={len(rows)} | new clean={added}", flush=True)

        print(f"\nTotal unique clean UniProt candidates: {len(cands)}", flush=True)
        print("  date_created >= 2026-05-01:", sum(1 for c in cands.values() if c["date_created"] >= CUTOFF_ISO), flush=True)
        print("  reviewed breakdown:", Counter(c["reviewed"] for c in cands.values()), flush=True)

        print("\n=== Step 2: Fetching ALL experimental QuickGO annotations for every candidate ===", flush=True)
        accs = sorted(cands.keys())
        anns_by_acc = fetch_all_quickgo_exp(client, accs, batch_size=25)

        strict_nk: List[str] = []
        strict_nk_and_new_uniprot: List[str] = []
        any_post_may: List[str] = []
        strict_nk_date_hist: Counter[str] = Counter()
        strict_nk_org_hist: Counter[str] = Counter()
        strict_nk_ann_total = 0

        for acc in accs:
            rows = anns_by_acc.get(acc, [])
            dates = [str(r.get("date") or "") for r in rows if r.get("date")]
            if not dates:
                continue
            mn, mx = min(dates), max(dates)
            if mx >= CUTOFF_QGO:
                any_post_may.append(acc)
            if mn >= CUTOFF_QGO:
                strict_nk.append(acc)
                strict_nk_date_hist[mn[:6]] += 1
                strict_nk_org_hist[cands[acc]["organism"]] += 1
                strict_nk_ann_total += len(rows)
                if cands[acc]["date_created"] >= CUTOFF_ISO:
                    strict_nk_and_new_uniprot.append(acc)

        qgo_totals = probe_quickgo_recent_annotations(client)

        summary = {
            "cutoff_iso": CUTOFF_ISO,
            "cutoff_qgo": CUTOFF_QGO,
            "uniprot_query_totals": query_totals,
            "quickgo_global_exp_counts": qgo_totals,
            "n_unique_candidates_queried": len(cands),
            "n_with_any_exp_go": sum(1 for a in accs if anns_by_acc.get(a)),
            "n_any_exp_go_post_may2026": len(any_post_may),
            "n_strict_nk_all_exp_go_post_may2026": len(strict_nk),
            "n_strict_nk_and_uniprot_created_post_may2026": len(strict_nk_and_new_uniprot),
            "strict_nk_total_exp_annotations": strict_nk_ann_total,
            "strict_nk_by_month": dict(sorted(strict_nk_date_hist.items())),
            "strict_nk_top_organisms": strict_nk_org_hist.most_common(15),
        }

        (OUT_DIR / "candidate_uniprot_entries.json").write_text(
            json.dumps(list(cands.values()), indent=2), encoding="utf-8"
        )
        (OUT_DIR / "quickgo_exp_annotations.json").write_text(
            json.dumps(anns_by_acc, indent=2), encoding="utf-8"
        )
        (OUT_DIR / "fetch_summary.json").write_text(
            json.dumps(summary, indent=2), encoding="utf-8"
        )

        print("\n=== POST-MAY-2026 EXPERIMENTAL GO SUMMARY ===", flush=True)
        print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
