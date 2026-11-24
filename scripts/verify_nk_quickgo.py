#!/usr/bin/env python3
"""Double-verify all 254 clean strict-NK candidates (50..800 AA) against live EBI QuickGO
(ECO:0000269) to guarantee ZERO pre-2026-05-01 experimental GO annotations anywhere in QuickGO."""

from __future__ import annotations

from collections import defaultdict
import json
from pathlib import Path
from typing import Any, Dict, List

import httpx

REPO = Path(__file__).resolve().parents[1]
OUT_DIR = REPO / "data" / "raw_stage2_cache" / "cafa_goa_post_may2026"
CUTOFF = "20260501"


def main() -> None:
    enr = json.loads((OUT_DIR / "enriched_strict_nk.json").read_text(encoding="utf-8"))
    cands = [x for x in enr if x["safe"] and x["canonical_aa"] and 50 <= x["length"] <= 800]
    accs = sorted({x["accession"] for x in cands})
    print(f"Double-checking {len(accs)} clean candidates (50..800 AA) against EBI QuickGO...", flush=True)

    pre_cutoff_found: Dict[str, str] = {}
    with httpx.Client(timeout=120.0, follow_redirects=True) as client:
        for i in range(0, len(accs), 25):
            batch = accs[i : i + 25]
            gp = ",".join(f"UniProtKB:{a}" for a in batch)
            page = 1
            while True:
                r = client.get(
                    "https://www.ebi.ac.uk/QuickGO/services/annotation/search",
                    params={
                        "geneProductId": gp,
                        "evidenceCode": "ECO:0000269",
                        "limit": 200,
                        "page": page,
                    },
                    headers={"Accept": "application/json"},
                )
                if r.status_code != 200:
                    print(f"  [WARN] HTTP {r.status_code} batch {i} p{page}", flush=True)
                    break
                js = r.json()
                res = js.get("results", [])
                for row in res:
                    if "NOT" in str(row.get("qualifier") or "").upper():
                        continue
                    if row.get("assignedBy") == "IntAct" or row.get("goId") == "GO:0005515":
                        continue
                    acc = str(row.get("geneProductId") or "").replace("UniProtKB:", "")
                    dt = str(row.get("date") or "")
                    if dt and dt < CUTOFF:
                        if acc not in pre_cutoff_found or dt < pre_cutoff_found[acc]:
                            pre_cutoff_found[acc] = dt
                pinfo = js.get("pageInfo") or {}
                if page >= int(pinfo.get("total") or 1) or page >= 20 or not res:
                    break
                page += 1

    verified = [x for x in cands if x["accession"] not in pre_cutoff_found]
    print(f"Pre-May-2026 non-IntAct annotations found in QuickGO: {len(pre_cutoff_found)}", flush=True)
    print(f"Double-verified strict-NK candidates (ZERO pre-May-2026 exp GO in GAF + QuickGO): {len(verified)}", flush=True)
    post_paper = [x for x in verified if x["min_pmid_date"] >= "2026-05-01"]
    print(f"  Of which PMID pubdate >= 2026-05-01: {len(post_paper)}", flush=True)
    (OUT_DIR / "verified_strict_nk.json").write_text(json.dumps(verified, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
