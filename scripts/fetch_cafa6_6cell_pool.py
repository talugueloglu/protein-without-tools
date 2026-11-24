#!/usr/bin/env python3
"""Scan live EBI GOA species GAFs (2026-11-03) to construct the exact CAFA-6 2x3 pool:
  - Pre-cutoff window  (t0' = 2025-05-01 -> t1' = 2025-11-01): pre_nk, pre_lk, pre_pk
  - Post-cutoff window (t0  = 2026-05-01 -> t1  = 2026-11-03): post_nk, post_lk, post_pk

Strictly follows CAFA-6 rules:
  - Curated experimental evidence codes only (EXP, IDA, IPI, IMP, IGI, IEP, HTP, HDA, HMP, HGI, HEP)
  - Excludes NOT qualifiers, IntAct bulk IPI re-stamps, and GO:0005515 (protein binding) only annotations
  - Computes propagated T0 and T1 GO term sets per aspect using go-basic.obo so PK strictly requires
    genuinely NEW non-ancestor terms (T1 \\ T0 != empty) and LK strictly requires a NEW aspect (T0[a] == empty, T1[a] != empty)
  - Fetches UniProtKB sequences & metadata for top candidates and verifies clean 50..800 AA canonical proteins.
"""

from __future__ import annotations

from collections import Counter, defaultdict
import gzip
import io
import json
from pathlib import Path
import re
import sys
from typing import Any, Dict, List, Set, Tuple

import httpx

REPO = Path(__file__).resolve().parents[1]
if str(REPO / "src") not in sys.path:
    sys.path.insert(0, str(REPO / "src"))

from fable_bench.stage2b_cafa import parse_go_obo

OUT_DIR = REPO / "data" / "raw_stage2_cache" / "cafa6_6cell"
EXP_CODES = {"EXP", "IDA", "IPI", "IMP", "IGI", "IEP", "HTP", "HDA", "HMP", "HGI", "HEP"}
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

SPECIES_GAFS = [
    ("human", "https://ftp.ebi.ac.uk/pub/databases/GO/goa/HUMAN/goa_human.gaf.gz"),
    ("mouse", "https://ftp.ebi.ac.uk/pub/databases/GO/goa/MOUSE/goa_mouse.gaf.gz"),
    ("arabidopsis", "https://ftp.ebi.ac.uk/pub/databases/GO/goa/ARABIDOPSIS/goa_arabidopsis.gaf.gz"),
    ("yeast", "https://ftp.ebi.ac.uk/pub/databases/GO/goa/YEAST/goa_yeast.gaf.gz"),
    ("fly", "https://ftp.ebi.ac.uk/pub/databases/GO/goa/FLY/goa_fly.gaf.gz"),
    ("worm", "https://ftp.ebi.ac.uk/pub/databases/GO/goa/WORM/goa_worm.gaf.gz"),
    ("zebrafish", "https://ftp.ebi.ac.uk/pub/databases/GO/goa/ZEBRAFISH/goa_zebrafish.gaf.gz"),
    ("rat", "https://ftp.ebi.ac.uk/pub/databases/GO/goa/RAT/goa_rat.gaf.gz"),
    ("chicken", "https://ftp.ebi.ac.uk/pub/databases/GO/goa/CHICKEN/goa_chicken.gaf.gz"),
    ("cow", "https://ftp.ebi.ac.uk/pub/databases/GO/goa/COW/goa_cow.gaf.gz"),
    ("dog", "https://ftp.ebi.ac.uk/pub/databases/GO/goa/DOG/goa_dog.gaf.gz"),
    ("pig", "https://ftp.ebi.ac.uk/pub/databases/GO/goa/PIG/goa_pig.gaf.gz"),
    ("dicty", "https://ftp.ebi.ac.uk/pub/databases/GO/goa/DICTY/goa_dicty.gaf.gz"),
]

WINDOWS = [
    ("pre", "20250501", "20251101"),
    ("post", "20260501", "20261231"),
]


def classify_window(
    dag: Any,
    anns: List[Tuple[str, str, str, str, str]],
    t0: str,
    t1: str,
) -> Dict[str, Any] | None:
    """Classify a single protein for window [t0, t1] into NK, LK, or PK per CAFA-6 rules."""
    # Partition raw annotations into <= t0 (known at t0) and (t0, t1] (acquired during window)
    raw_t0: Set[str] = set()
    raw_win: Set[str] = set()
    win_dates: List[str] = []
    win_refs: Set[str] = set()
    win_assigned: Set[str] = set()
    win_evs: Set[str] = set()

    for go_id, _asp, ev, dt, ref_assigned in anns:
        cid = dag.canonical(go_id)
        if cid is None:
            continue
        if dt < t0:
            raw_t0.add(cid)
        elif t0 <= dt <= t1:
            raw_win.add(cid)
            win_dates.append(dt)
            ref, assigned = ref_assigned.split("|", 1)
            if ref.startswith("PMID:"):
                win_refs.add(ref.split(":", 1)[1])
            win_assigned.add(assigned)
            win_evs.add(ev)

    if not raw_win:
        return None

    # Propagate T0 and T1 = T0 U raw_win
    prop_t0 = dag.propagate_set(raw_t0)
    prop_t1 = dag.propagate_set(raw_t0 | raw_win)

    t0_by_asp = {a: sorted(g for g in prop_t0 if dag.term_aspect.get(g) == a) for a in ("MF", "BP", "CC")}
    t1_by_asp = {a: sorted(g for g in prop_t1 if dag.term_aspect.get(g) == a) for a in ("MF", "BP", "CC")}
    delta_by_asp = {a: sorted(set(t1_by_asp[a]) - set(t0_by_asp[a])) for a in ("MF", "BP", "CC")}

    # Must have gained at least one strictly new propagated term!
    if not any(delta_by_asp.values()):
        return None

    has_any_t0 = any(bool(t0_by_asp[a]) for a in ("MF", "BP", "CC"))

    if not has_any_t0:
        # NK: ZERO experimental GO annotations in any aspect before t0
        eval_aspects = [a for a in ("MF", "BP", "CC") if t1_by_asp[a]]
        ktype = "NK"
    else:
        lk_aspects = [a for a in ("MF", "BP", "CC") if (not t0_by_asp[a]) and bool(t1_by_asp[a])]
        pk_aspects = [a for a in ("MF", "BP", "CC") if bool(t0_by_asp[a]) and bool(delta_by_asp[a])]
        if lk_aspects:
            # LK: already had >=1 aspect before t0, gained first experimental GO in new aspect(s)
            eval_aspects = lk_aspects
            ktype = "LK"
        elif pk_aspects:
            # PK: already had experimental GO in aspect(s) before t0, gained strictly new terms in same aspect(s)
            eval_aspects = pk_aspects
            ktype = "PK"
        else:
            return None

    return {
        "knowledge_type": ktype,
        "eval_aspects": eval_aspects,
        "known_t0": t0_by_asp,
        "true_t1": t1_by_asp,
        "delta_t1_minus_t0": delta_by_asp,
        "min_win_date": min(win_dates),
        "max_win_date": max(win_dates),
        "win_pmids": sorted(win_refs),
        "win_assigned_by": sorted(win_assigned),
        "win_evidence_codes": sorted(win_evs),
        "n_delta_terms": sum(len(delta_by_asp[a]) for a in eval_aspects),
    }


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    dag = parse_go_obo(str(REPO / "data" / "raw_stage2_cache" / "cafa_goa" / "go-basic.obo"))

    # Download/cache all 13 EBI GOA species GAFs locally so reruns are instant
    gaf_cache = OUT_DIR / "gafs"
    gaf_cache.mkdir(parents=True, exist_ok=True)

    prot_anns: Dict[str, List[Tuple[str, str, str, str, str]]] = defaultdict(list)
    prot_species: Dict[str, str] = {}

    with httpx.Client(timeout=180.0, follow_redirects=True) as client:
        for sp, url in SPECIES_GAFS:
            local_gz = gaf_cache / f"goa_{sp}.gaf.gz"
            if not local_gz.exists():
                print(f"  Downloading {sp} GAF...", flush=True)
                r = client.get(url)
                r.raise_for_status()
                local_gz.write_bytes(r.content)
            with gzip.open(local_gz, "rb") as gz:
                for raw_ln in gz:
                    if raw_ln.startswith(b"!"):
                        continue
                    cols = raw_ln.decode("utf-8", errors="ignore").rstrip("\n").split("\t")
                    if len(cols) < 15:
                        continue
                    db, acc, _sym, qual, go_id, ref, ev, _with, asp, _nm, _syn, _ot, _tax, dt, assigned = cols[:15]
                    if db != "UniProtKB" or ev not in EXP_CODES or "NOT" in qual.upper():
                        continue
                    # Drop IntAct bulk IPI & trivial protein-binding GO:0005515 per CAFA rules
                    if assigned == "IntAct" or go_id == "GO:0005515":
                        continue
                    prot_species.setdefault(acc, sp)
                    prot_anns[acc].append((go_id, asp, ev, dt, f"{ref}|{assigned}"))

        print(f"Parsed curated non-IntAct experimental annotations for {len(prot_anns):,} UniProt proteins.", flush=True)

        # Classify every protein for Pre and Post windows
        cells: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        for win_name, t0, t1 in WINDOWS:
            for acc, anns in prot_anns.items():
                res = classify_window(dag, anns, t0, t1)
                if res is None:
                    continue
                cell_key = f"{win_name}_{res['knowledge_type'].lower()}"
                res["accession"] = acc
                res["species_gaf"] = prot_species[acc]
                res["window"] = win_name
                res["cell"] = cell_key
                cells[cell_key].append(res)

        print("\n=== RAW CURATED NON-INTACT 2x3 CAFA-6 CELL COUNTS ===", flush=True)
        for ck in ("pre_nk", "pre_lk", "pre_pk", "post_nk", "post_lk", "post_pk"):
            arr = cells[ck]
            asps = Counter(a for x in arr for a in x["eval_aspects"])
            sp_c = Counter(x["species_gaf"] for x in arr).most_common(5)
            print(f"  {ck:8s}: {len(arr):5d} proteins | eval_aspects={dict(asps)} | top_sp={sp_c}", flush=True)

        # Pick a generous candidate pool (~120 per cell, prioritizing balanced MF/BP/CC eval aspects & species diversity)
        # so we only need to fetch UniProt sequences for ~600 proteins total!
        pool_by_cell: Dict[str, List[Dict[str, Any]]] = {}
        accs_to_fetch: Set[str] = set()

        # Always include our already-evaluated v2.0 Post-NK accessions!
        existing_post_nk = json.loads((REPO / "data" / "stage2b_cafa" / "manifest.json").read_text(encoding="utf-8"))["targets"]
        existing_post_nk_accs = {t["uniprot_accession"] for t in existing_post_nk}

        for ck in ("pre_nk", "pre_lk", "pre_pk", "post_nk", "post_lk", "post_pk"):
            arr = cells[ck]
            # Sort: for post_nk prefer already-evaluated targets first; otherwise rich eval aspects + balanced MF/BP/CC
            def _key(x: Dict[str, Any]) -> Tuple[int, int, int, int, str]:
                reuse = 0 if (ck == "post_nk" and x["accession"] in existing_post_nk_accs) else 1
                n_a = len(x["eval_aspects"])
                has_mf = 1 if "MF" in x["eval_aspects"] else 0
                has_cc = 1 if "CC" in x["eval_aspects"] else 0
                return (reuse, -n_a, -has_mf, -has_cc, -x["n_delta_terms"], x["accession"])  # type: ignore[return-value]

            # Take balanced per-aspect top candidates (50 MF + 50 BP + 50 CC) so every aspect has rich pool
            chosen_acc: Set[str] = set()
            chosen: List[Dict[str, Any]] = []
            # First take multi-aspect & reused
            for x in sorted(arr, key=_key):
                if x["accession"] in chosen_acc:
                    continue
                if ck == "post_nk" and x["accession"] in existing_post_nk_accs:
                    chosen_acc.add(x["accession"])
                    chosen.append(x)
            for asp in ("MF", "CC", "BP"):
                sub = [x for x in arr if asp in x["eval_aspects"] and x["accession"] not in chosen_acc]
                sub.sort(key=lambda x: (-len(x["eval_aspects"]), -x["n_delta_terms"], x["accession"]))
                for x in sub[:60]:
                    chosen_acc.add(x["accession"])
                    chosen.append(x)
            pool_by_cell[ck] = chosen
            accs_to_fetch.update(chosen_acc)

        print(f"\nFetching UniProt sequences & metadata for {len(accs_to_fetch)} pooled candidates...", flush=True)
        acc_list = sorted(accs_to_fetch)
        uniprot_meta: Dict[str, Dict[str, Any]] = {}
        for i in range(0, len(acc_list), 50):
            batch = acc_list[i : i + 50]
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
                    uniprot_meta[acc] = {
                        "accession": acc,
                        "entry_id": str(ent.get("uniProtkbId") or ""),
                        "organism": org,
                        "reviewed": str(ent.get("entryType") or "").lower().startswith("uniprotkb reviewed"),
                        "sequence": seq,
                        "length": len(seq),
                        "date_created": str(audit.get("firstPublicDate") or ""),
                        "safe": not bool(SAFETY_BLOCKLIST.search(f"{org} {pname}")),
                        "canonical_aa": bool(seq) and all(c in CANONICAL_AA_SET for c in seq),
                    }

        # Save enriched pool
        out_pool: Dict[str, List[Dict[str, Any]]] = {}
        print("\n=== CLEAN (50..800 AA, CANONICAL, SAFE) POOLED COUNTS PER CELL ===", flush=True)
        for ck in ("pre_nk", "pre_lk", "pre_pk", "post_nk", "post_lk", "post_pk"):
            clean = []
            for rec in pool_by_cell[ck]:
                u = uniprot_meta.get(rec["accession"])
                if not u or not (u["safe"] and u["canonical_aa"] and 50 <= u["length"] <= 800):
                    continue
                merged = dict(rec)
                merged.update(u)
                clean.append(merged)
            out_pool[ck] = clean
            asps = Counter(a for x in clean for a in x["eval_aspects"])
            print(f"  {ck:8s}: {len(clean):3d} clean | eval_aspects={dict(asps)}", flush=True)

        (OUT_DIR / "cafa6_6cell_pool.json").write_text(json.dumps(out_pool, indent=2), encoding="utf-8")
        print(f"\nSaved pooled 6-cell candidates -> {OUT_DIR / 'cafa6_6cell_pool.json'}", flush=True)


if __name__ == "__main__":
    main()
