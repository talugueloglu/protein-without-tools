#!/usr/bin/env python3
"""Rebuild Task 4 (stage2b_cafa) as a strictly post-May-2026 No-Knowledge (NK) experimental
Gene Ontology holdout (N = 60 non-redundant UniProtKB proteins):

Every single target satisfies:
  1. ZERO pre-2026-05-01 experimental GO annotations (double-verified against BOTH live EBI GOA
     FTP GAFs up to 2026-11-03 AND live EBI QuickGO ECO:0000269).
  2. Curated non-IntAct experimental evidence (IDA, IMP, IGI, IEP, EXP, HTP/HDA/HMP/HGI/HEP),
     excluding GO:0005515 (protein binding) only targets per official CAFA rules.
  3. Clean canonical protein sequence (50..800 AA), safety-screened, non-redundant (3-mer
     Jaccard <= 0.28), and zero accession overlap with the pre-2025 Swiss-Prot IC reference bank.

Two clean strata (30 + 30 = 60):
  - `post_may2026_paper`      (n=30): BOTH first experimental GO annotation >= 2026-05-01 AND
                                      underlying experimental PubMed paper >= 2026-05-01.
  - `post_may2026_go_curated` (n=30): First experimental GO annotation >= 2026-05-01 (zero prior
                                      GOA annotations), curated from earlier experimental papers.
"""

from __future__ import annotations

from collections import Counter
import json
from pathlib import Path
import shutil
import sys
from typing import Any, Dict, List, Set, Tuple

REPO = Path(__file__).resolve().parents[1]
if str(REPO / "src") not in sys.path:
    sys.path.insert(0, str(REPO / "src"))

from fable_bench.stage2b_cafa import (
    CAFATarget,
    GeneOntologyDAG,
    build_cafa_prompt,
    compute_term_information_content,
    parse_go_obo,
)


def kmer3(s: str) -> Set[str]:
    return {s[i : i + 3] for i in range(max(0, len(s) - 2))}


def write_compact_subdag_obo(dag: GeneOntologyDAG, keep_terms: Set[str], out_obo: Path) -> None:
    full_terms: Set[str] = set()
    for t in keep_terms:
        cid = dag.canonical(t)
        if cid:
            full_terms.add(cid)
            full_terms.update(dag.ancestors(cid, include_self=True))
    rev_asp = {"MF": "molecular_function", "BP": "biological_process", "CC": "cellular_component"}
    lines = ["format-version: 1.2", "ontology: go-subdag-stage2b", ""]
    for gid in sorted(full_terms):
        asp = dag.term_aspect.get(gid)
        if not asp:
            continue
        lines.append("[Term]")
        lines.append(f"id: {gid}")
        lines.append(f"name: {dag.term_name.get(gid, gid)}")
        lines.append(f"namespace: {rev_asp[asp]}")
        for pid in sorted(dag.parents.get(gid, ())):
            if pid in full_terms:
                lines.append(f"is_a: {pid}")
        lines.append("")
    out_obo.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    raw_cafa_dir = REPO / "data" / "raw_stage2_cache" / "cafa_goa"
    post_dir = REPO / "data" / "raw_stage2_cache" / "cafa_goa_post_may2026"
    out_s2b_dir = REPO / "data" / "stage2b_cafa"
    prompts_dir = REPO / "outputs" / "prompts"

    # Archive old v1 artifacts once if not already archived
    arch_dir = REPO / "outputs" / "archive_t4_v1"
    if not arch_dir.exists():
        arch_dir.mkdir(parents=True, exist_ok=True)
        for p in (
            out_s2b_dir / "manifest.json",
            out_s2b_dir / "go_subdag.obo",
            prompts_dir / "stage2b_cafa_prompts.jsonl",
            REPO / "outputs" / "results" / "baseline_t4_netgo4.json",
            REPO / "outputs" / "predictions" / "netgo_raw" / "ng4.0_results.txt",
            REPO / "outputs" / "predictions" / "gpt-6.1-sol_think4096_stage2b.jsonl",
            REPO / "outputs" / "predictions" / "gpt-6.2-sol_think4096_stage2b.jsonl",
            REPO / "outputs" / "predictions" / "gpt-6.2-astra_think4096_stage2b.jsonl",
        ):
            if p.exists():
                shutil.copy2(p, arch_dir / p.name)
        print(f"[Archive] Backed up v1 Task 4 files -> {arch_dir}", flush=True)

    dag = parse_go_obo(str(raw_cafa_dir / "go-basic.obo"))
    verified = json.loads((post_dir / "verified_strict_nk.json").read_text(encoding="utf-8"))
    raw_ref = json.loads((raw_cafa_dir / "pre2026_swissprot_go_reference.json").read_text(encoding="utf-8"))

    # Propagate GO terms for each verified strict-NK candidate
    candidates: List[Dict[str, Any]] = []
    for rec in verified:
        raw_go = {dag.canonical(str(a.get("goId") or "")) for a in rec["annotations"]}
        raw_go_clean = {g for g in raw_go if g is not None}
        mf_leaves = {g for g in raw_go_clean if dag.term_aspect.get(g) == "MF"}
        if mf_leaves == {"GO:0005515"}:
            raw_go_clean.discard("GO:0005515")
        prop = dag.propagate_set(raw_go_clean)
        mf = sorted(g for g in prop if dag.term_aspect.get(g) == "MF")
        bp = sorted(g for g in prop if dag.term_aspect.get(g) == "BP")
        cc = sorted(g for g in prop if dag.term_aspect.get(g) == "CC")
        if not (mf or bp or cc):
            continue
        assert rec["min_go_date"] >= "20260501", f"Non-post-cutoff candidate leaked: {rec['accession']}"
        item = dict(rec)
        item["mf"] = mf
        item["bp"] = bp
        item["cc"] = cc
        item["n_aspects"] = int(bool(mf)) + int(bool(bp)) + int(bool(cc))
        item["n_terms"] = len(mf) + len(bp) + len(cc)
        item["has_mf"] = bool(mf)
        item["has_cc"] = bool(cc)
        item["is_post_may_paper"] = bool(rec.get("min_pmid_date", "") >= "2026-05-01")
        candidates.append(item)

    # Split into the two strict-NK strata:
    #   Stratum A (`post_may2026_paper`, n=30): BOTH GO annotation >= 2026-05-01 AND PubMed paper >= 2026-05-01
    #   Stratum B (`post_may2026_go_curated`, n=30): GO annotation >= 2026-05-01 (zero prior GOA), earlier paper
    pool_a = [c for c in candidates if c["is_post_may_paper"]]
    pool_b = [c for c in candidates if not c["is_post_may_paper"]]

    # Sort to maximize multi-aspect balance (MF & CC especially, since BP is abundant) + annotation richness
    def sort_key(c: Dict[str, Any]) -> Tuple[int, int, int, int, str]:
        return (
            -c["n_aspects"],
            -(1 if c["has_mf"] else 0),
            -(1 if c["has_cc"] else 0),
            -c["n_terms"],
            c["accession"],
        )

    pool_a.sort(key=sort_key)
    pool_b.sort(key=sort_key)

    selected: List[Tuple[str, Dict[str, Any]]] = []
    kept_kmers: List[Set[str]] = []

    def take_non_redundant(pool: List[Dict[str, Any]], tier_name: str, quota: int) -> int:
        taken = 0
        for c in pool:
            if taken >= quota:
                break
            km = kmer3(c["sequence"])
            if any(len(km & okm) / float(max(1, len(km | okm))) > 0.28 for okm in kept_kmers):
                continue
            kept_kmers.append(km)
            selected.append((tier_name, c))
            taken += 1
        return taken

    n_a = take_non_redundant(pool_a, "post_may2026_paper", 30)
    n_b = take_non_redundant(pool_b, "post_may2026_go_curated", 30)
    assert n_a == 30 and n_b == 30, f"Expected 30+30, got {n_a}+{n_b}"

    holdout_accs = {c["accession"] for _, c in selected}

    # Build pre-2025 Swiss-Prot reference bank (zero accession overlap with holdouts)
    ref_bank: List[Dict[str, Any]] = []
    ref_prop_sets: List[Set[str]] = []
    all_needed_terms: Set[str] = set()
    for r in raw_ref:
        acc = r["accession"]
        if acc in holdout_accs:
            continue
        canon_terms = sorted({dag.canonical(g) for g in r["go_terms"] if dag.canonical(g) is not None})  # type: ignore[arg-type]
        if not canon_terms:
            continue
        prop_set = dag.propagate_set(canon_terms)
        if not prop_set:
            continue
        ref_bank.append({"accession": acc, "sequence": r["sequence"], "go_terms": canon_terms})
        ref_prop_sets.append(prop_set)
        all_needed_terms.update(prop_set)

    targets: List[CAFATarget] = []
    manifest_targets: List[Dict[str, Any]] = []
    for idx, (tier_name, c) in enumerate(selected, 1):
        tid = f"s2b_t{idx:03d}"
        mf, bp, cc = c["mf"], c["bp"], c["cc"]
        all_needed_terms.update(mf)
        all_needed_terms.update(bp)
        all_needed_terms.update(cc)
        t_obj = CAFATarget(
            target_id=tid,
            sequence=c["sequence"],
            length=len(c["sequence"]),
            true_terms_mf=mf,
            true_terms_bp=bp,
            true_terms_cc=cc,
            knowledge_type="NK",
        )
        targets.append(t_obj)
        manifest_targets.append(
            {
                "target_id": tid,
                "uniprot_accession": c["accession"],
                "entry_id": c["entry_id"],
                "organism": c["organism"],
                "species_gaf": c["species_gaf"],
                "length": len(c["sequence"]),
                "sequence": c["sequence"],
                "uniprot_first_public_date": c["date_created"],
                "quickgo_min_exp_date": c["min_go_date"],
                "quickgo_max_exp_date": c["max_go_date"],
                "pubmed_min_date": c["min_pmid_date"],
                "pubmed_max_date": c["max_pmid_date"],
                "pmids": c["pmids"],
                "assigned_by": sorted({a["assignedBy"] for a in c["annotations"]}),
                "evidence_codes": sorted({a["evidence"] for a in c["annotations"]}),
                "temporal_tier": tier_name,
                "knowledge_type": "NK",
                "true_terms_mf": mf,
                "true_terms_bp": bp,
                "true_terms_cc": cc,
            }
        )

    ic_map = compute_term_information_content(dag, ref_prop_sets)
    write_compact_subdag_obo(dag, all_needed_terms, out_s2b_dir / "go_subdag.obo")
    (out_s2b_dir / "reference_swissprot_go.json").write_text(json.dumps(ref_bank, indent=2), encoding="utf-8")

    manifest = {
        "benchmark": "fable-protein-bench-stage2b-cafa",
        "version": "2.0",
        "source": "Post-May-2026 Strict-NK EBI GOA (2026-11-03) + EBI QuickGO double-verified experimental GOA (non-IntAct)",
        "cutoff_date": "2026-05-01",
        "n_targets": len(targets),
        "n_reference_swissprot_pre2025": len(ref_bank),
        "aspect_target_counts": {
            "MF": sum(1 for t in targets if t.true_terms_mf),
            "BP": sum(1 for t in targets if t.true_terms_bp),
            "CC": sum(1 for t in targets if t.true_terms_cc),
        },
        "temporal_tier_counts": dict(Counter(m["temporal_tier"] for m in manifest_targets)),
        "species_counts": dict(Counter(m["species_gaf"] for m in manifest_targets)),
        "targets": manifest_targets,
    }
    (out_s2b_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    # Export strictly zero-leakage prompts
    p_lines = []
    for t_obj, m_row in zip(targets, manifest_targets):
        pr = build_cafa_prompt(t_obj)
        assert m_row["uniprot_accession"] not in pr
        if m_row["entry_id"]:
            assert m_row["entry_id"] not in pr
        p_lines.append(
            json.dumps(
                {
                    "target_id": t_obj.target_id,
                    "task": "stage2b_cafa",
                    "temporal_tier": m_row["temporal_tier"],
                    "length": t_obj.length,
                    "prompt": pr,
                }
            )
        )
    (prompts_dir / "stage2b_cafa_prompts.jsonl").write_text("\n".join(p_lines) + "\n", encoding="utf-8")

    spot_dir = prompts_dir / "web_chat_spotcheck"
    spot_dir.mkdir(parents=True, exist_ok=True)
    for m_row, t_obj in zip(manifest_targets[:2], targets[:2]):
        (spot_dir / f"{t_obj.target_id}_cafa_function.txt").write_text(build_cafa_prompt(t_obj), encoding="utf-8")

    print("\n=== REBUILT TASK 4 (v2.0 STRICT POST-MAY-2026 NK HOLDOUT) ===", flush=True)
    print(f"  Targets            : {len(targets)} (ALL min_exp_go_date >= 2026-05-01)", flush=True)
    print(f"  Temporal strata    : {manifest['temporal_tier_counts']}", flush=True)
    print(f"  Aspect counts      : {manifest['aspect_target_counts']}", flush=True)
    print(f"  Species breakdown  : {manifest['species_counts']}", flush=True)
    print(f"  Min/Max GO date    : {min(m['quickgo_min_exp_date'] for m in manifest_targets)} .. {max(m['quickgo_max_exp_date'] for m in manifest_targets)}", flush=True)
    print(f"  Paper>=2026-05     : {sum(1 for m in manifest_targets if m['pubmed_min_date'] >= '2026-05-01')}/60", flush=True)
    print(f"  Sub-DAG GO terms   : {len(all_needed_terms)} (IC entries: {len(ic_map)})", flush=True)


if __name__ == "__main__":
    main()
