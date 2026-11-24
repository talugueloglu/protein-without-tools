#!/usr/bin/env python3
"""Build Task 4 (stage2b_cafa v3.0) as a full CAFA-6-style 2x3 temporal & knowledge benchmark:
  - Pre-cutoff  (t0' = 2025-05-01 -> t1' = 2025-11-01): pre_nk (20), pre_lk (20), pre_pk (20)
  - Post-cutoff (t0  = 2026-05-01 -> t1  = 2026-11-03): post_nk (20), post_lk (20), post_pk (20)
Total: N = 120 non-redundant (3-mer Jaccard <= 0.28) UniProtKB proteins across 13 species.

Reuses 20 already-evaluated & QuickGO-double-verified strict Post-NK targets (zero wasted LLM spend!).
Exports `data/stage2b_cafa/manifest.json`, `go_subdag.obo`, `reference_swissprot_go.json`, and
zero-leakage `outputs/prompts/stage2b_cafa_prompts.jsonl`.
"""

from __future__ import annotations

from collections import Counter
import json
from pathlib import Path
import sys
from typing import Any, Dict, List, Set, Tuple

REPO = Path(__file__).resolve().parents[1]
if str(REPO / "src") not in sys.path:
    sys.path.insert(0, str(REPO / "src"))

from fable_bench.llm_client import UnifiedLLMRunner
from fable_bench.stage2b_cafa import (
    CAFATarget,
    GeneOntologyDAG,
    build_cafa_prompt,
    compute_term_information_accretion,
    parse_go_obo,
)

CELL_ORDER = ("pre_nk", "pre_lk", "pre_pk", "post_nk", "post_lk", "post_pk")
PER_CELL = 20


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
    pool_file = REPO / "data" / "raw_stage2_cache" / "cafa6_6cell" / "cafa6_6cell_pool.json"
    v2_manifest_file = REPO / "data" / "stage2b_cafa" / "manifest.json"
    out_s2b_dir = REPO / "data" / "stage2b_cafa"
    prompts_dir = REPO / "outputs" / "prompts"

    dag = parse_go_obo(str(raw_cafa_dir / "go-basic.obo"))
    pool_by_cell: Dict[str, List[Dict[str, Any]]] = json.loads(pool_file.read_text(encoding="utf-8"))
    raw_ref = json.loads((raw_cafa_dir / "pre2026_swissprot_go_reference.json").read_text(encoding="utf-8"))

    # Check which v2.0 Post-NK targets are ALREADY cached across all 3 LLMs (gpt-6.1-sol, gpt-6.2-sol, gpt-6.2-astra)
    v2_targets = json.loads(v2_manifest_file.read_text(encoding="utf-8"))["targets"]
    v2_by_acc = {t["uniprot_accession"]: t for t in v2_targets}
    runner = UnifiedLLMRunner()
    cached_all3: Set[str] = set()
    for t in v2_targets:
        t_obj = CAFATarget(
            target_id=t["target_id"],
            sequence=t["sequence"],
            length=len(t["sequence"]),
            true_terms_mf=t["true_terms_mf"],
            true_terms_bp=t["true_terms_bp"],
            true_terms_cc=t["true_terms_cc"],
        )
        pr = build_cafa_prompt(t_obj)
        if all(
            runner._cache_path("openai", mk, pr, 4096, 0.0).exists()
            for mk in ("gpt-6.1-sol", "gpt-6.2-sol", "gpt-6.2-astra")
        ):
            cached_all3.add(t["uniprot_accession"])
    print(f"Found {len(cached_all3)}/60 v2.0 Post-NK targets already cached for ALL 3 LLMs!", flush=True)

    # Select 20 non-redundant targets per cell, zero overlap across cells, balanced MF/BP/CC per cell!
    used_accs: Set[str] = set()
    kept_kmers: List[Set[str]] = []
    selected: List[Dict[str, Any]] = []

    def can_take(rec: Dict[str, Any]) -> bool:
        acc = rec["accession"]
        if acc in used_accs:
            return False
        km = kmer3(rec["sequence"])
        if any(len(km & okm) / float(max(1, len(km | okm))) > 0.28 for okm in kept_kmers):
            return False
        return True

    def take(rec: Dict[str, Any]) -> None:
        used_accs.add(rec["accession"])
        kept_kmers.append(kmer3(rec["sequence"]))
        selected.append(rec)

    # Process post_nk FIRST so we lock in our already-cached double-verified Post-NK targets!
    cell_process_order = ("post_nk", "post_lk", "post_pk", "pre_nk", "pre_lk", "pre_pk")
    by_cell_selected: Dict[str, List[Dict[str, Any]]] = {c: [] for c in CELL_ORDER}

    for ck in cell_process_order:
        cands = list(pool_by_cell[ck])
        if ck == "post_nk":
            # Restrict strictly to our QuickGO-double-verified v2.0 targets, preferring cached_all3 + post_may2026_paper!
            cands = [r for r in cands if r["accession"] in v2_by_acc]

        def score_cand(r: Dict[str, Any], asp_counts: Counter[str]) -> Tuple[int, int, int, int, int, str]:
            acc = r["accession"]
            # Prefer already-cached targets in post_nk
            not_cached = 0 if (ck == "post_nk" and acc in cached_all3) else 1
            # In post_nk, prefer post_may2026_paper too
            not_paper = 0 if (ck == "post_nk" and v2_by_acc.get(acc, {}).get("temporal_tier") == "post_may2026_paper") else 1
            # Greedy aspect-balancing score: reward covering whichever aspect(s) currently have fewest targets in this cell
            min_asp_cov = min(asp_counts[a] for a in ("MF", "BP", "CC"))
            covers_needy = sum(1 for a in r["eval_aspects"] if asp_counts[a] == min_asp_cov)
            n_asps = len(r["eval_aspects"])
            return (not_cached, not_paper, -covers_needy, -n_asps, -r["n_delta_terms"], acc)

        asp_counts: Counter[str] = Counter()
        chosen_cell: List[Dict[str, Any]] = []
        while len(chosen_cell) < PER_CELL:
            avail = [r for r in cands if can_take(r)]
            if not avail:
                raise RuntimeError(f"Ran out of candidates for cell {ck} at {len(chosen_cell)}/{PER_CELL}")
            avail.sort(key=lambda r: score_cand(r, asp_counts))
            best = avail[0]
            take(best)
            chosen_cell.append(best)
            for a in best["eval_aspects"]:
                asp_counts[a] += 1
        by_cell_selected[ck] = chosen_cell

    # Order cleanly by CELL_ORDER (pre_nk, pre_lk, pre_pk, post_nk, post_lk, post_pk)
    ordered_recs: List[Dict[str, Any]] = []
    for ck in CELL_ORDER:
        ordered_recs.extend(by_cell_selected[ck])

    holdout_accs = {r["accession"] for r in ordered_recs}

    # Rebuild pre-2025 Swiss-Prot reference bank with zero accession overlap
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
    for idx, rec in enumerate(ordered_recs, 1):
        tid = f"s2b_t{idx:03d}"
        t1_map = rec["true_t1"]
        t0_map = rec["known_t0"]
        for a in ("MF", "BP", "CC"):
            all_needed_terms.update(t1_map[a])
            all_needed_terms.update(t0_map[a])

        t_obj = CAFATarget(
            target_id=tid,
            sequence=rec["sequence"],
            length=len(rec["sequence"]),
            true_terms_mf=t1_map["MF"],
            true_terms_bp=t1_map["BP"],
            true_terms_cc=t1_map["CC"],
            knowledge_type=rec["knowledge_type"],
            known_terms_mf=t0_map["MF"],
            known_terms_bp=t0_map["BP"],
            known_terms_cc=t0_map["CC"],
            eval_aspects=rec["eval_aspects"],
        )
        targets.append(t_obj)
        manifest_targets.append(
            {
                "target_id": tid,
                "uniprot_accession": rec["accession"],
                "entry_id": rec["entry_id"],
                "organism": rec["organism"],
                "species_gaf": rec["species_gaf"],
                "length": len(rec["sequence"]),
                "sequence": rec["sequence"],
                "uniprot_first_public_date": rec["date_created"],
                "quickgo_min_exp_date": rec["min_win_date"],
                "quickgo_max_exp_date": rec["max_win_date"],
                "temporal_window": rec["window"],
                "knowledge_type": rec["knowledge_type"],
                "temporal_tier": rec["cell"],  # pre_nk | pre_lk | pre_pk | post_nk | post_lk | post_pk
                "eval_aspects": rec["eval_aspects"],
                "pmids": rec["win_pmids"],
                "assigned_by": rec["win_assigned_by"],
                "evidence_codes": rec["win_evidence_codes"],
                "known_terms_mf": t0_map["MF"],
                "known_terms_bp": t0_map["BP"],
                "known_terms_cc": t0_map["CC"],
                "true_terms_mf": t1_map["MF"],
                "true_terms_bp": t1_map["BP"],
                "true_terms_cc": t1_map["CC"],
                "delta_terms_mf": rec["delta_t1_minus_t0"]["MF"],
                "delta_terms_bp": rec["delta_t1_minus_t0"]["BP"],
                "delta_terms_cc": rec["delta_t1_minus_t0"]["CC"],
            }
        )

    ia_map = compute_term_information_accretion(dag, ref_prop_sets)
    write_compact_subdag_obo(dag, all_needed_terms, out_s2b_dir / "go_subdag.obo")
    (out_s2b_dir / "reference_swissprot_go.json").write_text(json.dumps(ref_bank, indent=2), encoding="utf-8")

    cell_aspect_counts = {
        ck: {
            a: sum(1 for m in manifest_targets if m["temporal_tier"] == ck and a in m["eval_aspects"])
            for a in ("MF", "BP", "CC")
        }
        for ck in CELL_ORDER
    }

    manifest = {
        "benchmark": "fable-protein-bench-stage2b-cafa",
        "version": "3.0",
        "source": "CAFA-6 2x3 Experimental GOA Benchmark (EBI GOA 2026-11-03; Pre 2025-05..11 vs Post 2026-05..11; NK/LK/PK)",
        "windows": {
            "pre": {"t0": "2025-05-01", "t1": "2025-11-01"},
            "post": {"t0": "2026-05-01", "t1": "2026-11-03"},
        },
        "n_targets": len(targets),
        "n_reference_swissprot_pre2025": len(ref_bank),
        "aspect_target_counts": {
            a: sum(1 for m in manifest_targets if a in m["eval_aspects"])
            for a in ("MF", "BP", "CC")
        },
        "temporal_tier_counts": dict(Counter(m["temporal_tier"] for m in manifest_targets)),
        "cell_aspect_counts": cell_aspect_counts,
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
                    "knowledge_type": m_row["knowledge_type"],
                    "length": t_obj.length,
                    "prompt": pr,
                }
            )
        )
    (prompts_dir / "stage2b_cafa_prompts.jsonl").write_text("\n".join(p_lines) + "\n", encoding="utf-8")

    print("\n=== REBUILT TASK 4 v3.0 (CAFA-6 2x3: Pre/Post x NK/LK/PK, N=120) ===", flush=True)
    for ck in CELL_ORDER:
        sub = [m for m in manifest_targets if m["temporal_tier"] == ck]
        reused = sum(1 for m in sub if m["uniprot_accession"] in cached_all3)
        print(
            f"  {ck:8s}: n={len(sub):2d} | eval_aspects={cell_aspect_counts[ck]} | "
            f"dates={min(m['quickgo_min_exp_date'] for m in sub)}..{max(m['quickgo_max_exp_date'] for m in sub)} | "
            f"cached_all3={reused}",
            flush=True,
        )
    print(f"  Species: {manifest['species_counts']}", flush=True)
    print(f"  Sub-DAG GO terms: {len(all_needed_terms)} | IA terms: {len(ia_map)}", flush=True)


if __name__ == "__main__":
    main()
