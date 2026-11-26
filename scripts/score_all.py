#!/usr/bin/env python3
"""Unified 13-Task Summary & Scoring Report for Proteins Without Tools (v1 + v2).

Reads all curated manifests, anti-cheat null floors, specialist SOTA baselines,
and any available LLM prediction JSONLs for Tasks 9–13 (while also summarizing
Tasks 1–8 manifests) to produce a single auditable suite report.
"""

from __future__ import annotations

import json
from pathlib import Path
import sys
from typing import Any, Dict, List, Optional

import numpy as np

REPO = Path(__file__).resolve().parents[1]
if str(REPO / "src") not in sys.path:
    sys.path.insert(0, str(REPO / "src"))

from protein_without_tools.task9_epistasis import DoubleMutantRecord, grade_epistasis_panel, parse_epistasis_response
from protein_without_tools.task10_ligand_site import SiteCandidate, grade_site_ranking_panel, parse_site_ranking_response
from protein_without_tools.task11_dynamics import DynamicsResidue, grade_dynamics_panel
from protein_without_tools.task13_developability import (
    DevelopabilityCandidate,
    grade_developability_panel,
    parse_developability_response,
)


def _load_json(path: Path) -> Optional[Dict[str, Any]]:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def score_v2_llm_predictions() -> Dict[str, Dict[str, Any]]:
    """Grade any LLM prediction files present in `outputs/predictions/` for Tasks 9–13."""
    pdir = REPO / "outputs/predictions"
    if not pdir.exists():
        return {}
    out: Dict[str, Dict[str, Any]] = {}

    # Task 9
    m9 = _load_json(REPO / "data/task9_epistasis/manifest.json")
    if m9:
        by_id = {t["target_id"]: t for t in m9["targets"]}
        for fp in sorted(pdir.glob("*_task9_epistasis.jsonl")):
            tag = fp.name.replace("_task9_epistasis.jsonl", "")
            rows = [json.loads(l) for l in fp.read_text(encoding="utf-8").splitlines() if l.strip()]
            aurocs, rhos, signs = [], [], []
            for r in rows:
                t = by_id.get(r["target_id"])
                if not t:
                    continue
                panel = [DoubleMutantRecord(**p) for p in t["pairs"]]
                ranked, sign_map, _ok = parse_epistasis_response(r.get("text", ""), [p.pair_id for p in panel])
                g = grade_epistasis_panel(panel, ranked, sign_map)
                aurocs.append(g["coupling_auroc"])
                rhos.append(g["spearman_rho_abs_dddg"])
                if g["sign_accuracy"] is not None:
                    signs.append(g["sign_accuracy"])
            if aurocs:
                out.setdefault("task9", {})[tag] = {
                    "n": len(aurocs),
                    "coupling_auroc": round(float(np.mean(aurocs)), 4),
                    "spearman_rho": round(float(np.mean(rhos)), 4),
                    "sign_accuracy": round(float(np.mean(signs)), 4),
                }

    # Tasks 10 & 12
    for tkey, mpath, suf in (
        ("task10", REPO / "data/task10_ligand_site/manifest.json", "_task10_ligand_site.jsonl"),
        ("task12", REPO / "data/task12_nucleic_acid/manifest.json", "_task12_nucleic_acid.jsonl"),
    ):
        m = _load_json(mpath)
        if not m:
            continue
        by_id = {t["target_id"]: t for t in m["targets"]}
        for fp in sorted(pdir.glob(f"*{suf}")):
            tag = fp.name.replace(suf, "")
            rows = [json.loads(l) for l in fp.read_text(encoding="utf-8").splitlines() if l.strip()]
            aurocs, wins = [], []
            for r in rows:
                t = by_id.get(r["target_id"])
                if not t:
                    continue
                panel = [SiteCandidate(**c) for c in t["candidates"]]
                ranked, _ok = parse_site_ranking_response(r.get("text", ""), [c.res_id for c in panel])
                g = grade_site_ranking_panel(panel, ranked)
                aurocs.append(g["auroc"])
                wins.append(g["same_aa_win_rate"])
            if aurocs:
                out.setdefault(tkey, {})[tag] = {
                    "n": len(aurocs),
                    "auroc": round(float(np.mean(aurocs)), 4),
                    "same_aa_win_rate": round(float(np.mean(wins)), 4),
                }

    # Task 11
    m11 = _load_json(REPO / "data/task11_dynamics/manifest.json")
    if m11:
        by_id = {t["target_id"]: t for t in m11["targets"]}
        for fp in sorted(pdir.glob("*_task11_dynamics.jsonl")):
            tag = fp.name.replace("_task11_dynamics.jsonl", "")
            rows = [json.loads(l) for l in fp.read_text(encoding="utf-8").splitlines() if l.strip()]
            aurocs, rhos = [], []
            for r in rows:
                t = by_id.get(r["target_id"])
                if not t:
                    continue
                panel = [DynamicsResidue(**c) for c in t["candidates"]]
                ranked, _ok = parse_site_ranking_response(r.get("text", ""), [c.res_id for c in panel])
                g = grade_dynamics_panel(panel, ranked)
                aurocs.append(g["auroc"])
                rhos.append(g["spearman_rho"])
            if aurocs:
                out.setdefault("task11", {})[tag] = {
                    "n": len(aurocs),
                    "auroc": round(float(np.mean(aurocs)), 4),
                    "spearman_rho": round(float(np.mean(rhos)), 4),
                }

    # Task 13
    m13 = _load_json(REPO / "data/task13_developability/manifest.json")
    if m13:
        by_id = {t["target_id"]: t for t in m13["targets"]}
        for fp in sorted(pdir.glob("*_task13_developability.jsonl")):
            tag = fp.name.replace("_task13_developability.jsonl", "")
            rows = [json.loads(l) for l in fp.read_text(encoding="utf-8").splitlines() if l.strip()]
            rhos, accs = [], []
            for r in rows:
                t = by_id.get(r["target_id"])
                if not t:
                    continue
                panel = [DevelopabilityCandidate(**c) for c in t["candidates"]]
                ranked, _ok = parse_developability_response(r.get("text", ""), [c.cand_id for c in panel])
                g = grade_developability_panel(panel, ranked)
                rhos.append(g["spearman_rho"])
                accs.append(g["pairwise_accuracy"])
            if rhos:
                out.setdefault("task13", {})[tag] = {
                    "n": len(rhos),
                    "spearman_rho": round(float(np.mean(rhos)), 4),
                    "pairwise_accuracy": round(float(np.mean(accs)), 4),
                }

    return out


def main() -> None:
    print("# Proteins Without Tools — Full 13-Task Suite Summary (v1 + v2)\n")

    v1_counts = [
        ("Task 1", "Forward Structure (Q3 / Contact F1)", "data/stage1_benchmark/manifest.json", "targets"),
        ("Task 2", "Inverse Folding (Seq Recovery / scTM)", "data/stage1_benchmark/manifest.json", "targets"),
        ("Task 3", "Remote Homology (Twilight Zone)", "data/stage2a_homology/manifest.json", "pairs"),
        ("Task 4", "Function Prediction (CAFA-6 NK/LK)", "data/stage2b_cafa/manifest.json", "targets"),
        ("Task 5", "Monomer Stability (Alanine Scan)", "data/alanine_scan_benchmark/manifest.json", "targets"),
        ("Task 6", "PPI Binding Hotspots (SKEMPI)", "data/binding_alanine_benchmark/manifest.json", "targets"),
        ("Task 7", "De Novo Motif Scaffolding", "data/stage3b_design/monomer_motif_manifest.json", "targets"),
        ("Task 8", "Closed-Loop Binder Design", "data/stage3b_design/epitope_binder_manifest.json", "targets"),
    ]
    v1_total = 0
    print("## v1 Tasks (Tasks 1–8)")
    print("| Task | Description | N |")
    print("|---|---|---:|")
    for tid, desc, mrel, key in v1_counts:
        m = _load_json(REPO / mrel)
        if isinstance(m, list):
            n = len(m)
        elif isinstance(m, dict):
            arr = m.get(key) or m.get("tasks") or m.get("epitopes") or m.get("designs") or m.get("pairs") or m.get("targets")
            n = len(arr) if isinstance(arr, list) else int(m.get("n_targets") or m.get("n_tasks") or 0)
        else:
            n = 0
        v1_total += n
        print(f"| {tid} | {desc} | {n} |")
    print(f"| **v1 Subtotal** | | **{v1_total}** |\n")

    v2_specs = [
        ("Task 9", "Double-Mutant Epistasis", "data/task9_epistasis/manifest.json", "outputs/results/baseline_t9_proteinmpnn.json"),
        ("Task 10", "Ligand / Cofactor / Metal Site", "data/task10_ligand_site/manifest.json", "outputs/results/baseline_t10_boltz2.json"),
        ("Task 11", "Conformational Dynamics (Apo–Holo)", "data/task11_dynamics/manifest.json", "outputs/results/baseline_t11_dynamics_boltz2_ensemble.json"),
        ("Task 12", "Protein–DNA / RNA Interface", "data/task12_nucleic_acid/manifest.json", "outputs/results/baseline_t12_boltz2.json"),
        ("Task 13", "Wet-Lab Antibody Developability", "data/task13_developability/manifest.json", "outputs/results/baseline_t13_developability.json"),
    ]
    v2_total = 0
    print("## v2 Tasks (Tasks 9–13 — Uncontaminated Specialist Baselines & Anti-Cheat Nulls)")
    print("| Task | Description | N | Anti-Cheat Null Floor | SOTA Specialist Baseline | Specialist Score |")
    print("|---|---|---:|---|---|---|")
    for tid, desc, mrel, brel in v2_specs:
        m = _load_json(REPO / mrel)
        b = _load_json(REPO / brel)
        n = int(m.get("n_targets", 0)) if m else 0
        v2_total += n
        null_str = json.dumps(m.get("anti_cheat_audit", {})) if m else "pending"
        if b:
            sp_name = b.get("specialist_model") or b.get("baseline") or b.get("model") or "specialist"
            ov = b.get("overall", {})
            n_eval = ov.get("n") or ov.get("n_targets") or b.get("n_evaluated") or n
            sc_str = ", ".join(f"{k}={v}" for k, v in ov.items() if k not in ("n", "n_targets"))
            sc_str = f"n={n_eval}: {sc_str}"
        else:
            sp_name = "pending"
            sc_str = "pending"
        print(f"| {tid} | {desc} | {n} | `{null_str}` | {sp_name} | `{sc_str}` |")

    print(f"| **v2 Subtotal** | | **{v2_total}** | | | |")
    print(f"| **Full Suite Total (13 tasks)** | | **{v1_total + v2_total}** | | | |")

    llm_scores = score_v2_llm_predictions()
    if llm_scores:
        print("\n## v2 LLM Predictions Graded")
        print(json.dumps(llm_scores, indent=2))


if __name__ == "__main__":
    main()
