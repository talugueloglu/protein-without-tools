#!/usr/bin/env python3
"""Evaluate Test 6 Interface Binding Alanine Scan Ranking (`*_stage6_binding.jsonl`) predictions."""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
import sys
from typing import Any, Dict, List

import numpy as np

REPO = Path("/Users/taluguloglu/.gemini/antigravity/scratch/fable-protein-bench")
if str(REPO / "src") not in sys.path:
    sys.path.insert(0, str(REPO / "src"))

from fable_bench.binding_alanine_scan import (
    BOGAN_THORN_BINDING_HOTSPOT_SCALE,
    BindingAlanineGrade,
    BindingAlanineParsed,
    BindingAlanineTarget,
    grade_binding_alanine_target,
    parse_binding_alanine_response,
)


def load_binding_targets(data_dir: Path) -> Dict[str, BindingAlanineTarget]:
    manifest = json.loads((data_dir / "manifest.json").read_text(encoding="utf-8"))
    out: Dict[str, BindingAlanineTarget] = {}
    for rec in manifest["targets"]:
        tid = rec["target_id"]
        tj = json.loads((data_dir / "targets" / f"{tid}.json").read_text(encoding="utf-8"))
        out[tid] = BindingAlanineTarget(
            target_id=tid,
            category=tj["category"],
            assay_type=tj["assay_type"],
            scanned_sequence=tj["scanned_sequence"],
            partner_sequence=tj["partner_sequence"],
            positions_1idx=tj["positions_1idx"],
            wt_aas=tj["wt_aas"],
            exp_binding_loss=tj["exp_binding_loss"],
            is_interface_flags=tj.get("is_interface_flags"),
            panel_roles=tj.get("panel_roles"),
            memorization_stratum=tj.get("memorization_stratum", "known_pdb_pre2026"),
        )
    return out


def summarize_grades(model: str, grades: List[BindingAlanineGrade]) -> Dict[str, Any]:
    def _agg(sub: List[BindingAlanineGrade]) -> Dict[str, Any]:
        if not sub:
            return {}
        return {
            "n": len(sub),
            "parse_ok_rate": round(float(np.mean([1.0 if g.parse_ok else 0.0 for g in sub])), 4),
            "spearman_rho": round(float(np.mean([g.spearman_rho for g in sub])), 4),
            "ndcg_at_5": round(float(np.mean([g.ndcg_at_5 for g in sub])), 4),
            "top_k_precision": round(float(np.mean([g.top_k_precision for g in sub])), 4),
            "same_aa_win_rate": round(float(np.mean([g.same_aa_win_rate for g in sub])), 4),
            "spearman_rho_aa_demeaned": round(float(np.mean([g.spearman_rho_aa_demeaned for g in sub])), 4),
            "binding_hotspot_auroc": round(float(np.mean([g.binding_hotspot_auroc for g in sub])), 4),
        }

    by_mem: Dict[str, List[BindingAlanineGrade]] = defaultdict(list)
    by_cat: Dict[str, List[BindingAlanineGrade]] = defaultdict(list)
    for g in grades:
        by_mem[g.memorization_stratum].append(g)
        by_cat[g.category].append(g)

    return {
        "model": model,
        "overall": _agg(grades),
        "by_memorization_stratum": {k: _agg(v) for k, v in sorted(by_mem.items())},
        "by_category": {k: _agg(v) for k, v in sorted(by_cat.items())},
        "per_target": [g.to_dict() for g in grades],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default=str(REPO / "data/binding_alanine_benchmark"))
    parser.add_argument("--pred-dir", default=str(REPO / "outputs/predictions"))
    parser.add_argument("--out-file", default=str(REPO / "outputs/results/live_stage6_binding_report.json"))
    args = parser.parse_args()

    targets = load_binding_targets(Path(args.data_dir))
    summaries: List[Dict[str, Any]] = []

    # Include Bogan-Thorn 1D reference baseline
    bt_grades = []
    for t in targets.values():
        bt = np.asarray([BOGAN_THORN_BINDING_HOTSPOT_SCALE.get(aa, 0.5) for aa in t.wt_aas], dtype=np.float64)
        bt_grades.append(grade_binding_alanine_target(t, BindingAlanineParsed(bt, len(bt), 1.0, True)))
    summaries.append(summarize_grades("Baseline_BoganThorn_1D", bt_grades))

    pdir = Path(args.pred_dir)
    if pdir.exists():
        for jf in sorted(pdir.glob("*_stage6_binding.jsonl")):
            mname = jf.name.replace("_stage6_binding.jsonl", "")
            rows = [json.loads(ln) for ln in jf.read_text(encoding="utf-8").splitlines() if ln.strip()]
            grades = [
                grade_binding_alanine_target(targets[r["target_id"]], parse_binding_alanine_response(r.get("text", ""), targets[r["target_id"]]))
                for r in rows
                if r.get("target_id") in targets
            ]
            if grades:
                summaries.append(summarize_grades(mname, grades))

    Path(args.out_file).write_text(json.dumps({"test6_binding_alanine_ranking": summaries}, indent=2), encoding="utf-8")

    print("=" * 112)
    print("TEST 6 — INTERFACE BINDING ALANINE RANKING SUMMARY (OVERALL + KNOWN PDB vs UNSEEN/POST-2026)")
    print("=" * 112)
    print(f"{'Model':<32s} | {'Rank ρ':>7s} | {'NDCG@5':>7s} | {'SameAA':>7s} | {'AA-Ctx':>7s} | {'KnownPDB ρ':>10s} | {'Unseen ρ':>9s}")
    print("-" * 112)
    for s in summaries:
        ov = s["overall"]
        kp = s["by_memorization_stratum"].get("known_pdb_pre2026", {}).get("spearman_rho", 0.0)
        up = s["by_memorization_stratum"].get("unseen_or_post2026", {}).get("spearman_rho", 0.0)
        print(
            f"{s['model']:<32s} | {ov.get('spearman_rho', 0.0):7.3f} | {ov.get('ndcg_at_5', 0.0):7.3f} | "
            f"{ov.get('same_aa_win_rate', 0.5):7.3f} | {ov.get('spearman_rho_aa_demeaned', 0.0):7.3f} | "
            f"{kp:10.3f} | {up:9.3f}"
        )


if __name__ == "__main__":
    main()
