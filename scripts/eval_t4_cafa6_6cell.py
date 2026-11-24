#!/usr/bin/env python3
"""Evaluate NetGO 4.0 (GOCurator) + all 3 frontier reasoning models (`gpt-6.1-sol`,
`gpt-6.2-sol`, `gpt-6.2-astra`) on Task 4 v3.0 (CAFA-6 2x3: Pre/Post x NK/LK/PK, N=120)
using exact CAFA-6 rules:
  - Conditional Information Accretion IA(v) = -log2((c(v)+1)/(c(parents(v))+1))
  - NK: full T1[aspect]
  - LK: newly gained aspect(s) only (`eval_aspects`)
  - PK: dual `-known` masking removing T0[aspect] from BOTH ground truth and DAG-propagated predictions
  - `full` mode protein-centric wF_max per aspect and macro-averaged across MF, BP, CC.

Writes `outputs/results/t4_cafa6_6cell_results.json` and updates `outputs/results/baseline_t4_netgo4.json`.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import sys
from typing import Any, Dict, List

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl_proteinbench")

import numpy as np

REPO = Path(__file__).resolve().parents[1]
if str(REPO / "src") not in sys.path:
    sys.path.insert(0, str(REPO / "src"))

from fable_bench.stage2b_cafa import (
    CAFATarget,
    compute_cafa_fmax,
    compute_term_information_accretion,
    parse_cafa_response,
    parse_go_obo,
)

CELLS = ("pre_nk", "pre_lk", "pre_pk", "post_nk", "post_lk", "post_pk")
MODELS = [
    ("baseline_t4", "NetGO 4.0", True),
    ("gpt-6.1-sol_think4096", "GPT-6.1 Sol", False),
    ("gpt-6.2-sol_think4096", "GPT-6.2 Sol", False),
    ("gpt-6.2-astra_think4096", "GPT-6.2 Astra", False),
]


def load_netgo_preds(raw_file: Path, valid_ids: set[str]) -> Dict[str, Dict[str, float]]:
    pred_map: Dict[str, Dict[str, float]] = {tid: {} for tid in valid_ids}
    for ln in raw_file.read_text(encoding="utf-8").splitlines():
        parts = ln.strip().split("\t")
        if len(parts) < 3:
            continue
        tid, go_id, sc_str = parts[0].strip(), parts[1].strip(), parts[2].strip()
        if tid not in valid_ids or not go_id.startswith("GO:"):
            continue
        try:
            sc = float(sc_str)
        except ValueError:
            continue
        if sc > pred_map[tid].get(go_id, 0.0):
            pred_map[tid][go_id] = sc
    return pred_map


def load_llm_preds(jsonl_path: Path, valid_ids: set[str]) -> Dict[str, Dict[str, float]]:
    pred_map: Dict[str, Dict[str, float]] = {tid: {} for tid in valid_ids}
    for ln in jsonl_path.read_text(encoding="utf-8").splitlines():
        if not ln.strip():
            continue
        row = json.loads(ln)
        tid = str(row.get("target_id") or "")
        if tid not in valid_ids:
            continue
        parsed, _ok = parse_cafa_response(str(row.get("text") or ""))
        pred_map[tid] = parsed
    return pred_map


def eval_model_on_targets(
    dag: Any,
    ia_map: Dict[str, float],
    cafa_targets: List[CAFATarget],
    tier_by_tid: Dict[str, str],
    pred_map: Dict[str, Dict[str, float]],
) -> Dict[str, Any]:
    # Overall (all 120 targets) across MF, BP, CC
    asp_all = {
        a: compute_cafa_fmax(dag, cafa_targets, pred_map, aspect=a, ic_map=ia_map, exclude_known=True)
        for a in ("MF", "BP", "CC")
    }
    macro_all = round(float(np.mean([asp_all[a]["wfmax"] for a in ("MF", "BP", "CC")])), 4)

    # Pre-cutoff vs Post-cutoff overall
    by_window: Dict[str, Any] = {}
    for w in ("pre", "post"):
        sub = [t for t in cafa_targets if tier_by_tid[t.target_id].startswith(w + "_")]
        w_asp = {a: compute_cafa_fmax(dag, sub, pred_map, aspect=a, ic_map=ia_map, exclude_known=True)["wfmax"] for a in ("MF", "BP", "CC")}
        by_window[w] = {
            "n": len(sub),
            "macro_wfmax": round(float(np.mean(list(w_asp.values()))), 4),
            "aspects": {a: round(float(v), 4) for a, v in w_asp.items()},
        }

    # Per 2x3 cell (pre_nk .. post_pk)
    by_cell: Dict[str, Any] = {}
    for ck in CELLS:
        sub = [t for t in cafa_targets if tier_by_tid[t.target_id] == ck]
        c_asp = {a: compute_cafa_fmax(dag, sub, pred_map, aspect=a, ic_map=ia_map, exclude_known=True) for a in ("MF", "BP", "CC")}
        by_cell[ck] = {
            "n": len(sub),
            "macro_wfmax": round(float(np.mean([c_asp[a]["wfmax"] for a in ("MF", "BP", "CC")])), 4),
            "macro_fmax": round(float(np.mean([c_asp[a]["fmax"] for a in ("MF", "BP", "CC")])), 4),
            "aspects": {a: round(float(c_asp[a]["wfmax"]), 4) for a in ("MF", "BP", "CC")},
            "n_by_aspect": {a: int(c_asp[a]["n_targets"]) for a in ("MF", "BP", "CC")},
        }

    # Per-target scores (respecting eval_aspects and dual -known masking!)
    per_target: Dict[str, Dict[str, Any]] = {}
    for t in cafa_targets:
        t_wfs = []
        row_t: Dict[str, Any] = {}
        for a in ("MF", "BP", "CC"):
            res = compute_cafa_fmax(dag, [t], pred_map, aspect=a, ic_map=ia_map, exclude_known=True)
            if res["n_targets"] > 0:
                row_t[f"wfmax_{a.lower()}"] = round(float(res["wfmax"]), 4)
                t_wfs.append(float(res["wfmax"]))
        row_t["wfmax"] = round(float(np.mean(t_wfs)) if t_wfs else 0.0, 4)
        per_target[t.target_id] = row_t

    return {
        "macro_wfmax": macro_all,
        "aspects": {a: round(float(asp_all[a]["wfmax"]), 4) for a in ("MF", "BP", "CC")},
        "by_window": by_window,
        "by_cell": by_cell,
        "per_target": per_target,
    }


def main() -> None:
    s2b_dir = REPO / "data" / "stage2b_cafa"
    pred_dir = REPO / "outputs" / "predictions"
    res_dir = REPO / "outputs" / "results"

    dag = parse_go_obo(str(s2b_dir / "go_subdag.obo"))
    ref_bank = json.loads((s2b_dir / "reference_swissprot_go.json").read_text(encoding="utf-8"))
    ia_map = compute_term_information_accretion(dag, [dag.propagate_set(r["go_terms"]) for r in ref_bank])

    m2b = json.loads((s2b_dir / "manifest.json").read_text(encoding="utf-8"))["targets"]
    cafa_targets: List[CAFATarget] = []
    tier_by_tid: Dict[str, str] = {}
    for t in m2b:
        tier_by_tid[t["target_id"]] = t["temporal_tier"]
        cafa_targets.append(
            CAFATarget(
                target_id=t["target_id"],
                sequence=t["sequence"],
                length=int(t["length"]),
                true_terms_mf=t["true_terms_mf"],
                true_terms_bp=t["true_terms_bp"],
                true_terms_cc=t["true_terms_cc"],
                knowledge_type=t["knowledge_type"],
                known_terms_mf=t.get("known_terms_mf", []),
                known_terms_bp=t.get("known_terms_bp", []),
                known_terms_cc=t.get("known_terms_cc", []),
                eval_aspects=t.get("eval_aspects"),
            )
        )
    valid_ids = {t.target_id for t in cafa_targets}

    all_results: Dict[str, Any] = {
        "benchmark": "stage2b_cafa6_2x3",
        "n_targets": len(cafa_targets),
        "cells": list(CELLS),
        "models": {},
    }

    for key, label, is_base in MODELS:
        if is_base:
            pm = load_netgo_preds(pred_dir / "netgo_raw" / "ng4.0_results.txt", valid_ids)
        else:
            pm = load_llm_preds(pred_dir / f"{key}_stage2b.jsonl", valid_ids)
        cov = sum(1 for v in pm.values() if v)
        res = eval_model_on_targets(dag, ia_map, cafa_targets, tier_by_tid, pm)
        res["display_name"] = label
        res["is_baseline"] = is_base
        res["covered"] = cov
        all_results["models"][key] = res

    (res_dir / "t4_cafa6_6cell_results.json").write_text(json.dumps(all_results, indent=2), encoding="utf-8")
    # Also refresh baseline_t4_netgo4.json for downstream compatibility
    netgo_res = dict(all_results["models"]["baseline_t4"])
    netgo_res["model"] = "baseline_netgo4"
    (res_dir / "baseline_t4_netgo4.json").write_text(json.dumps(netgo_res, indent=2), encoding="utf-8")

    print("\n=== TASK 4 v3.0: CAFA-6 2x3 IA-WEIGHTED wF_max (N=120) ===", flush=True)
    header = f"{'Model':<16s} | {'All':>5s} | {'Pre-NK':>6s} {'Pre-LK':>6s} {'Pre-PK':>6s} | {'Post-NK':>7s} {'Post-LK':>7s} {'Post-PK':>7s} | {'MF':>5s} {'BP':>5s} {'CC':>5s}"
    print(header, flush=True)
    print("-" * len(header), flush=True)
    for key, label, _ in MODELS:
        r = all_results["models"][key]
        bc = r["by_cell"]
        ap = r["aspects"]
        print(
            f"{label:<16s} | {r['macro_wfmax']:5.3f} | "
            f"{bc['pre_nk']['macro_wfmax']:6.3f} {bc['pre_lk']['macro_wfmax']:6.3f} {bc['pre_pk']['macro_wfmax']:6.3f} | "
            f"{bc['post_nk']['macro_wfmax']:7.3f} {bc['post_lk']['macro_wfmax']:7.3f} {bc['post_pk']['macro_wfmax']:7.3f} | "
            f"{ap['MF']:5.3f} {ap['BP']:5.3f} {ap['CC']:5.3f}",
            flush=True,
        )


if __name__ == "__main__":
    main()
