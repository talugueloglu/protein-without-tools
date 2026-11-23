#!/usr/bin/env python3
"""Grade official NetGO 4.0 (GOCurator, CAFA 5 1st-place) predictions on Task 4 (N=60)
with our exact IC-weighted CAFA wF_max evaluator, saving outputs/results/baseline_t4_netgo4.json."""

from __future__ import annotations

import json
import os
from pathlib import Path
import sys
from typing import Any, Dict

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl_proteinbench")

import numpy as np

REPO = Path(__file__).resolve().parents[1]
if str(REPO / "src") not in sys.path:
    sys.path.insert(0, str(REPO / "src"))

from fable_bench.stage2b_cafa import (
    CAFATarget,
    compute_cafa_fmax,
    compute_term_information_content,
    parse_go_obo,
)


def main() -> None:
    s2b_dir = REPO / "data" / "stage2b_cafa"
    raw_file = REPO / "outputs" / "predictions" / "netgo_raw" / "ng4.0_results.txt"
    assert raw_file.exists(), f"Missing NetGO 4.0 output: {raw_file}"

    dag = parse_go_obo(str(s2b_dir / "go_subdag.obo"))
    ref_bank = json.loads((s2b_dir / "reference_swissprot_go.json").read_text(encoding="utf-8"))
    ic_map = compute_term_information_content(dag, [dag.propagate_set(r["go_terms"]) for r in ref_bank])

    m2b = json.loads((s2b_dir / "manifest.json").read_text(encoding="utf-8"))["targets"]
    cafa_targets = [
        CAFATarget(
            target_id=t["target_id"],
            sequence=t["sequence"],
            length=int(t.get("length") or len(t["sequence"])),
            true_terms_mf=t["true_terms_mf"],
            true_terms_bp=t["true_terms_bp"],
            true_terms_cc=t["true_terms_cc"],
            knowledge_type=t.get("knowledge_type", "NK"),
        )
        for t in m2b
    ]
    valid_ids = {t.target_id for t in cafa_targets}

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

    covered = sum(1 for v in pred_map.values() if v)
    print(f"[NetGO 4.0] Parsed predictions for {covered}/{len(cafa_targets)} targets", flush=True)

    asp_res = {asp: compute_cafa_fmax(dag, cafa_targets, pred_map, aspect=asp, ic_map=ic_map) for asp in ("MF", "BP", "CC")}
    macro_wf = round(float(np.mean([asp_res[a]["wfmax"] for a in ("MF", "BP", "CC")])), 4)

    # Also compute split by post_may_2026 (true post-cutoff 7 targets vs all 11 vs early_2026)
    per_t: Dict[str, Dict[str, Any]] = {}
    for t in cafa_targets:
        t_wfs = []
        row_t: Dict[str, Any] = {}
        for asp, terms in (("MF", t.true_terms_mf), ("BP", t.true_terms_bp), ("CC", t.true_terms_cc)):
            if terms:
                v = compute_cafa_fmax(dag, [t], pred_map, aspect=asp, ic_map=ic_map)["wfmax"]
                row_t[f"wfmax_{asp.lower()}"] = round(float(v), 4)
                t_wfs.append(float(v))
        row_t["wfmax"] = round(float(np.mean(t_wfs)) if t_wfs else 0.0, 4)
        per_t[t.target_id] = row_t

    out = {
        "model": "baseline_netgo4",
        "display_name": "NetGO 4.0",
        "macro_wfmax": macro_wf,
        "aspects": {a: round(float(asp_res[a]["wfmax"]), 4) for a in ("MF", "BP", "CC")},
        "aspect_details": {
            a: {k: (round(float(v), 4) if isinstance(v, (int, float)) else v) for k, v in asp_res[a].items()}
            for a in ("MF", "BP", "CC")
        },
        "per_target": per_t,
    }
    out_p = REPO / "outputs" / "results" / "baseline_t4_netgo4.json"
    out_p.write_text(json.dumps(out, indent=2), encoding="utf-8")

    # Check true post-May-2026 7 targets vs pre-May-2026 53 targets
    post7 = [t for t in m2b if str(t.get("qgo_date", "")) > "20260501"]
    post7_ids = {t["target_id"] for t in post7}
    post7_targets = [t for t in cafa_targets if t.target_id in post7_ids]
    post7_asp = {a: compute_cafa_fmax(dag, post7_targets, pred_map, aspect=a, ic_map=ic_map)["wfmax"] for a in ("MF", "BP", "CC")}

    print("\n=== TASK 4 SOTA: Official NetGO 4.0 (GOCurator) ===", flush=True)
    print(f"  Overall (N=60)   : Macro wF_max = {macro_wf:.3f} | Aspects = {out['aspects']}", flush=True)
    print(
        f"  Post-May-2026 (7): Macro wF_max = {np.mean(list(post7_asp.values())):.3f} | "
        f"Aspects = { {k: round(v, 3) for k, v in post7_asp.items()} }",
        flush=True,
    )


if __name__ == "__main__":
    main()
