#!/usr/bin/env python3
"""Grade BindCraft 16-shot trajectory sequences (`outputs/predictions/bindcraft_t8_raw_seqs.json`)
using the IDENTICAL JapanFold `Boltz-2 + AF2-IG + ESMFold-2` oracle used for all 3 reasoning models,
saving `outputs/predictions/stage3b_binder_campaign_specialist_bindcraft_af2.json`."""

from __future__ import annotations

import json
from pathlib import Path
import sys
from typing import Any, Dict, List, Tuple

import numpy as np

REPO = Path(__file__).resolve().parents[1]
for p in (REPO / "src", REPO / "scripts"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from fable_bench.japanfold_client import JapanFoldClient
from fable_bench.stage3b_design import EpitopeTarget, evaluate_binder_campaign
from run_test8_af2_japanfold import evaluate_binders_af2_batch


def main() -> None:
    sdir = REPO / "data" / "stage3b_design"
    pdir = REPO / "outputs" / "predictions"
    raw_p = pdir / "bindcraft_t8_raw_seqs.json"
    assert raw_p.exists(), f"Missing BindCraft raw trajectories: {raw_p}"

    m2 = json.loads((sdir / "epitope_binder_manifest.json").read_text(encoding="utf-8"))["targets"]
    ep_by_id = {e["target_id"]: EpitopeTarget(**e) for e in m2}
    raw_rows = json.loads(raw_p.read_text(encoding="utf-8"))

    eval_items: List[Tuple[EpitopeTarget, str, int]] = []
    per_ep_order: List[Tuple[str, int]] = []
    for row in raw_rows:
        tid = row["target_id"]
        ep = ep_by_id[tid]
        seqs = row["sequences"][:16]
        per_ep_order.append((tid, len(seqs)))
        for r_idx, sq in enumerate(seqs, start=1):
            eval_items.append((ep, sq, r_idx))

    print(f"=== Grading {len(eval_items)} BindCraft trajectory binders on official JapanFold AF2-IG + ESMFold-2 oracle ===", flush=True)
    with JapanFoldClient(repo_root=REPO, max_usd_budget=25.0) as jf:
        bal = jf.get_account_balance()
        print(f"  [JapanFold] Starting balance: ${bal:.4f}", flush=True)
        feedbacks = evaluate_binders_af2_batch(jf, eval_items)
        print(f"  [JapanFold] Total grading cost: ${jf.cumulative_usd:.4f}", flush=True)

    camps: List[Dict[str, Any]] = []
    cursor = 0
    for tid, n_s in per_ep_order:
        hist = feedbacks[cursor : cursor + n_s]
        cursor += n_s
        camps.append(evaluate_binder_campaign(tid, hist, budget_k=16))

    hits = [c["hit_achieved"] for c in camps]
    shots_hit = [c["shots_to_first_hit"] for c in camps if c["shots_to_first_hit"] is not None]
    out_payload = {
        "model_tag": "Specialist_BindCraft_AF2_16shot",
        "summary": {
            "n_targets": len(camps),
            "hit_rate": round(float(np.mean(hits)), 4),
            "mean_shots_to_hit_among_hits": round(float(np.mean(shots_hit)), 2) if shots_hit else None,
            "mean_best_iptm": round(float(np.mean([c["best_iptm_on_epitope"] for c in camps])), 4),
            "mean_closed_loop_gain": round(float(np.mean([c["closed_loop_trajectory_gain"] for c in camps])), 4),
            "pass_at_k_curve": [round(float(x), 4) for x in np.mean([c["pass_at_k_curve"] for c in camps], axis=0)],
            "trajectory_mean": [round(float(x), 4) for x in np.mean([c["running_best_composite"] for c in camps], axis=0)],
            "trajectory_sem": [
                round(float(x), 4)
                for x in (np.std([c["running_best_composite"] for c in camps], axis=0) / np.sqrt(len(camps)))
            ],
            "per_target": camps,
        },
    }
    out_f = pdir / "stage3b_binder_campaign_specialist_bindcraft_af2.json"
    out_f.write_text(json.dumps(out_payload, indent=2), encoding="utf-8")

    s = out_payload["summary"]
    print("\n=== TASK 8 SOTA: Official BindCraft (16-Step Trajectory, JapanFold AF2-IG + ESMFold-2 Oracle) ===", flush=True)
    print(
        f"  Hit@16={s['hit_rate']*100:.1f}% | Mean Shots={s['mean_shots_to_hit_among_hits']} | "
        f"AF2 Composite={s['trajectory_mean'][0]:.3f} -> {s['trajectory_mean'][-1]:.3f} | "
        f"Best On-Epitope ipTM={s['mean_best_iptm']:.3f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
