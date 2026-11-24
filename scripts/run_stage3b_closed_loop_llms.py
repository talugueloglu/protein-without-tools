#!/usr/bin/env python3
"""Agentic Closed-Loop & Open-Loop (`Pass@K`) Epitope Binder Design Runner for Frontier LLMs.

Evaluates `gpt-6.2-astra`, `claude-opus-5-5`, `gpt-5.2`, `gemini-3.1-pro` on Stage 3B Part 2
(`N = 8` epitopes x `K = 16` shots) using either:
  - `--oracle modal_af2`: Live `ColabDesign` `AlphaFold2` (`ipTM`, `iPAE`, `monomer_RMSD`, hotspots) on Modal `L40S`
  - `--oracle surrogate`: Offline CPU surrogate oracle for $0 smoke-testing of prompts & parsers

Supports both:
  - `--mode closed_loop`: Full turn-by-turn `AF2` oracle feedback (`Shots-to-Hit` optimization)
  - `--mode open_loop`: Independent blind proposals (`Pass@16` baseline without `AF2` feedback)
"""

from __future__ import annotations

import argparse
import asyncio
from dataclasses import asdict
import json
from pathlib import Path
import sys
from typing import Any, Dict, List

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from protein_without_tools.llm_client import UnifiedLLMRunner
from protein_without_tools.stage3b_design import (
    BinderShotFeedback,
    EpitopeTarget,
    SurrogatePhysicsBinderOracle,
    build_binder_initial_prompt,
    build_binder_refinement_prompt,
    evaluate_binder_campaign,
    parse_designed_sequence,
    score_3d_complex_geometry,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Stage 3B Closed-Loop / Open-Loop Epitope Binder LLM Campaigns.")
    parser.add_argument("--provider", required=True, choices=["anthropic", "openai", "gemini", "openrouter"])
    parser.add_argument("--model", required=True, help="e.g. gpt-6.2-astra, claude-opus-5-5, gpt-5.2")
    parser.add_argument("--mode", default="closed_loop", choices=["closed_loop", "open_loop"])
    parser.add_argument("--oracle", default="modal_af2", choices=["modal_af2", "surrogate"])
    parser.add_argument("--shots", type=int, default=16, help="AF2 oracle budget K per epitope")
    parser.add_argument("--thinking-budget", type=int, default=0)
    parser.add_argument("--pilot", type=int, default=None, help="Optional limit N epitopes")
    parser.add_argument("--max-usd", type=float, default=30.0)
    args = parser.parse_args()

    sdir = REPO_ROOT / "data" / "stage3b_design"
    pdir = REPO_ROOT / "outputs" / "predictions"
    pdir.mkdir(parents=True, exist_ok=True)

    m2 = json.loads((sdir / "epitope_binder_manifest.json").read_text(encoding="utf-8"))
    targets = [EpitopeTarget(**e) for e in m2["targets"]]
    if args.pilot is not None:
        targets = targets[: args.pilot]

    runner = UnifiedLLMRunner(max_usd_budget=args.max_usd, assert_no_fallback=True)
    surrogate = SurrogatePhysicsBinderOracle()

    af2_fn = None
    if args.oracle == "modal_af2":
        import modal
        af2_fn = modal.Function.from_name("protein-without-tools-stage3b-af2", "evaluate_binder_batch_af2_gpu")

    async def _run_campaign() -> Dict[str, Any]:
        camps = []
        raw_trajectories: Dict[str, List[Dict[str, Any]]] = {}

        for ep in targets:
            pdb_txt = (sdir / "target_pdbs" / f"{ep.target_id}.pdb").read_text(encoding="utf-8")
            history: List[BinderShotFeedback] = []
            ep_logs: List[Dict[str, Any]] = []
            print(f"\n=== Target {ep.target_id} ({ep.tier}, L={len(ep.target_sequence)} AA, mode={args.mode}) ===")

            for shot in range(1, args.shots + 1):
                if args.mode == "closed_loop" and history:
                    prompt = build_binder_refinement_prompt(ep, history, total_budget=args.shots)
                    temp = 0.2
                else:
                    prompt = build_binder_initial_prompt(ep)
                    # Open-loop needs non-zero sampling temperature after Shot 1 for diverse Pass@K proposals
                    temp = 0.0 if shot == 1 else 0.7

                rec = await runner.complete(
                    provider=args.provider,
                    model=args.model,
                    prompt=prompt,
                    max_tokens=1536 + (args.thinking_budget if args.thinking_budget > 0 else 0),
                    thinking_budget=args.thinking_budget,
                    temperature=temp,
                )
                b_seq, rat, _ok = parse_designed_sequence(rec.text)

                if args.oracle == "surrogate" or not b_seq:
                    fb = surrogate.evaluate_binder(ep, b_seq, round_idx=shot)
                else:
                    ev = af2_fn.remote(pdb_txt, len(ep.target_sequence), ep.epitope_hotspots_1idx, [b_seq])[0]
                    if not ev.get("valid"):
                        fb = surrogate.evaluate_binder(ep, "", round_idx=shot)
                    else:
                        fb = score_3d_complex_geometry(
                            target=ep,
                            binder_seq=b_seq,
                            pred_target_cb=np.array(ev["pred_target_cb"]),
                            pred_binder_ca=np.array(ev["pred_binder_ca"]),
                            pred_binder_cb=np.array(ev["pred_binder_cb"]),
                            iptm=ev["iptm"],
                            ipae=ev["ipae"],
                            binder_plddt=ev["binder_plddt"],
                            monomer_rmsd=ev["monomer_rmsd"],
                            round_idx=shot,
                        )

                history.append(fb)
                ep_logs.append({"round": shot, "rationale": rat, "feedback": asdict(fb), "llm_meta": rec.to_dict()})
                hit_str = "HIT!" if fb.is_hit else "miss"
                print(
                    f"  Shot {shot:2d}/{args.shots}: [{hit_str}] ipTM={fb.iptm:.2f}, iPAE={fb.ipae:.1f}A, "
                    f"hotspots={len(fb.hotspots_contacted_1idx)}/5, score={fb.composite_score:.3f}"
                )

            camps.append(evaluate_binder_campaign(ep.target_id, history, budget_k=args.shots))
            raw_trajectories[ep.target_id] = ep_logs

        hits = [c["hit_achieved"] for c in camps]
        shots_hit = [c["shots_to_first_hit"] for c in camps if c["shots_to_first_hit"] is not None]
        safe_m = args.model.replace("/", "_").replace(":", "_")
        if args.thinking_budget > 0:
            safe_m += f"_think{args.thinking_budget}"
        tag = f"{safe_m}_{args.mode}_{args.oracle}"
        payload = {
            "model_tag": tag,
            "summary": {
                "n_targets": len(camps),
                "hit_rate": round(float(np.mean(hits)), 4),
                "mean_shots_to_hit_among_hits": round(float(np.mean(shots_hit)), 2) if shots_hit else None,
                "mean_best_iptm": round(float(np.mean([c["best_iptm_on_epitope"] for c in camps])), 4),
                "mean_closed_loop_gain": round(float(np.mean([c["closed_loop_trajectory_gain"] for c in camps])), 4),
                "pass_at_k_curve": [round(float(x), 4) for x in np.mean([c["pass_at_k_curve"] for c in camps], axis=0)],
                "trajectory_mean": [round(float(x), 4) for x in np.mean([c["running_best_composite"] for c in camps], axis=0)],
                "trajectory_sem": [round(float(x), 4) for x in (np.std([c["running_best_composite"] for c in camps], axis=0) / np.sqrt(len(camps)))],
                "per_target": camps,
            },
            "trajectories": raw_trajectories,
        }
        out_f = pdir / f"stage3b_binder_campaign_{tag}.json"
        out_f.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"\nSaved campaign -> {out_f} (Live spend: ${runner.cumulative_live_usd:.3f})")
        return payload

    asyncio.run(_run_campaign())


if __name__ == "__main__":
    main()
