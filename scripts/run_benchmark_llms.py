#!/usr/bin/env python3
"""Unified Frontier LLM Benchmark Runner across Stages 1A, 1B, 2A, 2B, and 3A (+ 3B).

Requires API keys in `.env` ONLY when invoked for live execution.
Supports `--stages 1a,1b,2a,2b,3a` (or `all`) and `--pilot N` for $1 smoke checks.
"""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path
import sys
from typing import Any, Dict, List, Tuple

from fable_bench.llm_client import UnifiedLLMRunner

STAGE_SPECS: Dict[str, Tuple[str, str, int]] = {
    "1a": ("stage1a_forward_prompts.jsonl", "stage1a", 4096),
    "1b": ("stage1b_inverse_prompts.jsonl", "stage1b", 2048),
    "2a": ("stage2a_homology_prompts.jsonl", "stage2a", 1024),
    "2b": ("stage2b_cafa_prompts.jsonl", "stage2b", 2048),
    "3a": ("stage3a_alanine_prompts.jsonl", "stage3a", 4096),
    "3b": ("stage3b_denovo_prompts.jsonl", "stage3b", 2048),
}


async def run_stage_jsonl(
    runner: UnifiedLLMRunner,
    prompt_jsonl: Path,
    out_jsonl: Path,
    provider: str,
    model: str,
    thinking_budget: int,
    max_tokens: int,
    concurrency: int,
    limit: int | None,
) -> None:
    if not prompt_jsonl.exists():
        print(f"Skipping {prompt_jsonl.name} (not built yet).")
        return

    rows = [
        json.loads(line)
        for line in prompt_jsonl.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if limit is not None:
        rows = rows[:limit]

    sem = asyncio.Semaphore(concurrency)
    results: List[Dict[str, Any]] = [{}] * len(rows)

    async def _worker(idx: int, item: Dict[str, Any]) -> None:
        async with sem:
            rec = await runner.complete(
                provider=provider,
                model=model,
                prompt=item["prompt"],
                max_tokens=max_tokens,
                thinking_budget=thinking_budget,
                temperature=0.0,
            )
            tid = item.get("target_id") or item.get("pair_id") or f"item_{idx}"
            out_row: Dict[str, Any] = {
                "target_id": tid,
                "task": item.get("task", ""),
                "length": item.get("length", 0),
                **rec.to_dict(),
            }
            if "pair_id" in item:
                out_row["pair_id"] = item["pair_id"]
            results[idx] = out_row
            tag = "CACHE" if rec.cached else f"${rec.est_cost_usd:.4f}"
            print(
                f"  [{idx + 1:3d}/{len(rows):3d}] {tid} ({out_row['task']}) "
                f"-> {rec.returned_model} [{tag}, {rec.latency_sec:.1f}s]"
            )

    await asyncio.gather(*(_worker(i, r) for i, r in enumerate(rows)))
    out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    out_jsonl.write_text("\n".join(json.dumps(r) for r in results) + "\n", encoding="utf-8")
    print(f"Saved {len(results)} responses -> {out_jsonl} (Cumulative spend: ${runner.cumulative_live_usd:.3f})")


def main() -> None:
    parser = argparse.ArgumentParser(description="Execute Frontier LLM protein benchmark prompts across stages.")
    parser.add_argument("--provider", required=True, choices=["anthropic", "openai", "gemini", "openrouter"])
    parser.add_argument("--model", required=True, help="e.g. claude-fable-5-2, claude-opus-5-5, gpt-5.2, gemini-3.1-pro")
    parser.add_argument("--stages", default="all", help="Comma-separated stages: 1a,1b,2a,2b,3a,3b or 'all'")
    parser.add_argument("--thinking-budget", type=int, default=0, help="Extended thinking token budget (0 = Direct)")
    parser.add_argument("--prompts-dir", default="outputs/prompts")
    parser.add_argument("--out-dir", default="outputs/predictions")
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--pilot", type=int, default=None, help="Optional pilot limit N items per stage")
    parser.add_argument("--max-usd", type=float, default=35.0)
    args = parser.parse_args()

    if args.stages.strip().lower() == "all":
        stage_keys = ["1a", "1b", "2a", "2b", "3a", "3b"]
    else:
        stage_keys = [s.strip().lower() for s in args.stages.split(",") if s.strip()]

    safe_tag = args.model.replace("/", "_").replace(":", "_")
    if args.thinking_budget > 0:
        safe_tag += f"_think{args.thinking_budget}"

    runner = UnifiedLLMRunner(max_usd_budget=args.max_usd, assert_no_fallback=True)
    pdir = Path(args.prompts_dir)
    odir = Path(args.out_dir)

    async def _run_selected() -> None:
        for sk in stage_keys:
            if sk not in STAGE_SPECS:
                raise ValueError(f"Unknown stage '{sk}'. Valid: {list(STAGE_SPECS)}")
            prompt_fn, out_suffix, base_max_toks = STAGE_SPECS[sk]
            eff_max_toks = base_max_toks + (args.thinking_budget if args.thinking_budget > 0 else 0)
            await run_stage_jsonl(
                runner=runner,
                prompt_jsonl=pdir / prompt_fn,
                out_jsonl=odir / f"{safe_tag}_{out_suffix}.jsonl",
                provider=args.provider,
                model=args.model,
                thinking_budget=args.thinking_budget,
                max_tokens=eff_max_toks,
                concurrency=args.concurrency,
                limit=args.pilot,
            )

    asyncio.run(_run_selected())


if __name__ == "__main__":
    sys.exit(main())
