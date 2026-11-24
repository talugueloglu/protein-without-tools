#!/usr/bin/env python3
"""Run Stage 1A (Forward Folding) & Stage 1B (Inverse Folding) across Frontier LLM APIs.

Requires API keys in `.env` ONLY when invoked for live execution.
Supports `--pilot N` (e.g., `--pilot 4`) to run a tiny $1 sanity check before full runs.
"""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path
import sys
from typing import Any, Dict, List

from protein_without_tools.llm_client import UnifiedLLMRunner


async def run_prompt_jsonl(
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
            results[idx] = {
                "target_id": item["target_id"],
                "task": item["task"],
                "length": item["length"],
                **rec.to_dict(),
            }
            tag = "CACHE" if rec.cached else f"${rec.est_cost_usd:.4f}"
            print(
                f"  [{idx + 1:3d}/{len(rows):3d}] {item['target_id']} ({item['task']}) "
                f"-> {rec.returned_model} [{tag}, {rec.latency_sec:.1f}s]"
            )

    await asyncio.gather(*(_worker(i, r) for i, r in enumerate(rows)))
    out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    out_jsonl.write_text("\n".join(json.dumps(r) for r in results) + "\n", encoding="utf-8")
    print(f"Saved {len(results)} responses -> {out_jsonl} (Live spend: ${runner.cumulative_live_usd:.3f})")


def main() -> None:
    parser = argparse.ArgumentParser(description="Execute Stage 1 LLM benchmark prompts.")
    parser.add_argument("--provider", required=True, choices=["anthropic", "openai", "gemini", "openrouter"])
    parser.add_argument("--model", required=True, help="e.g. gpt-6.2-astra, claude-opus-5-5, gpt-5.2")
    parser.add_argument("--thinking-budget", type=int, default=0, help="Extended thinking token budget (0 = Direct)")
    parser.add_argument("--tasks", default="both", choices=["forward", "inverse", "both"])
    parser.add_argument("--prompts-dir", default="outputs/prompts")
    parser.add_argument("--out-dir", default="outputs/predictions")
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--pilot", type=int, default=None, help="Optional pilot limit N targets")
    parser.add_argument("--max-usd", type=float, default=25.0)
    args = parser.parse_args()

    safe_tag = args.model.replace("/", "_").replace(":", "_")
    if args.thinking_budget > 0:
        safe_tag += f"_think{args.thinking_budget}"

    runner = UnifiedLLMRunner(max_usd_budget=args.max_usd, assert_no_fallback=True)
    pdir = Path(args.prompts_dir)
    odir = Path(args.out_dir)

    async def _run_all() -> None:
        if args.tasks in ("forward", "both"):
            await run_prompt_jsonl(
                runner=runner,
                prompt_jsonl=pdir / "stage1a_forward_prompts.jsonl",
                out_jsonl=odir / f"{safe_tag}_stage1a.jsonl",
                provider=args.provider,
                model=args.model,
                thinking_budget=args.thinking_budget,
                max_tokens=4096,
                concurrency=args.concurrency,
                limit=args.pilot,
            )
        if args.tasks in ("inverse", "both"):
            await run_prompt_jsonl(
                runner=runner,
                prompt_jsonl=pdir / "stage1b_inverse_prompts.jsonl",
                out_jsonl=odir / f"{safe_tag}_stage1b.jsonl",
                provider=args.provider,
                model=args.model,
                thinking_budget=args.thinking_budget,
                max_tokens=2048,
                concurrency=args.concurrency,
                limit=args.pilot,
            )

    asyncio.run(_run_all())


if __name__ == "__main__":
    sys.exit(main())
