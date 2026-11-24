#!/usr/bin/env python3
"""Import a manually pasted web-chat response (e.g. from Claude.ai or ChatGPT UI)
into `outputs/predictions/` so `evaluate_stage1.py`, `plot_stage1.py`, and
`generate_stage1_viewer.py` can grade and visualize it with ZERO API keys!

Example usage:
  1. Copy prompt from `outputs/prompts/web_chat_spotcheck/s1_t011_forward.txt`
  2. Paste into a frontier chat UI and save its reply to `my_reply.txt`
  3. Run:
     .venv/bin/python scripts/import_manual_response.py \
         --model gpt_6_2_astra_webui \
         --target-id s1_t011 \
         --task forward \
         --response-file my_reply.txt
  4. Grade & view in 3D:
     .venv/bin/python scripts/evaluate_stage1.py --run-calibration
     .venv/bin/python scripts/generate_stage1_viewer.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


def main() -> None:
    parser = argparse.ArgumentParser(description="Import manual web-chat reply into Stage 1 predictions.")
    parser.add_argument("--model", required=True, help="Model tag, e.g. gpt_6_2_astra_webui")
    parser.add_argument("--target-id", required=True, help="Anonymized target ID, e.g. s1_t001")
    parser.add_argument("--task", required=True, choices=["forward", "inverse"])
    parser.add_argument("--response-file", required=True, help="Path to text/JSON file containing model reply")
    parser.add_argument("--data-dir", default="data/stage1_benchmark")
    parser.add_argument("--pred-dir", default="outputs/predictions")
    args = parser.parse_args()

    manifest = json.loads((Path(args.data_dir) / "manifest.json").read_text(encoding="utf-8"))
    tmap = {t["target_id"]: t for t in manifest["targets"]}
    if args.target_id not in tmap:
        raise ValueError(f"Unknown target_id '{args.target_id}'. Valid IDs: {list(tmap)[:8]}...")

    raw_reply = Path(args.response_file).read_text(encoding="utf-8")
    suffix = "stage1a" if args.task == "forward" else "stage1b"
    task_full = "stage1a_forward_folding" if args.task == "forward" else "stage1b_inverse_folding"
    out_jsonl = Path(args.pred_dir) / f"{args.model}_{suffix}.jsonl"
    out_jsonl.parent.mkdir(parents=True, exist_ok=True)

    existing = []
    if out_jsonl.exists():
        existing = [
            json.loads(l)
            for l in out_jsonl.read_text(encoding="utf-8").splitlines()
            if l.strip() and json.loads(l).get("target_id") != args.target_id
        ]

    existing.append(
        {
            "target_id": args.target_id,
            "task": task_full,
            "length": tmap[args.target_id]["length"],
            "provider": "manual_webui",
            "requested_model": args.model,
            "returned_model": args.model,
            "text": raw_reply,
        }
    )
    out_jsonl.write_text("\n".join(json.dumps(x) for x in existing) + "\n", encoding="utf-8")
    print(f"Imported {args.target_id} ({args.task}) -> {out_jsonl}")


if __name__ == "__main__":
    sys.exit(main())
