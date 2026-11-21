#!/usr/bin/env python3
"""Export a single 370-line prompt bundle for Tests 1,3,4,5,6, or import external responses.

Usage:
  1. Export single-file 370-prompt bundle for any external harness / Batch API:
       .venv/bin/python scripts/import_external_predictions.py --export-bundle
     -> writes `outputs/prompts/local_13456_prompts_bundle.jsonl`

  2. Import a single-file external response JSONL (each line having `id`/`target_id` and `text`/`response`/`output`):
       .venv/bin/python scripts/import_external_predictions.py --model my-custom-model --input path/to/responses.jsonl
     -> splits into `outputs/predictions/my-custom-model_{stage}.jsonl` ready for `evaluate_local_13456.py`!
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]

LOCAL_STAGE_FILES = [
    (1, "stage1a", "stage1a_forward_prompts.jsonl"),
    (3, "stage2a", "stage2a_homology_prompts.jsonl"),
    (4, "stage2b", "stage2b_cafa_prompts.jsonl"),
    (5, "stage3a", "stage3a_alanine_prompts.jsonl"),
    (6, "stage6_binding", "stage6_binding_alanine_prompts.jsonl"),
]


def export_bundle(out_path: Path) -> None:
    pdir = REPO / "outputs" / "prompts"
    rows = []
    for test_num, stage_sfx, fn in LOCAL_STAGE_FILES:
        for ln in (pdir / fn).read_text(encoding="utf-8").splitlines():
            if not ln.strip():
                continue
            obj = json.loads(ln)
            tid = str(obj.get("target_id") or obj.get("pair_id"))
            rows.append(
                {
                    "id": f"{stage_sfx}__{tid}",
                    "test_num": test_num,
                    "stage": stage_sfx,
                    "target_id": tid,
                    "prompt": obj["prompt"],
                }
            )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    print(f"Exported {len(rows)} prompts -> {out_path} ({out_path.stat().st_size / 1024.0:.1f} KB)")


def import_responses(model_name: str, in_path: Path) -> None:
    safe_tag = model_name.replace("/", "_").replace(":", "_").replace(" ", "_")
    by_stage: dict[str, list[dict]] = defaultdict(list)

    # Map bare `target_id` -> `stage_sfx` in case external output kept `target_id` without `stage__` prefix
    pdir = REPO / "outputs" / "prompts"
    tid_to_stage: dict[str, str] = {}
    for _tnum, stage_sfx, fn in LOCAL_STAGE_FILES:
        for ln in (pdir / fn).read_text(encoding="utf-8").splitlines():
            if ln.strip():
                o = json.loads(ln)
                tid = str(o.get("target_id") or o.get("pair_id"))
                tid_to_stage[tid] = stage_sfx

    for ln in in_path.read_text(encoding="utf-8").splitlines():
        if not ln.strip():
            continue
        obj = json.loads(ln)
        raw_id = str(obj.get("id") or obj.get("custom_id") or obj.get("target_id") or obj.get("pair_id") or "")
        if "__" in raw_id:
            stage_sfx, tid = raw_id.split("__", 1)
        else:
            tid = raw_id
            stage_sfx = str(obj.get("stage") or tid_to_stage.get(tid, ""))
        if not stage_sfx:
            raise ValueError(f"Could not map id='{raw_id}' to a stage suffix.")
        txt = str(obj.get("text") or obj.get("response") or obj.get("output") or obj.get("completion") or "")
        rec = {"target_id": tid, "pair_id": tid, "requested_model": model_name, "returned_model": model_name, "text": txt}
        by_stage[stage_sfx].append(rec)

    odir = REPO / "outputs" / "predictions"
    odir.mkdir(parents=True, exist_ok=True)
    for stage_sfx, items in sorted(by_stage.items()):
        out_f = odir / f"{safe_tag}_{stage_sfx}.jsonl"
        out_f.write_text("\n".join(json.dumps(x) for x in items) + "\n", encoding="utf-8")
        print(f"Wrote {len(items):3d} predictions -> {out_f}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--export-bundle", action="store_true", help="Write single 370-prompt JSONL bundle")
    parser.add_argument("--bundle-out", default=str(REPO / "outputs/prompts/local_13456_prompts_bundle.jsonl"))
    parser.add_argument("--model", help="Model tag when importing external JSONL responses")
    parser.add_argument("--input", help="Path to external JSONL responses file")
    args = parser.parse_args()

    if args.export_bundle:
        export_bundle(Path(args.bundle_out))
    elif args.model and args.input:
        import_responses(args.model, Path(args.input))
    else:
        parser.error("Pass either `--export-bundle` or `--model <name> --input <responses.jsonl>`.")


if __name__ == "__main__":
    main()
