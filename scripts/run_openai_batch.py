#!/usr/bin/env python3
"""OpenAI 50%-Off Batch API Submitter, Monitor & Collector for Protein Structure Without Tools (Tests 1,3,4,5,6).

Workflow:
  1. Prepare JSONL locally (sandbox-safe):
       .venv/bin/python scripts/run_openai_batch.py --prepare --model gpt-6.2-astra --effort high
  2. Upload & create OpenAI Batch job (network):
       .venv/bin/python scripts/run_openai_batch.py --submit --model gpt-6.2-astra --effort high
  3. Check status or wait & auto-collect into `outputs/predictions/{model}_think4096_{stage}.jsonl`:
       .venv/bin/python scripts/run_openai_batch.py --status --batch-id <batch_id>
       .venv/bin/python scripts/run_openai_batch.py --collect --batch-id <batch_id>
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
import os
from pathlib import Path
import sys
import time
from typing import Any, Dict, List

from dotenv import load_dotenv
import httpx

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from protein_without_tools.llm_client import UnifiedLLMRunner

STAGE_ID_MAP = {
    "1": ("stage1a", "stage1a_forward_prompts.jsonl"),
    "2": ("stage1b", "stage1b_inverse_prompts.jsonl"),
    "3": ("stage2a", "stage2a_homology_prompts.jsonl"),
    "4": ("stage2b", "stage2b_cafa_prompts.jsonl"),
    "5": ("stage3a", "stage3a_alanine_prompts.jsonl"),
    "6": ("stage6_binding", "stage6_binding_alanine_prompts.jsonl"),
    "7": ("stage3b", "stage3b_denovo_prompts.jsonl"),
}

LOCAL_STAGE_FILES = [STAGE_ID_MAP[k] for k in ("1", "3", "4", "5", "6")]
ALL_STAGE_FILES = [STAGE_ID_MAP[k] for k in ("1", "2", "3", "4", "5", "6", "7")]

# 50%-discounted Batch API per-million-token rates (Input, Output)
BATCH_PRICING_USD_PER_MTOK: Dict[str, tuple[float, float]] = {
    "gpt-6.2-astra": (4.0, 20.0),
    "gpt-6-astra": (4.0, 20.0),
    "gpt-6.2-sol": (1.0, 5.0),
    "gpt-6.1-sol": (1.0, 5.0),
    "gpt-5.6-sol": (1.0, 5.0),
}

EFFORT_TO_TAG_BUDGET = {
    "low": 1024,
    "medium": 2048,
    "high": 4096,
    "xhigh": 12000,
}


def get_batch_input_path(model: str, effort: str, stages_key: str = "13456") -> Path:
    bdir = REPO_ROOT / "outputs" / "batches"
    bdir.mkdir(parents=True, exist_ok=True)
    safe = model.replace("/", "_")
    return bdir / f"{safe}_{effort}_{stages_key}_batch_input.jsonl"


def prepare_batch_jsonl(
    model: str,
    effort: str,
    max_completion_tokens: int = 32768,
    skip_cached: bool = False,
    stages: str = "1,3,4,5,6",
) -> Path:
    pdir = REPO_ROOT / "outputs" / "prompts"
    skeys = [s.strip() for s in stages.split(",") if s.strip()]
    stage_files = [STAGE_ID_MAP[k] for k in skeys]
    stages_tag = "".join(skeys)
    out_path = get_batch_input_path(model, effort, stages_tag)
    tb = EFFORT_TO_TAG_BUDGET.get(effort, 4096)
    runner = UnifiedLLMRunner()
    lines: List[str] = []
    skipped = 0
    for stage_sfx, fn in stage_files:
        for ln in (pdir / fn).read_text(encoding="utf-8").splitlines():
            if not ln.strip():
                continue
            obj = json.loads(ln)
            tid = str(obj.get("target_id") or obj.get("pair_id"))
            if skip_cached and runner._cache_path("openai", model, obj["prompt"], tb, 0.0).exists():
                skipped += 1
                continue
            req = {
                "custom_id": f"{stage_sfx}__{tid}",
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": {
                    "model": model,
                    "messages": [{"role": "user", "content": obj["prompt"]}],
                    "max_completion_tokens": max_completion_tokens,
                    "reasoning_effort": effort,
                },
            }
            lines.append(json.dumps(req))

    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(
        f"Prepared {len(lines)} Batch API requests (skipped {skipped} cached) -> "
        f"{out_path} ({out_path.stat().st_size / 1024:.1f} KB)"
    )
    return out_path


def _api_key() -> str:
    load_dotenv(REPO_ROOT / ".env")
    key = os.getenv("OPENAI_API_KEY", "").strip()
    if not key:
        raise EnvironmentError("OPENAI_API_KEY is missing in .env")
    return key


def submit_batch(model: str, effort: str, stages: str = "1,3,4,5,6") -> Dict[str, Any]:
    skeys = [s.strip() for s in stages.split(",") if s.strip()]
    in_path = get_batch_input_path(model, effort, "".join(skeys))
    if not in_path.exists():
        in_path = prepare_batch_jsonl(model, effort, stages=stages)

    key = _api_key()
    headers = {"Authorization": f"Bearer {key}"}

    with httpx.Client(timeout=120.0) as client:
        with in_path.open("rb") as fh:
            f_resp = client.post(
                "https://api.openai.com/v1/files",
                headers=headers,
                data={"purpose": "batch"},
                files={"file": (in_path.name, fh, "application/jsonl")},
            )
        if f_resp.status_code >= 400:
            raise RuntimeError(f"File upload HTTP {f_resp.status_code}: {f_resp.text[:500]}")
        file_obj = f_resp.json()
        file_id = file_obj["id"]
        print(f"Uploaded batch file -> {file_id} ({file_obj.get('bytes', 0)} bytes)")

        b_resp = client.post(
            "https://api.openai.com/v1/batches",
            headers={**headers, "Content-Type": "application/json"},
            json={
                "input_file_id": file_id,
                "endpoint": "/v1/chat/completions",
                "completion_window": "24h",
                "metadata": {
                    "benchmark": "protein_without_tools_local_13456",
                    "model": model,
                    "reasoning_effort": effort,
                },
            },
        )
        if b_resp.status_code >= 400:
            raise RuntimeError(f"Batch create HTTP {b_resp.status_code}: {b_resp.text[:500]}")
        batch_obj = b_resp.json()

    reg_path = REPO_ROOT / "outputs" / "batches" / "active_batches.json"
    existing = json.loads(reg_path.read_text(encoding="utf-8")) if reg_path.exists() else []
    existing.append(
        {
            "batch_id": batch_obj["id"],
            "input_file_id": file_id,
            "model": model,
            "effort": effort,
            "created_at": batch_obj.get("created_at"),
            "status": batch_obj.get("status"),
        }
    )
    reg_path.write_text(json.dumps(existing, indent=2), encoding="utf-8")
    print(f"Submitted OpenAI Batch -> batch_id={batch_obj['id']} (status={batch_obj.get('status')})")
    return batch_obj


def check_batch_status(batch_id: str) -> Dict[str, Any]:
    key = _api_key()
    with httpx.Client(timeout=60.0) as client:
        r = client.get(f"https://api.openai.com/v1/batches/{batch_id}", headers={"Authorization": f"Bearer {key}"})
    if r.status_code >= 400:
        raise RuntimeError(f"Batch status HTTP {r.status_code}: {r.text[:500]}")
    b = r.json()
    rc = b.get("request_counts", {})
    print(
        f"Batch {batch_id} | status={b.get('status')} | "
        f"completed={rc.get('completed', 0)}/{rc.get('total', 0)} (failed={rc.get('failed', 0)}) | "
        f"output_file_id={b.get('output_file_id')} | error_file_id={b.get('error_file_id')}",
        flush=True,
    )
    return b


def collect_batch(batch_id: str, model_override: str | None = None, effort_override: str | None = None) -> None:
    b = check_batch_status(batch_id)
    if b.get("status") != "completed":
        raise RuntimeError(f"Batch {batch_id} is not completed yet (status={b.get('status')}).")
    out_fid = b.get("output_file_id")
    if not out_fid:
        raise RuntimeError(f"Batch {batch_id} has no output_file_id (error_file_id={b.get('error_file_id')}).")

    meta = b.get("metadata") or {}
    model = model_override or meta.get("model") or "gpt-6.2-astra"
    effort = effort_override or meta.get("reasoning_effort") or "high"
    tb = EFFORT_TO_TAG_BUDGET.get(effort, 4096)
    in_rate, out_rate = BATCH_PRICING_USD_PER_MTOK.get(model, (4.0, 20.0))

    key = _api_key()
    with httpx.Client(timeout=180.0) as client:
        r = client.get(f"https://api.openai.com/v1/files/{out_fid}/content", headers={"Authorization": f"Bearer {key}"})
    if r.status_code >= 400:
        raise RuntimeError(f"Download HTTP {r.status_code}: {r.text[:500]}")

    raw_save = REPO_ROOT / "outputs" / "batches" / f"{batch_id}_output.jsonl"
    raw_save.write_text(r.text, encoding="utf-8")

    # Map prompt metadata (task, length) for full compatibility with `evaluate_local_13456.py`
    pdir = REPO_ROOT / "outputs" / "prompts"
    prompt_info: Dict[str, Dict[str, Any]] = {}
    for stage_sfx, fn in ALL_STAGE_FILES:
        for ln in (pdir / fn).read_text(encoding="utf-8").splitlines():
            if ln.strip():
                o = json.loads(ln)
                tid = str(o.get("target_id") or o.get("pair_id"))
                prompt_info[f"{stage_sfx}__{tid}"] = o

    by_stage: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    tot_cost = 0.0
    for ln in r.text.splitlines():
        if not ln.strip():
            continue
        row = json.loads(ln)
        cid = str(row["custom_id"])
        stage_sfx, tid = cid.split("__", 1)
        body = (row.get("response") or {}).get("body") or {}
        choices = body.get("choices") or [{}]
        msg = choices[0].get("message") or {}
        text = msg.get("content") or ""
        stop_reason = str(choices[0].get("finish_reason") or "")
        usage = body.get("usage") or {}
        in_tok = int(usage.get("prompt_tokens", 0))
        out_tok = int(usage.get("completion_tokens", 0))
        reas_tok = int((usage.get("completion_tokens_details") or {}).get("reasoning_tokens", 0))
        returned_model = str(body.get("model") or model)
        cost = (in_tok * in_rate + out_tok * out_rate) / 1_000_000.0
        tot_cost += cost

        pmeta = prompt_info.get(cid, {})
        pred_rec: Dict[str, Any] = {
            "target_id": tid,
            "task": pmeta.get("task", ""),
            "length": pmeta.get("length", 0),
            "provider": "openai_batch",
            "requested_model": model,
            "returned_model": returned_model,
            "thinking_budget": tb,
            "text": text,
            "thinking_text": f"[openai_reasoning_tokens={reas_tok}]" if reas_tok else "",
            "stop_reason": stop_reason,
            "input_tokens": in_tok,
            "output_tokens": out_tok,
            "est_cost_usd": round(cost, 6),
            "latency_sec": 0.0,
            "silent_fallback_detected": False,
            "refusal_detected": bool(msg.get("refusal")),
            "cached": False,
        }
        if "pair_id" in pmeta:
            pred_rec["pair_id"] = pmeta["pair_id"]
        by_stage[stage_sfx].append(pred_rec)

    safe_tag = f"{model.replace('/', '_')}_think{tb}"
    odir = REPO_ROOT / "outputs" / "predictions"
    odir.mkdir(parents=True, exist_ok=True)
    runner = UnifiedLLMRunner()

    for stage_sfx, fn in ALL_STAGE_FILES:
        if stage_sfx not in by_stage:
            continue
        items_by_tid = {x["target_id"]: x for x in by_stage.get(stage_sfx, [])}
        # Fill any previously cached live items so partial/skip-cached batches produce complete stage files
        merged_items: List[Dict[str, Any]] = []
        for ln in (pdir / fn).read_text(encoding="utf-8").splitlines():
            if not ln.strip():
                continue
            pmeta = json.loads(ln)
            tid = str(pmeta.get("target_id") or pmeta.get("pair_id"))
            if tid in items_by_tid:
                merged_items.append(items_by_tid[tid])
            else:
                cpath = runner._cache_path("openai", model, pmeta["prompt"], tb, 0.0)
                if cpath.exists():
                    cd = json.loads(cpath.read_text(encoding="utf-8"))
                    cd["cached"] = True
                    rec = {
                        "target_id": tid,
                        "task": pmeta.get("task", ""),
                        "length": pmeta.get("length", 0),
                        **cd,
                    }
                    if "pair_id" in pmeta:
                        rec["pair_id"] = pmeta["pair_id"]
                    merged_items.append(rec)
        if merged_items:
            out_f = odir / f"{safe_tag}_{stage_sfx}.jsonl"
            out_f.write_text("\n".join(json.dumps(x) for x in merged_items) + "\n", encoding="utf-8")
            print(f"Saved {len(merged_items):3d} predictions -> {out_f}")

    print(f"Total 50%-discounted Batch cost for {safe_tag} batch portion: ${tot_cost:.2f}")


def main() -> None:
    parser = argparse.ArgumentParser(description="OpenAI 50%-Off Batch API runner for Protein Structure Without Tools.")
    parser.add_argument("--prepare", action="store_true", help="Build batch input JSONL offline")
    parser.add_argument("--skip-cached", action="store_true", help="Exclude prompts already in outputs/api_cache/")
    parser.add_argument("--submit", action="store_true", help="Upload JSONL & create OpenAI Batch")
    parser.add_argument("--status", action="store_true", help="Check batch status")
    parser.add_argument("--collect", action="store_true", help="Download completed batch into outputs/predictions/")
    parser.add_argument("--wait", action="store_true", help="Poll every 60s until completed, then auto-collect")
    parser.add_argument("--submit-and-wait", action="store_true", help="Submit one or more --models and wait/collect all")
    parser.add_argument("--model", default="gpt-6.2-astra")
    parser.add_argument("--models", default="", help="Comma-separated models for --submit-and-wait")
    parser.add_argument("--effort", default="high", choices=["low", "medium", "high", "xhigh"])
    parser.add_argument("--stages", default="1,3,4,5,6", help="Comma-separated test IDs, e.g. 2,7")
    parser.add_argument("--batch-id", help="Batch ID (defaults to latest in outputs/batches/active_batches.json)")
    args = parser.parse_args()

    if args.submit_and_wait:
        mlist = [m.strip() for m in (args.models or args.model).split(",") if m.strip()]
        pending: Dict[str, str] = {}
        for m in mlist:
            b_obj = submit_batch(m, args.effort, stages=args.stages)
            pending[b_obj["id"]] = m
        while pending:
            done_ids = []
            for bid, m in list(pending.items()):
                b = check_batch_status(bid)
                st = b.get("status")
                if st == "completed":
                    collect_batch(bid, model_override=m, effort_override=args.effort)
                    done_ids.append(bid)
                elif st in ("failed", "expired", "cancelled"):
                    raise SystemExit(f"Batch {bid} ({m}) ended with status={st}: {b.get('errors')}")
            for bid in done_ids:
                pending.pop(bid, None)
            if pending:
                time.sleep(60.0)
        return

    if args.prepare:
        prepare_batch_jsonl(args.model, args.effort, skip_cached=args.skip_cached, stages=args.stages)
        return

    if args.submit:
        submit_batch(args.model, args.effort, stages=args.stages)
        return

    bid = args.batch_id
    if not bid:
        reg_path = REPO_ROOT / "outputs" / "batches" / "active_batches.json"
        if not reg_path.exists():
            raise SystemExit("No --batch-id given and no active_batches.json found.")
        bid = json.loads(reg_path.read_text(encoding="utf-8"))[-1]["batch_id"]

    if args.status:
        check_batch_status(bid)
    elif args.collect:
        collect_batch(bid)
    elif args.wait:
        while True:
            b = check_batch_status(bid)
            st = b.get("status")
            if st == "completed":
                collect_batch(bid)
                break
            if st in ("failed", "expired", "cancelled"):
                raise SystemExit(f"Batch {bid} ended with status={st}: {b.get('errors')}")
            time.sleep(60.0)
    else:
        parser.error("Specify one of --prepare, --submit, --status, --collect, or --wait")


if __name__ == "__main__":
    main()
