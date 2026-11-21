"""Unified multi-provider async HTTPX LLM client with caching, budget guard & fallback audit.

Works with zero vendor SDK bloat (pure `httpx`), supports:
  - `anthropic`: `claude-fable-5-2`, `claude-opus-5-5` (Direct & Extended Thinking + silent-fallback check)
  - `openai`: `gpt-5.2` (with reasoning_effort support)
  - `gemini`: Google Generative Language REST API
  - `openrouter`: Single-key access to Anthropic, OpenAI, Google, DeepSeek, Qwen
  - `offline`: Zero-key mode for importing manual web-chat / Batch API JSONL responses
"""

from __future__ import annotations

import asyncio
from dataclasses import asdict, dataclass
import hashlib
import json
import os
from pathlib import Path
import time
from typing import Any, Dict, Optional

from dotenv import load_dotenv
import httpx

REPO_ROOT = Path(__file__).resolve().parents[2]

# Approximate Nov 2026 per-million-token pricing for live cost tracking
PRICING_USD_PER_MTOK: Dict[str, tuple[float, float]] = {
    "claude-fable-5-2": (8.0, 40.0),
    "claude-opus-5-5": (5.0, 25.0),
    "claude-sonnet-4-6": (3.0, 15.0),
    "gpt-6.2-astra": (5.0, 25.0),
    "gpt-6.2-sol": (2.0, 10.0),
    "gpt-6-astra": (5.0, 25.0),
    "gpt-6-sol": (2.0, 10.0),
    "gpt-6-luna": (0.5, 2.5),
    "gpt-6.1-terra": (5.0, 25.0),
    "gpt-6.1-sol": (2.0, 10.0),
    "gpt-6.1": (2.0, 10.0),
    "gpt-5.6-sol": (2.0, 10.0),
    "gpt-5.2": (2.5, 15.0),
    "gpt-5": (1.25, 10.0),
    "gpt-4.1": (2.0, 8.0),
    "o3": (2.0, 8.0),
    "gemini-3.1-pro": (2.0, 12.0),
    "gemini-3.1-pro-preview": (2.0, 12.0),
    "gemini-3-pro": (2.0, 12.0),
    "gemini-2.5-pro": (1.25, 10.0),
}


async def _post_with_retries(
    url: str,
    headers: Optional[Dict[str, str]],
    json_body: Dict[str, Any],
    max_attempts: int = 5,
    timeout_sec: float = 420.0,
) -> Dict[str, Any]:
    """POST JSON with exponential backoff on transient 429/5xx/network errors."""
    last_err: Optional[Exception] = None
    for attempt in range(1, max_attempts + 1):
        try:
            async with httpx.AsyncClient(timeout=timeout_sec) as client:
                resp = await client.post(url, headers=headers, json=json_body)
            if resp.status_code in (429, 500, 502, 503, 504, 529) and attempt < max_attempts:
                wait_s = min(30.0, 1.8 * (2 ** (attempt - 1)))
                await asyncio.sleep(wait_s)
                continue
            if resp.status_code >= 400:
                raise RuntimeError(f"HTTP {resp.status_code}: {resp.text[:500]}")
            return resp.json()
        except (httpx.TimeoutException, httpx.TransportError) as exc:
            if "Errno 8" in str(exc) or "nodename nor servname" in str(exc):
                raise
            last_err = exc
            if attempt < max_attempts:
                await asyncio.sleep(min(30.0, 2.0 * (2 ** (attempt - 1))))
                continue
            raise
    raise RuntimeError(f"Exhausted {max_attempts} retries: {last_err}")


@dataclass
class LLMCallRecord:
    """Normalized record of a single LLM call (cached or live)."""

    provider: str
    requested_model: str
    returned_model: str
    thinking_budget: int
    text: str
    thinking_text: str
    stop_reason: str
    input_tokens: int
    output_tokens: int
    est_cost_usd: float
    latency_sec: float
    silent_fallback_detected: bool
    refusal_detected: bool
    cached: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def estimate_cost_usd(model: str, in_tok: int, out_tok: int) -> float:
    """Estimate API cost in USD from model name and token counts."""
    base = model.split("/")[-1]
    in_rate, out_rate = PRICING_USD_PER_MTOK.get(base, (5.0, 20.0))
    return (in_tok * in_rate + out_tok * out_rate) / 1_000_000.0


class UnifiedLLMRunner:
    """Cached async multi-provider LLM client with hard USD budget & fallback assertions."""

    def __init__(
        self,
        cache_dir: str | Path | None = None,
        max_usd_budget: Optional[float] = None,
        assert_no_fallback: bool = True,
    ) -> None:
        load_dotenv(REPO_ROOT / ".env")
        load_dotenv()
        self.cache_dir = Path(cache_dir) if cache_dir is not None else (REPO_ROOT / "outputs" / "api_cache")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        env_budget = float(os.getenv("MAX_USD_BUDGET", "25.0"))
        self.max_usd_budget = max_usd_budget if max_usd_budget is not None else env_budget
        self.assert_no_fallback = assert_no_fallback
        self.cumulative_live_usd = 0.0

    def _cache_path(
        self,
        provider: str,
        model: str,
        prompt: str,
        thinking_budget: int,
        temperature: float,
    ) -> Path:
        payload = json.dumps(
            {
                "provider": provider,
                "model": model,
                "prompt": prompt,
                "thinking_budget": thinking_budget,
                "temperature": round(temperature, 3),
            },
            sort_keys=True,
        )
        digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()[:24]
        safe_model = model.replace("/", "_").replace(":", "_")
        return self.cache_dir / f"{safe_model}_tb{thinking_budget}_{digest}.json"

    async def complete(
        self,
        provider: str,
        model: str,
        prompt: str,
        max_tokens: int = 4096,
        thinking_budget: int = 0,
        temperature: float = 0.0,
    ) -> LLMCallRecord:
        """Execute or load from disk cache a single prompt completion."""
        cpath = self._cache_path(provider, model, prompt, thinking_budget, temperature)
        if cpath.exists():
            data = json.loads(cpath.read_text(encoding="utf-8"))
            data["cached"] = True
            return LLMCallRecord(**data)

        if self.cumulative_live_usd >= self.max_usd_budget:
            raise RuntimeError(
                f"Hard budget cap reached (${self.cumulative_live_usd:.2f} >= ${self.max_usd_budget:.2f})."
            )

        t0 = time.perf_counter()
        if provider == "anthropic":
            rec = await self._call_anthropic(model, prompt, max_tokens, thinking_budget, temperature)
        elif provider == "openai":
            rec = await self._call_openai(model, prompt, max_tokens, thinking_budget, temperature)
        elif provider == "gemini":
            rec = await self._call_gemini(model, prompt, max_tokens, thinking_budget, temperature)
        elif provider == "openrouter":
            rec = await self._call_openrouter(model, prompt, max_tokens, thinking_budget, temperature)
        else:
            raise ValueError(f"Unsupported provider '{provider}'")

        rec.latency_sec = round(time.perf_counter() - t0, 3)
        self.cumulative_live_usd += rec.est_cost_usd

        if self.assert_no_fallback and rec.silent_fallback_detected:
            raise RuntimeError(
                f"SILENT FALLBACK DETECTED: requested '{rec.requested_model}' "
                f"but API served '{rec.returned_model}'! Check prompt bio-filter triggers."
            )

        cpath.write_text(json.dumps(rec.to_dict(), indent=2), encoding="utf-8")
        return rec

    async def _call_anthropic(
        self,
        model: str,
        prompt: str,
        max_tokens: int,
        thinking_budget: int,
        temperature: float,
    ) -> LLMCallRecord:
        api_key = os.getenv("ANTHROPIC_API_KEY", "").strip()
        if not api_key:
            raise EnvironmentError("ANTHROPIC_API_KEY is not set. Add it to `.env` to run live calls.")

        body: Dict[str, Any] = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
        }
        if thinking_budget > 0:
            body["max_tokens"] = max_tokens + thinking_budget
            body["temperature"] = 1.0  # Required by Anthropic Messages API when thinking is enabled
            body["thinking"] = {"type": "enabled", "budget_tokens": thinking_budget}
        else:
            body["max_tokens"] = max_tokens
            body["temperature"] = temperature

        data = await _post_with_retries(
            "https://api.anthropic.com/v1/messages",
            headers={
                "x-api-key": api_key,
                "anthropic-version": "2023-06-01",
                "content-type": "application/json",
            },
            json_body=body,
        )

        returned_model = str(data.get("model", model))
        stop_reason = str(data.get("stop_reason", ""))
        text_parts = []
        thinking_parts = []
        for block in data.get("content", []):
            if block.get("type") == "text":
                text_parts.append(block.get("text", ""))
            elif block.get("type") == "thinking":
                thinking_parts.append(block.get("thinking", ""))

        usage = data.get("usage", {})
        in_tok = int(usage.get("input_tokens", 0))
        out_tok = int(usage.get("output_tokens", 0))

        # Audit silent fallback (e.g., claude-fable-5-2 silently routed to claude-opus-*)
        req_family = model.split("-")[1] if "-" in model else model
        silent_fb = (req_family == "fable") and ("fable" not in returned_model.lower())
        refusal = stop_reason == "refusal"

        return LLMCallRecord(
            provider="anthropic",
            requested_model=model,
            returned_model=returned_model,
            thinking_budget=thinking_budget,
            text="\n".join(text_parts),
            thinking_text="\n".join(thinking_parts),
            stop_reason=stop_reason,
            input_tokens=in_tok,
            output_tokens=out_tok,
            est_cost_usd=estimate_cost_usd(model, in_tok, out_tok),
            latency_sec=0.0,
            silent_fallback_detected=silent_fb,
            refusal_detected=refusal,
        )

    async def _call_openai(
        self,
        model: str,
        prompt: str,
        max_tokens: int,
        thinking_budget: int,
        temperature: float,
    ) -> LLMCallRecord:
        api_key = os.getenv("OPENAI_API_KEY", "").strip()
        if not api_key:
            raise EnvironmentError("OPENAI_API_KEY is not set. Add it to `.env` to run live calls.")

        headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
        effort: Optional[str] = None
        if thinking_budget > 0:
            if thinking_budget >= 12000:
                effort = "xhigh"
            elif thinking_budget >= 4000:
                effort = "high"
            elif thinking_budget >= 1500:
                effort = "medium"
            else:
                effort = "low"

        # Ensure high/xhigh reasoning never truncates mid-thought before emitting JSON
        if effort in ("high", "xhigh"):
            eff_max = max(24576, max_tokens + thinking_budget)
        elif effort in ("medium", "low"):
            eff_max = max(12288, max_tokens + thinking_budget)
        else:
            eff_max = max_tokens + thinking_budget

        body: Dict[str, Any] = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "max_completion_tokens": eff_max,
        }
        if effort is not None:
            body["reasoning_effort"] = effort
        else:
            body["temperature"] = temperature

        try:
            data = await _post_with_retries(
                "https://api.openai.com/v1/chat/completions",
                headers=headers,
                json_body=body,
            )
        except RuntimeError as err:
            msg = str(err)
            # Reasoning-only OpenAI models may reject `temperature` or `xhigh` or require `/v1/responses`
            if "v1/responses" in msg:
                resp_body: Dict[str, Any] = {
                    "model": model,
                    "input": prompt,
                    "max_output_tokens": eff_max,
                }
                if effort is not None:
                    resp_body["reasoning"] = {"effort": "high" if effort == "xhigh" else effort}
                rdata = await _post_with_retries(
                    "https://api.openai.com/v1/responses",
                    headers=headers,
                    json_body=resp_body,
                )
                out_chunks = []
                for item in rdata.get("output", []):
                    for c in item.get("content", []):
                        if c.get("type") in ("output_text", "text"):
                            out_chunks.append(c.get("text", ""))
                text_out = rdata.get("output_text") or "\n".join(out_chunks)
                usage = rdata.get("usage", {})
                in_tok = int(usage.get("input_tokens", 0))
                out_tok = int(usage.get("output_tokens", 0))
                reas_tok = int((usage.get("output_tokens_details") or {}).get("reasoning_tokens", 0))
                returned_model = str(rdata.get("model", model))
                return LLMCallRecord(
                    provider="openai",
                    requested_model=model,
                    returned_model=returned_model,
                    thinking_budget=thinking_budget,
                    text=text_out,
                    thinking_text=f"[openai_reasoning_tokens={reas_tok}]" if reas_tok else "",
                    stop_reason=str(rdata.get("status", "completed")),
                    input_tokens=in_tok,
                    output_tokens=out_tok,
                    est_cost_usd=estimate_cost_usd(model, in_tok, out_tok),
                    latency_sec=0.0,
                    silent_fallback_detected=False,
                    refusal_detected=False,
                )
            if "temperature" in msg and "temperature" in body:
                body.pop("temperature", None)
                data = await _post_with_retries(
                    "https://api.openai.com/v1/chat/completions",
                    headers=headers,
                    json_body=body,
                )
            elif "xhigh" in msg and body.get("reasoning_effort") == "xhigh":
                body["reasoning_effort"] = "high"
                data = await _post_with_retries(
                    "https://api.openai.com/v1/chat/completions",
                    headers=headers,
                    json_body=body,
                )
            else:
                raise

        choice = data["choices"][0]
        text = choice["message"].get("content") or ""
        stop_reason = str(choice.get("finish_reason", ""))
        usage = data.get("usage", {})
        in_tok = int(usage.get("prompt_tokens", 0))
        out_tok = int(usage.get("completion_tokens", 0))
        reas_tok = int((usage.get("completion_tokens_details") or {}).get("reasoning_tokens", 0))
        returned_model = str(data.get("model", model))

        return LLMCallRecord(
            provider="openai",
            requested_model=model,
            returned_model=returned_model,
            thinking_budget=thinking_budget,
            text=text,
            thinking_text=f"[openai_reasoning_tokens={reas_tok}]" if reas_tok else "",
            stop_reason=stop_reason,
            input_tokens=in_tok,
            output_tokens=out_tok,
            est_cost_usd=estimate_cost_usd(model, in_tok, out_tok),
            latency_sec=0.0,
            silent_fallback_detected=False,
            refusal_detected=(choice["message"].get("refusal") is not None),
        )

    async def _call_gemini(
        self,
        model: str,
        prompt: str,
        max_tokens: int,
        thinking_budget: int,
        temperature: float,
    ) -> LLMCallRecord:
        api_key = os.getenv("GEMINI_API_KEY", "").strip()
        if not api_key:
            raise EnvironmentError("GEMINI_API_KEY is not set. Add it to `.env` to run live calls.")

        url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={api_key}"
        gen_cfg: Dict[str, Any] = {
            "temperature": temperature,
            "maxOutputTokens": max_tokens + thinking_budget,
        }
        if thinking_budget > 0:
            gen_cfg["thinkingConfig"] = {"thinkingBudget": thinking_budget}
        body = {
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": gen_cfg,
        }
        data = await _post_with_retries(url, headers=None, json_body=body)

        cand = data.get("candidates", [{}])[0]
        parts = cand.get("content", {}).get("parts", [])
        text = "\n".join(p.get("text", "") for p in parts if not p.get("thought"))
        thoughts = "\n".join(p.get("text", "") for p in parts if p.get("thought"))
        usage = data.get("usageMetadata", {})
        in_tok = int(usage.get("promptTokenCount", 0))
        out_tok = int(usage.get("candidatesTokenCount", 0)) + int(usage.get("thoughtsTokenCount", 0))

        return LLMCallRecord(
            provider="gemini",
            requested_model=model,
            returned_model=model,
            thinking_budget=thinking_budget,
            text=text,
            thinking_text=thoughts,
            stop_reason=str(cand.get("finishReason", "")),
            input_tokens=in_tok,
            output_tokens=out_tok,
            est_cost_usd=estimate_cost_usd(model, in_tok, out_tok),
            latency_sec=0.0,
            silent_fallback_detected=False,
            refusal_detected=False,
        )

    async def _call_openrouter(
        self,
        model: str,
        prompt: str,
        max_tokens: int,
        thinking_budget: int,
        temperature: float,
    ) -> LLMCallRecord:
        api_key = os.getenv("OPENROUTER_API_KEY", "").strip()
        if not api_key:
            raise EnvironmentError("OPENROUTER_API_KEY is not set. Add it to `.env` to run live calls.")

        body: Dict[str, Any] = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens + thinking_budget,
            "temperature": temperature,
        }
        if thinking_budget > 0:
            body["reasoning"] = {"max_tokens": thinking_budget}
        data = await _post_with_retries(
            "https://openrouter.ai/api/v1/chat/completions",
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            json_body=body,
        )

        choice = data["choices"][0]
        text = choice["message"].get("content") or ""
        thinking_text = str(choice["message"].get("reasoning") or "")
        usage = data.get("usage", {})
        in_tok = int(usage.get("prompt_tokens", 0))
        out_tok = int(usage.get("completion_tokens", 0))
        returned_model = str(data.get("model", model))

        return LLMCallRecord(
            provider="openrouter",
            requested_model=model,
            returned_model=returned_model,
            thinking_budget=thinking_budget,
            text=text,
            thinking_text=thinking_text,
            stop_reason=str(choice.get("finish_reason", "")),
            input_tokens=in_tok,
            output_tokens=out_tok,
            est_cost_usd=estimate_cost_usd(model, in_tok, out_tok),
            latency_sec=0.0,
            silent_fallback_detected=False,
            refusal_detected=False,
        )
