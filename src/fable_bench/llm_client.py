"""Unified multi-provider async HTTPX LLM client with caching, budget guard & fallback audit.

Works with zero vendor SDK bloat (pure `httpx`), supports:
  - `anthropic`: `claude-fable-5-2`, `claude-opus-5-5` (Direct & Extended Thinking + silent-fallback check)
  - `openai`: `gpt-5.2` (with reasoning_effort support)
  - `gemini`: Google Generative Language REST API
  - `openrouter`: Single-key access to Anthropic, OpenAI, Google, DeepSeek, Qwen
  - `offline`: Zero-key mode for importing manual web-chat / Batch API JSONL responses
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import os
from pathlib import Path
import time
from typing import Any, Dict, Optional

from dotenv import load_dotenv
import httpx

# Approximate Nov 2026 per-million-token pricing for live cost tracking
PRICING_USD_PER_MTOK: Dict[str, tuple[float, float]] = {
    "claude-fable-5-2": (8.0, 40.0),
    "claude-opus-5-5": (5.0, 25.0),
    "gpt-5.2": (2.5, 15.0),
    "gemini-3-pro": (2.0, 12.0),
}


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
        cache_dir: str | Path = "outputs/api_cache",
        max_usd_budget: Optional[float] = None,
        assert_no_fallback: bool = True,
    ) -> None:
        load_dotenv()
        self.cache_dir = Path(cache_dir)
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
            rec = await self._call_gemini(model, prompt, max_tokens, temperature)
        elif provider == "openrouter":
            rec = await self._call_openrouter(model, prompt, max_tokens, temperature)
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

        async with httpx.AsyncClient(timeout=240.0) as client:
            resp = await client.post(
                "https://api.anthropic.com/v1/messages",
                headers={
                    "x-api-key": api_key,
                    "anthropic-version": "2023-06-01",
                    "content-type": "application/json",
                },
                json=body,
            )
            resp.raise_for_status()
            data = resp.json()

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

        body: Dict[str, Any] = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "max_completion_tokens": max_tokens + thinking_budget,
        }
        if thinking_budget > 0:
            body["reasoning_effort"] = "high"
        else:
            body["temperature"] = temperature

        async with httpx.AsyncClient(timeout=240.0) as client:
            resp = await client.post(
                "https://api.openai.com/v1/chat/completions",
                headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
                json=body,
            )
            resp.raise_for_status()
            data = resp.json()

        choice = data["choices"][0]
        text = choice["message"].get("content") or ""
        stop_reason = str(choice.get("finish_reason", ""))
        usage = data.get("usage", {})
        in_tok = int(usage.get("prompt_tokens", 0))
        out_tok = int(usage.get("completion_tokens", 0))
        returned_model = str(data.get("model", model))

        return LLMCallRecord(
            provider="openai",
            requested_model=model,
            returned_model=returned_model,
            thinking_budget=thinking_budget,
            text=text,
            thinking_text="",
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
        temperature: float,
    ) -> LLMCallRecord:
        api_key = os.getenv("GEMINI_API_KEY", "").strip()
        if not api_key:
            raise EnvironmentError("GEMINI_API_KEY is not set. Add it to `.env` to run live calls.")

        url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={api_key}"
        body = {
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {"temperature": temperature, "maxOutputTokens": max_tokens},
        }
        async with httpx.AsyncClient(timeout=240.0) as client:
            resp = await client.post(url, json=body)
            resp.raise_for_status()
            data = resp.json()

        cand = data.get("candidates", [{}])[0]
        parts = cand.get("content", {}).get("parts", [])
        text = "\n".join(p.get("text", "") for p in parts if not p.get("thought"))
        usage = data.get("usageMetadata", {})
        in_tok = int(usage.get("promptTokenCount", 0))
        out_tok = int(usage.get("candidatesTokenCount", 0))

        return LLMCallRecord(
            provider="gemini",
            requested_model=model,
            returned_model=model,
            thinking_budget=0,
            text=text,
            thinking_text="",
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
        temperature: float,
    ) -> LLMCallRecord:
        api_key = os.getenv("OPENROUTER_API_KEY", "").strip()
        if not api_key:
            raise EnvironmentError("OPENROUTER_API_KEY is not set. Add it to `.env` to run live calls.")

        body = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        async with httpx.AsyncClient(timeout=240.0) as client:
            resp = await client.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
                json=body,
            )
            resp.raise_for_status()
            data = resp.json()

        choice = data["choices"][0]
        text = choice["message"].get("content") or ""
        usage = data.get("usage", {})
        in_tok = int(usage.get("prompt_tokens", 0))
        out_tok = int(usage.get("completion_tokens", 0))
        returned_model = str(data.get("model", model))

        return LLMCallRecord(
            provider="openrouter",
            requested_model=model,
            returned_model=returned_model,
            thinking_budget=0,
            text=text,
            thinking_text="",
            stop_reason=str(choice.get("finish_reason", "")),
            input_tokens=in_tok,
            output_tokens=out_tok,
            est_cost_usd=estimate_cost_usd(model, in_tok, out_tok),
            latency_sec=0.0,
            silent_fallback_detected=False,
            refusal_detected=False,
        )
