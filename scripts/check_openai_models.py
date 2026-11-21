#!/usr/bin/env python3
"""Zero-cost check of available OpenAI model IDs for `.env` OPENAI_API_KEY."""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv
import httpx

REPO_ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    load_dotenv(REPO_ROOT / ".env")
    key = os.getenv("OPENAI_API_KEY", "").strip()
    if not key:
        raise SystemExit("OPENAI_API_KEY missing in .env")

    resp = httpx.get(
        "https://api.openai.com/v1/models",
        headers={"Authorization": f"Bearer {key}"},
        timeout=20.0,
    )
    print(f"HTTP {resp.status_code}")
    if resp.status_code != 200:
        print(resp.text[:300])
        raise SystemExit(1)

    ids = sorted(m["id"] for m in resp.json().get("data", []))
    frontier = [
        m
        for m in ids
        if any(k in m for k in ("gpt-6", "gpt-5", "gpt-4.1", "o3", "o4", "sol", "astra"))
        and not any(x in m for x in ("audio", "realtime", "transcribe", "tts", "search", "image"))
    ]
    print(f"Total models visible: {len(ids)}")
    print("Frontier candidate IDs:")
    for m in frontier:
        print(f"  - {m}")


if __name__ == "__main__":
    main()
