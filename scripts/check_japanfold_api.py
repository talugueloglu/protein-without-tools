#!/usr/bin/env python3
"""Fetch JapanFold model catalog, pricing history, and account credits."""

from __future__ import annotations

import json
import os
from pathlib import Path

from dotenv import load_dotenv
import httpx

REPO = Path(__file__).resolve().parents[1]


def main() -> None:
    load_dotenv(REPO / ".env")
    key = os.getenv("JAPANFOLD_API_KEY", "").strip()
    base = os.getenv("JAPANFOLD_BASE_URL", "https://api.japanfold.aiand.com").rstrip("/")
    headers = {"Authorization": f"Bearer {key}"}

    out_dir = REPO / "outputs" / "colab"
    out_dir.mkdir(parents=True, exist_ok=True)

    with httpx.Client(timeout=30.0, follow_redirects=True) as client:
        acct = client.get(f"{base}/v1/account", headers=headers)
        print(f"Account HTTP {acct.status_code}")
        if acct.status_code == 200:
            ad = acct.json()
            print("  Balance USD:", ad.get("balance"), "| Credits:", ad.get("credits"))

        r = client.get(f"{base}/v1/models", headers=headers)
        print(f"Models HTTP {r.status_code} ({len(r.text)} bytes)")
        if r.status_code == 200:
            (out_dir / "japanfold_models.json").write_text(r.text, encoding="utf-8")
            cat = r.json()
            for group in ("models", "design_models", "protocols", "embed_models"):
                items = cat.get(group, [])
                if not items:
                    continue
                print(f"\n=== {group} ({len(items)}) ===")
                for m in items:
                    tc = m.get("typical_cost") or {}
                    print(
                        f"  {m.get('id'):22s} | caps={m.get('caps')} "
                        f"| p50/tgt=${tc.get('usd_per_target_p50')} p90/tgt=${tc.get('usd_per_target_p90')} "
                        f"| job_p50=${tc.get('usd_p50')} job_p90=${tc.get('usd_p90')} (n={tc.get('jobs')})"
                    )


if __name__ == "__main__":
    main()
