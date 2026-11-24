#!/usr/bin/env python3
"""Submit the rebuilt Task 4 (v2.0 strict post-May-2026 NK holdout, N=60) FASTA to official NetGO 4.0 (GOCurator)."""

from __future__ import annotations

from pathlib import Path
import sys

SCRIPTS_DIR = Path(__file__).resolve().parent
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from fetch_sota_t456_network import submit_and_poll_netgo

if __name__ == "__main__":
    res = submit_and_poll_netgo("ng4.0", "NG4.0", poll_timeout_s=1800.0)
    assert res is not None and res.exists(), "NetGO 4.0 failed to return results"
    print(f"[OK] NetGO 4.0 completed -> {res}", flush=True)
