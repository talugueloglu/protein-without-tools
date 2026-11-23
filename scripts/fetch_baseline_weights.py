#!/usr/bin/env python3
"""Download official open-source weights & pure-Python modules for ProteinMPNN (v_48_020) and ESM-2 (650M)."""

from __future__ import annotations

import io
from pathlib import Path
import tarfile
import time

import httpx

REPO = Path(__file__).resolve().parents[1]
CACHE = REPO / "outputs" / "proteinmpnn_cache"


def download_stream(client: httpx.Client, url: str, dest: Path, min_bytes: int = 1000) -> None:
    if dest.exists() and dest.stat().st_size >= min_bytes:
        print(f"[cached] {dest.name} ({dest.stat().st_size / 1e6:.2f} MB)", flush=True)
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    t0 = time.monotonic()
    print(f"[download] {url} -> {dest.name}", flush=True)
    with client.stream("GET", url) as r:
        r.raise_for_status()
        total = int(r.headers.get("Content-Length") or 0)
        got = 0
        last_log = t0
        with tmp.open("wb") as f:
            for chunk in r.iter_bytes(chunk_size=1024 * 1024):
                f.write(chunk)
                got += len(chunk)
                now = time.monotonic()
                if now - last_log >= 4.0:
                    pct = f" ({100.0 * got / total:.1f}%)" if total else ""
                    print(f"  ... {got / 1e6:.1f} MB{pct} in {now - t0:.1f}s", flush=True)
                    last_log = now
    if got < min_bytes:
        raise RuntimeError(f"Downloaded {got} bytes < {min_bytes} for {url}")
    tmp.replace(dest)
    print(f"[done] {dest.name} ({got / 1e6:.2f} MB in {time.monotonic() - t0:.1f}s)", flush=True)


def main() -> None:
    CACHE.mkdir(parents=True, exist_ok=True)
    with httpx.Client(timeout=httpx.Timeout(300.0, connect=30.0), follow_redirects=True) as client:
        # 1. ProteinMPNN official utils + vanilla v_48_020 weights (Dauparas et al., Science 2022)
        download_stream(
            client,
            "https://raw.githubusercontent.com/dauparas/ProteinMPNN/main/protein_mpnn_utils.py",
            CACHE / "protein_mpnn_utils.py",
            min_bytes=20_000,
        )
        download_stream(
            client,
            "https://raw.githubusercontent.com/dauparas/ProteinMPNN/main/vanilla_model_weights/v_48_020.pt",
            CACHE / "v_48_020.pt",
            min_bytes=5_000_000,
        )

        # 2. fair-esm v2.0.0 pure-Python source (Lin et al., Science 2023)
        esm_pkg_dir = CACHE / "fair_esm_src" / "esm"
        if not esm_pkg_dir.exists():
            tar_p = CACHE / "fair_esm_v2_0_0.tar.gz"
            download_stream(
                client,
                "https://github.com/facebookresearch/esm/archive/refs/tags/v2.0.0.tar.gz",
                tar_p,
                min_bytes=50_000,
            )
            with tarfile.open(tar_p, "r:gz") as tf:
                for member in tf.getmembers():
                    parts = Path(member.name).parts
                    if len(parts) >= 2 and parts[1] == "esm":
                        rel = Path(*parts[1:])
                        out_p = CACHE / "fair_esm_src" / rel
                        if member.isdir():
                            out_p.mkdir(parents=True, exist_ok=True)
                        elif member.isfile():
                            out_p.parent.mkdir(parents=True, exist_ok=True)
                            fobj = tf.extractfile(member)
                            if fobj is not None:
                                out_p.write_bytes(fobj.read())
            print(f"[extracted] {esm_pkg_dir}", flush=True)
        else:
            print(f"[cached] {esm_pkg_dir}", flush=True)

        # 3. Official ESM-2 650M checkpoint + regression weights
        download_stream(
            client,
            "https://dl.fbaipublicfiles.com/fair-esm/models/esm2_t33_650M_UR50D.pt",
            CACHE / "esm2_t33_650M_UR50D.pt",
            min_bytes=1_000_000_000,
        )
        download_stream(
            client,
            "https://dl.fbaipublicfiles.com/fair-esm/regression/esm2_t33_650M_UR50D-contact-regression.pt",
            CACHE / "esm2_t33_650M_UR50D-contact-regression.pt",
            min_bytes=1_000,
        )


if __name__ == "__main__":
    main()
