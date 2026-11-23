#!/usr/bin/env python3
"""Fetch all network assets and run remote folding / NetGO jobs for Tasks 4, 5, and 6 SOTA baselines:
  - Task 4 (N=60): Submit anonymised FASTA to official NetGO 4.0 (GOCurator) & NetGO 3.0 servers
  - Task 5 (N=52): Fold monomers via JapanFold ESMFold-2 (PDBs), download Foldseek macOS binary + SaProt_650M_AF2
  - Task 6 (N=38): Fold dimers via JapanFold Boltz-2 (two-chain PDBs), clone LDeng0205/StaB-ddG + stability_finetuned.pt
All downloads and API calls are idempotent and disk-cached.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tarfile
import time
from typing import Any, Dict, List, Optional

import httpx

REPO = Path(__file__).resolve().parents[1]
if str(REPO / "src") not in sys.path:
    sys.path.insert(0, str(REPO / "src"))

from fable_bench.japanfold_client import JapanFoldClient, find_pdb_for_target


def fetch_task5_esmfold2(jf: JapanFoldClient) -> None:
    out_dir = REPO / "outputs" / "japanfold_cache" / "t5_esmfold2_pdbs"
    out_dir.mkdir(parents=True, exist_ok=True)

    ddir = REPO / "data" / "alanine_scan_benchmark"
    manifest = json.loads((ddir / "manifest.json").read_text(encoding="utf-8"))["targets"]
    todo: List[tuple[str, str]] = []
    for rec in manifest:
        tid = rec["target_id"]
        if (out_dir / f"{tid}.pdb").exists():
            continue
        t_obj = json.loads((ddir / "targets" / f"{tid}.json").read_text(encoding="utf-8"))
        todo.append((tid, t_obj["sequence"].strip().upper()))

    if not todo:
        print(f"[T5 ESMFold-2] All {len(manifest)} PDBs already cached -> {out_dir}", flush=True)
        return

    print(f"\n=== [T5 ESMFold-2] Folding {len(todo)} monomers on JapanFold ===", flush=True)
    batch_size = 26
    for b0 in range(0, len(todo), batch_size):
        chunk = todo[b0 : b0 + batch_size]
        targets_payload = [{"name": tid, "content": f">{tid}\n{seq}"} for tid, seq in chunk]
        meta, files = jf.submit_and_collect(
            "/v1/predictions",
            {
                "model": "esmfold2",
                "name": f"t5_saprot_esmfold2_{b0}",
                "targets": targets_payload,
                "params": {"use_msa_server": False, "fast": False, "output_format": "pdb"},
            },
            cache_prefix=f"t5_esmfold2_b{b0}",
        )
        saved = 0
        for idx0, (tid, _seq) in enumerate(chunk):
            pdb_txt = find_pdb_for_target(files, tid, idx0)
            if not pdb_txt:
                raise RuntimeError(f"Missing ESMFold-2 PDB for {tid} (keys={list(files.keys())[:8]})")
            (out_dir / f"{tid}.pdb").write_text(pdb_txt, encoding="utf-8")
            saved += 1
        print(
            f"  [T5 ESMFold-2] Saved {saved} PDBs ({min(b0 + batch_size, len(todo))}/{len(todo)}, "
            f"cost=${meta['cost_usd']:.4f})",
            flush=True,
        )


def fetch_task6_boltz2(jf: JapanFoldClient) -> None:
    out_dir = REPO / "outputs" / "japanfold_cache" / "t6_boltz2_pdbs"
    out_dir.mkdir(parents=True, exist_ok=True)

    ddir = REPO / "data" / "binding_alanine_benchmark"
    manifest = json.loads((ddir / "manifest.json").read_text(encoding="utf-8"))["targets"]
    todo: List[tuple[str, str, str]] = []
    for rec in manifest:
        tid = rec["target_id"]
        if (out_dir / f"{tid}.pdb").exists():
            continue
        t_obj = json.loads((ddir / "targets" / f"{tid}.json").read_text(encoding="utf-8"))
        sa = t_obj["scanned_sequence"].strip().upper()
        sb = t_obj["partner_sequence"].strip().upper()
        todo.append((tid, sa, sb))

    if not todo:
        print(f"[T6 Boltz-2] All {len(manifest)} complex PDBs already cached -> {out_dir}", flush=True)
        return

    print(f"\n=== [T6 Boltz-2] Folding {len(todo)} two-chain complexes on JapanFold ===", flush=True)
    batch_size = 19
    for b0 in range(0, len(todo), batch_size):
        chunk = todo[b0 : b0 + batch_size]
        targets_payload = [
            {"name": tid, "content": f">A\n{sa}\n>B\n{sb}\n"}
            for tid, sa, sb in chunk
        ]
        meta, files = jf.submit_and_collect(
            "/v1/predictions",
            {
                "model": "boltz2",
                "name": f"t6_stabddg_boltz2_{b0}",
                "targets": targets_payload,
                "params": {"output_format": "pdb", "use_msa_server": True},
            },
            cache_prefix=f"t6_boltz2_b{b0}",
        )
        saved = 0
        for idx0, (tid, _sa, _sb) in enumerate(chunk):
            pdb_txt = find_pdb_for_target(files, tid, idx0)
            if not pdb_txt:
                raise RuntimeError(f"Missing Boltz-2 PDB for {tid} (keys={list(files.keys())[:8]})")
            (out_dir / f"{tid}.pdb").write_text(pdb_txt, encoding="utf-8")
            saved += 1
        print(
            f"  [T6 Boltz-2] Saved {saved} dimer PDBs ({min(b0 + batch_size, len(todo))}/{len(todo)}, "
            f"cost=${meta['cost_usd']:.4f})",
            flush=True,
        )


def fetch_foldseek_and_saprot() -> None:
    cache_dir = REPO / "outputs" / "proteinmpnn_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    # 1. Foldseek macOS static binary
    fs_bin = cache_dir / "foldseek" / "bin" / "foldseek"
    if not fs_bin.exists():
        print("\n=== [T5] Downloading Foldseek macOS binary ===", flush=True)
        tgz = cache_dir / "foldseek-osx-universal.tar.gz"
        with httpx.Client(timeout=180.0, follow_redirects=True) as c:
            with c.stream("GET", "https://mmseqs.com/foldseek/foldseek-osx-universal.tar.gz") as r:
                r.raise_for_status()
                with tgz.open("wb") as f:
                    for chunk in r.iter_bytes(chunk_size=1 << 18):
                        f.write(chunk)
        with tarfile.open(tgz, "r:gz") as tf:
            tf.extractall(cache_dir)
        tgz.unlink(missing_ok=True)
        assert fs_bin.exists(), f"Expected foldseek binary at {fs_bin}"
        fs_bin.chmod(0o755)
        print(f"  [T5] Installed Foldseek -> {fs_bin}", flush=True)
    else:
        print(f"[T5] Foldseek already present -> {fs_bin}", flush=True)

    # 2. SaProt 650M AF2 weights via huggingface_hub (skip redundant 2.6GB SaProt_650M_AF2.pt!)
    saprot_dir = cache_dir / "saprot_650m_af2"
    if not (saprot_dir / "config.json").exists() or not any(
        (saprot_dir / fn).exists() for fn in ("model.safetensors", "pytorch_model.bin")
    ):
        print("\n=== [T5] Downloading westlake-repl/SaProt_650M_AF2 (pytorch_model.bin only) ===", flush=True)
        from huggingface_hub import snapshot_download

        snapshot_download(
            repo_id="westlake-repl/SaProt_650M_AF2",
            local_dir=str(saprot_dir),
            ignore_patterns=["*.pt"],
        )
        print(f"  [T5] Downloaded SaProt_650M_AF2 -> {saprot_dir}", flush=True)
    else:
        print(f"[T5] SaProt_650M_AF2 already cached -> {saprot_dir}", flush=True)


def fetch_stabddg() -> None:
    cache_dir = REPO / "outputs" / "proteinmpnn_cache"
    stab_dir = cache_dir / "StaB-ddG"
    if not stab_dir.exists():
        print("\n=== [T6] Cloning LDeng0205/StaB-ddG ===", flush=True)
        subprocess.run(
            ["git", "clone", "--depth", "1", "https://github.com/LDeng0205/StaB-ddG.git", str(stab_dir)],
            check=True,
        )
    else:
        print(f"[T6] StaB-ddG repo already present -> {stab_dir}", flush=True)

    # Verify model_ckpts/stability_finetuned.pt is a real checkpoint (> 1 MB), not a Git LFS pointer
    for fn in ("stability_finetuned.pt", "proteinmpnn.pt"):
        ckpt = stab_dir / "model_ckpts" / fn
        if not ckpt.exists() or ckpt.stat().st_size < 1_000_000:
            print(f"  [T6] Fetching real binary for {fn} via GitHub media/LFS...", flush=True)
            ckpt.parent.mkdir(parents=True, exist_ok=True)
            url = f"https://media.githubusercontent.com/media/LDeng0205/StaB-ddG/main/model_ckpts/{fn}"
            with httpx.Client(timeout=180.0, follow_redirects=True) as c:
                with c.stream("GET", url) as r:
                    r.raise_for_status()
                    with ckpt.open("wb") as f:
                        for chunk in r.iter_bytes(chunk_size=1 << 18):
                            f.write(chunk)
        print(f"  [T6] Checkpoint {fn}: {ckpt.stat().st_size:,} bytes", flush=True)


def submit_and_poll_netgo(version: str = "ng4.0", model_val: str = "NG4.0", poll_timeout_s: float = 900.0) -> Optional[Path]:
    out_dir = REPO / "outputs" / "predictions" / "netgo_raw"
    out_dir.mkdir(parents=True, exist_ok=True)
    result_file = out_dir / f"{version}_results.txt"
    job_meta_file = out_dir / f"{version}_job.json"
    if result_file.exists() and result_file.stat().st_size > 200:
        print(f"[T4 {model_val}] Reusing cached results -> {result_file} ({result_file.stat().st_size:,} bytes)", flush=True)
        return result_file

    m2b = json.loads((REPO / "data" / "stage2b_cafa" / "manifest.json").read_text(encoding="utf-8"))["targets"]
    fasta_str = "\n".join(f">{t['target_id']}\n{t['sequence'].strip().upper()}" for t in m2b) + "\n"
    base_url = f"https://dmiip.sjtu.edu.cn/{version}"

    print(f"\n=== [T4 {model_val}] Submitting {len(m2b)} anonymised FASTA sequences to {base_url} ===", flush=True)
    with httpx.Client(timeout=60.0, follow_redirects=True, verify=False) as c:
        job_id: Optional[str] = None
        if job_meta_file.exists():
            job_id = json.loads(job_meta_file.read_text(encoding="utf-8")).get("job_id")
            if job_id:
                print(f"  [T4 {model_val}] Resuming existing job_id={job_id}", flush=True)

        if not job_id:
            # Fetch landing page first to get any cookies / CSRF token
            r0 = c.get(base_url)
            r0.raise_for_status()
            csrf_m = re.search(r'name=["\']csrfmiddlewaretoken["\']\s+value=["\']([^"\']+)["\']', r0.text)
            data_fields = {
                "fasta": fasta_str,
                "email": "",
                "model": model_val,
            }
            if csrf_m:
                data_fields["csrfmiddlewaretoken"] = csrf_m.group(1)

            r_sub = c.post(
                f"{base_url}/submit/",
                data=data_fields,
                headers={"Referer": base_url},
            )
            print(f"  [T4 {model_val}] Submit HTTP {r_sub.status_code} url={r_sub.url} len={len(r_sub.text)}", flush=True)
            (out_dir / f"{version}_submit_resp.html").write_text(r_sub.text, encoding="utf-8")

            # Extract job ID from URL or HTML (e.g. Job ID: 2026... or /result/<id> or checkID)
            for pat in (
                r"(?:job\s*id|task\s*id|your\s*id|id\s*is)\s*[:：#]?\s*<[^>]*>?\s*([A-Za-z0-9_-]{6,40})",
                r"result/([A-Za-z0-9_-]{6,40})",
                r"download/([A-Za-z0-9_-]{6,40})",
                r"value=[\"']([A-Za-z0-9_-]{8,40})[\"'][^>]*id=[\"']checkID",
                r"(\d{10,20}[A-Za-z0-9_-]*)",
            ):
                m = re.search(pat, r_sub.text, re.I)
                if m:
                    job_id = m.group(1).strip()
                    break
            if not job_id:
                # Try URL path segment
                um = re.search(r"/([A-Za-z0-9_-]{6,40})/?$", str(r_sub.url))
                if um and um.group(1) not in ("submit", "check", version):
                    job_id = um.group(1)
            print(f"  [T4 {model_val}] Extracted job_id={job_id!r}", flush=True)
            job_meta_file.write_text(
                json.dumps({"job_id": job_id, "submit_url": str(r_sub.url), "status_code": r_sub.status_code}, indent=2),
                encoding="utf-8",
            )
            if not job_id:
                print(f"  [T4 {model_val}] Submit response snippet:\n{r_sub.text[:1500]}", flush=True)
                return None

        # Poll /check/ until download link or results appear
        t0 = time.monotonic()
        while time.monotonic() - t0 < poll_timeout_s:
            r_chk = c.post(
                f"{base_url}/check/",
                data={"id": job_id, "model": model_val},
                headers={"Referer": base_url},
            )
            (out_dir / f"{version}_check_resp.html").write_text(r_chk.text, encoding="utf-8")
            # Look for download links (.txt, .zip, .tsv, /download/, /media/, /static/result/)
            dl_links = re.findall(
                r'href=["\']([^"\']*(?:download|result|predict|output|\.txt|\.zip|\.tsv)[^"\']*)["\']',
                r_chk.text,
                re.I,
            )
            dl_links = [
                lnk for lnk in dl_links
                if not any(bad in lnk.lower() for bad in (".css", ".js", "example", "demo", "sample", "help"))
            ]
            if dl_links:
                print(f"  [T4 {model_val}] Found download links: {dl_links[:5]}", flush=True)
                for lnk in dl_links:
                    full_u = lnk if lnk.startswith("http") else f"https://dmiip.sjtu.edu.cn{lnk if lnk.startswith('/') else '/' + version + '/' + lnk}"
                    r_dl = c.get(full_u)
                    if r_dl.status_code == 200 and len(r_dl.content) > 100 and ("GO:" in r_dl.text or r_dl.content[:2] == b"PK"):
                        if r_dl.content[:2] == b"PK":
                            zp = out_dir / f"{version}_results.zip"
                            zp.write_bytes(r_dl.content)
                            print(f"  [T4 {model_val}] Downloaded ZIP -> {zp} ({len(r_dl.content):,} bytes)", flush=True)
                            return zp
                        result_file.write_text(r_dl.text, encoding="utf-8")
                        print(f"  [T4 {model_val}] Downloaded results -> {result_file} ({len(r_dl.text):,} bytes)", flush=True)
                        return result_file

            # Check if GO terms are embedded directly in the check response
            if r_chk.text.count("GO:") >= 50 and "s2b_t001" in r_chk.text:
                result_file.write_text(r_chk.text, encoding="utf-8")
                print(f"  [T4 {model_val}] Captured inline results -> {result_file} ({len(r_chk.text):,} bytes)", flush=True)
                return result_file

            elapsed = time.monotonic() - t0
            status_m = re.search(r"(?:status|state|running|queued|waiting|finished|complete|error|fail)[^<\n]{0,80}", r_chk.text, re.I)
            st_hint = status_m.group(0).strip() if status_m else f"HTTP {r_chk.status_code} len={len(r_chk.text)}"
            print(f"  [T4 {model_val}] Poll {elapsed:.0f}s: {st_hint[:100]}", flush=True)
            time.sleep(15.0)

    print(f"  [T4 {model_val}] Timed out after {poll_timeout_s}s waiting for job_id={job_id}", flush=True)
    return None


def main() -> None:
    from concurrent.futures import ThreadPoolExecutor

    args = set(sys.argv[1:])
    run_all = not args

    def _run_fold() -> None:
        with JapanFoldClient(repo_root=REPO, max_usd_budget=25.0) as jf:
            bal = jf.get_account_balance()
            print(f"[JapanFold] Starting balance: ${bal:.4f}", flush=True)
            fetch_task5_esmfold2(jf)
            fetch_task6_boltz2(jf)
            print(f"[JapanFold] Total spend this run: ${jf.cumulative_usd:.4f}", flush=True)

    def _run_netgo() -> None:
        res4 = submit_and_poll_netgo("ng4.0", "NG4.0", poll_timeout_s=600.0)
        if not res4:
            submit_and_poll_netgo("ng3.0", "NG3.0", poll_timeout_s=600.0)

    if run_all:
        # Run all 4 independent network streams concurrently!
        with ThreadPoolExecutor(max_workers=4) as ex:
            futs = {
                "stabddg": ex.submit(fetch_stabddg),
                "fold": ex.submit(_run_fold),
                "netgo": ex.submit(_run_netgo),
                "saprot": ex.submit(fetch_foldseek_and_saprot),
            }
            for name, fut in futs.items():
                try:
                    fut.result()
                    print(f"[DONE] stream={name}", flush=True)
                except Exception as exc:
                    print(f"[ERROR] stream={name}: {exc}", flush=True)
    else:
        if "stabddg" in args or "weights" in args:
            fetch_stabddg()
        if "saprot" in args or "weights" in args:
            fetch_foldseek_and_saprot()
        if "fold" in args:
            _run_fold()
        if "netgo" in args:
            _run_netgo()


if __name__ == "__main__":
    main()
