"""JapanFold API client with SHA256 disk caching, idempotency keys, and 3D coordinate/PAE parsing."""

from __future__ import annotations

import hashlib
import io
import json
import os
from pathlib import Path
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple
import uuid
import zipfile

from dotenv import load_dotenv
import httpx
import numpy as np

from protein_without_tools.pdb_utils import AA3_TO_1, compute_kabsch_sander_q3, compute_virtual_cb


def parse_pdb_chain_with_bfactor(
    pdb_text: str,
    chain_id: str = "A",
) -> Dict[str, Any]:
    """Parse a single chain from PDB text, returning sequence, N/CA/C/O/CB coords, Q3, and CA B-factors (pLDDT)."""
    residues: Dict[Tuple[int, str], Dict[str, Tuple[str, np.ndarray, float]]] = {}
    for line in pdb_text.splitlines():
        if line.startswith("ENDMDL"):
            break
        if not line.startswith(("ATOM  ", "HETATM")):
            continue
        res_name = line[17:20].strip()
        if line.startswith("HETATM") and res_name != "MSE":
            continue
        if res_name not in AA3_TO_1:
            continue
        altloc = line[16]
        if altloc not in (" ", "A", "1"):
            continue
        ch = line[21].strip() or "A"
        if ch != chain_id:
            continue
        atom_name = line[12:16].strip()
        if atom_name not in ("N", "CA", "C", "O", "CB"):
            continue
        res_seq = int(line[22:26].strip())
        icode = line[26].strip()
        xyz = np.array(
            [float(line[30:38]), float(line[38:46]), float(line[46:54])],
            dtype=np.float64,
        )
        bfac_str = line[60:66].strip()
        bfac = float(bfac_str) if bfac_str else 0.0
        key = (res_seq, icode)
        if key not in residues:
            residues[key] = {}
        if atom_name not in residues[key]:
            residues[key][atom_name] = (res_name, xyz, bfac)

    seq_chars: List[str] = []
    n_l, ca_l, c_l, o_l, cb_l, b_l = [], [], [], [], [], []
    for _k, atoms in residues.items():
        if not all(a in atoms for a in ("N", "CA", "C", "O")):
            continue
        aa1 = AA3_TO_1[atoms["CA"][0]]
        n_xyz = atoms["N"][1]
        ca_xyz = atoms["CA"][1]
        c_xyz = atoms["C"][1]
        o_xyz = atoms["O"][1]
        ca_b = atoms["CA"][2]
        if aa1 == "G":
            cb_xyz = ca_xyz.copy()
        elif "CB" in atoms:
            cb_xyz = atoms["CB"][1]
        else:
            cb_xyz = compute_virtual_cb(n_xyz, ca_xyz, c_xyz)
        seq_chars.append(aa1)
        n_l.append(n_xyz)
        ca_l.append(ca_xyz)
        c_l.append(c_xyz)
        o_l.append(o_xyz)
        cb_l.append(cb_xyz)
        b_l.append(ca_b)

    if not seq_chars:
        raise ValueError(f"Chain {chain_id!r} not found or empty in PDB text")

    seq = "".join(seq_chars)
    coords_n = np.stack(n_l, axis=0)
    coords_ca = np.stack(ca_l, axis=0)
    coords_c = np.stack(c_l, axis=0)
    coords_o = np.stack(o_l, axis=0)
    coords_cb = np.stack(cb_l, axis=0)
    bfactors = np.array(b_l, dtype=np.float64)
    # Normalize pLDDT to [0, 100] if engine wrote [0, 1]
    plddt_res = bfactors * 100.0 if float(np.max(bfactors)) <= 1.05 and float(np.max(bfactors)) > 0.0 else bfactors
    q3 = compute_kabsch_sander_q3(seq, coords_n, coords_ca, coords_c, coords_o)
    return {
        "sequence": seq,
        "coords_n": coords_n,
        "coords_ca": coords_ca,
        "coords_c": coords_c,
        "coords_o": coords_o,
        "coords_cb": coords_cb,
        "plddt_per_res": plddt_res,
        "plddt_mean": float(np.mean(plddt_res)),
        "q3": q3,
    }


class JapanFoldClient:
    """Synchronous/batch client for JapanFold structure prediction & protein design."""

    def __init__(
        self,
        repo_root: Optional[Path] = None,
        cache_dir: Optional[Path] = None,
        max_usd_budget: float = 90.0,
    ) -> None:
        self.repo_root = repo_root or Path(__file__).resolve().parents[2]
        load_dotenv(self.repo_root / ".env")
        self.api_key = os.getenv("JAPANFOLD_API_KEY", "").strip()
        if not self.api_key:
            raise RuntimeError("JAPANFOLD_API_KEY missing from environment / .env")
        self.base_url = os.getenv("JAPANFOLD_BASE_URL", "https://api.japanfold.aiand.com").rstrip("/")
        self.cache_dir = cache_dir or (self.repo_root / "outputs" / "japanfold_cache")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.max_usd_budget = max_usd_budget
        self.cumulative_usd = 0.0
        self._client = httpx.Client(
            timeout=httpx.Timeout(95.0, connect=20.0),
            follow_redirects=True,
            headers={"Authorization": f"Bearer {self.api_key}"},
        )

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> "JapanFoldClient":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()

    def _request(
        self,
        method: str,
        path: str,
        *,
        json_body: Optional[Dict[str, Any]] = None,
        headers: Optional[Dict[str, str]] = None,
        max_retries: int = 6,
    ) -> httpx.Response:
        url = path if path.startswith("http") else f"{self.base_url}{path}"
        hdrs = dict(headers or {})
        for attempt in range(max_retries):
            try:
                resp = self._client.request(method, url, json=json_body, headers=hdrs)
            except httpx.ConnectError as exc:
                if "Errno 8" in str(exc) or "nodename nor servname" in str(exc):
                    raise
                if attempt + 1 >= max_retries:
                    raise
                time.sleep(min(2.0 ** attempt, 15.0))
                continue
            except httpx.TransportError:
                if attempt + 1 >= max_retries:
                    raise
                time.sleep(min(2.0 ** attempt, 15.0))
                continue

            if resp.status_code in (429, 502, 503, 504):
                retry_after = float(resp.headers.get("Retry-After", min(3.0 * (attempt + 1), 20.0)))
                if attempt + 1 >= max_retries:
                    resp.raise_for_status()
                time.sleep(retry_after)
                continue
            return resp
        raise RuntimeError(f"Unreachable retry loop for {method} {url}")

    def get_account_balance(self) -> float:
        r = self._request("GET", "/v1/account")
        r.raise_for_status()
        return float(r.json().get("balance", 0.0))

    def wait_for_job(self, job_id: str, timeout_s: float = 1800.0) -> Dict[str, Any]:
        t0 = time.monotonic()
        while True:
            r = self._request("GET", f"/v1/jobs/{job_id}?wait=60")
            r.raise_for_status()
            job = r.json()
            st = job.get("status")
            if st in ("succeeded", "failed", "canceled"):
                return job
            if time.monotonic() - t0 > timeout_s:
                raise TimeoutError(f"JapanFold job {job_id} timed out after {timeout_s}s (status={st})")
            if job.get("wait_refused"):
                time.sleep(3.0)

    def get_job_results(self, job_id: str) -> Dict[str, Any]:
        r = self._request("GET", f"/v1/jobs/{job_id}/results")
        r.raise_for_status()
        return r.json()

    def get_job_usage_usd(self, job_id: str) -> float:
        r = self._request("GET", f"/v1/jobs/{job_id}/usage")
        if r.status_code != 200:
            return 0.0
        u = r.json()
        micros = u.get("amount_micros")
        if micros is not None:
            return float(micros) / 1_000_000.0
        return float(u.get("amount") or 0.0)

    def download_artifact_bytes(self, job_id: str, rel_path: str) -> bytes:
        clean = rel_path.lstrip("/")
        r = self._request("GET", f"/v1/jobs/{job_id}/artifacts/{clean}")
        r.raise_for_status()
        return r.content

    def download_archive_files(self, job_id: str) -> Dict[str, bytes]:
        r = self._request("GET", f"/v1/jobs/{job_id}/archive")
        r.raise_for_status()
        out: Dict[str, bytes] = {}
        with zipfile.ZipFile(io.BytesIO(r.content), "r") as zf:
            for name in zf.namelist():
                if not name.endswith("/"):
                    out[name] = zf.read(name)
        return out

    def submit_and_collect(
        self,
        endpoint: str,
        payload: Dict[str, Any],
        cache_prefix: str = "job",
        download_all: bool = True,
    ) -> Tuple[Dict[str, Any], Dict[str, bytes]]:
        """Submit `/v1/predictions` or `/v1/designs` with deterministic disk caching."""
        canon = json.dumps({"endpoint": endpoint, "payload": payload}, sort_keys=True)
        digest = hashlib.sha256(canon.encode("utf-8")).hexdigest()[:24]
        cdir = self.cache_dir / f"{cache_prefix}_{digest}"
        meta_path = cdir / "_meta.json"
        files_dir = cdir / "files"

        if meta_path.exists():
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            files: Dict[str, bytes] = {}
            if files_dir.exists():
                for fp in files_dir.rglob("*"):
                    if fp.is_file():
                        files[str(fp.relative_to(files_dir))] = fp.read_bytes()
            return meta, files

        if self.cumulative_usd >= self.max_usd_budget:
            raise RuntimeError(
                f"JapanFold cumulative spend ${self.cumulative_usd:.2f} reached cap ${self.max_usd_budget:.2f}"
            )

        idem_key = str(uuid.uuid4())
        r = self._request("POST", endpoint, json_body=payload, headers={"Idempotency-Key": idem_key})
        if r.status_code not in (200, 201, 202):
            raise RuntimeError(f"POST {endpoint} failed ({r.status_code}): {r.text[:600]}")
        job_init = r.json()
        job_id = job_init["id"]
        job_final = self.wait_for_job(job_id)
        if job_final.get("status") != "succeeded":
            raise RuntimeError(f"JapanFold job {job_id} ended with status={job_final.get('status')}: {job_final.get('error')}")

        results = self.get_job_results(job_id)
        cost_usd = self.get_job_usage_usd(job_id)
        self.cumulative_usd += cost_usd

        files = {}
        if download_all:
            artifacts = results.get("artifacts") or []
            if len(artifacts) > 12:
                files = self.download_archive_files(job_id)
            else:
                for art in artifacts:
                    ap = art.get("path")
                    if ap:
                        files[ap] = self.download_artifact_bytes(job_id, ap)

        cdir.mkdir(parents=True, exist_ok=True)
        files_dir.mkdir(parents=True, exist_ok=True)
        for rel_p, data in files.items():
            dst = files_dir / rel_p
            dst.parent.mkdir(parents=True, exist_ok=True)
            dst.write_bytes(data)

        meta = {
            "job_id": job_id,
            "cost_usd": cost_usd,
            "job": job_final,
            "results": results,
        }
        meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
        return meta, files


AA1_TO_3: Dict[str, str] = {v: k for k, v in AA3_TO_1.items() if k != "MSE"}


def format_clean_two_chain_pdb(
    ch_a: Dict[str, Any],
    ch_b: Dict[str, Any],
) -> str:
    """Format clean N/CA/C/O/CB two-chain PDB (`Chain A` + `Chain B`) for `af2ig` initial-guess."""
    lines: List[str] = []
    serial = 1
    for cid, ch in (("A", ch_a), ("B", ch_b)):
        seq = ch["sequence"]
        for r_idx, aa1 in enumerate(seq, start=1):
            res3 = AA1_TO_3.get(aa1, "ALA")
            atoms = [
                ("N", ch["coords_n"][r_idx - 1], "N"),
                ("CA", ch["coords_ca"][r_idx - 1], "C"),
                ("C", ch["coords_c"][r_idx - 1], "C"),
                ("O", ch["coords_o"][r_idx - 1], "O"),
            ]
            if aa1 != "G":
                atoms.append(("CB", ch["coords_cb"][r_idx - 1], "C"))
            for aname, xyz, elem in atoms:
                lines.append(
                    f"ATOM  {serial:5d}  {aname:<3s} {res3:>3s} {cid}{r_idx:4d}    "
                    f"{xyz[0]:8.3f}{xyz[1]:8.3f}{xyz[2]:8.3f}  1.00 80.00           {elem:>2s}"
                )
                serial += 1
        lines.append("TER")
    lines.append("END")
    return "\n".join(lines) + "\n"


def find_pdb_for_target(files: Dict[str, bytes], name: str, idx0: int) -> Optional[str]:
    """Locate `<name>.pdb` (or `target_{idx0+1}.pdb` fallback) in JapanFold job artifacts."""
    candidates = [
        f"{name}.pdb",
        f"target_{idx0 + 1}.pdb",
    ]
    for c in candidates:
        if c in files:
            return files[c].decode("utf-8", errors="replace")
    for k, v in files.items():
        if k.endswith(f"/{name}.pdb") or k.endswith(f"{name}_model_0.pdb"):
            return v.decode("utf-8", errors="replace")
    # Single-target fallback
    pdb_keys = [k for k in files if k.endswith(".pdb")]
    if len(pdb_keys) == 1 and idx0 == 0:
        return files[pdb_keys[0]].decode("utf-8", errors="replace")
    return None

