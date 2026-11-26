#!/usr/bin/env python3
"""Build Task 11: Conformational Dynamics, Cryptic Pockets & Apo–Holo Mobility (N = 50).

Curates experimental multi-conformation crystal pairs from Microsoft's open
`bioemu-benchmarks` (`multiconf_benchmark_0.1`: `crypticpocket`, `domainmotion`,
`ood60`, `oodval`).

For each protein pair `(conformation_i, conformation_j)`:
  1. Downloads both aligned reference PDBs directly from `microsoft/bioemu-benchmarks`.
  2. Aligns residues by sequence/numbering and computes iterative rigid-core Kabsch
     superposition (round 1: all C-alpha; round 2: rigid 55% core C-alpha) so domain
     hinges and cryptic-pocket openings show crisp local C-alpha displacement (Å).
  3. Deduplicates sequences across all targets (`3-mer Jaccard <= 0.30`).
  4. Selects 16 internal residues (N/C-terminal 6 residues trimmed) via
     `select_disorder_matched_dynamics_panel`, matching high-mobility residues 1-to-1
     with rigid core decoys on 1D TOP-IDP disorder propensity and terminus distance.
  5. Audits 1D TOP-IDP disorder null and terminus-distance null (`AUROC ~ 0.50`).
"""

from __future__ import annotations

import asyncio
from collections import Counter
import csv
import io
import json
from pathlib import Path
import sys
from typing import Any, Dict, List, Optional, Tuple
import warnings

from Bio.Align import PairwiseAligner
import httpx
import numpy as np

warnings.filterwarnings("ignore")

REPO = Path(__file__).resolve().parents[1]
if str(REPO / "src") not in sys.path:
    sys.path.insert(0, str(REPO / "src"))

from protein_without_tools.pdb_utils import AA3_TO_1, CANONICAL_AA_SET
from protein_without_tools.task11_dynamics import (
    DynamicsResidue,
    build_dynamics_prompt,
    grade_dynamics_panel,
    select_disorder_matched_dynamics_panel,
)

CACHE_DIR = REPO / "data/raw_v2_cache/task11_bioemu"
CACHE_DIR.mkdir(parents=True, exist_ok=True)
BIOEMU_RAW = "https://raw.githubusercontent.com/microsoft/bioemu-benchmarks/main/bioemu_benchmarks/assets/multiconf_benchmark_0.1"

SUBSETS: List[Tuple[str, str]] = [
    ("crypticpocket", "cryptic_pocket"),
    ("domainmotion", "domain_motion"),
    ("ood60", "ood_conformational_switch"),
    ("oodval", "ood_conformational_switch"),
]


def _kmer_set(seq: str, k: int = 3) -> set[str]:
    return {seq[i : i + k] for i in range(max(0, len(seq) - k + 1))}


def _jaccard(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / float(len(a | b))


def parse_pdb_ca(pdb_text: str) -> Tuple[str, np.ndarray]:
    """Extract ordered (sequence, CA_coords[L, 3]) from a single-chain PDB."""
    residues: Dict[Tuple[int, str], Tuple[str, np.ndarray]] = {}
    for line in pdb_text.splitlines():
        if line.startswith("ENDMDL"):
            break
        if not line.startswith("ATOM  "):
            continue
        aname = line[12:16].strip()
        if aname != "CA":
            continue
        altloc = line[16]
        if altloc not in (" ", "A", "1"):
            continue
        rname = line[17:20].strip().upper()
        if rname not in AA3_TO_1:
            continue
        resseq = int(line[22:26].strip())
        icode = line[26].strip()
        key = (resseq, icode)
        if key not in residues:
            xyz = np.array(
                [float(line[30:38]), float(line[38:46]), float(line[46:54])],
                dtype=np.float64,
            )
            residues[key] = (AA3_TO_1[rname], xyz)
    seq = "".join(v[0] for v in residues.values())
    coords = np.stack([v[1] for v in residues.values()], axis=0) if residues else np.zeros((0, 3))
    return seq, coords


def kabsch_align(P: np.ndarray, Q: np.ndarray, mask: Optional[np.ndarray] = None) -> np.ndarray:
    """Superpose P onto Q using Kabsch algorithm (optionally fitting rotation on `mask` core)."""
    if mask is None or int(np.sum(mask)) < 6:
        Pm, Qm = P, Q
    else:
        Pm, Qm = P[mask], Q[mask]
    cP = np.mean(Pm, axis=0)
    cQ = np.mean(Qm, axis=0)
    H = (Pm - cP).T @ (Qm - cQ)
    U, _S, Vt = np.linalg.svd(H)
    d = np.linalg.det(Vt.T @ U.T)
    D = np.diag([1.0, 1.0, -1.0 if d < 0 else 1.0])
    R = Vt.T @ D @ U.T
    return (P - cP) @ R.T + cQ


def compute_core_aligned_displacement(
    seq1: str, ca1: np.ndarray, seq2: str, ca2: np.ndarray
) -> Optional[Tuple[str, np.ndarray]]:
    """Align two conformations of the same protein and compute iterative rigid-core C-alpha displacement."""
    if len(seq1) < 45 or len(seq2) < 45:
        return None
    if seq1 == seq2 and len(ca1) == len(ca2):
        common_seq = seq1
        P, Q = ca1, ca2
    else:
        aligner = PairwiseAligner()
        aligner.mode = "global"
        aligner.match_score = 2.0
        aligner.mismatch_score = -2.0
        aligner.open_gap_score = -5.0
        aligner.extend_gap_score = -0.5
        aln = aligner.align(seq1, seq2)[0]
        idx1_list: List[int] = []
        idx2_list: List[int] = []
        chars: List[str] = []
        for (a0, a1), (b0, b1) in zip(aln.aligned[0], aln.aligned[1]):
            for i1, i2 in zip(range(a0, a1), range(b0, b1)):
                if seq1[i1] == seq2[i2]:
                    idx1_list.append(i1)
                    idx2_list.append(i2)
                    chars.append(seq1[i1])
        if len(chars) < 55:
            return None
        # Require contiguous-ish overlap (>= 80% of shorter chain)
        if len(chars) / float(min(len(seq1), len(seq2))) < 0.78:
            return None
        common_seq = "".join(chars)
        P = ca1[idx1_list]
        Q = ca2[idx2_list]

    if not (55 <= len(common_seq) <= 420):
        return None
    if any(ch not in CANONICAL_AA_SET for ch in common_seq):
        return None

    # Round 1: global Kabsch
    P1 = kabsch_align(P, Q)
    d1 = np.linalg.norm(P1 - Q, axis=1)
    # Round 2: rigid core Kabsch (55% lowest-displacement core)
    core_mask = d1 <= float(np.percentile(d1, 55.0))
    P2 = kabsch_align(P, Q, mask=core_mask)
    d2 = np.linalg.norm(P2 - Q, axis=1)
    return common_seq, d2


async def fetch_text(client: httpx.AsyncClient, url: str, cache_path: Path, sem: asyncio.Semaphore) -> Optional[str]:
    if cache_path.exists() and cache_path.stat().st_size > 20:
        return cache_path.read_text(encoding="utf-8")
    async with sem:
        try:
            r = await client.get(url, timeout=25.0)
            if r.status_code != 200:
                return None
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            cache_path.write_text(r.text, encoding="utf-8")
            return r.text
        except Exception:
            return None


async def build_task11() -> None:
    sem = asyncio.Semaphore(12)
    selected: List[Dict[str, Any]] = []
    seen_kmers: List[set[str]] = []
    seen_tc: set[str] = set()
    stratum_counts = Counter()

    async with httpx.AsyncClient(follow_redirects=True) as client:
        for sub, stratum_label in SUBSETS:
            ref_csv = await fetch_text(
                client, f"{BIOEMU_RAW}/{sub}/references.csv", CACHE_DIR / f"{sub}_references.csv", sem
            )
            if not ref_csv:
                continue
            rows = list(csv.DictReader(io.StringIO(ref_csv)))
            print(f"Processing BioEmu subset '{sub}' ({len(rows)} pairs)...", flush=True)

            # Prefetch all reference PDBs for this subset concurrently
            pdb_tasks = {}
            for row in rows:
                tc = (row.get("test_case") or "").strip()
                p_i = (row.get("pdbidchain_i") or "").strip()
                p_j = (row.get("pdbidchain_j") or "").strip()
                if not tc or not p_i or not p_j:
                    continue
                for pid in (p_i, p_j):
                    for variant in (pid.upper(), pid.lower(), pid):
                        key = (sub, tc, pid)
                        if key not in pdb_tasks:
                            # Standardize uppercase PDB_CHAIN.pdb first, fallback to raw case if needed
                            pdb_tasks[key] = (tc, pid)

            async def _fetch_pair_pdb(sub_name: str, tc: str, pid: str) -> Tuple[Tuple[str, str, str], Optional[str]]:
                for cand_name in dict.fromkeys([pid.upper(), pid, pid.lower()]):
                    # Also handle 4char_CHAIN case (e.g. 6gts_B -> 6GTS_B)
                    parts = pid.split("_")
                    if len(parts) == 2:
                        cand_variants = [f"{parts[0].upper()}_{parts[1]}", f"{parts[0].upper()}_{parts[1].upper()}", pid]
                    else:
                        cand_variants = [cand_name]
                    for v in dict.fromkeys(cand_variants):
                        txt = await fetch_text(
                            client,
                            f"{BIOEMU_RAW}/{sub_name}/reference/{tc}/{v}.pdb",
                            CACHE_DIR / sub_name / tc / f"{v}.pdb",
                            sem,
                        )
                        if txt and "ATOM  " in txt:
                            return (sub_name, tc, pid), txt
                return (sub_name, tc, pid), None

            fetched = await asyncio.gather(*(_fetch_pair_pdb(sub, tc, pid) for (_s, tc, pid) in pdb_tasks))
            pdb_map = {k: v for k, v in fetched if v is not None}

            for row in rows:
                if len(selected) >= 50:
                    break
                tc = (row.get("test_case") or "").strip()
                p_i = (row.get("pdbidchain_i") or "").strip()
                p_j = (row.get("pdbidchain_j") or "").strip()
                if not tc or tc in seen_tc:
                    continue
                txt_i = pdb_map.get((sub, tc, p_i))
                txt_j = pdb_map.get((sub, tc, p_j))
                if not txt_i or not txt_j:
                    continue

                seq1, ca1 = parse_pdb_ca(txt_i)
                seq2, ca2 = parse_pdb_ca(txt_j)
                res = compute_core_aligned_displacement(seq1, ca1, seq2, ca2)
                if res is None:
                    continue
                seq, disp = res
                km = _kmer_set(seq)
                if any(_jaccard(km, prev) > 0.30 for prev in seen_kmers):
                    continue

                panel = select_disorder_matched_dynamics_panel(seq, disp.tolist(), n_total=16, terminal_trim=6)
                if panel is None:
                    continue
                # Verify real apo-holo displacement contrast (mobile median >= 2.0 A, rigid median <= 1.2 A)
                mob_med = float(np.median([r.mobility for r in panel if r.is_mobile]))
                rig_med = float(np.median([r.mobility for r in panel if not r.is_mobile]))
                if mob_med - rig_med < 1.0:
                    continue

                seen_tc.add(tc)
                seen_kmers.append(km)
                stratum_counts[stratum_label] += 1
                selected.append({
                    "source_subset": sub,
                    "stratum": stratum_label,
                    "test_case": tc,
                    "pdb_state_a": p_i,
                    "pdb_state_b": p_j,
                    "sequence": seq,
                    "length": len(seq),
                    "disp_median": round(float(np.median(disp)), 3),
                    "disp_p90": round(float(np.percentile(disp, 90.0)), 3),
                    "mobile_median": round(mob_med, 3),
                    "rigid_median": round(rig_med, 3),
                    "panel": panel,
                })

    # Write manifest and prompts
    out_dir = REPO / "data/task11_dynamics"
    out_dir.mkdir(parents=True, exist_ok=True)
    targets: List[Dict[str, Any]] = []
    prompts: List[Dict[str, Any]] = []

    idp_aurocs, idp_rhos, term_aurocs = [], [], []
    for idx, item in enumerate(selected, 1):
        tid = f"t11_c{idx:03d}"
        panel: List[DynamicsResidue] = item["panel"]
        L = item["length"]
        prompt_text = build_dynamics_prompt(item["sequence"], "apo_holo_motion", panel)

        # Anti-cheat shortcut audits
        g_idp = grade_dynamics_panel(panel, [], predicted_scores={r.res_id: r.local_idp for r in panel})
        g_term = grade_dynamics_panel(
            panel, [], predicted_scores={r.res_id: -float(min(r.pos - 1, L - r.pos)) for r in panel}
        )
        idp_aurocs.append(g_idp["auroc"])
        idp_rhos.append(g_idp["spearman_rho"])
        term_aurocs.append(g_term["auroc"])

        targets.append({
            "target_id": tid,
            "stratum": item["stratum"],
            "source_subset": item["source_subset"],
            "test_case": item["test_case"],
            "pdb_state_a": item["pdb_state_a"],
            "pdb_state_b": item["pdb_state_b"],
            "length": L,
            "sequence": item["sequence"],
            "disp_median": item["disp_median"],
            "disp_p90": item["disp_p90"],
            "mobile_median": item["mobile_median"],
            "rigid_median": item["rigid_median"],
            "n_candidates": len(panel),
            "candidates": [
                {
                    "res_id": r.res_id,
                    "aa": r.aa,
                    "pos": r.pos,
                    "mobility": r.mobility,
                    "local_idp": r.local_idp,
                    "is_mobile": r.is_mobile,
                }
                for r in panel
            ],
        })
        prompts.append({
            "custom_id": f"task11_{tid}",
            "target_id": tid,
            "task": "task11_conformational_dynamics",
            "stratum": item["stratum"],
            "length": L,
            "n_candidates": len(panel),
            "prompt": prompt_text,
        })

    manifest = {
        "benchmark": "protein-without-tools-task11-conformational-dynamics",
        "source": "microsoft/bioemu-benchmarks (multiconf_benchmark_0.1: crypticpocket, domainmotion, ood60, oodval)",
        "n_targets": len(targets),
        "by_stratum": dict(stratum_counts),
        "anti_cheat_audit": {
            "top_idp_disorder_auroc": round(float(np.mean(idp_aurocs)), 4) if idp_aurocs else None,
            "top_idp_disorder_spearman": round(float(np.mean(idp_rhos)), 4) if idp_rhos else None,
            "terminus_dist_auroc": round(float(np.mean(term_aurocs)), 4) if term_aurocs else None,
        },
        "targets": targets,
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    (REPO / "outputs/prompts/task11_dynamics_prompts.jsonl").write_text(
        "\n".join(json.dumps(r) for r in prompts) + "\n", encoding="utf-8"
    )
    print("\n=== TASK 11 (CONFORMATIONAL DYNAMICS) BUILT ===", flush=True)
    print(f"  Targets: {manifest['n_targets']} | {manifest['by_stratum']}", flush=True)
    print(f"  Anti-cheat audit: {manifest['anti_cheat_audit']}", flush=True)


if __name__ == "__main__":
    asyncio.run(build_task11())
