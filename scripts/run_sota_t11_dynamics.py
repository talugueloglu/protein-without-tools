#!/usr/bin/env python3
"""Run conformational-ensemble specialist baselines for Task 11 (Conformational Dynamics,
Cryptic Pockets & Apo–Holo Mobility, N = 50):

  1. Boltz-2 5-sample conformational ensemble (JapanFold, MSA + diffusion_samples=5):
     - Kabsch-aligns all 5 independent diffusion samples on the rigid 55% core and computes
       per-residue ensemble C-alpha RMSF (`ensemble_rmsf`)
     - Also records single-structure inverse confidence (`100 - pLDDT`) and the combined
       z-scored dynamics predictor (`z(RMSF) + z(-pLDDT)`)
  2. Reports exact breakdown across `cryptic_pocket`, `domain_motion`, and
     `ood_conformational_switch`, alongside the 1D TOP-IDP disorder & terminus nulls.
"""

from __future__ import annotations

from collections import defaultdict
import io
import json
from pathlib import Path
import sys
from typing import Any, Dict, List, Optional, Tuple
import warnings

from Bio.PDB import MMCIFParser, PDBParser
import numpy as np

warnings.filterwarnings("ignore")

REPO = Path(__file__).resolve().parents[1]
if str(REPO / "src") not in sys.path:
    sys.path.insert(0, str(REPO / "src"))

from protein_without_tools.japanfold_client import JapanFoldClient
from protein_without_tools.pdb_utils import AA3_TO_1
from protein_without_tools.task11_dynamics import DynamicsResidue, grade_dynamics_panel


def kabsch_align(P: np.ndarray, Q: np.ndarray, mask: Optional[np.ndarray] = None) -> np.ndarray:
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


def parse_ca_and_plddt(struct_text: str, is_cif: bool = True) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    parser = MMCIFParser(QUIET=True) if is_cif else PDBParser(QUIET=True)
    try:
        struct = parser.get_structure("pred", io.StringIO(struct_text))
    except Exception:
        return None
    model = next(iter(struct), None)
    if model is None:
        return None
    ca_list: List[np.ndarray] = []
    bf_list: List[float] = []
    chain = next((c for c in model if c.id == "A"), next(iter(model), None))
    if chain is None:
        return None
    for res in chain:
        het, _rseq, _ic = res.id
        rname = res.get_resname().strip().upper()
        if het == " " and rname in AA3_TO_1 and "CA" in res:
            atom = res["CA"]
            ca_list.append(np.asarray(atom.get_coord(), dtype=np.float64))
            bf_list.append(float(atom.get_bfactor()))
    if not ca_list:
        return None
    ca = np.stack(ca_list, axis=0)
    bf = np.array(bf_list, dtype=np.float64)
    if float(np.max(bf)) <= 1.05 and float(np.max(bf)) > 0.0:
        bf = bf * 100.0
    return ca, bf


def compute_ensemble_rmsf(samples: List[np.ndarray]) -> np.ndarray:
    """Iterative rigid-core Kabsch alignment across K conformations -> per-residue CA RMSF (A)."""
    ref = samples[0]
    aligned = [ref]
    for s in samples[1:]:
        s1 = kabsch_align(s, ref)
        d1 = np.linalg.norm(s1 - ref, axis=1)
        core = d1 <= float(np.percentile(d1, 55.0))
        aligned.append(kabsch_align(s, ref, mask=core))
    arr = np.stack(aligned, axis=0)  # (K, L, 3)
    mean_pos = np.mean(arr, axis=0, keepdims=True)
    sq_disp = np.sum((arr - mean_pos) ** 2, axis=-1)  # (K, L)
    return np.sqrt(np.mean(sq_disp, axis=0))


def build_monomer_yaml(seq: str) -> str:
    return (
        "version: 1\n"
        "sequences:\n"
        "  - protein:\n"
        "      id: A\n"
        f"      sequence: {seq}\n"
    )


def main() -> None:
    manifest = json.loads((REPO / "data/task11_dynamics/manifest.json").read_text(encoding="utf-8"))
    targets = manifest["targets"]

    per_target_rmsf: Dict[str, np.ndarray] = {}
    per_target_plddt: Dict[str, np.ndarray] = {}
    total_cost = 0.0

    with JapanFoldClient(max_usd_budget=25.0) as jf:
        print(f"JapanFold balance: ${jf.get_account_balance():.2f}", flush=True)
        print(f"=== Running Boltz-2 5-Sample Conformational Ensemble on Task 11 ({len(targets)} targets) ===", flush=True)

        batch_size = 10
        for b0 in range(0, len(targets), batch_size):
            chunk = targets[b0 : b0 + batch_size]
            payload = {
                "model": "boltz2",
                "targets": [{"name": t["target_id"], "content": build_monomer_yaml(t["sequence"])} for t in chunk],
                "params": {
                    "use_msa_server": True,
                    "recycling_steps": 3,
                    "sampling_steps": 200,
                    "diffusion_samples": 5,
                    "output_format": "cif",
                    "seed": 42,
                },
            }
            meta, files = jf.submit_and_collect(
                "/v1/predictions",
                payload,
                cache_prefix=f"boltz2_t11_ens5_b{b0:03d}",
                download_all=True,
            )
            total_cost += float(meta.get("cost_usd", 0.0))
            ok = 0
            for t in chunk:
                tid = t["target_id"]
                L = int(t["length"])
                ca_samples: List[np.ndarray] = []
                bf_samples: List[np.ndarray] = []
                for fname, raw in sorted(files.items()):
                    low = fname.lower()
                    if tid.lower() not in low:
                        continue
                    if not (low.endswith(".cif") or low.endswith(".pdb")):
                        continue
                    parsed = parse_ca_and_plddt(raw.decode("utf-8", errors="replace"), is_cif=low.endswith(".cif"))
                    if parsed and len(parsed[0]) == L:
                        ca_samples.append(parsed[0])
                        bf_samples.append(parsed[1])
                if len(ca_samples) >= 2:
                    per_target_rmsf[tid] = compute_ensemble_rmsf(ca_samples)
                    per_target_plddt[tid] = np.mean(np.stack(bf_samples, axis=0), axis=0)
                    ok += 1
            print(
                f"  [t11_ens5] batch {b0:02d}..{b0 + len(chunk) - 1:02d}: "
                f"parsed {ok}/{len(chunk)} ensembles (cost=${meta.get('cost_usd', 0.0):.4f})",
                flush=True,
            )

    def _eval_mode(mode: str) -> Dict[str, Any]:
        rows: List[Dict[str, Any]] = []
        by_stratum: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        for t in targets:
            tid = t["target_id"]
            if tid not in per_target_rmsf:
                continue
            rmsf = per_target_rmsf[tid]
            plddt = per_target_plddt[tid]
            panel = [
                DynamicsResidue(
                    res_id=c["res_id"],
                    aa=c["aa"],
                    pos=int(c["pos"]),
                    mobility=float(c["mobility"]),
                    local_idp=float(c["local_idp"]),
                    is_mobile=bool(c["is_mobile"]),
                )
                for c in t["candidates"]
            ]
            if mode == "ensemble_rmsf":
                scores = {c.res_id: float(rmsf[c.pos - 1]) for c in panel}
            elif mode == "inv_plddt":
                scores = {c.res_id: float(100.0 - plddt[c.pos - 1]) for c in panel}
            else:
                r_vals = np.array([rmsf[c.pos - 1] for c in panel], dtype=float)
                p_vals = np.array([100.0 - plddt[c.pos - 1] for c in panel], dtype=float)
                zr = (r_vals - np.mean(r_vals)) / (np.std(r_vals) + 1e-8)
                zp = (p_vals - np.mean(p_vals)) / (np.std(p_vals) + 1e-8)
                scores = {c.res_id: float(zr[i] + zp[i]) for i, c in enumerate(panel)}

            g = grade_dynamics_panel(panel, [], predicted_scores=scores)
            rec = {
                "target_id": tid,
                "stratum": t["stratum"],
                "length": t["length"],
                "auroc": round(g["auroc"], 4),
                "spearman_rho": round(g["spearman_rho"], 4),
                "ndcg_at_5": round(g["ndcg_at_5"], 4),
            }
            rows.append(rec)
            by_stratum[t["stratum"]].append(rec)

        def _agg(rs: List[Dict[str, Any]]) -> Dict[str, Any]:
            if not rs:
                return {"n": 0}
            return {
                "n": len(rs),
                "auroc": round(float(np.mean([r["auroc"] for r in rs])), 4),
                "spearman_rho": round(float(np.mean([r["spearman_rho"] for r in rs])), 4),
                "ndcg_at_5": round(float(np.mean([r["ndcg_at_5"] for r in rs])), 4),
            }

        return {
            "overall": _agg(rows),
            "by_stratum": {k: _agg(v) for k, v in sorted(by_stratum.items())},
            "per_target": rows,
        }

    ens_res = _eval_mode("ensemble_rmsf")
    plddt_res = _eval_mode("inv_plddt")
    comb_res = _eval_mode("combined")

    out = {
        "task": "task11_conformational_dynamics",
        "specialist_model": "Boltz-2 5-Sample Conformational Ensemble (MSA + Diffusion CA-RMSF)",
        "n_evaluated": ens_res["overall"]["n"],
        "n_total": manifest["n_targets"],
        "cost_usd": round(total_cost, 4),
        "overall": ens_res["overall"],
        "by_stratum": ens_res["by_stratum"],
        "modes": {
            "ensemble_rmsf": {"overall": ens_res["overall"], "by_stratum": ens_res["by_stratum"]},
            "inv_plddt": {"overall": plddt_res["overall"], "by_stratum": plddt_res["by_stratum"]},
            "combined_rmsf_plddt": {"overall": comb_res["overall"], "by_stratum": comb_res["by_stratum"]},
        },
        "anti_cheat_null": manifest["anti_cheat_audit"],
        "per_target": ens_res["per_target"],
    }
    out_path = REPO / "outputs/results/baseline_t11_dynamics_boltz2_ensemble.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print("\n=== TASK 11 BOLTZ-2 ENSEMBLE BASELINE COMPLETE ===", flush=True)
    print("  Ensemble RMSF:", ens_res["overall"], flush=True)
    print("  Inv pLDDT:    ", plddt_res["overall"], flush=True)
    print("  Combined:     ", comb_res["overall"], flush=True)
    print("  By stratum:   ", comb_res["by_stratum"], flush=True)


if __name__ == "__main__":
    main()
