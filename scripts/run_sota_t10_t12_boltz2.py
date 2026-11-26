#!/usr/bin/env python3
"""Run uncontaminated Boltz-2 (MSA + all-atom co-folding) SOTA specialist baselines
for Task 10 (Small-Molecule, Cofactor & Metal Binding-Site Ranking, N=60) and
Task 12 (Protein–DNA / RNA Interface Ranking, N=50) via JapanFold API.

All 110 complexes have RCSB initial_release_date >= 2026-05-01 (strictly post-cutoff).
For each complex, Boltz-2 co-folds the protein sequence with the exact ligand
(canonical SMILES for drugs/cofactors, CCD code for catalytic metal ions) or nucleic-acid
partner (dsDNA duplex / RNA strand), and ranks the candidate residues purely by
negative predicted minimum heavy-atom distance (-d_pred) to the ligand / nucleic acid.
"""

from __future__ import annotations

import argparse
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
from protein_without_tools.task10_ligand_site import SiteCandidate, grade_site_ranking_panel


_DNA_COMP = str.maketrans("ACGTacgt", "TGCAtgca")


def reverse_complement_dna(seq: str) -> str:
    return seq.translate(_DNA_COMP)[::-1]


def build_t10_boltz2_yaml(target: Dict[str, Any]) -> str:
    seq = target["sequence"]
    stratum = target["stratum"]
    comp_id = target["ligand_comp_id"]
    desc = target["ligand_descriptor"]
    if stratum == "metal_ion":
        lig_block = f"  - ligand:\n      id: B\n      ccd: {comp_id}\n"
    else:
        # Quote SMILES safely for YAML
        smiles_safe = desc.replace("'", "''")
        lig_block = f"  - ligand:\n      id: B\n      smiles: '{smiles_safe}'\n"
    return (
        "version: 1\n"
        "sequences:\n"
        "  - protein:\n"
        "      id: A\n"
        f"      sequence: {seq}\n"
        + lig_block
    )


def build_t12_boltz2_yaml(target: Dict[str, Any]) -> str:
    seq = target["sequence"]
    na_seq = target["na_sequence"]
    stratum = target["stratum"]
    if stratum == "dsDNA":
        # Clean non-ACGT chars to A/C/G/T just in case
        clean_dna = "".join(c if c in "ACGT" else "A" for c in na_seq.upper())
        rc = reverse_complement_dna(clean_dna)
        na_block = (
            "  - dna:\n"
            "      id: B\n"
            f"      sequence: {clean_dna}\n"
            "  - dna:\n"
            "      id: C\n"
            f"      sequence: {rc}\n"
        )
    else:
        clean_rna = "".join(c if c in "ACGU" else ("U" if c == "T" else "A") for c in na_seq.upper())
        na_block = (
            "  - rna:\n"
            "      id: B\n"
            f"      sequence: {clean_rna}\n"
        )
    return (
        "version: 1\n"
        "sequences:\n"
        "  - protein:\n"
        "      id: A\n"
        f"      sequence: {seq}\n"
        + na_block
    )


def parse_predicted_min_dists(struct_text: str, is_cif: bool = True) -> Optional[List[float]]:
    """Compute per-residue min heavy-atom distance from protein chain A to all non-A chains (B, C, ...)."""
    parser = MMCIFParser(QUIET=True) if is_cif else PDBParser(QUIET=True)
    try:
        struct = parser.get_structure("pred", io.StringIO(struct_text))
    except Exception:
        return None
    model = next(iter(struct), None)
    if model is None:
        return None

    chain_a_res: List[np.ndarray] = []
    partner_atoms: List[np.ndarray] = []
    for chain in model:
        if chain.id == "A":
            for res in chain:
                het, _resseq, _icode = res.id
                rname = res.get_resname().strip().upper()
                if het == " " and rname in AA3_TO_1:
                    atoms = [a.get_coord() for a in res.get_atoms() if a.element != "H"]
                    if atoms:
                        chain_a_res.append(np.array(atoms, dtype=float))
        else:
            for res in chain:
                for a in res.get_atoms():
                    if a.element != "H":
                        partner_atoms.append(a.get_coord())

    if not chain_a_res or not partner_atoms:
        return None
    p_arr = np.array(partner_atoms, dtype=float)
    dists: List[float] = []
    for rc in chain_a_res:
        diff = rc[:, None, :] - p_arr[None, :, :]
        d = float(np.sqrt(np.min(np.sum(diff * diff, axis=-1))))
        dists.append(d)
    return dists


def _find_target_structure_file(files: Dict[str, bytes], tid: str) -> Optional[Tuple[str, bytes]]:
    # Prefer model_0 .cif or .pdb matching target name
    cands = []
    for fname, raw in files.items():
        low = fname.lower()
        if not (low.endswith(".cif") or low.endswith(".pdb")):
            continue
        if tid.lower() in low:
            rank = 0 if "model_0" in low else 1
            cands.append((rank, fname, raw))
    if not cands:
        return None
    cands.sort(key=lambda x: x[0])
    return cands[0][1], cands[0][2]


def run_boltz2_batch(
    jf: JapanFoldClient,
    targets: List[Dict[str, Any]],
    yaml_fn,
    cache_tag: str,
    batch_size: int = 12,
) -> Tuple[Dict[str, List[float]], float]:
    """Run Boltz-2 on JapanFold in batches and return target_id -> per-residue predicted min dists."""
    pred_dists: Dict[str, List[float]] = {}
    total_cost = 0.0

    for b0 in range(0, len(targets), batch_size):
        chunk = targets[b0 : b0 + batch_size]
        payload_targets = [{"name": t["target_id"], "content": yaml_fn(t)} for t in chunk]
        payload = {
            "model": "boltz2",
            "targets": payload_targets,
            "params": {
                "use_msa_server": True,
                "recycling_steps": 3,
                "sampling_steps": 200,
                "diffusion_samples": 1,
                "output_format": "cif",
                "seed": 42,
            },
        }
        try:
            meta, files = jf.submit_and_collect(
                "/v1/predictions",
                payload,
                cache_prefix=f"boltz2_{cache_tag}_b{b0:03d}",
                download_all=True,
            )
            total_cost += float(meta.get("cost_usd", 0.0))
            ok_in_batch = 0
            for t in chunk:
                tid = t["target_id"]
                found = _find_target_structure_file(files, tid)
                if not found:
                    continue
                fname, raw = found
                dists = parse_predicted_min_dists(raw.decode("utf-8", errors="replace"), is_cif=fname.lower().endswith(".cif"))
                if dists and len(dists) == t["length"]:
                    pred_dists[tid] = dists
                    ok_in_batch += 1
            print(
                f"  [{cache_tag}] batch {b0:02d}..{b0 + len(chunk) - 1:02d}: "
                f"parsed {ok_in_batch}/{len(chunk)} structures (cost=${meta.get('cost_usd', 0.0):.4f})",
                flush=True,
            )
        except Exception as exc:
            print(f"  [{cache_tag}] batch {b0} failed ({exc}); retrying targets individually...", flush=True)
            for t in chunk:
                tid = t["target_id"]
                single_payload = {
                    "model": "boltz2",
                    "targets": [{"name": tid, "content": yaml_fn(t)}],
                    "params": {
                        "use_msa_server": True,
                        "recycling_steps": 3,
                        "sampling_steps": 200,
                        "diffusion_samples": 1,
                        "output_format": "cif",
                        "seed": 42,
                    },
                }
                try:
                    meta, files = jf.submit_and_collect(
                        "/v1/predictions",
                        single_payload,
                        cache_prefix=f"boltz2_{cache_tag}_{tid}",
                        download_all=True,
                    )
                    total_cost += float(meta.get("cost_usd", 0.0))
                    found = _find_target_structure_file(files, tid)
                    if found:
                        fname, raw = found
                        dists = parse_predicted_min_dists(
                            raw.decode("utf-8", errors="replace"), is_cif=fname.lower().endswith(".cif")
                        )
                        if dists and len(dists) == t["length"]:
                            pred_dists[tid] = dists
                except Exception as exc2:
                    print(f"    [WARN] {tid} failed: {exc2}", flush=True)

    return pred_dists, total_cost


def evaluate_and_save(
    manifest: Dict[str, Any],
    pred_dists: Dict[str, List[float]],
    out_path: Path,
    task_name: str,
    cost_usd: float,
) -> Dict[str, Any]:
    per_target: List[Dict[str, Any]] = []
    by_stratum: Dict[str, List[Dict[str, Any]]] = defaultdict(list)

    for t in manifest["targets"]:
        tid = t["target_id"]
        dists = pred_dists.get(tid)
        if not dists:
            continue
        panel = [
            SiteCandidate(
                res_id=c["res_id"],
                aa=c["aa"],
                pos=int(c["pos"]),
                min_dist_angstrom=float(c["min_dist_angstrom"]),
                is_contact=bool(c["is_contact"]),
            )
            for c in t["candidates"]
        ]
        # Higher score = closer predicted distance to ligand / nucleic acid
        scores = {c.res_id: -float(dists[c.pos - 1]) for c in panel}
        g = grade_site_ranking_panel(panel, [], predicted_scores=scores)
        rec = {
            "target_id": tid,
            "pdb_id": t["pdb_id"],
            "release_date": t["release_date"],
            "stratum": t["stratum"],
            "length": t["length"],
            "auroc": round(g["auroc"], 4),
            "same_aa_win_rate": round(g["same_aa_win_rate"], 4),
            "spearman_rho_prox": round(g["spearman_rho_prox"], 4),
            "ndcg_at_5": round(g["ndcg_at_5"], 4),
            "top5_precision": round(g["top5_precision"], 4),
        }
        per_target.append(rec)
        by_stratum[t["stratum"]].append(rec)

    def _agg(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
        if not rows:
            return {"n": 0}
        return {
            "n": len(rows),
            "auroc": round(float(np.mean([r["auroc"] for r in rows])), 4),
            "same_aa_win_rate": round(float(np.mean([r["same_aa_win_rate"] for r in rows])), 4),
            "spearman_rho_prox": round(float(np.mean([r["spearman_rho_prox"] for r in rows])), 4),
            "ndcg_at_5": round(float(np.mean([r["ndcg_at_5"] for r in rows])), 4),
            "top5_precision": round(float(np.mean([r["top5_precision"] for r in rows])), 4),
        }

    result = {
        "task": task_name,
        "specialist_model": "Boltz-2 (MSA + all-atom co-folding, JapanFold)",
        "n_evaluated": len(per_target),
        "n_total": manifest["n_targets"],
        "cost_usd": round(cost_usd, 4),
        "overall": _agg(per_target),
        "by_stratum": {k: _agg(v) for k, v in sorted(by_stratum.items())},
        "anti_cheat_null": manifest["anti_cheat_audit"],
        "per_target": per_target,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks", default="10,12", help="Comma-separated: 10,12")
    args = parser.parse_args()
    sel = {s.strip() for s in args.tasks.split(",") if s.strip()}

    with JapanFoldClient(max_usd_budget=25.0) as jf:
        print(f"JapanFold account balance: ${jf.get_account_balance():.2f}", flush=True)

        if "10" in sel:
            m10 = json.loads((REPO / "data/task10_ligand_site/manifest.json").read_text(encoding="utf-8"))
            print(f"\n=== Running Boltz-2 on Task 10 ({m10['n_targets']} post-May-2026 ligand/metal complexes) ===", flush=True)
            dists10, cost10 = run_boltz2_batch(jf, m10["targets"], build_t10_boltz2_yaml, cache_tag="t10", batch_size=10)
            res10 = evaluate_and_save(
                m10, dists10, REPO / "outputs/results/baseline_t10_boltz2.json", "task10_ligand_active_site", cost10
            )
            print("Task 10 Boltz-2 overall:", res10["overall"], flush=True)
            print("Task 10 by stratum:", res10["by_stratum"], flush=True)

        if "12" in sel:
            m12 = json.loads((REPO / "data/task12_nucleic_acid/manifest.json").read_text(encoding="utf-8"))
            print(f"\n=== Running Boltz-2 on Task 12 ({m12['n_targets']} post-May-2026 DNA/RNA complexes) ===", flush=True)
            dists12, cost12 = run_boltz2_batch(jf, m12["targets"], build_t12_boltz2_yaml, cache_tag="t12", batch_size=10)
            res12 = evaluate_and_save(
                m12, dists12, REPO / "outputs/results/baseline_t12_boltz2.json", "task12_protein_nucleic_acid", cost12
            )
            print("Task 12 Boltz-2 overall:", res12["overall"], flush=True)
            print("Task 12 by stratum:", res12["by_stratum"], flush=True)


if __name__ == "__main__":
    main()
