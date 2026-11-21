#!/usr/bin/env python3
"""Evaluate Stage 1A (Forward Folding) & Stage 1B (Inverse Folding) predictions and baselines.

Can run 100% offline (`--run-calibration`) to verify the entire Stage 1 dataset and establish
exact upper/lower reference bounds:
  - `Oracle_TrueContacts`: Exact crystal contacts + Q3 fed through our L-BFGS 3D reconstructor
  - `Oracle_50pctPrecision`: Simulated pLM contact map (50% true contacts + 50% false positives)
  - `Heuristic_1D_ChouFasman`: Local helix/strand propensities only (zero tertiary knowledge)
  - `Null_Random`: Random contact pairs (`|i - j| >= 6`) & random uniform 20-AA sequences
Plus grades all LLM (`outputs/predictions/*_stage1a.jsonl`, `*_stage1b.jsonl`) and Modal
specialist baseline (`esmfold`, `esm2_3b`, `proteinmpnn`) outputs.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
import json
import os
from pathlib import Path
import sys
from typing import Any, Dict, List, Tuple

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from fable_bench.pdb_utils import CANONICAL_AA_SET, ProteinStructure
from fable_bench.stage1_grader import (
    ForwardGradeResult,
    InverseGradeResult,
    grade_forward_prediction,
    grade_inverse_prediction,
)
from fable_bench.stage1_prompts_and_parsers import (
    ForwardPredictionParsed,
    InversePredictionParsed,
    parse_forward_folding_response,
    parse_inverse_folding_response,
)

AA_LIST = sorted(CANONICAL_AA_SET)
HELIX_PROP = set("AEHKLMQR")
STRAND_PROP = set("CFILTVWY")


def load_manifest_and_structures(
    data_dir: Path,
) -> Tuple[Dict[str, Any], Dict[str, ProteinStructure], Dict[str, Dict[str, Any]]]:
    manifest = json.loads((data_dir / "manifest.json").read_text(encoding="utf-8"))
    structures: Dict[str, ProteinStructure] = {}
    meta_by_id: Dict[str, Dict[str, Any]] = {}
    for rec in manifest["targets"]:
        tid = rec["target_id"]
        meta_by_id[tid] = rec
        d = np.load(data_dir / "ground_truth" / f"{tid}.npz", allow_pickle=False)
        structures[tid] = ProteinStructure(
            target_id=tid,
            pdb_id=rec["pdb_id"],
            chain_id=rec["chain_id"],
            sequence=str(d["sequence"]),
            res_numbers=d["res_numbers"].tolist(),
            coords_n=d["coords_n"],
            coords_ca=d["coords_ca"],
            coords_c=d["coords_c"],
            coords_o=d["coords_o"],
            coords_cb=d["coords_cb"],
            phi_deg=d["phi_deg"],
            psi_deg=d["psi_deg"],
            q3=str(d["q3"]),
            cb_dist_matrix=d["cb_dist_matrix"],
            ca_coord_num=d["ca_coord_num"],
        )
    return manifest, structures, meta_by_id


def make_calibration_forward_preds(
    struct: ProteinStructure,
    mode: str,
    rng: np.random.Generator,
) -> ForwardPredictionParsed:
    """Construct controlled calibration baselines for Stage 1A Forward Folding."""
    L = struct.length
    k_total = max(20, int(round(1.5 * L)))
    ii, jj = np.triu_indices(L, k=6)
    true_mask = struct.cb_dist_matrix[ii, jj] < 8.0
    true_pairs = [(int(i + 1), int(j + 1), float(struct.cb_dist_matrix[i, j])) for i, j in zip(ii[true_mask], jj[true_mask])]
    true_pairs.sort(key=lambda t: t[2])  # Tightest true Cβ contacts first
    false_pairs = [(int(i + 1), int(j + 1)) for i, j in zip(ii[~true_mask], jj[~true_mask])]

    if mode == "Oracle_TrueContacts":
        contacts = [(i, j, round(1.0 - 0.05 * (d / 8.0), 3)) for (i, j, d) in true_pairs[:k_total]]
        return ForwardPredictionParsed(q3=struct.q3, contacts=contacts, parse_ok=True, raw_q3_len=L)

    if mode == "NoisyOracle_50pctPrec":
        n_half = min(len(true_pairs), k_total // 2)
        chosen_true = [(i, j, 0.85) for (i, j, _d) in true_pairs[:n_half]]
        n_fp = max(0, k_total - n_half)
        fp_idx = rng.choice(len(false_pairs), size=min(n_fp, len(false_pairs)), replace=False)
        chosen_fp = [(false_pairs[k][0], false_pairs[k][1], 0.55) for k in fp_idx]
        mixed = chosen_true + chosen_fp
        rng.shuffle(mixed)
        # Re-assign confidence slightly favoring true contacts to mimic noisy pLM
        contacts = [
            (i, j, float(np.clip(c + rng.normal(0.0, 0.15), 0.1, 0.99)))
            for (i, j, c) in mixed
        ]
        contacts.sort(key=lambda t: t[2], reverse=True)
        # Flip 15% of Q3 states
        q3_chars = [
            ch if rng.random() > 0.15 else rng.choice(["H", "E", "C"])
            for ch in struct.q3
        ]
        return ForwardPredictionParsed(q3="".join(q3_chars), contacts=contacts, parse_ok=True, raw_q3_len=L)

    if mode == "Heuristic_1D_LocalOnly":
        # Predict Q3 from 5-residue sliding Chou-Fasman propensity window; only i, i+6..i+9 local contacts
        q3_out = []
        for idx in range(L):
            win = struct.sequence[max(0, idx - 2) : min(L, idx + 3)]
            h_cnt = sum(1 for a in win if a in HELIX_PROP)
            e_cnt = sum(1 for a in win if a in STRAND_PROP)
            if "P" in win[1:-1] or "G" in win[1:-1]:
                q3_out.append("C")
            elif h_cnt >= 3 and h_cnt >= e_cnt:
                q3_out.append("H")
            elif e_cnt >= 3:
                q3_out.append("E")
            else:
                q3_out.append("C")
        q3_s = "".join(q3_out)
        contacts = []
        for idx in range(1, L - 6):
            contacts.append((idx, idx + 6, 0.5))
        return ForwardPredictionParsed(q3=q3_s, contacts=contacts[:k_total], parse_ok=True, raw_q3_len=L)

    # Null_Random
    all_pairs = [(int(i + 1), int(j + 1)) for i, j in zip(ii, jj)]
    sel = rng.choice(len(all_pairs), size=min(k_total, len(all_pairs)), replace=False)
    contacts = [(all_pairs[s][0], all_pairs[s][1], round(1.0 - 0.8 * (r / k_total), 3)) for r, s in enumerate(sel)]
    rand_q3 = "".join(rng.choice(["H", "E", "C"], size=L))
    return ForwardPredictionParsed(q3=rand_q3, contacts=contacts, parse_ok=True, raw_q3_len=L)


def summarize_forward(
    model_name: str,
    grades: List[ForwardGradeResult],
    meta_by_id: Dict[str, Dict[str, Any]],
) -> Dict[str, Any]:
    """Aggregate Stage 1A metrics overall and across the 2x2 (Pre/Post x Size) matrix."""
    def _agg(sub: List[ForwardGradeResult]) -> Dict[str, float]:
        if not sub:
            return {}
        return {
            "n": len(sub),
            "tm_score": round(float(np.mean([g.tm_score for g in sub])), 4),
            "rmsd_ca": round(float(np.mean([g.rmsd_ca for g in sub])), 2),
            "lddt_ca": round(float(np.mean([g.lddt_ca for g in sub])), 4),
            "p_at_l_sep6": round(float(np.mean([g.p_at_l_sep6 for g in sub])), 4),
            "p_at_l_sep24": round(float(np.mean([g.p_at_l_sep24 for g in sub])), 4),
            "p_at_l5_sep24": round(float(np.mean([g.p_at_l5_sep24 for g in sub])), 4),
            "q3_accuracy": round(float(np.mean([g.q3_accuracy for g in sub])), 4),
        }

    by_split: Dict[str, List[ForwardGradeResult]] = defaultdict(list)
    for g in grades:
        m = meta_by_id[g.target_id]
        by_split[m["temporal_split"]].append(g)
        if "stratum" in m:
            by_split[m["stratum"]].append(g)
        by_split[f"{m['temporal_split']}__{m['size_bin']}"].append(g)
        # Stratify by PDB vs AFDB similarity regimes!
        pdb_sid = float(m.get("pdb_precutoff_seqid_max", 0.0))
        af_sid = float(m.get("afdb_max_seqid", 0.0))
        af_prob = float(m.get("afdb_foldseek_prob", 0.0))
        if m["temporal_split"] == "post_cutoff":
            if pdb_sid < 30.0 and af_sid < 30.0:
                by_split["novel_to_BOTH_pdb_and_afdb"].append(g)
            elif pdb_sid < 30.0 and af_sid >= 30.0:
                by_split["pdb_novel_but_in_afdb"].append(g)
            if af_prob < 0.50:
                by_split["foldseek_afdb_struct_novel"].append(g)
        if m["pdb_id"] == "6MRR":
            by_split["anchor_6MRR"].append(g)
        if m["pdb_id"] == "37NT":
            by_split["anchor_37NT_domains"].append(g)

    return {
        "model": model_name,
        "overall": _agg(grades),
        "splits": {k: _agg(v) for k, v in sorted(by_split.items())},
        "per_target": [g.to_dict() for g in grades],
    }


def summarize_inverse(
    model_name: str,
    grades: List[InverseGradeResult],
    meta_by_id: Dict[str, Dict[str, Any]],
) -> Dict[str, Any]:
    """Aggregate Stage 1B metrics overall and across Pre/Post-cutoff splits."""
    def _agg(sub: List[InverseGradeResult]) -> Dict[str, Any]:
        if not sub:
            return {}
        out: Dict[str, Any] = {
            "n": len(sub),
            "nsr_overall": round(float(np.mean([g.nsr_overall for g in sub])), 4),
            "nsr_core": round(float(np.mean([g.nsr_core for g in sub])), 4),
            "nsr_surface": round(float(np.mean([g.nsr_surface for g in sub])), 4),
            "biochem_group_recovery": round(float(np.mean([g.biochem_group_recovery for g in sub])), 4),
            "exact_len_rate": round(float(np.mean([1.0 if g.length_exact_match else 0.0 for g in sub])), 4),
        }
        sctms = [g.sc_tm_score for g in sub if g.sc_tm_score is not None]
        scrmsds = [g.sc_rmsd_ca for g in sub if g.sc_rmsd_ca is not None]
        if sctms:
            out["sc_tm_score"] = round(float(np.mean(sctms)), 4)
            out["sc_rmsd_ca"] = round(float(np.mean(scrmsds)), 2)
            out["designable_pass_rate_2A"] = round(
                float(np.mean([1.0 if (t > 0.5 and r < 2.0) else 0.0 for t, r in zip(sctms, scrmsds)])),
                4,
            )
        return out

    by_split: Dict[str, List[InverseGradeResult]] = defaultdict(list)
    for g in grades:
        m = meta_by_id[g.target_id]
        by_split[m["temporal_split"]].append(g)
        if "stratum" in m:
            by_split[m["stratum"]].append(g)
        pdb_sid = float(m.get("pdb_precutoff_seqid_max", 0.0))
        af_sid = float(m.get("afdb_max_seqid", 0.0))
        if m["temporal_split"] == "post_cutoff":
            if pdb_sid < 30.0 and af_sid < 30.0:
                by_split["novel_to_BOTH_pdb_and_afdb"].append(g)
            elif pdb_sid < 30.0 and af_sid >= 30.0:
                by_split["pdb_novel_but_in_afdb"].append(g)
        if m["pdb_id"] == "37NT":
            by_split["anchor_37NT_domains"].append(g)

    return {
        "model": model_name,
        "overall": _agg(grades),
        "splits": {k: _agg(v) for k, v in sorted(by_split.items())},
        "per_target": [g.to_dict() for g in grades],
    }


def _grade_single_forward_worker(args_tuple: Tuple[ProteinStructure, str, str, int]) -> ForwardGradeResult:
    struct, raw_text, pdb_out_str, restarts = args_tuple
    parsed = parse_forward_folding_response(raw_text, struct.length)
    return grade_forward_prediction(struct, parsed, out_pdb_path=Path(pdb_out_str), n_restarts=restarts)


def main() -> None:
    parser = argparse.ArgumentParser(description="Grade Stage 1 predictions and calibration baselines.")
    parser.add_argument("--data-dir", default=str(REPO_ROOT / "data" / "stage1_benchmark"))
    parser.add_argument("--pred-dir", default=str(REPO_ROOT / "outputs" / "predictions"))
    parser.add_argument("--out-dir", default=str(REPO_ROOT / "outputs" / "results"))
    parser.add_argument("--run-calibration", action="store_true", help="Run offline oracle & null baselines")
    parser.add_argument("--restarts", type=int, default=3, help="L-BFGS restarts per structure")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    pred_dir = Path(args.pred_dir)
    out_dir = Path(args.out_dir)
    recon_pdb_dir = out_dir / "reconstructed_pdbs"
    out_dir.mkdir(parents=True, exist_ok=True)
    recon_pdb_dir.mkdir(parents=True, exist_ok=True)

    _manifest, structures, meta_by_id = load_manifest_and_structures(data_dir)
    forward_summaries: List[Dict[str, Any]] = []
    inverse_summaries: List[Dict[str, Any]] = []

    prev_1a_by_model: Dict[str, Dict[str, Any]] = {}
    if args.run_calibration:
        rng = np.random.default_rng(20261118)
        for mode in ("Oracle_TrueContacts", "NoisyOracle_50pctPrec", "Heuristic_1D_LocalOnly", "Null_Random"):
            print(f"Grading Stage 1A calibration baseline: {mode} ...")
            f_grades: List[ForwardGradeResult] = []
            for tid, struct in structures.items():
                pred = make_calibration_forward_preds(struct, mode, rng)
                pdb_out = recon_pdb_dir / f"{mode}__{tid}.pdb"
                g = grade_forward_prediction(struct, pred, out_pdb_path=pdb_out, n_restarts=args.restarts)
                f_grades.append(g)
            forward_summaries.append(summarize_forward(mode, f_grades, meta_by_id))

        # Inverse folding null baseline (uniform 20 AA vs hydrophobic-core heuristic)
        for inv_mode in ("Heuristic_BurialPolarity", "Null_Uniform20AA"):
            i_grades: List[InverseGradeResult] = []
            for tid, struct in structures.items():
                if inv_mode == "Null_Uniform20AA":
                    rseq = "".join(rng.choice(AA_LIST, size=struct.length))
                else:
                    # Simple biophysical rule: core -> {L,V,I,A,F}, Gly at positive phi, surface -> {E,K,Q,S,D,R}
                    chars = []
                    for idx in range(struct.length):
                        if not np.isnan(struct.phi_deg[idx]) and struct.phi_deg[idx] > 30.0:
                            chars.append("G")
                        elif struct.is_core_mask[idx]:
                            chars.append(rng.choice(["L", "V", "I", "A", "F"]))
                        else:
                            chars.append(rng.choice(["E", "K", "Q", "S", "D", "R", "T"]))
                    rseq = "".join(chars)
                ipred = InversePredictionParsed(
                    sequence=rseq,
                    parse_ok=True,
                    raw_seq_len=struct.length,
                    length_exact_match=True,
                )
                i_grades.append(grade_inverse_prediction(struct, ipred))
            inverse_summaries.append(summarize_inverse(inv_mode, i_grades, meta_by_id))
    else:
        existing_file = out_dir / "stage1_evaluation_report.json"
        if existing_file.exists():
            prev = json.loads(existing_file.read_text(encoding="utf-8"))
            cal_1a = {"Oracle_TrueContacts", "NoisyOracle_50pctPrec", "Heuristic_1D_LocalOnly", "Null_Random"}
            cal_1b = {"Heuristic_BurialPolarity", "Null_Uniform20AA"}
            for x in prev.get("stage1a_forward_folding", []):
                if x.get("model") in cal_1a:
                    forward_summaries.append(x)
                elif x.get("model"):
                    prev_1a_by_model[x["model"]] = x
            inverse_summaries.extend(x for x in prev.get("stage1b_inverse_folding", []) if x.get("model") in cal_1b)

    # Grade any LLM or Modal specialist JSONL files in `outputs/predictions/`
    if pred_dir.exists():
        n_workers = min(8, os.cpu_count() or 4)
        for jf in sorted(pred_dir.glob("*_stage1a.jsonl")):
            mname = jf.name.replace("_stage1a.jsonl", "")
            rows = [json.loads(l) for l in jf.read_text(encoding="utf-8").splitlines() if l.strip()]
            if mname in prev_1a_by_model and prev_1a_by_model[mname].get("overall", {}).get("n") == len(rows):
                print(f"Reusing cached Stage 1A evaluation for {mname} (n={len(rows)}) ...", flush=True)
                forward_summaries.append(prev_1a_by_model[mname])
                continue
            print(f"Grading Stage 1A model predictions ({n_workers} workers): {mname} ...", flush=True)
            f_grades = []
            worker_jobs: List[Tuple[ProteinStructure, str, str, int]] = []
            for r in rows:
                tid = r["target_id"]
                if tid not in structures:
                    continue
                struct = structures[tid]
                if "precomputed_tm_score" in r:
                    f_grades.append(
                        ForwardGradeResult(
                            target_id=tid,
                            length=struct.length,
                            parse_ok=True,
                            n_predicted_contacts=r.get("n_predicted_contacts", 0),
                            q3_accuracy=r.get("q3_accuracy", 0.0),
                            p_at_l_sep6=r.get("p_at_l_sep6", 0.0),
                            p_at_l2_sep6=r.get("p_at_l2_sep6", 0.0),
                            p_at_l5_sep6=r.get("p_at_l5_sep6", 0.0),
                            p_at_l_sep12=r.get("p_at_l_sep12", 0.0),
                            p_at_l5_sep12=r.get("p_at_l5_sep12", 0.0),
                            p_at_l_sep24=r.get("p_at_l_sep24", 0.0),
                            p_at_l2_sep24=r.get("p_at_l2_sep24", 0.0),
                            p_at_l5_sep24=r.get("p_at_l5_sep24", 0.0),
                            tm_score=r["precomputed_tm_score"],
                            tm_score_mirror_best=r["precomputed_tm_score"],
                            rmsd_ca=r["precomputed_rmsd_ca"],
                            lddt_ca=r["precomputed_lddt_ca"],
                        )
                    )
                else:
                    pdb_out = str(recon_pdb_dir / f"{mname}__{tid}.pdb")
                    worker_jobs.append((struct, r["text"], pdb_out, args.restarts))
            if worker_jobs:
                with ProcessPoolExecutor(max_workers=n_workers) as pool:
                    f_grades.extend(pool.map(_grade_single_forward_worker, worker_jobs))
            forward_summaries.append(summarize_forward(mname, f_grades, meta_by_id))

        for jf in sorted(pred_dir.glob("*_stage1b.jsonl")):
            mname = jf.name.replace("_stage1b.jsonl", "")
            print(f"Grading Stage 1B inverse-folding predictions: {mname} ...")
            rows = [json.loads(l) for l in jf.read_text(encoding="utf-8").splitlines() if l.strip()]
            i_grades = []
            for r in rows:
                tid = r["target_id"]
                if tid not in structures:
                    continue
                struct = structures[tid]
                parsed = parse_inverse_folding_response(r["text"], struct.length)
                ig = grade_inverse_prediction(struct, parsed)
                ig.sc_tm_score = r.get("sc_tm_score")
                ig.sc_rmsd_ca = r.get("sc_rmsd_ca")
                ig.esmfold_plddt = r.get("esmfold_plddt")
                i_grades.append(ig)
            inverse_summaries.append(summarize_inverse(mname, i_grades, meta_by_id))

    report = {
        "stage1a_forward_folding": forward_summaries,
        "stage1b_inverse_folding": inverse_summaries,
    }
    out_file = out_dir / "stage1_evaluation_report.json"
    out_file.write_text(json.dumps(report, indent=2), encoding="utf-8")

    # Print crisp terminal tables
    print("\n" + "=" * 98)
    print("STAGE 1A — FORWARD FOLDING SUMMARY (PRE-CUTOFF vs. STRICT LATE-2026 POST-CUTOFF)")
    print("=" * 98)
    print(f"{'Model / Baseline':<26s} | {'All TM':>7s} | {'Pre TM':>7s} | {'Post TM':>7s} | {'P@L(|i-j|>=24)':>14s} | {'Q3 %':>6s} | {'37NT TM':>7s}")
    print("-" * 98)
    for fs in forward_summaries:
        ov = fs["overall"]
        pre = fs["splits"].get("pre_cutoff", {})
        post = fs["splits"].get("post_cutoff", {})
        t37 = fs["splits"].get("anchor_37NT_domains", {})
        print(
            f"{fs['model']:<26s} | {ov.get('tm_score', 0.0):7.3f} | "
            f"{pre.get('tm_score', 0.0):7.3f} | {post.get('tm_score', 0.0):7.3f} | "
            f"{ov.get('p_at_l_sep24', 0.0):14.3f} | {100.0 * ov.get('q3_accuracy', 0.0):5.1f}% | "
            f"{t37.get('tm_score', 0.0):7.3f}"
        )

    print("\n" + "=" * 98)
    print("STAGE 1B — INVERSE FOLDING SUMMARY (3D STRIPPED BACKBONE -> 1D SEQUENCE)")
    print("=" * 98)
    print(f"{'Model / Baseline':<26s} | {'NSR All':>8s} | {'NSR Core':>8s} | {'NSR Surf':>8s} | {'Post NSR':>8s} | {'Group %':>8s}")
    print("-" * 98)
    for iv in inverse_summaries:
        ov = iv["overall"]
        post = iv["splits"].get("post_cutoff", {})
        print(
            f"{iv['model']:<26s} | {100.0 * ov.get('nsr_overall', 0.0):7.2f}% | "
            f"{100.0 * ov.get('nsr_core', 0.0):7.2f}% | {100.0 * ov.get('nsr_surface', 0.0):7.2f}% | "
            f"{100.0 * post.get('nsr_overall', 0.0):7.2f}% | "
            f"{100.0 * ov.get('biochem_group_recovery', 0.0):7.2f}%"
        )
    print(f"\nSaved full Stage 1 report -> {out_file}")


if __name__ == "__main__":
    main()
