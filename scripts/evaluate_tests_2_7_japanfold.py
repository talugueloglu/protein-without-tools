#!/usr/bin/env python3
"""Evaluate Test 2 (Inverse Folding, N=100) and Test 7 (De Novo & Motif Scaffolding, N=40)
directly with JapanFold ESMFold-2 (MSA-free) — zero ProteinMPNN."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess
import sys
from typing import Any, Dict, List, Tuple

import numpy as np

REPO = Path(__file__).resolve().parents[1]
if str(REPO / "src") not in sys.path:
    sys.path.insert(0, str(REPO / "src"))

from protein_without_tools.japanfold_client import JapanFoldClient, find_pdb_for_target, parse_pdb_chain_with_bfactor
from protein_without_tools.pdb_utils import CANONICAL_AA_SET, load_structure
from protein_without_tools.stage1_prompts_and_parsers import parse_inverse_folding_response
from protein_without_tools.stage3b_design import (
    DesignTaskSpec,
    MotifSegmentSpec,
    grade_monomer_design,
    kabsch_rmsd,
    parse_designed_sequence,
)
from protein_without_tools.tmscore import evaluate_3d_superposition, tm_align_ca


def _seq_hash(model_name: str, seq: str) -> str:
    return hashlib.sha1(f"{model_name}:{seq}".encode("utf-8")).hexdigest()[:20]


def fold_sequences_cached(
    jf: JapanFoldClient,
    sequences: List[str],
    fold_model: str = "esmfold2",
    batch_size: int = 25,
) -> Dict[str, Dict[str, Any]]:
    """Fold unique canonical AA sequences on JapanFold (`esmfold2` or `openfold3`, MSA-free) with disk index cache."""
    idx_dir = REPO / "outputs" / "japanfold_cache" / f"seq_store_{fold_model}"
    idx_dir.mkdir(parents=True, exist_ok=True)

    uniq_seqs = sorted({s for s in sequences if len(s) >= 20 and all(c in CANONICAL_AA_SET for c in s)})
    results_by_seq: Dict[str, Dict[str, Any]] = {}
    todo: List[str] = []

    for s in uniq_seqs:
        h = _seq_hash(fold_model, s)
        npz_p = idx_dir / f"{h}.npz"
        json_p = idx_dir / f"{h}.json"
        if npz_p.exists() and json_p.exists():
            meta = json.loads(json_p.read_text(encoding="utf-8"))
            arr = np.load(npz_p)
            meta["coords_ca"] = arr["coords_ca"]
            meta["coords_cb"] = arr["coords_cb"]
            results_by_seq[s] = meta
        else:
            todo.append(s)

    if todo:
        print(f"[{fold_model}] Folding {len(todo)} uncached sequences in batches of {batch_size}...", flush=True)

    params: Dict[str, Any] = {"use_msa_server": False, "output_format": "pdb"}
    if fold_model == "esmfold2":
        params["fast"] = False

    for b_start in range(0, len(todo), batch_size):
        chunk = todo[b_start : b_start + batch_size]
        targets_payload = []
        for idx0, seq in enumerate(chunk):
            tname = f"s{idx0 + 1:03d}"
            targets_payload.append({"name": tname, "content": f">{tname}\n{seq}"})

        meta_job, files = jf.submit_and_collect(
            "/v1/predictions",
            {
                "model": fold_model,
                "name": f"batch_{fold_model}_{b_start}",
                "targets": targets_payload,
                "params": params,
            },
            cache_prefix=f"monobatch_{fold_model}",
        )
        rows = meta_job["results"].get("rows") or []
        row_by_id = {str(r.get("id")): r for r in rows}

        for idx0, seq in enumerate(chunk):
            tname = f"s{idx0 + 1:03d}"
            pdb_txt = find_pdb_for_target(files, tname, idx0)
            row = row_by_id.get(tname) or row_by_id.get(f"target_{idx0 + 1}") or (rows[idx0] if idx0 < len(rows) else {})
            if not pdb_txt:
                continue
            ch = parse_pdb_chain_with_bfactor(pdb_txt, "A")
            row_plddt = row.get("plddt")
            plddt_100 = float(row_plddt) * 100.0 if (row_plddt is not None and float(row_plddt) <= 1.05) else ch["plddt_mean"]
            ptm = float(row.get("ptm") or 0.0)

            h = _seq_hash(fold_model, seq)
            rec_meta = {
                "sequence": seq,
                "plddt_mean": round(float(plddt_100), 2),
                "ptm": round(ptm, 4),
                "q3": ch["q3"],
            }
            (idx_dir / f"{h}.json").write_text(json.dumps(rec_meta), encoding="utf-8")
            np.savez_compressed(idx_dir / f"{h}.npz", coords_ca=ch["coords_ca"], coords_cb=ch["coords_cb"])
            rec_full = dict(rec_meta)
            rec_full["coords_ca"] = ch["coords_ca"]
            rec_full["coords_cb"] = ch["coords_cb"]
            results_by_seq[seq] = rec_full

        print(
            f"  [{fold_model}] Completed {min(b_start + batch_size, len(todo))}/{len(todo)} "
            f"(job cost=${meta_job['cost_usd']:.4f}, cum=${jf.cumulative_usd:.4f})",
            flush=True,
        )

    return results_by_seq


def evaluate_test2(jf: JapanFoldClient) -> None:
    s1_dir = REPO / "data" / "stage1_benchmark"
    pred_dir = REPO / "outputs" / "predictions"
    manifest = json.loads((s1_dir / "manifest.json").read_text(encoding="utf-8"))["targets"]
    gt_by_id: Dict[str, Dict[str, Any]] = {}
    for m in manifest:
        tid = m["target_id"]
        z = np.load(s1_dir / "ground_truth" / f"{tid}.npz")
        gt_by_id[tid] = {
            "length": int(m["length"]),
            "sequence": str(z["sequence"]),
            "coords_ca": np.asarray(z["coords_ca"], dtype=np.float64),
            "q3": str(z["q3"]),
        }

    # Ensure native wild-type control JSONL exists so we measure ESMFold-2's own single-sequence ceiling!
    wt_file = pred_dir / "control_wildtype_native_stage1b.jsonl"
    if not wt_file.exists():
        wt_rows = [
            {
                "target_id": m["target_id"],
                "task": "stage1b_inverse_folding",
                "length": m["length"],
                "text": json.dumps({"sequence": gt_by_id[m["target_id"]]["sequence"]}),
            }
            for m in manifest
        ]
        wt_file.write_text("\n".join(json.dumps(r) for r in wt_rows) + "\n", encoding="utf-8")

    s1b_files = sorted(pred_dir.glob("*_stage1b.jsonl"))
    all_seqs: List[str] = []
    parsed_by_file: Dict[Path, Tuple[List[Dict[str, Any]], Dict[str, str]]] = {}

    for jf_path in s1b_files:
        rows = [json.loads(ln) for ln in jf_path.read_text(encoding="utf-8").splitlines() if ln.strip()]
        seq_map: Dict[str, str] = {}
        for r in rows:
            tid = r["target_id"]
            if tid not in gt_by_id:
                continue
            # Use raw cleaned sequence from response before pad/truncation if valid, else parsed
            raw_txt = str(r.get("text") or "")
            s_clean, _rat, ok = parse_designed_sequence(raw_txt)
            if not ok:
                p = parse_inverse_folding_response(raw_txt, gt_by_id[tid]["length"])
                s_clean = p.sequence if p.parse_ok else ""
            if len(s_clean) >= 20 and all(c in CANONICAL_AA_SET for c in s_clean):
                seq_map[tid] = s_clean
                all_seqs.append(s_clean)
        parsed_by_file[jf_path] = (rows, seq_map)

    folded = fold_sequences_cached(jf, all_seqs, fold_model="esmfold2", batch_size=25)

    print("\n=== TEST 2: INVERSE FOLDING ESMFold-2 SELF-CONSISTENCY (scTM & scRMSD) ===", flush=True)
    for jf_path, (rows, seq_map) in parsed_by_file.items():
        mtag = jf_path.name.replace("_stage1b.jsonl", "")
        sctms, rmsds, plddts = [], [], []
        for r in rows:
            tid = r["target_id"]
            if tid not in gt_by_id:
                continue
            gt = gt_by_id[tid]
            seq = seq_map.get(tid, "")
            f_rec = folded.get(seq)
            if f_rec is None:
                r["sc_tm_score"] = 0.0
                r["sc_rmsd_ca"] = 99.9
                r["esmfold_plddt"] = 0.0
            else:
                pred_ca = f_rec["coords_ca"]
                if len(pred_ca) == len(gt["coords_ca"]):
                    sup = evaluate_3d_superposition(pred_ca, gt["coords_ca"])
                    tm_v = round(float(sup.tm_score), 4)
                    rm_v = round(float(sup.rmsd_global), 3)
                else:
                    tm_v, rm_v, _ = tm_align_ca(
                        pred_ca, gt["coords_ca"], norm_length=len(gt["coords_ca"]), pred_q3=f_rec["q3"], true_q3=gt["q3"]
                    )
                r["sc_tm_score"] = tm_v
                r["sc_rmsd_ca"] = rm_v
                r["esmfold_plddt"] = f_rec["plddt_mean"]
            sctms.append(r["sc_tm_score"])
            rmsds.append(r["sc_rmsd_ca"])
            plddts.append(r["esmfold_plddt"])

        jf_path.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
        pass_tm05 = float(np.mean([1.0 if t >= 0.50 else 0.0 for t in sctms]))
        pass_2a = float(np.mean([1.0 if (t >= 0.50 and rm < 2.0) else 0.0 for t, rm in zip(sctms, rmsds)]))
        print(
            f"  {mtag:<32s} | n={len(sctms):3d} | Mean scTM={np.mean(sctms):.3f} | "
            f"scTM>=0.5: {pass_tm05*100:5.1f}% | scRMSD<2A: {pass_2a*100:5.1f}% | pLDDT={np.mean(plddts):5.1f}",
            flush=True,
        )

    # Refresh official Stage 1 report JSON
    subprocess.run([sys.executable, str(REPO / "scripts" / "evaluate_stage1.py")], check=True)


def evaluate_test7(jf: JapanFoldClient) -> None:
    s3b_dir = REPO / "data" / "stage3b_design"
    pred_dir = REPO / "outputs" / "predictions"
    res_dir = REPO / "outputs" / "results"
    res_dir.mkdir(parents=True, exist_ok=True)

    raw_tasks = json.loads((s3b_dir / "monomer_motif_manifest.json").read_text(encoding="utf-8"))["tasks"]
    specs: List[DesignTaskSpec] = []
    ref_structs: Dict[str, Any] = {}
    for t in raw_tasks:
        segs = [MotifSegmentSpec(**s) for s in t.get("motif_segments", [])]
        sp = DesignTaskSpec(
            task_id=t["task_id"],
            category=t["category"],
            subfamily=t["subfamily"],
            min_length=t["min_length"],
            max_length=t["max_length"],
            description=t["description"],
            ss_bounds=tuple(t.get("ss_bounds", [0.0, 1.0, 0.0, 1.0])),
            reference_pdb=t.get("reference_pdb", ""),
            motif_segments=segs,
            max_allowed_cys=t.get("max_allowed_cys", 1),
            forbidden_parent_sequences=t.get("forbidden_parent_sequences", []),
        )
        specs.append(sp)
        if sp.reference_pdb and sp.reference_pdb not in ref_structs:
            ref_p = s3b_dir / sp.reference_pdb
            if ref_p.exists():
                ref_structs[sp.reference_pdb] = load_structure(ref_p, sp.reference_pdb)

    s3b_files = sorted(pred_dir.glob("*_stage3b.jsonl"))
    all_seqs: List[str] = []
    parsed_by_model: Dict[str, Dict[str, str]] = {}

    for pf in s3b_files:
        mtag = pf.name.replace("_stage3b.jsonl", "")
        seq_by_tid: Dict[str, str] = {}
        for ln in pf.read_text(encoding="utf-8").splitlines():
            if not ln.strip():
                continue
            o = json.loads(ln)
            tid = str(o["target_id"])
            seq, _rat, ok = parse_designed_sequence(str(o.get("text") or ""))
            if ok and len(seq) >= 25 and all(c in CANONICAL_AA_SET for c in seq):
                seq_by_tid[tid] = seq
                all_seqs.append(seq)
        parsed_by_model[mtag] = seq_by_tid

    folded_esm2 = fold_sequences_cached(jf, all_seqs, fold_model="esmfold2", batch_size=25)
    folded_of3 = fold_sequences_cached(jf, all_seqs, fold_model="openfold3", batch_size=20)

    print("\n=== TEST 7: DE NOVO MONOMER & MOTIF SCAFFOLDING (ESMFold-2 + OpenFold3, N=40) ===", flush=True)
    report_models: Dict[str, Any] = {}

    for mtag, seq_by_tid in parsed_by_model.items():
        per_task: List[Dict[str, Any]] = []
        for sp in specs:
            seq = seq_by_tid.get(sp.task_id, "")
            f_e = folded_esm2.get(seq)
            f_o = folded_of3.get(seq)
            if f_e is None:
                g = grade_monomer_design(sp, seq)
                g["esmfold2_4gate_pass"] = False
                g["ptm_esmfold2"] = 0.0
                per_task.append(g)
                continue

            ca_e = f_e["coords_ca"]
            q3_e = f_e["q3"]
            plddt_e = f_e["plddt_mean"]

            tm_ref = None
            if sp.category == "fold_conditioned" and sp.reference_pdb in ref_structs:
                rst = ref_structs[sp.reference_pdb]
                tm_ref, _, _ = tm_align_ca(ca_e, rst.coords_ca, norm_length=len(ca_e), pred_q3=q3_e, true_q3=rst.q3)

            cross_rmsd = None
            if f_o is not None and len(f_o["coords_ca"]) == len(ca_e):
                cross_rmsd = round(float(kabsch_rmsd(ca_e, f_o["coords_ca"])), 3)

            g = grade_monomer_design(
                sp,
                seq,
                folded_ca_coords=ca_e,
                plddt_mean=plddt_e,
                ss_q3=q3_e,
                tm_to_target_fold=tm_ref,
                mpnn_sc_rmsd=cross_rmsd,
            )
            g["esmfold2_4gate_pass"] = bool(
                g["gate1_hygiene"] and g["gate2_foldable_globular"] and g["gate4_prompt_adherence"] and g["gate5_novelty_lt30"]
            )
            g["cross_oracle_sc_rmsd"] = cross_rmsd
            g["ptm_esmfold2"] = f_e["ptm"]
            per_task.append(g)

        fc_sub = [x for x in per_task if x["category"] == "fold_conditioned"]
        ms_sub = [x for x in per_task if x["category"] == "motif_scaffolding"]
        summary = {
            "n_tasks": len(per_task),
            "gate1_hygiene_rate": round(float(np.mean([x["gate1_hygiene"] for x in per_task])), 4),
            "gate2_foldable_rate": round(float(np.mean([x["gate2_foldable_globular"] for x in per_task])), 4),
            "gate3_cross_oracle_sc_rate": round(float(np.mean([x["gate3_mpnn_self_consistent"] for x in per_task])), 4),
            "gate4_adherence_rate": round(float(np.mean([x["gate4_prompt_adherence"] for x in per_task])), 4),
            "gate5_novelty_rate": round(float(np.mean([x["gate5_novelty_lt30"] for x in per_task])), 4),
            "esmfold2_4gate_success_rate": round(float(np.mean([x["esmfold2_4gate_pass"] for x in per_task])), 4),
            "design_success_rate_5gate": round(float(np.mean([x["all_5_gates_pass"] for x in per_task])), 4),
            "mean_plddt": round(float(np.mean([x["plddt_mean"] or 0.0 for x in per_task])), 2),
            "fold_conditioned_4gate_rate": round(float(np.mean([x["esmfold2_4gate_pass"] for x in fc_sub])), 4),
            "motif_scaffolding_4gate_rate": round(float(np.mean([x["esmfold2_4gate_pass"] for x in ms_sub])), 4),
            "mean_motif_ca_rmsd": round(
                float(np.mean([x["motif_ca_rmsd"] for x in ms_sub if x["motif_ca_rmsd"] is not None] or [99.9])), 3
            ),
            "per_task": per_task,
        }
        report_models[mtag] = summary
        print(
            f"  {mtag:<28s} | ESMFold2 4-Gate={summary['esmfold2_4gate_success_rate']*100:5.1f}% "
            f"(FoldCond={summary['fold_conditioned_4gate_rate']*100:4.1f}%, Motif={summary['motif_scaffolding_4gate_rate']*100:4.1f}%) | "
            f"5-Gate(+OF3)={summary['design_success_rate_5gate']*100:5.1f}% | Mean pLDDT={summary['mean_plddt']:5.1f}",
            flush=True,
        )

    out_f = res_dir / "test7_esmfold2_evaluation_report.json"
    out_f.write_text(json.dumps({"models": report_models}, indent=2), encoding="utf-8")
    print(f"Saved Test 7 report -> {out_f}", flush=True)


def main() -> None:
    with JapanFoldClient(repo_root=REPO, max_usd_budget=25.0) as jf:
        bal = jf.get_account_balance()
        print(f"JapanFold starting balance: ${bal:.2f} USD", flush=True)
        evaluate_test2(jf)
        evaluate_test7(jf)
        print(f"\nDone! Total JapanFold session spend: ${jf.cumulative_usd:.4f} USD", flush=True)


if __name__ == "__main__":
    main()
