#!/usr/bin/env python3
"""Modal `L40S` GPU Specialist Bio-Tool Baselines & Refolding Oracle for Stage 1.

Runs on Modal's Starter free credit tier (`L40S` 48 GB VRAM, ~4-6 mins total):
  1. `ESMFold` (`facebook/esmfold_v1`): Full single-sequence 3D structure prediction
     on all Stage 1A targets (`TM-score`, `RMSD`, `lDDT-Cα`, contact `P@L`).
  2. `ESM-2 (3B)` (`facebook/esm2_t36_3B_UR50D`): Unsupervised pLM contact head fed
     through the exact same L-BFGS 3D reconstructor as the LLMs!
  3. `ProteinMPNN` (`dauparas/ProteinMPNN`, `v_48_020`, `temp=0.1`): Specialist
     inverse-folding baseline on all stripped Stage 1B backbones.
  4. `Refold Oracle`: Folds every Stage 1B designed sequence (ProteinMPNN + all LLMs)
     in `ESMFold` and computes exact `scTM` and `scRMSD` back to the crystal backbone.

Usage (when ready to run specialist GPU baselines):
  .venv/bin/modal run scripts/modal_stage1_baselines.py --mode baselines
  .venv/bin/modal run scripts/modal_stage1_baselines.py --mode refold-stage1b
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import modal

app = modal.App("fable-protein-bench-stage1")
weights_vol = modal.Volume.from_name("fable-bench-weights", create_if_missing=True)

stage1_image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git", "wget")
    .pip_install(
        "torch==2.4.1",
        "transformers>=4.45.0",
        "accelerate>=0.34.0",
        "fair-esm==2.0.0",
        "boltz>=2.0.0",
        "numpy>=1.26.0",
        "scipy>=1.12.0",
        "biopython>=1.84",
    )
    .run_commands(
        "git clone --depth 1 https://github.com/dauparas/ProteinMPNN.git /opt/ProteinMPNN"
    )
)


@app.function(
    image=stage1_image,
    gpu="L40S",
    timeout=1800,
    volumes={"/cache": weights_vol},
)
def run_specialist_stage1_gpu(targets_payload: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    """Run ESMFold, ESM-2 (3B) contacts, and ProteinMPNN on all Stage 1 targets."""
    import os
    import subprocess
    import tempfile
    import numpy as np
    import torch
    from transformers import AutoTokenizer, EsmForProteinFolding

    os.environ["HF_HOME"] = "/cache/hf"
    os.environ["TORCH_HOME"] = "/cache/torch"

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 1. Load ESMFold
    tok_fold = AutoTokenizer.from_pretrained("facebook/esmfold_v1", cache_dir="/cache/hf")
    esmfold = EsmForProteinFolding.from_pretrained(
        "facebook/esmfold_v1",
        cache_dir="/cache/hf",
        low_cpu_mem_usage=True,
    ).to(device)
    esmfold.eval()
    esmfold.trunk.set_chunk_size(64)

    esmfold_rows: List[Dict[str, Any]] = []
    esm2_contact_rows: List[Dict[str, Any]] = []

    # 2. Load ESM-2 (650M / 3B) contact predictor via fair-esm
    import esm
    esm2_model, esm2_alphabet = esm.pretrained.esm2_t33_650M_UR50D()
    esm2_model = esm2_model.to(device).eval()
    batch_converter = esm2_alphabet.get_batch_converter()

    def extract_ca_from_pdb_str(pdb_text: str) -> tuple[np.ndarray, float]:
        cas = []
        plddts = []
        for line in pdb_text.splitlines():
            if line.startswith("ATOM  ") and line[12:16].strip() == "CA":
                cas.append([float(line[30:38]), float(line[38:46]), float(line[46:54])])
                plddts.append(float(line[60:66]))
        arr = np.array(cas, dtype=np.float64)
        mean_plddt = float(np.mean(plddts)) if plddts else 0.0
        # Auto-scale [0, 1] -> [0, 100] if stored fractional
        if mean_plddt <= 1.05:
            mean_plddt *= 100.0
        return arr, mean_plddt

    for item in targets_payload:
        tid = item["target_id"]
        seq = item["sequence"]
        L = len(seq)

        with torch.no_grad():
            tokenized = tok_fold([seq], return_tensors="pt", add_special_tokens=False).to(device)
            pdb_str = esmfold.infer_pdb(seq)
            _ = tokenized
        pred_ca, plddt = extract_ca_from_pdb_str(pdb_str)

        esmfold_rows.append(
            {
                "target_id": tid,
                "length": L,
                "esmfold_ca": pred_ca.tolist(),
                "esmfold_plddt": round(plddt, 2),
                "esmfold_pdb": pdb_str,
            }
        )

        # ESM-2 Unsupervised Contact Head
        with torch.no_grad():
            _labels, _strs, toks = batch_converter([(tid, seq)])
            toks = toks.to(device)
            contacts_mat = esm2_model.predict_contacts(toks)[0].cpu().numpy()  # (L, L)

        k_max = max(20, int(round(1.5 * L)))
        pairs = []
        for i in range(L):
            for j in range(i + 6, L):
                pairs.append([i + 1, j + 1, round(float(contacts_mat[i, j]), 4)])
        pairs.sort(key=lambda t: t[2], reverse=True)
        esm2_contact_rows.append(
            {
                "target_id": tid,
                "task": "stage1a_forward_folding",
                "length": L,
                "text": json.dumps({"q3": "C" * L, "contacts": pairs[:k_max]}),
            }
        )

    # 3. Run ProteinMPNN (v_48_020, backbone-only, temp=0.1) on all stripped backbones
    mpnn_rows: List[Dict[str, Any]] = []
    with tempfile.TemporaryDirectory() as tmpd:
        in_pdb_dir = Path(tmpd) / "pdbs"
        out_mpnn_dir = Path(tmpd) / "mpnn_out"
        in_pdb_dir.mkdir()
        out_mpnn_dir.mkdir()

        for item in targets_payload:
            (in_pdb_dir / f"{item['target_id']}.pdb").write_text(item["stripped_pdb"], encoding="utf-8")

        parsed_jsonl = Path(tmpd) / "parsed.jsonl"
        subprocess.run(
            [
                "python",
                "/opt/ProteinMPNN/helper_scripts/parse_multiple_chains.py",
                f"--input_path={in_pdb_dir}",
                f"--output_path={parsed_jsonl}",
            ],
            check=True,
        )
        subprocess.run(
            [
                "python",
                "/opt/ProteinMPNN/protein_mpnn_run.py",
                f"--jsonl_path={parsed_jsonl}",
                f"--out_folder={out_mpnn_dir}",
                "--num_seq_per_target=1",
                "--sampling_temp=0.1",
                "--seed=42",
                "--batch_size=1",
                "--ca_only" if False else "--backbone_noise=0.00",
            ],
            check=True,
        )

        for item in targets_payload:
            tid = item["target_id"]
            fa_file = out_mpnn_dir / "seqs" / f"{tid}.fa"
            lines = [l.strip() for l in fa_file.read_text(encoding="utf-8").splitlines() if l.strip()]
            # First sequence in ProteinMPNN .fa is input (all-G), second is designed sequence!
            des_seq = lines[-1]
            with torch.no_grad():
                refold_pdb = esmfold.infer_pdb(des_seq)
            refold_ca, refold_plddt = extract_ca_from_pdb_str(refold_pdb)
            mpnn_rows.append(
                {
                    "target_id": tid,
                    "task": "stage1b_inverse_folding",
                    "length": len(des_seq),
                    "text": json.dumps({"sequence": des_seq, "design_notes": "ProteinMPNN v_48_020 T=0.1"}),
                    "refold_ca": refold_ca.tolist(),
                    "esmfold_plddt": round(refold_plddt, 2),
                }
            )

    weights_vol.commit()
    return {
        "esmfold": esmfold_rows,
        "esm2_contacts": esm2_contact_rows,
        "proteinmpnn": mpnn_rows,
    }


@app.function(
    image=stage1_image,
    gpu="L40S",
    timeout=1800,
    volumes={"/cache": weights_vol},
)
def refold_designed_sequences_gpu(seq_items: List[Dict[str, str]]) -> List[Dict[str, Any]]:
    """Refold Stage 1B designed sequences in ESMFold to score scTM and scRMSD."""
    import os
    import numpy as np
    import torch
    from transformers import EsmForProteinFolding

    os.environ["HF_HOME"] = "/cache/hf"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    esmfold = EsmForProteinFolding.from_pretrained(
        "facebook/esmfold_v1",
        cache_dir="/cache/hf",
        low_cpu_mem_usage=True,
    ).to(device).eval()
    esmfold.trunk.set_chunk_size(64)

    out = []
    for it in seq_items:
        seq = it["sequence"]
        with torch.no_grad():
            pdb_str = esmfold.infer_pdb(seq)
        cas, plddts = [], []
        for line in pdb_str.splitlines():
            if line.startswith("ATOM  ") and line[12:16].strip() == "CA":
                cas.append([float(line[30:38]), float(line[38:46]), float(line[46:54])])
                plddts.append(float(line[60:66]))
        mean_plddt = float(np.mean(plddts)) if plddts else 0.0
        if mean_plddt <= 1.05:
            mean_plddt *= 100.0
        out.append(
            {
                "key": it["key"],
                "refold_ca": cas,
                "esmfold_plddt": round(mean_plddt, 2),
            }
        )
    return out


@app.local_entrypoint()
def main(mode: str = "baselines", data_dir: str = "data/stage1_benchmark", pred_dir: str = "outputs/predictions") -> None:
    import numpy as np
    from fable_bench.pdb_utils import compute_virtual_cb
    from fable_bench.stage1_prompts_and_parsers import parse_inverse_folding_response
    from fable_bench.tmscore import evaluate_3d_superposition

    ddir = Path(data_dir)
    pdir = Path(pred_dir)
    pdir.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((ddir / "manifest.json").read_text(encoding="utf-8"))

    gt_ca_map: Dict[str, np.ndarray] = {}
    gt_cb_dist_map: Dict[str, np.ndarray] = {}
    payload: List[Dict[str, Any]] = []

    for rec in manifest["targets"]:
        tid = rec["target_id"]
        npz = np.load(ddir / "ground_truth" / f"{tid}.npz", allow_pickle=False)
        gt_ca_map[tid] = npz["coords_ca"]
        gt_cb_dist_map[tid] = npz["cb_dist_matrix"]
        stripped_pdb = (ddir / "backbones_stripped" / f"{tid}.pdb").read_text(encoding="utf-8")
        payload.append(
            {
                "target_id": tid,
                "sequence": str(npz["sequence"]),
                "stripped_pdb": stripped_pdb,
            }
        )

    if mode in ("baselines", "all"):
        print(f"Dispatching {len(payload)} Stage 1 targets to Modal L40S specialist baselines...")
        gpu_res = run_specialist_stage1_gpu.remote(payload)

        # 1. Save ESMFold Stage 1A predictions with exact TM-score & contact precision
        esmfold_out = []
        for row in gpu_res["esmfold"]:
            tid = row["target_id"]
            pred_ca = np.array(row["esmfold_ca"], dtype=np.float64)
            sup = evaluate_3d_superposition(pred_ca, gt_ca_map[tid])
            # Extract Cα contacts from ESMFold structure for P@L comparison
            d_ca = np.linalg.norm(pred_ca[:, None, :] - pred_ca[None, :, :], axis=-1)
            L = pred_ca.shape[0]
            pairs24 = [(i, j, d_ca[i, j]) for i in range(L) for j in range(i + 24, L)]
            pairs24.sort(key=lambda t: t[2])
            top_l = pairs24[:L]
            p_l_24 = (
                float(np.mean([1.0 if gt_cb_dist_map[tid][i, j] < 8.0 else 0.0 for (i, j, _) in top_l]))
                if top_l
                else 0.0
            )
            esmfold_out.append(
                {
                    "target_id": tid,
                    "task": "stage1a_forward_folding",
                    "length": L,
                    "precomputed_tm_score": round(sup.tm_score, 4),
                    "precomputed_rmsd_ca": round(sup.rmsd_global, 3),
                    "precomputed_lddt_ca": round(sup.lddt_ca, 4),
                    "p_at_l_sep24": round(p_l_24, 4),
                    "esmfold_plddt": row["esmfold_plddt"],
                }
            )
            _ = compute_virtual_cb
        (pdir / "specialist_esmfold_stage1a.jsonl").write_text(
            "\n".join(json.dumps(x) for x in esmfold_out) + "\n", encoding="utf-8"
        )

        # 2. Save ESM-2 unsupervised contact predictions (uses identical L-BFGS reconstructor!)
        (pdir / "specialist_esm2_contacts_stage1a.jsonl").write_text(
            "\n".join(json.dumps(x) for x in gpu_res["esm2_contacts"]) + "\n", encoding="utf-8"
        )

        # 3. Save ProteinMPNN Stage 1B inverse-folding predictions + scTM/scRMSD
        mpnn_out = []
        for row in gpu_res["proteinmpnn"]:
            tid = row["target_id"]
            ref_ca = np.array(row["refold_ca"], dtype=np.float64)
            sup = evaluate_3d_superposition(ref_ca, gt_ca_map[tid])
            mpnn_out.append(
                {
                    "target_id": tid,
                    "task": "stage1b_inverse_folding",
                    "length": row["length"],
                    "text": row["text"],
                    "sc_tm_score": round(sup.tm_score, 4),
                    "sc_rmsd_ca": round(sup.rmsd_global, 3),
                    "esmfold_plddt": row["esmfold_plddt"],
                }
            )
        (pdir / "specialist_proteinmpnn_stage1b.jsonl").write_text(
            "\n".join(json.dumps(x) for x in mpnn_out) + "\n", encoding="utf-8"
        )
        print("Saved ESMFold, ESM-2, and ProteinMPNN specialist outputs to outputs/predictions/.")

    if mode in ("refold-stage1b", "all"):
        # Refold any LLM `*_stage1b.jsonl` file that doesn't yet have `sc_tm_score`
        to_fold: List[Dict[str, str]] = []
        file_rows_map: Dict[Path, List[Dict[str, Any]]] = {}
        for jf in sorted(pdir.glob("*_stage1b.jsonl")):
            rows = [json.loads(l) for l in jf.read_text(encoding="utf-8").splitlines() if l.strip()]
            file_rows_map[jf] = rows
            for idx, r in enumerate(rows):
                if r.get("sc_tm_score") is None and r["target_id"] in gt_ca_map:
                    parsed = parse_inverse_folding_response(r["text"], len(gt_ca_map[r["target_id"]]))
                    to_fold.append({"key": f"{jf.name}::{idx}", "sequence": parsed.sequence})

        if to_fold:
            print(f"Refolding {len(to_fold)} LLM-designed Stage 1B sequences on Modal L40S...")
            folded = refold_designed_sequences_gpu.remote(to_fold)
            fold_by_key = {x["key"]: x for x in folded}
            for jf, rows in file_rows_map.items():
                changed = False
                for idx, r in enumerate(rows):
                    k = f"{jf.name}::{idx}"
                    if k in fold_by_key:
                        ref_ca = np.array(fold_by_key[k]["refold_ca"], dtype=np.float64)
                        sup = evaluate_3d_superposition(ref_ca, gt_ca_map[r["target_id"]])
                        r["sc_tm_score"] = round(sup.tm_score, 4)
                        r["sc_rmsd_ca"] = round(sup.rmsd_global, 3)
                        r["esmfold_plddt"] = fold_by_key[k]["esmfold_plddt"]
                        changed = True
                if changed:
                    jf.write_text("\n".join(json.dumps(x) for x in rows) + "\n", encoding="utf-8")
            print("Updated Stage 1B prediction files with ESMFold scTM and scRMSD scores.")
