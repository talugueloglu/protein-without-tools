#!/usr/bin/env python3
"""Offline sandboxed runner for:
  - Task 1 (`Forward folding`, N=100): ESMFold-2 single-sequence baseline from cached folds
  - Task 2 (`Inverse folding`, N=100): ProteinMPNN (v_48_020, T=0.1) sequence generation on stripped UNK backbones
"""

from __future__ import annotations

from collections import defaultdict
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Dict, List, Tuple

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
if str(REPO / "src") not in sys.path:
    sys.path.insert(0, str(REPO / "src"))
if str(REPO / "outputs" / "proteinmpnn_cache") not in sys.path:
    sys.path.insert(0, str(REPO / "outputs" / "proteinmpnn_cache"))

from protein_without_tools.pdb_utils import ProteinStructure
from protein_without_tools.stage1_grader import _contact_precision_at_k, grade_inverse_prediction
from protein_without_tools.stage1_prompts_and_parsers import parse_inverse_folding_response
from protein_without_tools.tmscore import evaluate_3d_superposition
from protein_mpnn_utils import ProteinMPNN, _S_to_seq, parse_PDB, tied_featurize


def _seq_hash(model_name: str, seq: str) -> str:
    return hashlib.sha1(f"{model_name}:{seq}".encode("utf-8")).hexdigest()[:20]


def load_stage1_structures() -> Tuple[List[Dict[str, Any]], Dict[str, ProteinStructure]]:
    s1_dir = REPO / "data" / "stage1_benchmark"
    manifest = json.loads((s1_dir / "manifest.json").read_text(encoding="utf-8"))["targets"]
    structs: Dict[str, ProteinStructure] = {}
    for rec in manifest:
        tid = rec["target_id"]
        d = np.load(s1_dir / "ground_truth" / f"{tid}.npz", allow_pickle=False)
        structs[tid] = ProteinStructure(
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
    return manifest, structs


def evaluate_task1_esmfold2(manifest: List[Dict[str, Any]], structs: Dict[str, ProteinStructure]) -> None:
    store = REPO / "outputs" / "japanfold_cache" / "seq_store_esmfold2"
    per_target: Dict[str, Dict[str, Any]] = {}
    by_strat: Dict[str, List[Dict[str, float]]] = defaultdict(list)

    for rec in manifest:
        tid = rec["target_id"]
        st = rec["stratum"]
        gt = structs[tid]
        h = _seq_hash("esmfold2", gt.sequence)
        meta = json.loads((store / f"{h}.json").read_text(encoding="utf-8"))
        arr = np.load(store / f"{h}.npz")
        pred_ca = np.asarray(arr["coords_ca"], dtype=np.float64)
        pred_cb = np.asarray(arr["coords_cb"], dtype=np.float64)
        pred_q3 = str(meta["q3"])

        sup = evaluate_3d_superposition(pred_ca, gt.coords_ca)
        L = gt.length
        q3_acc = sum(1 for a, b in zip(pred_q3, gt.q3) if a == b) / float(L)

        # Rank pairwise Cβ distances (smallest distance = highest contact confidence) for |i-j|>=24
        diff = pred_cb[:, None, :] - pred_cb[None, :, :]
        dcb = np.sqrt(np.sum(diff * diff, axis=-1))
        ii, jj = np.triu_indices(L, k=24)
        order = np.argsort(dcb[ii, jj])
        contacts = [
            (int(ii[k] + 1), int(jj[k] + 1), float(max(0.0, 20.0 - dcb[ii[k], jj[k]])))
            for k in order[: max(L * 2, 50)]
        ]
        p_l_24 = _contact_precision_at_k(contacts, gt.cb_dist_matrix, 24, max(1, L))

        row = {
            "tm_score": round(float(sup.tm_score), 4),
            "rmsd_ca": round(float(sup.rmsd_global), 3),
            "lddt_ca": round(float(sup.lddt_ca), 4),
            "p_at_l_sep24": round(float(p_l_24), 4),
            "q3_accuracy": round(float(q3_acc), 4),
            "plddt_mean": float(meta["plddt_mean"]),
            "stratum": st,
        }
        per_target[tid] = row
        by_strat[st].append(row)

    all_rows = list(per_target.values())
    out = {
        "model": "baseline_esmfold2",
        "overall": {
            "n": len(all_rows),
            "tm_score": round(float(np.mean([r["tm_score"] for r in all_rows])), 4),
            "p_at_l_sep24": round(float(np.mean([r["p_at_l_sep24"] for r in all_rows])), 4),
            "q3_accuracy": round(float(np.mean([r["q3_accuracy"] for r in all_rows])), 4),
            "plddt_mean": round(float(np.mean([r["plddt_mean"] for r in all_rows])), 2),
        },
        "by_stratum": {
            k: {
                "n": len(v),
                "tm_score": round(float(np.mean([r["tm_score"] for r in v])), 4),
                "p_at_l_sep24": round(float(np.mean([r["p_at_l_sep24"] for r in v])), 4),
                "q3_accuracy": round(float(np.mean([r["q3_accuracy"] for r in v])), 4),
            }
            for k, v in sorted(by_strat.items())
        },
        "per_target": per_target,
    }
    out_p = REPO / "outputs" / "results" / "baseline_t1_esmfold2.json"
    out_p.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print("=== TASK 1: ESMFold-2 (Single-Sequence) ===", flush=True)
    print("Overall:", out["overall"], flush=True)
    for k, v in out["by_stratum"].items():
        print(f"  {k:<32s}: n={v['n']:2d} | TM={v['tm_score']:.3f} | P@L={v['p_at_l_sep24']:.3f} | Q3={v['q3_accuracy']:.3f}", flush=True)


def load_proteinmpnn_model(device: torch.device) -> ProteinMPNN:
    ckpt_path = REPO / "outputs" / "proteinmpnn_cache" / "v_48_020.pt"
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    hidden_dim = 128
    num_layers = 3
    model = ProteinMPNN(
        ca_only=False,
        num_letters=21,
        node_features=hidden_dim,
        edge_features=hidden_dim,
        hidden_dim=hidden_dim,
        num_encoder_layers=num_layers,
        num_decoder_layers=num_layers,
        augment_eps=0.0,
        k_neighbors=ckpt["num_edges"],
    )
    model.to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model


def run_task2_proteinmpnn(manifest: List[Dict[str, Any]], structs: Dict[str, ProteinStructure]) -> None:
    device = torch.device("cpu")
    model = load_proteinmpnn_model(device)
    alphabet = "ACDEFGHIKLMNPQRSTVWYX"
    omit_AAs_np = np.array([1.0 if a == "X" else 0.0 for a in alphabet], dtype=np.float32)
    bias_AAs_np = np.zeros(len(alphabet), dtype=np.float32)

    bb_dir = REPO / "data" / "stage1_benchmark" / "backbones_stripped"
    pred_rows: List[Dict[str, Any]] = []
    nsr_all, nsr_core_all = [], []

    torch.manual_seed(42)
    np.random.seed(42)

    with torch.no_grad():
        for idx, rec in enumerate(manifest):
            tid = rec["target_id"]
            gt = structs[tid]
            pdb_path = str(bb_dir / f"{tid}.pdb")
            pdb_dict_list = parse_PDB(pdb_path, ca_only=False)
            batch = [copy.deepcopy(pdb_dict_list[0])] if False else pdb_dict_list
            (
                X,
                S,
                mask,
                lengths,
                chain_M,
                chain_encoding_all,
                chain_list_list,
                visible_list_list,
                masked_list_list,
                masked_chain_length_list_list,
                chain_M_pos,
                omit_AA_mask,
                residue_idx,
                dihedral_mask,
                tied_pos_list_of_lists_list,
                pssm_coef,
                pssm_bias,
                pssm_log_odds_all,
                bias_by_res_all,
                tied_beta,
            ) = tied_featurize(batch, device, None, None, None, None, None, None, ca_only=False)

            g = torch.Generator(device=device)
            g.manual_seed(42 + idx)
            randn = torch.randn(chain_M.shape, generator=g, device=device)
            out_dict = model.sample(
                X,
                randn,
                S,
                chain_M,
                chain_encoding_all,
                residue_idx,
                mask=mask,
                temperature=0.1,
                omit_AAs_np=omit_AAs_np,
                bias_AAs_np=bias_AAs_np,
                chain_M_pos=chain_M_pos,
                omit_AA_mask=omit_AA_mask,
                pssm_coef=pssm_coef,
                pssm_bias=pssm_bias,
                pssm_multi=0.0,
                pssm_log_odds_flag=False,
                pssm_log_odds_mask=None,
                pssm_bias_flag=False,
                bias_by_res=bias_by_res_all,
            )
            seq = _S_to_seq(out_dict["S"][0], chain_M[0])
            assert len(seq) == gt.length, f"{tid}: {len(seq)} != {gt.length}"

            ig = grade_inverse_prediction(
                gt,
                parse_inverse_folding_response(json.dumps({"sequence": seq}), gt.length),
            )
            nsr_all.append(ig.nsr_overall)
            nsr_core_all.append(ig.nsr_core)
            pred_rows.append(
                {
                    "target_id": tid,
                    "task": "stage1b_inverse_folding",
                    "length": gt.length,
                    "text": json.dumps({"sequence": seq, "rationale": "ProteinMPNN v_48_020 (T=0.1)"}),
                    "nsr_overall": round(ig.nsr_overall, 4),
                    "nsr_core": round(ig.nsr_core, 4),
                }
            )

    out_jsonl = REPO / "outputs" / "predictions" / "baseline_proteinmpnn_stage1b.jsonl"
    out_jsonl.write_text("\n".join(json.dumps(r) for r in pred_rows) + "\n", encoding="utf-8")
    print("\n=== TASK 2: ProteinMPNN (v_48_020, T=0.1) Sequence Generation ===", flush=True)
    print(
        f"Generated {len(pred_rows)} sequences -> {out_jsonl.name} | "
        f"Mean NSR={np.mean(nsr_all):.3f} | Mean Core NSR={np.mean(nsr_core_all):.3f}",
        flush=True,
    )


if __name__ == "__main__":
    mf, st = load_stage1_structures()
    evaluate_task1_esmfold2(mf, st)
    run_task2_proteinmpnn(mf, st)
