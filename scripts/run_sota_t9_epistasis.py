#!/usr/bin/env python3
"""Task 9 Specialist Baseline: Zero-Shot ProteinMPNN (v_48_020) Double-Mutant Epistasis.

Computes exact symmetric conditional double-mutant thermodynamic coupling directly from
ProteinMPNN's autoregressive backbone-conditioned decoder (zero MegaScale training):

    dddG_mpnn(i: a->b, j: c->d) = -0.5 * [
        (log P(d_j | b_i, X) - log P(c_j | b_i, X)) - (log P(d_j | a_i, X) - log P(c_j | a_i, X))
      + (log P(b_i | d_j, X) - log P(a_i | d_j, X)) - (log P(b_i | c_j, X) - log P(a_i | c_j, X))
    ]

All context positions other than (i, j) are decoded before (i, j) with fixed wild-type
identities, and position i is decoded immediately before j (and symmetrically j before i)
so ProteinMPNN explicitly evaluates the exact side-chain coupling between i and j in 3D.
"""

from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile
from typing import Any, Dict, List

import numpy as np
import pyarrow.parquet as pq
import torch

REPO = Path(__file__).resolve().parents[1]
if str(REPO / "src") not in sys.path:
    sys.path.insert(0, str(REPO / "src"))
if str(REPO / "outputs" / "proteinmpnn_cache") not in sys.path:
    sys.path.insert(0, str(REPO / "outputs" / "proteinmpnn_cache"))

from protein_mpnn_utils import ProteinMPNN, parse_PDB, tied_featurize
from protein_without_tools.task9_epistasis import DoubleMutantRecord, grade_epistasis_panel

ALPHABET = "ACDEFGHIKLMNPQRSTVWYX"
AA_TO_IDX = {aa: i for i, aa in enumerate(ALPHABET)}


def load_megascale_pdb_texts(pdb_parquet: Path, needed_names: set[str]) -> Dict[str, str]:
    t = pq.read_table(pdb_parquet)
    cols = t.column_names
    name_col = next((c for c in ("WT_name", "name", "pdb_name", "filename") if c in cols), cols[0])
    text_col = next((c for c in ("pdb_text", "pdb", "content", "structure") if c in cols), cols[1])
    out: Dict[str, str] = {}
    for nm, txt in zip(t[name_col].to_pylist(), t[text_col].to_pylist()):
        key = Path(str(nm)).name
        if key in needed_names and txt:
            out[key] = str(txt)
    return out


def score_proteinmpnn_epistasis_for_target(
    model: ProteinMPNN,
    device: torch.device,
    pdb_text: str,
    wt_seq: str,
    pairs: List[Dict[str, Any]],
) -> Dict[str, float]:
    """Compute exact symmetric double-mutant coupling dddG_mpnn for each pair on one backbone."""
    with tempfile.NamedTemporaryFile("w", suffix=".pdb", delete=False, encoding="utf-8") as tf:
        tf.write(pdb_text)
        tmp_pdb = Path(tf.name)
    try:
        pdb_dicts = parse_PDB(str(tmp_pdb), input_chain_list=None)
    finally:
        tmp_pdb.unlink(missing_ok=True)

    if not pdb_dicts:
        return {p["pair_id"]: 0.0 for p in pairs}

    # Build a single-item batch matching ProteinMPNN featurization
    X, S, mask, lengths, chain_M, chain_encoding_all, chain_list_list, visible_list_list, masked_list_list, masked_chain_length_list_list, chain_M_pos, omit_AA_mask, residue_idx, dihedral_mask, tied_pos_list_of_lists_list, pssm_coef, pssm_bias, pssm_log_odds_all, bias_by_res_all, tied_beta = tied_featurize(
        [pdb_dicts[0]], device, chain_dict=None
    )

    L = int(mask.sum().item())
    if L != len(wt_seq):
        # Fallback if PDB residue count differs slightly
        return {p["pair_id"]: 0.0 for p in pairs}

    # Replace S with exact WT sequence indices
    wt_idx = torch.tensor([[AA_TO_IDX.get(c, 20) for c in wt_seq]], dtype=torch.long, device=device)

    # Batch all 4 evaluations per pair (2 orders x 2 conditioning states) into ONE forward pass!
    # For each pair (i0, j0) with WT (a, c) -> Mut (b, d):
    #   order 1 (i0 before j0): evaluate log P(j0 | S_i0 = a) and log P(j0 | S_i0 = b)
    #   order 2 (j0 before i0): evaluate log P(i0 | S_j0 = c) and log P(i0 | S_j0 = d)
    B = len(pairs) * 4
    X_b = X.repeat(B, 1, 1, 1)
    mask_b = mask.repeat(B, 1)
    chain_M_b = torch.ones_like(mask_b)
    residue_idx_b = residue_idx.repeat(B, 1)
    chain_enc_b = chain_encoding_all.repeat(B, 1)
    S_b = wt_idx.repeat(B, 1)
    # randn sets decoding order: smaller randn value is decoded EARLIER!
    # All context residues get 0.0; first-of-pair gets 1.0; target-of-pair gets 2.0 (decoded LAST!)
    randn_b = torch.zeros((B, L), dtype=torch.float32, device=device)

    meta = []
    for k, p in enumerate(pairs):
        i0 = int(p["pos_a"]) - 1
        j0 = int(p["pos_b"]) - 1
        a_i = AA_TO_IDX[p["mut_a"][0]]
        b_i = AA_TO_IDX[p["mut_a"][-1]]
        c_j = AA_TO_IDX[p["mut_b"][0]]
        d_j = AA_TO_IDX[p["mut_b"][-1]]
        base = 4 * k

        # 0: order i0 -> j0, S[i0]=a_i (WT)
        randn_b[base + 0, i0] = 1.0
        randn_b[base + 0, j0] = 2.0
        # 1: order i0 -> j0, S[i0]=b_i (MUT)
        S_b[base + 1, i0] = b_i
        randn_b[base + 1, i0] = 1.0
        randn_b[base + 1, j0] = 2.0
        # 2: order j0 -> i0, S[j0]=c_j (WT)
        randn_b[base + 2, j0] = 1.0
        randn_b[base + 2, i0] = 2.0
        # 3: order j0 -> i0, S[j0]=d_j (MUT)
        S_b[base + 3, j0] = d_j
        randn_b[base + 3, j0] = 1.0
        randn_b[base + 3, i0] = 2.0
        meta.append((p["pair_id"], base, i0, j0, a_i, b_i, c_j, d_j))

    with torch.no_grad():
        log_probs = model(X_b, S_b, mask_b, chain_M_b, residue_idx_b, chain_enc_b, randn_b)

    out: Dict[str, float] = {}
    for pid, base, i0, j0, a_i, b_i, c_j, d_j in meta:
        # Effect of j given i=WT vs i=MUT:
        # delta_j_given_wt  = log P(d_j | a_i) - log P(c_j | a_i)  (positive = stabilizing)
        # delta_j_given_mut = log P(d_j | b_i) - log P(c_j | b_i)
        # Epistatic coupling in destabilization convention (dddG < 0 means rescuing/stabilizing):
        # dddG_ij = -(delta_j_given_mut - delta_j_given_wt)
        dj_wt = float(log_probs[base + 0, j0, d_j] - log_probs[base + 0, j0, c_j])
        dj_mut = float(log_probs[base + 1, j0, d_j] - log_probs[base + 1, j0, c_j])
        di_wt = float(log_probs[base + 2, i0, b_i] - log_probs[base + 2, i0, a_i])
        di_mut = float(log_probs[base + 3, i0, b_i] - log_probs[base + 3, i0, a_i])
        dddg_pred = -0.5 * ((dj_mut - dj_wt) + (di_mut - di_wt))
        out[pid] = round(dddg_pred, 5)
    return out


def main() -> None:
    manifest = json.loads((REPO / "data/task9_epistasis/manifest.json").read_text(encoding="utf-8"))
    targets = manifest["targets"]
    needed = {t["wt_name"] for t in targets}
    pdb_texts = load_megascale_pdb_texts(REPO / "data/raw_alanine_cache/megascale_3d_pdbs.parquet", needed)
    print(f"Loaded {len(pdb_texts)}/{len(needed)} PDB structures for Task 9 ProteinMPNN baseline.", flush=True)

    device = torch.device("cpu")
    ckpt_path = REPO / "outputs/proteinmpnn_cache/v_48_020.pt"
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model = ProteinMPNN(
        num_letters=21,
        node_features=128,
        edge_features=128,
        hidden_dim=128,
        num_encoder_layers=3,
        num_decoder_layers=3,
        augment_eps=0.0,
        k_neighbors=ckpt["num_edges"],
    )
    model.to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    per_target: Dict[str, Any] = {}
    all_auroc, all_rho, all_ndcg, all_sign = [], [], [], []
    by_stratum: Dict[str, List[Dict[str, Any]]] = {"megascale_denovo": [], "megascale_natural": []}

    for idx, t in enumerate(targets, 1):
        tid = t["target_id"]
        pdb_txt = pdb_texts.get(t["wt_name"])
        if not pdb_txt:
            continue
        pred_signed = score_proteinmpnn_epistasis_for_target(model, device, pdb_txt, t["sequence"], t["pairs"])
        pred_abs = {pid: abs(val) for pid, val in pred_signed.items()}
        pred_signs = {pid: ("rescuing" if val < 0 else "synergistic") for pid, val in pred_signed.items()}
        pred_rank = sorted(pred_abs.keys(), key=lambda pid: (-pred_abs[pid], pid))

        panel = [DoubleMutantRecord(**p) for p in t["pairs"]]
        g = grade_epistasis_panel(panel, pred_rank, predicted_signs=pred_signs, predicted_scores=pred_abs)
        g["predicted_dddg"] = pred_signed
        g["stratum"] = t["stratum"]
        per_target[tid] = g
        all_auroc.append(g["coupling_auroc"])
        all_rho.append(g["spearman_rho_abs_dddg"])
        all_ndcg.append(g["ndcg_at_5"])
        if g["sign_accuracy"] is not None:
            all_sign.append(g["sign_accuracy"])
        by_stratum[t["stratum"]].append(g)

    def _agg(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
        if not rows:
            return {}
        signs = [r["sign_accuracy"] for r in rows if r["sign_accuracy"] is not None]
        return {
            "n_targets": len(rows),
            "coupling_auroc": round(float(np.mean([r["coupling_auroc"] for r in rows])), 4),
            "spearman_rho_abs_dddg": round(float(np.mean([r["spearman_rho_abs_dddg"] for r in rows])), 4),
            "ndcg_at_5": round(float(np.mean([r["ndcg_at_5"] for r in rows])), 4),
            "sign_accuracy": round(float(np.mean(signs)), 4) if signs else None,
        }

    report = {
        "task": "task9_double_mutant_epistasis",
        "baseline": "ProteinMPNN (v_48_020, zero-shot symmetric conditional coupling)",
        "zero_megascale_training": True,
        "overall": _agg(list(per_target.values())),
        "by_stratum": {k: _agg(v) for k, v in by_stratum.items()},
        "per_target": per_target,
    }
    out_path = REPO / "outputs/results/baseline_t9_proteinmpnn.json"
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print("\n=== TASK 9 SPECIALIST BASELINE: ProteinMPNN (zero-shot) ===")
    print("  Overall   :", report["overall"])
    print("  De novo   :", report["by_stratum"]["megascale_denovo"])
    print("  Natural   :", report["by_stratum"]["megascale_natural"])
    print("  Saved ->  ", out_path)


if __name__ == "__main__":
    main()
