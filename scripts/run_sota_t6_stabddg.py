#!/usr/bin/env python3
"""Offline Task 6 SOTA Baseline: Boltz-2 -> StaB-ddG (`stability_finetuned.pt`, ICML 2025).

Uses ONLY the exact two-chain sequence input (folded via Boltz-2, zero crystal structures)
and StaB-ddG's `stability_finetuned.pt` checkpoint (pre-trained on MegaScale folding stability
only — ZERO SKEMPI v2.0 binding training leakage!) via its exact thermodynamic cycle:
    ddG_bind = -(ddG_fold(complex) - ddG_fold(chain_A) - ddG_fold(chain_B))
averaged over 20 antithetic-variate MC decoding orders (noise_level=0.1).

Uses an exact O(L*K) rank-based decoder mask and single-pass encoder per MC sample
(mathematically identical to upstream StaBddG.folding_ddG, verified to <1e-5 on s6_b001).
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import sys
import tempfile
import time
from typing import Any, Dict, List

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl_proteinbench")

import numpy as np
import torch
import torch.nn.functional as F

torch.set_num_threads(max(4, os.cpu_count() or 8))

REPO = Path(__file__).resolve().parents[1]
STAB_DIR = REPO / "outputs" / "proteinmpnn_cache" / "StaB-ddG"
for p in (REPO / "src", REPO / "scripts", STAB_DIR):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from evaluate_binding_alanine import load_binding_targets, summarize_grades
from fable_bench.binding_alanine_scan import grade_binding_alanine_target, parse_binding_alanine_response
from stabddg.model import StaBddG
from stabddg.mpnn_utils import ProteinMPNN, cat_neighbors_nodes, featurize, gather_nodes, parse_PDB
from stabddg.utils import extract_chains


ALPHABET = "ACDEFGHIKLMNPQRSTVWYX"


def _prep_mpnn_dict(pdb_path: Path) -> Dict[str, Any]:
    d = parse_PDB(str(pdb_path))[0]
    all_chains = [k.split("_")[-1] for k in d.keys() if k.startswith("seq_chain_")]
    d["masked_list"] = all_chains
    d["visible_list"] = []
    return d


def _build_mut_tensor(struct_dict: Dict[str, Any], positions_1idx: List[int], target_chain: str) -> torch.Tensor:
    chain_offset: Dict[str, int] = {}
    offset = 0
    for k in struct_dict.keys():
        if k.startswith("seq_chain_"):
            chain_offset[k.split("_")[-1]] = offset
            offset += len(struct_dict[k])

    base_seq = list(struct_dict["seq"])
    rows = []
    for p1 in positions_1idx:
        mut_seq = list(base_seq)
        if target_chain in chain_offset:
            idx0 = chain_offset[target_chain] + (p1 - 1)
            mut_seq[idx0] = "A"
        idxs = [ALPHABET.index(a) if a in ALPHABET else ALPHABET.index("X") for a in mut_seq]
        rows.append(idxs)
    return torch.tensor(rows, dtype=torch.long)


@torch.no_grad()
def fast_folding_ddG(
    pmpnn: ProteinMPNN,
    domain: Dict[str, Any],
    mut_seqs: torch.Tensor,
    noise_level: float,
    device: torch.device,
    fixed_order: torch.Tensor | None = None,
    fixed_noise: torch.Tensor | None = None,
) -> torch.Tensor:
    """Exact bit-identical equivalent of StaBddG.folding_ddG with single encoder pass and O(L*K) mask."""
    X, wt_seq, mask, _, chain_M, residue_idx, _, chain_encoding_all = featurize([domain], device)
    # 1. Sample shared antithetic decoding order and backbone noise
    order = fixed_order if fixed_order is not None else torch.argsort(torch.abs(torch.randn(chain_M.shape, device=device)))
    noise = fixed_noise if fixed_noise is not None else noise_level * torch.randn_like(X, device=device)

    # 2. Run structure encoder ONCE (independent of sequence S!)
    E, E_idx = pmpnn.features(X, mask, residue_idx, chain_encoding_all, backbone_noise=noise)
    h_V_enc = torch.zeros((E.shape[0], E.shape[1], E.shape[-1]), device=device)
    h_E_enc = pmpnn.W_e(E)
    mask_attend_enc = gather_nodes(mask.unsqueeze(-1), E_idx).squeeze(-1)
    mask_attend_enc = mask.unsqueeze(-1) * mask_attend_enc
    for layer in pmpnn.encoder_layers:
        h_V_enc, h_E_enc = layer(h_V_enc, h_E_enc, E_idx, mask, mask_attend_enc)

    # 3. Exact O(L*K) backward/forward causal attention mask from permutation `order`
    # Upstream: order_mask_backward[q, p] = 1 iff rank(p) < rank(q), where rank = inverse permutation of order
    rank = torch.argsort(order, dim=-1)  # (1, L)
    # E_idx is (1, L, K); rank_neighbors[0, q, k] = rank[0, E_idx[0, q, k]]
    rank_nbr = torch.gather(rank.unsqueeze(1).expand(-1, E_idx.shape[1], -1), 2, E_idx)  # (1, L, K)
    mask_attend_dec = (rank_nbr < rank.unsqueeze(-1)).float().unsqueeze(-1)  # (1, L, K, 1)
    mask_1D = mask.view(1, mask.size(1), 1, 1)
    mask_bw = mask_1D * mask_attend_dec
    mask_fw = mask_1D * (1.0 - mask_attend_dec)

    # 4. Run decoder batched over [WT] + all M mutant sequences
    all_seqs = torch.cat([wt_seq.to(device), mut_seqs.to(device)], dim=0)  # (1+M, L)
    B = all_seqs.shape[0]
    h_V = h_V_enc.expand(B, -1, -1).contiguous()
    h_E = h_E_enc.expand(B, -1, -1, -1)
    E_idx_b = E_idx.expand(B, -1, -1)
    mask_b = mask.expand(B, -1)

    h_S = pmpnn.W_s(all_seqs)
    h_ES = cat_neighbors_nodes(h_S, h_E, E_idx_b)
    h_EX_encoder = cat_neighbors_nodes(torch.zeros_like(h_S[:1]), h_E_enc, E_idx)
    h_EXV_encoder_fw = (mask_fw * cat_neighbors_nodes(h_V_enc, h_EX_encoder, E_idx)).expand(B, -1, -1, -1)

    for layer in pmpnn.decoder_layers:
        h_ESV = cat_neighbors_nodes(h_V, h_ES, E_idx_b)
        h_ESV = mask_bw * h_ESV + h_EXV_encoder_fw
        h_V = layer(h_V, h_ESV, mask_b)

    log_probs = F.log_softmax(pmpnn.W_out(h_V), dim=-1)  # (1+M, L, 21)
    # Sum log_probs at each sequence's residues (identical to sum(one_hot(S, 21) * log_probs))
    dG_all = log_probs.gather(2, all_seqs.unsqueeze(-1)).squeeze(-1).sum(dim=1)  # (1+M,)
    return dG_all[1:] - dG_all[0]


def main() -> None:
    ckpt_path = STAB_DIR / "model_ckpts" / "stability_finetuned.pt"
    pdb_dir = REPO / "outputs" / "japanfold_cache" / "t6_boltz2_pdbs"
    s6_dir = REPO / "data" / "binding_alanine_benchmark"

    assert ckpt_path.exists(), f"Missing StaB-ddG checkpoint: {ckpt_path}"
    device = torch.device("cpu")
    torch.manual_seed(0)

    print(f"=== Loading StaB-ddG (stability_finetuned.pt, ZERO SKEMPI fine-tuning) on {device} ===", flush=True)
    pmpnn = ProteinMPNN(
        node_features=128,
        edge_features=128,
        hidden_dim=128,
        num_encoder_layers=3,
        num_decoder_layers=3,
        k_neighbors=48,
        dropout=0.0,
        augment_eps=0.0,
    )
    ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    pmpnn.load_state_dict(ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt)
    pmpnn = pmpnn.to(device).eval()

    t6_targets = load_binding_targets(s6_dir)
    mc_samples = 20
    t0 = time.monotonic()

    t6_rows: List[Dict[str, Any]] = []
    grades = []
    verified_exact = False

    print(f"=== [Task 6] Scoring {len(t6_targets)} Boltz-2 complexes with StaB-ddG (mc_samples={mc_samples}) ===", flush=True)
    with tempfile.TemporaryDirectory() as td:
        td_p = Path(td)
        for idx_t, (tid, t_obj) in enumerate(sorted(t6_targets.items()), start=1):
            complex_pdb = pdb_dir / f"{tid}.pdb"
            if not complex_pdb.exists():
                raise FileNotFoundError(f"Missing Boltz-2 complex PDB for {tid}: {complex_pdb}")

            a_pdb = td_p / f"{tid}_A.pdb"
            b_pdb = td_p / f"{tid}_B.pdb"
            extract_chains(str(complex_pdb), str(a_pdb), "A", str(b_pdb), "B")

            c_dict = _prep_mpnn_dict(complex_pdb)
            a_dict = _prep_mpnn_dict(a_pdb)

            sa = t_obj.scanned_sequence.strip().upper()
            sb = t_obj.partner_sequence.strip().upper()
            if c_dict.get("seq_chain_A") != sa or c_dict.get("seq_chain_B") != sb:
                raise RuntimeError(
                    f"Chain sequence mismatch for {tid}: "
                    f"A={len(c_dict.get('seq_chain_A',''))}/{len(sa)}, B={len(c_dict.get('seq_chain_B',''))}/{len(sb)}"
                )

            positions = t_obj.positions_1idx
            c_mut = _build_mut_tensor(c_dict, positions, "A").to(device)
            a_mut = _build_mut_tensor(a_dict, positions, "A").to(device)

            # One-time bit-exact verification against upstream StaBddG on monomer A of first target
            if not verified_exact:
                ref_model = StaBddG(pmpnn=pmpnn, use_antithetic_variates=True, noise_level=0.1, device=device).eval()
                X0, _, _, _, cm0, _, _, _ = featurize([a_dict], device)
                torch.manual_seed(123)
                ref_out = ref_model.folding_ddG(a_dict, a_mut[:2])
                torch.manual_seed(123)
                ord0 = torch.argsort(torch.abs(torch.randn(cm0.shape, device=device)))
                noi0 = 0.1 * torch.randn_like(X0, device=device)
                fast_out = fast_folding_ddG(pmpnn, a_dict, a_mut[:2], 0.1, device, fixed_order=ord0, fixed_noise=noi0)
                max_err = float((ref_out - fast_out).abs().max().item())
                print(f"  [Verify] Exact equivalence vs upstream StaBddG: max_abs_diff = {max_err:.2e}", flush=True)
                assert max_err < 1e-4, f"Mismatch vs upstream StaBddG: {max_err}"
                verified_exact = True

            # Chain B has zero mutations so binder2_ddG_fold == 0 identically
            preds_ens = []
            with torch.no_grad():
                for _ in range(mc_samples):
                    c_ddg = fast_folding_ddG(pmpnn, c_dict, c_mut, 0.1, device)
                    a_ddg = fast_folding_ddG(pmpnn, a_dict, a_mut, 0.1, device)
                    preds_ens.append((c_ddg - a_ddg).detach().cpu())

            # Negate so positive = binding loss (higher = stronger hotspot, matching run_stabddg.py line 37)
            mean_ddg = (-1.0 * torch.stack(preds_ens, dim=0).mean(dim=0)).numpy()
            scan_rows = [
                [p1, sa[p1 - 1], round(float(mean_ddg[k]), 4)]
                for k, p1 in enumerate(positions)
            ]
            txt = json.dumps({"binding_scan": scan_rows})
            t6_rows.append({"target_id": tid, "task": "test6_binding_alanine_scan", "text": txt})
            grades.append(grade_binding_alanine_target(t_obj, parse_binding_alanine_response(txt, t_obj)))

            if idx_t % 5 == 0 or idx_t == len(t6_targets):
                print(f"  [T6 StaB-ddG] {idx_t}/{len(t6_targets)} done ({time.monotonic()-t0:.1f}s)", flush=True)

    pred_out = REPO / "outputs" / "predictions" / "baseline_stabddg_stage6_binding.jsonl"
    pred_out.write_text("\n".join(json.dumps(r) for r in t6_rows) + "\n", encoding="utf-8")

    sum6 = summarize_grades("baseline_stabddg", grades)
    res_out = REPO / "outputs" / "results" / "baseline_t6_stabddg.json"
    res_out.write_text(json.dumps(sum6, indent=2), encoding="utf-8")

    print("\n=== TASK 6 SOTA: Boltz-2 -> StaB-ddG (stability_finetuned.pt, ZERO SKEMPI) ===", flush=True)
    print("  Overall:", sum6["overall"], flush=True)
    for k, v in sum6["by_memorization_stratum"].items():
        print(
            f"  {k:<24s}: n={v['n']:2d} | Rho={v['spearman_rho']:.3f} | "
            f"NDCG@5={v['ndcg_at_5']:.3f} | SameAA={v['same_aa_win_rate']:.3f}",
            flush=True,
        )


if __name__ == "__main__":
    main()
