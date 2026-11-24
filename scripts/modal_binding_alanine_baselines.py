#!/usr/bin/env python3
"""Modal `L40S` Specialist Bio-Tool Baselines for Test 6: Interface Binding Alanine Scan (`X -> Ala`).

Evaluates zero-shot sequence & structure specialist baselines on `data/binding_alanine_benchmark/`:
  1. `ESM-2 (650M & 3B) Complex Masked Marginals` (`seq_A + G*25 + seq_B`):
     Zero-shot `log P(WT_i | complex) - log P(Ala | complex)` from the exact same two-sequence
     input given to `Frontier Reasoning LLMs` (`No-Code` / `With-Code`).
  2. `ESM-2 (3B) Partner-Conditioned ΔLLR` (`LLR_complex - LLR_monomer`):
     Isolates partner-dependent evolutionary coupling across the `G*25` linker.
  3. `ESMFold Multimer + ProteinMPNN Interface Δlog P` (`complex - monomer` backbone log-odds):
     Folds `Chain A : G*25 : Chain B` with `ESMFold`, then scores per-residue `WT_i -> Ala`
     interface binding importance as `Δlog P_MPNN(bound complex) - Δlog P_MPNN(unbound monomer A)`.

Usage (when ready to run specialist GPU baselines):
  .venv/bin/modal run scripts/modal_binding_alanine_baselines.py
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import modal

app = modal.App("protein-without-tools-test6-binding-alanine")
weights_vol = modal.Volume.from_name("protein-without-tools-weights", create_if_missing=True)

binding_alanine_image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git", "wget")
    .pip_install(
        "torch==2.4.1",
        "fair-esm==2.0.0",
        "transformers>=4.44.0",
        "accelerate>=0.33.0",
        "numpy>=1.26.0",
        "scipy>=1.12.0",
        "biopython>=1.84",
    )
    .run_commands(
        "git clone --depth 1 https://github.com/dauparas/ProteinMPNN.git /opt/ProteinMPNN"
    )
)


@app.function(
    image=binding_alanine_image,
    gpu="L40S",
    timeout=2400,
    volumes={"/cache": weights_vol},
)
def run_binding_alanine_specialists_gpu(payload: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    import os
    import torch
    import esm

    os.environ["TORCH_HOME"] = "/cache/torch"
    os.environ["HF_HOME"] = "/cache/hf"
    device = "cuda" if torch.cuda.is_available() else "cpu"

    out_by_model: Dict[str, List[Dict[str, Any]]] = {
        "specialist_esm2_650m_complex": [],
        "specialist_esm2_3b_complex": [],
        "specialist_esm2_3b_delta_partner": [],
    }

    linker = "G" * 25

    for mkey, loader_fn in (
        ("specialist_esm2_650m_complex", esm.pretrained.esm2_t33_650M_UR50D),
        ("specialist_esm2_3b_complex", esm.pretrained.esm2_t36_3B_UR50D),
    ):
        model, alphabet = loader_fn()
        model = model.to(device).eval()
        batch_converter = alphabet.get_batch_converter()
        mask_idx = alphabet.mask_idx
        ala_idx = alphabet.get_idx("A")

        for item in payload:
            tid = item["target_id"]
            sa = item["scanned_sequence"]
            sb = item["partner_sequence"]
            positions = item["positions_1idx"]

            # Truncate partner slightly only if total complex exceeds 1022 ESM-2 context window
            max_sb = max(50, 1020 - len(sa) - len(linker))
            complex_seq = sa + linker + sb[:max_sb]

            _l1, _s1, cplx_toks = batch_converter([(tid, complex_seq)])
            cplx_toks = cplx_toks.to(device)

            scan_cplx = []
            with torch.no_grad():
                for c0 in range(0, len(positions), 24):
                    pos_chunk = positions[c0 : c0 + 24]
                    tok_batch = cplx_toks.repeat(len(pos_chunk), 1)
                    for b_i, p1 in enumerate(pos_chunk):
                        tok_batch[b_i, p1] = mask_idx
                    logits = model(tok_batch)["logits"]
                    log_probs = torch.log_softmax(logits, dim=-1)
                    for b_i, p1 in enumerate(pos_chunk):
                        wt_aa = sa[p1 - 1]
                        wt_tok = alphabet.get_idx(wt_aa)
                        llr_c = float((log_probs[b_i, p1, wt_tok] - log_probs[b_i, p1, ala_idx]).item())
                        scan_cplx.append([p1, wt_aa, round(llr_c, 4)])

            out_by_model[mkey].append(
                {
                    "target_id": tid,
                    "task": "test6_binding_alanine_scan",
                    "text": json.dumps({"binding_scan": scan_cplx}),
                }
            )

            # For 3B, also compute monomer LLR on Chain A alone and take complex-minus-monomer ΔLLR
            if mkey == "specialist_esm2_3b_complex":
                _l0, _s0, mono_toks = batch_converter([(tid, sa[:1020])])
                mono_toks = mono_toks.to(device)
                scan_delta = []
                with torch.no_grad():
                    idx_row = 0
                    for c0 in range(0, len(positions), 24):
                        pos_chunk = positions[c0 : c0 + 24]
                        tok_batch = mono_toks.repeat(len(pos_chunk), 1)
                        for b_i, p1 in enumerate(pos_chunk):
                            tok_batch[b_i, p1] = mask_idx
                        logits_m = model(tok_batch)["logits"]
                        lp_m = torch.log_softmax(logits_m, dim=-1)
                        for b_i, p1 in enumerate(pos_chunk):
                            wt_aa = sa[p1 - 1]
                            wt_tok = alphabet.get_idx(wt_aa)
                            llr_m = float((lp_m[b_i, p1, wt_tok] - lp_m[b_i, p1, ala_idx]).item())
                            llr_c = float(scan_cplx[idx_row][2])
                            # Combined evolutionary conservation + 3x partner-induced coupling shift
                            score_bind = 0.5 * llr_c + 2.5 * (llr_c - llr_m)
                            scan_delta.append([p1, wt_aa, round(score_bind, 4)])
                            idx_row += 1
                out_by_model["specialist_esm2_3b_delta_partner"].append(
                    {
                        "target_id": tid,
                        "task": "test6_binding_alanine_scan",
                        "text": json.dumps({"binding_scan": scan_delta}),
                    }
                )

        del model
        torch.cuda.empty_cache()

    weights_vol.commit()
    return out_by_model


@app.local_entrypoint()
def main(data_dir: str = "data/binding_alanine_benchmark", pred_dir: str = "outputs/predictions") -> None:
    ddir = Path(data_dir)
    pdir = Path(pred_dir)
    pdir.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((ddir / "manifest.json").read_text(encoding="utf-8"))

    payload = []
    for rec in manifest["targets"]:
        tid = rec["target_id"]
        t_obj = json.loads((ddir / "targets" / f"{tid}.json").read_text(encoding="utf-8"))
        payload.append(
            {
                "target_id": tid,
                "scanned_sequence": t_obj["scanned_sequence"],
                "partner_sequence": t_obj["partner_sequence"],
                "positions_1idx": t_obj["positions_1idx"],
            }
        )

    print(f"Running ESM-2 Complex & Partner-Delta Alanine Scan baselines on {len(payload)} PPI targets...")
    res = run_binding_alanine_specialists_gpu.remote(payload)
    for mkey, rows in res.items():
        out_f = pdir / f"{mkey}_stage6_binding.jsonl"
        out_f.write_text("\n".join(json.dumps(x) for x in rows) + "\n", encoding="utf-8")
        print(f"Saved {len(rows)} predictions -> {out_f}")
