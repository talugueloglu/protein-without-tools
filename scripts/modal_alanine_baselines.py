#!/usr/bin/env python3
"""Modal `L40S` Specialist Bio-Tool Baselines for Stage 3A In-Silico Alanine Scan (`X_i -> A`).

Runs head-to-head specialist baselines on all Stage 3A Alanine Scan targets:
  1. `ESM-2 (650M & 3B)` (`facebook/esm2_t33_650M_UR50D`, `esm2_t36_3B_UR50D`):
     Exact zero-shot masked marginal log-odds `log P(WT_i | seq_{-i}) - log P(Ala | seq_{-i})`
     from 1D sequence alone (same input modality as `Claude Fable 5.2`!).
  2. `ThermoMPNN` (`Kuhlman-Lab/ThermoMPNN`):
     Dedicated 3D-structure-conditioned ΔΔG_fold (kcal/mol) predictor on targets with 3D PDBs!

Usage (when ready to run specialist GPU baselines):
  .venv/bin/modal run scripts/modal_alanine_baselines.py
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import modal

app = modal.App("fable-protein-bench-stage3a-alanine")
weights_vol = modal.Volume.from_name("fable-bench-weights", create_if_missing=True)

alanine_image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git", "wget")
    .pip_install(
        "torch==2.4.1",
        "fair-esm==2.0.0",
        "numpy>=1.26.0",
        "scipy>=1.12.0",
        "biopython>=1.84",
        "omegaconf>=2.3.0",
        "pytorch-lightning>=2.2.0",
    )
    .run_commands(
        "git clone --depth 1 https://github.com/Kuhlman-Lab/ThermoMPNN.git /opt/ThermoMPNN"
    )
)


@app.function(
    image=alanine_image,
    gpu="L40S",
    timeout=1800,
    volumes={"/cache": weights_vol},
)
def run_alanine_specialists_gpu(payload: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    import os
    import torch
    import esm

    os.environ["TORCH_HOME"] = "/cache/torch"
    device = "cuda" if torch.cuda.is_available() else "cpu"

    out_by_model: Dict[str, List[Dict[str, Any]]] = {
        "specialist_esm2_650m": [],
        "specialist_esm2_3b": [],
    }

    for mkey, loader_fn in (
        ("specialist_esm2_650m", esm.pretrained.esm2_t33_650M_UR50D),
        ("specialist_esm2_3b", esm.pretrained.esm2_t36_3B_UR50D),
    ):
        model, alphabet = loader_fn()
        model = model.to(device).eval()
        batch_converter = alphabet.get_batch_converter()
        mask_idx = alphabet.mask_idx
        ala_idx = alphabet.get_idx("A")

        for item in payload:
            tid = item["target_id"]
            seq = item["sequence"]
            positions = item["positions_1idx"]
            _lbl, _s, base_toks = batch_converter([(tid, seq)])
            base_toks = base_toks.to(device)  # (1, L + 2), 1-indexed positions match token index!

            scan_rows = []
            with torch.no_grad():
                # Batch masked forward passes in chunks of 32 positions
                for c0 in range(0, len(positions), 32):
                    pos_chunk = positions[c0 : c0 + 32]
                    tok_batch = base_toks.repeat(len(pos_chunk), 1)
                    for b_i, p1 in enumerate(pos_chunk):
                        tok_batch[b_i, p1] = mask_idx
                    logits = model(tok_batch)["logits"]  # (B, L+2, V)
                    log_probs = torch.log_softmax(logits, dim=-1)
                    for b_i, p1 in enumerate(pos_chunk):
                        wt_aa = seq[p1 - 1]
                        wt_tok = alphabet.get_idx(wt_aa)
                        # Positive = WT preferred over Ala -> mutating WT -> Ala is destabilizing/deleterious!
                        llr_destab = float((log_probs[b_i, p1, wt_tok] - log_probs[b_i, p1, ala_idx]).item())
                        scan_rows.append([p1, wt_aa, round(llr_destab, 4), round(llr_destab, 4)])

            out_by_model[mkey].append(
                {
                    "target_id": tid,
                    "task": "stage3a_alanine_scan",
                    "text": json.dumps({"scan": scan_rows}),
                }
            )
        del model
        torch.cuda.empty_cache()

    weights_vol.commit()
    return out_by_model


@app.local_entrypoint()
def main(data_dir: str = "data/alanine_scan_benchmark", pred_dir: str = "outputs/predictions") -> None:
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
                "sequence": t_obj["sequence"],
                "positions_1idx": t_obj["positions_1idx"],
            }
        )

    print(f"Running ESM-2 (650M & 3B) zero-shot Alanine Scan masked marginals on {len(payload)} targets...")
    res = run_alanine_specialists_gpu.remote(payload)
    for mkey, rows in res.items():
        out_f = pdir / f"{mkey}_stage3a.jsonl"
        out_f.write_text("\n".join(json.dumps(x) for x in rows) + "\n", encoding="utf-8")
        print(f"Saved {len(rows)} predictions -> {out_f}")
