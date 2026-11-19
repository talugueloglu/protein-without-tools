#!/usr/bin/env python3
"""Modal `L40S` Specialist Bio-Tool Baselines for Stage 2A (`PickyBinders/tea` + `ESM-2`)
and Stage 2B (`ESM-2` Embedding k-NN GO Transfer).

Stage 2A (`Remote Homology`):
  1. `PickyBinders/tea` (`Tea.from_pretrained("PickyBinders/tea")` + `facebook/esm2_t33_650M_UR50D`):
     Converts every SCOPe domain sequence into its 20-letter TEA sequence using the exact
     `PickyBinders/tea` pipeline and aligns each pair with `tea/matcha.out` affine Smith-Waterman!
  2. `ESM-2 (650M)` Cosine Similarity: Mean-pooled residue embedding cosine similarity.

Stage 2B (`CAFA Function Prediction`):
  1. `ESM-2 (650M)` Embedding k-NN (`k = 15` cosine-weighted transfer from the pre-2025
     Swiss-Prot experimental GO reference bank).

Usage:
  .venv/bin/modal run scripts/modal_stage2_baselines.py
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Dict, List

import modal

app = modal.App("fable-protein-bench-stage2")
weights_vol = modal.Volume.from_name("fable-bench-weights", create_if_missing=True)

stage2_image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git")
    .pip_install(
        "torch==2.4.1",
        "transformers>=4.45.0",
        "accelerate>=0.34.0",
        "huggingface_hub>=0.25.0",
        "biotite>=1.0.0",
        "numpy>=1.26.0",
        "scipy>=1.12.0",
        "tqdm>=4.66.0",
        "git+https://github.com/PickyBinders/tea.git",
    )
)


@app.function(
    image=stage2_image,
    gpu="L40S",
    timeout=1800,
    volumes={"/cache": weights_vol},
)
def run_stage2_specialists_gpu(
    unique_scope_seqs: Dict[str, str],
    cafa_query_seqs: Dict[str, str],
    cafa_ref_bank: List[Dict[str, Any]],
) -> Dict[str, Any]:
    import os
    import re
    import numpy as np
    import torch
    from transformers import AutoModel, AutoTokenizer
    from tea.model import Tea

    os.environ["HF_HOME"] = "/cache/hf"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = AutoTokenizer.from_pretrained("facebook/esm2_t33_650M_UR50D", cache_dir="/cache/hf")
    esm2 = AutoModel.from_pretrained(
        "facebook/esm2_t33_650M_UR50D",
        cache_dir="/cache/hf",
        add_pooling_layer=False,
    ).to(device).eval()
    tea_model = Tea.from_pretrained("PickyBinders/tea", cache_dir="/cache/hf").to(device).eval()

    def embed_and_tea(seq_dict: Dict[str, str], need_tea: bool) -> tuple[Dict[str, np.ndarray], Dict[str, str]]:
        emb_out: Dict[str, np.ndarray] = {}
        tea_out: Dict[str, str] = {}
        items = list(seq_dict.items())
        for i in range(0, len(items), 24):
            chunk = items[i : i + 24]
            spaced = [" ".join(list(re.sub(r"[UZOBJ]", "X", s))) for _, s in chunk]
            toks = tokenizer(spaced, add_special_tokens=True, padding="longest", return_tensors="pt").to(device)
            with torch.no_grad():
                hidden = esm2(input_ids=toks["input_ids"], attention_mask=toks["attention_mask"]).last_hidden_state
                if need_tea:
                    tea_seqs = tea_model.to_sequences(embeddings=hidden, input_ids=toks["input_ids"])
                else:
                    tea_seqs = [""] * len(chunk)
                for idx_b, (k, orig_s) in enumerate(chunk):
                    L = len(orig_s)
                    vec = hidden[idx_b, 1 : 1 + L].mean(dim=0).cpu().numpy().astype(np.float32)
                    norm = float(np.linalg.norm(vec)) + 1e-8
                    emb_out[k] = vec / norm
                    if need_tea:
                        tea_out[k] = tea_seqs[idx_b]
        return emb_out, tea_out

    scope_embs, scope_tea_seqs = embed_and_tea(unique_scope_seqs, need_tea=True)

    # Embed CAFA reference bank & CAFA query targets for ESM-2 k-NN transfer
    ref_seq_map = {r["accession"]: r["sequence"] for r in cafa_ref_bank}
    ref_embs, _ = embed_and_tea(ref_seq_map, need_tea=False)
    q_embs, _ = embed_and_tea(cafa_query_seqs, need_tea=False)

    ref_ids = [r["accession"] for r in cafa_ref_bank if r["accession"] in ref_embs]
    ref_go_by_id = {r["accession"]: r["go_terms"] for r in cafa_ref_bank}
    ref_mat = np.stack([ref_embs[rid] for rid in ref_ids], axis=0)  # (N_ref, D)

    cafa_knn_preds: Dict[str, List[List[Any]]] = {}
    for tid, qvec in q_embs.items():
        sims = ref_mat @ qvec  # cosine similarities in [-1, 1]
        top_idx = np.argsort(-sims)[:15]
        term_weights: Dict[str, float] = {}
        denom = 0.0
        for idx_r in top_idx:
            w = max(0.0, float(sims[idx_r]) ** 4)
            denom += w
            for gid in ref_go_by_id[ref_ids[idx_r]]:
                term_weights[gid] = term_weights.get(gid, 0.0) + w
        if denom > 0:
            scored = sorted(
                ((gid, round(min(1.0, wt / denom), 4)) for gid, wt in term_weights.items() if wt / denom >= 0.05),
                key=lambda t: t[1],
                reverse=True,
            )[:35]
            cafa_knn_preds[tid] = [[g, s, "ESM-2 kNN"] for g, s in scored]
        else:
            cafa_knn_preds[tid] = []

    weights_vol.commit()
    return {
        "scope_tea_seqs": scope_tea_seqs,
        "scope_embs": {k: v.tolist() for k, v in scope_embs.items()},
        "cafa_knn_preds": cafa_knn_preds,
    }


@app.local_entrypoint()
def main(
    stage2a_dir: str = "data/stage2a_homology",
    stage2b_dir: str = "data/stage2b_cafa",
    pred_dir: str = "outputs/predictions",
) -> None:
    import numpy as np
    from fable_bench.stage2a_homology import parse_substitution_matrix_file, smith_waterman_affine

    s2a = Path(stage2a_dir)
    s2b = Path(stage2b_dir)
    pdir = Path(pred_dir)
    pdir.mkdir(parents=True, exist_ok=True)

    m2a = json.loads((s2a / "manifest.json").read_text(encoding="utf-8"))
    m2b = json.loads((s2b / "manifest.json").read_text(encoding="utf-8"))
    ref_bank = json.loads((s2b / "reference_swissprot_go.json").read_text(encoding="utf-8"))
    matcha_txt = (s2a / "matcha.out").read_text(encoding="utf-8")
    t_alph, t_mat, t_lam = parse_substitution_matrix_file(matcha_txt)

    unique_scope: Dict[str, str] = {}
    for p in m2a["pairs"]:
        unique_scope[p["domain_a"]] = p["seq_a"]
        unique_scope[p["domain_b"]] = p["seq_b"]

    cafa_queries = {t["target_id"]: t["sequence"] for t in m2b["targets"]}

    print(f"Dispatching {len(unique_scope)} SCOPe domains & {len(cafa_queries)} CAFA targets to Modal L40S (TEA + ESM-2)...")
    gpu_out = run_stage2_specialists_gpu.remote(unique_scope, cafa_queries, ref_bank)

    tea_seqs = gpu_out["scope_tea_seqs"]
    scope_embs = {k: np.array(v, dtype=np.float32) for k, v in gpu_out["scope_embs"].items()}

    tea_rows = []
    esm2_cos_rows = []
    for p in m2a["pairs"]:
        pid = p["pair_id"]
        da, db = p["domain_a"], p["domain_b"]
        ta, tb = tea_seqs[da], tea_seqs[db]
        bits, _ev, _sid = smith_waterman_affine(
            ta, tb, alphabet=t_alph, sub_matrix=t_mat, gap_open=9.0, gap_extend=1.5, karlin_lambda=t_lam
        )
        # Smooth logistic map from TEA bitscore -> [0, 1] probability (AUROC is rank-invariant!)
        prob_tea = 1.0 / (1.0 + math.exp(-(bits - 28.0) / 6.0))
        tea_rows.append(
            {
                "pair_id": pid,
                "task": "stage2a_homology",
                "tea_bitscore": bits,
                "text": json.dumps({"same_superfamily": prob_tea >= 0.5, "probability": round(prob_tea, 4)}),
            }
        )

        cos_sim = float(np.dot(scope_embs[da], scope_embs[db]))
        prob_cos = float(np.clip((cos_sim + 1.0) / 2.0, 0.0, 1.0))
        esm2_cos_rows.append(
            {
                "pair_id": pid,
                "task": "stage2a_homology",
                "text": json.dumps({"same_superfamily": cos_sim >= 0.85, "probability": round(prob_cos, 6)}),
            }
        )

    (pdir / "specialist_pickybinders_tea_stage2a.jsonl").write_text(
        "\n".join(json.dumps(x) for x in tea_rows) + "\n", encoding="utf-8"
    )
    (pdir / "specialist_esm2_cosine_stage2a.jsonl").write_text(
        "\n".join(json.dumps(x) for x in esm2_cos_rows) + "\n", encoding="utf-8"
    )

    cafa_rows = []
    for tid, preds in gpu_out["cafa_knn_preds"].items():
        cafa_rows.append(
            {
                "target_id": tid,
                "task": "stage2b_cafa",
                "text": json.dumps({"predictions": preds}),
            }
        )
    (pdir / "specialist_esm2_knn_stage2b.jsonl").write_text(
        "\n".join(json.dumps(x) for x in cafa_rows) + "\n", encoding="utf-8"
    )
    print("Saved PickyBinders/tea (Stage 2A) and ESM-2 (Stage 2A & 2B) specialist predictions!")
