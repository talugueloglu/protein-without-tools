#!/usr/bin/env python3
"""Offline Task 5 SOTA Baseline: ESMFold-2 -> Foldseek 3Di -> SaProt (650M, AF2) zero-shot masked marginals.

Uses only the exact 1D sequence input (folded via ESMFold-2, zero native crystal structures,
zero MegaScale training leakage), applying official SaProt_650M_AF2 pLDDT<70 masking and
exact AA-masked/3Di-unmasked (#<3di>) probability marginalization over the 21 structural tokens per AA.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
from typing import Any, Dict, List, Tuple

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl_proteinbench")
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

import numpy as np
import torch

torch.set_num_threads(max(4, os.cpu_count() or 8))

REPO = Path(__file__).resolve().parents[1]
if str(REPO / "src") not in sys.path:
    sys.path.insert(0, str(REPO / "src"))
if str(REPO / "scripts") not in sys.path:
    sys.path.insert(0, str(REPO / "scripts"))

from evaluate_alanine_scan import load_alanine_targets, summarize_model
from protein_without_tools.alanine_scan import grade_alanine_scan, parse_alanine_scan_response
from protein_without_tools.japanfold_client import parse_pdb_chain_with_bfactor


AA_ALPHABET = "ACDEFGHIKLMNPQRSTVWY"


def extract_foldseek_3di(foldseek_bin: Path, pdb_path: Path) -> Tuple[str, str]:
    """Run Foldseek `structureto3didescriptor` on a single-chain PDB and return (aa_seq, 3di_seq)."""
    with tempfile.TemporaryDirectory() as td:
        out_tsv = Path(td) / "out.tsv"
        subprocess.run(
            [
                str(foldseek_bin),
                "structureto3didescriptor",
                "-v",
                "0",
                "--threads",
                "4",
                "--chain-name-mode",
                "1",
                str(pdb_path),
                str(out_tsv),
            ],
            check=True,
        )
        lines = [ln for ln in out_tsv.read_text(encoding="utf-8").splitlines() if ln.strip()]
        if not lines:
            raise RuntimeError(f"Foldseek produced no output for {pdb_path}")
        cols = lines[0].split("\t")
        return cols[1].strip(), cols[2].strip()


def build_sa_tokens(pdb_path: Path, foldseek_bin: Path, plddt_threshold: float = 70.0) -> Tuple[str, List[str]]:
    """Return (aa_seq, list_of_3di_chars_masked) using official SaProt pLDDT<70 masking."""
    pdb_txt = pdb_path.read_text(encoding="utf-8")
    ch = parse_pdb_chain_with_bfactor(pdb_txt, "A")
    aa_fs, t3di = extract_foldseek_3di(foldseek_bin, pdb_path)
    plddts = ch["plddt_per_res"]
    if len(aa_fs) != len(plddts):
        raise RuntimeError(f"Length mismatch in {pdb_path.name}: foldseek={len(aa_fs)} vs pdb={len(plddts)}")
    masked_3di = [
        (s3.lower() if float(plddts[i]) >= plddt_threshold else "#")
        for i, s3 in enumerate(t3di)
    ]
    return aa_fs, masked_3di


def main() -> None:
    from transformers import EsmForMaskedLM, EsmTokenizer

    saprot_dir = REPO / "outputs" / "proteinmpnn_cache" / "saprot_650m_af2"
    foldseek_bin = REPO / "outputs" / "proteinmpnn_cache" / "foldseek" / "bin" / "foldseek"
    pdb_dir = REPO / "outputs" / "japanfold_cache" / "t5_esmfold2_pdbs"
    s5_dir = REPO / "data" / "alanine_scan_benchmark"

    assert saprot_dir.exists(), f"Missing SaProt directory: {saprot_dir}"
    assert foldseek_bin.exists(), f"Missing Foldseek binary: {foldseek_bin}"

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"=== Loading local SaProt_650M_AF2 on {device} (threads={torch.get_num_threads()}) ===", flush=True)
    tokenizer = EsmTokenizer.from_pretrained(str(saprot_dir), local_files_only=True)
    model = EsmForMaskedLM.from_pretrained(str(saprot_dir), local_files_only=True)
    model = model.to(device).eval()

    # Pre-verify 21-token slices per canonical AA in vocab.txt
    aa_slices: Dict[str, List[int]] = {}
    vocab = tokenizer.get_vocab()
    for aa in AA_ALPHABET:
        toks = [idx for tok, idx in vocab.items() if len(tok) == 2 and tok[0] == aa]
        assert len(toks) == 21, f"Expected 21 tokens for AA {aa}, got {len(toks)}"
        aa_slices[aa] = sorted(toks)
    ala_ids = torch.tensor(aa_slices["A"], dtype=torch.long, device=device)

    _, t5_targets, _ = load_alanine_targets(s5_dir)
    t5_rows: List[Dict[str, Any]] = []
    t5_grades = []
    t0 = time.monotonic()

    print(f"=== [Task 5] Scoring {len(t5_targets)} targets with ESMFold-2 -> SaProt (650M) ===", flush=True)
    for idx_t, (tid, t_obj) in enumerate(sorted(t5_targets.items()), start=1):
        pdb_path = pdb_dir / f"{tid}.pdb"
        if not pdb_path.exists():
            raise FileNotFoundError(f"Missing ESMFold-2 PDB for {tid}: {pdb_path}")

        aa_fs, masked_3di = build_sa_tokens(pdb_path, foldseek_bin, plddt_threshold=70.0)
        seq = t_obj.sequence.strip().upper()
        if aa_fs != seq:
            raise RuntimeError(f"Sequence mismatch for {tid}: foldseek={len(aa_fs)} vs target={len(seq)}")

        sa_seq = "".join(f"{aa}{s3}" for aa, s3 in zip(seq, masked_3di))
        enc = tokenizer(sa_seq, return_tensors="pt")
        input_ids = enc["input_ids"].to(device)  # (1, L+2): [cls, tok1..tokL, eos]
        attn_mask = enc["attention_mask"].to(device)
        assert input_ids.shape[1] == len(seq) + 2, f"Token len mismatch for {tid}: {input_ids.shape[1]} vs {len(seq)+2}"

        positions = [p for p in t_obj.positions_1idx if 1 <= p <= len(seq)]
        scan_rows = []
        with torch.no_grad():
            tok_batch = input_ids.repeat(len(positions), 1)
            mask_batch = attn_mask.repeat(len(positions), 1)
            for b_i, p1 in enumerate(positions):
                # Official SaProt masking: mask AA to '#', keep 3Di token unmasked!
                mask_tok = "#" + masked_3di[p1 - 1]
                tok_batch[b_i, p1] = vocab[mask_tok]

            logits = model(input_ids=tok_batch, attention_mask=mask_batch).logits  # (B, L+2, V)
            log_probs = torch.log_softmax(logits, dim=-1)
            for b_i, p1 in enumerate(positions):
                wt_aa = seq[p1 - 1]
                wt_ids = torch.tensor(aa_slices[wt_aa], dtype=torch.long, device=device)
                lp_wt = torch.logsumexp(log_probs[b_i, p1, wt_ids], dim=0)
                lp_ala = torch.logsumexp(log_probs[b_i, p1, ala_ids], dim=0)
                # Positive = WT preferred over Ala -> mutating WT -> Ala is destabilising
                llr_destab = float((lp_wt - lp_ala).item())
                scan_rows.append([p1, wt_aa, round(llr_destab, 4), round(llr_destab, 4)])

        txt = json.dumps({"scan": scan_rows})
        t5_rows.append({"target_id": tid, "task": "stage3a_alanine_scan", "text": txt})
        t5_grades.append(grade_alanine_scan(t_obj, parse_alanine_scan_response(txt, t_obj)))
        if idx_t % 10 == 0 or idx_t == len(t5_targets):
            print(f"  [T5 SaProt-650M] {idx_t}/{len(t5_targets)} done ({time.monotonic()-t0:.1f}s)", flush=True)

    pred_out = REPO / "outputs" / "predictions" / "baseline_saprot_650m_stage3a.jsonl"
    pred_out.write_text("\n".join(json.dumps(r) for r in t5_rows) + "\n", encoding="utf-8")

    sum5 = summarize_model("baseline_saprot_650m", t5_grades)
    res_out = REPO / "outputs" / "results" / "baseline_t5_saprot_650m.json"
    res_out.write_text(json.dumps(sum5, indent=2), encoding="utf-8")

    print("\n=== TASK 5 SOTA: ESMFold-2 -> SaProt (650M) ===", flush=True)
    print("Overall:", sum5["overall"], flush=True)
    for cat, cv in sum5["by_category"].items():
        print(
            f"  {cat:<28s}: n={cv['n_targets']:2d} | NDCG@5={cv['ndcg_at_5']:.3f} | "
            f"SameAA={cv['same_aa_win_rate']:.3f} | Rho={cv['spearman_rho']:.3f}",
            flush=True,
        )


if __name__ == "__main__":
    main()
