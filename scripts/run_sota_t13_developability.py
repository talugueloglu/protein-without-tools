#!/usr/bin/env python3
"""Run uncontaminated local specialist baselines for Task 13 (Wet-Lab Developability, N=50 panels):

  1. Zero-shot `ESM-2 (650M)` Sequence Log-Likelihood (`esm2_t33_650M_UR50D`, cached locally)
     averaged over VH + VL chains (higher log-likelihood = more natural / thermostable /
     high-expression antibody).
  2. Zero-shot Biophysical Antibody Developability Descriptor (TAP-inspired, pure sequence
     biophysics): net Fv charge at pH 7.4, positive-charge patch count, max 7-mer
     Kyte-Doolittle hydrophobic patch, and CDR-H3 aromatic/basic density.
  3. Zero-shot Physics-Aware Hybrid (z-scored ESM-2 + Biophysical TAP; no supervised training).
"""

from __future__ import annotations

from collections import defaultdict
import json
import os
from pathlib import Path
import sys
from typing import Any, Dict, List

import numpy as np
import torch
import torch.nn.functional as F

torch.set_num_threads(max(4, os.cpu_count() or 8))

REPO = Path(__file__).resolve().parents[1]
if str(REPO / "src") not in sys.path:
    sys.path.insert(0, str(REPO / "src"))
if str(REPO / "outputs" / "proteinmpnn_cache" / "fair_esm_src") not in sys.path:
    sys.path.insert(0, str(REPO / "outputs" / "proteinmpnn_cache" / "fair_esm_src"))

from protein_without_tools.task13_developability import (
    DevelopabilityCandidate,
    grade_developability_panel,
)


def load_local_esm2_650m(device: torch.device):
    import esm.pretrained

    cache_dir = REPO / "outputs" / "proteinmpnn_cache"
    ckpt_p = cache_dir / "esm2_t33_650M_UR50D.pt"
    reg_p = cache_dir / "esm2_t33_650M_UR50D-contact-regression.pt"
    model_data = torch.load(str(ckpt_p), map_location="cpu", weights_only=False)
    reg_data = torch.load(str(reg_p), map_location="cpu", weights_only=False) if reg_p.exists() else None
    model, alphabet = esm.pretrained.load_model_and_alphabet_core("esm2_t33_650M_UR50D", model_data, reg_data)
    model = model.to(device).eval()
    return model, alphabet


def score_chains_esm2_local(unique_chains: List[str]) -> Dict[str, float]:
    """Compute wild-type sequence log-likelihood (mean log P(aa_i | seq)) using local ESM-2 650M."""
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Loading local ESM-2 (650M) on device={device}...", flush=True)
    model, alphabet = load_local_esm2_650m(device)
    batch_converter = alphabet.get_batch_converter()

    out: Dict[str, float] = {}
    batch_size = 16
    for i0 in range(0, len(unique_chains), batch_size):
        chunk = unique_chains[i0 : i0 + batch_size]
        data = [(f"c{i0 + j}", s) for j, s in enumerate(chunk)]
        _labels, _strs, tokens = batch_converter(data)
        tokens = tokens.to(device)
        with torch.no_grad():
            res = model(tokens, repr_layers=[], return_contacts=False)
            logits = res["logits"].float()  # (B, L_max+2, V)
            lprobs = F.log_softmax(logits, dim=-1)
            for j, seq in enumerate(chunk):
                L = len(seq)
                tgt = tokens[j, 1 : L + 1]
                pos_lp = lprobs[j, 1 : L + 1, :]
                chosen = pos_lp[torch.arange(L, device=device), tgt]
                out[seq] = round(float(chosen.mean().item()), 6)
        if (i0 + len(chunk)) % 128 == 0 or (i0 + len(chunk)) == len(unique_chains):
            print(f"  [ESM-2 650M] scored {i0 + len(chunk)}/{len(unique_chains)} unique chains", flush=True)
    return out


# Kyte-Doolittle hydropathy scale
KD_SCALE: Dict[str, float] = {
    "I": 4.5, "V": 4.2, "L": 3.8, "F": 2.8, "C": 2.5, "M": 1.9, "A": 1.8,
    "G": -0.4, "T": -0.7, "S": -0.8, "W": -0.9, "Y": -1.3, "P": -1.6,
    "H": -3.2, "E": -3.5, "Q": -3.5, "D": -3.5, "N": -3.5, "K": -3.9, "R": -4.5,
}


def biophysical_tap_score(vh: str, vl: str, property_type: str) -> float:
    """Zero-shot textbook biophysical developability score (oriented: higher = more developable)."""
    fv = vh + vl
    cdr_h3 = vh[max(0, len(vh) - 24) : max(1, len(vh) - 9)]

    def net_charge(s: str) -> float:
        return s.count("R") + s.count("K") + 0.1 * s.count("H") - s.count("D") - s.count("E")

    def max_window_hydro(s: str, w: int = 7) -> float:
        if len(s) < w:
            return float(np.mean([KD_SCALE.get(a, 0.0) for a in s]))
        vals = [KD_SCALE.get(a, 0.0) for a in s]
        return float(max(np.mean(vals[i : i + w]) for i in range(len(vals) - w + 1)))

    def mean_hydro(s: str) -> float:
        return float(np.mean([KD_SCALE.get(a, 0.0) for a in s]))

    def pos_patch_max(s: str, w: int = 7) -> float:
        vals = [1.0 if a in ("R", "K") else (0.3 if a == "H" else 0.0) for a in s]
        if len(vals) < w:
            return float(sum(vals))
        return float(max(sum(vals[i : i + w]) for i in range(len(vals) - w + 1)))

    def arom_frac(s: str) -> float:
        return sum(1 for a in s if a in ("W", "Y", "F")) / float(max(1, len(s)))

    q_fv = net_charge(vh) + net_charge(vl)
    max_hp = max(max_window_hydro(vh, 7), max_window_hydro(vl, 7))
    pos_patch = max(pos_patch_max(vh, 7), pos_patch_max(vl, 7))
    h3_arom = arom_frac(cdr_h3)
    h3_pos = (cdr_h3.count("R") + cdr_h3.count("K")) / float(max(1, len(cdr_h3)))

    if property_type == "polyreactivity":
        return -(0.45 * q_fv + 1.2 * pos_patch + 6.0 * h3_pos + 3.0 * h3_arom)
    if property_type == "aggregation":
        return -(1.5 * max_hp + 12.0 * arom_frac(fv) + 0.25 * max(0.0, q_fv) + 2.0 * mean_hydro(fv))
    if property_type == "expression":
        return -(1.2 * max_hp + 0.20 * abs(q_fv - 2.0) + 5.0 * h3_arom)
    return -(0.8 * max_hp + 0.15 * abs(q_fv) + 0.05 * len(cdr_h3))


def main() -> None:
    manifest = json.loads((REPO / "data/task13_developability/manifest.json").read_text(encoding="utf-8"))
    targets = manifest["targets"]

    unique_chains_set: set[str] = set()
    for t in targets:
        for c in t["candidates"]:
            vh, vl = [x.strip() for x in c["sequence"].split("/")]
            unique_chains_set.add(vh)
            unique_chains_set.add(vl)
    unique_chains = sorted(unique_chains_set)
    print(f"Scoring {len(unique_chains)} unique VH/VL antibody chains with local ESM-2 650M...", flush=True)

    cache_p = REPO / "data/raw_v2_cache/task13_flab/esm2_650m_ll_cache.json"
    if cache_p.exists():
        chain_ll = json.loads(cache_p.read_text(encoding="utf-8"))
        missing = [s for s in unique_chains if s not in chain_ll]
        if missing:
            chain_ll.update(score_chains_esm2_local(missing))
            cache_p.write_text(json.dumps(chain_ll, indent=2), encoding="utf-8")
    else:
        chain_ll = score_chains_esm2_local(unique_chains)
        cache_p.parent.mkdir(parents=True, exist_ok=True)
        cache_p.write_text(json.dumps(chain_ll, indent=2), encoding="utf-8")

    def _eval(mode: str) -> Dict[str, Any]:
        rows: List[Dict[str, Any]] = []
        by_prop: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        for t in targets:
            prop = t["property_type"]
            panel = [
                DevelopabilityCandidate(
                    cand_id=c["cand_id"],
                    sequence=c["sequence"],
                    value=float(c["value"]),
                    desirability=float(c["desirability"]),
                )
                for c in t["candidates"]
            ]
            esm_raw: Dict[str, float] = {}
            bio_raw: Dict[str, float] = {}
            for c in panel:
                vh, vl = [x.strip() for x in c.sequence.split("/")]
                esm_raw[c.cand_id] = 0.5 * (chain_ll[vh] + chain_ll[vl])
                bio_raw[c.cand_id] = biophysical_tap_score(vh, vl, prop)

            if mode == "esm2_650m_ll":
                scores = esm_raw
            elif mode == "biophysical_tap":
                scores = bio_raw
            else:
                ev = np.array([esm_raw[c.cand_id] for c in panel], dtype=float)
                bv = np.array([bio_raw[c.cand_id] for c in panel], dtype=float)
                ze = (ev - np.mean(ev)) / (np.std(ev) + 1e-8)
                zb = (bv - np.mean(bv)) / (np.std(bv) + 1e-8)
                w_e = 0.7 if prop in ("thermostability", "expression") else 0.3
                scores = {c.cand_id: float(w_e * ze[i] + (1.0 - w_e) * zb[i]) for i, c in enumerate(panel)}

            g = grade_developability_panel(panel, [], predicted_scores=scores)
            rec = {
                "target_id": t["target_id"],
                "property_type": prop,
                "source_assay": t["source_assay"],
                "spearman_rho": round(g["spearman_rho"], 4),
                "pairwise_accuracy": round(g["pairwise_accuracy"], 4),
                "ndcg_at_5": round(g["ndcg_at_5"], 4),
            }
            rows.append(rec)
            by_prop[prop].append(rec)

        def _agg(rs: List[Dict[str, Any]]) -> Dict[str, Any]:
            if not rs:
                return {"n": 0}
            return {
                "n": len(rs),
                "spearman_rho": round(float(np.mean([r["spearman_rho"] for r in rs])), 4),
                "pairwise_accuracy": round(float(np.mean([r["pairwise_accuracy"] for r in rs])), 4),
                "ndcg_at_5": round(float(np.mean([r["ndcg_at_5"] for r in rs])), 4),
            }

        return {
            "overall": _agg(rows),
            "by_property": {k: _agg(v) for k, v in sorted(by_prop.items())},
            "per_target": rows,
        }

    esm_res = _eval("esm2_650m_ll")
    bio_res = _eval("biophysical_tap")
    hyb_res = _eval("hybrid_esm2_tap")

    out = {
        "task": "task13_wet_lab_developability",
        "specialist_model": "Zero-Shot ESM-2 (650M) Log-Likelihood + Biophysical TAP-Style Descriptors",
        "n_evaluated": manifest["n_targets"],
        "n_total": manifest["n_targets"],
        "overall": hyb_res["overall"],
        "by_property": hyb_res["by_property"],
        "baselines": {
            "esm2_650m_ll": {"overall": esm_res["overall"], "by_property": esm_res["by_property"]},
            "biophysical_tap": {"overall": bio_res["overall"], "by_property": bio_res["by_property"]},
            "hybrid_esm2_tap": {"overall": hyb_res["overall"], "by_property": hyb_res["by_property"]},
        },
        "anti_cheat_null": manifest["anti_cheat_audit"],
        "per_target": hyb_res["per_target"],
    }
    out_path = REPO / "outputs/results/baseline_t13_developability.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print("\n=== TASK 13 DEVELOPABILITY BASELINES COMPLETE ===", flush=True)
    print("  ESM-2 650M LL: ", esm_res["overall"], "|", {k: v["spearman_rho"] for k, v in esm_res["by_property"].items()})
    print("  Biophys TAP:   ", bio_res["overall"], "|", {k: v["spearman_rho"] for k, v in bio_res["by_property"].items()})
    print("  Hybrid ZeroShot:", hyb_res["overall"], "|", {k: v["spearman_rho"] for k, v in hyb_res["by_property"].items()})


if __name__ == "__main__":
    main()
