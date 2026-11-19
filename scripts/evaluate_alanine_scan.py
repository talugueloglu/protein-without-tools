#!/usr/bin/env python3
"""Evaluate Stage 3A In-Silico Alanine Scan (`X_i -> A`) predictions & calibration baselines.

Computes:
  - Raw Spearman ρ & Pearson r vs. experimental `X -> A` ΔΔG_fold (kcal/mol) / DMS loss
  - **AA-Demeaned Context Spearman ρ (`rho_aa_demeaned`)**: isolates true 3D context awareness
    (e.g., buried core Leu vs. solvent-exposed surface Leu) from static 20-AA hydrophobicity!
  - Top-20% Destabilizing Hotspot AUROC
  - 3D Hydrophobic Core Burial AUROC (predicting 3D core burial purely from 1D alanine scan!)
  - Function-vs-Stability Disentanglement AUROC on dual-assay targets (active/binding sites where
    `X -> A` leaves folding intact `ΔΔG ≈ 0` yet abolishes binding/catalytic activity!)
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
from typing import Any, Dict, List

import matplotlib.pyplot as plt
import numpy as np

from fable_bench.alanine_scan import (
    AA_TO_ALA_LOOKUP_DDG,
    AlanineScanGrade,
    AlanineScanParsed,
    AlanineScanTarget,
    grade_alanine_scan,
    parse_alanine_scan_response,
)

KD_HYDROPATHY: Dict[str, float] = {
    "I": 4.5, "V": 4.2, "L": 3.8, "F": 2.8, "C": 2.5, "M": 1.9, "A": 1.8,
    "G": -0.4, "T": -0.7, "S": -0.8, "W": -0.9, "Y": -1.3, "P": -1.6, "H": -3.2,
    "E": -3.5, "Q": -3.5, "D": -3.5, "N": -3.5, "K": -3.9, "R": -4.5,
}


def load_alanine_targets(data_dir: Path) -> tuple[Dict[str, Any], Dict[str, AlanineScanTarget], Dict[str, Dict[str, Any]]]:
    manifest = json.loads((data_dir / "manifest.json").read_text(encoding="utf-8"))
    targets: Dict[str, AlanineScanTarget] = {}
    meta_by_id: Dict[str, Dict[str, Any]] = {}
    for rec in manifest["targets"]:
        tid = rec["target_id"]
        meta_by_id[tid] = rec
        t_json = json.loads((data_dir / "targets" / f"{tid}.json").read_text(encoding="utf-8"))
        targets[tid] = AlanineScanTarget(
            target_id=tid,
            category=t_json["category"],
            assay_type=t_json["assay_type"],
            sequence=t_json["sequence"],
            length=t_json["length"],
            positions_1idx=t_json["positions_1idx"],
            wt_aas=t_json["wt_aas"],
            exp_destab_scores=t_json["exp_destab_scores"],
            exp_func_loss_scores=t_json.get("exp_func_loss_scores"),
            is_core_flags=t_json.get("is_core_flags"),
        )
    return manifest, targets, meta_by_id


def make_calibration_pred(target: AlanineScanTarget, mode: str, rng: np.random.Generator) -> AlanineScanParsed:
    n = len(target.positions_1idx)
    if mode == "Oracle_Experimental":
        ddg = np.asarray(target.exp_destab_scores, dtype=np.float64)
        fn = (
            np.asarray(target.exp_func_loss_scores, dtype=np.float64)
            if target.exp_func_loss_scores is not None
            else ddg.copy()
        )
        return AlanineScanParsed(ddg, fn, n, 1.0, True)

    if mode == "Baseline_AA_LookupOnly":
        ddg = np.array([AA_TO_ALA_LOOKUP_DDG.get(aa, 0.5) for aa in target.wt_aas], dtype=np.float64)
        return AlanineScanParsed(ddg, ddg / 3.0, n, 1.0, True)

    if mode == "Baseline_LocalWindowKD":
        # Combines AA lookup with 9-residue local hydropathy + amphipathic helix moment (i±3, i±4)
        seq = target.sequence
        L = len(seq)
        ddg = np.zeros(n, dtype=np.float64)
        for idx, (pos1, aa) in enumerate(zip(target.positions_1idx, target.wt_aas)):
            i0 = pos1 - 1
            base = AA_TO_ALA_LOOKUP_DDG.get(aa, 0.5)
            neigh = [KD_HYDROPATHY.get(seq[j], 0.0) for j in range(max(0, i0 - 4), min(L, i0 + 5)) if j != i0]
            helix_face = [KD_HYDROPATHY.get(seq[j], 0.0) for j in (i0 - 4, i0 - 3, i0 + 3, i0 + 4) if 0 <= j < L]
            ddg[idx] = base + 0.12 * float(np.mean(neigh or [0.0])) + 0.15 * float(np.mean(helix_face or [0.0]))
        return AlanineScanParsed(ddg, ddg / 3.0, n, 1.0, True)

    # Null_Random
    r_ddg = rng.normal(0.8, 1.0, size=n)
    r_fn = rng.uniform(0.0, 1.0, size=n)
    return AlanineScanParsed(r_ddg, r_fn, n, 1.0, True)


def summarize_model(model_name: str, grades: List[AlanineScanGrade]) -> Dict[str, Any]:
    def _agg(sub: List[AlanineScanGrade]) -> Dict[str, Any]:
        if not sub:
            return {}
        core_vals = [g.core_burial_auroc for g in sub if g.core_burial_auroc is not None]
        func_vals = [g.func_only_hotspot_auroc for g in sub if g.func_only_hotspot_auroc is not None]
        out: Dict[str, Any] = {
            "n_targets": len(sub),
            "n_mutations_total": int(sum(g.n_assayed for g in sub)),
            "spearman_rho": round(float(np.mean([g.spearman_rho for g in sub])), 4),
            "pearson_r": round(float(np.mean([g.pearson_r for g in sub])), 4),
            "spearman_rho_aa_demeaned": round(float(np.mean([g.spearman_rho_aa_demeaned for g in sub])), 4),
            "hotspot_auroc_top20pct": round(float(np.mean([g.hotspot_auroc_top20pct for g in sub])), 4),
        }
        if core_vals:
            out["core_burial_auroc"] = round(float(np.mean(core_vals)), 4)
        if func_vals:
            out["func_only_hotspot_auroc"] = round(float(np.mean(func_vals)), 4)
        return out

    by_cat: Dict[str, List[AlanineScanGrade]] = defaultdict(list)
    for g in grades:
        by_cat[g.category].append(g)

    return {
        "model": model_name,
        "overall": _agg(grades),
        "by_category": {k: _agg(v) for k, v in sorted(by_cat.items())},
        "per_target": [g.to_dict() for g in grades],
    }


def plot_alanine_overview(summaries: List[Dict[str, Any]], out_png: Path) -> None:
    plt.rcParams.update({"font.family": "sans-serif", "font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.8), dpi=180, facecolor="#ffffff")

    labels = [s["model"].replace("Baseline_", "").replace("specialist_", "") for s in summaries]
    raw_rho = [s["overall"].get("spearman_rho", 0.0) for s in summaries]
    ctx_rho = [s["overall"].get("spearman_rho_aa_demeaned", 0.0) for s in summaries]
    core_auc = [s["overall"].get("core_burial_auroc", 0.5) for s in summaries]

    x = np.arange(len(labels))
    w = 0.36
    axes[0].bar(x - w / 2, raw_rho, width=w, color="#0ea5e9", label="Raw Spearman ρ")
    axes[0].bar(x + w / 2, ctx_rho, width=w, color="#ec4899", label="AA-Demeaned Context ρ")
    axes[0].axhline(0.0, color="#64748b", linewidth=0.8)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(labels, rotation=18, ha="right", fontsize=8.5)
    axes[0].set_ylim(-0.1, 1.08)
    axes[0].set_ylabel("Mean Spearman Correlation")
    axes[0].set_title("A. Overall vs. Context-Only (AA-Demeaned) ρ", fontweight="bold", loc="left")
    axes[0].legend(frameon=False, fontsize=8.5)

    # Panel B: De Novo Zero-Homolog ΔΔG vs Natural Stability/DMS
    denovo_ctx = [s["by_category"].get("de_novo_ddg", {}).get("spearman_rho_aa_demeaned", 0.0) for s in summaries]
    nat_ctx = [
        s["by_category"].get("natural_ddg", s["by_category"].get("dual_stability_function", {})).get("spearman_rho_aa_demeaned", 0.0)
        for s in summaries
    ]
    axes[1].bar(x - w / 2, denovo_ctx, width=w, color="#8b5cf6", label="Synthetic De Novo (0 MSA)")
    axes[1].bar(x + w / 2, nat_ctx, width=w, color="#10b981", label="Natural Domains")
    axes[1].axhline(0.0, color="#64748b", linewidth=0.8)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels, rotation=18, ha="right", fontsize=8.5)
    axes[1].set_ylim(-0.1, 1.08)
    axes[1].set_ylabel("AA-Demeaned Spearman ρ")
    axes[1].set_title("B. Zero-Homolog De Novo vs. Natural Domains", fontweight="bold", loc="left")
    axes[1].legend(frameon=False, fontsize=8.5)

    # Panel C: 3D Core Burial AUROC from 1D Alanine Scan
    bars = axes[2].bar(x, core_auc, width=0.52, color="#f59e0b")
    for b, v in zip(bars, core_auc):
        axes[2].text(b.get_x() + b.get_width() / 2, v + 0.015, f"{v:.2f}", ha="center", va="bottom", fontsize=8.5, fontweight="bold")
    axes[2].axhline(0.50, color="#0f172a", linestyle="--", linewidth=1.0, alpha=0.55, label="Random Chance (0.50)")
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(labels, rotation=18, ha="right", fontsize=8.5)
    axes[2].set_ylim(0.35, 1.08)
    axes[2].set_ylabel("3D Core Burial AUROC")
    axes[2].set_title("C. 3D Core Recovery from 1D X→A Scan", fontweight="bold", loc="left")
    axes[2].legend(frameon=False, fontsize=8.5)

    fig.suptitle("Stage 3A — In-Silico Alanine Scan (`X → A`) Benchmark & Calibration", fontsize=13, fontweight="bold", y=1.02)
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, bbox_inches="tight", dpi=180)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data/alanine_scan_benchmark")
    parser.add_argument("--pred-dir", default="outputs/predictions")
    parser.add_argument("--out-dir", default="outputs/results")
    parser.add_argument("--run-calibration", action="store_true")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    pred_dir = Path(args.pred_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    _manifest, targets, _meta = load_alanine_targets(data_dir)
    summaries: List[Dict[str, Any]] = []

    if args.run_calibration:
        rng = np.random.default_rng(20261119)
        for mode in ("Oracle_Experimental", "Baseline_LocalWindowKD", "Baseline_AA_LookupOnly", "Null_Random"):
            grades = [grade_alanine_scan(t, make_calibration_pred(t, mode, rng)) for t in targets.values()]
            summaries.append(summarize_model(mode, grades))

    if pred_dir.exists():
        for jf in sorted(pred_dir.glob("*_stage3a.jsonl")):
            mname = jf.name.replace("_stage3a.jsonl", "")
            rows = [json.loads(l) for l in jf.read_text(encoding="utf-8").splitlines() if l.strip()]
            grades = []
            for r in rows:
                tid = r["target_id"]
                if tid in targets:
                    parsed = parse_alanine_scan_response(r["text"], targets[tid])
                    grades.append(grade_alanine_scan(targets[tid], parsed))
            if grades:
                summaries.append(summarize_model(mname, grades))

    out_report = out_dir / "stage3a_alanine_report.json"
    out_report.write_text(json.dumps({"stage3a_alanine_scan": summaries}, indent=2), encoding="utf-8")
    plot_alanine_overview(summaries, data_dir.parent.parent / "outputs/figures/stage3a_alanine_overview.png")

    print("=" * 98)
    print("STAGE 3A — IN-SILICO ALANINE SCAN (`X -> A`) SUMMARY")
    print("=" * 98)
    print(f"{'Model / Baseline':<26s} | {'Raw ρ':>7s} | {'AA-Ctx ρ':>8s} | {'Hotspot AUC':>11s} | {'3D Core AUC':>11s} | {'FuncOnly AUC':>12s}")
    print("-" * 98)
    for s in summaries:
        ov = s["overall"]
        print(
            f"{s['model']:<26s} | {ov.get('spearman_rho', 0.0):7.3f} | "
            f"{ov.get('spearman_rho_aa_demeaned', 0.0):8.3f} | "
            f"{ov.get('hotspot_auroc_top20pct', 0.0):11.3f} | "
            f"{ov.get('core_burial_auroc', 0.0):11.3f} | "
            f"{ov.get('func_only_hotspot_auroc', 0.0):12.3f}"
        )


if __name__ == "__main__":
    main()
