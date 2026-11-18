#!/usr/bin/env python3
"""Generate publication-grade Stage 1 SVG & PNG plots for the blog post & Twitter/X."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import matplotlib.pyplot as plt
import numpy as np


def plot_stage1_overview(report: Dict[str, Any], out_dir: Path) -> None:
    """Plot Panel A: Pre- vs Post-Cutoff Forward Folding TM-score & P@L; Panel B: Inverse Folding NSR."""
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.size": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )

    fwd = report.get("stage1a_forward_folding", [])
    inv = report.get("stage1b_inverse_folding", [])

    fig, axes = plt.subplots(1, 3, figsize=(15.5, 5.0), dpi=180)

    # Panel 1: Forward Folding TM-score (Pre-Cutoff vs Strict Late-2026 Post-Cutoff)
    ax0 = axes[0]
    labels = [m["model"].replace("specialist_", "").replace("_", " ") for m in fwd]
    pre_tm = [m["splits"].get("pre_cutoff", {}).get("tm_score", 0.0) for m in fwd]
    post_tm = [m["splits"].get("post_cutoff", {}).get("tm_score", 0.0) for m in fwd]
    x = np.arange(len(labels))
    w = 0.36
    ax0.bar(x - w / 2, pre_tm, width=w, label="Pre-2025 Controls (incl. 6MRR)", color="#3b82f6", alpha=0.9)
    ax0.bar(x + w / 2, post_tm, width=w, label="Sept-Nov 2026 Post-Cutoff (incl. 37NT)", color="#ef4444", alpha=0.9)
    ax0.axhline(0.50, linestyle="--", color="#10b981", linewidth=1.1, label="Correct Fold Threshold (TM = 0.50)")
    ax0.axhline(0.17, linestyle=":", color="#6b7280", linewidth=1.0, label="Random Chain Floor (TM ~ 0.17)")
    ax0.set_xticks(x)
    ax0.set_xticklabels(labels, rotation=28, ha="right")
    ax0.set_ylim(0.0, 1.0)
    ax0.set_ylabel("Mean TM-score (SO(3))")
    ax0.set_title("A. Forward Folding: 3D TM-score", fontweight="bold")
    ax0.legend(frameon=False, fontsize=8, loc="upper right")

    # Panel 2: Strict Long-Range Contact Precision P@L (|i - j| >= 24)
    ax1 = axes[1]
    p_lr24 = [100.0 * m["overall"].get("p_at_l_sep24", 0.0) for m in fwd]
    q3_acc = [100.0 * m["overall"].get("q3_accuracy", 0.0) for m in fwd]
    ax1.bar(x - w / 2, p_lr24, width=w, label="Long-Range Contact P@L (|i-j| >= 24)", color="#8b5cf6")
    ax1.bar(x + w / 2, q3_acc, width=w, label="Secondary Structure Q3 (%)", color="#f59e0b")
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, rotation=28, ha="right")
    ax1.set_ylim(0.0, 105.0)
    ax1.set_ylabel("Accuracy / Precision (%)")
    ax1.set_title("B. Long-Range Contacts vs. 1D Q3", fontweight="bold")
    ax1.legend(frameon=False, fontsize=8, loc="upper right")

    # Panel 3: Inverse Folding Native Sequence Recovery (Core vs Surface)
    ax2 = axes[2]
    if inv:
        ilabels = [m["model"].replace("specialist_", "").replace("_", " ") for m in inv]
        xi = np.arange(len(ilabels))
        core_nsr = [100.0 * m["overall"].get("nsr_core", 0.0) for m in inv]
        surf_nsr = [100.0 * m["overall"].get("nsr_surface", 0.0) for m in inv]
        ax2.bar(xi - w / 2, core_nsr, width=w, label="Buried Core NSR (%)", color="#0f766e")
        ax2.bar(xi + w / 2, surf_nsr, width=w, label="Exposed Surface NSR (%)", color="#64748b")
        ax2.axhline(5.0, linestyle=":", color="#6b7280", linewidth=1.0, label="Uniform 20-AA Chance (5%)")
        ax2.set_xticks(xi)
        ax2.set_xticklabels(ilabels, rotation=28, ha="right")
        ax2.set_ylim(0.0, max(65.0, max(core_nsr + [20.0]) + 10.0))
        ax2.set_ylabel("Native Sequence Recovery (%)")
        ax2.set_title("C. Inverse Folding: Core vs. Surface NSR", fontweight="bold")
        ax2.legend(frameon=False, fontsize=8, loc="upper right")

    fig.suptitle(
        "Stage 1 Benchmark: Forward Folding & Inverse Folding Across Pre- vs. Post-Cutoff PDBs",
        fontsize=13,
        fontweight="bold",
        y=1.02,
    )
    fig.tight_layout()
    for ext in ("svg", "png"):
        fig.savefig(out_dir / f"stage1_overview.{ext}", bbox_inches="tight")
    plt.close(fig)


def plot_anchor_contact_maps(data_dir: Path, out_dir: Path) -> None:
    """Plot side-by-side ground-truth contact maps for @sokrypton's 6MRR vs @WUFang's Oct-2026 37NT."""
    manifest = json.loads((data_dir / "manifest.json").read_text(encoding="utf-8"))
    anchors: List[Dict[str, Any]] = []
    for rec in manifest["targets"]:
        if rec["pdb_id"] == "6MRR":
            anchors.append(rec)
        elif rec["pdb_id"] == "37NT" and len(anchors) < 2:
            anchors.append(rec)
    for rec in manifest["targets"]:
        if rec["temporal_split"] == "post_cutoff" and rec["pdb_id"] != "37NT" and len(anchors) < 3:
            anchors.append(rec)

    if not anchors:
        return

    fig, axes = plt.subplots(1, len(anchors), figsize=(5.0 * len(anchors), 4.6), dpi=180)
    if len(anchors) == 1:
        axes = [axes]

    for ax, rec in zip(axes, anchors):
        tid = rec["target_id"]
        d = np.load(data_dir / "ground_truth" / f"{tid}.npz", allow_pickle=False)
        cb_mat = d["cb_dist_matrix"]
        L = cb_mat.shape[0]
        ii, jj = np.where((cb_mat < 8.0) & (np.abs(np.arange(L)[:, None] - np.arange(L)[None, :]) >= 6))
        dists = cb_mat[ii, jj]
        sc = ax.scatter(jj + 1, ii + 1, c=dists, cmap="viridis_r", s=14, vmin=3.5, vmax=8.0)
        ax.plot([1, L], [1, L], color="#9ca3af", linewidth=0.8)
        ax.set_xlim(1, L)
        ax.set_ylim(L, 1)
        ax.set_aspect("equal")
        ax.set_xlabel("Residue j")
        ax.set_ylabel("Residue i")
        ax.set_title(
            f"{tid} ({rec['pdb_id']}:{rec['chain_id']}, L={L})\n"
            f"[{rec['temporal_split']}, {rec['fold_class']}]",
            fontsize=9.5,
            fontweight="bold",
        )
        plt.colorbar(sc, ax=ax, fraction=0.046, pad=0.04, label="Cβ–Cβ dist (Å)")

    fig.tight_layout()
    for ext in ("svg", "png"):
        fig.savefig(out_dir / f"stage1_anchor_contact_maps.{ext}", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot Stage 1 benchmark figures.")
    parser.add_argument("--data-dir", default="data/stage1_benchmark")
    parser.add_argument("--report", default="outputs/results/stage1_evaluation_report.json")
    parser.add_argument("--out-dir", default="outputs/figures")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    report_path = Path(args.report)
    if report_path.exists():
        report = json.loads(report_path.read_text(encoding="utf-8"))
        plot_stage1_overview(report, out_dir)
        print(f"Saved overview plots -> {out_dir}/stage1_overview.[svg|png]")
    plot_anchor_contact_maps(Path(args.data_dir), out_dir)
    print(f"Saved anchor contact maps -> {out_dir}/stage1_anchor_contact_maps.[svg|png]")


if __name__ == "__main__":
    main()
