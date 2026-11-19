#!/usr/bin/env python3
"""Plot True 3D TM-score (mode='tmalign') vs Sequence Identity across Pre-Cutoff PDB & AlphaFold DB."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def main() -> None:
    repo = Path("/Users/taluguloglu/.gemini/antigravity/scratch/fable-protein-bench")
    manifest = json.loads((repo / "data/stage1_benchmark/manifest.json").read_text(encoding="utf-8"))
    targets = manifest["targets"]

    out_art = Path("/Users/taluguloglu/.gemini/antigravity/brain/4775d422-bdcb-488c-afa6-e25f2cd600e6/stage1_true_tm_landscape.png")
    out_repo = repo / "outputs/figures/stage1_true_tm_landscape.png"

    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.size": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )

    fig, axes = plt.subplots(1, 3, figsize=(16.0, 5.2), dpi=180, facecolor="#ffffff")

    # Classify cleanly using TRUE combined SeqID so leaked re-crystallizations are visibly separated!
    def true_group(t: dict) -> str:
        if t["temporal_split"] == "pre_cutoff":
            return "pre_cutoff"
        ps = float(t.get("pdb_precutoff_seqid_combined", t.get("pdb_precutoff_seqid_max", 0.0)))
        afs = max(float(t.get("afdb_max_seqid", 0.0)), float(t.get("afdb_tm_seqid", 0.0)))
        if ps >= 30.0:
            return "post_homolog_or_recryst"
        if t["stratum"] == "post_cutoff_de_novo" or afs < 30.0:
            return "post_de_novo"
        return "post_pdb_novel_afdb_known"

    groups = [
        ("pre_cutoff", "Pre-Cutoff Controls", "#3b82f6", "o"),
        ("post_homolog_or_recryst", "2026 PDB Homologs / Re-cryst (>=30% PDB SeqID)", "#f59e0b", "^"),
        ("post_pdb_novel_afdb_known", "2026 Natural (<30% PDB SeqID, in AFDB)", "#10b981", "s"),
        ("post_de_novo", "2026 Synthetic De Novo (<30% PDB & AFDB SeqID)", "#ec4899", "D"),
    ]

    # Panel A: Pre-Cutoff PDB SeqID vs True Pre-Cutoff PDB TM-score (Query-Normalized)
    ax0 = axes[0]
    for gkey, glbl, col, mk in groups:
        sub = [t for t in targets if true_group(t) == gkey]
        xs = [float(t.get("pdb_precutoff_seqid_combined", 0.0)) for t in sub]
        ys = [float(t.get("pdb_precutoff_tm_query", 0.0)) for t in sub]
        ax0.scatter(xs, ys, c=col, marker=mk, s=52, alpha=0.86, edgecolors="white", linewidths=0.6, label=f"{glbl} (n={len(sub)})")
    ax0.axhline(0.50, color="#0f172a", linestyle="--", linewidth=1.0, alpha=0.55)
    ax0.axvline(30.0, color="#64748b", linestyle=":", linewidth=1.0, alpha=0.6)
    ax0.set_xlim(-3, 104)
    ax0.set_ylim(0.25, 1.04)
    ax0.set_xlabel("Max Sequence Identity to Pre-Cutoff PDB (%)")
    ax0.set_ylabel("Closest Pre-Cutoff PDB TM-score (Query-Norm)")
    ax0.set_title("A. Pre-Cutoff Experimental PDB (< May 2026)", fontweight="bold", loc="left")

    # Panel B: AlphaFold DB SeqID vs True AlphaFold DB TM-score (Query-Normalized)
    ax1 = axes[1]
    for gkey, glbl, col, mk in groups:
        sub = [t for t in targets if true_group(t) == gkey]
        xs = [max(float(t.get("afdb_max_seqid", 0.0)), float(t.get("afdb_tm_seqid", 0.0))) for t in sub]
        ys = [float(t.get("afdb_tm_query", 0.0)) for t in sub]
        ax1.scatter(xs, ys, c=col, marker=mk, s=52, alpha=0.86, edgecolors="white", linewidths=0.6, label=glbl)
    ax1.axhline(0.50, color="#0f172a", linestyle="--", linewidth=1.0, alpha=0.55)
    ax1.axvline(30.0, color="#64748b", linestyle=":", linewidth=1.0, alpha=0.6)
    ax1.set_xlim(-3, 104)
    ax1.set_ylim(0.25, 1.04)
    ax1.set_xlabel("Max Sequence Identity to AlphaFold DB (%)")
    ax1.set_ylabel("Closest AlphaFold DB TM-score (Query-Norm)")
    ax1.set_title("B. AlphaFold DB Predicted Structures (AFDB)", fontweight="bold", loc="left")

    # Panel C: Closest Pre-Cutoff PDB TM-score vs Closest AFDB TM-score (Query-Normalized TM)
    ax2 = axes[2]
    for gkey, glbl, col, mk in groups:
        sub = [t for t in targets if true_group(t) == gkey]
        xs = [float(t.get("pdb_precutoff_tm_query", 0.0)) for t in sub]
        ys = [float(t.get("afdb_tm_query", 0.0)) for t in sub]
        ax2.scatter(xs, ys, c=col, marker=mk, s=52, alpha=0.86, edgecolors="white", linewidths=0.6, label=f"{glbl} ({len(sub)})")
    for t in targets:
        if t["pdb_id"] in ("9YOY", "9YOX", "11CM", "11CP", "9ZR2", "47WY", "6MRR"):
            x, y = float(t.get("pdb_precutoff_tm_query", 0.0)), float(t.get("afdb_tm_query", 0.0))
            if x > 0:
                ax2.annotate(t["pdb_id"], (x, y), xytext=(4, 2), textcoords="offset points", fontsize=8, fontweight="bold")
    ax2.axhline(0.50, color="#0f172a", linestyle="--", linewidth=1.0, alpha=0.55)
    ax2.axvline(0.50, color="#0f172a", linestyle="--", linewidth=1.0, alpha=0.55)
    ax2.fill_between([0, 0.50], 0, 0.50, color="#ec4899", alpha=0.09)
    ax2.set_xlim(0.22, 1.04)
    ax2.set_ylim(0.22, 1.04)
    ax2.set_xlabel("Closest Pre-Cutoff PDB TM-score")
    ax2.set_ylabel("Closest AlphaFold DB TM-score")
    ax2.set_title("C. Pure 3D Structural Similarity: PDB TM vs. AFDB TM", fontweight="bold", loc="left")
    ax2.legend(frameon=True, facecolor="#f8fafc", edgecolor="#e2e8f0", fontsize=7.6, loc="lower right")

    fig.suptitle("Stage 1 True 3D Structural Similarity Audit (Foldseek TM-align Mode, N = 81)", fontsize=13, fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(out_art, bbox_inches="tight", dpi=180)
    fig.savefig(out_repo, bbox_inches="tight", dpi=180)
    print(f"Saved true TM-score landscape -> {out_art}")


if __name__ == "__main__":
    main()
