#!/usr/bin/env python3
"""Plot the N=100 Stage 1 Sequence & Structure Similarity Landscape vs. (a) PDB and (b) AFDB."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


STRATUM_STYLE = {
    "pre_cutoff_control": ("Pre-2025 Controls", "#2563eb", "o"),
    "post_cutoff_homolog_gradient": ("2026 Homolog Gradient (30-95% PDB SeqID)", "#f59e0b", "^"),
    "post_cutoff_pdb_novel": ("2026 PDB-Novel Natural (<30% PDB SeqID)", "#10b981", "s"),
    "post_cutoff_de_novo": ("2026 Synthetic De Novo (<30% PDB & AFDB)", "#ef4444", "D"),
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot 100-target PDB vs AFDB similarity landscape.")
    parser.add_argument("--manifest", default="data/stage1_benchmark/manifest.json")
    parser.add_argument("--out-dir", default="outputs/figures")
    args = parser.parse_args()

    manifest = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    targets = manifest["targets"]
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.size": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )

    fig, axes = plt.subplots(1, 3, figsize=(16.2, 5.1), dpi=180)
    fig.patch.set_facecolor("#ffffff")
    rng = np.random.default_rng(7)

    # Panel A: Sequence vs Structure Similarity against Pre-Cutoff Experimental PDB
    ax0 = axes[0]
    for s_key, (lbl, col, mk) in STRATUM_STYLE.items():
        sub = [t for t in targets if t.get("stratum") == s_key]
        if not sub:
            continue
        xs = [float(t.get("pdb_precutoff_seqid_max", 0.0)) + rng.uniform(-0.8, 0.8) for t in sub]
        ys = [float(t.get("pdb100_precutoff_foldseek_prob", 0.0)) + rng.uniform(-0.012, 0.012) for t in sub]
        ax0.scatter(xs, np.clip(ys, 0, 1.02), label=f"{lbl} (n={len(sub)})", c=col, marker=mk, s=46, alpha=0.84, edgecolors="white", linewidths=0.6)
    ax0.axvline(30.0, linestyle="--", color="#64748b", linewidth=1.0)
    ax0.axhline(0.50, linestyle=":", color="#94a3b8", linewidth=1.0)
    ax0.set_xlim(-3, 105)
    ax0.set_ylim(-0.04, 1.06)
    ax0.set_xlabel("Max Sequence Identity to Pre-Cutoff PDB (%) [MMseqs2]")
    ax0.set_ylabel("Max Structure Homology Prob to Pre-Cutoff PDB100 [Foldseek]")
    ax0.set_title("A. Similarity to Experimental PDB (< May 2026)", fontweight="bold")
    ax0.legend(frameon=True, facecolor="#f8fafc", edgecolor="none", fontsize=7.8, loc="lower right")

    # Panel B: Sequence vs Structure Similarity against AlphaFold Predicted Structure DB (AFDB)
    ax1 = axes[1]
    for s_key, (lbl, col, mk) in STRATUM_STYLE.items():
        sub = [t for t in targets if t.get("stratum") == s_key]
        if not sub:
            continue
        xs = [float(t.get("afdb_max_seqid", 0.0)) + rng.uniform(-0.8, 0.8) for t in sub]
        ys = [float(t.get("afdb_foldseek_prob", 0.0)) + rng.uniform(-0.012, 0.012) for t in sub]
        ax1.scatter(xs, np.clip(ys, 0, 1.02), label=lbl, c=col, marker=mk, s=46, alpha=0.84, edgecolors="white", linewidths=0.6)
    ax1.axvline(30.0, linestyle="--", color="#64748b", linewidth=1.0)
    ax1.axhline(0.50, linestyle=":", color="#94a3b8", linewidth=1.0)
    ax1.set_xlim(-3, 105)
    ax1.set_ylim(-0.04, 1.06)
    ax1.set_xlabel("Max Sequence Identity to AlphaFold DB (%) [AFDB50/SwissProt]")
    ax1.set_ylabel("Max Structure Homology Prob to AlphaFold DB [Foldseek]")
    ax1.set_title("B. Similarity to AF-Predicted Structures (AFDB)", fontweight="bold")

    # Panel C: PDB SeqID vs AFDB SeqID (Isolating PDB-Novel/UniRef-Known from True Zero-Prior Orphans!)
    ax2 = axes[2]
    for s_key, (lbl, col, mk) in STRATUM_STYLE.items():
        sub = [t for t in targets if t.get("stratum") == s_key]
        if not sub:
            continue
        xs = [float(t.get("pdb_precutoff_seqid_max", 0.0)) + rng.uniform(-0.8, 0.8) for t in sub]
        ys = [float(t.get("afdb_max_seqid", 0.0)) + rng.uniform(-0.8, 0.8) for t in sub]
        ax2.scatter(xs, np.clip(ys, 0, 102), label=lbl, c=col, marker=mk, s=46, alpha=0.84, edgecolors="white", linewidths=0.6)
    ax2.axvline(30.0, linestyle="--", color="#64748b", linewidth=1.0)
    ax2.axhline(30.0, linestyle="--", color="#64748b", linewidth=1.0)
    ax2.fill_between([0, 30], 0, 30, color="#fee2e2", alpha=0.32, label="True Zero-Prior Zone (<30% PDB & AFDB)")
    ax2.fill_between([0, 30], 30, 105, color="#d1fae5", alpha=0.28, label="PDB-Novel, In UniRef/AFDB")
    ax2.set_xlim(-3, 105)
    ax2.set_ylim(-3, 105)
    ax2.set_xlabel("Max SeqID to Pre-Cutoff Experimental PDB (%)")
    ax2.set_ylabel("Max SeqID to AlphaFold DB / UniProt (%)")
    ax2.set_title("C. Disentangling PDB Leakage vs. UniRef/AFDB Exposure", fontweight="bold")
    ax2.legend(frameon=True, facecolor="#f8fafc", edgecolor="none", fontsize=7.5, loc="lower right")

    fig.suptitle(
        f"Stage 1 Dataset Audit (N = {len(targets)}): Sequence & Foldseek Structural Similarity to Pre-Cutoff PDB vs. AlphaFold DB",
        fontsize=13,
        fontweight="bold",
        y=1.02,
    )
    fig.tight_layout()
    for ext in ("png", "svg"):
        fig.savefig(out_dir / f"stage1_100_similarity_landscape.{ext}", bbox_inches="tight", facecolor="#ffffff")
    plt.close(fig)
    print(f"Saved N={len(targets)} similarity landscape -> {out_dir}/stage1_100_similarity_landscape.[png|svg]")


if __name__ == "__main__":
    main()
