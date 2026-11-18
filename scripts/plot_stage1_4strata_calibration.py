#!/usr/bin/env python3
"""Plot 4-stratum calibration bounds + 4 Crystal vs Chiral L-BFGS 3D overlays for Stage 1 (N=81)."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from fable_bench.tmscore import evaluate_3d_superposition


def parse_ca(pdb_path: Path) -> np.ndarray:
    pts = []
    for line in pdb_path.read_text(encoding="utf-8").splitlines():
        if line.startswith("ATOM  ") and line[12:16].strip() == "CA":
            pts.append([float(line[30:38]), float(line[38:46]), float(line[46:54])])
    return np.array(pts, dtype=np.float64)


def project_pca_2d(ca: np.ndarray, other: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mu = ca.mean(axis=0)
    x = ca - mu
    y = other - mu
    _u, _s, vt = np.linalg.svd(x, full_matrices=False)
    basis = vt[:2].T  # (3, 2)
    return x @ basis, y @ basis


def main() -> None:
    root = Path("/Users/taluguloglu/.gemini/antigravity/scratch/fable-protein-bench")
    data_dir = root / "data" / "stage1_benchmark"
    res_dir = root / "outputs" / "results"
    out_png = Path("/Users/taluguloglu/.gemini/antigravity/brain/4775d422-bdcb-488c-afa6-e25f2cd600e6/stage1_4strata_calibration_and_3d.png")

    manifest = json.loads((data_dir / "manifest.json").read_text(encoding="utf-8"))
    rep = json.loads((res_dir / "stage1_evaluation_report.json").read_text(encoding="utf-8"))

    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.size": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )

    fig = plt.figure(figsize=(15.5, 8.6), dpi=180, facecolor="#ffffff")
    gs = fig.add_gridspec(2, 4, height_ratios=[1.05, 1.0], hspace=0.38, wspace=0.28)

    # Top-Left (cols 0..1): 4-Stratum Forward Folding Calibration Ladder
    ax_bar = fig.add_subplot(gs[0, :2])
    strata = [
        ("pre_cutoff_control", "Pre-Cutoff\nControl (n=15)"),
        ("post_cutoff_homolog_gradient", "2026 Homolog\nGradient (n=15)"),
        ("post_cutoff_pdb_novel", "2026 PDB-Novel /\nAFDB-Known (n=21)"),
        ("post_cutoff_de_novo", "2026 Synthetic\nDe Novo (n=30)"),
    ]
    models = [
        ("Oracle_TrueContacts", "Oracle Contacts + Chiral L-BFGS", "#0ea5e9"),
        ("NoisyOracle_50pctPrec", "50% Precision Noisy Contacts", "#f59e0b"),
        ("Null_Random", "Random Contacts Null", "#94a3b8"),
        ("Heuristic_1D_LocalOnly", "1D Chou-Fasman Only (0 Tertiary)", "#ef4444"),
    ]
    fwd_by_m = {m["model"]: m for m in rep["stage1a_forward_folding"]}
    x = np.arange(len(strata))
    w = 0.19
    for idx, (mkey, mlabel, col) in enumerate(models):
        vals = [fwd_by_m[mkey]["splits"][sk]["tm_score"] for sk, _ in strata]
        bars = ax_bar.bar(x + (idx - 1.5) * w, vals, width=w, color=col, label=mlabel, edgecolor="white", linewidth=0.6)
        if mkey in ("Oracle_TrueContacts", "Heuristic_1D_LocalOnly"):
            for b, v in zip(bars, vals):
                ax_bar.text(b.get_x() + b.get_width() / 2, v + 0.015, f"{v:.2f}", ha="center", va="bottom", fontsize=8, fontweight="bold", color="#1e293b")

    ax_bar.axhline(0.50, color="#0f172a", linestyle="--", linewidth=1.0, alpha=0.55, label="Correct Fold Threshold (TM = 0.50)")
    ax_bar.set_xticks(x)
    ax_bar.set_xticklabels([lbl for _, lbl in strata], fontsize=9)
    ax_bar.set_ylim(0, 1.02)
    ax_bar.set_ylabel("Mean TM-score (SO(3) Exact)")
    ax_bar.set_title("A. Stage 1A Calibration Bounds Across All 4 Novelty Strata (N = 81)", fontweight="bold", loc="left")
    ax_bar.legend(frameon=False, fontsize=8, ncol=2, loc="upper left")

    # Top-Right (cols 2..3): Per-Target Oracle TM-score vs Sequence Length colored by Stratum
    ax_sc = fig.add_subplot(gs[0, 2:])
    meta_by_id = {t["target_id"]: t for t in manifest["targets"]}
    s_colors = {
        "pre_cutoff_control": ("#3b82f6", "Pre-Cutoff (15)"),
        "post_cutoff_homolog_gradient": ("#f59e0b", "2026 Homologs (15)"),
        "post_cutoff_pdb_novel": ("#10b981", "2026 PDB-Novel / AFDB (21)"),
        "post_cutoff_de_novo": ("#ec4899", "2026 Synthetic De Novo (30)"),
    }
    oracle_pts = fwd_by_m["Oracle_TrueContacts"]["per_target"]
    plotted_lbl = set()
    for pt in oracle_pts:
        m = meta_by_id[pt["target_id"]]
        st = m["stratum"]
        col, lbl = s_colors[st]
        ax_sc.scatter(
            pt["length"],
            pt["tm_score"],
            c=col,
            s=46,
            alpha=0.85,
            edgecolors="#0f172a",
            linewidths=0.4,
            label=lbl if st not in plotted_lbl else None,
        )
        plotted_lbl.add(st)
        if m["pdb_id"] in ("6MRR", "23WS", "23WT", "36BS"):
            ax_sc.annotate(m["pdb_id"], (pt["length"], pt["tm_score"]), xytext=(4, 3), textcoords="offset points", fontsize=8, fontweight="bold")

    ax_sc.axhline(0.50, color="#0f172a", linestyle="--", linewidth=1.0, alpha=0.5)
    ax_sc.set_xlabel("Monomer / Domain Length (Residues)")
    ax_sc.set_ylabel("Chiral L-BFGS Oracle TM-score")
    ax_sc.set_ylim(0.35, 1.00)
    ax_sc.set_title("B. 3D Reconstructor Fidelity Across Length & Novelty (Mean TM = 0.74)", fontweight="bold", loc="left")
    ax_sc.legend(frameon=False, fontsize=8.5, loc="lower left")

    # Bottom Row (4 panels): Crystal (solid dark) vs Chiral L-BFGS Reconstructed Fold (cyan) across all 4 strata
    spotlights = [
        ("6MRR", "Pre-Cutoff (@sokrypton)\n6MRR · βαβαβ (68 AA)"),
        ("37NT", "2026 Homolog (@WUFang)\n37NT D1 · MERTK Ig (100 AA)"),
        ("36DO", "2026 PDB-Novel / AFDB-Known\n36DO · FlbD (53 AA)"),
        ("23WT", "2026 Zero-Prior De Novo\n23WT · SOD1 Binder (91 AA)"),
    ]
    for col_idx, (pdb_code, panel_title) in enumerate(spotlights):
        ax = fig.add_subplot(gs[1, col_idx])
        rec = next(t for t in manifest["targets"] if t["pdb_id"] == pdb_code)
        tid = rec["target_id"]
        npz = np.load(data_dir / "ground_truth" / f"{tid}.npz")
        ca = npz["coords_ca"] - npz["coords_ca"].mean(axis=0)
        rc = parse_ca(res_dir / "reconstructed_pdbs" / f"Oracle_TrueContacts__{tid}.pdb")
        sup = evaluate_3d_superposition(rc, ca)
        p_cryst, p_recon = project_pca_2d(ca, sup.aligned_pred_ca)

        ax.plot(p_cryst[:, 0], p_cryst[:, 1], color="#1e293b", linewidth=2.8, label="Crystal Cα", alpha=0.9)
        ax.plot(p_recon[:, 0], p_recon[:, 1], color="#06b6d4", linewidth=2.0, linestyle="--", label="Chiral L-BFGS", alpha=0.95)
        ax.scatter(p_cryst[:, 0], p_cryst[:, 1], c=np.linspace(0, 1, len(ca)), cmap="Spectral", s=16, zorder=3)
        ax.set_aspect("equal", adjustable="datalim")
        ax.set_xticks([])
        ax.set_yticks([])
        for sp in ("left", "bottom", "top", "right"):
            ax.spines[sp].set_visible(True)
            ax.spines[sp].set_color("#e2e8f0")
        ax.set_title(panel_title, fontsize=9.2, fontweight="bold")
        ax.text(
            0.03,
            0.04,
            f"TM = {sup.tm_score:.3f} · RMSD = {sup.rmsd_global:.2f} Å\nPDB SeqID: {rec['pdb_precutoff_seqid_max']:.0f}% | AFDB: {rec['afdb_max_seqid']:.0f}%",
            transform=ax.transAxes,
            fontsize=8,
            bbox=dict(boxstyle="round,pad=0.25", facecolor="#f8fafc", edgecolor="#cbd5e1", alpha=0.92),
        )
        if col_idx == 0:
            ax.legend(frameon=True, fontsize=7.8, loc="upper right")

    fig.suptitle("Stage 1 End-to-End Verification: 4 Novelty Regimes & Chiral 3D Contact Folding (N = 81)", fontsize=13, fontweight="bold", y=0.98)
    fig.savefig(out_png, bbox_inches="tight", dpi=180)
    print(f"Saved 4-stratum calibration + 3D superposition showcase -> {out_png}")


if __name__ == "__main__":
    main()
