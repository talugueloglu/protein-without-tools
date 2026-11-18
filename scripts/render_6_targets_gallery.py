#!/usr/bin/env python3
"""Render high-resolution 3D shaded ribbon-tube + contact-map visuals for 6 diverse Stage 1 targets."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import numpy as np
from scipy.interpolate import make_interp_spline

SHOWCASE_IDS = [
    "s1_t001",  # 6MRR (68 AA, Pre-cutoff 2019 Baker de novo βαβαβ — @sokrypton tweet)
    "s1_t005",  # 6CZH (105 AA, Pre-cutoff 2018 Baker mFAP0 all-β barrel)
    "s1_t011",  # 37NT (100 AA, Post-cutoff 2026-10-07 MERTK Ig-like all-β — @WUFang tweet)
    "s1_t025",  # 47WY (72 AA, Post-cutoff 2026-11-18 de novo helical miniprotein)
    "s1_t020",  # 36DO (53 AA, Post-cutoff 2026-11-18 B. burgdorferi FlbD α/β)
    "s1_t019",  # 36BS (207 AA, Post-cutoff 2026-11-18 synthetic designed desB α/β)
]


def pca_orient_coords(ca: np.ndarray) -> np.ndarray:
    """Center and align principal axes of Cα coordinates so the fold faces the camera cleanly."""
    centered = ca - ca.mean(axis=0)
    _u, _s, vt = np.linalg.svd(centered, full_matrices=False)
    if np.linalg.det(vt) < 0:
        vt[-1, :] *= -1.0
    oriented = centered @ vt.T
    # Tilt slightly around X and Y for rich 3D depth perception
    ax, ay = np.deg2rad(22.0), np.deg2rad(-18.0)
    rx = np.array([[1, 0, 0], [0, np.cos(ax), -np.sin(ax)], [0, np.sin(ax), np.cos(ax)]])
    ry = np.array([[np.cos(ay), 0, np.sin(ay)], [0, 1, 0], [-np.sin(ay), 0, np.cos(ay)]])
    return oriented @ rx.T @ ry.T


def draw_pseudo3d_tube(ax: plt.Axes, ca: np.ndarray, q3: str) -> None:
    """Render a publication-grade depth-sorted smooth spline protein tube colored N->C rainbow."""
    coords = pca_orient_coords(ca)
    L = coords.shape[0]

    # Smooth cubic spline upsampling (6x per residue) for crisp helical coils & beta ribbons
    t_orig = np.linspace(0.0, 1.0, L)
    n_fine = max(200, L * 7)
    t_fine = np.linspace(0.0, 1.0, n_fine)
    spl = make_interp_spline(t_orig, coords, k=min(3, L - 1))
    pts = spl(t_fine)  # (n_fine, 3)

    # Map each fine segment to rainbow color + depth (z) cueing + secondary-structure width
    segs = np.stack([pts[:-1, :2], pts[1:, :2]], axis=1)  # (n_fine-1, 2, 2)
    z_mid = 0.5 * (pts[:-1, 2] + pts[1:, 2])
    z_min, z_max = float(z_mid.min()), float(z_mid.max())
    z_norm = (z_mid - z_min) / max(1e-5, z_max - z_min)  # 0 = back, 1 = front

    t_mid = 0.5 * (t_fine[:-1] + t_fine[1:])
    res_idx = np.clip((t_mid * (L - 1)).astype(int), 0, L - 1)

    cmap = plt.get_cmap("turbo")
    base_rgba = cmap(0.08 + 0.84 * t_mid)

    # Painter's algorithm: sort segments from back (lowest z) to front (highest z)
    order = np.argsort(z_mid)
    segs_sorted = segs[order]

    # Dark outline halo pass + lit core pass for clean 3D occlusion borders
    halo_colors = []
    core_colors = []
    halo_widths = []
    core_widths = []

    for k in order:
        ss = q3[res_idx[k]]
        base_w = 5.0 if ss == "H" else (4.4 if ss == "E" else 2.5)
        depth_scale = 0.72 + 0.45 * z_norm[k]
        w = base_w * depth_scale

        # Mix slightly toward white/bright in foreground, muted in background
        shade = 0.52 + 0.48 * z_norm[k]
        r, g, b, _ = base_rgba[k]
        core_colors.append((min(1.0, r * shade), min(1.0, g * shade), min(1.0, b * shade), 0.98))
        halo_colors.append((0.06, 0.09, 0.16, 0.88))
        core_widths.append(w)
        halo_widths.append(w + 1.8)

    # Draw subtle long-range Cβ contact chords in background first
    for i in range(0, L, 2):
        for j in range(i + 18, L, 2):
            if np.linalg.norm(coords[i] - coords[j]) < 7.8:
                ax.plot(
                    [coords[i, 0], coords[j, 0]],
                    [coords[i, 1], coords[j, 1]],
                    color="#94a3b8",
                    alpha=0.16,
                    linewidth=0.7,
                    linestyle="--",
                    zorder=1,
                )

    lc_halo = LineCollection(segs_sorted, colors=halo_colors, linewidths=halo_widths, capstyle="round", joinstyle="round", zorder=2)
    lc_core = LineCollection(segs_sorted, colors=core_colors, linewidths=core_widths, capstyle="round", joinstyle="round", zorder=3)
    ax.add_collection(lc_halo)
    ax.add_collection(lc_core)

    # Mark N and C termini clearly
    ax.scatter([coords[0, 0]], [coords[0, 1]], s=55, color="#2563eb", edgecolors="white", linewidths=1.2, zorder=5)
    ax.scatter([coords[-1, 0]], [coords[-1, 1]], s=55, color="#dc2626", edgecolors="white", linewidths=1.2, zorder=5)
    ax.text(coords[0, 0], coords[0, 1] + 1.4, "N", fontsize=8.5, fontweight="bold", color="#1d4ed8", ha="center", zorder=6)
    ax.text(coords[-1, 0], coords[-1, 1] + 1.4, "C", fontsize=8.5, fontweight="bold", color="#b91c1c", ha="center", zorder=6)

    span = max(float(np.ptp(coords[:, 0])), float(np.ptp(coords[:, 1]))) * 0.60 + 3.0
    cx = 0.5 * (float(coords[:, 0].min()) + float(coords[:, 0].max()))
    cy = 0.5 * (float(coords[:, 1].min()) + float(coords[:, 1].max()))
    ax.set_xlim(cx - span, cx + span)
    ax.set_ylim(cy - span, cy + span)
    ax.set_aspect("equal")
    ax.axis("off")


def render_single_card(rec: Dict[str, Any], npz: Any, out_png: Path) -> None:
    """Render a crisp 2-panel card (Left: 3D Structure Tube, Right: Cβ < 8 Å Contact Map + Q3 bar)."""
    ca = npz["coords_ca"]
    cb_mat = npz["cb_dist_matrix"]
    q3 = str(npz["q3"])
    L = ca.shape[0]

    fig, (ax_3d, ax_cm) = plt.subplots(
        1, 2, figsize=(8.8, 4.1), dpi=180, gridspec_kw={"width_ratios": [1.15, 1.0]}
    )
    fig.patch.set_facecolor("#ffffff")
    ax_3d.set_facecolor("#f8fafc")

    draw_pseudo3d_tube(ax_3d, ca, q3)
    split_lbl = "PRE-2025 CONTROL" if rec["temporal_split"] == "pre_cutoff" else f"POST-CUTOFF ({rec['release_date']})"
    ax_3d.set_title(
        f"3D Crystal Fold — {rec['pdb_id']}:{rec['chain_id']} ({rec['target_id']})\n"
        f"L = {L} AA | {rec['fold_class']} | {split_lbl}",
        fontsize=9.5,
        fontweight="bold",
        pad=8,
    )

    # Contact map panel
    ii, jj = np.where((cb_mat < 8.0) & (np.abs(np.arange(L)[:, None] - np.arange(L)[None, :]) >= 6))
    dists = cb_mat[ii, jj]
    sc = ax_cm.scatter(jj + 1, ii + 1, c=dists, cmap="viridis_r", s=max(4.0, 26.0 - 0.09 * L), vmin=3.5, vmax=8.0)
    ax_cm.plot([1, L], [1, L], color="#cbd5e1", linewidth=0.9)
    ax_cm.set_xlim(1, L)
    ax_cm.set_ylim(L, 1)
    ax_cm.set_aspect("equal")
    ax_cm.set_xlabel("Residue j", fontsize=8.5)
    ax_cm.set_ylabel("Residue i", fontsize=8.5)
    ax_cm.set_title(
        f"True Contact Map (|i-j| ≥ 6, Cβ < 8 Å)\n"
        f"{rec['description'][:40]}",
        fontsize=9.5,
        fontweight="bold",
        pad=8,
    )
    plt.colorbar(sc, ax=ax_cm, fraction=0.046, pad=0.04, label="Cβ–Cβ (Å)")

    fig.tight_layout()
    fig.savefig(out_png, bbox_inches="tight", facecolor="#ffffff")
    plt.close(fig)


def main() -> None:
    repo_root = Path("/Users/taluguloglu/.gemini/antigravity/scratch/fable-protein-bench")
    data_dir = repo_root / "data" / "stage1_benchmark"
    repo_fig_dir = repo_root / "outputs" / "figures" / "showcase_6"
    artifact_dir = Path("/Users/taluguloglu/.gemini/antigravity/brain/4775d422-bdcb-488c-afa6-e25f2cd600e6")
    repo_fig_dir.mkdir(parents=True, exist_ok=True)

    manifest = json.loads((data_dir / "manifest.json").read_text(encoding="utf-8"))
    rec_by_id = {r["target_id"]: r for r in manifest["targets"]}

    # 1. Render 6 individual high-res 2-panel cards AND a 2x3 master overview grid
    fig_grid, axes_grid = plt.subplots(2, 3, figsize=(15.0, 9.4), dpi=180)
    fig_grid.patch.set_facecolor("#ffffff")

    widget_targets: List[Dict[str, Any]] = []

    for idx, tid in enumerate(SHOWCASE_IDS):
        rec = rec_by_id[tid]
        npz = np.load(data_dir / "ground_truth" / f"{tid}.npz", allow_pickle=False)
        card_name = f"card_{tid}_{rec['pdb_id']}.png"
        render_single_card(rec, npz, repo_fig_dir / card_name)
        render_single_card(rec, npz, artifact_dir / card_name)

        # Also draw onto master 2x3 grid
        ax = axes_grid[idx // 3, idx % 3]
        ax.set_facecolor("#f8fafc")
        draw_pseudo3d_tube(ax, npz["coords_ca"], str(npz["q3"]))
        tag = "Pre-2025" if rec["temporal_split"] == "pre_cutoff" else f"Released {rec['release_date']}"
        res_s = f"{rec['resolution_A']:.2f} Å" if rec["resolution_A"] else "NMR"
        ax.set_title(
            f"{idx + 1}. {rec['pdb_id']}:{rec['chain_id']} ({tid}, L={rec['length']})\n"
            f"{tag} | {rec['fold_class']} ({res_s})\n"
            f"{rec['description'][:38]}",
            fontsize=9.5,
            fontweight="bold",
        )

        # Export compact JSON geometry for inline interactive 3D HTML5 widget
        ca_oriented = np.round(pca_orient_coords(npz["coords_ca"]), 2).tolist()
        cb_mat = npz["cb_dist_matrix"]
        L = len(ca_oriented)
        lr_contacts = [
            [int(i), int(j), round(float(cb_mat[i, j]), 1)]
            for i in range(L)
            for j in range(i + 6, L)
            if cb_mat[i, j] < 8.0
        ]
        widget_targets.append(
            {
                "target_id": tid,
                "pdb_id": rec["pdb_id"],
                "chain_id": rec["chain_id"],
                "length": L,
                "temporal_split": rec["temporal_split"],
                "release_date": rec["release_date"],
                "resolution_A": rec["resolution_A"],
                "fold_class": rec["fold_class"],
                "organism": rec["organism"],
                "description": rec["description"],
                "note": rec["note"],
                "q3": str(npz["q3"]),
                "ca": ca_oriented,
                "contacts": lr_contacts,
            }
        )

    fig_grid.suptitle(
        "6 Representative Targets from Stage 1 (Pre-2025 Controls vs. Oct/Nov 2026 Post-Cutoff Releases)",
        fontsize=14,
        fontweight="bold",
        y=0.99,
    )
    fig_grid.tight_layout()
    fig_grid.savefig(repo_fig_dir / "stage1_6_targets_grid.png", bbox_inches="tight", facecolor="#ffffff")
    fig_grid.savefig(artifact_dir / "stage1_6_targets_grid.png", bbox_inches="tight", facecolor="#ffffff")
    plt.close(fig_grid)

    (repo_fig_dir / "widget_6_targets.json").write_text(json.dumps(widget_targets), encoding="utf-8")
    print("Rendered 6 individual cards, master 2x3 grid, and widget JSON payload.")


if __name__ == "__main__":
    main()
