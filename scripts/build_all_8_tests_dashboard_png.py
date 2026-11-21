#!/usr/bin/env python3
"""Render a 3x3 9-panel publication-grade static PNG dashboard covering ALL 8 ProteinBench tests
for GPT-6.1 Sol, GPT-6.2 Sol, and GPT-6.2 Astra (including ESMFold-2 scTM & 16-shot AF2 trajectories)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parents[1]

MODELS = [
    ("gpt-6.1-sol_think4096", "GPT-6.1 Sol", "#0284C7"),
    ("gpt-6.2-sol_think4096", "GPT-6.2 Sol", "#4F46E5"),
    ("gpt-6.2-astra_think4096", "GPT-6.2 Astra", "#E11D48"),
]


def main() -> None:
    rdir = REPO / "outputs" / "results"
    pdir = REPO / "outputs" / "predictions"
    fdir = REPO / "outputs" / "figures"
    fdir.mkdir(parents=True, exist_ok=True)

    s1_rep = json.loads((rdir / "stage1_evaluation_report.json").read_text(encoding="utf-8"))
    t1 = {m["model"]: m for m in s1_rep["stage1a_forward_folding"]}
    t2 = {m["model"]: m for m in s1_rep["stage1b_inverse_folding"]}
    loc = json.loads((rdir / "local_13456_unified_summary.json").read_text(encoding="utf-8"))
    t3 = loc["test3_remote_homology"]
    t4 = loc["test4_cafa_function"]
    t5 = {m["model"]: m for m in loc["test5_monomer_alanine_ranking"]}
    t6 = {m["model"]: m for m in loc["test6_binding_alanine_ranking"]}
    t7 = json.loads((rdir / "test7_esmfold2_evaluation_report.json").read_text(encoding="utf-8"))["models"]

    t8: Dict[str, Any] = {}
    for mtag, _mlbl, _mcol in MODELS:
        cf = pdir / f"stage3b_binder_campaign_{mtag}_closed_loop_af2.json"
        if cf.exists():
            t8[mtag] = json.loads(cf.read_text(encoding="utf-8"))

    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.size": 9.2,
            "axes.titlesize": 10.6,
            "axes.titleweight": "bold",
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )
    fig, axes = plt.subplots(3, 3, figsize=(17.6, 13.4), dpi=200, facecolor="#FAFAFC")

    # 1. Test 1 Forward Folding
    ax = axes[0, 0]
    ax.set_facecolor("#FFFFFF")
    strata1 = [
        ("pre_cutoff_control", "Known\n(15)"),
        ("post_cutoff_de_novo", "DeNovo\n(28)"),
        ("post_cutoff_homolog_gradient", "Homol\n(12)"),
        ("post_cutoff_pdb_novel", "PDBNov\n(27)"),
        ("post_cutoff_struct_novel", "TrueNov\n(18)"),
    ]
    x = np.arange(len(strata1))
    pm1 = [("Heuristic_1D_LocalOnly", "1D Chou-Fasman", "#CBD5E1")] + MODELS
    w = 0.78 / len(pm1)
    for i, (mk, ml, mc) in enumerate(pm1):
        if mk not in t1:
            continue
        vals = [t1[mk]["splits"].get(sk, {}).get("tm_score", 0.0) for sk, _ in strata1]
        ov = t1[mk]["overall"]["tm_score"]
        ax.bar(x + (i - (len(pm1) - 1) / 2) * w, vals, width=w * 0.92, color=mc, label=f"{ml} ({ov:.3f})")
    ax.axhline(0.50, color="#334155", linestyle="--", linewidth=0.9, alpha=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels([l for _, l in strata1], fontsize=8.2)
    ax.set_ylim(0, 0.56)
    ax.set_ylabel("Reconstructed 3D TM-Score")
    ax.set_title("1. Forward Folding (N=100, Sequence -> Contacts -> 3D)")
    ax.legend(fontsize=7.4, loc="upper right", frameon=True)
    ax.grid(True, axis="y", alpha=0.2)

    # 2. Test 2 Inverse Folding (ESMFold-2 scTM)
    ax = axes[0, 1]
    ax.set_facecolor("#FFFFFF")
    pm2 = MODELS + [("control_wildtype_native", "Wild-Type Ceiling", "#10B981")]
    w2 = 0.78 / len(pm2)
    for i, (mk, ml, mc) in enumerate(pm2):
        if mk not in t2:
            continue
        vals = [t2[mk]["splits"].get(sk, {}).get("sc_tm_score", 0.0) for sk, _ in strata1]
        ov = t2[mk]["overall"].get("sc_tm_score", 0.0)
        ax.bar(x + (i - (len(pm2) - 1) / 2) * w2, vals, width=w2 * 0.92, color=mc, label=f"{ml} (scTM={ov:.3f})")
    ax.axhline(0.50, color="#334155", linestyle="--", linewidth=0.9, alpha=0.6, label="Fold Match (scTM=0.5)")
    ax.set_xticks(x)
    ax.set_xticklabels([l for _, l in strata1], fontsize=8.2)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("ESMFold-2 Refolding scTM")
    ax.set_title("2. Inverse Folding (N=100, Backbone -> Seq -> ESMFold-2)")
    ax.legend(fontsize=7.3, loc="upper right", frameon=True)
    ax.grid(True, axis="y", alpha=0.2)

    # 3. Test 3 Remote Homology
    ax = axes[0, 2]
    ax.set_facecolor("#FFFFFF")
    rungs = [("1_seq_easy", "Rung 1\nSeq-Easy"), ("2_foldseek_3di", "Rung 2\nFoldseek 3Di"), ("3_deep_tea_zone", "Rung 3\nDeep TEA")]
    x3 = np.arange(len(rungs))
    pm3 = [("BLOSUM62_SmithWaterman", "BLOSUM62 SW", "#94A3B8"), ("Foldseek_3Di_Structure", "Foldseek 3Di", "#0D9488")] + MODELS
    w3 = 0.78 / len(pm3)
    for i, (mk, ml, mc) in enumerate(pm3):
        if mk not in t3:
            continue
        vals = [t3[mk]["by_rung"][rk]["recall_at_fpr10"] for rk, _ in rungs]
        auc = t3[mk]["overall_auroc"]
        ax.bar(x3 + (i - (len(pm3) - 1) / 2) * w3, vals, width=w3 * 0.92, color=mc, label=f"{ml} (AUC={auc:.2f})")
    ax.set_xticks(x3)
    ax.set_xticklabels([l for _, l in rungs])
    ax.set_ylim(0, 1.15)
    ax.set_ylabel("Recall @ 10% FPR")
    ax.set_title("3. Remote Homology Detection (N=120 SCOPe Pairs)")
    ax.legend(fontsize=7.3, loc="upper right", frameon=True)
    ax.grid(True, axis="y", alpha=0.2)

    # 4. Test 4 CAFA Function
    ax = axes[1, 0]
    ax.set_facecolor("#FFFFFF")
    asps = ["MF", "BP", "CC"]
    x4 = np.arange(len(asps))
    pm4 = [("CAFA_SequenceHomology_SW", "SW Homology", "#94A3B8"), ("CAFA_NaivePrior", "CAFA Naive", "#F59E0B")] + MODELS
    w4 = 0.78 / len(pm4)
    for i, (mk, ml, mc) in enumerate(pm4):
        if mk not in t4:
            continue
        vals = [t4[mk][a]["wfmax"] for a in asps]
        mw = t4[mk]["macro_wfmax"]
        ax.bar(x4 + (i - (len(pm4) - 1) / 2) * w4, vals, width=w4 * 0.92, color=mc, label=f"{ml} ({mw:.3f})")
    ax.set_xticks(x4)
    ax.set_xticklabels(["Molecular Function\n(MF)", "Biological Process\n(BP)", "Cellular Component\n(CC)"])
    ax.set_ylim(0, 0.78)
    ax.set_ylabel("IC-Weighted CAFA wF_max")
    ax.set_title("4. Protein Function Prediction (N=60 2026 GO Holdouts)")
    ax.legend(fontsize=7.4, loc="upper left", frameon=True)
    ax.grid(True, axis="y", alpha=0.2)

    # 5. Test 5 Monomer Alanine Scan
    ax = axes[1, 1]
    ax.set_facecolor("#FFFFFF")
    pm5 = [("Baseline_AA_LookupOnly", "1D AA Lookup", "#CBD5E1"), ("Baseline_LocalWindowKD", "Local KD Win", "#94A3B8")] + MODELS
    x5 = np.arange(len(pm5))
    nd5 = [t5[mk]["overall"]["ndcg_at_5"] for mk, _, _ in pm5 if mk in t5]
    wn5 = [t5[mk]["overall"]["same_aa_win_rate"] for mk, _, _ in pm5 if mk in t5]
    cols5 = [mc for mk, _, mc in pm5 if mk in t5]
    lbls5 = [ml.replace(" ", "\n") for mk, ml, _ in pm5 if mk in t5]
    ax.bar(x5 - 0.17, nd5, width=0.32, color=cols5, label="NDCG@5")
    ax.bar(x5 + 0.17, wn5, width=0.32, color=cols5, alpha=0.55, hatch="//", label="Same-AA Win Rate")
    for idx_b, (v1, v2) in enumerate(zip(nd5, wn5)):
        ax.text(idx_b - 0.17, v1 + 0.012, f"{v1:.2f}", ha="center", fontsize=7.6, fontweight="bold")
        ax.text(idx_b + 0.17, v2 + 0.012, f"{v2*100:.0f}%", ha="center", fontsize=7.4)
    ax.axhline(0.50, color="#64748B", linestyle=":", linewidth=0.9)
    ax.set_xticks(x5)
    ax.set_xticklabels(lbls5, fontsize=8.2)
    ax.set_ylim(0, 0.88)
    ax.set_ylabel("Score / Win Rate")
    ax.set_title("5. Monomer Alanine Scan Ranking (N=52, 18-Site Panels)")
    ax.legend(fontsize=7.6, loc="upper left", frameon=True)
    ax.grid(True, axis="y", alpha=0.2)

    # 6. Test 6 Interface Binding Alanine Scan (Known PDB vs Unseen/Post-2026)
    ax = axes[1, 2]
    ax.set_facecolor("#FFFFFF")
    pm6 = [("Baseline_BoganThorn_1D", "Bogan-Thorn 1D", "#94A3B8")] + MODELS
    x6 = np.arange(len(pm6))
    kn6 = [t6[mk]["by_memorization_stratum"]["known_pdb_pre2026"]["spearman_rho"] for mk, _, _ in pm6 if mk in t6]
    un6 = [t6[mk]["by_memorization_stratum"]["unseen_or_post2026"]["spearman_rho"] for mk, _, _ in pm6 if mk in t6]
    cols6 = [mc for mk, _, mc in pm6 if mk in t6]
    lbls6 = [ml.replace(" ", "\n") for mk, ml, _ in pm6 if mk in t6]
    ax.bar(x6 - 0.17, kn6, width=0.32, color=cols6, label="Known Complex PDB (N=29)")
    ax.bar(x6 + 0.17, un6, width=0.32, color=cols6, alpha=0.48, hatch="//", label="Unseen / Post-2026 (N=9)")
    for idx_b, (v1, v2) in enumerate(zip(kn6, un6)):
        ax.text(idx_b - 0.17, v1 + 0.012, f"{v1:.2f}", ha="center", fontsize=7.6, fontweight="bold")
        ax.text(idx_b + 0.17, max(0.01, v2) + 0.012, f"{v2:.2f}", ha="center", fontsize=7.4)
    ax.set_xticks(x6)
    ax.set_xticklabels(lbls6, fontsize=8.4)
    ax.set_ylim(-0.02, 0.65)
    ax.set_ylabel("Spearman Rank ρ")
    ax.set_title("6. Interface Binding Ala Ranking: Known vs. Unseen")
    ax.legend(fontsize=7.6, loc="upper left", frameon=True)
    ax.grid(True, axis="y", alpha=0.2)

    # 7. Test 7 De Novo Monomer & Motif Scaffolding Gate Breakdown
    ax = axes[2, 0]
    ax.set_facecolor("#FFFFFF")
    g7_keys = [
        ("gate1_hygiene_rate", "Gate 1\nComplexity"),
        ("gate5_novelty_rate", "Gate 5\n<30% Novel"),
        ("gate3_cross_oracle_sc_rate", "Gate 3\nOF3 scRMSD<=2Å"),
        ("gate4_adherence_rate", "Gate 4\nFold/Motif"),
        ("gate2_foldable_rate", "Gate 2\npLDDT>=80"),
    ]
    x7 = np.arange(len(g7_keys))
    w7 = 0.76 / len(MODELS)
    for i, (mk, ml, mc) in enumerate(MODELS):
        if mk not in t7:
            continue
        vals = [t7[mk][gk] for gk, _ in g7_keys]
        pl = t7[mk]["mean_plddt"]
        ax.bar(x7 + (i - (len(MODELS) - 1) / 2) * w7, vals, width=w7 * 0.92, color=mc, label=f"{ml} (pLDDT={pl:.1f})")
    ax.set_xticks(x7)
    ax.set_xticklabels([gl for _, gl in g7_keys], fontsize=8.1)
    ax.set_ylim(0, 1.15)
    ax.set_ylabel("Pass Fraction (N=40)")
    ax.set_title("7. De Novo & Motif Design Gates (ESMFold-2 + OpenFold3)")
    ax.legend(fontsize=7.5, loc="upper right", frameon=True)
    ax.grid(True, axis="y", alpha=0.2)

    # 8. Test 8 Closed-Loop AF2 Optimization Trajectory (Rounds 1..16)
    ax = axes[2, 1]
    ax.set_facecolor("#FFFFFF")
    rx = np.arange(1, 17)
    for mk, ml, mc in MODELS:
        if mk not in t8:
            continue
        sm = t8[mk]["summary"]
        mu = np.array(sm["trajectory_mean"])
        se = np.array(sm["trajectory_sem"])
        gain = mu[-1] - mu[0]
        ax.plot(rx, mu, color=mc, marker="o", markersize=4.0, linewidth=2.2, label=f"{ml} ({mu[0]:.2f}->{mu[-1]:.2f}, Δ+{gain:.2f})")
        ax.fill_between(rx, mu - se, mu + se, color=mc, alpha=0.14)
    ax.axhline(0.76, color="#64748B", linestyle="--", linewidth=1.0, label="Strict BindCraft Hit Regime (~0.76)")
    ax.set_xticks([1, 4, 8, 12, 16])
    ax.set_ylim(0.10, 0.82)
    ax.set_xlabel("Closed-Loop AlphaFold2 (`af2ig`) Turn (Shot 1 .. 16)")
    ax.set_ylabel("Running-Best Interface Score (N=12)")
    ax.set_title("8a. Closed-Loop AF2 Binder Optimization Trajectory")
    ax.legend(fontsize=7.4, loc="upper left", frameon=True)
    ax.grid(True, alpha=0.2)

    # 9. Test 8 Shot-1 vs Best-of-16 AF2 ipTM across the 12 Epitopes
    ax = axes[2, 2]
    ax.set_facecolor("#FFFFFF")
    x9 = np.arange(len(MODELS))
    s1_iptms, best_iptms, on_ep_iptms = [], [], []
    for mk, _ml, _mc in MODELS:
        if mk not in t8:
            s1_iptms.append(0.0)
            best_iptms.append(0.0)
            on_ep_iptms.append(0.0)
            continue
        trajs = t8[mk]["trajectories"]
        s1_iptms.append(float(np.mean([v[0]["feedback"]["iptm"] for v in trajs.values()])))
        best_iptms.append(float(np.mean([max(s["feedback"]["iptm"] for s in v) for v in trajs.values()])))
        on_ep_iptms.append(float(t8[mk]["summary"]["mean_best_iptm"]))
    cols9 = [mc for _, _, mc in MODELS]
    ax.bar(x9 - 0.24, s1_iptms, width=0.22, color=cols9, alpha=0.45, label="Shot 1 AF2 ipTM (Blind)")
    ax.bar(x9, best_iptms, width=0.22, color=cols9, label="Best-of-16 AF2 ipTM (All)")
    ax.bar(x9 + 0.24, on_ep_iptms, width=0.22, color=cols9, hatch="//", alpha=0.82, label="Best-of-16 On-Epitope (>=2 HS)")
    for idx_b in range(len(MODELS)):
        ax.text(idx_b - 0.24, s1_iptms[idx_b] + 0.01, f"{s1_iptms[idx_b]:.2f}", ha="center", fontsize=7.6)
        ax.text(idx_b, best_iptms[idx_b] + 0.01, f"{best_iptms[idx_b]:.2f}", ha="center", fontsize=7.8, fontweight="bold")
        ax.text(idx_b + 0.24, on_ep_iptms[idx_b] + 0.01, f"{on_ep_iptms[idx_b]:.2f}", ha="center", fontsize=7.6)
    ax.axhline(0.75, color="#334155", linestyle="--", linewidth=0.9, alpha=0.6, label="Hit Threshold (ipTM=0.75)")
    ax.set_xticks(x9)
    ax.set_xticklabels([ml for _, ml, _ in MODELS])
    ax.set_ylim(0, 0.85)
    ax.set_ylabel("Official AlphaFold2 (`af2ig`) ipTM")
    ax.set_title("8b. Epitope Binder AF2 ipTM: Shot 1 vs. Closed-Loop")
    ax.legend(fontsize=7.3, loc="upper right", frameon=True)
    ax.grid(True, axis="y", alpha=0.2)

    fig.suptitle(
        "ProteinBench Complete 8-Test Live Evaluation — GPT-6.1 Sol vs. GPT-6.2 Sol vs. GPT-6.2 Astra (Closed-Book Weights + ESMFold-2 & Official AF2 Oracles)",
        fontsize=13.0,
        fontweight="bold",
        y=0.992,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.974))
    out_png = fdir / "all_8_tests_live_dashboard.png"
    fig.savefig(out_png, dpi=200)
    plt.close(fig)
    print(f"Saved 9-panel complete dashboard -> {out_png}")


if __name__ == "__main__":
    main()
