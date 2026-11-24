#!/usr/bin/env python3
"""Unified Offline CPU Grader & Visual Dashboard for Local Protein Structure Without Tools Tests 1, 3, 4, 5, 6.

Executes all local graders end-to-end (zero network/GPU needed):
  1. Forward Folding (N=100, Chiral L-BFGS 3D reconstruction + 5 novelty strata)
  3. Remote Homology Detection (N=120 SCOPe 2.08 pairs across 4 rungs)
  4. Protein Function Prediction (N=60 2026 QuickGO experimental holdouts, CAFA wF_max)
  5. Monomer Alanine Scan Ranking (N=52 balanced 18-site panels, NDCG@5 & Same-AA Win Rate)
  6. Interface Binding Alanine Scan Ranking (N=38 monomer-stable complexes, Known PDB vs Unseen/Post-2026)

Outputs:
  - `outputs/results/local_13456_unified_summary.json`
  - `outputs/results/local_13456_leaderboard.md`
  - `outputs/figures/local_13456_live_dashboard.png`
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
import re
import subprocess
import sys
from typing import Any, Dict, List

import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]


def run_step(label: str, cmd: List[str]) -> None:
    print(f"\n>>> Running {label} ...", flush=True)
    subprocess.run(cmd, cwd=str(REPO_ROOT), check=True)


def collect_api_telemetry(pred_dir: Path) -> Dict[str, Any]:
    """Summarize token, reasoning, latency, and cost telemetry per model and per stage."""
    re_reas = re.compile(r"openai_reasoning_tokens=(\d+)")
    by_model: Dict[str, Dict[str, Any]] = defaultdict(
        lambda: {
            "total_calls": 0,
            "total_input_tokens": 0,
            "total_output_tokens": 0,
            "total_reasoning_tokens": 0,
            "total_est_cost_usd": 0.0,
            "refusals": 0,
            "silent_fallbacks": 0,
            "latencies_sec": [],
            "by_stage": {},
        }
    )

    stage_suffixes = ("stage1a", "stage2a", "stage2b", "stage3a", "stage6_binding")
    for sfx in stage_suffixes:
        for pf in sorted(pred_dir.glob(f"*_{sfx}.jsonl")):
            mtag = pf.name[: -len(f"_{sfx}.jsonl")]
            rows = [json.loads(ln) for ln in pf.read_text(encoding="utf-8").splitlines() if ln.strip()]
            if not rows:
                continue
            in_tok = sum(int(r.get("input_tokens", 0)) for r in rows)
            out_tok = sum(int(r.get("output_tokens", 0)) for r in rows)
            reas_tok = 0
            for r in rows:
                m = re_reas.search(str(r.get("thinking_text") or ""))
                if m:
                    reas_tok += int(m.group(1))
            cost = sum(float(r.get("est_cost_usd", 0.0)) for r in rows)
            lats = [float(r.get("latency_sec", 0.0)) for r in rows if float(r.get("latency_sec", 0.0)) > 0]
            refs = sum(1 for r in rows if r.get("refusal_detected"))
            fbs = sum(1 for r in rows if r.get("silent_fallback_detected"))
            stop_len = sum(1 for r in rows if str(r.get("stop_reason", "")) in ("length", "max_tokens"))

            rec = by_model[mtag]
            rec["total_calls"] += len(rows)
            rec["total_input_tokens"] += in_tok
            rec["total_output_tokens"] += out_tok
            rec["total_reasoning_tokens"] += reas_tok
            rec["total_est_cost_usd"] += cost
            rec["refusals"] += refs
            rec["silent_fallbacks"] += fbs
            rec["latencies_sec"].extend(lats)
            rec["by_stage"][sfx] = {
                "n_calls": len(rows),
                "input_tokens": in_tok,
                "output_tokens": out_tok,
                "reasoning_tokens": reas_tok,
                "mean_reasoning_tokens": round(reas_tok / max(1, len(rows)), 1),
                "est_cost_usd": round(cost, 4),
                "mean_latency_sec": round(float(np.mean(lats)) if lats else 0.0, 2),
                "truncated_length_stops": stop_len,
            }

    out: Dict[str, Any] = {}
    for mtag, d in sorted(by_model.items()):
        lats = d.pop("latencies_sec")
        d["total_est_cost_usd"] = round(d["total_est_cost_usd"], 4)
        d["mean_latency_sec"] = round(float(np.mean(lats)) if lats else 0.0, 2)
        out[mtag] = d
    return out


def plot_dashboard(summary: Dict[str, Any], out_png: Path) -> None:
    """Render 6-panel publication overview comparing live LLM(s) vs calibration/structural baselines."""
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.size": 9.5,
            "axes.titlesize": 10.8,
            "axes.titleweight": "bold",
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )
    fig, axes = plt.subplots(2, 3, figsize=(16.8, 9.6), dpi=200, facecolor="#FAFAFC")

    def _pretty(name: str) -> str:
        return (
            name.replace("_think4096", " (High)")
            .replace("_think16000", " (xHigh)")
            .replace("gpt-6.2-sol", "GPT-6.2 Sol")
            .replace("gpt-6.1-sol", "GPT-6.1 Sol")
            .replace("gpt-6.2-astra", "GPT-6.2 Astra")
            .replace("Oracle_TrueContacts", "Oracle Contacts")
            .replace("NoisyOracle_50pctPrec", "50%-Prec Contacts")
            .replace("Heuristic_1D_LocalOnly", "1D Chou-Fasman")
            .replace("Foldseek_3Di_Structure", "Foldseek 3Di")
            .replace("BLOSUM62_SmithWaterman", "BLOSUM62 SW")
            .replace("Baseline_AAComposition_Cosine", "20-AA Comp")
            .replace("CAFA_SequenceHomology_SW", "SW Homology")
            .replace("Baseline_3mer_kNN", "3-mer kNN")
            .replace("CAFA_NaivePrior", "CAFA Naive")
            .replace("Baseline_LocalWindowKD", "Local KD Window")
            .replace("Baseline_AA_LookupOnly", "1D AA Lookup")
            .replace("Oracle_3D_Interface_Contacts", "3D Crystal Contacts")
            .replace("Baseline_BoganThorn_1D", "Bogan-Thorn 1D")
        )

    def _llm_col(name: str) -> str:
        if "astra" in name:
            return "#E11D48"
        if "6.1" in name:
            return "#0284C7"
        return "#4F46E5"

    # --- Panel A: Test 1 Forward Folding TM-Score across Novelty Strata ---
    ax = axes[0, 0]
    ax.set_facecolor("#FFFFFF")
    t1 = {m["model"]: m for m in summary["test1_forward_folding"]}
    llm_keys_t1 = [k for k in t1 if k not in ("Oracle_TrueContacts", "NoisyOracle_50pctPrec", "Heuristic_1D_LocalOnly", "Null_Random")]
    strata_order = [
        ("pre_cutoff_control", "Pre-Cutoff\nKnown (15)"),
        ("post_cutoff_de_novo", "Post\nDe Novo (28)"),
        ("post_cutoff_homolog_gradient", "Post\nHomolog (12)"),
        ("post_cutoff_pdb_novel", "Post\nPDB-Novel (27)"),
        ("post_cutoff_struct_novel", "Post\nTrue Novel (18)"),
    ]
    plot_models_t1 = [("Heuristic_1D_LocalOnly", "#CBD5E1"), ("NoisyOracle_50pctPrec", "#94A3B8")] + [
        (lk, _llm_col(lk)) for lk in llm_keys_t1
    ]
    x = np.arange(len(strata_order))
    w = 0.78 / max(1, len(plot_models_t1))
    for idx, (mk, col) in enumerate(plot_models_t1):
        if mk not in t1:
            continue
        vals = [
            t1[mk]["splits"].get(sk, t1[mk]["splits"].get("pre_cutoff" if sk == "pre_cutoff_control" else "post_cutoff", {})).get("tm_score", 0.0)
            for sk, _ in strata_order
        ]
        off = (idx - (len(plot_models_t1) - 1) / 2.0) * w
        ov_tm = t1[mk]["overall"].get("tm_score", 0.0)
        bars = ax.bar(x + off, vals, width=w * 0.92, color=col, label=f"{_pretty(mk)} (All={ov_tm:.3f})")
        if mk in llm_keys_t1:
            for b, v in zip(bars, vals):
                ax.text(b.get_x() + b.get_width() / 2, v + 0.012, f"{v:.2f}", ha="center", va="bottom", fontsize=7.5, fontweight="bold", color="#312E81")
    ax.axhline(0.50, color="#0F172A", linestyle="--", linewidth=0.9, alpha=0.55, label="Correct Fold Threshold (TM=0.5)")
    ax.set_xticks(x)
    ax.set_xticklabels([lbl for _, lbl in strata_order], fontsize=8.2)
    ax.set_ylim(0, 0.88)
    ax.set_ylabel("Reconstructed 3D TM-Score")
    ax.set_title("1. Forward Folding: 3D TM-Score by Novelty")
    ax.legend(fontsize=7.6, frameon=True, loc="upper right")
    ax.grid(True, axis="y", alpha=0.2)

    # --- Panel B: Test 3 Remote Homology Recall@10%FPR across Rungs ---
    ax = axes[0, 1]
    ax.set_facecolor("#FFFFFF")
    t3 = summary["test3_remote_homology"]
    llm_keys_t3 = [k for k in t3 if k not in ("Foldseek_3Di_Structure", "BLOSUM62_SmithWaterman", "Baseline_AAComposition_Cosine", "Null_Random")]
    rungs = [("1_seq_easy", "Rung 1\n(Seq-Easy)"), ("2_foldseek_3di", "Rung 2\n(Foldseek 3Di)"), ("3_deep_tea_zone", "Rung 3\n(Deep TEA)")]
    plot_models_t3 = [("Foldseek_3Di_Structure", "#0D9488"), ("BLOSUM62_SmithWaterman", "#64748B")] + [
        (lk, _llm_col(lk)) for lk in llm_keys_t3
    ]
    x3 = np.arange(len(rungs))
    w3 = 0.76 / max(1, len(plot_models_t3))
    for idx, (mk, col) in enumerate(plot_models_t3):
        if mk not in t3:
            continue
        vals = [t3[mk]["by_rung"].get(rk, {}).get("recall_at_fpr10", 0.0) for rk, _ in rungs]
        off = (idx - (len(plot_models_t3) - 1) / 2.0) * w3
        auc = t3[mk].get("overall_auroc", 0.0)
        bars = ax.bar(x3 + off, vals, width=w3 * 0.92, color=col, label=f"{_pretty(mk)} (AUC={auc:.2f})")
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + 0.015, f"{v:.2f}", ha="center", va="bottom", fontsize=7.6, fontweight="bold")
    ax.set_xticks(x3)
    ax.set_xticklabels([lbl for _, lbl in rungs])
    ax.set_ylim(0, 1.15)
    ax.set_ylabel("Recall @ 10% FPR (vs. Hard Fold Decoys)")
    ax.set_title("3. Remote Homology Across Evolutionary Depth")
    ax.legend(fontsize=7.6, frameon=True, loc="upper right")
    ax.grid(True, axis="y", alpha=0.2)

    # --- Panel C: Test 4 Protein Function Prediction (CAFA IC-Weighted wF_max) ---
    ax = axes[0, 2]
    ax.set_facecolor("#FFFFFF")
    t4 = summary["test4_cafa_function"]
    llm_keys_t4 = [k for k in t4 if k not in ("Oracle_GroundTruth", "CAFA_NaivePrior", "CAFA_SequenceHomology_SW", "Baseline_3mer_kNN")]
    aspects = ["MF", "BP", "CC"]
    plot_models_t4 = [("CAFA_SequenceHomology_SW", "#94A3B8"), ("CAFA_NaivePrior", "#F59E0B")] + [
        (lk, _llm_col(lk)) for lk in llm_keys_t4
    ]
    x4 = np.arange(len(aspects))
    w4 = 0.76 / max(1, len(plot_models_t4))
    for idx, (mk, col) in enumerate(plot_models_t4):
        if mk not in t4:
            continue
        vals = [t4[mk][a]["wfmax"] for a in aspects]
        off = (idx - (len(plot_models_t4) - 1) / 2.0) * w4
        mw = t4[mk]["macro_wfmax"]
        bars = ax.bar(x4 + off, vals, width=w4 * 0.92, color=col, label=f"{_pretty(mk)} (Macro={mw:.2f})")
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + 0.015, f"{v:.2f}", ha="center", va="bottom", fontsize=7.6, fontweight="bold")
    ax.set_xticks(x4)
    ax.set_xticklabels(["Molecular Function\n(MF)", "Biological Process\n(BP)", "Cellular Component\n(CC)"])
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("IC-Weighted CAFA wF_max")
    ax.set_title("4. 2026 Experimental GO Holdouts (N=60)")
    ax.legend(fontsize=7.6, frameon=True, loc="upper right")
    ax.grid(True, axis="y", alpha=0.2)

    # --- Panel D: Test 5 Monomer Alanine Scan Ranking (NDCG@5 & Same-AA Win Rate) ---
    ax = axes[1, 0]
    ax.set_facecolor("#FFFFFF")
    t5 = {m["model"]: m for m in summary["test5_monomer_alanine_ranking"]}
    llm_keys_t5 = [k for k in t5 if k not in ("Oracle_Experimental", "Baseline_LocalWindowKD", "Baseline_AA_LookupOnly", "Null_Random")]
    order_t5 = ["Null_Random", "Baseline_AA_LookupOnly", "Baseline_LocalWindowKD"] + llm_keys_t5
    order_t5 = [k for k in order_t5 if k in t5]
    x5 = np.arange(len(order_t5))
    ndcg5 = [t5[k]["overall"]["ndcg_at_5"] for k in order_t5]
    win5 = [t5[k]["overall"]["same_aa_win_rate"] for k in order_t5]
    b1 = ax.bar(x5 - 0.18, ndcg5, width=0.34, color="#8B5CF6", label="NDCG@5 (Top-5 Hotspot Ranking)")
    b2 = ax.bar(x5 + 0.18, win5, width=0.34, color="#F59E0B", label="Same-AA Core vs. Surface Win Rate")
    for b, v in zip(b1, ndcg5):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.012, f"{v:.2f}", ha="center", va="bottom", fontsize=7.6, fontweight="bold")
    for b, v in zip(b2, win5):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.012, f"{v:.2f}", ha="center", va="bottom", fontsize=7.6, fontweight="bold")
    ax.axhline(0.50, color="#0F172A", linestyle="--", linewidth=0.9, alpha=0.55, label="1D Lookup Coin-Flip (0.50)")
    ax.set_xticks(x5)
    ax.set_xticklabels([_pretty(k) for k in order_t5], rotation=14, ha="right")
    ax.set_ylim(0.30, 1.08)
    ax.set_ylabel("Score")
    ax.set_title("5. Monomer Alanine Scan Ranking (N=52, 936 Mutants)")
    ax.legend(fontsize=7.6, frameon=True, loc="upper left")
    ax.grid(True, axis="y", alpha=0.2)

    # --- Panel E: Test 6 Interface Binding Alanine Ranking (NDCG@5 & Same-AA Win Rate) ---
    ax = axes[1, 1]
    ax.set_facecolor("#FFFFFF")
    t6 = {m["model"]: m for m in summary["test6_binding_alanine_ranking"]}
    llm_keys_t6 = [k for k in t6 if k not in ("Baseline_BoganThorn_1D", "Oracle_3D_Interface_Contacts", "Oracle_NoisyReplicate", "Null_Uniform")]
    order_t6 = ["Null_Uniform", "Baseline_BoganThorn_1D"] + llm_keys_t6 + ["Oracle_3D_Interface_Contacts"]
    order_t6 = [k for k in order_t6 if k in t6]
    x6 = np.arange(len(order_t6))
    ndcg6 = [t6[k]["overall"]["ndcg_at_5"] for k in order_t6]
    win6 = [t6[k]["overall"]["same_aa_win_rate"] for k in order_t6]
    b1 = ax.bar(x6 - 0.18, ndcg6, width=0.34, color="#4F46E5", label="NDCG@5 (Interface Hotspot Ranking)")
    b2 = ax.bar(x6 + 0.18, win6, width=0.34, color="#10B981", label="Same-AA Interface vs. Decoy Win Rate")
    for b, v in zip(b1, ndcg6):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.012, f"{v:.2f}", ha="center", va="bottom", fontsize=7.6, fontweight="bold")
    for b, v in zip(b2, win6):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.012, f"{v:.2f}", ha="center", va="bottom", fontsize=7.6, fontweight="bold")
    ax.axhline(0.50, color="#0F172A", linestyle="--", linewidth=0.9, alpha=0.55, label="1D Bogan-Thorn Coin-Flip (0.50)")
    ax.set_xticks(x6)
    ax.set_xticklabels([_pretty(k) for k in order_t6], rotation=14, ha="right")
    ax.set_ylim(0.30, 1.08)
    ax.set_ylabel("Score")
    ax.set_title("6. Interface Binding Alanine Scan (N=38 Complexes)")
    ax.legend(fontsize=7.6, frameon=True, loc="upper left")
    ax.grid(True, axis="y", alpha=0.2)

    # --- Panel F: Test 6 Memorization Audit (`Known PDB Pre-2026` vs `Unseen / Post-May-2026`) ---
    ax = axes[1, 2]
    ax.set_facecolor("#FFFFFF")
    mem_models = ["Baseline_BoganThorn_1D"] + llm_keys_t6 + ["Oracle_3D_Interface_Contacts"]
    mem_models = [k for k in mem_models if k in t6]
    xm = np.arange(len(mem_models))
    known_ndcg = [t6[k]["by_memorization_stratum"].get("known_pdb_pre2026", {}).get("ndcg_at_5", 0.0) for k in mem_models]
    unseen_ndcg = [t6[k]["by_memorization_stratum"].get("unseen_or_post2026", {}).get("ndcg_at_5", 0.0) for k in mem_models]
    bk1 = ax.bar(xm - 0.18, known_ndcg, width=0.34, color="#2563EB", label="Known Pre-2026 PDBs (n=31, NDCG@5)")
    bk2 = ax.bar(xm + 0.18, unseen_ndcg, width=0.34, color="#EA580C", label="Unseen / Post-May-2026 DMS (n=7, NDCG@5)")
    for b, v in zip(bk1, known_ndcg):
        if v > 0:
            ax.text(b.get_x() + b.get_width() / 2, v + 0.012, f"{v:.2f}", ha="center", va="bottom", fontsize=7.6, fontweight="bold")
    for b, v in zip(bk2, unseen_ndcg):
        if v > 0:
            ax.text(b.get_x() + b.get_width() / 2, v + 0.012, f"{v:.2f}", ha="center", va="bottom", fontsize=7.6, fontweight="bold")
    ax.set_xticks(xm)
    ax.set_xticklabels([_pretty(k) for k in mem_models], rotation=14, ha="right")
    ax.set_ylim(0.30, 1.08)
    ax.set_ylabel("Mean NDCG@5")
    ax.set_title("6b. Binding Memorization Audit: Known PDB vs. Unseen")
    ax.legend(fontsize=7.6, frameon=True, loc="upper left")
    ax.grid(True, axis="y", alpha=0.2)

    fig.suptitle(
        "Protein Structure Without Tools Local Suite (Tests 1, 3, 4, 5, 6 — 370 Tasks): Live Frontier Reasoning Evaluation",
        fontsize=14,
        fontweight="bold",
        y=0.985,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.955))
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=200)
    plt.close(fig)


def _norm_b6_dict(d: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "n": d.get("n") or d.get("n_targets", 0),
        "spearman_rho": d.get("spearman_rho", d.get("mean_spearman_rho", 0.0)),
        "ndcg_at_5": d.get("ndcg_at_5", d.get("mean_ndcg_at_5", 0.0)),
        "top_k_precision": d.get("top_k_precision", d.get("mean_top_k_precision", 0.0)),
        "same_aa_win_rate": d.get("same_aa_win_rate", d.get("mean_same_aa_win_rate", 0.5)),
        "spearman_rho_aa_demeaned": d.get("spearman_rho_aa_demeaned", d.get("mean_spearman_rho_aa_demeaned", 0.0)),
        "binding_hotspot_auroc": d.get("binding_hotspot_auroc", d.get("mean_binding_hotspot_auroc", 0.5)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run all 5 local CPU graders (Tests 1, 3, 4, 5, 6) and build summary.")
    parser.add_argument("--restarts", type=int, default=3, help="L-BFGS restarts for Test 1 3D reconstruction")
    parser.add_argument("--combine-only", action="store_true", help="Combine already-graded JSON reports & plot dashboard")
    args = parser.parse_args()

    py = sys.executable
    scripts = REPO_ROOT / "scripts"
    rdir = REPO_ROOT / "outputs" / "results"
    pdir = REPO_ROOT / "outputs" / "predictions"
    fdir = REPO_ROOT / "outputs" / "figures"

    if not args.combine_only:
        s1_cmd = [py, str(scripts / "evaluate_stage1.py"), "--restarts", str(args.restarts)]
        if not (rdir / "stage1_evaluation_report.json").exists():
            s1_cmd.append("--run-calibration")
        run_step("Test 1 (Forward Folding)", s1_cmd)
        run_step("Tests 3 & 4 (Remote Homology + CAFA Function)", [py, str(scripts / "build_and_evaluate_stage2.py"), "--eval-only"])
        run_step("Test 5 (Monomer Alanine Scan Ranking)", [py, str(scripts / "evaluate_alanine_scan.py"), "--run-calibration"])
        run_step("Test 6 (Interface Binding Alanine Scan Ranking)", [py, str(scripts / "evaluate_binding_alanine.py")])

    s1 = json.loads((rdir / "stage1_evaluation_report.json").read_text(encoding="utf-8"))["stage1a_forward_folding"]
    s2 = json.loads((rdir / "stage2_evaluation_report.json").read_text(encoding="utf-8"))
    s5 = json.loads((rdir / "stage3a_alanine_report.json").read_text(encoding="utf-8"))["stage3a_alanine_scan"]
    s6_live = json.loads((rdir / "live_stage6_binding_report.json").read_text(encoding="utf-8"))["test6_binding_alanine_ranking"]

    # Merge Oracle & Null calibration baselines from `binding_alanine_evaluation_report.json` into Test 6 list
    s6_base_path = rdir / "binding_alanine_evaluation_report.json"
    s6_merged: List[Dict[str, Any]] = []
    seen_s6 = {m["model"] for m in s6_live}
    if s6_base_path.exists():
        b6 = json.loads(s6_base_path.read_text(encoding="utf-8")).get("models", {})
        for bname in ("Oracle_NoisyReplicate", "Oracle_3D_Interface_Contacts", "Null_Uniform"):
            if bname in b6 and bname not in seen_s6:
                raw_m = b6[bname]
                s6_merged.append(
                    {
                        "model": bname,
                        "overall": _norm_b6_dict(raw_m),
                        "by_memorization_stratum": {
                            sk: _norm_b6_dict(sv) for sk, sv in raw_m.get("by_memorization_stratum", {}).items()
                        },
                        "by_category": {
                            ck: _norm_b6_dict(cv) for ck, cv in raw_m.get("by_category", {}).items()
                        },
                    }
                )
    s6_merged.extend(s6_live)

    telemetry = collect_api_telemetry(pdir)
    unified = {
        "telemetry": telemetry,
        "test1_forward_folding": s1,
        "test3_remote_homology": s2["stage2a_remote_homology"],
        "test4_cafa_function": s2["stage2b_cafa_function"],
        "test5_monomer_alanine_ranking": s5,
        "test6_binding_alanine_ranking": s6_merged,
    }

    out_json = rdir / "local_13456_unified_summary.json"
    out_json.write_text(json.dumps(unified, indent=2), encoding="utf-8")

    dash_png = fdir / "local_13456_live_dashboard.png"
    plot_dashboard(unified, dash_png)
    print(f"\nSaved unified local summary -> {out_json}")
    print(f"Saved 6-panel dashboard figure -> {dash_png}")


if __name__ == "__main__":
    main()
