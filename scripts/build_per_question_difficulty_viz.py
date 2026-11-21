#!/usr/bin/env python3
"""Build per-question × difficulty × split representations across all 370 evaluated tasks (Tests 1,3,4,5,6).

Outputs:
  1. `outputs/results/local_13456_per_question_records.json`: All 370 questions with score, difficulty tier,
     continuous difficulty metric, temporal/memorization split, and reasoning token count.
  2. `outputs/figures/gpt62_sol_per_question_difficulty.png`: 6-panel publication strip + median + continuous
     difficulty plot showing every single question dot.
  3. Artifact HTML interactive beeswarm/strip & continuous scatter widget for inline chat + side-pane inspection.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
import re
import sys
from typing import Any, Dict, List

import matplotlib.pyplot as plt
import numpy as np

REPO = Path("/Users/taluguloglu/.gemini/antigravity/scratch/fable-protein-bench")
if str(REPO / "src") not in sys.path:
    sys.path.insert(0, str(REPO / "src"))

from fable_bench.stage2a_homology import parse_remote_homology_response
from fable_bench.stage2b_cafa import (
    CAFATarget,
    compute_cafa_fmax,
    compute_term_information_content,
    parse_cafa_response,
    parse_go_obo,
)

MODEL_TAG = "gpt-6.2-sol_think4096"
RE_REAS = re.compile(r"openai_reasoning_tokens=(\d+)")


def _load_pred_meta(jsonl_path: Path) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    for ln in jsonl_path.read_text(encoding="utf-8").splitlines():
        if not ln.strip():
            continue
        obj = json.loads(ln)
        tid = str(obj.get("target_id") or obj.get("pair_id") or "")
        m = RE_REAS.search(str(obj.get("thinking_text") or ""))
        obj["reasoning_tokens"] = int(m.group(1)) if m else 0
        out[tid] = obj
    return out


def compile_370_question_records() -> List[Dict[str, Any]]:
    rdir = REPO / "outputs" / "results"
    pdir = REPO / "outputs" / "predictions"
    records: List[Dict[str, Any]] = []

    # =========================================================================
    # TEST 1: Forward Folding (N = 100)
    # =========================================================================
    m1_meta = {
        t["target_id"]: t
        for t in json.loads((REPO / "data/stage1_benchmark/manifest.json").read_text(encoding="utf-8"))["targets"]
    }
    s1_rep = json.loads((rdir / "stage1_evaluation_report.json").read_text(encoding="utf-8"))["stage1a_forward_folding"]
    s1_sol = next(x for x in s1_rep if x["model"] == MODEL_TAG)
    pred1 = _load_pred_meta(pdir / f"{MODEL_TAG}_stage1a.jsonl")

    t1_tier_map = {
        "pre_cutoff_control": (1, "1. Easy: Pre-Cutoff Known"),
        "post_cutoff_de_novo": (2, "2. Med-Easy: Post De Novo"),
        "post_cutoff_homolog_gradient": (3, "3. Medium: Post Homolog"),
        "post_cutoff_pdb_novel": (4, "4. Hard: Post PDB-Novel"),
        "post_cutoff_struct_novel": (5, "5. Hardest: Post True Novel"),
    }
    for g in s1_sol["per_target"]:
        tid = g["target_id"]
        mt = m1_meta[tid]
        st = mt.get("stratum", "post_cutoff_pdb_novel")
        d_ord, d_lbl = t1_tier_map.get(st, (4, st))
        pdb_tm = float(mt.get("pdb_precutoff_tm_bilateral_max", 0.5))
        split = "Pre-Cutoff Known" if mt["temporal_split"] == "pre_cutoff" else "Post-Cutoff (2026)"
        pm = pred1.get(tid, {})
        records.append(
            {
                "test": 1,
                "test_label": "1. Forward Folding",
                "id": tid,
                "name": f"PDB {mt['pdb_id']}:{mt['chain_id']} ({mt.get('description','')[:34]})",
                "split": split,
                "stratum": st,
                "diff_order": d_ord,
                "diff_tier": d_lbl,
                "cont_x": round(1.0 - min(1.0, pdb_tm), 4),  # 0 = identical in PDB, 1 = completely novel fold
                "cont_x_label": "Structural Novelty (1 − Max Pre-Cutoff PDB TM)",
                "score": round(float(g["tm_score"]), 4),
                "score_label": "3D TM-Score",
                "secondary_score": round(float(g["p_at_l_sep24"]), 4),
                "secondary_label": "Long-Range P@L(|i-j|≥24)",
                "q3_acc": round(float(g["q3_accuracy"]), 4),
                "length": int(mt["length"]),
                "reasoning_tokens": pm.get("reasoning_tokens", 0),
                "abstained": False,
            }
        )

    # =========================================================================
    # TEST 3: Remote Homology Detection (N = 120)
    # =========================================================================
    m3_pairs = json.loads((REPO / "data/stage2a_homology/manifest.json").read_text(encoding="utf-8"))["pairs"]
    pred3 = _load_pred_meta(pdir / f"{MODEL_TAG}_stage2a.jsonl")

    # Compute exact 10% FPR threshold on Rung 4 negative decoys
    neg_probs = []
    parsed_p3: Dict[str, float] = {}
    for p in m3_pairs:
        pid = p["pair_id"]
        pr = parse_remote_homology_response(pred3.get(pid, {}).get("text", ""))
        parsed_p3[pid] = float(pr.prob_homolog)
        if p["label_homolog"] == 0:
            neg_probs.append(float(pr.prob_homolog))
    neg_sorted = sorted(neg_probs, reverse=True)
    fpr10_thr = neg_sorted[max(0, int(math.floor(0.10 * len(neg_sorted))) - 1)] if neg_sorted else 0.5

    t3_tier_map = {
        "1_seq_easy": (1, "1. Easy: Rung 1 Seq-Easy"),
        "2_foldseek_3di": (2, "2. Med: Rung 2 Foldseek-3Di"),
        "3_deep_tea_zone": (3, "3. Hardest: Rung 3 Deep TEA"),
        "4_hard_negative_decoy": (4, "4. Control: Hard Fold Decoys"),
    }
    for p in m3_pairs:
        pid = p["pair_id"]
        prob = parsed_p3[pid]
        rung = p["rung"]
        d_ord, d_lbl = t3_tier_map[rung]
        is_pos = int(p["label_homolog"]) == 1
        split = "True Homolog (Superfamily)" if is_pos else "Class-Matched Decoy (Neg)"
        # Continuous difficulty for homologs: lower SeqID = harder (1 - SeqID/45)
        seqid = float(p["seq_identity_pct"])
        cont_d = round(max(0.0, min(1.0, 1.0 - seqid / 40.0)), 4)
        pm = pred3.get(pid, {})
        records.append(
            {
                "test": 3,
                "test_label": "3. Remote Homology",
                "id": pid,
                "name": f"{p['domain_a']} × {p['domain_b']} ({p['scop_a']} vs {p['scop_b']})",
                "split": split,
                "stratum": rung,
                "diff_order": d_ord,
                "diff_tier": d_lbl,
                "cont_x": cont_d,
                "cont_x_label": "Divergence Difficulty (1 − SeqID% / 40)",
                "score": round(prob, 4),
                "score_label": f"Predicted Homology Prob (10% FPR thr={fpr10_thr:.2f})",
                "secondary_score": round(1.0 - abs((1.0 if is_pos else 0.0) - prob), 4),
                "secondary_label": "Calibration Accuracy (1 − |y − p|)",
                "seq_identity_pct": seqid,
                "fpr10_threshold": round(fpr10_thr, 4),
                "length": len(p["seq_a"]) + len(p["seq_b"]),
                "reasoning_tokens": pm.get("reasoning_tokens", 0),
                "abstained": False,
            }
        )

    # =========================================================================
    # TEST 4: Protein Function Prediction / CAFA (N = 60)
    # =========================================================================
    s2b_dir = REPO / "data/stage2b_cafa"
    dag = parse_go_obo(str(s2b_dir / "go_subdag.obo"))
    m4 = json.loads((s2b_dir / "manifest.json").read_text(encoding="utf-8"))["targets"]
    ref_bank = json.loads((s2b_dir / "reference_swissprot_go.json").read_text(encoding="utf-8"))
    ref_prop = [dag.propagate_set(r["go_terms"]) for r in ref_bank]
    ic_map = compute_term_information_content(dag, ref_prop)
    ref_kmers = [{r["sequence"][i : i + 3] for i in range(max(0, len(r["sequence"]) - 2))} for r in ref_bank]

    pred4 = _load_pred_meta(pdir / f"{MODEL_TAG}_stage2b.jsonl")

    # Compute max 3-mer cosine similarity to pre-2025 Swiss-Prot reference bank per target as homology difficulty!
    max_sims = []
    t4_temp = []
    for t in m4:
        qs = t["sequence"]
        qks = {qs[i : i + 3] for i in range(max(0, len(qs) - 2))}
        qd = math.sqrt(max(1, len(qks)))
        msim = max(len(qks & rk) / (qd * math.sqrt(max(1, len(rk)))) for rk in ref_kmers)
        max_sims.append(msim)
        t_obj = CAFATarget(
            target_id=t["target_id"],
            sequence=t["sequence"],
            length=t["length"],
            true_terms_mf=t["true_terms_mf"],
            true_terms_bp=t["true_terms_bp"],
            true_terms_cc=t["true_terms_cc"],
            knowledge_type=t.get("knowledge_type", "NK"),
        )
        pmap, _ = parse_cafa_response(pred4.get(t["target_id"], {}).get("text", ""))
        # Per-protein IC-weighted wF_max across present aspects
        asp_scores = []
        for asp, terms in (("MF", t["true_terms_mf"]), ("BP", t["true_terms_bp"]), ("CC", t["true_terms_cc"])):
            if terms:
                res_a = compute_cafa_fmax(dag, [t_obj], {t["target_id"]: pmap}, aspect=asp, ic_map=ic_map)
                asp_scores.append(float(res_a["wfmax"]))
        wf_i = float(np.mean(asp_scores)) if asp_scores else 0.0
        t4_temp.append((t, msim, wf_i))

    med_sim = float(np.median(max_sims))
    for t, msim, wf_i in t4_temp:
        tid = t["target_id"]
        is_post = t.get("temporal_tier") == "post_may_2026"
        is_hard_hom = msim < med_sim
        if not is_hard_hom and not is_post:
            d_ord, d_lbl = 1, "1. Easy: Homolog + Early-2026"
        elif not is_hard_hom and is_post:
            d_ord, d_lbl = 2, "2. Med-Easy: Homolog + Post-May26"
        elif is_hard_hom and not is_post:
            d_ord, d_lbl = 3, "3. Med-Hard: Low-Hom + Early-2026"
        else:
            d_ord, d_lbl = 4, "4. Hardest: Low-Hom + Post-May26"

        split = "Post-May-2026 Holdout" if is_post else "Early-2026 Holdout"
        pm = pred4.get(tid, {})
        records.append(
            {
                "test": 4,
                "test_label": "4. Function (CAFA)",
                "id": tid,
                "name": f"UniProt {t['uniprot_accession']} ({t.get('organism','')[:28]})",
                "split": split,
                "stratum": t.get("temporal_tier", ""),
                "diff_order": d_ord,
                "diff_tier": d_lbl,
                "cont_x": round(max(0.0, min(1.0, 1.0 - msim / 0.45)), 4),
                "cont_x_label": "Sequence Novelty vs Swiss-Prot (1 − 3mer Sim / 0.45)",
                "score": round(wf_i, 4),
                "score_label": "Per-Protein IC-Weighted wF_max",
                "secondary_score": round(msim, 4),
                "secondary_label": "Max Swiss-Prot 3-mer Cosine",
                "length": int(t["length"]),
                "reasoning_tokens": pm.get("reasoning_tokens", 0),
                "abstained": False,
            }
        )

    # =========================================================================
    # TEST 5: Monomer Alanine Scan Ranking (N = 52)
    # =========================================================================
    m5_meta = {
        t["target_id"]: t
        for t in json.loads((REPO / "data/alanine_scan_benchmark/manifest.json").read_text(encoding="utf-8"))["targets"]
    }
    s5_rep = json.loads((rdir / "stage3a_alanine_report.json").read_text(encoding="utf-8"))["stage3a_alanine_scan"]
    s5_sol = next(x for x in s5_rep if x["model"] == MODEL_TAG)
    pred5 = _load_pred_meta(pdir / f"{MODEL_TAG}_stage3a.jsonl")

    for g in s5_sol["per_target"]:
        tid = g["target_id"]
        mt = m5_meta[tid]
        cat = mt["category"]
        sname = mt.get("source_name", "")
        if cat == "de_novo_ddg":
            d_ord, d_lbl = 1, "1. Easy: De Novo Mini-Fold ΔΔG"
            split = "Pre-Cutoff MegaScale ΔΔG"
        elif cat == "natural_ddg":
            d_ord, d_lbl = 2, "2. Med: Natural Domain ΔΔG"
            split = "Pre-Cutoff MegaScale ΔΔG"
        elif cat == "dual_stability_function":
            d_ord, d_lbl = 3, "3. Med-Hard: Paired Dual-Assay"
            split = "Post-May-2026 MaveDB"
        else:
            d_ord, d_lbl = 4, "4. Hardest: Post-May-2026 MaveDB"
            split = "Post-May-2026 MaveDB"

        pm = pred5.get(tid, {})
        # Longer natural proteins with larger sequence context are harder than 45-AA mini-bundles
        cont_d = round(min(1.0, max(0.0, (int(mt["length"]) - 40) / 300.0)), 4)
        records.append(
            {
                "test": 5,
                "test_label": "5. Monomer Ala Ranking",
                "id": tid,
                "name": f"{sname[:40]}",
                "split": split,
                "stratum": cat,
                "diff_order": d_ord,
                "diff_tier": d_lbl,
                "cont_x": cont_d,
                "cont_x_label": "Chain Complexity ((Length − 40) / 300)",
                "score": round(float(g["ndcg_at_5"]), 4),
                "score_label": "NDCG@5 (18-Candidate Panel)",
                "secondary_score": round(float(g["same_aa_win_rate"]), 4),
                "secondary_label": "Same-AA Decoy Win Rate",
                "spearman_rho": round(float(g["spearman_rho"]), 4),
                "length": int(mt["length"]),
                "reasoning_tokens": pm.get("reasoning_tokens", 0),
                "abstained": False,
            }
        )

    # =========================================================================
    # TEST 6: Interface Binding Alanine Scan Ranking (N = 38)
    # =========================================================================
    m6_meta = {
        t["target_id"]: t
        for t in json.loads((REPO / "data/binding_alanine_benchmark/manifest.json").read_text(encoding="utf-8"))["targets"]
    }
    s6_rep = json.loads((rdir / "live_stage6_binding_report.json").read_text(encoding="utf-8"))["test6_binding_alanine_ranking"]
    s6_sol = next(x for x in s6_rep if x["model"] == MODEL_TAG)
    pred6 = _load_pred_meta(pdir / f"{MODEL_TAG}_stage6_binding.jsonl")

    for g in s6_sol["per_target"]:
        tid = g["target_id"]
        mt = m6_meta[tid]
        mem = mt.get("memorization_stratum", "known_pdb_pre2026")
        cat = mt["category"]
        sname = mt.get("source_name", tid)
        abstained = not bool(g["parse_ok"])

        if mem == "known_pdb_pre2026" and cat == "skempi_biophysical_ddg":
            d_ord, d_lbl = 1, "1. Known PDB: SKEMPI SPR/ITC"
            split = "Known Pre-2026 PDB (n=31)"
        elif mem == "known_pdb_pre2026":
            d_ord, d_lbl = 2, "2. Known PDB: Display Kd/DMS"
            split = "Known Pre-2026 PDB (n=31)"
        elif any(k in sname for k in ("KRAS", "GRB2", "RAF1", "MLH1")):
            d_ord, d_lbl = 3, "3. Post-2026: Known Pathway PPI"
            split = "Unseen / Post-2026 (n=7)"
        else:
            d_ord, d_lbl = 4, "4. Hardest: Unseen scFv/Peptide"
            split = "Unseen / Post-2026 (n=7)"

        tot_len = int(mt["scanned_length"]) + int(mt["partner_length"])
        cont_d = round(min(1.0, max(0.0, (0.55 if mem != "known_pdb_pre2026" else 0.0) + tot_len / 1800.0)), 4)
        pm = pred6.get(tid, {})
        records.append(
            {
                "test": 6,
                "test_label": "6. Binding Ala Ranking",
                "id": tid,
                "name": sname[:44],
                "split": split,
                "stratum": mem,
                "diff_order": d_ord,
                "diff_tier": d_lbl,
                "cont_x": cont_d,
                "cont_x_label": "Memorization & Complex Size Difficulty",
                "score": round(float(g["ndcg_at_5"]), 4),
                "score_label": "NDCG@5 (Interface Hotspot Ranking)",
                "secondary_score": round(float(g["same_aa_win_rate"]), 4),
                "secondary_label": "Same-AA Interface Win Rate",
                "spearman_rho": round(float(g["spearman_rho"]), 4),
                "length": tot_len,
                "reasoning_tokens": pm.get("reasoning_tokens", 0),
                "abstained": abstained,
            }
        )

    return records


def render_6panel_per_question_png(records: List[Dict[str, Any]], out_png: Path) -> None:
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
    fig, axes = plt.subplots(2, 3, figsize=(17.5, 10.2), dpi=200, facecolor="#FAFAFC")
    rng = np.random.default_rng(20261121)

    split_palette = {
        "Pre-Cutoff Known": "#2563EB",
        "Post-Cutoff (2026)": "#EA580C",
        "True Homolog (Superfamily)": "#4F46E5",
        "Class-Matched Decoy (Neg)": "#DC2626",
        "Early-2026 Holdout": "#0284C7",
        "Post-May-2026 Holdout": "#D97706",
        "Pre-Cutoff MegaScale ΔΔG": "#2563EB",
        "Post-May-2026 MaveDB": "#EA580C",
        "Known Pre-2026 PDB (n=31)": "#2563EB",
        "Unseen / Post-2026 (n=7)": "#EA580C",
    }

    panel_specs = [
        (axes[0, 0], 1, "1. Forward Folding (N=100): Per-Protein TM by Novelty", "3D TM-Score", 0.50, "Fold Threshold (0.50)"),
        (axes[0, 1], 3, "3. Remote Homology (N=120): Per-Pair Prob by Rung", "Predicted P(Homolog)", None, "10% FPR Threshold"),
        (axes[0, 2], 4, "4. Function CAFA (N=60): Per-Protein wF_max by Split", "IC-Weighted wF_max", 0.251, "SW Homology Mean (0.25)"),
        (axes[1, 0], 5, "5. Monomer Ala Scan (N=52): Per-Protein NDCG@5", "NDCG@5", 0.615, "1D AA Lookup Mean (0.615)"),
        (axes[1, 1], 6, "6. Interface Binding Ala (N=38): Per-Complex NDCG@5", "NDCG@5", 0.526, "1D Bogan-Thorn (0.526)"),
    ]

    for ax, tnum, title, ylabel, ref_y, ref_lbl in panel_specs:
        ax.set_facecolor("#FFFFFF")
        sub = [r for r in records if r["test"] == tnum]
        tiers = sorted({(r["diff_order"], r["diff_tier"]) for r in sub})
        if tnum == 3 and sub:
            ref_y = sub[0].get("fpr10_threshold", 0.5)
            ref_lbl = f"10% Decoy FPR Line ({ref_y:.2f})"

        seen_splits = set()
        for idx_t, (d_ord, d_lbl) in enumerate(tiers):
            tsub = [r for r in sub if r["diff_order"] == d_ord]
            ys = np.array([r["score"] for r in tsub], dtype=np.float64)
            # Draw horizontal mean + IQRwhisker box
            q25, med, q75 = np.percentile(ys, [25, 50, 75])
            mean_v = float(np.mean(ys))
            ax.fill_between([idx_t - 0.26, idx_t + 0.26], q25, q75, color="#E2E8F0", alpha=0.65, zorder=1)
            ax.plot([idx_t - 0.28, idx_t + 0.28], [mean_v, mean_v], color="#0F172A", linewidth=2.1, zorder=4)
            top_lbl_y = 0.75 if tnum == 1 else 1.055
            ax.text(idx_t, top_lbl_y, f"μ={mean_v:.2f}", ha="center", va="bottom", fontsize=7.8, fontweight="bold", color="#1E293B")

            for r in tsub:
                jit = rng.uniform(-0.19, 0.19)
                sp = r["split"]
                col = split_palette.get(sp, "#4F46E5")
                lbl_s = sp if sp not in seen_splits else None
                seen_splits.add(sp)
                if r.get("abstained"):
                    ax.scatter(idx_t + jit, r["score"], facecolors="none", edgecolors=col, marker="o", s=46, linewidths=1.5, zorder=5, label=lbl_s)
                else:
                    ax.scatter(idx_t + jit, r["score"], c=col, s=36, alpha=0.82, edgecolors="white", linewidth=0.45, zorder=5, label=lbl_s)

        if ref_y is not None:
            ax.axhline(ref_y, color="#64748B", linestyle="--", linewidth=1.0, label=ref_lbl)

        short_ticks = [
            lbl.split(": ", 1)[-1].replace("Post ", "").replace("Rung ", "R") + f"\n(n={sum(1 for r in sub if r['diff_order']==do)})"
            for do, lbl in tiers
        ]
        ax.set_xticks(range(len(tiers)))
        ax.set_xticklabels(short_ticks, fontsize=7.8, rotation=12, ha="right")
        ax.set_ylim(-0.04, 1.24 if tnum == 4 else (0.90 if tnum == 1 else 1.15))
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend(loc="upper right" if tnum in (1, 4) else "lower left", fontsize=7.3, frameon=True, framealpha=0.92)
        ax.grid(True, axis="y", alpha=0.22)

    # Panel 6: All 370 Questions — Score vs. Normalized Difficulty Tier (1=Easiest -> 4/5=Hardest), Split Pre vs Post/Unseen
    ax6 = axes[1, 2]
    ax6.set_facecolor("#FFFFFF")
    # Exclude negative decoys in Test 3 so higher score = better on all 340 positive/ranking questions
    pos_recs = [r for r in records if not (r["test"] == 3 and r["stratum"] == "4_hard_negative_decoy")]
    is_novel_split = lambda r: any(k in r["split"] for k in ("Post-", "Unseen")) or r["stratum"] in ("2_foldseek_3di", "3_deep_tea_zone")

    pre_pts = [r for r in pos_recs if not is_novel_split(r)]
    nov_pts = [r for r in pos_recs if is_novel_split(r)]

    ax6.scatter(
        [r["cont_x"] + rng.normal(0, 0.012) for r in pre_pts],
        [r["score"] for r in pre_pts],
        c="#2563EB",
        s=28,
        alpha=0.65,
        label=f"Known / Pre-Cutoff / Easy-Homolog (n={len(pre_pts)})",
    )
    ax6.scatter(
        [r["cont_x"] + rng.normal(0, 0.012) for r in nov_pts],
        [r["score"] for r in nov_pts],
        c="#EA580C",
        s=28,
        alpha=0.68,
        label=f"Post-Cutoff / Unseen / Twilight (n={len(nov_pts)})",
    )

    # Binned trend curve across continuous difficulty [0, 1]
    bins = np.linspace(0.0, 1.0, 6)
    bx, by = [], []
    for b0, b1 in zip(bins[:-1], bins[1:]):
        vals = [r["score"] for r in pos_recs if b0 <= r["cont_x"] <= b1]
        if vals:
            bx.append(0.5 * (b0 + b1))
            by.append(float(np.mean(vals)))
    ax6.plot(bx, by, color="#0F172A", linewidth=2.4, marker="D", markersize=5.5, label="Mean Score vs. Difficulty Trend")
    ax6.set_xlim(-0.04, 1.04)
    ax6.set_ylim(-0.04, 1.08)
    ax6.set_xlabel("Normalized Question Difficulty / Novelty (0 = Easiest / Known → 1 = Hardest / Novel)")
    ax6.set_ylabel("Per-Question Primary Score")
    ax6.set_title("All Positive/Ranking Tasks (N=340): Continuous Difficulty Decay")
    ax6.legend(loc="lower left", fontsize=7.5, frameon=True)
    ax6.grid(True, alpha=0.22)

    fig.suptitle(
        "GPT-6.2 Sol (High Reasoning) — Per-Question Score Distribution Across Difficulty Tiers & Memorization Splits (N = 370)",
        fontsize=13.5,
        fontweight="bold",
        y=0.988,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=200)
    plt.close(fig)


WIDGET_HTML_TEMPLATE = r"""<!DOCTYPE html>
<html>
<head>
  <meta charset="UTF-8" />
  <script src="https://www.gstatic.com/antigravity/web/dev/tailwindcss.min.js"></script>
</head>
<body class="bg-transparent text-[var(--foreground)] antialiased p-3">
  <div class="bg-[var(--card)] text-[var(--foreground)] border border-[var(--border)] rounded-xl p-4 shadow-sm space-y-3">
    <div class="flex flex-wrap items-center justify-between gap-2">
      <div>
        <h2 class="text-[var(--foreground)] font-bold text-base leading-tight">Per-Question Score × Difficulty &amp; Split Explorer (370 Questions)</h2>
        <p id="sub-lbl" class="text-[var(--muted-foreground)] text-xs mt-0.5"></p>
      </div>
      <div class="flex flex-wrap items-center gap-1.5">
        <div id="test-tabs" class="flex flex-wrap gap-1"></div>
        <button id="mode-toggle" class="px-2.5 py-1 rounded-lg border border-[var(--border)] bg-[var(--secondary)] text-xs font-semibold hover:opacity-80 transition">
          Axis: Difficulty Tiers
        </button>
      </div>
    </div>

    <div id="legend-row" class="flex flex-wrap items-center justify-between gap-2 text-xs"></div>

    <div class="relative">
      <canvas id="plot-canvas" width="760" height="245" class="w-full h-[245px] rounded-lg bg-[var(--background)] border border-[var(--border)] cursor-crosshair"></canvas>
    </div>

    <div id="hover-card" class="rounded-lg bg-[var(--secondary)]/60 border border-[var(--border)] px-3 py-2 text-xs flex flex-wrap items-center justify-between gap-2">
      <span class="text-[var(--muted-foreground)]">Hover or click any of the dots above to inspect that individual question's ID, split, difficulty, reasoning tokens, and score.</span>
    </div>
  </div>

  <script>
    const RECORDS = __RECORDS_JSON__;

    const SPLIT_COLORS = {
      'Pre-Cutoff Known': '#3b82f6',
      'Post-Cutoff (2026)': '#f97316',
      'True Homolog (Superfamily)': '#6366f1',
      'Class-Matched Decoy (Neg)': '#ef4444',
      'Early-2026 Holdout': '#0ea5e9',
      'Post-May-2026 Holdout': '#f59e0b',
      'Pre-Cutoff MegaScale ΔΔG': '#3b82f6',
      'Post-May-2026 MaveDB': '#f97316',
      'Known Pre-2026 PDB (n=31)': '#3b82f6',
      'Unseen / Post-2026 (n=7)': '#f97316'
    };

    const TEST_INFO = {
      1: { tab: 'T1 Folding (100)', refY: 0.50, refLbl: 'Fold Threshold (0.50)', maxY: 0.76 },
      3: { tab: 'T3 Homology (120)', refY: null, refLbl: '10% FPR Line', maxY: 1.05 },
      4: { tab: 'T4 CAFA (60)', refY: 0.251, refLbl: 'SW Homology (0.25)', maxY: 1.05 },
      5: { tab: 'T5 Mono Ala (52)', refY: 0.615, refLbl: '1D Lookup (0.615)', maxY: 1.05 },
      6: { tab: 'T6 Bind Ala (38)', refY: 0.526, refLbl: 'Bogan-Thorn 1D (0.526)', maxY: 1.05 }
    };

    let activeTest = 1;
    let continuousMode = false;
    let plottedPts = [];
    let pinnedIdx = -1;

    function seededJitter(str) {
      let h = 2166136261;
      for (let i = 0; i < str.length; i++) {
        h ^= str.charCodeAt(i);
        h = Math.imul(h, 16777619);
      }
      return ((h >>> 0) / 4294967295) - 0.5;
    }

    function draw() {
      const sub = RECORDS.filter(r => r.test === activeTest);
      const info = TEST_INFO[activeTest];
      const tabsEl = document.getElementById('test-tabs');
      tabsEl.innerHTML = Object.entries(TEST_INFO).map(([k, v]) => {
        const act = Number(k) === activeTest
          ? 'bg-[var(--primary)] text-[var(--primary-foreground)] border-transparent'
          : 'bg-[var(--background)] text-[var(--foreground)] border-[var(--border)] hover:opacity-80';
        return `<button data-t="${k}" class="px-2.5 py-1 rounded-lg border text-xs font-semibold transition ${act}">${v.tab}</button>`;
      }).join('');
      tabsEl.querySelectorAll('button').forEach(b => {
        b.addEventListener('click', () => { activeTest = Number(b.getAttribute('data-t')); pinnedIdx = -1; draw(); });
      });

      document.getElementById('mode-toggle').textContent = continuousMode ? 'Axis: Continuous Difficulty' : 'Axis: Difficulty Tiers';
      document.getElementById('sub-lbl').textContent = continuousMode
        ? `${sub[0].score_label} vs. ${sub[0].cont_x_label}`
        : `Each dot = 1 question (${sub[0].score_label}), grouped by difficulty tier & colored by split (horizontal bar = tier mean)`;

      const splits = [...new Set(sub.map(r => r.split))];
      const legendHtml = splits.map(s => {
        const c = SPLIT_COLORS[s] || '#6366f1';
        const cnt = sub.filter(r => r.split === s).length;
        return `<span class="inline-flex items-center gap-1.5"><span class="w-2.5 h-2.5 rounded-full inline-block" style="background:${c}"></span><span>${s} (${cnt})</span></span>`;
      }).join('<span class="mx-2">•</span>');
      const abstNote = sub.some(r => r.abstained) ? '<span class="text-[var(--muted-foreground)]">Hollow ring = Abstained</span>' : '';
      document.getElementById('legend-row').innerHTML = `<div class="flex flex-wrap items-center gap-2">${legendHtml}</div><div>${abstNote}</div>`;

      const canvas = document.getElementById('plot-canvas');
      const dpr = window.devicePixelRatio || 1;
      const cssW = canvas.clientWidth || 760;
      const cssH = 245;
      canvas.width = cssW * dpr;
      canvas.height = cssH * dpr;
      const ctx = canvas.getContext('2d');
      ctx.scale(dpr, dpr);
      ctx.clearRect(0, 0, cssW, cssH);

      const isLight = document.documentElement.classList.contains('light') ||
        getComputedStyle(document.body).color.includes('15, 23, 42') ||
        !getComputedStyle(document.documentElement).getPropertyValue('--background').includes('#0');
      const fgCol = getComputedStyle(document.body).color || (isLight ? '#0f172a' : '#f8fafc');
      const gridCol = 'rgba(148, 163, 184, 0.24)';

      const padL = 46, padR = 20, padT = 18, padB = 42;
      const plotW = cssW - padL - padR;
      const plotH = cssH - padT - padB;
      const maxY = info.maxY;

      const yToPx = y => padT + plotH * (1 - Math.max(0, Math.min(maxY, y)) / maxY);

      // Horizontal Y gridlines
      ctx.font = '12px monospace';
      ctx.fillStyle = fgCol;
      const ySteps = maxY <= 0.7 ? [0, 0.2, 0.4, 0.6] : [0, 0.25, 0.5, 0.75, 1.0];
      ySteps.forEach(yv => {
        const py = yToPx(yv);
        ctx.strokeStyle = gridCol;
        ctx.lineWidth = 1;
        ctx.beginPath(); ctx.moveTo(padL, py); ctx.lineTo(cssW - padR, py); ctx.stroke();
        ctx.globalAlpha = 0.7;
        ctx.fillText(yv.toFixed(2), 6, py + 4);
        ctx.globalAlpha = 1.0;
      });

      // Reference threshold line
      let refY = info.refY;
      if (activeTest === 3 && sub.length) refY = sub[0].fpr10_threshold;
      if (refY !== null) {
        const ry = yToPx(refY);
        ctx.save();
        ctx.setLineDash([5, 4]);
        ctx.strokeStyle = '#64748b';
        ctx.lineWidth = 1.3;
        ctx.beginPath(); ctx.moveTo(padL, ry); ctx.lineTo(cssW - padR, ry); ctx.stroke();
        ctx.restore();
      }

      plottedPts = [];
      if (!continuousMode) {
        const tierOrders = [...new Set(sub.map(r => r.diff_order))].sort((a, b) => a - b);
        const nT = tierOrders.length;
        tierOrders.forEach((doVal, idx) => {
          const tsub = sub.filter(r => r.diff_order === doVal);
          const cx = padL + plotW * ((idx + 0.5) / nT);
          const meanY = tsub.reduce((a, b) => a + b.score, 0) / tsub.length;
          const my = yToPx(meanY);

          // Mean bar
          ctx.strokeStyle = fgCol;
          ctx.lineWidth = 2.4;
          ctx.beginPath(); ctx.moveTo(cx - 24, my); ctx.lineTo(cx + 24, my); ctx.stroke();

          ctx.font = 'bold 12px monospace';
          ctx.fillStyle = fgCol;
          const mTxt = meanY.toFixed(2);
          ctx.fillText(mTxt, cx - ctx.measureText(mTxt).width / 2, Math.max(13, my - 7));

          // X-axis tier label
          const rawLbl = tsub[0].diff_tier.split(': ')[1] || tsub[0].diff_tier;
          const shortLbl = rawLbl.length > 20 ? rawLbl.slice(0, 19) + '…' : rawLbl;
          ctx.font = '12px sans-serif';
          ctx.globalAlpha = 0.85;
          ctx.fillText(shortLbl, cx - ctx.measureText(shortLbl).width / 2, cssH - 20);
          const nLbl = `(n=${tsub.length})`;
          ctx.fillText(nLbl, cx - ctx.measureText(nLbl).width / 2, cssH - 6);
          ctx.globalAlpha = 1.0;

          tsub.forEach(r => {
            const jx = cx + seededJitter(r.id) * Math.min(46, (plotW / nT) * 0.52);
            const py = yToPx(r.score);
            plottedPts.push({ x: jx, y: py, rec: r });
          });
        });
      } else {
        // Continuous X mode [0, 1]
        [0, 0.25, 0.5, 0.75, 1.0].forEach(xv => {
          const px = padL + plotW * xv;
          ctx.font = '12px monospace';
          ctx.globalAlpha = 0.75;
          ctx.fillText(xv.toFixed(2), px - 12, cssH - 20);
          ctx.globalAlpha = 1.0;
        });
        const xl = sub[0].cont_x_label;
        ctx.font = '12px sans-serif';
        ctx.fillText(xl, padL + plotW / 2 - ctx.measureText(xl).width / 2, cssH - 5);

        sub.forEach(r => {
          const px = padL + plotW * Math.max(0.02, Math.min(0.98, r.cont_x + seededJitter(r.id) * 0.02));
          const py = yToPx(r.score);
          plottedPts.push({ x: px, y: py, rec: r });
        });
      }

      // Render points
      plottedPts.forEach((pt, idx) => {
        const col = SPLIT_COLORS[pt.rec.split] || '#6366f1';
        const isPin = idx === pinnedIdx;
        ctx.beginPath();
        ctx.arc(pt.x, pt.y, isPin ? 6.5 : 4.6, 0, Math.PI * 2);
        if (pt.rec.abstained) {
          ctx.strokeStyle = col;
          ctx.lineWidth = 2.0;
          ctx.stroke();
        } else {
          ctx.fillStyle = col;
          ctx.globalAlpha = isPin ? 1.0 : 0.85;
          ctx.fill();
          ctx.globalAlpha = 1.0;
          ctx.strokeStyle = isPin ? fgCol : 'rgba(255,255,255,0.7)';
          ctx.lineWidth = isPin ? 2.0 : 0.8;
          ctx.stroke();
        }
      });

      if (pinnedIdx >= 0 && plottedPts[pinnedIdx]) {
        showHover(plottedPts[pinnedIdx].rec, true);
      }
    }

    function showHover(r, pinned) {
      const el = document.getElementById('hover-card');
      const abst = r.abstained ? ' • <span class="font-bold text-rose-500">ABSTAINED</span>' : '';
      el.innerHTML = `
        <div class="space-y-0.5">
          <div><strong class="font-mono">${r.id}</strong> — <strong>${r.name}</strong>${abst}</div>
          <div class="text-[var(--muted-foreground)]">Tier: <strong>${r.diff_tier}</strong> • Split: <strong>${r.split}</strong> • Len: ${r.length} AA • Reasoning: ${r.reasoning_tokens.toLocaleString()} tok</div>
        </div>
        <div class="font-mono font-bold text-sm">
          ${r.score_label.split(' (')[0]}: ${r.score.toFixed(3)} <span class="text-xs font-normal text-[var(--muted-foreground)]">(${r.secondary_label}: ${r.secondary_score.toFixed(2)})</span>
        </div>`;
    }

    const canvasEl = document.getElementById('plot-canvas');
    canvasEl.addEventListener('mousemove', ev => {
      if (pinnedIdx >= 0) return;
      const rect = canvasEl.getBoundingClientRect();
      const mx = ev.clientX - rect.left, my = ev.clientY - rect.top;
      let best = -1, bestD = 18;
      plottedPts.forEach((p, idx) => {
        const d = Math.hypot(p.x - mx, p.y - my);
        if (d < bestD) { bestD = d; best = idx; }
      });
      if (best >= 0) showHover(plottedPts[best].rec, false);
    });
    canvasEl.addEventListener('click', ev => {
      const rect = canvasEl.getBoundingClientRect();
      const mx = ev.clientX - rect.left, my = ev.clientY - rect.top;
      let best = -1, bestD = 18;
      plottedPts.forEach((p, idx) => {
        const d = Math.hypot(p.x - mx, p.y - my);
        if (d < bestD) { bestD = d; best = idx; }
      });
      pinnedIdx = best === pinnedIdx ? -1 : best;
      draw();
    });

    document.getElementById('mode-toggle').addEventListener('click', () => {
      continuousMode = !continuousMode;
      draw();
    });
    window.addEventListener('resize', draw);
    draw();
  </script>
</body>
</html>
"""


def main() -> None:
    records = compile_370_question_records()
    out_json = REPO / "outputs/results/local_13456_per_question_records.json"
    out_json.write_text(json.dumps(records, indent=2), encoding="utf-8")

    out_png = REPO / "outputs/figures/gpt62_sol_per_question_difficulty.png"
    render_6panel_per_question_png(records, out_png)

    out_html = REPO / "outputs/figures/gpt62_sol_per_question_difficulty.html"
    html_txt = WIDGET_HTML_TEMPLATE.replace("__RECORDS_JSON__", json.dumps(records, separators=(",", ":")))
    out_html.write_text(html_txt, encoding="utf-8")

    print(f"Compiled {len(records)} per-question records -> {out_json}")
    print(f"Rendered 6-panel per-question figure -> {out_png}")
    print(f"Rendered interactive HTML widget -> {out_html} ({out_html.stat().st_size / 1024.0:.1f} KB)")


if __name__ == "__main__":
    main()
