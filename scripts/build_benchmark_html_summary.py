#!/usr/bin/env python3
"""Compile real benchmark manifests, 3D coordinates, calibration scores & prompts into a
self-contained interactive HTML Benchmark Explorer (`protein_without_tools_summary.html`).
"""

from __future__ import annotations

import json
from pathlib import Path
import sys
from typing import Any, Dict, List

import numpy as np

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from protein_without_tools.pdb_utils import load_structure


def extract_ca_pdb_simple(pdb_path: Path) -> List[List[float]]:
    coords = []
    for line in pdb_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        if line.startswith("ATOM  ") and line[12:16].strip() == "CA":
            coords.append([
                round(float(line[30:38]), 1),
                round(float(line[38:46]), 1),
                round(float(line[46:54]), 1),
            ])
    return coords


def build_data_payload() -> Dict[str, Any]:
    # 1. Stage 1 (`N = 100`)
    m1 = json.loads((REPO / "data/stage1_benchmark/manifest.json").read_text(encoding="utf-8"))
    s1_points = []
    showcase_pids = {"9YOY", "9YOX", "11CY", "21VU", "11CM", "9VFK", "9P40", "35AW", "23AA", "37NT", "6CZH", "6MRR"}
    s1_3d: Dict[str, Any] = {}

    gt_dir = REPO / "data/stage1_benchmark/ground_truth"
    for t in m1["targets"]:
        tid = t["target_id"]
        pid = t["pdb_id"]
        pt = {
            "tid": tid,
            "pdb": pid,
            "chain": t.get("chain_id", "A"),
            "stratum": t["stratum"],
            "len": t["length"],
            "date": t.get("release_date", ""),
            "desc": str(t.get("description") or t.get("note") or pid)[:80],
            "pdb_seqid": round(float(t.get("pdb_precutoff_seqid_combined", t.get("pdb_precutoff_seqid_max", 0.0))), 1),
            "pdb_tm": round(float(t.get("pdb_precutoff_tm_query", 0.5)), 3),
            "afdb_seqid": round(float(t.get("afdb_max_seqid", 0.0)), 1),
            "afdb_tm": round(float(t.get("afdb_tm_query", 0.5)), 3),
            "closest_pdb": str(t.get("pdb_precutoff_tm_closest_id") or "none"),
            "has3d": False,
        }
        if pid in showcase_pids and pid not in s1_3d:
            matches = sorted(gt_dir.glob(f"{tid}*.pdb"))
            if matches:
                ca = extract_ca_pdb_simple(matches[0])
                if len(ca) >= 35:
                    s1_3d[tid] = ca
                    pt["has3d"] = True
        s1_points.append(pt)

    # 2. Stage 2A & 2B
    rep2 = json.loads((REPO / "outputs/results/stage2_evaluation_report.json").read_text(encoding="utf-8"))
    m2a = json.loads((REPO / "data/stage2a_homology/manifest.json").read_text(encoding="utf-8"))
    s2a_samples = []
    for rk in ("1_seq_easy", "2_foldseek_3di", "3_deep_tea_zone", "4_hard_negative_decoy"):
        ex = next(p for p in m2a["pairs"] if p["rung"] == rk)
        s2a_samples.append({
            "pair_id": ex["pair_id"],
            "rung": rk,
            "label": ex["label_homolog"],
            "scop_a": ex["scop_a"],
            "scop_b": ex["scop_b"],
            "seqid": ex["seq_identity_pct"],
            "bits": ex["blosum_bitscore"],
            "fs_ev": ex["foldseek_pdb100_evalue"],
            "len_a": len(ex["seq_a"]),
            "len_b": len(ex["seq_b"]),
        })

    # 3. Stage 3A (`N = 52`, `5,348` X->A mutations)
    m3a = json.loads((REPO / "data/alanine_scan_benchmark/manifest.json").read_text(encoding="utf-8"))
    rep3a = json.loads((REPO / "outputs/results/stage3a_alanine_report.json").read_text(encoding="utf-8"))["stage3a_alanine_scan"]
    s3a_examples = []
    picked_cats: Dict[str, int] = {}
    for meta in m3a["targets"]:
        cat = meta["category"]
        if picked_cats.get(cat, 0) >= 2:
            continue
        tfile = REPO / "data/alanine_scan_benchmark/targets" / f"{meta['target_id']}.json"
        if not tfile.exists():
            continue
        tobj = json.loads(tfile.read_text(encoding="utf-8"))
        # Cap display to first 130 assayed positions for crisp barcode rendering
        lim = min(130, len(tobj["positions_1idx"]))
        s3a_examples.append({
            "tid": meta["target_id"],
            "name": meta["source_name"],
            "cat": cat,
            "date": meta["published_date"],
            "assay": meta["assay_type"],
            "len": meta["length"],
            "n_mut": meta["n_assayed_positions"],
            "pos": tobj["positions_1idx"][:lim],
            "wt": tobj["wt_aas"][:lim],
            "scores": [round(float(x), 3) for x in tobj["exp_destab_scores"][:lim]],
            "core": (tobj["is_core_flags"][:lim] if tobj.get("is_core_flags") else []),
        })
        picked_cats[cat] = picked_cats.get(cat, 0) + 1

    # 4. Stage 3B (`N = 40` Monomer/Motif + `N = 12` Proven-Bindable Epitopes)
    m3b_ep = json.loads((REPO / "data/stage3b_design/epitope_binder_manifest.json").read_text(encoding="utf-8"))
    rep3b = json.loads((REPO / "outputs/results/stage3b_evaluation_report.json").read_text(encoding="utf-8"))
    ep_list = []
    for e in m3b_ep["targets"]:
        # Round Cα to 1 decimal place for compact JSON
        t_ca = [[round(float(c), 1) for c in row] for row in e["target_ca_coords"]]
        p_ca = [[round(float(c), 1) for c in row] for row in e["heldout_partner_ca_coords"]] if e["heldout_partner_ca_coords"] else []
        ep_list.append({
            "tid": e["target_id"],
            "tier": e["tier"],
            "name": e["name"],
            "pdb": e["pdb_id"],
            "date": e["release_date"],
            "len_t": len(e["target_sequence"]),
            "len_p": len(e["heldout_partner_sequence"]),
            "hotspots": e["epitope_hotspots_1idx"],
            "hotspot_aas": [f"{e['target_sequence'][h - 1]}{h}" for h in e["epitope_hotspots_1idx"]],
            "t_ca": t_ca,
            "p_ca": p_ca,
        })

    # 5. Sample Zero-Leakage Prompts across all stages
    def _first_prompt(fn: str) -> Dict[str, Any]:
        line = (REPO / "outputs/prompts" / fn).read_text(encoding="utf-8").splitlines()[0]
        return json.loads(line)

    prompts_showcase = [
        {"stage": "1A — Forward Folding (Seq → Contacts + Q3)", "file": "stage1a_forward_prompts.jsonl", **_first_prompt("stage1a_forward_prompts.jsonl")},
        {"stage": "1B — Inverse Folding (Stripped Backbone → Seq)", "file": "stage1b_inverse_prompts.jsonl", **_first_prompt("stage1b_inverse_prompts.jsonl")},
        {"stage": "2A — Remote Homology (SCOPe + TEA Twilight Zone)", "file": "stage2a_homology_prompts.jsonl", **_first_prompt("stage2a_homology_prompts.jsonl")},
        {"stage": "2B — CAFA Function Prediction (GO MF/BP/CC)", "file": "stage2b_cafa_prompts.jsonl", **_first_prompt("stage2b_cafa_prompts.jsonl")},
        {"stage": "3A — Full-Protein In-Silico Alanine Scan (X → A)", "file": "stage3a_alanine_prompts.jsonl", **_first_prompt("stage3a_alanine_prompts.jsonl")},
        {"stage": "3B.1 — De Novo Fold / Crystal Motif Scaffolding", "file": "stage3b_denovo_prompts.jsonl", **_first_prompt("stage3b_denovo_prompts.jsonl")},
        {"stage": "3B.2 — Closed-Loop AF2 Epitope Binder (Shot 1)", "file": "stage3b_binder_round1_prompts.jsonl", **_first_prompt("stage3b_binder_round1_prompts.jsonl")},
    ]

    return {
        "s1_points": s1_points,
        "s1_3d": s1_3d,
        "s2_report": rep2,
        "s2a_samples": s2a_samples,
        "s3a_report": rep3a,
        "s3a_examples": s3a_examples,
        "s3b_report": rep3b,
        "s3b_epitopes": ep_list,
        "prompts": prompts_showcase,
    }


HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>Protein Structure Without Tools — Complete 6-Stage Benchmark Architecture</title>
  <script src="https://www.gstatic.com/antigravity/web/dev/tailwindcss.min.js"></script>
  <style>
    .tab-btn-active {
      background-color: var(--primary);
      color: var(--primary-foreground);
      font-weight: 600;
    }
    canvas { touch-action: none; }
  </style>
</head>
<body class="bg-[var(--background)] text-[var(--foreground)] antialiased p-6">
  <div class="max-w-7xl mx-auto space-y-6">

    <!-- HERO HEADER -->
    <div class="bg-[var(--card)] border border-[var(--border)] rounded-2xl p-6 shadow-sm">
      <div class="flex flex-col lg:flex-row lg:items-center lg:justify-between gap-4">
        <div class="space-y-2">
          <div class="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-[var(--secondary)] text-[var(--secondary-foreground)] text-xs font-semibold">
            <span>Zero-Leakage Post-May-2026 Protein Biology Audit</span>
            <span>•</span>
            <span>All 6 Stages Calibrated &amp; Tested (11/11 Passing)</span>
          </div>
          <h1 class="text-2xl lg:text-3xl font-bold tracking-tight text-[var(--foreground)]">
            Protein Structure Without Tools: Auditing Frontier Reasoning LLMs vs. Specialist Bio-Tools
          </h1>
          <p class="text-sm text-[var(--muted-foreground)] max-w-4xl">
            From sequence memorization to folding physics: a reproducible 6-stage benchmark comparing frontier reasoning LLMs
            (<code class="px-1.5 py-0.5 rounded bg-[var(--secondary)] text-xs">gpt-6.2-astra</code>, <code class="px-1.5 py-0.5 rounded bg-[var(--secondary)] text-xs">claude-opus-5-5</code>, <code class="px-1.5 py-0.5 rounded bg-[var(--secondary)] text-xs">gpt-5.2</code>, <code class="px-1.5 py-0.5 rounded bg-[var(--secondary)] text-xs">gemini-3.1-pro</code>)
            head-to-head against specialist structural biology models (<code class="px-1.5 py-0.5 rounded bg-[var(--secondary)] text-xs">ESMFold</code>, <code class="px-1.5 py-0.5 rounded bg-[var(--secondary)] text-xs">Boltz-2</code>, <code class="px-1.5 py-0.5 rounded bg-[var(--secondary)] text-xs">ProteinMPNN</code>, <code class="px-1.5 py-0.5 rounded bg-[var(--secondary)] text-xs">PickyBinders/tea</code>, <code class="px-1.5 py-0.5 rounded bg-[var(--secondary)] text-xs">ThermoMPNN</code>, <code class="px-1.5 py-0.5 rounded bg-[var(--secondary)] text-xs">BindCraft AF2</code>).
          </p>
        </div>

        <!-- KPI PILLS -->
        <div class="grid grid-cols-2 sm:grid-cols-4 lg:grid-cols-2 gap-3 shrink-0">
          <div class="bg-[var(--background)] border border-[var(--border)] rounded-xl px-4 py-2.5">
            <div class="text-xs text-[var(--muted-foreground)]">Total Curated Tasks</div>
            <div class="text-xl font-bold text-[var(--foreground)]">484</div>
          </div>
          <div class="bg-[var(--background)] border border-[var(--border)] rounded-xl px-4 py-2.5">
            <div class="text-xs text-[var(--muted-foreground)]">Wet-Lab X→A Mutants</div>
            <div class="text-xl font-bold text-[var(--foreground)]">5,348</div>
          </div>
          <div class="bg-[var(--background)] border border-[var(--border)] rounded-xl px-4 py-2.5">
            <div class="text-xs text-[var(--muted-foreground)]">Novel Folds (TM &lt; 0.52)</div>
            <div class="text-xl font-bold text-emerald-500">18 PDBs</div>
          </div>
          <div class="bg-[var(--background)] border border-[var(--border)] rounded-xl px-4 py-2.5">
            <div class="text-xs text-[var(--muted-foreground)]">Closed-Loop AF2 Binders</div>
            <div class="text-xl font-bold text-teal-500">12 × 16 Shots</div>
          </div>
        </div>
      </div>

      <!-- NAVIGATION TABS -->
      <div class="mt-6 pt-4 border-t border-[var(--border)] flex flex-wrap gap-2" id="nav-tabs">
        <button data-tab="overview" class="tab-btn tab-btn-active px-3.5 py-2 rounded-lg text-xs border border-[var(--border)] transition">1. Architecture &amp; Anti-Cheat</button>
        <button data-tab="stage1" class="tab-btn px-3.5 py-2 rounded-lg text-xs border border-[var(--border)] bg-[var(--background)] text-[var(--foreground)] hover:opacity-90 transition">2. Stage 1A/1B: Folding &amp; Inverse (N=100)</button>
        <button data-tab="stage2" class="tab-btn px-3.5 py-2 rounded-lg text-xs border border-[var(--border)] bg-[var(--background)] text-[var(--foreground)] hover:opacity-90 transition">3. Stage 2A/2B: Homology &amp; CAFA (N=180)</button>
        <button data-tab="stage3a" class="tab-btn px-3.5 py-2 rounded-lg text-xs border border-[var(--border)] bg-[var(--background)] text-[var(--foreground)] hover:opacity-90 transition">4. Stage 3A: Alanine Scan ΔΔG (5,348 muts)</button>
        <button data-tab="stage3b" class="tab-btn px-3.5 py-2 rounded-lg text-xs border border-[var(--border)] bg-[var(--background)] text-[var(--foreground)] hover:opacity-90 transition">5. Stage 3B: De Novo &amp; AF2 Binders (N=12×16)</button>
        <button data-tab="prompts" class="tab-btn px-3.5 py-2 rounded-lg text-xs border border-[var(--border)] bg-[var(--background)] text-[var(--foreground)] hover:opacity-90 transition">6. Zero-Leakage Prompts &amp; CLI</button>
      </div>
    </div>

    <!-- =================================================================== -->
    <!-- TAB 1: ARCHITECTURE & ANTI-CHEAT MATRIX                              -->
    <!-- =================================================================== -->
    <section id="tab-overview" class="tab-panel space-y-6">
      <div class="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
        <div onclick="switchTab('stage1')" class="cursor-pointer bg-[var(--card)] border border-[var(--border)] rounded-xl p-5 hover:border-blue-500 transition space-y-2">
          <div class="flex items-center justify-between">
            <span class="px-2.5 py-0.5 rounded bg-blue-500/15 text-blue-500 text-xs font-bold">STAGE 1A</span>
            <span class="text-xs text-[var(--muted-foreground)]">N = 100 PDBs</span>
          </div>
          <h3 class="font-bold text-base">Forward Folding (Seq → 3D)</h3>
          <p class="text-xs text-[var(--muted-foreground)]">Predict residue-residue Cβ &lt; 8Å contacts + Q3 secondary structure, folded deterministically via chiral L-BFGS into 3D coordinates.</p>
          <div class="pt-2 border-t border-[var(--border)] text-xs space-y-1">
            <div><span class="font-semibold">Specialists:</span> ESMFold, Boltz-2 (AF3-class), ESM-2</div>
            <div><span class="font-semibold">Stratification:</span> Exact Foldseek TM-align TM-score &amp; MMseqs2 SeqID to pre-May-2026 PDB &amp; AFDB</div>
          </div>
        </div>

        <div onclick="switchTab('stage1')" class="cursor-pointer bg-[var(--card)] border border-[var(--border)] rounded-xl p-5 hover:border-indigo-500 transition space-y-2">
          <div class="flex items-center justify-between">
            <span class="px-2.5 py-0.5 rounded bg-indigo-500/15 text-indigo-500 text-xs font-bold">STAGE 1B</span>
            <span class="text-xs text-[var(--muted-foreground)]">N = 100 Backbones</span>
          </div>
          <h3 class="font-bold text-base">Inverse Folding (3D → Seq → Refold)</h3>
          <p class="text-xs text-[var(--muted-foreground)]">Design foldable amino-acid sequences from stripped <code class="text-xs">UNK</code> backbone <code class="text-xs">(N, Cα, C, O)</code> coordinates with zero residue-identity leakage.</p>
          <div class="pt-2 border-t border-[var(--border)] text-xs space-y-1">
            <div><span class="font-semibold">Specialists:</span> ProteinMPNN (v_48_020) + ESMFold</div>
            <div><span class="font-semibold">Metrics:</span> Core/Surface NSR %, refolded scTM &amp; scRMSD &lt; 2.0 Å</div>
          </div>
        </div>

        <div onclick="switchTab('stage2')" class="cursor-pointer bg-[var(--card)] border border-[var(--border)] rounded-xl p-5 hover:border-purple-500 transition space-y-2">
          <div class="flex items-center justify-between">
            <span class="px-2.5 py-0.5 rounded bg-purple-500/15 text-purple-500 text-xs font-bold">STAGE 2A</span>
            <span class="text-xs text-[var(--muted-foreground)]">N = 120 SCOPe Pairs</span>
          </div>
          <h3 class="font-bold text-base">Remote Homology Ladder (TEA)</h3>
          <p class="text-xs text-[var(--muted-foreground)]">4-rung evolutionary depth test from sequence-easy homologs down into the &lt;15% twilight zone where both BLAST &amp; Foldseek pdb100 fail.</p>
          <div class="pt-2 border-t border-[var(--border)] text-xs space-y-1">
            <div><span class="font-semibold">Specialists:</span> PickyBinders/tea (ESM-2 → 20-AA), Foldseek 3Di</div>
            <div><span class="font-semibold">Decoys:</span> Length- &amp; SCOPe-class-matched different-fold hard negatives</div>
          </div>
        </div>

        <div onclick="switchTab('stage2')" class="cursor-pointer bg-[var(--card)] border border-[var(--border)] rounded-xl p-5 hover:border-amber-500 transition space-y-2">
          <div class="flex items-center justify-between">
            <span class="px-2.5 py-0.5 rounded bg-amber-500/15 text-amber-500 text-xs font-bold">STAGE 2B</span>
            <span class="text-xs text-[var(--muted-foreground)]">N = 60 Holdouts</span>
          </div>
          <h3 class="font-bold text-base">CAFA Experimental Function (GO)</h3>
          <p class="text-xs text-[var(--muted-foreground)]">Blind Gene Ontology prediction across Molecular Function (MF), Biological Process (BP), and Cellular Component (CC) on 2026 QuickGO holdouts.</p>
          <div class="pt-2 border-t border-[var(--border)] text-xs space-y-1">
            <div><span class="font-semibold">Specialists:</span> ESM-2 Embedding k-NN, Smith-Waterman, CAFA Naive</div>
            <div><span class="font-semibold">Metrics:</span> Information-Content-weighted wF_max &amp; F_max</div>
          </div>
        </div>

        <div onclick="switchTab('stage3a')" class="cursor-pointer bg-[var(--card)] border border-[var(--border)] rounded-xl p-5 hover:border-rose-500 transition space-y-2">
          <div class="flex items-center justify-between">
            <span class="px-2.5 py-0.5 rounded bg-rose-500/15 text-rose-500 text-xs font-bold">STAGE 3A</span>
            <span class="text-xs text-[var(--muted-foreground)]">52 Proteins • 5,348 Muts</span>
          </div>
          <h3 class="font-bold text-base">In-Silico Alanine Scan (X → A)</h3>
          <p class="text-xs text-[var(--muted-foreground)]">Inspired by @sokrypton &amp; @WUFang: predict full-sequence X→A destabilization on synthetic de novo Rocklin miniproteins &amp; May–Nov 2026 MaveDB DMS.</p>
          <div class="pt-2 border-t border-[var(--border)] text-xs space-y-1">
            <div><span class="font-semibold">Specialists:</span> ThermoMPNN, ESM-2 (650M &amp; 3B) Masked Marginals</div>
            <div><span class="font-semibold">Metrics:</span> AA-Demeaned Context ρ (core vs surface Leu), Hotspot AUC</div>
          </div>
        </div>

        <div onclick="switchTab('stage3b')" class="cursor-pointer bg-[var(--card)] border border-[var(--border)] rounded-xl p-5 hover:border-teal-500 transition space-y-2">
          <div class="flex items-center justify-between">
            <span class="px-2.5 py-0.5 rounded bg-teal-500/15 text-teal-500 text-xs font-bold">STAGE 3B</span>
            <span class="text-xs text-[var(--muted-foreground)]">40 Monomers + 12×16 Binders</span>
          </div>
          <h3 class="font-bold text-base">De Novo &amp; Closed-Loop AF2 Binders</h3>
          <p class="text-xs text-[var(--muted-foreground)]">5-gate fold/motif scaffolding + BindCraft-style closed-loop AlphaFold2 epitope binder optimization across 12 proven-bindable targets.</p>
          <div class="pt-2 border-t border-[var(--border)] text-xs space-y-1">
            <div><span class="font-semibold">Specialists:</span> BindCraft (ColabDesign AF2), RFD4/ProtDBench, BoltzProt-1</div>
            <div><span class="font-semibold">Metrics:</span> Shots-to-First-Hit (Pass@1..16), ipTM ≥ 0.75, iPAE &lt; 10Å</div>
          </div>
        </div>
      </div>

      <!-- ANTI-CHEAT ENGINEERING TABLE -->
      <div class="bg-[var(--card)] border border-[var(--border)] rounded-2xl p-6 space-y-4">
        <div>
          <h2 class="text-lg font-bold">6 Subtle Protein Benchmark Pitfalls We Caught &amp; Engineered Out</h2>
          <p class="text-xs text-[var(--muted-foreground)]">Why naive LLM biology benchmarks overestimate reasoning—and how Protein Structure Without Tools closes every loophole.</p>
        </div>
        <div class="overflow-x-auto">
          <table class="w-full text-left border-collapse text-xs">
            <thead>
              <tr class="border-b border-[var(--border)] text-[var(--muted-foreground)]">
                <th class="py-2.5 pr-4 font-semibold">Pitfall / Shortcut</th>
                <th class="py-2.5 pr-4 font-semibold">What Goes Wrong Without Guardrails</th>
                <th class="py-2.5 font-semibold">Our Implemented Fix</th>
              </tr>
            </thead>
            <tbody class="divide-y divide-[var(--border)]">
              <tr>
                <td class="py-2.5 pr-4 font-semibold">1. Foldseek 3diaa E-value vs. True 3D TM-score</td>
                <td class="py-2.5 pr-4 text-[var(--muted-foreground)]"><code class="text-xs">47WY</code> (14.8% SeqID) gets weak Foldseek E=6.50, yet its true 3D TM-score to pre-cutoff PDB is <code class="text-xs">0.84</code> (1.55Å RMSD).</td>
                <td class="py-2.5">Computed exact <code class="text-xs">Foldseek mode="tmalign"</code> + Kabsch SO(3) TM-score across all 100 Stage 1 targets; mined <strong>18 genuinely novel folds (mean PDB TM = 0.431)</strong>.</td>
              </tr>
              <tr>
                <td class="py-2.5 pr-4 font-semibold">2. Post-Cutoff Re-Crystallization Leaks</td>
                <td class="py-2.5 pr-4 text-[var(--muted-foreground)]">RCSB top-35 pagination hid pre-2026 structures of ubiquitous proteins like Lysozyme (<code class="text-xs">38BH</code>) &amp; KRAS (<code class="text-xs">11GH</code>).</td>
                <td class="py-2.5">Combined <code class="text-xs">MMseqs2 (rows=100, return_explain_metadata=True)</code> + Foldseek <code class="text-xs">pdb100</code> pre-cutoff SeqID; purged all leaked re-crystallizations.</td>
              </tr>
              <tr>
                <td class="py-2.5 pr-4 font-semibold">3. Static 1D Hydrophobicity Cheat on Alanine Scans</td>
                <td class="py-2.5 pr-4 text-[var(--muted-foreground)]">Predicting <code class="text-xs">Trp/Phe/Leu → Ala</code> is always destabilizing yields <strong>Raw Spearman ρ = 0.339</strong> with zero 3D understanding!</td>
                <td class="py-2.5">Added <strong>AA-Demeaned Context Correlation (<code class="text-xs">AA-Ctx ρ</code>)</strong>, which scores <code class="text-xs">0.000</code> on 1D tables and isolates true buried-core vs. surface-exposed context.</td>
              </tr>
              <tr>
                <td class="py-2.5 pr-4 font-semibold">4. Generic "Binding" Spam in CAFA GO Prediction</td>
                <td class="py-2.5 pr-4 text-[var(--muted-foreground)]">Guessing high-prior root neighbors ties BLAST on unweighted <code class="text-xs">F_max</code> (0.322 vs 0.323).</td>
                <td class="py-2.5">Primary metric is <strong>Information-Content-weighted <code class="text-xs">wF_max</code></strong> (bits), where homology beats Naive <strong>0.357 vs 0.127</strong> on Molecular Function.</td>
              </tr>
              <tr>
                <td class="py-2.5 pr-4 font-semibold">5. EAAAK Coiled-Coil Rods in De Novo Design</td>
                <td class="py-2.5 pr-4 text-[var(--muted-foreground)]">Periodic <code class="text-xs">AEAAAKEAAAK...</code> helices trick ESMFold into <code class="text-xs">pLDDT &gt; 90</code> despite being non-globular infinite rods.</td>
                <td class="py-2.5">5-Gate filter enforces Shannon entropy ≥ 3.2 bits, 3-mer repeat cap, globular <code class="text-xs">Rg ≤ 2.35·L^0.38</code>, MPNN <code class="text-xs">scRMSD ≤ 2Å</code>, and <code class="text-xs">&lt;30%</code> SeqID.</td>
              </tr>
              <tr>
                <td class="py-2.5 pr-4 font-semibold">6. Unbindable Dimers &amp; Anthropic Eval Overlap in Binders</td>
                <td class="py-2.5 pr-4 text-[var(--muted-foreground)]">Obligate enzyme dimers may be unbindable by miniproteins; <code class="text-xs">TREM2/15-PGDH/RBX1</code> were used in Anthropic's Aug-2026 binder study.</td>
                <td class="py-2.5">Curated <strong>12 proven-bindable epitopes</strong> (4 <code class="text-xs">RFD4/ProtDBench</code> + 5 June-2026 <code class="text-xs">BoltzProt-1</code> novel + 3 July–Aug 2026 wet-lab de novo minibinder crystals), excluding Anthropic's internal eval set.</td>
              </tr>
            </tbody>
          </table>
        </div>
      </div>
    </section>

    <!-- =================================================================== -->
    <!-- TAB 2: STAGE 1A & 1B — TRUE TM-SCORE LANDSCAPE & 3D VIEWER          -->
    <!-- =================================================================== -->
    <section id="tab-stage1" class="tab-panel hidden space-y-6">
      <div class="grid grid-cols-1 lg:grid-cols-12 gap-6">
        <!-- Interactive SVG Scatter (7 cols) -->
        <div class="lg:col-span-7 bg-[var(--card)] border border-[var(--border)] rounded-2xl p-5 space-y-3">
          <div class="flex flex-wrap items-center justify-between gap-2">
            <div>
              <h2 class="text-base font-bold">Stage 1 v3 (N = 100): Pre-Cutoff PDB SeqID % vs. Exact 3D TM-Score</h2>
              <p class="text-xs text-[var(--muted-foreground)]">Click highlighted points (ringed) or quick-select pills below to inspect the real 3D Cα fold on the right.</p>
            </div>
          </div>
          <div id="s1-scatter-wrap" class="w-full overflow-x-auto"></div>
          <div class="flex flex-wrap gap-1.5 pt-1" id="s1-pills"></div>
        </div>

        <!-- Live 3D Canvas Viewer (5 cols) -->
        <div class="lg:col-span-5 bg-[var(--card)] border border-[var(--border)] rounded-2xl p-5 flex flex-col justify-between space-y-3">
          <div>
            <div class="flex items-center justify-between">
              <h3 class="text-base font-bold" id="s1-3d-title">9YOY — Novel Fold (PDB TM = 0.28)</h3>
              <span class="px-2 py-0.5 rounded bg-emerald-500/15 text-emerald-500 text-xs font-semibold" id="s1-3d-badge">post_cutoff_struct_novel</span>
            </div>
            <p class="text-xs text-[var(--muted-foreground)] mt-1" id="s1-3d-desc"></p>
          </div>

          <div class="relative rounded-xl border border-[var(--border)] bg-[var(--background)] overflow-hidden">
            <canvas id="s1-canvas" width="440" height="300" class="w-full h-[280px] cursor-grab active:cursor-grabbing"></canvas>
            <div class="absolute bottom-2 left-3 text-xs text-[var(--muted-foreground)] pointer-events-none">Drag to rotate 3D Cα trace (N→C rainbow)</div>
          </div>

          <div class="grid grid-cols-2 gap-2 text-xs" id="s1-3d-metrics"></div>
        </div>
      </div>
    </section>

    <!-- =================================================================== -->
    <!-- TAB 3: STAGE 2A (TEA HOMOLOGY) & STAGE 2B (CAFA FUNCTION)           -->
    <!-- =================================================================== -->
    <section id="tab-stage2" class="tab-panel hidden space-y-6">
      <div class="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <!-- Stage 2A -->
        <div class="bg-[var(--card)] border border-[var(--border)] rounded-2xl p-5 space-y-4">
          <div>
            <span class="px-2 py-0.5 rounded bg-purple-500/15 text-purple-500 text-xs font-bold">STAGE 2A • N = 120 SCOPe 2.08 PAIRS</span>
            <h2 class="text-base font-bold mt-1">Remote Homology Sensitivity Step-Down (Recall @ 10% FPR)</h2>
            <p class="text-xs text-[var(--muted-foreground)]">Calibrated against 30 SCOPe-class-matched &amp; length-matched (±10%) different-fold hard negative decoys.</p>
          </div>
          <div id="s2a-chart"></div>
          <div class="space-y-2 pt-2 border-t border-[var(--border)]">
            <div class="text-xs font-semibold">Sample Pairs by Evolutionary Rung:</div>
            <div id="s2a-samples" class="grid grid-cols-1 sm:grid-cols-2 gap-2 text-xs"></div>
          </div>
        </div>

        <!-- Stage 2B -->
        <div class="bg-[var(--card)] border border-[var(--border)] rounded-2xl p-5 space-y-4">
          <div class="flex flex-wrap items-start justify-between gap-2">
            <div>
              <span class="px-2 py-0.5 rounded bg-amber-500/15 text-amber-500 text-xs font-bold">STAGE 2B • N = 60 2026 QUICKGO HOLDOUTS</span>
              <h2 class="text-base font-bold mt-1">CAFA Function Prediction (1,728 Pre-2025 Swiss-Prot Ref)</h2>
              <p class="text-xs text-[var(--muted-foreground)]">Toggle between unweighted <code class="text-xs">F_max</code> and Information-Content-weighted <code class="text-xs">wF_max</code>:</p>
            </div>
            <div class="inline-flex rounded-lg border border-[var(--border)] p-0.5 bg-[var(--background)]">
              <button id="btn-wfmax" onclick="renderStage2B('wfmax')" class="px-2.5 py-1 rounded-md text-xs font-semibold bg-[var(--primary)] text-[var(--primary-foreground)]">IC-Weighted wF_max</button>
              <button id="btn-fmax" onclick="renderStage2B('fmax')" class="px-2.5 py-1 rounded-md text-xs text-[var(--muted-foreground)]">Unweighted F_max</button>
            </div>
          </div>
          <div id="s2b-chart"></div>
          <div class="p-3 rounded-xl bg-[var(--background)] border border-[var(--border)] text-xs text-[var(--muted-foreground)]" id="s2b-note"></div>
        </div>
      </div>
    </section>

    <!-- =================================================================== -->
    <!-- TAB 4: STAGE 3A — WET-LAB ALANINE SCAN EXPLORER                     -->
    <!-- =================================================================== -->
    <section id="tab-stage3a" class="tab-panel hidden space-y-6">
      <div class="grid grid-cols-1 lg:grid-cols-12 gap-6">
        <div class="lg:col-span-8 bg-[var(--card)] border border-[var(--border)] rounded-2xl p-5 space-y-4">
          <div class="flex flex-wrap items-center justify-between gap-2">
            <div>
              <span class="px-2 py-0.5 rounded bg-rose-500/15 text-rose-500 text-xs font-bold">STAGE 3A • 52 PROTEINS • 5,348 WET-LAB X→A MUTATIONS</span>
              <h2 class="text-base font-bold mt-1">Per-Residue Experimental X → Ala Mutational Scan Explorer</h2>
            </div>
            <select id="s3a-select" onchange="renderStage3AProfile(this.value)" class="px-3 py-1.5 rounded-lg border border-[var(--border)] bg-[var(--background)] text-[var(--foreground)] text-xs font-medium"></select>
          </div>
          <div id="s3a-barcode" class="w-full overflow-x-auto"></div>
          <p class="text-xs text-[var(--muted-foreground)]" id="s3a-caption"></p>
        </div>

        <div class="lg:col-span-4 bg-[var(--card)] border border-[var(--border)] rounded-2xl p-5 space-y-4">
          <div>
            <h3 class="text-base font-bold">Why AA-Demeaned Context ρ Matters</h3>
            <p class="text-xs text-[var(--muted-foreground)] mt-1">
              A static 20-amino-acid hydrophobicity table achieves <strong>Raw ρ = 0.339</strong> because mutating bulky hydrophobics (<code class="text-xs">W/F/I/L → A</code>) hurts on average—yet it has <strong>zero</strong> knowledge of whether a <code class="text-xs">Leu</code> is buried in the 3D core or exposed on the surface!
            </p>
          </div>
          <div id="s3a-calib-bars" class="space-y-3 pt-2"></div>
        </div>
      </div>
    </section>

    <!-- =================================================================== -->
    <!-- TAB 5: STAGE 3B — DE NOVO & CLOSED-LOOP AF2 EPITOPE BINDERS         -->
    <!-- =================================================================== -->
    <section id="tab-stage3b" class="tab-panel hidden space-y-6">
      <div class="grid grid-cols-1 lg:grid-cols-12 gap-6">
        <!-- 3D Epitope & Held-Out Binder Viewer (6 cols) -->
        <div class="lg:col-span-6 bg-[var(--card)] border border-[var(--border)] rounded-2xl p-5 space-y-3">
          <div class="flex flex-wrap items-center justify-between gap-2">
            <div>
              <span class="px-2 py-0.5 rounded bg-teal-500/15 text-teal-500 text-xs font-bold">N = 12 PROVEN-BINDABLE EPITOPES</span>
              <h2 class="text-base font-bold mt-1">3D Target Epitope Hotspots &amp; Held-Out Minibinders</h2>
            </div>
            <select id="s3b-ep-select" onchange="selectEpitope3D(this.value)" class="px-3 py-1.5 rounded-lg border border-[var(--border)] bg-[var(--background)] text-[var(--foreground)] text-xs font-medium"></select>
          </div>

          <div class="relative rounded-xl border border-[var(--border)] bg-[var(--background)] overflow-hidden">
            <canvas id="s3b-canvas" width="520" height="310" class="w-full h-[290px] cursor-grab active:cursor-grabbing"></canvas>
            <div class="absolute bottom-2 left-3 flex flex-wrap gap-3 text-xs pointer-events-none">
              <span class="inline-flex items-center gap-1"><span class="w-2.5 h-2.5 rounded-full bg-sky-500 inline-block"></span>Target Backbone</span>
              <span class="inline-flex items-center gap-1"><span class="w-2.5 h-2.5 rounded-full bg-rose-500 inline-block"></span>5 Epitope Hotspots</span>
              <span class="inline-flex items-center gap-1"><span class="w-2.5 h-2.5 rounded-full bg-emerald-500 inline-block"></span>Held-Out Partner/Binder</span>
            </div>
          </div>

          <div id="s3b-ep-meta" class="grid grid-cols-2 sm:grid-cols-3 gap-2 text-xs"></div>
        </div>

        <!-- Shots-to-Hit Curve & ProtDBench Specialist Bars (6 cols) -->
        <div class="lg:col-span-6 space-y-6">
          <div class="bg-[var(--card)] border border-[var(--border)] rounded-2xl p-5 space-y-3">
            <h3 class="text-base font-bold">Closed-Loop AF2 Feedback vs. Blind Pass@16 (Shots-to-Hit)</h3>
            <p class="text-xs text-[var(--muted-foreground)]">Each round returns <code class="text-xs">ipTM</code>, <code class="text-xs">iPAE</code>, <code class="text-xs">binder pLDDT</code>, <code class="text-xs">monomer RMSD</code>, and exact hotspot contacts (≥25% SeqID copy-paste disqualified).</p>
            <div id="s3b-passk-svg"></div>
          </div>

          <div class="bg-[var(--card)] border border-[var(--border)] rounded-2xl p-5 space-y-3">
            <h3 class="text-base font-bold">Published ProtDBench (2026) Specialist Single-Shot AF2 Pass Rates</h3>
            <div id="s3b-protd-bars" class="space-y-2 text-xs"></div>
          </div>
        </div>
      </div>
    </section>

    <!-- =================================================================== -->
    <!-- TAB 6: ZERO-LEAKAGE PROMPT INSPECTOR & TURNKEY CLI                  -->
    <!-- =================================================================== -->
    <section id="tab-prompts" class="tab-panel hidden space-y-6">
      <div class="grid grid-cols-1 lg:grid-cols-12 gap-6">
        <div class="lg:col-span-7 bg-[var(--card)] border border-[var(--border)] rounded-2xl p-5 space-y-3">
          <div class="flex flex-wrap items-center justify-between gap-2">
            <div>
              <h2 class="text-base font-bold">Live Zero-Leakage Prompt Inspector</h2>
              <p class="text-xs text-[var(--muted-foreground)]">Every PDB ID, UniProt accession, SCOPe code, and native binder sequence is stripped &amp; verified by pytest.</p>
            </div>
            <div class="flex items-center gap-2">
              <select id="prompt-select" onchange="renderPromptBox(this.value)" class="px-2.5 py-1.5 rounded-lg border border-[var(--border)] bg-[var(--background)] text-[var(--foreground)] text-xs font-medium"></select>
              <button id="copy-prompt-btn" onclick="copyCurrentPrompt()" class="px-3 py-1.5 rounded-lg bg-[var(--primary)] text-[var(--primary-foreground)] text-xs font-semibold hover:opacity-90 transition">Copy Prompt</button>
            </div>
          </div>
          <pre id="prompt-pre" class="p-4 rounded-xl bg-[var(--background)] border border-[var(--border)] text-xs font-mono whitespace-pre-wrap overflow-y-auto max-h-[420px] text-[var(--foreground)]"></pre>
        </div>

        <div class="lg:col-span-5 bg-[var(--card)] border border-[var(--border)] rounded-2xl p-5 space-y-4">
          <h2 class="text-base font-bold">Turnkey Execution Commands</h2>
          <div class="space-y-3 text-xs">
            <div>
              <div class="font-semibold mb-1">1. Offline Verification &amp; Calibration (0 API Keys, &lt;2s)</div>
              <pre class="p-3 rounded-lg bg-[var(--background)] border border-[var(--border)] font-mono overflow-x-auto">.venv/bin/pytest -v</pre>
            </div>
            <div>
              <div class="font-semibold mb-1">2. Specialist GPU Bio-Tools (Modal L40S)</div>
              <pre class="p-3 rounded-lg bg-[var(--background)] border border-[var(--border)] font-mono overflow-x-auto">.venv/bin/modal run scripts/modal_stage1_baselines.py --mode baselines
.venv/bin/modal run scripts/modal_stage2_baselines.py
.venv/bin/modal run scripts/modal_alanine_baselines.py
.venv/bin/modal run scripts/modal_stage3b_af2_bindcraft.py --mode bindcraft-baseline</pre>
            </div>
            <div>
              <div class="font-semibold mb-1">3. Frontier LLMs (Single-Turn Stages 1A–3B.1 + Closed-Loop 3B.2)</div>
              <pre class="p-3 rounded-lg bg-[var(--background)] border border-[var(--border)] font-mono overflow-x-auto">.venv/bin/python scripts/run_benchmark_llms.py \
  --provider openai --model gpt-6.2-astra \
  --stages all --thinking-budget 8000

.venv/bin/python scripts/run_stage3b_closed_loop_llms.py \
  --provider openai --model gpt-6.2-astra \
  --mode closed_loop --oracle modal_af2 --shots 16</pre>
            </div>
          </div>
        </div>
      </div>
    </section>

  </div>

  <script>
    const DATA = __BENCHMARK_DATA_JSON__;

    // =========================================================================
    // Tab Switching
    // =========================================================================
    function switchTab(tabId) {
      document.querySelectorAll('.tab-panel').forEach(el => el.classList.add('hidden'));
      document.getElementById('tab-' + tabId).classList.remove('hidden');
      document.querySelectorAll('.tab-btn').forEach(btn => {
        if (btn.dataset.tab === tabId) {
          btn.classList.add('tab-btn-active');
          btn.classList.remove('bg-[var(--background)]', 'text-[var(--foreground)]');
        } else {
          btn.classList.remove('tab-btn-active');
          btn.classList.add('bg-[var(--background)]', 'text-[var(--foreground)]');
        }
      });
      if (tabId === 'stage1') drawS1Canvas();
      if (tabId === 'stage3b') drawS3BCanvas();
    }
    document.querySelectorAll('.tab-btn').forEach(btn => {
      btn.addEventListener('click', () => switchTab(btn.dataset.tab));
    });

    function getThemeColors() {
      const st = getComputedStyle(document.documentElement);
      const fg = st.getPropertyValue('--foreground').trim() || '#0f172a';
      const muted = st.getPropertyValue('--muted-foreground').trim() || '#64748b';
      const border = st.getPropertyValue('--border').trim() || '#cbd5e1';
      return { fg, muted, border };
    }

    // =========================================================================
    // Simple Interactive 3D Viewer Engine (<canvas>)
    // =========================================================================
    function makeRotator(canvas, redrawFn) {
      const state = { rotX: -0.35, rotY: 0.65, dragging: false, lx: 0, ly: 0 };
      canvas.addEventListener('pointerdown', e => {
        state.dragging = true; state.lx = e.clientX; state.ly = e.clientY;
        canvas.setPointerCapture(e.pointerId);
      });
      canvas.addEventListener('pointermove', e => {
        if (!state.dragging) return;
        state.rotY += (e.clientX - state.lx) * 0.012;
        state.rotX += (e.clientY - state.ly) * 0.012;
        state.lx = e.clientX; state.ly = e.clientY;
        redrawFn();
      });
      canvas.addEventListener('pointerup', () => { state.dragging = false; });
      return state;
    }

    function project3D(pt, center, rotX, rotY, scale, w, h) {
      let x = pt[0] - center[0], y = pt[1] - center[1], z = pt[2] - center[2];
      const cy = Math.cos(rotY), sy = Math.sin(rotY);
      let x1 = x * cy + z * sy, z1 = -x * sy + z * cy;
      const cx = Math.cos(rotX), sx = Math.sin(rotX);
      let y2 = y * cx - z1 * sx, z2 = y * sx + z1 * cx;
      const persp = 240 / (240 + z2 * 0.35);
      return [w / 2 + x1 * scale * persp, h / 2 + y2 * scale * persp, z2];
    }

    // =========================================================================
    // Tab 2: Stage 1 Scatter + 3D Novel Fold Viewer
    // =========================================================================
    const STRATUM_COLORS = {
      post_cutoff_struct_novel: '#10b981',
      post_cutoff_de_novo: '#0ea5e9',
      post_cutoff_pdb_novel: '#8b5cf6',
      post_cutoff_homolog_gradient: '#f59e0b',
      pre_cutoff_control: '#ef4444'
    };
    let currentS1Tid = Object.keys(DATA.s1_3d)[0];
    const s1Canvas = document.getElementById('s1-canvas');
    const s1Rot = makeRotator(s1Canvas, drawS1Canvas);

    function selectS1Target(tid) {
      if (!DATA.s1_3d[tid]) return;
      currentS1Tid = tid;
      const pt = DATA.s1_points.find(p => p.tid === tid);
      document.getElementById('s1-3d-title').textContent = `${pt.pdb} (${pt.len} AA) — PDB TM = ${pt.pdb_tm.toFixed(2)}`;
      document.getElementById('s1-3d-badge').textContent = pt.stratum;
      document.getElementById('s1-3d-desc').textContent = `${pt.desc} • Released ${pt.date}`;
      document.getElementById('s1-3d-metrics').innerHTML = `
        <div class="p-2.5 rounded-lg bg-[var(--background)] border border-[var(--border)]">
          <div class="text-[var(--muted-foreground)]">Closest Pre-Cutoff PDB</div>
          <div class="font-bold">TM = ${pt.pdb_tm.toFixed(3)} (${pt.pdb_seqid}% SeqID)</div>
        </div>
        <div class="p-2.5 rounded-lg bg-[var(--background)] border border-[var(--border)]">
          <div class="text-[var(--muted-foreground)]">Closest AlphaFold DB</div>
          <div class="font-bold">TM = ${pt.afdb_tm.toFixed(3)} (${pt.afdb_seqid}% SeqID)</div>
        </div>
      `;
      renderS1Scatter();
      drawS1Canvas();
    }

    function drawS1Canvas() {
      const coords = DATA.s1_3d[currentS1Tid];
      if (!coords || !coords.length) return;
      const ctx = s1Canvas.getContext('2d');
      const w = s1Canvas.width, h = s1Canvas.height;
      ctx.clearRect(0, 0, w, h);

      let cx = 0, cy = 0, cz = 0;
      coords.forEach(p => { cx += p[0]; cy += p[1]; cz += p[2]; });
      cx /= coords.length; cy /= coords.length; cz /= coords.length;
      let maxR = 1;
      coords.forEach(p => {
        const d = Math.hypot(p[0] - cx, p[1] - cy, p[2] - cz);
        if (d > maxR) maxR = d;
      });
      const scale = Math.min(w, h) * 0.39 / maxR;
      const proj = coords.map(p => project3D(p, [cx, cy, cz], s1Rot.rotX, s1Rot.rotY, scale, w, h));

      ctx.lineCap = 'round';
      ctx.lineJoin = 'round';
      for (let i = 0; i < proj.length - 1; i++) {
        const hue = Math.round(240 * (1 - i / Math.max(1, proj.length - 1)));
        ctx.strokeStyle = `hsl(${hue}, 82%, 54%)`;
        ctx.lineWidth = 3.6;
        ctx.beginPath();
        ctx.moveTo(proj[i][0], proj[i][1]);
        ctx.lineTo(proj[i + 1][0], proj[i + 1][1]);
        ctx.stroke();
      }
      proj.forEach((p, idx) => {
        const hue = Math.round(240 * (1 - idx / Math.max(1, proj.length - 1)));
        ctx.fillStyle = `hsl(${hue}, 85%, 56%)`;
        ctx.beginPath();
        ctx.arc(p[0], p[1], 2.6, 0, Math.PI * 2);
        ctx.fill();
      });
    }

    function renderS1Scatter() {
      const { fg, muted, border } = getThemeColors();
      const W = 580, H = 310, padL = 48, padR = 18, padT = 18, padB = 40;
      const xPix = s => padL + (Math.min(100, s) / 100) * (W - padL - padR);
      const yPix = tm => padT + (1 - (tm - 0.20) / 0.85) * (H - padT - padB);

      let circles = '';
      DATA.s1_points.forEach(p => {
        const cx = xPix(p.pdb_seqid).toFixed(1);
        const cy = yPix(p.pdb_tm).toFixed(1);
        const col = STRATUM_COLORS[p.stratum] || '#64748b';
        const isSel = p.tid === currentS1Tid;
        const r = isSel ? 7.5 : (p.has3d ? 5.6 : 4.2);
        const stroke = isSel ? fg : (p.has3d ? '#ffffff' : 'none');
        const sw = isSel ? 2.4 : (p.has3d ? 1.3 : 0);
        const cur = p.has3d ? 'cursor:pointer;' : '';
        const click = p.has3d ? `onclick="selectS1Target('${p.tid}')"` : '';
        circles += `<circle cx="${cx}" cy="${cy}" r="${r}" fill="${col}" fill-opacity="0.88" stroke="${stroke}" stroke-width="${sw}" style="${cur}" ${click}><title>${p.pdb} (${p.stratum}): SeqID=${p.pdb_seqid}%, PDB TM=${p.pdb_tm}, AFDB TM=${p.afdb_tm}</title></circle>`;
      });

      const y50 = yPix(0.50).toFixed(1);
      const x30 = xPix(30).toFixed(1);
      document.getElementById('s1-scatter-wrap').innerHTML = `
        <svg viewBox="0 0 ${W} ${H}" class="w-full h-auto">
          <rect x="${padL}" y="${y50}" width="${x30 - padL}" height="${H - padB - y50}" fill="#10b981" fill-opacity="0.10" />
          <line x1="${padL}" y1="${y50}" x2="${W - padR}" y2="${y50}" stroke="#10b981" stroke-dasharray="4 3" stroke-width="1.2" />
          <line x1="${x30}" y1="${padT}" x2="${x30}" y2="${H - padB}" stroke="${border}" stroke-dasharray="4 3" stroke-width="1.1" />
          <text x="${padL + 6}" y="${H - padB - 8}" fill="#10b981" font-size="12" font-weight="600">True Novel Fold Zone (TM &lt; 0.50, SeqID &lt; 30%)</text>
          <line x1="${padL}" y1="${H - padB}" x2="${W - padR}" y2="${H - padB}" stroke="${muted}" />
          <line x1="${padL}" y1="${padT}" x2="${padL}" y2="${H - padB}" stroke="${muted}" />
          ${[0, 20, 40, 60, 80, 100].map(v => `<text x="${xPix(v)}" y="${H - 14}" fill="${muted}" font-size="12" text-anchor="middle">${v}%</text>`).join('')}
          ${[0.3, 0.5, 0.7, 0.9, 1.0].map(v => `<text x="${padL - 8}" y="${yPix(v) + 4}" fill="${muted}" font-size="12" text-anchor="end">${v.toFixed(1)}</text>`).join('')}
          <text x="${(padL + W - padR)/2}" y="${H - 1}" fill="${fg}" font-size="12" font-weight="600" text-anchor="middle">Pre-Cutoff PDB Max Sequence Identity (%)</text>
          <text x="14" y="${(padT + H - padB)/2}" fill="${fg}" font-size="12" font-weight="600" text-anchor="middle" transform="rotate(-90 14 ${(padT + H - padB)/2})">Closest PDB TM-Score</text>
          ${circles}
        </svg>
      `;
    }

    function initS1Pills() {
      const c = document.getElementById('s1-pills');
      const pts3d = DATA.s1_points.filter(p => p.has3d).sort((a, b) => a.pdb_tm - b.pdb_tm);
      c.innerHTML = pts3d.map(p => `
        <button onclick="selectS1Target('${p.tid}')" class="px-2.5 py-1 rounded-md border border-[var(--border)] bg-[var(--background)] hover:border-emerald-500 text-xs font-medium transition">
          ${p.pdb} <span class="text-[var(--muted-foreground)]">(TM=${p.pdb_tm.toFixed(2)})</span>
        </button>
      `).join('');
      selectS1Target(pts3d[0].tid);
    }

    // =========================================================================
    // Tab 3: Stage 2A & Stage 2B
    // =========================================================================
    function initStage2() {
      const s2a = DATA.s2_report.stage2a_remote_homology;
      const rungs = [
        ['1_seq_easy', 'Rung 1: Sequence-Easy Homologs'],
        ['2_foldseek_3di', 'Rung 2: Foldseek 3Di Caught, Seq Missed'],
        ['3_deep_tea_zone', 'Rung 3: Deep PickyBinders/tea Zone (Both Missed)']
      ];
      const methods = [
        ['Foldseek_3Di_Structure', 'Foldseek 3Di (3D Structure)', 'bg-teal-500'],
        ['BLOSUM62_SmithWaterman', 'BLOSUM62 Smith-Waterman', 'bg-blue-500'],
        ['Baseline_AAComposition_Cosine', '20-AA Composition Cosine', 'bg-slate-400']
      ];
      document.getElementById('s2a-chart').innerHTML = rungs.map(([rk, rlbl]) => `
        <div class="mb-3 space-y-1">
          <div class="text-xs font-semibold">${rlbl}</div>
          ${methods.map(([mk, mlbl, mcls]) => {
            const v = s2a[mk].by_rung[rk].recall_at_fpr10;
            return `
              <div class="flex items-center gap-2 text-xs">
                <div class="w-44 truncate text-[var(--muted-foreground)]">${mlbl}</div>
                <div class="flex-1 h-3 rounded-full bg-[var(--background)] border border-[var(--border)] overflow-hidden">
                  <div class="h-full ${mcls}" style="width:${Math.round(v * 100)}%"></div>
                </div>
                <div class="w-10 text-right font-mono font-semibold">${v.toFixed(2)}</div>
              </div>
            `;
          }).join('')}
        </div>
      `).join('');

      document.getElementById('s2a-samples').innerHTML = DATA.s2a_samples.map(s => `
        <div class="p-2.5 rounded-lg bg-[var(--background)] border border-[var(--border)]">
          <div class="font-semibold">${s.rung}</div>
          <div class="text-[var(--muted-foreground)]">SCOPe: ${s.scop_a} vs ${s.scop_b} • SeqID=${s.seqid}% • Bits=${s.bits}</div>
        </div>
      `).join('');

      renderStage2B('wfmax');
    }

    function renderStage2B(metricKey) {
      const s2b = DATA.s2_report.stage2b_cafa_function;
      const isW = metricKey === 'wfmax';
      document.getElementById('btn-wfmax').className = isW
        ? 'px-2.5 py-1 rounded-md text-xs font-semibold bg-[var(--primary)] text-[var(--primary-foreground)]'
        : 'px-2.5 py-1 rounded-md text-xs text-[var(--muted-foreground)]';
      document.getElementById('btn-fmax').className = !isW
        ? 'px-2.5 py-1 rounded-md text-xs font-semibold bg-[var(--primary)] text-[var(--primary-foreground)]'
        : 'px-2.5 py-1 rounded-md text-xs text-[var(--muted-foreground)]';

      const aspects = [['MF', 'Molecular Function (n=28)'], ['BP', 'Biological Process (n=18)'], ['CC', 'Cellular Component (n=49)']];
      const models = [
        ['CAFA_SequenceHomology_SW', 'Smith-Waterman Homology', 'bg-blue-500'],
        ['Baseline_3mer_kNN', 'Alignment-Free 3-mer k-NN', 'bg-teal-500'],
        ['CAFA_NaivePrior', 'CAFA Naive Frequency Prior', 'bg-amber-500']
      ];
      document.getElementById('s2b-chart').innerHTML = aspects.map(([ak, albl]) => `
        <div class="mb-3 space-y-1">
          <div class="text-xs font-semibold">${albl}</div>
          ${models.map(([mk, mlbl, mcls]) => {
            const val = s2b[mk][ak][metricKey];
            const pct = Math.min(100, Math.round((val / 0.55) * 100));
            return `
              <div class="flex items-center gap-2 text-xs">
                <div class="w-44 truncate text-[var(--muted-foreground)]">${mlbl}</div>
                <div class="flex-1 h-3 rounded-full bg-[var(--background)] border border-[var(--border)] overflow-hidden">
                  <div class="h-full ${mcls}" style="width:${pct}%"></div>
                </div>
                <div class="w-12 text-right font-mono font-semibold">${val.toFixed(3)}</div>
              </div>
            `;
          }).join('')}
        </div>
      `).join('');

      document.getElementById('s2b-note').innerHTML = isW
        ? '<strong>Specificity-Weighted wF_max:</strong> Weights rare catalytic/binding GO terms by Information Content IC(t) = -log2 P(t). Notice how Smith-Waterman beats CAFA Naive nearly <strong>3× on Molecular Function (0.357 vs 0.127)</strong>!'
        : '<strong>Unweighted F_max Warning:</strong> Without IC weighting, CAFA Naive scores <strong>0.322 macro F_max</strong> (tying Smith-Waterman 0.323!) purely by predicting generic high-frequency terms on every protein.';
    }

    // =========================================================================
    // Tab 4: Stage 3A Alanine Scan Explorer
    // =========================================================================
    function initStage3A() {
      const sel = document.getElementById('s3a-select');
      sel.innerHTML = DATA.s3a_examples.map((ex, i) => `
        <option value="${i}">[${ex.cat}] ${ex.name} (${ex.n_mut} X→A muts, ${ex.date})</option>
      `).join('');
      renderStage3AProfile(0);

      const rep = DATA.s3a_report;
      const rows = [
        ['Oracle_NoisyExperimental', 'Wet-Lab Replicate Ceiling'],
        ['Baseline_AA_LookupOnly', 'Static 20-AA Hydrophobicity Table'],
        ['Baseline_LocalWindow9_KD', '9-Residue Local KD Window'],
        ['Null_Uniform', 'Null Flat Baseline']
      ];
      document.getElementById('s3a-calib-bars').innerHTML = rows.filter(([k]) => rep[k]).map(([k, lbl]) => {
        const rawR = rep[k].mean_spearman_rho || 0;
        const ctxR = rep[k].mean_spearman_aa_demeaned || 0;
        return `
          <div class="p-3 rounded-xl bg-[var(--background)] border border-[var(--border)] space-y-1.5 text-xs">
            <div class="font-semibold">${lbl}</div>
            <div class="flex justify-between"><span>Raw Spearman ρ:</span><span class="font-mono font-bold">${rawR.toFixed(3)}</span></div>
            <div class="flex justify-between text-emerald-500"><span>AA-Demeaned Context ρ:</span><span class="font-mono font-bold">${ctxR.toFixed(3)}</span></div>
          </div>
        `;
      }).join('');
    }

    function renderStage3AProfile(idx) {
      const ex = DATA.s3a_examples[Number(idx)];
      const { fg, muted, border } = getThemeColors();
      const n = ex.scores.length;
      const W = Math.max(560, n * 6.2 + 55), H = 220, padL = 42, padB = 30, padT = 18;
      const maxAbs = Math.max(1.5, ...ex.scores.map(v => Math.abs(v)));
      const zeroY = padT + (maxAbs / (maxAbs * 1.35)) * (H - padT - padB);
      const scaleY = (H - padT - padB) / (maxAbs * 1.35);

      let bars = '';
      ex.scores.forEach((sc, i) => {
        const x = padL + i * ((W - padL - 12) / Math.max(1, n));
        const bw = Math.max(2.5, ((W - padL - 12) / Math.max(1, n)) - 1.2);
        const h = Math.abs(sc) * scaleY;
        const y = sc >= 0 ? zeroY - h : zeroY;
        const isCore = ex.core && ex.core[i] === 1;
        const col = isCore ? '#f43f5e' : (sc >= 0 ? '#3b82f6' : '#94a3b8');
        bars += `<rect x="${x.toFixed(1)}" y="${y.toFixed(1)}" width="${bw.toFixed(1)}" height="${Math.max(1, h).toFixed(1)}" fill="${col}"><title>${ex.wt[i]}${ex.pos[i]}A: effect=${sc} ${isCore ? '(3D Buried Core)' : '(Surface/Assay)'}</title></rect>`;
      });

      document.getElementById('s3a-barcode').innerHTML = `
        <svg viewBox="0 0 ${W} ${H}" class="w-full h-auto">
          <line x1="${padL}" y1="${zeroY.toFixed(1)}" x2="${W - 10}" y2="${zeroY.toFixed(1)}" stroke="${muted}" stroke-width="1" />
          <line x1="${padL}" y1="${padT}" x2="${padL}" y2="${H - padB}" stroke="${border}" />
          <text x="12" y="${(H - padB + padT)/2}" fill="${fg}" font-size="12" font-weight="600" text-anchor="middle" transform="rotate(-90 12 ${(H - padB + padT)/2})">Destabilization / Loss</text>
          <text x="${(W + padL)/2}" y="${H - 6}" fill="${muted}" font-size="12" text-anchor="middle">Sequence Position (hover bars for WT→Ala effect; rose = 3D buried core)</text>
          ${bars}
        </svg>
      `;
      document.getElementById('s3a-caption').textContent =
        `${ex.name} (${ex.cat}, ${ex.assay}, published ${ex.date}): showing ${n} wet-lab X→Ala single-mutant measurements.`;
    }

    // =========================================================================
    // Tab 5: Stage 3B 3D Epitope Viewer + Shots-to-Hit Curve
    // =========================================================================
    let currentEpIdx = 9; // Default to 9S83 (Aug 2026 LEDGF PWWP + JUBO4 de novo minibinder!)
    const s3bCanvas = document.getElementById('s3b-canvas');
    const s3bRot = makeRotator(s3bCanvas, drawS3BCanvas);

    function initStage3B() {
      const sel = document.getElementById('s3b-ep-select');
      sel.innerHTML = DATA.s3b_epitopes.map((e, i) => `
        <option value="${i}" ${i === currentEpIdx ? 'selected' : ''}>[${e.tier}] ${e.pdb} — ${e.name}</option>
      `).join('');
      selectEpitope3D(currentEpIdx);

      // Render Pass@k SVG
      const { fg, muted, border } = getThemeColors();
      const p2 = DATA.s3b_report.stage3b_part2_closed_loop_binders;
      const W = 500, H = 205, padL = 42, padR = 16, padT = 16, padB = 34;
      const xP = k => padL + ((k - 1) / 15) * (W - padL - padR);
      const yP = v => padT + (1 - v) * (H - padT - padB);

      function poly(arr, col, dash) {
        const pts = arr.map((v, idx) => `${xP(idx + 1).toFixed(1)},${yP(v).toFixed(1)}`).join(' ');
        return `<polyline fill="none" stroke="${col}" stroke-width="2.6" ${dash ? `stroke-dasharray="${dash}"` : ''} points="${pts}" />`;
      }

      const cl = p2.ClosedLoop_OracleHillClimber_16shot.pass_at_k_curve;
      const ol = p2.OpenLoop_BlindSampling_PassAt16.pass_at_k_curve;
      const ch = p2.Cheat_HeldoutPartnerCopyPaste.pass_at_k_curve;

      document.getElementById('s3b-passk-svg').innerHTML = `
        <svg viewBox="0 0 ${W} ${H}" class="w-full h-auto">
          <line x1="${padL}" y1="${H - padB}" x2="${W - padR}" y2="${H - padB}" stroke="${muted}" />
          <line x1="${padL}" y1="${padT}" x2="${padL}" y2="${H - padB}" stroke="${muted}" />
          ${[1, 4, 8, 12, 16].map(k => `<text x="${xP(k)}" y="${H - 12}" fill="${muted}" font-size="12" text-anchor="middle">Shot ${k}</text>`).join('')}
          ${[0, 0.5, 1.0].map(v => `<text x="${padL - 6}" y="${yP(v) + 4}" fill="${muted}" font-size="12" text-anchor="end">${Math.round(v*100)}%</text>`).join('')}
          ${poly(cl, '#10b981', '')}
          ${poly(ol, '#3b82f6', '6 4')}
          ${poly(ch, '#ef4444', '3 3')}
          <text x="${padL + 10}" y="${padT + 14}" fill="#10b981" font-size="12" font-weight="600">Closed-Loop Oracle Refinement (Hit@16 = 92%)</text>
          <text x="${padL + 10}" y="${padT + 30}" fill="#3b82f6" font-size="12" font-weight="600">Open-Loop Blind Pass@16 (Hit@16 = 33%)</text>
          <text x="${padL + 10}" y="${padT + 46}" fill="#ef4444" font-size="12" font-weight="600">Copy-Paste Native/Known Cheat (0% — Blocked)</text>
        </svg>
      `;

      const protd = [
        ['BindCraft (AF2 Hallucination)', 53.1, 'bg-teal-500'],
        ['PXDesign (2026)', 32.1, 'bg-blue-500'],
        ['BoltzDesign-1', 30.0, 'bg-purple-500'],
        ['BoltzGen', 22.4, 'bg-amber-500'],
        ['RFdiffusion-3', 16.2, 'bg-slate-500']
      ];
      document.getElementById('s3b-protd-bars').innerHTML = protd.map(([nm, meanSr, cls]) => `
        <div class="flex items-center gap-2">
          <div class="w-48 truncate text-[var(--muted-foreground)]">${nm}</div>
          <div class="flex-1 h-3 rounded-full bg-[var(--background)] border border-[var(--border)] overflow-hidden">
            <div class="h-full ${cls}" style="width:${Math.round(meanSr)}%"></div>
          </div>
          <div class="w-14 text-right font-mono font-semibold">${meanSr.toFixed(1)}%</div>
        </div>
      `).join('');
    }

    function selectEpitope3D(idx) {
      currentEpIdx = Number(idx);
      const e = DATA.s3b_epitopes[currentEpIdx];
      document.getElementById('s3b-ep-meta').innerHTML = `
        <div class="p-2.5 rounded-lg bg-[var(--background)] border border-[var(--border)]">
          <div class="text-[var(--muted-foreground)]">Target PDB &amp; Date</div>
          <div class="font-bold">${e.pdb} (${e.date})</div>
        </div>
        <div class="p-2.5 rounded-lg bg-[var(--background)] border border-[var(--border)]">
          <div class="text-[var(--muted-foreground)]">Chain Lengths</div>
          <div class="font-bold">Target ${e.len_t} AA ${e.len_p ? `• Held-Out ${e.len_p} AA` : '(Apo Target)'}</div>
        </div>
        <div class="p-2.5 rounded-lg bg-[var(--background)] border border-[var(--border)] col-span-2 sm:col-span-1">
          <div class="text-[var(--muted-foreground)]">5 Hotspots</div>
          <div class="font-bold truncate">${e.hotspot_aas.join(', ')}</div>
        </div>
      `;
      drawS3BCanvas();
    }

    function drawS3BCanvas() {
      const e = DATA.s3b_epitopes[currentEpIdx];
      if (!e) return;
      const ctx = s3bCanvas.getContext('2d');
      const w = s3bCanvas.width, h = s3bCanvas.height;
      ctx.clearRect(0, 0, w, h);

      const allPts = e.t_ca.concat(e.p_ca || []);
      let cx = 0, cy = 0, cz = 0;
      allPts.forEach(p => { cx += p[0]; cy += p[1]; cz += p[2]; });
      cx /= allPts.length; cy /= allPts.length; cz /= allPts.length;
      let maxR = 1;
      allPts.forEach(p => {
        const d = Math.hypot(p[0] - cx, p[1] - cy, p[2] - cz);
        if (d > maxR) maxR = d;
      });
      const scale = Math.min(w, h) * 0.40 / maxR;

      const tProj = e.t_ca.map(p => project3D(p, [cx, cy, cz], s3bRot.rotX, s3bRot.rotY, scale, w, h));
      ctx.strokeStyle = '#0ea5e9';
      ctx.lineWidth = 3.2;
      ctx.beginPath();
      tProj.forEach((p, i) => { if (i === 0) ctx.moveTo(p[0], p[1]); else ctx.lineTo(p[0], p[1]); });
      ctx.stroke();

      if (e.p_ca && e.p_ca.length > 0) {
        const pProj = e.p_ca.map(p => project3D(p, [cx, cy, cz], s3bRot.rotX, s3bRot.rotY, scale, w, h));
        ctx.strokeStyle = 'rgba(16, 185, 129, 0.80)';
        ctx.lineWidth = 3.2;
        ctx.beginPath();
        pProj.forEach((p, i) => { if (i === 0) ctx.moveTo(p[0], p[1]); else ctx.lineTo(p[0], p[1]); });
        ctx.stroke();
      }

      // Draw glowing red Epitope Hotspot spheres on top
      const hsSet = new Set(e.hotspots);
      tProj.forEach((p, idx0) => {
        if (hsSet.has(idx0 + 1)) {
          ctx.fillStyle = '#f43f5e';
          ctx.beginPath();
          ctx.arc(p[0], p[1], 6.4, 0, Math.PI * 2);
          ctx.fill();
          ctx.strokeStyle = '#ffffff';
          ctx.lineWidth = 1.6;
          ctx.stroke();
        }
      });
    }

    // =========================================================================
    // Tab 6: Zero-Leakage Prompt Inspector
    // =========================================================================
    function initPrompts() {
      const sel = document.getElementById('prompt-select');
      sel.innerHTML = DATA.prompts.map((p, i) => `<option value="${i}">${p.stage}</option>`).join('');
      renderPromptBox(0);
    }

    function renderPromptBox(idx) {
      const p = DATA.prompts[Number(idx)];
      document.getElementById('prompt-pre').textContent = p.prompt;
    }

    function copyCurrentPrompt() {
      const txt = document.getElementById('prompt-pre').textContent;
      navigator.clipboard.writeText(txt);
      const btn = document.getElementById('copy-prompt-btn');
      btn.textContent = 'Copied!';
      setTimeout(() => { btn.textContent = 'Copy Prompt'; }, 1500);
    }

    // Initialize all views
    initS1Pills();
    initStage2();
    initStage3A();
    initStage3B();
    initPrompts();
  </script>
</body>
</html>
"""


def main() -> None:
    payload = build_data_payload()
    html_out = HTML_TEMPLATE.replace("__BENCHMARK_DATA_JSON__", json.dumps(payload, separators=(",", ":")))
    out_repo = REPO / "outputs/figures/protein_without_tools_summary.html"
    out_repo.parent.mkdir(parents=True, exist_ok=True)
    out_repo.write_text(html_out, encoding="utf-8")
    print(f"Built {out_repo} ({out_repo.stat().st_size / 1024:.1f} KB)")


if __name__ == "__main__":
    main()
