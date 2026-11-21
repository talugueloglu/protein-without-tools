#!/usr/bin/env python3
"""Build the academic research paper & benchmark website for ProteinBench.

Inspired by Stanford's `memory.stanford.edu` (Memorilla) and Bo Wang Lab's `BioReason`
(`bowang-lab.github.io/BioReason`) academic project/benchmark portals.

Outputs:
  1. `website/index.html`: Standalone GitHub-Pages-ready academic paper & benchmark website.
  2. Artifact-compatible HTML (using semantic `var(--background)` / `var(--card)` tokens without `:root`
     overrides) for inline preview inside Antigravity.
"""

from __future__ import annotations

import json
from pathlib import Path
import sys
from typing import Any, Dict, List

import numpy as np

REPO = Path("/Users/taluguloglu/.gemini/antigravity/scratch/fable-protein-bench")
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
if str(REPO / "src") not in sys.path:
    sys.path.insert(0, str(REPO / "src"))

from scripts.build_visual_benchmark_html import compile_visual_atlas_data


def load_sample_prompts() -> Dict[str, str]:
    pdir = REPO / "outputs" / "prompts"

    def _first(fn: str, idx: int = 0, max_lines: int = 26) -> str:
        rows = [json.loads(ln) for ln in (pdir / fn).read_text(encoding="utf-8").splitlines() if ln.strip()]
        txt = rows[min(idx, len(rows) - 1)]["prompt"]
        lines = txt.splitlines()
        if len(lines) > max_lines:
            keep_top = max_lines - 8
            txt = "\n".join(
                lines[:keep_top]
                + [f"... [{len(lines) - max_lines} coordinate lines condensed in preview] ..."]
                + lines[-7:]
            )
        return txt

    return {
        "1": _first("stage1a_forward_prompts.jsonl", 0),
        "2": _first("stage1b_inverse_prompts.jsonl", 0),
        "3": _first("stage2a_homology_prompts.jsonl", 60),  # Rung 3 Deep TEA zone pair!
        "4": _first("stage2b_cafa_prompts.jsonl", 0),
        "5": _first("stage3a_alanine_prompts.jsonl", 0),
        "6": _first("stage6_binding_alanine_prompts.jsonl", 34),  # Oct-2026 KRAS x LZTR1!
        "7": _first("stage3b_denovo_prompts.jsonl", 24),  # EF-hand crystal motif scaffolding!
        "8": _first("stage3b_binder_round1_prompts.jsonl", 4),  # BoltzProt-1 hard novel target!
    }


WEBSITE_TEMPLATE = r"""<!DOCTYPE html>
<html lang="en" class="scroll-smooth">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>ProteinBench: Auditing Folding &amp; Binding Physics in Frontier Reasoning Models</title>
  <script src="__TAILWIND_SCRIPT_URL__"></script>
  <style>
__STANDALONE_FALLBACK_VARS__
    body {
      font-family: Inter, -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    }
    .font-academic-serif {
      font-family: Charter, "Bitstream Charter", Georgia, Cambria, "Times New Roman", serif;
    }
    canvas {
      touch-action: none;
    }
    .tab-btn-active {
      background-color: #0f172a;
      color: #ffffff;
      border-color: #0f172a;
    }
  </style>
</head>
<body class="bg-[var(--background)] text-[var(--foreground)] antialiased selection:bg-indigo-500/20">

  <!-- ===================================================================== -->
  <!-- STICKY ACADEMIC NAVBAR (`memory.stanford.edu` style)                  -->
  <!-- ===================================================================== -->
  <nav class="sticky top-0 z-40 backdrop-blur-md bg-[var(--background)]/90 border-b border-[var(--border)]">
    <div class="max-w-5xl mx-auto px-5 py-3 flex items-center justify-between gap-4">
      <a href="#top" class="flex items-center gap-2.5 font-extrabold tracking-tight text-base">
        <span class="w-7 h-7 rounded-lg bg-indigo-600 text-white inline-flex items-center justify-center text-xs font-mono shadow-sm">PB</span>
        <span>ProteinBench</span>
        <span class="px-2 py-0.5 rounded-full bg-[var(--secondary)] text-[var(--muted-foreground)] text-xs font-medium">v1.0</span>
      </a>
      <div class="flex items-center gap-4">
        <div class="hidden md:flex items-center gap-5 text-xs font-semibold text-[var(--muted-foreground)]">
          <a href="#abstract" class="hover:text-[var(--foreground)] transition">Abstract</a>
          <a href="#overview" class="hover:text-[var(--foreground)] transition">Architecture</a>
          <a href="#explorer" class="hover:text-[var(--foreground)] transition">8 Tasks Explorer</a>
          <a href="#results" class="hover:text-[var(--foreground)] transition">Live Leaderboard</a>
          <a href="#dataset" class="hover:text-[var(--foreground)] transition">All 522 Datapoints</a>
          <a href="#quickstart" class="hover:text-[var(--foreground)] transition">Quickstart</a>
          <a href="#citation" class="hover:text-[var(--foreground)] transition">BibTeX</a>
        </div>
        <button id="copy-html-btn" class="px-3 py-1.5 rounded-full bg-indigo-600 text-white hover:bg-indigo-500 text-xs font-semibold shadow-sm transition shrink-0">
          Copy HTML Source
        </button>
      </div>
    </div>
  </nav>

  <main id="top" class="max-w-5xl mx-auto px-5 py-10 md:py-14 space-y-16">

    <!-- ===================================================================== -->
    <!-- 1. PAPER HERO HEADER (`BioReason` + `memory.stanford.edu` fusion)     -->
    <!-- ===================================================================== -->
    <header class="text-center space-y-6">
      <div class="inline-flex items-center gap-2 px-3.5 py-1 rounded-full border border-indigo-500/30 bg-indigo-500/10 text-indigo-500 text-xs font-semibold tracking-wide">
        <span>OPEN BENCHMARK &amp; PREPRINT</span>
        <span>•</span>
        <span>NOVEMBER 2026</span>
      </div>

      <h1 class="text-3xl md:text-4xl lg:text-[2.6rem] font-extrabold tracking-tight leading-[1.15] max-w-4xl mx-auto">
        ProteinBench: From Sequence Memorization to Folding &amp; Binding Physics in Frontier Reasoning Models
      </h1>

      <p class="text-base md:text-lg text-[var(--muted-foreground)] max-w-3xl mx-auto leading-relaxed">
        An eight-task, zero-leakage structural biology evaluation suite comparing raw-sequence biological LLMs, frontier reasoning models, and specialist bio-tools across <strong>No-Code (Weights)</strong> and <strong>With-Code (Air-Gapped NumPy)</strong> regimes.
      </p>

      <!-- Author & Affiliation Block (BioReason academic style) -->
      <div class="space-y-1.5 pt-1">
        <div class="text-sm font-semibold flex flex-wrap items-center justify-center gap-x-5 gap-y-1">
          <span>ProteinBench Research Team<sup>1,2</sup></span>
        </div>
        <div class="text-xs text-[var(--muted-foreground)] flex flex-wrap items-center justify-center gap-x-4 gap-y-1">
          <span><sup>1</sup>Computational Structural Biology &amp; AI</span>
          <span><sup>2</sup>Open-Source Protein Engineering Initiative</span>
        </div>
      </div>

      <!-- Iconic Academic Action Pill Buttons (`BioReason` / `Nerfies` / `Memorilla`) -->
      <div class="flex flex-wrap items-center justify-center gap-2.5 pt-2">
        <a href="#abstract" class="inline-flex items-center gap-2 px-4 py-2 rounded-full bg-slate-900 text-white hover:bg-slate-800 text-xs font-semibold shadow-sm transition">
          <svg class="w-3.5 h-3.5" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/><polyline points="14 2 14 8 20 8"/></svg>
          <span>Paper Overview</span>
        </a>
        <a href="#quickstart" class="inline-flex items-center gap-2 px-4 py-2 rounded-full bg-[var(--card)] border border-[var(--border)] hover:bg-[var(--secondary)] text-xs font-semibold transition">
          <svg class="w-3.5 h-3.5" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polyline points="16 18 22 12 16 6"/><polyline points="8 6 2 12 8 18"/></svg>
          <span>Code &amp; Runners</span>
        </a>
        <a href="#explorer" class="inline-flex items-center gap-2 px-4 py-2 rounded-full bg-indigo-600 text-white hover:bg-indigo-500 text-xs font-semibold shadow-sm transition">
          <svg class="w-3.5 h-3.5" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="12" r="10"/><polygon points="10 8 16 12 10 16 10 8"/></svg>
          <span>Interactive 8-Task Explorer</span>
        </a>
        <a href="#results" class="inline-flex items-center gap-2 px-4 py-2 rounded-full bg-[var(--card)] border border-[var(--border)] hover:bg-[var(--secondary)] text-xs font-semibold transition">
          <svg class="w-3.5 h-3.5" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><line x1="18" y1="20" x2="18" y2="10"/><line x1="12" y1="20" x2="12" y2="4"/><line x1="6" y1="20" x2="6" y2="14"/></svg>
          <span>Live Leaderboard</span>
        </a>
        <a href="#dataset" class="inline-flex items-center gap-2 px-4 py-2 rounded-full bg-emerald-600 text-white hover:bg-emerald-500 text-xs font-semibold shadow-sm transition">
          <svg class="w-3.5 h-3.5" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><ellipse cx="12" cy="5" rx="9" ry="3"/><path d="M21 12c0 1.66-4 3-9 3s-9-1.34-9-3"/><path d="M3 5v14c0 1.66 4 3 9 3s9-1.34 9-3V5"/></svg>
          <span>Browse All 522 Datapoints</span>
        </a>
        <a href="#citation" class="inline-flex items-center gap-2 px-4 py-2 rounded-full bg-[var(--card)] border border-[var(--border)] hover:bg-[var(--secondary)] text-xs font-semibold transition">
          <span>BibTeX</span>
        </a>
      </div>

      <!-- 4-Card Stanford Metric Strip -->
      <div class="grid grid-cols-2 md:grid-cols-4 gap-3 pt-4 text-left">
        <div class="rounded-xl bg-[var(--card)] border border-[var(--border)] p-4">
          <div class="text-2xl font-extrabold text-indigo-500">8 Tasks</div>
          <div class="text-xs font-semibold mt-0.5">Folding, Homology, Function, Ranking &amp; Design</div>
          <div class="text-xs text-[var(--muted-foreground)] mt-0.5">Sequential 1. to 8. suite</div>
        </div>
        <div class="rounded-xl bg-[var(--card)] border border-[var(--border)] p-4">
          <div class="text-2xl font-extrabold text-emerald-500">522 Items</div>
          <div class="text-xs font-semibold mt-0.5">7,414 Wet-Lab Ala Mutants</div>
          <div class="text-xs text-[var(--muted-foreground)] mt-0.5">May–Nov 2026 holdouts</div>
        </div>
        <div class="rounded-xl bg-[var(--card)] border border-[var(--border)] p-4">
          <div class="text-2xl font-extrabold text-sky-500">2 Tracks</div>
          <div class="text-xs font-semibold mt-0.5">No-Code vs. Air-Gapped NumPy</div>
          <div class="text-xs text-[var(--muted-foreground)] mt-0.5">Isolates math vs. physics</div>
        </div>
        <div class="rounded-xl bg-[var(--card)] border border-[var(--border)] p-4">
          <div class="text-2xl font-extrabold text-amber-500">50.0%</div>
          <div class="text-xs font-semibold mt-0.5">1D Lookup Coin-Flip Floor</div>
          <div class="text-xs text-[var(--muted-foreground)] mt-0.5">Via Same-AA neutral decoys</div>
        </div>
      </div>
    </header>

    <!-- ===================================================================== -->
    <!-- 2. TEASER ARCHITECTURE FIGURE (`BioReason` Fig. 1 style)              -->
    <!-- ===================================================================== -->
    <section id="overview" class="space-y-3">
      <div class="rounded-2xl bg-[var(--card)] border border-[var(--border)] p-5 md:p-7 shadow-sm space-y-5">
        <div class="flex items-center justify-between flex-wrap gap-2">
          <div class="text-xs font-bold uppercase tracking-wider text-indigo-500">Figure 1 • Benchmark Architecture &amp; Disentanglement Protocol</div>
          <div class="text-xs text-[var(--muted-foreground)]">Strict ID Stripping • Temporal Holdouts (May–Nov 2026) • Structural Novelty (TM &lt; 0.50)</div>
        </div>

        <div class="grid grid-cols-1 lg:grid-cols-12 gap-4 items-stretch">
          <!-- Column 1: Zero-Leakage Inputs -->
          <div class="lg:col-span-4 rounded-xl border border-[var(--border)] bg-[var(--background)] p-4 flex flex-col justify-between space-y-3">
            <div>
              <div class="text-xs font-bold text-indigo-500 uppercase">A. Anonymized Inputs (522 Tasks)</div>
              <p class="text-xs text-[var(--muted-foreground)] mt-1">All PDB IDs, gene symbols, organism names, and MSAs stripped.</p>
            </div>
            <div class="space-y-2 text-xs">
              <div class="p-2.5 rounded-lg bg-[var(--card)] border border-[var(--border)]">
                <strong>1D Sequence Only:</strong> Tasks <strong>1.</strong> Folding, <strong>3.</strong> Homology, <strong>4.</strong> Function, <strong>5.</strong> Monomer Ala, <strong>6.</strong> Binding Ala
              </div>
              <div class="p-2.5 rounded-lg bg-[var(--card)] border border-[var(--border)]">
                <strong>3D Coordinates (UNK / Patch):</strong> Tasks <strong>2.</strong> Inverse Folding, <strong>7.</strong> Motif Design, <strong>8.</strong> Epitope Binders
              </div>
            </div>
            <div class="text-xs px-2.5 py-1.5 rounded bg-emerald-500/10 text-emerald-500 font-semibold">
              Stratified: Pre-2026 Known PDB vs. Post-Cutoff / Zero-PDB
            </div>
          </div>

          <!-- Column 2: Evaluation Regimes -->
          <div class="lg:col-span-4 rounded-xl border-2 border-indigo-500/30 bg-[var(--background)] p-4 flex flex-col justify-between space-y-3">
            <div>
              <div class="text-xs font-bold text-sky-500 uppercase">B. Three Head-to-Head Regimes</div>
              <p class="text-xs text-[var(--muted-foreground)] mt-1">Separates floating-point arithmetic limits from learned folding physics.</p>
            </div>
            <div class="space-y-2 text-xs">
              <div class="p-2.5 rounded-lg bg-amber-500/10 border border-amber-500/20">
                <strong class="text-amber-500">Track 1 • No-Code (Weights + CoT):</strong> Pure neural weights &amp; reasoning tokens; no external execution.
              </div>
              <div class="p-2.5 rounded-lg bg-sky-500/10 border border-sky-500/20">
                <strong class="text-sky-500">Track 2 • With-Code (Air-Gapped NumPy):</strong> Unlocks 3D coordinate trigonometry (2, 7, 8); cannot fake 1D sequence physics (1, 3, 5, 6).
              </div>
              <div class="p-2.5 rounded-lg bg-teal-500/10 border border-teal-500/20">
                <strong class="text-teal-500">Track 3 • Specialist Bio-Models:</strong> Boltz-2, ESMFold, ProteinMPNN, TEA, ThermoMPNN, BindCraft AF2.
              </div>
            </div>
          </div>

          <!-- Column 3: Physics & Wet-Lab Oracles -->
          <div class="lg:col-span-4 rounded-xl border border-[var(--border)] bg-[var(--background)] p-4 flex flex-col justify-between space-y-3">
            <div>
              <div class="text-xs font-bold text-rose-500 uppercase">C. Anti-Cheat Physics Oracles</div>
              <p class="text-xs text-[var(--muted-foreground)] mt-1">Every prediction is verified against wet-lab assays or 3D co-folding.</p>
            </div>
            <div class="space-y-1.5 text-xs">
              <div class="flex justify-between py-1 border-b border-[var(--border)]"><span>1. Chiral L-BFGS 3D</span><span class="font-mono font-semibold">TM-score</span></div>
              <div class="flex justify-between py-1 border-b border-[var(--border)]"><span>2. ESMFold Refold</span><span class="font-mono font-semibold">scRMSD &lt; 2Å</span></div>
              <div class="flex justify-between py-1 border-b border-[var(--border)]"><span>3. SCOPe Twilight</span><span class="font-mono font-semibold">Rec@10%FPR</span></div>
              <div class="flex justify-between py-1 border-b border-[var(--border)]"><span>4. QuickGO 2026 DAG</span><span class="font-mono font-semibold">IC wF_max</span></div>
              <div class="flex justify-between py-1 border-b border-[var(--border)]"><span>5–6. Same-AA Decoys</span><span class="font-mono font-semibold">NDCG@5 / Win%</span></div>
              <div class="flex justify-between py-1"><span>7–8. AF2 Closed-Loop</span><span class="font-mono font-semibold">Shots-to-Hit</span></div>
            </div>
          </div>
        </div>

        <p class="text-xs text-[var(--muted-foreground)] leading-relaxed pt-1 border-t border-[var(--border)]">
          <strong>Figure 1:</strong> Overview of <strong>ProteinBench</strong>. Unlike retrospective literature QA benchmarks where high scores can arise from pretraining exposure to published papers, ProteinBench combines strict post-May-2026 experimental holdouts, Foldseek <code class="text-xs">TM &lt; 0.50</code> structurally novel folds, monomer-stability-filtered alanine ranking panels with matched same-amino-acid decoys, and closed-loop <code class="text-xs">AlphaFold2</code> binder design campaigns.
        </p>
      </div>
    </section>

    <!-- ===================================================================== -->
    <!-- 3. ABSTRACT & KEY CONTRIBUTIONS (`BioReason` + `Memorilla` style)     -->
    <!-- ===================================================================== -->
    <section id="abstract" class="space-y-8">
      <div class="max-w-3xl mx-auto space-y-3">
        <h2 class="text-2xl font-bold tracking-tight text-center">Abstract</h2>
        <div class="rounded-2xl bg-[var(--card)] border border-[var(--border)] p-6 md:p-7 font-academic-serif text-sm md:text-[15px] leading-relaxed text-[var(--foreground)] space-y-3">
          <p>
            Frontier large language models increasingly incorporate raw biological sequences alongside scientific text during pretraining, raising a fundamental question: <em>do general-purpose reasoning models acquire genuine representations of protein folding thermodynamics and molecular recognition, or do they primarily rely on sequence-to-literature recall and 1D amino-acid heuristics?</em>
          </p>
          <p>
            We introduce <strong>ProteinBench</strong>, a reproducible eight-task evaluation suite spanning forward folding (<em>N</em> = 100), inverse folding on stripped <code class="font-sans text-xs">UNK</code> backbones (<em>N</em> = 100), deep twilight-zone remote homology detection down to &lt;15% sequence identity (<em>N</em> = 120), 2026 CAFA experimental function prediction (<em>N</em> = 60), monomer folding alanine scan ranking (<em>N</em> = 52), monomer-stability-controlled interface binding alanine scan ranking (<em>N</em> = 38), <em>de novo</em> fold and crystal-motif scaffolding (<em>N</em> = 40), and 16-round closed-loop epitope binder design with an <code class="font-sans text-xs">AlphaFold2</code> oracle (<em>N</em> = 12 × 16 turns).
          </p>
          <p>
            By evaluating models across both <strong>No-Code</strong> (pure parametric weights and chain-of-thought) and <strong>With-Code</strong> (air-gapped <code class="font-sans text-xs">Python + NumPy/SciPy</code>) tracks—and stratifying targets by exact 3D structural novelty (<code class="font-sans text-xs">Foldseek TM-score</code>) and pretraining memorization exposure—ProteinBench cleanly separates arithmetic coordinate bottlenecks, literature de-anonymization, and true biophysical generalization against nine specialist structural biology baselines.
          </p>
        </div>
      </div>

      <!-- 3 Key Scientific Design Pillars -->
      <div class="grid grid-cols-1 md:grid-cols-3 gap-4">
        <div class="rounded-xl bg-[var(--card)] border border-[var(--border)] p-5 space-y-2">
          <div class="text-xs font-bold text-indigo-500 uppercase">Pillar 1 • Anti-Memorization</div>
          <h3 class="text-sm font-bold">Sequence De-Anonymization vs. True Novelty</h3>
          <p class="text-xs text-[var(--muted-foreground)] leading-relaxed">
            Stripping PDB IDs is not enough—LLMs recognize famous sequences like Keytruda (<code class="text-xs">5B8C</code>) from memory. ProteinBench explicitly contrasts <strong>Known Pre-2026 PDBs</strong> against <strong>May–Nov 2026 experimental releases</strong> (<code class="text-xs">9YOY</code> TM=0.28, <code class="text-xs">KRAS×LZTR1</code>, <code class="text-xs">9S83</code>) and unpublished synthetic antibodies (<code class="text-xs">PP489×TIGIT</code>).
          </p>
        </div>
        <div class="rounded-xl bg-[var(--card)] border border-[var(--border)] p-5 space-y-2">
          <div class="text-xs font-bold text-emerald-500 uppercase">Pillar 2 • Biophysical Controls</div>
          <h3 class="text-sm font-bold">Same-AA Decoys &amp; Pure Handshake Filtering</h3>
          <p class="text-xs text-[var(--muted-foreground)] leading-relaxed">
            Static 1D lookup tables (<code class="text-xs">Kyte–Doolittle</code>, <code class="text-xs">Bogan–Thorn Trp/Tyr/Arg</code>) inflate raw correlations. Our balanced 18-site ranking panels pair every hotspot with a <strong>same-amino-acid neutral decoy</strong> (forcing 1D tables to a <strong>50.0% coin flip</strong>) and filter out monomer-melting mutations via paired abundance scans.
          </p>
        </div>
        <div class="rounded-xl bg-[var(--card)] border border-[var(--border)] p-5 space-y-2">
          <div class="text-xs font-bold text-sky-500 uppercase">Pillar 3 • Tool Disentanglement</div>
          <h3 class="text-sm font-bold">No-Code Weights vs. Air-Gapped NumPy</h3>
          <p class="text-xs text-[var(--muted-foreground)] leading-relaxed">
            Giving an LLM <code class="text-xs">numpy/scipy</code> with zero network or bio-libraries lets it compute exact 3D virtual-Cβ burial and Ramachandran angles in Tasks <strong>2, 7, 8</strong>—while proving whether code alone can or cannot rescue pure 1D sequence physics in Tasks <strong>1, 3, 5, 6</strong>.
          </p>
        </div>
      </div>
    </section>

    <!-- ===================================================================== -->
    <!-- 4. INTERACTIVE 8-TASK EXPLORER (Click 1..8 to inspect data + prompts) -->
    <!-- ===================================================================== -->
    <section id="explorer" class="space-y-5">
      <div class="space-y-1.5">
        <div class="text-xs font-bold uppercase tracking-wider text-indigo-500">Interactive Benchmark Explorer</div>
        <h2 class="text-2xl font-bold tracking-tight">Explore All 8 Tasks, Real 3D Targets &amp; Verbatim Prompts</h2>
        <p class="text-xs text-[var(--muted-foreground)]">Select any task below to inspect its input modality, grading oracle, live interactive 3D/sequence examples, and copyable zero-leakage LLM prompt.</p>
      </div>

      <!-- Task Selector Tabs (1. to 8.) -->
      <div class="flex flex-wrap gap-2" id="task-tabs"></div>

      <!-- Active Task Detail Card -->
      <div class="rounded-2xl bg-[var(--card)] border border-[var(--border)] p-5 md:p-6 shadow-sm space-y-5" id="task-panel"></div>
    </section>

    <!-- ===================================================================== -->
    <!-- 5. BIOREASON-STYLE MECHANISTIC REASONING TRACE AUDIT                  -->
    <!-- ===================================================================== -->
    <section id="reasoning" class="space-y-5">
      <div class="space-y-1.5">
        <div class="text-xs font-bold uppercase tracking-wider text-emerald-500">Qualitative &amp; Quantitative Error Analysis</div>
        <h2 class="text-2xl font-bold tracking-tight">Inside a Frontier Reasoning Trace: When Memorized Crystal Contacts Mislead Energetics</h2>
        <p class="text-xs text-[var(--muted-foreground)]">
          Like <em>BioReason</em>, ProteinBench audits step-by-step mechanistic rationales—revealing a striking gap between <strong>recalled PDB contact surface area</strong> and <strong>wet-lab binding free energy (ΔΔG_bind)</strong>.
        </p>
      </div>

      <div class="rounded-2xl bg-[var(--card)] border border-[var(--border)] p-5 md:p-6 grid grid-cols-1 lg:grid-cols-12 gap-5 items-start">
        <div class="lg:col-span-7 space-y-3">
          <div class="flex items-center justify-between gap-2 flex-wrap">
            <span class="px-2.5 py-0.5 rounded bg-amber-500/15 text-amber-500 text-xs font-bold">CASE STUDY • PEMBROLIZUMAB-scFv × HUMAN PD-1 (TEST 6)</span>
            <span class="text-xs font-mono text-[var(--muted-foreground)]">CDR NDCG@5 = 0.570 • Same-AA Win = 100%</span>
          </div>
          <div class="rounded-xl bg-[var(--background)] border border-[var(--border)] p-4 space-y-2.5 text-xs leading-relaxed">
            <div class="font-bold text-[var(--foreground)]">Excerpt from Frontier Model Reasoning Trace (Zero Names Given in Prompt):</div>
            <p class="text-[var(--muted-foreground)] font-mono text-xs bg-[var(--card)] p-2.5 rounded border border-[var(--border)]">
              "Protein A (246 AA): Pembrolizumab scFv (VH 1–120, GGGGS×3, VL 136–246, murine parent 409A11). Protein B (122 AA): Human PD-1 IgV (L25–E146, C93S, matching PDBs 5GGS, 5JXE, 5B8C)..."
            </p>
            <div class="grid grid-cols-1 md:grid-cols-2 gap-2.5 pt-1">
              <div class="p-3 rounded-lg border border-emerald-500/30 bg-emerald-500/10 space-y-1">
                <div class="font-bold text-emerald-500">Where Structural Reasoning Succeeded</div>
                <p class="text-[var(--foreground)]">Correctly identified <code class="font-mono">103F→A</code> (#2), <code class="font-mono">98R→A</code> (#3 CDR-H3 kink salt-bridge), and <code class="font-mono">106G→A</code> (#4), while pushing affinity-improving loop rims <code class="font-mono">57G</code>, <code class="font-mono">54S</code>, and <code class="font-mono">30T</code> to the bottom (#7–#9).</p>
              </div>
              <div class="p-3 rounded-lg border border-rose-500/30 bg-rose-500/10 space-y-1">
                <div class="font-bold text-rose-500">The Crystal-Area vs. ΔΔG Trap (#1 Miss)</div>
                <p class="text-[var(--foreground)]">Ranked <code class="font-mono">101Y→A</code> (Kabat H97) as <strong>#1 Catastrophic Hotspot</strong> due to large buried area in <code class="font-mono">5B8C</code>—yet in wet-lab AlphaSeq, <code class="font-mono">101Y→A</code> binds <strong>tighter than WT</strong> (114 nM vs. 166 nM)!</p>
              </div>
            </div>
          </div>
        </div>

        <div class="lg:col-span-5 rounded-xl bg-[var(--background)] border border-[var(--border)] p-4 space-y-2.5">
          <div class="text-xs font-bold uppercase tracking-wider text-[var(--muted-foreground)]">CDR Paratope Rank Audit (Wet-Lab vs. Model)</div>
          <div class="overflow-x-auto">
            <table class="w-full text-xs text-left border-collapse">
              <thead>
                <tr class="border-b border-[var(--border)] text-[var(--muted-foreground)]">
                  <th class="py-1.5 pr-2">Mutant</th>
                  <th class="py-1.5 pr-2">Exp Δlog₁₀Kd</th>
                  <th class="py-1.5 pr-2">True</th>
                  <th class="py-1.5">Model</th>
                </tr>
              </thead>
              <tbody class="font-mono">
                <tr class="border-b border-[var(--border)]/60"><td class="py-1.5">51I→A</td><td class="text-rose-500 font-semibold">+2.54</td><td>#1</td><td class="text-emerald-500 font-bold">#5 ✓</td></tr>
                <tr class="border-b border-[var(--border)]/60"><td class="py-1.5">106G→A</td><td class="text-rose-500 font-semibold">+2.34</td><td>#2</td><td class="text-emerald-500 font-bold">#4 ✓</td></tr>
                <tr class="border-b border-[var(--border)]/60"><td class="py-1.5">60F→A</td><td class="text-rose-500 font-semibold">+1.80</td><td>#3</td><td>#6</td></tr>
                <tr class="border-b border-[var(--border)]/60"><td class="py-1.5">103F→A</td><td class="text-rose-500 font-semibold">+1.59</td><td>#4</td><td class="text-emerald-500 font-bold">#2 ✓</td></tr>
                <tr class="border-b border-[var(--border)]/60"><td class="py-1.5">98R→A</td><td class="text-rose-500 font-semibold">+1.53</td><td>#5</td><td class="text-emerald-500 font-bold">#3 ✓</td></tr>
                <tr class="border-b border-[var(--border)]/60"><td class="py-1.5">30T→A</td><td class="text-sky-500">-0.18</td><td>#6</td><td class="text-emerald-500 font-bold">#9 ✓</td></tr>
                <tr class="border-b border-[var(--border)]/60 bg-rose-500/10"><td class="py-1.5 font-bold">101Y→A</td><td class="text-sky-500 font-semibold">-0.31</td><td>#7</td><td class="text-rose-500 font-bold">#1 ✗</td></tr>
                <tr class="border-b border-[var(--border)]/60"><td class="py-1.5">57G→A</td><td class="text-sky-500">-0.55</td><td>#8</td><td class="text-emerald-500 font-bold">#7 ✓</td></tr>
                <tr><td class="py-1.5">54S→A</td><td class="text-sky-500">-0.61</td><td>#9</td><td class="text-emerald-500 font-bold">#8 ✓</td></tr>
              </tbody>
            </table>
          </div>
          <p class="text-xs text-[var(--muted-foreground)] pt-1">
            Motivates our <strong>Memorization Split</strong>: comparing <code class="text-xs">known_pdb_pre2026</code> (31 complexes) against <code class="text-xs">unseen_or_post2026</code> (<code class="text-xs">PP489×TIGIT</code>, <code class="text-xs">KRAS×LZTR1</code>).
          </p>
        </div>
      </div>
    </section>

    <!-- ===================================================================== -->
    <!-- 6. BENCHMARK SUITE & CALIBRATED BASELINE TABLES (`memory.stanford`)   -->
    <!-- ===================================================================== -->
    <section id="results" class="space-y-6">
      <div class="space-y-1.5">
        <div class="text-xs font-bold uppercase tracking-wider text-indigo-500">Benchmark Suite &amp; Reference Calibration</div>
        <h2 class="text-2xl font-bold tracking-tight">Task Specifications &amp; Calibrated Baseline Separation</h2>
        <p class="text-xs text-[var(--muted-foreground)]">All 8 tasks include verified positive/oracle ceilings, domain specialist baselines, and anti-cheat null floors.</p>
      </div>

      <!-- Live Frontier Evaluation Leaderboard (dynamically rendered from live_summary) -->
      <div id="live-leaderboard-container" class="space-y-4"></div>

      <!-- Table 1: Complete 8-Task Architecture -->
      <div class="rounded-2xl bg-[var(--card)] border border-[var(--border)] overflow-hidden shadow-sm">
        <div class="px-5 py-3.5 border-b border-[var(--border)] flex items-center justify-between flex-wrap gap-2">
          <span class="text-xs font-bold uppercase tracking-wider">Table 1 • Complete ProteinBench 8-Task Suite (522 Targets, 702 Calls / Track)</span>
          <span class="text-xs text-[var(--muted-foreground)]">14 / 14 Unit &amp; Zero-Leakage Tests Passing</span>
        </div>
        <div class="overflow-x-auto">
          <table class="w-full text-xs text-left border-collapse">
            <thead>
              <tr class="border-b border-[var(--border)] bg-[var(--secondary)]/50 text-[var(--muted-foreground)]">
                <th class="py-2.5 px-4 font-bold">#</th>
                <th class="py-2.5 px-4 font-bold">Task</th>
                <th class="py-2.5 px-4 font-bold">Targets &amp; Holdout Design</th>
                <th class="py-2.5 px-4 font-bold">Model Input</th>
                <th class="py-2.5 px-4 font-bold">Specialist Baselines</th>
                <th class="py-2.5 px-4 font-bold">Primary Metrics</th>
              </tr>
            </thead>
            <tbody class="divide-y divide-[var(--border)]">
              <tr>
                <td class="py-2.5 px-4 font-bold">1.</td>
                <td class="py-2.5 px-4 font-semibold">Forward Folding</td>
                <td class="py-2.5 px-4">100 PDBs (85 post-May-2026 incl. 18 novel folds TM&lt;0.52 + 15 controls)</td>
                <td class="py-2.5 px-4">1D AA sequence</td>
                <td class="py-2.5 px-4">Boltz-2, ESMFold, ESM-2</td>
                <td class="py-2.5 px-4 font-mono">TM-score, lDDT, P@L</td>
              </tr>
              <tr>
                <td class="py-2.5 px-4 font-bold">2.</td>
                <td class="py-2.5 px-4 font-semibold">Inverse Folding</td>
                <td class="py-2.5 px-4">100 stripped UNK 3D backbones (same 5 novelty strata)</td>
                <td class="py-2.5 px-4">3D Cα/φ/ψ/burial table</td>
                <td class="py-2.5 px-4">ProteinMPNN + ESMFold</td>
                <td class="py-2.5 px-4 font-mono">scRMSD&lt;2Å %, Core NSR</td>
              </tr>
              <tr>
                <td class="py-2.5 px-4 font-bold">3.</td>
                <td class="py-2.5 px-4 font-semibold">Remote Homology</td>
                <td class="py-2.5 px-4">120 SCOPe pairs (Seq-Easy → Foldseek → Deep TEA → Decoys)</td>
                <td class="py-2.5 px-4">2 AA sequences (A, B)</td>
                <td class="py-2.5 px-4">PickyBinders/tea, Foldseek</td>
                <td class="py-2.5 px-4 font-mono">Recall@10%FPR, AUROC</td>
              </tr>
              <tr>
                <td class="py-2.5 px-4 font-bold">4.</td>
                <td class="py-2.5 px-4 font-semibold">Function (CAFA)</td>
                <td class="py-2.5 px-4">60 2026 QuickGO experimental holdouts (MF, BP, CC)</td>
                <td class="py-2.5 px-4">1D AA sequence</td>
                <td class="py-2.5 px-4">ESM-2 k-NN, Smith–Waterman</td>
                <td class="py-2.5 px-4 font-mono">IC-weighted wF_max</td>
              </tr>
              <tr>
                <td class="py-2.5 px-4 font-bold">5.</td>
                <td class="py-2.5 px-4 font-semibold">Monomer Ala Ranking</td>
                <td class="py-2.5 px-4">52 proteins × 18-site panels (from 5,348 wet-lab X→A mutants)</td>
                <td class="py-2.5 px-4">1D seq + 18 X→A candidates</td>
                <td class="py-2.5 px-4">ThermoMPNN, ESM-2 (3B)</td>
                <td class="py-2.5 px-4 font-mono">NDCG@5, SameAA Win%, ρ</td>
              </tr>
              <tr>
                <td class="py-2.5 px-4 font-bold">6.</td>
                <td class="py-2.5 px-4 font-semibold">Binding Ala Ranking</td>
                <td class="py-2.5 px-4">38 complexes × 8–18 panels (monomer-stability filtered; PDB vs Unseen)</td>
                <td class="py-2.5 px-4">Seq A + Partner B + panel</td>
                <td class="py-2.5 px-4">ESM-2 Complex ΔLLR, MPNN</td>
                <td class="py-2.5 px-4 font-mono">NDCG@5, SameAA Win%, ρ</td>
              </tr>
              <tr>
                <td class="py-2.5 px-4 font-bold">7.</td>
                <td class="py-2.5 px-4 font-semibold">Monomer &amp; Motif Design</td>
                <td class="py-2.5 px-4">40 specs (24 fold-conditioned + 16 crystal-motif scaffolds)</td>
                <td class="py-2.5 px-4">Fold spec or motif 3D Cα</td>
                <td class="py-2.5 px-4">ESM3-open, ProGen2</td>
                <td class="py-2.5 px-4 font-mono">5-Gate % Designable</td>
              </tr>
              <tr>
                <td class="py-2.5 px-4 font-bold">8.</td>
                <td class="py-2.5 px-4 font-semibold">Closed-Loop Binders</td>
                <td class="py-2.5 px-4">12 proven epitopes (RFD4, BoltzProt-1, Aug-2026 crystals) × 16 shots</td>
                <td class="py-2.5 px-4">Target 3D epitope + AF2 loop</td>
                <td class="py-2.5 px-4">BindCraft, RFD3, BoltzGen</td>
                <td class="py-2.5 px-4 font-mono">Hit@16 %, Mean Shots</td>
              </tr>
            </tbody>
          </table>
        </div>
      </div>

      <!-- Calibrated Highlight Cards Across Key Tasks -->
      <div class="grid grid-cols-1 md:grid-cols-3 gap-4">
        <div class="rounded-xl bg-[var(--card)] border border-[var(--border)] p-4 space-y-2">
          <div class="text-xs font-bold text-purple-500">TASK 3 • REMOTE HOMOLOGY STEP-DOWN</div>
          <div class="text-xs text-[var(--muted-foreground)]">Recall @ 10% FPR across SCOPe Rungs 1 → 2 → 3:</div>
          <div class="space-y-1 text-xs font-mono pt-1">
            <div class="flex justify-between"><span>BLOSUM62 SW:</span><span>1.00 → 0.30 → <strong>0.23</strong></span></div>
            <div class="flex justify-between"><span>Foldseek 3Di:</span><span>0.83 → 1.00 → <strong>0.03</strong></span></div>
            <div class="flex justify-between text-emerald-500 font-bold"><span>Rung 3 Target:</span><span>PickyBinders / TEA Zone</span></div>
          </div>
        </div>
        <div class="rounded-xl bg-[var(--card)] border border-[var(--border)] p-4 space-y-2">
          <div class="text-xs font-bold text-emerald-500">TASKS 5 &amp; 6 • ALANINE RANKING CALIBRATION</div>
          <div class="text-xs text-[var(--muted-foreground)]">18-Candidate Balanced Panels (Same-AA Decoys):</div>
          <div class="space-y-1 text-xs font-mono pt-1">
            <div class="flex justify-between"><span>Wet-Lab Replicate:</span><span>NDCG@5 <strong>0.96</strong> • Win <strong>95.9%</strong></span></div>
            <div class="flex justify-between"><span>1D Hydro/Bogan:</span><span>AA-Ctx <strong>0.00</strong> • Win <strong>50.0%</strong></span></div>
            <div class="flex justify-between text-muted"><span>Null Uniform:</span><span>NDCG@5 0.34 • Win 50.0%</span></div>
          </div>
        </div>
        <div class="rounded-xl bg-[var(--card)] border border-[var(--border)] p-4 space-y-2">
          <div class="text-xs font-bold text-teal-500">TASK 8 • CLOSED-LOOP AF2 ORACLE</div>
          <div class="text-xs text-[var(--muted-foreground)]">12 Proven-Bindable Epitopes × 16 Shots:</div>
          <div class="space-y-1 text-xs font-mono pt-1">
            <div class="flex justify-between text-emerald-500 font-bold"><span>Closed-Loop Agent:</span><span>Hit@16 <strong>91.7%</strong> (7.6 shots)</span></div>
            <div class="flex justify-between"><span>Blind Open-Loop:</span><span>Pass@16 <strong>33.3%</strong></span></div>
            <div class="flex justify-between text-rose-500"><span>Copy-Paste Cheat:</span><span>Hit@16 <strong>0.0%</strong> (Blocked)</span></div>
          </div>
        </div>
      </div>
    </section>

    <!-- ===================================================================== -->
    <!-- 6B. ALL 522 DATAPOINTS INTERACTIVE BROWSER (`#dataset`)               -->
    <!-- ===================================================================== -->
    <section id="dataset" class="space-y-4">
      <div class="flex flex-col md:flex-row md:items-end justify-between gap-3">
        <div class="space-y-1">
          <div class="text-xs font-bold uppercase tracking-wider text-emerald-500">Complete Benchmark Explorer</div>
          <h2 class="text-2xl font-bold tracking-tight">Browse All 522 Datapoints (Prompts, Ground Truth &amp; Live Scores)</h2>
          <p class="text-xs text-[var(--muted-foreground)]">Filter by task (Tests 1–8), search by ID, PDB/UniProt/SCOPe, stratum, or sequence, and click any row to inspect its sequence, zero-leakage prompt, ground truth, and live model score.</p>
        </div>
        <button id="download-jsonl-btn" class="px-3.5 py-2 rounded-xl bg-emerald-600 text-white hover:bg-emerald-500 text-xs font-semibold shadow-sm shrink-0 transition">
          Download All 522 Datapoints (.json)
        </button>
      </div>

      <div class="rounded-2xl bg-[var(--card)] border border-[var(--border)] p-4 md:p-5 space-y-4 shadow-sm">
        <div class="flex flex-col md:flex-row md:items-center justify-between gap-3">
          <div id="ds-test-filters" class="flex flex-wrap gap-1.5"></div>
          <input id="ds-search-input" type="text" placeholder="Search ID, PDB, UniProt, organism, stratum, or sequence..." class="w-full md:w-72 px-3 py-1.5 rounded-lg bg-[var(--background)] border border-[var(--border)] text-xs focus:outline-none focus:ring-2 focus:ring-indigo-500/40" />
        </div>

        <div class="grid grid-cols-1 lg:grid-cols-12 gap-4 items-start">
          <div class="lg:col-span-7 space-y-2">
            <div class="overflow-x-auto rounded-xl border border-[var(--border)] max-h-[460px] overflow-y-auto">
              <table class="w-full text-xs text-left border-collapse">
                <thead class="sticky top-0 bg-[var(--secondary)] text-[var(--muted-foreground)] border-b border-[var(--border)]">
                  <tr>
                    <th class="py-2 px-3 font-bold">Test</th>
                    <th class="py-2 px-3 font-bold">ID</th>
                    <th class="py-2 px-3 font-bold">Source / Target</th>
                    <th class="py-2 px-3 font-bold">Stratum</th>
                    <th class="py-2 px-3 font-bold">Len</th>
                    <th class="py-2 px-3 font-bold">Live Score</th>
                  </tr>
                </thead>
                <tbody id="ds-table-body" class="divide-y divide-[var(--border)]"></tbody>
              </table>
            </div>
            <div class="flex items-center justify-between text-xs text-[var(--muted-foreground)] pt-1">
              <span id="ds-count-label"></span>
              <div class="flex items-center gap-2">
                <button id="ds-prev-btn" class="px-2.5 py-1 rounded border border-[var(--border)] hover:bg-[var(--secondary)] disabled:opacity-40">Prev</button>
                <span id="ds-page-label" class="font-mono"></span>
                <button id="ds-next-btn" class="px-2.5 py-1 rounded border border-[var(--border)] hover:bg-[var(--secondary)] disabled:opacity-40">Next</button>
              </div>
            </div>
          </div>

          <div id="ds-detail-pane" class="lg:col-span-5 rounded-xl bg-[var(--background)] border border-[var(--border)] p-4 space-y-3 text-xs"></div>
        </div>
      </div>
    </section>

    <!-- ===================================================================== -->
    <!-- 7. QUICKSTART & REPRODUCIBILITY (`memory.stanford.edu` Section 5)     -->
    <!-- ===================================================================== -->
    <section id="quickstart" class="space-y-4">
      <div class="space-y-1">
        <div class="text-xs font-bold uppercase tracking-wider text-indigo-500">Open-Source Reproducibility</div>
        <h2 class="text-2xl font-bold tracking-tight">Quickstart: Run &amp; Grade in Three Commands</h2>
      </div>

      <div class="rounded-2xl bg-slate-950 text-slate-100 p-5 font-mono text-xs overflow-x-auto space-y-1 leading-relaxed border border-slate-800">
        <div class="text-slate-400"># 1. Verify all mathematical, anti-cheat &amp; zero-leakage tests (14/14 in &lt; 2s)</div>
        <div>.venv/bin/pytest -v</div>
        <div class="pt-2 text-slate-400"># 2. Run any frontier model on CPU-graded Tasks 1, 3, 4, 5, 6 (cached + hard USD cap)</div>
        <div>.venv/bin/python scripts/run_benchmark_llms.py --provider openai --model gpt-6.1-sol --stages 1,3,4,5,6 --thinking-budget 4000</div>
        <div class="pt-2 text-slate-400"># 3. Launch 16-shot Closed-Loop AlphaFold2 Epitope Binder Campaigns &amp; Specialist GPU Baselines</div>
        <div>.venv/bin/python scripts/run_stage3b_closed_loop_llms.py --provider anthropic --model claude-fable-5-2 --mode closed_loop --oracle modal_af2</div>
      </div>
    </section>

    <!-- ===================================================================== -->
    <!-- 8. COPYABLE BIBTEX (`BioReason` Section G)                            -->
    <!-- ===================================================================== -->
    <section id="citation" class="space-y-3">
      <div class="flex items-center justify-between">
        <h2 class="text-xl font-bold tracking-tight">Citation (BibTeX)</h2>
        <button id="copy-bib-btn" class="px-3 py-1.5 rounded-lg bg-[var(--card)] border border-[var(--border)] hover:bg-[var(--secondary)] text-xs font-semibold transition">
          Copy BibTeX
        </button>
      </div>
      <pre id="bibtex-block" class="rounded-2xl bg-[var(--card)] border border-[var(--border)] p-5 font-mono text-xs overflow-x-auto leading-relaxed">@article{proteinbench2026,
  title   = {ProteinBench: From Sequence Memorization to Folding and Binding Physics in Frontier Reasoning Models},
  author  = {ProteinBench Research Team},
  journal = {arXiv preprint},
  year    = {2026},
  note    = {Eight-task zero-leakage structural biology benchmark across No-Code and With-Code regimes}
}</pre>
    </section>

    <footer class="pt-6 border-t border-[var(--border)] flex flex-col sm:flex-row items-center justify-between gap-3 text-xs text-[var(--muted-foreground)]">
      <div><strong>ProteinBench v1.0</strong> • All 522 prompts audited for zero ID/name leakage &amp; biosecurity compliance.</div>
      <div>Layout inspired by <a href="https://memory.stanford.edu/" target="_blank" rel="noopener" class="underline">Stanford Memorilla</a> &amp; <a href="https://bowang-lab.github.io/BioReason/" target="_blank" rel="noopener" class="underline">BioReason</a>.</div>
    </footer>

  </main>

  <script>
    const DATA = __WEBSITE_DATA_JSON__;

    const TASK_META = [
      {
        id: '1',
        short: '1. Forward Folding',
        badge: 'N = 100 PDBs • 5 Novelty Strata',
        title: '1. Forward Folding (1D Sequence → Contacts & Q3 → 3D Structure)',
        input: '1 anonymized amino-acid sequence (no MSA, no templates)',
        eval: 'Predicted Cβ < 8 Å contacts + Q3 folded into 3D via chiral L-BFGS and aligned to native PDB',
        metrics: 'Exact TM-score, lDDT-Cα, Long-Range Contact P@L (|i-j| ≥ 24), Q3 %',
        specialists: 'Boltz-2, ESMFold, ESM-2 Contact Head',
        codeNote: 'Pure Biology Test: NumPy cannot derive 3D folding contacts from a novel 1D string.'
      },
      {
        id: '2',
        short: '2. Inverse Folding',
        badge: 'N = 100 Stripped UNK Backbones',
        title: '2. Inverse Folding (Stripped 3D Backbone → Designed Sequence → ESMFold)',
        input: 'Centered Cα (x,y,z), (φ,ψ) dihedrals, 10Å burial & non-local contacts (all AAs masked)',
        eval: 'Designed sequence refolded with ESMFold and aligned back to target input backbone',
        metrics: 'Self-Consistency scRMSD < 2.0 Å Pass Rate, scTM, Core & Surface NSR %',
        specialists: 'ProteinMPNN (v_48_020)',
        codeNote: 'High Code Lift: With-Code (NumPy) lets models compute exact packing & H-bond geometry.'
      },
      {
        id: '3',
        short: '3. Remote Homology',
        badge: 'N = 120 SCOPe 2.08 Pairs',
        title: '3. Remote Homology Detection (Seq-Easy → Foldseek → Deep TEA Twilight Zone)',
        input: '2 anonymized amino-acid sequences (Protein A & Protein B)',
        eval: 'Calibrated same-superfamily probability graded across 3 homology depths + class/length-matched decoys',
        metrics: 'Rung-Stratified Recall @ 10% FPR (esp. Rung 3 Deep TEA Zone), AUROC, AUPRC',
        specialists: 'PickyBinders/tea (Entropy Representation), Foldseek 3Di, BLOSUM62 SW',
        codeNote: 'Pure Biology Test: Coding Smith–Waterman in Python only reaches 0.23 Recall@10% on Rung 3.'
      },
      {
        id: '4',
        short: '4. Function (CAFA)',
        badge: 'N = 60 2026 QuickGO Holdouts',
        title: '4. Protein Function Prediction (CAFA Experimental Gene Ontology)',
        input: '1 anonymized amino-acid sequence (organism & gene stripped)',
        eval: 'Predicted GO:XXXXXXX terms propagated up the GO DAG against 2026 experimental annotations',
        metrics: 'Information-Content-Weighted wF_max & Standard F_max (MF, BP, CC)',
        specialists: 'ESM-2 Embedding k-NN, Smith–Waterman Homology Transfer, CAFA Naive',
        codeNote: 'Rewarding Specificity: IC-weighted wF_max penalizes generic root-term guessing (Homology 0.36 vs Naive 0.13).'
      },
      {
        id: '5',
        short: '5. Monomer Ala Ranking',
        badge: 'N = 52 Proteins × 18-Site Panels',
        title: '5. Monomer Alanine Scan Ranking (#1 Core Hotspot → #18 Same-AA Surface Decoy)',
        input: '1 anonymized sequence + 18 candidate X→Ala mutations (6 hotspots, 6 moderate, 6 same-AA decoys)',
        eval: 'Submitted #1..#18 ranking compared against wet-lab ΔΔG_fold (MegaScale) & May–Nov 2026 MaveDB DMS',
        metrics: 'NDCG@5, Top-K Hotspot Precision, Same-AA Decoy Win Rate (1D = 50%), AA-Demeaned Rank ρ',
        specialists: 'ThermoMPNN (3D), ESM-2 (650M & 3B) Masked Marginals',
        codeNote: 'Anti-Lookup Design: Pairing buried core Leu against surface Leu forces 1D hydrophobicity scales to 50.0%.'
      },
      {
        id: '6',
        short: '6. Binding Ala Ranking',
        badge: 'N = 38 Complexes • Pure Handshake',
        title: '6. Interface Binding Alanine Scan Ranking (Chain A + Partner B → Binding Hotspots)',
        input: 'Scanned Chain A + Partner B sequences + 8–18 monomer-stable candidate X→Ala mutations',
        eval: 'Ranking graded against SKEMPI v2.0 SPR/ITC, Open-AlphaSeq PD-1/TIGIT Kd, & Oct-2026 KRAS×LZTR1 DMS',
        metrics: 'NDCG@5, Same-AA Decoy Win Rate (Bogan–Thorn 1D = 50%), AA-Demeaned Rank ρ, Memorization Split',
        specialists: 'ESM-2 Complex & Partner-Δ Marginals, ProteinMPNN Complex ΔlogP',
        codeNote: 'Pure Handshake + Memorization Split: Monomer-melting mutants excluded; Known Pre-2026 PDB vs. Unseen/Post-2026.'
      },
      {
        id: '7',
        short: '7. Monomer & Motif',
        badge: 'N = 40 De Novo Design Specs',
        title: '7. De Novo Monomer Generation & Functional Crystal Motif Scaffolding',
        input: 'Topology spec (4-helix, ferredoxin, Rossmann, all-β) OR crystal motif sequence + 3D Cα coordinates',
        eval: '5-gate pipeline: sequence entropy → ESMFold pLDDT & Rg → MPNN scRMSD → TM/motif RMSD → <30% natural SeqID',
        metrics: '5-Gate % Designable Pass Rate (motif Cα RMSD ≤ 1.50 Å, scRMSD ≤ 2.0 Å)',
        specialists: 'ESM3-open, ProGen2, ProteinMPNN Positive Control',
        codeNote: 'Anti-Cheat Hygiene: Blocks homopolymer & EAAAK coiled-coil pLDDT hacks (0.0% cheat pass rate).'
      },
      {
        id: '8',
        short: '8. Epitope Binders',
        badge: '12 Epitopes × 16 AF2 Shots',
        title: '8. Closed-Loop Epitope Binder Design (AlphaFold2 Oracle Shots-to-Hit)',
        input: 'Target sequence + 3D epitope Cα coordinates + 5 hotspots + iterative AF2 oracle scores (ipTM, iPAE, monoRMSD)',
        eval: '16-turn closed-loop campaign on RFD4 classics, June-2026 BoltzProt-1 hard targets, & Aug-2026 binder crystals',
        metrics: 'Hit@16 (% Epitopes Solved) & Mean Shots-to-First-Hit (ipTM ≥ 0.75, iPAE < 10 Å, ≥3/5 hotspots)',
        specialists: 'BindCraft (ColabDesign AF2), RFD3, PXDesign, BoltzGen',
        codeNote: 'Agentic Optimization: Tests whether models actively learn from negative oracle feedback across 16 DMTA rounds.'
      }
    ];

    function attach3D(canvas, drawFn, initRy = 0.65) {
      if (!canvas) return;
      const st = { rx: -0.3, ry: initRy, drag: false, lx: 0, ly: 0 };
      canvas.addEventListener('pointerdown', e => {
        st.drag = true; st.lx = e.clientX; st.ly = e.clientY;
        canvas.setPointerCapture(e.pointerId);
      });
      canvas.addEventListener('pointermove', e => {
        if (!st.drag) return;
        st.ry += (e.clientX - st.lx) * 0.013;
        st.rx += (e.clientY - st.ly) * 0.013;
        st.lx = e.clientX; st.ly = e.clientY;
        drawFn(st);
      });
      canvas.addEventListener('pointerup', () => { st.drag = false; });
      drawFn(st);
    }

    function proj(pt, ctr, rx, ry, sc, w, h) {
      let x = pt[0] - ctr[0], y = pt[1] - ctr[1], z = pt[2] - ctr[2];
      const cy = Math.cos(ry), sy = Math.sin(ry);
      let x1 = x * cy + z * sy, z1 = -x * sy + z * cy;
      const cx = Math.cos(rx), sx = Math.sin(rx);
      let y2 = y * cx - z1 * sx, z2 = y * sx + z1 * cx;
      const p = 240 / (240 + z2 * 0.32);
      return [w / 2 + x1 * sc * p, h / 2 + y2 * sc * p];
    }

    function bounds(pts) {
      let cx = 0, cy = 0, cz = 0;
      pts.forEach(p => { cx += p[0]; cy += p[1]; cz += p[2]; });
      cx /= pts.length; cy /= pts.length; cz /= pts.length;
      let r = 1;
      pts.forEach(p => { const d = Math.hypot(p[0]-cx, p[1]-cy, p[2]-cz); if (d > r) r = d; });
      return { ctr: [cx, cy, cz], r };
    }

    function pillsHtml(panel) {
      const n = panel.length;
      const shown = n <= 12 ? panel : [...panel.slice(0, 5), { dots: true }, ...panel.slice(n - 5)];
      return shown.map(x => {
        if (x.dots) return '<span class="px-1 text-xs text-[var(--muted-foreground)]">⋯</span>';
        const c = x.role === 'hotspot'
          ? 'bg-rose-500/15 text-rose-500 border-rose-500/30'
          : (x.role === 'same_aa_decoy' ? 'bg-sky-500/15 text-sky-500 border-sky-500/30' : 'bg-amber-500/15 text-amber-500 border-amber-500/30');
        return `<span class="px-1.5 py-0.5 rounded border font-mono text-xs font-semibold ${c}">#${x.rank} ${x.mut}</span>`;
      }).join('');
    }

    let activeTask = '1';

    function renderExplorer() {
      const tabsEl = document.getElementById('task-tabs');
      tabsEl.innerHTML = TASK_META.map(t => {
        const isAct = t.id === activeTask;
        const cls = isAct
          ? 'tab-btn-active shadow-sm'
          : 'bg-[var(--card)] text-[var(--foreground)] border-[var(--border)] hover:bg-[var(--secondary)]';
        return `<button data-tid="${t.id}" class="px-3 py-1.5 rounded-full border text-xs font-semibold transition ${cls}">${t.short}</button>`;
      }).join('');

      tabsEl.querySelectorAll('button').forEach(btn => {
        btn.addEventListener('click', () => {
          activeTask = btn.getAttribute('data-tid');
          renderExplorer();
        });
      });

      const meta = TASK_META.find(t => t.id === activeTask);
      const promptTxt = DATA.prompts[activeTask] || '';
      const panelEl = document.getElementById('task-panel');

      let visualHtml = '';
      if (activeTask === '1') {
        const ex = DATA.atlas.s1_examples[0];
        visualHtml = `
          <div class="rounded-xl bg-[var(--background)] border border-[var(--border)] p-4 space-y-2">
            <div class="flex justify-between items-center text-xs">
              <span class="font-bold text-indigo-500">Novel Fold Example: ${ex.pdb} (${ex.len} AA, Pre-Cutoff TM = ${ex.pdb_tm})</span>
              <span class="text-[var(--muted-foreground)]">Drag 3D trace</span>
            </div>
            <div class="grid grid-cols-3 gap-2 items-center">
              <canvas id="exp-c1" width="260" height="165" class="col-span-2 w-full h-[155px] rounded-lg bg-[var(--card)] border border-[var(--border)] cursor-grab"></canvas>
              <div class="col-span-1 flex flex-col items-center p-2 rounded-lg bg-[var(--card)] border border-[var(--border)]">
                <canvas id="exp-cm1" width="100" height="100" class="w-[92px] h-[92px]"></canvas>
                <span class="text-xs text-[var(--muted-foreground)] mt-1">Cβ &lt; 8Å Map</span>
              </div>
            </div>
          </div>`;
      } else if (activeTask === '2') {
        const ex = DATA.atlas.s1b_example;
        visualHtml = `
          <div class="rounded-xl bg-[var(--background)] border border-[var(--border)] p-4 space-y-2">
            <div class="flex justify-between items-center text-xs">
              <span class="font-bold text-indigo-500">Stripped UNK Backbone: ${ex.pdb} (${ex.len} AA, TM = ${ex.pdb_tm})</span>
              <span class="text-[var(--muted-foreground)]">Red = Core • Sky = Surface</span>
            </div>
            <canvas id="exp-c2" width="380" height="170" class="w-full h-[160px] rounded-lg bg-[var(--card)] border border-[var(--border)] cursor-grab"></canvas>
          </div>`;
      } else if (activeTask === '3') {
        const ex = DATA.atlas.s2a_examples[2];
        visualHtml = `
          <div class="rounded-xl bg-[var(--background)] border border-[var(--border)] p-4 space-y-2.5 text-xs">
            <div class="font-bold text-purple-500">${ex.title} (${ex.scop_a} vs. ${ex.scop_b})</div>
            <div class="text-[var(--muted-foreground)]">Shared 3D superfamily at <strong>${ex.seqid}% sequence identity</strong> where both BLAST and Foldseek pdb100 miss:</div>
            <pre class="p-3 rounded-lg bg-[var(--card)] border border-[var(--border)] font-mono text-xs overflow-x-auto">Seq A (${ex.len_a} AA): ${ex.snip_a}...\n               ${ex.snip_mid}\nSeq B (${ex.len_b} AA): ${ex.snip_b}...</pre>
          </div>`;
      } else if (activeTask === '4') {
        const ex = DATA.atlas.s2b_examples[0];
        visualHtml = `
          <div class="rounded-xl bg-[var(--background)] border border-[var(--border)] p-4 space-y-2.5 text-xs">
            <div class="font-bold text-amber-500">${ex.tier_label}</div>
            <div><strong>${ex.tid}</strong> • ${ex.organism} (${ex.len} AA) • Experimental Date: <strong>${ex.qgo_date}</strong></div>
            <div class="flex flex-wrap gap-1.5 pt-1">
              ${ex.top_terms.map(t => `<span class="px-2 py-1 rounded bg-[var(--card)] border border-[var(--border)]"><strong>[${t.aspect}]</strong> ${t.name} <span class="text-emerald-500 font-semibold">(${t.ic} bits)</span></span>`).join('')}
            </div>
          </div>`;
      } else if (activeTask === '5') {
        const ex = DATA.atlas.s3a_examples[0];
        visualHtml = `
          <div class="rounded-xl bg-[var(--background)] border border-[var(--border)] p-4 space-y-2.5 text-xs">
            <div class="font-bold text-rose-500">${ex.name} (Synthetic Zero-Homolog De Novo Miniprotein)</div>
            <canvas id="exp-c5" width="380" height="145" class="w-full h-[135px] rounded-lg bg-[var(--card)] border border-[var(--border)] cursor-grab"></canvas>
            <div class="font-semibold">Ground-Truth 18-Site Ranking (#1 Core Hotspot → #18 Neutral Decoy):</div>
            <div class="flex flex-wrap gap-1">${pillsHtml(ex.ranked_panel)}</div>
          </div>`;
      } else if (activeTask === '6') {
        const ex = DATA.atlas.s6_examples[0];
        const exPost = DATA.atlas.s6_examples[3];
        visualHtml = `
          <div class="rounded-xl bg-[var(--background)] border border-[var(--border)] p-4 space-y-2.5 text-xs">
            <div class="flex justify-between items-center">
              <span class="font-bold text-emerald-500">3D Complex: ${ex.name}</span>
              <span class="text-[var(--muted-foreground)]">Red = Wet-Lab Hotspots</span>
            </div>
            <canvas id="exp-c6" width="380" height="145" class="w-full h-[135px] rounded-lg bg-[var(--card)] border border-[var(--border)] cursor-grab"></canvas>
            <div class="font-semibold pt-1">Post-Cutoff Example (${exPost.name}):</div>
            <div class="flex flex-wrap gap-1">${pillsHtml(exPost.ranked_panel)}</div>
          </div>`;
      } else if (activeTask === '7') {
        const ex = DATA.atlas.s3b_part1_examples[1];
        visualHtml = `
          <div class="rounded-xl bg-[var(--background)] border border-[var(--border)] p-4 space-y-2 text-xs">
            <div class="font-bold text-indigo-500">${ex.title}</div>
            <div class="text-[var(--muted-foreground)]">${ex.subtitle}</div>
            <canvas id="exp-c7" width="380" height="155" class="w-full h-[145px] rounded-lg bg-[var(--card)] border border-[var(--border)] cursor-grab"></canvas>
          </div>`;
      } else if (activeTask === '8') {
        const ex = DATA.atlas.s3b_part2_examples[2];
        visualHtml = `
          <div class="rounded-xl bg-[var(--background)] border border-[var(--border)] p-4 space-y-2 text-xs">
            <div class="font-bold text-teal-500">${ex.subtitle}</div>
            <div class="text-[var(--muted-foreground)]">Sky = Target • Red = 5 Hotspots (${ex.hs_labels.join(', ')}) • Green = Held-Out Minibinder</div>
            <canvas id="exp-c8" width="380" height="155" class="w-full h-[145px] rounded-lg bg-[var(--card)] border border-[var(--border)] cursor-grab"></canvas>
          </div>`;
      }

      panelEl.innerHTML = `
        <div class="flex flex-col md:flex-row md:items-center justify-between gap-2 pb-4 border-b border-[var(--border)]">
          <div>
            <span class="px-2.5 py-0.5 rounded bg-indigo-500/15 text-indigo-500 text-xs font-bold">${meta.badge}</span>
            <h3 class="text-lg font-bold mt-1">${meta.title}</h3>
          </div>
          <button id="copy-prompt-btn" class="px-3 py-1.5 rounded-lg bg-[var(--background)] border border-[var(--border)] hover:bg-[var(--secondary)] text-xs font-semibold shrink-0 transition">
            Copy Sample Prompt
          </button>
        </div>

        <div class="grid grid-cols-1 lg:grid-cols-12 gap-5 items-start">
          <div class="lg:col-span-6 space-y-3 text-xs">
            <div class="grid grid-cols-1 gap-2">
              <div class="p-3 rounded-lg bg-[var(--background)] border border-[var(--border)]"><strong class="text-indigo-500">Model Input:</strong> ${meta.input}</div>
              <div class="p-3 rounded-lg bg-[var(--background)] border border-[var(--border)]"><strong class="text-emerald-500">Evaluation Oracle:</strong> ${meta.eval}</div>
              <div class="p-3 rounded-lg bg-[var(--background)] border border-[var(--border)]"><strong class="text-amber-500">Scoring Metrics:</strong> ${meta.metrics}</div>
              <div class="p-3 rounded-lg bg-[var(--background)] border border-[var(--border)]"><strong class="text-sky-500">No-Code vs. With-Code:</strong> ${meta.codeNote}</div>
            </div>
            ${visualHtml}
          </div>

          <div class="lg:col-span-6 space-y-1.5">
            <div class="text-xs font-bold uppercase tracking-wider text-[var(--muted-foreground)]">Verbatim Zero-Leakage Prompt Preview</div>
            <pre id="active-prompt-pre" class="p-4 rounded-xl bg-[var(--background)] border border-[var(--border)] font-mono text-xs overflow-x-auto whitespace-pre-wrap max-h-[390px] overflow-y-auto leading-relaxed"></pre>
          </div>
        </div>
      `;

      document.getElementById('active-prompt-pre').textContent = promptTxt;
      document.getElementById('copy-prompt-btn').addEventListener('click', ev => {
        navigator.clipboard.writeText(promptTxt);
        ev.currentTarget.textContent = 'Copied Prompt!';
        setTimeout(() => { ev.currentTarget.textContent = 'Copy Sample Prompt'; }, 1600);
      });

      // Attach interactive 3D canvases if present
      if (activeTask === '1') {
        const ex = DATA.atlas.s1_examples[0];
        const c = document.getElementById('exp-c1');
        const b = bounds(ex.ca);
        attach3D(c, st => {
          const ctx = c.getContext('2d'), w = c.width, h = c.height;
          ctx.clearRect(0, 0, w, h);
          const sc = Math.min(w, h) * 0.41 / b.r;
          const P = ex.ca.map(p => proj(p, b.ctr, st.rx, st.ry, sc, w, h));
          ctx.lineWidth = 3.0;
          for (let i = 0; i < P.length - 1; i++) {
            const q = ex.q3[i];
            ctx.strokeStyle = q === 'H' ? '#f43f5e' : (q === 'E' ? '#f59e0b' : '#0ea5e9');
            ctx.beginPath(); ctx.moveTo(P[i][0], P[i][1]); ctx.lineTo(P[i+1][0], P[i+1][1]); ctx.stroke();
          }
        });
        const cm = document.getElementById('exp-cm1'), mctx = cm.getContext('2d'), mw = cm.width, L = ex.shown_len;
        mctx.strokeStyle = '#94a3b855'; mctx.beginPath(); mctx.moveTo(0, 0); mctx.lineTo(mw, mw); mctx.stroke();
        mctx.fillStyle = '#4f46e5';
        ex.contacts.forEach(([i, j]) => {
          mctx.fillRect((i/L)*mw, (j/L)*mw, 2.2, 2.2);
          mctx.fillRect((j/L)*mw, (i/L)*mw, 2.2, 2.2);
        });
      } else if (activeTask === '2') {
        const ex = DATA.atlas.s1b_example, c = document.getElementById('exp-c2'), b = bounds(ex.ca);
        const med = [...ex.nbrs].sort((x,y)=>x-y)[Math.floor(ex.nbrs.length * 0.58)];
        attach3D(c, st => {
          const ctx = c.getContext('2d'), w = c.width, h = c.height;
          ctx.clearRect(0, 0, w, h);
          const sc = Math.min(w, h) * 0.42 / b.r;
          const P = ex.ca.map(p => proj(p, b.ctr, st.rx, st.ry, sc, w, h));
          ctx.strokeStyle = '#94a3b888'; ctx.lineWidth = 2.6;
          ctx.beginPath(); P.forEach((p,i)=>{ if(i===0) ctx.moveTo(p[0],p[1]); else ctx.lineTo(p[0],p[1]); }); ctx.stroke();
          P.forEach((p,i)=>{
            const core = ex.nbrs[i] >= med;
            ctx.fillStyle = core ? '#f43f5e' : '#0ea5e9';
            ctx.beginPath(); ctx.arc(p[0], p[1], core ? 3.8 : 2.5, 0, Math.PI*2); ctx.fill();
          });
        });
      } else if (activeTask === '5') {
        const ex = DATA.atlas.s3a_examples[0], c = document.getElementById('exp-c5'), b = bounds(ex.ca);
        const pmap = {}; ex.full_pos.forEach((p1, k) => { pmap[p1] = ex.full_destab[k]; });
        attach3D(c, st => {
          const ctx = c.getContext('2d'), w = c.width, h = c.height;
          ctx.clearRect(0, 0, w, h);
          const sc = Math.min(w, h) * 0.42 / b.r;
          const P = ex.ca.map(p => proj(p, b.ctr, st.rx, st.ry, sc, w, h));
          ctx.lineWidth = 3.0;
          for (let i = 0; i < P.length - 1; i++) {
            const v = pmap[i+1] || 0;
            ctx.strokeStyle = v > 1.0 ? '#f43f5e' : (v > 0.4 ? '#fb923c' : '#0ea5e9');
            ctx.beginPath(); ctx.moveTo(P[i][0], P[i][1]); ctx.lineTo(P[i+1][0], P[i+1][1]); ctx.stroke();
          }
        });
      } else if (activeTask === '6') {
        const ex = DATA.atlas.s6_examples[0], c = document.getElementById('exp-c6');
        const allP = ex.ca_a.concat(ex.ca_b), b = bounds(allP);
        const lmap = {}; ex.full_pos.forEach((p1, k) => { lmap[p1] = ex.full_loss[k]; });
        attach3D(c, st => {
          const ctx = c.getContext('2d'), w = c.width, h = c.height;
          ctx.clearRect(0, 0, w, h);
          const sc = Math.min(w, h) * 0.42 / b.r;
          const PA = ex.ca_a.map(p => proj(p, b.ctr, st.rx, st.ry, sc, w, h));
          const PB = ex.ca_b.map(p => proj(p, b.ctr, st.rx, st.ry, sc, w, h));
          ctx.strokeStyle = '#10b981'; ctx.lineWidth = 2.6;
          ctx.beginPath(); PB.forEach((p,i)=>{ if(i===0) ctx.moveTo(p[0],p[1]); else ctx.lineTo(p[0],p[1]); }); ctx.stroke();
          ctx.strokeStyle = '#0ea5e9'; ctx.lineWidth = 2.9;
          ctx.beginPath(); PA.forEach((p,i)=>{ if(i===0) ctx.moveTo(p[0],p[1]); else ctx.lineTo(p[0],p[1]); }); ctx.stroke();
          PA.forEach((p,i)=>{
            if ((lmap[i+1] || 0) >= 1.2) {
              ctx.fillStyle = '#f43f5e'; ctx.beginPath(); ctx.arc(p[0], p[1], 4.8, 0, Math.PI*2); ctx.fill();
            }
          });
        });
      } else if (activeTask === '7') {
        const ex = DATA.atlas.s3b_part1_examples[1], c = document.getElementById('exp-c7'), b = bounds(ex.ca);
        const hi = new Set(ex.highlight_0idx || []);
        attach3D(c, st => {
          const ctx = c.getContext('2d'), w = c.width, h = c.height;
          ctx.clearRect(0, 0, w, h);
          const sc = Math.min(w, h) * 0.42 / b.r;
          const P = ex.ca.map(p => proj(p, b.ctr, st.rx, st.ry, sc, w, h));
          for (let i = 0; i < P.length - 1; i++) {
            const m = hi.has(i) && hi.has(i+1);
            ctx.strokeStyle = m ? '#eab308' : '#6366f1'; ctx.lineWidth = m ? 4.4 : 2.8;
            ctx.beginPath(); ctx.moveTo(P[i][0], P[i][1]); ctx.lineTo(P[i+1][0], P[i+1][1]); ctx.stroke();
          }
        });
      } else if (activeTask === '8') {
        const ex = DATA.atlas.s3b_part2_examples[2], c = document.getElementById('exp-c8');
        const b = bounds(ex.t_ca.concat(ex.p_ca || [])), hs = new Set(ex.hotspots);
        attach3D(c, st => {
          const ctx = c.getContext('2d'), w = c.width, h = c.height;
          ctx.clearRect(0, 0, w, h);
          const sc = Math.min(w, h) * 0.42 / b.r;
          const PT = ex.t_ca.map(p => proj(p, b.ctr, st.rx, st.ry, sc, w, h));
          ctx.strokeStyle = '#0ea5e9'; ctx.lineWidth = 2.8;
          ctx.beginPath(); PT.forEach((p,i)=>{ if(i===0) ctx.moveTo(p[0],p[1]); else ctx.lineTo(p[0],p[1]); }); ctx.stroke();
          if (ex.p_ca && ex.p_ca.length) {
            const PP = ex.p_ca.map(p => proj(p, b.ctr, st.rx, st.ry, sc, w, h));
            ctx.strokeStyle = '#10b981'; ctx.lineWidth = 3.2;
            ctx.beginPath(); PP.forEach((p,i)=>{ if(i===0) ctx.moveTo(p[0],p[1]); else ctx.lineTo(p[0],p[1]); }); ctx.stroke();
          }
          PT.forEach((p,i)=>{
            if (hs.has(i+1)) { ctx.fillStyle = '#f43f5e'; ctx.beginPath(); ctx.arc(p[0], p[1], 5.2, 0, Math.PI*2); ctx.fill(); }
          });
        });
      }
    }

    document.getElementById('copy-bib-btn').addEventListener('click', ev => {
      navigator.clipboard.writeText(document.getElementById('bibtex-block').textContent);
      ev.currentTarget.textContent = 'Copied!';
      setTimeout(() => { ev.currentTarget.textContent = 'Copy BibTeX'; }, 1600);
    });

    document.getElementById('copy-html-btn').addEventListener('click', ev => {
      const rootVars = ':root{--background:#f8fafc;--foreground:#0f172a;--card:#ffffff;--border:#e2e8f0;--secondary:#f1f5f9;--muted-foreground:#475569;}';
      let raw = '<!DOCTYPE html>\n' + document.documentElement.outerHTML
        .replace('https://www.gstatic.com/antigravity/web/dev/tailwindcss.min.js', 'https://cdn.tailwindcss.com');
      if (!raw.includes('--background:')) {
        raw = raw.replace('<style>', '<style>\n    ' + rootVars);
      }
      navigator.clipboard.writeText(raw);
      ev.currentTarget.textContent = 'Copied Full HTML!';
      setTimeout(() => { ev.currentTarget.textContent = 'Copy HTML Source'; }, 1800);
    });

    function prettyModel(m) {
      return m
        .replace('gpt-6.2-sol_think4096', 'GPT-6.2 Sol (High Reasoning)')
        .replace('gpt-6.1-sol_think4096', 'GPT-6.1 Sol (High Reasoning)')
        .replace('gpt-6.2-astra_think4096', 'GPT-6.2 Astra (High Reasoning)')
        .replace('_think4096', ' (High)');
    }

    function renderLiveLeaderboard() {
      const el = document.getElementById('live-leaderboard-container');
      const S = DATA.live_summary;
      if (!el || !S) return;

      const t1 = {}; (S.test1_forward_folding || []).forEach(x => { t1[x.model] = x; });
      const t3 = S.test3_remote_homology || {};
      const t4 = S.test4_cafa_function || {};
      const t5 = {}; (S.test5_monomer_alanine_ranking || []).forEach(x => { t5[x.model] = x; });
      const t6 = {}; (S.test6_binding_alanine_ranking || []).forEach(x => { t6[x.model] = x; });

      const llmModels = Object.keys(S.telemetry || {});
      if (!llmModels.length) return;

      const rowsHtml = llmModels.map(m => {
        const tel = S.telemetry[m] || {};
        const f1 = t1[m] || { overall: {}, splits: {} };
        const f3 = t3[m] || { by_rung: {} };
        const f4 = t4[m] || {};
        const f5 = (t5[m] || {}).overall || {};
        const f6Obj = t6[m] || { overall: {}, by_memorization_stratum: {} };
        const f6 = f6Obj.overall || {};
        const f6Known = (f6Obj.by_memorization_stratum || {}).known_pdb_pre2026 || {};
        const f6Unseen = (f6Obj.by_memorization_stratum || {}).unseen_or_post2026 || {};

        const tmAll = (f1.overall.tm_score || 0).toFixed(3);
        const tmPre = ((f1.splits.pre_cutoff || {}).tm_score || 0).toFixed(3);
        const tmNov = ((f1.splits.post_cutoff_struct_novel || {}).tm_score || 0).toFixed(3);
        const auc3 = (f3.overall_auroc || 0).toFixed(3);
        const r3Tea = (((f3.by_rung || {})['3_deep_tea_zone'] || {}).recall_at_fpr10 || 0).toFixed(2);
        const wf4 = (f4.macro_wfmax || 0).toFixed(3);
        const n5 = (f5.ndcg_at_5 || 0).toFixed(3);
        const w5 = (((f5.same_aa_win_rate || 0) * 100)).toFixed(1) + '%';
        const n6 = (f6.ndcg_at_5 || 0).toFixed(3);
        const w6 = (((f6.same_aa_win_rate || 0) * 100)).toFixed(1) + '%';
        const memSplit = `${(f6Known.ndcg_at_5 || 0).toFixed(2)} / ${(f6Unseen.ndcg_at_5 || 0).toFixed(2)}`;

        return `
          <tr class="bg-indigo-500/5 font-semibold">
            <td class="py-3 px-4">
              <div class="font-bold text-indigo-600">${prettyModel(m)}</div>
              <div class="text-[11px] text-[var(--muted-foreground)] font-normal">${tel.total_calls || 370} calls • $${(tel.total_est_cost_usd || 0).toFixed(2)} • ${Math.round((tel.total_reasoning_tokens || 0)/1000)}k reasoning tok</div>
            </td>
            <td class="py-3 px-4 font-mono">${tmAll} <span class="text-[11px] text-[var(--muted-foreground)]">(${tmPre}→${tmNov})</span></td>
            <td class="py-3 px-4 font-mono">${auc3} <span class="text-[11px] text-[var(--muted-foreground)]">(TEA ${r3Tea})</span></td>
            <td class="py-3 px-4 font-mono">${wf4}</td>
            <td class="py-3 px-4 font-mono">${n5} <span class="text-[11px] text-[var(--muted-foreground)]">(${w5})</span></td>
            <td class="py-3 px-4 font-mono">${n6} <span class="text-[11px] text-[var(--muted-foreground)]">(${w6})</span></td>
            <td class="py-3 px-4 font-mono">${memSplit}</td>
          </tr>`;
      }).join('');

      // Reference baseline comparison rows
      const h1 = (t1['Heuristic_1D_LocalOnly'] || { overall: {}, splits: {} });
      const sw3 = t3['BLOSUM62_SmithWaterman'] || { by_rung: {} };
      const fs3 = t3['Foldseek_3Di_Structure'] || { by_rung: {} };
      const sw4 = t4['CAFA_SequenceHomology_SW'] || {};
      const lk5 = (t5['Baseline_AA_LookupOnly'] || {}).overall || {};
      const bt6Obj = t6['Baseline_BoganThorn_1D'] || { overall: {}, by_memorization_stratum: {} };
      const or6Obj = t6['Oracle_3D_Interface_Contacts'] || { overall: {}, by_memorization_stratum: {} };

      el.innerHTML = `
        <div class="rounded-2xl bg-[var(--card)] border-2 border-indigo-500/40 overflow-hidden shadow-sm">
          <div class="px-5 py-3.5 border-b border-[var(--border)] bg-indigo-500/10 flex items-center justify-between flex-wrap gap-2">
            <span class="text-xs font-bold uppercase tracking-wider text-indigo-600">Live Evaluation Results • Local Suite (Tests 1, 3, 4, 5, 6 — 370 Tasks, Zero-Code Weights Track)</span>
            <span class="text-xs font-mono text-[var(--muted-foreground)]">366/370 Answered (4 Abstentions on T6) • Zero Fallbacks</span>
          </div>
          <div class="overflow-x-auto">
            <table class="w-full text-xs text-left border-collapse">
              <thead>
                <tr class="border-b border-[var(--border)] bg-[var(--secondary)]/60 text-[var(--muted-foreground)]">
                  <th class="py-2.5 px-4 font-bold">Model / Reference Baseline</th>
                  <th class="py-2.5 px-4 font-bold">1. Fold TM (Pre→Novel)</th>
                  <th class="py-2.5 px-4 font-bold">3. Homology AUC (TEA Rec)</th>
                  <th class="py-2.5 px-4 font-bold">4. CAFA wF_max</th>
                  <th class="py-2.5 px-4 font-bold">5. Mono Ala NDCG@5 (Win%)</th>
                  <th class="py-2.5 px-4 font-bold">6. Bind Ala NDCG@5 (Win%)</th>
                  <th class="py-2.5 px-4 font-bold">6b. Known PDB / Unseen</th>
                </tr>
              </thead>
              <tbody class="divide-y divide-[var(--border)]">
                ${rowsHtml}
                <tr>
                  <td class="py-2.5 px-4 font-semibold text-[var(--muted-foreground)]">3D Structure / Homology Reference (Foldseek / Crystal / SW)</td>
                  <td class="py-2.5 px-4 font-mono">${((t1['NoisyOracle_50pctPrec']||{}).overall||{}).tm_score?.toFixed(3) || '—'} <span class="text-[11px] text-[var(--muted-foreground)]">(50%-Prec)</span></td>
                  <td class="py-2.5 px-4 font-mono">${(fs3.overall_auroc||0).toFixed(3)} <span class="text-[11px] text-[var(--muted-foreground)]">(TEA ${(((fs3.by_rung||{})['3_deep_tea_zone']||{}).recall_at_fpr10||0).toFixed(2)})</span></td>
                  <td class="py-2.5 px-4 font-mono">${(sw4.macro_wfmax||0).toFixed(3)} <span class="text-[11px] text-[var(--muted-foreground)]">(SW)</span></td>
                  <td class="py-2.5 px-4 font-mono">${(((t5['Baseline_LocalWindowKD']||{}).overall||{}).ndcg_at_5||0).toFixed(3)} <span class="text-[11px] text-[var(--muted-foreground)]">(KD Win)</span></td>
                  <td class="py-2.5 px-4 font-mono">${((or6Obj.overall||{}).ndcg_at_5||0).toFixed(3)} <span class="text-[11px] text-[var(--muted-foreground)]">(3D Contacts)</span></td>
                  <td class="py-2.5 px-4 font-mono">${(((or6Obj.by_memorization_stratum||{}).known_pdb_pre2026||{}).ndcg_at_5||0).toFixed(2)} / ${(((or6Obj.by_memorization_stratum||{}).unseen_or_post2026||{}).ndcg_at_5||0).toFixed(2)}</td>
                </tr>
                <tr>
                  <td class="py-2.5 px-4 text-[var(--muted-foreground)]">1D Sequence / Hydrophobicity Lookup Baseline</td>
                  <td class="py-2.5 px-4 font-mono">${(h1.overall.tm_score||0).toFixed(3)}</td>
                  <td class="py-2.5 px-4 font-mono">${(sw3.overall_auroc||0).toFixed(3)} <span class="text-[11px] text-[var(--muted-foreground)]">(TEA ${(((sw3.by_rung||{})['3_deep_tea_zone']||{}).recall_at_fpr10||0).toFixed(2)})</span></td>
                  <td class="py-2.5 px-4 font-mono">${((t4['CAFA_NaivePrior']||{}).macro_wfmax||0).toFixed(3)} <span class="text-[11px] text-[var(--muted-foreground)]">(Naive)</span></td>
                  <td class="py-2.5 px-4 font-mono">${(lk5.ndcg_at_5||0).toFixed(3)} <span class="text-[11px] text-[var(--muted-foreground)]">(50.0%)</span></td>
                  <td class="py-2.5 px-4 font-mono">${((bt6Obj.overall||{}).ndcg_at_5||0).toFixed(3)} <span class="text-[11px] text-[var(--muted-foreground)]">(50.0%)</span></td>
                  <td class="py-2.5 px-4 font-mono">${(((bt6Obj.by_memorization_stratum||{}).known_pdb_pre2026||{}).ndcg_at_5||0).toFixed(2)} / ${(((bt6Obj.by_memorization_stratum||{}).unseen_or_post2026||{}).ndcg_at_5||0).toFixed(2)}</td>
                </tr>
              </tbody>
            </table>
          </div>
        </div>
        ${renderFoldingDesignLeaderboard(S, llmModels)}`;
    }

    function renderFoldingDesignLeaderboard(S, llmModels) {
      const G = DATA.gpu_summary;
      if (!G) return '';
      const t2 = G.test2 || {};
      const t7 = G.test7 || {};
      const t8 = G.test8 || {};
      const rows278 = llmModels.map(m => {
        const g2 = (t2[m] || {}).overall || {};
        const g7 = t7[m] || {};
        const g8 = t8[m] || {};
        const sctm = g2.sc_tm_score !== undefined ? g2.sc_tm_score.toFixed(3) : '—';
        const pass2a = g2.designable_pass_rate_2A !== undefined ? (g2.designable_pass_rate_2A * 100).toFixed(1) + '%' : '—';
        const plddt7 = g7.mean_plddt !== undefined ? g7.mean_plddt.toFixed(1) : '—';
        const sc7 = g7.gate3_cross_oracle_sc_rate !== undefined ? (g7.gate3_cross_oracle_sc_rate * 100).toFixed(1) + '%' : '—';
        const mrmsd7 = g7.mean_motif_ca_rmsd !== undefined ? g7.mean_motif_ca_rmsd.toFixed(2) + 'Å' : '—';
        const hit8 = g8.hit_rate !== undefined ? (g8.hit_rate * 100).toFixed(1) + '%' : 'Running';
        const iptm8 = g8.mean_best_iptm !== undefined ? g8.mean_best_iptm.toFixed(3) : '—';
        const gain8 = g8.mean_closed_loop_gain !== undefined ? '+' + g8.mean_closed_loop_gain.toFixed(3) : '—';
        return `
          <tr class="bg-rose-500/5 font-semibold">
            <td class="py-3 px-4 font-bold text-rose-600">${prettyModel(m)}</td>
            <td class="py-3 px-4 font-mono">${sctm} <span class="text-[11px] text-[var(--muted-foreground)]">(&lt;2Å: ${pass2a})</span></td>
            <td class="py-3 px-4 font-mono">${plddt7} <span class="text-[11px] text-[var(--muted-foreground)]">(OF3-SC: ${sc7}, Motif: ${mrmsd7})</span></td>
            <td class="py-3 px-4 font-mono">${hit8} <span class="text-[11px] text-[var(--muted-foreground)]">(ipTM ${iptm8}, Δ=${gain8})</span></td>
          </tr>`;
      }).join('');
      const wt2 = ((t2['control_wildtype_native'] || {}).overall) || {};
      return `
        <div class="rounded-2xl bg-[var(--card)] border-2 border-rose-500/35 overflow-hidden shadow-sm">
          <div class="px-5 py-3.5 border-b border-[var(--border)] bg-rose-500/10 flex items-center justify-between flex-wrap gap-2">
            <span class="text-xs font-bold uppercase tracking-wider text-rose-600">Live 3D Folding &amp; Closed-Loop Design Suite (Tests 2, 7, 8 — ESMFold-2, OpenFold3 &amp; Official AlphaFold2)</span>
            <span class="text-xs font-mono text-[var(--muted-foreground)]">152 Tasks • 16-Shot Closed-Loop AF2 Binder Campaigns</span>
          </div>
          <div class="overflow-x-auto">
            <table class="w-full text-xs text-left border-collapse">
              <thead>
                <tr class="border-b border-[var(--border)] bg-[var(--secondary)]/60 text-[var(--muted-foreground)]">
                  <th class="py-2.5 px-4 font-bold">Model / Reference Control</th>
                  <th class="py-2.5 px-4 font-bold">2. Inverse Fold ESMFold-2 scTM (&lt;2Å%)</th>
                  <th class="py-2.5 px-4 font-bold">7. De Novo/Motif pLDDT (Cross-SC / Motif RMSD)</th>
                  <th class="py-2.5 px-4 font-bold">8. Closed-Loop AF2 Binder Hit@16 (ipTM / ΔGain)</th>
                </tr>
              </thead>
              <tbody class="divide-y divide-[var(--border)]">
                ${rows278}
                <tr>
                  <td class="py-2.5 px-4 text-[var(--muted-foreground)]">Wild-Type Native Sequence Control (ESMFold-2 Oracle Ceiling)</td>
                  <td class="py-2.5 px-4 font-mono">${wt2.sc_tm_score !== undefined ? wt2.sc_tm_score.toFixed(3) : '—'} <span class="text-[11px] text-[var(--muted-foreground)]">(&lt;2Å: ${wt2.designable_pass_rate_2A !== undefined ? (wt2.designable_pass_rate_2A*100).toFixed(1)+'%' : '—'})</span></td>
                  <td class="py-2.5 px-4 font-mono text-[var(--muted-foreground)]">—</td>
                  <td class="py-2.5 px-4 font-mono text-[var(--muted-foreground)]">—</td>
                </tr>
              </tbody>
            </table>
          </div>
        </div>`;
    }

    // --- All 522 Datapoints Interactive Browser ---
    let dsFilterTest = 0; // 0 = All
    let dsQuery = '';
    let dsPage = 0;
    let dsSelectedKey = null;
    const DS_PAGE_SIZE = 18;

    function formatRowScore(row) {
      const sc = row.sc || {};
      const keys = Object.keys(sc);
      if (!keys.length) return '<span class="text-[var(--muted-foreground)]">Pending</span>';
      const v = sc[keys[0]];
      if (v.tm_score !== undefined) return `<span class="font-mono font-semibold text-indigo-600">TM ${v.tm_score.toFixed(3)}</span>`;
      if (v.sc_tm_score !== undefined && v.sc_tm_score !== null) return `<span class="font-mono font-semibold text-sky-600">scTM ${v.sc_tm_score.toFixed(3)}</span>`;
      if (v.plddt_mean !== undefined && v.plddt_mean !== null) return `<span class="font-mono font-semibold text-teal-600">pLDDT ${Number(v.plddt_mean).toFixed(1)}</span>`;
      if (v.best_iptm_on_epitope !== undefined) return `<span class="font-mono font-semibold text-rose-600">ipTM ${Number(v.best_iptm_on_epitope).toFixed(2)}</span>`;
      if (v.ndcg_at_5 !== undefined) return `<span class="font-mono font-semibold text-emerald-600">NDCG ${v.ndcg_at_5.toFixed(2)}</span>`;
      if (v.raw !== undefined) {
        const m = v.raw.match(/"prob_homolog"\s*:\s*([0-9.]+)/);
        if (m) return `<span class="font-mono font-semibold text-purple-600">p=${Number(m[1]).toFixed(2)}</span>`;
        return `<span class="font-mono font-semibold text-amber-600">Scored ✓</span>`;
      }
      return '—';
    }

    function renderDatasetBrowser() {
      const rows = DATA.datapoints || [];
      const filterEl = document.getElementById('ds-test-filters');
      if (!filterEl) return;

      const filterBtns = [{ t: 0, lbl: `All (${rows.length})` }].concat(
        [1,2,3,4,5,6,7,8].map(n => ({ t: n, lbl: `T${n} (${rows.filter(r => r.t === n).length})` }))
      );
      filterEl.innerHTML = filterBtns.map(b => {
        const act = b.t === dsFilterTest ? 'tab-btn-active' : 'bg-[var(--background)] border-[var(--border)] hover:bg-[var(--secondary)]';
        return `<button data-dst="${b.t}" class="px-2.5 py-1 rounded-full border text-xs font-semibold transition ${act}">${b.lbl}</button>`;
      }).join('');
      filterEl.querySelectorAll('button').forEach(btn => {
        btn.addEventListener('click', () => {
          dsFilterTest = Number(btn.getAttribute('data-dst'));
          dsPage = 0;
          renderDatasetBrowser();
        });
      });

      const q = dsQuery.trim().toLowerCase();
      const filtered = rows.filter(r => {
        if (dsFilterTest !== 0 && r.t !== dsFilterTest) return false;
        if (!q) return true;
        return (
          r.id.toLowerCase().includes(q) ||
          (r.src || '').toLowerCase().includes(q) ||
          (r.desc || '').toLowerCase().includes(q) ||
          (r.st || '').toLowerCase().includes(q) ||
          (r.seq || '').toLowerCase().includes(q)
        );
      });

      const totalPages = Math.max(1, Math.ceil(filtered.length / DS_PAGE_SIZE));
      if (dsPage >= totalPages) dsPage = totalPages - 1;
      const pageRows = filtered.slice(dsPage * DS_PAGE_SIZE, (dsPage + 1) * DS_PAGE_SIZE);

      if (!dsSelectedKey || !filtered.some(r => `${r.t}:${r.id}` === dsSelectedKey)) {
        dsSelectedKey = pageRows.length ? `${pageRows[0].t}:${pageRows[0].id}` : null;
      }

      const tbody = document.getElementById('ds-table-body');
      tbody.innerHTML = pageRows.map(r => {
        const key = `${r.t}:${r.id}`;
        const selCls = key === dsSelectedKey ? 'bg-indigo-500/10 font-semibold' : 'hover:bg-[var(--secondary)]/60';
        return `
          <tr data-rkey="${key}" class="cursor-pointer transition ${selCls}">
            <td class="py-2 px-3 font-mono font-bold text-indigo-500">T${r.t}</td>
            <td class="py-2 px-3 font-mono">${r.id}</td>
            <td class="py-2 px-3 truncate max-w-[170px]" title="${r.src}">${r.src}</td>
            <td class="py-2 px-3 font-mono text-[11px] text-[var(--muted-foreground)]">${r.st}</td>
            <td class="py-2 px-3 font-mono">${r.len}</td>
            <td class="py-2 px-3">${formatRowScore(r)}</td>
          </tr>`;
      }).join('');

      tbody.querySelectorAll('tr').forEach(tr => {
        tr.addEventListener('click', () => {
          dsSelectedKey = tr.getAttribute('data-rkey');
          renderDatasetBrowser();
        });
      });

      document.getElementById('ds-count-label').textContent = `Showing ${pageRows.length} of ${filtered.length} datapoints`;
      document.getElementById('ds-page-label').textContent = `${dsPage + 1} / ${totalPages}`;
      document.getElementById('ds-prev-btn').disabled = dsPage <= 0;
      document.getElementById('ds-next-btn').disabled = dsPage >= totalPages - 1;

      const detailEl = document.getElementById('ds-detail-pane');
      const sel = rows.find(r => `${r.t}:${r.id}` === dsSelectedKey);
      if (!sel) {
        detailEl.innerHTML = '<div class="text-[var(--muted-foreground)]">No matching datapoint.</div>';
      } else {
        const scJson = Object.keys(sel.sc || {}).length ? JSON.stringify(sel.sc, null, 2) : 'Not yet evaluated (GPU task)';
        detailEl.innerHTML = `
          <div class="flex items-center justify-between gap-2 pb-2 border-b border-[var(--border)]">
            <div>
              <span class="px-2 py-0.5 rounded bg-indigo-500/15 text-indigo-500 font-mono font-bold">Test ${sel.t} • ${sel.id}</span>
              <div class="font-bold text-sm mt-1">${sel.src}</div>
              <div class="text-[var(--muted-foreground)]">${sel.desc} • Stratum: <code class="font-mono">${sel.st}</code></div>
            </div>
          </div>
          <div class="space-y-1">
            <div class="font-bold uppercase tracking-wider text-[10px] text-emerald-600">Ground Truth</div>
            <div class="p-2.5 rounded-lg bg-[var(--card)] border border-[var(--border)] font-mono text-[11px] leading-relaxed break-words">${sel.gt}</div>
          </div>
          <div class="space-y-1">
            <div class="font-bold uppercase tracking-wider text-[10px] text-indigo-600">Live Model Output / Grade</div>
            <pre class="p-2.5 rounded-lg bg-[var(--card)] border border-[var(--border)] font-mono text-[11px] overflow-x-auto whitespace-pre-wrap max-h-28 overflow-y-auto">${scJson}</pre>
          </div>
          <div class="space-y-1">
            <div class="font-bold uppercase tracking-wider text-[10px] text-[var(--muted-foreground)]">Sequence (${sel.len} AA)</div>
            <pre class="p-2.5 rounded-lg bg-[var(--card)] border border-[var(--border)] font-mono text-[11px] overflow-x-auto whitespace-pre-wrap break-all max-h-28 overflow-y-auto">${sel.seq}</pre>
          </div>
          <div class="space-y-1">
            <div class="font-bold uppercase tracking-wider text-[10px] text-[var(--muted-foreground)]">Prompt Preview</div>
            <pre class="p-2.5 rounded-lg bg-[var(--card)] border border-[var(--border)] font-mono text-[11px] overflow-x-auto whitespace-pre-wrap max-h-36 overflow-y-auto">${sel.pr}</pre>
          </div>`;
      }
    }

    const dsInput = document.getElementById('ds-search-input');
    if (dsInput) {
      dsInput.addEventListener('input', ev => {
        dsQuery = ev.target.value;
        dsPage = 0;
        renderDatasetBrowser();
      });
      document.getElementById('ds-prev-btn').addEventListener('click', () => { if (dsPage > 0) { dsPage--; renderDatasetBrowser(); } });
      document.getElementById('ds-next-btn').addEventListener('click', () => { dsPage++; renderDatasetBrowser(); });
      document.getElementById('download-jsonl-btn').addEventListener('click', () => {
        const blob = new Blob([JSON.stringify(DATA.datapoints || [], null, 2)], { type: 'application/json' });
        const a = document.createElement('a');
        a.href = URL.createObjectURL(blob);
        a.download = 'proteinbench_v1_522_datapoints.json';
        a.click();
      });
    }

    renderExplorer();
    renderLiveLeaderboard();
    renderDatasetBrowser();
  </script>
</body>
</html>
"""


def trim_explorer_atlas(full_atlas: Dict[str, Any]) -> Dict[str, Any]:
    """Keep only the exact showcase records rendered by the 8-tab explorer so `website/index.html` stays ~35 KB!"""
    return {
        "s1_examples": [full_atlas["s1_examples"][0]],
        "s1b_example": full_atlas["s1b_example"],
        "s2a_examples": [full_atlas["s2a_examples"][0], full_atlas["s2a_examples"][1], full_atlas["s2a_examples"][2]],
        "s2b_examples": [full_atlas["s2b_examples"][0]],
        "s3a_examples": [full_atlas["s3a_examples"][0]],
        "s6_examples": [
            full_atlas["s6_examples"][0],
            full_atlas["s6_examples"][1],
            full_atlas["s6_examples"][2],
            full_atlas["s6_examples"][3],
        ],
        "s3b_part1_examples": [full_atlas["s3b_part1_examples"][0], full_atlas["s3b_part1_examples"][1]],
        "s3b_part2_examples": [
            full_atlas["s3b_part2_examples"][0],
            full_atlas["s3b_part2_examples"][1],
            full_atlas["s3b_part2_examples"][2],
        ],
    }


def main() -> None:
    live_sum_path = REPO / "outputs" / "results" / "local_13456_unified_summary.json"
    live_sum = json.loads(live_sum_path.read_text(encoding="utf-8")) if live_sum_path.exists() else None
    # Strip bulky per_target arrays from live_sum inside HTML since `datapoints` already carries per-row scores
    if live_sum:
        for k in ("test1_forward_folding", "test5_monomer_alanine_ranking", "test6_binding_alanine_ranking"):
            for item in live_sum.get(k, []):
                item.pop("per_target", None)

    dp_path = REPO / "website" / "data" / "proteinbench_v1_explorer.json"
    datapoints = json.loads(dp_path.read_text(encoding="utf-8")) if dp_path.exists() else []

    # Compile Tests 2, 7, 8 live GPU/JapanFold summary
    gpu_sum: Dict[str, Any] = {"test2": {}, "test7": {}, "test8": {}}
    s1_rep_p = REPO / "outputs" / "results" / "stage1_evaluation_report.json"
    if s1_rep_p.exists():
        s1_rep = json.loads(s1_rep_p.read_text(encoding="utf-8"))
        for m in s1_rep.get("stage1b_inverse_folding", []):
            mc = dict(m)
            mc.pop("per_target", None)
            gpu_sum["test2"][mc["model"]] = mc
    s7_rep_p = REPO / "outputs" / "results" / "test7_esmfold2_evaluation_report.json"
    if s7_rep_p.exists():
        s7_rep = json.loads(s7_rep_p.read_text(encoding="utf-8"))
        for mname, mobj in s7_rep.get("models", {}).items():
            mc = dict(mobj)
            mc.pop("per_task", None)
            gpu_sum["test7"][mname] = mc
    for cf in sorted((REPO / "outputs" / "predictions").glob("stage3b_binder_campaign_*_closed_loop_af2.json")):
        cobj = json.loads(cf.read_text(encoding="utf-8"))
        base_m = cobj.get("model_tag", "").replace("_closed_loop_af2", "")
        sm = dict(cobj.get("summary", {}))
        sm.pop("per_target", None)
        gpu_sum["test8"][base_m] = sm

    payload = {
        "atlas": trim_explorer_atlas(compile_visual_atlas_data()),
        "prompts": load_sample_prompts(),
        "live_summary": live_sum,
        "gpu_summary": gpu_sum,
        "datapoints": datapoints,
    }
    json_blob = json.dumps(payload, separators=(",", ":"))

    # 1. Standalone GitHub-Pages / Browser website (`website/index.html`) using public Tailwind CDN + light theme
    standalone_vars = """    :root {
      --background: #f8fafc;
      --foreground: #0f172a;
      --card: #ffffff;
      --border: #e2e8f0;
      --secondary: #f1f5f9;
      --muted-foreground: #475569;
    }"""
    web_dir = REPO / "website"
    web_dir.mkdir(parents=True, exist_ok=True)
    standalone_html = (
        WEBSITE_TEMPLATE.replace("__TAILWIND_SCRIPT_URL__", "https://cdn.tailwindcss.com")
        .replace("__STANDALONE_FALLBACK_VARS__", standalone_vars)
        .replace("__WEBSITE_DATA_JSON__", json_blob)
    )
    (web_dir / "index.html").write_text(standalone_html, encoding="utf-8")

    # 2. Antigravity inline artifact version (gstatic allowlisted Tailwind + zero `:root` overrides)
    artifact_html = (
        WEBSITE_TEMPLATE.replace("__TAILWIND_SCRIPT_URL__", "https://www.gstatic.com/antigravity/web/dev/tailwindcss.min.js")
        .replace("__STANDALONE_FALLBACK_VARS__", "")
        .replace("__WEBSITE_DATA_JSON__", json_blob)
    )
    (web_dir / "proteinbench_artifact.html").write_text(artifact_html, encoding="utf-8")
    print(
        f"Built ProteinBench website -> {web_dir / 'index.html'} "
        f"({(web_dir / 'index.html').stat().st_size / 1024:.1f} KB)"
    )


if __name__ == "__main__":
    main()
