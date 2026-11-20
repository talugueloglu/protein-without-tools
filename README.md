# Fable-Protein-Bench

**From Sequence Memorization to Folding Physics? Auditing `Claude Fable 5.2` (`claude-fable-5-2`) & Frontier Reasoning LLMs Head-to-Head Against Specialist Structural Biology Tools (`ESMFold`, `Boltz-2`, `ProteinMPNN`, `PickyBinders/tea`, `ESM-2`, `ThermoMPNN`, `BindCraft` / `ColabDesign AF2`) Across 6 Zero-Leakage Benchmarks.**

---

## Complete 6-Benchmark Suite (All 6 Ready Offline + Turnkey Live Runners)

| Stage | Benchmark | Curated Dataset (Leak-Free & Stratified) | Specialist GPU Baselines | Primary Metrics |
| :---: | :--- | :--- | :--- | :--- |
| **1A** | **Forward Folding** | **`N = 100`** (`85` post-May-2026 PDB incl. **`18` true novel folds `TM < 0.52`** + `15` pre-cutoff controls) | `ESMFold`, `Boltz-2` (AF3-class), `ESM-2` | Exact `TM-score`, `Cα RMSD`, `lDDT-Cα`, `P@L` (`\|i-j\| ≥ 24`), `Q3` |
| **1B** | **Inverse Folding** | **`N = 100`** stripped `UNK` 3D backbones (`N, Cα, C, O` only) | `ProteinMPNN` (`v_48_020`) + `ESMFold` | `NSR %` (All / Core / Surface), refolded `scTM` & `scRMSD < 2.0 Å` |
| **2A** | **Remote Homology (`TEA`)** | **`N = 120`** `SCOPe 2.08` pairs (`Seq-Easy` $\to$ `Foldseek 3Di` $\to$ **`Deep TEA Zone`** $\to$ class-matched decoys) | `Foldseek 3Di`, `PickyBinders/tea`, `ESM-2` | Stratified `AUROC`, `AUPRC`, & Rung `Recall @ 10% FPR` |
| **2B** | **Function (`CAFA`)** | **`N = 60`** 2026 experimental `QuickGO` holdouts vs. `1,728` pre-2025 Swiss-Prot ref bank | `Smith-Waterman`, `ESM-2` $k$-NN, `CAFA Naive` | Official `CAFA` `wF_max` & `F_max` (`MF`, `BP`, `CC`) |
| **3A** | **Alanine Scan (`X -> A`)** | **`N = 52`** proteins (**`5,348`** wet-lab `X -> A` mutants: `24` *de novo* $\Delta\Delta G$, `14` natural $\Delta\Delta G$, `14` May–Nov 2026 `MaveDB` DMS) | `ThermoMPNN`, `ESM-2` (650M & 3B) marginals | `Spearman ρ`, **`AA-Demeaned Context ρ`**, Hotspot & Core `AUROC` |
| **3B** | ***De Novo* & Closed-Loop Binders** | **`N = 40`** fold/motif prompts + **`N = 8` epitopes $\times$ `16` `AF2` shots** (`2` classic + **`6` post-May-2026 stripped complexes**) | `BindCraft` (`ColabDesign AF2`), `ESM3`, `MPNN` | 5-Gate `% Designable` + Closed-Loop **`Shots-to-Hit`** (`Pass@1..16`) |

---

## Offline Verification & Calibration (100% Zero-Key Ready)

```bash
# 1. Run full mathematical, parser, anti-cheat & zero-leakage test suite (11/11 in < 2s)
.venv/bin/pytest -v

# 2. Re-evaluate offline baselines & regenerate figures for any stage
.venv/bin/python scripts/evaluate_stage1.py --run-calibration
.venv/bin/python scripts/build_and_evaluate_stage2.py --eval-only
.venv/bin/python scripts/evaluate_alanine_scan.py
.venv/bin/python scripts/build_and_evaluate_stage3b.py --eval-only
```

* **Manual Web-UI Spotchecks (`$0` API spend):** Ready-to-paste zero-leakage `.txt` prompts across all stages live in `outputs/prompts/web_chat_spotcheck/`.

---

## Running Live Frontier LLMs & Modal `L40S` Specialist Baselines

```bash
cp .env.example .env   # Add ANTHROPIC_API_KEY, OPENAI_API_KEY, GEMINI_API_KEY

# 1. Single-turn stages (1A, 1B, 2A, 2B, 3A, and 3B Part-1 Monomer/Motif) — supports --pilot N & --max-usd
.venv/bin/python scripts/run_benchmark_llms.py --provider anthropic --model claude-fable-5-2 --stages all --pilot 4
.venv/bin/python scripts/run_benchmark_llms.py --provider anthropic --model claude-fable-5-2 --stages all --thinking-budget 8000

# 2. Stage 3B Part-2 Closed-Loop vs Open-Loop Epitope Binder Campaigns (16 AF2 shots / epitope)
.venv/bin/python scripts/run_stage3b_closed_loop_llms.py --provider anthropic --model claude-fable-5-2 --mode closed_loop --oracle modal_af2 --shots 16
.venv/bin/python scripts/run_stage3b_closed_loop_llms.py --provider anthropic --model claude-fable-5-2 --mode open_loop  --oracle modal_af2 --shots 16

# 3. Modal L40S Specialist Bio-Tools
.venv/bin/modal run scripts/modal_stage1_baselines.py --mode baselines
.venv/bin/modal run scripts/modal_stage2_baselines.py
.venv/bin/modal run scripts/modal_alanine_baselines.py
.venv/bin/modal run scripts/modal_stage3b_af2_bindcraft.py --mode bindcraft-baseline
```
