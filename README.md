# Fable-Protein-Bench

**From Sequence Memorization to Folding & Binding Physics? Auditing `Claude Fable 5.2` (`claude-fable-5-2`) & Frontier Reasoning LLMs Head-to-Head Against Specialist Structural Biology Tools (`ESMFold`, `Boltz-2`, `ProteinMPNN`, `PickyBinders/tea`, `ESM-2`, `ThermoMPNN`, `BindCraft` / `ColabDesign AF2`) Across 8 Zero-Leakage Benchmarks (`No-Code` vs. `With-Code`).**

---

## Complete 8-Test Suite (`1.` to `8.` — All Ready Offline + Turnkey Live Runners)

| # | Test | Curated Dataset (Leak-Free & Stratified) | Specialist GPU Baselines | Primary Metrics |
| :---: | :--- | :--- | :--- | :--- |
| **1.** | **Forward Folding** | **`N = 100`** (`85` post-May-2026 PDB incl. **`18` true novel folds `TM < 0.52`** + `15` pre-cutoff controls) | `ESMFold`, `Boltz-2` (AF3-class), `ESM-2` | Exact `TM-score`, `Cα RMSD`, `lDDT-Cα`, `P@L` (`\|i-j\| ≥ 24`), `Q3` |
| **2.** | **Inverse Folding** | **`N = 100`** stripped `UNK` 3D backbones (`N, Cα, C, O` only) | `ProteinMPNN` (`v_48_020`) + `ESMFold` | `NSR %` (All / Core / Surface), refolded `scTM` & `scRMSD < 2.0 Å` |
| **3.** | **Remote Homology (`TEA`)** | **`N = 120`** `SCOPe 2.08` pairs (`Seq-Easy` $\to$ `Foldseek 3Di` $\to$ **`Deep TEA Zone`** $\to$ class-matched decoys) | `Foldseek 3Di`, `PickyBinders/tea`, `ESM-2` | Stratified `AUROC`, `AUPRC`, & Rung `Recall @ 10% FPR` |
| **4.** | **Function (`CAFA`)** | **`N = 60`** 2026 experimental `QuickGO` holdouts vs. `1,728` pre-2025 Swiss-Prot ref bank | `Smith-Waterman`, `ESM-2` $k$-NN, `CAFA Naive` | Official `CAFA` `wF_max` & `F_max` (`MF`, `BP`, `CC`) |
| **5.** | **Monomer Alanine Scan (`X -> A`)** | **`N = 52`** proteins (**`5,348`** wet-lab `X -> A` mutants: `24` *de novo* $\Delta\Delta G$, `14` natural $\Delta\Delta G$, `14` May–Nov 2026 `MaveDB` DMS) | `ThermoMPNN`, `ESM-2` (650M & 3B) marginals | `Spearman ρ`, **`AA-Demeaned Context ρ`**, Hotspot & Core `AUROC` |
| **6.** | **Interface Binding Alanine Scan** | **`N = 38`** PPI complexes (**`2,767`** wet-lab `X -> A` binding mutants: `26` `SKEMPI v2.0` SPR/ITC $\Delta\Delta G_{\text{bind}}$, `3` `Open-AlphaSeq` `PD-1`/`TIGIT` $K_D$, `5` PPI binding DMS, `4` Oct–Nov 2026 `MaveDB` binding incl. `KRAS × LZTR1`) | `ESM-2` Complex & Partner-$\Delta$ Marginals, `ProteinMPNN` $\Delta\log P$ | `Spearman ρ`, **`AA-Demeaned Context ρ`** (`Bogan–Thorn 1D = 0.000`), Binding Hotspot & 3D Interface `AUROC` |
| **7.** | ***De Novo* Monomer & Motif** | **`N = 40`** tasks (`24` fold-conditioned across 4 topologies + `16` crystal-motif scaffolds) | `ESM3-open`, `ProGen2`, `ProteinMPNN` | 5-Gate `% Designable` (`pLDDT ≥ 80`, `scRMSD ≤ 2 Å`, `motif RMSD ≤ 1.5 Å`, `<30%` SeqID) |
| **8.** | **Closed-Loop Epitope Binders** | **`N = 12` proven-bindable epitopes $\times$ `16` `AF2` shots** (`4` `RFD4`/`ProtDBench` + `5` `BoltzProt-1` hard novel + `3` 2026 *de novo* binder crystals) | `BindCraft` (`ColabDesign AF2`), `ProtDBench` (`RFD3`, `PXDesign`, `BoltzGen`) | Closed-Loop **`Shots-to-Hit`** (`Pass@1..16`, `ipTM ≥ 0.75`, `iPAE < 10 Å`) |

---

## Offline Verification & Calibration (100% Zero-Key Ready)

```bash
# 1. Run full mathematical, parser, anti-cheat & zero-leakage test suite (13/13 in < 2s)
.venv/bin/pytest -v

# 2. Re-evaluate offline baselines & regenerate figures for any test
.venv/bin/python scripts/evaluate_stage1.py --run-calibration
.venv/bin/python scripts/build_and_evaluate_stage2.py --eval-only
.venv/bin/python scripts/evaluate_alanine_scan.py
.venv/bin/python scripts/build_and_evaluate_binding_alanine.py
.venv/bin/python scripts/build_and_evaluate_stage3b.py --eval-only
```

* **Manual Web-UI Spotchecks (`$0` API spend):** Ready-to-paste zero-leakage `.txt` prompts across all tests live in `outputs/prompts/web_chat_spotcheck/`.

---

## Running Live Frontier LLMs & Modal `L40S` Specialist Baselines

```bash
cp .env.example .env   # Add ANTHROPIC_API_KEY, OPENAI_API_KEY, GEMINI_API_KEY

# 1. Single-turn tests (1..7) — supports --pilot N & --max-usd
.venv/bin/python scripts/run_benchmark_llms.py --provider anthropic --model claude-fable-5-2 --stages all --pilot 4
.venv/bin/python scripts/run_benchmark_llms.py --provider anthropic --model claude-fable-5-2 --stages all --thinking-budget 8000

# 2. Test 8 Closed-Loop vs Open-Loop Epitope Binder Campaigns (16 AF2 shots / epitope)
.venv/bin/python scripts/run_stage3b_closed_loop_llms.py --provider anthropic --model claude-fable-5-2 --mode closed_loop --oracle modal_af2 --shots 16
.venv/bin/python scripts/run_stage3b_closed_loop_llms.py --provider anthropic --model claude-fable-5-2 --mode open_loop  --oracle modal_af2 --shots 16

# 3. Modal L40S Specialist Bio-Tools
.venv/bin/modal run scripts/modal_stage1_baselines.py --mode baselines
.venv/bin/modal run scripts/modal_stage2_baselines.py
.venv/bin/modal run scripts/modal_alanine_baselines.py
.venv/bin/modal run scripts/modal_binding_alanine_baselines.py
.venv/bin/modal run scripts/modal_stage3b_af2_bindcraft.py --mode bindcraft-baseline
```
