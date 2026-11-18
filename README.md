# Fable-Protein-Bench

**From Sequence Memorization to Folding Physics? Benchmarking Frontier Reasoning LLMs (`Claude Fable 5.2`, `Opus 5.5`, `GPT-5.2`, `Gemini`) Head-to-Head Against Specialist Structural Biology Tools (`ESMFold`, `ProteinMPNN`, `ESM-2`, `PickyBinders/TEA`, `ESM3`) on Strict Post-Cutoff 2026 Proteins.**

---

## Benchmark Roadmap (6 Canonical Pillars)

| Stage | Benchmark | Dataset (Uncontaminated / Stratified) | Specialist Bio-Tools | Community Metrics |
| :---: | :--- | :--- | :--- | :--- |
| **1A** *(Ready)* | **Forward Folding** | Pre-2025 controls (`6MRR`, etc.) vs. **`Sept–Nov 2026` PDB Monomers** (incl. `37NT` domains) | `ESMFold`, `ESM-2` contacts | `TM-score`, `Cα RMSD`, `lDDT-Cα`, `P@L` (`\|i-j\| ≥ 24`), `Q3` |
| **1B** *(Ready)* | **Inverse Folding** | Stripped `Sept–Nov 2026` PDB Backbones (`N, Cα, C, O` only) | `ProteinMPNN` (`v_48_020`) | `NSR %` (All / Core / Surface), Refolded `scTM` & `scRMSD < 2 Å` |
| **2A** *(Next)* | **Remote Homology (`TEA`)** | `SCOPe` / `ECOD` (`BLAST` $\to$ `Foldseek` $\to$ **`PickyBinders/tea`-only**) | `MMseqs2`, `Foldseek`, `TEA` | Tiered `AUROC` & `Top-1 Retrieval` |
| **2B** *(Next)* | **Function (`CAFA`)** | Newly `GOA` experimentally annotated late-2026 proteins | `MMseqs2`/`Diamond`, `ESM-2` | Official CAFA `F_max` & `wF_max` (`MF`, `BP`, `CC`) |
| **3A** *(Planned)* | **Variant Fitness & $\Delta\Delta G$** | `ProteinGym` 2026 Holdout + Synthetic $\Delta\Delta G$ + Alanine Scan | `ESM-2`, `ProGen3`, `ThermoMPNN` | `Spearman ρ`, `Top-10% Recall`, Burial `AUROC` |
| **3B** *(Planned)* | ***De Novo* Generation** | Unconditional + Architecture (`4H`, `βαβαβ`, `all-β`) + Motif Scaffolding | `ESM3-open`, `ProGen3` | `% Designable` (`pLDDT > 80` & `scRMSD < 2 Å`), `MMseqs2`/`Foldseek` Novelty |

---

## Stage 1 Quickstart (Works 100% Without Any API Credits!)

### 1. Run Unit & Mathematical Verification Suite (Offline, < 3s)
Verifies `.pdb`/`.cif` parsing, Kabsch–Sander `Q3` H-bonds, proper $SO(3)$ `TM-score` vs. mirror enantiomers, `L-BFGS` 3D contact reconstruction (`TM > 0.80` from true contacts vs. `< 0.30` random), and strict zero-leakage prompt assertions:
```bash
.venv/bin/pytest -v
```

### 2. Curate Stage 1 PDB Dataset & Export Zero-Leakage Prompts (Free Public RCSB API)
Fetches Pre-2025 controls (including `@sokrypton`'s `6MRR`), `@WUFang40615703`'s Oct-2026 `37NT` (`MERTK` ectodomain) individual domains, and newly released **`Sept–Nov 2026` non-viral, non-toxin monomers**:
```bash
.venv/bin/python scripts/curate_stage1_pdbs.py
.venv/bin/python scripts/export_stage1_prompts.py
```
* **Want to test manually in Claude.ai / ChatGPT Web UI with \$0 API spend?** Open any `.txt` file in `outputs/prompts/web_chat_spotcheck/` (all PDB IDs and headers are stripped so the model cannot cheat!).

### 3. Run Offline Oracle & Null Calibration Across All Curated PDBs
Establishes the exact upper bound of our deterministic `L-BFGS` 3D reconstructor (`Oracle_TrueContacts`), `50%`-precision pLM simulation, 1D Chou–Fasman heuristic, and random null floor—and generates baseline figures:
```bash
.venv/bin/python scripts/evaluate_stage1.py --run-calibration
.venv/bin/python scripts/plot_stage1.py
```

---

## Running Live Models & Modal Specialist Baselines (When Keys/Credits Are Ready)

### A. Specialist GPU Tools (`ESMFold`, `ESM-2`, `ProteinMPNN` on Modal `L40S`, ~5 mins)
```bash
.venv/bin/modal run scripts/modal_stage1_baselines.py --mode baselines
```

### B. Frontier LLMs (`Claude Fable 5.2`, `Opus 5.5`, `GPT-5.2`, `Gemini`)
Copy `.env.example` to `.env`, add your API key(s), and start with a `--pilot 4` sanity check (~`\$0.50`) before running the full set:
```bash
cp .env.example .env

# 1. Pilot check (4 proteins, verifies 0% silent safety-fallback on claude-fable-5-2)
.venv/bin/python scripts/run_stage1_llms.py --provider anthropic --model claude-fable-5-2 --pilot 4

# 2. Full Stage 1A + 1B run (Direct vs Extended Thinking)
.venv/bin/python scripts/run_stage1_llms.py --provider anthropic --model claude-fable-5-2 --thinking-budget 0
.venv/bin/python scripts/run_stage1_llms.py --provider anthropic --model claude-fable-5-2 --thinking-budget 8000
.venv/bin/python scripts/run_stage1_llms.py --provider openai --model gpt-5.2 --thinking-budget 8000

# 3. Refold LLM Stage 1B designed sequences in ESMFold on Modal for scTM & scRMSD
.venv/bin/modal run scripts/modal_stage1_baselines.py --mode refold-stage1b

# 4. Score everything & regenerate figures
.venv/bin/python scripts/evaluate_stage1.py --run-calibration
.venv/bin/python scripts/plot_stage1.py
```
