# Protein Structure Without Tools

**Can frontier reasoning models solve real structural biology problems from weights and chain-of-thought alone — with no alignments, no databases, and no structure predictors?**

This repository contains the complete 8-task benchmark (**582 targets**), zero-leakage anonymised prompts, automated evaluators, specialist baseline pipelines, and interactive HTML report (`website/index.html`).

---

## Benchmark Overview (8 Tasks · 582 Targets)

| # | Task | Dataset & Holdout Design | Specialist Baseline | Primary Metrics |
| :---: | :--- | :--- | :--- | :--- |
| **1** | **Forward folding** | **`N = 100`** monomeric RCSB PDB structures stratified by Foldseek structural TM-score to pre-May-2026 PDB (`15` known, `12` homolog gradient, `27` PDB-novel, `28` *de novo*, `18` truly novel `TM < 0.53`) | `ESMFold-2` (single-seq) | `TM-score`, `P@L (|i−j| ≥ 24)`, `Q3` |
| **2** | **Inverse folding** | **Same `N = 100`** structures stripped to bare `Cα` coordinates (`UNK` residues) + backbone angles & burial counts | `ProteinMPNN` (`v_48_020`) | `scTM` (ESMFold-2 refold), `scRMSD < 2 Å`, `NSR` |
| **3** | **Remote homology** | **`N = 120`** SCOPe 2.08 domain pairs via TEA & Foldseek (`30` sequence-easy, `30` Foldseek 3Di, `30` deep twilight `<15%` ID, `30` class/ID-matched decoys) | `ESM-2 (650M)` cosine | `Recall @ 10% FPR`, `AUROC`, `Accuracy` |
| **4** | **Function prediction** | **`N = 120`** UniProtKB / EBI GOA proteins in a CAFA-6 2×3 split (`Pre May–Oct 2025` vs `Post May–Oct 2026` × `NK / LK / PK`, `20` per cell; strict Post-NK double-verified via QuickGO) | `NetGO 4.0` (GOCurator) | CAFA-6 `IA`-weighted `wF_max` (`MF`, `BP`, `CC`; dual `-known` masking on PK) |
| **5** | **Monomer alanine ranking** | **`N = 52`** 18-mutation `X→Ala` panels (`MegaScale` *de novo* `24`, natural `14`; `MaveDB` DMS `12`, dual `2`), pairing hotspots with same-AA neutral decoys (`50%` 1D floor) | `ESMFold-2 → SaProt (650M)` | `NDCG@5`, `Same-AA win`, `Spearman ρ` |
| **6** | **Interface binding ranking** | **`N = 38`** PPI `X→Ala` panels (`SKEMPI v2.0` `28`, `Open-AlphaSeq` `3`, `MaveDB` PPI DMS `7`; `31` pre-2026 PDB vs `7` unseen/post-2026), monomer-destabilising sites removed | `Boltz-2 → StaB-ddG` *(zero SKEMPI training)* | `Spearman ρ`, `NDCG@5`, `Same-AA win` |
| **7** | **De novo & motif design** | **`N = 40`** single-shot sequence prompts (`24` fold-conditioned + `16` crystal motif scaffolds), folded with ESMFold-2 & OpenFold3 | `RFdiffusion3 + ProteinMPNN` | Proteina-Complexa criteria: Fold `TM ≥ 0.50`, Motif `RMSD ≤ 1.0 Å`, `pLDDT ≥ 70`, `scRMSD ≤ 2 Å`, `<30%` SeqID |
| **8** | **Closed-loop binder design** | **`N = 12`** RCSB PDB epitopes (`4` ProtDBench classic, `5` BoltzProt1 novel, `3` post-May-2026 crystals) × `16` AlphaFold2/ESMFold-2 feedback rounds | `BoltzGen` (16 all-atom designs) | `Hit@16` (`ipTM ≥ 0.75`, `iPAE < 10 Å`, `≥3/5` hotspots), running-best AF2 score |

---

## Quick Start

```bash
git clone https://github.com/talugueloglu/protein-without-tools.git
cd protein-without-tools

# Install into a local virtualenv (Python 3.11+)
uv venv && uv pip install -e ".[dev,cloud]"

# Run unit, anti-cheat, CAFA-6 IA/masking, and zero-leakage tests (14/14 in < 2s)
.venv/bin/pytest -v
```

---

## Running Models & Rebuilding the Report

```bash
cp .env.example .env   # add OPENAI_API_KEY (and ANTHROPIC_API_KEY / GEMINI_API_KEY as needed)

# Single-shot tasks 1–7 (e.g. gpt-6.1-sol, gpt-6.2-sol, gpt-6.2-astra)
.venv/bin/python scripts/run_benchmark_llms.py \
  --provider openai --model gpt-6.2-astra --stages 1,2,3,4,5,6,7 --thinking-budget 4096

# Task 8: 16-round closed-loop AlphaFold2 binder campaigns
.venv/bin/python scripts/run_stage3b_closed_loop_llms.py \
  --provider openai --model gpt-6.2-astra --mode closed_loop --oracle modal_af2 --shots 16 --thinking-budget 4096

# Re-score Task 4 (CAFA-6 2×3 IA-weighted wF_max) and rebuild website/index.html
.venv/bin/python scripts/eval_t4_cafa6_6cell.py
.venv/bin/python scripts/build_website.py
```

Open `website/index.html` locally (e.g., `.venv/bin/python -m http.server 8765 --directory website`) to view all interactive charts, 3D Mol\* structures, and all 582 datapoints.

---

## Repository Layout

- `src/protein_without_tools/` — Core scoring modules (TM-score, contact reconstruction, SCOPe homology, CAFA-6 IA & NK/LK/PK evaluator, alanine-scan ranking, design & binder evaluators, unified LLM client).
- `data/` — Curated benchmark manifests, holdout metadata, and compact `go_subdag.obo` (large raw PDB/GAF download caches are gitignored).
- `outputs/prompts/` — Zero-leakage anonymised JSONL prompts for all tasks (`web_chat_spotcheck/` has ready-to-paste `.txt` examples).
- `outputs/predictions/` & `outputs/results/` — Saved predictions and per-target evaluation JSONs for `gpt-6.1-sol`, `gpt-6.2-sol`, `gpt-6.2-astra`, and all 8 specialist baselines.
- `scripts/` — Dataset builders, model runners, specialist baseline scripts, and the single-file site compiler (`build_website.py`).
- `tests/` — Pytest suite checking parsers, CAFA-6 `IA` and `-known` masking, anti-cheat decoy balance, and zero-leakage prompt guarantees.
- `website/` — Standalone interactive report (`website/index.html`), source template (`website/template.html`), and flat 582-datapoint export (`website/data/datapoints_v1.jsonl`).

## License

Code is released under the [MIT License](LICENSE). Benchmark inputs derived from public databases (PDB, SCOPe, EBI GOA / UniProt, MegaScale, SKEMPI v2) remain subject to their respective upstream licenses.

## Citation

```bibtex
@misc{gueloglu2026proteinstructure,
  title  = {Protein Structure Without Tools: Folding and Binding Physics in Frontier Reasoning Models},
  author = {Gueloglu, Talu},
  year   = {2026},
  url    = {https://github.com/talugueloglu/protein-without-tools}
}
```
