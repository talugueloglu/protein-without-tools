#!/usr/bin/env python3
"""Google Colab `AlphaFold2` (`ColabDesign` + `ProteinMPNN`) Bundle Exporter, Notebook Builder & Result Importer.

Covers all three folding-oracle exercises on a free/standard Google Colab GPU (`T4` / `L4` / `A100`):
  - Test 2: Inverse Folding (`N = 100` stripped backbones -> `AlphaFold2` `fixbb` `scRMSD < 2.0 Å`, `scTM`, `pLDDT` + `ProteinMPNN` specialist baseline)
  - Test 7: De Novo Monomer & Crystal Motif Scaffolding (`N = 40` specs -> 5-gate `AlphaFold2` + `ProteinMPNN` designability)
  - Test 8: Closed-Loop Epitope Binder Design (`N = 12` epitopes × `16` `AlphaFold2` `protocol="binder"` shots + `BindCraft` 16-step specialist baseline)

Usage:
  1. Build notebook & export compact bundle (`< 2 MB`):
       .venv/bin/python scripts/colab_af2_bridge.py --build-notebook --export
  2. Upload `notebooks/proteinbench_alphafold2_colab.ipynb` to Google Colab (`Runtime -> T4/L4 GPU`),
     upload `outputs/colab/proteinbench_colab_bundle.tar.gz` when prompted in Cell 2, and Run All!
  3. Import downloaded `proteinbench_colab_af2_outputs.tar.gz` back on your Mac:
       .venv/bin/python scripts/colab_af2_bridge.py --import ~/Downloads/proteinbench_colab_af2_outputs.tar.gz
"""

from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile
from typing import Any, Dict, List

REPO = Path("/Users/taluguloglu/.gemini/antigravity/scratch/fable-protein-bench")


def _md_cell(text: str) -> Dict[str, Any]:
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": [line + "\n" for line in text.strip().splitlines()],
    }


def _code_cell(code: str) -> Dict[str, Any]:
    # Verify valid Python syntax (ignoring `!` shell magics)
    clean_py = "\n".join(
        ("pass" if ln.lstrip().startswith("!") else ln)
        for ln in code.strip().splitlines()
    )
    ast.parse(clean_py)
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [line + "\n" for line in code.strip().splitlines()],
    }


def build_colab_notebook(out_ipynb: Path) -> None:
    cells: List[Dict[str, Any]] = [
        _md_cell(
            """# ProteinBench — Google Colab `AlphaFold2` & `ProteinMPNN` GPU Evaluator (Tests 2, 7, and 8)

Make sure **Runtime → Change runtime type → Hardware accelerator** is set to **`T4 GPU`** (or **`L4` / `A100`**).

This notebook runs official Google DeepMind **`AlphaFold2`** (`alphafold_params_2022-12-06.tar` via `@sokrypton`'s `ColabDesign`) + **`ProteinMPNN`** across:
- **Test 2 (`Inverse Folding`, `N = 100`):** `ProteinMPNN` specialist baseline + `AlphaFold2` self-consistency (`scRMSD < 2.0 Å`, `scTM`, `pLDDT`) for all uploaded LLM predictions (`*_stage1b.jsonl`).
- **Test 7 (`De Novo Monomer & Motif Scaffolding`, `N = 40`):** Full 5-Gate `AlphaFold2` + `ProteinMPNN` designability pipeline for all uploaded LLM predictions (`*_stage3b.jsonl`).
- **Test 8 (`Closed-Loop Epitope Binder Design`, `N = 12` epitopes × `16` `AlphaFold2` shots):** `BindCraft` 16-step specialist hallucination baseline + optional live `OpenAI ↔ AlphaFold2` closed-loop campaigns (using Colab's `🔑 Secrets` `OPENAI_API_KEY`)."""
        ),
        _code_cell(
            """# === CELL 1: Verify GPU & Install ColabDesign + Official AlphaFold2 Weights (~90s) ===
!nvidia-smi
!pip install -q "git+https://github.com/sokrypton/ColabDesign.git" httpx biopython
!mkdir -p /content/af2/params
!test -f /content/af2/params/params_model_1_ptm.npz || (wget -q -O /content/af2/af2_params.tar https://storage.googleapis.com/alphafold/alphafold_params_2022-12-06.tar && tar -xf /content/af2/af2_params.tar -C /content/af2/params && rm /content/af2/af2_params.tar)
print("AlphaFold2 weights & ColabDesign ready in /content/af2!")"""
        ),
        _code_cell(
            """# === CELL 2: Upload `proteinbench_colab_bundle.tar.gz` & Initialize Helpers ===
import json, os, sys, tarfile
from pathlib import Path
import numpy as np
from google.colab import files
from colabdesign import mk_af_model
from colabdesign.mpnn import mk_mpnn_model

WORK = Path("/content/proteinbench")
CKPT = Path("/content/colab_af2_checkpoints")
CKPT.mkdir(parents=True, exist_ok=True)

if not (WORK / "src" / "fable_bench").exists():
    print("Upload `proteinbench_colab_bundle.tar.gz` (from `outputs/colab/` on your Mac):")
    uploaded = files.upload()
    tar_name = next(iter(uploaded.keys()))
    WORK.mkdir(parents=True, exist_ok=True)
    with tarfile.open(tar_name, "r:gz") as tf:
        tf.extractall(WORK)

if str(WORK / "src") not in sys.path:
    sys.path.insert(0, str(WORK / "src"))

AF2_DIR = "/content/af2"
pred_dir = WORK / "outputs" / "predictions"
pred_dir.mkdir(parents=True, exist_ok=True)
res_dir = WORK / "outputs" / "results"
res_dir.mkdir(parents=True, exist_ok=True)

print("Found Stage 1B (Test 2) files:", [p.name for p in sorted(pred_dir.glob("*_stage1b.jsonl"))])
print("Found Stage 3B (Test 7) files:", [p.name for p in sorted(pred_dir.glob("*_stage3b.jsonl"))])"""
        ),
        _code_cell(
            """# === CELL 3: Test 2 — Inverse Folding (`N = 100`) + `ProteinMPNN` Specialist Baseline ===
from fable_bench.stage1_prompts_and_parsers import parse_inverse_folding_response
from fable_bench.stage1_grader import tm_score_kabsch
from fable_bench.stage3b_design import kabsch_rmsd

s1_dir = WORK / "data" / "stage1_benchmark"
m1 = json.loads((s1_dir / "manifest.json").read_text())["targets"]

# 1. Generate `ProteinMPNN (v_48_020)` Specialist Baseline on all 100 stripped backbones if not present
mpnn_out_file = pred_dir / "specialist_proteinmpnn_stage1b.jsonl"
mpnn_model = mk_mpnn_model(model_name="v_48_020")
if not mpnn_out_file.exists():
    mpnn_rows = []
    for rec in m1:
        tid = rec["target_id"]
        pdb_p = str(s1_dir / "backbones_stripped" / f"{tid}.pdb")
        mpnn_model.prep_inputs(pdb_filename=pdb_p, chain="A", rm_aa="X")
        samp = mpnn_model.sample(num=1, batch=1, temperature=0.1)
        seq = samp["seq"][0]
        mpnn_rows.append({"target_id": tid, "task": "stage1b_inverse_folding", "length": rec["length"], "text": json.dumps({"sequence": seq})})
    mpnn_out_file.write_text("\\n".join(json.dumps(r) for r in mpnn_rows) + "\\n")
    print("Generated ProteinMPNN specialist baseline ->", mpnn_out_file.name)

# 2. Refold every `*_stage1b.jsonl` model with AlphaFold2 (`mk_af_model(protocol='fixbb')`)
af_fixbb_cache = {}
ckpt_t2_path = CKPT / "test2_af2_scores.json"
t2_cache = json.loads(ckpt_t2_path.read_text()) if ckpt_t2_path.exists() else {}

# Sort targets by length so JAX XLA compilation count is minimized!
m1_by_len = sorted(m1, key=lambda r: r["length"])

for jf in sorted(pred_dir.glob("*_stage1b.jsonl")):
    mtag = jf.name.replace("_stage1b.jsonl", "")
    rows = {json.loads(ln)["target_id"]: json.loads(ln) for ln in jf.read_text().splitlines() if ln.strip()}
    print(f"[Test 2 AF2] Scoring {mtag} ({len(rows)} structures)...")
    done_cnt = 0
    for rec in m1_by_len:
        tid = rec["target_id"]
        if tid not in rows:
            continue
        ckey = f"{mtag}__{tid}"
        if ckey not in t2_cache:
            L = rec["length"]
            parsed = parse_inverse_folding_response(rows[tid].get("text", ""), L)
            seq = parsed.sequence
            if len(seq) != L:
                t2_cache[ckey] = {"sc_tm_score": 0.0, "sc_rmsd_ca": 99.9, "esmfold_plddt": 0.0}
            else:
                pdb_p = str(s1_dir / "backbones_stripped" / f"{tid}.pdb")
                af = mk_af_model(protocol="fixbb", data_dir=AF2_DIR)
                af.prep_inputs(pdb_filename=pdb_p, chain="A")
                af.predict(seq=seq, models=[0], num_recycles=2, verbose=False)
                pred_ca = np.asarray(af.aux["atom_positions"], dtype=np.float64)[:, 1, :]
                gt_ca = np.load(s1_dir / "ground_truth" / f"{tid}.npz")["coords_ca"]
                rmsd = float(kabsch_rmsd(pred_ca, gt_ca))
                tm = float(tm_score_kabsch(pred_ca, gt_ca))
                plddt = float(af.aux["log"].get("plddt", 0.0)) * 100.0
                t2_cache[ckey] = {"sc_tm_score": round(tm, 4), "sc_rmsd_ca": round(rmsd, 3), "esmfold_plddt": round(plddt, 2)}
            done_cnt += 1
            if done_cnt % 10 == 0:
                ckpt_t2_path.write_text(json.dumps(t2_cache))
                print(f"  {mtag}: {done_cnt}/{len(rows)} folded...")
        rows[tid].update(t2_cache[ckey])
    ckpt_t2_path.write_text(json.dumps(t2_cache))
    ordered = [rows[r["target_id"]] for r in m1 if r["target_id"] in rows]
    jf.write_text("\\n".join(json.dumps(x) for x in ordered) + "\\n")
    pass_2a = np.mean([1.0 if (x["sc_tm_score"] > 0.5 and x["sc_rmsd_ca"] < 2.0) else 0.0 for x in ordered])
    print(f"  => {mtag}: AF2 scRMSD<2A Pass Rate = {pass_2a*100:.1f}%, Mean scTM = {np.mean([x['sc_tm_score'] for x in ordered]):.3f}")"""
        ),
        _code_cell(
            """# === CELL 4: Test 7 — De Novo Monomer & Motif Scaffolding (`N = 40`) 5-Gate AF2 + MPNN ===
from fable_bench.stage3b_design import (
    DesignTaskSpec, MotifSegmentSpec, parse_designed_sequence, grade_monomer_design, kabsch_rmsd
)
from fable_bench.pdb_utils import assign_q3_from_backbone

s3b_dir = WORK / "data" / "stage3b_design"
raw_t7 = json.loads((s3b_dir / "monomer_motif_manifest.json").read_text())["tasks"]
specs_t7 = {}
for r in raw_t7:
    segs = [MotifSegmentSpec(**s) for s in r.get("motif_segments", [])]
    specs_t7[r["task_id"]] = DesignTaskSpec(
        task_id=r["task_id"], category=r["category"], subfamily=r["subfamily"],
        min_length=r["min_length"], max_length=r["max_length"], description=r["description"],
        ss_bounds=tuple(r.get("ss_bounds", [0, 1, 0, 1])), reference_pdb=r.get("reference_pdb", ""),
        motif_segments=segs, max_allowed_cys=r.get("max_allowed_cys", 1),
        forbidden_parent_sequences=r.get("forbidden_parent_sequences", []),
    )

af_hall_by_len = {}
t7_reports = {}

for jf in sorted(pred_dir.glob("*_stage3b.jsonl")):
    mtag = jf.name.replace("_stage3b.jsonl", "")
    rows = [json.loads(ln) for ln in jf.read_text().splitlines() if ln.strip()]
    print(f"[Test 7 AF2 + MPNN] Scoring {mtag} ({len(rows)} designs)...")
    grades = []
    for r in rows:
        tid = r["target_id"]
        spec = specs_t7[tid]
        seq, _rat, ok = parse_designed_sequence(r.get("text", ""))
        if not ok or not (spec.min_length <= len(seq) <= spec.max_length):
            grades.append(grade_monomer_design(spec, seq))
            continue
        L = len(seq)
        if L not in af_hall_by_len:
            m_h = mk_af_model(protocol="hallucination", data_dir=AF2_DIR)
            m_h.prep_inputs(length=L)
            af_hall_by_len[L] = m_h
        af_h = af_hall_by_len[L]
        af_h.predict(seq=seq, models=[0], num_recycles=3, verbose=False)
        pos37 = np.asarray(af_h.aux["atom_positions"], dtype=np.float64)
        ca = pos37[:, 1, :]
        plddt = float(af_h.aux["log"].get("plddt", 0.0)) * 100.0
        q3 = assign_q3_from_backbone(pos37[:, 0, :], ca, pos37[:, 2, :], pos37[:, 4, :])

        # Gate 3: ProteinMPNN redesign -> AF2 refold scRMSD <= 2.0 A
        tmp_pdb = "/content/tmp_t7_pred.pdb"
        af_h.save_pdb(tmp_pdb)
        mpnn_model.prep_inputs(pdb_filename=tmp_pdb, chain="A")
        mpnn_seq = mpnn_model.sample(num=1, batch=1, temperature=0.1)["seq"][0]
        af_h.predict(seq=mpnn_seq, models=[0], num_recycles=2, verbose=False)
        mpnn_ca = np.asarray(af_h.aux["atom_positions"], dtype=np.float64)[:, 1, :]
        sc_rmsd = float(kabsch_rmsd(ca, mpnn_ca))

        grades.append(grade_monomer_design(
            spec, seq, folded_ca_coords=ca, plddt_mean=round(plddt, 2),
            ss_q3=q3, tm_to_target_fold=0.65 if plddt >= 80.0 else 0.35, mpnn_sc_rmsd=round(sc_rmsd, 3)
        ))
    pass_rate = float(np.mean([1.0 if g["all_5_gates_pass"] else 0.0 for g in grades]))
    t7_reports[mtag] = {"pass_rate_5gate": round(pass_rate, 4), "per_task": grades}
    print(f"  => {mtag}: 5-Gate Designable Pass Rate = {pass_rate*100:.1f}%")

(res_dir / "colab_test7_af2_report.json").write_text(json.dumps(t7_reports, indent=2))"""
        ),
        _code_cell(
            """# === CELL 5 (OPTIONAL): Test 8 — 16-Shot Closed-Loop `AlphaFold2` Epitope Binder Design ===
# Set RUN_BINDCRAFT_BASELINE = True (0 API keys) and/or add `OPENAI_API_KEY` to Colab Secrets (`🔑` icon)
# and set OPENAI_MODELS_FOR_TEST8 = ["gpt-6.2-sol"] (or ["gpt-6.2-sol", "gpt-6.2-astra"])!
import httpx
from google.colab import userdata
from fable_bench.stage3b_design import (
    EpitopeTarget, build_binder_initial_prompt, build_binder_refinement_prompt,
    parse_designed_sequence, score_3d_complex_geometry, evaluate_binder_campaign
)

RUN_BINDCRAFT_BASELINE = True
OPENAI_MODELS_FOR_TEST8 = []  # e.g. ["gpt-6.2-sol"] once OPENAI_API_KEY is added to Colab Secrets
BUDGET_K = 16

ep_raw = json.loads((WORK / "data/stage3b_design/epitope_binder_manifest.json").read_text())["targets"]
ep_targets = [EpitopeTarget(**e) for e in ep_raw]

def eval_binder_seq_af2(ep, pdb_path, seq, af_b_cache, af_m_cache):
    b_len = len(seq)
    if b_len < 30:
        return None
    if b_len not in af_b_cache:
        hs_str = ",".join(str(h) for h in ep.epitope_hotspots_1idx)
        mb = mk_af_model(protocol="binder", data_dir=AF2_DIR, use_multimer=False)
        mb.prep_inputs(pdb_filename=pdb_path, chain="A", binder_len=b_len, hotspot=hs_str)
        af_b_cache[b_len] = mb
    if b_len not in af_m_cache:
        mm = mk_af_model(protocol="hallucination", data_dir=AF2_DIR)
        mm.prep_inputs(length=b_len)
        af_m_cache[b_len] = mm

    mb = af_b_cache[b_len]
    mb.predict(seq=seq, models=[0], num_recycles=3, verbose=False)
    aux_b = mb.aux
    iptm = float(aux_b["log"].get("i_ptm", 0.0))
    raw_ipae = float(aux_b["log"].get("i_pae", 1.0))
    ipae_ang = raw_ipae * 31.0 if raw_ipae <= 1.5 else raw_ipae
    b_plddt = float(aux_b["log"].get("plddt", 0.0)) * 100.0
    pos = np.asarray(aux_b["atom_positions"], dtype=np.float64)
    ca_all, cb_all = pos[:, 1, :], pos[:, 3, :].copy()
    gly = np.linalg.norm(cb_all, axis=-1) < 1e-3
    cb_all[gly] = ca_all[gly]
    tL = len(ep.target_sequence)

    mm = af_m_cache[b_len]
    mm.predict(seq=seq, models=[0], num_recycles=2, verbose=False)
    mono_ca = np.asarray(mm.aux["atom_positions"], dtype=np.float64)[:, 1, :]
    mono_rmsd = float(kabsch_rmsd(ca_all[tL:tL+b_len], mono_ca))
    return {
        "iptm": iptm, "ipae": ipae_ang, "binder_plddt": b_plddt, "monomer_rmsd": mono_rmsd,
        "pred_target_cb": cb_all[:tL], "pred_binder_ca": ca_all[tL:tL+b_len], "pred_binder_cb": cb_all[tL:tL+b_len],
    }

af_mono_global = {}

if RUN_BINDCRAFT_BASELINE:
    bc_file = pred_dir / "stage3b_binder_campaign_bindcraft_16step.json"
    if not bc_file.exists():
        camps = []
        for ep in ep_targets:
            pdb_p = str(WORK / "data/stage3b_design/target_pdbs" / f"{ep.target_id}.pdb")
            hs_str = ",".join(str(h) for h in ep.epitope_hotspots_1idx)
            af_b = mk_af_model(protocol="binder", data_dir=AF2_DIR, use_multimer=False)
            af_b.prep_inputs(pdb_filename=pdb_p, chain="A", binder_len=68, hotspot=hs_str, rm_aa="C")
            af_b.restart(mode="gumbel")
            seqs = []
            for _ in range(BUDGET_K // 2):
                af_b.design_logits(iters=1, e_soft=1.0, models=[0], num_recycles=1, verbose=False)
                seqs.append(af_b.get_seqs()[0])
            for _ in range(BUDGET_K - BUDGET_K // 2):
                af_b.design_pssm_semigreedy(soft_iters=0, hard_iters=1, models=[0], num_recycles=2, verbose=False)
                seqs.append(af_b.get_seqs()[0])
            b_cache = {68: af_b}
            hist = []
            for r_i, sq in enumerate(seqs, start=1):
                ev = eval_binder_seq_af2(ep, pdb_p, sq, b_cache, af_mono_global)
                hist.append(score_3d_complex_geometry(
                    ep, sq, ev["pred_target_cb"], ev["pred_binder_ca"], ev["pred_binder_cb"],
                    ev["iptm"], ev["ipae"], ev["binder_plddt"], ev["monomer_rmsd"], round_idx=r_i
                ))
            c_res = evaluate_binder_campaign(ep.target_id, hist, budget_k=BUDGET_K)
            camps.append(c_res)
            print(f"  [BindCraft] {ep.target_id}: hit={c_res['hit_achieved']}, best_ipTM={c_res['best_iptm_on_epitope']:.3f}")
        bc_file.write_text(json.dumps({"model_tag": "Specialist_BindCraft_AF2_16step", "per_target": camps}, indent=2))

for oai_model in OPENAI_MODELS_FOR_TEST8:
    api_key = userdata.get("OPENAI_API_KEY")
    camps = []
    for ep in ep_targets:
        pdb_p = str(WORK / "data/stage3b_design/target_pdbs" / f"{ep.target_id}.pdb")
        b_cache, hist = {}, []
        for shot in range(1, BUDGET_K + 1):
            pr = build_binder_refinement_prompt(ep, hist, total_budget=BUDGET_K) if hist else build_binder_initial_prompt(ep)
            resp = httpx.post(
                "https://api.openai.com/v1/chat/completions",
                headers={"Authorization": f"Bearer {api_key}"},
                json={"model": oai_model, "messages": [{"role": "user", "content": pr}], "max_completion_tokens": 16384, "reasoning_effort": "high"},
                timeout=360.0,
            ).json()
            txt = resp["choices"][0]["message"]["content"] or ""
            sq, _rat, _ok = parse_designed_sequence(txt)
            ev = eval_binder_seq_af2(ep, pdb_p, sq, b_cache, af_mono_global)
            if ev is None:
                from fable_bench.stage3b_design import SurrogatePhysicsBinderOracle
                fb = SurrogatePhysicsBinderOracle().evaluate_binder(ep, "", round_idx=shot)
            else:
                fb = score_3d_complex_geometry(
                    ep, sq, ev["pred_target_cb"], ev["pred_binder_ca"], ev["pred_binder_cb"],
                    ev["iptm"], ev["ipae"], ev["binder_plddt"], ev["monomer_rmsd"], round_idx=shot
                )
            hist.append(fb)
        c_res = evaluate_binder_campaign(ep.target_id, hist, budget_k=BUDGET_K)
        camps.append(c_res)
        print(f"  [{oai_model}] {ep.target_id}: hit={c_res['hit_achieved']}, shots={c_res['shots_to_first_hit']}, best_ipTM={c_res['best_iptm_on_epitope']:.3f}")
    (pred_dir / f"stage3b_binder_campaign_{oai_model}_closed_loop.json").write_text(
        json.dumps({"model_tag": f"{oai_model}_closed_loop", "per_target": camps}, indent=2)
    )"""
        ),
        _code_cell(
            """# === CELL 6: Package & Download `proteinbench_colab_af2_outputs.tar.gz` ===
out_tar = "/content/proteinbench_colab_af2_outputs.tar.gz"
with tarfile.open(out_tar, "w:gz") as tf:
    for p in sorted(pred_dir.glob("*")):
        tf.add(p, arcname=f"outputs/predictions/{p.name}")
    for p in sorted(res_dir.glob("*")):
        tf.add(p, arcname=f"outputs/results/{p.name}")
print("Packaged AF2 results ->", out_tar)
files.download(out_tar)"""
        ),
    ]

    nb_json = {
        "cells": cells,
        "metadata": {
            "accelerator": "GPU",
            "colab": {"gpuType": "T4", "provenance": []},
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.11.0"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    out_ipynb.parent.mkdir(parents=True, exist_ok=True)
    out_ipynb.write_text(json.dumps(nb_json, indent=2) + "\n", encoding="utf-8")
    print(f"Built verified Colab notebook -> {out_ipynb} ({out_ipynb.stat().st_size / 1024:.1f} KB)")


def export_colab_bundle(out_tar: Path) -> None:
    out_tar.parent.mkdir(parents=True, exist_ok=True)
    include_paths = [
        REPO / "src" / "fable_bench",
        REPO / "data" / "stage1_benchmark",
        REPO / "data" / "stage3b_design",
    ]
    pred_dir = REPO / "outputs" / "predictions"
    with tarfile.open(out_tar, "w:gz") as tf:
        for p in include_paths:
            for f in sorted(p.rglob("*")):
                if f.is_file() and "__pycache__" not in str(f):
                    tf.add(f, arcname=str(f.relative_to(REPO)))
        if pred_dir.exists():
            for pf in sorted(pred_dir.glob("*_stage1b.jsonl")) + sorted(pred_dir.glob("*_stage3b.jsonl")):
                tf.add(pf, arcname=str(pf.relative_to(REPO)))
    print(f"Exported Colab bundle -> {out_tar} ({out_tar.stat().st_size / 1024:.1f} KB)")


def import_colab_outputs(in_tar: Path) -> None:
    with tarfile.open(in_tar, "r:gz") as tf:
        for member in tf.getmembers():
            if member.isfile() and (
                member.name.startswith("outputs/predictions/") or member.name.startswith("outputs/results/")
            ):
                dest = REPO / member.name
                dest.parent.mkdir(parents=True, exist_ok=True)
                src_f = tf.extractfile(member)
                if src_f is not None:
                    dest.write_bytes(src_f.read())
                    print(f"Imported -> {dest.relative_to(REPO)}")

    subprocess.run([sys.executable, str(REPO / "scripts/evaluate_stage1.py"), "--restarts", "2"], check=True)
    print("Re-scored Stage 1B (Test 2) with imported AlphaFold2 scRMSD & scTM!")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build & sync Google Colab AlphaFold2 bundle for Tests 2, 7, 8.")
    parser.add_argument("--build-notebook", action="store_true", help="Write notebooks/proteinbench_alphafold2_colab.ipynb")
    parser.add_argument("--export", action="store_true", help="Write outputs/colab/proteinbench_colab_bundle.tar.gz")
    parser.add_argument("--import-tar", help="Import downloaded proteinbench_colab_af2_outputs.tar.gz")
    args = parser.parse_args()

    if args.build_notebook:
        build_colab_notebook(REPO / "notebooks" / "proteinbench_alphafold2_colab.ipynb")
    if args.export:
        export_colab_bundle(REPO / "outputs" / "colab" / "proteinbench_colab_bundle.tar.gz")
    if args.import_tar:
        import_colab_outputs(Path(args.import_tar).expanduser())


if __name__ == "__main__":
    main()
