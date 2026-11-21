#!/usr/bin/env python3
"""Export all 622 ProteinBench datapoints (across Tests 1..8) for the website & public release.

Outputs:
  1. `website/data/proteinbench_v1_all_datapoints.jsonl`: Complete self-contained JSONL with
     full sequences, provenance metadata, novelty/memorization strata, prompts, ground-truth
     summaries, and per-datapoint live LLM scores/outputs when available.
  2. `website/data/proteinbench_v1_explorer.json`: Compact browser-ready JSON index (~350 KB)
     powering the interactive "Browse All 622 Datapoints" table & inspector on `website/index.html`.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

REPO = Path("/Users/taluguloglu/.gemini/antigravity/scratch/fable-protein-bench")


def _load_jsonl_by_id(path: Path) -> Dict[str, Dict[str, Any]]:
    if not path.exists():
        return {}
    out: Dict[str, Dict[str, Any]] = {}
    for ln in path.read_text(encoding="utf-8").splitlines():
        if ln.strip():
            obj = json.loads(ln)
            tid = str(obj.get("target_id") or obj.get("pair_id") or obj.get("task_id") or "")
            if tid:
                out[tid] = obj
    return out


def build_all_datapoints() -> tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    pdir = REPO / "outputs" / "prompts"
    rdir = REPO / "outputs" / "results"
    pred_dir = REPO / "outputs" / "predictions"

    p_1a = _load_jsonl_by_id(pdir / "stage1a_forward_prompts.jsonl")
    p_1b = _load_jsonl_by_id(pdir / "stage1b_inverse_prompts.jsonl")
    p_2a = _load_jsonl_by_id(pdir / "stage2a_homology_prompts.jsonl")
    p_2b = _load_jsonl_by_id(pdir / "stage2b_cafa_prompts.jsonl")
    p_3a = _load_jsonl_by_id(pdir / "stage3a_alanine_prompts.jsonl")
    p_6 = _load_jsonl_by_id(pdir / "stage6_binding_alanine_prompts.jsonl")
    p_7 = _load_jsonl_by_id(pdir / "stage3b_denovo_prompts.jsonl")
    p_8 = _load_jsonl_by_id(pdir / "stage3b_binder_round1_prompts.jsonl")

    # Load per-target live LLM grades if evaluated
    s1_per_tid: Dict[str, Dict[str, Any]] = {}
    if (rdir / "stage1_evaluation_report.json").exists():
        s1_rep = json.loads((rdir / "stage1_evaluation_report.json").read_text(encoding="utf-8"))
        for m in s1_rep.get("stage1a_forward_folding", []):
            mname = m["model"]
            if mname not in ("Oracle_TrueContacts", "NoisyOracle_50pctPrec", "Heuristic_1D_LocalOnly", "Null_Random"):
                for g in m.get("per_target", []):
                    s1_per_tid.setdefault(g["target_id"], {})[mname] = {
                        "tm_score": round(float(g["tm_score"]), 3),
                        "p_at_l_sep24": round(float(g["p_at_l_sep24"]), 3),
                        "q3_accuracy": round(float(g["q3_accuracy"]), 3),
                    }

    s5_per_tid: Dict[str, Dict[str, Any]] = {}
    if (rdir / "stage3a_alanine_report.json").exists():
        s5_rep = json.loads((rdir / "stage3a_alanine_report.json").read_text(encoding="utf-8"))
        for m in s5_rep.get("stage3a_alanine_scan", []):
            mname = m["model"]
            if mname not in ("Oracle_Experimental", "Baseline_LocalWindowKD", "Baseline_AA_LookupOnly", "Null_Random"):
                for g in m.get("per_target", []):
                    s5_per_tid.setdefault(g["target_id"], {})[mname] = {
                        "ndcg_at_5": round(float(g["ndcg_at_5"]), 3),
                        "same_aa_win_rate": round(float(g["same_aa_win_rate"]), 3),
                        "spearman_rho": round(float(g["spearman_rho"]), 3),
                    }

    s6_per_tid: Dict[str, Dict[str, Any]] = {}
    if (rdir / "live_stage6_binding_report.json").exists():
        s6_rep = json.loads((rdir / "live_stage6_binding_report.json").read_text(encoding="utf-8"))
        for m in s6_rep.get("test6_binding_alanine_ranking", []):
            mname = m["model"]
            if mname != "Baseline_BoganThorn_1D":
                for g in m.get("per_target", []):
                    s6_per_tid.setdefault(g["target_id"], {})[mname] = {
                        "ndcg_at_5": round(float(g["ndcg_at_5"]), 3),
                        "same_aa_win_rate": round(float(g["same_aa_win_rate"]), 3),
                        "spearman_rho": round(float(g["spearman_rho"]), 3),
                    }

    # Also load raw LLM predictions for Tests 3 & 4 per-datapoint display
    pred_2a: Dict[str, Dict[str, str]] = {}
    for pf in sorted(pred_dir.glob("*_stage2a.jsonl")):
        mtag = pf.name[: -len("_stage2a.jsonl")]
        pred_2a[mtag] = {k: str(v.get("text", ""))[:260] for k, v in _load_jsonl_by_id(pf).items()}

    pred_2b: Dict[str, Dict[str, str]] = {}
    for pf in sorted(pred_dir.glob("*_stage2b.jsonl")):
        mtag = pf.name[: -len("_stage2b.jsonl")]
        pred_2b[mtag] = {k: str(v.get("text", ""))[:260] for k, v in _load_jsonl_by_id(pf).items()}

    full_rows: List[Dict[str, Any]] = []
    compact_rows: List[Dict[str, Any]] = []

    def _add(full_obj: Dict[str, Any], comp_obj: Dict[str, Any]) -> None:
        full_rows.append(full_obj)
        compact_rows.append(comp_obj)

    # --- Test 1 & Test 2 (100 structures each) ---
    m1 = json.loads((REPO / "data/stage1_benchmark/manifest.json").read_text(encoding="utf-8"))
    for rec in m1["targets"]:
        tid = rec["target_id"]
        npz = np.load(REPO / "data/stage1_benchmark/ground_truth" / f"{tid}.npz", allow_pickle=False)
        seq = str(npz["sequence"])
        q3 = str(npz["q3"])
        pdb_tm = round(float(rec.get("pdb_precutoff_tm_bilateral_max", 0.0)), 3)
        stratum = rec.get("stratum", rec.get("temporal_split", ""))
        src = f"PDB {rec['pdb_id']}:{rec['chain_id']} ({rec.get('release_date', '')})"

        # Test 1: Forward Folding
        pr1 = p_1a.get(tid, {}).get("prompt", "")
        gt1 = f"L={rec['length']}, Fold={rec.get('fold_class','')}, Max Pre-Cutoff PDB TM={pdb_tm}, Long-range contacts(|i-j|>=24)={rec.get('n_contacts_sep24',0)}"
        sc1 = s1_per_tid.get(tid, {})
        _add(
            {
                "test_num": 1,
                "test_name": "Forward Folding",
                "datapoint_id": f"T1:{tid}",
                "target_id": tid,
                "source": src,
                "description": rec.get("description", ""),
                "stratum": stratum,
                "length": rec["length"],
                "sequence": seq,
                "q3_ground_truth": q3,
                "ground_truth_summary": gt1,
                "prompt": pr1,
                "live_scores": sc1,
            },
            {
                "t": 1,
                "id": tid,
                "src": src,
                "desc": rec.get("description", "")[:80],
                "st": stratum,
                "len": rec["length"],
                "seq": seq,
                "gt": gt1,
                "pr": "\n".join(pr1.splitlines()[:18]),
                "sc": sc1,
            },
        )

        # Test 2: Inverse Folding
        pr2 = p_1b.get(tid, {}).get("prompt", "")
        pr2_lines = pr2.splitlines()
        pr2_short = "\n".join(pr2_lines[:14] + ([f"... [{len(pr2_lines)-20} 3D backbone rows] ..."] + pr2_lines[-5:] if len(pr2_lines) > 22 else []))
        gt2 = f"Native 1D sequence (L={rec['length']}), Fold={rec.get('fold_class','')}, Pre-Cutoff PDB TM={pdb_tm}"
        _add(
            {
                "test_num": 2,
                "test_name": "Inverse Folding",
                "datapoint_id": f"T2:{tid}",
                "target_id": tid,
                "source": src,
                "description": rec.get("description", ""),
                "stratum": stratum,
                "length": rec["length"],
                "ground_truth_sequence": seq,
                "ground_truth_summary": gt2,
                "prompt": pr2,
                "live_scores": {},
            },
            {
                "t": 2,
                "id": tid,
                "src": src,
                "desc": rec.get("description", "")[:80],
                "st": stratum,
                "len": rec["length"],
                "seq": seq,
                "gt": gt2,
                "pr": pr2_short,
                "sc": {},
            },
        )

    # --- Test 3: Remote Homology Detection (120 SCOPe pairs) ---
    m3 = json.loads((REPO / "data/stage2a_homology/manifest.json").read_text(encoding="utf-8"))
    for rec in m3["pairs"]:
        pid = rec["pair_id"]
        pr3 = p_2a.get(pid, {}).get("prompt", "")
        lbl = "Homolog (Same Superfamily)" if rec["label_homolog"] == 1 else "Non-Homolog Class-Matched Decoy"
        gt3 = f"{lbl} | SCOPe {rec['scop_a']} vs {rec['scop_b']} | SeqID={rec['seq_identity_pct']:.1f}%, SW={rec['blosum_bitscore']:.1f} bits"
        sc3 = {m: {"raw": tx} for m, mp in pred_2a.items() for k, tx in mp.items() if k == pid}
        _add(
            {
                "test_num": 3,
                "test_name": "Remote Homology Detection",
                "datapoint_id": f"T3:{pid}",
                "target_id": pid,
                "source": f"SCOPe {rec['domain_a']} × {rec['domain_b']}",
                "description": f"{rec['scop_a']} vs {rec['scop_b']}",
                "stratum": rec["rung"],
                "length": len(rec["seq_a"]) + len(rec["seq_b"]),
                "sequence_a": rec["seq_a"],
                "sequence_b": rec["seq_b"],
                "label_homolog": rec["label_homolog"],
                "ground_truth_summary": gt3,
                "prompt": pr3,
                "live_scores": sc3,
            },
            {
                "t": 3,
                "id": pid,
                "src": f"SCOPe {rec['domain_a']} × {rec['domain_b']}",
                "desc": lbl,
                "st": rec["rung"],
                "len": len(rec["seq_a"]) + len(rec["seq_b"]),
                "seq": f"A: {rec['seq_a']}\nB: {rec['seq_b']}",
                "gt": gt3,
                "pr": "\n".join(pr3.splitlines()[:18]),
                "sc": sc3,
            },
        )

    # --- Test 4: Protein Function Prediction (60 2026 QuickGO holdouts) ---
    m4 = json.loads((REPO / "data/stage2b_cafa/manifest.json").read_text(encoding="utf-8"))
    for rec in m4["targets"]:
        tid = rec["target_id"]
        pr4 = p_2b.get(tid, {}).get("prompt", "")
        gt4 = (
            f"MF ({len(rec['true_terms_mf'])}): {', '.join(rec['true_terms_mf'][:5])} | "
            f"BP ({len(rec['true_terms_bp'])}): {', '.join(rec['true_terms_bp'][:5])} | "
            f"CC ({len(rec['true_terms_cc'])}): {', '.join(rec['true_terms_cc'][:5])}"
        )
        sc4 = {m: {"raw": tx} for m, mp in pred_2b.items() for k, tx in mp.items() if k == tid}
        _add(
            {
                "test_num": 4,
                "test_name": "Protein Function Prediction (CAFA)",
                "datapoint_id": f"T4:{tid}",
                "target_id": tid,
                "source": f"UniProt {rec['uniprot_accession']} ({rec.get('quickgo_max_exp_date','')})",
                "description": rec.get("organism", ""),
                "stratum": rec.get("temporal_tier", ""),
                "length": rec["length"],
                "sequence": rec["sequence"],
                "true_terms_mf": rec["true_terms_mf"],
                "true_terms_bp": rec["true_terms_bp"],
                "true_terms_cc": rec["true_terms_cc"],
                "ground_truth_summary": gt4,
                "prompt": pr4,
                "live_scores": sc4,
            },
            {
                "t": 4,
                "id": tid,
                "src": f"UniProt {rec['uniprot_accession']}",
                "desc": f"{rec.get('organism','')} ({rec.get('knowledge_type','NK')})",
                "st": rec.get("temporal_tier", ""),
                "len": rec["length"],
                "seq": rec["sequence"],
                "gt": gt4,
                "pr": "\n".join(pr4.splitlines()[:18]),
                "sc": sc4,
            },
        )

    # --- Test 5: Monomer Alanine Scan Ranking (52 proteins) ---
    m5 = json.loads((REPO / "data/alanine_scan_benchmark/manifest.json").read_text(encoding="utf-8"))
    for rec in m5["targets"]:
        tid = rec["target_id"]
        tj = json.loads((REPO / "data/alanine_scan_benchmark/targets" / f"{tid}.json").read_text(encoding="utf-8"))
        pr5 = p_3a.get(tid, {}).get("prompt", "")
        order = sorted(range(len(tj["exp_destab_scores"])), key=lambda k: -tj["exp_destab_scores"][k])
        top5 = ", ".join(f"{tj['positions_1idx'][k]}{tj['wt_aas'][k]}->A ({tj['exp_destab_scores'][k]:.2f})" for k in order[:5])
        gt5 = f"Top-5 experimental destabilizing X->A: {top5}"
        sc5 = s5_per_tid.get(tid, {})
        _add(
            {
                "test_num": 5,
                "test_name": "Monomer Alanine Scan Ranking",
                "datapoint_id": f"T5:{tid}",
                "target_id": tid,
                "source": f"{rec['source_db']}: {rec['source_name']}",
                "description": rec["assay_type"],
                "stratum": rec["category"],
                "length": rec["length"],
                "sequence": tj["sequence"],
                "panel_positions_1idx": tj["positions_1idx"],
                "panel_wt_aas": tj["wt_aas"],
                "exp_destab_scores": tj["exp_destab_scores"],
                "ground_truth_summary": gt5,
                "prompt": pr5,
                "live_scores": sc5,
            },
            {
                "t": 5,
                "id": tid,
                "src": rec["source_name"][:42],
                "desc": rec["assay_type"][:75],
                "st": rec["category"],
                "len": rec["length"],
                "seq": tj["sequence"],
                "gt": gt5,
                "pr": "\n".join(pr5.splitlines()[:20]),
                "sc": sc5,
            },
        )

    # --- Test 6: Interface Binding Alanine Scan Ranking (38 complexes) ---
    m6 = json.loads((REPO / "data/binding_alanine_benchmark/manifest.json").read_text(encoding="utf-8"))
    for rec in m6["targets"]:
        tid = rec["target_id"]
        tj = json.loads((REPO / "data/binding_alanine_benchmark/targets" / f"{tid}.json").read_text(encoding="utf-8"))
        pr6 = p_6.get(tid, {}).get("prompt", "")
        order = sorted(range(len(tj["exp_binding_loss"])), key=lambda k: -tj["exp_binding_loss"][k])
        top5 = ", ".join(f"{tj['positions_1idx'][k]}{tj['wt_aas'][k]}->A ({tj['exp_binding_loss'][k]:.2f})" for k in order[:5])
        gt6 = f"Top-5 interface binding hotspots: {top5} | Filter: {rec.get('monomer_stability_filter','')}"
        sc6 = s6_per_tid.get(tid, {})
        _add(
            {
                "test_num": 6,
                "test_name": "Interface Binding Alanine Scan Ranking",
                "datapoint_id": f"T6:{tid}",
                "target_id": tid,
                "source": rec.get("source_name", tid),
                "description": f"{rec.get('assay_type','')} ({rec.get('published_date','')})",
                "stratum": rec.get("memorization_stratum", ""),
                "length": rec["scanned_length"] + rec["partner_length"],
                "scanned_sequence": tj["scanned_sequence"],
                "partner_sequence": tj["partner_sequence"],
                "panel_positions_1idx": tj["positions_1idx"],
                "panel_wt_aas": tj["wt_aas"],
                "exp_binding_loss": tj["exp_binding_loss"],
                "ground_truth_summary": gt6,
                "prompt": pr6,
                "live_scores": sc6,
            },
            {
                "t": 6,
                "id": tid,
                "src": rec.get("source_name", tid)[:44],
                "desc": rec.get("assay_type", "")[:75],
                "st": rec.get("memorization_stratum", ""),
                "len": rec["scanned_length"] + rec["partner_length"],
                "seq": f"A (scanned): {tj['scanned_sequence']}\nB (partner): {tj['partner_sequence']}",
                "gt": gt6,
                "pr": "\n".join(pr6.splitlines()[:20]),
                "sc": sc6,
            },
        )

    # --- Test 7: De Novo Monomer & Motif Scaffolding (40 tasks) ---
    m7 = json.loads((REPO / "data/stage3b_design/monomer_motif_manifest.json").read_text(encoding="utf-8"))
    for rec in m7["tasks"]:
        tid = rec["task_id"]
        pr7 = p_7.get(tid, {}).get("prompt", "")
        gt7 = f"Length {rec['min_length']}–{rec['max_length']} AA | 5-gate ESMFold+MPNN designability (pLDDT>=75, scTM>=0.80, Motif RMSD<=1.5Å)"
        _add(
            {
                "test_num": 7,
                "test_name": "De Novo Monomer & Motif Scaffolding",
                "datapoint_id": f"T7:{tid}",
                "target_id": tid,
                "source": rec.get("reference_pdb") or "Topology Spec",
                "description": rec.get("description", ""),
                "stratum": rec.get("category", ""),
                "length": int((rec["min_length"] + rec["max_length"]) // 2),
                "ground_truth_summary": gt7,
                "prompt": pr7,
                "live_scores": {},
            },
            {
                "t": 7,
                "id": tid,
                "src": rec.get("subfamily", tid),
                "desc": rec.get("description", "")[:80],
                "st": rec.get("category", ""),
                "len": int((rec["min_length"] + rec["max_length"]) // 2),
                "seq": "(Generative de novo sequence design task)",
                "gt": gt7,
                "pr": "\n".join(pr7.splitlines()[:20]),
                "sc": {},
            },
        )

    # --- Test 8: Closed-Loop Epitope Binder Design (12 targets) ---
    m8 = json.loads((REPO / "data/stage3b_design/epitope_binder_manifest.json").read_text(encoding="utf-8"))
    for rec in m8["targets"]:
        tid = rec["target_id"]
        pr8 = p_8.get(tid, {}).get("prompt", "")
        gt8 = (
            f"Target PDB {rec['pdb_id']}:{rec['target_chain']} ({rec.get('release_date','')}) | "
            f"Epitope hotspots: {rec['epitope_hotspots_1idx']} | Binder {rec['binder_min_len']}–{rec['binder_max_len']} AA"
        )
        _add(
            {
                "test_num": 8,
                "test_name": "Closed-Loop Epitope Binder Design",
                "datapoint_id": f"T8:{tid}",
                "target_id": tid,
                "source": f"{rec['name']} (PDB {rec['pdb_id']})",
                "description": f"Epitope {rec['epitope_hotspots_1idx']}",
                "stratum": rec["tier"],
                "length": len(rec["target_sequence"]),
                "target_sequence": rec["target_sequence"],
                "epitope_hotspots_1idx": rec["epitope_hotspots_1idx"],
                "ground_truth_summary": gt8,
                "prompt": pr8,
                "live_scores": {},
            },
            {
                "t": 8,
                "id": tid,
                "src": f"{rec['name']} ({rec['pdb_id']})",
                "desc": f"Hotspots {rec['epitope_hotspots_1idx']}",
                "st": rec["tier"],
                "len": len(rec["target_sequence"]),
                "seq": rec["target_sequence"],
                "gt": gt8,
                "pr": "\n".join(pr8.splitlines()[:20]),
                "sc": {},
            },
        )

    return full_rows, compact_rows


def main() -> None:
    out_dir = REPO / "website" / "data"
    out_dir.mkdir(parents=True, exist_ok=True)
    full_rows, compact_rows = build_all_datapoints()

    full_jsonl = out_dir / "proteinbench_v1_all_datapoints.jsonl"
    full_jsonl.write_text("\n".join(json.dumps(r) for r in full_rows) + "\n", encoding="utf-8")

    comp_json = out_dir / "proteinbench_v1_explorer.json"
    comp_json.write_text(json.dumps(compact_rows, separators=(",", ":")), encoding="utf-8")

    print(f"Exported {len(full_rows)} total datapoints:")
    print(f"  Full JSONL   -> {full_jsonl} ({full_jsonl.stat().st_size / 1024.0:.1f} KB)")
    print(f"  Explorer JSON-> {comp_json} ({comp_json.stat().st_size / 1024.0:.1f} KB)")


if __name__ == "__main__":
    main()
