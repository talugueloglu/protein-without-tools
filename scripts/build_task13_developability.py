#!/usr/bin/env python3
"""Build Task 13: Wet-Lab Developability Ranking Benchmark (N = 50 panels × 10 antibodies).

Curates strictly within-assay 10-antibody panels from the open Graylab/FLAb wet-lab
developability repository across 4 distinct biophysical properties:
  - thermostability (14 panels, higher Tm = better)
  - polyreactivity  (12 panels, lower PSR / ELISA / Heparin RT = better)
  - aggregation     (12 panels, lower AC-SINS / HIC RT = better)
  - expression      (12 panels, higher HEK titer / expression yield = better)

Zero-leakage guarantees:
  1. Every clinical INN name, germline label, paper author, and raw value is stripped.
  2. Every panel draws all 10 candidates from the exact same experimental assay/CSV
     (no cross-assay unit mixing) stratified across experimental quintiles.
  3. Candidates are assigned shuffled anonymized IDs (`C01`..`C10`), and no sequence
     is reused across panels of the same property.
  4. Audits 1D length-only shortcut baseline (`Spearman rho ~ 0.0`).
"""

from __future__ import annotations

import csv
import io
import json
import math
from pathlib import Path
import random
import sys
from typing import Any, Dict, List, Optional, Tuple

import httpx
import numpy as np

REPO = Path(__file__).resolve().parents[1]
if str(REPO / "src") not in sys.path:
    sys.path.insert(0, str(REPO / "src"))

from protein_without_tools.pdb_utils import CANONICAL_AA_SET
from protein_without_tools.task13_developability import (
    DevelopabilityCandidate,
    build_developability_prompt,
    grade_developability_panel,
)

CACHE_DIR = REPO / "data/raw_v2_cache/task13_flab"
CACHE_DIR.mkdir(parents=True, exist_ok=True)
FLAB_RAW = "https://raw.githubusercontent.com/Graylab/FLAb/main/data"

# (property_type, relative_csv_path, higher_is_better, target_panels, assay_tag)
ASSAY_SPECS: List[Tuple[str, str, bool, int, str]] = [
    # Thermostability (14 panels)
    ("thermostability", "thermostability/garbinski2023_tm1.csv", True, 4, "garbinski2023_tm1"),
    ("thermostability", "thermostability/hutchinson2023enhancement_top200tm1_igg.csv", True, 4, "hutchinson2023_tm1"),
    ("thermostability", "thermostability/jain2024assessment_Tm.csv", True, 2, "jain2024_tm"),
    ("thermostability", "thermostability/jain2017biophysical_Tm.csv", True, 4, "jain2017_tm"),
    # Polyreactivity (12 panels; lower score = cleaner / better developability)
    ("polyreactivity", "polyreactivity/jain2024assessment_PSR.csv", False, 2, "jain2024_psr"),
    ("polyreactivity", "polyreactivity/jain2024assessment_Herapin_RT.csv", False, 2, "jain2024_heparin"),
    ("polyreactivity", "polyreactivity/jain2023identifying_DNP.csv", False, 3, "jain2023_dnp"),
    ("polyreactivity", "polyreactivity/jain2023identifying_Heme2.csv", False, 2, "jain2023_heme"),
    ("polyreactivity", "polyreactivity/jain2017biophysical_BVPELISA.csv", False, 2, "jain2017_bvp"),
    ("polyreactivity", "polyreactivity/jain2017biophysical_ELISA.csv", False, 1, "jain2017_elisa"),
    ("polyreactivity", "polyreactivity/shanehsazzadeh2023unlocking_polyspecificity.csv", False, 1, "shanehsazzadeh2023_poly"),
    # Aggregation / self-association (12 panels; lower AC-SINS / HIC RT = less aggregation)
    ("aggregation", "aggregation/jain2024assessment_ACSINS.csv", False, 2, "jain2024_acsins"),
    ("aggregation", "aggregation/jain2024assessment_HIC.csv", False, 2, "jain2024_hic"),
    ("aggregation", "aggregation/jain2017biophysical_ACSINS.csv", False, 4, "jain2017_acsins"),
    ("aggregation", "aggregation/jain2017biophyscial_HICRT.csv", False, 3, "jain2017_hic"),
    ("aggregation", "aggregation/shanehsazzadeh2023unlocking_ACSINS.csv", False, 1, "shanehsazzadeh2023_acsins"),
    # Expression / solubility yield (12 panels; higher titer / ER = better)
    ("expression", "expression/garbinski2023_exp.csv", True, 4, "garbinski2023_exp"),
    ("expression", "expression/jain2017biophysical_HEK.csv", True, 4, "jain2017_hek"),
    ("expression", "expression/koenig2017mutational_er_g6.csv", True, 4, "koenig2017_er"),
]


def _clean_aa(seq: str) -> Optional[str]:
    s = "".join(ch for ch in (seq or "").strip().upper() if ch.isalpha())
    if len(s) < 60 or len(s) > 160:
        return None
    if any(ch not in CANONICAL_AA_SET for ch in s):
        return None
    return s


def fetch_flab_csv(client: httpx.Client, rel_path: str) -> str:
    dst = CACHE_DIR / rel_path.replace("/", "__")
    if dst.exists() and dst.stat().st_size > 200:
        return dst.read_text(encoding="utf-8")
    r = client.get(f"{FLAB_RAW}/{rel_path}", timeout=25.0)
    r.raise_for_status()
    dst.write_text(r.text, encoding="utf-8")
    return r.text


def parse_flab_rows(csv_text: str, higher_is_better: bool) -> List[Dict[str, Any]]:
    reader = csv.DictReader(io.StringIO(csv_text))
    if not reader.fieldnames:
        return []
    fields = list(reader.fieldnames)
    # Identify heavy, light, and primary measurement column (3rd column typically)
    h_col = next((c for c in fields if c.strip().lower() == "heavy"), fields[0])
    l_col = next((c for c in fields if c.strip().lower() == "light"), fields[1] if len(fields) > 1 else None)
    val_col = next(
        (c for c in fields if c not in (h_col, l_col) and c.strip().lower() != "fitness"),
        fields[2] if len(fields) > 2 else fields[-1],
    )

    dedup: Dict[str, List[float]] = {}
    for row in reader:
        vh = _clean_aa(row.get(h_col, ""))
        vl = _clean_aa(row.get(l_col, "")) if l_col else None
        if not vh or not vl:
            continue
        raw_v = (row.get(val_col) or "").strip()
        if not raw_v:
            continue
        try:
            v = float(raw_v)
        except ValueError:
            continue
        if math.isnan(v) or math.isinf(v):
            continue
        seq_key = f"{vh} / {vl}"
        dedup.setdefault(seq_key, []).append(v)

    out: List[Dict[str, Any]] = []
    for seq_key, vals in dedup.items():
        med_v = float(np.median(vals))
        desirability = med_v if higher_is_better else -med_v
        out.append({
            "sequence": seq_key,
            "value": round(med_v, 4),
            "desirability": round(desirability, 4),
            "val_col": val_col,
        })
    return out


def sample_stratified_panels(
    rows: List[Dict[str, Any]],
    n_panels: int,
    panel_size: int,
    used_seqs: set[str],
    rng: random.Random,
) -> List[List[Dict[str, Any]]]:
    """Build disjoint panels stratified across experimental quintiles so every panel has strong dynamic range."""
    avail = [r for r in rows if r["sequence"] not in used_seqs]
    if len(avail) < panel_size:
        return []

    # Cap how many panels we can build without repeating sequences
    max_possible = len(avail) // panel_size
    k_panels = min(n_panels, max_possible)
    if k_panels == 0:
        return []

    # Sort by desirability and slice into `panel_size` quantile strata (each stratum contributes 1 candidate/panel)
    avail_sorted = sorted(avail, key=lambda r: r["desirability"])
    n_avail = len(avail_sorted)
    strata: List[List[Dict[str, Any]]] = []
    for s in range(panel_size):
        i0 = int(round(s * n_avail / panel_size))
        i1 = int(round((s + 1) * n_avail / panel_size))
        bucket = list(avail_sorted[i0:max(i0 + 1, i1)])
        rng.shuffle(bucket)
        strata.append(bucket)

    panels: List[List[Dict[str, Any]]] = []
    for p_idx in range(k_panels):
        chosen: List[Dict[str, Any]] = []
        chosen_set: set[str] = set()
        for s in range(panel_size):
            pick = None
            for cand in strata[s]:
                if cand["sequence"] not in used_seqs and cand["sequence"] not in chosen_set:
                    pick = cand
                    break
            if pick is None:
                # fallback to any remaining unused sequence
                for cand in avail_sorted:
                    if cand["sequence"] not in used_seqs and cand["sequence"] not in chosen_set:
                        pick = cand
                        break
            if pick is None:
                break
            chosen.append(pick)
            chosen_set.add(pick["sequence"])

        if len(chosen) < panel_size:
            break
        # Check non-degenerate dynamic range (at least 6 distinct measured values)
        distinct_vals = len({round(c["desirability"], 4) for c in chosen})
        if distinct_vals < 6:
            continue
        for c in chosen:
            used_seqs.add(c["sequence"])
        rng.shuffle(chosen)
        panels.append(chosen)

    return panels


def build_task13() -> None:
    rng = random.Random(20261126)
    targets: List[Dict[str, Any]] = []
    prompts: List[Dict[str, Any]] = []
    used_by_prop: Dict[str, set[str]] = {}
    counts_by_prop: Dict[str, int] = {}

    null_len_rhos: List[float] = []
    null_len_concs: List[float] = []

    with httpx.Client(follow_redirects=True) as client:
        idx = 0
        for prop, rel_path, higher_is_better, n_panels, assay_tag in ASSAY_SPECS:
            csv_text = fetch_flab_csv(client, rel_path)
            rows = parse_flab_rows(csv_text, higher_is_better=higher_is_better)
            used_set = used_by_prop.setdefault(prop, set())
            panels = sample_stratified_panels(rows, n_panels=n_panels, panel_size=10, used_seqs=used_set, rng=rng)
            print(f"  [{prop:16s}] {assay_tag:26s}: {len(rows):4d} unique Fvs -> {len(panels)}/{n_panels} panels", flush=True)

            for p_items in panels:
                idx += 1
                tid = f"t13_d{idx:03d}"
                panel = [
                    DevelopabilityCandidate(
                        cand_id=f"C{j + 1:02d}",
                        sequence=item["sequence"],
                        value=item["value"],
                        desirability=item["desirability"],
                    )
                    for j, item in enumerate(p_items)
                ]
                prompt_text = build_developability_prompt(prop, "antibody_fv", panel)

                # Anti-cheat null audit: total sequence length alone
                len_scores = {c.cand_id: float(len(c.sequence)) for c in panel}
                g_len = grade_developability_panel(panel, [], predicted_scores=len_scores)
                null_len_rhos.append(g_len["spearman_rho"])
                null_len_concs.append(g_len["pairwise_accuracy"])

                counts_by_prop[prop] = counts_by_prop.get(prop, 0) + 1
                targets.append({
                    "target_id": tid,
                    "property_type": prop,
                    "stratum": prop,
                    "modality": "antibody_fv",
                    "source_assay": assay_tag,
                    "source_csv": rel_path,
                    "metric_column": p_items[0]["val_col"],
                    "higher_raw_is_better": higher_is_better,
                    "n_candidates": len(panel),
                    "value_range": [round(min(c.value for c in panel), 3), round(max(c.value for c in panel), 3)],
                    "candidates": [
                        {
                            "cand_id": c.cand_id,
                            "sequence": c.sequence,
                            "value": c.value,
                            "desirability": c.desirability,
                        }
                        for c in panel
                    ],
                })
                prompts.append({
                    "custom_id": f"task13_{tid}",
                    "target_id": tid,
                    "task": "task13_wet_lab_developability",
                    "stratum": prop,
                    "property_type": prop,
                    "n_candidates": len(panel),
                    "prompt": prompt_text,
                })

    out_dir = REPO / "data/task13_developability"
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "benchmark": "protein-without-tools-task13-wet-lab-developability",
        "source": "Graylab/FLAb (open wet-lab antibody developability datasets)",
        "n_targets": len(targets),
        "by_property": counts_by_prop,
        "anti_cheat_audit": {
            "length_null_spearman": round(float(np.mean(null_len_rhos)), 4),
            "length_null_concordance": round(float(np.mean(null_len_concs)), 4),
        },
        "targets": targets,
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    (REPO / "outputs/prompts/task13_developability_prompts.jsonl").write_text(
        "\n".join(json.dumps(r) for r in prompts) + "\n", encoding="utf-8"
    )
    print("\n=== TASK 13 (DEVELOPABILITY) BUILT ===")
    print(f"  Targets: {manifest['n_targets']} | {manifest['by_property']}")
    print(f"  Anti-cheat audit: {manifest['anti_cheat_audit']}")


if __name__ == "__main__":
    build_task13()
