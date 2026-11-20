"""Unit & zero-leakage tests for Test 6: Interface Binding Alanine Scan (`X -> Ala`)."""

from __future__ import annotations

import json
from pathlib import Path
import re

import numpy as np

from fable_bench.binding_alanine_scan import (
    BOGAN_THORN_BINDING_HOTSPOT_SCALE,
    BindingAlanineTarget,
    build_binding_alanine_prompt,
    grade_binding_alanine_target,
    parse_binding_alanine_response,
)

REPO = Path("/Users/taluguloglu/.gemini/antigravity/scratch/fable-protein-bench")
SAFETY_BLOCKLIST = re.compile(
    r"(virus|viral|virion|capsid|phage|toxin|venom|ricin|abrin|"
    r"botulinum|anthrax|anthracis|ebola|marburg|variola|influenza|corona|sars|\bhiv\b|gp120|cov2)",
    re.IGNORECASE,
)


def test_binding_alanine_prompt_parser_and_aa_demeaned_separation() -> None:
    # Construct a synthetic complex where Trp/Tyr/Arg appear BOTH inside the binding interface
    # (high binding loss) AND outside the binding interface (near-zero binding loss).
    scanned_seq = "MWYRKLDEFGHISTVNPQMWYRKLDEFGHISTVNPQ"  # 36 AA: 1..18 = interface, 19..36 = non-interface
    partner_seq = "ACDEFGHIKLMNPQRSTVWYACDEFGHIKLMNPQRS"
    positions = [2, 3, 4, 5, 6, 7, 20, 21, 22, 23, 24, 25]
    wt_aas = [scanned_seq[p - 1] for p in positions]
    iface_flags = [1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0]
    exp_loss = [3.2, 2.8, 2.3, 1.1, 1.5, 1.2, 0.1, 0.0, -0.1, 0.05, 0.15, 0.0]

    target = BindingAlanineTarget(
        target_id="s6_test999",
        category="skempi_biophysical_ddg",
        assay_type="spr_itc_ddg_kcal",
        scanned_sequence=scanned_seq,
        partner_sequence=partner_seq,
        positions_1idx=positions,
        wt_aas=wt_aas,
        exp_binding_loss=exp_loss,
        is_interface_flags=iface_flags,
    )

    prompt = build_binding_alanine_prompt(target)
    assert "s6_test999" not in prompt
    assert scanned_seq in prompt
    assert partner_seq in prompt

    # 1. Static Bogan-Thorn 1D hotspot table scores identically on interface vs non-interface W/Y/R -> 0.000 demeaned!
    bt_rows = [[p, aa, BOGAN_THORN_BINDING_HOTSPOT_SCALE[aa]] for p, aa in zip(positions, wt_aas)]
    bt_parsed = parse_binding_alanine_response("```json\n" + json.dumps({"binding_scan": bt_rows}) + "\n```", target)
    assert bt_parsed.parse_ok
    assert bt_parsed.wt_match_rate == 1.0
    g_bt = grade_binding_alanine_target(target, bt_parsed)
    assert abs(g_bt.spearman_rho_aa_demeaned) < 1e-6
    assert abs((g_bt.interface_3d_auroc or 0.5) - 0.5) < 1e-6

    # 2. Interface-aware predictor separates interface W/Y/R from surface W/Y/R -> high demeaned rho & 3D AUROC!
    true_rows = [[p, aa, round(v * 0.95, 3)] for p, aa, v in zip(positions, wt_aas, exp_loss)]
    true_parsed = parse_binding_alanine_response(json.dumps({"binding_scan": true_rows}), target)
    g_true = grade_binding_alanine_target(target, true_parsed)
    assert g_true.spearman_rho > 0.95
    assert g_true.spearman_rho_aa_demeaned > 0.90
    assert g_true.binding_hotspot_auroc > 0.95
    assert (g_true.interface_3d_auroc or 0.0) > 0.95


def test_built_binding_alanine_manifest_and_zero_leakage() -> None:
    bdir = REPO / "data" / "binding_alanine_benchmark"
    manifest = json.loads((bdir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["n_targets"] >= 35
    assert manifest["total_assayed_alanine_mutants"] >= 2500
    for req_cat in ("skempi_biophysical_ddg", "oncology_alphaseq_kd", "ppi_binding_dms", "post_may2026_mavedb_binding"):
        assert manifest["category_counts"].get(req_cat, 0) >= 3

    prompts_path = REPO / "outputs" / "prompts" / "stage6_binding_alanine_prompts.jsonl"
    prompt_lines = [json.loads(ln) for ln in prompts_path.read_text(encoding="utf-8").splitlines() if ln.strip()]
    assert len(prompt_lines) == manifest["n_targets"]
    prompt_by_id = {r["target_id"]: r["prompt"] for r in prompt_lines}

    for rec in manifest["targets"]:
        tid = rec["target_id"]
        assert not SAFETY_BLOCKLIST.search(str(rec.get("source_name") or ""))
        tobj = json.loads((bdir / "targets" / f"{tid}.json").read_text(encoding="utf-8"))
        sa = tobj["scanned_sequence"]
        sb = tobj["partner_sequence"]
        assert len(sa) >= 25 and len(sb) >= 25
        positions = tobj["positions_1idx"]
        wt_aas = tobj["wt_aas"]
        losses = tobj["exp_binding_loss"]
        assert len(positions) >= 8
        assert len(positions) == len(wt_aas) == len(losses)
        for p, wt in zip(positions, wt_aas):
            assert wt != "A"
            assert sa[p - 1] == wt

        ptxt = prompt_by_id[tid]
        assert tid not in ptxt
        if rec.get("pdb_id"):
            assert str(rec["pdb_id"]) not in ptxt
        if rec.get("mavedb_urn"):
            assert str(rec["mavedb_urn"]) not in ptxt
