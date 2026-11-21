"""Unit & zero-leakage tests for Test 6: Interface Binding Alanine Scan Ranking (`X -> Ala`)."""

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


def test_binding_alanine_ranking_prompt_parser_and_same_aa_separation() -> None:
    scanned_seq = "MWYRKLDEFGHISTVNPQMWYRKLDEFGHISTVNPQ"
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
    assert '"ranking"' in prompt

    # 1. Static Bogan-Thorn 1D hotspot table scores 0.500 coin-flip on same-AA pairs & 0.000 demeaned!
    bt_rows = [[p, aa, BOGAN_THORN_BINDING_HOTSPOT_SCALE[aa]] for p, aa in zip(positions, wt_aas)]
    bt_parsed = parse_binding_alanine_response(json.dumps({"binding_scan": bt_rows}), target)
    assert bt_parsed.parse_ok
    g_bt = grade_binding_alanine_target(target, bt_parsed)
    assert abs(g_bt.same_aa_win_rate - 0.5) < 1e-6
    assert abs(g_bt.spearman_rho_aa_demeaned) < 1e-6
    assert abs((g_bt.interface_3d_auroc or 0.5) - 0.5) < 1e-6

    # 2. Ranked list output (`{"ranking": ["2W->A", "3Y->A", ...]}`) achieves near-1.0 across all ranking metrics!
    desc_idx = np.argsort(-np.asarray(exp_loss))
    ranked_tokens = [f"{positions[k]}{wt_aas[k]}->A" for k in desc_idx]
    true_parsed = parse_binding_alanine_response("```json\n" + json.dumps({"ranking": ranked_tokens}) + "\n```", target)
    assert true_parsed.parse_ok
    assert true_parsed.wt_match_rate == 1.0
    g_true = grade_binding_alanine_target(target, true_parsed)
    assert g_true.spearman_rho > 0.98
    assert g_true.ndcg_at_5 > 0.99
    assert g_true.top_k_precision > 0.99
    assert g_true.same_aa_win_rate > 0.99
    assert g_true.spearman_rho_aa_demeaned > 0.95
    assert g_true.binding_hotspot_auroc > 0.95
    assert (g_true.interface_3d_auroc or 0.0) > 0.95


def test_built_binding_alanine_manifest_and_zero_leakage() -> None:
    bdir = REPO / "data" / "binding_alanine_benchmark"
    manifest = json.loads((bdir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["n_targets"] >= 35
    assert manifest["total_underlying_wetlab_alanine_mutants"] >= 1600
    assert manifest["memorization_stratum_counts"].get("known_pdb_pre2026", 0) >= 20
    assert manifest["memorization_stratum_counts"].get("unseen_or_post2026", 0) >= 6
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
        assert 8 <= len(positions) <= 20
        assert positions == sorted(positions)
        assert len(positions) == len(wt_aas) == len(losses)
        for p, wt in zip(positions, wt_aas):
            assert wt != "A"
            assert sa[p - 1] == wt

        ptxt = prompt_by_id[tid]
        assert tid not in ptxt
        assert '"ranking"' in ptxt
        if rec.get("pdb_id"):
            assert str(rec["pdb_id"]) not in ptxt
        if rec.get("mavedb_urn"):
            assert str(rec["mavedb_urn"]) not in ptxt
