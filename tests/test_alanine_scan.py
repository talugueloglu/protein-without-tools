"""End-to-end tests for Stage 3A In-Silico Alanine Scan (`X_i -> A`) module."""

from __future__ import annotations

import json
import numpy as np

from fable_bench.alanine_scan import (
    AA_TO_ALA_LOOKUP_DDG,
    AlanineScanTarget,
    build_alanine_scan_prompt,
    grade_alanine_scan,
    non_alanine_positions,
    parse_alanine_scan_response,
)


def _make_synthetic_alanine_target() -> AlanineScanTarget:
    # 30-residue amphipathic sequence with buried vs surface Leucines, Valines, Phenylalanines
    seq = "MKLAEKLLEALKEFVEAGKLLQEFVKALGK"
    pos = non_alanine_positions(seq)
    wt = [seq[p - 1] for p in pos]
    # Alternating core burial pattern (`i % 3 == 0` or `i % 4 == 0` buried)
    core = [1 if (p in (3, 7, 10, 14, 15, 20, 24, 25)) else 0 for p in pos]
    exp_ddg = [
        round(AA_TO_ALA_LOOKUP_DDG[aa] * (1.35 if c == 1 else 0.12) + (0.25 if c == 1 else -0.05), 3)
        for aa, c in zip(wt, core)
    ]
    # Active-site surface residues at pos 6 (K) and 23 (E): stable fold (ddg ~ 0), high functional loss!
    exp_func = [
        round(0.95 if p in (6, 23) else min(1.0, max(0.0, d / 3.5)), 3)
        for p, d in zip(pos, exp_ddg)
    ]
    return AlanineScanTarget(
        target_id="s3a_t001",
        category="dual_stability_function",
        assay_type="ddg_kcal",
        sequence=seq,
        length=len(seq),
        positions_1idx=pos,
        wt_aas=wt,
        exp_destab_scores=exp_ddg,
        exp_func_loss_scores=exp_func,
        is_core_flags=core,
    )


def test_prompt_and_parser_roundtrip() -> None:
    t = _make_synthetic_alanine_target()
    prompt = build_alanine_scan_prompt(t)
    assert "s3a_t001" not in prompt
    assert t.sequence in prompt

    # Simulate realistic LLM JSON output wrapped in markdown fences
    rows = [[p, aa, d, fn] for p, aa, d, fn in zip(t.positions_1idx, t.wt_aas, t.exp_destab_scores, t.exp_func_loss_scores)]
    raw_reply = "```json\n" + json.dumps({"scan": rows}) + "\n```"
    parsed = parse_alanine_scan_response(raw_reply, t)
    assert parsed.parse_ok
    assert parsed.n_matched_positions == len(t.positions_1idx)
    assert parsed.wt_aa_match_rate == 1.0

    grade = grade_alanine_scan(t, parsed)
    assert grade.spearman_rho > 0.98
    assert grade.spearman_rho_aa_demeaned > 0.95
    assert grade.core_burial_auroc is not None and grade.core_burial_auroc > 0.90
    assert grade.func_only_hotspot_auroc is not None and grade.func_only_hotspot_auroc > 0.90


def test_static_aa_lookup_scores_zero_on_aa_demeaned_metric() -> None:
    """Verify that a static 20-AA hydrophobicity table gets >0 raw rho but EXACTLY 0.0 AA-demeaned rho!"""
    t = _make_synthetic_alanine_target()
    static_rows = [
        [p, aa, AA_TO_ALA_LOOKUP_DDG[aa], AA_TO_ALA_LOOKUP_DDG[aa] / 3.0]
        for p, aa in zip(t.positions_1idx, t.wt_aas)
    ]
    parsed = parse_alanine_scan_response(json.dumps({"scan": static_rows}), t)
    grade = grade_alanine_scan(t, parsed)
    assert grade.spearman_rho > 0.30
    assert abs(grade.spearman_rho_aa_demeaned) < 1e-6
    _ = np
