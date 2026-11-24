"""End-to-end tests for Test 5: Monomer Alanine Scan Ranking (`X_i -> A`)."""

from __future__ import annotations

import json
from pathlib import Path
import numpy as np

from protein_without_tools.alanine_scan import (
    AA_TO_ALA_LOOKUP_DDG,
    AlanineScanTarget,
    build_alanine_scan_prompt,
    grade_alanine_scan,
    non_alanine_positions,
    parse_alanine_scan_response,
    select_balanced_alanine_panel,
)

REPO = Path(__file__).resolve().parents[1]


def _make_synthetic_alanine_target() -> AlanineScanTarget:
    seq = "MKLAEKLLEALKEFVEAGKLLQEFVKALGK"
    pos = non_alanine_positions(seq)
    wt = [seq[p - 1] for p in pos]
    core = [1 if (p in (3, 7, 10, 14, 15, 20, 24, 25)) else 0 for p in pos]
    exp_ddg = [
        round(AA_TO_ALA_LOOKUP_DDG[aa] * (1.35 if c == 1 else 0.12) + (0.25 if c == 1 else -0.05) + 0.002 * p, 4)
        for p, aa, c in zip(pos, wt, core)
    ]
    sel_i, roles = select_balanced_alanine_panel(pos, wt, exp_ddg, target_panel_size=18)
    return AlanineScanTarget(
        target_id="s3a_t001",
        category="de_novo_ddg",
        assay_type="ddg_kcal",
        sequence=seq,
        length=len(seq),
        positions_1idx=[pos[k] for k in sel_i],
        wt_aas=[wt[k] for k in sel_i],
        exp_destab_scores=[exp_ddg[k] for k in sel_i],
        is_core_flags=[core[k] for k in sel_i],
        panel_roles=roles,
    )


def test_ranking_prompt_and_parser_roundtrip() -> None:
    t = _make_synthetic_alanine_target()
    assert len(t.positions_1idx) == 18
    assert t.positions_1idx == sorted(t.positions_1idx)

    prompt = build_alanine_scan_prompt(t)
    assert "s3a_t001" not in prompt
    assert t.sequence in prompt
    assert '"ranking"' in prompt

    # True experimental ranking (#1 most destabilizing first)
    desc_idx = np.argsort(-np.asarray(t.exp_destab_scores))
    ranked_tokens = [f"{t.positions_1idx[k]}{t.wt_aas[k]}->A" for k in desc_idx]
    raw_reply = "```json\n" + json.dumps({"ranking": ranked_tokens}) + "\n```"
    parsed = parse_alanine_scan_response(raw_reply, t)
    assert parsed.parse_ok
    assert parsed.n_matched_positions == 18
    assert parsed.wt_aa_match_rate == 1.0

    grade = grade_alanine_scan(t, parsed)
    assert grade.spearman_rho > 0.98
    assert grade.ndcg_at_5 > 0.99
    assert grade.top_k_precision > 0.99
    assert grade.same_aa_win_rate > 0.99
    assert grade.spearman_rho_aa_demeaned > 0.95
    assert grade.core_burial_auroc is not None and grade.core_burial_auroc > 0.90


def test_static_aa_lookup_scores_coinflip_on_same_aa_and_zero_demeaned() -> None:
    t = _make_synthetic_alanine_target()
    static_rows = [
        [p, aa, AA_TO_ALA_LOOKUP_DDG[aa], AA_TO_ALA_LOOKUP_DDG[aa]]
        for p, aa in zip(t.positions_1idx, t.wt_aas)
    ]
    parsed = parse_alanine_scan_response(json.dumps({"scan": static_rows}), t)
    grade = grade_alanine_scan(t, parsed)
    assert abs(grade.same_aa_win_rate - 0.5) < 1e-6
    assert abs(grade.spearman_rho_aa_demeaned) < 1e-6


def test_built_test5_alanine_ranking_manifest_and_prompts() -> None:
    adir = REPO / "data" / "alanine_scan_benchmark"
    manifest = json.loads((adir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["total_targets"] == 52
    assert manifest["total_underlying_wetlab_alanine_mutants"] >= 5000
    prompts = [
        json.loads(ln)
        for ln in (REPO / "outputs" / "prompts" / "stage3a_alanine_prompts.jsonl").read_text(encoding="utf-8").splitlines()
        if ln.strip()
    ]
    assert len(prompts) == 52
    for rec, pr in zip(manifest["targets"], prompts):
        assert 8 <= rec["n_assayed_positions"] <= 20
        assert rec["target_id"] not in pr["prompt"]
        assert '"ranking"' in pr["prompt"]
