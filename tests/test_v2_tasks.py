"""Comprehensive anti-cheat, zero-leakage, manifest integrity, and parser/grader unit tests
for Proteins Without Tools v2 (Tasks 9–13).
"""

from __future__ import annotations

import json
from pathlib import Path
import re
import sys
from typing import Any, Dict, List

import pytest

REPO = Path(__file__).resolve().parents[1]
if str(REPO / "src") not in sys.path:
    sys.path.insert(0, str(REPO / "src"))

from protein_without_tools.task9_epistasis import DoubleMutantRecord, grade_epistasis_panel, parse_epistasis_response
from protein_without_tools.task10_ligand_site import SiteCandidate, grade_site_ranking_panel, parse_site_ranking_response
from protein_without_tools.task11_dynamics import DynamicsResidue, grade_dynamics_panel
from protein_without_tools.task13_developability import (
    DevelopabilityCandidate,
    grade_developability_panel,
    parse_developability_response,
)

FORBIDDEN_INN_AND_SOURCE_TOKENS = [
    "adalimumab", "trastuzumab", "pembrolizumab", "nivolumab", "golimumab",
    "rituximab", "bevacizumab", "cetuximab", "infliximab", "omalizumab",
    "elotuzumab", "ixekizumab", "garbinski", "hutchinson", "shanehsazzadeh",
    "koenig2017", "jain2017", "jain2023", "jain2024", "megascale", "bioemu",
]


def _load_jsonl(path: Path) -> List[Dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


# ---------------------------------------------------------------------------
# 1. Manifest Counts & Prompt 1-to-1 Alignment
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "manifest_rel,prompt_rel,expected_n",
    [
        ("data/task9_epistasis/manifest.json", "outputs/prompts/task9_epistasis_prompts.jsonl", 60),
        ("data/task10_ligand_site/manifest.json", "outputs/prompts/task10_ligand_site_prompts.jsonl", 60),
        ("data/task11_dynamics/manifest.json", "outputs/prompts/task11_dynamics_prompts.jsonl", 50),
        ("data/task12_nucleic_acid/manifest.json", "outputs/prompts/task12_nucleic_acid_prompts.jsonl", 50),
        ("data/task13_developability/manifest.json", "outputs/prompts/task13_developability_prompts.jsonl", 50),
    ],
)
def test_v2_manifest_and_prompt_alignment(manifest_rel: str, prompt_rel: str, expected_n: int) -> None:
    manifest = json.loads((REPO / manifest_rel).read_text(encoding="utf-8"))
    prompts = _load_jsonl(REPO / prompt_rel)
    assert manifest["n_targets"] == expected_n
    assert len(manifest["targets"]) == expected_n
    assert len(prompts) == expected_n
    m_ids = [t["target_id"] for t in manifest["targets"]]
    p_ids = [p["target_id"] for p in prompts]
    assert m_ids == p_ids


# ---------------------------------------------------------------------------
# 2. Zero-Leakage Audits Across All 270 New Prompts
# ---------------------------------------------------------------------------

def test_task9_zero_leakage() -> None:
    manifest = json.loads((REPO / "data/task9_epistasis/manifest.json").read_text(encoding="utf-8"))
    prompts = {p["target_id"]: p["prompt"] for p in _load_jsonl(REPO / "outputs/prompts/task9_epistasis_prompts.jsonl")}
    for t in manifest["targets"]:
        pr = prompts[t["target_id"]]
        low = pr.lower()
        wt_name = str(t.get("wt_name", ""))
        if len(wt_name) >= 4:
            assert wt_name.lower() not in low, f"Leaked wt_name {wt_name} in {t['target_id']}"
        for tok in FORBIDDEN_INN_AND_SOURCE_TOKENS:
            assert tok not in low
        # Ensure ground-truth coupling delta_delta_dG is not leaked
        assert "delta_delta_dg" not in low
        assert "is_strong_epistasis" not in low


def test_task10_and_12_temporal_cutoff_and_zero_leakage() -> None:
    for mrel, prel in (
        ("data/task10_ligand_site/manifest.json", "outputs/prompts/task10_ligand_site_prompts.jsonl"),
        ("data/task12_nucleic_acid/manifest.json", "outputs/prompts/task12_nucleic_acid_prompts.jsonl"),
    ):
        manifest = json.loads((REPO / mrel).read_text(encoding="utf-8"))
        prompts = {p["target_id"]: p["prompt"] for p in _load_jsonl(REPO / prel)}
        for t in manifest["targets"]:
            assert t["release_date"] >= "2026-05-01", f"Pre-cutoff PDB {t['pdb_id']} ({t['release_date']})"
            pr = prompts[t["target_id"]]
            # PDB ID must never appear in the prompt
            assert re.search(rf"\b{re.escape(t['pdb_id'])}\b", pr, re.I) is None, f"Leaked PDB {t['pdb_id']}"
            # Chemical component ID (except ion formula in metal_ion) must not leak for drugs
            if t.get("stratum") == "small_molecule_drug" and len(t.get("ligand_comp_id", "")) >= 3:
                # Note: 3-letter comp IDs can rarely appear as an amino-acid triplet inside the protein sequence,
                # so check only outside the protein sequence block.
                pr_no_seq = pr.replace(t["sequence"], "")
                assert re.search(rf"\b{re.escape(t['ligand_comp_id'])}\b", pr_no_seq) is None


def test_task11_zero_leakage() -> None:
    manifest = json.loads((REPO / "data/task11_dynamics/manifest.json").read_text(encoding="utf-8"))
    prompts = {p["target_id"]: p["prompt"] for p in _load_jsonl(REPO / "outputs/prompts/task11_dynamics_prompts.jsonl")}
    for t in manifest["targets"]:
        pr = prompts[t["target_id"]]
        low = pr.lower()
        assert t["test_case"].lower() not in low, f"Leaked UniProt {t['test_case']}"
        for pdb_key in ("pdb_state_a", "pdb_state_b"):
            pdb4 = t[pdb_key].split("_")[0]
            assert re.search(rf"\b{re.escape(pdb4)}\b", pr, re.I) is None, f"Leaked PDB {pdb4}"
        for tok in FORBIDDEN_INN_AND_SOURCE_TOKENS:
            assert tok not in low


def test_task13_zero_leakage_and_within_assay_consistency() -> None:
    manifest = json.loads((REPO / "data/task13_developability/manifest.json").read_text(encoding="utf-8"))
    prompts = {p["target_id"]: p["prompt"] for p in _load_jsonl(REPO / "outputs/prompts/task13_developability_prompts.jsonl")}
    for t in manifest["targets"]:
        pr = prompts[t["target_id"]]
        low = pr.lower()
        assert t["source_assay"].lower() not in low
        for tok in FORBIDDEN_INN_AND_SOURCE_TOKENS:
            assert tok not in low, f"Leaked token {tok} in {t['target_id']}"
        assert len(t["candidates"]) == 10
        # Distinct values ensure non-degenerate panel
        assert len({c["desirability"] for c in t["candidates"]}) >= 6


# ---------------------------------------------------------------------------
# 3. Anti-Cheat Null Shortcut Floors
# ---------------------------------------------------------------------------

def test_v2_anti_cheat_null_floors() -> None:
    m9 = json.loads((REPO / "data/task9_epistasis/manifest.json").read_text(encoding="utf-8"))
    assert m9["anti_cheat_audit"]["additive_null_mean_auroc"] == pytest.approx(0.5, abs=0.01)
    assert m9["anti_cheat_audit"]["additive_null_mean_rho"] == pytest.approx(0.0, abs=0.01)
    assert m9["anti_cheat_audit"]["native_3d_ca_proximity_mean_auroc"] < 0.56

    m10 = json.loads((REPO / "data/task10_ligand_site/manifest.json").read_text(encoding="utf-8"))
    assert m10["anti_cheat_audit"]["aa_1d_lookup_same_aa_win"] == pytest.approx(0.5, abs=0.005)
    assert m10["anti_cheat_audit"]["aa_1d_lookup_auroc"] == pytest.approx(0.5, abs=0.005)

    m11 = json.loads((REPO / "data/task11_dynamics/manifest.json").read_text(encoding="utf-8"))
    assert 0.44 <= m11["anti_cheat_audit"]["top_idp_disorder_auroc"] <= 0.56
    assert 0.44 <= m11["anti_cheat_audit"]["terminus_dist_auroc"] <= 0.56

    m12 = json.loads((REPO / "data/task12_nucleic_acid/manifest.json").read_text(encoding="utf-8"))
    assert m12["anti_cheat_audit"]["charge_1d_lookup_same_aa_win"] == pytest.approx(0.5, abs=0.005)
    assert m12["anti_cheat_audit"]["charge_1d_lookup_auroc"] == pytest.approx(0.5, abs=0.005)

    m13 = json.loads((REPO / "data/task13_developability/manifest.json").read_text(encoding="utf-8"))
    assert abs(m13["anti_cheat_audit"]["length_null_spearman"]) <= 0.12


# ---------------------------------------------------------------------------
# 4. Parser & Grader Round-Trip Tests (Tasks 9–13)
# ---------------------------------------------------------------------------

def test_task9_parser_and_grader_roundtrip() -> None:
    panel = [
        DoubleMutantRecord("A5V+L12I", "A5V", "L12I", 5, 12, 7, 1.0, 0.8, 0.4, -1.4, 1.4, 6.2, True),
        DoubleMutantRecord("G8A+K19E", "G8A", "K19E", 8, 19, 11, 0.7, 0.9, 2.8, 1.2, 1.2, 7.1, True),
        DoubleMutantRecord("D3N+S20T", "D3N", "S20T", 3, 20, 17, 0.5, 0.5, 1.05, 0.05, 0.05, 6.8, False),
        DoubleMutantRecord("V6I+R22K", "V6I", "R22K", 6, 22, 16, 0.6, 0.4, 0.98, -0.02, 0.02, 7.4, False),
    ]
    perfect_json = json.dumps({
        "ranking": ["A5V+L12I", "G8A+K19E", "D3N+S20T", "V6I+R22K"],
        "signs": {"A5V+L12I": "rescuing", "G8A+K19E": "synergistic"},
    })
    ranked, signs, ok = parse_epistasis_response(f"```json\n{perfect_json}\n```", [p.pair_id for p in panel])
    assert ok and ranked == ["A5V+L12I", "G8A+K19E", "D3N+S20T", "V6I+R22K"]
    g = grade_epistasis_panel(panel, ranked, signs)
    assert g["coupling_auroc"] == pytest.approx(1.0)
    assert g["spearman_rho_abs_dddg"] == pytest.approx(1.0)
    assert g["sign_accuracy"] == pytest.approx(1.0)


def test_task10_12_site_parser_and_grader_roundtrip() -> None:
    panel = [
        SiteCandidate("R10", "R", 10, 2.8, True),
        SiteCandidate("R45", "R", 45, 16.0, False),
        SiteCandidate("H18", "H", 18, 2.4, True),
        SiteCandidate("H62", "H", 62, 18.2, False),
    ]
    resp = '{"ranking": ["H18", "R10", "R45", "H62"]}'
    ranked, ok = parse_site_ranking_response(resp, [c.res_id for c in panel])
    assert ok and ranked == ["H18", "R10", "R45", "H62"]
    g = grade_site_ranking_panel(panel, ranked)
    assert g["auroc"] == pytest.approx(1.0)
    assert g["same_aa_win_rate"] == pytest.approx(1.0)


def test_task11_dynamics_grader_roundtrip() -> None:
    panel = [
        DynamicsResidue("G20", "G", 20, 5.2, 0.1, True),
        DynamicsResidue("A30", "A", 30, 3.8, 0.0, True),
        DynamicsResidue("G50", "G", 50, 0.4, 0.1, False),
        DynamicsResidue("A60", "A", 60, 0.3, 0.0, False),
    ]
    g_perf = grade_dynamics_panel(panel, ["G20", "A30", "G50", "A60"])
    g_rev = grade_dynamics_panel(panel, ["A60", "G50", "A30", "G20"])
    assert g_perf["auroc"] == pytest.approx(1.0)
    assert g_perf["spearman_rho"] == pytest.approx(1.0)
    assert g_rev["auroc"] == pytest.approx(0.0)
    assert g_rev["spearman_rho"] == pytest.approx(-1.0)


def test_task13_developability_parser_and_grader_roundtrip() -> None:
    panel = [
        DevelopabilityCandidate("C01", "AAA / BBB", 80.0, 80.0),
        DevelopabilityCandidate("C02", "CCC / DDD", 72.0, 72.0),
        DevelopabilityCandidate("C03", "EEE / FFF", 65.0, 65.0),
        DevelopabilityCandidate("C04", "GGG / HHH", 55.0, 55.0),
    ]
    resp = 'Here is the ranking:\n```json\n{"ranking": ["C01", "C02", "C03", "C04"]}\n```'
    ranked, ok = parse_developability_response(resp, [c.cand_id for c in panel])
    assert ok and ranked == ["C01", "C02", "C03", "C04"]
    g = grade_developability_panel(panel, ranked)
    assert g["spearman_rho"] == pytest.approx(1.0)
    assert g["pairwise_accuracy"] == pytest.approx(1.0)
