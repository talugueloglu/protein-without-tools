"""Unit & integration tests for Stage 2A (Remote Homology / TEA) and Stage 2B (CAFA GO Function)."""

from __future__ import annotations

import json

from protein_without_tools.stage2a_homology import (
    HomologyPair,
    build_remote_homology_prompt,
    evaluate_homology_predictions,
    parse_remote_homology_response,
    parse_substitution_matrix_file,
    smith_waterman_affine,
)
from protein_without_tools.stage2b_cafa import (
    CAFATarget,
    build_cafa_prompt,
    compute_cafa_fmax,
    compute_term_information_accretion,
    compute_term_information_content,
    parse_cafa_response,
    parse_go_obo,
)


def test_stage2a_smith_waterman_and_evaluation() -> None:
    seq1 = "MKLAVILGAGGVGKSALTIQLIQNHFVDEYDPTIEDSYRKQVVIDGETCLLDILDTAGQEEYSAMRDQYMRTGEGFLCVFAINNTKSFEDIHQYREQIKRVK"
    seq2 = "MTEYKLVVVGAGGVGKSALTIQLIQNHFVDEYDPTIEDSYRKQVVIDGETCLLDILDTAGQEEYSAMRDQYMRTGEGFLCVFAINNTKSFEDIHHYREQIKR"
    seq_unrelated = "AEAAAKEAAAKEAAAKEAAAKPGGGSDEAAAKEAAAKEAAAKEAAAKPGGGSDEAAAKEAAAKEAAAKEAAAKPGGGSD"

    bits_hom, eval_hom, sid_hom = smith_waterman_affine(seq1, seq2)
    bits_unr, eval_unr, _ = smith_waterman_affine(seq1, seq_unrelated)
    assert bits_hom > 120.0 and eval_hom < 1e-20 and sid_hom > 80.0
    assert bits_unr < 30.0 and eval_unr > 0.5

    # Test MATCHA-style custom matrix parser
    sample_mat_txt = (
        "# Lambda (precomputed): 0.3217\n"
        "   A   C\n"
        "A  5.0 -2.0\n"
        "C -2.0  6.0\n"
    )
    alph, mat, lam = parse_substitution_matrix_file(sample_mat_txt)
    assert alph == "AC" and mat.shape == (2, 2) and abs(lam - 0.3217) < 1e-6

    p_pos = HomologyPair(
        pair_id="s2a_p001",
        rung="3_deep_tea_zone",
        label_homolog=1,
        seq_a=seq1,
        seq_b=seq2,
        scop_a="c.37.1.8",
        scop_b="c.37.1.1",
        domain_a="d5p21a_",
        domain_b="d1efua2",
        seq_identity_pct=12.4,
        blosum_bitscore=22.0,
        blosum_evalue=5.0,
    )
    p_neg = HomologyPair(
        pair_id="s2a_p002",
        rung="4_hard_negative_decoy",
        label_homolog=0,
        seq_a=seq1,
        seq_b=seq_unrelated,
        scop_a="c.37.1.8",
        scop_b="a.24.1.1",
        domain_a="d5p21a_",
        domain_b="d1f4na_",
        seq_identity_pct=9.1,
        blosum_bitscore=18.0,
        blosum_evalue=20.0,
    )
    prompt = build_remote_homology_prompt(p_pos)
    assert "d5p21a_" not in prompt and "c.37.1" not in prompt

    parsed = parse_remote_homology_response('```json\n{"same_superfamily": true, "probability": 0.88}\n```')
    assert parsed.parse_ok and parsed.predicted_same_superfamily and abs(parsed.prob_homolog - 0.88) < 1e-6

    metrics = evaluate_homology_predictions([p_pos, p_neg], [0.88, 0.12])
    assert metrics["overall_auroc"] == 1.0
    assert metrics["by_rung"]["3_deep_tea_zone"]["recall_at_fpr10"] == 1.0


def test_stage2b_go_dag_propagation_and_fmax() -> None:
    mini_obo = """format-version: 1.2
[Term]
id: GO:0003674
name: molecular_function
namespace: molecular_function

[Term]
id: GO:0003824
name: catalytic activity
namespace: molecular_function
is_a: GO:0003674 ! molecular_function

[Term]
id: GO:0016740
name: transferase activity
namespace: molecular_function
is_a: GO:0003824 ! catalytic activity

[Term]
id: GO:0004672
name: protein kinase activity
alt_id: GO:0004673
namespace: molecular_function
is_a: GO:0016740 ! transferase activity
"""
    dag = parse_go_obo(mini_obo)
    assert dag.canonical("GO:0004673") == "GO:0004672"
    anc = dag.ancestors("GO:0004673", include_self=True)
    # Trivial root GO:0003674 excluded; intermediate ancestors included!
    assert anc == {"GO:0004672", "GO:0016740", "GO:0003824"}

    ic_map = compute_term_information_content(
        dag,
        [{"GO:0003824", "GO:0016740"}, {"GO:0003824"}],
    )
    assert ic_map["GO:0004672"] > ic_map["GO:0003824"]

    t1 = CAFATarget(
        target_id="s2b_t001",
        sequence="MKLAVILGAGGVGKSALTIQLIQNHFVDEYDPTIEDSYRKQVVIDGETCLLD",
        length=52,
        true_terms_mf=sorted(anc),
        true_terms_bp=[],
        true_terms_cc=[],
    )
    prompt = build_cafa_prompt(t1)
    assert "s2b_t001" not in prompt

    raw_reply = json.dumps({"predictions": [["GO:0004673", 0.91, "protein kinase"]]})
    p_dict, ok = parse_cafa_response(raw_reply)
    assert ok

    res_exact = compute_cafa_fmax(dag, [t1], {"s2b_t001": p_dict}, aspect="MF", ic_map=ic_map)
    assert res_exact["fmax"] == 1.0 and res_exact["wfmax"] == 1.0

    # Partial ancestor prediction (`GO:0016740` misses leaf `GO:0004672` -> 0 < fmax < 1)
    res_partial = compute_cafa_fmax(dag, [t1], {"s2b_t001": {"GO:0016740": 0.8}}, aspect="MF", ic_map=ic_map)
    assert 0.5 < res_partial["fmax"] < 0.95
    assert res_partial["wfmax"] < res_partial["fmax"]

    # CAFA-6 conditional Information Accretion (IA)
    ia_map = compute_term_information_accretion(
        dag,
        [{"GO:0003824", "GO:0016740"}, {"GO:0003824", "GO:0016740"}, {"GO:0003824"}],
    )
    assert ia_map["GO:0003674"] == 0.0
    assert ia_map["GO:0004672"] > ia_map["GO:0016740"] > 0.0

    # CAFA-6 PK dual `-known` masking: known T0 = {GO:0003824, GO:0016740}, newly gained T1\T0 = {GO:0004672}
    t_pk = CAFATarget(
        target_id="s2b_pk",
        sequence="MKLAVILGAGGVGKSALTIQLIQNHFVDEYDPTIEDSYRKQVVIDGETCLLD",
        length=52,
        true_terms_mf=["GO:0003824", "GO:0016740", "GO:0004672"],
        true_terms_bp=[],
        true_terms_cc=[],
        knowledge_type="PK",
        known_terms_mf=["GO:0003824", "GO:0016740"],
        eval_aspects=["MF"],
    )
    # Regurgitating ONLY known T0 terms scores 0.0 under CAFA-6 PK dual masking!
    res_pk_known_only = compute_cafa_fmax(dag, [t_pk], {"s2b_pk": {"GO:0016740": 0.95}}, aspect="MF", ic_map=ia_map)
    assert res_pk_known_only["fmax"] == 0.0 and res_pk_known_only["wfmax"] == 0.0
    # Predicting the novel leaf GO:0004672 (which propagates up to known parents) scores 1.0 without FP penalty for known parents!
    res_pk_novel = compute_cafa_fmax(dag, [t_pk], {"s2b_pk": {"GO:0004672": 0.9}}, aspect="MF", ic_map=ia_map)
    assert res_pk_novel["fmax"] == 1.0 and res_pk_novel["wfmax"] == 1.0

    # CAFA-6 LK gained-aspect-only evaluation: eval_aspects=['MF'] ignores pre-existing CC terms
    t_lk = CAFATarget(
        target_id="s2b_lk",
        sequence="MKLAVILGAGGVGKSALTIQLIQNHFVDEYDPTIEDSYRKQVVIDGETCLLD",
        length=52,
        true_terms_mf=["GO:0003824"],
        true_terms_bp=[],
        true_terms_cc=["GO:0005634"],
        knowledge_type="LK",
        eval_aspects=["MF"],
    )
    assert compute_cafa_fmax(dag, [t_lk], {"s2b_lk": {"GO:0003824": 0.9}}, aspect="MF", ic_map=ia_map)["n_targets"] == 1
    assert compute_cafa_fmax(dag, [t_lk], {"s2b_lk": {"GO:0005634": 0.9}}, aspect="CC", ic_map=ia_map)["n_targets"] == 0


def test_built_stage2_manifests_and_zero_leakage() -> None:
    from pathlib import Path

    repo = Path(__file__).resolve().parents[1]
    m2a_path = repo / "data" / "stage2a_homology" / "manifest.json"
    m2b_path = repo / "data" / "stage2b_cafa" / "manifest.json"
    if not (m2a_path.exists() and m2b_path.exists()):
        return

    m2a = json.loads(m2a_path.read_text(encoding="utf-8"))
    assert m2a["n_pairs"] == 120
    assert m2a["rung_counts"] == {
        "1_seq_easy": 30,
        "2_foldseek_3di": 30,
        "3_deep_tea_zone": 30,
        "4_hard_negative_decoy": 30,
    }
    alph, mat, lam = parse_substitution_matrix_file((repo / "data" / "stage2a_homology" / "matcha.out").read_text())
    assert alph == "ACDEFGHIKLMNPQRSTVWYX" and mat.shape == (21, 21) and lam > 0.2

    p2a_lines = (repo / "outputs" / "prompts" / "stage2a_homology_prompts.jsonl").read_text().splitlines()
    assert len(p2a_lines) == 120
    for row_meta, pline in zip(m2a["pairs"], p2a_lines):
        pr_txt = json.loads(pline)["prompt"]
        assert row_meta["domain_a"] not in pr_txt
        assert row_meta["domain_b"] not in pr_txt
        assert row_meta["scop_a"] not in pr_txt

    m2b = json.loads(m2b_path.read_text(encoding="utf-8"))
    assert m2b["n_targets"] == 120
    assert m2b["temporal_tier_counts"] == {
        "pre_nk": 20,
        "pre_lk": 20,
        "pre_pk": 20,
        "post_nk": 20,
        "post_lk": 20,
        "post_pk": 20,
    }
    sub_dag = parse_go_obo(str(repo / "data" / "stage2b_cafa" / "go_subdag.obo"))
    assert len(sub_dag.term_aspect) > 500

    p2b_lines = (repo / "outputs" / "prompts" / "stage2b_cafa_prompts.jsonl").read_text().splitlines()
    assert len(p2b_lines) == 120
    for row_meta, pline in zip(m2b["targets"], p2b_lines):
        pr_txt = json.loads(pline)["prompt"]
        assert row_meta["uniprot_accession"] not in pr_txt
