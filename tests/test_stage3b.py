"""Unit and integration tests for Stage 3B (De Novo Design & Closed-Loop Epitope Binder Design)."""

from __future__ import annotations

import json
import numpy as np

from fable_bench.stage3b_design import (
    DesignTaskSpec,
    EpitopeTarget,
    MotifSegmentSpec,
    SurrogatePhysicsBinderOracle,
    build_binder_initial_prompt,
    build_binder_refinement_prompt,
    build_stage3b_monomer_prompt,
    evaluate_binder_campaign,
    evaluate_sequence_hygiene,
    grade_monomer_design,
    kabsch_rmsd,
    parse_designed_sequence,
    score_3d_complex_geometry,
)


def test_stage3b_hygiene_blocks_repeat_helix_and_validates_motifs() -> None:
    # Classic `@sokrypton` EAAAK repeat-helix cheat must be rejected by Gate 1!
    repeat_cheat = "AEAAAKEAAAKEAAAKEAAAKPGGGSDEAAAKEAAAKEAAAKEAAAKPGGGSDEAAAKEAAAKEAAAKEAAAK"
    rep_hyg = evaluate_sequence_hygiene(repeat_cheat, min_len=65, max_len=90)
    assert not rep_hyg.hygiene_pass
    assert any("entropy" in r or "repetitive" in r or "single_aa" in r for r in rep_hyg.failure_reasons)

    # Diverse globular sequence with embedded 12-AA EF-hand motif `DKDGDGTITTKE`
    ef_motif = MotifSegmentSpec(
        segment_id="ef_hand",
        sequence="DKDGDGTITTKE",
        min_linker_before=15,
        max_linker_before=45,
        native_ca_coords=[[float(i) * 3.8, math.sin(i), math.cos(i)] for i in range(12)] if False else [
            [i * 3.2, float(i % 3), float((i * 2) % 5)] for i in range(12)
        ],
    )
    good_scaffold = (
        "MDEELKRKLEALVKKYNPDAVLVIK"  # 25 AA N-flank
        + "DKDGDGTITTKE"             # 12 AA exact EF-hand motif
        + "ELVKKFEEKLAQMGDPNTVRKLIEALKKQG"  # 30 AA C-flank (total 67 AA)
    )
    ok_hyg = evaluate_sequence_hygiene(good_scaffold, min_len=60, max_len=85, motif_segments=[ef_motif])
    assert ok_hyg.hygiene_pass
    assert ok_hyg.motif_present
    assert len(ok_hyg.motif_0idx_positions) == 12

    # Verify Kabsch SO(3) invariance under exact rigid rotation + translation
    ca_full = np.zeros((len(good_scaffold), 3), dtype=np.float64)
    for k in range(len(good_scaffold)):
        theta = 0.55 * k
        ca_full[k] = [4.5 * np.cos(theta), 4.5 * np.sin(theta), 0.18 * k]
    native_m = np.array(ef_motif.native_ca_coords, dtype=np.float64)
    # Apply 90-degree rotation around z + translation to place motif into folded structure
    rot90 = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    ca_full[ok_hyg.motif_0idx_positions] = (native_m @ rot90.T) + np.array([12.0, -7.5, 3.0])

    spec = DesignTaskSpec(
        task_id="s3b_m001",
        category="motif_scaffolding",
        subfamily="EF_hand_Ca_loop",
        min_length=60,
        max_length=85,
        description="Scaffold 12-residue Ca2+-binding EF-hand loop",
        motif_segments=[ef_motif],
        forbidden_parent_sequences=["ADQLTEEQIAEFKEAFSLFDKDGDGTITTKELGTVMRSLGQNPTEAELQDMINEVDADGNGTIDFPEFLTMMARKMKDTDSEEEIREAFRVFDKDGNGYISAAELRHVMTNLGEKLTDEEVDEMIREADIDGDGQVNYEEFVQMMTAK"],
    )
    pr = build_stage3b_monomer_prompt(spec)
    assert "DKDGDGTITTKE" in pr and "1CLL" not in pr

    grade = grade_monomer_design(
        spec=spec,
        designed_seq=good_scaffold,
        folded_ca_coords=ca_full,
        plddt_mean=86.5,
        mpnn_sc_rmsd=1.25,
        external_max_seqid_pct=19.0,
    )
    assert grade["motif_ca_rmsd"] == 0.0
    assert grade["all_5_gates_pass"] is True

    # Memorizing native Calmodulin parent must fail Gate 5 novelty!
    calmodulin_crop = "ADQLTEEQIAEFKEAFSLFDKDGDGTITTKELGTVMRSLGQNPTEAELQDMINEVDADGNGTIDFPEFLTMM"
    grade_mem = grade_monomer_design(
        spec=spec,
        designed_seq=calmodulin_crop,
        folded_ca_coords=np.zeros((len(calmodulin_crop), 3)),
        plddt_mean=92.0,
        mpnn_sc_rmsd=0.8,
    )
    assert grade_mem["gate5_novelty_lt30"] is False
    assert grade_mem["all_5_gates_pass"] is False


def test_stage3b_closed_loop_binder_oracle_and_anti_cheat_gates() -> None:
    L_t = 90
    t_seq = ("MKLAVILGAGGVGKSALTIQLIQNHFVDEYDPTIEDSYRKQVVIDGET"
             "CLLDILDTAGQEEYSAMRDQYMRTGEGFLCVFAINNTKSFED")[:L_t]
    t_ca = np.array([[3.8 * np.cos(0.4 * i), 3.8 * np.sin(0.4 * i), 0.25 * i] for i in range(L_t)])
    t_cb = t_ca + np.array([1.2, 0.0, 0.0])

    heldout_partner = "MDEEVKRKLEALVKKYNPDAVLVIKELVKKFEEKLAQMGDPNTVRKLIEALKKQGWYPEELK"
    b_len = len(heldout_partner)
    # Place binder cleanly outside the 3.8A target cylinder (`x ~ 8.8A`) within 8A Cbeta of hotspots!
    hs_1idx = [1, 16, 17, 31, 32]
    b_ca = np.array(
        [[9.2, 1.5 * np.cos(0.5 * j), 3.0 + 0.12 * j] for j in range(b_len)]
    )
    b_cb = b_ca - np.array([1.4, 0.0, 0.0])

    target = EpitopeTarget(
        target_id="s3b_ep01",
        tier="post_may_2026_complex",
        name="Novel_Target_01",
        pdb_id="9ZZZ",
        release_date="2026-08-19",
        target_chain="A",
        target_sequence=t_seq,
        epitope_hotspots_1idx=hs_1idx,
        binder_min_len=50,
        binder_max_len=85,
        target_ca_coords=t_ca.tolist(),
        target_cb_coords=t_cb.tolist(),
        heldout_partner_sequence=heldout_partner,
        heldout_partner_ca_coords=b_ca.tolist(),
        heldout_partner_cb_coords=b_cb.tolist(),
    )

    p1 = build_binder_initial_prompt(target)
    assert "9ZZZ" not in p1 and heldout_partner[:12] not in p1

    # 1. Held-out partner copy-paste cheat -> must be disqualified (`novelty_ok == False`)!
    fb_cheat = score_3d_complex_geometry(
        target=target,
        binder_seq=heldout_partner,
        pred_target_cb=t_cb,
        pred_binder_ca=b_ca,
        pred_binder_cb=b_cb,
        iptm=0.89,
        ipae=5.2,
        binder_plddt=91.0,
        monomer_rmsd=0.9,
        round_idx=1,
    )
    assert fb_cheat.novelty_ok is False and fb_cheat.is_hit is False and fb_cheat.composite_score == 0.0

    # 2. Novel binder on backside (far from hotspots) -> must fail hotspot gate even with ipTM=0.85!
    novel_binder = "SWQERIDKLMNAVYKFHGPDSLEQRIKAMVNWYKEHGPDSTLRQFIEDMVKAWYNHGPDSL"
    far_ca = b_ca + np.array([40.0, 40.0, 40.0])
    fb_backside = score_3d_complex_geometry(
        target=target,
        binder_seq=novel_binder,
        pred_target_cb=t_cb,
        pred_binder_ca=far_ca,
        pred_binder_cb=far_ca,
        iptm=0.85,
        ipae=6.5,
        binder_plddt=88.0,
        monomer_rmsd=1.2,
        round_idx=1,
    )
    assert len(fb_backside.hotspots_contacted_1idx) == 0 and fb_backside.is_hit is False

    # 3. Novel binder on target epitope passing all BindCraft gates -> HIT on Shot 2!
    fb_hit = score_3d_complex_geometry(
        target=target,
        binder_seq=novel_binder,
        pred_target_cb=t_cb,
        pred_binder_ca=b_ca,
        pred_binder_cb=b_cb,
        iptm=0.82,
        ipae=7.4,
        binder_plddt=87.5,
        monomer_rmsd=1.35,
        round_idx=2,
    )
    assert fb_hit.is_hit is True and len(fb_hit.hotspots_contacted_1idx) >= 3

    p2 = build_binder_refinement_prompt(target, [fb_backside], total_budget=16)
    assert "Shot 2 of 16" in p2 and "ipTM=0.85" in p2

    camp = evaluate_binder_campaign(target.target_id, [fb_backside, fb_hit], budget_k=16)
    assert camp["hit_achieved"] == 1
    assert camp["shots_to_first_hit"] == 2
    assert camp["pass_at_k_curve"][0] == 0 and camp["pass_at_k_curve"][1] == 1

    # 4. Parser + Surrogate CPU oracle sanity check
    seq_p, rat_p, ok_p = parse_designed_sequence(json.dumps({"binder_sequence": novel_binder, "rationale": "pack hotspots"}))
    assert ok_p and seq_p == novel_binder and "pack" in rat_p
    sur = SurrogatePhysicsBinderOracle()
    fb_sur = sur.evaluate_binder(target, novel_binder, round_idx=1)
    assert fb_sur.hygiene_ok and fb_sur.novelty_ok
    assert kabsch_rmsd(b_ca, b_ca + 5.0) < 1e-8


def test_built_stage3b_manifests_and_zero_leakage() -> None:
    from pathlib import Path

    repo = Path(__file__).resolve().parents[1]
    m1_p = repo / "data" / "stage3b_design" / "monomer_motif_manifest.json"
    m2_p = repo / "data" / "stage3b_design" / "epitope_binder_manifest.json"
    if not (m1_p.exists() and m2_p.exists()):
        return

    m1 = json.loads(m1_p.read_text(encoding="utf-8"))
    assert m1["n_tasks"] == 40
    p1_lines = (repo / "outputs" / "prompts" / "stage3b_denovo_prompts.jsonl").read_text().splitlines()
    assert len(p1_lines) == 40
    for pl in p1_lines:
        txt = json.loads(pl)["prompt"]
        for pdb_leak in ("1CLL", "1AAY", "5P21", "1SHG", "6MRR", "6CZH", "9S83"):
            assert pdb_leak not in txt

    m2 = json.loads(m2_p.read_text(encoding="utf-8"))
    assert m2["n_targets"] == 12
    p2_lines = (repo / "outputs" / "prompts" / "stage3b_binder_round1_prompts.jsonl").read_text().splitlines()
    assert len(p2_lines) == 12
    for ep_meta, pl in zip(m2["targets"], p2_lines):
        txt = json.loads(pl)["prompt"]
        assert ep_meta["pdb_id"] not in txt
        if ep_meta["heldout_partner_sequence"]:
            assert ep_meta["heldout_partner_sequence"][:14] not in txt
        assert (repo / "data" / "stage3b_design" / "target_pdbs" / f"{ep_meta['target_id']}.pdb").exists()
