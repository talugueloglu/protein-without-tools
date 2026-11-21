#!/usr/bin/env python3
"""Build a richly visual, illustrated HTML Atlas of Fable-Protein-Bench (`fable_protein_bench_summary.html`).

Visual-first design featuring:
  1. Visual SVG Architecture Diagrams of the Core Scientific Question (`No-Code Weights` vs
     `With-Code Air-Gapped NumPy` vs `3D/MSA Specialist Bio-Tools`) and each Stage's Test Pipeline.
  2. One Concrete, Real-Data Visual Example Card for EVERY Category across all 6 Stages:
     - Stage 1A (5 Strata): Live 3D rotating Cα trace + real Cβ < 8Å 2D contact-map heatmap + Q3 ribbon
       for `post_cutoff_struct_novel` (`9YOY` TM=0.28), `post_cutoff_de_novo` (`47WY`),
       `post_cutoff_pdb_novel`, `post_cutoff_homolog_gradient` (`37NT`), and `pre_cutoff_control` (`6MRR`).
     - Stage 1B (Inverse Folding): Live 3D stripped `UNK` backbone (`11CY` TM=0.35) colored by exact
       virtual-Cβ core burial (red buried core vs sky-blue solvent surface) showing what `NumPy` unlocks.
     - Stage 2A (4 Homology Rungs): Visual alignment & fold-comparison strips for `1_seq_easy`,
       `2_foldseek_3di`, `3_deep_tea_zone` (`PickyBinders/tea`), and `4_hard_negative_decoy`.
     - Stage 2B (2 CAFA Temporal Tiers + Visual GO DAG): `post_may_2026` (`s2b_t001`) and
       `early_2026_uniprot` (`s2b_t012`) with real high-IC GO term hierarchy & `wF_max` vs `F_max` visual.
     - Stage 3A (4 Alanine-Scan Categories): Live 3D ΔΔG-colored structures + color heatstrips for
       `de_novo_ddg` (`HEEH_KT_rd6_0746`), `natural_ddg` (`1PSE`), `post_cutoff_mavedb` (`GBA1` Nov 11, 2026),
       and dual-track stability-vs-signaling `dual_stability_function` (`OPRM1` Nov 9, 2026).
     - Stage 3B (2 Part-1 Categories + 3 Part-2 Epitope Tiers): Live 3D fold/motif scaffolds (`6MRR`,
       `1SHG` all-β barrel, `1CLL` gold `EF-hand` motif) + 3 live 3D epitope-hotspot complexes across
       `rfd4_proteina_classic` (`3MJG` PDGFRβ), `boltzprot1_hard_novel` (`3QKG` AMBP), and
       `post_may2026_denovo_crystal` (`9S83` LEDGF + de novo minibinder `JUBO4`).
"""

from __future__ import annotations

import json
from pathlib import Path
import sys
from typing import Any, Dict, List

import numpy as np

REPO = Path("/Users/taluguloglu/.gemini/antigravity/scratch/fable-protein-bench")
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from fable_bench.pdb_utils import load_structure
from fable_bench.stage2b_cafa import compute_term_information_content, parse_go_obo


def load_ca_and_contacts(npz_path: Path, max_res: int = 160) -> Dict[str, Any]:
    z = np.load(npz_path)
    seq_full = str(z["sequence"].item())
    q3_full = str(z["q3"].item())
    full_len = len(seq_full)
    L = min(full_len, max_res)
    ca = np.round(z["coords_ca"][:L], 1).tolist()
    dmat = z["cb_dist_matrix"][:L, :L]
    contacts = []
    for i in range(L):
        for j in range(i + 6, L):
            if dmat[i, j] < 8.0:
                contacts.append([i, j])
    nbrs = (np.sum(dmat < 10.0, axis=1) - 1).tolist()
    return {
        "len": full_len,
        "shown_len": L,
        "seq": seq_full[:L],
        "q3": q3_full[:L],
        "ca": ca,
        "contacts": contacts,
        "nbrs": nbrs,
    }


def compile_visual_atlas_data() -> Dict[str, Any]:
    # ==========================================================================
    # 1. Stage 1A (5 Strata — 1 visual card per stratum) + Stage 1B (`11CY`)
    # ==========================================================================
    m1 = json.loads((REPO / "data/stage1_benchmark/manifest.json").read_text(encoding="utf-8"))
    gt_dir = REPO / "data/stage1_benchmark/ground_truth"

    def _pick_s1(stratum: str, pref_pdb: str | None = None) -> Dict[str, Any]:
        cands = [t for t in m1["targets"] if t["stratum"] == stratum]
        chosen = next((t for t in cands if pref_pdb and t["pdb_id"] == pref_pdb), cands[0])
        pfile = gt_dir / f"{chosen['target_id']}.npz"
        geom = load_ca_and_contacts(pfile)
        return {
            "tid": chosen["target_id"],
            "pdb": chosen["pdb_id"],
            "stratum": stratum,
            "date": chosen.get("release_date", ""),
            "desc": str(chosen.get("description") or chosen.get("note") or chosen["pdb_id"])[:72],
            "pdb_seqid": round(float(chosen.get("pdb_precutoff_seqid_combined", 0.0)), 1),
            "pdb_tm": round(float(chosen.get("pdb_precutoff_tm_query", 0.5)), 2),
            "afdb_seqid": round(float(chosen.get("afdb_max_seqid", 0.0)), 1),
            "afdb_tm": round(float(chosen.get("afdb_tm_query", 0.5)), 2),
            "n_in_stratum": len(cands),
            **geom,
        }

    s1_examples = [
        _pick_s1("post_cutoff_struct_novel", "9YOY"),
        _pick_s1("post_cutoff_de_novo", "47WY"),
        _pick_s1("post_cutoff_pdb_novel", None),
        _pick_s1("post_cutoff_homolog_gradient", "37NT"),
        _pick_s1("pre_cutoff_control", "6MRR"),
    ]
    s1b_example = _pick_s1("post_cutoff_struct_novel", "11CY")

    # ==========================================================================
    # 2. Stage 2A (4 Rungs — 1 real SCOPe pair per rung)
    # ==========================================================================
    m2a = json.loads((REPO / "data/stage2a_homology/manifest.json").read_text(encoding="utf-8"))
    rep2 = json.loads((REPO / "outputs/results/stage2_evaluation_report.json").read_text(encoding="utf-8"))
    s2a_examples = []
    rung_titles = {
        "1_seq_easy": ("Rung 1: Sequence-Easy Homolog", "Same Superfamily • Caught by BLOSUM62 & Code"),
        "2_foldseek_3di": ("Rung 2: Structure-Caught Homolog", "Same Superfamily, Diff Family • BLAST Fails, Foldseek Hits"),
        "3_deep_tea_zone": ("Rung 3: Deep TEA Twilight Zone", "Same Superfamily • Both BLAST & pdb100 Fail (TEA Target!)"),
        "4_hard_negative_decoy": ("Rung 4: Class & Length-Matched Decoy", "Different 3D Fold, Same Class & Length • Blocks Cheats"),
    }
    for rk in ("1_seq_easy", "2_foldseek_3di", "3_deep_tea_zone", "4_hard_negative_decoy"):
        p = next(x for x in m2a["pairs"] if x["rung"] == rk)
        title, sub = rung_titles[rk]
        sa, sb = p["seq_a"][:44], p["seq_b"][:44]
        matches = "".join("|" if a == b else ("·" if a in "LIVFMWYA" and b in "LIVFMWYA" else " ") for a, b in zip(sa, sb))
        s2a_examples.append({
            "pair_id": p["pair_id"],
            "rung": rk,
            "title": title,
            "subtitle": sub,
            "same_sf": bool(p["label_homolog"] == 1),
            "scop_a": p["scop_a"],
            "scop_b": p["scop_b"],
            "len_a": len(p["seq_a"]),
            "len_b": len(p["seq_b"]),
            "seqid": p["seq_identity_pct"],
            "bits": p["blosum_bitscore"],
            "fs_ev": p["foldseek_pdb100_evalue"],
            "snip_a": sa,
            "snip_b": sb,
            "snip_mid": matches,
        })

    # ==========================================================================
    # 3. Stage 2B (2 Temporal Tiers — real 2026 proteins + specific high-IC GO terms)
    # ==========================================================================
    dag = parse_go_obo(str(REPO / "data/stage2b_cafa/go_subdag.obo"))
    m2b = json.loads((REPO / "data/stage2b_cafa/manifest.json").read_text(encoding="utf-8"))
    ref2b = json.loads((REPO / "data/stage2b_cafa/reference_swissprot_go.json").read_text(encoding="utf-8"))
    ic_map = compute_term_information_content(dag, [dag.propagate_set(r["go_terms"]) for r in ref2b])

    s2b_examples = []
    for tier, lbl in (
        ("post_may_2026", "Tier A: Post-May-2026 QuickGO Experimental Holdout (n = 11)"),
        ("early_2026_uniprot", "Tier B: Early-2026 UniProt Experimental Holdout (n = 49)"),
    ):
        t = next(x for x in m2b["targets"] if x["temporal_tier"] == tier and x["true_terms_mf"] and x["true_terms_cc"])
        top_terms = []
        for asp, arr in (("MF", t["true_terms_mf"]), ("BP", t["true_terms_bp"]), ("CC", t["true_terms_cc"])):
            ranked = sorted(arr, key=lambda g: ic_map.get(g, 5.0), reverse=True)[:2]
            for g in ranked:
                top_terms.append({
                    "id": g,
                    "aspect": asp,
                    "name": dag.term_name.get(g, g),
                    "ic": round(float(ic_map.get(g, 5.0)), 1),
                })
        s2b_examples.append({
            "tid": t["target_id"],
            "tier_label": lbl,
            "organism": t["organism"],
            "len": t["length"],
            "uniprot_date": t["uniprot_first_public_date"],
            "qgo_date": t["quickgo_max_exp_date"],
            "seq_snip": t["sequence"][:56] + "...",
            "n_mf": len(t["true_terms_mf"]),
            "n_bp": len(t["true_terms_bp"]),
            "n_cc": len(t["true_terms_cc"]),
            "top_terms": top_terms,
        })

    # ==========================================================================
    # 4. Test 5: Monomer Alanine Scan Ranking (4 Categories — 1 per category)
    # ==========================================================================
    m3a = json.loads((REPO / "data/alanine_scan_benchmark/manifest.json").read_text(encoding="utf-8"))
    s3a_picks = [
        ("de_novo_ddg", "s3a_t001", "Category 1: Zero-Homolog De Novo Miniproteins (n = 24)", "Synthetic Rocklin HEEH design • Rank 18 X→Ala candidates"),
        ("natural_ddg", "s3a_t026", "Category 2: Natural Protein Domains (n = 14)", "Tsuboyama natural domain (1PSE) • Rank 18 X→Ala candidates"),
        ("post_cutoff_mavedb", "s3a_t042", "Category 3: May–Nov 2026 MaveDB DMS Scans (n = 12)", "Human GBA1 (published Nov 11, 2026) • 18-site panel from 290 X→Ala"),
        ("dual_stability_function", "s3a_t048", "Category 4: Dual Stability vs. Signaling DMS (n = 2)", "Human μ-Opioid Receptor OPRM1 (Nov 9, 2026) • 18-site ranking panel"),
    ]
    s3a_examples = []
    for cat, tid, ctitle, csub in s3a_picks:
        meta = next(x for x in m3a["targets"] if x["target_id"] == tid)
        tobj = json.loads((REPO / "data/alanine_scan_benchmark/targets" / f"{tid}.json").read_text(encoding="utf-8"))
        pdb_p = REPO / "data/alanine_scan_benchmark/structures" / f"{tid}.pdb"
        ca_coords = []
        if pdb_p.exists():
            st_a = load_structure(pdb_p, pdb_id=tid)
            ca_coords = np.round(st_a.coords_ca, 1).tolist()
        f_pos = tobj.get("full_positions_1idx") or tobj["positions_1idx"]
        f_wt = tobj.get("full_wt_aas") or tobj["wt_aas"]
        f_dest = tobj.get("full_exp_destab_scores") or tobj["exp_destab_scores"]
        f_func = tobj.get("full_exp_func_loss_scores") or tobj.get("exp_func_loss_scores")
        lim = min(72, len(f_pos))
        # Build true #1..#K ground-truth ranking order on the 18-site panel
        pan_order = np.argsort(-np.asarray(tobj["exp_destab_scores"], dtype=np.float64)).tolist()
        roles = tobj.get("panel_roles") or ["moderate"] * len(pan_order)
        ranked_panel = [
            {
                "rank": r_i + 1,
                "mut": f"{tobj['positions_1idx'][k]}{tobj['wt_aas'][k]}→A",
                "val": round(float(tobj["exp_destab_scores"][k]), 2),
                "role": roles[k],
            }
            for r_i, k in enumerate(pan_order)
        ]
        s3a_examples.append({
            "tid": tid,
            "cat": cat,
            "title": ctitle,
            "subtitle": csub,
            "name": meta["source_name"],
            "date": meta["published_date"],
            "len": tobj["length"],
            "n_panel": len(tobj["positions_1idx"]),
            "n_full": len(f_pos),
            "pos": f_pos[:lim],
            "wt": f_wt[:lim],
            "destab": [round(float(v), 2) for v in f_dest[:lim]],
            "func": ([round(float(v), 2) for v in f_func[:lim]] if f_func else []),
            "ranked_panel": ranked_panel,
            "ca": ca_coords,
            "full_pos": f_pos,
            "full_destab": [round(float(v), 2) for v in f_dest],
        })

    # ==========================================================================
    # 4B. Test 6: Interface Binding Alanine Scan Ranking (4 Categories — 1 per category)
    # ==========================================================================
    m6 = json.loads((REPO / "data/binding_alanine_benchmark/manifest.json").read_text(encoding="utf-8"))
    s6_picks = [
        ("skempi_biophysical_ddg", "s6_b003", "Category 1: SKEMPI v2.0 Purified SPR/ITC ΔΔG_bind (n = 28)", "Human Growth Hormone × hGH Receptor (1A22) • Non-interior surface/interface"),
        ("oncology_alphaseq_kd", "s6_b030", "Category 2: Oncology Checkpoint AlphaSeq Kd (n = 3)", "Anti-TIGIT PP489 × Human TIGIT (Zero PDB!) • CDR/surface-filtered"),
        ("ppi_binding_dms", "s6_b032", "Category 3: Abundance-Controlled PPI Binding DMS (n = 3)", "Human MLH1 CTD × PMS2 MutLα • Filtered by paired MLH1 abundance"),
        ("post_may2026_mavedb_binding", "s6_b035", "Category 4: Oct–Nov 2026 Post-Cutoff Binding DMS (n = 4)", "Human KRAS × LZTR1 Direct Binding (Oct 16, 2026) • Filtered by KRAS abundance"),
    ]
    s6_examples = []
    for cat, tid, ctitle, csub in s6_picks:
        meta6 = next(x for x in m6["targets"] if x["target_id"] == tid)
        tobj6 = json.loads((REPO / "data/binding_alanine_benchmark/targets" / f"{tid}.json").read_text(encoding="utf-8"))
        pdb6 = REPO / "data/binding_alanine_benchmark/structures" / f"{tid}.pdb"
        ca_a6, ca_b6 = [], []
        if pdb6.exists():
            for ln in pdb6.read_text(encoding="utf-8").splitlines():
                if ln.startswith("ATOM") and ln[12:16].strip() == "CA":
                    ch_id = ln[21]
                    xyz = [round(float(ln[30:38]), 1), round(float(ln[38:46]), 1), round(float(ln[46:54]), 1)]
                    if ch_id == "A":
                        ca_a6.append(xyz)
                    elif ch_id == "B":
                        ca_b6.append(xyz)
        f6_pos = tobj6.get("full_positions_1idx") or tobj6["positions_1idx"]
        f6_wt = tobj6.get("full_wt_aas") or tobj6["wt_aas"]
        f6_loss = np.asarray(tobj6.get("full_exp_binding_loss") or tobj6["exp_binding_loss"], dtype=np.float64)
        lim6 = min(85, len(f6_pos))
        vis_loss = (f6_loss - float(np.median(f6_loss))) if tobj6["assay_type"] == "binding_dms_loss" else f6_loss
        pan6_order = np.argsort(-np.asarray(tobj6["exp_binding_loss"], dtype=np.float64)).tolist()
        roles6 = tobj6.get("panel_roles") or ["moderate"] * len(pan6_order)
        ranked_panel6 = [
            {
                "rank": r_i + 1,
                "mut": f"{tobj6['positions_1idx'][k]}{tobj6['wt_aas'][k]}→A",
                "val": round(float(tobj6["exp_binding_loss"][k]), 2),
                "role": roles6[k],
            }
            for r_i, k in enumerate(pan6_order)
        ]
        s6_examples.append({
            "tid": tid,
            "cat": cat,
            "mem_stratum": tobj6.get("memorization_stratum", "known_pdb_pre2026"),
            "title": ctitle,
            "subtitle": csub,
            "name": meta6["source_name"],
            "date": meta6["published_date"],
            "len_a": tobj6["scanned_length"],
            "len_b": tobj6["partner_length"],
            "n_panel": len(tobj6["positions_1idx"]),
            "n_full": len(f6_pos),
            "pos": f6_pos[:lim6],
            "wt": f6_wt[:lim6],
            "loss": [round(float(v), 2) for v in vis_loss[:lim6]],
            "raw_loss": [round(float(v), 2) for v in f6_loss[:lim6]],
            "ranked_panel": ranked_panel6,
            "ca_a": ca_a6,
            "ca_b": ca_b6,
            "full_pos": f6_pos,
            "full_loss": [round(float(v), 2) for v in vis_loss],
        })

    # ==========================================================================
    # 5. Stage 3B Part 1 (2 Categories: Fold-Conditioned & Motif Scaffolding)
    #    + Part 2 (3 Proven-Bindable Epitope Tiers)
    # ==========================================================================
    ref_3b = REPO / "data/stage3b_design/reference_pdbs"
    st_sh3 = load_structure(ref_3b / "all_beta_1SHG.pdb", pdb_id="1SHG")
    # Also extract 1CLL with exact EF-hand indices `[16..28]` for the motif-scaffolding visual!
    from scripts.download_stage3b_raw import parse_cif_all_protein_chains
    c1cll = parse_cif_all_protein_chains((REPO / "data/raw_stage3b_cache/cifs/1CLL.cif").read_text(encoding="utf-8"))["A"]
    ef_start = c1cll["sequence"].index("DKDGDGTITTKE")

    s3b_part1_examples = [
        {
            "cat": "fold_conditioned",
            "title": "Category 1: Fold-Conditioned Monomer Generation (n = 24)",
            "subtitle": "Example: SH3-like 5-Stranded All-β Barrel (72–82 AA, 0% helix cheat allowed)",
            "ca": np.round(st_sh3.coords_ca, 1).tolist(),
            "highlight_0idx": [],
            "badge": "4 Topologies × 6 Specs (4-Helix, Ferredoxin, Rossmann, All-β)",
        },
        {
            "cat": "motif_scaffolding",
            "title": "Category 2: Functional Crystal Motif Scaffolding (n = 16)",
            "subtitle": "Example: 12-AA EF-Hand Ca²⁺ Loop (DKDGDGTITTKE, gold) held within ≤ 1.50 Å RMSD",
            "ca": np.round(c1cll["ca"][:82], 1).tolist(),
            "highlight_0idx": list(range(ef_start, ef_start + 12)),
            "badge": "4 Motifs × 4 Specs (EF-Hand, C2H2 Zn, Walker-A P-Loop, Bi-Segment)",
        },
    ]

    m3b_ep = json.loads((REPO / "data/stage3b_design/epitope_binder_manifest.json").read_text(encoding="utf-8"))
    tier_picks = [
        ("rfd4_proteina_classic", "3MJG", "Category 1: RFD4-Proteina / ProtDBench Classics (n = 4)", "PDGFRβ (3MJG) • Proteina-Complexa 63.5% wet-lab & BindCraft 54.2% AF2 target"),
        ("boltzprot1_hard_novel", "3QKG", "Category 2: BoltzProt-1 June-2026 Hard Novel Panel (n = 5)", "Human AMBP (3QKG) • Zero prior PDB binders, proven nM bindable in June 2026"),
        ("post_may2026_denovo_crystal", "9S83", "Category 3: July–Aug 2026 Wet-Lab De Novo Crystals (n = 3)", "LEDGF PWWP + De Novo Minibinder JUBO4 (9S83, released Aug 5, 2026)"),
    ]
    s3b_part2_examples = []
    for tier_k, pdb_k, ttitle, tsub in tier_picks:
        ep = next(e for e in m3b_ep["targets"] if e["tier"] == tier_k and e["pdb_id"] == pdb_k)
        s3b_part2_examples.append({
            "tid": ep["target_id"],
            "pdb": ep["pdb_id"],
            "date": ep["release_date"],
            "title": ttitle,
            "subtitle": tsub,
            "len_t": len(ep["target_sequence"]),
            "len_p": len(ep["heldout_partner_sequence"]),
            "hotspots": ep["epitope_hotspots_1idx"],
            "hs_labels": [f"{ep['target_sequence'][h - 1]}{h}" for h in ep["epitope_hotspots_1idx"]],
            "t_ca": [[round(float(c), 1) for c in row] for row in ep["target_ca_coords"]],
            "p_ca": ([[round(float(c), 1) for c in row] for row in ep["heldout_partner_ca_coords"]] if ep["heldout_partner_ca_coords"] else []),
        })

    return {
        "s1_examples": s1_examples,
        "s1b_example": s1b_example,
        "s2_report": rep2,
        "s2a_examples": s2a_examples,
        "s2b_examples": s2b_examples,
        "s3a_examples": s3a_examples,
        "s6_examples": s6_examples,
        "s3b_part1_examples": s3b_part1_examples,
        "s3b_part2_examples": s3b_part2_examples,
    }


VISUAL_HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>Fable-Protein-Bench — Visual Atlas of Ideas, Tests &amp; Category Examples</title>
  <script src="https://www.gstatic.com/antigravity/web/dev/tailwindcss.min.js"></script>
  <style>
    canvas { touch-action: none; }
    .pill-nav:hover { opacity: 0.9; }
  </style>
</head>
<body class="bg-[var(--background)] text-[var(--foreground)] antialiased p-5 md:p-8">
  <div class="max-w-7xl mx-auto space-y-10">

    <!-- =================================================================== -->
    <!-- 0. VISUAL HERO & CORE HYPOTHESIS DIAGRAM (NO-CODE vs WITH-CODE)     -->
    <!-- =================================================================== -->
    <header class="bg-[var(--card)] border border-[var(--border)] rounded-2xl p-6 md:p-8 shadow-sm space-y-6">
      <div class="flex flex-col lg:flex-row lg:items-end justify-between gap-4">
        <div class="space-y-2">
          <div class="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-emerald-500/15 text-emerald-500 text-xs font-bold">
            <span>VISUAL BENCHMARK ATLAS</span>
            <span>•</span>
            <span>8 TESTS (1. TO 8.) • 25 CATEGORY EXAMPLES • ZERO LEAKAGE</span>
          </div>
          <h1 class="text-2xl md:text-3xl font-extrabold tracking-tight">
            Does Raw Protein Pretraining Teach Folding &amp; Binding Physics—Or Just Sequence Patterns?
          </h1>
          <p class="text-sm text-[var(--muted-foreground)] max-w-4xl">
            Auditing <strong>Claude Fable 5.2</strong> (<code class="px-1.5 py-0.5 rounded bg-[var(--secondary)] text-xs">claude-fable-5-2</code>) against general reasoning LLMs (<code class="px-1.5 py-0.5 rounded bg-[var(--secondary)] text-xs">Opus 5.5</code>, <code class="px-1.5 py-0.5 rounded bg-[var(--secondary)] text-xs">GPT-5.2</code>, <code class="px-1.5 py-0.5 rounded bg-[var(--secondary)] text-xs">Gemini 3.1 Pro</code>) and specialist structural biology models across <strong>No-Code (Weights Only)</strong> vs. <strong>With-Code (Air-Gapped NumPy)</strong>.
          </p>
        </div>
        <div class="flex flex-wrap gap-1.5 text-xs">
          <a href="#test-1" class="pill-nav px-2.5 py-1.5 rounded-lg bg-[var(--secondary)] text-[var(--secondary-foreground)] font-semibold">1. Forward Folding</a>
          <a href="#test-2" class="pill-nav px-2.5 py-1.5 rounded-lg bg-[var(--secondary)] text-[var(--secondary-foreground)] font-semibold">2. Inverse Folding</a>
          <a href="#test-3" class="pill-nav px-2.5 py-1.5 rounded-lg bg-[var(--secondary)] text-[var(--secondary-foreground)] font-semibold">3. Remote Homology</a>
          <a href="#test-4" class="pill-nav px-2.5 py-1.5 rounded-lg bg-[var(--secondary)] text-[var(--secondary-foreground)] font-semibold">4. Function (CAFA)</a>
          <a href="#test-5" class="pill-nav px-2.5 py-1.5 rounded-lg bg-[var(--secondary)] text-[var(--secondary-foreground)] font-semibold">5. Monomer Ala Scan</a>
          <a href="#test-6" class="pill-nav px-2.5 py-1.5 rounded-lg bg-[var(--secondary)] text-[var(--secondary-foreground)] font-semibold">6. Binding Ala Scan</a>
          <a href="#test-7" class="pill-nav px-2.5 py-1.5 rounded-lg bg-[var(--secondary)] text-[var(--secondary-foreground)] font-semibold">7. Monomer &amp; Motif</a>
          <a href="#test-8" class="pill-nav px-2.5 py-1.5 rounded-lg bg-[var(--secondary)] text-[var(--secondary-foreground)] font-semibold">8. Epitope Binders</a>
        </div>
      </div>

      <!-- VISUAL HYPOTHESIS SPLIT: NO-CODE vs WITH-CODE vs SPECIALISTS -->
      <div class="grid grid-cols-1 lg:grid-cols-3 gap-4 pt-2">
        <div class="rounded-xl border border-[var(--border)] bg-[var(--background)] p-4 space-y-2">
          <div class="flex items-center justify-between">
            <span class="px-2.5 py-0.5 rounded bg-amber-500/15 text-amber-500 text-xs font-bold">TRACK A • NO-CODE (PURE WEIGHTS)</span>
            <span class="text-xs text-[var(--muted-foreground)]">Direct &amp; Thinking</span>
          </div>
          <div class="text-sm font-bold">Mental Geometry + Implicit Fold Intuition</div>
          <p class="text-xs text-[var(--muted-foreground)]">
            No tools. Tests raw representations inside neural weights—just like ESM-2 or ESMFold—plus text chain-of-thought. Vulnerable to mental 3D floating-point math &amp; residue-counting errors.
          </p>
          <div class="pt-1 flex flex-wrap gap-1.5 text-xs">
            <span class="px-2 py-0.5 rounded bg-emerald-500/15 text-emerald-500 font-medium">Tests Weight Biology</span>
            <span class="px-2 py-0.5 rounded bg-rose-500/15 text-rose-500 font-medium">Hard on Raw (x,y,z) Floats</span>
          </div>
        </div>

        <div class="rounded-xl border-2 border-sky-500/50 bg-[var(--background)] p-4 space-y-2">
          <div class="flex items-center justify-between">
            <span class="px-2.5 py-0.5 rounded bg-sky-500/15 text-sky-500 text-xs font-bold">TRACK B • WITH-CODE (AIR-GAPPED NUMPY)</span>
            <span class="text-xs text-[var(--muted-foreground)]">No Net • No Bio-Libs</span>
          </div>
          <div class="text-sm font-bold">Separates Math Arithmetic from Missing Biology</div>
          <p class="text-xs text-[var(--muted-foreground)]">
            Gives models <code class="text-xs">python + numpy/scipy</code> only:
            <strong>Unlocks 3D coordinate geometry</strong> (Tests 2, 7, 8), <strong>but CANNOT fake 1D folding or binding physics</strong> (Tests 1, 3, 5, 6)!
          </p>
          <div class="pt-1 flex flex-wrap gap-1.5 text-xs">
            <span class="px-2 py-0.5 rounded bg-sky-500/15 text-sky-500 font-medium">Tests 2, 7, 8 Big Boost</span>
            <span class="px-2 py-0.5 rounded bg-purple-500/15 text-purple-500 font-medium">Tests 1, 3, 5, 6 Pure Biology</span>
          </div>
        </div>

        <div class="rounded-xl border border-[var(--border)] bg-[var(--background)] p-4 space-y-2">
          <div class="flex items-center justify-between">
            <span class="px-2.5 py-0.5 rounded bg-teal-500/15 text-teal-500 text-xs font-bold">TRACK C • SPECIALIST BIO-TOOLS</span>
            <span class="text-xs text-[var(--muted-foreground)]">Modal L40S GPUs</span>
          </div>
          <div class="text-sm font-bold">Purpose-Built Structural &amp; Evolutionary Ceilings</div>
          <p class="text-xs text-[var(--muted-foreground)]">
            Every test has its domain gold standard: <strong>ESMFold &amp; Boltz-2</strong> (1.), <strong>ProteinMPNN</strong> (2.), <strong>PickyBinders/tea</strong> (3.), <strong>ESM-2 kNN</strong> (4.), <strong>ThermoMPNN</strong> (5.), <strong>ESM-2 Complex &amp; MPNN ΔlogP</strong> (6.), <strong>ESM3</strong> (7.), and <strong>BindCraft AF2</strong> (8.).
          </p>
          <div class="pt-1 flex flex-wrap gap-1.5 text-xs">
            <span class="px-2 py-0.5 rounded bg-teal-500/15 text-teal-500 font-medium">Calibrated Head-to-Head Bar</span>
          </div>
        </div>
      </div>
    </header>

    <!-- =================================================================== -->
    <!-- TEST 1. FORWARD FOLDING                                             -->
    <!-- =================================================================== -->
    <section id="test-1" class="bg-[var(--card)] border border-[var(--border)] rounded-2xl p-6 space-y-5">
      <div class="flex flex-col md:flex-row md:items-center justify-between gap-4 border-b border-[var(--border)] pb-4">
        <div>
          <span class="px-2.5 py-0.5 rounded bg-blue-500/15 text-blue-500 text-xs font-bold">TEST 1 • N = 100 STRUCTURES • SPECIALISTS: ESMFOLD &amp; BOLTZ-2</span>
          <h2 class="text-xl font-bold mt-1">1. Forward Folding (Sequence → Contacts &amp; Q3 → 3D Structure)</h2>
          <p class="text-xs text-[var(--muted-foreground)]">Stratified into 5 novelty categories by <strong>exact Foldseek TM-align 3D TM-score</strong> and <strong>MMseqs2 sequence identity</strong> to pre-May-2026 PDB &amp; AlphaFold DB. Drag any 3D fold below!</p>
        </div>
        <div class="flex items-center gap-2 text-xs font-semibold shrink-0 flex-wrap">
          <span class="px-2.5 py-1.5 rounded-lg bg-[var(--background)] border border-[var(--border)]">1D Sequence</span>
          <span class="text-[var(--muted-foreground)]">→</span>
          <span class="px-2.5 py-1.5 rounded-lg bg-blue-500/15 text-blue-500">Cβ &lt; 8Å Map + Q3</span>
          <span class="text-[var(--muted-foreground)]">→</span>
          <span class="px-2.5 py-1.5 rounded-lg bg-[var(--background)] border border-[var(--border)]">Chiral L-BFGS 3D</span>
          <span class="text-[var(--muted-foreground)]">→</span>
          <span class="px-2.5 py-1.5 rounded-lg bg-emerald-500/15 text-emerald-500">Exact TM-Score</span>
        </div>
      </div>
      <div>
        <div class="text-xs font-bold uppercase tracking-wider text-[var(--muted-foreground)] mb-3">
          One Real Example Per Novelty Category (3D Fold + True 2D Contact Map + Q3 Ribbon)
        </div>
        <div class="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-5 gap-4" id="s1a-cards"></div>
      </div>
    </section>

    <!-- =================================================================== -->
    <!-- TEST 2. INVERSE FOLDING                                             -->
    <!-- =================================================================== -->
    <section id="test-2" class="bg-[var(--card)] border border-[var(--border)] rounded-2xl p-6 space-y-5">
      <div class="flex flex-col md:flex-row md:items-center justify-between gap-4 border-b border-[var(--border)] pb-4">
        <div>
          <span class="px-2.5 py-0.5 rounded bg-indigo-500/15 text-indigo-500 text-xs font-bold">TEST 2 • N = 100 STRIPPED BACKBONES • SPECIALIST: PROTEINMPNN</span>
          <h2 class="text-xl font-bold mt-1">2. Inverse Folding (Stripped 3D Backbone → Sequence → ESMFold Refold)</h2>
          <p class="text-xs text-[var(--muted-foreground)]">Design a foldable sequence from pure backbone <code class="text-xs">(N, Cα, C, O)</code> geometry with all amino-acid identities stripped to <code class="text-xs">UNK</code>.</p>
        </div>
        <div class="flex items-center gap-2 text-xs font-semibold shrink-0 flex-wrap">
          <span class="px-2.5 py-1.5 rounded-lg bg-[var(--background)] border border-[var(--border)]">Stripped UNK (x,y,z)</span>
          <span class="text-[var(--muted-foreground)]">→</span>
          <span class="px-2.5 py-1.5 rounded-lg bg-indigo-500/15 text-indigo-500">Designed Sequence</span>
          <span class="text-[var(--muted-foreground)]">→</span>
          <span class="px-2.5 py-1.5 rounded-lg bg-emerald-500/15 text-emerald-500">ESMFold scRMSD &lt; 2.0 Å</span>
        </div>
      </div>

      <div class="rounded-xl border border-[var(--border)] bg-[var(--background)] p-5 grid grid-cols-1 lg:grid-cols-12 gap-5 items-center">
        <div class="lg:col-span-5 space-y-2">
          <span class="px-2 py-0.5 rounded bg-indigo-500/15 text-indigo-500 text-xs font-bold">VISUAL EXAMPLE • NOVEL FOLD 11CY</span>
          <h3 class="text-base font-bold" id="s1b-title">Inverse Folding on Novel Fold 11CY (PDB TM = 0.35)</h3>
          <p class="text-xs text-[var(--muted-foreground)]">
            In <strong>No-Code</strong>, the model reads raw floating-point lines in text. In <strong>With-Code (NumPy)</strong>, it can compute exact virtual-Cβ burial (<strong>red = buried hydrophobic core</strong>, <strong>sky-blue = solvent-exposed surface</strong>) and Ramachandran <code class="text-xs">(φ, ψ)</code> angles—directly testing whether an LLM can program a physical packing function!
          </p>
        </div>
        <div class="lg:col-span-4 rounded-xl border border-[var(--border)] bg-[var(--card)] p-2 relative">
          <canvas id="s1b-canvas" width="340" height="200" class="w-full h-[185px] cursor-grab"></canvas>
          <div class="absolute bottom-2 left-3 text-xs text-[var(--muted-foreground)] pointer-events-none">Red = Buried Core • Blue = Surface</div>
        </div>
        <div class="lg:col-span-3 space-y-2 text-xs">
          <div class="p-3 rounded-lg bg-[var(--card)] border border-[var(--border)]">
            <div class="font-bold text-indigo-500">Input Shown to Model</div>
            <div class="font-mono mt-1 text-[var(--muted-foreground)]">ATOM 1 N UNK A 1 ...<br/>ATOM 2 CA UNK A 1 ...<br/>ATOM 3 C UNK A 1 ...</div>
          </div>
          <div class="p-3 rounded-lg bg-[var(--card)] border border-[var(--border)]">
            <div class="font-bold text-emerald-500">Graded By</div>
            <div class="mt-1 text-[var(--muted-foreground)]">Core NSR %, Surface NSR %, &amp; ESMFold refolded <strong>scRMSD &lt; 2.0 Å</strong> vs. ProteinMPNN</div>
          </div>
        </div>
      </div>
    </section>

    <!-- =================================================================== -->
    <!-- TEST 3. REMOTE HOMOLOGY DETECTION                                   -->
    <!-- =================================================================== -->
    <section id="test-3" class="bg-[var(--card)] border border-[var(--border)] rounded-2xl p-6 space-y-5">
      <div class="border-b border-[var(--border)] pb-4">
        <span class="px-2.5 py-0.5 rounded bg-purple-500/15 text-purple-500 text-xs font-bold">TEST 3 • N = 120 SCOPe 2.08 PAIRS • SPECIALIST: PICKYBINDERS/TEA &amp; FOLDSEEK</span>
        <h2 class="text-xl font-bold mt-1">3. Remote Homology Detection (Sequence-Easy → Foldseek → Deep TEA Twilight Zone)</h2>
        <p class="text-xs text-[var(--muted-foreground)]">Testing whether raw protein pretraining recognizes shared 3D superfamily topology at &lt;15% SeqID where even coding Smith–Waterman in Python fails.</p>
      </div>
      <div>
        <div class="text-xs font-bold uppercase tracking-wider text-[var(--muted-foreground)] mb-3">
          One Real SCOPe 2.08 Pair Per Category (4 Categories: 3 Homology Depths + 1 Class/Length-Matched Decoy)
        </div>
        <div class="grid grid-cols-1 md:grid-cols-2 gap-4" id="s2a-rung-cards"></div>
      </div>
    </section>

    <!-- =================================================================== -->
    <!-- TEST 4. PROTEIN FUNCTION PREDICTION (CAFA)                          -->
    <!-- =================================================================== -->
    <section id="test-4" class="bg-[var(--card)] border border-[var(--border)] rounded-2xl p-6 space-y-5">
      <div class="border-b border-[var(--border)] pb-4">
        <span class="px-2.5 py-0.5 rounded bg-amber-500/15 text-amber-500 text-xs font-bold">TEST 4 • N = 60 2026 QUICKGO HOLDOUTS • SPECIALIST: ESM-2 k-NN &amp; HOMOLOGY</span>
        <h2 class="text-xl font-bold mt-1">4. Protein Function Prediction (CAFA Experimental Gene Ontology)</h2>
        <p class="text-xs text-[var(--muted-foreground)]">Predicting experimental Molecular Function (MF), Biological Process (BP), and Cellular Component (CC) annotations curated in 2026.</p>
      </div>
      <div class="grid grid-cols-1 lg:grid-cols-12 gap-5 items-center">
        <div class="lg:col-span-5 rounded-xl border border-[var(--border)] bg-[var(--background)] p-4 space-y-2">
          <div class="text-xs font-bold text-amber-500">CORE IDEA • WHY IC-WEIGHTED wF_max BEATS GENERIC GUESSING</div>
          <svg viewBox="0 0 380 200" class="w-full h-auto">
            <line x1="190" y1="34" x2="105" y2="92" stroke="#94a3b8" stroke-width="1.8" />
            <line x1="190" y1="34" x2="275" y2="92" stroke="#94a3b8" stroke-width="1.8" />
            <line x1="275" y1="116" x2="275" y2="162" stroke="#10b981" stroke-width="2.4" />
            <rect x="115" y="10" width="150" height="28" rx="7" fill="#64748b" fill-opacity="0.18" stroke="#64748b" />
            <text x="190" y="28" font-size="12" text-anchor="middle" fill="currentColor">GO:0003674 Root (Excluded)</text>
            <rect x="20" y="90" width="165" height="34" rx="7" fill="#f59e0b" fill-opacity="0.18" stroke="#f59e0b" />
            <text x="102" y="105" font-size="12" font-weight="600" text-anchor="middle" fill="#f59e0b">binding (IC = 0.8 bits)</text>
            <text x="102" y="119" font-size="12" text-anchor="middle" fill="#94a3b8">CAFA Naive spams this</text>
            <rect x="195" y="90" width="165" height="30" rx="7" fill="#3b82f6" fill-opacity="0.16" stroke="#3b82f6" />
            <text x="277" y="109" font-size="12" font-weight="600" text-anchor="middle" fill="#3b82f6">transferase (IC = 3.1 bits)</text>
            <rect x="180" y="156" width="190" height="36" rx="7" fill="#10b981" fill-opacity="0.20" stroke="#10b981" stroke-width="1.8" />
            <text x="275" y="172" font-size="12" font-weight="700" text-anchor="middle" fill="#10b981">Ser/Thr kinase (IC = 6.8 bits)</text>
            <text x="275" y="186" font-size="12" text-anchor="middle" fill="#10b981">MF wF_max: Homology 0.36 vs Naive 0.13</text>
          </svg>
        </div>
        <div class="lg:col-span-7 space-y-3">
          <div class="text-xs font-bold uppercase tracking-wider text-[var(--muted-foreground)]">
            One Real Experimental Holdout Protein Per Temporal Category
          </div>
          <div class="grid grid-cols-1 md:grid-cols-2 gap-4" id="s2b-cards"></div>
        </div>
      </div>
    </section>

    <!-- =================================================================== -->
    <!-- TEST 5. MONOMER ALANINE SCAN RANKING                                -->
    <!-- =================================================================== -->
    <section id="test-5" class="bg-[var(--card)] border border-[var(--border)] rounded-2xl p-6 space-y-6">
      <div class="flex flex-col md:flex-row md:items-center justify-between gap-4 border-b border-[var(--border)] pb-4">
        <div>
          <span class="px-2.5 py-0.5 rounded bg-rose-500/15 text-rose-500 text-xs font-bold">TEST 5 • 52 PROTEINS × 18-CANDIDATE RANKING PANELS (FROM 5,348 MUTANTS) • SPECIALISTS: THERMOMPNN &amp; ESM-2</span>
          <h2 class="text-xl font-bold mt-1">5. Monomer Alanine Scan Ranking (Rank #1 Core Hotspot → #18 Neutral Decoy)</h2>
          <p class="text-xs text-[var(--muted-foreground)]">
            Models rank a balanced 18-mutation panel (<strong>6 hotspots + 6 moderate + 6 same-AA neutral decoys</strong>, e.g. buried core Leu vs. surface Leu). Graded by <strong>NDCG@5</strong>, <strong>Top-K Precision</strong>, <strong>Same-AA Decoy Win Rate</strong> (1D lookup = <strong>50% coin-flip</strong>), &amp; <strong>AA-Demeaned Rank ρ</strong>.
          </p>
        </div>
        <div class="flex items-center gap-3 text-xs shrink-0 flex-wrap">
          <span class="inline-flex items-center gap-1.5"><span class="w-3 h-3 rounded bg-rose-500 inline-block"></span>Hotspot</span>
          <span class="inline-flex items-center gap-1.5"><span class="w-3 h-3 rounded bg-amber-500 inline-block"></span>Moderate</span>
          <span class="inline-flex items-center gap-1.5"><span class="w-3 h-3 rounded bg-sky-500 inline-block"></span>Same-AA Neutral Decoy</span>
        </div>
      </div>
      <div class="grid grid-cols-1 lg:grid-cols-2 gap-5" id="s3a-cards"></div>
    </section>

    <!-- =================================================================== -->
    <!-- TEST 6. INTERFACE BINDING ALANINE SCAN RANKING                      -->
    <!-- =================================================================== -->
    <section id="test-6" class="bg-[var(--card)] border border-[var(--border)] rounded-2xl p-6 space-y-6">
      <div class="flex flex-col md:flex-row md:items-center justify-between gap-4 border-b border-[var(--border)] pb-4">
        <div>
          <span class="px-2.5 py-0.5 rounded bg-emerald-500/15 text-emerald-500 text-xs font-bold">TEST 6 • 38 COMPLEXES (31 KNOWN PDB vs. 7 UNSEEN/POST-2026) • MONOMER-STABILITY FILTERED</span>
          <h2 class="text-xl font-bold mt-1">6. Interface Binding Alanine Scan Ranking (Pure Handshake Hotspots vs. Memorization Split)</h2>
          <p class="text-xs text-[var(--muted-foreground)]">
            Monomer-melting core sites are excluded (via CDR/surface masks &amp; paired abundance scans) so 100% of candidates test <strong>direct partner binding</strong>. Stratified into <strong>Known Pre-2026 PDBs</strong> (where LLMs can recall crystal structures like <code class="text-xs">5B8C</code>) vs. <strong>Unseen / Post-2026 Complexes</strong> (<code class="text-xs">PP489×TIGIT</code>, <code class="text-xs">KRAS×LZTR1</code>).
          </p>
        </div>
        <div class="flex items-center gap-3 text-xs shrink-0 flex-wrap">
          <span class="inline-flex items-center gap-1.5"><span class="w-3 h-3 rounded bg-rose-500 inline-block"></span>Interface Hotspot</span>
          <span class="inline-flex items-center gap-1.5"><span class="w-3 h-3 rounded bg-sky-500 inline-block"></span>Same-AA Non-Interface Decoy</span>
        </div>
      </div>
      <div>
        <div class="text-xs font-bold uppercase tracking-wider text-[var(--muted-foreground)] mb-3">
          One Real Protein–Protein Complex Per Category (3D Complex + Full Scan + True #1..#K Ground-Truth Ranking Panel)
        </div>
        <div class="grid grid-cols-1 lg:grid-cols-2 gap-5" id="s6-cards"></div>
      </div>
    </section>

    <!-- =================================================================== -->
    <!-- TEST 7. DE NOVO MONOMER & MOTIF SCAFFOLDING                         -->
    <!-- =================================================================== -->
    <section id="test-7" class="bg-[var(--card)] border border-[var(--border)] rounded-2xl p-6 space-y-5">
      <div class="flex flex-col md:flex-row md:items-center justify-between gap-4 border-b border-[var(--border)] pb-4">
        <div>
          <span class="px-2.5 py-0.5 rounded bg-indigo-500/15 text-indigo-500 text-xs font-bold">TEST 7 • N = 40 DESIGN TASKS • SPECIALISTS: ESM3-OPEN &amp; PROGEN2</span>
          <h2 class="text-xl font-bold mt-1">7. De Novo Monomer Generation &amp; Crystal Motif Scaffolding</h2>
          <p class="text-xs text-[var(--muted-foreground)]">Must pass all 5 anti-cheat gates: (1) entropy/no EAAAK repeats, (2) ESMFold pLDDT ≥ 80 &amp; globular Rg, (3) MPNN scRMSD ≤ 2Å, (4) fold TM ≥ 0.50 or motif RMSD ≤ 1.5Å, (5) &lt;30% natural SeqID.</p>
        </div>
      </div>
      <div class="grid grid-cols-1 md:grid-cols-2 gap-5" id="s3b-p1-cards"></div>
    </section>

    <!-- =================================================================== -->
    <!-- TEST 8. CLOSED-LOOP EPITOPE BINDER DESIGN (AF2 ORACLE)              -->
    <!-- =================================================================== -->
    <section id="test-8" class="bg-[var(--card)] border border-[var(--border)] rounded-2xl p-6 space-y-6">
      <div class="border-b border-[var(--border)] pb-4 space-y-3">
        <div class="flex flex-wrap items-center justify-between gap-2">
          <span class="px-2.5 py-0.5 rounded bg-teal-500/15 text-teal-500 text-xs font-bold">TEST 8 • 12 PROVEN-BINDABLE EPITOPES × 16 AF2 SHOTS • SPECIALIST: BINDCRAFT &amp; PROTDBENCH</span>
          <span class="text-xs font-semibold text-teal-500">Closed-Loop Hit@16 = 92% vs. Blind Pass@16 = 33%</span>
        </div>
        <h2 class="text-xl font-bold">8. Closed-Loop Epitope Binder Design (AlphaFold2 Oracle Shots-to-Hit)</h2>

        <div class="rounded-xl border border-[var(--border)] bg-[var(--background)] p-4 flex flex-wrap items-center justify-between gap-2 text-xs font-semibold">
          <div class="px-3 py-2 rounded-lg bg-[var(--card)] border border-[var(--border)]">Target 3D Surface + 5 Red Hotspots</div>
          <span class="text-teal-500 font-bold">→</span>
          <div class="px-3 py-2 rounded-lg bg-sky-500/15 text-sky-500">LLM Proposes 55–85 AA Binder</div>
          <span class="text-teal-500 font-bold">→</span>
          <div class="px-3 py-2 rounded-lg bg-purple-500/15 text-purple-500">Modal L40S ColabDesign AF2 Co-Fold (~4s)</div>
          <span class="text-teal-500 font-bold">→</span>
          <div class="px-3 py-2 rounded-lg bg-amber-500/15 text-amber-500">Oracle Returns ipTM, iPAE, monoRMSD &amp; Hotspots</div>
          <span class="text-teal-500 font-bold">→</span>
          <div class="px-3 py-2 rounded-lg bg-emerald-500/15 text-emerald-500">16 Turns → Shots-to-First-Hit!</div>
        </div>
      </div>

      <div>
        <div class="text-xs font-bold uppercase tracking-wider text-[var(--muted-foreground)] mb-3">
          One Live 3D Epitope Example Per Proven-Bindable Category (Sky = Target, Red = 5 Hotspots, Green = Held-Out Binder)
        </div>
        <div class="grid grid-cols-1 lg:grid-cols-3 gap-5" id="s3b-p2-cards"></div>
      </div>
    </section>

  </div>

  <script>
    const ATLAS = __ATLAS_JSON__;

    // =========================================================================
    // Shared Interactive 3D Mini-Canvas Renderer
    // =========================================================================
    function attach3DViewer(canvas, renderCallback, initRotY = 0.6) {
      const st = { rotX: -0.32, rotY: initRotY, drag: false, lx: 0, ly: 0 };
      canvas.addEventListener('pointerdown', e => {
        st.drag = true; st.lx = e.clientX; st.ly = e.clientY;
        canvas.setPointerCapture(e.pointerId);
      });
      canvas.addEventListener('pointermove', e => {
        if (!st.drag) return;
        st.rotY += (e.clientX - st.lx) * 0.013;
        st.rotX += (e.clientY - st.ly) * 0.013;
        st.lx = e.clientX; st.ly = e.clientY;
        renderCallback(st);
      });
      canvas.addEventListener('pointerup', () => { st.drag = false; });
      renderCallback(st);
    }

    function projPt(pt, ctr, rx, ry, scale, w, h) {
      let x = pt[0] - ctr[0], y = pt[1] - ctr[1], z = pt[2] - ctr[2];
      const cy = Math.cos(ry), sy = Math.sin(ry);
      let x1 = x * cy + z * sy, z1 = -x * sy + z * cy;
      const cx = Math.cos(rx), sx = Math.sin(rx);
      let y2 = y * cx - z1 * sx, z2 = y * sx + z1 * cx;
      const p = 240 / (240 + z2 * 0.32);
      return [w / 2 + x1 * scale * p, h / 2 + y2 * scale * p, z2];
    }

    function computeBounds(pts) {
      let cx = 0, cy = 0, cz = 0;
      pts.forEach(p => { cx += p[0]; cy += p[1]; cz += p[2]; });
      cx /= pts.length; cy /= pts.length; cz /= pts.length;
      let maxR = 1;
      pts.forEach(p => {
        const d = Math.hypot(p[0] - cx, p[1] - cy, p[2] - cz);
        if (d > maxR) maxR = d;
      });
      return { ctr: [cx, cy, cz], maxR };
    }

    // =========================================================================
    // Render Test 1 & Test 2
    // =========================================================================
    const S1_LABELS = {
      post_cutoff_struct_novel: ['1. Structurally Novel', 'bg-emerald-500/15 text-emerald-500'],
      post_cutoff_de_novo: ['2. Post-Cutoff De Novo', 'bg-sky-500/15 text-sky-500'],
      post_cutoff_pdb_novel: ['3. PDB-Novel / In AFDB', 'bg-purple-500/15 text-purple-500'],
      post_cutoff_homolog_gradient: ['4. Homolog Gradient', 'bg-amber-500/15 text-amber-500'],
      pre_cutoff_control: ['5. Pre-Cutoff Control', 'bg-rose-500/15 text-rose-500']
    };

    function renderStage1Section() {
      const wrap = document.getElementById('s1a-cards');
      wrap.innerHTML = ATLAS.s1_examples.map((ex, idx) => {
        const [lbl, badgeCls] = S1_LABELS[ex.stratum];
        // Build Q3 secondary structure color strip (H=rose, E=amber, C=slate)
        const q3Spans = ex.q3.split('').map(c => {
          const cl = c === 'H' ? '#f43f5e' : (c === 'E' ? '#f59e0b' : '#94a3b844');
          return `<span style="flex:1;background:${cl};height:6px;display:inline-block"></span>`;
        }).join('');

        return `
          <div class="rounded-xl border border-[var(--border)] bg-[var(--background)] p-3.5 flex flex-col justify-between space-y-2.5">
            <div class="space-y-1">
              <div class="flex items-center justify-between gap-1">
                <span class="px-2 py-0.5 rounded text-xs font-bold ${badgeCls}">${lbl}</span>
                <span class="text-xs text-[var(--muted-foreground)]">n=${ex.n_in_stratum}</span>
              </div>
              <div class="font-bold text-sm">${ex.pdb} <span class="font-normal text-xs text-[var(--muted-foreground)]">(${ex.len} AA)</span></div>
              <div class="text-xs text-[var(--muted-foreground)] truncate" title="${ex.desc}">${ex.desc}</div>
            </div>

            <!-- Side-by-side 3D Canvas + 2D Contact Map Canvas -->
            <div class="grid grid-cols-3 gap-1.5 items-center">
              <div class="col-span-2 rounded-lg border border-[var(--border)] bg-[var(--card)] overflow-hidden">
                <canvas id="s1-3d-${idx}" width="180" height="135" class="w-full h-[120px] cursor-grab"></canvas>
              </div>
              <div class="col-span-1 rounded-lg border border-[var(--border)] bg-[var(--card)] p-1 flex flex-col items-center">
                <canvas id="s1-cmap-${idx}" width="76" height="76" class="w-[68px] h-[68px]"></canvas>
                <span class="text-xs text-[var(--muted-foreground)] mt-0.5">Contacts</span>
              </div>
            </div>

            <!-- Q3 Secondary Structure Strip -->
            <div class="space-y-1">
              <div class="flex rounded overflow-hidden">${q3Spans}</div>
              <div class="flex justify-between text-xs pt-1 border-t border-[var(--border)]">
                <span>PDB TM: <strong>${ex.pdb_tm.toFixed(2)}</strong></span>
                <span>SeqID: <strong>${ex.pdb_seqid}%</strong></span>
              </div>
            </div>
          </div>
        `;
      }).join('');

      ATLAS.s1_examples.forEach((ex, idx) => {
        const c3d = document.getElementById(`s1-3d-${idx}`);
        const { ctr, maxR } = computeBounds(ex.ca);
        attach3DViewer(c3d, st => {
          const ctx = c3d.getContext('2d');
          const w = c3d.width, h = c3d.height;
          ctx.clearRect(0, 0, w, h);
          const scale = Math.min(w, h) * 0.40 / maxR;
          const proj = ex.ca.map(p => projPt(p, ctr, st.rotX, st.rotY, scale, w, h));
          ctx.lineWidth = 2.8;
          ctx.lineCap = 'round';
          for (let i = 0; i < proj.length - 1; i++) {
            const q = ex.q3[i];
            ctx.strokeStyle = q === 'H' ? '#f43f5e' : (q === 'E' ? '#f59e0b' : '#38bdf8');
            ctx.beginPath();
            ctx.moveTo(proj[i][0], proj[i][1]);
            ctx.lineTo(proj[i + 1][0], proj[i + 1][1]);
            ctx.stroke();
          }
        }, 0.5 + idx * 0.4);

        // Draw symmetric 2D contact map
        const cm = document.getElementById(`s1-cmap-${idx}`);
        const mctx = cm.getContext('2d');
        const mw = cm.width, L = ex.shown_len;
        mctx.strokeStyle = '#64748b44';
        mctx.beginPath(); mctx.moveTo(0, 0); mctx.lineTo(mw, mw); mctx.stroke();
        mctx.fillStyle = '#3b82f6';
        ex.contacts.forEach(([i, j]) => {
          const x1 = (i / L) * mw, y1 = (j / L) * mw;
          mctx.fillRect(x1, y1, 2.1, 2.1);
          mctx.fillRect(y1, x1, 2.1, 2.1);
        });
      });

      // Test 2 Stripped UNK Backbone Viewer (colored by Core vs Surface!)
      const bEx = ATLAS.s1b_example;
      document.getElementById('s1b-title').textContent = `Inverse Folding on Novel Fold ${bEx.pdb} (${bEx.len} AA, PDB TM = ${bEx.pdb_tm})`;
      const c1b = document.getElementById('s1b-canvas');
      const bBounds = computeBounds(bEx.ca);
      const medNbr = [...bEx.nbrs].sort((a, b) => a - b)[Math.floor(bEx.nbrs.length * 0.58)];
      attach3DViewer(c1b, st => {
        const ctx = c1b.getContext('2d');
        const w = c1b.width, h = c1b.height;
        ctx.clearRect(0, 0, w, h);
        const scale = Math.min(w, h) * 0.41 / bBounds.maxR;
        const proj = bEx.ca.map(p => projPt(p, bBounds.ctr, st.rotX, st.rotY, scale, w, h));
        ctx.strokeStyle = '#94a3b888';
        ctx.lineWidth = 2.6;
        ctx.beginPath();
        proj.forEach((p, i) => { if (i === 0) ctx.moveTo(p[0], p[1]); else ctx.lineTo(p[0], p[1]); });
        ctx.stroke();
        proj.forEach((p, i) => {
          const isCore = bEx.nbrs[i] >= medNbr;
          ctx.fillStyle = isCore ? '#f43f5e' : '#38bdf8';
          ctx.beginPath();
          ctx.arc(p[0], p[1], isCore ? 3.8 : 2.5, 0, Math.PI * 2);
          ctx.fill();
        });
      }, 0.8);
    }

    // =========================================================================
    // Render Test 3 & Test 4
    // =========================================================================
    function renderStage2Section() {
      const s2aRep = ATLAS.s2_report.stage2a_remote_homology;
      const wrap2a = document.getElementById('s2a-rung-cards');
      wrap2a.innerHTML = ATLAS.s2a_examples.map(ex => {
        const isPos = ex.same_sf;
        const badge = isPos
          ? '<span class="px-2 py-0.5 rounded bg-emerald-500/15 text-emerald-500 text-xs font-bold">SAME SUPERFAMILY (POS)</span>'
          : '<span class="px-2 py-0.5 rounded bg-rose-500/15 text-rose-500 text-xs font-bold">DIFFERENT FOLD DECOY (NEG)</span>';

        let perfLine = '';
        if (isPos) {
          const rBlos = s2aRep.BLOSUM62_SmithWaterman.by_rung[ex.rung].recall_at_fpr10;
          const rFs = s2aRep.Foldseek_3Di_Structure.by_rung[ex.rung].recall_at_fpr10;
          perfLine = `
            <div class="flex items-center gap-4 text-xs pt-2 border-t border-[var(--border)]">
              <span>BLOSUM62 Rec@10%: <strong>${rBlos.toFixed(2)}</strong></span>
              <span>Foldseek 3Di Rec@10%: <strong>${rFs.toFixed(2)}</strong></span>
              <span class="text-[var(--muted-foreground)]">SeqID: ${ex.seqid}%</span>
            </div>
          `;
        } else {
          perfLine = `
            <div class="flex items-center gap-4 text-xs pt-2 border-t border-[var(--border)]">
              <span>Lengths: <strong>${ex.len_a} vs ${ex.len_b} AA</strong></span>
              <span>SCOPe: <strong>${ex.scop_a} vs ${ex.scop_b}</strong></span>
              <span class="text-[var(--muted-foreground)]">SeqID: ${ex.seqid}%</span>
            </div>
          `;
        }

        return `
          <div class="rounded-xl border border-[var(--border)] bg-[var(--background)] p-4 space-y-2.5">
            <div class="flex items-center justify-between gap-2 flex-wrap">
              <div class="font-bold text-sm">${ex.title}</div>
              ${badge}
            </div>
            <div class="text-xs text-[var(--muted-foreground)]">${ex.subtitle} (${ex.scop_a} vs. ${ex.scop_b})</div>
            <pre class="p-2.5 rounded-lg bg-[var(--card)] border border-[var(--border)] font-mono text-xs overflow-x-auto leading-tight">Seq A: ${ex.snip_a}...
       ${ex.snip_mid}
Seq B: ${ex.snip_b}...</pre>
            ${perfLine}
          </div>
        `;
      }).join('');

      const wrap2b = document.getElementById('s2b-cards');
      wrap2b.innerHTML = ATLAS.s2b_examples.map(ex => `
        <div class="rounded-xl border border-[var(--border)] bg-[var(--background)] p-4 space-y-2.5">
          <div class="text-xs font-bold text-amber-500">${ex.tier_label}</div>
          <div class="font-bold text-sm">${ex.tid} <span class="font-normal text-xs text-[var(--muted-foreground)]">• ${ex.organism} (${ex.len} AA)</span></div>
          <div class="text-xs text-[var(--muted-foreground)]">UniProt Public: <strong>${ex.uniprot_date}</strong> • QuickGO Exp: <strong>${ex.qgo_date}</strong></div>
          <div class="p-2 rounded bg-[var(--card)] border border-[var(--border)] font-mono text-xs truncate">${ex.seq_snip}</div>
          <div class="flex flex-wrap gap-1.5 pt-1">
            ${ex.top_terms.map(t => `
              <span class="px-2 py-0.5 rounded bg-[var(--card)] border border-[var(--border)] text-xs">
                <strong>[${t.aspect}]</strong> ${t.name} <span class="text-emerald-500 font-semibold">(${t.ic}b)</span>
              </span>
            `).join('')}
          </div>
        </div>
      `).join('');
    }

    function renderRankPills(rankedPanel) {
      // Show top 5 (#1..#5) + middle 2 + bottom 5 (same-AA neutral decoys) so card stays super compact!
      const n = rankedPanel.length;
      const shown = n <= 12
        ? rankedPanel
        : [...rankedPanel.slice(0, 5), { ellipsis: true }, ...rankedPanel.slice(n - 5)];
      return shown.map(item => {
        if (item.ellipsis) {
          return '<span class="px-1.5 py-0.5 text-xs text-[var(--muted-foreground)]">⋯</span>';
        }
        const cls = item.role === 'hotspot'
          ? 'bg-rose-500/15 text-rose-500 border-rose-500/30'
          : (item.role === 'same_aa_decoy'
              ? 'bg-sky-500/15 text-sky-500 border-sky-500/30'
              : 'bg-amber-500/15 text-amber-500 border-amber-500/30');
        return `<span title="True Rank #${item.rank} (${item.role}, exp=${item.val})" class="px-1.5 py-0.5 rounded border text-xs font-mono font-semibold ${cls}">#${item.rank} ${item.mut}</span>`;
      }).join('');
    }

    // =========================================================================
    // Render Test 5
    // =========================================================================
    function renderStage3ASection() {
      const wrap = document.getElementById('s3a-cards');
      wrap.innerHTML = ATLAS.s3a_examples.map((ex, idx) => {
        const maxV = Math.max(1.2, ...ex.destab.map(v => Math.abs(v)));
        const cells1 = ex.destab.map((v, i) => {
          const norm = Math.max(0, Math.min(1, v / (maxV * 0.85)));
          const bg = v > 0.25 ? `rgba(244, 63, 94, ${(0.22 + 0.78 * norm).toFixed(2)})` : 'rgba(56, 189, 248, 0.28)';
          return `<div title="${ex.wt[i]}${ex.pos[i]}A: ${v}" style="flex:1;height:16px;background:${bg};min-width:3px"></div>`;
        }).join('');

        const has3d = ex.ca && ex.ca.length > 0;
        return `
          <div class="rounded-xl border border-[var(--border)] bg-[var(--background)] p-4 space-y-3">
            <div class="flex items-start justify-between gap-2">
              <div>
                <div class="text-xs font-bold text-rose-500">${ex.title}</div>
                <div class="font-bold text-sm mt-0.5">${ex.name}</div>
                <div class="text-xs text-[var(--muted-foreground)]">${ex.subtitle} (${ex.n_panel} ranked from ${ex.n_full} assayed)</div>
              </div>
              <span class="px-2 py-0.5 rounded bg-[var(--card)] border border-[var(--border)] text-xs shrink-0">${ex.date}</span>
            </div>

            ${has3d ? `
              <div class="rounded-lg border border-[var(--border)] bg-[var(--card)] overflow-hidden relative">
                <canvas id="s3a-3d-${idx}" width="420" height="140" class="w-full h-[130px] cursor-grab"></canvas>
                <div class="absolute bottom-1.5 left-2.5 text-xs text-[var(--muted-foreground)] pointer-events-none">3D Backbone Colored by Wet-Lab X→A ΔΔG_fold (Red = Core Hotspot)</div>
              </div>
            ` : ''}

            <div class="space-y-1">
              <div class="text-xs font-semibold">Full Wet-Lab Landscape (${ex.n_full} X→A sites):</div>
              <div class="flex rounded overflow-hidden border border-[var(--border)]">${cells1}</div>
            </div>

            <div class="space-y-1 pt-1 border-t border-[var(--border)]">
              <div class="text-xs font-semibold">Ground-Truth Ranking Order (#1 Most Disruptive → #${ex.n_panel} Neutral Decoy):</div>
              <div class="flex flex-wrap gap-1">${renderRankPills(ex.ranked_panel)}</div>
            </div>
          </div>
        `;
      }).join('');

      ATLAS.s3a_examples.forEach((ex, idx) => {
        if (!ex.ca || !ex.ca.length) return;
        const c = document.getElementById(`s3a-3d-${idx}`);
        const { ctr, maxR } = computeBounds(ex.ca);
        const posMap = {};
        ex.full_pos.forEach((p1, k) => { posMap[p1] = ex.full_destab[k]; });
        attach3DViewer(c, st => {
          const ctx = c.getContext('2d');
          const w = c.width, h = c.height;
          ctx.clearRect(0, 0, w, h);
          const scale = Math.min(w, h) * 0.42 / maxR;
          const proj = ex.ca.map(p => projPt(p, ctr, st.rotX, st.rotY, scale, w, h));
          ctx.lineWidth = 3.0;
          for (let i = 0; i < proj.length - 1; i++) {
            const ddg = posMap[i + 1] || 0;
            ctx.strokeStyle = ddg > 1.0 ? '#f43f5e' : (ddg > 0.4 ? '#fb923c' : '#38bdf8');
            ctx.beginPath(); ctx.moveTo(proj[i][0], proj[i][1]); ctx.lineTo(proj[i + 1][0], proj[i + 1][1]); ctx.stroke();
          }
          proj.forEach((p, i) => {
            const ddg = posMap[i + 1] || 0;
            if (ddg > 1.1) {
              ctx.fillStyle = '#f43f5e';
              ctx.beginPath(); ctx.arc(p[0], p[1], 4.2, 0, Math.PI * 2); ctx.fill();
            }
          });
        }, 0.7);
      });
    }

    // =========================================================================
    // Render Test 6 (Interface Binding Alanine Scanning)
    // =========================================================================
    function renderTest6BindingSection() {
      const wrap6 = document.getElementById('s6-cards');
      wrap6.innerHTML = ATLAS.s6_examples.map((ex, idx) => {
        const sortedAbs = [...ex.loss].sort((a, b) => a - b);
        const q80 = sortedAbs[Math.floor(sortedAbs.length * 0.80)] || 0.4;
        const maxL = Math.max(0.5, ...ex.loss.map(v => Math.abs(v)));
        const cells = ex.loss.map((v, i) => {
          const norm = Math.max(0, Math.min(1, v / (maxL * 0.85)));
          const isHot = v >= Math.max(0.18, q80);
          const bg = isHot
            ? `rgba(244, 63, 94, ${(0.35 + 0.65 * norm).toFixed(2)})`
            : (v > 0.05 ? 'rgba(251, 146, 60, 0.35)' : 'rgba(56, 189, 248, 0.25)');
          return `<div title="${ex.wt[i]}${ex.pos[i]}A binding loss: ${ex.raw_loss[i]}" style="flex:1;height:18px;background:${bg};min-width:3px"></div>`;
        }).join('');

        const has3d = ex.ca_a && ex.ca_a.length > 0 && ex.ca_b && ex.ca_b.length > 0;
        const memBadge = ex.mem_stratum === 'unseen_or_post2026'
          ? '<span class="px-2 py-0.5 rounded bg-purple-500/15 text-purple-500 text-xs font-bold shrink-0">UNSEEN / POST-2026</span>'
          : '<span class="px-2 py-0.5 rounded bg-amber-500/15 text-amber-500 text-xs font-bold shrink-0">KNOWN PRE-2026 PDB</span>';
        return `
          <div class="rounded-xl border border-[var(--border)] bg-[var(--background)] p-4 space-y-3">
            <div class="flex items-start justify-between gap-2">
              <div>
                <div class="text-xs font-bold text-emerald-500">${ex.title}</div>
                <div class="font-bold text-sm mt-0.5">${ex.name}</div>
                <div class="text-xs text-[var(--muted-foreground)]">${ex.subtitle} • Chain A (${ex.len_a} AA) × Partner B (${ex.len_b} AA)</div>
              </div>
              <div class="flex flex-col items-end gap-1 shrink-0">
                ${memBadge}
                <span class="px-2 py-0.5 rounded bg-[var(--card)] border border-[var(--border)] text-xs">${ex.date}</span>
              </div>
            </div>

            ${has3d ? `
              <div class="rounded-lg border border-[var(--border)] bg-[var(--card)] overflow-hidden relative">
                <canvas id="s6-3d-${idx}" width="420" height="155" class="w-full h-[145px] cursor-grab"></canvas>
                <div class="absolute bottom-1.5 left-2.5 text-xs text-[var(--muted-foreground)] pointer-events-none">Sky = Scanned Chain A • Green = Partner B • Red Spheres = Wet-Lab ΔΔG_bind Hotspots</div>
              </div>
            ` : ''}

            <div class="space-y-1">
              <div class="text-xs font-semibold">Full Wet-Lab Binding Landscape (${ex.n_full} X→A sites):</div>
              <div class="flex rounded overflow-hidden border border-[var(--border)]">${cells}</div>
            </div>

            <div class="space-y-1 pt-1 border-t border-[var(--border)]">
              <div class="text-xs font-semibold">Ground-Truth Binding Ranking (#1 Critical Hotspot → #${ex.n_panel} Non-Interface Decoy):</div>
              <div class="flex flex-wrap gap-1">${renderRankPills(ex.ranked_panel)}</div>
            </div>
          </div>
        `;
      }).join('');

      ATLAS.s6_examples.forEach((ex, idx) => {
        if (!ex.ca_a || !ex.ca_a.length || !ex.ca_b || !ex.ca_b.length) return;
        const c = document.getElementById(`s6-3d-${idx}`);
        const allPts = ex.ca_a.concat(ex.ca_b);
        const { ctr, maxR } = computeBounds(allPts);
        const lossMap = {};
        ex.full_pos.forEach((p1, k) => { lossMap[p1] = ex.full_loss[k]; });
        attach3DViewer(c, st => {
          const ctx = c.getContext('2d');
          const w = c.width, h = c.height;
          ctx.clearRect(0, 0, w, h);
          const scale = Math.min(w, h) * 0.42 / maxR;
          const pA = ex.ca_a.map(p => projPt(p, ctr, st.rotX, st.rotY, scale, w, h));
          const pB = ex.ca_b.map(p => projPt(p, ctr, st.rotX, st.rotY, scale, w, h));

          ctx.strokeStyle = '#10b981';
          ctx.lineWidth = 2.6;
          ctx.beginPath();
          pB.forEach((p, i) => { if (i === 0) ctx.moveTo(p[0], p[1]); else ctx.lineTo(p[0], p[1]); });
          ctx.stroke();

          ctx.strokeStyle = '#0ea5e9';
          ctx.lineWidth = 2.9;
          ctx.beginPath();
          pA.forEach((p, i) => { if (i === 0) ctx.moveTo(p[0], p[1]); else ctx.lineTo(p[0], p[1]); });
          ctx.stroke();

          pA.forEach((p, i0) => {
            const ddg = lossMap[i0 + 1] || 0;
            if (ddg >= 1.2) {
              ctx.fillStyle = '#f43f5e';
              ctx.beginPath(); ctx.arc(p[0], p[1], 5.0, 0, Math.PI * 2); ctx.fill();
              ctx.strokeStyle = '#ffffff'; ctx.lineWidth = 1.2; ctx.stroke();
            }
          });
        }, 0.65);
      });
    }

    // =========================================================================
    // Render Test 7 & Test 8
    // =========================================================================
    function renderStage3BSection() {
      const p1Wrap = document.getElementById('s3b-p1-cards');
      p1Wrap.innerHTML = ATLAS.s3b_part1_examples.map((ex, idx) => `
        <div class="rounded-xl border border-[var(--border)] bg-[var(--background)] p-4 space-y-2.5">
          <div class="flex items-center justify-between gap-2 flex-wrap">
            <div class="font-bold text-sm">${ex.title}</div>
            <span class="px-2 py-0.5 rounded bg-teal-500/15 text-teal-500 text-xs font-semibold">${ex.badge}</span>
          </div>
          <div class="text-xs text-[var(--muted-foreground)]">${ex.subtitle}</div>
          <div class="rounded-lg border border-[var(--border)] bg-[var(--card)] overflow-hidden relative">
            <canvas id="s3b-p1-${idx}" width="440" height="165" class="w-full h-[155px] cursor-grab"></canvas>
            <div class="absolute bottom-1.5 left-2.5 text-xs text-[var(--muted-foreground)] pointer-events-none">Drag to rotate 3D reference topology / gold crystal motif</div>
          </div>
        </div>
      `).join('');

      ATLAS.s3b_part1_examples.forEach((ex, idx) => {
        const c = document.getElementById(`s3b-p1-${idx}`);
        const { ctr, maxR } = computeBounds(ex.ca);
        const hi = new Set(ex.highlight_0idx || []);
        attach3DViewer(c, st => {
          const ctx = c.getContext('2d');
          const w = c.width, h = c.height;
          ctx.clearRect(0, 0, w, h);
          const scale = Math.min(w, h) * 0.41 / maxR;
          const proj = ex.ca.map(p => projPt(p, ctr, st.rotX, st.rotY, scale, w, h));
          for (let i = 0; i < proj.length - 1; i++) {
            const isM = hi.has(i) && hi.has(i + 1);
            ctx.strokeStyle = isM ? '#eab308' : '#6366f1';
            ctx.lineWidth = isM ? 4.4 : 2.8;
            ctx.beginPath(); ctx.moveTo(proj[i][0], proj[i][1]); ctx.lineTo(proj[i + 1][0], proj[i + 1][1]); ctx.stroke();
          }
          proj.forEach((p, i) => {
            if (hi.has(i)) {
              ctx.fillStyle = '#eab308';
              ctx.beginPath(); ctx.arc(p[0], p[1], 4.8, 0, Math.PI * 2); ctx.fill();
            }
          });
        }, 0.5 + idx * 0.8);
      });

      const p2Wrap = document.getElementById('s3b-p2-cards');
      p2Wrap.innerHTML = ATLAS.s3b_part2_examples.map((ex, idx) => `
        <div class="rounded-xl border border-[var(--border)] bg-[var(--background)] p-4 flex flex-col justify-between space-y-2.5">
          <div class="space-y-1">
            <div class="text-xs font-bold text-teal-500">${ex.title}</div>
            <div class="font-bold text-sm">${ex.pdb} <span class="font-normal text-xs text-[var(--muted-foreground)]">(${ex.date})</span></div>
            <div class="text-xs text-[var(--muted-foreground)]">${ex.subtitle}</div>
          </div>
          <div class="rounded-lg border border-[var(--border)] bg-[var(--card)] overflow-hidden relative">
            <canvas id="s3b-p2-${idx}" width="340" height="190" class="w-full h-[175px] cursor-grab"></canvas>
          </div>
          <div class="text-xs pt-1 border-t border-[var(--border)] flex justify-between gap-2">
            <span>Hotspots: <strong>${ex.hs_labels.join(', ')}</strong></span>
            <span class="text-[var(--muted-foreground)]">${ex.len_p ? `Held-out: ${ex.len_p}AA` : 'Apo Surface Patch'}</span>
          </div>
        </div>
      `).join('');

      ATLAS.s3b_part2_examples.forEach((ex, idx) => {
        const c = document.getElementById(`s3b-p2-${idx}`);
        const allPts = ex.t_ca.concat(ex.p_ca || []);
        const { ctr, maxR } = computeBounds(allPts);
        const hsSet = new Set(ex.hotspots);
        attach3DViewer(c, st => {
          const ctx = c.getContext('2d');
          const w = c.width, h = c.height;
          ctx.clearRect(0, 0, w, h);
          const scale = Math.min(w, h) * 0.41 / maxR;
          const tP = ex.t_ca.map(p => projPt(p, ctr, st.rotX, st.rotY, scale, w, h));
          ctx.strokeStyle = '#0ea5e9';
          ctx.lineWidth = 2.8;
          ctx.beginPath();
          tP.forEach((p, i) => { if (i === 0) ctx.moveTo(p[0], p[1]); else ctx.lineTo(p[0], p[1]); });
          ctx.stroke();

          if (ex.p_ca && ex.p_ca.length > 0) {
            const pP = ex.p_ca.map(p => projPt(p, ctr, st.rotX, st.rotY, scale, w, h));
            ctx.strokeStyle = '#10b981';
            ctx.lineWidth = 3.2;
            ctx.beginPath();
            pP.forEach((p, i) => { if (i === 0) ctx.moveTo(p[0], p[1]); else ctx.lineTo(p[0], p[1]); });
            ctx.stroke();
          }

          tP.forEach((p, i0) => {
            if (hsSet.has(i0 + 1)) {
              ctx.fillStyle = '#f43f5e';
              ctx.beginPath(); ctx.arc(p[0], p[1], 5.6, 0, Math.PI * 2); ctx.fill();
              ctx.strokeStyle = '#ffffff'; ctx.lineWidth = 1.4; ctx.stroke();
            }
          });
        }, 0.8 + idx * 0.5);
      });
    }

    renderStage1Section();
    renderStage2Section();
    renderStage3ASection();
    renderTest6BindingSection();
    renderStage3BSection();
  </script>
</body>
</html>
"""


def main() -> None:
    atlas_data = compile_visual_atlas_data()
    html = VISUAL_HTML.replace("__ATLAS_JSON__", json.dumps(atlas_data, separators=(",", ":")))
    out_repo = REPO / "outputs/figures/fable_protein_bench_summary.html"
    out_repo.write_text(html, encoding="utf-8")
    print(f"Built visual benchmark atlas -> {out_repo} ({out_repo.stat().st_size / 1024:.1f} KB)")


if __name__ == "__main__":
    main()
