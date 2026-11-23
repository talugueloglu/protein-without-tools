#!/usr/bin/env python3
"""Run JapanFold-backed established baselines across Tasks 2, 3, 4, 7, and 8:
  - Task 2: Refold ProteinMPNN (v_48_020) 100 inverse-folding designs with ESMFold-2
  - Tasks 3 & 4: Compute ESMC-600M sequence embeddings (/v1/embeddings) for Stage 2A & Stage 2B
  - Task 7: Run RFdiffusion3 (`rfd3-scaffold`) + local ProteinMPNN (v_48_020), then grade with ESMFold-2 + OpenFold3
  - Task 8: Run BoltzGen (`protein-anything`, 16 designs/epitope) and grade with AF2-IG + ESMFold-2 oracle
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict
import hashlib
import io
import json
from pathlib import Path
import random
import re
import sys
import tempfile
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
if str(REPO / "src") not in sys.path:
    sys.path.insert(0, str(REPO / "src"))
if str(REPO / "scripts") not in sys.path:
    sys.path.insert(0, str(REPO / "scripts"))
if str(REPO / "outputs" / "proteinmpnn_cache") not in sys.path:
    sys.path.insert(0, str(REPO / "outputs" / "proteinmpnn_cache"))

from evaluate_tests_2_7_japanfold import evaluate_test2, evaluate_test7
from fable_bench.japanfold_client import JapanFoldClient, parse_pdb_chain_with_bfactor
from fable_bench.pdb_utils import CANONICAL_AA_SET
from fable_bench.stage2a_homology import smith_waterman_affine
from fable_bench.stage3b_design import EpitopeTarget, evaluate_binder_campaign
from protein_mpnn_utils import ProteinMPNN, _S_to_seq, parse_PDB, tied_featurize
from run_local_t1_t2_mpnn import load_proteinmpnn_model
from run_test8_af2_japanfold import evaluate_binders_af2_batch


def _seq_id(seq: str) -> str:
    return "s_" + hashlib.sha1(seq.encode("utf-8")).hexdigest()[:18]


def run_esmc_embeddings(jf: JapanFoldClient) -> None:
    out_npz = REPO / "outputs" / "japanfold_cache" / "esmc_600m_pooled.npz"
    out_idx = REPO / "outputs" / "japanfold_cache" / "esmc_600m_index.json"
    if out_npz.exists() and out_idx.exists():
        print(f"[ESMC-600M] Reusing cached embeddings -> {out_npz.name}", flush=True)
        return

    m2a = json.loads((REPO / "data" / "stage2a_homology" / "manifest.json").read_text(encoding="utf-8"))["pairs"]
    m2b = json.loads((REPO / "data" / "stage2b_cafa" / "manifest.json").read_text(encoding="utf-8"))["targets"]
    ref2b = json.loads((REPO / "data" / "stage2b_cafa" / "reference_swissprot_go.json").read_text(encoding="utf-8"))

    raw_seqs: set[str] = set()
    for p in m2a:
        raw_seqs.add(p["seq_a"].strip().upper()[:1990])
        raw_seqs.add(p["seq_b"].strip().upper()[:1990])
    for t in m2b:
        raw_seqs.add(t["sequence"].strip().upper()[:1990])
    for r in ref2b:
        raw_seqs.add(r["sequence"].strip().upper()[:1990])

    ordered_seqs = sorted(s for s in raw_seqs if len(s) >= 10)
    print(f"\n=== [Tasks 3 & 4] Computing ESMC-600M Embeddings for {len(ordered_seqs)} sequences ===", flush=True)

    vecs_by_sid: Dict[str, np.ndarray] = {}
    chunk_size = 350
    for c0 in range(0, len(ordered_seqs), chunk_size):
        chunk = ordered_seqs[c0 : c0 + chunk_size]
        seq_payload = [{"id": _seq_id(s), "sequence": s} for s in chunk]
        meta, files = jf.submit_and_collect(
            "/v1/embeddings",
            {
                "model": "esmc-600m",
                "sequences": seq_payload,
                "params": {"pool": "mean", "format": "npz"},
            },
            cache_prefix=f"esmc600m_b{c0}",
            download_all=True,
        )
        seq_rows = meta["results"].get("sequences") or []
        file_by_id = {str(r.get("id")): str(r.get("file") or f"{r.get('id')}.npz") for r in seq_rows}
        for s in chunk:
            sid = _seq_id(s)
            rel_f = file_by_id.get(sid, f"{sid}.npz")
            raw_b = files.get(rel_f)
            if raw_b is None:
                for fk, fv in files.items():
                    if fk.endswith(f"{sid}.npz"):
                        raw_b = fv
                        break
            if raw_b is None:
                raise RuntimeError(f"Missing NPZ embedding for {sid} (files sample: {list(files.keys())[:5]})")
            z = np.load(io.BytesIO(raw_b))
            vec: Optional[np.ndarray] = None
            for k in ("pooled", "mean", "embedding", "cls", "embeddings"):
                if k in z and z[k].ndim == 1:
                    vec = np.asarray(z[k], dtype=np.float32)
                    break
            if vec is None:
                for k in z.files:
                    arr = np.asarray(z[k], dtype=np.float32)
                    if arr.ndim == 1 and arr.shape[0] >= 256:
                        vec = arr
                        break
                    if arr.ndim == 2 and arr.shape[-1] >= 256:
                        vec = np.mean(arr, axis=0).astype(np.float32)
                        break
            if vec is None:
                raise RuntimeError(f"Could not extract 1D vector from {sid}.npz keys={z.files}")
            vecs_by_sid[sid] = vec
        print(
            f"  [ESMC-600M] Embedded {min(c0 + chunk_size, len(ordered_seqs))}/{len(ordered_seqs)} "
            f"(cost=${meta['cost_usd']:.5f})",
            flush=True,
        )

    sids = sorted(vecs_by_sid.keys())
    mat = np.stack([vecs_by_sid[k] for k in sids], axis=0)
    idx_map = {s: _seq_id(s) for s in ordered_seqs}
    np.savez_compressed(out_npz, sids=np.array(sids), embeddings=mat)
    out_idx.write_text(json.dumps(idx_map), encoding="utf-8")
    print(f"Saved ESMC-600M embeddings {mat.shape} -> {out_npz.name}", flush=True)


def _design_with_mpnn_on_pdb_text(
    mpnn_model: ProteinMPNN,
    pdb_or_cif_text: str,
    spec: Dict[str, Any],
    seed: int,
) -> str:
    device = torch.device("cpu")
    alphabet = "ACDEFGHIKLMNPQRSTVWYX"
    aa_to_idx = {a: i for i, a in enumerate(alphabet)}

    if "_atom_site." in pdb_or_cif_text or pdb_or_cif_text.lstrip().startswith("data_"):
        from build_and_evaluate_stage3b import parse_cif_all_protein_chains

        chs = parse_cif_all_protein_chains(pdb_or_cif_text)
        c0 = chs.get("A") or next(iter(chs.values()))
        ch_a = {
            "sequence": c0["sequence"],
            "coords_n": c0["n"],
            "coords_ca": c0["ca"],
            "coords_c": c0["c"],
            "coords_o": c0["o"],
        }
    else:
        ch_a = parse_pdb_chain_with_bfactor(pdb_or_cif_text, "A")
    L = len(ch_a["sequence"])
    fixed_map: Dict[int, str] = {}
    if spec["category"] == "motif_scaffolding" and spec["fixed_motif_seqs"]:
        cursor = 0
        for mseq in spec["fixed_motif_seqs"]:
            pos = ch_a["sequence"].find(mseq, cursor)
            if pos != -1:
                for k_i, m_aa in enumerate(mseq):
                    fixed_map[pos + k_i] = m_aa
                cursor = pos + len(mseq)
        if not fixed_map and spec["fixed_0idx"]:
            flat_motif = "".join(spec["fixed_motif_seqs"])
            if len(flat_motif) == len(spec["fixed_0idx"]) and max(spec["fixed_0idx"]) < L:
                for idx0, m_aa in zip(spec["fixed_0idx"], flat_motif):
                    fixed_map[int(idx0)] = m_aa

    # Rewrite clean PDB with motif residues placed on fixed_0idx so S_true holds exact motif letters!
    from fable_bench.japanfold_client import AA1_TO_3

    lines = []
    serial = 1
    for r0 in range(L):
        aa1 = fixed_map.get(r0, ch_a["sequence"][r0] if ch_a["sequence"][r0] in AA1_TO_3 else "A")
        res3 = AA1_TO_3.get(aa1, "ALA")
        for aname, arr, elem in (
            ("N", ch_a["coords_n"][r0], "N"),
            ("CA", ch_a["coords_ca"][r0], "C"),
            ("C", ch_a["coords_c"][r0], "C"),
            ("O", ch_a["coords_o"][r0], "O"),
        ):
            xyz = arr
            lines.append(
                f"ATOM  {serial:5d}  {aname:<3s} {res3:>3s} A{r0 + 1:4d}    "
                f"{xyz[0]:8.3f}{xyz[1]:8.3f}{xyz[2]:8.3f}  1.00 80.00           {elem:>2s}"
            )
            serial += 1
    lines.append("TER\nEND\n")
    clean_pdb = "\n".join(lines)

    with tempfile.NamedTemporaryFile("w", suffix=".pdb", delete=False) as tf:
        tf.write(clean_pdb)
        tmp_path = tf.name

    try:
        pdb_dict_list = parse_PDB(tmp_path, input_chain_list=["A"], ca_only=False)
        bname = pdb_dict_list[0]["name"]
        # ProteinMPNN fixed_position_dict uses 1-based positions within chain A!
        fixed_1idx = sorted(i0 + 1 for i0 in fixed_map.keys())
        fixed_pos_dict = {bname: {"A": fixed_1idx}} if fixed_1idx else None

        # Omit Cys on designed positions so motif cysteines (e.g. C2H2 Zn-finger) are preserved while scaffold adds 0 extra Cys!
        omit_AAs_np = np.array([1.0 if a in ("C", "X") else 0.0 for a in alphabet], dtype=np.float32)
        # Mild anti-Ala (-0.35) and anti-Pro (-0.6 on motif scaffolds) so MPNN builds clean secondary structure instead of poly-Pro/Ala coils
        is_motif = spec["category"] == "motif_scaffolding"
        bias_AAs_np = np.array(
            [-0.35 if a == "A" else (-0.65 if (is_motif and a == "P") else (-0.25 if (is_motif and a == "G") else 0.0)) for a in alphabet],
            dtype=np.float32,
        )

        (
            X,
            S,
            mask,
            lengths,
            chain_M,
            chain_encoding_all,
            chain_list_list,
            visible_list_list,
            masked_list_list,
            masked_chain_length_list_list,
            chain_M_pos,
            omit_AA_mask,
            residue_idx,
            dihedral_mask,
            tied_pos_list_of_lists_list,
            pssm_coef,
            pssm_bias,
            pssm_log_odds_all,
            bias_by_res_all,
            tied_beta,
        ) = tied_featurize(pdb_dict_list, device, None, fixed_pos_dict, None, None, None, None, ca_only=False)

        # Compute Cα neighbor count (10 Å) so anti-parent novelty bias ONLY touches exposed surface residues (<14 nbrs),
        # leaving buried hydrophobic core & structural turns 100% free for optimal ProteinMPNN packing!
        ca_arr = np.asarray(ch_a["coords_ca"], dtype=np.float64)
        dmat_ca = np.linalg.norm(ca_arr[:, None, :] - ca_arr[None, :, :], axis=-1)
        nbr10 = np.sum((dmat_ca < 10.0) & (dmat_ca > 0.1), axis=1)

        best_seq = ""
        configs = (
            [(0.1, 0.0, 0.0), (0.12, 0.0, 0.0), (0.15, 0.0, 0.0)]
            if is_motif
            else [(0.1, -2.5, 0.0), (0.1, -4.0, -0.35), (0.12, -5.0, -0.6), (0.15, -6.0, -0.9)]
        )
        from fable_bench.stage3b_design import MotifSegmentSpec, evaluate_sequence_hygiene

        segs_hyg = [
            MotifSegmentSpec(f"m{k}", ms, 0, L, [[0.0, 0.0, 0.0]] * len(ms))
            for k, ms in enumerate(spec.get("fixed_motif_seqs", []))
        ]
        with torch.no_grad():
            for t_i, (temp, surf_pen, core_pen) in enumerate(configs):
                bias_res = bias_by_res_all.clone()
                if surf_pen < 0.0 or core_pen < 0.0:
                    for r0, orig_aa in enumerate(ch_a["sequence"]):
                        if r0 not in fixed_map and orig_aa in aa_to_idx and orig_aa not in ("G", "P"):
                            pen = surf_pen if nbr10[r0] < 14 else core_pen
                            bias_res[0, r0, aa_to_idx[orig_aa]] = pen
                g = torch.Generator(device=device)
                g.manual_seed(seed + t_i * 101)
                randn = torch.randn(chain_M.shape, generator=g, device=device)
                out_d = mpnn_model.sample(
                    X,
                    randn,
                    S,
                    chain_M,
                    chain_encoding_all,
                    residue_idx,
                    mask=mask,
                    temperature=temp,
                    omit_AAs_np=omit_AAs_np,
                    bias_AAs_np=bias_AAs_np,
                    chain_M_pos=chain_M_pos,
                    omit_AA_mask=omit_AA_mask,
                    pssm_coef=pssm_coef,
                    pssm_bias=pssm_bias,
                    pssm_multi=0.0,
                    pssm_log_odds_flag=False,
                    pssm_log_odds_mask=None,
                    pssm_bias_flag=False,
                    bias_by_res=bias_res,
                )
                cand = _S_to_seq(out_d["S"][0], chain_M[0])
                best_seq = cand
                hyg = evaluate_sequence_hygiene(cand, L, L, spec.get("max_allowed_cys", 1), segs_hyg)
                scaffold_only = "".join(c for i, c in enumerate(cand) if i not in fixed_map)
                max_sid = 0.0
                for pseq in spec.get("forbidden_parent_sequences", []):
                    if len(scaffold_only) >= 25:
                        _, _, sid = smith_waterman_affine(scaffold_only, pseq)
                        max_sid = max(max_sid, sid)
                if hyg.hygiene_pass and max_sid < 29.0:
                    break
        return best_seq
    finally:
        Path(tmp_path).unlink(missing_ok=True)


def run_task7_rfd3_mpnn(jf: JapanFoldClient) -> None:
    out_jsonl = REPO / "outputs" / "predictions" / "baseline_rfd3_mpnn_stage3b.jsonl"
    specs = json.loads((REPO / "outputs" / "japanfold_cache" / "t7_rfd3_specs.json").read_text(encoding="utf-8"))
    mpnn_model = load_proteinmpnn_model(torch.device("cpu"))

    print(f"\n=== [Task 7] Running RFdiffusion3 + ProteinMPNN on {len(specs)} Design Specifications ===", flush=True)

    def _run_one_rfd3(idx_sp: Tuple[int, Dict[str, Any]]) -> Tuple[int, Dict[str, Any], str, float]:
        idx, sp = idx_sp
        tid = sp["task_id"]
        pdb_input = Path(sp["pdb_path"]).read_text(encoding="utf-8")
        with JapanFoldClient(repo_root=REPO, max_usd_budget=25.0) as w_jf:
            meta, files = w_jf.submit_and_collect(
                "/v1/designs",
                {
                    "model": "rfd3",
                    "protocol": "rfd3-scaffold",
                    "name": f"t7_{tid}",
                    "structure": pdb_input,
                    "contig": sp["contig"],
                    "params": {"num_designs": 2, "num_timesteps": 200 if sp["category"] == "motif_scaffolding" else 100, "seed": 42 + idx},
                },
                cache_prefix=f"t7_rfd3_{tid}",
                download_all=True,
            )
            spent = w_jf.cumulative_usd
        designs = meta["results"].get("designs") or []
        # Sort by continuous 1-chain (`num_chains == 1`) then `final_rank`
        designs_sorted = sorted(
            designs,
            key=lambda d: (0 if d.get("num_chains", 1) == 1 else 1, int(d.get("final_rank") or 999)),
        )
        chosen_pdb = ""
        for d in designs_sorted:
            f_rel = str(d.get("file") or d.get("structure") or d.get("path") or "")
            if f_rel and f_rel in files:
                chosen_pdb = files[f_rel].decode("utf-8", errors="replace")
                break
            did = str(d.get("id") or d.get("name") or "")
            if did:
                for fk, fv in files.items():
                    if did in fk and fk.endswith(".pdb"):
                        chosen_pdb = fv.decode("utf-8", errors="replace")
                        break
            if chosen_pdb:
                break
        if not chosen_pdb:
            for fk, fv in sorted(files.items()):
                if fk.endswith(".pdb"):
                    chosen_pdb = fv.decode("utf-8", errors="replace")
                    break
        if not chosen_pdb:
            # Fallback to RFdiffusion3's own sequence head if only CIF returned
            cif_keys = [fk for fk in sorted(files.keys()) if fk.endswith(".cif")]
            if cif_keys:
                from build_and_evaluate_stage3b import parse_cif_all_protein_chains, write_backbone_pdb

                chs = parse_cif_all_protein_chains(files[cif_keys[0]].decode("utf-8", errors="replace"))
                ch0 = chs.get("A") or next(iter(chs.values()))
                with tempfile.NamedTemporaryFile("w", suffix=".pdb", delete=False) as tf2:
                    t2p = Path(tf2.name)
                write_backbone_pdb(ch0["sequence"], ch0["n"], ch0["ca"], ch0["c"], ch0["o"], t2p)
                chosen_pdb = t2p.read_text(encoding="utf-8")
                t2p.unlink(missing_ok=True)
        return idx, sp, chosen_pdb, spent

    rfd_results: List[Tuple[int, Dict[str, Any], str]] = []
    # JapanFold holds up to 2 jobs concurrently per account (`?wait=60` rule)
    with ThreadPoolExecutor(max_workers=2) as pool:
        futs = [pool.submit(_run_one_rfd3, (i, sp)) for i, sp in enumerate(specs)]
        for done_cnt, fut in enumerate(as_completed(futs), start=1):
            idx, sp, pdb_str, spent = fut.result()
            jf.cumulative_usd += spent
            rfd_results.append((idx, sp, pdb_str))
            if done_cnt % 5 == 0 or done_cnt == len(specs):
                print(f"  [RFdiffusion3] Completed {done_cnt}/{len(specs)} backbones (JF cum=${jf.cumulative_usd:.4f})", flush=True)

    rfd_results.sort(key=lambda x: x[0])
    rows_out: List[Dict[str, Any]] = []
    for idx, sp, pdb_str in rfd_results:
        tid = sp["task_id"]
        seq = _design_with_mpnn_on_pdb_text(mpnn_model, pdb_str, sp, seed=1000 + idx)
        rows_out.append(
            {
                "target_id": tid,
                "task": "stage3b_monomer_design",
                "text": json.dumps({"sequence": seq, "rationale": "RFdiffusion3 + ProteinMPNN v_48_020"}),
            }
        )

    out_jsonl.write_text("\n".join(json.dumps(r) for r in rows_out) + "\n", encoding="utf-8")
    print(f"Saved {len(rows_out)} RFdiffusion3 + ProteinMPNN designs -> {out_jsonl.name}", flush=True)
    evaluate_test7(jf)


def _extract_sample_num(d: Dict[str, Any], fallback_idx: int) -> Tuple[int, str]:
    raw_id = str(d.get("id") or d.get("design_id") or d.get("name") or f"idx_{fallback_idx}")
    nums = re.findall(r"\d+", raw_id)
    if nums:
        return int(nums[-1]), raw_id
    return fallback_idx, raw_id


def run_task8_boltzgen(jf: JapanFoldClient, shots: int = 16) -> None:
    pdir = REPO / "outputs" / "predictions"
    out_f = pdir / "stage3b_binder_campaign_specialist_boltzgen_af2.json"
    if out_f.exists():
        print(f"[BoltzGen] Reusing completed AF2 baseline -> {out_f.name}", flush=True)
        return
    m2 = json.loads((REPO / "data" / "stage3b_design" / "epitope_binder_manifest.json").read_text(encoding="utf-8"))
    targets = [EpitopeTarget(**e) for e in m2["targets"]]

    print(f"\n=== [Task 8] Running BoltzGen Specialist Baseline ({len(targets)} epitopes x {shots} designs) ===", flush=True)
    eval_items: List[Tuple[Any, str, int]] = []
    per_ep_seqs: Dict[str, List[str]] = {}

    def _run_one_bgen(ep: Any) -> Tuple[str, List[Dict[str, Any]], float, float]:
        hs = sorted(ep.epitope_hotspots_1idx)
        hs_span = f"{hs[0]}..{hs[-1]}"
        spec_yaml = (
            "entities:\n"
            "  - protein:\n"
            "      id: A\n"
            f"      sequence: {ep.target_sequence}\n"
            f"      binding_types: {{binding: '{hs_span}'}}\n"
            "  - protein:\n"
            "      id: B\n"
            f"      sequence: {ep.binder_min_len}..{ep.binder_max_len}\n"
        )
        with JapanFoldClient(repo_root=REPO, max_usd_budget=25.0) as w_jf:
            meta_d, _ = w_jf.submit_and_collect(
                "/v1/designs",
                {
                    "model": "boltzgen",
                    "protocol": "protein-anything",
                    "name": f"bgen_{ep.target_id}",
                    "spec": spec_yaml,
                    "params": {"num_designs": shots, "budget": shots, "fast": True, "seed": 42},
                },
                cache_prefix=f"bgen_{ep.target_id}",
                download_all=False,
            )
            spent = w_jf.cumulative_usd
        return ep.target_id, meta_d["results"].get("designs") or [], float(meta_d["cost_usd"]), spent

    designs_by_tid: Dict[str, List[Dict[str, Any]]] = {}
    with ThreadPoolExecutor(max_workers=2) as pool:
        futs = {pool.submit(_run_one_bgen, ep): ep for ep in targets}
        for fut in as_completed(futs):
            tid, designs, cost, spent = fut.result()
            jf.cumulative_usd += spent
            designs_by_tid[tid] = designs
            print(f"  [BoltzGen] {tid}: generated {len(designs)} designs (cost=${cost:.4f})", flush=True)

    for ep in targets:
        designs = designs_by_tid[ep.target_id]
        if designs and ep.target_id == targets[0].target_id:
            print(f"  [BoltzGen sample keys] {list(designs[0].keys())} id={designs[0].get('id')}", flush=True)
        # Order by raw sample index (not BoltzGen's post-filter rank!) so Shot 1 = unbiased single draw & Shot 16 = Best-of-16
        indexed_designs = sorted(enumerate(designs), key=lambda t: _extract_sample_num(t[1], t[0]))
        # Check if IDs were reassigned 1..16 in rank order; if monotonically identical to rank order, permute deterministically
        extracted_nums = [_extract_sample_num(d, i)[0] for i, d in enumerate(designs)]
        if extracted_nums == list(range(len(designs))) or extracted_nums == list(range(1, len(designs) + 1)):
            rng = random.Random(f"unbiased_draw_order_{ep.target_id}")
            indexed_designs = list(enumerate(designs))
            rng.shuffle(indexed_designs)

        seqs: List[str] = []
        for _i, d in indexed_designs:
            sq = str(d.get("designed_sequence") or d.get("binder_sequence") or d.get("sequence") or "").strip().upper()
            if sq and all(c in CANONICAL_AA_SET for c in sq):
                seqs.append(sq)
        while len(seqs) < shots and seqs:
            seqs.append(seqs[-1])
        per_ep_seqs[ep.target_id] = seqs[:shots]
        for r_idx, sq in enumerate(per_ep_seqs[ep.target_id], start=1):
            eval_items.append((ep, sq, r_idx))

    all_fbs = evaluate_binders_af2_batch(jf, eval_items)
    idx_cursor = 0
    camps = []
    trajs: Dict[str, List[Dict[str, Any]]] = {}
    for ep in targets:
        n_s = len(per_ep_seqs[ep.target_id])
        ep_fbs = all_fbs[idx_cursor : idx_cursor + n_s]
        idx_cursor += n_s
        camps.append(evaluate_binder_campaign(ep.target_id, ep_fbs, budget_k=shots))
        trajs[ep.target_id] = [
            {"round": fb.round_idx, "rationale": "BoltzGen all-atom binder diffusion", "feedback": asdict(fb)}
            for fb in ep_fbs
        ]

    hits = [c["hit_achieved"] for c in camps]
    shots_hit = [c["shots_to_first_hit"] for c in camps if c["shots_to_first_hit"] is not None]
    traj_mean = [round(float(x), 4) for x in np.mean([c["running_best_composite"] for c in camps], axis=0)]
    traj_sem = [round(float(x), 4) for x in (np.std([c["running_best_composite"] for c in camps], axis=0) / np.sqrt(len(camps)))]
    payload = {
        "model_tag": "specialist_boltzgen_closed_loop_af2",
        "summary": {
            "n_targets": len(camps),
            "hit_rate": round(float(np.mean(hits)), 4),
            "mean_shots_to_hit_among_hits": round(float(np.mean(shots_hit)), 2) if shots_hit else None,
            "mean_best_iptm": round(float(np.mean([c["best_iptm_on_epitope"] for c in camps])), 4),
            "mean_best_composite": traj_mean[-1],
            "mean_shot1_composite": traj_mean[0],
            "mean_closed_loop_gain": round(float(np.mean([c["closed_loop_trajectory_gain"] for c in camps])), 4),
            "pass_at_k_curve": [round(float(x), 4) for x in np.mean([c["pass_at_k_curve"] for c in camps], axis=0)],
            "trajectory_mean": traj_mean,
            "trajectory_sem": traj_sem,
            "per_target": camps,
        },
        "trajectories": trajs,
    }
    out_f.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(
        f"Saved BoltzGen AF2 baseline -> {out_f.name} | Hit@{shots}={payload['summary']['hit_rate']*100:.1f}% | "
        f"Composite {traj_mean[0]:.3f} -> {traj_mean[-1]:.3f} | Best On-Epitope ipTM={payload['summary']['mean_best_iptm']:.3f}",
        flush=True,
    )


def main() -> None:
    with JapanFoldClient(repo_root=REPO, max_usd_budget=25.0) as jf:
        bal = jf.get_account_balance()
        print(f"JapanFold starting balance: ${bal:.2f} USD", flush=True)
        evaluate_test2(jf)
        run_task7_rfd3_mpnn(jf)
        run_task8_boltzgen(jf, shots=16)
        print(f"\nAll JapanFold baseline runs complete! Session spend: ${jf.cumulative_usd:.4f} USD", flush=True)


if __name__ == "__main__":
    main()
