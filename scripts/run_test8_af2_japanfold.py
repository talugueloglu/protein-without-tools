#!/usr/bin/env python3
"""Synchronized 16-Shot Closed-Loop Epitope Binder Design (Test 8, N=12 targets)
scored by Official Google DeepMind AlphaFold2 (`af2ig`) on JapanFold.

Evaluates `gpt-6.2-astra`, `gpt-6.2-sol`, and `gpt-6.1-sol` (`reasoning_effort="high"`)
plus optional `BoltzGen` 16-shot specialist baseline under identical `AF2` (`af2ig`)
interface + hotspot + unbound monomer stability + `<25%` SeqID anti-memorization rules.
"""

from __future__ import annotations

import argparse
import asyncio
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

REPO = Path(__file__).resolve().parents[1]
if str(REPO / "src") not in sys.path:
    sys.path.insert(0, str(REPO / "src"))

from fable_bench.japanfold_client import (
    JapanFoldClient,
    find_pdb_for_target,
    format_clean_two_chain_pdb,
    parse_pdb_chain_with_bfactor,
)
from fable_bench.llm_client import UnifiedLLMRunner
from fable_bench.pdb_utils import CANONICAL_AA_SET
from fable_bench.stage3b_design import (
    BinderShotFeedback,
    EpitopeTarget,
    build_binder_initial_prompt,
    build_binder_refinement_prompt,
    evaluate_binder_campaign,
    kabsch_rmsd,
    parse_designed_sequence,
    score_3d_complex_geometry,
)
from fable_bench.tmscore import kabsch_superpose


def _binder_cache_key(target_id: str, binder_seq: str) -> str:
    return hashlib.sha1(f"af2v1:{target_id}:{binder_seq}".encode("utf-8")).hexdigest()[:22]


def _make_af2ig_yaml(two_chain_pdb: str, binder_seq: str) -> str:
    indented = "\n".join("    " + ln for ln in two_chain_pdb.splitlines() if ln.strip())
    return (
        "target:\n"
        "  structure: |\n"
        f"{indented}\n"
        "  chain: A\n"
        "binder:\n"
        f"  sequence: {binder_seq}\n"
        "  chain: B\n"
    )


def evaluate_binders_af2_batch(
    jf: JapanFoldClient,
    items: List[Tuple[EpitopeTarget, str, int]],
) -> List[BinderShotFeedback]:
    """Batch-score `(EpitopeTarget, binder_seq, round_idx)` via JapanFold `AF2-IG` (`AlphaFold2`)."""
    store_dir = REPO / "outputs" / "japanfold_cache" / "af2_binder_store"
    store_dir.mkdir(parents=True, exist_ok=True)

    geom_by_key: Dict[str, Dict[str, Any]] = {}
    todo_pairs: Dict[str, Tuple[EpitopeTarget, str]] = {}

    for ep, raw_seq, _rnd in items:
        b_seq = raw_seq.strip().upper()
        if not (25 <= len(b_seq) <= 160) or any(c not in CANONICAL_AA_SET for c in b_seq):
            continue
        ck = _binder_cache_key(ep.target_id, b_seq)
        cp = store_dir / f"{ck}.json"
        npz_p = store_dir / f"{ck}.npz"
        if cp.exists() and npz_p.exists():
            m = json.loads(cp.read_text(encoding="utf-8"))
            z = np.load(npz_p)
            m["pred_target_cb"] = z["pred_target_cb"]
            m["pred_binder_ca"] = z["pred_binder_ca"]
            m["pred_binder_cb"] = z["pred_binder_cb"]
            geom_by_key[ck] = m
        elif ck not in todo_pairs:
            todo_pairs[ck] = (ep, b_seq)

    if todo_pairs:
        ordered_keys = sorted(todo_pairs.keys())
        print(f"    [AF2 Oracle] Scoring {len(ordered_keys)} uncached complex candidates on JapanFold...", flush=True)

        # Process in chunks of up to 36 targets per job
        for c_start in range(0, len(ordered_keys), 36):
            sub_keys = ordered_keys[c_start : c_start + 36]

            # Step 1: Initial 3D complex coordinates (`Chain A` + `Chain B`) via Boltz-2
            boltz_targets = []
            for idx0, ck in enumerate(sub_keys):
                ep, b_seq = todo_pairs[ck]
                tname = f"c{idx0 + 1:03d}"
                boltz_targets.append({"name": tname, "content": f">A\n{ep.target_sequence}\n>B\n{b_seq}\n"})

            meta_b, files_b = jf.submit_and_collect(
                "/v1/predictions",
                {
                    "model": "boltz2",
                    "name": f"t8_init_complex_{c_start}",
                    "targets": boltz_targets,
                    "params": {"output_format": "pdb"},
                },
                cache_prefix="t8_boltz2_init",
            )

            # Build clean two-chain PDB inputs for Official AlphaFold2 (`af2ig`)
            af2_targets = []
            mono_targets = []
            valid_sub: List[Tuple[int, str]] = []
            for idx0, ck in enumerate(sub_keys):
                ep, b_seq = todo_pairs[ck]
                tname = f"c{idx0 + 1:03d}"
                pdb_b = find_pdb_for_target(files_b, tname, idx0)
                if not pdb_b:
                    continue
                try:
                    ch_a0 = parse_pdb_chain_with_bfactor(pdb_b, "A")
                    ch_b0 = parse_pdb_chain_with_bfactor(pdb_b, "B")
                except Exception:
                    continue
                if len(ch_b0["sequence"]) != len(b_seq):
                    continue
                clean_pdb = format_clean_two_chain_pdb(ch_a0, ch_b0)
                aname = f"a{len(valid_sub) + 1:03d}"
                af2_targets.append({"name": aname, "content": _make_af2ig_yaml(clean_pdb, b_seq)})
                mono_targets.append({"name": aname, "content": f">{aname}\n{b_seq}"})
                valid_sub.append((len(valid_sub), ck))

            if not af2_targets:
                continue

            # Step 2: Official Google DeepMind AlphaFold2 (`af2ig`) Complex Re-Prediction & Interface Scoring
            meta_af, files_af = jf.submit_and_collect(
                "/v1/predictions",
                {
                    "model": "af2ig",
                    "name": f"t8_af2ig_{c_start}",
                    "targets": af2_targets,
                    "params": {"output_format": "pdb"},
                },
                cache_prefix="t8_af2ig_score",
            )
            af_rows = meta_af["results"].get("rows") or []
            af_row_by_id = {str(r.get("id")): r for r in af_rows}

            # Step 3: Unbound Monomer Fold (`ESMFold-2` MSA-free) for `monomer_RMSD`
            meta_m, files_m = jf.submit_and_collect(
                "/v1/predictions",
                {
                    "model": "esmfold2",
                    "name": f"t8_mono_{c_start}",
                    "targets": mono_targets,
                    "params": {"fast": False, "use_msa_server": False, "output_format": "pdb"},
                },
                cache_prefix="t8_esm2_mono",
            )

            for v_idx, ck in valid_sub:
                ep, b_seq = todo_pairs[ck]
                aname = f"a{v_idx + 1:03d}"
                af_pdb = find_pdb_for_target(files_af, aname, v_idx)
                mono_pdb = find_pdb_for_target(files_m, aname, v_idx)
                row = af_row_by_id.get(aname) or af_row_by_id.get(f"target_{v_idx + 1}") or (af_rows[v_idx] if v_idx < len(af_rows) else {})
                if not af_pdb:
                    continue
                try:
                    af_a = parse_pdb_chain_with_bfactor(af_pdb, "A")
                    af_b = parse_pdb_chain_with_bfactor(af_pdb, "B")
                except Exception:
                    continue

                # Rigid-body Kabsch superposition of AF2 complex onto crystal target Cα frame!
                t_ca_gt = np.asarray(ep.target_ca_coords, dtype=np.float64)
                if len(af_a["coords_ca"]) == len(t_ca_gt):
                    rot, trans, _ = kabsch_superpose(af_a["coords_ca"], t_ca_gt)
                    pred_t_cb = af_a["coords_cb"] @ rot + trans
                    pred_b_ca = af_b["coords_ca"] @ rot + trans
                    pred_b_cb = af_b["coords_cb"] @ rot + trans
                else:
                    pred_t_cb = af_a["coords_cb"]
                    pred_b_ca = af_b["coords_ca"]
                    pred_b_cb = af_b["coords_cb"]

                iptm = float(row.get("iptm") or 0.0)
                raw_ipae = row.get("interface_pae")
                if raw_ipae is None:
                    raw_ipae = row.get("ipae")
                if raw_ipae is None:
                    raw_ipae = row.get("pae")
                ipae_val = float(raw_ipae) if raw_ipae is not None else 28.0
                # AF2-IG reports interface_pae directly in Angstroms (per JapanFold guide)
                if 0.0 < ipae_val <= 1.2 and row.get("interface_pae") is None:
                    ipae_val *= 31.0

                b_plddt = float(af_b["plddt_mean"])
                mono_rmsd = 9.99
                if mono_pdb:
                    try:
                        m_ch = parse_pdb_chain_with_bfactor(mono_pdb, "A")
                        if len(m_ch["coords_ca"]) == len(af_b["coords_ca"]):
                            mono_rmsd = float(kabsch_rmsd(af_b["coords_ca"], m_ch["coords_ca"]))
                            # Use max/consistent binder pLDDT or AF2 bound pLDDT
                    except Exception:
                        pass

                meta_item = {
                    "iptm": round(iptm, 4),
                    "ipae": round(ipae_val, 2),
                    "binder_plddt": round(b_plddt, 2),
                    "monomer_rmsd": round(mono_rmsd, 3),
                    "af2_row": row,
                }
                (store_dir / f"{ck}.json").write_text(json.dumps(meta_item), encoding="utf-8")
                np.savez_compressed(
                    store_dir / f"{ck}.npz",
                    pred_target_cb=pred_t_cb,
                    pred_binder_ca=pred_b_ca,
                    pred_binder_cb=pred_b_cb,
                )
                m_full = dict(meta_item)
                m_full["pred_target_cb"] = pred_t_cb
                m_full["pred_binder_ca"] = pred_b_ca
                m_full["pred_binder_cb"] = pred_b_cb
                geom_by_key[ck] = m_full

            print(
                f"    [AF2 Oracle] Batch done: Boltz2=${meta_b['cost_usd']:.4f} + AF2-IG=${meta_af['cost_usd']:.4f} "
                f"+ Mono=${meta_m['cost_usd']:.4f} (JF cum=${jf.cumulative_usd:.4f})",
                flush=True,
            )

    # Assemble BinderShotFeedback for each input item in exact order
    feedbacks: List[BinderShotFeedback] = []
    for ep, raw_seq, rnd in items:
        b_seq = raw_seq.strip().upper()
        ck = _binder_cache_key(ep.target_id, b_seq) if (25 <= len(b_seq) <= 160 and all(c in CANONICAL_AA_SET for c in b_seq)) else ""
        g = geom_by_key.get(ck) if ck else None
        if g is None:
            feedbacks.append(
                BinderShotFeedback(
                    round_idx=rnd,
                    binder_sequence=b_seq,
                    hygiene_ok=False,
                    novelty_ok=True,
                    max_forbidden_seqid_pct=0.0,
                    iptm=0.05,
                    ipae=28.0,
                    binder_plddt=30.0,
                    monomer_rmsd=9.99,
                    hotspots_contacted_1idx=[],
                    hotspots_missed_1idx=list(ep.epitope_hotspots_1idx),
                    clash_count=0,
                    composite_score=0.02,
                    is_hit=False,
                    diagnostic="Failed to parse valid canonical binder sequence or fold failed.",
                )
            )
        else:
            feedbacks.append(
                score_3d_complex_geometry(
                    target=ep,
                    binder_seq=b_seq,
                    pred_target_cb=g["pred_target_cb"],
                    pred_binder_ca=g["pred_binder_ca"],
                    pred_binder_cb=g["pred_binder_cb"],
                    iptm=g["iptm"],
                    ipae=g["ipae"],
                    binder_plddt=g["binder_plddt"],
                    monomer_rmsd=g["monomer_rmsd"],
                    round_idx=rnd,
                )
            )
    return feedbacks


def run_boltzgen_specialist_baseline(
    jf: JapanFoldClient,
    targets: List[EpitopeTarget],
    shots: int,
    pdir: Path,
) -> None:
    """Run BoltzGen (`protein-anything`) specialist baseline on all 12 epitopes and grade with the exact same AF2 oracle."""
    out_f = pdir / "stage3b_binder_campaign_specialist_boltzgen_af2.json"
    if out_f.exists():
        print(f"Reusing cached BoltzGen AF2 campaign -> {out_f.name}", flush=True)
        return

    print(f"\n=== Running BoltzGen Specialist Baseline ({len(targets)} epitopes x {shots} designs) ===", flush=True)
    eval_items: List[Tuple[EpitopeTarget, str, int]] = []
    per_ep_seqs: Dict[str, List[str]] = {}

    for ep in targets:
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
        meta_d, _files_d = jf.submit_and_collect(
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
        designs = meta_d["results"].get("designs") or []
        seqs: List[str] = []
        for d in designs:
            sq = str(d.get("designed_sequence") or d.get("binder_sequence") or d.get("sequence") or "").strip().upper()
            if sq:
                seqs.append(sq)
        while len(seqs) < shots and seqs:
            seqs.append(seqs[-1])
        per_ep_seqs[ep.target_id] = seqs[:shots]
        for r_idx, sq in enumerate(per_ep_seqs[ep.target_id], start=1):
            eval_items.append((ep, sq, r_idx))
        print(f"  [BoltzGen] {ep.target_id}: generated {len(designs)} designs (cost=${meta_d['cost_usd']:.4f})", flush=True)

    all_fbs = evaluate_binders_af2_batch(jf, eval_items)
    idx_cursor = 0
    camps = []
    trajs: Dict[str, List[Dict[str, Any]]] = {}
    for ep in targets:
        n_s = len(per_ep_seqs[ep.target_id])
        ep_fbs = all_fbs[idx_cursor : idx_cursor + n_s]
        idx_cursor += n_s
        camps.append(evaluate_binder_campaign(ep.target_id, ep_fbs, budget_k=shots))
        trajs[ep.target_id] = [{"round": fb.round_idx, "rationale": "BoltzGen all-atom diffusion", "feedback": asdict(fb)} for fb in ep_fbs]

    hits = [c["hit_achieved"] for c in camps]
    shots_hit = [c["shots_to_first_hit"] for c in camps if c["shots_to_first_hit"] is not None]
    payload = {
        "model_tag": "specialist_boltzgen_open_loop_af2",
        "summary": {
            "n_targets": len(camps),
            "hit_rate": round(float(np.mean(hits)), 4),
            "mean_shots_to_hit_among_hits": round(float(np.mean(shots_hit)), 2) if shots_hit else None,
            "mean_best_iptm": round(float(np.mean([c["best_iptm_on_epitope"] for c in camps])), 4),
            "mean_closed_loop_gain": round(float(np.mean([c["closed_loop_trajectory_gain"] for c in camps])), 4),
            "pass_at_k_curve": [round(float(x), 4) for x in np.mean([c["pass_at_k_curve"] for c in camps], axis=0)],
            "trajectory_mean": [round(float(x), 4) for x in np.mean([c["running_best_composite"] for c in camps], axis=0)],
            "trajectory_sem": [round(float(x), 4) for x in (np.std([c["running_best_composite"] for c in camps], axis=0) / np.sqrt(len(camps)))],
            "per_target": camps,
        },
        "trajectories": trajs,
    }
    out_f.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Saved BoltzGen AF2 specialist baseline -> {out_f.name} (Hit@{shots}={payload['summary']['hit_rate']*100:.1f}%)", flush=True)


async def run_synchronized_closed_loop(
    models: List[str],
    targets: List[EpitopeTarget],
    shots: int,
    thinking_budget: int,
    max_openai_usd: float,
    run_boltzgen: bool,
) -> None:
    pdir = REPO / "outputs" / "predictions"
    pdir.mkdir(parents=True, exist_ok=True)

    runner = UnifiedLLMRunner(max_usd_budget=max_openai_usd, assert_no_fallback=True)
    sem_llm = asyncio.Semaphore(18)

    # State: histories[model][target_id] -> List[BinderShotFeedback]
    histories: Dict[str, Dict[str, List[BinderShotFeedback]]] = {
        m: {ep.target_id: [] for ep in targets} for m in models
    }
    ep_logs: Dict[str, Dict[str, List[Dict[str, Any]]]] = {
        m: {ep.target_id: [] for ep in targets} for m in models
    }

    with JapanFoldClient(repo_root=REPO, max_usd_budget=40.0) as jf:
        for shot in range(1, shots + 1):
            print(f"\n================ ROUND {shot:2d} / {shots} ({len(models)} models x {len(targets)} epitopes) ================", flush=True)

            async def _propose_one(m_name: str, ep_obj: EpitopeTarget) -> Tuple[str, EpitopeTarget, str, str, Dict[str, Any]]:
                hist = histories[m_name][ep_obj.target_id]
                prompt = build_binder_refinement_prompt(ep_obj, hist, total_budget=shots) if hist else build_binder_initial_prompt(ep_obj)
                async with sem_llm:
                    rec = await runner.complete(
                        provider="openai",
                        model=m_name,
                        prompt=prompt,
                        max_tokens=1536,
                        thinking_budget=thinking_budget,
                        temperature=0.0,
                    )
                b_seq, rat, _ok = parse_designed_sequence(rec.text)
                return m_name, ep_obj, b_seq, rat, rec.to_dict()

            tasks = [_propose_one(m, ep) for m in models for ep in targets]
            proposals = await asyncio.gather(*tasks)

            eval_items = [(ep_obj, b_seq, shot) for (_m, ep_obj, b_seq, _rat, _meta) in proposals]
            feedbacks = evaluate_binders_af2_batch(jf, eval_items)

            for (m_name, ep_obj, _b_seq, rat, llm_meta), fb in zip(proposals, feedbacks):
                histories[m_name][ep_obj.target_id].append(fb)
                ep_logs[m_name][ep_obj.target_id].append(
                    {"round": shot, "rationale": rat, "feedback": asdict(fb), "llm_meta": llm_meta}
                )

            # Print concise per-model progress after each round & save partial checkpoint!
            for m_name in models:
                r_fbs = [histories[m_name][ep.target_id][-1] for ep in targets]
                cum_hits = sum(1 for ep in targets if any(x.is_hit for x in histories[m_name][ep.target_id]))
                mean_iptm = float(np.mean([x.iptm for x in r_fbs]))
                mean_comp = float(np.mean([max(x.composite_score for x in histories[m_name][ep.target_id]) for ep in targets]))
                print(
                    f"  [{m_name:<15s}] Round {shot:2d}: CumHits={cum_hits:2d}/{len(targets)} | "
                    f"Round_ipTM={mean_iptm:.3f} | RunningBestScore={mean_comp:.3f} | OpenAI=${runner.cumulative_live_usd:.2f}",
                    flush=True,
                )

        # Save official campaign files for each model
        for m_name in models:
            camps = [evaluate_binder_campaign(ep.target_id, histories[m_name][ep.target_id], budget_k=shots) for ep in targets]
            hits = [c["hit_achieved"] for c in camps]
            shots_hit = [c["shots_to_first_hit"] for c in camps if c["shots_to_first_hit"] is not None]
            tag = f"{m_name}_think{thinking_budget}_closed_loop_af2"
            if len(targets) < 12 or shots < 16:
                tag += f"_p{len(targets)}s{shots}"
            payload = {
                "model_tag": tag,
                "summary": {
                    "n_targets": len(camps),
                    "hit_rate": round(float(np.mean(hits)), 4),
                    "mean_shots_to_hit_among_hits": round(float(np.mean(shots_hit)), 2) if shots_hit else None,
                    "mean_best_iptm": round(float(np.mean([c["best_iptm_on_epitope"] for c in camps])), 4),
                    "mean_closed_loop_gain": round(float(np.mean([c["closed_loop_trajectory_gain"] for c in camps])), 4),
                    "pass_at_k_curve": [round(float(x), 4) for x in np.mean([c["pass_at_k_curve"] for c in camps], axis=0)],
                    "trajectory_mean": [round(float(x), 4) for x in np.mean([c["running_best_composite"] for c in camps], axis=0)],
                    "trajectory_sem": [round(float(x), 4) for x in (np.std([c["running_best_composite"] for c in camps], axis=0) / np.sqrt(len(camps)))],
                    "per_target": camps,
                },
                "trajectories": ep_logs[m_name],
            }
            out_f = pdir / f"stage3b_binder_campaign_{tag}.json"
            out_f.write_text(json.dumps(payload, indent=2), encoding="utf-8")
            print(
                f"\nSaved {out_f.name}: Hit@{shots}={payload['summary']['hit_rate']*100:.1f}%, "
                f"Best_ipTM={payload['summary']['mean_best_iptm']:.3f}, Gain=+{payload['summary']['mean_closed_loop_gain']:.3f}",
                flush=True,
            )

        if run_boltzgen:
            run_boltzgen_specialist_baseline(jf, targets, shots, pdir)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Test 8 Closed-Loop Epitope Binder Design with Official AF2 (`af2ig`).")
    parser.add_argument("--models", default="gpt-6.2-sol,gpt-6.1-sol,gpt-6.2-astra")
    parser.add_argument("--shots", type=int, default=16)
    parser.add_argument("--thinking-budget", type=int, default=4096)
    parser.add_argument("--max-openai-usd", type=float, default=85.0)
    parser.add_argument("--pilot", type=int, default=None, help="Optional limit N epitopes for smoke testing")
    parser.add_argument("--boltzgen", action="store_true", help="Also evaluate 16-design BoltzGen specialist baseline via AF2")
    args = parser.parse_args()

    sdir = REPO / "data" / "stage3b_design"
    m2 = json.loads((sdir / "epitope_binder_manifest.json").read_text(encoding="utf-8"))
    targets = [EpitopeTarget(**e) for e in m2["targets"]]
    if args.pilot is not None:
        targets = targets[: args.pilot]
    models = [m.strip() for m in args.models.split(",") if m.strip()]

    asyncio.run(
        run_synchronized_closed_loop(
            models=models,
            targets=targets,
            shots=args.shots,
            thinking_budget=args.thinking_budget,
            max_openai_usd=args.max_openai_usd,
            run_boltzgen=args.boltzgen,
        )
    )


if __name__ == "__main__":
    main()
