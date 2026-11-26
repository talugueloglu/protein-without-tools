#!/usr/bin/env python3
"""Build Task 9: Double-Mutant Epistasis Benchmark (N = 60 balanced panels).

Reads Tsuboyama et al. MegaScale Dataset 3 (`data/raw_v2_cache/task9_epistasis/dataset3.parquet`)
and MegaScale 3D PDB structures (`data/raw_alanine_cache/megascale_3d_pdbs.parquet`), selects
strictly cluster-non-redundant domains (all qualifying de novo Rocklin designs + diverse natural
domains), builds 16-pair panels matched on sequence separation |i - j| and single-mutant
magnitude, exports `data/task9_epistasis/manifest.json` + zero-leakage prompts
`outputs/prompts/task9_epistasis_prompts.jsonl`, and audits all anti-cheat null baselines.
"""

from __future__ import annotations

from collections import defaultdict
import json
from pathlib import Path
import sys
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pyarrow.parquet as pq

REPO = Path(__file__).resolve().parents[1]
if str(REPO / "src") not in sys.path:
    sys.path.insert(0, str(REPO / "src"))

from protein_without_tools.task9_epistasis import (
    MUT_TOKEN_RE,
    DoubleMutantRecord,
    build_epistasis_prompt,
    canonicalize_double_mutant,
    grade_epistasis_panel,
    select_balanced_epistasis_panel,
)


def _load_megascale_ca_coords(pdb_parquet: Path) -> Dict[str, np.ndarray]:
    """Extract C-alpha coordinate arrays keyed by WT_name (e.g. 'HHH_rd1_0493.pdb')."""
    if not pdb_parquet.exists():
        return {}
    t = pq.read_table(pdb_parquet)
    cols = t.column_names
    name_col = next((c for c in ("WT_name", "name", "pdb_name", "filename") if c in cols), cols[0])
    text_col = next((c for c in ("pdb_text", "pdb", "content", "structure") if c in cols), cols[1])
    out: Dict[str, np.ndarray] = {}
    for nm, txt in zip(t[name_col].to_pylist(), t[text_col].to_pylist()):
        if not nm or not txt:
            continue
        coords: List[List[float]] = []
        seen_res = set()
        for ln in str(txt).splitlines():
            if ln.startswith("ATOM  ") and ln[12:16].strip() == "CA":
                res_key = (ln[21], ln[22:27])
                if res_key in seen_res:
                    continue
                seen_res.add(res_key)
                try:
                    coords.append([float(ln[30:38]), float(ln[38:46]), float(ln[46:54])])
                except ValueError:
                    pass
        if coords:
            key = Path(str(nm)).name
            out[key] = np.array(coords, dtype=float)
    return out


def _is_denovo_wt(wt_name: str) -> bool:
    return not (len(wt_name) == 8 and wt_name.endswith(".pdb") and wt_name[0].isdigit())


def main() -> None:
    ds3_path = REPO / "data/raw_v2_cache/task9_epistasis/dataset3.parquet"
    if not ds3_path.exists():
        raise FileNotFoundError(f"Missing {ds3_path}. Download RosettaCommons/MegaScale dataset3 first.")

    print("Loading MegaScale 3D PDB coordinates...", flush=True)
    ca_map = _load_megascale_ca_coords(REPO / "data/raw_alanine_cache/megascale_3d_pdbs.parquet")
    print(f"  Loaded C-alpha structures for {len(ca_map)} MegaScale domains.", flush=True)

    print("Reading MegaScale dataset3.parquet...", flush=True)
    tbl = pq.read_table(
        ds3_path,
        columns=["WT_name", "WT_cluster", "mut_type", "aa_seq", "dG_ML", "ddG_ML"],
    )
    wts = tbl["WT_name"].to_pylist()
    clusters = tbl["WT_cluster"].to_pylist()
    muts = tbl["mut_type"].to_pylist()
    seqs = tbl["aa_seq"].to_pylist()
    dgs = tbl["dG_ML"].to_pylist()
    ddgs = tbl["ddG_ML"].to_pylist()

    wt_seq: Dict[str, str] = {}
    wt_cluster: Dict[str, str] = {}
    wt_dg_acc: Dict[str, List[float]] = defaultdict(list)
    # Store (destab_ddG, abs_dG_ML) per (wt, single_mut) and (wt, double_mut)
    singles_acc: Dict[str, Dict[str, List[Tuple[float, float]]]] = defaultdict(lambda: defaultdict(list))
    doubles_acc: Dict[str, Dict[str, List[Tuple[float, float]]]] = defaultdict(lambda: defaultdict(list))

    for w, cl, m, sq, g, d in zip(wts, clusters, muts, seqs, dgs, ddgs):
        if not w or not m:
            continue
        if w not in wt_cluster and cl is not None:
            wt_cluster[w] = str(cl)
        if m == "wt":
            if sq and w not in wt_seq:
                wt_seq[w] = str(sq).strip().upper()
            if g not in (None, "-"):
                try:
                    wt_dg_acc[w].append(float(g))
                except ValueError:
                    pass
            continue
        if d in (None, "-") or g in (None, "-"):
            continue
        try:
            destab = -float(d)  # positive = destabilizing (dG_WT - dG_mut)
            dg_val = float(g)
        except ValueError:
            continue
        if ":" in m:
            parts = m.split(":")
            if len(parts) != 2:
                continue
            try:
                canon, _, _, _, _ = canonicalize_double_mutant(parts[0], parts[1])
            except ValueError:
                continue
            doubles_acc[w][canon].append((destab, dg_val))
        elif MUT_TOKEN_RE.match(m):
            singles_acc[w][m].append((destab, dg_val))

    # Infer WT sequence from single-mutant aa_seq if `wt` row was missing
    for w, m, sq in zip(wts, muts, seqs):
        if w in doubles_acc and w not in wt_seq and m and ":" not in m and sq and MUT_TOKEN_RE.match(m):
            mm = MUT_TOKEN_RE.match(m)
            pos = int(mm.group(2))
            if 1 <= pos <= len(sq):
                arr = list(str(sq).strip().upper())
                arr[pos - 1] = mm.group(1)
                wt_seq[w] = "".join(arr)

    print(f"Domains with double mutants: {len(doubles_acc)} | with WT sequences: {sum(1 for w in doubles_acc if w in wt_seq)}")

    # Build candidate panels per domain
    domain_panels: List[Tuple[bool, str, str, str, List[DoubleMutantRecord], float]] = []
    for w, dbl_dict in sorted(doubles_acc.items()):
        seq = wt_seq.get(w)
        if not seq or not (35 <= len(seq) <= 120):
            continue
        s_dict = singles_acc.get(w, {})
        ca = ca_map.get(w)
        if ca is not None and len(ca) != len(seq):
            ca = None

        cands: List[DoubleMutantRecord] = []
        for canon, vals_ab in dbl_dict.items():
            ma, mb = canon.split("+")
            if ma not in s_dict or mb not in s_dict:
                continue
            m1, m2 = MUT_TOKEN_RE.match(ma), MUT_TOKEN_RE.match(mb)
            if not m1 or not m2:
                continue
            p1, p2 = int(m1.group(2)), int(m2.group(2))
            if not (1 <= p1 <= len(seq) and 1 <= p2 <= len(seq)):
                continue
            # Sanity-check wild-type letters match WT sequence
            if seq[p1 - 1] != m1.group(1) or seq[p2 - 1] != m2.group(1):
                continue
            ddg_a = float(np.mean([x[0] for x in s_dict[ma]]))
            dg_a = float(np.mean([x[1] for x in s_dict[ma]]))
            ddg_b = float(np.mean([x[0] for x in s_dict[mb]]))
            dg_b = float(np.mean([x[1] for x in s_dict[mb]]))
            ddg_ab = float(np.mean([x[0] for x in vals_ab]))
            dg_ab = float(np.mean([x[1] for x in vals_ab]))
            # Dynamic-range filter (Tsuboyama et al. Fig 4): exclude assay-floor / ceiling
            # clamped mutations so non-additivity reflects true 3D energetic coupling, not unfolding-floor saturation.
            dg_wt_est = dg_a + ddg_a
            dg_add_pred = dg_wt_est - (ddg_a + ddg_b)
            if min(dg_a, dg_b, dg_ab) < 0.75 or max(dg_a, dg_b, dg_ab) > 4.75 or dg_add_pred < 0.50:
                continue
            dddg = ddg_ab - (ddg_a + ddg_b)
            ca_d = float(np.linalg.norm(ca[p1 - 1] - ca[p2 - 1])) if ca is not None else None
            cands.append(
                DoubleMutantRecord(
                    pair_id=canon,
                    mut_a=ma,
                    mut_b=mb,
                    pos_a=p1,
                    pos_b=p2,
                    seq_sep=abs(p2 - p1),
                    ddg_a=round(ddg_a, 4),
                    ddg_b=round(ddg_b, 4),
                    ddg_ab=round(ddg_ab, 4),
                    dddg=round(dddg, 4),
                    abs_dddg=round(abs(dddg), 4),
                    ca_dist=round(ca_d, 2) if ca_d is not None else None,
                )
            )

        panel = select_balanced_epistasis_panel(cands, n_pairs=16, coupled_thresh=0.80, additive_max=0.25, min_seq_sep=4)
        if panel is None:
            continue
        # Score panel quality by anti-cheat neutrality (seq-sep AUROC close to 0.50 & single-mag AUROC close to 0.50)
        g_sep = grade_epistasis_panel(panel, [], predicted_scores={r.pair_id: -float(r.seq_sep) for r in panel})
        g_mag = grade_epistasis_panel(panel, [], predicted_scores={r.pair_id: abs(r.ddg_a) + abs(r.ddg_b) for r in panel})
        # Also prefer having 3D structure available + both positive & negative epistasis
        n_pos = sum(1 for r in panel if r.is_coupled and r.dddg < 0)
        n_neg = sum(1 for r in panel if r.is_coupled and r.dddg > 0)
        sign_bal = min(n_pos, n_neg)
        anti_cheat_err = abs(g_sep["coupling_auroc"] - 0.5) + abs(g_mag["coupling_auroc"] - 0.5)
        quality = (1.0 if ca is not None else 0.0) + 0.15 * sign_bal - 2.0 * anti_cheat_err
        domain_panels.append((_is_denovo_wt(w), wt_cluster.get(w, w), w, seq, panel, quality))

    print(f"Domains yielding valid balanced 16-pair panels: {len(domain_panels)}")
    print(f"  De novo panels: {sum(1 for x in domain_panels if x[0])} | Natural panels: {sum(1 for x in domain_panels if not x[0])}")

    # Select up to N=60 cluster-unique domains: all de novo first, then best-balanced natural clusters
    denovo_sel = sorted([x for x in domain_panels if x[0]], key=lambda x: -x[5])
    nat_sorted = sorted([x for x in domain_panels if not x[0]], key=lambda x: -x[5])

    selected: List[Tuple[bool, str, str, str, List[DoubleMutantRecord], float]] = []
    seen_clusters: set[str] = set()
    for item in denovo_sel:
        selected.append(item)
        seen_clusters.add(item[1])
    for item in nat_sorted:
        if len(selected) >= 60:
            break
        if item[1] in seen_clusters:
            continue
        selected.append(item)
        seen_clusters.add(item[1])

    # If still < 60 after strict cluster dedup, allow additional distinct domains
    if len(selected) < 60:
        seen_wts = {x[2] for x in selected}
        for item in nat_sorted:
            if len(selected) >= 60:
                break
            if item[2] not in seen_wts:
                selected.append(item)
                seen_wts.add(item[2])

    out_dir = REPO / "data/task9_epistasis"
    out_dir.mkdir(parents=True, exist_ok=True)
    prompts_dir = REPO / "outputs/prompts"
    prompts_dir.mkdir(parents=True, exist_ok=True)

    manifest_targets: List[Dict[str, Any]] = []
    prompt_rows: List[Dict[str, Any]] = []

    null_add_auroc, null_sep_auroc, null_mag_auroc, oracle_3d_auroc = [], [], [], []
    null_add_rho, oracle_3d_rho = [], []

    for idx, (is_dn, cl, wt_name, seq, panel, _qual) in enumerate(selected, start=1):
        tid = f"t9_e{idx:03d}"
        stratum = "megascale_denovo" if is_dn else "megascale_natural"
        prompt_text = build_epistasis_prompt(seq, panel)

        # Anti-cheat audits
        g_add = grade_epistasis_panel(panel, [], predicted_scores={r.pair_id: 0.0 for r in panel})
        g_sep = grade_epistasis_panel(panel, [], predicted_scores={r.pair_id: -float(r.seq_sep) for r in panel})
        g_mag = grade_epistasis_panel(panel, [], predicted_scores={r.pair_id: abs(r.ddg_a) + abs(r.ddg_b) for r in panel})
        null_add_auroc.append(g_add["coupling_auroc"])
        null_add_rho.append(g_add["spearman_rho_abs_dddg"])
        null_sep_auroc.append(g_sep["coupling_auroc"])
        null_mag_auroc.append(g_mag["coupling_auroc"])
        if all(r.ca_dist is not None for r in panel):
            g_3d = grade_epistasis_panel(panel, [], predicted_scores={r.pair_id: -float(r.ca_dist) for r in panel})
            oracle_3d_auroc.append(g_3d["coupling_auroc"])
            oracle_3d_rho.append(g_3d["spearman_rho_abs_dddg"])

        manifest_targets.append({
            "target_id": tid,
            "wt_name": wt_name,
            "wt_cluster": cl,
            "stratum": stratum,
            "is_denovo": is_dn,
            "length": len(seq),
            "sequence": seq,
            "n_pairs": len(panel),
            "n_coupled": sum(1 for r in panel if r.is_coupled),
            "n_rescuing": sum(1 for r in panel if r.is_coupled and r.dddg < 0),
            "n_synergistic": sum(1 for r in panel if r.is_coupled and r.dddg > 0),
            "has_3d_ca": all(r.ca_dist is not None for r in panel),
            "pairs": [
                {
                    "pair_id": r.pair_id,
                    "mut_a": r.mut_a,
                    "mut_b": r.mut_b,
                    "pos_a": r.pos_a,
                    "pos_b": r.pos_b,
                    "seq_sep": r.seq_sep,
                    "ddg_a": r.ddg_a,
                    "ddg_b": r.ddg_b,
                    "ddg_ab": r.ddg_ab,
                    "dddg": r.dddg,
                    "abs_dddg": r.abs_dddg,
                    "ca_dist": r.ca_dist,
                    "is_coupled": r.is_coupled,
                }
                for r in panel
            ],
        })
        prompt_rows.append({
            "custom_id": f"task9_{tid}",
            "target_id": tid,
            "task": "task9_double_mutant_epistasis",
            "stratum": stratum,
            "length": len(seq),
            "n_pairs": len(panel),
            "prompt": prompt_text,
        })

    manifest = {
        "benchmark": "protein-without-tools-task9-double-mutant-epistasis",
        "n_targets": len(manifest_targets),
        "n_denovo": sum(1 for t in manifest_targets if t["is_denovo"]),
        "n_natural": sum(1 for t in manifest_targets if not t["is_denovo"]),
        "n_unique_clusters": len({t["wt_cluster"] for t in manifest_targets}),
        "anti_cheat_audit": {
            "additive_null_mean_auroc": round(float(np.mean(null_add_auroc)), 4),
            "additive_null_mean_rho": round(float(np.mean(null_add_rho)), 4),
            "seq_sep_1d_mean_auroc": round(float(np.mean(null_sep_auroc)), 4),
            "single_mag_1d_mean_auroc": round(float(np.mean(null_mag_auroc)), 4),
            "native_3d_ca_proximity_mean_auroc": round(float(np.mean(oracle_3d_auroc)), 4) if oracle_3d_auroc else None,
            "native_3d_ca_proximity_mean_rho": round(float(np.mean(oracle_3d_rho)), 4) if oracle_3d_rho else None,
            "n_with_3d_ca": len(oracle_3d_auroc),
        },
        "targets": manifest_targets,
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    (prompts_dir / "task9_epistasis_prompts.jsonl").write_text(
        "\n".join(json.dumps(r) for r in prompt_rows) + "\n", encoding="utf-8"
    )

    print("\n=== TASK 9 EPISTASIS BENCHMARK BUILT ===", flush=True)
    print(f"  Targets: {manifest['n_targets']} ({manifest['n_denovo']} de novo + {manifest['n_natural']} natural, {manifest['n_unique_clusters']} clusters)")
    print("  Anti-cheat audit:", json.dumps(manifest["anti_cheat_audit"], indent=2))


if __name__ == "__main__":
    main()
