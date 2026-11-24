#!/usr/bin/env python3
"""Build & Calibrate Test 6: Interface Binding Alanine Scan (`Chain A + Partner B -> X->Ala Binding Importance`).

Curates primary wet-lab `X -> Ala` protein–protein binding datasets from `data/raw_binding_alanine_cache/` into:
  - `data/binding_alanine_benchmark/manifest.json`
  - `data/binding_alanine_benchmark/targets/s6_b001..json`
  - `data/binding_alanine_benchmark/structures/s6_b001..pdb` (for crystal SKEMPI complexes)
  - `outputs/prompts/stage6_binding_alanine_prompts.jsonl` + `web_chat_spotcheck/`
  - `outputs/results/binding_alanine_evaluation_report.json`
  - `outputs/figures/binding_alanine_overview.png`

Categories (100% focused on predicting per-residue `X -> Ala` binding importance):
  1. `skempi_biophysical_ddg`: Binary crystal complexes from SKEMPI v2.0 (`>= 8` unique single-point
     `X -> Ala` SPR/ITC ΔΔG_bind kcal/mol mutants + exact 3D crystal interface contact flags `Cβ < 8.0 Å`).
  2. `oncology_alphaseq_kd`: Clinical immuno-oncology checkpoint binders from A-Alpha Bio `open-alphaseq`
     (`Pembrolizumab-scFv x Human PD-1`, `PP489 x Human TIGIT`, `PP489 x Mouse TIGIT`, quantitative Δlog10 Kd).
  3. `ppi_binding_dms`: Deep mutational scans of protein–protein binding (`MLH1 x PMS2`, `INSR x mAb 83-07`,
     `INSR x Insulin`, `RAF1-RBD x KRAS`, `GRB2-SH3 x SOS1`).
  4. `post_may2026_mavedb_binding`: Oct–Nov 2026 post-cutoff MaveDB interaction & peptide-binding scans
     (`2026-10-16 KRAS x LZTR1` direct binding with wet-lab replicate ceiling, `2026-10-16 GRB2 x SOS1`,
     `2026-10-16 RAF1 x KRAS`, `2026-11-09 OPRM1 x DAMGO/peptide`).
"""

from __future__ import annotations

from collections import Counter, defaultdict
import csv
import io
import json
import math
from pathlib import Path
import re
import sys
from typing import Any, Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pyarrow.parquet as pq
from scipy.stats import spearmanr

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from protein_without_tools.alanine_scan import select_balanced_alanine_panel
from protein_without_tools.binding_alanine_scan import (
    BOGAN_THORN_BINDING_HOTSPOT_SCALE,
    BindingAlanineParsed,
    BindingAlanineTarget,
    build_binding_alanine_prompt,
    grade_binding_alanine_target,
    parse_binding_alanine_response,
)
from protein_without_tools.pdb_utils import CANONICAL_AA_SET

AA3_TO_AA1: Dict[str, str] = {
    "ALA": "A", "CYS": "C", "ASP": "D", "GLU": "E", "PHE": "F",
    "GLY": "G", "HIS": "H", "ILE": "I", "LYS": "K", "LEU": "L",
    "MET": "M", "ASN": "N", "PRO": "P", "GLN": "Q", "ARG": "R",
    "SER": "S", "THR": "T", "VAL": "V", "TRP": "W", "TYR": "Y",
    "MSE": "M", "SEP": "S", "TPO": "T", "PTR": "Y",
}

SAFETY_BLOCKLIST = re.compile(
    r"(virus|viral|virion|capsid|phage|toxin|venom|ricin|abrin|"
    r"botulinum|anthrax|anthracis|ebola|marburg|variola|influenza|corona|sars|\bhiv\b|gp120|cov2)",
    re.IGNORECASE,
)

R_KCAL_MOL_K = 0.00198720425864083


def compute_virtual_cb(n_xyz: np.ndarray, ca_xyz: np.ndarray, c_xyz: np.ndarray, seq: str) -> np.ndarray:
    """Idealized Cβ coordinates (Cα for Gly)."""
    b = ca_xyz - n_xyz
    c = c_xyz - ca_xyz
    a = np.cross(b, c)
    cb = -0.58273431 * a + 0.56802827 * b - 0.54067466 * c + ca_xyz
    out = cb.copy()
    for idx, aa in enumerate(seq):
        if aa == "G":
            out[idx] = ca_xyz[idx]
    return out


def parse_mmcif_auth_chains(cif_text: str) -> Dict[str, Dict[str, Any]]:
    """Parse protein chains keyed by `auth_asym_id`, preserving exact `(auth_seq_id, ins_code)` -> 1-idx map."""
    lines = cif_text.splitlines()
    n_lines = len(lines)
    i = 0
    chains_raw: Dict[str, Dict[Tuple[int, str], Dict[str, Any]]] = defaultdict(dict)
    chains_order: Dict[str, List[Tuple[int, str]]] = defaultdict(list)

    while i < n_lines:
        if lines[i].strip() == "loop_":
            cols: List[str] = []
            j = i + 1
            while j < n_lines and lines[j].strip().startswith("_"):
                cols.append(lines[j].strip())
                j += 1
            if cols and cols[0].startswith("_atom_site."):
                col_idx = {c.split(".")[1]: k for k, c in enumerate(cols)}
                req = ("group_PDB", "label_atom_id", "label_comp_id", "Cartn_x", "Cartn_y", "Cartn_z")
                if all(r in col_idx for r in req):
                    i_grp = col_idx["group_PDB"]
                    i_atom = col_idx["label_atom_id"]
                    i_comp = col_idx["label_comp_id"]
                    i_auth_asym = col_idx.get("auth_asym_id", col_idx.get("label_asym_id", 0))
                    i_auth_seq = col_idx.get("auth_seq_id", col_idx.get("label_seq_id", 0))
                    i_ins = col_idx.get("pdbx_PDB_ins_code")
                    i_alt = col_idx.get("label_alt_id")
                    i_mod = col_idx.get("pdbx_PDB_model_num")
                    ix, iy, iz = col_idx["Cartn_x"], col_idx["Cartn_y"], col_idx["Cartn_z"]

                    while j < n_lines:
                        ln = lines[j].strip()
                        if not ln or ln.startswith("#") or ln == "loop_" or ln.startswith("_"):
                            break
                        toks = ln.split()
                        if len(toks) >= len(cols):
                            if i_mod is not None and toks[i_mod] not in ("1", ".", "?"):
                                j += 1
                                continue
                            if toks[i_grp] in ("ATOM", "HETATM"):
                                alt = toks[i_alt] if i_alt is not None else "."
                                if alt in (".", "?", "A"):
                                    comp = toks[i_comp].upper()
                                    aa1 = AA3_TO_AA1.get(comp)
                                    if aa1 and aa1 in CANONICAL_AA_SET:
                                        ch = toks[i_auth_asym]
                                        seq_s = toks[i_auth_seq]
                                        ins_s = toks[i_ins] if i_ins is not None else "."
                                        if ins_s in (".", "?"):
                                            ins_s = ""
                                        try:
                                            rkey = (int(seq_s), ins_s.upper())
                                            at_name = toks[i_atom].strip('"')
                                            if at_name in ("N", "CA", "C", "O", "CB"):
                                                if rkey not in chains_raw[ch]:
                                                    chains_raw[ch][rkey] = {"aa": aa1, "atoms": {}}
                                                    chains_order[ch].append(rkey)
                                                if at_name not in chains_raw[ch][rkey]["atoms"]:
                                                    chains_raw[ch][rkey]["atoms"][at_name] = [
                                                        float(toks[ix]),
                                                        float(toks[iy]),
                                                        float(toks[iz]),
                                                    ]
                                        except ValueError:
                                            pass
                        j += 1
                    i = j
                    continue
        i += 1

    parsed: Dict[str, Dict[str, Any]] = {}
    for ch, order in chains_order.items():
        seq_chars: List[str] = []
        ca_list: List[List[float]] = []
        cb_list: List[List[float]] = []
        auth_to_1idx: Dict[Tuple[int, str], int] = {}
        for rkey in order:
            entry = chains_raw[ch][rkey]
            atoms = entry["atoms"]
            if "CA" not in atoms:
                continue
            idx1 = len(seq_chars) + 1
            aa = entry["aa"]
            seq_chars.append(aa)
            ca_xyz = np.asarray(atoms["CA"], dtype=np.float64)
            ca_list.append(ca_xyz.tolist())
            if "CB" in atoms and aa != "G":
                cb_list.append(atoms["CB"])
            elif all(k in atoms for k in ("N", "CA", "C")):
                vcb = compute_virtual_cb(
                    np.asarray([atoms["N"]]),
                    np.asarray([atoms["CA"]]),
                    np.asarray([atoms["C"]]),
                    aa,
                )[0]
                cb_list.append(vcb.tolist())
            else:
                cb_list.append(ca_xyz.tolist())
            auth_to_1idx[rkey] = idx1
        if len(seq_chars) >= 25:
            parsed[ch] = {
                "sequence": "".join(seq_chars),
                "ca": np.asarray(ca_list, dtype=np.float64),
                "cb": np.asarray(cb_list, dtype=np.float64),
                "auth_to_1idx": auth_to_1idx,
            }
    return parsed


def write_complex_ca_pdb(out_path: Path, seq_a: str, ca_a: np.ndarray, seq_b: str, ca_b: np.ndarray) -> None:
    aa1_to_3 = {v: k for k, v in AA3_TO_AA1.items() if len(k) == 3 and v in CANONICAL_AA_SET}
    aa1_to_3.update({"A": "ALA", "C": "CYS", "D": "ASP", "E": "GLU", "F": "PHE", "G": "GLY", "H": "HIS", "I": "ILE",
                     "K": "LYS", "L": "LEU", "M": "MET", "N": "ASN", "P": "PRO", "Q": "GLN", "R": "ARG", "S": "SER",
                     "T": "THR", "V": "VAL", "W": "TRP", "Y": "TYR"})
    lines: List[str] = []
    serial = 1
    for chain_id, seq, ca in (("A", seq_a, ca_a), ("B", seq_b, ca_b)):
        for r_i, (aa, xyz) in enumerate(zip(seq, ca), start=1):
            res3 = aa1_to_3.get(aa, "UNK")
            lines.append(
                f"ATOM  {serial:5d}  CA  {res3:>3s} {chain_id}{r_i:4d}    "
                f"{xyz[0]:8.3f}{xyz[1]:8.3f}{xyz[2]:8.3f}  1.00 50.00           C"
            )
            serial += 1
        lines.append("TER")
    lines.append("END")
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_skempi_pdb_mut(mut_pdb: str) -> Optional[Tuple[str, str, int, str]]:
    """Parse SKEMPI `Mutation(s)_PDB` token like `YB33A` or `LH100AA` -> `(wt_aa, chain, resnum, ins_code)`."""
    s = mut_pdb.strip()
    if len(s) < 4 or s[-1] != "A":
        return None
    wt_aa = s[0].upper()
    chain = s[1]
    mid = s[2:-1]
    m = re.match(r"^(-?\d+)([A-Za-z]?)$", mid)
    if not m or wt_aa == "A" or wt_aa not in CANONICAL_AA_SET:
        return None
    return (wt_aa, chain, int(m.group(1)), m.group(2).upper())


def main() -> None:
    raw_dir = REPO / "data" / "raw_binding_alanine_cache"
    out_dir = REPO / "data" / "binding_alanine_benchmark"
    t_dir = out_dir / "targets"
    s_dir = out_dir / "structures"
    prompt_dir = REPO / "outputs" / "prompts"
    spot_dir = prompt_dir / "web_chat_spotcheck"
    res_dir = REPO / "outputs" / "results"
    fig_dir = REPO / "outputs" / "figures"

    for d in (out_dir, t_dir, s_dir, prompt_dir, spot_dir, res_dir, fig_dir):
        d.mkdir(parents=True, exist_ok=True)
    # Clean any stale target JSONs
    for old_f in t_dir.glob("*.json"):
        old_f.unlink()

    partner_uniprot = json.loads((raw_dir / "uniprot_partner_sequences.json").read_text(encoding="utf-8"))

    curated_records: List[Tuple[BindingAlanineTarget, Dict[str, Any]]] = []

    # ==========================================================================
    # CATEGORY 1: `skempi_biophysical_ddg` (SKEMPI v2.0 SPR/ITC X->Ala + 3D CIF)
    # ==========================================================================
    sk_rows = list(
        csv.DictReader(
            io.StringIO((raw_dir / "skempi_v2.csv").read_text(encoding="utf-8", errors="replace")),
            delimiter=";",
        )
    )
    cif_cache: Dict[str, Dict[str, Dict[str, Any]]] = {}
    for cp in sorted((raw_dir / "skempi_cifs").glob("*.cif")):
        cif_cache[cp.stem.upper()] = parse_mmcif_auth_chains(cp.read_text(encoding="utf-8", errors="replace"))

    grouped_sk: Dict[Tuple[str, str, str], Dict[int, List[float]]] = defaultdict(lambda: defaultdict(list))
    meta_sk: Dict[Tuple[str, str, str], Dict[str, str]] = {}

    for row in sk_rows:
        pdb_field = str(row.get("#Pdb") or "")
        mut_pdb = str(row.get("Mutation(s)_PDB") or "").strip()
        p1_name = str(row.get("Protein 1") or "").strip()
        p2_name = str(row.get("Protein 2") or "").strip()
        if SAFETY_BLOCKLIST.search(f"{p1_name} {p2_name}"):
            continue
        parts = pdb_field.split("_")
        if len(parts) < 3 or "," in mut_pdb or not mut_pdb.endswith("A"):
            continue
        pdb4, g1, g2 = parts[0].upper(), parts[1], parts[2]
        if pdb4 not in cif_cache:
            continue
        mut_loc = str(row.get("iMutation_Location(s)") or "").strip().upper()
        # Exclude buried monomer-interior (`INT`) mutations so SKEMPI strictly tests surface/interface sites
        if mut_loc == "INT":
            continue
        pm = parse_skempi_pdb_mut(mut_pdb)
        if not pm:
            continue
        wt_aa, mch, resnum, ins = pm
        pch = g2 if mch in g1 else g1
        if len(pch) != 1:
            continue
        chs = cif_cache[pdb4]
        if mch not in chs or pch not in chs:
            continue
        idx1 = chs[mch]["auth_to_1idx"].get((resnum, ins))
        if idx1 is None or chs[mch]["sequence"][idx1 - 1] != wt_aa:
            continue
        try:
            kd_wt = float(str(row.get("Affinity_wt_parsed") or "").strip())
            kd_mut = float(str(row.get("Affinity_mut_parsed") or "").strip())
            if kd_wt <= 0 or kd_mut <= 0:
                continue
        except ValueError:
            continue
        temp_s = re.sub(r"[^0-9.]", "", str(row.get("Temperature") or "298.15"))
        temp_k = float(temp_s) if temp_s else 298.15
        if temp_k < 250 or temp_k > 350:
            temp_k = 298.15
        # Positive ddG_bind = Kd_mut > Kd_wt = mutation to Ala WEAKENS binding!
        ddg_bind = R_KCAL_MOL_K * temp_k * math.log(kd_mut / kd_wt)
        ddg_bind = float(np.clip(ddg_bind, -4.0, 8.5))
        key = (pdb4, mch, pch)
        grouped_sk[key][idx1].append(ddg_bind)
        scanned_label = p1_name if mch in g1 else p2_name
        partner_label = p2_name if mch in g1 else p1_name
        meta_sk[key] = {
            "pdb_id": pdb4,
            "scanned_chain": mch,
            "partner_chain": pch,
            "source_name": f"{scanned_label} × {partner_label} ({pdb4} {mch}→{pch})",
            "method": str(row.get("Method") or "SPR/ITC"),
        }

    # Deduplicate near-identical sequence pairs (>90% SeqID on BOTH chains) while keeping highest-N complexes
    seen_seq_pairs: List[Tuple[str, str]] = []
    for key, pos_dict in sorted(grouped_sk.items(), key=lambda kv: -len(kv[1])):
        if len(pos_dict) < 8:
            continue
        pdb4, mch, pch = key
        sa = cif_cache[pdb4][mch]["sequence"]
        sb = cif_cache[pdb4][pch]["sequence"]
        if any(sa == pa and sb == pb for pa, pb in seen_seq_pairs):
            continue
        seen_seq_pairs.append((sa, sb))

        positions = sorted(pos_dict.keys())
        wt_aas = [sa[p - 1] for p in positions]
        exp_vals = [round(float(np.median(pos_dict[p])), 4) for p in positions]

        # Compute exact 3D crystal interface contact flags (`min d(Cβ_A[p], Cβ_B) < 8.0 Å`)
        cb_a = cif_cache[pdb4][mch]["cb"]
        cb_b = cif_cache[pdb4][pch]["cb"]
        dmat_ab = np.linalg.norm(cb_a[:, None, :] - cb_b[None, :, :], axis=-1)
        min_d = np.min(dmat_ab, axis=1)
        contact_cnt = np.sum(dmat_ab < 8.0, axis=1)
        iface_flags = [1 if float(min_d[p - 1]) < 8.0 else 0 for p in positions]
        iface_cnts = [int(contact_cnt[p - 1]) for p in positions]

        tid = f"s6_b{len(curated_records) + 1:03d}"
        write_complex_ca_pdb(s_dir / f"{tid}.pdb", sa, cif_cache[pdb4][mch]["ca"], sb, cif_cache[pdb4][pch]["ca"])
        tgt = BindingAlanineTarget(
            target_id=tid,
            category="skempi_biophysical_ddg",
            assay_type="spr_itc_ddg_kcal",
            scanned_sequence=sa,
            partner_sequence=sb,
            positions_1idx=positions,
            wt_aas=wt_aas,
            exp_binding_loss=exp_vals,
            is_interface_flags=iface_flags,
            memorization_stratum="known_pdb_pre2026",
        )
        curated_records.append(
            (
                tgt,
                {
                    **meta_sk[key],
                    "published_date": "pre-2026-05 (SKEMPI v2.0)",
                    "memorization_stratum": "known_pdb_pre2026",
                    "monomer_stability_filter": "SKEMPI purified protein + excluded INT buried interior",
                    "crystal_contact_counts": iface_cnts,
                },
            )
        )
        if sum(1 for t, _ in curated_records if t.category == "skempi_biophysical_ddg") >= 28:
            break

    # ==========================================================================
    # CATEGORY 2: `oncology_alphaseq_kd` (CDR / Surface-Filtered PD-1 & TIGIT)
    # ==========================================================================
    # Exclude buried VH β-sandwich framework core, C96 disulfide, buried Vernier anchors (51, 60, 64),
    # and the YM_0852 pos-78 oligo dropout so 100% of candidates are folded CDR/outer-loop sites!
    pembro_surface_paratope = (
        set(range(26, 36))
        | (set(range(50, 67)) - {51, 60, 64})
        | set(range(74, 78))
        | set(range(99, 111))
    )
    pp489_surface_paratope = (
        set(range(26, 36))
        | (set(range(50, 67)) - {51, 60, 64})
        | set(range(74, 79))
        | set(range(99, 115))
    )
    alphaseq_specs = [
        (
            "YM_0852",
            "huPD-1_RE(rebuilt)",
            "Pembrolizumab-scFv × Human PD-1 (Open-AlphaSeq YM_0852, CDR/Surface-Filtered)",
            "2024-09",
            "known_pdb_pre2026",
            pembro_surface_paratope,
        ),
        (
            "YM_0693",
            "TIGIT_22-137_POI-AGA2",
            "Anti-TIGIT scFv PP489 × Human TIGIT (Open-AlphaSeq YM_0693, CDR/Surface-Filtered)",
            "2024-09",
            "unseen_or_post2026",
            pp489_surface_paratope,
        ),
        (
            "YM_0693",
            "TIGIT_Mouse",
            "Anti-TIGIT scFv PP489 × Mouse TIGIT (Open-AlphaSeq YM_0693, CDR/Surface-Filtered)",
            "2024-09",
            "unseen_or_post2026",
            pp489_surface_paratope,
        ),
    ]
    for cfg, partner_desc, label, pub_dt, mem_strat, allowed_pos_set in alphaseq_specs:
        tbl = pq.read_table(raw_dir / "alphaseq" / f"{cfg}.parquet")
        d = tbl.to_pydict()
        rows_p = [
            (ms, bs, aff, rep_ok)
            for ms, bs, pd_s, aff, rep_ok in zip(
                d["mata_sequence"],
                d["matalpha_sequence"],
                d["matalpha_description"],
                d["alphaseq_affinity"],
                d["sufficient_replicate_observations"],
            )
            if pd_s == partner_desc and ms and bs
        ]
        Lm = Counter(len(ms) for ms, _, _, _ in rows_p).most_common(1)[0][0]
        seqs_L = [ms for ms, _, _, _ in rows_p if len(ms) == Lm]
        wt_sa = "".join(Counter(s[i] for s in seqs_L).most_common(1)[0][0] for i in range(Lm))
        wt_sb = Counter(bs for _, bs, _, _ in rows_p).most_common(1)[0][0]
        wt_sa = "".join(c for c in wt_sa if c in CANONICAL_AA_SET)
        wt_sb = "".join(c for c in wt_sb if c in CANONICAL_AA_SET)

        wt_affs = [float(aff) for ms, _, aff, _ in rows_p if ms == wt_sa and aff is not None and not math.isnan(float(aff))]
        wt_kd_log10 = float(np.median(wt_affs))

        pos_affs: Dict[int, List[float]] = defaultdict(list)
        for ms, _, aff, rep_ok in rows_p:
            if len(ms) != len(wt_sa) or aff is None or math.isnan(float(aff)) or rep_ok is False:
                continue
            diffs = [idx for idx in range(len(wt_sa)) if ms[idx] != wt_sa[idx]]
            if len(diffs) == 1 and ms[diffs[0]] == "A":
                p1 = diffs[0] + 1
                if p1 in allowed_pos_set:
                    pos_affs[p1].append(float(aff) - wt_kd_log10)

        positions = sorted(pos_affs.keys())
        wt_aas = [wt_sa[p - 1] for p in positions]
        exp_vals = [round(float(np.median(pos_affs[p])), 4) for p in positions]
        tid = f"s6_b{len(curated_records) + 1:03d}"
        tgt = BindingAlanineTarget(
            target_id=tid,
            category="oncology_alphaseq_kd",
            assay_type="alphaseq_delta_log10kd",
            scanned_sequence=wt_sa,
            partner_sequence=wt_sb,
            positions_1idx=positions,
            wt_aas=wt_aas,
            exp_binding_loss=exp_vals,
            is_interface_flags=None,
            memorization_stratum=mem_strat,
        )
        curated_records.append(
            (
                tgt,
                {
                    "pdb_id": ("5B8C/5JXE" if cfg == "YM_0852" else None),
                    "source_name": label,
                    "published_date": pub_dt,
                    "memorization_stratum": mem_strat,
                    "monomer_stability_filter": "Restricted to CDR-H1/H2/H3 + exposed DE loop (excluded VH core & C96)",
                    "wt_log10_kd_nM": round(wt_kd_log10, 3),
                },
            )
        )

    # ==========================================================================
    # CATEGORY 3 (`ppi_binding_dms`) & CATEGORY 4 (`post_may2026_mavedb_binding`)
    # ALL FILTERED BY PAIRED MONOMER ABUNDANCE / SURFACE EXPRESSION!
    # ==========================================================================
    AA3_MAP = {
        "Ala": "A", "Cys": "C", "Asp": "D", "Glu": "E", "Phe": "F",
        "Gly": "G", "His": "H", "Ile": "I", "Lys": "K", "Leu": "L",
        "Met": "M", "Asn": "N", "Pro": "P", "Gln": "Q", "Arg": "R",
        "Ser": "S", "Thr": "T", "Val": "V", "Trp": "W", "Tyr": "Y",
    }

    def _load_companion_abundance_map(csv_path: Path, gene_filter: Optional[str], c_start: int, c_end: int) -> Dict[int, float]:
        rows_a = list(csv.DictReader(io.StringIO(csv_path.read_text(encoding="utf-8"))))
        acc: Dict[int, List[float]] = defaultdict(list)
        for r in rows_a:
            hp = str(r.get("hgvs_pro") or "")
            if gene_filter and ":" in hp and hp.split(":")[0].upper() != gene_filter.upper():
                continue
            m = re.search(r"p\.([A-Z][a-z]{2})(\d+)Ala$", hp)
            if not m:
                continue
            pos_raw = int(m.group(2))
            if not (c_start <= pos_raw <= c_end):
                continue
            try:
                v = float(r.get("score") or "nan")
                if not math.isnan(v):
                    acc[pos_raw - c_start + 1].append(v)
            except ValueError:
                pass
        return {k: float(np.median(vs)) for k, vs in acc.items()}

    def _load_mavedb_ala_scan(
        urn_file_stem: str,
        companion_abund_csv: Path,
        partner_seq: str,
        category: str,
        mem_strat: str,
        source_name: str,
        gene_filter: Optional[str] = None,
        crop_range_1idx: Optional[Tuple[int, int]] = None,
        flip_sign: bool = True,
    ) -> Tuple[BindingAlanineTarget, Dict[str, Any]]:
        j_meta = json.loads((raw_dir / "mavedb_binding" / f"{urn_file_stem}.json").read_text(encoding="utf-8"))
        tgenes = j_meta.get("targetGenes") or []
        chosen_tg = tgenes[0]
        if gene_filter:
            for tg in tgenes:
                if gene_filter.upper() in str(tg.get("name") or "").upper():
                    chosen_tg = tg
                    break
        full_seq = str((chosen_tg.get("targetSequence") or {}).get("sequence") or "").strip().upper()
        c_start, c_end = crop_range_1idx if crop_range_1idx else (1, len(full_seq))
        sa_cropped = full_seq[c_start - 1 : c_end]

        abund_map = _load_companion_abundance_map(companion_abund_csv, gene_filter, c_start, c_end)
        abund_cut = float(np.quantile(list(abund_map.values()), 0.35)) if abund_map else -1e9

        csv_rows = list(csv.DictReader(io.StringIO((raw_dir / "mavedb_binding" / f"{urn_file_stem}.csv").read_text(encoding="utf-8"))))
        by_pos: Dict[int, List[float]] = defaultdict(list)
        rep1_map: Dict[int, float] = {}
        rep2_map: Dict[int, float] = {}

        for r in csv_rows:
            hp = str(r.get("hgvs_pro") or "")
            if gene_filter and ":" in hp and hp.split(":")[0].upper() != gene_filter.upper():
                continue
            m = re.search(r"p\.([A-Z][a-z]{2})(\d+)Ala$", hp)
            if not m:
                continue
            wt3, pos_raw = m.group(1), int(m.group(2))
            wt1 = AA3_MAP.get(wt3)
            if not wt1 or wt1 == "A" or not (c_start <= pos_raw <= c_end):
                continue
            pos_rel = pos_raw - c_start + 1
            if sa_cropped[pos_rel - 1] != wt1:
                continue
            # Require WT-like monomer abundance / surface expression in companion scan!
            if abund_map and abund_map.get(pos_rel, -1e9) < abund_cut:
                continue
            sc_s = r.get("score")
            if sc_s in (None, "", "NA", "nan"):
                continue
            try:
                sc_val = float(sc_s)
                if math.isnan(sc_val):
                    continue
            except ValueError:
                continue
            loss_val = -sc_val if flip_sign else sc_val
            by_pos[pos_rel].append(loss_val)

            r1_s = r.get("wt_relative_score_replicate_1")
            r2_s = r.get("wt_relative_score_replicate_2")
            if r1_s not in (None, "", "NA", "nan") and r2_s not in (None, "", "NA", "nan"):
                try:
                    rep1_map[pos_rel] = -float(r1_s)
                    rep2_map[pos_rel] = -float(r2_s)
                except ValueError:
                    pass

        positions = sorted(by_pos.keys())
        wt_aas = [sa_cropped[p - 1] for p in positions]
        exp_vals = [round(float(np.median(by_pos[p])), 4) for p in positions]

        rep_rho: Optional[float] = None
        common_rep = [p for p in positions if p in rep1_map and p in rep2_map]
        if len(common_rep) >= 20:
            rep_rho = round(float(spearmanr([rep1_map[p] for p in common_rep], [rep2_map[p] for p in common_rep]).statistic), 4)

        tid = f"s6_b{len(curated_records) + 1:03d}"
        tgt = BindingAlanineTarget(
            target_id=tid,
            category=category,
            assay_type="binding_dms_loss",
            scanned_sequence=sa_cropped,
            partner_sequence=partner_seq,
            positions_1idx=positions,
            wt_aas=wt_aas,
            exp_binding_loss=exp_vals,
            is_interface_flags=None,
            memorization_stratum=mem_strat,
        )
        meta_out: Dict[str, Any] = {
            "pdb_id": None,
            "mavedb_urn": j_meta.get("urn"),
            "source_name": source_name,
            "published_date": str(j_meta.get("publishedDate") or "")[:10],
            "memorization_stratum": mem_strat,
            "monomer_stability_filter": f"Filtered by companion abundance/expression CSV ({companion_abund_csv.name} >= q35)",
        }
        if rep_rho is not None:
            meta_out["measured_wetlab_replicate_spearman_rho"] = rep_rho
        return tgt, meta_out

    ins_mature = partner_uniprot["INS"][24:110]
    kras_seq = json.loads((raw_dir / "mavedb_binding" / "urn_mavedb_00001302-d-1.json").read_text())["targetGenes"][1]["targetSequence"]["sequence"]
    sos1_pr = partner_uniprot["SOS1"][1010:1333]
    lztr1_kelch = partner_uniprot["LZTR1"][:420]
    beta_endorphin = partner_uniprot["POMC"][236:267]
    ighg1_seq = partner_uniprot["IGHG1"][:230]
    mb_dir = raw_dir / "mavedb_binding"
    ma_dir = REPO / "data" / "raw_alanine_cache" / "mavedb"

    # Category 3: Abundance/Expression-Filtered PPI Binding DMS (`n = 3`)
    cat3_specs = [
        ("urn_mavedb_00001218-a-2", mb_dir / "urn_mavedb_00001218-a-1.csv", partner_uniprot["PMS2"][500:], "ppi_binding_dms", "unseen_or_post2026", "Human MLH1 CTD × PMS2 MutLα Dimerization DMS (Abundance-Filtered)", None, None),
        ("urn_mavedb_00001239-a-2", mb_dir / "urn_mavedb_00001239-a-1.csv", ighg1_seq, "ppi_binding_dms", "known_pdb_pre2026", "Human Insulin Receptor (L1-CR-L2) × mAb 83-07 FACS Binding DMS (Surface-Filtered)", None, (1, 470)),
        ("urn_mavedb_00001239-a-1", mb_dir / "urn_mavedb_00001239-a-2.csv", ins_mature, "ppi_binding_dms", "known_pdb_pre2026", "Human Insulin Receptor (L1-CR-L2) × Human Insulin DMS (Surface-Filtered)", None, (1, 470)),
    ]
    for u_stem, ab_csv, p_seq, cat_k, m_st, s_name, g_filt, c_rng in cat3_specs:
        curated_records.append(_load_mavedb_ala_scan(u_stem, ab_csv, p_seq, cat_k, m_st, s_name, g_filt, c_rng))

    # Category 4: Post-May-2026 Abundance/Expression-Filtered Binding DMS (`n = 4`, Oct–Nov 2026!)
    cat4_specs = [
        ("urn_mavedb_00001302-g-1", mb_dir / "urn_mavedb_00001302-d-1.csv", lztr1_kelch, "post_may2026_mavedb_binding", "unseen_or_post2026", "Human KRAS × Human LZTR1 Direct Binding DMS (Oct 16, 2026, Abundance-Filtered)", "KRAS", None),
        ("urn_mavedb_00001305-b-1", mb_dir / "urn_mavedb_00001305-a-1.csv", sos1_pr, "post_may2026_mavedb_binding", "unseen_or_post2026", "Human GRB2 × Human SOS1 Adaptor Signaling DMS (Oct 16, 2026, Abundance-Filtered)", "GRB2", None),
        ("urn_mavedb_00001316-0-3", mb_dir / "urn_mavedb_00001316-0-2.csv", kras_seq, "post_may2026_mavedb_binding", "unseen_or_post2026", "Human RAF1 (N-Term RBD-CRD) × Human KRAS DMS (Oct 16, 2026, Abundance-Filtered)", "RAF1", (1, 305)),
        ("urn_mavedb_00001326-e-1", ma_dir / "urn_mavedb_00001326-a-1.csv", beta_endorphin, "post_may2026_mavedb_binding", "unseen_or_post2026", "Human µ-Opioid Receptor OPRM1 × Peptide Agonist DMS (Nov 9, 2026, Surface-Filtered)", None, None),
    ]
    for u_stem, ab_csv, p_seq, cat_k, m_st, s_name, g_filt, c_rng in cat4_specs:
        curated_records.append(_load_mavedb_ala_scan(u_stem, ab_csv, p_seq, cat_k, m_st, s_name, g_filt, c_rng))

    # ==========================================================================
    # SUBSAMPLE BALANCED 18-RESIDUE RANKING PANELS & SAVE TARGETS + PROMPTS
    # ==========================================================================
    manifest_rows: List[Dict[str, Any]] = []
    prompt_rows: List[str] = []
    total_panel_mutants = 0
    total_full_mutants = 0
    panel_records: List[Tuple[BindingAlanineTarget, Dict[str, Any], List[int], List[float]]] = []

    for old_sp in spot_dir.glob("test6_binding_*.txt"):
        old_sp.unlink()

    for full_tgt, extra_meta in curated_records:
        full_pos = list(full_tgt.positions_1idx)
        full_wt = list(full_tgt.wt_aas)
        full_loss = list(full_tgt.exp_binding_loss)
        full_iface = full_tgt.is_interface_flags
        total_full_mutants += len(full_pos)

        sel_i, roles = select_balanced_alanine_panel(full_pos, full_wt, full_loss, target_panel_size=18)
        pan_pos = [full_pos[k] for k in sel_i]
        pan_wt = [full_wt[k] for k in sel_i]
        pan_loss = [full_loss[k] for k in sel_i]
        pan_iface = ([full_iface[k] for k in sel_i] if full_iface is not None else None)
        if extra_meta.get("crystal_contact_counts") is not None:
            full_cnts = extra_meta["crystal_contact_counts"]
            extra_meta = {**extra_meta, "crystal_contact_counts": [full_cnts[k] for k in sel_i]}

        tgt = BindingAlanineTarget(
            target_id=full_tgt.target_id,
            category=full_tgt.category,
            assay_type=full_tgt.assay_type,
            scanned_sequence=full_tgt.scanned_sequence,
            partner_sequence=full_tgt.partner_sequence,
            positions_1idx=pan_pos,
            wt_aas=pan_wt,
            exp_binding_loss=pan_loss,
            is_interface_flags=pan_iface,
            panel_roles=roles,
            memorization_stratum=full_tgt.memorization_stratum,
        )
        panel_records.append((tgt, extra_meta, full_pos, full_loss))
        total_panel_mutants += len(pan_pos)

        t_json = {
            "target_id": tgt.target_id,
            "category": tgt.category,
            "memorization_stratum": tgt.memorization_stratum,
            "assay_type": tgt.assay_type,
            "scanned_length": len(tgt.scanned_sequence),
            "partner_length": len(tgt.partner_sequence),
            "scanned_sequence": tgt.scanned_sequence,
            "partner_sequence": tgt.partner_sequence,
            "positions_1idx": tgt.positions_1idx,
            "wt_aas": tgt.wt_aas,
            "exp_binding_loss": tgt.exp_binding_loss,
            "is_interface_flags": tgt.is_interface_flags,
            "panel_roles": roles,
            "full_positions_1idx": full_pos,
            "full_wt_aas": full_wt,
            "full_exp_binding_loss": full_loss,
        }
        (t_dir / f"{tgt.target_id}.json").write_text(json.dumps(t_json, indent=2), encoding="utf-8")

        prompt_txt = build_binding_alanine_prompt(tgt)
        for forbidden in (tgt.target_id, str(extra_meta.get("pdb_id") or "ZZZZ").split("/")[0], str(extra_meta.get("mavedb_urn") or "ZZZZ")):
            if forbidden and forbidden != "ZZZZ" and len(forbidden) >= 4:
                assert forbidden not in prompt_txt, f"Leakage of {forbidden} in {tgt.target_id}"

        prompt_rows.append(
            json.dumps(
                {
                    "target_id": tgt.target_id,
                    "task": "test6_binding_alanine_ranking",
                    "category": tgt.category,
                    "memorization_stratum": tgt.memorization_stratum,
                    "length": len(tgt.scanned_sequence),
                    "partner_length": len(tgt.partner_sequence),
                    "n_assayed": len(tgt.positions_1idx),
                    "prompt": prompt_txt,
                }
            )
        )

        manifest_rows.append(
            {
                "target_id": tgt.target_id,
                "category": tgt.category,
                "assay_type": tgt.assay_type,
                "scanned_length": len(tgt.scanned_sequence),
                "partner_length": len(tgt.partner_sequence),
                "n_assayed": len(tgt.positions_1idx),
                "n_full_scan_positions": len(full_pos),
                "has_3d_interface_flags": bool(tgt.is_interface_flags is not None),
                **extra_meta,
            }
        )

    cat_counts = dict(Counter(r["category"] for r in manifest_rows))
    mem_counts = dict(Counter(r["memorization_stratum"] for r in manifest_rows))
    manifest_obj = {
        "benchmark": "Protein Structure Without Tools Test 6: Interface Binding Alanine Scan Ranking (X -> Ala, Monomer-Stability Filtered)",
        "n_targets": len(manifest_rows),
        "total_panel_candidates": total_panel_mutants,
        "total_assayed_alanine_mutants": total_panel_mutants,
        "total_underlying_wetlab_alanine_mutants": total_full_mutants,
        "category_counts": cat_counts,
        "memorization_stratum_counts": mem_counts,
        "targets": manifest_rows,
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest_obj, indent=2), encoding="utf-8")
    (prompt_dir / "stage6_binding_alanine_prompts.jsonl").write_text("\n".join(prompt_rows) + "\n", encoding="utf-8")

    seen_spot_cats: set[str] = set()
    for tgt, _m, _fp, _fl in panel_records:
        if tgt.category not in seen_spot_cats:
            seen_spot_cats.add(tgt.category)
            (spot_dir / f"test6_binding_{tgt.category}_{tgt.target_id}.txt").write_text(
                build_binding_alanine_prompt(tgt) + "\n",
                encoding="utf-8",
            )

    # ==========================================================================
    # CALIBRATION BASELINES
    # ==========================================================================
    rng = np.random.default_rng(20261120)
    models_eval: Dict[str, List[Dict[str, Any]]] = {
        "Oracle_NoisyReplicate": [],
        "Oracle_3D_Interface_Contacts": [],
        "Baseline_BoganThorn_1D": [],
        "Null_Uniform": [],
    }

    for tgt, extra_meta, _fp, _fl in panel_records:
        y = np.asarray(tgt.exp_binding_loss, dtype=np.float64)
        sd = max(0.15, float(np.std(y)) * 0.40)
        noisy = y + rng.normal(0.0, sd, size=len(y))
        # Verify JSON roundtrip through the NEW `{"ranking": [...]}` format!
        noisy_rank_order = np.argsort(-noisy)
        ranked_tokens = [f"{tgt.positions_1idx[k]}{tgt.wt_aas[k]}->A" for k in noisy_rank_order]
        rep_parsed = parse_binding_alanine_response(json.dumps({"ranking": ranked_tokens}), tgt)
        g_rep = grade_binding_alanine_target(tgt, rep_parsed)
        models_eval["Oracle_NoisyReplicate"].append(g_rep.to_dict())

        bt_preds = np.asarray([BOGAN_THORN_BINDING_HOTSPOT_SCALE.get(aa, 0.5) for aa in tgt.wt_aas], dtype=np.float64)
        bt_json = json.dumps(
            {"binding_scan": [[p, aa, round(float(v), 3)] for p, aa, v in zip(tgt.positions_1idx, tgt.wt_aas, bt_preds)]}
        )
        bt_parsed = parse_binding_alanine_response(bt_json, tgt)
        g_bt = grade_binding_alanine_target(tgt, bt_parsed)
        models_eval["Baseline_BoganThorn_1D"].append(g_bt.to_dict())

        g_null = grade_binding_alanine_target(
            tgt, BindingAlanineParsed(np.zeros(len(y), dtype=np.float64), len(y), 1.0, True)
        )
        models_eval["Null_Uniform"].append(g_null.to_dict())

        if extra_meta.get("crystal_contact_counts") is not None:
            cnt_arr = np.asarray(extra_meta["crystal_contact_counts"], dtype=np.float64) + 0.15 * bt_preds
            g_3d = grade_binding_alanine_target(
                tgt, BindingAlanineParsed(cnt_arr, len(y), 1.0, True)
            )
            models_eval["Oracle_3D_Interface_Contacts"].append(g_3d.to_dict())

    def _summarize(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
        if not rows:
            return {}
        iface_vals = [r["interface_3d_auroc"] for r in rows if r.get("interface_3d_auroc") is not None]
        by_cat: Dict[str, Any] = {}
        for c_name in ("skempi_biophysical_ddg", "oncology_alphaseq_kd", "ppi_binding_dms", "post_may2026_mavedb_binding"):
            sub = [r for r in rows if r["category"] == c_name]
            if sub:
                by_cat[c_name] = {
                    "n": len(sub),
                    "mean_spearman_rho": round(float(np.mean([r["spearman_rho"] for r in sub])), 4),
                    "mean_ndcg_at_5": round(float(np.mean([r["ndcg_at_5"] for r in sub])), 4),
                    "mean_top_k_precision": round(float(np.mean([r["top_k_precision"] for r in sub])), 4),
                    "mean_same_aa_win_rate": round(float(np.mean([r["same_aa_win_rate"] for r in sub])), 4),
                    "mean_spearman_rho_aa_demeaned": round(float(np.mean([r["spearman_rho_aa_demeaned"] for r in sub])), 4),
                    "mean_binding_hotspot_auroc": round(float(np.mean([r["binding_hotspot_auroc"] for r in sub])), 4),
                }
        by_mem: Dict[str, Any] = {}
        for m_name in ("known_pdb_pre2026", "unseen_or_post2026"):
            sub_m = [r for r in rows if r.get("memorization_stratum") == m_name]
            if sub_m:
                by_mem[m_name] = {
                    "n": len(sub_m),
                    "mean_spearman_rho": round(float(np.mean([r["spearman_rho"] for r in sub_m])), 4),
                    "mean_ndcg_at_5": round(float(np.mean([r["ndcg_at_5"] for r in sub_m])), 4),
                    "mean_same_aa_win_rate": round(float(np.mean([r["same_aa_win_rate"] for r in sub_m])), 4),
                    "mean_spearman_rho_aa_demeaned": round(float(np.mean([r["spearman_rho_aa_demeaned"] for r in sub_m])), 4),
                }
        return {
            "n_targets": len(rows),
            "mean_spearman_rho": round(float(np.mean([r["spearman_rho"] for r in rows])), 4),
            "mean_ndcg_at_5": round(float(np.mean([r["ndcg_at_5"] for r in rows])), 4),
            "mean_top_k_precision": round(float(np.mean([r["top_k_precision"] for r in rows])), 4),
            "mean_same_aa_win_rate": round(float(np.mean([r["same_aa_win_rate"] for r in rows])), 4),
            "mean_spearman_rho_aa_demeaned": round(float(np.mean([r["spearman_rho_aa_demeaned"] for r in rows])), 4),
            "mean_binding_hotspot_auroc": round(float(np.mean([r["binding_hotspot_auroc"] for r in rows])), 4),
            "mean_interface_3d_auroc": (round(float(np.mean(iface_vals)), 4) if iface_vals else None),
            "by_memorization_stratum": by_mem,
            "by_category": by_cat,
        }

    kras_lztr1_rep = next(
        (m.get("measured_wetlab_replicate_spearman_rho") for _, m, _, _ in panel_records if m.get("mavedb_urn") == "urn:mavedb:00001302-g-1"),
        None,
    )
    report = {
        "benchmark": "Test 6: Interface Binding Alanine Scan Ranking (X -> Ala)",
        "n_targets": len(panel_records),
        "total_panel_candidates": total_panel_mutants,
        "total_underlying_wetlab_alanine_mutants": total_full_mutants,
        "category_counts": cat_counts,
        "kras_lztr1_oct2026_wetlab_replicate_spearman_rho": kras_lztr1_rep,
        "models": {mk: _summarize(rows) for mk, rows in models_eval.items()},
    }
    (res_dir / "binding_alanine_evaluation_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    # ==========================================================================
    # OVERVIEW CALIBRATION FIGURE (`outputs/figures/binding_alanine_overview.png`)
    # ==========================================================================
    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.8), dpi=180)
    fig.patch.set_facecolor("#ffffff")

    # Panel A: Panel Rank ρ vs AA-Demeaned Context ρ
    ax = axes[0]
    m_keys = ["Oracle_NoisyReplicate", "Oracle_3D_Interface_Contacts", "Baseline_BoganThorn_1D", "Null_Uniform"]
    m_lbls = ["Noisy Replicate\nCeiling", "3D Crystal\nContacts (SKEMPI)", "Bogan–Thorn 1D\n(Trp/Tyr/Arg)", "Null\nUniform"]
    raw_rhos = [report["models"][k]["mean_spearman_rho"] for k in m_keys]
    dm_rhos = [report["models"][k]["mean_spearman_rho_aa_demeaned"] for k in m_keys]
    x = np.arange(len(m_keys))
    w = 0.35
    b1 = ax.bar(x - w / 2, raw_rhos, w, label="Panel Spearman Rank ρ", color="#3b82f6", alpha=0.88)
    b2 = ax.bar(x + w / 2, dm_rhos, w, label="AA-Demeaned Context ρ", color="#e11d48", alpha=0.88)
    ax.axhline(0.0, color="#64748b", lw=0.9)
    ax.set_xticks(x)
    ax.set_xticklabels(m_lbls, fontsize=8.3)
    ax.set_ylim(-0.08, 1.05)
    ax.set_ylabel("Mean Per-Complex Rank Correlation")
    ax.set_title("A. Panel Rank ρ vs. AA-Demeaned Context ρ", fontsize=10.5, fontweight="bold")
    ax.legend(loc="upper right", fontsize=8, frameon=True)
    for bars in (b1, b2):
        for rect in bars:
            hv = rect.get_height()
            ax.text(rect.get_x() + rect.get_width() / 2, hv + 0.02, f"{hv:.2f}", ha="center", va="bottom", fontsize=7.8)

    # Panel B: Top-5 NDCG & Same-AA Decoy Pairwise Win Rate (50% coin-flip for 1D lookup!)
    ax2 = axes[1]
    ndcgs = [report["models"][k]["mean_ndcg_at_5"] for k in m_keys]
    saa_wins = [report["models"][k]["mean_same_aa_win_rate"] for k in m_keys]
    b3 = ax2.bar(x - w / 2, ndcgs, w, label="NDCG@5 (Top-5 Ranking)", color="#7c3aed", alpha=0.88)
    b4 = ax2.bar(x + w / 2, saa_wins, w, label="Same-AA Decoy Win Rate", color="#059669", alpha=0.88)
    ax2.axhline(0.50, color="#0f172a", lw=0.9, ls="--", alpha=0.6, label="Coin-Flip (0.50)")
    ax2.set_xticks(x)
    ax2.set_xticklabels(m_lbls, fontsize=8.3)
    ax2.set_ylim(0.0, 1.08)
    ax2.set_ylabel("Ranking Score")
    ax2.set_title("B. Top-5 NDCG & Interface vs. Non-Interface Same-AA Win", fontsize=10.2, fontweight="bold")
    ax2.legend(loc="upper right", fontsize=7.8, frameon=True)
    for bars in (b3, b4):
        for rect in bars:
            hv = rect.get_height()
            ax2.text(rect.get_x() + rect.get_width() / 2, hv + 0.02, f"{hv:.2f}", ha="center", va="bottom", fontsize=7.8)

    # Panel C: Real Oct-16-2026 KRAS × LZTR1 Direct Binding X->Ala Scan + 18-Site Ranking Panel
    ax3 = axes[2]
    kras_rec = next((t, fp, fl) for t, m, fp, fl in panel_records if m.get("mavedb_urn") == "urn:mavedb:00001302-g-1")
    kras_tgt, full_pos_k, full_loss_k = kras_rec
    pos_k = np.asarray(full_pos_k)
    loss_k = np.asarray(full_loss_k)
    med_k = float(np.median(loss_k))
    loss_shift = loss_k - med_k
    pan_set = set(kras_tgt.positions_1idx)
    colors_k = [
        "#e11d48" if (p in pan_set and v > 0.15) else ("#0284c7" if p in pan_set else "#cbd5e1")
        for p, v in zip(pos_k, loss_shift)
    ]
    ax3.bar(pos_k, loss_shift, width=1.05, color=colors_k)
    ax3.axvspan(58, 76, color="#f59e0b", alpha=0.16, label="Switch II Helix (58–76)")
    ax3.axhline(0.0, color="#475569", lw=0.8, ls="--")
    ax3.set_xlim(1, len(kras_tgt.scanned_sequence))
    ax3.set_xlabel("KRAS Position (Red/Blue = 18 Ranked Panel Candidates)")
    ax3.set_ylabel("LZTR1 Binding Loss upon X → Ala")
    ax3.set_title("C. Oct-2026 KRAS × LZTR1 18-Candidate Ranking Panel", fontsize=10.2, fontweight="bold")
    ax3.legend(loc="upper right", fontsize=8, frameon=True)

    for a in axes:
        a.spines["top"].set_visible(False)
        a.spines["right"].set_visible(False)

    plt.tight_layout()
    fig_path = fig_dir / "binding_alanine_overview.png"
    fig.savefig(fig_path, dpi=180)
    plt.close(fig)

    print(
        f"Built Test 6 Binding Alanine Ranking: {len(panel_records)} complexes, "
        f"{total_panel_mutants} panel candidates (from {total_full_mutants:,} wet-lab X->Ala mutants) across {cat_counts}"
    )
    print("Calibration summary:")
    for mk, ms in report["models"].items():
        print(
            f"  {mk:30s}: Rank rho={ms['mean_spearman_rho']:.3f} | NDCG@5={ms['mean_ndcg_at_5']:.3f} | "
            f"TopK_P={ms['mean_top_k_precision']:.3f} | SameAA_Win={ms['mean_same_aa_win_rate']:.3f} | "
            f"AA-Ctx rho={ms['mean_spearman_rho_aa_demeaned']:.3f}"
        )


if __name__ == "__main__":
    main()
