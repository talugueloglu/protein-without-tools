"""Task 10: Small-Molecule, Cofactor & Metal Binding-Site Ranking Benchmark.

Tests whether frontier reasoning models (without tools) and all-atom co-folding
specialists (Boltz-2) can identify which residues directly contact a bound small
molecule, cofactor, or catalytic metal ion — strictly controlling for 1D amino-acid
identity shortcuts by pairing every binding-site residue with a same-AA non-pocket decoy.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
import json
import math
import re
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy.stats import spearmanr

from protein_without_tools.alanine_scan import binary_auroc, compute_ndcg_at_k

RES_TOKEN_RE = re.compile(r"^([ACDEFGHIKLMNPQRSTVWY])(\d+)$")
RES_TOKEN_ALT_RE = re.compile(r"^(\d+)([ACDEFGHIKLMNPQRSTVWY])$")

# Common crystallographic artifacts / buffers / cryoprotectants to exclude
CRYSTAL_ARTIFACT_COMP_IDS = {
    "HOH", "DOD", "WAT", "SO4", "PO4", "GOL", "EDO", "PEG", "PGE", "PG4",
    "1PE", "P6G", "DMS", "DMSO", "ACT", "ACY", "TRS", "MES", "HEP", "EPE",
    "CIT", "FMT", "MPD", "BME", "DTT", "CL", "BR", "IOD", "NA", "K", "CS",
    "NH4", "NO3", "CO3", "SCN", "ACE", "NME", "MSE", "SEP", "TPO", "PTR",
    "UNX", "UNL", "BCT", "IMD", "IPA", "EOH", "MOH", "TAR", "SUC", "GLC",
    "NAG", "MAN", "BMA", "FUC", "GAL", "NDG", "SO3", "SUL", "AZI", "POP",
}

CATALYTIC_METALS = {
    "ZN": "Zn2+",
    "MG": "Mg2+",
    "MN": "Mn2+",
    "FE": "Fe2+/Fe3+",
    "FE2": "Fe2+",
    "CU": "Cu2+",
    "CU1": "Cu+",
    "CA": "Ca2+",
    "CO": "Co2+",
    "NI": "Ni2+",
}

COFACTOR_COMP_IDS = {
    "ATP", "ADP", "AMP", "GTP", "GDP", "ANP", "ACP", "GNP",
    "NAD", "NAP", "NDP", "NAI", "FAD", "FMN",
    "HEM", "HEC", "HEA", "SAM", "SAH", "PLP", "TPP", "COA", "ACO", "BTN",
}


@dataclass(frozen=True)
class SiteCandidate:
    res_id: str          # canonical "H142" (1-indexed)
    aa: str              # single-letter AA
    pos: int             # 1-based sequence index
    min_dist_angstrom: float  # minimum heavy-atom distance to ligand/metal (A)
    is_contact: bool     # True if min_dist <= contact_cutoff


def select_same_aa_site_panel(
    candidates: Sequence[SiteCandidate],
    n_total: int = 16,
    contact_cutoff: float = 4.2,
    decoy_min_dist: float = 10.0,
) -> Optional[List[SiteCandidate]]:
    """Build a strictly same-AA-balanced panel of `n_total` residues (`n_total // 2`
    binding-site contacts paired 1-to-1 by amino-acid letter with non-pocket decoys).

    Because every AA type appears equally often as contact and decoy, any 1D
    residue-letter frequency table (e.g., 'always pick His/Cys/Asp/Trp') is
    mathematically locked to 50.0% same-AA win rate and ~0.50 AUROC.
    """
    half = n_total // 2
    by_aa_pos: Dict[str, List[SiteCandidate]] = defaultdict(list)
    by_aa_neg: Dict[str, List[SiteCandidate]] = defaultdict(list)

    for c in candidates:
        if c.min_dist_angstrom <= contact_cutoff:
            by_aa_pos[c.aa].append(c)
        elif c.min_dist_angstrom >= decoy_min_dist:
            by_aa_neg[c.aa].append(c)

    # Sort contacts closest-first; sort decoys moderately far (spread across sequence)
    for aa in by_aa_pos:
        by_aa_pos[aa].sort(key=lambda x: (x.min_dist_angstrom, x.pos))
    for aa in by_aa_neg:
        by_aa_neg[aa].sort(key=lambda x: (-x.min_dist_angstrom, x.pos))

    paired_pos: List[SiteCandidate] = []
    paired_neg: List[SiteCandidate] = []

    # Round-robin across AA types with both contacts and decoys for AA diversity
    aa_order = sorted(
        [aa for aa in by_aa_pos if by_aa_neg.get(aa)],
        key=lambda a: by_aa_pos[a][0].min_dist_angstrom,
    )
    cursors = {aa: 0 for aa in aa_order}
    while len(paired_pos) < half:
        progressed = False
        for aa in aa_order:
            if len(paired_pos) >= half:
                break
            k = cursors[aa]
            if k < len(by_aa_pos[aa]) and k < len(by_aa_neg[aa]):
                p = by_aa_pos[aa][k]
                # Pick decoy of same AA farthest in 1D from p to avoid local-neighborhood leakage
                d_pool = [d for d in by_aa_neg[aa] if d.res_id not in {x.res_id for x in paired_neg}]
                if not d_pool:
                    continue
                d_pick = max(d_pool, key=lambda d: (abs(d.pos - p.pos) >= 12, d.min_dist_angstrom))
                paired_pos.append(SiteCandidate(p.res_id, p.aa, p.pos, p.min_dist_angstrom, True))
                paired_neg.append(SiteCandidate(d_pick.res_id, d_pick.aa, d_pick.pos, d_pick.min_dist_angstrom, False))
                cursors[aa] = k + 1
                progressed = True
        if not progressed:
            break

    if len(paired_pos) < half:
        return None

    panel = paired_pos + paired_neg
    panel.sort(key=lambda x: x.pos)
    return panel


def build_ligand_site_prompt(
    sequence: str,
    ligand_category: str,
    ligand_descriptor: str,
    panel: Sequence[SiteCandidate],
) -> str:
    """Build a zero-leakage prompt for small-molecule / cofactor / metal binding-site ranking."""
    L = len(sequence)
    n = len(panel)
    toks = [f'"{c.res_id}"' for c in panel]
    if ligand_category == "metal_ion":
        lig_block = f"Bound Metal Ion:\n  {ligand_descriptor}"
    else:
        lig_block = f"Bound Small Molecule / Cofactor (canonical SMILES):\n  {ligand_descriptor}"

    ex1 = panel[0].res_id if panel else "H42"
    ex2 = panel[1].res_id if len(panel) > 1 else "D88"
    return (
        f"You are predicting the 3D ligand/metal binding site of an anonymized protein "
        f"from its primary amino-acid sequence and the bound chemical species alone.\n\n"
        f"Protein Sequence (1-indexed, length L = {L}):\n{sequence}\n\n"
        f"{lig_block}\n\n"
        f"Candidate Residues ({n} total, listed in ascending sequence order):\n"
        f"[{', '.join(toks)}]\n\n"
        f"Task:\n"
        f"Rank ALL {n} candidate residues from #1 (MOST likely direct binding-site / coordinating residue "
        f"within 4.2 A heavy-atom distance of the bound ligand or metal ion) down to #{n} "
        f"(LEAST likely — distant non-pocket residue > 10 A away).\n"
        f"Note: the panel contains an exact 1-to-1 match of true binding-site residues and non-pocket decoys "
        f"of the SAME amino-acid types, so rank by 3D fold and pocket context rather than residue identity alone.\n\n"
        f"Return ONLY a single compact JSON object with exact schema:\n"
        f'{{"ranking": ["{ex1}", "{ex2}", ...]}}\n'
        f"ordered strictly from #1 (closest binding-site contact) to #{n} (distant decoy), "
        f"containing all {n} residues with no text outside JSON."
    )


def _normalize_res_token(raw: str, valid_set: Dict[str, str]) -> Optional[str]:
    s = str(raw).strip().upper()
    if s in valid_set:
        return valid_set[s]
    m1 = RES_TOKEN_RE.match(s)
    if m1 and f"{m1.group(1)}{m1.group(2)}" in valid_set:
        return valid_set[f"{m1.group(1)}{m1.group(2)}"]
    m2 = RES_TOKEN_ALT_RE.match(s)
    if m2 and f"{m2.group(2)}{m2.group(1)}" in valid_set:
        return valid_set[f"{m2.group(2)}{m2.group(1)}"]
    return None


def parse_site_ranking_response(
    raw_text: str,
    valid_res_ids: Sequence[str],
) -> Tuple[List[str], bool]:
    valid_map = {r.upper(): r for r in valid_res_ids}
    text = raw_text.strip()
    m_fence = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    json_str = m_fence.group(1) if m_fence else text
    if not json_str.startswith("{"):
        i0, i1 = json_str.find("{"), json_str.rfind("}")
        if i0 != -1 and i1 > i0:
            json_str = json_str[i0 : i1 + 1]

    ranking: List[str] = []
    parsed_ok = False
    try:
        obj = json.loads(json_str)
        if isinstance(obj, dict):
            raw_list = obj.get("ranking") or obj.get("residues") or []
            seen = set()
            for item in raw_list:
                tok = _normalize_res_token(item, valid_map)
                if tok and tok not in seen:
                    ranking.append(tok)
                    seen.add(tok)
            parsed_ok = len(ranking) >= max(4, len(valid_res_ids) // 2)
    except Exception:
        parsed_ok = False

    if not ranking:
        seen = set()
        for m in re.finditer(r"\b([ACDEFGHIKLMNPQRSTVWY]\d+|\d+[ACDEFGHIKLMNPQRSTVWY])\b", raw_text.upper()):
            tok = _normalize_res_token(m.group(1), valid_map)
            if tok and tok not in seen:
                ranking.append(tok)
                seen.add(tok)
        parsed_ok = len(ranking) >= max(4, len(valid_res_ids) // 2)

    return ranking, parsed_ok


def grade_site_ranking_panel(
    panel: Sequence[SiteCandidate],
    predicted_ranking: Sequence[str],
    predicted_scores: Optional[Dict[str, float]] = None,
) -> Dict[str, Any]:
    """Grade binding-site ranking with strict same-AA win rate, AUROC, NDCG@5, and Top-5 precision."""
    by_id = {c.res_id: c for c in panel}
    n = len(panel)
    if predicted_scores is not None:
        scores = {rid: float(predicted_scores.get(rid, 0.0)) for rid in by_id}
    else:
        scores = {}
        seen = set()
        for r_idx, rid in enumerate(predicted_ranking):
            if rid in by_id and rid not in seen:
                scores[rid] = float(n - r_idx)
                seen.add(rid)
        for rid in by_id:
            if rid not in seen:
                scores[rid] = 0.0

    rids = list(by_id.keys())
    y_bin = [1 if by_id[r].is_contact else 0 for r in rids]
    y_pred = np.array([scores[r] for r in rids], dtype=float)
    # Continuous proximity target: max(0, 12.0 - min_dist) so closer contacts score higher
    y_prox = np.array([max(0.0, 12.0 - by_id[r].min_dist_angstrom) for r in rids], dtype=float)

    auroc = float(binary_auroc(y_bin, [scores[r] for r in rids]))
    ndcg5 = float(compute_ndcg_at_k(y_prox, y_pred, k=5))
    ordered = sorted(rids, key=lambda r: (-scores[r], r))
    top5_prec = sum(1 for r in ordered[:5] if by_id[r].is_contact) / 5.0

    # Same-AA win rate: across all (contact, decoy) pairs sharing the exact same AA letter
    wins = 0.0
    total_pairs = 0
    by_aa: Dict[str, List[SiteCandidate]] = defaultdict(list)
    for c in panel:
        by_aa[c.aa].append(c)
    for aa, grp in by_aa.items():
        pos_g = [c for c in grp if c.is_contact]
        neg_g = [c for c in grp if not c.is_contact]
        for p in pos_g:
            for d in neg_g:
                total_pairs += 1
                sp, sd = scores[p.res_id], scores[d.res_id]
                if sp > sd:
                    wins += 1.0
                elif sp == sd:
                    wins += 0.5
    same_aa_win = (wins / total_pairs) if total_pairs > 0 else 0.5

    if np.std(y_pred) < 1e-12 or np.std(y_prox) < 1e-12:
        rho = 0.0
    else:
        rv = spearmanr(y_pred, y_prox).statistic
        rho = 0.0 if (rv is None or math.isnan(rv)) else float(rv)

    return {
        "n_candidates": n,
        "coverage": len([r for r in predicted_ranking if r in by_id]) / float(max(1, n)),
        "same_aa_win_rate": round(same_aa_win, 4),
        "auroc": round(auroc, 4),
        "ndcg_at_5": round(ndcg5, 4),
        "top5_precision": round(top5_prec, 4),
        "spearman_rho_prox": round(rho, 4),
    }
