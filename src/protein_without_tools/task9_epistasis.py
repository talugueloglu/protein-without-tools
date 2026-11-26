"""Task 9: Double-Mutant Epistasis & Non-Additive Coupling Benchmark.

Tests whether frontier reasoning models (without tools) and specialist structural
models can predict true 3D energetic coupling between pairs of mutations:

    ddG_epistasis (dddG) = ddG_AB - (ddG_A + ddG_B)

where ddG > 0 denotes destabilization (dG_WT - dG_mut, kcal/mol).
By supplying the measured single-mutant effects (ddG_A, ddG_B) directly in the
prompt and matching coupled vs. additive pairs on sequence separation |i - j| and
single-mutant effect magnitude, any purely additive model, 1D lookup table, or
naive sequence-distance heuristic is mathematically held near random chance.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
import re
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
from scipy.stats import spearmanr

from protein_without_tools.alanine_scan import binary_auroc, compute_ndcg_at_k

MUT_TOKEN_RE = re.compile(r"^([ACDEFGHIKLMNPQRSTVWY])(\d+)([ACDEFGHIKLMNPQRSTVWY])$")
PAIR_CANON_RE = re.compile(
    r"([ACDEFGHIKLMNPQRSTVWY]\d+[ACDEFGHIKLMNPQRSTVWY])\s*[:+/,&]\s*([ACDEFGHIKLMNPQRSTVWY]\d+[ACDEFGHIKLMNPQRSTVWY])"
)


@dataclass(frozen=True)
class DoubleMutantRecord:
    pair_id: str          # canonical "A12V+K34E" (sorted by residue index)
    mut_a: str            # e.g. "A12V" (1-indexed)
    mut_b: str            # e.g. "K34E" (1-indexed)
    pos_a: int
    pos_b: int
    seq_sep: int          # abs(pos_b - pos_a)
    ddg_a: float          # destabilization kcal/mol (>0 destabilizing)
    ddg_b: float          # destabilization kcal/mol (>0 destabilizing)
    ddg_ab: float         # measured double-mutant destabilization kcal/mol
    dddg: float           # ddg_ab - (ddg_a + ddg_b); <0 rescuing/positive, >0 synergistic/negative
    abs_dddg: float       # |dddg|
    ca_dist: Optional[float] = None  # native 3D Ca-Ca distance (A), metadata only (never in prompt)
    is_coupled: bool = False         # True if |dddg| >= threshold


def canonicalize_double_mutant(mut_a: str, mut_b: str) -> Tuple[str, str, str, int, int]:
    """Normalize a double mutant pair into sorted 1-indexed canonical form `X12Y+Z34W`."""
    ma = mut_a.strip().upper()
    mb = mut_b.strip().upper()
    m1 = MUT_TOKEN_RE.match(ma)
    m2 = MUT_TOKEN_RE.match(mb)
    if not m1 or not m2:
        raise ValueError(f"Invalid mutation token(s): {mut_a!r}, {mut_b!r}")
    p1, p2 = int(m1.group(2)), int(m2.group(2))
    if p1 == p2:
        raise ValueError(f"Double mutant cannot target same position: {ma}, {mb}")
    if p1 > p2:
        ma, mb = mb, ma
        p1, p2 = p2, p1
    return f"{ma}+{mb}", ma, mb, p1, p2


def select_balanced_epistasis_panel(
    candidates: Sequence[DoubleMutantRecord],
    n_pairs: int = 16,
    coupled_thresh: float = 0.80,
    additive_max: float = 0.25,
    min_seq_sep: int = 4,
) -> Optional[List[DoubleMutantRecord]]:
    """Select a balanced panel of strongly coupled vs. additive double mutants.

    Anti-shortcut constraints:
      1. Requires `n_pairs // 2` strongly coupled pairs (|dddg| >= coupled_thresh)
         and `n_pairs // 2` additive decoy pairs (|dddg| <= additive_max), all with
         sequence separation >= min_seq_sep.
      2. Each coupled pair is greedily matched to an additive decoy minimizing
         difference in sequence separation |i - j| AND single-mutant magnitude
         (|ddg_a| + |ddg_b|), defeating both 1D distance and magnitude heuristics.
      3. Prefers balancing positive/rescuing (dddg < 0) and negative/synergistic
         (dddg > 0) epistasis when both signs are present.
    """
    half = n_pairs // 2
    pool = [c for c in candidates if c.seq_sep >= min_seq_sep]
    coupled_all = [c for c in pool if c.abs_dddg >= coupled_thresh]
    additive_all = [c for c in pool if c.abs_dddg <= additive_max]
    if len(coupled_all) < half or len(additive_all) < half:
        return None

    # Prefer sign diversity among coupled pairs (rescuing vs synergistic)
    pos_epi = sorted([c for c in coupled_all if c.dddg < 0], key=lambda x: -x.abs_dddg)
    neg_epi = sorted([c for c in coupled_all if c.dddg > 0], key=lambda x: -x.abs_dddg)

    chosen_coupled: List[DoubleMutantRecord] = []
    used_ids: set[str] = set()
    used_pos_pairs: set[Tuple[int, int]] = set()

    # Round-robin across positive & negative epistasis for sign balance
    # Cap duplicate exact position-pairs so panel probes diverse 3D contacts
    pi, ni = 0, 0
    while len(chosen_coupled) < half and (pi < len(pos_epi) or ni < len(neg_epi)):
        progressed = False
        for bucket_name in ("pos", "neg"):
            if len(chosen_coupled) >= half:
                break
            if bucket_name == "pos":
                while pi < len(pos_epi):
                    cand = pos_epi[pi]
                    pi += 1
                    if cand.pair_id not in used_ids and (cand.pos_a, cand.pos_b) not in used_pos_pairs:
                        chosen_coupled.append(cand)
                        used_ids.add(cand.pair_id)
                        used_pos_pairs.add((cand.pos_a, cand.pos_b))
                        progressed = True
                        break
            else:
                while ni < len(neg_epi):
                    cand = neg_epi[ni]
                    ni += 1
                    if cand.pair_id not in used_ids and (cand.pos_a, cand.pos_b) not in used_pos_pairs:
                        chosen_coupled.append(cand)
                        used_ids.add(cand.pair_id)
                        used_pos_pairs.add((cand.pos_a, cand.pos_b))
                        progressed = True
                        break
        if not progressed:
            break

    # If unique position-pairs weren't enough, allow distinct mutations on same positions
    if len(chosen_coupled) < half:
        rem = sorted([c for c in coupled_all if c.pair_id not in used_ids], key=lambda x: -x.abs_dddg)
        for cand in rem:
            if len(chosen_coupled) >= half:
                break
            chosen_coupled.append(cand)
            used_ids.add(cand.pair_id)

    if len(chosen_coupled) < half:
        return None

    # Match each coupled pair to an additive decoy minimizing |seq_sep diff| + |single-effect sum diff|
    chosen_additive: List[DoubleMutantRecord] = []
    avail_add = [c for c in additive_all if c.pair_id not in used_ids]
    for c in chosen_coupled:
        c_mag = abs(c.ddg_a) + abs(c.ddg_b)
        best_idx = None
        best_cost = float("inf")
        for idx, a in enumerate(avail_add):
            a_mag = abs(a.ddg_a) + abs(a.ddg_b)
            sep_cost = abs(c.seq_sep - a.seq_sep) / max(10.0, float(c.seq_sep))
            mag_cost = abs(c_mag - a_mag) / max(1.0, c_mag)
            pos_pen = 0.3 if (a.pos_a, a.pos_b) in used_pos_pairs else 0.0
            cost = 1.2 * sep_cost + 3.2 * mag_cost + pos_pen
            if cost < best_cost:
                best_cost = cost
                best_idx = idx
        if best_idx is None:
            return None
        pick = avail_add.pop(best_idx)
        chosen_additive.append(pick)
        used_ids.add(pick.pair_id)
        used_pos_pairs.add((pick.pos_a, pick.pos_b))

    panel = [
        DoubleMutantRecord(
            pair_id=c.pair_id, mut_a=c.mut_a, mut_b=c.mut_b,
            pos_a=c.pos_a, pos_b=c.pos_b, seq_sep=c.seq_sep,
            ddg_a=c.ddg_a, ddg_b=c.ddg_b, ddg_ab=c.ddg_ab,
            dddg=c.dddg, abs_dddg=c.abs_dddg, ca_dist=c.ca_dist,
            is_coupled=True,
        )
        for c in chosen_coupled
    ] + [
        DoubleMutantRecord(
            pair_id=a.pair_id, mut_a=a.mut_a, mut_b=a.mut_b,
            pos_a=a.pos_a, pos_b=a.pos_b, seq_sep=a.seq_sep,
            ddg_a=a.ddg_a, ddg_b=a.ddg_b, ddg_ab=a.ddg_ab,
            dddg=a.dddg, abs_dddg=a.abs_dddg, ca_dist=a.ca_dist,
            is_coupled=False,
        )
        for a in chosen_additive
    ]
    # Deterministic ascending sequence order so prompt order leaks zero signal
    panel.sort(key=lambda r: (r.pos_a, r.pos_b, r.pair_id))
    return panel


def build_epistasis_prompt(
    sequence: str,
    panel: Sequence[DoubleMutantRecord],
) -> str:
    """Build a zero-leakage prompt for double-mutant epistasis ranking & sign prediction."""
    n = len(panel)
    L = len(sequence)
    pair_lines = [
        f'  {idx + 1:>2d}. "{r.pair_id}"  (single-mutant ddG: {r.mut_a} = {r.ddg_a:+.2f}, {r.mut_b} = {r.ddg_b:+.2f} kcal/mol)'
        for idx, r in enumerate(panel)
    ]
    ex1 = panel[0].pair_id if panel else "A12V+K34E"
    ex2 = panel[1].pair_id if len(panel) > 1 else "L18A+I45V"
    return (
        f"You are predicting 3D thermodynamic epistasis (non-additive energetic coupling) "
        f"between pairs of point mutations in an anonymized monomeric protein of length L = {L} residues.\n\n"
        f"Sequence (1-indexed, length {L}):\n{sequence}\n\n"
        f"Sign convention: ddG > 0 means destabilizing (loss of folding stability, kcal/mol); "
        f"ddG < 0 means stabilizing.\n"
        f"Epistatic coupling is defined as: dddG = ddG_AB - (ddG_A + ddG_B).\n"
        f"  - dddG < 0 ('rescuing' / positive epistasis): double mutant is MORE stable than the sum of singles.\n"
        f"  - dddG > 0 ('synergistic' / negative epistasis): double mutant is MORE destabilizing than the sum of singles.\n"
        f"  - dddG ~ 0 ('additive'): the two mutations act independently.\n\n"
        f"Candidate Double Mutants ({n} total, listed in ascending sequence order):\n"
        + "\n".join(pair_lines)
        + "\n\n"
        f"Task:\n"
        f"1. Rank ALL {n} double-mutant pairs from #1 (STRONGEST epistatic coupling, largest |dddG|) "
        f"down to #{n} (MOST purely additive, |dddG| near 0).\n"
        f"2. For each pair, predict the direction of coupling: \"rescuing\" (dddG < 0), "
        f"\"synergistic\" (dddG > 0), or \"additive\" (dddG ~ 0).\n"
        f"Note: coupled and additive pairs are matched on sequence separation and single-mutant magnitudes, "
        f"so rank by 3D structural proximity and side-chain interaction physics rather than 1D distance.\n\n"
        f"Return ONLY a single compact JSON object with exact schema:\n"
        f'{{"ranking": ["{ex1}", "{ex2}", ...], "signs": {{"{ex1}": "rescuing", "{ex2}": "additive", ...}}}}\n'
        f"ordered strictly from #1 (strongest |dddG|) to #{n} (most additive), containing all {n} pairs "
        f"with no text outside JSON."
    )


def _normalize_pair_str(raw: str, valid_map: Dict[str, str]) -> Optional[str]:
    s = str(raw).strip().upper().replace("->", "").replace(" ", "")
    if s in valid_map:
        return valid_map[s]
    m = PAIR_CANON_RE.search(str(raw).upper())
    if m:
        try:
            canon, _, _, _, _ = canonicalize_double_mutant(m.group(1), m.group(2))
            if canon in valid_map:
                return valid_map[canon]
        except ValueError:
            pass
    return None


def parse_epistasis_response(
    raw_text: str,
    valid_pair_ids: Sequence[str],
) -> Tuple[List[str], Dict[str, str], bool]:
    """Parse JSON ranking + sign dictionary cleanly from an LLM response."""
    valid_map: Dict[str, str] = {}
    for pid in valid_pair_ids:
        a, b = pid.split("+")
        valid_map[pid.upper()] = pid
        valid_map[f"{a}:{b}".upper()] = pid
        valid_map[f"{b}+{a}".upper()] = pid
        valid_map[f"{b}:{a}".upper()] = pid

    text = raw_text.strip()
    m_fence = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    json_str = m_fence.group(1) if m_fence else text
    if not json_str.startswith("{"):
        i0, i1 = json_str.find("{"), json_str.rfind("}")
        if i0 != -1 and i1 > i0:
            json_str = json_str[i0 : i1 + 1]

    ranking: List[str] = []
    signs: Dict[str, str] = {}
    parsed_ok = False
    try:
        obj = json.loads(json_str)
        if isinstance(obj, dict):
            raw_rank = obj.get("ranking") or obj.get("ranked_pairs") or []
            seen: set[str] = set()
            for item in raw_rank:
                tok = _normalize_pair_str(item, valid_map)
                if tok and tok not in seen:
                    ranking.append(tok)
                    seen.add(tok)
            raw_signs = obj.get("signs") or {}
            if isinstance(raw_signs, dict):
                for k, v in raw_signs.items():
                    tok = _normalize_pair_str(k, valid_map)
                    sv = str(v).strip().lower()
                    if tok:
                        if sv in ("rescuing", "positive", "alleviating", "negative_dddg", "-"):
                            signs[tok] = "rescuing"
                        elif sv in ("synergistic", "negative", "destabilizing", "positive_dddg", "+"):
                            signs[tok] = "synergistic"
                        else:
                            signs[tok] = "additive"
            parsed_ok = len(ranking) >= max(4, len(valid_pair_ids) // 2)
    except Exception:
        parsed_ok = False

    # Fallback token scan if JSON failed
    if not ranking:
        seen = set()
        for m in PAIR_CANON_RE.finditer(raw_text.upper()):
            tok = _normalize_pair_str(f"{m.group(1)}+{m.group(2)}", valid_map)
            if tok and tok not in seen:
                ranking.append(tok)
                seen.add(tok)
        parsed_ok = len(ranking) >= max(4, len(valid_pair_ids) // 2)

    return ranking, signs, parsed_ok


def grade_epistasis_panel(
    panel: Sequence[DoubleMutantRecord],
    predicted_ranking: Sequence[str],
    predicted_signs: Optional[Dict[str, str]] = None,
    predicted_scores: Optional[Dict[str, float]] = None,
) -> Dict[str, Any]:
    """Score a predicted epistasis ranking (and optional signed scores) on one protein panel."""
    by_id = {r.pair_id: r for r in panel}
    n = len(panel)
    if predicted_scores is not None:
        scores = {pid: float(predicted_scores.get(pid, 0.0)) for pid in by_id}
    else:
        scores = {}
        seen = set()
        for rank_idx, pid in enumerate(predicted_ranking):
            if pid in by_id and pid not in seen:
                scores[pid] = float(n - rank_idx)
                seen.add(pid)
        for pid in by_id:
            if pid not in seen:
                scores[pid] = 0.0

    pids = list(by_id.keys())
    y_true_abs = np.array([by_id[p].abs_dddg for p in pids], dtype=float)
    y_pred_abs = np.array([scores[p] for p in pids], dtype=float)
    y_bin = [1 if by_id[p].is_coupled else 0 for p in pids]

    if np.std(y_pred_abs) < 1e-12 or np.std(y_true_abs) < 1e-12:
        rho = 0.0
    else:
        r_val = spearmanr(y_pred_abs, y_true_abs).statistic
        rho = 0.0 if (r_val is None or math.isnan(r_val)) else float(r_val)

    auroc = float(binary_auroc(y_bin, [scores[p] for p in pids]))

    # NDCG@5 over true |dddG|
    ndcg5 = float(compute_ndcg_at_k(y_true_abs, y_pred_abs, k=5))

    # Sign accuracy on strongly coupled pairs (rescuing vs synergistic)
    sign_correct = 0
    sign_total = 0
    ps = predicted_signs or {}
    for p in pids:
        rec = by_id[p]
        if rec.is_coupled:
            true_sign = "rescuing" if rec.dddg < 0 else "synergistic"
            pred_sign = ps.get(p)
            if pred_sign in ("rescuing", "synergistic"):
                sign_total += 1
                if pred_sign == true_sign:
                    sign_correct += 1
    sign_acc = (sign_correct / sign_total) if sign_total > 0 else None

    return {
        "n_pairs": n,
        "coverage": len([p for p in predicted_ranking if p in by_id]) / float(max(1, n)),
        "spearman_rho_abs_dddg": round(rho, 4),
        "coupling_auroc": round(auroc, 4),
        "ndcg_at_5": round(ndcg5, 4),
        "sign_accuracy": round(sign_acc, 4) if sign_acc is not None else None,
        "sign_evaluated": sign_total,
    }
