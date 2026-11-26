"""Task 13: Wet-Lab Developability Ranking Benchmark.

Tests whether frontier reasoning models (without tools) and biophysical / PLM
specialists can rank anonymized proteins and therapeutic antibodies on experimental
wet-lab developability properties that structure confidence (pLDDT) alone misses:

  - Thermostability (Tm / melting temperature, higher is better)
  - Polyreactivity / non-specific binding (lower is better)
  - Aggregation / self-interaction (HIC / AC-SINS / CIC, lower is better)
  - Expression / solubility yield (higher is better)

Every antibody INN name, clinical drug name, PDB ID, and germline label is stripped.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
import re
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy.stats import spearmanr

from protein_without_tools.alanine_scan import compute_ndcg_at_k

CAND_ID_RE = re.compile(r"\b(C\d{2}|SEQ_?\d{1,2})\b", re.I)


@dataclass(frozen=True)
class DevelopabilityCandidate:
    cand_id: str          # anonymized "C01".."C10"
    sequence: str         # VH/VL concatenated with "/" for antibodies, or single chain
    value: float          # raw measured assay value
    desirability: float   # oriented so HIGHER = MORE DEVELOPABLE (higher Tm/expr, lower poly/agg)


PROPERTY_SPECS: Dict[str, Dict[str, str]] = {
    "thermostability": {
        "label": "Thermostability (Tm)",
        "higher_desc": "HIGHEST thermal melting temperature (most thermostable)",
        "lower_desc": "LOWEST melting temperature (least thermostable)",
    },
    "polyreactivity": {
        "label": "Low Polyreactivity (Non-Specific Binding)",
        "higher_desc": "CLEANEST / LOWEST non-specific polyreactivity (most developable)",
        "lower_desc": "HIGHEST sticky non-specific polyreactivity (worst developability)",
    },
    "aggregation": {
        "label": "Low Aggregation / Self-Association",
        "higher_desc": "LOWEST hydrophobic aggregation / self-interaction (most developable)",
        "lower_desc": "HIGHEST aggregation / self-interaction propensity (worst developability)",
    },
    "expression": {
        "label": "Expression & Solubility Yield",
        "higher_desc": "HIGHEST recombinant expression titer / solubility",
        "lower_desc": "LOWEST expression yield / solubility",
    },
}


def build_developability_prompt(
    property_type: str,
    modality: str,
    panel: Sequence[DevelopabilityCandidate],
) -> str:
    """Build a zero-leakage developability ranking prompt for anonymized sequences."""
    spec = PROPERTY_SPECS[property_type]
    n = len(panel)
    if modality == "antibody_fv":
        mod_note = "Antibody variable domains shown as `VH / VL` (heavy chain / light chain)."
    else:
        mod_note = "Single-chain monomeric proteins."

    seq_lines = [f"  {c.cand_id}: {c.sequence}" for c in panel]
    c1 = panel[0].cand_id if panel else "C01"
    c2 = panel[1].cand_id if len(panel) > 1 else "C02"
    return (
        f"You are evaluating experimental wet-lab developability ({spec['label']}) "
        f"across {n} anonymized protein candidates from sequence alone.\n\n"
        f"Modality: {mod_note}\n\n"
        f"Candidates ({n} total, anonymized IDs):\n"
        + "\n".join(seq_lines)
        + "\n\n"
        f"Task:\n"
        f"Rank ALL {n} candidates from #1 ({spec['higher_desc']}) down to #{n} ({spec['lower_desc']}).\n"
        f"Base your ranking on biophysical sequence determinants (surface hydrophobic patches, "
        f"net charge/pI asymmetry, CDR aromatic/positively charged clusters, core packing, and "
        f"aggregation-prone motifs).\n\n"
        f"Return ONLY a single compact JSON object with exact schema:\n"
        f'{{"ranking": ["{c1}", "{c2}", ...]}}\n'
        f"ordered strictly from #1 (best developability) to #{n} (worst developability), "
        f"containing all {n} candidate IDs with no text outside JSON."
    )


def parse_developability_response(
    raw_text: str,
    valid_ids: Sequence[str],
) -> Tuple[List[str], bool]:
    valid_map = {v.upper(): v for v in valid_ids}
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
            raw_list = obj.get("ranking") or obj.get("candidates") or []
            seen = set()
            for item in raw_list:
                tok = str(item).strip().upper()
                if tok in valid_map and valid_map[tok] not in seen:
                    ranking.append(valid_map[tok])
                    seen.add(valid_map[tok])
            parsed_ok = len(ranking) >= max(4, len(valid_ids) // 2)
    except Exception:
        parsed_ok = False

    if not ranking:
        seen = set()
        for m in CAND_ID_RE.finditer(raw_text.upper()):
            tok = m.group(1).upper()
            if tok in valid_map and valid_map[tok] not in seen:
                ranking.append(valid_map[tok])
                seen.add(valid_map[tok])
        parsed_ok = len(ranking) >= max(4, len(valid_ids) // 2)

    return ranking, parsed_ok


def grade_developability_panel(
    panel: Sequence[DevelopabilityCandidate],
    predicted_ranking: Sequence[str],
    predicted_scores: Optional[Dict[str, float]] = None,
) -> Dict[str, Any]:
    by_id = {c.cand_id: c for c in panel}
    n = len(panel)
    if predicted_scores is not None:
        scores = {cid: float(predicted_scores.get(cid, 0.0)) for cid in by_id}
    else:
        scores = {}
        seen = set()
        for idx, cid in enumerate(predicted_ranking):
            if cid in by_id and cid not in seen:
                scores[cid] = float(n - idx)
                seen.add(cid)
        for cid in by_id:
            if cid not in seen:
                scores[cid] = 0.0

    cids = list(by_id.keys())
    y_true = np.array([by_id[c].desirability for c in cids], dtype=float)
    y_pred = np.array([scores[c] for c in cids], dtype=float)

    if np.std(y_pred) < 1e-12 or np.std(y_true) < 1e-12:
        rho = 0.0
    else:
        rv = spearmanr(y_pred, y_true).statistic
        rho = 0.0 if (rv is None or math.isnan(rv)) else float(rv)

    # Shift desirability to >= 0 for NDCG@5
    y_shift = y_true - float(np.min(y_true))
    ndcg5 = float(compute_ndcg_at_k(y_shift, y_pred, k=min(5, n)))

    # Concordance (pairwise accuracy across all distinct pairs with |dy| > 1e-6)
    conc = 0.0
    tot = 0
    for i in range(n):
        for j in range(i + 1, n):
            dy = y_true[i] - y_true[j]
            if abs(dy) <= 1e-6:
                continue
            tot += 1
            dp = y_pred[i] - y_pred[j]
            if dp * dy > 0:
                conc += 1.0
            elif dp == 0:
                conc += 0.5
    pair_acc = (conc / tot) if tot > 0 else 0.5

    return {
        "n_candidates": n,
        "coverage": len([c for c in predicted_ranking if c in by_id]) / float(max(1, n)),
        "spearman_rho": round(rho, 4),
        "ndcg_at_5": round(ndcg5, 4),
        "pairwise_accuracy": round(pair_acc, 4),
    }
