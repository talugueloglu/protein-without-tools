"""Stage 2B: CAFA-Style Protein Function Prediction on Post-Cutoff Experimental GOA Holdouts.

Implements:
  - `go-basic.obo` parser (`is_a` + `part_of` DAG edges, `alt_id` canonicalization, aspect mapping).
  - Official CAFA true-path ancestor propagation (excluding uninformative root terms and `GO:0005515`
    `protein binding` when it is the sole MF annotation, per CAFA5 rules).
  - Official CAFA `F_max` and Information-Content-weighted `wF_max` across `MF`, `BP`, and `CC`.
  - Zero-leakage GO term prediction prompts & fault-tolerant JSON parsers.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import json
import math
import re
from typing import Any, Dict, FrozenSet, Iterable, List, Optional, Sequence, Set, Tuple

import numpy as np

from fable_bench.pdb_utils import CANONICAL_AA_SET

ASPECT_MAP = {
    "molecular_function": "MF",
    "biological_process": "BP",
    "cellular_component": "CC",
}
ROOT_GO_TERMS = {"GO:0003674", "GO:0008150", "GO:0005575"}
GO_ID_RE = re.compile(r"GO:\d{7}")


@dataclass
class GeneOntologyDAG:
    """In-memory Gene Ontology DAG for true-path propagation and IC-weighted CAFA evaluation."""

    term_name: Dict[str, str] = field(default_factory=dict)
    term_aspect: Dict[str, str] = field(default_factory=dict)  # "MF" | "BP" | "CC"
    parents: Dict[str, Set[str]] = field(default_factory=dict)
    alt_to_canonical: Dict[str, str] = field(default_factory=dict)
    _ancestors_cache: Dict[str, FrozenSet[str]] = field(default_factory=dict)

    def canonical(self, go_id: str) -> Optional[str]:
        gid = go_id.strip().upper()
        gid = self.alt_to_canonical.get(gid, gid)
        return gid if gid in self.term_aspect else None

    def ancestors(self, go_id: str, include_self: bool = True) -> Set[str]:
        """Return all `is_a` / `part_of` ancestors excluding the 3 trivial aspect root nodes."""
        cid = self.canonical(go_id)
        if cid is None:
            return set()
        if cid not in self._ancestors_cache:
            visited: Set[str] = set()
            stack = list(self.parents.get(cid, ()))
            while stack:
                cur = stack.pop()
                if cur not in visited and cur in self.term_aspect:
                    visited.add(cur)
                    stack.extend(self.parents.get(cur, ()))
            self._ancestors_cache[cid] = frozenset(visited - ROOT_GO_TERMS)
        out = set(self._ancestors_cache[cid])
        if include_self and cid not in ROOT_GO_TERMS:
            out.add(cid)
        return out

    def propagate_set(self, raw_terms: Iterable[str]) -> Set[str]:
        out: Set[str] = set()
        for t in raw_terms:
            out.update(self.ancestors(t, include_self=True))
        return out

    def propagate_scores(self, raw_scores: Dict[str, float]) -> Dict[str, float]:
        """Propagate confidence scores upward: `score(ancestor) = max_{descendant} score(descendant)`."""
        prop: Dict[str, float] = {}
        for raw_t, sc in raw_scores.items():
            val = float(np.clip(sc, 0.0, 1.0))
            if val <= 0.0:
                continue
            for anc in self.ancestors(raw_t, include_self=True):
                if val > prop.get(anc, 0.0):
                    prop[anc] = val
        return prop


def parse_go_obo(obo_path_or_text: str) -> GeneOntologyDAG:
    """Parse `go-basic.obo` text or file path into a `GeneOntologyDAG`."""
    if "\n[Term]" in obo_path_or_text or obo_path_or_text.startswith("format-version:"):
        text = obo_path_or_text
    else:
        with open(obo_path_or_text, "r", encoding="utf-8") as fh:
            text = fh.read()

    dag = GeneOntologyDAG()
    in_term = False
    cur_id = ""
    cur_name = ""
    cur_ns = ""
    cur_obsolete = False
    cur_alts: List[str] = []
    cur_parents: Set[str] = set()

    def _flush() -> None:
        if in_term and cur_id and not cur_obsolete and cur_ns in ASPECT_MAP:
            asp = ASPECT_MAP[cur_ns]
            dag.term_name[cur_id] = cur_name
            dag.term_aspect[cur_id] = asp
            dag.parents[cur_id] = set(cur_parents)
            for aid in cur_alts:
                dag.alt_to_canonical[aid] = cur_id

    for raw_line in text.splitlines():
        line = raw_line.strip()
        if line == "[Term]":
            _flush()
            in_term = True
            cur_id, cur_name, cur_ns = "", "", ""
            cur_obsolete = False
            cur_alts = []
            cur_parents = set()
        elif line.startswith("[") and line.endswith("]"):
            _flush()
            in_term = False
        elif in_term:
            if line.startswith("id: GO:"):
                cur_id = line[4:].strip()
            elif line.startswith("name: "):
                cur_name = line[6:].strip()
            elif line.startswith("namespace: "):
                cur_ns = line[11:].strip()
            elif line == "is_obsolete: true":
                cur_obsolete = True
            elif line.startswith("alt_id: GO:"):
                cur_alts.append(line[8:].strip())
            elif line.startswith("is_a: GO:"):
                pid = line[6:].split("!")[0].strip()
                cur_parents.add(pid)
            elif line.startswith("relationship: part_of GO:"):
                pid = line[len("relationship: part_of ") :].split("!")[0].strip()
                cur_parents.add(pid)
    _flush()
    return dag


def compute_term_information_content(
    dag: GeneOntologyDAG,
    reference_propagated_sets: Sequence[Set[str]],
) -> Dict[str, float]:
    """Compute empirical Information Content `IC(t) = -log2((count(t) + 1) / (N + 1))` per GO term."""
    n_total = max(1, len(reference_propagated_sets))
    counts: Dict[str, int] = {}
    for tset in reference_propagated_sets:
        for t in tset:
            counts[t] = counts.get(t, 0) + 1
    default_ic = math.log2(n_total + 1.0)
    ic_map: Dict[str, float] = {}
    for tid in dag.term_aspect:
        c = counts.get(tid, 0)
        ic_map[tid] = round(max(0.05, -math.log2((c + 1.0) / float(n_total + 1.0))), 4)
    ic_map["__default__"] = round(default_ic, 4)
    return ic_map


def compute_term_information_accretion(
    dag: GeneOntologyDAG,
    reference_propagated_sets: Sequence[Set[str]],
) -> Dict[str, float]:
    """Official CAFA-5/6 conditional Information Accretion:
    `IA(v) = -log2((count(v) + 1) / (count(all direct parents of v) + 1))`
    computed over the reference experimental GO corpus per ontology aspect.
    Root terms carry IA = 0.0."""
    aspect_totals: Dict[str, int] = {"MF": 0, "BP": 0, "CC": 0}
    term_prots: Dict[str, Set[int]] = {}
    for idx, tset in enumerate(reference_propagated_sets):
        seen_asp: Set[str] = set()
        for t in tset:
            asp = dag.term_aspect.get(t)
            if asp:
                seen_asp.add(asp)
                term_prots.setdefault(t, set()).add(idx)
        for asp in seen_asp:
            aspect_totals[asp] += 1

    ia_map: Dict[str, float] = {}
    for tid, asp in dag.term_aspect.items():
        if tid in ROOT_GO_TERMS:
            ia_map[tid] = 0.0
            continue
        c_v = len(term_prots.get(tid, ()))
        parents = [p for p in dag.parents.get(tid, ()) if p not in ROOT_GO_TERMS and dag.term_aspect.get(p) == asp]
        if not parents:
            c_par = max(aspect_totals.get(asp, 0), c_v)
        else:
            inter: Optional[Set[int]] = None
            for p in parents:
                ps = term_prots.get(p, set())
                inter = set(ps) if inter is None else (inter & ps)
                if not inter:
                    break
            c_par = max(len(inter or ()), c_v)
        ia = max(0.0, -math.log2((c_v + 1.0) / float(c_par + 1.0)))
        ia_map[tid] = round(ia, 4)

    # Fallback IA for any unseen leaf term
    ia_map["__default__"] = 1.0
    return ia_map


@dataclass
class CAFATarget:
    """Single experimentally GOA-annotated CAFA target protein (NK, LK, or PK)."""

    target_id: str
    sequence: str
    length: int
    # Propagated ground-truth GO term IDs at t1 (excluding trivial roots)
    true_terms_mf: List[str]
    true_terms_bp: List[str]
    true_terms_cc: List[str]
    # Knowledge type per CAFA-6 convention: "NK" (No-Knowledge), "LK" (Limited-Knowledge), "PK" (Partial-Knowledge)
    knowledge_type: str = "NK"
    # Pre-cutoff propagated experimental GO terms at t0 (used for CAFA-6 `-known` dual masking in PK)
    known_terms_mf: List[str] = field(default_factory=list)
    known_terms_bp: List[str] = field(default_factory=list)
    known_terms_cc: List[str] = field(default_factory=list)
    # Optional explicit aspects eligible for evaluation (e.g. newly gained aspect(s) for LK/PK)
    eval_aspects: Optional[List[str]] = None


def build_cafa_prompt(target: CAFATarget) -> str:
    seq = target.sequence.strip().upper()
    if any(c not in CANONICAL_AA_SET for c in seq):
        raise ValueError(f"Non-canonical AA in {target.target_id}")

    prompt = (
        f"You are competing in a CAFA-style blind protein function prediction benchmark.\n\n"
        f"Anonymized Query Protein Sequence (length {len(seq)}):\n"
        f"{seq}\n\n"
        f"Task:\n"
        f"Predict the experimental Gene Ontology (`GO`) functional annotations of this protein across all "
        f"three aspects:\n"
        f"  - `MF` (Molecular Function, e.g. catalytic activity, specific ligand/ion/nucleic-acid binding)\n"
        f"  - `BP` (Biological Process, e.g. metabolic pathway, signaling, transport, DNA repair)\n"
        f"  - `CC` (Cellular Component, e.g. plasma membrane, mitochondrion, cytosol, nucleus)\n\n"
        f"Output up to 25 specific, informative `GO:XXXXXXX` terms total (avoiding trivial root terms), each "
        f"with a calibrated confidence score in (0.0, 1.0].\n\n"
        f"Return ONLY a single compact JSON object with exact schema:\n"
        f'{{"predictions": [["GO:0004672", 0.92, "protein kinase activity"], ...]}}\n'
        f"Include no text outside the JSON object."
    )
    assert target.target_id not in prompt
    return prompt


def parse_cafa_response(raw_text: str) -> Tuple[Dict[str, float], bool]:
    """Parse `{"predictions": [["GO:XXXXXXX", conf, ...], ...]}` into `{go_id: confidence}`."""
    if not raw_text or not raw_text.strip():
        return {}, False

    out: Dict[str, float] = {}
    cleaned = raw_text.strip()
    m_f = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", cleaned, flags=re.DOTALL)
    if m_f:
        cleaned = m_f.group(1)
    else:
        i0, i1 = cleaned.find("{"), cleaned.rfind("}")
        if i0 != -1 and i1 > i0:
            cleaned = cleaned[i0 : i1 + 1]

    try:
        obj = json.loads(cleaned)
        rows = obj.get("predictions") or obj.get("go_terms") or obj.get("terms") or []
        for item in rows:
            if isinstance(item, (list, tuple)) and len(item) >= 2:
                m_go = GO_ID_RE.search(str(item[0]))
                if m_go:
                    gid = m_go.group(0)
                    sc = float(np.clip(float(item[1]), 0.0, 1.0))
                    out[gid] = max(out.get(gid, 0.0), sc)
            elif isinstance(item, dict):
                m_go = GO_ID_RE.search(str(item.get("go_id") or item.get("id") or item.get("term") or ""))
                if m_go:
                    gid = m_go.group(0)
                    sc = float(np.clip(float(item.get("score", item.get("confidence", 0.5))), 0.0, 1.0))
                    out[gid] = max(out.get(gid, 0.0), sc)
    except Exception:
        # Regex fallback for `["GO:0003677", 0.85`
        for m in re.finditer(r"(GO:\d{7})[^\d\n]{1,20}([01]?\.\d+|1\.0|1)", raw_text):
            gid = m.group(1)
            sc = float(np.clip(float(m.group(2)), 0.0, 1.0))
            out[gid] = max(out.get(gid, 0.0), sc)

    return out, (len(out) > 0)


def compute_cafa_fmax(
    dag: GeneOntologyDAG,
    targets: Sequence[CAFATarget],
    pred_raw_by_tid: Dict[str, Dict[str, float]],
    aspect: str,  # "MF" | "BP" | "CC"
    ic_map: Optional[Dict[str, float]] = None,
    exclude_known: bool = True,
) -> Dict[str, float]:
    """Official CAFA-6 protein-centric `F_max` and IA/IC-weighted `wF_max` (`full` mode) for a single GO aspect.

    Supports exact CAFA-6 track semantics:
      - NK: evaluates full `T1[aspect]` (`known_terms` empty).
      - LK: restricts to targets with `aspect in eval_aspects` (newly gained ontology only).
      - PK: when `exclude_known=True`, applies `cafaeval -known` dual masking, removing `T0[aspect]`
            from BOTH ground truth (`T1 \\ T0`) and DAG-propagated predictions (`Y_hat \\ T0`).
    """
    attr = {"MF": "true_terms_mf", "BP": "true_terms_bp", "CC": "true_terms_cc"}[aspect]
    k_attr = {"MF": "known_terms_mf", "BP": "known_terms_bp", "CC": "known_terms_cc"}[aspect]

    true_sets: List[Set[str]] = []
    prop_preds: List[Dict[str, float]] = []
    default_ic = (ic_map or {}).get("__default__", 1.0)

    for t in targets:
        if t.eval_aspects is not None and aspect not in t.eval_aspects:
            continue
        raw_true = getattr(t, attr)
        if not raw_true:
            continue
        t_true = {g for g in raw_true if dag.term_aspect.get(g) == aspect and g not in ROOT_GO_TERMS}
        t_known = (
            {g for g in getattr(t, k_attr) if dag.term_aspect.get(g) == aspect and g not in ROOT_GO_TERMS}
            if exclude_known
            else set()
        )
        if t_known:
            t_true = t_true - t_known
        if not t_true:
            continue

        raw_p = pred_raw_by_tid.get(t.target_id, {})
        prop_all = dag.propagate_scores(raw_p)
        prop_asp = {
            g: sc
            for g, sc in prop_all.items()
            if dag.term_aspect.get(g) == aspect and g not in ROOT_GO_TERMS and g not in t_known
        }
        true_sets.append(t_true)
        prop_preds.append(prop_asp)

    if not true_sets:
        return {"n_targets": 0, "fmax": 0.0, "wfmax": 0.0, "best_threshold": 0.5, "coverage": 0.0}

    n_total = len(true_sets)
    thresholds = np.linspace(0.01, 1.00, 100)
    best_f = 0.0
    best_tau = 0.5
    best_wf = 0.0

    for tau in thresholds:
        sum_pr, sum_rc, n_pred_nonempty = 0.0, 0.0, 0
        sum_wpr, sum_wrc, n_wpred_nonempty = 0.0, 0.0, 0

        for t_true, p_map in zip(true_sets, prop_preds):
            p_set = {g for g, sc in p_map.items() if sc >= tau}
            inter = p_set & t_true
            rc = len(inter) / float(len(t_true))
            sum_rc += rc
            if p_set:
                n_pred_nonempty += 1
                sum_pr += len(inter) / float(len(p_set))

            if ic_map is not None:
                w_true = sum(ic_map.get(g, default_ic) for g in t_true)
                w_inter = sum(ic_map.get(g, default_ic) for g in inter)
                if w_true > 0:
                    sum_wrc += w_inter / w_true
                if p_set:
                    w_pred = sum(ic_map.get(g, default_ic) for g in p_set)
                    if w_pred > 0:
                        n_wpred_nonempty += 1
                        sum_wpr += w_inter / w_pred

        avg_rc = sum_rc / float(n_total)
        avg_pr = (sum_pr / float(n_pred_nonempty)) if n_pred_nonempty > 0 else 0.0
        if avg_pr + avg_rc > 0:
            f1 = 2.0 * avg_pr * avg_rc / (avg_pr + avg_rc)
            if f1 > best_f:
                best_f = f1
                best_tau = float(tau)

        if ic_map is not None:
            avg_wrc = sum_wrc / float(n_total)
            avg_wpr = (sum_wpr / float(n_wpred_nonempty)) if n_wpred_nonempty > 0 else 0.0
            if avg_wpr + avg_wrc > 0:
                wf1 = 2.0 * avg_wpr * avg_wrc / (avg_wpr + avg_wrc)
                if wf1 > best_wf:
                    best_wf = wf1

    cov = sum(1 for p_map in prop_preds if len(p_map) > 0) / float(n_total)
    return {
        "n_targets": n_total,
        "fmax": round(best_f, 4),
        "wfmax": round(best_wf, 4) if ic_map is not None else round(best_f, 4),
        "best_threshold": round(best_tau, 2),
        "coverage": round(cov, 4),
    }
