"""Zero-leakage prompt builders and fault-tolerant output parsers for Stage 1.

Covers:
  - Stage 1A (Forward Folding): Sequence -> Secondary Structure (Q3) + Ranked Contact Map
  - Stage 1B (Inverse Folding): Stripped 3D Backbone Geometry -> Designed Amino Acid Sequence
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
import re
from typing import Any, List, Optional, Tuple

from protein_without_tools.pdb_utils import CANONICAL_AA_SET, ProteinStructure


@dataclass
class ForwardPredictionParsed:
    """Parsed LLM output for Stage 1A Forward Folding."""

    q3: str
    contacts: List[Tuple[int, int, float]]  # 1-indexed (i, j, confidence), i < j, j - i >= 6
    parse_ok: bool
    raw_q3_len: int
    notes: str = ""


@dataclass
class InversePredictionParsed:
    """Parsed LLM output for Stage 1B Inverse Folding."""

    sequence: str
    parse_ok: bool
    raw_seq_len: int
    length_exact_match: bool
    notes: str = ""


def build_forward_folding_prompt(struct: ProteinStructure) -> str:
    """Build a zero-metadata-leakage Forward Folding prompt for `struct`."""
    L = struct.length
    n_contacts = max(20, int(round(1.5 * L)))

    # Numbered ruler every 10 residues to prevent trivial off-by-one character counting errors
    blocks = [struct.sequence[k : k + 10] for k in range(0, L, 10)]
    numbered_lines = []
    for line_start in range(0, len(blocks), 5):
        sub_blocks = blocks[line_start : line_start + 5]
        start_res = line_start * 10 + 1
        end_res = min(L, start_res + sum(len(b) for b in sub_blocks) - 1)
        numbered_lines.append(f"  {start_res:4d} - {end_res:<4d}: {' '.join(sub_blocks)}")
    ruler_block = "\n".join(numbered_lines)

    prompt = f"""You are predicting the 3D fold topology and residue-residue contact map of a monomeric protein from its primary amino acid sequence alone.

Rules:
- Work strictly from the sequence below. Do not use external search or database lookups.
- Residues are 1-indexed from 1 to {L}.
- A residue-residue contact is defined by C-beta atoms (C-alpha for Glycine) within 8.0 Å in the native 3D structure, with sequence separation |i - j| >= 6 (1 <= i < j <= {L}).

Sequence (length L = {L}):
{struct.sequence}

1-Indexed Residue Ruler:
{ruler_block}

Return ONLY a valid JSON object with the exact schema below (no markdown prose outside the JSON):
{{
  "fold_summary": "Brief 1-2 sentence hypothesis of secondary structure topology and sheet/helix packing",
  "q3": "<exact {L}-character string using ONLY H (helix), E (strand), C (coil)>",
  "contacts": [
    [i, j, confidence],
    ...
  ]
}}

Requirements for "contacts":
- Provide up to {n_contacts} predicted contacts with |i - j| >= 6 and 1 <= i < j <= {L}.
- Sort from highest confidence (near 1.0) to lowest confidence (near 0.1).
- Include all key long-range contacts (paired beta-strands in exact parallel/antiparallel register, packed helix-helix and sheet-helix core interfaces)."""

    # Anti-leakage check: ensure PDB ID does not appear anywhere in prompt
    if len(struct.pdb_id) >= 4 and struct.pdb_id.upper() in prompt.upper():
        raise AssertionError(f"PDB ID {struct.pdb_id} leaked into forward prompt!")
    return prompt


def build_inverse_folding_prompt(struct: ProteinStructure) -> str:
    """Build a zero-sequence-leakage Inverse Folding prompt from backbone `(N, CA, C, O)` only."""
    L = struct.length
    center = struct.coords_ca.mean(axis=0)
    ca_centered = struct.coords_ca - center
    ca_dist = np.linalg.norm(ca_centered[:, None, :] - ca_centered[None, :, :], axis=-1)

    table_rows: List[str] = [
        "idx |   Ca_x   Ca_y   Ca_z |    phi    psi | nbr10A | nonlocal_Ca_contacts(|i-j|>=4, d<9.0A)"
    ]
    for i in range(L):
        phi_s = "   NA" if math.isnan(float(struct.phi_deg[i])) else f"{struct.phi_deg[i]:6.1f}"
        psi_s = "   NA" if math.isnan(float(struct.psi_deg[i])) else f"{struct.psi_deg[i]:6.1f}"
        nbr_cnt = int(struct.ca_coord_num[i])

        # Non-local Cα neighbors within 9.0 Å (strictly backbone Cα geometry, zero sidechain info)
        cand = [
            (j + 1, float(ca_dist[i, j]))
            for j in range(L)
            if abs(i - j) >= 4 and ca_dist[i, j] < 9.0
        ]
        cand.sort(key=lambda t: t[1])
        nbr_str = ",".join(f"{j1}:{d:.1f}" for j1, d in cand[:6]) if cand else "-"

        table_rows.append(
            f"{i + 1:3d} | {ca_centered[i, 0]:6.1f} {ca_centered[i, 1]:6.1f} {ca_centered[i, 2]:6.1f} "
            f"| {phi_s} {psi_s} | {nbr_cnt:6d} | {nbr_str}"
        )

    backbone_table = "\n".join(table_rows)

    prompt = f"""You are performing de novo Inverse Folding (3D backbone structure -> 1D amino acid sequence design), analogous to ProteinMPNN.

Given ONLY the stripped backbone geometry of a monomeric target fold of length L = {L} residues (all side chains removed), design a single canonical amino acid sequence of exact length {L} that will fold stably and specifically into this exact 3D backbone.

Column definitions (all derived purely from N, C-alpha, C, O backbone coordinates):
- idx: 1-based residue index (1..{L})
- Ca_x, Ca_y, Ca_z: Centered C-alpha coordinates in Å
- phi, psi: Backbone Ramachandran dihedral angles in degrees
- nbr10A: Number of other C-alpha atoms within 10.0 Å (high >= 14 indicates buried hydrophobic core; low <= 9 indicates solvent-exposed surface)
- nonlocal_Ca_contacts: Nearest non-local residues (|i - j| >= 4) with C-alpha distance < 9.0 Å (`j:dist_Å`)

Backbone Geometry Table (L = {L}):
{backbone_table}

Return ONLY a valid JSON object with this exact schema:
{{
  "design_notes": "Brief 1-2 sentence summary of fold architecture, core packing, turn/glycine/proline choices, and surface polarity",
  "sequence": "<exact {L}-character string using ONLY the 20 standard 1-letter amino acids ACDEFGHIKLMNPQRSTVWY>"
}}"""

    # Strict anti-leakage assertions: neither native sequence nor PDB ID may appear in prompt
    if L >= 12 and struct.sequence in prompt:
        raise AssertionError("Native sequence leaked into inverse folding prompt!")
    if len(struct.pdb_id) >= 4 and struct.pdb_id.upper() in prompt.upper():
        raise AssertionError(f"PDB ID {struct.pdb_id} leaked into inverse prompt!")
    return prompt


# Needed locally above in build_inverse_folding_prompt
import numpy as np  # noqa: E402


def _extract_json_dict(raw_text: str) -> Optional[dict[str, Any]]:
    """Extract a JSON object from raw LLM output (handling markdown code fences & CoT preamble)."""
    text = raw_text.strip()
    fence_matches = re.findall(r"```(?:json)?\s*(\{.*?\})\s*```", text, flags=re.DOTALL)
    candidates = list(reversed(fence_matches))
    first_brace = text.find("{")
    last_brace = text.rfind("}")
    if first_brace != -1 and last_brace > first_brace:
        candidates.append(text[first_brace : last_brace + 1])

    for cand in candidates:
        try:
            obj = json.loads(cand)
            if isinstance(obj, dict):
                return obj
        except json.JSONDecodeError:
            # Remove trailing commas before ] or }
            cleaned = re.sub(r",\s*([\]}])", r"\1", cand)
            try:
                obj = json.loads(cleaned)
                if isinstance(obj, dict):
                    return obj
            except json.JSONDecodeError:
                continue
    return None


def parse_forward_folding_response(raw_text: str, length: int) -> ForwardPredictionParsed:
    """Parse Stage 1A LLM output into validated `(q3, contacts)`."""
    obj = _extract_json_dict(raw_text)
    if obj is None:
        return ForwardPredictionParsed(
            q3="C" * length,
            contacts=[],
            parse_ok=False,
            raw_q3_len=0,
            notes="Failed to parse JSON object from response.",
        )

    raw_q3 = re.sub(r"[^HEC]", "", str(obj.get("q3", "")).upper())
    raw_q3_len = len(raw_q3)
    if raw_q3_len == length:
        q3 = raw_q3
    elif raw_q3_len > length:
        q3 = raw_q3[:length]
    else:
        q3 = raw_q3 + "C" * (length - raw_q3_len)

    raw_contacts = obj.get("contacts", [])
    parsed_pairs: List[Tuple[int, int, float]] = []
    seen = set()
    if isinstance(raw_contacts, list):
        for rank_idx, item in enumerate(raw_contacts):
            i_val, j_val, conf_val = None, None, None
            if isinstance(item, (list, tuple)) and len(item) >= 2:
                i_val, j_val = item[0], item[1]
                conf_val = item[2] if len(item) >= 3 else max(0.1, 1.0 - 0.005 * rank_idx)
            elif isinstance(item, dict):
                i_val = item.get("i", item.get("res1"))
                j_val = item.get("j", item.get("res2"))
                conf_val = item.get(
                    "confidence",
                    item.get("p", item.get("prob", max(0.1, 1.0 - 0.005 * rank_idx))),
                )
            try:
                i_int = int(i_val)
                j_int = int(j_val)
                c_flt = float(conf_val)
            except (TypeError, ValueError):
                continue
            if i_int > j_int:
                i_int, j_int = j_int, i_int
            if not (1 <= i_int < j_int <= length):
                continue
            if (j_int - i_int) < 6:
                continue
            if (i_int, j_int) in seen:
                continue
            seen.add((i_int, j_int))
            parsed_pairs.append((i_int, j_int, float(np.clip(c_flt, 0.01, 1.0))))

    parsed_pairs.sort(key=lambda t: t[2], reverse=True)
    return ForwardPredictionParsed(
        q3=q3,
        contacts=parsed_pairs,
        parse_ok=(len(parsed_pairs) > 0 or raw_q3_len > 0),
        raw_q3_len=raw_q3_len,
        notes=str(obj.get("fold_summary", ""))[:300],
    )


def parse_inverse_folding_response(raw_text: str, length: int) -> InversePredictionParsed:
    """Parse Stage 1B LLM output into a validated length-`L` canonical amino acid sequence."""
    obj = _extract_json_dict(raw_text)
    raw_seq = ""
    notes = ""
    if obj is not None and "sequence" in obj:
        raw_seq = str(obj["sequence"]).upper()
        notes = str(obj.get("design_notes", ""))[:300]
    else:
        # Fallback: look for an uppercase amino-acid block near target length
        tokens = re.findall(r"[ACDEFGHIKLMNPQRSTVWY]{20,}", raw_text.upper())
        if tokens:
            tokens.sort(key=lambda s: abs(len(s) - length))
            raw_seq = tokens[0]

    cleaned = "".join(ch for ch in raw_seq if ch in CANONICAL_AA_SET)
    raw_len = len(cleaned)
    if raw_len == 0:
        return InversePredictionParsed(
            sequence="A" * length,
            parse_ok=False,
            raw_seq_len=0,
            length_exact_match=False,
            notes="Failed to extract valid amino acid sequence.",
        )

    exact = raw_len == length
    if raw_len > length:
        final_seq = cleaned[:length]
    elif raw_len < length:
        final_seq = cleaned + ("G" * (length - raw_len))
    else:
        final_seq = cleaned

    return InversePredictionParsed(
        sequence=final_seq,
        parse_ok=True,
        raw_seq_len=raw_len,
        length_exact_match=exact,
        notes=notes,
    )
