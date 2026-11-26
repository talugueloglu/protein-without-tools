"""Task 12: Protein–DNA / RNA Interface Ranking Benchmark.

Tests whether frontier reasoning models (without tools) and nucleic-acid co-folding
specialists (Boltz-2) can pinpoint the exact DNA/RNA-contacting residues on an
anonymized protein — pairing every interface residue (including positively charged
Arg/Lys/His) with a same-AA non-interface surface decoy so 1D electrostatics
heuristics ('always pick Arg/Lys') are locked to 50% same-AA win rate.
"""

from __future__ import annotations

from typing import Sequence

from protein_without_tools.task10_ligand_site import (
    SiteCandidate,
    grade_site_ranking_panel,
    parse_site_ranking_response,
    select_same_aa_site_panel,
)


def build_nucleic_acid_prompt(
    protein_sequence: str,
    na_type: str,
    na_sequence: str,
    panel: Sequence[SiteCandidate],
) -> str:
    """Build a zero-leakage prompt for protein–DNA/RNA interface residue ranking."""
    L = len(protein_sequence)
    n = len(panel)
    toks = [f'"{c.res_id}"' for c in panel]
    na_label = "Double-stranded DNA (one strand shown, 5'->3')" if na_type == "dsDNA" else "RNA (5'->3')"
    ex1 = panel[0].res_id if panel else "R24"
    ex2 = panel[1].res_id if len(panel) > 1 else "K51"
    return (
        f"You are predicting the 3D protein-nucleic acid binding interface of an anonymized "
        f"protein complex from primary sequences alone.\n\n"
        f"Protein Sequence (1-indexed, length L = {L}):\n{protein_sequence}\n\n"
        f"Bound {na_label} (length {len(na_sequence)} nt):\n{na_sequence}\n\n"
        f"Candidate Protein Residues ({n} total, listed in ascending sequence order):\n"
        f"[{', '.join(toks)}]\n\n"
        f"Task:\n"
        f"Rank ALL {n} candidate protein residues from #1 (MOST likely direct {na_type}-contacting "
        f"interface residue within 4.5 A heavy-atom distance of the nucleic acid) down to #{n} "
        f"(LEAST likely — distant non-interface residue > 10 A away).\n"
        f"Note: the panel pairs true nucleic-acid-binding residues 1-to-1 with non-interface decoys "
        f"of the EXACT SAME amino-acid types (e.g., interface Arg/Lys paired with non-binding Arg/Lys), "
        f"so rank by 3D fold and binding-surface geometry rather than residue charge alone.\n\n"
        f"Return ONLY a single compact JSON object with exact schema:\n"
        f'{{"ranking": ["{ex1}", "{ex2}", ...]}}\n'
        f"ordered strictly from #1 (strongest interface contact) to #{n} (distant decoy), "
        f"containing all {n} residues with no text outside JSON."
    )


__all__ = [
    "SiteCandidate",
    "build_nucleic_acid_prompt",
    "grade_site_ranking_panel",
    "parse_site_ranking_response",
    "select_same_aa_site_panel",
]
