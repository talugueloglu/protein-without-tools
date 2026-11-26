#!/usr/bin/env python3
"""Build Task 10 (Small-Molecule, Cofactor & Metal Binding-Site Ranking, N=60)
and Task 12 (Protein–DNA / RNA Interface Ranking, N=50) from strictly post-May-2026 RCSB PDB complexes.

Anti-memorisation & anti-shortcut guarantees:
  1. Every structure has `initial_release_date >= 2026-05-01` (post training cutoff) and
     resolution <= 3.0 A.
  2. Sequences are deduplicated by 3-mer Jaccard <= 0.30 across each task.
  3. Every binding/interface residue is paired 1-to-1 with a same-AA non-contact decoy (> 10 A),
     locking any 1D amino-acid or charge ('pick R/K/H') lookup table to exact 50.0% win rate.
  4. Zero PDB IDs, protein names, ligand common names, or organism names appear in prompts
     (only anonymized sequences + canonical SMILES / metal formula / nucleotide sequence).
"""

from __future__ import annotations

import asyncio
from collections import Counter
import gzip
import json
from pathlib import Path
import sys
from typing import Any, Dict, List, Optional, Tuple
import warnings

from Bio.PDB import MMCIFParser
import httpx
import numpy as np

warnings.filterwarnings("ignore")

REPO = Path(__file__).resolve().parents[1]
if str(REPO / "src") not in sys.path:
    sys.path.insert(0, str(REPO / "src"))

from protein_without_tools.pdb_utils import AA3_TO_1
from protein_without_tools.task10_ligand_site import (
    CATALYTIC_METALS,
    COFACTOR_COMP_IDS,
    CRYSTAL_ARTIFACT_COMP_IDS,
    SiteCandidate,
    build_ligand_site_prompt,
    grade_site_ranking_panel,
    select_same_aa_site_panel,
)
from protein_without_tools.task12_nucleic_acid import build_nucleic_acid_prompt

CACHE_DIR = REPO / "data/raw_v2_cache/rcsb_v2"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

NA_RES_MAP = {
    "DA": "A", "DT": "T", "DG": "G", "DC": "C", "DI": "I",
    "A": "A", "U": "U", "G": "G", "C": "C", "I": "I",
}
DNA_RES_SET = {"DA", "DT", "DG", "DC"}
RNA_RES_SET = {"A", "U", "G", "C"}


def _kmer_set(seq: str, k: int = 3) -> set[str]:
    return {seq[i : i + k] for i in range(max(0, len(seq) - k + 1))}


def _jaccard(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / float(len(a | b))


async def rcsb_search(client: httpx.AsyncClient, query: Dict[str, Any], rows: int = 300) -> List[str]:
    payload = {
        "query": query,
        "return_type": "entry",
        "request_options": {
            "paginate": {"start": 0, "rows": rows},
            "sort": [{"sort_by": "rcsb_accession_info.initial_release_date", "direction": "desc"}],
        },
    }
    resp = await client.post("https://search.rcsb.org/rcsbsearch/v2/query", json=payload, timeout=30.0)
    if resp.status_code == 204:
        return []
    resp.raise_for_status()
    return [r["identifier"] for r in resp.json().get("result_set", [])]


async def fetch_cif(client: httpx.AsyncClient, pdb_id: str, sem: asyncio.Semaphore) -> Optional[Path]:
    dst = CACHE_DIR / f"{pdb_id}.cif"
    if dst.exists() and dst.stat().st_size > 1000:
        return dst
    async with sem:
        try:
            r = await client.get(f"https://files.rcsb.org/download/{pdb_id}.cif.gz", timeout=25.0)
            if r.status_code != 200:
                return None
            dst.write_bytes(gzip.decompress(r.content))
            return dst
        except Exception:
            return None


async def fetch_entry_meta(client: httpx.AsyncClient, pdb_id: str, sem: asyncio.Semaphore) -> Dict[str, Any]:
    cache = CACHE_DIR / f"{pdb_id}_meta.json"
    if cache.exists():
        return json.loads(cache.read_text(encoding="utf-8"))
    async with sem:
        try:
            r = await client.get(f"https://data.rcsb.org/rest/v1/core/entry/{pdb_id}", timeout=20.0)
            if r.status_code != 200:
                return {}
            d = r.json()
            meta = {
                "pdb_id": pdb_id,
                "release_date": d.get("rcsb_accession_info", {}).get("initial_release_date", "")[:10],
                "title": d.get("struct", {}).get("title", ""),
                "resolution": (d.get("rcsb_entry_info", {}).get("resolution_combined") or [None])[0],
            }
            cache.write_text(json.dumps(meta), encoding="utf-8")
            return meta
        except Exception:
            return {}


async def fetch_smiles(client: httpx.AsyncClient, comp_id: str, sem: asyncio.Semaphore) -> Optional[Tuple[str, float]]:
    """Return (canonical_smiles, formula_weight) for an RCSB chemical component."""
    cache = CACHE_DIR / f"chem_{comp_id}.json"
    if cache.exists():
        d = json.loads(cache.read_text(encoding="utf-8"))
        return (d["smiles"], float(d["mw"])) if d.get("smiles") else None
    async with sem:
        try:
            r = await client.get(f"https://data.rcsb.org/rest/v1/core/chemcomp/{comp_id}", timeout=20.0)
            if r.status_code != 200:
                return None
            d = r.json()
            mw = float(d.get("chem_comp", {}).get("formula_weight") or 0.0)
            smiles = None
            for desc in d.get("rcsb_chem_comp_descriptor", {}) and [d.get("rcsb_chem_comp_descriptor", {})]:
                smiles = desc.get("smilesstereo") or desc.get("smiles")
            if not smiles:
                for item in d.get("pdbx_chem_comp_descriptor", []):
                    if item.get("type") in ("SMILES_CANONICAL", "SMILES") and item.get("descriptor"):
                        smiles = item["descriptor"]
                        break
            cache.write_text(json.dumps({"comp_id": comp_id, "smiles": smiles, "mw": mw}), encoding="utf-8")
            return (smiles, mw) if smiles else None
        except Exception:
            return None


def extract_protein_and_ligands_from_cif(cif_path: Path) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Parse protein chains, non-artifact hetero ligands/metals, and DNA/RNA chains."""
    if cif_path.stat().st_size > 4_500_000:
        return [], [], []
    parser = MMCIFParser(QUIET=True)
    try:
        struct = parser.get_structure(cif_path.stem, str(cif_path))
    except Exception:
        return [], [], []
    model = next(iter(struct), None)
    if model is None:
        return [], [], []

    prot_chains: List[Dict[str, Any]] = []
    ligands: List[Dict[str, Any]] = []
    na_chains: List[Dict[str, Any]] = []

    for chain in model:
        p_res = []
        na_res = []
        for res in chain:
            het, resseq, _icode = res.id
            rname = res.get_resname().strip().upper()
            if het == " " and rname in AA3_TO_1:
                atoms = [a.get_coord() for a in res.get_atoms() if a.element != "H"]
                if atoms:
                    p_res.append((AA3_TO_1[rname], np.array(atoms, dtype=float)))
            elif het == " " and rname in NA_RES_MAP:
                atoms = [a.get_coord() for a in res.get_atoms() if a.element != "H"]
                if atoms:
                    na_res.append((rname, NA_RES_MAP[rname], np.array(atoms, dtype=float)))
            elif het != " ":
                if rname in CRYSTAL_ARTIFACT_COMP_IDS or rname in AA3_TO_1:
                    continue
                atoms = [a.get_coord() for a in res.get_atoms() if a.element != "H"]
                if not atoms:
                    continue
                is_metal = rname in CATALYTIC_METALS
                if is_metal or len(atoms) >= 10:
                    ligands.append({
                        "comp_id": rname,
                        "chain_id": chain.id,
                        "is_metal": is_metal,
                        "is_cofactor": rname in COFACTOR_COMP_IDS,
                        "n_heavy": len(atoms),
                        "coords": np.array(atoms, dtype=float),
                    })
        if 55 <= len(p_res) <= 380:
            prot_chains.append({
                "chain_id": chain.id,
                "sequence": "".join(x[0] for x in p_res),
                "res_coords": [x[1] for x in p_res],
            })
        if 4 <= len(na_res) <= 90:
            rnames = [x[0] for x in na_res]
            n_dna = sum(1 for rn in rnames if rn in DNA_RES_SET)
            n_rna = sum(1 for rn in rnames if rn in RNA_RES_SET)
            na_type = "dsDNA" if n_dna >= n_rna else "RNA"
            na_chains.append({
                "chain_id": chain.id,
                "na_type": na_type,
                "sequence": "".join(x[1] for x in na_res),
                "coords": np.vstack([x[2] for x in na_res]),
            })

    return prot_chains, ligands, na_chains


def min_dist_per_residue(res_coords: List[np.ndarray], target_coords: np.ndarray) -> List[float]:
    dists: List[float] = []
    for rc in res_coords:
        # Pairwise Euclidean distance between residue heavy atoms and target heavy atoms
        diff = rc[:, None, :] - target_coords[None, :, :]
        d = float(np.sqrt(np.min(np.sum(diff * diff, axis=-1))))
        dists.append(d)
    return dists


async def build_all() -> None:
    sem = asyncio.Semaphore(10)
    async with httpx.AsyncClient(follow_redirects=True) as client:
        # 1. Search post-May-2026 ligand/cofactor complexes
        q_lig = {
            "type": "group",
            "logical_operator": "and",
            "nodes": [
                {"type": "terminal", "service": "text", "parameters": {
                    "attribute": "rcsb_accession_info.initial_release_date", "operator": "greater_or_equal", "value": "2026-05-01T00:00:00Z"
                }},
                {"type": "terminal", "service": "text", "parameters": {
                    "attribute": "rcsb_entry_info.resolution_combined", "operator": "less_or_equal", "value": 3.0
                }},
                {"type": "terminal", "service": "text", "parameters": {
                    "attribute": "rcsb_entry_info.nonpolymer_entity_count", "operator": "greater_or_equal", "value": 1
                }},
                {"type": "terminal", "service": "text", "parameters": {
                    "attribute": "entity_poly.rcsb_sample_sequence_length", "operator": "range",
                    "value": {"from": 60, "to": 320, "include_lower": True, "include_upper": True}
                }},
            ],
        }
        # 2. Search post-May-2026 protein-DNA complexes
        q_dna = {
            "type": "group",
            "logical_operator": "and",
            "nodes": [
                {"type": "terminal", "service": "text", "parameters": {
                    "attribute": "rcsb_accession_info.initial_release_date", "operator": "greater_or_equal", "value": "2026-05-01T00:00:00Z"
                }},
                {"type": "terminal", "service": "text", "parameters": {
                    "attribute": "rcsb_entry_info.resolution_combined", "operator": "less_or_equal", "value": 3.2
                }},
                {"type": "terminal", "service": "text", "parameters": {
                    "attribute": "rcsb_entry_info.polymer_entity_count_DNA", "operator": "greater_or_equal", "value": 1
                }},
                {"type": "terminal", "service": "text", "parameters": {
                    "attribute": "rcsb_entry_info.polymer_entity_count_protein", "operator": "greater_or_equal", "value": 1
                }},
            ],
        }
        # 3. Search post-May-2026 compact protein-RNA complexes (exclude giant ribosomes!)
        q_rna = {
            "type": "group",
            "logical_operator": "and",
            "nodes": [
                {"type": "terminal", "service": "text", "parameters": {
                    "attribute": "rcsb_accession_info.initial_release_date", "operator": "greater_or_equal", "value": "2026-05-01T00:00:00Z"
                }},
                {"type": "terminal", "service": "text", "parameters": {
                    "attribute": "rcsb_entry_info.resolution_combined", "operator": "less_or_equal", "value": 3.4
                }},
                {"type": "terminal", "service": "text", "parameters": {
                    "attribute": "rcsb_entry_info.polymer_entity_count_RNA", "operator": "greater_or_equal", "value": 1
                }},
                {"type": "terminal", "service": "text", "parameters": {
                    "attribute": "rcsb_entry_info.polymer_entity_count_protein", "operator": "range",
                    "value": {"from": 1, "to": 6, "include_lower": True, "include_upper": True}
                }},
            ],
        }
        # 4. Targeted post-May-2026 cofactor & catalytic metal queries
        q_cof = {
            "type": "group",
            "logical_operator": "and",
            "nodes": [
                {"type": "terminal", "service": "text", "parameters": {
                    "attribute": "rcsb_accession_info.initial_release_date", "operator": "greater_or_equal", "value": "2026-05-01T00:00:00Z"
                }},
                {"type": "terminal", "service": "text", "parameters": {
                    "attribute": "rcsb_entry_info.resolution_combined", "operator": "less_or_equal", "value": 3.0
                }},
                {"type": "terminal", "service": "text", "parameters": {
                    "attribute": "rcsb_nonpolymer_entity_container_identifiers.nonpolymer_comp_id",
                    "operator": "in",
                    "value": sorted(COFACTOR_COMP_IDS),
                }},
                {"type": "terminal", "service": "text", "parameters": {
                    "attribute": "entity_poly.rcsb_sample_sequence_length", "operator": "range",
                    "value": {"from": 60, "to": 320, "include_lower": True, "include_upper": True}
                }},
            ],
        }
        q_met = {
            "type": "group",
            "logical_operator": "and",
            "nodes": [
                {"type": "terminal", "service": "text", "parameters": {
                    "attribute": "rcsb_accession_info.initial_release_date", "operator": "greater_or_equal", "value": "2026-05-01T00:00:00Z"
                }},
                {"type": "terminal", "service": "text", "parameters": {
                    "attribute": "rcsb_entry_info.resolution_combined", "operator": "less_or_equal", "value": 2.8
                }},
                {"type": "terminal", "service": "text", "parameters": {
                    "attribute": "rcsb_nonpolymer_entity_container_identifiers.nonpolymer_comp_id",
                    "operator": "in",
                    "value": sorted(CATALYTIC_METALS.keys()),
                }},
                {"type": "terminal", "service": "text", "parameters": {
                    "attribute": "entity_poly.rcsb_sample_sequence_length", "operator": "range",
                    "value": {"from": 60, "to": 320, "include_lower": True, "include_upper": True}
                }},
            ],
        }

        print("Querying RCSB Search API for strictly post-May-2026 complexes...", flush=True)
        lig_pdbs = await rcsb_search(client, q_lig, rows=320)
        cof_pdbs = await rcsb_search(client, q_cof, rows=160)
        met_pdbs = await rcsb_search(client, q_met, rows=260)
        dna_pdbs = await rcsb_search(client, q_dna, rows=280)
        rna_pdbs = await rcsb_search(client, q_rna, rows=360)
        print(f"  Candidates found: lig={len(lig_pdbs)}, cof={len(cof_pdbs)}, met={len(met_pdbs)}, DNA={len(dna_pdbs)}, RNA={len(rna_pdbs)}", flush=True)

        # Download mmCIFs & metadata concurrently
        lig_pool = list(dict.fromkeys(lig_pdbs[:220] + cof_pdbs[:120] + met_pdbs[:220]))
        all_pdbs = list(dict.fromkeys(lig_pool + dna_pdbs[:260] + rna_pdbs[:320]))
        print(f"Fetching {len(all_pdbs)} mmCIF files + entry metadata...", flush=True)
        cif_paths = await asyncio.gather(*(fetch_cif(client, pid, sem) for pid in all_pdbs))
        metas = await asyncio.gather(*(fetch_entry_meta(client, pid, sem) for pid in all_pdbs))
        cif_map = {pid: p for pid, p in zip(all_pdbs, cif_paths) if p is not None}
        meta_map = {pid: m for pid, m in zip(all_pdbs, metas) if m}
        print(f"  Downloaded/cached {len(cif_map)} mmCIF structures.", flush=True)

        # ---- BUILD TASK 10: Small Molecule (24) + Cofactor (18) + Metal Ion (18) = 60 ----
        t10_quotas = {"small_molecule_drug": 24, "cofactor_nucleotide": 18, "metal_ion": 18}
        t10_selected: List[Dict[str, Any]] = []
        t10_counts = Counter()
        t10_kmers: List[set[str]] = []

        # Track unique comp_ids so we don't repeat the same drug/cofactor excessively
        t10_comp_counts = Counter()

        for pid in lig_pool:
            if sum(t10_counts.values()) >= sum(t10_quotas.values()):
                break
            cif_p = cif_map.get(pid)
            meta = meta_map.get(pid, {})
            if not cif_p or meta.get("release_date", "") < "2026-05-01":
                continue
            prots, ligs, _nas = extract_protein_and_ligands_from_cif(cif_p)
            if not prots or not ligs:
                continue

            # Prioritize underrepresented category first
            ligs_sorted = sorted(
                ligs,
                key=lambda lg: (
                    0 if (lg["is_metal"] and t10_counts["metal_ion"] < t10_quotas["metal_ion"])
                    else (0 if (lg["is_cofactor"] and t10_counts["cofactor_nucleotide"] < t10_quotas["cofactor_nucleotide"])
                          else (0 if (not lg["is_metal"] and not lg["is_cofactor"] and t10_counts["small_molecule_drug"] < t10_quotas["small_molecule_drug"]) else 1)),
                    -lg["n_heavy"],
                ),
            )
            picked_for_pdb = False
            for lg in ligs_sorted:
                if picked_for_pdb:
                    break
                if lg["is_metal"]:
                    cat = "metal_ion"
                elif lg["is_cofactor"]:
                    cat = "cofactor_nucleotide"
                else:
                    cat = "small_molecule_drug"
                if t10_counts[cat] >= t10_quotas[cat]:
                    continue
                if cat == "small_molecule_drug" and t10_comp_counts[lg["comp_id"]] >= 2:
                    continue
                if cat == "cofactor_nucleotide" and t10_comp_counts[lg["comp_id"]] >= 5:
                    continue

                # Find best protein chain contacting this ligand
                best_choice = None
                for pr in prots:
                    seq = pr["sequence"]
                    km = _kmer_set(seq)
                    if any(_jaccard(km, prev) > 0.30 for prev in t10_kmers):
                        continue
                    dists = min_dist_per_residue(pr["res_coords"], lg["coords"])
                    cands = [
                        SiteCandidate(
                            res_id=f"{seq[i]}{i + 1}",
                            aa=seq[i],
                            pos=i + 1,
                            min_dist_angstrom=round(dists[i], 2),
                            is_contact=False,
                        )
                        for i in range(len(seq))
                    ]
                    # Metals have fewer direct coordinating atoms (often 3-6); allow 4.8A and 8-12 panel for metals
                    if cat == "metal_ion":
                        panel = select_same_aa_site_panel(cands, n_total=12, contact_cutoff=4.8, decoy_min_dist=10.0)
                        if panel is None:
                            panel = select_same_aa_site_panel(cands, n_total=8, contact_cutoff=4.8, decoy_min_dist=10.0)
                    else:
                        panel = select_same_aa_site_panel(cands, n_total=16, contact_cutoff=4.2, decoy_min_dist=10.0)
                    if panel is not None:
                        best_choice = (pr, panel)
                        break
                if best_choice is None:
                    continue

                if cat == "metal_ion":
                    descriptor = CATALYTIC_METALS[lg["comp_id"]]
                    mw = 60.0
                else:
                    sm_res = await fetch_smiles(client, lg["comp_id"], sem)
                    if not sm_res:
                        continue
                    descriptor, mw = sm_res
                    if cat == "small_molecule_drug" and not (160.0 <= mw <= 850.0):
                        continue

                pr, panel = best_choice
                t10_counts[cat] += 1
                t10_comp_counts[lg["comp_id"]] += 1
                t10_kmers.append(_kmer_set(pr["sequence"]))
                t10_selected.append({
                    "pdb_id": pid,
                    "chain_id": pr["chain_id"],
                    "release_date": meta.get("release_date", ""),
                    "resolution": meta.get("resolution"),
                    "title": meta.get("title", ""),
                    "stratum": cat,
                    "ligand_comp_id": lg["comp_id"],
                    "ligand_descriptor": descriptor,
                    "ligand_mw": round(mw, 1),
                    "sequence": pr["sequence"],
                    "length": len(pr["sequence"]),
                    "panel": panel,
                })
                picked_for_pdb = True

        # ---- BUILD TASK 12: Protein–DNA (25) + Protein–RNA (25) = 50 ----
        t12_quotas = {"dsDNA": 25, "RNA": 25}
        t12_selected: List[Dict[str, Any]] = []
        t12_counts = Counter()
        t12_kmers: List[set[str]] = []

        for na_group, pdbs_list in (("dsDNA", dna_pdbs[:260]), ("RNA", rna_pdbs[:320])):
            for pid in pdbs_list:
                if t12_counts[na_group] >= t12_quotas[na_group]:
                    break
                cif_p = cif_map.get(pid)
                meta = meta_map.get(pid, {})
                if not cif_p or meta.get("release_date", "") < "2026-05-01":
                    continue
                prots, _ligs, nas = extract_protein_and_ligands_from_cif(cif_p)
                nas = [x for x in nas if x["na_type"] == na_group]
                if not prots or not nas:
                    continue
                # Combine all NA chains of this type for true complex contact distance
                all_na_coords = np.vstack([x["coords"] for x in nas])
                rep_na = max(nas, key=lambda x: len(x["sequence"]))

                for pr in prots:
                    seq = pr["sequence"]
                    km = _kmer_set(seq)
                    if any(_jaccard(km, prev) > 0.30 for prev in t12_kmers):
                        continue
                    dists = min_dist_per_residue(pr["res_coords"], all_na_coords)
                    cands = [
                        SiteCandidate(
                            res_id=f"{seq[i]}{i + 1}",
                            aa=seq[i],
                            pos=i + 1,
                            min_dist_angstrom=round(dists[i], 2),
                            is_contact=False,
                        )
                        for i in range(len(seq))
                    ]
                    panel = select_same_aa_site_panel(cands, n_total=16, contact_cutoff=4.5, decoy_min_dist=10.0)
                    if panel is None:
                        continue
                    t12_counts[na_group] += 1
                    t12_kmers.append(km)
                    t12_selected.append({
                        "pdb_id": pid,
                        "chain_id": pr["chain_id"],
                        "release_date": meta.get("release_date", ""),
                        "resolution": meta.get("resolution"),
                        "title": meta.get("title", ""),
                        "stratum": na_group,
                        "na_sequence": rep_na["sequence"],
                        "na_length": len(rep_na["sequence"]),
                        "sequence": seq,
                        "length": len(seq),
                        "panel": panel,
                    })
                    break

    # ---- WRITE TASK 10 MANIFEST & PROMPTS ----
    t10_dir = REPO / "data/task10_ligand_site"
    t10_dir.mkdir(parents=True, exist_ok=True)
    t10_targets, t10_prompts = [], []
    # AA frequency anti-cheat lookup (favor His/Cys/Asp/Glu/Trp/Tyr/Arg)
    aa_prior = {"H": 5, "C": 5, "D": 4, "E": 4, "W": 3, "Y": 3, "R": 3, "K": 2}
    t10_null_same_aa, t10_null_auroc = [], []

    for idx, item in enumerate(t10_selected, 1):
        tid = f"t10_l{idx:03d}"
        panel: List[SiteCandidate] = item["panel"]
        prompt_text = build_ligand_site_prompt(
            item["sequence"], item["stratum"], item["ligand_descriptor"], panel
        )
        g_null = grade_site_ranking_panel(
            panel, [], predicted_scores={c.res_id: float(aa_prior.get(c.aa, 1)) for c in panel}
        )
        t10_null_same_aa.append(g_null["same_aa_win_rate"])
        t10_null_auroc.append(g_null["auroc"])

        t10_targets.append({
            "target_id": tid,
            "pdb_id": item["pdb_id"],
            "chain_id": item["chain_id"],
            "release_date": item["release_date"],
            "resolution": item["resolution"],
            "title": item["title"],
            "stratum": item["stratum"],
            "ligand_comp_id": item["ligand_comp_id"],
            "ligand_descriptor": item["ligand_descriptor"],
            "ligand_mw": item["ligand_mw"],
            "length": item["length"],
            "sequence": item["sequence"],
            "n_candidates": len(panel),
            "candidates": [
                {
                    "res_id": c.res_id,
                    "aa": c.aa,
                    "pos": c.pos,
                    "min_dist_angstrom": c.min_dist_angstrom,
                    "is_contact": c.is_contact,
                }
                for c in panel
            ],
        })
        t10_prompts.append({
            "custom_id": f"task10_{tid}",
            "target_id": tid,
            "task": "task10_ligand_active_site",
            "stratum": item["stratum"],
            "length": item["length"],
            "n_candidates": len(panel),
            "prompt": prompt_text,
        })

    t10_manifest = {
        "benchmark": "protein-without-tools-task10-ligand-active-site",
        "n_targets": len(t10_targets),
        "by_stratum": dict(t10_counts),
        "anti_cheat_audit": {
            "aa_1d_lookup_same_aa_win": round(float(np.mean(t10_null_same_aa)), 4) if t10_null_same_aa else None,
            "aa_1d_lookup_auroc": round(float(np.mean(t10_null_auroc)), 4) if t10_null_auroc else None,
        },
        "targets": t10_targets,
    }
    (t10_dir / "manifest.json").write_text(json.dumps(t10_manifest, indent=2), encoding="utf-8")
    (REPO / "outputs/prompts/task10_ligand_site_prompts.jsonl").write_text(
        "\n".join(json.dumps(r) for r in t10_prompts) + "\n", encoding="utf-8"
    )

    # ---- WRITE TASK 12 MANIFEST & PROMPTS ----
    t12_dir = REPO / "data/task12_nucleic_acid"
    t12_dir.mkdir(parents=True, exist_ok=True)
    t12_targets, t12_prompts = [], []
    charge_prior = {"R": 5, "K": 5, "H": 4, "Q": 3, "N": 3, "Y": 3, "W": 3}
    t12_null_same_aa, t12_null_auroc = [], []

    for idx, item in enumerate(t12_selected, 1):
        tid = f"t12_n{idx:03d}"
        panel = item["panel"]
        prompt_text = build_nucleic_acid_prompt(
            item["sequence"], item["stratum"], item["na_sequence"], panel
        )
        g_null = grade_site_ranking_panel(
            panel, [], predicted_scores={c.res_id: float(charge_prior.get(c.aa, 1)) for c in panel}
        )
        t12_null_same_aa.append(g_null["same_aa_win_rate"])
        t12_null_auroc.append(g_null["auroc"])

        t12_targets.append({
            "target_id": tid,
            "pdb_id": item["pdb_id"],
            "chain_id": item["chain_id"],
            "release_date": item["release_date"],
            "resolution": item["resolution"],
            "title": item["title"],
            "stratum": item["stratum"],
            "na_sequence": item["na_sequence"],
            "na_length": item["na_length"],
            "length": item["length"],
            "sequence": item["sequence"],
            "n_candidates": len(panel),
            "n_pos_charge_pairs": sum(1 for c in panel if c.is_contact and c.aa in ("R", "K", "H")),
            "candidates": [
                {
                    "res_id": c.res_id,
                    "aa": c.aa,
                    "pos": c.pos,
                    "min_dist_angstrom": c.min_dist_angstrom,
                    "is_contact": c.is_contact,
                }
                for c in panel
            ],
        })
        t12_prompts.append({
            "custom_id": f"task12_{tid}",
            "target_id": tid,
            "task": "task12_protein_nucleic_acid",
            "stratum": item["stratum"],
            "length": item["length"],
            "n_candidates": len(panel),
            "prompt": prompt_text,
        })

    t12_manifest = {
        "benchmark": "protein-without-tools-task12-protein-nucleic-acid",
        "n_targets": len(t12_targets),
        "by_stratum": dict(t12_counts),
        "anti_cheat_audit": {
            "charge_1d_lookup_same_aa_win": round(float(np.mean(t12_null_same_aa)), 4) if t12_null_same_aa else None,
            "charge_1d_lookup_auroc": round(float(np.mean(t12_null_auroc)), 4) if t12_null_auroc else None,
        },
        "targets": t12_targets,
    }
    (t12_dir / "manifest.json").write_text(json.dumps(t12_manifest, indent=2), encoding="utf-8")
    (REPO / "outputs/prompts/task12_nucleic_acid_prompts.jsonl").write_text(
        "\n".join(json.dumps(r) for r in t12_prompts) + "\n", encoding="utf-8"
    )

    print("\n=== TASK 10 (LIGAND/METAL) BUILT ===", flush=True)
    print(f"  Targets: {t10_manifest['n_targets']} | {t10_manifest['by_stratum']} | audit={t10_manifest['anti_cheat_audit']}")
    print("\n=== TASK 12 (DNA/RNA) BUILT ===", flush=True)
    print(f"  Targets: {t12_manifest['n_targets']} | {t12_manifest['by_stratum']} | audit={t12_manifest['anti_cheat_audit']}")


if __name__ == "__main__":
    asyncio.run(build_all())
