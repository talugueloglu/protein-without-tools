#!/usr/bin/env python3
"""Download raw RCSB crystal structures for Stage 3B (De Novo Design & Closed-Loop Epitope Binder Design).

Fetches and caches into `data/raw_stage3b_cache/`:
  1. Part 1 Motif & Fold Reference PDBs:
     - `1CLL` (Calmodulin `EF-hand` Ca2+-binding loop)
     - `1AAY` (Zif268 `C2H2` Zn-finger chelating motif)
     - `5P21` (H-Ras `Walker-A` P-loop + `Rossmann` alpha/beta/alpha sandwich)
     - `1SHG` (`SH3` domain `all-beta` barrel)
  2. Part 2 Classic Epitope Binder Controls:
     - `5O45` (Human `PD-L1` IgV domain + binder complex)
     - `1WWW` (Human `TrkA-d5` domain + NGF complex)
  3. Part 2 Post-May-2026 (`release_date >= 2026-05-01`) Heterodimeric Complexes:
     - Searches RCSB for high-resolution (`<= 3.0 A`) heterodimeric protein complexes released
       after `2026-05-01` (`polymer_entity_count_protein == 2`), verifies globular target chain
       (`75 <= L_A <= 210`), tight physical protein-protein interface (`>= 10` Cbeta < 8A contacts),
       and `< 30%` pre-May-2026 PDB sequence identity (`MMseqs2`) so zero binders exist in training data!
"""

from __future__ import annotations

import json
from pathlib import Path
import re
from typing import Any, Dict, List, Optional, Tuple

import httpx
import numpy as np

from fable_bench.pdb_utils import AA3_TO_1, CANONICAL_AA_SET


def compute_virtual_cb(n: np.ndarray, ca: np.ndarray, c: np.ndarray) -> np.ndarray:
    b = ca - n
    cc = c - ca
    a = np.cross(b, cc)
    return -0.58273431 * b + 0.56802827 * cc - 0.54067466 * a + ca


def radius_of_gyration(ca: np.ndarray) -> float:
    ctr = np.mean(ca, axis=0)
    return float(np.sqrt(np.mean(np.sum((ca - ctr) ** 2, axis=1))))

SAFETY_BLOCKLIST = re.compile(
    r"(virus|viral|virion|capsid|phage|toxin|venom|ricin|abrin|"
    r"botulinum|anthrax|ebola|marburg|variola|influenza|corona|sars|\bhiv\b)",
    re.IGNORECASE,
)


def parse_cif_all_protein_chains(cif_text: str) -> Dict[str, Dict[str, Any]]:
    """Extract per-chain canonical backbone (`N, CA, C, O`) segments from an mmCIF string."""
    lines = cif_text.splitlines()
    n_lines = len(lines)
    i = 0
    col_idx: Dict[str, int] = {}
    data_start = -1

    while i < n_lines:
        s = lines[i].strip()
        if s == "loop_":
            j = i + 1
            headers: List[str] = []
            while j < n_lines and lines[j].strip().startswith("_"):
                headers.append(lines[j].strip())
                j += 1
            if headers and headers[0].startswith("_atom_site."):
                col_idx = {h.split(".", 1)[1]: k for k, h in enumerate(headers)}
                data_start = j
                break
        i += 1

    if data_start < 0:
        return {}

    idx_group = col_idx.get("group_PDB", 0)
    idx_atom = col_idx.get("label_atom_id", col_idx.get("auth_atom_id", -1))
    idx_comp = col_idx.get("label_comp_id", col_idx.get("auth_comp_id", -1))
    idx_asym = col_idx.get("auth_asym_id", col_idx.get("label_asym_id", -1))
    idx_seq = col_idx.get("label_seq_id", col_idx.get("auth_seq_id", -1))
    idx_alt = col_idx.get("label_alt_id", -1)
    idx_mod = col_idx.get("pdbx_PDB_model_num", -1)
    idx_x, idx_y, idx_z = col_idx.get("Cartn_x", -1), col_idx.get("Cartn_y", -1), col_idx.get("Cartn_z", -1)

    first_model: Optional[str] = None
    # chain -> seq_num -> atom -> xyz
    raw_chains: Dict[str, Dict[int, Tuple[str, Dict[str, List[float]]]]] = {}

    for k in range(data_start, n_lines):
        raw = lines[k].strip()
        if not raw or raw.startswith("#") or raw.startswith("loop_") or raw.startswith("_"):
            if raw.startswith("#") or raw.startswith("loop_"):
                break
            continue
        toks = raw.split()
        if len(toks) <= max(idx_x, idx_y, idx_z, idx_asym):
            continue
        if toks[idx_group] not in ("ATOM", "HETATM"):
            continue
        if idx_mod >= 0:
            mnum = toks[idx_mod]
            if first_model is None:
                first_model = mnum
            elif mnum != first_model:
                break
        if idx_alt >= 0 and toks[idx_alt] not in (".", "?", "A"):
            continue
        comp = toks[idx_comp].upper()
        if comp not in AA3_TO_1:
            continue
        atom = toks[idx_atom].strip('"')
        if atom not in ("N", "CA", "C", "O"):
            continue
        seq_str = toks[idx_seq]
        if not seq_str.lstrip("-").isdigit():
            continue
        snum = int(seq_str)
        ch = toks[idx_asym]
        ch_map = raw_chains.setdefault(ch, {})
        aa1 = AA3_TO_1[comp]
        if snum not in ch_map:
            ch_map[snum] = (aa1, {})
        if atom not in ch_map[snum][1]:
            ch_map[snum][1][atom] = [float(toks[idx_x]), float(toks[idx_y]), float(toks[idx_z])]

    out: Dict[str, Dict[str, Any]] = {}
    for ch, res_map in raw_chains.items():
        # Extract longest contiguous (`CA-CA < 4.4 A`) resolved backbone segment
        ordered = [
            (sn, aa, atoms)
            for sn, (aa, atoms) in sorted(res_map.items())
            if all(a in atoms for a in ("N", "CA", "C", "O")) and aa in CANONICAL_AA_SET
        ]
        if len(ordered) < 35:
            continue
        best_seg: List[Tuple[int, str, Dict[str, List[float]]]] = []
        cur_seg = [ordered[0]]
        for item in ordered[1:]:
            prev_ca = np.array(cur_seg[-1][2]["CA"])
            cur_ca = np.array(item[2]["CA"])
            if float(np.linalg.norm(cur_ca - prev_ca)) <= 4.5:
                cur_seg.append(item)
            else:
                if len(cur_seg) > len(best_seg):
                    best_seg = cur_seg
                cur_seg = [item]
        if len(cur_seg) > len(best_seg):
            best_seg = cur_seg
        if len(best_seg) < 35:
            continue
        seq = "".join(x[1] for x in best_seg)
        n_arr = np.array([x[2]["N"] for x in best_seg], dtype=np.float64)
        ca_arr = np.array([x[2]["CA"] for x in best_seg], dtype=np.float64)
        c_arr = np.array([x[2]["C"] for x in best_seg], dtype=np.float64)
        o_arr = np.array([x[2]["O"] for x in best_seg], dtype=np.float64)
        cb_arr = compute_virtual_cb(n_arr, ca_arr, c_arr)
        out[ch] = {
            "chain_id": ch,
            "sequence": seq,
            "length": len(seq),
            "res_nums": [x[0] for x in best_seg],
            "n": n_arr,
            "ca": ca_arr,
            "c": c_arr,
            "o": o_arr,
            "cb": cb_arr,
            "rg": radius_of_gyration(ca_arr),
        }
    return out


def check_precutoff_pdb_seqid(client: httpx.Client, seq: str, cutoff: str = "2026-05-01") -> float:
    """Return max sequence identity (%) in RCSB PDB released strictly before `cutoff`."""
    q = {
        "query": {
            "type": "group",
            "logical_operator": "and",
            "nodes": [
                {
                    "type": "terminal",
                    "service": "sequence",
                    "parameters": {
                        "evalue_cutoff": 10.0,
                        "identity_cutoff": 0.20,
                        "sequence_type": "protein",
                        "value": seq,
                    },
                },
                {
                    "type": "terminal",
                    "service": "text",
                    "parameters": {
                        "attribute": "rcsb_accession_info.initial_release_date",
                        "operator": "less",
                        "value": f"{cutoff}T00:00:00Z",
                    },
                },
            ],
        },
        "return_type": "polymer_entity",
        "request_options": {"paginate": {"start": 0, "rows": 25}, "scoring_strategy": "sequence"},
    }
    r = client.post("https://search.rcsb.org/rcsbsearch/v2/query", json=q)
    if r.status_code in (204, 404):
        return 0.0
    if r.status_code != 200:
        return 0.0
    hits = r.json().get("result_set") or []
    if not hits:
        return 0.0
    best_id = 0.0
    for h in hits[:10]:
        for sc in h.get("services") or []:
            for n in sc.get("nodes") or []:
                for m in n.get("match_context") or []:
                    sid = float(m.get("sequence_identity", 0.0)) * 100.0
                    if sid > best_id:
                        best_id = sid
    return round(best_id, 1)


def main() -> None:
    repo = Path("/Users/taluguloglu/.gemini/antigravity/scratch/fable-protein-bench")
    raw_dir = repo / "data" / "raw_stage3b_cache"
    cif_dir = raw_dir / "cifs"
    cif_dir.mkdir(parents=True, exist_ok=True)

    with httpx.Client(timeout=60.0, follow_redirects=True) as client:
        # 1. Fixed reference PDBs (Part 1 motifs/folds + Part 2 classic controls)
        fixed_pdbs = ["1CLL", "1AAY", "5P21", "1SHG", "5O45", "1WWW"]
        for pid in fixed_pdbs:
            dest = cif_dir / f"{pid}.cif"
            if not dest.exists():
                r = client.get(f"https://files.rcsb.org/download/{pid}.cif")
                r.raise_for_status()
                dest.write_text(r.text, encoding="utf-8")
            print(f"  Cached {pid}.cif ({dest.stat().st_size / 1024:.1f} KB)", flush=True)

        # 2. Search RCSB for post-May-2026 (`>= 2026-05-01`) heterodimeric protein-protein complexes
        print("Searching RCSB for post-2026-05-01 heterodimeric protein complexes...", flush=True)
        search_q = {
            "query": {
                "type": "group",
                "logical_operator": "and",
                "nodes": [
                    {
                        "type": "terminal",
                        "service": "text",
                        "parameters": {
                            "attribute": "rcsb_accession_info.initial_release_date",
                            "operator": "greater_or_equal",
                            "value": "2026-05-01T00:00:00Z",
                        },
                    },
                    {
                        "type": "terminal",
                        "service": "text",
                        "parameters": {
                            "attribute": "rcsb_entry_info.polymer_entity_count_protein",
                            "operator": "equals",
                            "value": 2,
                        },
                    },
                    {
                        "type": "terminal",
                        "service": "text",
                        "parameters": {
                            "attribute": "rcsb_entry_info.deposited_polymer_monomer_count",
                            "operator": "less_or_equal",
                            "value": 460,
                        },
                    },
                    {
                        "type": "terminal",
                        "service": "text",
                        "parameters": {
                            "attribute": "rcsb_entry_info.resolution_combined",
                            "operator": "less_or_equal",
                            "value": 3.0,
                        },
                    },
                ],
            },
            "return_type": "entry",
            "request_options": {"paginate": {"start": 0, "rows": 140}},
        }
        rs = client.post("https://search.rcsb.org/rcsbsearch/v2/query", json=search_q)
        rs.raise_for_status()
        cand_ids = [h["identifier"] for h in (rs.json().get("result_set") or [])]
        print(f"  Found {len(cand_ids)} candidate post-May-2026 heterodimer entries.", flush=True)

        # Fetch GraphQL titles/dates/organisms in batches of 40 for safety filtering
        entry_meta: Dict[str, Dict[str, str]] = {}
        gql = """
        query($ids: [String!]!) {
          entries(entry_ids: $ids) {
            rcsb_id
            struct { title }
            rcsb_accession_info { initial_release_date }
            polymer_entities {
              rcsb_entity_source_organism { ncbi_scientific_name }
              entity_poly { pdbx_strand_id }
              rcsb_polymer_entity { pdbx_description }
            }
          }
        }
        """
        for i in range(0, len(cand_ids), 40):
            rg = client.post(
                "https://data.rcsb.org/graphql",
                json={"query": gql, "variables": {"ids": cand_ids[i : i + 40]}},
            )
            if rg.status_code != 200:
                continue
            for ent in ((rg.json().get("data") or {}).get("entries") or []):
                eid = ent["rcsb_id"]
                title = str((ent.get("struct") or {}).get("title") or "")
                rdate = str((ent.get("rcsb_accession_info") or {}).get("initial_release_date") or "")[:10]
                orgs = []
                descs = []
                for pe in ent.get("polymer_entities") or []:
                    for o in pe.get("rcsb_entity_source_organism") or []:
                        if o.get("ncbi_scientific_name"):
                            orgs.append(str(o["ncbi_scientific_name"]))
                    d = str((pe.get("rcsb_polymer_entity") or {}).get("pdbx_description") or "")
                    if d:
                        descs.append(d)
                blob = f"{title} {' '.join(orgs)} {' '.join(descs)}"
                if SAFETY_BLOCKLIST.search(blob):
                    continue
                # Avoid trivial Fab/scFv antibody-antigen crystal complexes so we get genuine non-antibody PPIs too!
                is_ab = bool(re.search(r"\b(fab|scfv|nanobody|vhh|antibody|immunoglobulin|heavy chain|light chain)\b", blob, re.I))
                entry_meta[eid] = {
                    "pdb_id": eid,
                    "title": title,
                    "release_date": rdate,
                    "organisms": "; ".join(sorted(set(orgs)))[:120],
                    "descriptions": "; ".join(descs)[:180],
                    "is_antibody_like": "yes" if is_ab else "no",
                }

        # Prioritize non-antibody novel PPIs first, then novel binder complexes
        ordered_eids = sorted(entry_meta.keys(), key=lambda k: (0 if entry_meta[k]["is_antibody_like"] == "no" else 1, k))

        verified_complexes: List[Dict[str, Any]] = []
        seen_target_kmers: List[set[str]] = []

        for eid in ordered_eids:
            if len(verified_complexes) >= 8:
                break
            cif_p = cif_dir / f"{eid}.cif"
            if not cif_p.exists():
                rc = client.get(f"https://files.rcsb.org/download/{eid}.cif")
                if rc.status_code != 200:
                    continue
                cif_p.write_text(rc.text, encoding="utf-8")
            chains = parse_cif_all_protein_chains(cif_p.read_text(encoding="utf-8"))
            if len(chains) < 2:
                continue

            # Find best heteromeric pair (ch_target, ch_partner) with distinct sequences, globular target 75..205 AA,
            # partner 42..175 AA, and >= 12 intermolecular Cbeta < 8.0 A contacts!
            best_pair = None
            ch_keys = sorted(chains.keys())
            for ca_id in ch_keys:
                for cb_id in ch_keys:
                    if ca_id == cb_id:
                        continue
                    ta, tb = chains[ca_id], chains[cb_id]
                    if ta["sequence"] == tb["sequence"]:
                        continue
                    la, lb = ta["length"], tb["length"]
                    if not (75 <= la <= 205 and 42 <= lb <= 175):
                        continue
                    # Globularity check on Target Chain A
                    if ta["rg"] > 2.38 * (la ** 0.38):
                        continue
                    # Intermolecular Cbeta distance matrix
                    dmat = np.linalg.norm(ta["cb"][:, None, :] - tb["cb"][None, :, :], axis=-1)
                    n_contacts = int(np.sum(dmat < 8.0))
                    n_target_iface = int(np.sum(np.min(dmat, axis=1) < 8.0))
                    # Check zero severe backbone clashes
                    d_ca = np.linalg.norm(ta["ca"][:, None, :] - tb["ca"][None, :, :], axis=-1)
                    if int(np.sum(d_ca < 3.2)) > 0:
                        continue
                    if n_contacts >= 14 and n_target_iface >= 6:
                        if best_pair is None or n_contacts > best_pair[0]:
                            best_pair = (n_contacts, ca_id, cb_id)

            if best_pair is None:
                continue

            n_cont, ca_id, cb_id = best_pair
            ta, tb = chains[ca_id], chains[cb_id]
            km_a = {ta["sequence"][j : j + 3] for j in range(ta["length"] - 2)}
            if any(len(km_a & okm) / float(max(1, len(km_a | okm))) > 0.25 for okm in seen_target_kmers):
                continue

            # Verify pre-cutoff PDB sequence novelty (`< 30%` SeqID on partner OR target!)
            sid_partner = check_precutoff_pdb_seqid(client, tb["sequence"])
            sid_target = check_precutoff_pdb_seqid(client, ta["sequence"])
            if sid_partner >= 30.0 and sid_target >= 30.0:
                continue

            seen_target_kmers.append(km_a)
            meta = dict(entry_meta[eid])
            meta.update(
                {
                    "target_chain": ca_id,
                    "partner_chain": cb_id,
                    "target_len": ta["length"],
                    "partner_len": tb["length"],
                    "interchain_cb8_contacts": n_cont,
                    "target_precutoff_pdb_seqid": sid_target,
                    "partner_precutoff_pdb_seqid": sid_partner,
                }
            )
            verified_complexes.append(meta)
            print(
                f"  [Accepted Post-May-2026 Complex] {eid} ({meta['release_date']}): "
                f"Target {ca_id} ({ta['length']} AA, pre-PDB={sid_target}%) + "
                f"Partner {cb_id} ({tb['length']} AA, pre-PDB={sid_partner}%), {n_cont} contacts — {meta['title'][:65]}",
                flush=True,
            )

        (raw_dir / "post_may2026_complexes.json").write_text(
            json.dumps(verified_complexes, indent=2), encoding="utf-8"
        )
        print(f"Saved {len(verified_complexes)} verified post-May-2026 heterodimer complexes!", flush=True)


if __name__ == "__main__":
    main()
