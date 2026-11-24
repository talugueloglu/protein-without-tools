#!/usr/bin/env python3
"""Download raw RCSB structures & benchmark configs for Stage 3B (N = 12 Proven-Bindable Epitopes).

Fetches & caches into `data/raw_stage3b_cache/`:
  1. Part 1 Motif & Fold Reference PDBs (`1CLL`, `1AAY`, `5P21`, `1SHG`).
  2. Part 2 `N = 12` Proven-De-Novo-Bindable Epitope Panel:
     - Tier 1 (`RFD4-Proteina` / `Proteina-Complexa` / `ProtDBench` Classics, n=4):
       `5O45` (`PD-L1`), `1WWW` (`TrkA`), `3DI3` (`IL-7Ra`), `3MJG` (`PDGFRb`).
     - Tier 2 (`BoltzProt-1` / `BoltzGen` June-2026 Hard Novel Panel, n=5):
       `3QKG` (`AMBP`), `1G13` (`GM2A`), `1JQD` (`HNMT`), `7AAH` (`MZB1`), `1NB0` (`RFK`),
       plus optional YAML hotspot specs from `HannesStark/boltzgen` / `congliuUvA/ProtDBench` if public.
     - Tier 3 (Post-May-2026 Wet-Lab Synthetic De Novo Binder Complexes, n=3):
       `9S83` (`2026-08-05` `JUBO4` de novo binder + `LEDGF PWWP`) + RCSB search for
       `release_date >= 2026-05-01` + `taxonomy_id == 32630 ("synthetic construct")`
       with `return_explain_metadata: True` verifying `< 30%` pre-May-2026 binder SeqID!
"""

from __future__ import annotations

import json
from pathlib import Path
import re
from typing import Any, Dict, List, Optional, Tuple

import httpx
import numpy as np

from protein_without_tools.pdb_utils import AA3_TO_1, CANONICAL_AA_SET


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
    r"botulinum|anthrax|ebola|marburg|variola|influenza|corona|sars|\bhiv\b|nipah)",
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
    idx_auth_seq = col_idx.get("auth_seq_id", idx_seq)
    idx_alt = col_idx.get("label_alt_id", -1)
    idx_mod = col_idx.get("pdbx_PDB_model_num", -1)
    idx_x, idx_y, idx_z = col_idx.get("Cartn_x", -1), col_idx.get("Cartn_y", -1), col_idx.get("Cartn_z", -1)

    first_model: Optional[str] = None
    raw_chains: Dict[str, Dict[int, Tuple[str, int, Dict[str, List[float]]]]] = {}

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
        auth_snum = int(toks[idx_auth_seq]) if toks[idx_auth_seq].lstrip("-").isdigit() else snum
        ch = toks[idx_asym]
        ch_map = raw_chains.setdefault(ch, {})
        aa1 = AA3_TO_1[comp]
        if snum not in ch_map:
            ch_map[snum] = (aa1, auth_snum, {})
        if atom not in ch_map[snum][2]:
            ch_map[snum][2][atom] = [float(toks[idx_x]), float(toks[idx_y]), float(toks[idx_z])]

    out: Dict[str, Dict[str, Any]] = {}
    for ch, res_map in raw_chains.items():
        ordered = [
            (sn, aa, asn, atoms)
            for sn, (aa, asn, atoms) in sorted(res_map.items())
            if all(a in atoms for a in ("N", "CA", "C", "O")) and aa in CANONICAL_AA_SET
        ]
        if len(ordered) < 35:
            continue
        best_seg: List[Tuple[int, str, int, Dict[str, List[float]]]] = []
        cur_seg = [ordered[0]]
        for item in ordered[1:]:
            prev_ca = np.array(cur_seg[-1][3]["CA"])
            cur_ca = np.array(item[3]["CA"])
            if float(np.linalg.norm(cur_ca - prev_ca)) <= 4.6:
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
        n_arr = np.array([x[3]["N"] for x in best_seg], dtype=np.float64)
        ca_arr = np.array([x[3]["CA"] for x in best_seg], dtype=np.float64)
        c_arr = np.array([x[3]["C"] for x in best_seg], dtype=np.float64)
        o_arr = np.array([x[3]["O"] for x in best_seg], dtype=np.float64)
        cb_arr = compute_virtual_cb(n_arr, ca_arr, c_arr)
        out[ch] = {
            "chain_id": ch,
            "sequence": seq,
            "length": len(seq),
            "res_nums": [x[0] for x in best_seg],
            "auth_res_nums": [x[2] for x in best_seg],
            "n": n_arr,
            "ca": ca_arr,
            "c": c_arr,
            "o": o_arr,
            "cb": cb_arr,
            "rg": radius_of_gyration(ca_arr),
        }
    return out


def check_precutoff_pdb_seqid(client: httpx.Client, seq: str, cutoff: str = "2026-05-01") -> Tuple[float, List[str]]:
    """Return `(max_seqid_pct, hit_entity_ids)` in RCSB PDB released strictly before `cutoff` (WITH explain metadata!)."""
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
                        "identity_cutoff": 0.25,
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
        "request_options": {
            "paginate": {"start": 0, "rows": 25},
            "scoring_strategy": "sequence",
            "return_explain_metadata": True,
        },
    }
    r = client.post("https://search.rcsb.org/rcsbsearch/v2/query", json=q)
    if r.status_code in (204, 404):
        return 0.0, []
    if r.status_code != 200:
        return 100.0, ["http_err"]
    hits = r.json().get("result_set") or []
    if not hits:
        return 0.0, []
    hit_ids = [str(h.get("identifier", "")) for h in hits[:10]]
    best_id = 30.0  # conservative non-zero floor whenever >=25% identity hits exist!
    for h in hits[:10]:
        for sc in h.get("services") or []:
            for n in sc.get("nodes") or []:
                for m in n.get("match_context") or []:
                    sid = float(m.get("sequence_identity", 0.0)) * 100.0
                    if sid > best_id:
                        best_id = sid
    return round(best_id, 1), hit_ids


def main() -> None:
    repo = Path(__file__).resolve().parents[1]
    raw_dir = repo / "data" / "raw_stage3b_cache"
    cif_dir = raw_dir / "cifs"
    cfg_dir = raw_dir / "upstream_configs"
    cif_dir.mkdir(parents=True, exist_ok=True)
    cfg_dir.mkdir(parents=True, exist_ok=True)

    with httpx.Client(timeout=60.0, follow_redirects=True) as client:
        # 1. Fixed PDBs across Part 1 + Part 2 Tier 1 (`ProtDBench`/`RFD4`) + Part 2 Tier 2 (`BoltzProt-1`) + `9S83`
        fixed_pdbs = [
            # Part 1 motifs/folds
            "1CLL", "1AAY", "5P21", "1SHG",
            # Tier 1: RFD4-Proteina / Proteina-Complexa / ProtDBench classics
            "5O45", "1WWW", "3DI3", "3MJG",
            # Tier 2: BoltzProt-1 / BoltzGen 5 hard novel targets
            "3QKG", "1G13", "1JQD", "7AAH", "1NB0",
            # Tier 3 anchor: Aug-2026 wet-lab de novo minibinder JUBO4 + LEDGF PWWP
            "9S83",
        ]
        for pid in fixed_pdbs:
            dest = cif_dir / f"{pid}.cif"
            if not dest.exists():
                r = client.get(f"https://files.rcsb.org/download/{pid}.cif")
                r.raise_for_status()
                dest.write_text(r.text, encoding="utf-8")
            print(f"  Cached {pid}.cif ({dest.stat().st_size / 1024:.1f} KB)", flush=True)

        # 2. Check upstream `HannesStark/boltzgen` & `congliuUvA/ProtDBench` trees for any public YAML target specs
        for repo_slug, local_tag in (("HannesStark/boltzgen", "boltzgen"), ("congliuUvA/ProtDBench", "protd_bench")):
            try:
                rt = client.get(f"https://api.github.com/repos/{repo_slug}/git/trees/HEAD?recursive=1")
                if rt.status_code == 200:
                    paths = [
                        item["path"]
                        for item in (rt.json().get("tree") or [])
                        if any(k in item["path"].lower() for k in ("3qkg", "1g13", "1jqd", "7aah", "1nb0", "3di3", "3mjg", "1www", "5o45", "target", "benchmark"))
                        and item["path"].endswith((".yaml", ".yml", ".json", ".csv", ".tsv"))
                    ]
                    (cfg_dir / f"{local_tag}_paths.json").write_text(json.dumps(paths, indent=2), encoding="utf-8")
                    print(f"  [{repo_slug}] Found {len(paths)} matching config paths", flush=True)
                    for p in paths[:15]:
                        rcf = client.get(f"https://raw.githubusercontent.com/{repo_slug}/HEAD/{p}")
                        if rcf.status_code == 200 and len(rcf.text) < 250_000:
                            safe_n = p.replace("/", "__")
                            (cfg_dir / f"{local_tag}__{safe_n}").write_text(rcf.text, encoding="utf-8")
            except Exception as exc:
                print(f"  [{repo_slug}] skipped config probe: {exc}", flush=True)

        # 3. Search RCSB specifically for post-May-2026 (`>= 2026-05-01`) Wet-Lab Synthetic De Novo Binder Complexes!
        #    (`ncbi_taxonomy_id == 32630` "synthetic construct" + `polymer_entity_count_protein >= 2`)
        print("Searching RCSB for post-2026-05-01 wet-lab synthetic de novo binder complexes...", flush=True)
        q_denovo = {
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
                            "attribute": "rcsb_entity_source_organism.ncbi_scientific_name",
                            "operator": "exact_match",
                            "value": "synthetic construct",
                        },
                    },
                    {
                        "type": "terminal",
                        "service": "text",
                        "parameters": {
                            "attribute": "rcsb_entry_info.polymer_entity_count_protein",
                            "operator": "greater_or_equal",
                            "value": 2,
                        },
                    },
                    {
                        "type": "terminal",
                        "service": "text",
                        "parameters": {
                            "attribute": "rcsb_entry_info.deposited_polymer_monomer_count",
                            "operator": "less_or_equal",
                            "value": 900,
                        },
                    },
                ],
            },
            "return_type": "entry",
            "request_options": {"paginate": {"start": 0, "rows": 180}},
        }
        rd = client.post("https://search.rcsb.org/rcsbsearch/v2/query", json=q_denovo)
        rd.raise_for_status()
        cand_ids = [h["identifier"] for h in (rd.json().get("result_set") or [])]
        print(f"  Found {len(cand_ids)} post-May-2026 synthetic-construct protein complexes.", flush=True)

        gql = """
        query($ids: [String!]!) {
          entries(entry_ids: $ids) {
            rcsb_id
            struct { title }
            rcsb_accession_info { initial_release_date }
            polymer_entities {
              rcsb_entity_source_organism { ncbi_scientific_name ncbi_taxonomy_id }
              entity_poly { pdbx_strand_id pdbx_seq_one_letter_code_can }
              rcsb_polymer_entity { pdbx_description }
            }
          }
        }
        """
        ab_regex = re.compile(r"\b(fab|scfv|nanobody|vhh|antibody|immunoglobulin|heavy chain|light chain|sybody|darpin)\b", re.I)
        denovo_kw = re.compile(r"\b(de novo|designed|minibinder|miniprotein|binder|rfdiffusion|hallucin|scaffold)\b", re.I)

        cands_filtered = []
        for i in range(0, len(cand_ids), 40):
            rg = client.post("https://data.rcsb.org/graphql", json={"query": gql, "variables": {"ids": cand_ids[i : i + 40]}})
            if rg.status_code != 200:
                continue
            for ent in ((rg.json().get("data") or {}).get("entries") or []):
                eid = ent["rcsb_id"]
                title = str((ent.get("struct") or {}).get("title") or "")
                rdate = str((ent.get("rcsb_accession_info") or {}).get("initial_release_date") or "")[:10]
                pes = ent.get("polymer_entities") or []
                blob_parts = [title]
                has_synth_miniprotein = False
                has_target = False
                for pe in pes:
                    orgs = pe.get("rcsb_entity_source_organism") or []
                    txids = {o.get("ncbi_taxonomy_id") for o in orgs}
                    onames = [str(o.get("ncbi_scientific_name") or "") for o in orgs]
                    desc = str((pe.get("rcsb_polymer_entity") or {}).get("pdbx_description") or "")
                    seq = str((pe.get("entity_poly") or {}).get("pdbx_seq_one_letter_code_can") or "").replace("\n", "")
                    blob_parts.extend(onames + [desc])
                    if 32630 in txids and 42 <= len(seq) <= 125:
                        has_synth_miniprotein = True
                    elif 32630 not in txids and 65 <= len(seq) <= 320:
                        has_target = True
                blob = " ".join(blob_parts)
                if SAFETY_BLOCKLIST.search(blob) or ab_regex.search(blob):
                    continue
                if has_synth_miniprotein and (has_target or denovo_kw.search(blob)):
                    prio = 0 if denovo_kw.search(blob) else 1
                    cands_filtered.append((prio, eid, rdate, title, pes))

        cands_filtered.sort()
        print(f"  {len(cands_filtered)} non-viral, non-antibody synthetic miniprotein complex candidates after title filter.", flush=True)

        accepted_denovo: List[Dict[str, Any]] = []
        for _prio, eid, rdate, title, pes in cands_filtered:
            if len(accepted_denovo) >= 6:
                break
            # Identify synthetic binder entity (42..125 AA) with STRICT `< 30%` pre-May-2026 PDB sequence identity!
            synth_chains: List[str] = []
            target_chains: List[str] = []
            binder_sid = 100.0
            for pe in pes:
                txids = {o.get("ncbi_taxonomy_id") for o in (pe.get("rcsb_entity_source_organism") or [])}
                strands = [s.strip() for s in str((pe.get("entity_poly") or {}).get("pdbx_strand_id") or "").split(",") if s.strip()]
                seq = str((pe.get("entity_poly") or {}).get("pdbx_seq_one_letter_code_can") or "").replace("\n", "")
                if 32630 in txids and 42 <= len(seq) <= 125:
                    sid, hits = check_precutoff_pdb_seqid(client, seq)
                    if sid < 30.0 and len(hits) == 0:
                        synth_chains.extend(strands)
                        binder_sid = sid
                elif 65 <= len(seq) <= 320:
                    target_chains.extend(strands)

            if not synth_chains or not target_chains:
                continue

            cif_p = cif_dir / f"{eid}.cif"
            if not cif_p.exists():
                rc = client.get(f"https://files.rcsb.org/download/{eid}.cif")
                if rc.status_code != 200:
                    continue
                cif_p.write_text(rc.text, encoding="utf-8")

            chs = parse_cif_all_protein_chains(cif_p.read_text(encoding="utf-8"))
            best_pair = None
            for tc in target_chains:
                for bc in synth_chains:
                    if tc not in chs or bc not in chs:
                        continue
                    ta, tb = chs[tc], chs[bc]
                    if not (65 <= ta["length"] <= 280 and 42 <= tb["length"] <= 120):
                        continue
                    if ta["rg"] > 2.45 * (ta["length"] ** 0.38):
                        continue
                    dmat = np.linalg.norm(ta["cb"][:, None, :] - tb["cb"][None, :, :], axis=-1)
                    nc = int(np.sum(dmat < 8.0))
                    nhs = int(np.sum(np.min(dmat, axis=1) < 8.0))
                    if nc >= 14 and nhs >= 5:
                        if best_pair is None or nc > best_pair[0]:
                            best_pair = (nc, tc, bc)
            if best_pair:
                nc, tc, bc = best_pair
                rec = {
                    "pdb_id": eid,
                    "release_date": rdate,
                    "title": title,
                    "target_chain": tc,
                    "target_len": chs[tc]["length"],
                    "binder_chain": bc,
                    "binder_len": chs[bc]["length"],
                    "binder_precutoff_pdb_seqid": binder_sid,
                    "interchain_cb8_contacts": nc,
                }
                accepted_denovo.append(rec)
                print(
                    f"  [Accepted Post-May-2026 De Novo Binder Complex] {eid} ({rdate}): "
                    f"Target {tc} ({chs[tc]['length']} AA) + Synthetic Binder {bc} ({chs[bc]['length']} AA, pre-PDB={binder_sid}%), "
                    f"{nc} contacts — {title[:75]}",
                    flush=True,
                )

        (raw_dir / "post_may2026_denovo_binder_complexes.json").write_text(
            json.dumps(accepted_denovo, indent=2), encoding="utf-8"
        )
        print(f"Saved {len(accepted_denovo)} verified post-May-2026 wet-lab de novo binder complexes!", flush=True)


if __name__ == "__main__":
    main()
