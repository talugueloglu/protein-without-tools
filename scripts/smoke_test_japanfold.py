#!/usr/bin/env python3
"""1-monomer + 1-complex smoke test for JapanFold API (< $0.01 total)."""

from __future__ import annotations

import io
import json
from pathlib import Path
import sys

import numpy as np

REPO = Path(__file__).resolve().parents[1]
if str(REPO / "src") not in sys.path:
    sys.path.insert(0, str(REPO / "src"))

from protein_without_tools.japanfold_client import JapanFoldClient, parse_pdb_chain_with_bfactor


def main() -> None:
    with JapanFoldClient(repo_root=REPO) as jf:
        # 0. Fetch root discovery / guide (free)
        r_root = jf._request("GET", "/")
        if r_root.status_code == 200:
            (REPO / "outputs" / "colab" / "japanfold_root.json").write_text(r_root.text, encoding="utf-8")
            root_j = r_root.json()
            print("Root keys:", list(root_j.keys()))
            if "guide" in root_j:
                g_url = root_j["guide"]
                rg = jf._request("GET", g_url)
                if rg.status_code == 200:
                    (REPO / "outputs" / "colab" / "japanfold_guide.txt").write_text(rg.text, encoding="utf-8")
                    print(f"Saved guide ({len(rg.text)} bytes)")

        # 1. Smoke test ESMFold-2 monomer (Test 2 / Test 7 single-sequence oracle)
        gb1_seq = "MTYKLILNGKTLKGETTTEAVDAATAEKVFKQYANDNGVDGEWTYDDATKTFTVTE"
        meta_m, files_m = jf.submit_and_collect(
            "/v1/predictions",
            {
                "model": "esmfold2",
                "name": "smoke_gb1",
                "sequence": gb1_seq,
                "params": {"fast": False, "use_msa_server": False, "output_format": "pdb"},
            },
            cache_prefix="smoke_esm2",
        )
        print("\n=== ESMFold-2 Monomer Smoke ===")
        print("Cost USD:", meta_m["cost_usd"])
        print("Results rows:", json.dumps(meta_m["results"].get("rows"), indent=2))
        print("Files:", list(files_m.keys()))
        pdb_files = [k for k in files_m if k.endswith(".pdb")]
        if pdb_files:
            ch = parse_pdb_chain_with_bfactor(files_m[pdb_files[0]].decode("utf-8"), "A")
            print(f"Parsed PDB chain A: L={len(ch['sequence'])}, mean_pLDDT={ch['plddt_mean']:.2f}, Q3={ch['q3'][:25]}...")

        # 2. Smoke test Boltz-2 2-chain complex with PAE (Test 8 binder oracle)
        ep_raw = json.loads((REPO / "data" / "stage3b_design" / "epitope_binder_manifest.json").read_text())["targets"][0]
        fasta_2ch = f">A\n{ep_raw['target_sequence']}\n>B\n{ep_raw['heldout_partner_sequence'][:70]}\n"
        meta_c, files_c = jf.submit_and_collect(
            "/v1/predictions",
            {
                "model": "boltz2",
                "name": "smoke_complex",
                "input": fasta_2ch,
                "params": {"use_msa_server": True, "output_format": "pdb", "write_pae": True},
            },
            cache_prefix="smoke_boltz2",
        )
        print("\n=== Boltz-2 Complex Smoke ===")
        print("Cost USD:", meta_c["cost_usd"])
        print("Results rows:", json.dumps(meta_c["results"].get("rows"), indent=2))
        print("Files:", list(files_c.keys()))
        for k, data in files_c.items():
            if k.endswith(".npz"):
                z = np.load(io.BytesIO(data))
                print("NPZ keys:", list(z.keys()), {ak: (z[ak].shape, str(z[ak].dtype), float(np.min(z[ak])), float(np.max(z[ak]))) for ak in z.keys()})
            elif k.endswith("_pae.json"):
                print("PAE sidecar:", data.decode("utf-8")[:500])
            elif k.endswith(".pdb"):
                ch_a = parse_pdb_chain_with_bfactor(data.decode("utf-8"), "A")
                ch_b = parse_pdb_chain_with_bfactor(data.decode("utf-8"), "B")
                print(f"Complex PDB: Chain A L={len(ch_a['sequence'])} pLDDT={ch_a['plddt_mean']:.1f} | Chain B L={len(ch_b['sequence'])} pLDDT={ch_b['plddt_mean']:.1f}")


if __name__ == "__main__":
    main()
