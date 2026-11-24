#!/usr/bin/env python3
"""Export zero-leakage Stage 1A (Forward) and Stage 1B (Inverse) prompts.

Generates:
  1. `outputs/prompts/stage1a_forward_prompts.jsonl`
  2. `outputs/prompts/stage1b_inverse_prompts.jsonl`
  3. `outputs/prompts/web_chat_spotcheck/` — Individual `.md` files for key targets
     (e.g. anonymized `6MRR`, `37NT` Domain 1, and a compact Oct/Nov-2026 monomer)
     so you can even paste 2-3 prompts straight into Claude.ai / ChatGPT web chat
     with ZERO API credits and grade the replies offline!
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import numpy as np

from protein_without_tools.pdb_utils import ProteinStructure
from protein_without_tools.stage1_prompts_and_parsers import (
    build_forward_folding_prompt,
    build_inverse_folding_prompt,
)


def load_cached_npz_structure(npz_path: Path, target_id: str, pdb_id: str, chain_id: str) -> ProteinStructure:
    """Rehydrate a `ProteinStructure` from a curated `.npz` archive."""
    d = np.load(npz_path, allow_pickle=False)
    return ProteinStructure(
        target_id=target_id,
        pdb_id=pdb_id,
        chain_id=chain_id,
        sequence=str(d["sequence"]),
        res_numbers=d["res_numbers"].tolist(),
        coords_n=d["coords_n"],
        coords_ca=d["coords_ca"],
        coords_c=d["coords_c"],
        coords_o=d["coords_o"],
        coords_cb=d["coords_cb"],
        phi_deg=d["phi_deg"],
        psi_deg=d["psi_deg"],
        q3=str(d["q3"]),
        cb_dist_matrix=d["cb_dist_matrix"],
        ca_coord_num=d["ca_coord_num"],
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Export Stage 1 zero-leakage LLM prompt files.")
    parser.add_argument("--data-dir", default="data/stage1_benchmark")
    parser.add_argument("--out-dir", default="outputs/prompts")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir)
    spot_dir = out_dir / "web_chat_spotcheck"
    out_dir.mkdir(parents=True, exist_ok=True)
    spot_dir.mkdir(parents=True, exist_ok=True)

    manifest = json.loads((data_dir / "manifest.json").read_text(encoding="utf-8"))
    fwd_lines = []
    inv_lines = []

    for idx, rec in enumerate(manifest["targets"]):
        tid = rec["target_id"]
        struct = load_cached_npz_structure(
            data_dir / "ground_truth" / f"{tid}.npz",
            target_id=tid,
            pdb_id=rec["pdb_id"],
            chain_id=rec["chain_id"],
        )
        p_fwd = build_forward_folding_prompt(struct)
        p_inv = build_inverse_folding_prompt(struct)

        fwd_lines.append(
            json.dumps(
                {
                    "custom_id": f"{tid}_forward",
                    "target_id": tid,
                    "task": "stage1a_forward_folding",
                    "length": struct.length,
                    "prompt": p_fwd,
                }
            )
        )
        inv_lines.append(
            json.dumps(
                {
                    "custom_id": f"{tid}_inverse",
                    "target_id": tid,
                    "task": "stage1b_inverse_folding",
                    "length": struct.length,
                    "prompt": p_inv,
                }
            )
        )

        # Export first 2 pre-cutoff, first 2 37NT, and first 4 post-cutoff monomers for manual web UI spot-checks
        if rec["pdb_id"] in ("6MRR", "37NT") or idx < 4 or (rec["temporal_split"] == "post_cutoff" and idx < 20):
            (spot_dir / f"{tid}_forward.txt").write_text(p_fwd + "\n", encoding="utf-8")
            (spot_dir / f"{tid}_inverse.txt").write_text(p_inv + "\n", encoding="utf-8")

    fwd_path = out_dir / "stage1a_forward_prompts.jsonl"
    inv_path = out_dir / "stage1b_inverse_prompts.jsonl"
    fwd_path.write_text("\n".join(fwd_lines) + "\n", encoding="utf-8")
    inv_path.write_text("\n".join(inv_lines) + "\n", encoding="utf-8")
    print(f"Exported {len(fwd_lines)} Forward prompts -> {fwd_path}")
    print(f"Exported {len(inv_lines)} Inverse prompts -> {inv_path}")
    print(f"Exported ready-to-paste web-chat prompts -> {spot_dir}/")


if __name__ == "__main__":
    main()
