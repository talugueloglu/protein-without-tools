"""PDB and mmCIF backbone/contact/secondary-structure extraction utilities.

Supports both legacy `.pdb` and modern RCSB `.cif` (`mmCIF`) files (including 2026
entries), extracts backbone `(N, CA, C, O)` and `CB` coordinates, computes exact
Ramachandran `(phi, psi)` angles, Kabsch-Sander H-bond `Q3` secondary structure
(`H`/`E`/`C`), `Cβ < 8 Å` (`Cα` for Gly) contact maps, and stripped `UNK` backbones
for leak-free Inverse Folding evaluation.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
import shlex
from typing import Dict, List, Optional, Tuple

import numpy as np

AA3_TO_1: Dict[str, str] = {
    "ALA": "A",
    "ARG": "R",
    "ASN": "N",
    "ASP": "D",
    "CYS": "C",
    "GLN": "Q",
    "GLU": "E",
    "GLY": "G",
    "HIS": "H",
    "ILE": "I",
    "LEU": "L",
    "LYS": "K",
    "MET": "M",
    "MSE": "M",  # Selenomethionine mapped to canonical Met
    "PHE": "F",
    "PRO": "P",
    "SER": "S",
    "THR": "T",
    "TRP": "W",
    "TYR": "Y",
    "VAL": "V",
}
CANONICAL_AA_SET = set("ACDEFGHIKLMNPQRSTVWY")


@dataclass
class ProteinStructure:
    """Clean single-chain protein structure with backbone geometry and contacts."""

    target_id: str
    pdb_id: str
    chain_id: str
    sequence: str
    res_numbers: List[int]
    coords_n: np.ndarray   # (L, 3)
    coords_ca: np.ndarray  # (L, 3)
    coords_c: np.ndarray   # (L, 3)
    coords_o: np.ndarray   # (L, 3)
    coords_cb: np.ndarray  # (L, 3) — true CB or CA for Gly (CASP standard)
    phi_deg: np.ndarray    # (L,) degrees, NaN at N-terminus / chain breaks
    psi_deg: np.ndarray    # (L,) degrees, NaN at C-terminus / chain breaks
    q3: str                # Length-L string over {'H', 'E', 'C'}
    cb_dist_matrix: np.ndarray  # (L, L) Å
    ca_coord_num: np.ndarray    # (L,) number of Cα neighbors within 10 Å (excluding self)

    @property
    def length(self) -> int:
        return len(self.sequence)

    @property
    def is_core_mask(self) -> np.ndarray:
        """Residues with >= 14 Cα neighbors within 10 Å (standard buried core proxy)."""
        return self.ca_coord_num >= 14


def compute_virtual_cb(n: np.ndarray, ca: np.ndarray, c: np.ndarray) -> np.ndarray:
    """Construct idealized Cβ coordinates from backbone (N, Cα, C) tetrahedral geometry."""
    b = ca - n
    c_vec = c - ca
    a = np.cross(b, c_vec)
    return -0.58273431 * a + 0.56802827 * b - 0.54067466 * c_vec + ca


def _dihedral_deg(p0: np.ndarray, p1: np.ndarray, p2: np.ndarray, p3: np.ndarray) -> float:
    """Compute signed IUPAC dihedral angle in degrees."""
    b0 = p0 - p1
    b1 = p2 - p1
    b2 = p3 - p2
    norm_b1 = np.linalg.norm(b1)
    if norm_b1 < 1e-6:
        return float("nan")
    b1_u = b1 / norm_b1
    v = b0 - np.dot(b0, b1_u) * b1_u
    w = b2 - np.dot(b2, b1_u) * b1_u
    if np.linalg.norm(v) < 1e-6 or np.linalg.norm(w) < 1e-6:
        return float("nan")
    x = np.dot(v, w)
    y = np.dot(np.cross(b1_u, v), w)
    return math.degrees(math.atan2(y, x))


def compute_phi_psi(
    coords_n: np.ndarray,
    coords_ca: np.ndarray,
    coords_c: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute per-residue Ramachandran (phi, psi) angles in degrees, respecting chain breaks."""
    L = coords_ca.shape[0]
    phi = np.full(L, np.nan, dtype=np.float64)
    psi = np.full(L, np.nan, dtype=np.float64)
    for i in range(L):
        if i > 0 and np.linalg.norm(coords_n[i] - coords_c[i - 1]) < 2.0:
            phi[i] = _dihedral_deg(coords_c[i - 1], coords_n[i], coords_ca[i], coords_c[i])
        if i < L - 1 and np.linalg.norm(coords_n[i + 1] - coords_c[i]) < 2.0:
            psi[i] = _dihedral_deg(coords_n[i], coords_ca[i], coords_c[i], coords_n[i + 1])
    return phi, psi


def compute_kabsch_sander_q3(
    sequence: str,
    coords_n: np.ndarray,
    coords_ca: np.ndarray,
    coords_c: np.ndarray,
    coords_o: np.ndarray,
) -> str:
    """Assign 3-state secondary structure ('H', 'E', 'C') via Kabsch-Sander (DSSP) H-bonds."""
    L = len(sequence)
    if L < 5:
        return "C" * L

    # 1. Reconstruct idealized amide H positions for residues i >= 1 (except Proline)
    coords_h = np.full((L, 3), np.nan, dtype=np.float64)
    for i in range(1, L):
        if sequence[i] == "P":
            continue
        cn = coords_n[i] - coords_c[i - 1]
        can = coords_n[i] - coords_ca[i]
        cn_n = np.linalg.norm(cn)
        can_n = np.linalg.norm(can)
        if cn_n > 2.2 or cn_n < 1e-6 or can_n < 1e-6:
            continue
        bisector = (cn / cn_n) + (can / can_n)
        b_norm = np.linalg.norm(bisector)
        if b_norm > 1e-6:
            coords_h[i] = coords_n[i] + 1.01 * (bisector / b_norm)

    # 2. Pairwise Kabsch-Sander electrostatic H-bond matrix: hbond[donor_nh, acceptor_co]
    # E = 0.084 * (1/r_ON + 1/r_CH - 1/r_OH - 1/r_CN) * 332 kcal/mol < -0.5 kcal/mol
    hbond = np.zeros((L, L), dtype=bool)
    for i in range(1, L):  # donor N-H at i
        if np.isnan(coords_h[i, 0]):
            continue
        for j in range(L):  # acceptor C=O at j
            if abs(i - j) < 2:
                continue
            r_on = np.linalg.norm(coords_o[j] - coords_n[i])
            if r_on > 5.2 or r_on < 0.5:
                continue
            r_ch = np.linalg.norm(coords_c[j] - coords_h[i])
            r_oh = np.linalg.norm(coords_o[j] - coords_h[i])
            r_cn = np.linalg.norm(coords_c[j] - coords_n[i])
            if min(r_ch, r_oh, r_cn) < 0.5:
                continue
            energy = 27.888 * (1.0 / r_on + 1.0 / r_ch - 1.0 / r_oh - 1.0 / r_cn)
            if energy < -0.5:
                hbond[i, j] = True

    # 3. Assign Helices (4-turns i<-i+4, 3-turns i<-i+3, 5-turns i<-i+5) & Beta bridges
    is_helix = np.zeros(L, dtype=bool)
    for n_turn in (4, 3, 5):
        turn = np.zeros(L, dtype=bool)
        for j in range(L - n_turn):
            if hbond[j + n_turn, j]:
                turn[j] = True
        # Consecutive n-turns define a helix (DSSP rule)
        for j in range(L - n_turn - 1):
            if turn[j] and turn[j + 1]:
                is_helix[j + 1 : j + n_turn + 1] = True

    is_strand = np.zeros(L, dtype=bool)
    for i in range(1, L - 1):
        for j in range(1, L - 1):
            if abs(i - j) < 3:
                continue
            # Parallel bridge: (i-1 -> j and j -> i+1) or (j-1 -> i and i -> j+1)
            par = (hbond[j, i - 1] and hbond[i + 1, j]) or (hbond[i, j - 1] and hbond[j + 1, i])
            # Antiparallel bridge: (i -> j and j -> i) or (i-1 -> j+1 and j-1 -> i+1)
            anti = (hbond[j, i] and hbond[i, j]) or (hbond[j + 1, i - 1] and hbond[i + 1, j - 1])
            if par or anti:
                is_strand[i] = True
                is_strand[j] = True

    ss = []
    for idx in range(L):
        if is_helix[idx]:
            ss.append("H")
        elif is_strand[idx]:
            ss.append("E")
        else:
            ss.append("C")
    return "".join(ss)


def _parse_pdb_atoms(text: str, chain_id: Optional[str]) -> Tuple[str, Dict[Tuple[int, str], Dict[str, Tuple[str, np.ndarray]]]]:
    """Extract first-model ATOM records from legacy PDB text."""
    residues: Dict[Tuple[int, str], Dict[str, Tuple[str, np.ndarray]]] = {}
    chosen_chain: Optional[str] = chain_id
    for line in text.splitlines():
        if line.startswith("ENDMDL"):
            break
        if not line.startswith(("ATOM  ", "HETATM")):
            continue
        res_name = line[17:20].strip()
        if line.startswith("HETATM") and res_name != "MSE":
            continue
        if res_name not in AA3_TO_1:
            continue
        altloc = line[16]
        if altloc not in (" ", "A", "1"):
            continue
        ch = line[21].strip() or "A"
        if chosen_chain is None:
            chosen_chain = ch
        if ch != chosen_chain:
            continue
        atom_name = line[12:16].strip()
        if atom_name not in ("N", "CA", "C", "O", "CB"):
            continue
        res_seq = int(line[22:26].strip())
        icode = line[26].strip()
        xyz = np.array(
            [float(line[30:38]), float(line[38:46]), float(line[46:54])],
            dtype=np.float64,
        )
        key = (res_seq, icode)
        if key not in residues:
            residues[key] = {}
        if atom_name not in residues[key]:
            residues[key][atom_name] = (res_name, xyz)
    return chosen_chain or "A", residues


def _parse_mmcif_atoms(text: str, chain_id: Optional[str]) -> Tuple[str, Dict[Tuple[int, str], Dict[str, Tuple[str, np.ndarray]]]]:
    """Extract first-model `_atom_site` records from mmCIF text."""
    lines = text.splitlines()
    n_lines = len(lines)
    i = 0
    residues: Dict[Tuple[int, str], Dict[str, Tuple[str, np.ndarray]]] = {}
    chosen_chain: Optional[str] = chain_id
    first_model_num: Optional[str] = None

    while i < n_lines:
        if lines[i].strip() == "loop_":
            i += 1
            headers: List[str] = []
            while i < n_lines and lines[i].strip().startswith("_"):
                headers.append(lines[i].strip())
                i += 1
            if headers and headers[0].startswith("_atom_site."):
                col = {h.split(".", 1)[1]: idx for idx, h in enumerate(headers)}
                while i < n_lines:
                    s = lines[i].strip()
                    if not s or s.startswith("#") or s == "loop_" or s.startswith("_"):
                        break
                    toks = shlex.split(s)
                    i += 1
                    if len(toks) < len(headers):
                        continue
                    group = toks[col["group_PDB"]] if "group_PDB" in col else "ATOM"
                    comp_id = toks[col.get("auth_comp_id", col.get("label_comp_id", 0))]
                    if group == "HETATM" and comp_id != "MSE":
                        continue
                    if comp_id not in AA3_TO_1:
                        continue
                    if "pdbx_PDB_model_num" in col:
                        mnum = toks[col["pdbx_PDB_model_num"]]
                        if first_model_num is None:
                            first_model_num = mnum
                        elif mnum != first_model_num:
                            continue
                    if "label_alt_id" in col:
                        alt = toks[col["label_alt_id"]]
                        if alt not in (".", "?", "A", "1"):
                            continue
                    ch_col = "auth_asym_id" if "auth_asym_id" in col else "label_asym_id"
                    ch = toks[col[ch_col]]
                    if chosen_chain is None:
                        chosen_chain = ch
                    if ch != chosen_chain:
                        continue
                    atom_col = "auth_atom_id" if "auth_atom_id" in col else "label_atom_id"
                    atom_name = toks[col[atom_col]]
                    if atom_name not in ("N", "CA", "C", "O", "CB"):
                        continue
                    seq_col = "auth_seq_id" if "auth_seq_id" in col else "label_seq_id"
                    seq_str = toks[col[seq_col]]
                    if seq_str in (".", "?"):
                        continue
                    res_seq = int(seq_str)
                    icode = ""
                    if "pdbx_PDB_ins_code" in col:
                        ic = toks[col["pdbx_PDB_ins_code"]]
                        if ic not in (".", "?"):
                            icode = ic
                    xyz = np.array(
                        [
                            float(toks[col["Cartn_x"]]),
                            float(toks[col["Cartn_y"]]),
                            float(toks[col["Cartn_z"]]),
                        ],
                        dtype=np.float64,
                    )
                    key = (res_seq, icode)
                    if key not in residues:
                        residues[key] = {}
                    if atom_name not in residues[key]:
                        residues[key][atom_name] = (comp_id, xyz)
                break
        else:
            i += 1
    return chosen_chain or "A", residues


def load_structure(
    path: str | Path,
    pdb_id: str,
    target_id: Optional[str] = None,
    chain_id: Optional[str] = None,
    res_range: Optional[Tuple[int, int]] = None,
) -> ProteinStructure:
    """Load a `.pdb` or `.cif` file into a canonical `ProteinStructure`."""
    p = Path(path)
    text = p.read_text(encoding="utf-8", errors="replace")
    if p.suffix.lower() in (".cif", ".mmcif") or "_atom_site." in text:
        used_chain, raw_res = _parse_mmcif_atoms(text, chain_id)
    else:
        used_chain, raw_res = _parse_pdb_atoms(text, chain_id)

    seq_chars: List[str] = []
    res_nums: List[int] = []
    n_list, ca_list, c_list, o_list, cb_list = [], [], [], [], []

    for (res_num, _icode), atoms in raw_res.items():
        if res_range is not None and not (res_range[0] <= res_num <= res_range[1]):
            continue
        if not all(k in atoms for k in ("N", "CA", "C", "O")):
            continue
        res_name = atoms["CA"][0]
        aa1 = AA3_TO_1[res_name]
        n_xyz = atoms["N"][1]
        ca_xyz = atoms["CA"][1]
        c_xyz = atoms["C"][1]
        o_xyz = atoms["O"][1]
        if aa1 == "G":
            cb_xyz = ca_xyz.copy()  # CASP / Rao standard: Cα used for Glycine
        elif "CB" in atoms:
            cb_xyz = atoms["CB"][1]
        else:
            cb_xyz = compute_virtual_cb(n_xyz, ca_xyz, c_xyz)

        seq_chars.append(aa1)
        res_nums.append(res_num)
        n_list.append(n_xyz)
        ca_list.append(ca_xyz)
        c_list.append(c_xyz)
        o_list.append(o_xyz)
        cb_list.append(cb_xyz)

    if not seq_chars:
        raise ValueError(f"No complete backbone residues extracted from {path} (chain={chain_id})")

    sequence = "".join(seq_chars)
    coords_n = np.stack(n_list, axis=0)
    coords_ca = np.stack(ca_list, axis=0)
    coords_c = np.stack(c_list, axis=0)
    coords_o = np.stack(o_list, axis=0)
    coords_cb = np.stack(cb_list, axis=0)

    phi_deg, psi_deg = compute_phi_psi(coords_n, coords_ca, coords_c)
    q3 = compute_kabsch_sander_q3(sequence, coords_n, coords_ca, coords_c, coords_o)

    diff_cb = coords_cb[:, None, :] - coords_cb[None, :, :]
    cb_dist = np.linalg.norm(diff_cb, axis=-1)

    diff_ca = coords_ca[:, None, :] - coords_ca[None, :, :]
    ca_dist = np.linalg.norm(diff_ca, axis=-1)
    ca_coord_num = np.sum((ca_dist < 10.0) & (ca_dist > 1e-4), axis=1)

    return ProteinStructure(
        target_id=target_id or f"{pdb_id}_{used_chain}",
        pdb_id=pdb_id,
        chain_id=used_chain,
        sequence=sequence,
        res_numbers=res_nums,
        coords_n=coords_n,
        coords_ca=coords_ca,
        coords_c=coords_c,
        coords_o=coords_o,
        coords_cb=coords_cb,
        phi_deg=phi_deg,
        psi_deg=psi_deg,
        q3=q3,
        cb_dist_matrix=cb_dist,
        ca_coord_num=ca_coord_num,
    )


def write_backbone_only_pdb(struct: ProteinStructure, out_path: str | Path) -> None:
    """Write a stripped backbone-only PDB (`GLY` / `N, CA, C, O` only, 1..L renumbered).

    Ensures zero sequence identity leakage when passing structures to ProteinMPNN,
    ESM-IF1, or downstream visualizers.
    """
    lines: List[str] = []
    atom_serial = 1
    center = struct.coords_ca.mean(axis=0)
    for idx in range(struct.length):
        res_idx = idx + 1
        for atom_name, arr, elem in (
            ("N", struct.coords_n[idx] - center, "N"),
            ("CA", struct.coords_ca[idx] - center, "C"),
            ("C", struct.coords_c[idx] - center, "C"),
            ("O", struct.coords_o[idx] - center, "O"),
        ):
            lines.append(
                f"ATOM  {atom_serial:5d}  {atom_name:<3s} GLY A{res_idx:4d}    "
                f"{arr[0]:8.3f}{arr[1]:8.3f}{arr[2]:8.3f}  1.00  0.00           {elem:>2s}"
            )
            atom_serial += 1
    lines.append("TER")
    lines.append("END")
    Path(out_path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_ca_trace_pdb(sequence: str, coords_ca: np.ndarray, out_path: str | Path) -> None:
    """Write a Cα trace PDB for 3D visualization (3Dmol.js / PyMOL) and TMalign."""
    aa1_to_3 = {v: k for k, v in AA3_TO_1.items() if k != "MSE"}
    lines: List[str] = []
    center = coords_ca.mean(axis=0)
    for idx, aa in enumerate(sequence):
        res3 = aa1_to_3.get(aa, "UNK")
        xyz = coords_ca[idx] - center
        lines.append(
            f"ATOM  {idx + 1:5d}  CA  {res3} A{idx + 1:4d}    "
            f"{xyz[0]:8.3f}{xyz[1]:8.3f}{xyz[2]:8.3f}  1.00  0.00           C"
        )
    lines.append("TER")
    lines.append("END")
    Path(out_path).write_text("\n".join(lines) + "\n", encoding="utf-8")
