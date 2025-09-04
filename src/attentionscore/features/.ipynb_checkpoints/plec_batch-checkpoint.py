from __future__ import annotations
import os, glob
from typing import List, Tuple
import numpy as np
import oddt
from oddt.fingerprints import PLEC

__all__ = ["plec_features_from_dir"]

def _numeric_key(path: str):
    b = os.path.basename(path)
    digits = "".join(ch for ch in b if ch.isdigit())
    return (int(digits) if digits else float("inf"), b)

def plec_features_from_dir(
    docked_dir: str,
    receptor_pdb: str,
    *,
    plec_size: int = 4092,
    depth_protein: int = 4,
    depth_ligand: int = 2,
    distance_cutoff: float = 4.5,
    sparse: bool = False,
    pattern: str = "*.sdf",
    numeric_sort: bool = True,
) -> Tuple[np.ndarray, List[str]]:
    """
    Build a (N, plec_size) array over SDF files in docked_dir.
    Assumes 1 docked pose per SDF (common when --num_modes 1).
    Returns (plec_array, files_sorted).
    """
    files = glob.glob(os.path.join(docked_dir, pattern))
    files.sort(key=_numeric_key if numeric_sort else None)

    protein = next(oddt.toolkit.readfile("pdb", receptor_pdb))

    feats: List[np.ndarray] = []
    for f in files:
        lig = next(oddt.toolkit.readfile("sdf", f))
        vec = PLEC(lig, protein=protein, size=plec_size,
                   depth_protein=depth_protein, depth_ligand=depth_ligand,
                   distance_cutoff=distance_cutoff, sparse=sparse)
        feats.append(np.asarray(vec, dtype=np.float32))
    X = np.vstack(feats) if feats else np.zeros((0, plec_size), dtype=np.float32)
    return X, files
