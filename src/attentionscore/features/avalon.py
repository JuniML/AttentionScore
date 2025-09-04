from __future__ import annotations

import numpy as np
from rdkit import Chem, DataStructs
from rdkit.Avalon import pyAvalonTools

def avalon_fp_array(smiles: str, nBits: int = 512) -> np.ndarray:
    """Return Avalon fingerprint as numpy int array of shape (nBits,)."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES: {smiles}")
    fp = pyAvalonTools.GetAvalonFP(mol, nBits=nBits)
    arr = np.zeros((nBits,), dtype=int)
    DataStructs.ConvertToNumpyArray(fp, arr)
    return arr
