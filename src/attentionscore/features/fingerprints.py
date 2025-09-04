from __future__ import annotations
import numpy as np
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem
from rdkit.Avalon import pyAvalonTools
import base64
from typing import Literal
import pandas as pd
from rdkit.Chem.rdFingerprintGenerator import GetMorganGenerator

def calculate_avalon_array(smiles: str, nBits: int = 512) -> np.ndarray:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES string: {smiles}")
    fingerprint = pyAvalonTools.GetAvalonFP(mol, nBits=nBits)
    array = np.zeros((nBits,), dtype=int)
    DataStructs.ConvertToNumpyArray(fingerprint, array)
    return array


def ecfp4_dataframe(
    df: pd.DataFrame,
    smiles_col: str = "SMILES_STD",
    id_col: str = "ID",
    n_bits: int = 2048,
    radius: int = 2,                     # ECFP4 => radius=2
    drop_invalid: bool = True,
    output: Literal["base64", "numpy", "bitvect"] = "base64",
) -> pd.DataFrame:
    """
    Compute ECFP4 (Morgan) fingerprints for SMILES in a DataFrame.

    Parameters
    ----------
    df : DataFrame with at least [smiles_col] and optionally [id_col]
    smiles_col : column name with SMILES (default: 'SMILES_STD')
    id_col : identifier column (default: 'ID'), if missing, uses row index
    n_bits : fingerprint length (default: 2048)
    radius : Morgan radius (ECFP4 uses 2)
    drop_invalid : drop invalid SMILES if True, else raise ValueError
    output : one of {'base64','numpy','bitvect'}

    Returns
    -------
    DataFrame with columns:
      - id, smiles, ecfp4_bits
      - plus one of:
          * ecfp4_b64 (if output='base64')
          * ecfp4_np  (if output='numpy')
          * ecfp4_bv  (if output='bitvect', RDKit ExplicitBitVect objects)
    """
    # 1) Parse SMILES (collect valid/invalid)
    mols, valid_idx, invalid_idx = [], [], []
    for i, smi in enumerate(df[smiles_col].astype(str).values):
        m = Chem.MolFromSmiles(smi)
        if m is None:
            invalid_idx.append(i)
            if not drop_invalid:
                raise ValueError(f"Invalid SMILES at index {i}: {smi}")
        else:
            mols.append(m)
            valid_idx.append(i)

    if invalid_idx:
        print(f"[info] Dropped {len(invalid_idx)} invalid SMILES."
              if drop_invalid else f"[warn] Found {len(invalid_idx)} invalid SMILES.")

    if not mols:
        raise ValueError("No valid molecules to fingerprint.")

    df_valid = df.iloc[valid_idx].reset_index(drop=True)

    # 2) Generate ECFP4
    morgan = GetMorganGenerator(radius=radius, fpSize=n_bits)
    fps = [morgan.GetFingerprint(m) for m in mols]

    def bv_to_numpy(bv):
        arr = np.zeros((n_bits,), dtype=np.int8)
        DataStructs.ConvertToNumpyArray(bv, arr)
        return arr

    def pack_bits(arr: np.ndarray) -> bytes:
        return np.packbits(arr.astype(np.uint8)).tobytes()

    rows = []
    for i, (bv, smi) in enumerate(zip(fps, df_valid[smiles_col].values)):
        rid = df_valid[id_col].iloc[i] if id_col in df_valid.columns else i
        base = {"id": str(rid), "smiles": smi, "ecfp4_bits": n_bits}
        if output == "numpy":
            rows.append({**base, "ecfp4_np": bv_to_numpy(bv)})
        elif output == "bitvect":
            rows.append({**base, "ecfp4_bv": bv})
        else:  # 'base64' (compact, Parquet-friendly)
            b64 = base64.b64encode(pack_bits(bv_to_numpy(bv))).decode("ascii")
            rows.append({**base, "ecfp4_b64": b64})

    return pd.DataFrame(rows)
