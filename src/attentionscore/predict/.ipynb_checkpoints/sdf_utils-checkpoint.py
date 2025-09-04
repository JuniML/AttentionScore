from __future__ import annotations

import os
import glob
from typing import List, Dict, Optional
import pandas as pd
from rdkit import Chem

__all__ = ["sdf_to_smiles_file", "sdf_dir_to_smiles"]

def sdf_to_smiles_file(
    sdf_path: str,
    *,
    canonical: bool = True,
    isomeric: bool = True,
    kekule: bool = False,
    sanitize: bool = True,
) -> List[Dict[str, str]]:
    """
    Convert all molecules in one SDF file to SMILES.
    Returns a list of dicts: {"file","mol_index","name","smiles"}.
    """
    rows: List[Dict[str, str]] = []
    suppl = Chem.SDMolSupplier(sdf_path, removeHs=False, sanitize=sanitize)
    base = os.path.basename(sdf_path)
    for i, mol in enumerate(suppl):
        if mol is None:
            continue
        name = mol.GetProp("_Name") if mol.HasProp("_Name") else f"{base}#{i}"
        smi = Chem.MolToSmiles(mol, isomericSmiles=isomeric, kekuleSmiles=kekule, canonical=canonical)
        rows.append({"file": sdf_path, "mol_index": i, "name": name, "smiles": smi})
    return rows

def _numeric_key(path: str):
    b = os.path.basename(path)
    digits = "".join(ch for ch in b if ch.isdigit())
    return (int(digits) if digits else float("inf"), b)

def sdf_dir_to_smiles(
    docked_dir: str,
    pattern: str = "*.sdf",
    *,
    canonical: bool = True,
    isomeric: bool = True,
    kekule: bool = False,
    sanitize: bool = True,
    numeric_sort: bool = True,
    output_csv: Optional[str] = None,
    output_smi: Optional[str] = None,
) -> pd.DataFrame:
    """
    Convert all SDFs in a directory to SMILES.
    - docked_dir: folder containing docked SDF files.
    - pattern: glob pattern (default '*.sdf').
    - numeric_sort: if True, sorts files by the numeric part of their names.
    - output_csv: optional path to save a CSV with columns [file, mol_index, name, smiles].
    - output_smi: optional path to save a .smi file ('SMILES name' per line).

    Returns: pandas DataFrame with one row per molecule.
    """
    files = glob.glob(os.path.join(docked_dir, pattern))
    files.sort(key=_numeric_key if numeric_sort else None)

    all_rows: List[Dict[str, str]] = []
    for f in files:
        all_rows.extend(
            sdf_to_smiles_file(
                f,
                canonical=canonical,
                isomeric=isomeric,
                kekule=kekule,
                sanitize=sanitize,
            )
        )

    df = pd.DataFrame(all_rows, columns=["file", "mol_index", "name", "smiles"])
    if output_csv:
        os.makedirs(os.path.dirname(output_csv) or ".", exist_ok=True)
        df.to_csv(output_csv, index=False)
    if output_smi:
        os.makedirs(os.path.dirname(output_smi) or ".", exist_ok=True)
        with open(output_smi, "w", encoding="utf-8") as w:
            for _, row in df.iterrows():
                w.write(f"{row['smiles']} {row['name']}\n")
    return df
