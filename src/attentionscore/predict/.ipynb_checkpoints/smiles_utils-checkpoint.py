from __future__ import annotations
import os, glob
from typing import List, Dict, Optional
import pandas as pd
from rdkit import Chem

__all__ = ["mol2_dir_to_smiles", "sdf_dir_to_smiles"]

def _numeric_key(path: str):
    b = os.path.basename(path)
    digits = "".join(ch for ch in b if ch.isdigit())
    return (int(digits) if digits else float("inf"), b)

def mol2_dir_to_smiles(
    mol2_dir: str,
    pattern: str = "*.mol2",
    *,
    sanitize: bool = True,
    numeric_sort: bool = True,
    output_csv: Optional[str] = None,
    output_smi: Optional[str] = None,
) -> pd.DataFrame:
    """
    Read all MOL2s in a directory and return DataFrame: [file, name, smiles].
    """
    files = glob.glob(os.path.join(mol2_dir, pattern))
    files.sort(key=_numeric_key if numeric_sort else None)

    rows: List[Dict[str, str]] = []
    for f in files:
        # RDKit MOL2 reader (works for most docking exports)
        mol = Chem.MolFromMol2File(f, sanitize=sanitize, removeHs=False)
        if mol is None:
            # skip but keep a row with NA for traceability
            rows.append({"file": f, "name": os.path.basename(f), "smiles": None})
            continue
        name = mol.GetProp("_Name") if mol.HasProp("_Name") else os.path.basename(f)
        smi = Chem.MolToSmiles(mol, isomericSmiles=True, canonical=True)
        rows.append({"file": f, "name": name, "smiles": smi})

    df = pd.DataFrame(rows, columns=["file", "name", "smiles"])
    if output_csv:
        os.makedirs(os.path.dirname(output_csv) or ".", exist_ok=True)
        df.to_csv(output_csv, index=False)
    if output_smi:
        os.makedirs(os.path.dirname(output_smi) or ".", exist_ok=True)
        with open(output_smi, "w", encoding="utf-8") as w:
            for _, r in df.dropna(subset=["smiles"]).iterrows():
                w.write(f"{r['smiles']} {r['name']}\n")
    return df

# Re-export your existing SDF->SMILES helper if you already have it:
try:
    from attentionscore.predict.sdf_utils import sdf_dir_to_smiles  # already created earlier
except Exception:
    # Minimal fallback
    def sdf_dir_to_smiles(
        docked_dir: str,
        pattern: str = "*.sdf",
        *,
        sanitize: bool = True,
        numeric_sort: bool = True,
        output_csv: Optional[str] = None,
        output_smi: Optional[str] = None,
    ) -> pd.DataFrame:
        files = glob.glob(os.path.join(docked_dir, pattern))
        files.sort(key=_numeric_key if numeric_sort else None)
        rows: List[Dict[str, str]] = []
        for f in files:
            suppl = Chem.SDMolSupplier(f, removeHs=False, sanitize=sanitize)
            for i, mol in enumerate(suppl):
                if mol is None:
                    continue
                name = mol.GetProp("_Name") if mol.HasProp("_Name") else f"{os.path.basename(f)}#{i}"
                smi = Chem.MolToSmiles(mol, isomericSmiles=True, canonical=True)
                rows.append({"file": f, "mol_index": i, "name": name, "smiles": smi})
        df = pd.DataFrame(rows, columns=["file", "mol_index", "name", "smiles"])
        if output_csv:
            os.makedirs(os.path.dirname(output_csv) or ".", exist_ok=True)
            df.to_csv(output_csv, index=False)
        if output_smi:
            os.makedirs(os.path.dirname(output_smi) or ".", exist_ok=True)
            with open(output_smi, "w", encoding="utf-8") as w:
                for _, r in df.iterrows():
                    w.write(f"{r['smiles']} {r['name']}\n")
        return df
