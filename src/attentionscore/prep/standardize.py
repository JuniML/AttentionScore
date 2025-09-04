from __future__ import annotations
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit.Chem.MolStandardize import rdMolStandardize
from rdkit.rdBase import BlockLogs

def calculate_ro5_properties(smiles: str, fullfill: int = 4) -> bool:
    if not isinstance(smiles, str) or not smiles.strip():
        return False
    m = Chem.MolFromSmiles(smiles)
    if m is None:
        return False
    mw = Descriptors.MolWt(m)
    from rdkit.Chem import Crippen, rdMolDescriptors
    logp = Crippen.MolLogP(m)
    hbd = rdMolDescriptors.CalcNumHBD(m)
    hba = rdMolDescriptors.CalcNumHBA(m)
    satisfied = sum([mw <= 500, logp <= 5, hbd <= 5, hba <= 10])
    return satisfied >= fullfill

class standardization:
    def __init__(self, data: pd.DataFrame, ID: str, smiles_col: str, active_col: str, ro5: int = 4):
        self.data = data.copy()
        self.ID = ID
        self.smiles_col = smiles_col
        self.active_col = active_col
        self.ro5 = ro5

    def standardize(self, smiles: str):
        mol = Chem.MolFromSmiles(smiles)
        clean_mol = rdMolStandardize.Cleanup(mol) 
        parent_clean_mol = rdMolStandardize.FragmentParent(clean_mol)
        uncharger = rdMolStandardize.Uncharger()
        uncharged_parent_clean_mol = uncharger.uncharge(parent_clean_mol)
        te = rdMolStandardize.TautomerEnumerator()
        taut_uncharged_parent_clean_mol = te.Canonicalize(uncharged_parent_clean_mol)
        return taut_uncharged_parent_clean_mol
    
    def filter_data(self) -> pd.DataFrame:
        self.data['Canonicalsmiles'] = self.data[self.smiles_col].apply(Chem.CanonSmiles)
        self.data = self.data[self.data['Canonicalsmiles'].apply(calculate_ro5_properties, fullfill=self.ro5)]
        block = BlockLogs()
        self.data['Molecule'] = self.data['Canonicalsmiles'].apply(self.standardize)
        self.data["Standardize_smile"] = self.data["Molecule"].apply(Chem.MolToSmiles)
        return self.data

def standardize_dataframe(df: pd.DataFrame, id_col="ID", smiles_col="SMILES", active_col="pIC50", ro5: int = 4) -> pd.DataFrame:
    std = standardization(df, ID=id_col, smiles_col=smiles_col, active_col=active_col, ro5=ro5)
    return std.filter_data()