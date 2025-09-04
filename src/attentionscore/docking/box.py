from __future__ import annotations

from typing import Dict, Tuple

def vina_box_from_ligand(ligand_path: str, extending: float = 6.0) -> Tuple[Dict[str, float], Dict[str, float]]:
    """
    Compute a Vina-style box around a ligand using PyMOL if available,
    otherwise fall back to RDKit (PDB/SDF) or Open Babel (MOL2).
    """
    try:
        from pymol import cmd
        cmd.reinitialize()
        ext = ligand_path.split(".")[-1].lower()
        fmt = "pdb" if ext not in ("sdf","mol2","pdb","pdbqt") else ext
        obj = "lig"
        cmd.load(filename=ligand_path, format=fmt, object=obj)
        (min_xyz, max_xyz) = cmd.get_extent(obj)
        cmd.delete("all")
        minX, minY, minZ = [c - float(extending) for c in min_xyz]
        maxX, maxY, maxZ = [c + float(extending) for c in max_xyz]
    except Exception:
        ext = ligand_path.split(".")[-1].lower()
        if ext in ("pdb", "sdf"):
            from rdkit import Chem
            import numpy as np
            if ext == "pdb":
                mol = Chem.MolFromPDBFile(ligand_path, removeHs=False)
            else:
                suppl = Chem.SDMolSupplier(ligand_path, removeHs=False)
                mol = suppl[0] if len(suppl) else None
            if mol is None or not mol.GetNumConformers():
                raise RuntimeError("Could not read 3D coords from ligand for box computation.")
            conf = mol.GetConformer(0)
            coords = np.array([list(conf.GetAtomPosition(i)) for i in range(mol.GetNumAtoms())], dtype=float)
            minX, minY, minZ = coords.min(axis=0) - extending
            maxX, maxY, maxZ = coords.max(axis=0) + extending
        elif ext == "mol2":
            import numpy as np
            from openbabel import pybel
            mol = next(pybel.readfile("mol2", ligand_path))
            coords = np.array([a.coords for a in mol.atoms], dtype=float)
            minX, minY, minZ = coords.min(axis=0) - extending
            maxX, maxY, maxZ = coords.max(axis=0) + extending
        else:
            raise RuntimeError("Unsupported ligand format. Use PDB/SDF/MOL2 or install PyMOL for broader support.")

    SizeX, SizeY, SizeZ = maxX - minX, maxY - minY, maxZ - minZ
    CenterX, CenterY, CenterZ = (maxX + minX)/2, (maxY + minY)/2, (maxZ + minZ)/2
    return {"center_x": CenterX, "center_y": CenterY, "center_z": CenterZ}, {"size_x": SizeX, "size_y": SizeY, "size_z": SizeZ}
