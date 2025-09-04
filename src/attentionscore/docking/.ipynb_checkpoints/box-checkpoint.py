# src/attentionscore/docking/box.py
from __future__ import annotations

from typing import Dict, Tuple

def getbox(selection: str = "sele", extending: float = 6.0, software: str = "vina"):
    """
    Compute a docking box from a loaded PyMOL selection using cmd.get_extent.

    Returns
    -------
    If software == 'vina':
      (center: {center_x,center_y,center_z}, size: {size_x,size_y,size_z})
    If software == 'ledock':
      ({minX,maxX}, {minY,maxY}, {minZ,maxZ})
    If software == 'both':
      ((vina_center, vina_size), (ledock_x, ledock_y, ledock_z))
    """
    from pymol import cmd

    (min_xyz, max_xyz) = cmd.get_extent(selection)
    minX, minY, minZ = min_xyz
    maxX, maxY, maxZ = max_xyz

    minX -= float(extending)
    minY -= float(extending)
    minZ -= float(extending)
    maxX += float(extending)
    maxY += float(extending)
    maxZ += float(extending)

    SizeX = maxX - minX
    SizeY = maxY - minY
    SizeZ = maxZ - minZ
    CenterX = (maxX + minX) / 2
    CenterY = (maxY + minY) / 2
    CenterZ = (maxZ + minZ) / 2

    if software == "vina":
        return (
            {"center_x": CenterX, "center_y": CenterY, "center_z": CenterZ},
            {"size_x": SizeX, "size_y": SizeY, "size_z": SizeZ},
        )
    elif software == "ledock":
        return ({"minX": minX, "maxX": maxX}, {"minY": minY, "maxY": maxY}, {"minZ": minZ, "maxZ": maxZ})
    elif software == "both":
        vina = (
            {"center_x": CenterX, "center_y": CenterY, "center_z": CenterZ},
            {"size_x": SizeX, "size_y": SizeY, "size_z": SizeZ},
        )
        ledock = (
            {"minX": minX, "maxX": maxX},
            {"minY": minY, "maxY": maxY},
            {"minZ": minZ, "maxZ": maxZ},
        )
        return vina, ledock
    else:
        raise ValueError('software must be "vina", "ledock" or "both"')

def compute_vina_box_from_files(
    receptor_file: str | None,
    ligand_file: str,
    extending: float = 6.0,
    clear: bool = True,
):
    """
    Load receptor/ligand into PyMOL, compute a Vina-style box around the ligand selection 'lig'.

    Parameters
    ----------
    receptor_file : optional receptor (for visual reference only)
    ligand_file   : path to ligand (PDB/SDF/MOL2/PDBQT)
    extending     : Å padding added to the ligand extent
    clear         : delete all PyMOL objects at the end (recommended in notebooks)

    Returns
    -------
    (center_dict, size_dict)
    """
    from pymol import cmd

    cmd.reinitialize()
    # infer formats from extension
    def _fmt(path: str):
        ext = path.split(".")[-1].lower()
        return "pdb" if ext not in ("pdb", "sdf", "mol2", "pdbqt") else ext

    if receptor_file:
        cmd.load(filename=receptor_file, format=_fmt(receptor_file), object="prot")
    cmd.load(filename=ligand_file, format=_fmt(ligand_file), object="lig")

    center, size = getbox(selection="lig", extending=extending, software="vina")

    if clear:
        cmd.delete("all")

    return center, size
