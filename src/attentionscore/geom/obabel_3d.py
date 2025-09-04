# src/attentionscore/geom/obabel_3d.py
from __future__ import annotations

import os
from typing import Iterable, List, Optional

def smiles_to_mol2(
    smiles: Iterable[str],
    dest_dir: str,
    *,
    ids: Optional[Iterable[str]] = None,   # if provided, used as file basenames & titles
    name_prefix: str = "mol_",
    start_index: int = 0,
    forcefield: str = "mmff94s",
    steps: int = 500,
    overwrite: bool = True,
    drop_invalid: bool = True,
) -> List[str]:
    """
    Generate 3D conformers from SMILES with Pybel/Open Babel and write MOL2 files.

    Parameters
    ----------
    smiles        : iterable of SMILES strings
    dest_dir      : directory to write output MOL2 files
    ids           : optional iterable of IDs; if given, each output will be <id>.mol2
    name_prefix   : used when `ids` is None → filenames become f"{name_prefix}{start_index+i}.mol2"
    start_index   : starting index for filenames when `ids` is None
    forcefield    : 'mmff94s' (default) or any Open Babel-supported FF
    steps         : local optimization steps
    overwrite     : whether to overwrite existing files
    drop_invalid  : if False, raise ValueError on invalid SMILES; if True, skip them

    Returns
    -------
    List[str] : list of absolute paths to written MOL2 files
    """
    try:
        from openbabel import pybel
    except Exception as e:
        raise RuntimeError(
            "Open Babel Python bindings (pybel) are required. "
            "Install via your package manager or conda (e.g., `conda install -c conda-forge openbabel`)."
        ) from e

    os.makedirs(dest_dir, exist_ok=True)
    written: List[str] = []

    id_list = list(ids) if ids is not None else None

    for i, smi in enumerate(smiles):
        try:
            mol = pybel.readstring("smi", str(smi))
        except Exception:
            if drop_invalid:
                continue
            raise ValueError(f"Invalid SMILES at index {i}: {smi}")

        # Title / basename
        if id_list is not None:
            title = str(id_list[i])
        else:
            title = f"{name_prefix}{start_index + i}"

        mol.title = title
        # Build 3D and optimize
        mol.make3D(forcefield)
        mol.localopt(forcefield=forcefield, steps=steps)

        out_path = os.path.abspath(os.path.join(dest_dir, f"{title}.mol2"))
        writer = pybel.Outputfile("mol2", out_path, overwrite=overwrite)
        writer.write(mol)
        writer.close()

        written.append(out_path)

    return written
