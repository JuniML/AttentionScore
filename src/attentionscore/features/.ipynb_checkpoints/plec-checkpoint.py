from __future__ import annotations
import os, glob, re, base64
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from tqdm import tqdm
from typing import Literal






def _plec_for_file(lig_sdf: str, protein, size=4092, depth_protein=4, depth_ligand=2, distance_cutoff=4.5, sparse=False):
    import oddt
    from oddt.fingerprints import PLEC
    ligand = next(oddt.toolkit.readfile('sdf', lig_sdf))
    feature = PLEC(ligand, protein=protein, size=size, 
                   depth_protein=depth_protein, depth_ligand=depth_ligand,
                   distance_cutoff=distance_cutoff, sparse=sparse)
    arr = np.packbits(feature.astype(np.bool_))
    return arr


_num = re.compile(r"\d+")

def _numeric_key(path: str) -> tuple[int, ...]:
    """Sort key: extract all digit groups so 'lig2.sdf' < 'lig10.sdf'."""
    nums = _num.findall(os.path.basename(path))
    return tuple(int(x) for x in nums) if nums else (float("inf"),)

def _plec_for_file(
    lig_sdf: str,
    protein,
    size: int,
    depth_protein: int,
    depth_ligand: int,
    distance_cutoff: float,
    sparse: bool,
):
    import oddt
    from oddt.fingerprints import PLEC

    ligand = next(oddt.toolkit.readfile("sdf", lig_sdf))
    fp = PLEC(
        ligand,
        protein=protein,
        size=size,
        depth_protein=depth_protein,
        depth_ligand=depth_ligand,
        distance_cutoff=distance_cutoff,
        sparse=sparse,
    )
    # Ensure boolean-ish array, then return as numpy 0/1 (uint8)
    arr = np.asarray(fp, dtype=bool).astype(np.uint8)
    return arr

def plec_from_dir(
    docked_dir: str,
    protein_path: str,
    pattern: str = "*.sdf",
    *,
    n_jobs: int = 20,
    size: int = 4092,
    depth_protein: int = 4,
    depth_ligand: int = 2,
    distance_cutoff: float = 4.5,
    sparse: bool = False,
    sort_numeric: bool = True,
    backend: Literal["multiprocessing", "loky", "threading"] = "multiprocessing",
    output: Literal["numpy", "bytes", "base64"] = "numpy",
) -> pd.DataFrame:
    """
    Compute PLEC for all SDFs in a directory against a protein file.

    Returns a DataFrame with:
      id, plec_len, and one of:
        - plec_np  (numpy 0/1 array), if output='numpy'
        - plec_bytes (packed bytes), if output='bytes'
        - plec_b64  (base64 str of packed bytes), if output='base64'
    """
    import oddt

    # Load protein (infer format from extension)
    ext = os.path.splitext(protein_path)[1][1:]
    if not ext:
        raise ValueError(f"Could not infer protein format from: {protein_path}")
    protein = next(oddt.toolkit.readfile(ext, protein_path))
    protein.protein = True

    # Gather SDFs
    sdf_files = glob.glob(os.path.join(docked_dir, pattern))
    if not sdf_files:
        raise FileNotFoundError(f"No SDF files matching '{pattern}' in {docked_dir}")

    if sort_numeric:
        sdf_files.sort(key=_numeric_key)
    else:
        sdf_files.sort()

    # Parallel compute
    feats = Parallel(n_jobs=n_jobs, backend=backend)(
        delayed(_plec_for_file)(
            lig_sdf=f,
            protein=protein,
            size=size,
            depth_protein=depth_protein,
            depth_ligand=depth_ligand,
            distance_cutoff=distance_cutoff,
            sparse=sparse,
        )
        for f in sdf_files
    )

    # Build rows
    rows = []
    for f, arr in zip(sdf_files, feats):
        lig_id = os.path.splitext(os.path.basename(f))[0]
        base = {"id": lig_id, "plec_len": size}
        if output == "numpy":
            rows.append({**base, "plec_np": arr})
        else:
            packed = np.packbits(arr).tobytes()
            if output == "bytes":
                rows.append({**base, "plec_bytes": packed})
            else:
                rows.append({**base, "plec_b64": base64.b64encode(packed).decode("ascii")})
    return pd.DataFrame(rows)

# Optional: Parquet-saving wrapper (handy for pipelines/CLIs)
def plec_to_parquet(
    docked_dir: str,
    protein_path: str,
    out_path: str,
    **kwargs,
) -> pd.DataFrame:
    """
    Convenience wrapper that computes PLEC (default output='base64') and saves to Parquet.
    """
    import pyarrow as pa, pyarrow.parquet as pq
    if "output" not in kwargs:
        kwargs["output"] = "base64"
    df = plec_from_dir(docked_dir, protein_path, **kwargs)
    table = pa.Table.from_pandas(df)
    pq.write_table(table, out_path)
    return df
