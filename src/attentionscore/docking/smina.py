# src/attentionscore/docking/smina.py
from __future__ import annotations

import os, glob, re, subprocess
from typing import Dict, List, Tuple, Optional, Sequence
from joblib import Parallel, delayed
from tqdm import tqdm

_NUM = re.compile(r"\d+")


def numeric_sort_key(path: str) -> tuple:
    """Numeric-aware sort key so lig2.mol2 comes before lig10.mol2."""
    nums = _NUM.findall(os.path.basename(path))
    return tuple(int(n) for n in nums) if nums else (float("inf"),)


def build_smina_cmd(
    receptor: str,
    ligand: str,
    out_path: str,
    center: Dict[str, float],
    size: Dict[str, float],
    *,
    smina_bin: str = "smina",
    exhaustiveness: int = 8,
    num_modes: int = 1,
    extra_args: Optional[Sequence[str]] = None,
) -> List[str]:
    """
    Build the smina CLI. Receptor may be PDBQT or PDB (smina supports both on many builds).
    """
    cmd = [
        smina_bin,
        "-r", receptor,           # receptor (PDBQT or PDB)
        "-l", ligand,             # ligand (SDF/MOL2/PDBQT/...)
        "-o", out_path,           # output SDF
        "--center_x", str(center["center_x"]),
        "--center_y", str(center["center_y"]),
        "--center_z", str(center["center_z"]),
        "--size_x", str(size["size_x"]),
        "--size_y", str(size["size_y"]),
        "--size_z", str(size["size_z"]),
        "--exhaustiveness", str(exhaustiveness),
        "--num_modes", str(num_modes),
    ]
    if extra_args:
        cmd.extend(list(extra_args))
    return cmd


def run_smina(
    input_file: str,
    receptor_path: str,
    center: Dict[str, float],
    size: Dict[str, float],
    *,
    out_path: Optional[str] = None,
    smina_bin: str = "smina",
    exhaustiveness: int = 8,
    num_modes: int = 1,
    extra_args: Optional[Sequence[str]] = None,
) -> Tuple[str, int]:
    """
    Dock a single ligand with smina. Returns (out_path, return_code).

    Parameters
    ----------
    input_file   : ligand file path (e.g., .mol2 / .sdf / .pdbqt)
    receptor_path: receptor path (PDBQT or PDB)
    center/size  : dicts with center_{x,y,z} / size_{x,y,z}
    out_path     : where to write <SDF> (defaults next to input with _docked.sdf)
    smina_bin    : path to smina binary or 'smina' if on PATH
    extra_args   : sequence of extra smina flags, e.g. ["--seed", "42"]
    """
    if out_path is None:
        base = os.path.splitext(os.path.basename(input_file))[0]
        out_path = os.path.join(os.path.dirname(input_file), f"{base}_docked.sdf")

    cmd = build_smina_cmd(
        receptor=receptor_path,
        ligand=input_file,
        out_path=out_path,
        center=center,
        size=size,
        smina_bin=smina_bin,
        exhaustiveness=exhaustiveness,
        num_modes=num_modes,
        extra_args=extra_args,
    )
    try:
        subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        return out_path, 0
    except FileNotFoundError:
        raise RuntimeError(
            "smina not found. Pass the full path via smina_bin='/full/path/to/smina' "
            "or `conda install -c bioconda smina`."
        )
    except subprocess.CalledProcessError as e:
        # Return non-zero code so the caller can inspect failures per ligand
        return out_path, e.returncode


# --- MODULE-LEVEL helper (picklable) for multiprocessing ---
def _dock_single_file(
    f: str,
    receptor_path: str,
    center: Dict[str, float],
    size: Dict[str, float],
    out_dir: str,
    smina_bin: str,
    exhaustiveness: int,
    num_modes: int,
    extra_args: Optional[Sequence[str]],
) -> Tuple[str, int]:
    base = os.path.splitext(os.path.basename(f))[0]
    out_path = os.path.join(out_dir, f"{base}_docked.sdf")
    return run_smina(
        input_file=f,
        receptor_path=receptor_path,
        center=center,
        size=size,
        out_path=out_path,
        smina_bin=smina_bin,
        exhaustiveness=exhaustiveness,
        num_modes=num_modes,
        extra_args=extra_args,
    )


def run_smina_dir(
    lig_dir: str,
    receptor_path: str,
    center: Dict[str, float],
    size: Dict[str, float],
    *,
    out_dir: Optional[str] = None,
    pattern: str = "*.mol2",
    smina_bin: str = "smina",
    exhaustiveness: int = 8,
    num_modes: int = 1,
    n_jobs: int = 20,
    backend: str = "multiprocessing",
    numeric_sort: bool = True,
    extra_args: Optional[Sequence[str]] = None,
) -> List[Tuple[str, int]]:
    """
    Dock all ligands in a directory matching a glob pattern (parallel).

    Returns a list of (out_path, return_code) tuples in the same order as `files`.
    """
    files = glob.glob(os.path.join(lig_dir, pattern))
    if not files:
        raise FileNotFoundError(f"No files matching {pattern} in {lig_dir}")

    if numeric_sort:
        files.sort(key=numeric_sort_key)
    else:
        files.sort()

    if out_dir is None:
        out_dir = os.path.join(lig_dir, "docked")
    os.makedirs(out_dir, exist_ok=True)

    results = Parallel(n_jobs=n_jobs, backend=backend)(
        delayed(_dock_single_file)(
            f, receptor_path, center, size, out_dir, smina_bin, exhaustiveness, num_modes, extra_args
        )
        for f in tqdm(files, desc="smina")
    )
    return results
