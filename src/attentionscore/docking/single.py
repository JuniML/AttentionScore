from __future__ import annotations

import os, subprocess
from typing import Dict, Optional, Sequence, Tuple, List

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
    cmd = [
        smina_bin,
        "-r", receptor,
        "-l", ligand,
        "-o", out_path,
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

def run_smina_single(
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
) -> Tuple[str, int, str, str]:
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    cmd = build_smina_cmd(
        receptor=receptor, ligand=ligand, out_path=out_path,
        center=center, size=size, smina_bin=smina_bin,
        exhaustiveness=exhaustiveness, num_modes=num_modes,
        extra_args=extra_args
    )
    try:
        proc = subprocess.run(cmd, check=False, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        return out_path, proc.returncode, proc.stdout, proc.stderr
    except FileNotFoundError:
        raise RuntimeError("smina not found. Install it (e.g. `conda install -c bioconda smina`) "
                           "or pass full path via smina_bin='/path/to/smina'.")
