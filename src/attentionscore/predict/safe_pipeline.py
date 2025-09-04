from __future__ import annotations
import os, glob
from typing import List, Tuple, Optional
import numpy as np
import pandas as pd
import torch

# --- thread & env safety ------------------------------------------------------
def set_low_thread_env(max_threads: int = 1) -> None:
    """Cap math libs threads to avoid kernel crashes in notebooks."""
    for k in ["OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
              "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"]:
        os.environ[k] = str(max_threads)
    # RDKit sometimes respects this hint:
    os.environ.setdefault("RDKIT_THREADCOUNT", str(max_threads))

# --- imports from your package ------------------------------------------------
from attentionscore.geom.obabel_3d import smiles_to_mol2
from attentionscore.docking.single import run_smina_single
from attentionscore.docking.box import vina_box_from_ligand
from attentionscore.predict.sdf_utils import sdf_dir_to_smiles
from attentionscore.predict.smiles_utils import mol2_dir_to_smiles
from attentionscore.features.avalon import avalon_fp_array
from attentionscore.features.plec_batch import plec_features_from_dir
from attentionscore.predict.infer import predict_proba_arrays, predict_classes_arrays

# small results helper (use your existing one if already added)
def build_and_save_results(
    names: List[str],
    smiles: List[Optional[str]],
    probs: np.ndarray,
    preds: Optional[np.ndarray] = None,
    *,
    threshold: float = 0.5,
    out_csv: str = "predictions.csv",
    out_xlsx: Optional[str] = None,
) -> pd.DataFrame:
    probs = np.asarray(probs).reshape(-1)
    if preds is None:
        preds = (probs >= float(threshold)).astype(int)
    df = pd.DataFrame({
        "molecule": names,
        "smiles": smiles,
        "probability": probs.round(4),
        "activity": preds.astype(int),
    })
    df["activity_label"] = df["activity"].map({1: "Active", 0: "Inactive"})
    os.makedirs(os.path.dirname(out_csv) or ".", exist_ok=True)
    df.to_csv(out_csv, index=False)
    if out_xlsx:
        df.to_excel(out_xlsx, index=False)
    return df

# ------------------------------------------------------------------------------
# 1) SMILES -> MOL2 -> Dock (sequential) -> Features -> Predict
# ------------------------------------------------------------------------------
def pipeline_from_smiles_safe(
    smiles: List[str],
    out_dir: str,
    receptor_path: str,
    *,
    smina_bin: str = "smina",
    box_ligand: Optional[str] = None,   # if provided, use this file to define box
    box_extend: float = 6.0,
    plec_size: int = 4092,
    avalon_bits: int = 512,
    model=None,
    device: torch.device = torch.device("cpu"),
    threshold: float = 0.5,
) -> pd.DataFrame:
    set_low_thread_env(1)
    os.makedirs(out_dir, exist_ok=True)
    mol2_dir = os.path.join(out_dir, "mol2"); os.makedirs(mol2_dir, exist_ok=True)
    dock_dir = os.path.join(out_dir, "docked"); os.makedirs(dock_dir, exist_ok=True)

    # Box: from provided ref ligand or from the first SMILES we’ll generate
    if box_ligand is not None:
        center, size = vina_box_from_ligand(box_ligand, extending=box_extend)
    else:
        seed_mol2 = smiles_to_mol2([smiles[0]], dest_dir=mol2_dir, name_prefix="seed_")[0]
        center, size = vina_box_from_ligand(seed_mol2, extending=box_extend)

    # 1) SMILES -> MOL2 (named lig_0, lig_1, ...)
    lig_paths = smiles_to_mol2(smiles, dest_dir=mol2_dir, name_prefix="lig_")

    # 2) Dock sequentially (avoid multiprocessing in notebooks)
    docked_files = []
    for p in lig_paths:
        base = os.path.splitext(os.path.basename(p))[0]
        out_sdf = os.path.join(dock_dir, f"{base}_docked.sdf")
        _, code, _, stderr = run_smina_single(
            receptor=receptor_path, ligand=p, out_path=out_sdf,
            center=center, size=size,
            smina_bin=smina_bin, exhaustiveness=8, num_modes=1,
        )
        if code != 0:
            raise RuntimeError(f"smina failed for {p} (code {code}):\n{stderr}")
        docked_files.append(out_sdf)

    # 3) Features: PLEC from SDFs, Avalon from original SMILES (aligned order)
    plec_arr, files = plec_features_from_dir(dock_dir, receptor_pdb=receptor_path, plec_size=plec_size)
    # align smiles by extracting the numeric id from 'lig_N'
    order = [int("".join(filter(str.isdigit, os.path.basename(f)))) for f in files]
    smiles_sorted = [smiles[i] for i in order]
    avalon_arr = np.vstack([avalon_fp_array(s, nBits=avalon_bits) for s in smiles_sorted]).astype(np.float32)

    # 4) Predict
    probs = predict_proba_arrays(model, plec_arr, avalon_arr, batch_size=128, device=device)
    preds = predict_classes_arrays(model, plec_arr, avalon_arr, threshold=threshold, batch_size=128, device=device)

    return build_and_save_results(
        names=[os.path.basename(f) for f in files],
        smiles=smiles_sorted,
        probs=probs,
        preds=preds,
        out_csv=os.path.join(out_dir, "predictions.csv"),
    )

# ------------------------------------------------------------------------------
# 2) MOL2 dir -> Dock (sequential) -> Features -> Predict (auto SMILES)
# ------------------------------------------------------------------------------
def pipeline_from_mol2_safe(
    mol2_dir: str,
    receptor_path: str,
    *,
    smina_bin: str = "smina",
    center: Optional[dict] = None,
    size: Optional[dict] = None,
    box_extend: float = 6.0,
    ref_ligand_for_box: Optional[str] = None,
    plec_size: int = 4092,
    avalon_bits: int = 512,
    model=None,
    device: torch.device = torch.device("cpu"),
    threshold: float = 0.5,
) -> pd.DataFrame:
    set_low_thread_env(1)
    dock_dir = os.path.join(mol2_dir, "docked"); os.makedirs(dock_dir, exist_ok=True)

    # Box: if not provided, make from first MOL2 or a provided ref ligand
    if center is None or size is None:
        if ref_ligand_for_box:
            center, size = vina_box_from_ligand(ref_ligand_for_box, extending=box_extend)
        else:
            first = sorted(glob.glob(os.path.join(mol2_dir, "*.mol2")))[0]
            center, size = vina_box_from_ligand(first, extending=box_extend)

    # Dock sequentially
    mol2_files = sorted(glob.glob(os.path.join(mol2_dir, "*.mol2")),
                        key=lambda p: (int("".join(filter(str.isdigit, os.path.basename(p))) or 1e9), os.path.basename(p)))
    docked_files = []
    for lig in mol2_files:
        base = os.path.splitext(os.path.basename(lig))[0]
        out_sdf = os.path.join(dock_dir, f"{base}_docked.sdf")
        _, code, _, stderr = run_smina_single(
            receptor=receptor_path, ligand=lig, out_path=out_sdf,
            center=center, size=size,
            smina_bin=smina_bin, exhaustiveness=8, num_modes=1,
        )
        if code != 0:
            raise RuntimeError(f"smina failed for {lig} (code {code}):\n{stderr}")
        docked_files.append(out_sdf)

    # SMILES from MOL2s for Avalon
    df_mol2 = mol2_dir_to_smiles(mol2_dir)
    base2smi = {os.path.splitext(os.path.basename(p))[0]: smi for p, smi in zip(df_mol2["file"], df_mol2["smiles"])}

    # Features
    plec_arr, files = plec_features_from_dir(dock_dir, receptor_pdb=receptor_path, plec_size=plec_size)
    smiles_ordered = [base2smi.get(os.path.splitext(os.path.basename(f))[0].replace("_docked", ""), None) for f in files]
    avalon_arr = np.vstack([avalon_fp_array(s, nBits=avalon_bits) if isinstance(s, str) else np.zeros((avalon_bits,), dtype=int)
                            for s in smiles_ordered]).astype(np.float32)

    # Predict
    probs = predict_proba_arrays(model, plec_arr, avalon_arr, batch_size=128, device=device)
    preds = predict_classes_arrays(model, plec_arr, avalon_arr, threshold=threshold, batch_size=128, device=device)

    return build_and_save_results(
        names=[os.path.basename(f) for f in files],
        smiles=smiles_ordered,
        probs=probs,
        preds=preds,
        out_csv=os.path.join(dock_dir, "predictions.csv"),
    )

# ------------------------------------------------------------------------------
# 3) Docked SDF dir -> Features -> Predict (auto SMILES from SDF)
# ------------------------------------------------------------------------------
def pipeline_from_docked_safe(
    docked_dir: str,
    receptor_path: str,
    *,
    plec_size: int = 4092,
    avalon_bits: int = 512,
    model=None,
    device: torch.device = torch.device("cpu"),
    threshold: float = 0.5,
) -> pd.DataFrame:
    set_low_thread_env(1)
    # SMILES from SDFs (first mol per file)
    df_sdf = sdf_dir_to_smiles(docked_dir, pattern="*.sdf")
    first = (df_sdf.sort_values(["file", "mol_index"]).drop_duplicates(subset=["file"], keep="first"))
    file2smi = dict(zip(first["file"], first["smiles"]))

    # PLEC
    plec_arr, files = plec_features_from_dir(docked_dir, receptor_pdb=receptor_path, plec_size=plec_size)
    smiles_ordered = [file2smi.get(f) for f in files]

    # Avalon
    avalon_arr = np.vstack([avalon_fp_array(s, nBits=avalon_bits) if isinstance(s, str) else np.zeros((avalon_bits,), dtype=int)
                            for s in smiles_ordered]).astype(np.float32)

    # Predict
    probs = predict_proba_arrays(model, plec_arr, avalon_arr, batch_size=128, device=device)
    preds = predict_classes_arrays(model, plec_arr, avalon_arr, threshold=threshold, batch_size=128, device=device)

    return build_and_save_results(
        names=[os.path.basename(f) for f in files],
        smiles=smiles_ordered,
        probs=probs,
        preds=preds,
        out_csv=os.path.join(docked_dir, "predictions.csv"),
    )
