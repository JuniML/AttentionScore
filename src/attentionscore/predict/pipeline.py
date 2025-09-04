from __future__ import annotations
import os, glob
from typing import List, Optional, Tuple
import numpy as np
import pandas as pd
import torch

# building blocks you already have / we created earlier
from attentionscore.geom.obabel_3d import smiles_to_mol2
from attentionscore.docking.single import run_smina_single   # single-run fallback
from attentionscore.docking.smina import run_smina_dir       # parallel dir runner you used
from attentionscore.features.plec_batch import plec_features_from_dir
# from attentionscore.features.avalon import avalon_fp_array
from attentionscore.features.fingerprints import calculate_avalon_array
from attentionscore.predict.smiles_utils import mol2_dir_to_smiles, sdf_dir_to_smiles
from attentionscore.predict.infer import predict_proba_arrays, predict_classes_arrays
from attentionscore.predict.infer import PairDataset, make_pred_loader  # optional
from attentionscore.predict.infer import _check_dims  # internal check (optional)
from attentionscore.predict.infer import np as _np  # if you exported; else ignore

# Optional results builder (from earlier step); paste in if not present
try:
    from attentionscore.predict.infer import build_and_save_results
except Exception:
    def build_and_save_results(names, smiles, probs, preds=None, *, threshold=0.5,
                               sort_by="probability", descending=True, round_digits=4,
                               out_csv="predictions.csv", out_xlsx=None) -> pd.DataFrame:
        if preds is None:
            preds = (np.asarray(probs) >= threshold).astype(int)
        df = pd.DataFrame({"molecule": names, "smiles": smiles, "probability": np.asarray(probs),
                           "activity": np.asarray(preds, dtype=int)})
        df["probability"] = df["probability"].round(round_digits)
        df["activity_label"] = df["activity"].map({1: "Active", 0: "Inactive"})
        df = df.sort_values(sort_by, ascending=not descending).reset_index(drop=True)
        os.makedirs(os.path.dirname(out_csv) or ".", exist_ok=True)
        df.to_csv(out_csv, index=False)
        if out_xlsx:
            df.to_excel(out_xlsx, index=False)
        return df

# ---------------------------
# 1) From SMILES list
# ---------------------------
def pipeline_from_smiles(
    smiles: List[str],
    out_dir: str,
    receptor_path: str,
    center: dict,
    size: dict,
    model,
    *,
    avalon_bits: int = 512,
    plec_size: int = 4092,
    smina_bin: str = "smina",
    exhaustiveness: int = 8,
    num_modes: int = 1,
    device: torch.device = torch.device("cpu"),
    threshold: float = 0.5,
) -> pd.DataFrame:
    os.makedirs(out_dir, exist_ok=True)
    mol2_dir = os.path.join(out_dir, "mol2")
    docked_dir = os.path.join(out_dir, "docked")
    os.makedirs(mol2_dir, exist_ok=True)
    os.makedirs(docked_dir, exist_ok=True)

    # 1) SMILES -> MOL2 files
    smiles_to_mol2(smiles, dest_dir=mol2_dir, name_prefix="lig_")

    # 2) Dock MOL2 -> SDF (parallel)
    run_smina_dir(
        lig_dir=mol2_dir,
        receptor_path=receptor_path,
        center=center, size=size,
        out_dir=docked_dir,
        pattern="*.mol2",
        smina_bin=smina_bin,
        exhaustiveness=exhaustiveness,
        num_modes=num_modes,
        n_jobs=8,
        backend="multiprocessing",
    )

    # 3) Features
    plec_arr, files = plec_features_from_dir(docked_dir, receptor_path, plec_size=plec_size)
    # Align smiles to file order by numeric sort of created names (lig_0, lig_1, ...)
    # If you used name_prefix="lig_", the sort matches enumeration order:
    smiles_sorted = [smiles[int("".join(filter(str.isdigit, os.path.basename(f))))] for f in files]
    avalon_arr = np.vstack([calculate_avalon_array(smi, nBits=avalon_bits) for smi in smiles_sorted]).astype(np.float32)

    # 4) Predict
    probs = predict_proba_arrays(model, plec_arr, avalon_arr, device=device)
    preds = predict_classes_arrays(model, plec_arr, avalon_arr, threshold=threshold, device=device)

    return build_and_save_results(
        names=[os.path.basename(f) for f in files],
        smiles=smiles_sorted,
        probs=probs,
        preds=preds,
        out_csv=os.path.join(out_dir, "predictions.csv"),
    )

# ---------------------------
# 2) From a MOL2 directory
# ---------------------------
def pipeline_from_mol2(
    mol2_dir: str,
    receptor_path: str,
    center: dict,
    size: dict,
    model,
    *,
    avalon_bits: int = 512,
    plec_size: int = 4092,
    smina_bin: str = "smina",
    exhaustiveness: int = 8,
    num_modes: int = 1,
    device: torch.device = torch.device("cpu"),
    threshold: float = 0.5,
) -> pd.DataFrame:
    docked_dir = os.path.join(mol2_dir, "docked")
    os.makedirs(docked_dir, exist_ok=True)

    # 1) Dock MOL2 -> SDF
    run_smina_dir(
        lig_dir=mol2_dir,
        receptor_path=receptor_path,
        center=center, size=size,
        out_dir=docked_dir,
        pattern="*.mol2",
        smina_bin=smina_bin,
        exhaustiveness=exhaustiveness,
        num_modes=num_modes,
        n_jobs=8,
        backend="multiprocessing",
    )

    # 2) Extract SMILES from the original MOL2s (for Avalon)
    df_mol2 = mol2_dir_to_smiles(mol2_dir, pattern="*.mol2")
    # Match order with docked SDF files
    sdf_files = sorted(glob.glob(os.path.join(docked_dir, "*.sdf")), key=lambda p: (int("".join(filter(str.isdigit, os.path.basename(p))) or 1e9), os.path.basename(p)))
    # Map file stem -> smiles from mol2 DF (assumes same base names)
    base_to_smi = {os.path.splitext(os.path.basename(p))[0]: smi for p, smi in zip(df_mol2["file"], df_mol2["smiles"])}
    smiles_ordered = [ base_to_smi.get(os.path.splitext(os.path.basename(sdf))[0], None) for sdf in sdf_files ]

    # 3) Features
    plec_arr, _ = plec_features_from_dir(docked_dir, receptor_pdb=receptor_path, plec_size=plec_size)
    avalon_arr = np.vstack([avalon_fp_array(s, nBits=avalon_bits) if isinstance(s, str) else np.zeros((avalon_bits,), dtype=int)
                            for s in smiles_ordered]).astype(np.float32)

    # 4) Predict
    probs = predict_proba_arrays(model, plec_arr, avalon_arr, device=device)
    preds = predict_classes_arrays(model, plec_arr, avalon_arr, threshold=threshold, device=device)

    return build_and_save_results(
        names=[os.path.basename(f) for f in sdf_files],
        smiles=smiles_ordered,
        probs=probs,
        preds=preds,
        out_csv=os.path.join(docked_dir, "predictions.csv"),
    )

# ---------------------------
# 3) From a docked SDF directory
# ---------------------------
def pipeline_from_docked(
    docked_dir: str,
    receptor_path: str,
    model,
    *,
    avalon_bits: int = 512,
    plec_size: int = 4092,
    device: torch.device = torch.device("cpu"),
    threshold: float = 0.5,
) -> pd.DataFrame:
    # 1) Extract SMILES from SDFs (for Avalon)
    df_sdf = sdf_dir_to_smiles(docked_dir, pattern="*.sdf")
    # We assume 1 mol per sdf; if multiple, we’ll still work but file order duplicates
    sdf_files = sorted(glob.glob(os.path.join(docked_dir, "*.sdf")),
                       key=lambda p: (int("".join(filter(str.isdigit, os.path.basename(p))) or 1e9), os.path.basename(p)))
    # Build smiles list that matches file order; take first occurrence per file
    first_by_file = (df_sdf.sort_values(["file", "mol_index"])
                          .drop_duplicates(subset=["file"], keep="first"))
    file_to_smi = dict(zip(first_by_file["file"], first_by_file["smiles"]))
    smiles_ordered = [file_to_smi.get(f) for f in sdf_files]

    # 2) PLEC features
    plec_arr, files = plec_features_from_dir(docked_dir, receptor_pdb=receptor_path, plec_size=plec_size)

    # 3) Avalon features (some SMILES may be None; fill zeros)
    avalon_arr = np.vstack([avalon_fp_array(s, nBits=avalon_bits) if isinstance(s, str) else np.zeros((avalon_bits,), dtype=int)
                            for s in smiles_ordered]).astype(np.float32)

    # 4) Predict
    probs = predict_proba_arrays(model, plec_arr, avalon_arr, device=device)
    preds = predict_classes_arrays(model, plec_arr, avalon_arr, threshold=threshold, device=device)

    return build_and_save_results(
        names=[os.path.basename(f) for f in files],
        smiles=smiles_ordered,
        probs=probs,
        preds=preds,
        out_csv=os.path.join(docked_dir, "predictions.csv"),
    )
