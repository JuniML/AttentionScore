from __future__ import annotations

from typing import Optional
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


__all__ = [
    "PairDataset",
    "make_pred_loader",
    "predict_proba_arrays",
    "predict_classes_arrays",
    "predict_proba_loader",
]


# -----------------------------
# Dataset & Loader
# -----------------------------
class PairDataset(Dataset):
    """
    Minimal dataset for prediction: pairs of feature rows (XA=PLEC, XB=Avalon).
    """
    def __init__(self, XA_np: np.ndarray, XB_np: np.ndarray, dtype=np.float32):
        XA_np = np.asarray(XA_np, dtype=dtype)
        XB_np = np.asarray(XB_np, dtype=dtype)
        if XA_np.ndim == 1:
            XA_np = XA_np.reshape(1, -1)
        if XB_np.ndim == 1:
            XB_np = XB_np.reshape(1, -1)
        if XA_np.shape[0] != XB_np.shape[0]:
            raise ValueError(f"Row mismatch: XA shape={XA_np.shape}, XB shape={XB_np.shape}")
        self.XA = XA_np
        self.XB = XB_np

    def __len__(self) -> int:
        return self.XA.shape[0]

    def __getitem__(self, idx: int):
        # return tensors so the DataLoader doesn’t need to convert every batch
        return (
            torch.from_numpy(self.XA[idx]).float(),
            torch.from_numpy(self.XB[idx]).float(),
        )


def make_pred_loader(
    XA_np: np.ndarray,
    XB_np: np.ndarray,
    batch_size: int = 256,
    shuffle: bool = False,
    seed: int = 42,
) -> DataLoader:
    ds = PairDataset(XA_np, XB_np)
    if shuffle:
        g = torch.Generator()
        g.manual_seed(seed)
        return DataLoader(ds, batch_size=batch_size, shuffle=True, generator=g)
    return DataLoader(ds, batch_size=batch_size, shuffle=False)


# -----------------------------
# Prediction (no labels)
# -----------------------------
def _check_dims(model, XA: np.ndarray, XB: np.ndarray) -> None:
    if XA.shape[0] != XB.shape[0]:
        raise ValueError(f"Row mismatch: XA={XA.shape}, XB={XB.shape}")
    # sanity-check against model attributes if present
    if hasattr(model, "input_dim_A") and XA.shape[1] != getattr(model, "input_dim_A"):
        raise ValueError(f"XA has {XA.shape[1]} features, but model.input_dim_A={model.input_dim_A}")
    if hasattr(model, "input_dim_B") and XB.shape[1] != getattr(model, "input_dim_B"):
        raise ValueError(f"XB has {XB.shape[1]} features, but model.input_dim_B={model.input_dim_B}")


@torch.no_grad()
def predict_proba_loader(
    model,
    loader: DataLoader,
    device: Optional[torch.device] = None,
) -> np.ndarray:
    """
    Predict positive-class probabilities from a DataLoader that yields (XA, XB).
    """
    if device is None:
        device = torch.device("cpu")
    model.to(device).eval()

    out = []
    for XA_b, XB_b in loader:
        XA_b = XA_b.to(device).float()
        XB_b = XB_b.to(device).float()
        probs, *_ = model(XA_b, XB_b)      # model returns sigmoid probs in shape [B, 1]
        out.append(probs.squeeze(-1).cpu().numpy())
    return np.concatenate(out, axis=0) if out else np.empty((0,), dtype=np.float32)


def predict_proba_arrays(
    model,
    XA_np: np.ndarray,
    XB_np: np.ndarray,
    *,
    batch_size: int = 256,
    device: Optional[torch.device] = None,
) -> np.ndarray:
    """
    Convenience wrapper: take raw numpy arrays (XA=PLEC, XB=Avalon),
    build a loader, and return probabilities (shape: N,).
    """
    _check_dims(model, np.asarray(XA_np), np.asarray(XB_np))
    loader = make_pred_loader(XA_np, XB_np, batch_size=batch_size, shuffle=False)
    return predict_proba_loader(model, loader, device=device)


def predict_classes_arrays(
    model,
    XA_np: np.ndarray,
    XB_np: np.ndarray,
    *,
    threshold: float = 0.5,
    batch_size: int = 256,
    device: Optional[torch.device] = None,
) -> np.ndarray:
    """
    Return 0/1 predictions using a probability threshold (default 0.5).
    """
    probs = predict_proba_arrays(model, XA_np, XB_np, batch_size=batch_size, device=device)
    return (probs >= float(threshold)).astype(np.int64)


import numpy as np
import pandas as pd

def build_and_save_results(
    names,
    smiles,
    probs: np.ndarray,
    preds: np.ndarray = None,
    *,
    threshold: float = 0.5,
    sort_by: str = "probability",
    descending: bool = True,
    round_digits: int = 4,
    out_csv: str = "predictions.csv",
    out_xlsx: str = None,
) -> pd.DataFrame:
    """
    Create a human-readable table with columns:
    [molecule, smiles, probability, activity, activity_label]
    and save to disk.

    - names/smiles: lists aligned to your feature rows
    - probs: (N,) array of predicted probabilities
    - preds: optional (N,) 0/1 array; if None, computed via threshold
    """
    names = list(names)
    smiles = list(smiles)
    probs = np.asarray(probs, dtype=float).reshape(-1)

    if len(names) != len(smiles) or len(names) != len(probs):
        raise ValueError(f"Length mismatch: names={len(names)}, smiles={len(smiles)}, probs={len(probs)}")

    if preds is None:
        preds = (probs >= float(threshold)).astype(int)
    else:
        preds = np.asarray(preds, dtype=int).reshape(-1)
        if len(preds) != len(probs):
            raise ValueError(f"Length mismatch: preds={len(preds)}, probs={len(probs)}")

    df = pd.DataFrame({
        "molecule": names,
        "smiles": smiles,
        "probability": probs,
        "activity": preds,
    })
    df["probability"] = df["probability"].round(round_digits)
    df["activity_label"] = df["activity"].map({1: "Active", 0: "Inactive"})

    df = df.sort_values(sort_by, ascending=not descending).reset_index(drop=True)

    # save
    df.to_csv(out_csv, index=False)
    if out_xlsx:
        df.to_excel(out_xlsx, index=False)

    return df

