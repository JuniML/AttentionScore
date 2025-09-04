from __future__ import annotations

import os, random
from typing import Dict, List, Tuple, Optional
import numpy as np
import torch
from torch import nn
from torch.utils.data import Subset, DataLoader
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, average_precision_score, matthews_corrcoef

from .models import Model
from .data import CustomDataset

def set_global_seed(seed: int = 42):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def compute_metrics_from_probs(y_true, y_prob, threshold: float = 0.5):
    y_true = np.asarray(y_true, dtype=float)
    y_prob = np.asarray(y_prob, dtype=float)
    y_pred = (y_prob >= threshold).astype(float)

    acc  = (y_pred == y_true).mean()
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec  = recall_score(y_true, y_pred, zero_division=0)
    f1   = f1_score(y_true, y_pred, zero_division=0)
    try:
        roc = roc_auc_score(y_true, y_prob)
    except ValueError:
        roc = float("nan")
    pr   = average_precision_score(y_true, y_prob)
    mcc  = matthews_corrcoef(y_true, y_pred)
    return acc, prec, rec, f1, roc, pr, mcc

def evaluate_on_loader(model: torch.nn.Module, loader: DataLoader, device: torch.device, criterion_bce: nn.Module, criterion_mse: nn.Module):
    model.eval()
    loss_sum = 0.0
    count = 0
    probs_all, labels_all = [], []
    with torch.no_grad():
        for XA_batch, XB_batch, targets in loader:
            XA_batch = XA_batch.to(device).float()
            XB_batch = XB_batch.to(device).float()
            targets  = targets.to(device).unsqueeze(1)

            predictions, XC, XDC, XAC, XBC = model(XA_batch, XB_batch)
            loss1 = criterion_bce(predictions, targets)
            loss2 = criterion_mse(XA_batch, XAC)
            loss3 = criterion_mse(XB_batch, XBC)
            tot = 1*loss1 + loss2 + loss3

            loss_sum += tot.item() * targets.size(0)
            count += targets.size(0)

            probs = predictions.detach().squeeze().cpu().numpy()
            lbls  = targets.detach().cpu().numpy().squeeze()
            probs_all.extend(np.atleast_1d(probs))
            labels_all.extend(np.atleast_1d(lbls))

    avg_loss = loss_sum / max(1, count)
    metrics = compute_metrics_from_probs(labels_all, probs_all, threshold=0.5)
    return avg_loss, metrics, np.asarray(probs_all), np.asarray(labels_all)

def train_kfold(
    XA: torch.Tensor,
    XB: torch.Tensor,
    y: torch.Tensor,
    input_dim_A: int,
    input_dim_B: int,
    n_heads: int = 1,
    n_layers: int = 1,
    num_epochs: int = 10,
    batch_size: int = 64,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    seed: int = 42,
    device: Optional[torch.device] = None,
    n_splits: int = 5,
) -> Dict[str, object]:
    set_global_seed(seed)
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = CustomDataset(XA.float(), XB.float(), y.float())
    N = len(dataset)
    y_all = np.array([float(dataset[i][2]) for i in range(N)])

    kf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    splits = list(kf.split(np.arange(N), y_all))

    history = {
        "train_losses": [], "val_losses": [],
        "train_metrics": [], "val_metrics": [],
        "fold_probs": [], "fold_labels": [],
    }

    for fold, (train_idx, val_idx) in enumerate(splits, start=1):
        print(f"\\n===== Fold {fold}/{n_splits} =====", flush=True)

        model = Model(input_dim_A, input_dim_B, n_heads, n_layers).to(device)
        criterion_bce = nn.BCELoss()
        criterion_mse = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

        train_subset = Subset(dataset, train_idx)
        val_subset   = Subset(dataset, val_idx)

        g = torch.Generator()
        g.manual_seed(seed)
        train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True, generator=g)
        val_loader   = DataLoader(val_subset,   batch_size=batch_size, shuffle=False)

        train_losses, val_losses = [], []
        train_metrics_hist, val_metrics_hist = [], []

        for epoch in range(num_epochs):
            model.train()
            running_loss = 0.0
            total_samples = 0

            for XA_batch, XB_batch, targets in train_loader:
                XA_batch = XA_batch.to(device).float()
                XB_batch = XB_batch.to(device).float()
                targets  = targets.to(device).unsqueeze(1)

                optimizer.zero_grad()
                predictions, XC, XDC, XAC, XBC = model(XA_batch, XB_batch)

                loss1 = criterion_bce(predictions, targets)
                loss2 = criterion_mse(XA_batch, XAC)
                loss3 = criterion_mse(XB_batch, XBC)
                total_loss = 1*loss1 + loss2 + loss3

                total_loss.backward()
                optimizer.step()

                running_loss += total_loss.item() * targets.size(0)
                total_samples += targets.size(0)

            train_avg_loss, train_mets, _, _ = evaluate_on_loader(model, train_loader, device, criterion_bce, criterion_mse)
            val_avg_loss,   val_mets,   _, _ = evaluate_on_loader(model, val_loader,   device, criterion_bce, criterion_mse)

            train_losses.append(train_avg_loss)
            val_losses.append(val_avg_loss)
            train_metrics_hist.append(train_mets)
            val_metrics_hist.append(val_mets)

            print(
                f"Epoch {epoch+1}/{num_epochs} | TrainLoss {train_avg_loss:.4f} | ValLoss {val_avg_loss:.4f} | "
                f"Val Acc {val_mets[0]:.4f} | F1 {val_mets[3]:.4f} | ROC-AUC {val_mets[4]:.4f} | PR-AUC {val_mets[5]:.4f}",
                flush=True
            )

        history["train_losses"].append(train_losses)
        history["val_losses"].append(val_losses)
        history["train_metrics"].append(train_metrics_hist)
        history["val_metrics"].append(val_metrics_hist)

        _, _, probs, labels = evaluate_on_loader(model, val_loader, device, criterion_bce, criterion_mse)
        history["fold_probs"].append(probs)
        history["fold_labels"].append(labels)

    return history