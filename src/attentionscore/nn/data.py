from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Tuple, Optional

class CustomDataset(Dataset):
    def __init__(self, XA: torch.Tensor, XB: torch.Tensor, labels: torch.Tensor):
        assert len(XA) == len(XB) == len(labels), "XA, XB, labels must have same length"
        self.XA = XA
        self.XB = XB
        self.labels = labels

    def __len__(self) -> int:
        return len(self.XA)

    def __getitem__(self, idx: int):
        return self.XA[idx], self.XB[idx], self.labels[idx]

def tensors_from_numpy(XA, XB, y, device: Optional[torch.device] = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    def to_tensor(x):
        if isinstance(x, torch.Tensor):
            return x
        return torch.tensor(x, dtype=torch.float32)
    XA = to_tensor(XA)
    XB = to_tensor(XB)
    y  = torch.tensor(np.ravel(y), dtype=torch.float32)
    if device is not None:
        XA = XA.to(device)
        XB = XB.to(device)
        y  = y.to(device)
    return XA, XB, y

def make_dataloader(dataset: Dataset, batch_size: int = 128, shuffle: bool = True, seed: int = 42) -> DataLoader:
    g = torch.Generator()
    g.manual_seed(seed)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, generator=g)