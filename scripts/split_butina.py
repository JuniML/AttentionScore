#!/usr/bin/env python
import argparse, base64, json, os
import numpy as np
import pandas as pd
from rdkit import DataStructs
from rdkit.ML.Cluster import Butina

def b64_to_bv(b64, n_bits):
    b = base64.b64decode(b64)
    bitstr = bin(int.from_bytes(b, "big"))[2:].zfill(n_bits)
    return DataStructs.CreateFromBitString(bitstr)

def load_fps(parquet_path, id_col="id"):
    df = pd.read_parquet(parquet_path)
    fps, ids = [], []
    for r in df.itertuples():
        fps.append(b64_to_bv(getattr(r, "ecfp4_b64"), getattr(r, "ecfp4_bits")))
        ids.append(str(getattr(r, id_col)))
    return ids, fps

def cluster_ids(ids, fps, sim_thresh):
    dists = []
    nfps = len(fps)
    for i in range(1, nfps):
        sims = DataStructs.BulkTanimotoSimilarity(fps[i], fps[:i])
        dists.extend([1 - x for x in sims])
    cs = Butina.ClusterData(dists, nfps, cutoff=1 - sim_thresh, isDistData=True)
    return [[ids[i] for i in c] for c in cs]

def split_from_clusters(clusters, ratios=(0.8,0.1,0.1)):
    train, val, test = [], [], []
    total = sum(len(c) for c in clusters)
    target = [int(total*r) for r in ratios]
    counts = [0,0,0]
    for c in sorted(clusters, key=len, reverse=True):
        deficits = [t - counts[i] for i,t in enumerate(target)]
        idx = int(np.argmin(deficits))
        (train if idx==0 else val if idx==1 else test).extend(c)
        counts[idx]+=len(c)
    return train, val, test

def compute_hard_test(train_ids, all_ids, all_fps, hard_thresh=0.5):
    id_to_fp = {i:fp for i,fp in zip(all_ids, all_fps)}
    train_fps = [id_to_fp[i] for i in train_ids]
    hard = []
    for tid in all_ids:
        if tid in train_ids: continue
        sims = DataStructs.BulkTanimotoSimilarity(id_to_fp[tid], train_fps) if train_fps else [0.0]
        if max(sims) <= hard_thresh: hard.append(tid)
    return hard

def main():
    ap = argparse.ArgumentParser(description="Create Butina-based splits and hard test subset.")
    ap.add_argument("--lig_fps", required=True)
    ap.add_argument("--id_col", default="id")
    ap.add_argument("--tau", type=float, default=0.50)
    ap.add_argument("--hard_thresh", type=float, default=0.50)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    ids, fps = load_fps(args.lig_fps, id_col=args.id_col)
    clusters = cluster_ids(ids, fps, sim_thresh=args.tau)
    train, val, test = split_from_clusters(clusters, ratios=(0.8,0.1,0.1))
    hard_pool = compute_hard_test(train, ids, fps, hard_thresh=args.hard_thresh)
    split = {"train": train, "val": val, "test": test, "hard_test": [i for i in test if i in hard_pool]}
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(split, f, indent=2)
    print(f"Wrote split to {args.out}")
    print({k: len(v) for k,v in split.items()})

if __name__ == "__main__":
    main()