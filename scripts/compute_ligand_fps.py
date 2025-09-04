#!/usr/bin/env python
import argparse, base64, pandas as pd
from attentionscore.features.fingerprints import calculate_avalon_array, morgan_ecfp4_array
import pyarrow as pa, pyarrow.parquet as pq
import numpy as np

def pack_bits(arr):
    return np.packbits(arr.astype('uint8')).tobytes()

def main():
    ap = argparse.ArgumentParser(description="Compute ECFP4 and Avalon fingerprints to Parquet.")
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--smiles_col", default="Standardize_smile")
    ap.add_argument("--id_col", default="ID")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    df = pd.read_csv(args.inp)
    rows = []
    for r in df.itertuples():
        smi = getattr(r, args.smiles_col)
        lid = getattr(r, args.id_col)
        ecfp = morgan_ecfp4_array(smi, nBits=2048, radius=2)
        aval = calculate_avalon_array(smi, nBits=512)
        rows.append({
            "id": str(lid),
            "smiles": smi,
            "ecfp4_bits": 2048,
            "avalon_bits": 512,
            "ecfp4_b64": base64.b64encode(pack_bits(ecfp)).decode("ascii"),
            "avalon_b64": base64.b64encode(pack_bits(aval)).decode("ascii"),
        })
    table = pa.Table.from_pandas(pd.DataFrame(rows))
    pq.write_table(table, args.out)
    print(f"Saved {len(rows)} ligands to {args.out}")

if __name__ == "__main__":
    main()