#!/usr/bin/env python
import argparse, pandas as pd
from attentionscore.prep.standardize import standardize_dataframe

def main():
    ap = argparse.ArgumentParser(description="Standardize ligand SMILES using RDKit + RO5 filter.")
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--smiles_col", default="SMILES")
    ap.add_argument("--id_col", default="ID")
    ap.add_argument("--active_col", default="pIC50")
    ap.add_argument("--ro5", type=int, default=4)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    df = pd.read_csv(args.inp)
    out = standardize_dataframe(df, id_col=args.id_col, smiles_col=args.smiles_col,
                                active_col=args.active_col, ro5=args.ro5)
    out[[args.id_col, "Standardize_smile"]].to_csv(args.out, index=False)
    print(f"Wrote {len(out)} standardized ligands to {args.out}")

if __name__ == "__main__":
    main()