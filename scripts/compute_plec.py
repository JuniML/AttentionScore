#!/usr/bin/env python
import argparse
from attentionscore.features.plec import compute_plec_dir

def main():
    ap = argparse.ArgumentParser(description="Compute PLEC from docked SDFs using ODDT.")
    ap.add_argument("--receptor", required=True, help="Protein file (.pdb, .mol2, etc.)")
    ap.add_argument("--docked_dir", required=True, help="Directory with docked SDFs")
    ap.add_argument("--out", required=True, help="Parquet output path")
    ap.add_argument("--n_jobs", type=int, default=8)
    ap.add_argument("--size", type=int, default=4092)
    ap.add_argument("--prot_depth", type=int, default=4)
    ap.add_argument("--lig_depth", type=int, default=2)
    ap.add_argument("--cutoff", type=float, default=4.5)
    args = ap.parse_args()

    compute_plec_dir(
        docked_dir=args.docked_dir, protein_path=args.receptor, out_parquet=args.out,
        n_jobs=args.n_jobs, size=args.size, depth_protein=args.prot_depth,
        depth_ligand=args.lig_depth, distance_cutoff=args.cutoff, sparse=False
    )
    print(f"Wrote PLEC parquet to {args.out}")

if __name__ == "__main__":
    main()