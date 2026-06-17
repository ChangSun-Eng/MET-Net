"""
Stage-2: fine-tune the assembled MET-Net on the scarce bilayer target domain.

Runs the 5-fold x 5-seed (25-run) cross-validation reported in the paper and
exports per-run training curves, per-fold/seed metrics, reliability bins, test
predictions, the aggregate ``training_summary_metrics_met_net.csv``, and the
trained 25-model ensemble.

Prerequisites (Stage-1 outputs, see scripts/train_se_net.py and train_slsp_net.py):
  * se_net_pretrained.pth     - pretrained SE-Net (equivalent-section adaptive head)
  * slsp_net_pretrained.pth   - pretrained SLSP-Net (+ input/output scalers)

Example
-------
    python scripts/train_met_net.py \
        --data data/bilayer_target_OE.xlsx \
        --se-ckpt se_net_pretrained.pth \
        --slsp-ckpt slsp_net_pretrained.pth \
        --out-dir runs/met_net

Paper settings (defaults below): n_splits=5, n_seeds=5, test_size=0.15,
warmup_epochs=30, max_epochs=1000, Huber beta=0.4, lambda_f_max=1.0, kappa_f=3.0.
"""
import argparse
import os
import numpy as np
import pandas as pd
import torch

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from metnet.met_net import (
    set_seed,
    reorder_dataframe_columns,
    train_kfold_ensemble_with_metrics,
)


def main():
    ap = argparse.ArgumentParser(description="Stage-2 MET-Net fine-tuning (25-run CV).")
    ap.add_argument("--data", default="data/bilayer_target_OE.xlsx",
                    help="Bilayer target (OE) dataset: 14 input features + springback.")
    ap.add_argument("--se-ckpt", default="models/se_net_pretrained.pth")
    ap.add_argument("--slsp-ckpt", default="models/slsp_net_pretrained.pth")
    ap.add_argument("--out-dir", default=".")
    ap.add_argument("--n-splits", type=int, default=5)
    ap.add_argument("--n-seeds", type=int, default=5)
    ap.add_argument("--test-size", type=float, default=0.15)
    ap.add_argument("--warmup-epochs", type=int, default=30)
    ap.add_argument("--global-seed", type=int, default=20250928)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--no-plots", action="store_true")
    args = ap.parse_args()

    # Resolve all inputs to absolute paths BEFORE changing directory, so that outputs
    # can be written under --out-dir while the inputs still resolve correctly.
    data_path = os.path.abspath(args.data)
    se_ckpt = os.path.abspath(args.se_ckpt)
    slsp_ckpt = os.path.abspath(args.slsp_ckpt)
    for p in (data_path, se_ckpt, slsp_ckpt):
        if not os.path.exists(p):
            raise FileNotFoundError(f"Required input not found: {p}")

    os.makedirs(args.out_dir, exist_ok=True)
    if args.out_dir != ".":
        os.chdir(args.out_dir)

    set_seed(args.global_seed)

    df = reorder_dataframe_columns(pd.read_excel(data_path))
    X = df.iloc[:, :14].values
    y = df.iloc[:, 14].values
    print(f"Loaded {len(df)} bilayer (OE) samples; springback range "
          f"[{y.min():.4f}, {y.max():.4f}], mean {y.mean():.4f} +/- {y.std():.4f}")

    device = torch.device(args.device)
    print(f"Device: {device}")

    # The Stage-1 subnets are loaded inside each fold via METNet.load_pretrained();
    # pass the resolved checkpoint paths through (load_pretrained raises if missing).
    ensemble, test_results, _ = train_kfold_ensemble_with_metrics(
        X, y,
        n_splits=args.n_splits,
        n_seeds=args.n_seeds,
        test_size=args.test_size,
        device=device,
        verbose=True,
        warmup_epochs=args.warmup_epochs,
        make_all_plots=not args.no_plots,
        se_path=se_ckpt,
        slsp_path=slsp_ckpt,
    )

    ensemble.save("met_net_ensemble.pkl")
    np.savez(
        "met_net_ensemble_results.npz",
        predictions=test_results["predictions"],
        uncertainties=test_results["uncertainties"],
        rmse=test_results["rmse"],
        r2=test_results["r2"],
    )
    print("\nDone. Ensemble test RMSE %.4f, R2 %.4f"
          % (test_results["rmse"], test_results["r2"]))
    print("Aggregate metrics -> training_summary_metrics_met_net.csv")


if __name__ == "__main__":
    main()
