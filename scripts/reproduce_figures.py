"""
Reproduce Fig. 7 (MET-Net Stage-2 fine-tuning dynamics) from the frozen per-epoch
record, without retraining.

Reads reference_results/training_dynamics/met_net_fold3_seed3_per_epoch.npz (the
median-validation-RMSE model, Fold3-Seed3, as used in the paper) and redraws the
training-dynamics panels: validation RMSE, physics-weight lambda_f schedule, mean PDPM
monitoring weight, physics-violation rate (PCAR), calibrator scale/shift, SLSP last-layer
drift, and MCI.

    python scripts/reproduce_figures.py            # -> figures/fig7_training_dynamics.png
    python scripts/reproduce_figures.py --out my.png
"""
import argparse
import os
import sys
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from metnet.met_net import plot_training_curves

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_NPZ = os.path.join(HERE, "..", "reference_results", "training_dynamics",
                           "met_net_fold3_seed3_per_epoch.npz")

# map the saved per-epoch keys to the names expected by plot_training_curves()
KEY_MAP = {
    "val_rmse": "val_rmse",
    "lambda_f": "lambda_f",
    "mean_w_monitor": "w_monitor_val_mean",
    "violation_rate": "violations_val_rate",
    "calib_scale_a": "calib_scale",
    "calib_shift_b": "calib_shift",
    "slsp_drift_w": "slsp_w_drift",
    "slsp_drift_b": "slsp_b_drift",
    "ece_equal_mass": "mci",
    "physics_auc": "physics_auc_curve",
    "epoch": "epoch",
}


def main():
    ap = argparse.ArgumentParser(description="Redraw Fig. 7 training dynamics from the frozen npz.")
    ap.add_argument("--npz", default=DEFAULT_NPZ)
    ap.add_argument("--out", default=os.path.join(HERE, "..", "figures", "fig7_training_dynamics.png"))
    args = ap.parse_args()

    z = np.load(args.npz, allow_pickle=True)
    metrics = {}
    for k in z.files:
        metrics[KEY_MAP.get(k, k)] = list(z[k])

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    plot_training_curves(metrics, save_path=args.out)

    def last_finite(key):
        a = np.asarray(metrics[key], dtype=float)
        a = a[np.isfinite(a)]
        return a[-1] if a.size else float("nan")

    print(f"[OK] Saved Fig. 7 (training dynamics) -> {args.out}")
    print(f"  final validation RMSE: {last_finite('val_rmse'):.4f}; "
          f"calibrator (alpha,beta)=({last_finite('calib_scale'):.3f},{last_finite('calib_shift'):.3f}); "
          f"final MCI: {last_finite('mci'):.3f}")


if __name__ == "__main__":
    main()
