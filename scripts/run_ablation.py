"""
Ablation study (manuscript Table 3) for MET-Net.

Each variant differs from the full MET-Net by a SINGLE, documented modification to the
Stage-2 fine-tuning of `scripts/train_met_net.py`. The authoritative per-run results
(5 folds x 5 seeds = 25 runs each) are shipped under
``reference_results/ablation_runs/`` and were verified to reproduce manuscript Table 3
exactly (see ``scripts/reproduce_tables.py``).

Running this script (no arguments) prints the variant definitions and aggregates the
shipped per-run CSVs into Table 3. To RE-TRAIN a variant from scratch, apply its
one-line modification (column "modification") to the corresponding place in
``train_met_net.py`` / ``src/metnet/met_net.py`` and re-run; the change points are
indicated below.
"""
import os
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
REF = os.path.join(HERE, "..", "reference_results", "ablation_runs")

# variant -> (clean csv stem, one-line definition, exact code modification vs. base)
VARIANTS = [
    ("MET-Net", "met_net",
     "Full model (baseline).",
     "none (baseline)"),
    ("MET-Net-NoCal", "met_net_nocal",
     "Output calibrator fixed at identity (alpha_cal=1, beta_cal=0).",
     "exclude output_calibrator from the Stage-2 optimizer (do not train it)"),
    ("MET-Net-FreezeSE", "met_net_freezese",
     "SE-Net frozen during fine-tuning.",
     "exclude se_net from the Stage-2 optimizer"),
    ("MET-Net-FreezeSLSP", "met_net_freezeslsp",
     "SLSP-Net fully frozen, including its last layer.",
     "do not unfreeze slsp_net.last_layer; exclude it from the optimizer"),
    ("MET-Net-FWP", "met_net_fwp",
     "Fine-tune without the physics-residual guardrail.",
     "set lambda_f = 0 throughout Stage-2 (drop L_phys)"),
    ("MET-Net-SEN", "met_net_sen",
     "SE-Net not pretrained in Stage 1 (random init).",
     "skip loading se_net_pretrained.pth in METNet.load_pretrained()"),
    ("MET-Net-SLN", "met_net_sln",
     "SLSP-Net not pretrained in Stage 1.",
     "skip loading slsp_net_pretrained.pth weights (scalers still applied)"),
    ("MET-Net-BN", "met_net_bn",
     "Both subnets not pretrained in Stage 1.",
     "skip loading both se_net and slsp_net pretrained weights"),
    ("MET-Net-TheorySE", "met_net_theoryse",
     "SE-Net frozen at pure theory (adaptive corrections disabled: g_lambda=g_t=0).",
     "zero and freeze se_net.head.out (theory-only), and freeze se_net"),
]

# manuscript Table 3 ground truth (median test RMSE / R2)
PAPER = {
    "MET-Net": (0.4835, 0.8570), "MET-Net-FreezeSLSP": (0.4965, 0.8492),
    "MET-Net-SEN": (0.5666, 0.8037), "MET-Net-FWP": (0.5798, 0.7944),
    "MET-Net-FreezeSE": (0.6541, 0.7384), "MET-Net-TheorySE": (0.6634, 0.7308),
    "MET-Net-NoCal": (1.0256, 0.3566), "MET-Net-BN": (2.7053, -3.4762),
    "MET-Net-SLN": (2.7114, -3.4965),
}


def main():
    print("=" * 96)
    print("MET-Net ablation variants (manuscript Table 3)")
    print("=" * 96)
    rows, all_ok = [], True
    for name, stem, definition, mod in VARIANTS:
        csv = os.path.join(REF, f"{stem}_25runs.csv")
        df = pd.read_csv(csv)
        rmse = np.median(pd.to_numeric(df["test_rmse"], errors="coerce").dropna())
        r2 = np.median(pd.to_numeric(df["test_r2"], errors="coerce").dropna())
        p_rmse, p_r2 = PAPER[name]
        ok = abs(rmse - p_rmse) < 1e-3 and abs(r2 - p_r2) < 1e-3
        all_ok &= ok
        rows.append((name, rmse, r2, "OK" if ok else "MISMATCH", definition, mod))
    print(f"{'Variant':20s} {'RMSE':>8s} {'R2':>9s}  {'chk':4s}  definition")
    for name, rmse, r2, chk, definition, mod in rows:
        print(f"{name:20s} {rmse:8.4f} {r2:9.4f}  {chk:4s}  {definition}")
    print(f"\nAll variants reproduce manuscript Table 3: {all_ok}")
    print("\nTo re-train a variant from scratch, apply its single modification to the base")
    print("Stage-2 pipeline (train_met_net.py / met_net.py):")
    for name, _, _, mod in [(r[0], None, None, r[5]) for r in rows]:
        print(f"  - {name:20s}: {mod}")
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
