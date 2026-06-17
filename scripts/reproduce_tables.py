"""
Reproduce the paper's quantitative tables from the released per-run reference CSVs.

Aggregates reference_results/ablation_runs/*_25runs.csv into Table 3 (accuracy +
ablation), prints the EPE Table 4 / Table 5 reference files, and asserts the headline
numbers against the published values.

    python scripts/reproduce_tables.py
"""
import os
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
REF = os.path.join(HERE, "..", "reference_results")

# paper Table 3 (median test RMSE / R2) -- ground truth for the assertion
PAPER_T3 = {
    "MET-Net": (0.4835, 0.8570), "MET-Net-FreezeSLSP": (0.4965, 0.8492),
    "MET-Net-SEN": (0.5666, 0.8037), "MET-Net-FWP": (0.5798, 0.7944),
    "MET-Net-FreezeSE": (0.6541, 0.7384), "MET-Net-TheorySE": (0.6634, 0.7308),
    "MET-Net-NoCal": (1.0256, 0.3566), "MET-Net-BN": (2.7053, -3.4762),
    "MET-Net-SLN": (2.7114, -3.4965),
}
FILES = {
    "MET-Net": "met_net", "MET-Net-FreezeSLSP": "met_net_freezeslsp",
    "MET-Net-SEN": "met_net_sen", "MET-Net-FWP": "met_net_fwp",
    "MET-Net-FreezeSE": "met_net_freezese", "MET-Net-TheorySE": "met_net_theoryse",
    "MET-Net-NoCal": "met_net_nocal", "MET-Net-BN": "met_net_bn", "MET-Net-SLN": "met_net_sln",
}


def med_iqr(s):
    s = pd.to_numeric(s, errors="coerce").dropna()
    return np.median(s), np.percentile(s, 25), np.percentile(s, 75)


def main():
    print("=" * 78)
    print("TABLE 3  -  Accuracy and ablation (median test RMSE / R^2 over 25 runs)")
    print("=" * 78)
    print(f"{'Variant':22s} {'RMSE median [IQR]':24s} {'R2 median [IQR]':24s} {'check'}")
    all_ok = True
    for name, clean in FILES.items():
        df = pd.read_csv(os.path.join(REF, "ablation_runs", f"{clean}_25runs.csv"))
        rm, rlo, rhi = med_iqr(df["test_rmse"])
        r2m, r2lo, r2hi = med_iqr(df["test_r2"])
        p_rmse, p_r2 = PAPER_T3[name]
        ok = abs(rm - p_rmse) < 1e-3 and abs(r2m - p_r2) < 1e-3
        all_ok &= ok
        print(f"{name:22s} {rm:.4f} [{rlo:.4f}, {rhi:.4f}]   {r2m:7.4f} [{r2lo:.4f}, {r2hi:.4f}]   "
              f"{'OK' if ok else 'MISMATCH'}")
    print(f"\nTable 3 reproduces the published values exactly: {all_ok}")

    for fn, title in [("table4_epe_per_variant.csv", "TABLE 4  -  EPE per-variant metrics (Full row: held-out test protocol; intervention rows: full paired sets)"),
                      ("table5_module_effects.csv", "TABLE 5  -  Inference-time module effects"),
                      ("stage1_pretraining_metrics.csv", "STAGE-1 pretraining metrics")]:
        p = os.path.join(REF, fn)
        if os.path.exists(p):
            print("\n" + "=" * 78); print(title); print("=" * 78)
            print(pd.read_csv(p).to_string(index=False))

    # Assert the released EPE reference values (Table 4 Full row + Table 5 effects),
    # so a passing run guarantees the shipped tables match the documented numbers.
    EPE_OK = True
    t4 = pd.read_csv(os.path.join(REF, "table4_epe_per_variant.csv")).set_index("variant")
    for col, exp in [("RMSE_OE", 0.4835), ("R2_OE", 0.8570), ("RMSE_PE", 0.5537), ("R2_PE", 0.8522)]:
        EPE_OK &= abs(float(t4.loc["Full", col]) - exp) < 1e-3
    t5 = pd.read_csv(os.path.join(REF, "table5_module_effects.csv")).set_index("effect")
    for eff, col, exp in [("Output Calibrator", "dRMSE_OE", 0.4104),
                          ("SLSP Last Layer", "dRMSE_OE", 0.0085),
                          ("SE-Net Adaptive", "dRMSE_OE", 0.5592)]:
        EPE_OK &= abs(float(t5.loc[eff, col]) - exp) < 1e-3
    print(f"\nTable 4 (Full row) and Table 5 effects match the released reference values: {EPE_OK}")
    all_ok &= EPE_OK

    print("\nNote: the Full-row accuracy (RMSE/R2) is the held-out test result (Table-3")
    print("protocol) for OE and PE, consistent with manuscript Table 4; the other EPE")
    print("variants use the full paired-set evaluation.")
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
