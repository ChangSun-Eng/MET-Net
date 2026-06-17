# Reproducibility guide

Every reported result maps to a command below. The fast path (no GPU) reproduces the
tables from the shipped per-run reference data; the full path retrains from scratch.

Environment: `conda env create -f environment.yml` (Python 3.10, PyTorch 2.6.0+cu126,
captum 0.8.0, shap 0.49.1). All training uses fixed seeds; the held-out test split is
fixed (`random_state=42`).

## Fast path - reproduce tables from shipped results (seconds, CPU)

| Manuscript item | Command | Expected |
|---|---|---|
| Table 3 (accuracy + ablation) | `python scripts/reproduce_tables.py` | every row `OK`; MET-Net 0.4835 / 0.8570 |
| Table 3 (variant view) | `python scripts/run_ablation.py` | all 9 variants `OK` |
| Tables 4-5 (EPE) | `python scripts/reproduce_tables.py` | per-variant OE/PE metrics + module effects |
| Stage-1 metrics | `reference_results/stage1_pretraining_metrics.csv` | SE-Net De 0.011/te 0.042; SLSP 0.453/0.911 |

Per-run sources: `reference_results/ablation_runs/<variant>_25runs.csv` (25 rows each:
fold, seed, test_rmse, test_r2, mci, pcar, ...).

## Full path - retrain from scratch (GPU)

### Stage 1 - independent subnet pretraining
```bash
python scripts/train_se_net.py --checkpoint models/se_net_pretrained.pth   # De RMSE ~0.011, te ~0.042
```
SE-Net pretraining generates its synthetic equivalent-section data in-script. The
pretrained **SLSP-Net** (`models/slsp_net_pretrained.pth`, RMSE ~0.453 / R^2 ~0.911) is
provided directly; the single-layer source dataset it was trained on is not
redistributed, so `train_slsp_net.py` is included for transparency of the method only.
Stage 2 below uses the provided pretrained subnets and the bilayer target data.

### Stage 2 - MET-Net fine-tuning (25 runs)
```bash
python scripts/train_met_net.py --data data/bilayer_target_OE.xlsx \
    --se-ckpt models/se_net_pretrained.pth --slsp-ckpt models/slsp_net_pretrained.pth \
    --out-dir runs/met_net
```
Outputs `met_net_ensemble.pkl`, `training_summary_metrics_met_net.csv`, and per-run
`per_epoch_*`, `final_metrics_*`, `reliability_bins_*`, `test_predictions_*`. The median
test RMSE over the 25 runs is 0.4835 (R^2 0.8570, MCI 0.306). Typical convergence ~600
epochs; ~minutes per run on an RTX 4060.

### Model selection (as in the paper)
- **Fold3-Seed3** (median validation RMSE) -> training dynamics (Fig. 7).
- **Fold0-Seed0** (best comprehensive RMSE + MCI) -> explainability (Figs 8-12).
Both are provided in `models/met_net_fold{0_seed0,3_seed3}.pth`.

## Figures

| Figure | Command | Notes |
|---|---|---|
| Fig. 7 (training dynamics) | `python scripts/reproduce_figures.py` | redrawn from the frozen Fold3-Seed3 per-epoch record: RMSE, calibrator, drift, lambda_f, monitoring weight, PCAR, MCI |
| Figs 8-9 (GradientSHAP) | `python scripts/explain_gradientshap.py` | Fold0-Seed0; 256 background, 100 path samples, 5 runs, robust MASV |
| Figs 10-11 (ALE) | `python scripts/explain_ale_regime.py` (ALE section) | 40 quantile bins on g_lambda, g_t |
| Fig. 12 (Error Regime Map) | `python scripts/explain_ale_regime.py` (regime section) | 100x100 grid over r, D; t at 10/50/90 pct |

## Tables 4-5 (EPE) from scratch
```bash
python scripts/run_epe.py
# defaults: --ensemble models/met_net_ensemble.pkl --oe data/bilayer_target_OE.xlsx
#           --pe data/bilayer_target_PE.xlsx --out-dir out_epe (all overridable)
```
Reproduces RMSE/MAE/ME for all five inference-time variants on the full paired OE/PE sets,
plus the module-effect deltas (Table 5). The "Full" row accuracy (RMSE/R2) is reported on
the held-out test split (Table-3 protocol) for OE and PE, consistent with manuscript Table 4.
