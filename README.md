# MET-Net

**MET-Net** is a mechanism-guided explainable transfer-learning network for **bilayer
metal tube springback prediction** under data scarcity and incomplete physical
mechanisms. It reuses abundant single-layer springback knowledge for data-scarce bilayer
prediction through a differentiable equivalent-section module whose adaptation is
confined to two physically meaningful correction parameters, followed by conservative
fine-tuning with sparse physics-residual guardrails and a lightweight affine calibrator.

This repository contains the **source code and reproducibility package** for:

> C. Sun, H. Huo, Z. Wang, S. Zhang, Z. Li, Y. Xiang, J. Tan.
> *MET-Net: An equivalent-section mechanism-guided neural framework for explainable
> transfer learning in bilayer tube springback prediction under data scarcity and
> incomplete physical mechanisms.* Journal of Manufacturing Systems, 2026.
> https://doi.org/10.1016/j.jmsy.2026.06.003

On a copper-aluminium rotary-draw-bending case, MET-Net attains a **median test RMSE of
0.4835 degrees** using only 100 bilayer samples, approaching a single-layer baseline
trained on ten times more data.

---

## Architecture

```
bilayer (D, t, r) --> SE-Net --> (De, te) --+--> SLSP-Net --> affine calibrator --> springback
                       |                     |
              equivalent-section theory   manufacturing params x_manu
              + 2 corrections (g_lambda, g_t)
```

- **SE-Net** (Section Equivalent Network): non-trainable equivalent-section theory layer
  + a compact adaptive head learning two corrections `g_lambda`, `g_t`.
- **SLSP-Net** (Single-Layer Springback Prediction Network): MLP pretrained on the
  abundant single-layer source domain.
- **Output calibrator**: lightweight affine correction (`alpha_cal`, `beta_cal`).
- **PDPM** (Physics-Data Paradigm Monitor): a passive, DeepLIFT-attribution training
  monitor reporting MAPD / PCAR / MCI (it does not affect predictions; the physics
  residual is applied uniformly, manuscript Eq. 13).

## Repository layout

```
MET-Net/
  src/metnet/met_net.py      core model + losses + PDPM + training pipeline
  scripts/
    train_se_net.py          Stage-1: SE-Net pretraining          -> se_net_pretrained.pth
    train_slsp_net.py        Stage-1: SLSP-Net pretraining         -> slsp_net_pretrained.pth
    train_met_net.py         Stage-2: MET-Net fine-tuning (25-run CV) -> met_net_ensemble.pkl
    run_ablation.py          ablation study (Table 3)
    run_epe.py               paired environment-perturbation eval (Tables 4-5)
    explain_gradientshap.py  GradientSHAP attribution (Figs 8-9)
    explain_ale_regime.py    ALE curves + Error Regime Maps (Figs 10-12)
    reproduce_tables.py      aggregate reference results -> Tables 3-5
    reproduce_figures.py     redraw Fig. 7 (training dynamics) from the frozen record
  configs/met_net.yaml       all hyperparameters (as run)
  data/                      bilayer target datasets (OE / PE)
  models/                    pretrained subnets + paper-selected MET-Net checkpoints
  reference_results/         verified per-run metrics + Tables 3-5 + training dynamics
  docs/                      reproducibility, data dictionary, naming map
```

> The single-layer source domain is used only to pretrain SLSP-Net; the **trained
> SLSP-Net checkpoint** (`models/slsp_net_pretrained.pth`) is provided directly, so the
> single-layer dataset itself is not redistributed. Everything needed to reproduce the
> bilayer results (Stage-2, ablation, EPE, explainability) is included. The released
> bilayer datasets contain 98 (OE) and 60 (PE) usable samples; the manuscript refers to
> the target domain as ~100 samples.

The trained 25-model (5-fold x 5-seed) ensemble (`models/met_net_ensemble.pkl`) and the
two paper-selected checkpoints are included in this repository, so every reported table
and figure can be reproduced directly.

## Install

```bash
conda env create -f environment.yml   # or: pip install -r requirements.txt
conda activate met-net
pip install -e .                       # makes `import metnet` work (uses src/ layout)
```
- **CPU is enough** for all reproduction commands below (they run in minutes on a laptop).
- The fast paths install a CPU build of PyTorch. For GPU training, install a CUDA build,
  e.g. `pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu126`.
- Reference versions: Python 3.10, PyTorch 2.6.0, captum 0.8.0, shap 0.49.1,
  scikit-learn 1.7.2, numpy 1.26.4. Reported results were produced on an NVIDIA RTX 4060.

> If you prefer not to install the package, every script also works after
> `export PYTHONPATH=$PWD/src` (Windows: `set PYTHONPATH=%CD%\src`).

## Quick start: reproduce the tables

Run all commands **from the repository root** (script defaults are relative to it).

```bash
python scripts/reproduce_tables.py    # Tables 3, 4, 5 from the shipped per-run results
python scripts/run_ablation.py        # ablation Table 3 + variant definitions
```
Both assert the published numbers and print `OK` for every row.

## Full pipeline (from scratch)

```bash
# Stage 1: SE-Net pretraining (synthetic equivalent-section data is generated in-script)
python scripts/train_se_net.py --checkpoint models/se_net_pretrained.pth
# Stage 1: SLSP-Net pretraining is included for transparency only; the single-layer source
#          data are not redistributed, so the pretrained models/slsp_net_pretrained.pth is
#          provided directly and used below.

# Stage 2 (assembled MET-Net fine-tuning, 5 folds x 5 seeds = 25 runs)
python scripts/train_met_net.py \
    --data data/bilayer_target_OE.xlsx \
    --se-ckpt models/se_net_pretrained.pth \
    --slsp-ckpt models/slsp_net_pretrained.pth \
    --out-dir runs/met_net

# Evaluation / figures (use the paper-selected checkpoints in models/)
python scripts/run_epe.py                 # Tables 4-5 (uses models/met_net_ensemble.pkl)
python scripts/reproduce_figures.py       # Fig. 7  (training dynamics, from frozen record)
python scripts/explain_gradientshap.py    # Figs 8-9   (Fold0-Seed0)
python scripts/explain_ale_regime.py      # Figs 10-12 (Fold0-Seed0)
```
The two paper-selected models are provided: **Fold0-Seed0** (explainability) and
**Fold3-Seed3** (training dynamics, Fig. 7). Load them with:
```python
from metnet.met_net import METNet
model = METNet.load_paper_model("models/met_net_fold0_seed0.pth")
```

## Reproducibility & transparency

- `docs/reproducibility.md` - step-by-step mapping from each table/figure to a command.
- `docs/data_dictionary.md` - dataset columns, symbols, units, ranges.
- `docs/naming_map.md` - development-code names -> manuscript names (e.g. ECM-Net ->
  MET-Net; the removed "gate" -> the PDPM monitor).
- The EPE "Full" row accuracy (RMSE/R2) is computed on the held-out test split (the
  Table-3 protocol); the other inference-time variants use the full paired-set
  evaluation. Each row's protocol is recorded in the `protocol` column of
  `reference_results/table4_epe_per_variant.csv`.

## License & citation

Code: **MIT** (`LICENSE`). Data: **CC BY 4.0** (`data/LICENSE.txt`). Please cite the
article above (`CITATION.cff`).
