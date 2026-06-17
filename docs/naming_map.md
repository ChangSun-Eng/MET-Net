# Naming map: development code -> published manuscript

The original research code evolved through several names. This release renames
everything to match the manuscript. The table below is the authoritative mapping
(useful when comparing this release to the original project, or to the paper).

## Models and modules

| Development code | Released / manuscript name | Notes |
|---|---|---|
| `ECM-Net`, `ECM_NET`, `EMT-Net` | **MET-Net** | Mechanism-guided Explainable Transfer-learning Network |
| class `ECMNetWithGate` | class `METNet` | the assembled model |
| class `ECMNetEnsembleAdvanced` | class `METNetEnsemble` | 25-model ensemble |
| `SENet` / "SE-NET" | `SENet` / **SE-Net** | Section Equivalent Network |
| `SLSPNet` / "SLSP-NET" | `SLSPNet` / **SLSP-Net** | Single-Layer Springback Prediction Network |
| `OutputCalibrator` | `OutputCalibrator` | affine output calibrator (alpha_cal, beta_cal) |

## The "gate" -> PDPM monitor

The early architecture had a DeepLIFT-attribution **gate** that re-weighted the physics
loss per sample. This gate was **removed**: the final MET-Net applies the physics-residual
hinge **uniformly** (manuscript Eq. 13). The development code base named the final model
`ECM-Net-NoGate`; that is exactly the published **MET-Net**. The DeepLIFT attribution
machinery survives only as the **passive PDPM monitor** (it does not affect predictions).

| Development code | Released name | Meaning |
|---|---|---|
| `...-NoGate` suffix | (dropped) | the no-gate model **is** MET-Net |
| `GateCalibrator` | `MonitorCalibrator` | PDPM monitoring-weight calibration |
| `compute_deeplift_gate` | `compute_pdpm_monitor` | DeepLIFT attribution -> pi_theory + monitoring weight |
| `w_gate` | `w_monitor` | calibrated monitoring weight (passive) |
| `gate_calibrator` / `update_gate_calibrator` | `monitor_calibrator` / `update_monitor_calibrator` | |
| CSV column `final_ece`, `ece_gate` | `mci` | Monitor Calibration Index (paper Eq. 21) |
| `final_phys_active` | `pcar` | Physics Constraint Activation Rate (Eq. 22) |
| (mean of pi_theory) | `MAPD` | Mean Attribution-based Physics Dependence (Eq. 23) |

## Experiment / evaluation terminology

| Development code | Released / manuscript name |
|---|---|
| CME (cross-manufacturing-environment) | **EPE** (paired zero-shot environment-perturbation evaluation) |
| IME / `Factory_A` | **OE** (original environment) |
| CME / `Factory_B` / `FactoryB` | **PE** (perturbed environment) |
| `SEoff`, `SE-CorrectionOFF` | **SE-AdaptiveOFF@test** |
| `NoCal`, `FrozenHead`, `FH_NoCal` | NoCal@test, FrozenHead@test, FH_NoCal@test |

## Files

| Development file/name | Released name |
|---|---|
| `DB3_0923001_ENG.xlsx`, `Bilayer_dataset_01.xlsx`, `Factory_A.xlsx` | `bilayer_target_OE.xlsx` |
| `Bilayer_dataset_01_FactoryB.xlsx`, `Factory_B.xlsx` | `bilayer_target_PE.xlsx` |
| `Singlelayer_dataset_01.xlsx` | (single-layer source; not redistributed - `slsp_net_pretrained.pth` provided) |
| `senet_trained_optA.pth` | `se_net_pretrained.pth` |
| `slsp_net_for_ecm.pth` | `slsp_net_pretrained.pth` |
| `ecm_net_ensemble_NoGate_ablation_A2_01.pkl` | `met_net_ensemble.pkl` |
| `ECM_net_NoGate_ablation_A2_01.py` | `src/metnet/met_net.py` + `scripts/train_met_net.py` |
| `Modularity_claim_NoGate_*.py` | `scripts/run_epe.py` |
| `SE_NET_20250916_C1_01.py` | `scripts/train_se_net.py` |
| `step2_tree_guided_nn_Final_02.py` | `scripts/train_slsp_net.py` |
| `GradientSHAP_*_B1_03.py` | `scripts/explain_gradientshap.py` |
| `ALE_Mechanism_Discovery_*` + `Error_Regime_Map_*` | `scripts/explain_ale_regime.py` |
