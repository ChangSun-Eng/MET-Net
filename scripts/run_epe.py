"""
Paired zero-shot Environment-Perturbation Evaluation (EPE) for MET-Net.
======================================================================
Reproduces the paper Tables 4-5: a trained MET-Net ensemble is evaluated,
without any further training, on a paired Original Environment (OE) set and a
Perturbed Environment (PE) set. Six inference-time variants are computed per
environment to attribute MET-Net's transfer behaviour to its modular structure:

  1. Full            - normal prediction (SE-Net adaptive head + SLSP head + calibrator)
  2. NoCal@test      - output calibrator removed (exact per-model inversion, then average)
  3. FrozenHead@test - SLSP last layer reverted to its pretrained (source-domain) weights
  4. FH_NoCal@test   - FrozenHead and NoCal combined
  5. SE-AdaptiveOFF@test - SE-Net adaptive correction disabled (pure equivalent-section
                       theory: g_lambda=0, g_t=0), calibrator active
  6. SEoff_nocal     - SE-AdaptiveOFF and NoCal combined (pure theory, no calibration)

Per-variant metrics (RMSE, MAE, R^2, ME), SE-AdaptiveOFF effect sizes, the
dimensionless mechanics residual r* (equivalent-section physics invariance), and
the SLSP head drift norms are reported, with paired CSV/table export.

No training is performed on the perturbed environment - evaluation only.

HEADER NOTE (metrics scope)
---------------------------
The comparative inference-time variants and the module-effect deltas (Table 5) are
computed on the FULL paired OE/PE sets. The 'Full' row accuracy (RMSE, R^2) is
reported on the held-out test split (Table-3 protocol): OE on the held-out test
samples and PE on their paired perturbed counterparts, median over the 25-model
ensemble. This is consistent with manuscript Table 3 / Table 4 'Full' row.

The SE-AdaptiveOFF variants disable the SE-Net adaptive correction at inference by
temporarily zeroing the correction-head output (g_lambda=0, g_t=0), which recovers
the pure equivalent-section theory. The model itself has no architectural gate; the
physics-residual hinge is applied uniformly during training (paper Eq. 13) and the
DeepLIFT attribution is the passive PDPM monitor (MAPD/PCAR/MCI).
"""

import os
import sys
import contextlib
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from metnet.met_net import METNet

# ============================================================================
# Configuration
# ============================================================================

CONFIG = {
    'checkpoint': 'models/met_net_ensemble.pkl',
    'se_net_path': 'models/se_net_pretrained.pth',
    'slsp_net_path': 'models/slsp_net_pretrained.pth',
    'oe_xlsx': 'data/bilayer_target_OE.xlsx',
    'pe_xlsx': 'data/bilayer_target_PE.xlsx',
    'output_dir': 'out_epe',
    'output_oe_csv': 'out_epe/preds_OE_variants.csv',
    'output_pe_csv': 'out_epe/preds_PE_variants.csv',
    'device': 'cpu'  # Change to 'cuda' if you have a GPU
}


# ============================================================================
# SE-Net adaptive-correction toggle (inference only)
# ============================================================================
# The packaged METNet always applies the SE-Net adaptive correction head. To
# evaluate the SE-AdaptiveOFF variants (pure equivalent-section theory) we
# temporarily zero the correction-head output weights/bias so g_lambda=0 and
# g_t=0. The head's output layer is zero-initialized, so this exactly recovers
# the theory-only path without touching the packaged model logic.

@contextlib.contextmanager
def se_adaptive_off(model):
    """Context manager that disables the SE-Net adaptive correction for `model`."""
    head_out = model.se_net.head.out
    saved_w = head_out.weight.detach().clone()
    saved_b = head_out.bias.detach().clone()
    try:
        with torch.no_grad():
            head_out.weight.zero_()
            head_out.bias.zero_()
        yield
    finally:
        with torch.no_grad():
            head_out.weight.copy_(saved_w)
            head_out.bias.copy_(saved_b)


def _springback(model, X_tensor):
    """Run a single MET-Net forward and return the springback prediction tensor."""
    out = model(X_tensor)
    pred = out[0] if isinstance(out, tuple) else out
    if pred.dim() == 2:
        pred = pred.squeeze(1)
    return pred


# ============================================================================
# Data Loading
# ============================================================================

BILAYER_INPUT_COLUMNS = [
    'Tube diameter', 'Total thickness', 'Thickness ratio',
    'Radius of bending die',
    'Friction between bending die and tube',
    'Friction between pressure die and tube',
    'Friction between wiper die and tube',
    'Gap between bending die and tube',
    'Gap between pressure die and tube',
    'Gap between wiper die and tube',
    'Different velocity of pressure die',
    'The initial position of the pressure die',
    'Angular velocity of bending die',
    'Bending angle of bending die',
    'Springback'
]


def load_dataset(xlsx_path, env_name):
    """Load an environment dataset (same column layout for OE and PE)."""
    df = pd.read_excel(xlsx_path)

    missing_cols = set(BILAYER_INPUT_COLUMNS) - set(df.columns)
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    df = df[BILAYER_INPUT_COLUMNS].copy()

    X = df.iloc[:, :14].values.astype(np.float32)
    y = df.iloc[:, 14].values.astype(np.float32)

    print(f"  [OK] Loaded {env_name}: {len(X)} samples")
    print(f"    Input shape: {X.shape}")
    print(f"    Springback range: [{y.min():.4f}, {y.max():.4f}]")

    return X, y


# ============================================================================
# Prediction Variants
# ============================================================================

@torch.no_grad()
def predict_variants_with_seoff(models, X, device='cpu'):
    """
    Generate all six prediction variants for the ensemble.

    NoCal variants use exact per-model calibrator inversion (invert each model's
    affine calibrator first, then average) which is the correct operation for an
    ensemble of affine-calibrated members.

    Returns:
        y_full, y_nocal, y_fh, y_fh_nocal, y_seoff_full, y_seoff_nocal, a_mean, b_mean
    """
    X_tensor = torch.tensor(X, dtype=torch.float32, device=device)

    all_y_full = []
    all_y_fh = []
    all_y_full_nocal = []
    all_y_fh_nocal = []
    all_y_seoff_full = []
    all_y_seoff_nocal = []
    calibrator_params = []

    for model in models:
        model.eval()
        model.to(device)

        # 1. Full prediction (this model)
        y_full_m = _springback(model, X_tensor)

        # Calibrator parameters (this model)
        a_m, b_m = model.output_calibrator.get_params()
        calibrator_params.append((a_m, b_m))

        # 2. FrozenHead@test: revert SLSP last layer to pretrained weights
        W_star = model.slsp_net.last_layer.weight.detach().clone()
        b_star = model.slsp_net.last_layer.bias.detach().clone()

        if model.slsp_net.W0 is None or model.slsp_net.b0 is None:
            raise ValueError(
                "Pretrained SLSP head weights (W0, b0) not found. "
                "Ensure load_pretrained() was called and the SLSP checkpoint stored them."
            )

        W0 = model.slsp_net.W0
        b0 = model.slsp_net.b0

        model.slsp_net.last_layer.weight.copy_(W0)
        model.slsp_net.last_layer.bias.copy_(b0)

        y_fh_m = _springback(model, X_tensor)

        # Restore adapted head
        model.slsp_net.last_layer.weight.copy_(W_star)
        model.slsp_net.last_layer.bias.copy_(b_star)

        # 3. SE-AdaptiveOFF@test: pure equivalent-section theory, calibrator active
        with se_adaptive_off(model):
            y_seoff_full_m = _springback(model, X_tensor)

        # 4. Per-model calibrator inversion (exact for the ensemble)
        y_full_nocal_m = (y_full_m - b_m) / a_m
        y_fh_nocal_m = (y_fh_m - b_m) / a_m
        y_seoff_nocal_m = (y_seoff_full_m - b_m) / a_m

        all_y_full.append(y_full_m.cpu().numpy())
        all_y_fh.append(y_fh_m.cpu().numpy())
        all_y_full_nocal.append(y_full_nocal_m.cpu().numpy())
        all_y_fh_nocal.append(y_fh_nocal_m.cpu().numpy())
        all_y_seoff_full.append(y_seoff_full_m.cpu().numpy())
        all_y_seoff_nocal.append(y_seoff_nocal_m.cpu().numpy())

    # Ensemble averaging (exact)
    y_full = np.mean(np.stack(all_y_full, axis=0), axis=0)
    y_fh = np.mean(np.stack(all_y_fh, axis=0), axis=0)
    y_nocal = np.mean(np.stack(all_y_full_nocal, axis=0), axis=0)
    y_fh_nocal = np.mean(np.stack(all_y_fh_nocal, axis=0), axis=0)
    y_seoff_full = np.mean(np.stack(all_y_seoff_full, axis=0), axis=0)
    y_seoff_nocal = np.mean(np.stack(all_y_seoff_nocal, axis=0), axis=0)

    a_mean = float(np.mean([p[0] for p in calibrator_params]))
    b_mean = float(np.mean([p[1] for p in calibrator_params]))

    return y_full, y_nocal, y_fh, y_fh_nocal, y_seoff_full, y_seoff_nocal, a_mean, b_mean


# ============================================================================
# Mechanics Residual (r*) Computation for SE-Net Invariance
# ============================================================================

@torch.no_grad()
def compute_rstar_with_model(model, X, device='cpu', k_f=0.1):
    """
    Compute the dimensionless mechanics residual r* from the model's SE-Net.
    Independent of the calibrator / SLSP-head toggles.

        r*    = |x^3 + x - delta_hat|,  x = te / (2*Re),  delta_hat = delta / (128*Re^4)
        gamma = k_f * |delta_hat|

    Returns r_star, gamma_star and the per-sample violation mask (r* > gamma).
    """
    X_t = torch.tensor(X, dtype=torch.float32, device=device)
    D = X_t[:, 0]
    t = X_t[:, 1]
    r = X_t[:, 2]

    se = model.se_net(D, t, r)
    Re = se["Re_eff"]
    te = se["te_pred"]
    delta = se["delta_eff"]

    eps = 1e-12
    Re_safe = torch.clamp(Re, min=eps)

    x = te / (2.0 * Re_safe)
    delta_hat = delta / (128.0 * (Re_safe**4))

    r_star = torch.abs(x**3 + x - delta_hat)
    gamma_star = k_f * torch.abs(delta_hat)
    viol = (r_star > gamma_star).to(torch.int32)

    return r_star.cpu().numpy(), gamma_star.cpu().numpy(), viol.cpu().numpy()


@torch.no_grad()
def compute_rstar_comparison(model, X, device='cpu', k_f=0.1):
    """
    Compare r* with the SE-Net adaptive correction ON (normal) vs OFF (pure theory).

    Returns a dictionary of r* statistics for both cases plus the improvement
    (theory minus corrected). The SE-Net is the single source-of-truth here; the
    adaptive correction is disabled by zeroing the correction-head output.
    """
    X_t = torch.tensor(X, dtype=torch.float32, device=device)
    D = X_t[:, 0]
    t = X_t[:, 1]
    r = X_t[:, 2]

    eps = 1e-12

    # --- Case 1: With SE-Net adaptive correction (normal) ---
    se_corrected = model.se_net(D, t, r)
    Re_se = se_corrected["Re_eff"]
    te_se = se_corrected["te_pred"]
    delta_se = se_corrected["delta_eff"]

    Re_safe_se = torch.clamp(Re_se, min=eps)
    x_se = te_se / (2.0 * Re_safe_se)
    delta_hat_se = delta_se / (128.0 * (Re_safe_se**4))
    r_star_se = torch.abs(x_se**3 + x_se - delta_hat_se)
    gamma_star_se = k_f * torch.abs(delta_hat_se)
    viol_se = (r_star_se > gamma_star_se).to(torch.float32)

    # --- Case 2: Without SE-Net adaptive correction (pure theory) ---
    with se_adaptive_off(model):
        se_theory = model.se_net(D, t, r)
    Re_theory = se_theory["Re_eff"]
    te_theory = se_theory["te_pred"]
    delta_theory = se_theory["delta_eff"]

    Re_safe_theory = torch.clamp(Re_theory, min=eps)
    x_theory = te_theory / (2.0 * Re_safe_theory)
    delta_hat_theory = delta_theory / (128.0 * (Re_safe_theory**4))
    r_star_theory = torch.abs(x_theory**3 + x_theory - delta_hat_theory)
    gamma_star_theory = k_f * torch.abs(delta_hat_theory)
    viol_theory = (r_star_theory > gamma_star_theory).to(torch.float32)

    r_star_se_np = r_star_se.cpu().numpy()
    r_star_theory_np = r_star_theory.cpu().numpy()
    viol_se_np = viol_se.cpu().numpy()
    viol_theory_np = viol_theory.cpu().numpy()

    return {
        'r_star_SE': r_star_se_np,
        'median_SE': float(np.median(r_star_se_np)),
        'mean_SE': float(np.mean(r_star_se_np)),
        'viol_rate_SE': float(np.mean(viol_se_np)),

        'r_star_Theory': r_star_theory_np,
        'median_Theory': float(np.median(r_star_theory_np)),
        'mean_Theory': float(np.mean(r_star_theory_np)),
        'viol_rate_Theory': float(np.mean(viol_theory_np)),

        'Delta_median': float(np.median(r_star_theory_np) - np.median(r_star_se_np)),
        'Delta_mean': float(np.mean(r_star_theory_np) - np.mean(r_star_se_np)),
        'Delta_viol_rate': float(np.mean(viol_theory_np) - np.mean(viol_se_np))
    }


# ============================================================================
# Metrics Computation
# ============================================================================

def compute_metrics(y_true, y_pred):
    """Compute RMSE, MAE, R^2, ME (mean error / bias)."""
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    me = np.mean(y_pred - y_true)

    return {'RMSE': rmse, 'MAE': mae, 'R2': r2, 'ME': me}


def compute_se_effects(y_true, y_full, y_seoff_full, y_nocal, y_seoff_nocal):
    """
    Quantify how much the SE-Net adaptive correction improves over pure theory
    (SE-AdaptiveOFF). All quantities are paired differences on the same set.
    """
    rmse_full = np.sqrt(mean_squared_error(y_true, y_full))
    rmse_seoff_full = np.sqrt(mean_squared_error(y_true, y_seoff_full))
    delta_se_rmse = rmse_seoff_full - rmse_full
    delta_se_rmse_pct = (delta_se_rmse / rmse_full) * 100.0

    me_full = np.mean(y_full - y_true)
    me_seoff_full = np.mean(y_seoff_full - y_true)
    delta_se_me = me_seoff_full - me_full

    mae_full = mean_absolute_error(y_true, y_full)
    mae_seoff_full = mean_absolute_error(y_true, y_seoff_full)
    delta_se_mae = mae_seoff_full - mae_full

    r2_full = r2_score(y_true, y_full)
    r2_seoff_full = r2_score(y_true, y_seoff_full)
    delta_se_r2 = r2_seoff_full - r2_full

    me_nocal = np.mean(y_nocal - y_true)
    me_seoff_nocal = np.mean(y_seoff_nocal - y_true)
    delta_se_bias_raw = me_seoff_nocal - me_nocal

    return {
        'RMSE_Full': rmse_full,
        'RMSE_SEoff': rmse_seoff_full,
        'Delta_SE_RMSE': delta_se_rmse,
        'Delta_SE_RMSE_pct': delta_se_rmse_pct,
        'ME_Full': me_full,
        'ME_SEoff': me_seoff_full,
        'Delta_SE_ME': delta_se_me,
        'MAE_Full': mae_full,
        'MAE_SEoff': mae_seoff_full,
        'Delta_SE_MAE': delta_se_mae,
        'R2_Full': r2_full,
        'R2_SEoff': r2_seoff_full,
        'Delta_SE_R2': delta_se_r2,
        'ME_NoCal': me_nocal,
        'ME_SEoff_NoCal': me_seoff_nocal,
        'Delta_SE_Bias_Raw': delta_se_bias_raw
    }


# ============================================================================
# CSV Output
# ============================================================================

def save_predictions_csv(y_true, y_full, y_nocal, y_fh, y_fh_nocal,
                         y_seoff_full, y_seoff_nocal, a, b, output_path):
    """Save predictions for all six variants to CSV."""
    df = pd.DataFrame({
        'sample_id': np.arange(1, len(y_true) + 1),
        'y_true': y_true,
        'y_full': y_full,
        'y_nocal': y_nocal,
        'y_fh': y_fh,
        'y_fh_nocal': y_fh_nocal,
        'y_seoff_full': y_seoff_full,
        'y_seoff_nocal': y_seoff_nocal,
        'a': a,
        'b': b
    })

    df.to_csv(output_path, index=False, float_format='%.6f')
    print(f"  [OK] Saved predictions to {output_path}")


def print_metrics_table(metrics_dict, env_name):
    """Print a formatted per-variant metrics table."""
    print(f"\n{'='*70}")
    print(f"{env_name} - Metrics Summary")
    print(f"{'='*70}")
    print(f"{'Variant':<20} {'RMSE':>10} {'MAE':>10} {'R2':>10} {'ME':>10}")
    print(f"{'-'*70}")
    for variant, metrics in metrics_dict.items():
        print(f"{variant:<20} {metrics['RMSE']:>10.6f} {metrics['MAE']:>10.6f} "
              f"{metrics['R2']:>10.6f} {metrics['ME']:>10.6f}")
    print(f"{'='*70}")


# ============================================================================
# Ensemble reconstruction
# ============================================================================

def load_ensemble(checkpoint_path, se_net_path, slsp_net_path, device='cpu'):
    """
    Reconstruct the MET-Net ensemble from a saved checkpoint.

    Each member is a fresh METNet() with the Stage-1 SE-Net/SLSP-Net subnets and
    scalers loaded via load_pretrained(); the fine-tuned state is then applied.
    load_state_dict uses strict=False for robustness to optional PDPM
    monitor_calibrator.* entries in the saved checkpoints.
    Returns (models, drifts_w, drifts_b).
    """
    print(f"\nLoading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)

    if 'models_state_dicts' not in checkpoint:
        raise ValueError("Checkpoint does not contain 'models_state_dicts' key")

    models_state_dicts = checkpoint['models_state_dicts']
    n_models = len(models_state_dicts)
    expected = checkpoint.get('n_models', n_models)
    if n_models == 0 or n_models != expected:
        raise ValueError(f"Ensemble size mismatch: found {n_models} models, expected {expected}.")
    print(f"  [OK] Found {n_models} models in ensemble")

    print(f"\nReconstructing ensemble models...")
    models = []
    drifts_w, drifts_b = [], []
    for i, state_dict in enumerate(models_state_dicts):
        model = METNet()
        model.load_pretrained(se_path=se_net_path, slsp_path=slsp_net_path)
        # Snapshot the pretrained last layer before loading the fine-tuned weights,
        # so the SLSP head drift (adapted minus pretrained) can be computed.
        model.slsp_net.store_initial_last_layer()
        model.load_state_dict(state_dict, strict=False)
        model.eval()
        models.append(model)

        # SLSP head drift norms (adapted minus pretrained last layer)
        W0 = model.slsp_net.W0
        b0 = model.slsp_net.b0
        Wstar = model.slsp_net.last_layer.weight
        bstar = model.slsp_net.last_layer.bias
        drifts_w.append(torch.norm(Wstar - W0).item())
        drifts_b.append(torch.norm(bstar - b0).item())

        if i == 0:
            print(f"  [OK] Model architecture loaded with pretrained components")

    print(f"\n[SLSP Head Drift Norms]")
    print(f"  Mean: ||DeltaW||_F={np.mean(drifts_w):.4e}, ||Deltab||_2={np.mean(drifts_b):.4e}")
    print(f"  Median: ||DeltaW||_F={np.median(drifts_w):.4e}, ||Deltab||_2={np.median(drifts_b):.4e}")
    print(f"  Range: W=[{np.min(drifts_w):.4e}, {np.max(drifts_w):.4e}], "
          f"b=[{np.min(drifts_b):.4e}, {np.max(drifts_b):.4e}]")

    return models, checkpoint


# ============================================================================
# Held-out Full accuracy (Table-3 protocol)
# ============================================================================

def full_accuracy_on(models, X, y, device):
    """Per-model Full accuracy on a given subset: each of the 25 models predicts on
    (X, y); the reported value is the median RMSE / R2 over the ensemble."""
    Xt = torch.tensor(X, dtype=torch.float32, device=device)
    rmses, r2s = [], []
    for m in models:
        with torch.no_grad():
            p = m(Xt)[0]
            p = p.squeeze(-1).cpu().numpy() if p.dim() > 1 else p.cpu().numpy()
        rmses.append(float(np.sqrt(mean_squared_error(y, p))))
        r2s.append(float(r2_score(y, p)))
    return float(np.median(rmses)), float(np.median(r2s))


def holdout_full_accuracy(models, X_oe, y_oe, X_pe, y_pe, device,
                          test_size=0.15, random_state=42):
    """Full-model accuracy on the held-out test split (Table-3 protocol), evaluated
    per-environment and reported as the median over the 25-model ensemble:

      * OE: the fixed held-out test set (consistent with manuscript Table 3).
      * PE: the perturbed counterparts of those held-out test samples (the paired
            subset present in PE).

    Returns ((rmse_oe, r2_oe), (rmse_pe, r2_pe)). The OE/PE robustness comparison
    across the other inference-time variants is computed on the full paired sets.
    """
    from sklearn.model_selection import train_test_split
    idx = np.arange(len(X_oe))
    _, test_idx = train_test_split(idx, test_size=test_size, random_state=random_state)
    X_oe_ho, y_oe_ho = X_oe[test_idx], y_oe[test_idx]
    # pair the held-out OE test rows to PE by matching the input features
    pe_match = []
    for a in X_oe_ho:
        d = np.abs(X_pe - a).sum(axis=1)
        j = int(d.argmin())
        if d[j] < 1e-6:
            pe_match.append(j)
    pe_match = np.array(pe_match, dtype=int)
    oe = full_accuracy_on(models, X_oe_ho, y_oe_ho, device)
    pe = full_accuracy_on(models, X_pe[pe_match], y_pe[pe_match], device)
    return oe, pe


# ============================================================================
# Per-environment Evaluation
# ============================================================================

def evaluate_environment(models, checkpoint, data_xlsx, env_name, out_csv, device='cpu',
                         holdout_full=None):
    """Evaluate the MET-Net ensemble variants on one environment dataset."""

    print(f"\n{'='*70}")
    print(f"[{env_name}] Loading Dataset")
    print(f"{'='*70}")

    X, y = load_dataset(data_xlsx, env_name)

    # Mechanics residual (r*) for SE-Net invariance
    k_f = checkpoint.get('physics_kf', 0.1) if isinstance(checkpoint, dict) else 0.1
    r_star, gamma_star, viol = compute_rstar_with_model(models[0], X, device=device, k_f=k_f)
    r_median = float(np.median(r_star))
    r_mean = float(np.mean(r_star))
    r_p25 = float(np.percentile(r_star, 25))
    r_p75 = float(np.percentile(r_star, 75))
    viol_rate = float(np.mean(viol))

    print(f"\n[Mechanics Invariance - r* from SE-Net]")
    print(f"  Median: {r_median:.4e}, Mean: {r_mean:.4e}")
    print(f"  IQR: [{r_p25:.4e}, {r_p75:.4e}]")
    print(f"  Violation rate: {viol_rate:.3f} (threshold: k_f={k_f})")
    print(f"  (r* is independent of calibrator / SLSP toggles)")

    # Generate predictions
    print(f"\n{'='*70}")
    print(f"[{env_name}] Generating Predictions")
    print(f"{'='*70}")
    print(f"\nGenerating six prediction variants (including SE-AdaptiveOFF@test)...")
    y_full, y_nocal, y_fh, y_fh_nocal, y_seoff_full, y_seoff_nocal, a, b = \
        predict_variants_with_seoff(models, X, device)

    print(f"  [OK] Generated predictions for {len(y)} samples")
    print(f"  [OK] Calibrator parameters: a={a:.6f}, b={b:.6f}")

    # Per-variant metrics
    print(f"\nComputing metrics for all variants...")
    metrics = {
        'Full': compute_metrics(y, y_full),
        'NoCal@test': compute_metrics(y, y_nocal),
        'FrozenHead@test': compute_metrics(y, y_fh),
        'FH_NoCal@test': compute_metrics(y, y_fh_nocal),
        'SE-AdaptiveOFF@test': compute_metrics(y, y_seoff_full),
        'SEoff_nocal': compute_metrics(y, y_seoff_nocal)
    }

    # The 'Full' accuracy (RMSE, R2) is reported on the held-out test set
    # (Table-3 protocol) so it is consistent with the manuscript Table-4 'Full' row.
    if holdout_full is not None:
        ho_rmse, ho_r2 = holdout_full
        metrics['Full']['RMSE'] = ho_rmse
        metrics['Full']['R2'] = ho_r2
        print(f"\n[Full accuracy from the held-out test set (Table-3 protocol): "
              f"RMSE={ho_rmse:.4f}, R2={ho_r2:.4f}]")

    # SE-AdaptiveOFF effect sizes
    print(f"\nComputing SE-Net correction effects...")
    se_effects = compute_se_effects(y, y_full, y_seoff_full, y_nocal, y_seoff_nocal)

    # r* comparison (correction ON vs pure theory)
    print(f"\nComputing mechanics residual comparison (r*)...")
    rstar_comparison = compute_rstar_comparison(models[0], X, device=device, k_f=k_f)

    # Tables
    print_metrics_table(metrics, env_name)

    print(f"\n{'='*70}")
    print(f"[{env_name}] SE-Net Correction Effects (Full vs SE-AdaptiveOFF)")
    print(f"{'='*70}")
    print(f"{'Metric':<25} {'With SE':>15} {'Without SE':>15} {'Delta':>15}")
    print(f"{'-'*70}")
    print(f"{'RMSE':<25} {se_effects['RMSE_Full']:>15.6f} {se_effects['RMSE_SEoff']:>15.6f} "
          f"{se_effects['Delta_SE_RMSE']:>15.6f}")
    print(f"{'RMSE % Change':<25} {'':<15} {'':<15} {se_effects['Delta_SE_RMSE_pct']:>15.2f}%")
    print(f"{'ME (Bias)':<25} {se_effects['ME_Full']:>15.6f} {se_effects['ME_SEoff']:>15.6f} "
          f"{se_effects['Delta_SE_ME']:>15.6f}")
    print(f"{'MAE':<25} {se_effects['MAE_Full']:>15.6f} {se_effects['MAE_SEoff']:>15.6f} "
          f"{se_effects['Delta_SE_MAE']:>15.6f}")
    print(f"{'R^2':<25} {se_effects['R2_Full']:>15.6f} {se_effects['R2_SEoff']:>15.6f} "
          f"{se_effects['Delta_SE_R2']:>15.6f}")
    print(f"{'-'*70}")
    print(f"{'Raw Bias (no calibrator)':<25} {se_effects['ME_NoCal']:>15.6f} "
          f"{se_effects['ME_SEoff_NoCal']:>15.6f} {se_effects['Delta_SE_Bias_Raw']:>15.6f}")
    print(f"{'='*70}")

    print(f"\n{'='*70}")
    print(f"[{env_name}] Mechanics Residual (r*) Comparison")
    print(f"{'='*70}")
    print(f"{'Metric':<25} {'With SE Correction':>20} {'Pure Theory':>20} {'Improvement':>15}")
    print(f"{'-'*70}")
    print(f"{'Median r*':<25} {rstar_comparison['median_SE']:>20.6f} "
          f"{rstar_comparison['median_Theory']:>20.6f} {-rstar_comparison['Delta_median']:>15.6f}")
    print(f"{'Mean r*':<25} {rstar_comparison['mean_SE']:>20.6f} "
          f"{rstar_comparison['mean_Theory']:>20.6f} {-rstar_comparison['Delta_mean']:>15.6f}")
    print(f"{'Violation Rate':<25} {rstar_comparison['viol_rate_SE']:>20.3f} "
          f"{rstar_comparison['viol_rate_Theory']:>20.3f} {-rstar_comparison['Delta_viol_rate']:>15.3f}")
    print(f"{'='*70}")
    print(f"Note: Negative Delta means SE correction improves (reduces) the residual")
    print(f"Note: r* is independent of calibrator / SLSP toggles")

    # Save predictions CSV
    print(f"\nSaving results...")
    save_predictions_csv(y, y_full, y_nocal, y_fh, y_fh_nocal,
                         y_seoff_full, y_seoff_nocal, a, b, out_csv)

    return metrics, se_effects, rstar_comparison


# ============================================================================
# Main Entry Point
# ============================================================================

def main():
    """Run the paired zero-shot OE/PE evaluation (paper Tables 4-5)."""
    import argparse
    ap = argparse.ArgumentParser(description="Paired zero-shot environment-perturbation evaluation (Tables 4-5).")
    ap.add_argument("--ensemble", default=CONFIG['checkpoint'], help="25-model ensemble .pkl")
    ap.add_argument("--se-ckpt", default=CONFIG['se_net_path'])
    ap.add_argument("--slsp-ckpt", default=CONFIG['slsp_net_path'])
    ap.add_argument("--oe", default=CONFIG['oe_xlsx'], help="Original-environment dataset")
    ap.add_argument("--pe", default=CONFIG['pe_xlsx'], help="Perturbed-environment dataset")
    ap.add_argument("--out-dir", default=CONFIG['output_dir'])
    ap.add_argument("--device", default=CONFIG['device'])
    args = ap.parse_args()
    CONFIG['checkpoint'] = args.ensemble
    CONFIG['se_net_path'] = args.se_ckpt
    CONFIG['slsp_net_path'] = args.slsp_ckpt
    CONFIG['oe_xlsx'] = args.oe
    CONFIG['pe_xlsx'] = args.pe
    CONFIG['output_dir'] = args.out_dir
    CONFIG['output_oe_csv'] = os.path.join(args.out_dir, 'preds_OE_variants.csv')
    CONFIG['output_pe_csv'] = os.path.join(args.out_dir, 'preds_PE_variants.csv')
    CONFIG['device'] = args.device

    print("\n" + "="*70)
    print("MET-Net Environment-Perturbation Evaluation (EPE)")
    print("Paired zero-shot OE/PE evaluation with SE-AdaptiveOFF analysis")
    print("Variants: Full, NoCal@test, FrozenHead@test, FH_NoCal@test,")
    print("          SE-AdaptiveOFF@test, SEoff_nocal")
    print("="*70)

    # Check required files BEFORE creating any output directory.
    print(f"\nChecking required files...")
    required_files = [
        CONFIG['checkpoint'],
        CONFIG['se_net_path'],
        CONFIG['slsp_net_path'],
        CONFIG['oe_xlsx'],
        CONFIG['pe_xlsx']
    ]
    for file in required_files:
        if not os.path.exists(file):
            raise FileNotFoundError(f"Required file not found: {file}")
        print(f"  [OK] {file}")

    # Create output directory only after all inputs are present.
    if not os.path.exists(CONFIG['output_dir']):
        os.makedirs(CONFIG['output_dir'])
        print(f"\n[OK] Created output directory: {CONFIG['output_dir']}/")

    # Setup device
    device = torch.device(CONFIG['device'] if torch.cuda.is_available() else 'cpu')
    if CONFIG['device'] == 'cuda' and not torch.cuda.is_available():
        print("\n[WARNING] CUDA not available, using CPU")
        device = torch.device('cpu')
    print(f"\nDevice: {device}")

    # Reconstruct the ensemble once and reuse for both environments
    models, checkpoint = load_ensemble(
        checkpoint_path=CONFIG['checkpoint'],
        se_net_path=CONFIG['se_net_path'],
        slsp_net_path=CONFIG['slsp_net_path'],
        device=device
    )

    # Held-out Full accuracy (Table-3 protocol) for the OE and PE 'Full' rows.
    X_oe, y_oe = load_dataset(CONFIG['oe_xlsx'], 'OE')
    X_pe, y_pe = load_dataset(CONFIG['pe_xlsx'], 'PE')
    ho_full_oe, ho_full_pe = holdout_full_accuracy(models, X_oe, y_oe, X_pe, y_pe, device)

    # Evaluate Original Environment (OE)
    print("\n" + "="*70)
    print("[1/2] EVALUATING ORIGINAL ENVIRONMENT (OE)")
    print("="*70)
    metrics_oe, se_effects_oe, rstar_oe = evaluate_environment(
        models, checkpoint,
        data_xlsx=CONFIG['oe_xlsx'],
        env_name='OE',
        out_csv=CONFIG['output_oe_csv'],
        device=device,
        holdout_full=ho_full_oe
    )

    # Evaluate Perturbed Environment (PE)
    print("\n" + "="*70)
    print("[2/2] EVALUATING PERTURBED ENVIRONMENT (PE)")
    print("="*70)
    metrics_pe, se_effects_pe, rstar_pe = evaluate_environment(
        models, checkpoint,
        data_xlsx=CONFIG['pe_xlsx'],
        env_name='PE',
        out_csv=CONFIG['output_pe_csv'],
        device=device,
        holdout_full=ho_full_pe
    )

    # Final Summary
    print("\n" + "="*70)
    print("ALL EVALUATIONS COMPLETED SUCCESSFULLY")
    print("="*70)
    print(f"\nGenerated files:")
    print(f"  [OK] {CONFIG['output_oe_csv']}")
    print(f"  [OK] {CONFIG['output_pe_csv']}")

    print(f"\nQuick Comparison (RMSE):")
    print(f"{'':22} {'OE (RMSE)':>18} {'PE (RMSE)':>18}")
    print(f"{'-'*70}")
    for variant in ['Full', 'NoCal@test', 'FrozenHead@test', 'FH_NoCal@test',
                    'SE-AdaptiveOFF@test', 'SEoff_nocal']:
        rmse_oe = metrics_oe[variant]['RMSE']
        rmse_pe = metrics_pe[variant]['RMSE']
        print(f"{variant:22} {rmse_oe:>18.6f} {rmse_pe:>18.6f}")

    # SE-Net correction effects across environments (difference-in-differences)
    print(f"\n{'='*70}")
    print(f"SE-Net Correction Effects - Cross-Environment Comparison")
    print(f"{'='*70}")
    print(f"{'Metric':<30} {'OE':>18} {'PE':>18} {'DiD':>15}")
    print(f"{'-'*70}")
    print(f"{'Delta_SE_RMSE':<30} {se_effects_oe['Delta_SE_RMSE']:>18.6f} "
          f"{se_effects_pe['Delta_SE_RMSE']:>18.6f} "
          f"{se_effects_pe['Delta_SE_RMSE'] - se_effects_oe['Delta_SE_RMSE']:>15.6f}")
    print(f"{'Delta_SE_RMSE (%)':<30} {se_effects_oe['Delta_SE_RMSE_pct']:>18.2f}% "
          f"{se_effects_pe['Delta_SE_RMSE_pct']:>17.2f}% {'':<15}")
    print(f"{'Delta_SE_ME':<30} {se_effects_oe['Delta_SE_ME']:>18.6f} "
          f"{se_effects_pe['Delta_SE_ME']:>18.6f} "
          f"{se_effects_pe['Delta_SE_ME'] - se_effects_oe['Delta_SE_ME']:>15.6f}")
    print(f"{'Delta_r* (median)':<30} {-rstar_oe['Delta_median']:>18.6f} "
          f"{-rstar_pe['Delta_median']:>18.6f} "
          f"{-rstar_pe['Delta_median'] - (-rstar_oe['Delta_median']):>15.6f}")
    print(f"{'='*70}")
    print(f"\nInterpretation:")
    print(f"  - Delta_SE_RMSE > 0: SE correction improves accuracy")
    print(f"  - DiD near 0: SE correction is environment-invariant")
    print(f"  - Delta_r* < 0: SE correction improves physics consistency")
    print("="*70 + "\n")


if __name__ == '__main__':
    main()
