"""
ALE curves and Error Regime Maps for the SE-Net theory corrections (MET-Net)
============================================================================

This script characterizes WHEN and WHERE the equivalent-section theory fails by
probing the two physically meaningful SE-Net correction parameters:

  * g_lambda - stiffness correction (equivalent modulus ratio)
  * g_t      - thickness correction (equivalent wall thickness)

It produces the two diagnostics used in the paper (Figs. 10-12):

  Section A - ALE curves
      Accumulated Local Effects (ALE) of each engineered input feature on
      g_lambda and g_t, computed with 40 quantile bins. g_lambda and g_t are
      overlaid on the same axes for direct comparison.
      Interpretation of the correction magnitude g:
        g > 0  : theory underestimates (needs a positive correction)
        g < 0  : theory overestimates (needs a negative correction)
        |g| ~ 0: theory is accurate

  Section B - Error Regime Maps
      A 100x100 grid over the thickness ratio r and the tube diameter D, with
      the wall thickness t fixed at its 10th / 50th / 90th data percentile,
      producing a 2x3 grid of heatmaps (row 1: g_lambda, row 2: g_t).

The diagnostics use a single deterministic model: Fold 0, Seed 0 of the trained
MET-Net ensemble. The default input data is the original-environment (OE)
bilayer target dataset bilayer_target_OE.xlsx.
"""

import os
import sys
import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from sklearn.model_selection import KFold, train_test_split

# Make the metnet package importable from ../src
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
from metnet.met_net import METNet, SENet  # noqa: E402

# ============================================================================
# Configuration
# ============================================================================

SEED = 0
FOLD_IDX = 0
ALE_BINS = 40  # Number of quantile intervals for the ALE calculation
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(SCRIPT_DIR)
DATA_DIR = os.path.join(ROOT_DIR, 'data')
MODELS_DIR = os.path.join(ROOT_DIR, 'models')

FILES = {
    'data': os.path.join(DATA_DIR, 'bilayer_target_OE.xlsx'),
    'model': os.path.join(MODELS_DIR, 'met_net_fold0_seed0.pth'),
}

OUTPUT_DIR = os.path.join(ROOT_DIR, "figures")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Engineered SE-Net input features (matches SENet.forward feature order)
FEATURE_NAMES_ENGINEERED = [
    r'$D$', r'$t$', r'$r$',
    r'$t/D$', r'$t_1/D$', r'$t_2/D$'
]
FEATURE_NAMES_PLAIN = ['D', 't', 'r', 't/D', 't1/D', 't2/D']

# ----------------------------------------------------------------------------
# Plot layout
# ----------------------------------------------------------------------------
ALE_LAYOUT = {
    'fig_size': (18, 11),
    'font_family': 'Times New Roman',
    'font_size_label': 16,
    'font_size_tick': 14,
    'font_size_legend': 12,
    'color_g_lambda': '#1f77b4',  # Blue
    'color_g_t': '#ff7f0e',       # Orange
    'linewidth': 2.5,
    'marker_size': 20,
}

REGIME_LAYOUT = {
    'fig_size': (20, 14),
    'font_family': 'Times New Roman',
    'font_size_label': 14,
    'font_size_tick': 12,
    'font_size_title': 14,
    'font_size_colorbar': 12,
    'font_size_colorbar_label': 16,
    'font_size_panel': 18,
    'row_label_x': 0.03,
    'grid_resolution': 100,
    'contour_levels': [-2.0, -1.0, -0.5, 0, 0.5, 1.0, 2.0],
    'show_data_points': True,
    'data_point_color': 'black',
    'data_point_size': 10,
    'data_point_alpha': 0.3,
}

plt.rcParams['font.family'] = ALE_LAYOUT['font_family']
plt.rcParams['mathtext.fontset'] = 'stix'


# ============================================================================
# Model and data loading (shared)
# ============================================================================

def load_met_net_fold0_seed0():
    """Load the paper-selected MET-Net Fold0-Seed0 model (self-sufficient checkpoint
    bundling the model weights and the SLSP-Net scalers)."""
    print("Loading MET-Net Fold0-Seed0...")
    return METNet.load_paper_model(FILES['model'], device=DEVICE)


def compute_engineered_features(X_raw, theory_layer, device):
    """Compute the 6 engineered SE-Net input features from raw (D, t, r)."""
    D = torch.tensor(X_raw[:, 0], dtype=torch.float32, device=device)
    t = torch.tensor(X_raw[:, 1], dtype=torch.float32, device=device)
    r = torch.tensor(X_raw[:, 2], dtype=torch.float32, device=device)

    with torch.no_grad():
        th = theory_layer(D, t, r)

    eps = 1e-12
    features = np.column_stack([
        D.cpu().numpy(),
        t.cpu().numpy(),
        r.cpu().numpy(),
        (t / torch.clamp(D, min=eps)).cpu().numpy(),
        (th["t1"] / torch.clamp(D, min=eps)).cpu().numpy(),
        (th["t2"] / torch.clamp(D, min=eps)).cpu().numpy(),
    ])
    return features


def load_raw_data():
    """Load raw (D, t, r) inputs from the bilayer target (OE) dataset."""
    df = pd.read_excel(FILES['data'])
    X_raw_all = df.iloc[:, :3].values  # D, t, r
    return X_raw_all


def load_engineered_features(theory_layer):
    """Load the full set of engineered SE-Net features for the ALE analysis."""
    X_raw_all = load_raw_data()

    # Reproduce the same CV split bookkeeping as training (Fold0-Seed0).
    # The ALE curves are evaluated over the full feature distribution.
    df = pd.read_excel(FILES['data'])
    y_all = df.iloc[:, 14].values
    X_raw_cv, _, _, _ = train_test_split(X_raw_all, y_all, test_size=0.15, random_state=42)
    kf = KFold(n_splits=5, shuffle=True, random_state=42 + SEED)
    for fold_idx, (train_idx, _) in enumerate(kf.split(X_raw_cv)):
        if fold_idx == FOLD_IDX:
            break

    print("Computing engineered features...")
    X_all_eng = compute_engineered_features(X_raw_all, theory_layer, DEVICE)
    return X_all_eng


# ============================================================================
# SECTION A - ALE curves
# ============================================================================

def compute_ale(head, X, feature_idx, bins=40, output_target='g_lambda'):
    """Compute Accumulated Local Effects (ALE) for a single feature.

    Args:
        head: the SE-Net CorrectionHead (returns (g_lambda, g_t))
        X: engineered feature matrix (numpy array)
        feature_idx: index of the feature to analyze
        bins: number of quantile intervals
        output_target: 'g_lambda' (index 0) or 'g_t' (index 1)

    Returns:
        bin_edges: quantile bin edges (x-axis)
        ale_y_centered: accumulated, centered effect values (y-axis)
        feature_vals: original feature values (for the rug plot)
    """
    # 1. Define intervals (quantiles)
    feature_vals = X[:, feature_idx]
    quantiles = np.linspace(0, 1, bins + 1)
    bin_edges = np.unique(np.quantile(feature_vals, quantiles))

    if len(bin_edges) < bins:
        bin_edges = np.sort(np.unique(feature_vals))

    # 2. Compute local effects
    ale_effects = []
    samples_count = []

    target_out_idx = 0 if output_target == 'g_lambda' else 1

    for i in range(len(bin_edges) - 1):
        lower, upper = bin_edges[i], bin_edges[i + 1]

        mask = (feature_vals >= lower) & (feature_vals <= upper)
        X_bin = X[mask].copy()

        if len(X_bin) == 0:
            continue

        X_lower = X_bin.copy()
        X_lower[:, feature_idx] = lower

        X_upper = X_bin.copy()
        X_upper[:, feature_idx] = upper

        with torch.no_grad():
            t_lower = torch.tensor(X_lower, dtype=torch.float32).to(DEVICE)
            t_upper = torch.tensor(X_upper, dtype=torch.float32).to(DEVICE)

            pred_lower = head(t_lower)
            pred_upper = head(t_upper)

            if isinstance(pred_lower, tuple):
                pred_lower = torch.stack(pred_lower, dim=1)
            if isinstance(pred_upper, tuple):
                pred_upper = torch.stack(pred_upper, dim=1)

            diff = pred_upper[:, target_out_idx] - pred_lower[:, target_out_idx]
            avg_diff = torch.mean(diff).item()

        ale_effects.append(avg_diff)
        samples_count.append(len(X_bin))

    # 3. Accumulate and center
    ale_y = np.cumsum([0] + ale_effects)
    weights = np.array([0] + samples_count)
    ale_y_centered = ale_y - np.average(ale_y, weights=weights) if np.sum(weights) > 0 else ale_y

    return bin_edges, ale_y_centered, feature_vals


def plot_ale_combined(head, X_data, save_path):
    """Create one 2x3 figure; each subplot overlays g_lambda and g_t ALE curves."""
    print("Generating combined ALE plot (g_lambda and g_t overlaid)...")

    fig, axes = plt.subplots(2, 3, figsize=ALE_LAYOUT['fig_size'])
    axes = axes.flatten()

    ale_summary = {'g_lambda': {}, 'g_t': {}}

    for i, ax in enumerate(axes):
        feature_name = FEATURE_NAMES_ENGINEERED[i]

        ale_x_lambda, ale_y_lambda, raw_x = compute_ale(
            head, X_data, i, bins=ALE_BINS, output_target='g_lambda'
        )
        ale_x_t, ale_y_t, _ = compute_ale(
            head, X_data, i, bins=ALE_BINS, output_target='g_t'
        )

        ale_summary['g_lambda'][i] = {
            'min': ale_y_lambda.min(),
            'max': ale_y_lambda.max(),
            'range': ale_y_lambda.max() - ale_y_lambda.min()
        }
        ale_summary['g_t'][i] = {
            'min': ale_y_t.min(),
            'max': ale_y_t.max(),
            'range': ale_y_t.max() - ale_y_t.min()
        }

        # Rug plot (data density) at the bottom of the panel
        y_min = min(ale_y_lambda.min(), ale_y_t.min())
        ax.plot(raw_x, np.full_like(raw_x, y_min - 0.05 * abs(y_min)),
                '|', color='gray', alpha=0.3, markersize=4)

        # g_lambda ALE curve
        f_interp_lambda = interp1d(ale_x_lambda, ale_y_lambda, kind='linear',
                                   fill_value="extrapolate")
        x_smooth = np.linspace(ale_x_lambda.min(), ale_x_lambda.max(), 200)
        y_smooth_lambda = f_interp_lambda(x_smooth)

        ax.plot(x_smooth, y_smooth_lambda,
                color=ALE_LAYOUT['color_g_lambda'],
                linewidth=ALE_LAYOUT['linewidth'],
                label=r'$g_\lambda$')
        ax.scatter(ale_x_lambda, ale_y_lambda,
                   s=ALE_LAYOUT['marker_size'],
                   color=ALE_LAYOUT['color_g_lambda'],
                   alpha=0.6)

        # g_t ALE curve
        f_interp_t = interp1d(ale_x_t, ale_y_t, kind='linear',
                              fill_value="extrapolate")
        y_smooth_t = f_interp_t(x_smooth)

        ax.plot(x_smooth, y_smooth_t,
                color=ALE_LAYOUT['color_g_t'],
                linewidth=ALE_LAYOUT['linewidth'],
                label=r'$g_t$')
        ax.scatter(ale_x_t, ale_y_t,
                   s=ALE_LAYOUT['marker_size'],
                   color=ALE_LAYOUT['color_g_t'],
                   alpha=0.6)

        # Reference line at y=0 (perfect theory)
        ax.axhline(0, color='black', linestyle='-', linewidth=0.8, alpha=0.5)

        ax.set_xlabel(feature_name, fontsize=ALE_LAYOUT['font_size_label'])
        if i % 3 == 0:
            ax.set_ylabel("Theory Error\n(Correction Magnitude)",
                          fontsize=ALE_LAYOUT['font_size_label'])

        ax.grid(True, linestyle='--', alpha=0.3)
        ax.tick_params(labelsize=ALE_LAYOUT['font_size_tick'])

        if i == 0:
            ax.legend(loc='best', fontsize=ALE_LAYOUT['font_size_legend'],
                      framealpha=0.9)

    plt.tight_layout()
    plt.savefig(save_path + ".png", dpi=300, bbox_inches='tight')
    plt.savefig(save_path + ".svg", format='svg', bbox_inches='tight')
    plt.close()
    print(f"  [OK] Saved combined ALE plot: {save_path}.png")

    return ale_summary


def print_ale_summary(ale_summary):
    """Print a summary table of the ALE effect ranges per feature."""
    print("\n" + "=" * 80)
    print("ALE SUMMARY: Theory Error Characterization")
    print("=" * 80)

    print("\nTable: ALE Effect Ranges for Each Feature")
    print("-" * 80)
    print(f"{'Feature':<12} {'g_lambda Min':>12} {'g_lambda Max':>12} {'g_lambda Range':>15} "
          f"{'g_t Min':>10} {'g_t Max':>10} {'g_t Range':>12}")
    print("-" * 80)

    for i, feat in enumerate(FEATURE_NAMES_PLAIN):
        gl = ale_summary['g_lambda'][i]
        gt = ale_summary['g_t'][i]
        print(f"{feat:<12} {gl['min']:>12.4f} {gl['max']:>12.4f} {gl['range']:>15.4f} "
              f"{gt['min']:>10.4f} {gt['max']:>10.4f} {gt['range']:>12.4f}")

    print("-" * 80)
    print("\nKey Findings:")

    gl_ranges = [(i, ale_summary['g_lambda'][i]['range']) for i in range(6)]
    gl_ranges_sorted = sorted(gl_ranges, key=lambda x: x[1], reverse=True)
    print(f"  - Dominant feature for g_lambda: {FEATURE_NAMES_PLAIN[gl_ranges_sorted[0][0]]} "
          f"(range = {gl_ranges_sorted[0][1]:.4f})")

    gt_ranges = [(i, ale_summary['g_t'][i]['range']) for i in range(6)]
    gt_ranges_sorted = sorted(gt_ranges, key=lambda x: x[1], reverse=True)
    print(f"  - Dominant feature for g_t: {FEATURE_NAMES_PLAIN[gt_ranges_sorted[0][0]]} "
          f"(range = {gt_ranges_sorted[0][1]:.4f})")

    negligible = [FEATURE_NAMES_PLAIN[i] for i in range(6)
                  if ale_summary['g_lambda'][i]['range'] < 0.1
                  and ale_summary['g_t'][i]['range'] < 0.1]
    if negligible:
        print(f"  - Negligible features (range < 0.1): {', '.join(negligible)}")


def save_ale_summary_csv(ale_summary, save_path):
    """Save the ALE summary table to CSV."""
    data = []
    for i, feat in enumerate(FEATURE_NAMES_PLAIN):
        gl = ale_summary['g_lambda'][i]
        gt = ale_summary['g_t'][i]
        data.append({
            'Feature': feat,
            'g_lambda_min': gl['min'],
            'g_lambda_max': gl['max'],
            'g_lambda_range': gl['range'],
            'g_t_min': gt['min'],
            'g_t_max': gt['max'],
            'g_t_range': gt['range'],
        })
    df = pd.DataFrame(data)
    df.to_csv(save_path + "_Summary.csv", index=False)
    print(f"  [OK] Saved ALE summary: {save_path}_Summary.csv")


def run_ale(model):
    """Section A entry point: compute and plot the ALE curves."""
    print("=" * 70)
    print("Section A - ALE curves (g_lambda and g_t)")
    print("=" * 70)

    head = model.se_net.head.to(DEVICE).eval()
    X_eng = load_engineered_features(model.se_net.theory)
    print(f"  Data shape: {X_eng.shape}")

    save_path = os.path.join(OUTPUT_DIR, f"ALE_Combined_Fold{FOLD_IDX}_Seed{SEED}")
    ale_summary = plot_ale_combined(head, X_eng, save_path)
    print_ale_summary(ale_summary)
    save_ale_summary_csv(ale_summary, save_path)


# ============================================================================
# SECTION B - Error Regime Maps
# ============================================================================

def compute_engineered_features_batch(D, t, r, theory_layer):
    """Compute the 6 engineered features for a batch of (D, t, r) tensors."""
    with torch.no_grad():
        th = theory_layer(D, t, r)

    eps = 1e-12
    features = torch.stack([
        D,
        t,
        r,
        t / torch.clamp(D, min=eps),
        th["t1"] / torch.clamp(D, min=eps),
        th["t2"] / torch.clamp(D, min=eps),
    ], dim=-1)
    return features


def predict_corrections(se_net, D_grid, t_fixed, r_grid):
    """Predict g_lambda and g_t over a (D, r) grid at fixed t."""
    D_flat = D_grid.flatten()
    r_flat = r_grid.flatten()
    t_flat = np.full_like(D_flat, t_fixed)

    D_tensor = torch.tensor(D_flat, dtype=torch.float32, device=DEVICE)
    t_tensor = torch.tensor(t_flat, dtype=torch.float32, device=DEVICE)
    r_tensor = torch.tensor(r_flat, dtype=torch.float32, device=DEVICE)

    features = compute_engineered_features_batch(D_tensor, t_tensor, r_tensor, se_net.theory)

    with torch.no_grad():
        g_lambda, g_t = se_net.head(features)

    g_lambda_grid = g_lambda.cpu().numpy().reshape(D_grid.shape)
    g_t_grid = g_t.cpu().numpy().reshape(D_grid.shape)
    return g_lambda_grid, g_t_grid


def plot_error_regime_map_grid(se_net, X_raw, save_path):
    """Create a 2x3 grid of Error Regime Maps.

    Columns: t fixed at the 10th / 50th / 90th percentile.
    Row 1: g_lambda. Row 2: g_t.
    """
    print("Generating Error Regime Map (2x3 grid)...")

    D_data = X_raw[:, 0]
    t_data = X_raw[:, 1]
    r_data = X_raw[:, 2]

    D_min, D_max = D_data.min(), D_data.max()
    r_min, r_max = r_data.min(), r_data.max()

    t_slices = {
        'Thin': np.percentile(t_data, 10),
        'Median': np.percentile(t_data, 50),
        'Thick': np.percentile(t_data, 90),
    }

    print("  Parameter ranges:")
    print(f"    D: [{D_min:.2f}, {D_max:.2f}]")
    print(f"    r: [{r_min:.2f}, {r_max:.2f}]")
    print(f"    t slices: Thin={t_slices['Thin']:.2f}, "
          f"Median={t_slices['Median']:.2f}, Thick={t_slices['Thick']:.2f}")

    resolution = REGIME_LAYOUT['grid_resolution']
    r_range = np.linspace(r_min, r_max, resolution)
    D_range = np.linspace(D_min, D_max, resolution)
    r_grid, D_grid = np.meshgrid(r_range, D_range)

    predictions = {}
    for name, t_val in t_slices.items():
        g_lambda_grid, g_t_grid = predict_corrections(se_net, D_grid, t_val, r_grid)
        predictions[name] = {'g_lambda': g_lambda_grid, 'g_t': g_t_grid}

    all_g_lambda = np.concatenate([predictions[name]['g_lambda'].flatten() for name in t_slices])
    all_g_t = np.concatenate([predictions[name]['g_t'].flatten() for name in t_slices])

    g_lambda_abs_max = max(abs(all_g_lambda.min()), abs(all_g_lambda.max()))
    g_t_abs_max = max(abs(all_g_t.min()), abs(all_g_t.max()))

    fig = plt.figure(figsize=REGIME_LAYOUT['fig_size'])
    gs = fig.add_gridspec(2, 4, width_ratios=[1, 1, 1, 0.08], wspace=0.25, hspace=0.25)

    axes = np.empty((2, 3), dtype=object)
    for row in range(2):
        for col in range(3):
            axes[row, col] = fig.add_subplot(gs[row, col])

    cax1 = fig.add_subplot(gs[0, 3])  # Colorbar for row 1
    cax2 = fig.add_subplot(gs[1, 3])  # Colorbar for row 2

    cmap = plt.cm.RdBu_r

    slice_names = ['Thin', 'Median', 'Thick']
    slice_labels = ["Thin tubes", "Median tubes", "Thick tubes"]

    # --- Row 1: g_lambda ---
    for col, (name, label) in enumerate(zip(slice_names, slice_labels)):
        ax = axes[0, col]

        im = ax.pcolormesh(r_grid, D_grid, predictions[name]['g_lambda'],
                           cmap=cmap,
                           vmin=-g_lambda_abs_max,
                           vmax=g_lambda_abs_max,
                           shading='auto')

        contour_levels = [lv for lv in REGIME_LAYOUT['contour_levels']
                          if -g_lambda_abs_max <= lv <= g_lambda_abs_max]
        if contour_levels:
            cs = ax.contour(r_grid, D_grid, predictions[name]['g_lambda'],
                            levels=contour_levels,
                            colors='black',
                            linewidths=0.8,
                            alpha=0.7)
            ax.clabel(cs, inline=True, fontsize=9, fmt='%.1f')

        if REGIME_LAYOUT['show_data_points']:
            ax.scatter(r_data, D_data,
                       c=REGIME_LAYOUT['data_point_color'],
                       s=REGIME_LAYOUT['data_point_size'],
                       alpha=REGIME_LAYOUT['data_point_alpha'],
                       marker='o')

        if col == 0:
            ax.set_ylabel(r'$D$ (mm)', fontsize=REGIME_LAYOUT['font_size_label'])
        ax.tick_params(labelsize=REGIME_LAYOUT['font_size_tick'])
        ax.set_title(label, fontsize=REGIME_LAYOUT['font_size_title'])

    cbar1 = fig.colorbar(im, cax=cax1)
    cbar1.set_label(r'$g_\lambda$', fontsize=REGIME_LAYOUT['font_size_colorbar_label'])
    cbar1.ax.tick_params(labelsize=REGIME_LAYOUT['font_size_colorbar'])

    # --- Row 2: g_t ---
    for col, name in enumerate(slice_names):
        ax = axes[1, col]

        im = ax.pcolormesh(r_grid, D_grid, predictions[name]['g_t'],
                           cmap=cmap,
                           vmin=-g_t_abs_max,
                           vmax=g_t_abs_max,
                           shading='auto')

        contour_levels_t = [lv for lv in REGIME_LAYOUT['contour_levels']
                            if -g_t_abs_max <= lv <= g_t_abs_max]
        if contour_levels_t:
            cs = ax.contour(r_grid, D_grid, predictions[name]['g_t'],
                            levels=contour_levels_t,
                            colors='black',
                            linewidths=0.8,
                            alpha=0.7)
            ax.clabel(cs, inline=True, fontsize=9, fmt='%.1f')

        if REGIME_LAYOUT['show_data_points']:
            ax.scatter(r_data, D_data,
                       c=REGIME_LAYOUT['data_point_color'],
                       s=REGIME_LAYOUT['data_point_size'],
                       alpha=REGIME_LAYOUT['data_point_alpha'],
                       marker='o')

        ax.set_xlabel(r'$r$', fontsize=REGIME_LAYOUT['font_size_label'])
        if col == 0:
            ax.set_ylabel(r'$D$ (mm)', fontsize=REGIME_LAYOUT['font_size_label'])
        ax.tick_params(labelsize=REGIME_LAYOUT['font_size_tick'])

    cbar2 = fig.colorbar(im, cax=cax2)
    cbar2.set_label(r'$g_t$', fontsize=REGIME_LAYOUT['font_size_colorbar_label'])
    cbar2.ax.tick_params(labelsize=REGIME_LAYOUT['font_size_colorbar'])

    row_label_x = REGIME_LAYOUT['row_label_x']
    fig.text(row_label_x, 0.72, '(a)', fontsize=REGIME_LAYOUT['font_size_panel'],
             fontweight='bold', ha='center', va='center')
    fig.text(row_label_x, 0.28, '(b)', fontsize=REGIME_LAYOUT['font_size_panel'],
             fontweight='bold', ha='center', va='center')

    plt.savefig(save_path + ".png", dpi=300, bbox_inches='tight')
    plt.savefig(save_path + ".svg", format='svg', bbox_inches='tight')
    plt.close()
    print(f"  [OK] Saved Error Regime Map grid: {save_path}.png")

    print("\n  Summary statistics:")
    print(f"    g_lambda global range: [{all_g_lambda.min():.4f}, {all_g_lambda.max():.4f}]")
    print(f"    g_t global range: [{all_g_t.min():.4f}, {all_g_t.max():.4f}]")

    return predictions


def run_regime_map(model):
    """Section B entry point: compute and plot the Error Regime Maps."""
    print("=" * 70)
    print("Section B - Error Regime Maps (r-D, t at 10/50/90 percentile)")
    print("=" * 70)

    se_net = model.se_net.to(DEVICE).eval()
    X_raw = load_raw_data()
    print(f"  Data shape: {X_raw.shape}")

    save_path = os.path.join(OUTPUT_DIR, f"Error_Regime_Map_r-D_Fold{FOLD_IDX}_Seed{SEED}")
    plot_error_regime_map_grid(se_net, X_raw, save_path)

    print("\nLayout:")
    print("  Columns: Thin (10th pct) | Median (50th pct) | Thick (90th pct)")
    print("  Row 1 (a): g_lambda (stiffness correction)")
    print("  Row 2 (b): g_t (thickness correction)")
    print("Interpretation:")
    print("  - Red: theory underestimates (g > 0)")
    print("  - Blue: theory overestimates (g < 0)")
    print("  - White: theory is accurate (g ~ 0)")


# ============================================================================
# Main
# ============================================================================

def main():
    print("=" * 70)
    print("MET-Net SE-Net theory-error diagnostics (Fold0-Seed0)")
    print("=" * 70)

    model = load_met_net_fold0_seed0()

    # Section A: ALE curves
    run_ale(model)

    # Section B: Error Regime Maps
    run_regime_map(model)

    print("\n" + "=" * 70)
    print("Diagnostics complete.")
    print("=" * 70)


if __name__ == "__main__":
    main()
