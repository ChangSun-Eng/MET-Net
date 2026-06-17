"""
GradientSHAP feature attribution for MET-Net (Fold 0 - Seed 0)
==============================================================

Reproduces the GradientSHAP analysis behind the paper feature-attribution
figures (Figs 8-9). The single representative model (Fold 0, Seed 0) is loaded
and explained with Captum's GradientShap explainer:

  * 256 background (baseline) samples drawn from the fold-0 training split,
  * 100 path samples per attribution,
  * 5 independent runs averaged for a robust estimate,
  * robust global importance via the inter-quartile (IQR) trimmed mean of the
    mean absolute SHAP value (MASV).

The script renders a publication-style composite figure: a left beeswarm/summary
panel with a shared colour bar and a right grid of the six most important
feature-dependence plots. Both a 300-DPI PNG and an SVG vector file are saved.

Inputs (defaults):
  * data       : data/bilayer_target_OE.xlsx (14 input features + springback)
  * checkpoint : met_net_fold0_seed0.pth (clean state_dict, loaded via
                 METNet.load_state_dict(strict=False))
  * Stage-1 subnets/scalers : se_net_pretrained.pth, slsp_net_pretrained.pth

The MET-Net architecture is imported from the shared core library
(metnet.met_net); no model is defined inline here.
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.stats import gaussian_kde
from sklearn.model_selection import KFold, train_test_split
from captum.attr import GradientShap

# Import the shared MET-Net core library (../src/metnet/met_net.py)
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
from metnet.met_net import METNet

# ============================================================================
# 1. Layout configuration
# ============================================================================
LAYOUT_CONFIG = {
    'fig_size': (20, 10),

    # Width ratios [Left panel, Colorbar, Right col 1, Right col 2]
    'width_ratios': [2.2, 0.05, 0.7, 0.7],

    'wspace': 0.25,
    'hspace': 0.4,

    'left_margin': 0.05,
    'bottom_margin': 0.1,
    'right_margin': 0.95,
    'top_margin': 0.95,

    'font_family': 'Times New Roman',
    'font_size_label': 12,
    'font_size_tick': 10,
    'font_size_caption': 14,
}

plt.rcParams['font.family'] = LAYOUT_CONFIG['font_family']
plt.rcParams['mathtext.fontset'] = 'stix'

# ============================================================================
# Configuration
# ============================================================================

SEED = 0
FOLD_IDX = 0
N_BACKGROUND_SAMPLES = 256
N_SHAP_SAMPLES = 100
N_RUNS = 5
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
OUTPUT_SUFFIX = "f0s0"

# Default paths (relative to this script's parent package root)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(SCRIPT_DIR)

DEFAULT_FILES = {
    'data': os.path.join(ROOT_DIR, 'data', 'bilayer_target_OE.xlsx'),
    'checkpoint': os.path.join(ROOT_DIR, 'models', 'met_net_fold0_seed0.pth'),
    'senet_weights': os.path.join(ROOT_DIR, 'models', 'se_net_pretrained.pth'),
    'slsp_weights': os.path.join(ROOT_DIR, 'models', 'slsp_net_pretrained.pth'),
}

FEATURE_NAME_MAP = {
    'Tube diameter': r'$D$',
    'Total thickness': r'$t$',
    'Thickness ratio': r'$r$',
    'Radius of bending die': r'$R_B$',
    'Friction between bending die and tube': r'$f_B$',
    'Friction between pressure die and tube': r'$f_P$',
    'Friction between wiper die and tube': r'$f_W$',
    'Gap between bending die and tube': r'$G_B$',
    'Gap between pressure die and tube': r'$G_P$',
    'Gap between wiper die and tube': r'$G_W$',
    'Different velocity of pressure die': r'$v_B^d$',
    'The initial position of the pressure die': r'$L_P$',
    'Angular velocity of bending die': r'$\omega_B$',
    'Bending angle of bending die': r'$\alpha_B$',
}

def map_feature_names(original_names):
    return [FEATURE_NAME_MAP.get(name, name) for name in original_names]

# ============================================================================
# Model wrapper
# ============================================================================

class METNetWrapper(nn.Module):
    """Expose the scalar springback prediction as a 2-D tensor for Captum."""
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        output = self.model(x)
        pred = output[0]
        if pred.dim() == 1:
            pred = pred.unsqueeze(1)
        return pred

# ============================================================================
# Data and model loading
# ============================================================================

def load_data_all_samples(files):
    df = pd.read_excel(files['data'])
    X_all = df.iloc[:, :14].values
    y_all = df.iloc[:, 14].values
    feat_orig = df.columns[:14].tolist()
    feat_latex = map_feature_names(feat_orig)

    # Reconstruct the Fold-0 training split (same protocol as training)
    X_cv, _, _, _ = train_test_split(X_all, y_all, test_size=0.15, random_state=42)
    kf = KFold(n_splits=5, shuffle=True, random_state=42 + SEED)
    X_train = None
    for fold_idx, (train_idx, _) in enumerate(kf.split(X_cv)):
        if fold_idx == FOLD_IDX:
            X_train = X_cv[train_idx]
            break
    return X_train, X_all, feat_orig, feat_latex

def load_model(files):
    model = METNet()
    model.load_pretrained(se_path=files['senet_weights'], slsp_path=files['slsp_weights'])
    state_dict = torch.load(files['checkpoint'], map_location='cpu', weights_only=False)
    if isinstance(state_dict, dict) and 'model_state_dict' in state_dict:
        state_dict = state_dict['model_state_dict']
    model.load_state_dict(state_dict, strict=False)
    model = model.to(DEVICE).eval()
    return METNetWrapper(model).to(DEVICE)

# ============================================================================
# GradientSHAP
# ============================================================================

def run_gradient_shap(model, X_train, X_explain):
    print(f"Running GradientSHAP ensemble ({N_RUNS} runs)...")
    explain = torch.tensor(X_explain, dtype=torch.float32).to(DEVICE)
    gradient_shap = GradientShap(model)
    accumulated_attributions = None

    for i in range(N_RUNS):
        n_base = min(N_BACKGROUND_SAMPLES, len(X_train))
        baseline_idx = np.random.choice(len(X_train), n_base, replace=False)
        baseline = torch.tensor(X_train[baseline_idx], dtype=torch.float32).to(DEVICE)

        attr = gradient_shap.attribute(explain, baselines=baseline, n_samples=N_SHAP_SAMPLES)
        if accumulated_attributions is None:
            accumulated_attributions = attr
        else:
            accumulated_attributions += attr

    mean_attributions = accumulated_attributions / N_RUNS
    return mean_attributions.cpu().detach().numpy()

# ============================================================================
# Plotting logic
# ============================================================================

def calculate_jitter(values):
    try:
        kde = gaussian_kde(values)
        density = kde(values)
    except Exception:
        density = np.ones_like(values)
    if density.max() > 0:
        density = density / density.max()
    return density * (np.random.random(len(values)) - 0.5) * 0.6

def get_robust_importance(shap_values):
    """Robust mean absolute SHAP value (MASV) using the IQR-trimmed mean."""
    abs_values = np.abs(shap_values)
    robust_means = []
    for i in range(abs_values.shape[1]):
        feature_vals = abs_values[:, i]
        q25, q75 = np.percentile(feature_vals, [25, 75])
        mask = (feature_vals >= q25) & (feature_vals <= q75)
        robust_means.append(np.mean(feature_vals[mask]) if np.sum(mask) > 0 else np.mean(feature_vals))
    return np.array(robust_means)

def plot_composite_style(shap_values, X_data, feature_names_latex, feature_names_orig, save_path_base):
    print(f"Generating composite visualization (style {OUTPUT_SUFFIX})...")

    robust_importance = get_robust_importance(shap_values)
    sorted_idx = np.argsort(robust_importance)
    top_indices = sorted_idx[::-1][:6]

    fig = plt.figure(figsize=LAYOUT_CONFIG['fig_size'])

    gs = gridspec.GridSpec(
        3, 4,
        width_ratios=LAYOUT_CONFIG['width_ratios'],
        wspace=LAYOUT_CONFIG['wspace'],
        hspace=LAYOUT_CONFIG['hspace']
    )

    plt.subplots_adjust(
        left=LAYOUT_CONFIG['left_margin'],
        bottom=LAYOUT_CONFIG['bottom_margin'],
        right=LAYOUT_CONFIG['right_margin'],
        top=LAYOUT_CONFIG['top_margin']
    )

    # ========================================================================
    # Left panel: beeswarm summary + global-importance bars
    # ========================================================================
    ax_left = fig.add_subplot(gs[:, 0])
    y_pos = np.arange(len(feature_names_latex))

    ax_bar = ax_left.twiny()
    ax_bar.barh(y_pos, robust_importance[sorted_idx], color='#E0E0E0', alpha=0.6, height=0.7, align='center')

    # Top axis label intentionally omitted for the publication layout
    ax_bar.set_xlabel("")

    ax_bar.tick_params(axis='x', labelsize=LAYOUT_CONFIG['font_size_tick'])
    for spine in ax_bar.spines.values():
        spine.set_visible(False)

    cmap = plt.get_cmap('RdYlBu_r')

    for i, idx in enumerate(sorted_idx):
        s_vals = shap_values[:, idx]
        f_vals = X_data[:, idx]
        if f_vals.max() - f_vals.min() > 0:
            norm_f = (f_vals - f_vals.min()) / (f_vals.max() - f_vals.min())
        else:
            norm_f = np.zeros_like(f_vals) + 0.5
        y_jitter = y_pos[i] + calculate_jitter(s_vals)
        ax_left.scatter(s_vals, y_jitter, s=15, c=norm_f, cmap=cmap, alpha=0.7, linewidth=0.1, edgecolor='grey')

    ax_left.set_yticks(y_pos)
    ax_left.set_yticklabels(np.array(feature_names_latex)[sorted_idx], fontsize=LAYOUT_CONFIG['font_size_label'])
    ax_left.set_xlabel('SHAP value (impact on model output)', fontsize=LAYOUT_CONFIG['font_size_label'])
    ax_left.axvline(x=0, color='k', linestyle='-', linewidth=0.5, alpha=0.3)
    ax_left.grid(axis='x', linestyle='--', alpha=0.3)
    ax_left.tick_params(labelsize=LAYOUT_CONFIG['font_size_tick'])

    # Colorbar
    cax = fig.add_subplot(gs[:, 1])

    # Manual nudge of the colorbar slightly to the left
    pos = cax.get_position()
    new_pos = [pos.x0 - 0.02, pos.y0, pos.width, pos.height]
    cax.set_position(new_pos)

    norm = matplotlib.colors.Normalize(vmin=0, vmax=1)
    cb = matplotlib.colorbar.ColorbarBase(cax, cmap=cmap, norm=norm, orientation='vertical')
    cb.set_label('Feature value', rotation=90, labelpad=5, fontsize=10)
    cb.set_ticks([0, 1])
    cb.set_ticklabels(['Low', 'High'])

    fig.text(0.30, 0.02, '(a)', ha='center', fontsize=LAYOUT_CONFIG['font_size_caption'], fontweight='bold')

    # ========================================================================
    # Right panel: top-6 feature-dependence plots
    # ========================================================================
    right_grid_locs = [(0, 2), (0, 3), (1, 2), (1, 3), (2, 2), (2, 3)]

    for i, feat_idx in enumerate(top_indices):
        if i >= 6:
            break
        row, col = right_grid_locs[i]
        ax_dep = fig.add_subplot(gs[row, col])

        feature_name = feature_names_latex[feat_idx]
        x_vals = X_data[:, feat_idx]
        y_vals = shap_values[:, feat_idx]

        ax_dep.scatter(x_vals, y_vals, c=x_vals, cmap=cmap, s=20, alpha=0.8, edgecolors='none')

        ax_dep.set_xlabel(feature_name, fontsize=LAYOUT_CONFIG['font_size_label'])
        ax_dep.set_ylabel('SHAP value', fontsize=10)
        ax_dep.tick_params(labelsize=8)

    fig.text(0.78, 0.02, '(b)', ha='center', fontsize=LAYOUT_CONFIG['font_size_caption'], fontweight='bold')

    # Save as PNG
    png_path = save_path_base + ".png"
    plt.savefig(png_path, dpi=300, bbox_inches='tight')
    print(f"  [OK] Saved PNG: {png_path}")

    # Save as SVG
    svg_path = save_path_base + ".svg"
    plt.savefig(svg_path, format='svg', bbox_inches='tight')
    print(f"  [OK] Saved SVG: {svg_path}")

    plt.close()

# ============================================================================
# Main
# ============================================================================

def parse_args():
    ap = argparse.ArgumentParser(
        description="GradientSHAP feature attribution for MET-Net (Fold 0 - Seed 0)."
    )
    ap.add_argument("--data", default=DEFAULT_FILES['data'],
                    help="Bilayer target (OE) dataset: 14 input features + springback.")
    ap.add_argument("--checkpoint", default=DEFAULT_FILES['checkpoint'],
                    help="Clean MET-Net Fold-0/Seed-0 state_dict (load_state_dict strict=False).")
    ap.add_argument("--se-ckpt", default=DEFAULT_FILES['senet_weights'],
                    help="Pretrained SE-Net checkpoint.")
    ap.add_argument("--slsp-ckpt", default=DEFAULT_FILES['slsp_weights'],
                    help="Pretrained SLSP-Net checkpoint (+ scalers).")
    ap.add_argument("--out-dir", default=os.path.join(ROOT_DIR, "figures"),
                    help="Output directory for the composite figure.")
    return ap.parse_args()

def main():
    args = parse_args()
    files = {
        'data': args.data,
        'checkpoint': args.checkpoint,
        'senet_weights': args.se_ckpt,
        'slsp_weights': args.slsp_ckpt,
    }
    os.makedirs(args.out_dir, exist_ok=True)

    print(f"GradientSHAP analysis - MET-Net (style {OUTPUT_SUFFIX})")
    X_train, X_all, feat_orig, feat_latex = load_data_all_samples(files)
    model = load_model(files)
    shap_values = run_gradient_shap(model, X_train, X_all)

    base_path = os.path.join(
        args.out_dir,
        f"GradientSHAP_{OUTPUT_SUFFIX}_Fold{FOLD_IDX}_Seed{SEED}_Composite"
    )

    plot_composite_style(shap_values, X_all, feat_latex, feat_orig, base_path)
    print("Done.")

if __name__ == "__main__":
    main()
