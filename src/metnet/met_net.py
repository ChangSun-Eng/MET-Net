"""
MET-Net core implementation
===========================
Mechanism-guided Explainable Transfer-learning Network (MET-Net) for bilayer tube
springback prediction. This module defines the full model and training pipeline:

  * SE-Net   - Section Equivalent Network: a non-trainable equivalent-section theory
               layer plus a compact adaptive head learning two physically meaningful
               correction parameters (g_lambda, g_t) -> equivalent diameter/thickness
               (De, te).
  * SLSP-Net - Single-Layer Springback Prediction Network (MLP), pretrained on the
               abundant single-layer source domain.
  * OutputCalibrator - lightweight affine output calibrator (alpha_cal, beta_cal).
  * PDPM     - Physics-Data Paradigm Monitor: a passive, DeepLIFT-attribution based
               training-time monitor reporting MAPD, PCAR and MCI. The monitoring
               weight does NOT weight the training loss; the physics-residual hinge is
               applied uniformly (paper Eq. 13).

Stage-1 pretrains SE-Net and SLSP-Net independently; Stage-2 fine-tunes the assembled
MET-Net on the scarce bilayer target domain. See scripts/train_met_net.py for the CLI
entry point and docs/reproducibility.md for exact settings.
"""

import os
import csv
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import KFold, train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import warnings
from copy import deepcopy
from collections import defaultdict

# Set matplotlib backend BEFORE importing pyplot
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for saving plots
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
warnings.filterwarnings('ignore')

# Try to import Captum for DeepLIFT
try:
    from captum.attr import DeepLift
    HAS_CAPTUM = True
    pass  # Captum available - DeepLIFT used for PDPM attribution
except ImportError:
    HAS_CAPTUM = False
    pass  # Captum not available - gradient fallback used for PDPM attribution

# ============================================================================
# Random Seed Setting Function
# ============================================================================

def set_seed(s: int):
    """Set all random seeds for reproducibility"""
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(s)
    # Set deterministic behavior for CUDA (may impact performance)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# ============================================================================
# Metrics Tracking System
# ============================================================================

class MetricsTracker:
    """Comprehensive metrics tracking for training analysis"""
    def __init__(self):
        self.store = defaultdict(list)
        self.epoch_store = defaultdict(list)

    def log(self, **kwargs):
        """Log batch-level metrics"""
        for k, v in kwargs.items():
            if v is not None:
                if torch.is_tensor(v):
                    v = v.item() if v.numel() == 1 else v.cpu().numpy()
                self.store[k].append(float(v) if isinstance(v, (int, float)) else v)

    def log_epoch(self, **kwargs):
        """Log epoch-level metrics"""
        epoch = kwargs.get('epoch', None)
        for k, v in kwargs.items():
            if v is None:
                continue
            if torch.is_tensor(v):
                v = v.item() if v.numel() == 1 else v.cpu().numpy()
            self.epoch_store.setdefault(k, []).append(
                float(v) if isinstance(v, (int, float)) else v
            )
        # Ensure epoch is stored
        if epoch is not None:
            self.epoch_store.setdefault('epoch', []).append(int(epoch))

    def save_npz(self, path):
        """Save metrics to npz file"""
        all_metrics = dict(self.store)
        for k, v in self.epoch_store.items():
            all_metrics[f'epoch_{k}'] = v
        np.savez(path, **all_metrics)
        print(f"Metrics saved to {path}")

# ============================================================================
# Core Components
# ============================================================================

class EarlyStopping:
    def __init__(self, patience=30, min_delta=0.0001):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0
        return self.early_stop

def cbrt(x):
    """Stable cube root that preserves sign"""
    eps = torch.finfo(x.dtype).eps
    return torch.sign(x) * torch.pow(torch.clamp(torch.abs(x), min=eps), 1.0 / 3.0)

# ============================================================================
# SE-NET Components
# ============================================================================

class TheoryLayer(nn.Module):
    def __init__(self, E1=120.0, E2=69.0):
        super().__init__()
        self.register_buffer("E1", torch.tensor(E1, dtype=torch.float32))
        self.register_buffer("E2", torch.tensor(E2, dtype=torch.float32))

    def forward(self, D, t, r):
        lam = self.E2 / self.E1
        t1 = (r / (r + 1.0)) * t
        t2 = (1.0 / (r + 1.0)) * t

        c = (t1**2 + 2.0 * lam * t1 * t2 + lam * t2**2) / (2.0 * (t1 + lam * t2))
        Re = D * 0.5 - c

        d1 = D - 2.0 * t1
        d2 = d1 - 2.0 * t2
        delta = (D**4 - d1**4) + lam * (d1**4 - d2**4)

        Re_safe = torch.clamp(Re, min=1e-12)
        p = 4.0 * (Re_safe**2)
        q = -delta / (16.0 * Re_safe)
        half_q = 0.5 * q
        disc = torch.clamp(half_q**2 + (p / 3.0)**3, min=0.0)
        sqrt_disc = torch.sqrt(disc)
        te_th = cbrt(-half_q + sqrt_disc) + cbrt(-half_q - sqrt_disc)

        De_th = 2.0 * Re + te_th

        return {"t1": t1, "t2": t2, "Re": Re, "te_th": te_th, "De_th": De_th, "delta": delta}

class CorrectionHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.inp = nn.Linear(6, 64)
        self.hid = nn.Linear(64, 64)
        self.out = nn.Linear(64, 2)
        self.act = nn.SiLU()
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(self, features):
        x = self.act(self.inp(features))
        x = self.act(self.hid(x))
        out = self.out(x)
        return out[..., 0], out[..., 1]

class SENet(nn.Module):
    def __init__(self):
        super().__init__()
        self.theory = TheoryLayer()
        self.head = CorrectionHead()

    def forward(self, D, t, r):
        th = self.theory(D, t, r)

        eps = 1e-12
        feats = torch.stack([
            D, t, r,
            t / torch.clamp(D, min=eps),
            th["t1"] / torch.clamp(D, min=eps),
            th["t2"] / torch.clamp(D, min=eps),
        ], dim=-1)

        g_lambda, g_t = self.head(feats)

        lam_eff = (self.theory.E2 / self.theory.E1) * torch.exp(g_lambda)

        t1 = th["t1"]
        t2 = th["t2"]
        c_eff = (t1**2 + 2.0 * lam_eff * t1 * t2 + lam_eff * t2**2) / (2.0 * (t1 + lam_eff * t2))
        Re_eff = D * 0.5 - c_eff

        d1 = D - 2.0 * t1
        d2 = d1 - 2.0 * t2
        delta_eff = (D**4 - d1**4) + lam_eff * (d1**4 - d2**4)

        Re_safe = torch.clamp(Re_eff, min=1e-12)
        p = 4.0 * (Re_safe**2)
        q = -delta_eff / (16.0 * Re_safe)
        half_q = 0.5 * q
        disc = torch.clamp(half_q**2 + (p / 3.0)**3, min=0.0)
        sqrt_disc = torch.sqrt(disc)
        te_th_eff = cbrt(-half_q + sqrt_disc) + cbrt(-half_q - sqrt_disc)

        te_pred = te_th_eff * torch.exp(g_t)
        De_pred = 2.0 * Re_eff + te_pred

        return {
            "De_pred": De_pred, "te_pred": te_pred,
            "Re_eff": Re_eff, "delta_eff": delta_eff,
            "g_lambda": g_lambda, "g_t": g_t
        }

# ============================================================================
# SLSP-NET with trackable last layer
# ============================================================================

class SLSPNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(13, 256)
        self.bn1 = nn.BatchNorm1d(256)
        self.act1 = nn.ReLU()
        self.drop1 = nn.Dropout(0.2)

        self.layer2 = nn.Linear(256, 128)
        self.bn2 = nn.BatchNorm1d(128)
        self.act2 = nn.ReLU()
        self.drop2 = nn.Dropout(0.2)

        self.layer3 = nn.Linear(128, 64)
        self.bn3 = nn.BatchNorm1d(64)
        self.act3 = nn.ReLU()
        self.drop3 = nn.Dropout(0.2)

        self.layer4 = nn.Linear(64, 32)
        self.act4 = nn.ReLU()

        self.last_layer = nn.Linear(32, 1)

        self.W0 = None
        self.b0 = None

    def forward(self, x):
        x = self.drop1(self.act1(self.bn1(self.layer1(x))))
        x = self.drop2(self.act2(self.bn2(self.layer2(x))))
        x = self.drop3(self.act3(self.bn3(self.layer3(x))))
        x = self.act4(self.layer4(x))
        x = self.last_layer(x)
        return x

    def store_initial_last_layer(self):
        self.W0 = self.last_layer.weight.detach().clone()
        self.b0 = self.last_layer.bias.detach().clone()

    def get_last_layer_drift(self):
        if self.W0 is None or self.b0 is None:
            return 0.0, 0.0
        w_drift = torch.norm(self.last_layer.weight - self.W0).item()
        b_drift = torch.norm(self.last_layer.bias - self.b0).item()
        return w_drift, b_drift

# ============================================================================
# Output Calibrator
# ============================================================================

class OutputCalibrator(nn.Module):
    def __init__(self, init_scale=1.0, init_shift=0.0):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor([init_scale], dtype=torch.float32))
        self.shift = nn.Parameter(torch.tensor([init_shift], dtype=torch.float32))

    def forward(self, y):
        return self.scale * y + self.shift

    def regularization_loss(self, w=0.01):
        return w * ((self.scale - 1.0)**2 + (self.shift - 0.0)**2)

    def get_params(self):
        return self.scale.item(), self.shift.item()

# ============================================================================
# Monitor calibrator (PDPM) with identity prior
# ============================================================================

class MonitorCalibrator(nn.Module):
    """PDPM monitoring-weight calibration: w = sigmoid(alpha * pi_theory + beta)."""
    def __init__(self, init_alpha=1.0, init_beta=0.0):
        super().__init__()
        self.alpha = nn.Parameter(torch.tensor([init_alpha], dtype=torch.float32))
        self.beta = nn.Parameter(torch.tensor([init_beta], dtype=torch.float32))

    def forward(self, raw_scores):
        """Map raw attribution proportion to the monitoring weight."""
        return torch.sigmoid(self.alpha * raw_scores + self.beta)

    def regularization_loss(self, weight=0.001):
        """L2 penalty to identity map (alpha=1, beta=0)"""
        return weight * ((self.alpha - 1.0)**2 + self.beta**2)

# ============================================================================
# Helper functions
# ============================================================================

def get_scaler_params(scaler):
    """Extract center and scale from any sklearn scaler type."""
    if scaler is None:
        return None, None

    if hasattr(scaler, 'center_'):
        center = scaler.center_
    elif hasattr(scaler, 'mean_'):
        center = scaler.mean_
    elif hasattr(scaler, 'data_min_'):
        center = scaler.data_min_
    else:
        center = None

    if hasattr(scaler, 'scale_'):
        scale = scaler.scale_
    elif hasattr(scaler, 'data_range_'):
        scale = scaler.data_range_
    else:
        scale = None

    return center, scale

# ============================================================================
# MET-Net: assembled model with DeepLIFT-based PDPM monitor
# ============================================================================

class METNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.se_net = SENet()
        self.slsp_net = SLSPNet()
        self.output_calibrator = OutputCalibrator()
        self.monitor_calibrator = MonitorCalibrator()

        # For physics override during evaluation
        self._override_lambda_f = None

        # Scalers
        self.scaler_slsp_X = None
        self.scaler_slsp_y = None
        self.scaler_X_center = None
        self.scaler_X_scale = None
        self.scaler_y_center = None
        self.scaler_y_scale = None

        # Store baseline
        self.deeplift_baseline = None

    def load_pretrained(self, se_path='se_net_pretrained.pth', slsp_path='slsp_net_pretrained.pth'):
        # Fail loudly if a requested Stage-1 checkpoint is missing (no silent skip).
        for _p, _name in ((se_path, 'SE-Net'), (slsp_path, 'SLSP-Net')):
            if not os.path.exists(_p):
                raise FileNotFoundError(
                    f"{_name} pretrained checkpoint not found: {_p}. Provide the Stage-1 "
                    f"checkpoint (see scripts/train_se_net.py / train_slsp_net.py) or pass the "
                    f"correct path via --se-ckpt / --slsp-ckpt.")
        # Load SE-NET
        if os.path.exists(se_path):
            ckpt = torch.load(se_path, map_location='cpu', weights_only=False)
            self.se_net.load_state_dict(ckpt.get('model_state_dict', ckpt))
            print(f"[OK] Loaded SE-NET from {se_path}")

        # Load SLSP-NET
        if os.path.exists(slsp_path):
            ckpt = torch.load(slsp_path, map_location='cpu', weights_only=False)

            if 'model_state_dict' in ckpt:
                slsp_state = ckpt['model_state_dict']
            else:
                slsp_state = ckpt

            try:
                self.slsp_net.load_state_dict(slsp_state)
            except:
                # Map from Sequential format
                new_state = {}
                layer_mapping = {
                    'network.0': 'layer1',
                    'network.1': 'bn1',
                    'network.4': 'layer2',
                    'network.5': 'bn2',
                    'network.8': 'layer3',
                    'network.9': 'bn3',
                    'network.12': 'layer4',
                    'network.14': 'last_layer'
                }

                for old_key, value in slsp_state.items():
                    mapped = False
                    for old_prefix, new_name in layer_mapping.items():
                        if old_key.startswith(old_prefix + '.'):
                            suffix = old_key[len(old_prefix):]
                            new_key = new_name + suffix
                            new_state[new_key] = value
                            mapped = True
                            break
                    if not mapped:
                        new_state[old_key] = value

                self.slsp_net.load_state_dict(new_state, strict=False)

            print(f"[OK] Loaded SLSP-NET from {slsp_path}")

            # Load scalers
            self.scaler_slsp_X = ckpt.get('scaler_X')
            self.scaler_slsp_y = ckpt.get('scaler_y')

            if self.scaler_slsp_X is not None:
                center_X, scale_X = get_scaler_params(self.scaler_slsp_X)
                self.scaler_X_center = center_X
                self.scaler_X_scale = scale_X

            if self.scaler_slsp_y is not None:
                center_y, scale_y = get_scaler_params(self.scaler_slsp_y)
                self.scaler_y_center = center_y
                self.scaler_y_scale = scale_y

    def freeze_slsp_except_last(self):
        """Freeze all SLSP parameters except last layer"""
        for param in self.slsp_net.parameters():
            param.requires_grad = False

        for param in self.slsp_net.last_layer.parameters():
            param.requires_grad = True

        self.slsp_net.store_initial_last_layer()

        total_params = sum(p.numel() for p in self.slsp_net.parameters())
        unfrozen_params = sum(p.numel() for p in self.slsp_net.last_layer.parameters())
        print(f"SLSP: Frozen {total_params - unfrozen_params}/{total_params} params")
        print(f"      Unfrozen last layer: {unfrozen_params} params")

    def forward_slsp_only(self, slsp_input):
        """Forward pass through SLSP only"""
        if self.scaler_X_center is not None and self.scaler_X_scale is not None:
            center_X = torch.as_tensor(self.scaler_X_center, dtype=torch.float32, device=slsp_input.device)
            scale_X = torch.as_tensor(self.scaler_X_scale, dtype=torch.float32, device=slsp_input.device)
            slsp_input_scaled = (slsp_input - center_X) / scale_X
        else:
            slsp_input_scaled = slsp_input

        springback_scaled = self.slsp_net(slsp_input_scaled)

        if self.scaler_y_center is not None and self.scaler_y_scale is not None:
            center_y = torch.as_tensor(self.scaler_y_center, dtype=torch.float32, device=springback_scaled.device)
            scale_y = torch.as_tensor(self.scaler_y_scale, dtype=torch.float32, device=springback_scaled.device)

            if springback_scaled.dim() == 2 and springback_scaled.shape[1] == 1:
                if center_y.ndim == 1:
                    center_y = center_y.view(1, 1)
                    scale_y = scale_y.view(1, 1)

            springback = springback_scaled * scale_y + center_y
        else:
            springback = springback_scaled

        if springback.dim() == 2:
            springback = springback.squeeze(1)

        return springback

    def forward(self, bilayer_inputs, return_monitor=False):
        """Full forward pass with enhanced metrics support"""
        D = bilayer_inputs[:, 0]
        t = bilayer_inputs[:, 1]
        r = bilayer_inputs[:, 2]

        # SE-NET forward
        se_out = self.se_net(D, t, r)
        De = se_out["De_pred"]
        te = se_out["te_pred"]
        Re_eff = se_out["Re_eff"]
        delta_eff = se_out["delta_eff"]

        # Build SLSP input
        slsp_input = torch.cat([
            De.unsqueeze(1),
            te.unsqueeze(1),
            bilayer_inputs[:, 3:14]
        ], dim=1)

        # Get SLSP prediction
        slsp_pred = self.forward_slsp_only(slsp_input)

        # Apply calibrator
        springback = self.output_calibrator(slsp_pred)

        if return_monitor:
            # Dimensionless physics residual (cubic-equilibrium consistency)
            eps = 1e-12
            Re_safe = torch.clamp(Re_eff, min=eps)

            # Dimensionless thickness ratio (consistent with theory)
            x = te / (2.0 * Re_safe)  # x = te/D = te/(2*Re)

            # Normalization factor 128 (matches the theory-layer cubic)
            delta_hat = delta_eff / (128.0 * Re_safe**4)

            # Dimensionless residual
            r_star = torch.abs(x**3 + x - delta_hat)

            # Dimensionless threshold
            gamma_star = 3.0 * torch.abs(delta_hat)  # kf=5.0

            # PDPM monitoring weight (passive; not used to weight the loss)
            baseline = getattr(self, 'deeplift_baseline', None)
            w_monitor, pi_theory = self.compute_pdpm_monitor(slsp_input, baseline=baseline)

            monitor_info = {
                'De_pred': De,
                'te_pred': te,
                'Re_eff': Re_eff,
                'delta_eff': delta_eff,
                'residual': r_star,  # Correctly normalized
                'gamma': gamma_star,  # Correctly normalized
                'w_monitor': w_monitor,
                'pi_theory': pi_theory,
                'violations': (r_star > gamma_star).float(),
                'slsp_input': slsp_input
            }
            return springback, monitor_info
        else:
            return springback, se_out

    def compute_pdpm_monitor(self, slsp_input, baseline=None):
        """Compute the DeepLIFT attribution proportion pi_theory and the calibrated PDPM monitoring weight."""
        self.eval()

        with torch.no_grad():
            if HAS_CAPTUM:
                try:
                    class SLSPWrapper(nn.Module):
                        def __init__(self, ecm_model):
                            super().__init__()
                            self.ecm_model = ecm_model

                        def forward(self, x):
                            return self.ecm_model.forward_slsp_only(x)

                    wrapper = SLSPWrapper(self)
                    dl = DeepLift(wrapper)

                    if baseline is None:
                        baseline = torch.zeros_like(slsp_input)
                    attrs = dl.attribute(slsp_input, baselines=baseline)
                except Exception:
                    slsp_input = slsp_input.clone().detach().requires_grad_(True)
                    yhat = self.forward_slsp_only(slsp_input)
                    if yhat.dim() > 1:
                        yhat = yhat.sum()
                    else:
                        yhat = yhat.sum()
                    (grad,) = torch.autograd.grad(yhat, slsp_input, retain_graph=False)
                    attrs = grad.abs()
            else:
                slsp_input = slsp_input.clone().detach().requires_grad_(True)
                yhat = self.forward_slsp_only(slsp_input)
                if yhat.dim() > 1:
                    yhat = yhat.sum()
                else:
                    yhat = yhat.sum()
                (grad,) = torch.autograd.grad(yhat, slsp_input, retain_graph=False)
                attrs = grad.abs()

            # Compute attribution proportion for [De, te]
            num = attrs[:, [0, 1]].abs().sum(dim=1)
            den = attrs.abs().sum(dim=1) + 1e-12
            pi_theory = (num / den).clamp(0, 1)

            # Calibrated monitoring weight (PDPM)
            w_monitor = self.monitor_calibrator(pi_theory)

        return w_monitor, pi_theory

    @classmethod
    def load_paper_model(cls, ckpt_path, device="cpu"):
        """Load a self-sufficient paper-selected MET-Net checkpoint.

        The checkpoint bundles ``model_state_dict`` together with the fixed
        SLSP-Net input/output scalers, so no Stage-1 files are required.
        Returns a ready-to-evaluate ``METNet`` in eval mode.
        """
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        model = cls()
        model.load_state_dict(ckpt["model_state_dict"], strict=True)
        model.scaler_X_center = ckpt.get("scaler_X_center")
        model.scaler_X_scale = ckpt.get("scaler_X_scale")
        model.scaler_y_center = ckpt.get("scaler_y_center")
        model.scaler_y_scale = ckpt.get("scaler_y_scale")
        model.to(device).eval()
        return model

# ============================================================================
# Physics-residual loss functions (dimensionless cubic-equilibrium)
# ============================================================================

def physics_residual_and_gamma_normalized(Re, te, delta, kf=3.0, eps=1e-12):
    """
    Compute the dimensionless physics residual and tolerance threshold
    Consistent with theory layer cubic: te^3 + 4*Re^2*te - delta/(16*Re) = 0
    """
    Re_safe = torch.clamp(Re, min=eps)

    # Dimensionless thickness ratio
    x = te / (2.0 * Re_safe)  # x = te/D = te/(2*Re)

    # Normalization factor 128 (matches the theory-layer cubic)
    delta_hat = delta / (128.0 * Re_safe**4)

    # Dimensionless residual
    r_star = torch.abs(x**3 + x - delta_hat)

    # Dimensionless threshold
    gamma_star = kf * torch.abs(delta_hat)

    return r_star, gamma_star

def physics_hinge_loss(r, gamma):
    """Soft hinge loss for physics constraint"""
    return F.relu(r - gamma)

def slsp_last_layer_drift(slsp_net, alpha_w=1e-3, alpha_b=1e-3):
    """Compute drift penalty for SLSP last layer"""
    if slsp_net.W0 is None or slsp_net.b0 is None:
        return torch.tensor(0.0)

    w_drift = (slsp_net.last_layer.weight - slsp_net.W0).pow(2).sum()
    b_drift = (slsp_net.last_layer.bias - slsp_net.b0).pow(2).sum()

    return alpha_w * w_drift + alpha_b * b_drift

# ============================================================================
# Compute baseline in RAW domain
# ============================================================================

def compute_fold_baseline(model, X_train, device):
    """Compute fold-specific baseline for DeepLIFT in RAW SLSP input domain"""
    with torch.no_grad():
        X = torch.tensor(X_train, dtype=torch.float32, device=device)
        D, t, r = X[:, 0], X[:, 1], X[:, 2]
        se = model.se_net(D, t, r)
        De, te = se["De_pred"], se["te_pred"]
        slsp_in = torch.cat([De.unsqueeze(1), te.unsqueeze(1), X[:, 3:14]], dim=1)
        # DO NOT SCALE HERE; wrapper will scale internally
        baseline = slsp_in.median(dim=0).values.unsqueeze(0)
    return baseline

# ============================================================================
# Update PDPM monitor calibrator (class-balanced)
# ============================================================================

def update_monitor_calibrator(model, X_val, y_val, device, lr=0.01):
    """
    Update the PDPM monitor calibrator with class-balanced BCE and an identity prior
    """
    model.eval()
    X_val_tensor = torch.tensor(X_val, dtype=torch.float32, device=device)

    with torch.no_grad():
        _, monitor_info = model(X_val_tensor, return_monitor=True)
        violations = monitor_info['violations'].detach()  # Target labels
        pi_theory = monitor_info['pi_theory'].detach()  # Raw scores

    # Optimize alpha and beta
    monitor_opt = torch.optim.Adam(model.monitor_calibrator.parameters(), lr=lr)

    for _ in range(10):  # Few steps of optimization
        monitor_opt.zero_grad()

        # Forward through calibrator
        w_monitor_pred = model.monitor_calibrator(pi_theory)

        # Class balancing
        pos = violations.sum().clamp(min=1.0)
        neg = (violations.numel() - pos).clamp(min=1.0)
        pos_weight = (neg / pos).detach()

        # BCE loss with class balancing
        bce_loss = F.binary_cross_entropy(
            w_monitor_pred, violations,
            reduction='none'
        )
        # Apply pos_weight to positive samples
        weighted_bce = torch.where(violations > 0.5, bce_loss * pos_weight, bce_loss)
        bce_loss = weighted_bce.mean()

        # Identity prior regularization (alpha~1, beta~0)
        reg_loss = model.monitor_calibrator.regularization_loss()

        loss = bce_loss + reg_loss
        loss.backward()
        monitor_opt.step()

    return loss.item()

# ============================================================================
# Calibration and Utility Analysis Functions
# ============================================================================

def calibration_bins(w_monitor, violations, n_bins=10, method='equal_width'):
    """
    Compute calibration statistics for the PDPM monitoring weight (MCI)

    Args:
        w_monitor: PDPM monitoring weights
        violations: Binary violation indicators
        n_bins: Number of bins
        method: 'equal_width' or 'equal_mass'
    """
    if method == 'equal_mass':
        # Equal-mass bins (quantile-based)
        quantiles = np.linspace(0, 1, n_bins + 1)
        bins = np.percentile(w_monitor, quantiles * 100)
        bins[0] = 0.0  # Ensure first bin starts at 0
        bins[-1] = 1.0  # Ensure last bin ends at 1
    else:
        # Equal-width bins
        bins = np.linspace(0, 1, n_bins + 1)

    bin_idx = np.digitize(w_monitor, bins) - 1
    bin_idx = np.clip(bin_idx, 0, n_bins - 1)  # Handle edge cases

    p_hat = []
    w_bar = []
    counts = []

    for i in range(n_bins):
        mask = (bin_idx == i)
        n = mask.sum()
        counts.append(n)

        if n > 0:
            p_hat.append(violations[mask].mean())
            w_bar.append(w_monitor[mask].mean())
        else:
            p_hat.append(np.nan)
            w_bar.append(np.nan)

    # ECE calculation
    valid = ~np.isnan(p_hat)
    if valid.sum() > 0:
        weights = np.array(counts)[valid] / np.sum(counts)
        mci = np.sum(weights * np.abs(np.array(p_hat)[valid] - np.array(w_bar)[valid]))
    else:
        mci = np.nan

    return np.array(p_hat), np.array(w_bar), np.array(counts), mci

def physics_gain_curve(model, X_val, y_val, device, lambda_on,
                      percentiles=[10, 20, 30, 40, 50, 60, 70, 80, 90, 100]):
    """Compute physics utility: RMSE improvement for samples ranked by the PDPM monitoring weight"""
    X_val_tensor = torch.tensor(X_val, dtype=torch.float32).to(device)

    with torch.no_grad():
        # Predictions with physics OFF
        model._override_lambda_f = 0.0
        y_phys_off, _ = model(X_val_tensor, return_monitor=False)
        y_phys_off = y_phys_off.cpu().numpy()

        # Predictions with physics ON
        model._override_lambda_f = lambda_on
        y_phys_on, monitor_info = model(X_val_tensor, return_monitor=True)
        y_phys_on = y_phys_on.cpu().numpy()
        w_monitors = monitor_info['w_monitor'].cpu().numpy()

        # Reset override
        model._override_lambda_f = None

    # Sort by monitoring weight (descending)
    order = np.argsort(-w_monitors)

    gains = []
    for pct in percentiles:
        n_samples = max(1, int(len(w_monitors) * pct / 100))
        idx = order[:n_samples]

        rmse_off = np.sqrt(np.mean((y_val[idx] - y_phys_off[idx])**2))
        rmse_on = np.sqrt(np.mean((y_val[idx] - y_phys_on[idx])**2))
        gain = rmse_off - rmse_on
        gains.append(gain)

    # Area Under Gain curve
    aug = np.trapz(gains, np.array(percentiles) / 100.0)

    return percentiles, gains, aug

# ============================================================================
# Export functions for paper plots/tables
# ============================================================================

def save_per_epoch_npz(metrics_dict, fold, seed, save_dir='.'):
    """
    Export A: Per-epoch arrays for training grid plots
    """
    epochs = metrics_dict.get('epoch', [])
    if len(epochs) == 0:
        return

    # Helper to safely get array, filling with NaN if missing
    def get_array(key, default_val=np.nan):
        data = metrics_dict.get(key, [])
        if len(data) == 0:
            return np.full(len(epochs), default_val)
        elif len(data) < len(epochs):
            # Pad with NaN
            return np.array(list(data) + [default_val] * (len(epochs) - len(data)))
        else:
            return np.array(data[:len(epochs)])

    # Create standardized export
    export_data = {
        'epoch': np.array(epochs, dtype=int),
        'val_rmse': get_array('val_rmse'),
        'lambda_f': get_array('lambda_f'),
        'mean_w_monitor': get_array('w_monitor_val_mean'),
        'violation_rate': get_array('violations_val_rate'),
        'calib_scale_a': get_array('calib_scale'),
        'calib_shift_b': get_array('calib_shift'),
        'slsp_drift_w': get_array('slsp_w_drift'),
        'slsp_drift_b': get_array('slsp_b_drift'),
        'ece_equal_mass': get_array('mci'),
        'physics_auc': get_array('physics_auc_curve')
    }

    save_path = os.path.join(save_dir, f'per_epoch_fold{fold}_seed{seed}.npz')
    np.savez(save_path, **export_data)
    print(f"  Saved per-epoch data: {save_path}")

def save_final_metrics_csv(fold, seed, best_val_rmse, test_rmse, test_r2,
                           mean_uncertainty, final_mci, final_phys_active,
                           final_calib_scale_a, final_calib_shift_b,
                           final_slsp_drift_w, final_slsp_drift_b,
                           physics_auc_final, save_dir='.'):
    """
    Export B: Final metrics per fold/seed for aggregate tables
    """
    save_path = os.path.join(save_dir, f'final_metrics_fold{fold}_seed{seed}.csv')

    with open(save_path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['fold', 'seed', 'best_val_rmse', 'test_rmse', 'test_r2', 'mean_uncertainty',
                   'final_mci', 'final_phys_active', 'final_calib_scale_a', 'final_calib_shift_b',
                   'final_slsp_drift_w', 'final_slsp_drift_b', 'physics_auc_final'])
        w.writerow([fold, seed, best_val_rmse, test_rmse, test_r2, mean_uncertainty,
                   final_mci, final_phys_active, final_calib_scale_a, final_calib_shift_b,
                   final_slsp_drift_w, final_slsp_drift_b, physics_auc_final])

    print(f"  Saved final metrics: {save_path}")

def save_reliability_bins_csv(w_monitor, violations, fold, seed, n_bins=10, save_dir='.'):
    """
    Export C: Reliability bins for calibration plots
    """
    # Use equal-mass bins as recommended
    p_hat, w_bar, counts, mci = calibration_bins(w_monitor, violations, n_bins=n_bins, method='equal_mass')

    save_path = os.path.join(save_dir, f'reliability_bins_fold{fold}_seed{seed}.csv')

    df = pd.DataFrame({
        'bin_mean_w': w_bar,
        'bin_violation_rate': p_hat,
        'bin_size': counts
    })
    df.to_csv(save_path, index=False)
    print(f"  Saved reliability bins: {save_path}")

    return mci

def save_test_predictions_csv(y_true, y_pred, fold, seed, save_dir='.'):
    """
    Export D: Test predictions for parity/residual plots
    """
    save_path = os.path.join(save_dir, f'test_predictions_fold{fold}_seed{seed}.csv')

    np.savetxt(save_path, np.c_[y_true, y_pred],
              delimiter=',', header='y_true,y_pred', comments='')

    print(f"  Saved test predictions: {save_path}")

# ============================================================================
# Complete Plotting Functions
# ============================================================================

def plot_training_curves(metrics_dict, save_path=None):
    """Create comprehensive training curves figure"""
    val_rmse = metrics_dict.get('val_rmse', [])
    if len(val_rmse) == 0:
        print("No validation data to plot")
        return None

    ep = list(range(1, len(val_rmse) + 1))

    def g(key, default=None):
        data = metrics_dict.get(key, default if default is not None else [None]*len(ep))
        if len(data) > len(ep):
            data = data[:len(ep)]
        elif len(data) < len(ep):
            data = list(data) + [None]*(len(ep) - len(data))
        return data

    fig, axes = plt.subplots(3, 3, figsize=(15, 12))

    def safe_plot(ax, x, y, *args, **kwargs):
        if y is None:
            return
        valid_pairs = [(xi, yi) for xi, yi in zip(x, y) if yi is not None]
        if valid_pairs:
            x_valid, y_valid = zip(*valid_pairs)
            ax.plot(x_valid, y_valid, *args, **kwargs)

    # Row 1
    safe_plot(axes[0,0], ep, val_rmse, 'b-', label='Val RMSE')
    axes[0,0].set_title('Validation RMSE')
    axes[0,0].set_xlabel('Epoch')
    axes[0,0].set_ylabel('RMSE')

    safe_plot(axes[0,1], ep, g('lambda_f'), 'g-', label='lambda_f')
    axes[0,1].set_title('lambda_f Schedule')
    axes[0,1].set_xlabel('Epoch')
    axes[0,1].set_ylabel('lambda_f')

    safe_plot(axes[0,2], ep, g('w_monitor_val_mean'), 'r-', label='Mean w_monitor')
    axes[0,2].set_title('Mean monitoring weight')
    axes[0,2].set_xlabel('Epoch')
    axes[0,2].set_ylabel('Weight')

    # Row 2
    safe_plot(axes[1,0], ep, g('violations_val_rate'), 'm-')
    axes[1,0].set_title('Physics Violations Rate')
    axes[1,0].set_xlabel('Epoch')
    axes[1,0].set_ylabel('Rate')

    safe_plot(axes[1,1], ep, g('calib_scale'), 'c-')
    axes[1,1].set_title('Calibrator Scale')
    axes[1,1].set_xlabel('Epoch')
    axes[1,1].set_ylabel('Scale')

    safe_plot(axes[1,2], ep, g('calib_shift'), 'y-')
    axes[1,2].set_title('Calibrator Shift')
    axes[1,2].set_xlabel('Epoch')
    axes[1,2].set_ylabel('Shift')

    # Row 3
    safe_plot(axes[2,0], ep, g('slsp_w_drift'), 'b--')
    axes[2,0].set_title('SLSP ||DeltaW||')
    axes[2,0].set_xlabel('Epoch')
    axes[2,0].set_ylabel('Drift')

    safe_plot(axes[2,1], ep, g('mci', []), 'g--')
    axes[2,1].set_title('Monitor MCI')
    axes[2,1].set_xlabel('Epoch')
    axes[2,1].set_ylabel('ECE')

    aug_data = g('physics_auc_curve', [])
    if aug_data and any(v is not None for v in aug_data):
        safe_plot(axes[2,2], ep, aug_data, 'r--')
    axes[2,2].set_title('Physics Utility AUC')
    axes[2,2].set_xlabel('Epoch')
    axes[2,2].set_ylabel('AUG')

    for ax in axes.flat:
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved training curves to {save_path}")
    plt.close(fig)
    return fig

def plot_calibration_analysis(model, X_val, y_val, device, save_path=None):
    """Create calibration and utility plots"""
    model.eval()
    X = torch.tensor(X_val, dtype=torch.float32, device=device)

    with torch.no_grad():
        y_pred, info = model(X, return_monitor=True)
        y_pred = y_pred.cpu().numpy()
        w = info['w_monitor'].cpu().numpy()
        viol = info['violations'].cpu().numpy()

    # Reliability curve (equal-mass bins)
    p_hat, w_bar, counts, mci = calibration_bins(w, viol, n_bins=10, method='equal_mass')

    # Save calibration data
    if save_path:
        calib_csv = save_path.replace('.png', '_calibration_data.csv')
        calib_df = pd.DataFrame({
            'bin_mean_w': w_bar,
            'violation_rate': p_hat,
            'count': counts
        })
        calib_df['ece_contribution'] = np.abs(np.array(p_hat) - np.array(w_bar)) * np.array(counts) / len(w)
        calib_df.to_csv(calib_csv, index=False)

    # Physics gain curve (using lambda_f=1.0 for consistency with training)
    pct, gains, aug = physics_gain_curve(model, X_val, y_val, device, lambda_on=1.0)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    # Plot 1: Calibration
    if len(w_bar) > 0:
        # Filter out NaN values
        valid_mask = ~np.isnan(w_bar) & ~np.isnan(p_hat)
        if valid_mask.sum() > 0:
            axes[0].plot(w_bar[valid_mask], p_hat[valid_mask], 'bo-', label='Empirical')
    axes[0].plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Perfect')
    axes[0].set_title(f'PDPM monitor reliability (MCI={mci:.3f})')
    axes[0].set_xlabel('Mean w_monitor per bin')
    axes[0].set_ylabel('Violation rate')
    axes[0].legend()

    # Plot 2: Utility
    axes[1].plot(pct, gains, 'g^-')
    axes[1].set_title(f'Physics Utility (AUG={aug:.3f})')
    axes[1].set_xlabel('% highest-w samples')
    axes[1].set_ylabel('RMSE gain (off->on)')
    axes[1].axhline(y=0, color='k', linestyle='--', alpha=0.3)

    for ax in axes:
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved calibration analysis to {save_path}")
    plt.close(fig)
    return fig

def create_summary_table(all_metrics, n_folds=5, n_seeds=5):
    """Create summary table of key metrics"""
    rows = []
    for k, m in enumerate(all_metrics):
        ep = m.get('epoch', [])
        vr = m.get('val_rmse', [])
        lf = m.get('lambda_f', [])
        mci = m.get('mci', [])

        if len(vr) == 0:
            continue

        best_i = int(np.argmin(vr))
        rows.append({
            'fold': (k % n_folds) + 1,
            'seed': (k // n_folds) + 1,
            'best_epoch': ep[best_i] if len(ep) == len(vr) else best_i,
            'best_val_rmse': vr[best_i],
            'lambda_f_at_best': lf[best_i] if len(lf) == len(vr) else None,
            'final_mci': mci[-1] if len(mci) > 0 else None
        })

    if not rows:
        return None

    df = pd.DataFrame(rows)
    return df.sort_values(['seed', 'fold'])

# ============================================================================
# Training Function with All Improvements + Enhanced Exports
# ============================================================================

def train_single_fold_advanced_with_metrics(X_train, y_train, X_val, y_val, X_test, y_test, device,
                                           fold_idx=0, seed=0, verbose=True,
                                           max_epochs=1000, warmup_epochs=30,
                                           make_plots=False,
                                           se_path='se_net_pretrained.pth',
                                           slsp_path='slsp_net_pretrained.pth'):
    """
    Train one fold/seed: Stage-2 fine-tuning with uniform physics-residual guardrail and PDPM monitoring,
    and comprehensive exports for paper plots/tables.

    Key settings:
    - ECE computed every epoch
    - AUG computed every 20 epochs
    - lambda_f_target = 1.0
    - max_epochs = 1000
    """

    # Set seed for this fold
    set_seed(20250928 + seed * 100 + fold_idx)

    # Initialize metrics tracker
    metrics = MetricsTracker()

    # Create model
    model = METNet().to(device)
    model.load_pretrained(se_path=se_path, slsp_path=slsp_path)

    # Phase 1: Warm-up (SLSP fully frozen)
    for param in model.slsp_net.parameters():
        param.requires_grad = False

    warmup_opt = torch.optim.AdamW([
        {'params': model.se_net.parameters(), 'lr': 1e-4, 'weight_decay': 1e-4},
        {'params': model.output_calibrator.parameters(), 'lr': 1e-4, 'weight_decay': 1e-4}
    ])

    batch_size = 16
    huber = nn.SmoothL1Loss(beta=0.4)

    X_train_tensor = torch.tensor(X_train, dtype=torch.float32).to(device)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32).to(device)
    X_val_tensor = torch.tensor(X_val, dtype=torch.float32).to(device)
    y_val_tensor = torch.tensor(y_val, dtype=torch.float32)

    if verbose:
        print(f"\n--- Fold {fold_idx + 1} Seed {seed + 1} Training ---")
        print(f"Phase 1: Warm-up ({warmup_epochs} epochs) - NO PHYSICS")

    # WARM-UP
    for epoch in range(warmup_epochs):
        model.train()
        model.slsp_net.eval()

        epoch_loss = 0
        n_batches = 0

        indices = torch.randperm(len(X_train))
        for i in range(0, len(X_train), batch_size):
            batch_idx = indices[i:i+batch_size]
            X_batch = X_train_tensor[batch_idx]
            y_batch = y_train_tensor[batch_idx]

            warmup_opt.zero_grad()

            y_pred, monitor_info = model(X_batch, return_monitor=True)

            if y_pred.dim() == 2:
                y_pred = y_pred.squeeze(1)

            L_data = huber(y_pred, y_batch)
            L_cal = model.output_calibrator.regularization_loss()

            loss = L_data + L_cal
            loss.backward()
            warmup_opt.step()

            epoch_loss += loss.item()
            n_batches += 1

            metrics.log(
                epoch=epoch,
                phase='warmup',
                loss=loss.item(),
                L_data=L_data.item(),
                L_cal=L_cal.item()
            )

        if verbose and (epoch + 1) % 5 == 0:
            scale, shift = model.output_calibrator.get_params()
            print(f"  Warmup {epoch+1:2d}: Loss={epoch_loss/n_batches:.4f}, "
                  f"Scale={scale:.3f}, Shift={shift:.3f}")

    # Phase 2: Main training
    if verbose:
        print(f"Phase 2: Main training with corrected physics (delta/128Re^4)")

    model.freeze_slsp_except_last()

    optimizer = torch.optim.AdamW([
        {'params': filter(lambda p: p.requires_grad, model.se_net.parameters()),
         'lr': 1e-4, 'weight_decay': 1e-4},
        {'params': model.output_calibrator.parameters(),
         'lr': 1e-4, 'weight_decay': 1e-4},
        {'params': model.slsp_net.last_layer.parameters(),
         'lr': 1e-6, 'weight_decay': 1e-3},
        {'params': model.monitor_calibrator.parameters(),
         'lr': 1e-3, 'weight_decay': 1e-3}
    ])

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=20, min_lr=1e-7
    )

    early_stopping = EarlyStopping(patience=50)

    # Compute baseline and store in model
    baseline = compute_fold_baseline(model, X_train, device)
    model.deeplift_baseline = baseline

    lambda_f_target = 1.0
    ramp_epochs = max(1, (max_epochs - warmup_epochs) // 3)

    best_val_rmse = float('inf')
    best_model_state = None
    best_epoch = -1

    for epoch in range(max_epochs - warmup_epochs):
        model.train()
        model.slsp_net.eval()
        model.slsp_net.last_layer.train()

        # Determine phase and lambda_f
        if epoch < ramp_epochs:
            phase = 'ramp'
            lam_f = lambda_f_target * min(1.0, (epoch + 1) / ramp_epochs)
        else:
            phase = 'stable'
            lam_f = lambda_f_target

        if hasattr(model, '_override_lambda_f') and model._override_lambda_f is not None:
            lam_f = model._override_lambda_f

        # Update PDPM monitor calibrator every 10 epochs (passive; does not affect predictions)
        if (epoch + 1) % 10 == 0 and epoch > 0:
            monitor_loss = update_monitor_calibrator(model, X_val, y_val, device)
            if verbose:
                alpha = model.monitor_calibrator.alpha.item()
                beta = model.monitor_calibrator.beta.item()
                print(f"  PDPM monitor calibrator updated: alpha={alpha:.3f}, beta={beta:.3f}, loss={monitor_loss:.4f}")

        epoch_metrics = defaultdict(list)

        indices = torch.randperm(len(X_train))
        for i in range(0, len(X_train), batch_size):
            batch_idx = indices[i:i+batch_size]
            X_batch = X_train_tensor[batch_idx]
            y_batch = y_train_tensor[batch_idx]

            optimizer.zero_grad()

            y_pred, monitor_info = model(X_batch, return_monitor=True)

            if y_pred.dim() == 2:
                y_pred = y_pred.squeeze(1)

            # PDPM monitoring weight (passive monitor; NOT used to weight the physics loss)
            w_monitor = monitor_info['w_monitor']
            pi_theory = monitor_info['pi_theory']

            # Losses
            L_data = huber(y_pred, y_batch)

            # Physics-residual hinge applied uniformly across samples (paper Eq. 13)
            Re = monitor_info["Re_eff"]
            te = monitor_info["te_pred"]
            delta = monitor_info["delta_eff"]
            r_star, gamma_star = physics_residual_and_gamma_normalized(Re, te, delta, kf=3.0)
            L_phys_hinge = physics_hinge_loss(r_star, gamma_star)
            # Uniform physics-residual guardrail (no per-sample weighting)
            L_phys = torch.clamp(L_phys_hinge, max=10.0).mean()

            L_cal = model.output_calibrator.regularization_loss()
            L_monitor = model.monitor_calibrator.regularization_loss()
            L_drift = slsp_last_layer_drift(model.slsp_net)

            loss = L_data + lam_f * L_phys + L_cal + L_monitor + L_drift
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            # Log batch metrics
            violations = (r_star > gamma_star).float()

            metrics.log(
                epoch=epoch + warmup_epochs,
                phase=phase,
                loss=loss.item(),
                L_data=L_data.item(),
                L_phys=L_phys.item(),
                L_cal=L_cal.item(),
                L_monitor=L_monitor.item(),
                L_drift=L_drift.item(),
                pi_theory_mean=pi_theory.mean().item(),
                pi_theory_std=pi_theory.std().item(),
                w_monitor_mean=w_monitor.mean().item(),
                w_monitor_std=w_monitor.std().item(),
                violations_rate=violations.mean().item(),
                phys_active=violations.mean().item()
            )

            epoch_metrics['loss'].append(loss.item())
            epoch_metrics['violations'].extend(violations.cpu().numpy())
            epoch_metrics['w_monitor'].extend(w_monitor.cpu().numpy())

        # Validation
        model.eval()
        with torch.no_grad():
            val_preds = []
            val_r = []
            val_w = []
            val_De = []
            val_te = []
            val_violations = []

            for i in range(0, len(X_val), batch_size):
                X_batch = X_val_tensor[i:i+batch_size]
                pred, monitor_info = model(X_batch, return_monitor=True)

                if pred.dim() == 2:
                    pred = pred.squeeze(1)

                val_preds.append(pred.cpu())
                val_r.append(monitor_info['residual'].cpu())
                val_w.append(monitor_info['w_monitor'].cpu())
                val_De.append(monitor_info['De_pred'].cpu())
                val_te.append(monitor_info['te_pred'].cpu())
                val_violations.append(monitor_info['violations'].cpu())

            val_preds = torch.cat(val_preds).numpy()
            val_r = torch.cat(val_r).numpy()
            val_w = torch.cat(val_w).numpy()
            val_De = torch.cat(val_De).numpy()
            val_te = torch.cat(val_te).numpy()
            val_violations = torch.cat(val_violations).numpy()

            val_rmse = np.sqrt(mean_squared_error(y_val, val_preds))

            # Robust statistics
            r_abs = np.abs(val_r)

            # Log epoch-level metrics
            scale, shift = model.output_calibrator.get_params()
            w_drift, b_drift = model.slsp_net.get_last_layer_drift()

            metrics.log_epoch(
                epoch=epoch + warmup_epochs,
                phase=phase,
                val_rmse=val_rmse,
                lambda_f=lam_f,
                r_median=np.median(r_abs),
                r_mad=np.median(np.abs(val_r - np.median(val_r))),
                r_p10=np.percentile(r_abs, 10),
                r_p90=np.percentile(r_abs, 90),
                De_median=np.median(val_De),
                De_p10=np.percentile(val_De, 10),
                De_p90=np.percentile(val_De, 90),
                te_median=np.median(val_te),
                te_p10=np.percentile(val_te, 10),
                te_p90=np.percentile(val_te, 90),
                w_monitor_val_mean=val_w.mean(),
                violations_val_rate=val_violations.mean(),
                calib_scale=scale,
                calib_shift=shift,
                monitor_alpha=model.monitor_calibrator.alpha.item(),
                monitor_beta=model.monitor_calibrator.beta.item(),
                slsp_w_drift=w_drift,
                slsp_b_drift=b_drift
            )

            # Calibration analysis every epoch
            p_hat, w_bar, counts, mci = calibration_bins(val_w, val_violations, n_bins=10, method='equal_mass')
            metrics.log_epoch(epoch=epoch + warmup_epochs, mci=mci)
            if verbose and (epoch + 1) % 5 == 0 and mci is not None and not np.isnan(mci):
                print(f"  ECE: {mci:.3f}")

            # Physics utility analysis every 20 epochs
            if (epoch + 1) % 20 == 0 and lam_f > 0:
                percentiles, gains, aug = physics_gain_curve(
                    model, X_val, y_val, device, lambda_on=1.0  # Use lambda_f=1.0 for consistency
                )
                metrics.log_epoch(epoch=epoch + warmup_epochs, physics_auc_curve=aug)

        # Track best model
        if val_rmse < best_val_rmse:
            best_val_rmse = val_rmse
            best_model_state = deepcopy(model.state_dict())
            best_epoch = epoch + warmup_epochs
            marker = "*"
        else:
            marker = " "

        # Print progress
        if verbose and ((epoch + 1) % 10 == 0 or epoch == 0):
            print(f"  Epoch {epoch+warmup_epochs+1:3d}: "
                  f"Loss={np.mean(epoch_metrics['loss']):.4f} | "
                  f"Val RMSE={val_rmse:.4f} {marker} | "
                  f"lambda_f={lam_f:.2f}")
            print(f"           pi_theory={pi_theory.mean():.3f} | "
                  f"w_monitor={w_monitor.mean():.3f} | "
                  f"Violations={val_violations.mean()*100:.1f}%")

        scheduler.step(val_rmse)

        if early_stopping(val_rmse):
            if verbose:
                print(f"  Early stopping at epoch {epoch+warmup_epochs+1}")
            break

    # Load best model
    model.load_state_dict(best_model_state)

    # ========================================================================
    # Test-set evaluation for this fold/seed
    # ========================================================================
    model.eval()
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)

    with torch.no_grad():
        test_preds = []
        test_w_monitor = []
        test_violations = []

        for i in range(0, len(X_test), batch_size):
            X_batch = X_test_tensor[i:i+batch_size]
            pred, monitor_info = model(X_batch, return_monitor=True)

            if pred.dim() == 2:
                pred = pred.squeeze(1)

            test_preds.append(pred.cpu())
            test_w_monitor.append(monitor_info['w_monitor'].cpu())
            test_violations.append(monitor_info['violations'].cpu())

        test_preds = torch.cat(test_preds).numpy()
        test_w_monitor = torch.cat(test_w_monitor).numpy()
        test_violations = torch.cat(test_violations).numpy()

    test_rmse = np.sqrt(mean_squared_error(y_test, test_preds))
    test_r2 = r2_score(y_test, test_preds)

    # Final metrics from validation set (for calibration)
    final_mci = save_reliability_bins_csv(val_w, val_violations, fold_idx, seed)
    final_phys_active = val_violations.mean()
    final_calib_scale_a, final_calib_shift_b = model.output_calibrator.get_params()
    final_slsp_drift_w, final_slsp_drift_b = model.slsp_net.get_last_layer_drift()

    # Physics utility AUC
    physics_auc_final = metrics.epoch_store.get('physics_auc_curve', [np.nan])[-1]
    if isinstance(physics_auc_final, list):
        physics_auc_final = physics_auc_final[-1] if len(physics_auc_final) > 0 else np.nan

    # ========================================================================
    # Enhanced exports for paper tables and plots
    # ========================================================================

    if verbose:
        print(f"\n  Exporting data for paper plots/tables...")

    # Export A: Per-epoch NPZ
    save_per_epoch_npz(metrics.epoch_store, fold_idx, seed)

    # Export B: Final metrics CSV
    # Note: mean_uncertainty will be computed at ensemble level; use 0.0 as placeholder
    save_final_metrics_csv(
        fold=fold_idx, seed=seed,
        best_val_rmse=best_val_rmse,
        test_rmse=test_rmse,
        test_r2=test_r2,
        mean_uncertainty=0.0,  # Placeholder; will be computed at ensemble level
        final_mci=final_mci if not np.isnan(final_mci) else 0.0,
        final_phys_active=final_phys_active,
        final_calib_scale_a=final_calib_scale_a,
        final_calib_shift_b=final_calib_shift_b,
        final_slsp_drift_w=final_slsp_drift_w,
        final_slsp_drift_b=final_slsp_drift_b,
        physics_auc_final=physics_auc_final if not np.isnan(physics_auc_final) else 0.0
    )

    # Export C: Reliability bins CSV (already done above)

    # Export D: Test predictions CSV
    save_test_predictions_csv(y_test, test_preds, fold_idx, seed)

    # Save metrics (legacy format for compatibility)
    metrics.save_npz(f'metrics_fold{fold_idx}_seed{seed}.npz')

    # Create plots if requested
    if make_plots:
        plot_training_curves(metrics.epoch_store,
                           f'training_curves_fold{fold_idx}_seed{seed}.png')
        plot_calibration_analysis(model, X_val, y_val, device,
                                f'calibration_fold{fold_idx}_seed{seed}.png')
        if verbose:
            print(f"  Plots saved for fold {fold_idx} seed {seed}")

    if verbose:
        print(f"  Best Val RMSE: {best_val_rmse:.4f} at epoch {best_epoch}")
        print(f"  Test RMSE: {test_rmse:.4f}, Test R^2: {test_r2:.4f}")

    return model, best_val_rmse, test_rmse, test_r2, metrics

# ============================================================================
# Ensemble Class
# ============================================================================

class METNetEnsemble:
    def __init__(self, models, training_metrics=None):
        self.models = models
        self.n_models = len(models)
        self.training_metrics = training_metrics

    def predict(self, X, device='cpu', return_std=True):
        if isinstance(X, np.ndarray):
            X = torch.tensor(X, dtype=torch.float32)

        X = X.to(device)
        predictions = []

        for model in self.models:
            model.eval()
            model.to(device)
            with torch.no_grad():
                pred, _ = model(X)
                if pred.dim() == 2:
                    pred = pred.squeeze(1)
                predictions.append(pred.cpu().numpy())

        predictions = np.array(predictions)
        mean_pred = predictions.mean(axis=0)

        if return_std:
            std_pred = predictions.std(axis=0)
            return mean_pred, std_pred
        else:
            return mean_pred

    def evaluate(self, X, y, device='cpu'):
        mean_pred, std_pred = self.predict(X, device, return_std=True)
        rmse = np.sqrt(mean_squared_error(y, mean_pred))
        r2 = r2_score(y, mean_pred)
        mean_std = std_pred.mean()
        return {
            'rmse': rmse,
            'r2': r2,
            'mean_std': mean_std,
            'predictions': mean_pred,
            'uncertainties': std_pred
        }

    def save(self, path='met_net_ensemble_metrics.pkl'):
        ensemble_data = {
            'models_state_dicts': [model.state_dict() for model in self.models],
            'n_models': self.n_models,
            'training_metrics': self.training_metrics
        }
        torch.save(ensemble_data, path)
        print(f"Ensemble saved to {path}")

    def get_median_model_idx(self, X_val, y_val, device='cpu'):
        """Find the model with median performance"""
        rmses = []
        for i, model in enumerate(self.models):
            model.eval()
            model.to(device)
            X_tensor = torch.tensor(X_val, dtype=torch.float32, device=device)
            with torch.no_grad():
                pred, _ = model(X_tensor)
                if pred.dim() == 2:
                    pred = pred.squeeze(1)
                pred = pred.cpu().numpy()
            rmse = np.sqrt(mean_squared_error(y_val, pred))
            rmses.append((i, rmse))

        rmses.sort(key=lambda x: x[1])
        median_idx = rmses[len(rmses)//2][0]
        return median_idx

# ============================================================================
# K-Fold Training with All Improvements + Enhanced Summary Table
# ============================================================================

def train_kfold_ensemble_with_metrics(X, y, n_splits=5, n_seeds=5, test_size=0.15,
                                     device='cpu', verbose=True, warmup_epochs=15,
                                     make_all_plots=False,
                                     se_path='se_net_pretrained.pth',
                                     slsp_path='slsp_net_pretrained.pth'):
    """Train ensemble with correct normalization, improved calibration, and full exports"""

    X_cv, X_test, y_cv, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

    print("="*70)
    print("MET-Net fine-tuning")
    print("="*70)
    print("\nABLATION: Physics loss applied uniformly (no gating)")
    print("  - PDPM monitor: computed for diagnostics (MAPD/PCAR/MCI)")
    print("  - Physics-residual guardrail: applied uniformly (Eq. 13)")
    print("  - Effective behavior: w_monitor = 1.0 for all samples")
    print(f"\nSetup: {n_splits} folds x {n_seeds} seeds = {n_splits*n_seeds} models")
    print(f"Data: {len(X_cv)} CV samples, {len(X_test)} test samples")
    print(f"Physics: lambda_f=1.0, kf=5.0, CORRECT normalization (delta/128Re^4)")
    print(f"PDPM monitor calibrator: class-balanced BCE with identity prior (alpha~1, beta~0)")
    print(f"Metrics: ECE every epoch, AUG every 20 epochs")
    print(f"Training: max_epochs=1000, warmup_epochs={warmup_epochs}")
    print(f"Plots: {'All folds/seeds' if make_all_plots else 'First fold only'}")
    print("="*70)

    all_models = []
    all_metrics = []
    fold_scores = []

    # Track per-fold test performance for the summary table
    per_fold_test_results = []

    for seed in range(n_seeds):
        set_seed(20250928 + seed)

        if verbose:
            print(f"\n=== Seed {seed + 1}/{n_seeds} ===")

        kf = KFold(n_splits=n_splits, shuffle=True, random_state=42 + seed)

        for fold_idx, (train_idx, val_idx) in enumerate(kf.split(X_cv)):
            X_train_fold = X_cv[train_idx]
            y_train_fold = y_cv[train_idx]
            X_val_fold = X_cv[val_idx]
            y_val_fold = y_cv[val_idx]

            make_plots = make_all_plots or (seed == 0 and fold_idx == 0)

            model, val_rmse, test_rmse, test_r2, fold_metrics = train_single_fold_advanced_with_metrics(
                X_train_fold, y_train_fold,
                X_val_fold, y_val_fold,
                X_test, y_test,  # held-out test set
                device=device,
                fold_idx=fold_idx,
                seed=seed,
                verbose=verbose,
                warmup_epochs=warmup_epochs,
                make_plots=make_plots,
                se_path=se_path,
                slsp_path=slsp_path
            )

            all_models.append(model)
            all_metrics.append(fold_metrics.epoch_store)
            fold_scores.append(val_rmse)
            per_fold_test_results.append({
                'fold': fold_idx,
                'seed': seed,
                'test_rmse': test_rmse,
                'test_r2': test_r2
            })

    # Create ensemble
    ensemble = METNetEnsemble(all_models, all_metrics)

    # Evaluate ensemble
    print("\n" + "="*70)
    print("Final Ensemble Evaluation")
    print("="*70)

    cv_mean_rmse = np.mean(fold_scores)
    cv_std_rmse = np.std(fold_scores)
    print(f"CV Performance: {cv_mean_rmse:.4f} +/- {cv_std_rmse:.4f}")

    test_results = ensemble.evaluate(X_test, y_test, device)
    print(f"Ensemble Test RMSE: {test_results['rmse']:.4f}")
    print(f"Ensemble Test R^2: {test_results['r2']:.4f}")
    print(f"Mean Uncertainty: {test_results['mean_std']:.4f}")

    # Find median model for paper
    median_idx = ensemble.get_median_model_idx(X_test, y_test, device)
    fold = median_idx % n_splits
    seed_num = median_idx // n_splits
    print(f"\nMedian model: Fold {fold+1}, Seed {seed_num+1} (index {median_idx})")
    print(f"Use for paper plots:")
    print(f"  - training_curves_fold{fold}_seed{seed_num}.png")
    print(f"  - calibration_fold{fold}_seed{seed_num}.png")
    print(f"  - per_epoch_fold{fold}_seed{seed_num}.npz")
    print(f"  - reliability_bins_fold{fold}_seed{seed_num}.csv")

    # ========================================================================
    # Summary table with all columns
    # ========================================================================
    print("\n" + "="*70)
    print("Creating Enhanced Summary Table")
    print("="*70)

    # Update final_metrics CSVs with mean_uncertainty from ensemble
    print("Updating final_metrics CSVs with ensemble uncertainties...")
    for fold_idx in range(n_splits):
        for seed in range(n_seeds):
            csv_path = f'final_metrics_fold{fold_idx}_seed{seed}.csv'
            if os.path.exists(csv_path):
                df_fold = pd.read_csv(csv_path)
                df_fold['mean_uncertainty'] = test_results['mean_std']
                df_fold.to_csv(csv_path, index=False)

    # Create comprehensive summary table
    summary_rows = []
    for fold_idx in range(n_splits):
        for seed in range(n_seeds):
            csv_path = f'final_metrics_fold{fold_idx}_seed{seed}.csv'
            if os.path.exists(csv_path):
                df_fold = pd.read_csv(csv_path)
                if len(df_fold) > 0:
                    summary_rows.append(df_fold.iloc[0].to_dict())

    if summary_rows:
        summary_df = pd.DataFrame(summary_rows)
        summary_df = summary_df.sort_values(['seed', 'fold'])
        summary_df.to_csv('training_summary_metrics_met_net.csv', index=False)
        print(f"[OK] Saved training_summary_metrics_met_net.csv ({len(summary_df)} rows)")

        # Display summary statistics
        print("\n--- Summary Statistics ---")
        print(f"CV RMSE: {summary_df['best_val_rmse'].mean():.4f} +/- {summary_df['best_val_rmse'].std():.4f}")
        print(f"Test RMSE (per-fold): {summary_df['test_rmse'].mean():.4f} +/- {summary_df['test_rmse'].std():.4f}")
        print(f"Test R^2: {summary_df['test_r2'].mean():.4f} +/- {summary_df['test_r2'].std():.4f}")
        print(f"Final ECE: {np.median(summary_df['final_mci']):.3f} [{np.percentile(summary_df['final_mci'], 25):.3f}, {np.percentile(summary_df['final_mci'], 75):.3f}]")
        print(f"Physics Active %: {np.median(summary_df['final_phys_active']*100):.1f}% [{np.percentile(summary_df['final_phys_active']*100, 25):.1f}%, {np.percentile(summary_df['final_phys_active']*100, 75):.1f}%]")

        # Display first few rows
        print("\n--- Training Summary Metrics (first 10 rows) ---")
        print(summary_df.head(10).to_string(index=False))

    return ensemble, test_results, all_metrics

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

def reorder_dataframe_columns(df):
    missing_cols = set(BILAYER_INPUT_COLUMNS) - set(df.columns)
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    df_reordered = df[BILAYER_INPUT_COLUMNS].copy()
    print("[OK] Data loaded and columns verified")
    return df_reordered

# ============================================================================
# Main Execution
# ============================================================================
