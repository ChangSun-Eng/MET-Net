"""
Stage-1 SE-Net pretraining (Section-Equivalent Network)
-------------------------------------------------------
This standalone script pretrains the SE-Net subnetwork used as Stage-1 of
MET-Net. SE-Net maps a bilayer tube cross-section (D, t, r) onto an equivalent
single-layer section by learning two small, data-driven corrections on top of
the closed-form equivalent-section theory:

  - A correction to the effective modulus ratio:
        lambda_eff = lambda * exp(g_lambda)
    with baseline lambda = E2 / E1. Centroid c, equivalent neutral radius Re,
    and the inertia term delta are recomputed with lambda_eff so that Re and c
    become fine-tunable and data-driven.

  - A multiplicative thickness correction solved from the corrected cubic:
        te_pred = te_th_eff * exp(g_t)

The equivalent diameters are composed from the corrected radius and thickness:
        De_pred = 2*Re_eff + te_pred
        de_pred = 2*Re_eff - te_pred

Training objective (per batch):
  - Data loss:    MSE(De_pred, De_data) + MSE(te_pred, te_data)
                  (+ optional MSE(de_pred, de_data) when measured de is given)
  - Theory loss:  MSE(te_pred, te_th_eff) + MSE(De_pred, De_th_eff)
  - Physics loss: mean( | te_pred^3 + 4*Re_eff^2*te_pred - delta_eff/(16*Re_eff) | )
  - Regularizer:  beta_lambda * mean(g_lambda^2) + beta_t * mean(g_t^2)

  L = L_data + lambda_theory * L_theory + lambda_phys * L_phys + L_reg

The pretraining set is generated on the fly: triplets (D, t, r) are sampled
uniformly over the section domain and the equivalent targets (De, te) are
produced by the baseline closed-form theory (optionally with measurement
noise). No Excel input is required.

The final layer of the correction head is zero-initialized so that initial
predictions coincide exactly with the theory.

Output checkpoint: se_net_pretrained.pth
"""

import argparse
import math
import warnings
from dataclasses import dataclass
from typing import Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

warnings.filterwarnings("ignore")


# -----------------------------------------------------------------------------
# Utilities
# -----------------------------------------------------------------------------
def cbrt(x: torch.Tensor) -> torch.Tensor:
    """Sign-preserving cube root, differentiable and stable near zero."""
    eps = torch.finfo(x.dtype).eps
    return torch.sign(x) * torch.pow(torch.clamp(torch.abs(x), min=eps), 1.0 / 3.0)


# -----------------------------------------------------------------------------
# Core Model (Theory + Learnable Corrections)
# -----------------------------------------------------------------------------
class TheoryLayer(nn.Module):
    def __init__(self, E1: float = 120.0, E2: float = 69.0):
        super().__init__()
        # Buffers (moved to device with .to())
        self.register_buffer("E1", torch.tensor(float(E1)))
        self.register_buffer("E2", torch.tensor(float(E2)))

    def forward(self, D: torch.Tensor, t: torch.Tensor, r: torch.Tensor) -> Dict[str, torch.Tensor]:
        lam = self.E2 / self.E1  # baseline lambda = E2/E1

        # Layer thicknesses
        t1 = (r / (r + 1.0)) * t
        t2 = (1.0 / (r + 1.0)) * t

        # E-weighted centroid through the wall
        c = (t1**2 + 2.0 * lam * t1 * t2 + lam * t2**2) / (2.0 * (t1 + lam * t2))
        Re = D * 0.5 - c

        # Diameters
        d1 = D - 2.0 * t1
        d2 = d1 - 2.0 * t2

        # delta and IE (bilayer exact E-weighted inertia)
        delta = (D**4 - d1**4) + lam * (d1**4 - d2**4)
        IE_bi = (math.pi / 64.0) * delta

        # Solve te^3 + 4 Re^2 te - delta/(16 Re) = 0  (baseline theory)
        Re_safe = torch.clamp(Re, min=1e-12)
        p = 4.0 * (Re_safe**2)
        q = -delta / (16.0 * Re_safe)
        half_q = 0.5 * q
        disc = torch.clamp(half_q**2 + (p / 3.0) ** 3, min=0.0)
        sqrt_disc = torch.sqrt(disc)
        te_th = cbrt(-half_q + sqrt_disc) + cbrt(-half_q - sqrt_disc)

        De_th = 2.0 * Re + te_th
        de_th = 2.0 * Re - te_th
        IE_eq = math.pi * (Re**3 * te_th + Re * te_th**3 / 4.0)

        return {
            "lam": lam,
            "t1": t1,
            "t2": t2,
            "c": c,
            "Re": Re,
            "d1": d1,
            "d2": d2,
            "delta": delta,
            "te_th": te_th,
            "De_th": De_th,
            "de_th": de_th,
            "IE_bilayer": IE_bi,
            "IE_eq": IE_eq,
        }


class CorrectionHead(nn.Module):
    """Small MLP that outputs two corrections: g_lambda and g_t"""
    def __init__(self, hidden: int = 64, act: str = "silu"):
        super().__init__()
        self.inp = nn.Linear(6, hidden)
        self.hid = nn.Linear(hidden, hidden)
        self.out = nn.Linear(hidden, 2)  # [g_lambda, g_t]

        if act.lower() == "relu":
            self.act = nn.ReLU()
        elif act.lower() == "gelu":
            self.act = nn.GELU()
        elif act.lower() == "tanh":
            self.act = nn.Tanh()
        else:
            self.act = nn.SiLU()

        # Zero-init the final layer => initial predictions equal theory
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(self, features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.act(self.inp(features))
        x = self.act(self.hid(x))
        out = self.out(x)  # (N,2)
        g_lambda = out[..., 0].squeeze(-1)
        g_t = out[..., 1].squeeze(-1)
        return g_lambda, g_t


class SENet(nn.Module):
    """
    SE-Net (Section-Equivalent Network):
      - Learn lambda_eff = lambda * exp(g_lambda)
      - Recompute c, Re, delta with lambda_eff
      - Solve cubic -> te_th_eff
      - Learn multiplicative thickness scale te_pred = te_th_eff * exp(g_t)
      - Compose De_pred, de_pred from Re_eff and te_pred
    """
    def __init__(self, E1: float = 120.0, E2: float = 69.0, hidden: int = 64, act: str = "silu"):
        super().__init__()
        self.theory = TheoryLayer(E1, E2)
        self.head = CorrectionHead(hidden=hidden, act=act)

    def forward(self, D: torch.Tensor, t: torch.Tensor, r: torch.Tensor) -> Dict[str, torch.Tensor]:
        th = self.theory(D, t, r)

        # 6-D features: [D, t, r, t/D, t1/D, t2/D]
        eps = 1e-12
        feats = torch.stack([
            D, t, r,
            t / torch.clamp(D, min=eps),
            th["t1"] / torch.clamp(D, min=eps),
            th["t2"] / torch.clamp(D, min=eps),
        ], dim=-1)

        # Learn corrections
        g_lambda, g_t = self.head(feats)

        # Effective lambda (always positive; equals lam when g_lambda=0)
        lam_eff = th["lam"] * torch.exp(g_lambda)

        # Recompute centroid, Re, delta with lambda_eff
        t1, t2 = th["t1"], th["t2"]
        c_eff = (t1**2 + 2.0 * lam_eff * t1 * t2 + lam_eff * t2**2) / (2.0 * (t1 + lam_eff * t2))
        Re_eff = D * 0.5 - c_eff
        d1, d2 = th["d1"], th["d2"]
        delta_eff = (D**4 - d1**4) + lam_eff * (d1**4 - d2**4)

        # Solve cubic with (Re_eff, delta_eff) -> te_th_eff
        Re_safe = torch.clamp(Re_eff, min=1e-12)
        p = 4.0 * (Re_safe**2)
        q = -delta_eff / (16.0 * Re_safe)
        half_q = 0.5 * q
        disc = torch.clamp(half_q**2 + (p / 3.0)**3, min=0.0)
        sqrt_disc = torch.sqrt(disc)
        te_th_eff = cbrt(-half_q + sqrt_disc) + cbrt(-half_q - sqrt_disc)

        # Multiplicative thickness correction (positivity + equality at init)
        te_pred = te_th_eff * torch.exp(g_t)

        # Compose equivalent diameters from corrected Re and te
        De_th_eff = 2.0 * Re_eff + te_th_eff
        de_th_eff = 2.0 * Re_eff - te_th_eff
        De_pred = 2.0 * Re_eff + te_pred
        de_pred = 2.0 * Re_eff - te_pred

        # Inertia check (optional)
        IE_eq_eff = math.pi * (Re_eff**3 * te_th_eff + Re_eff * te_th_eff**3 / 4.0)

        return {
            **th,  # baseline theory (lam, Re, delta, te_th, De_th, ...)
            "g_lambda": g_lambda,
            "g_t": g_t,
            "lam_eff": lam_eff,
            "c_eff": c_eff,
            "Re_eff": Re_eff,
            "delta_eff": delta_eff,
            "te_th_eff": te_th_eff,
            "De_th_eff": De_th_eff,
            "de_th_eff": de_th_eff,
            "te_pred": te_pred,
            "De_pred": De_pred,
            "de_pred": de_pred,
            "IE_eq_eff": IE_eq_eff,
        }


# -----------------------------------------------------------------------------
# Synthetic data generation (no Excel required)
# -----------------------------------------------------------------------------
@dataclass
class Domain:
    D: Tuple[float, float] = (20.0, 120.0)
    t: Tuple[float, float] = (0.5, 6.0)
    r: Tuple[float, float] = (0.3, 3.0)


def sample_uniform(n: int, dom: Domain, device: torch.device):
    D = torch.rand(n, device=device) * (dom.D[1] - dom.D[0]) + dom.D[0]
    t = torch.rand(n, device=device) * (dom.t[1] - dom.t[0]) + dom.t[0]
    r = torch.rand(n, device=device) * (dom.r[1] - dom.r[0]) + dom.r[0]
    return D, t, r


class SyntheticDataLoader:
    def __init__(self, model_for_theory: TheoryLayer, n_samples: int = 20000, test_size: float = 0.2,
                 noise_std: float = 0.0, random_state: int = 42, device: torch.device = torch.device("cpu")):
        """
        Generate synthetic dataset by sampling (D, t, r), computing (De, te) from baseline theory,
        and optionally adding Gaussian noise to simulate measurements.
        """
        self.dom = Domain()
        self.theory = model_for_theory
        self.n_samples = n_samples
        self.test_size = test_size
        self.noise_std = noise_std
        self.random_state = random_state
        self.device = device
        self._prepare()

    def _prepare(self):
        torch.manual_seed(self.random_state)
        np.random.seed(self.random_state)

        with torch.no_grad():
            D_all, t_all, r_all = sample_uniform(self.n_samples, self.dom, self.device)
            th = self.theory(D_all, t_all, r_all)
            De_all = th["De_th"].clone()
            te_all = th["te_th"].clone()

            if self.noise_std > 0:
                De_all += torch.randn_like(De_all) * self.noise_std
                te_all += torch.randn_like(te_all) * self.noise_std

        # Train/test split (contiguous for determinism)
        n_test = int(self.n_samples * self.test_size)
        n_train = self.n_samples - n_test

        self.train_data = {
            "D": D_all[:n_train].cpu(),
            "t": t_all[:n_train].cpu(),
            "r": r_all[:n_train].cpu(),
            "De": De_all[:n_train].cpu(),
            "te": te_all[:n_train].cpu(),
        }
        self.test_data = {
            "D": D_all[n_train:].cpu(),
            "t": t_all[n_train:].cpu(),
            "r": r_all[n_train:].cpu(),
            "De": De_all[n_train:].cpu(),
            "te": te_all[n_train:].cpu(),
        }

        print("Data prepared successfully (synthetic)")
        print(f"Total samples: {self.n_samples}")
        print(f"Training samples: {n_train}")
        print(f"Testing samples: {n_test}")
        print("Input ranges:")
        print(f"  D: [{self.train_data['D'].min().item():.2f}, {self.train_data['D'].max().item():.2f}] mm")
        print(f"  t: [{self.train_data['t'].min().item():.2f}, {self.train_data['t'].max().item():.2f}] mm")
        print(f"  r: [{self.train_data['r'].min().item():.2f}, {self.train_data['r'].max().item():.2f}]")

    def get_tensors(self, subset: str, device: torch.device):
        data = self.train_data if subset == "train" else self.test_data
        return {
            "D": data["D"].to(device),
            "t": data["t"].to(device),
            "r": data["r"].to(device),
            "De": data["De"].to(device),
            "te": data["te"].to(device),
        }


# -----------------------------------------------------------------------------
# Losses, Training, Evaluation
# -----------------------------------------------------------------------------
@dataclass
class LossWeights:
    data_De: float = 1.0
    data_de: float = 0.0   # set >0 when you have measured de; 0 for synthetic
    data_te: float = 1.0
    theory: float = 0.1    # keep small in fine-tune (keeps correction small)
    physics: float = 0.5   # stronger during fine-tune (soft constraint)
    reg_lambda: float = 1e-4
    reg_t: float = 1e-4


def compute_losses(outputs: Dict[str, torch.Tensor],
                   targets: Dict[str, torch.Tensor],
                   w: LossWeights,
                   mode: str = "train"):
    """Compute all losses using corrected (eff) quantities and predicted values."""
    losses: Dict[str, torch.Tensor] = {}

    # Data losses (De, de, te) -- for synthetic we have De, te only
    losses["De_mse"] = F.mse_loss(outputs["De_pred"], targets["De"])
    losses["te_mse"] = F.mse_loss(outputs["te_pred"], targets["te"])
    # de_data not available in synthetic set; keep placeholder
    if w.data_de > 0.0 and "de" in targets:
        losses["de_mse"] = F.mse_loss(outputs["de_pred"], targets["de"])
    else:
        losses["de_mse"] = torch.zeros_like(losses["De_mse"])

    losses["data"] = w.data_De * losses["De_mse"] + w.data_te * losses["te_mse"] + w.data_de * losses["de_mse"]

    # Theory replication around corrected theory (keeps g_t small)
    losses["th_De"] = F.mse_loss(outputs["De_pred"], outputs["De_th_eff"])
    losses["th_te"] = F.mse_loss(outputs["te_pred"], outputs["te_th_eff"])
    losses["theory"] = losses["th_De"] + losses["th_te"]

    # Physics residual with corrected quantities
    Re_safe = torch.clamp(outputs["Re_eff"], min=1e-12)
    residual = torch.abs(outputs["te_pred"]**3 + 4.0 * Re_safe**2 * outputs["te_pred"]
                         - outputs["delta_eff"] / (16.0 * Re_safe))
    losses["physics"] = torch.mean(residual)

    # Regularization for small corrections
    losses["reg"] = w.reg_lambda * torch.mean(outputs["g_lambda"]**2) + w.reg_t * torch.mean(outputs["g_t"]**2)

    # Total objective (stage-agnostic; adjust weights w for pretrain vs finetune)
    losses["total"] = losses["data"] + w.theory * losses["theory"] + w.physics * losses["physics"] + losses["reg"]
    return losses


class Trainer:
    def __init__(self, model: nn.Module, data_loader: SyntheticDataLoader, device: torch.device,
                 weights: LossWeights):
        self.model = model.to(device)
        self.data_loader = data_loader
        self.device = device
        self.w = weights
        self.history = {"train": [], "test": []}

    def train_epoch(self, optimizer: torch.optim.Optimizer, train_data: Dict[str, torch.Tensor]):
        self.model.train()
        outputs = self.model(train_data["D"], train_data["t"], train_data["r"])
        losses = compute_losses(outputs, train_data, self.w, mode="train")
        optimizer.zero_grad()
        losses["total"].backward()
        optimizer.step()
        return losses

    @torch.no_grad()
    def evaluate(self, test_data: Dict[str, torch.Tensor]):
        self.model.eval()
        outputs = self.model(test_data["D"], test_data["t"], test_data["r"])
        # For testing we still compute the same combined losses (with same weights)
        losses = compute_losses(outputs, test_data, self.w, mode="test")

        De_pred = outputs["De_pred"].cpu().numpy()
        te_pred = outputs["te_pred"].cpu().numpy()
        De_true = test_data["De"].cpu().numpy()
        te_true = test_data["te"].cpu().numpy()

        metrics = {
            "De_rmse": float(np.sqrt(mean_squared_error(De_true, De_pred))),
            "De_mae": float(mean_absolute_error(De_true, De_pred)),
            "De_r2": float(r2_score(De_true, De_pred)),
            "te_rmse": float(np.sqrt(mean_squared_error(te_true, te_pred))),
            "te_mae": float(mean_absolute_error(te_true, te_pred)),
            "te_r2": float(r2_score(te_true, te_pred)),
        }
        return losses, metrics, outputs

    def train(self, epochs: int = 500, lr: float = 1e-3, print_every: int = 50):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=25)

        train_data = self.data_loader.get_tensors("train", self.device)
        test_data = self.data_loader.get_tensors("test", self.device)

        print("\n" + "=" * 60)
        print("Starting Training (learn lambda_eff and thickness scale)")
        print("=" * 60)

        best_loss = float("inf")
        best_state = None

        for epoch in range(1, epochs + 1):
            train_losses = self.train_epoch(optimizer, train_data)
            self.history["train"].append({k: v.detach().cpu() for k, v in train_losses.items()})

            test_losses, metrics, _ = self.evaluate(test_data)
            self.history["test"].append({k: v.detach().cpu() for k, v in test_losses.items()})

            scheduler.step(test_losses["total"])

            if test_losses["total"] < best_loss:
                best_loss = test_losses["total"]
                best_state = self.model.state_dict().copy()

            if epoch % print_every == 0:
                print(f"\nEpoch {epoch}/{epochs}")
                print(
                    f"  Train Loss: {train_losses['total']:.6f} "
                    f"(Data: {train_losses['data']:.6f}, "
                    f"Theory: {train_losses['theory']:.6f}, "
                    f"Physics: {train_losses['physics']:.6f}, "
                    f"Reg: {train_losses['reg']:.6f})"
                )
                print(f"  Test  Loss: {test_losses['total']:.6f}")
                print("  Metrics:")
                print(
                    f"    De: RMSE={metrics['De_rmse']:.6f}, "
                    f"MAE={metrics['De_mae']:.6f}, R^2={metrics['De_r2']:.6f}"
                )
                print(
                    f"    te: RMSE={metrics['te_rmse']:.6f}, "
                    f"MAE={metrics['te_mae']:.6f}, R^2={metrics['te_r2']:.6f}"
                )

        if best_state is not None:
            self.model.load_state_dict(best_state)
            print(f"\nBest model loaded (test loss: {best_loss:.6f})")

        return self.history


# -----------------------------------------------------------------------------
# Visualization
# -----------------------------------------------------------------------------
@torch.no_grad()
def plot_results(model: nn.Module, data_loader: SyntheticDataLoader, device: torch.device, save_path: str = None):
    model.eval()
    test_data = data_loader.get_tensors("test", device)
    outputs = model(test_data["D"], test_data["t"], test_data["r"])

    De_true = test_data["De"].cpu().numpy()
    te_true = test_data["te"].cpu().numpy()

    De_pred = outputs["De_pred"].cpu().numpy()
    te_pred = outputs["te_pred"].cpu().numpy()

    # Baseline theory (lam) and corrected theory (lam_eff)
    De_theory = outputs["De_th"].cpu().numpy()
    te_theory = outputs["te_th"].cpu().numpy()
    De_th_eff = outputs["De_th_eff"].cpu().numpy()
    te_th_eff = outputs["te_th_eff"].cpu().numpy()

    fig, axes = plt.subplots(2, 3, figsize=(16, 10))

    # De vs truth
    ax = axes[0, 0]
    ax.scatter(De_true, De_pred, alpha=0.6, label="Network", s=25)
    ax.scatter(De_true, De_th_eff, alpha=0.6, label="Theory (lambda_eff)", s=25)
    ax.scatter(De_true, De_theory, alpha=0.3, label="Theory (baseline lambda)", s=20)
    mn, mx = De_true.min(), De_true.max()
    ax.plot([mn, mx], [mn, mx], "r--", label="Perfect")
    ax.set_xlabel("True De [mm]")
    ax.set_ylabel("Predicted / Theory De [mm]")
    ax.set_title("Equivalent Diameter Predictions")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # te vs truth
    ax = axes[0, 1]
    ax.scatter(te_true, te_pred, alpha=0.6, label="Network", s=25)
    ax.scatter(te_true, te_th_eff, alpha=0.6, label="Theory (lambda_eff)", s=25)
    ax.scatter(te_true, te_theory, alpha=0.3, label="Theory (baseline lambda)", s=20)
    mn, mx = te_true.min(), te_true.max()
    ax.plot([mn, mx], [mn, mx], "r--", label="Perfect")
    ax.set_xlabel("True te [mm]")
    ax.set_ylabel("Predicted / Theory te [mm]")
    ax.set_title("Equivalent Thickness Predictions")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Error distributions
    ax = axes[0, 2]
    De_error = De_pred - De_true
    te_error = te_pred - te_true
    ax.hist(De_error, bins=25, alpha=0.6, label="De error", density=True)
    ax.hist(te_error, bins=25, alpha=0.6, label="te error", density=True)
    ax.set_xlabel("Prediction Error [mm]")
    ax.set_ylabel("Density")
    ax.set_title("Error Distributions")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # De residuals
    ax = axes[1, 0]
    ax.scatter(De_true, De_error, alpha=0.6, s=20)
    ax.axhline(y=0, color="r", linestyle="--")
    ax.set_xlabel("True De [mm]")
    ax.set_ylabel("Residual [mm]")
    ax.set_title("De Residuals vs True Values")
    ax.grid(True, alpha=0.3)

    # te residuals
    ax = axes[1, 1]
    ax.scatter(te_true, te_error, alpha=0.6, s=20)
    ax.axhline(y=0, color="r", linestyle="--")
    ax.set_xlabel("True te [mm]")
    ax.set_ylabel("Residual [mm]")
    ax.set_title("te Residuals vs True Values")
    ax.grid(True, alpha=0.3)

    # Correction distributions
    ax = axes[1, 2]
    g_lambda = outputs["g_lambda"].cpu().numpy()
    g_t = outputs["g_t"].cpu().numpy()
    ax.hist(g_lambda, bins=30, alpha=0.7, label="g_lambda", color="purple")
    ax.hist(g_t, bins=30, alpha=0.7, label="g_t", color="green")
    ax.set_xlabel("Correction Value")
    ax.set_ylabel("Frequency")
    ax.set_title(f"Correction Distributions\n"
                 f"g_lambda: mean={g_lambda.mean():.3e}, std={g_lambda.std():.3e}\n"
                 f"g_t: mean={g_t.mean():.3e}, std={g_t.std():.3e}")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Plot saved to {save_path}")
    plt.show()
    return fig


def plot_training_history(history: Dict[str, list], save_path: str = None):
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))

    train_total = [h["total"].item() for h in history["train"]]
    test_total  = [h["total"].item() for h in history["test"]]
    train_data  = [h["data"].item()  for h in history["train"]]
    test_data   = [h["data"].item()  for h in history["test"]]
    train_theory= [h["theory"].item()for h in history["train"]]
    train_phys  = [h["physics"].item()for h in history["train"]]
    train_reg   = [h["reg"].item()   for h in history["train"]]

    epochs = range(1, len(train_total) + 1)

    # Total loss
    ax = axes[0, 0]
    ax.plot(epochs, train_total, label="Train", alpha=0.8)
    ax.plot(epochs, test_total,  label="Test",  alpha=0.8)
    ax.set_xlabel("Epoch"); ax.set_ylabel("Loss"); ax.set_title("Total Loss")
    ax.legend(); ax.grid(True, alpha=0.3); ax.set_yscale("log")

    # Data loss
    ax = axes[0, 1]
    ax.plot(epochs, train_data, label="Train", alpha=0.8)
    ax.plot(epochs, test_data,  label="Test",  alpha=0.8)
    ax.set_xlabel("Epoch"); ax.set_ylabel("Loss"); ax.set_title("Data Loss")
    ax.legend(); ax.grid(True, alpha=0.3); ax.set_yscale("log")

    # Theory loss
    ax = axes[0, 2]
    ax.plot(epochs, train_theory, color="orange", alpha=0.8)
    ax.set_xlabel("Epoch"); ax.set_ylabel("Loss"); ax.set_title("Theory Loss")
    ax.grid(True, alpha=0.3); ax.set_yscale("log")

    # Physics loss
    ax = axes[1, 0]
    ax.plot(epochs, train_phys, color="green", alpha=0.8)
    ax.set_xlabel("Epoch"); ax.set_ylabel("Loss"); ax.set_title("Physics Loss")
    ax.grid(True, alpha=0.3); ax.set_yscale("log")

    # Regularization
    ax = axes[1, 1]
    ax.plot(epochs, train_reg, color="purple", alpha=0.8)
    ax.set_xlabel("Epoch"); ax.set_ylabel("Loss"); ax.set_title("Regularization")
    ax.grid(True, alpha=0.3); ax.set_yscale("log")

    # Empty panel for spacing / notes
    axes[1, 2].axis('off')

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Training history plot saved to {save_path}")
    plt.show()
    return fig


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(
        description="Stage-1 SE-Net pretraining on synthetic equivalent-section data."
    )
    parser.add_argument("--E1", type=float, default=120.0,
                        help="Inner-layer modulus E1 (default: 120.0).")
    parser.add_argument("--E2", type=float, default=69.0,
                        help="Outer-layer modulus E2 (default: 69.0).")
    parser.add_argument("--hidden", type=int, default=64,
                        help="Hidden width of the correction head (default: 64).")
    parser.add_argument("--act", type=str, default="silu",
                        choices=["silu", "relu", "gelu", "tanh"],
                        help="Activation in the correction head (default: silu).")
    parser.add_argument("--n-samples", type=int, default=20000,
                        help="Number of synthetic samples (default: 20000).")
    parser.add_argument("--test-size", type=float, default=0.2,
                        help="Fraction held out for testing (default: 0.2).")
    parser.add_argument("--noise-std", type=float, default=0.0,
                        help="Gaussian noise std on synthetic targets (default: 0.0).")
    parser.add_argument("--epochs", type=int, default=500,
                        help="Number of training epochs (default: 500).")
    parser.add_argument("--lr", type=float, default=1e-3,
                        help="Adam learning rate (default: 1e-3).")
    parser.add_argument("--print-every", type=int, default=50,
                        help="Logging interval in epochs (default: 50).")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed (default: 42).")
    parser.add_argument("--checkpoint", type=str, default="se_net_pretrained.pth",
                        help="Output checkpoint path (default: se_net_pretrained.pth).")
    parser.add_argument("--pred-plot", type=str, default="se_net_predictions.png",
                        help="Path for the prediction plot (default: se_net_predictions.png).")
    parser.add_argument("--history-plot", type=str, default="se_net_training.png",
                        help="Path for the training-history plot (default: se_net_training.png).")
    return parser.parse_args()


def main():
    args = parse_args()

    # Reproducibility and numeric stability
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.set_default_dtype(torch.float64)

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {DEVICE}")

    print("\n" + "=" * 60)
    print("Preparing Synthetic Data")
    print("=" * 60)
    # Use theory layer standalone for generating targets
    theory = TheoryLayer(E1=args.E1, E2=args.E2).to(DEVICE)
    data_loader = SyntheticDataLoader(
        theory, n_samples=args.n_samples, test_size=args.test_size,
        noise_std=args.noise_std, random_state=args.seed, device=DEVICE
    )

    print("\n" + "=" * 60)
    print("Creating SE-Net")
    print("=" * 60)
    model = SENet(E1=args.E1, E2=args.E2, hidden=args.hidden, act=args.act)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {n_params}")

    # Loss weights: defaults for pretraining on synthetic theory
    weights = LossWeights(
        data_De=1.0, data_de=0.0, data_te=1.0,
        theory=0.1, physics=0.5,
        reg_lambda=1e-5, reg_t=1e-5
    )
    trainer = Trainer(model, data_loader, device=DEVICE, weights=weights)

    history = trainer.train(epochs=args.epochs, lr=args.lr, print_every=args.print_every)

    print("\n" + "=" * 60)
    print("Final Evaluation on Test Set")
    print("=" * 60)
    test_data = data_loader.get_tensors("test", DEVICE)
    test_losses, metrics, outputs = trainer.evaluate(test_data)

    print("\nFinal Test Metrics:")
    print(f"  De: RMSE={metrics['De_rmse']:.6f} mm, MAE={metrics['De_mae']:.6f} mm, R^2={metrics['De_r2']:.6f}")
    print(f"  te: RMSE={metrics['te_rmse']:.6f} mm, MAE={metrics['te_mae']:.6f} mm, R^2={metrics['te_r2']:.6f}")

    print("\n" + "=" * 60)
    print("Creating Visualizations")
    print("=" * 60)
    plot_results(model, data_loader, device=DEVICE, save_path=args.pred_plot)
    plot_training_history(history, save_path=args.history_plot)

    torch.save(
        {"model_state_dict": model.state_dict(), "history": history, "final_metrics": metrics},
        args.checkpoint
    )
    print(f"\nModel saved to {args.checkpoint}")

    return model, history, metrics


if __name__ == "__main__":
    main()
