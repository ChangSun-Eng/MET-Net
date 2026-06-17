"""
Stage-1: pretrain SLSP-Net (Single-Layer Springback Prediction Network).

SLSP-Net is a compact MLP that maps the manufacturing parameters of a single-layer
tube (13 features) to its springback angle. It is pretrained here on the abundant
single-layer source domain using a physics/tree-guided strategy: a previously fitted
tree ensemble (XGBoost + LightGBM) supplies dense pseudo-labels and a collocation-point
consistency penalty (weight lambda_physics = 0.1) that regularizes the network toward
the tree's response surface. The resulting subnet and its input/output scalers are saved
for assembly into MET-Net (scripts/train_met_net.py loads them via METNet.load_pretrained).

Companion script
----------------
The tree ensemble consumed here (tree_ensemble_simple.pkl) is produced by the Stage-1
tree-fitting step (step1_best_tree_C1_01.py in the source package): it fits XGBoost and
LightGBM on the single-layer source data with light feature engineering and pickles a
SimpleTreePredictor. That predictor class is reproduced below so the pickle can be loaded
without external dependencies; the two steps are kept separate and are not merged.

Inputs
------
  * models/tree_ensemble_simple.pkl - pickled SimpleTreePredictor (tree pseudo-label model).
  * single_layer_source.xlsx - single-layer source dataset (13 features + springback).
                               NOT redistributed; supply your own path via --data.

Output
------
  * slsp_net_pretrained.pth - model_state_dict + input/output scalers (scaler_X, scaler_y)
                              plus architecture metadata, consumed by MET-Net assembly.

Example
-------
    python scripts/train_slsp_net.py \
        --data /path/to/single_layer_source.xlsx \
        --tree-model models/tree_ensemble_simple.pkl \
        --out-dir runs/slsp_net

Paper settings (defaults below): hidden dims [256, 128, 64, 32], dropout 0.2,
n_epochs 400, batch_size 256, lr 1e-3, weight_decay 1e-5, lambda_physics 0.1,
n_synthetic 20000, n_collocation 500, patience 40.

NOTE: This script is included for transparency of the Stage-1 SLSP-Net pretraining
method. It cannot be rerun end-to-end as shipped, because the single-layer SOURCE
dataset is not redistributed; supply your own via --data. The trained checkpoint
models/slsp_net_pretrained.pth is provided directly and is what the rest of the
pipeline (Stage-2, EPE, explainability) uses.
"""

import argparse
import os
import json
import pickle
import warnings
from datetime import datetime

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')


# ============================================================================
# Configuration
# ============================================================================

class Config:
    """Configuration for SLSP-Net pretraining."""
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Network architecture
    input_dim = 13  # Manufacturing parameters
    hidden_dims = [256, 128, 64, 32]
    output_dim = 1  # Springback angle
    dropout_rate = 0.2
    activation = 'relu'

    # Training parameters
    n_epochs = 400
    batch_size = 256
    learning_rate = 1e-3
    weight_decay = 1e-5

    # Data generation
    n_synthetic = 20000  # synthetic sample count for tree-guided coverage
    val_split = 0.2

    # Physics constraint
    lambda_physics = 0.1  # collocation-consistency weight
    n_collocation = 500

    # Early stopping
    patience = 40
    min_delta = 1e-6

    # Logging
    print_every = 20
    save_dir = "./slsp_net_results"


# ============================================================================
# Tree model loader (required for the physics/tree-guided constraint)
# ============================================================================

class SimpleTreePredictor:
    """Tree ensemble model used to supply pseudo-labels and the collocation target.

    Reproduced here so tree_ensemble_simple.pkl can be unpickled without external
    dependencies; it mirrors the predictor saved by the Stage-1 tree-fitting step.
    """
    def __init__(self, xgb_model, lgb_model, xgb_weight, lgb_weight,
                 scaler_X, scaler_y):
        self.xgb_model = xgb_model
        self.lgb_model = lgb_model
        self.xgb_weight = xgb_weight
        self.lgb_weight = lgb_weight
        self.scaler_X = scaler_X
        self.scaler_y = scaler_y

    def create_features(self, X):
        """Feature engineering matching the original tree model."""
        features = [X]

        # Key ratios
        D, t, R_B, alpha_B = X[:, 0:1], X[:, 1:2], X[:, 2:3], X[:, 12:13]

        features.append(t / np.maximum(D, 0.1))
        features.append(D / np.maximum(R_B, 0.1))
        features.append(R_B / np.maximum(D, 0.1))
        features.append(alpha_B * D / np.maximum(R_B, 0.1))
        features.append(np.sin(alpha_B * np.pi / 180))
        features.append(np.cos(alpha_B * np.pi / 180))

        return np.hstack(features)

    def predict(self, X):
        """Make predictions using the weighted ensemble."""
        X_eng = self.create_features(X)
        X_scaled = self.scaler_X.transform(X_eng)

        # XGBoost prediction
        xgb_pred = self.xgb_model.predict(X_scaled)
        xgb_pred = self.scaler_y.inverse_transform(xgb_pred.reshape(-1, 1)).ravel()

        # LightGBM prediction
        lgb_pred = self.lgb_model.predict(X_scaled)
        lgb_pred = self.scaler_y.inverse_transform(lgb_pred.reshape(-1, 1)).ravel()

        # Weighted average
        return self.xgb_weight * xgb_pred + self.lgb_weight * lgb_pred


# ============================================================================
# SLSP-Net architecture
# ============================================================================

class SLSPNet(nn.Module):
    """
    Single-Layer Springback Prediction Network.
    Maps manufacturing parameters (13D) to springback angle (1D).
    """
    def __init__(self, config):
        super(SLSPNet, self).__init__()

        self.config = config
        layers = []
        prev_dim = config.input_dim

        # Build hidden layers
        for i, hidden_dim in enumerate(config.hidden_dims):
            layers.append(nn.Linear(prev_dim, hidden_dim))

            # Batch norm for all but last hidden layer
            if i < len(config.hidden_dims) - 1:
                layers.append(nn.BatchNorm1d(hidden_dim))

            # Activation
            if config.activation == 'relu':
                layers.append(nn.ReLU())
            elif config.activation == 'leaky_relu':
                layers.append(nn.LeakyReLU(0.1))
            elif config.activation == 'elu':
                layers.append(nn.ELU())
            else:
                layers.append(nn.ReLU())

            # Dropout for regularization
            if i < len(config.hidden_dims) - 1:
                layers.append(nn.Dropout(config.dropout_rate))

            prev_dim = hidden_dim

        # Output layer
        layers.append(nn.Linear(prev_dim, config.output_dim))

        self.network = nn.Sequential(*layers)
        self._initialize_weights()

    def _initialize_weights(self):
        """He initialization for ReLU networks."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        """Forward pass: manufacturing parameters -> springback angle."""
        return self.network(x)

    def get_architecture_info(self):
        """Return architecture information for saving."""
        return {
            'input_dim': self.config.input_dim,
            'hidden_dims': self.config.hidden_dims,
            'output_dim': self.config.output_dim,
            'activation': self.config.activation,
            'dropout_rate': self.config.dropout_rate
        }


# ============================================================================
# Data generation and loading
# ============================================================================

class DataManager:
    """Manages data loading and synthetic generation."""

    def __init__(self, config, tree_model_path, data_path):
        self.config = config
        self.tree_model_path = tree_model_path
        self.data_path = data_path
        self.tree_model = None
        self.X_real = None
        self.y_real = None
        self.scaler_X = RobustScaler()
        self.scaler_y = RobustScaler()

    def load_tree_model(self):
        """Load the tree model used for the physics/tree-guided constraint."""
        try:
            with open(self.tree_model_path, 'rb') as f:
                self.tree_model = pickle.load(f)
            print(f"[OK] Successfully loaded tree model from {self.tree_model_path}")
            return True
        except Exception as e:
            print(f"Error loading tree model: {e}")
            return False

    def load_real_data(self):
        """Load the single-layer source dataset for distribution reference."""
        try:
            df = pd.read_excel(self.data_path)
            X = df.iloc[:, :13].values
            y = df.iloc[:, 13].values

            # Remove outliers
            z_scores = np.abs((y - y.mean()) / y.std())
            mask = z_scores < 3
            self.X_real = X[mask]
            self.y_real = y[mask]

            print(f"[OK] Loaded {len(self.X_real)} real samples from {self.data_path}")
            return True
        except Exception as e:
            print(f"Warning: Could not load real data: {e}")
            print("Will use default ranges for synthetic data generation")
            return False

    def generate_synthetic_data(self, n_samples):
        """Generate synthetic data using the tree model."""
        if self.X_real is not None:
            # Use real data distribution
            mins = np.percentile(self.X_real, 5, axis=0)
            maxs = np.percentile(self.X_real, 95, axis=0)

            X_synthetic = np.random.rand(n_samples, 13)
            for i in range(13):
                X_synthetic[:, i] = X_synthetic[:, i] * (maxs[i] - mins[i]) + mins[i]
        else:
            # Default ranges
            X_synthetic = np.random.rand(n_samples, 13) * 50

        # Get tree predictions
        y_synthetic = self.tree_model.predict(X_synthetic)

        return X_synthetic, y_synthetic

    def prepare_training_data(self):
        """Prepare training and validation data."""
        # Generate synthetic data
        X_syn, y_syn = self.generate_synthetic_data(self.config.n_synthetic)

        # Split for validation
        n_val = int(self.config.n_synthetic * self.config.val_split)
        X_train = X_syn[:-n_val]
        X_val = X_syn[-n_val:]
        y_train = y_syn[:-n_val]
        y_val = y_syn[-n_val:]

        # Scale data
        X_train_scaled = self.scaler_X.fit_transform(X_train)
        y_train_scaled = self.scaler_y.fit_transform(y_train.reshape(-1, 1)).ravel()
        X_val_scaled = self.scaler_X.transform(X_val)
        y_val_scaled = self.scaler_y.transform(y_val.reshape(-1, 1)).ravel()

        print(f"Training samples: {len(X_train)}")
        print(f"Validation samples: {len(X_val)}")
        print(f"Output range: [{y_train.min():.3f}, {y_train.max():.3f}]")

        return {
            'X_train': X_train, 'y_train': y_train,
            'X_val': X_val, 'y_val': y_val,
            'X_train_scaled': X_train_scaled, 'y_train_scaled': y_train_scaled,
            'X_val_scaled': X_val_scaled, 'y_val_scaled': y_val_scaled
        }


# ============================================================================
# Training with the physics/tree-guided constraint
# ============================================================================

class Trainer:
    """Handles model training with the collocation-consistency constraint."""

    def __init__(self, model, data_manager, config):
        self.model = model.to(config.device)
        self.data_manager = data_manager
        self.config = config
        self.history = {
            'epoch': [],
            'train_loss': [],
            'val_loss': [],
            'physics_loss': [],
            'data_loss': [],
            'learning_rate': []
        }

    def compute_physics_loss(self, X_batch_original):
        """Compute the collocation-consistency (physics/tree-guided) loss."""
        # Generate collocation points
        X_colloc, _ = self.data_manager.generate_synthetic_data(self.config.n_collocation)
        X_colloc_scaled = self.data_manager.scaler_X.transform(X_colloc)
        X_colloc_tensor = torch.tensor(X_colloc_scaled, dtype=torch.float32).to(self.config.device)

        # NN predictions on collocation points
        pred_colloc = self.model(X_colloc_tensor).squeeze()

        # Tree predictions (physics target)
        y_colloc_tree = self.data_manager.tree_model.predict(X_colloc)
        y_colloc_scaled = self.data_manager.scaler_y.transform(y_colloc_tree.reshape(-1, 1)).ravel()
        y_colloc_tensor = torch.tensor(y_colloc_scaled, dtype=torch.float32).to(self.config.device)

        # MSE between NN and tree
        physics_loss = nn.MSELoss()(pred_colloc, y_colloc_tensor)
        return physics_loss

    def train(self, data):
        """Train model with the physics/tree-guided constraint."""
        # Create datasets
        train_dataset = TensorDataset(
            torch.tensor(data['X_train_scaled'], dtype=torch.float32),
            torch.tensor(data['y_train_scaled'], dtype=torch.float32)
        )
        val_dataset = TensorDataset(
            torch.tensor(data['X_val_scaled'], dtype=torch.float32),
            torch.tensor(data['y_val_scaled'], dtype=torch.float32)
        )

        train_loader = DataLoader(train_dataset, batch_size=self.config.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=self.config.batch_size)

        # Optimizer and scheduler
        optimizer = optim.AdamW(self.model.parameters(),
                                lr=self.config.learning_rate,
                                weight_decay=self.config.weight_decay)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                         mode='min',
                                                         factor=0.5,
                                                         patience=20,
                                                         min_lr=1e-6)

        criterion = nn.MSELoss()
        best_val_loss = float('inf')
        best_model_state = None
        patience_counter = 0

        print("\n" + "=" * 70)
        print("Training SLSP-Net with the physics/tree-guided constraint")
        print(f"Physics weight (lambda): {self.config.lambda_physics}")
        print("=" * 70)

        for epoch in range(self.config.n_epochs):
            # Training phase
            self.model.train()
            train_losses = []
            data_losses = []
            physics_losses = []

            for X_batch, y_batch in train_loader:
                X_batch = X_batch.to(self.config.device)
                y_batch = y_batch.to(self.config.device)

                optimizer.zero_grad()

                # Data loss
                predictions = self.model(X_batch).squeeze()
                data_loss = criterion(predictions, y_batch)

                # Physics loss
                physics_loss = self.compute_physics_loss(None)

                # Total loss
                total_loss = data_loss + self.config.lambda_physics * physics_loss

                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                optimizer.step()

                train_losses.append(total_loss.item())
                data_losses.append(data_loss.item())
                physics_losses.append(physics_loss.item())

            # Validation phase
            self.model.eval()
            val_losses = []

            with torch.no_grad():
                for X_batch, y_batch in val_loader:
                    X_batch = X_batch.to(self.config.device)
                    y_batch = y_batch.to(self.config.device)
                    predictions = self.model(X_batch).squeeze()
                    loss = criterion(predictions, y_batch)
                    val_losses.append(loss.item())

            # Calculate epoch metrics
            avg_train_loss = np.mean(train_losses)
            avg_val_loss = np.mean(val_losses)
            avg_data_loss = np.mean(data_losses)
            avg_physics_loss = np.mean(physics_losses)
            current_lr = optimizer.param_groups[0]['lr']

            # Update scheduler
            scheduler.step(avg_val_loss)

            # Early stopping check
            if avg_val_loss < best_val_loss - self.config.min_delta:
                best_val_loss = avg_val_loss
                best_model_state = self.model.state_dict().copy()
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= self.config.patience:
                    print(f"\nEarly stopping triggered at epoch {epoch}")
                    break

            # Record history
            self.history['epoch'].append(epoch)
            self.history['train_loss'].append(avg_train_loss)
            self.history['val_loss'].append(avg_val_loss)
            self.history['data_loss'].append(avg_data_loss)
            self.history['physics_loss'].append(avg_physics_loss)
            self.history['learning_rate'].append(current_lr)

            # Print progress
            if epoch % self.config.print_every == 0:
                print(f"Epoch {epoch:4d} | Train: {avg_train_loss:.6f} | Val: {avg_val_loss:.6f} | "
                      f"Data: {avg_data_loss:.6f} | Physics: {avg_physics_loss:.6f} | "
                      f"LR: {current_lr:.2e}")

        # Load best model
        if best_model_state is not None:
            self.model.load_state_dict(best_model_state)
            print(f"\nBest model restored (val_loss: {best_val_loss:.6f})")

        return best_val_loss


# ============================================================================
# Evaluation
# ============================================================================

class Evaluator:
    """Handles model evaluation."""

    def __init__(self, model, data_manager, config):
        self.model = model
        self.data_manager = data_manager
        self.config = config

    def evaluate_on_synthetic(self, n_test=2000):
        """Evaluate on synthetic test data (tree predictions)."""
        self.model.eval()

        # Generate test data
        X_test, y_test = self.data_manager.generate_synthetic_data(n_test)
        X_test_scaled = self.data_manager.scaler_X.transform(X_test)

        with torch.no_grad():
            X_tensor = torch.tensor(X_test_scaled, dtype=torch.float32).to(self.config.device)
            y_pred_scaled = self.model(X_tensor).cpu().numpy().ravel()
            y_pred = self.data_manager.scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).ravel()

        # Calculate metrics
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        return {
            'rmse': rmse,
            'mae': mae,
            'r2': r2,
            'predictions': y_pred,
            'targets': y_test
        }

    def evaluate_on_real(self):
        """Evaluate on real test data if available."""
        if self.data_manager.X_real is None:
            return None

        self.model.eval()

        # Split real data
        X_train, X_test, y_train, y_test = train_test_split(
            self.data_manager.X_real, self.data_manager.y_real,
            test_size=0.2, random_state=42
        )

        # Scale and predict
        X_test_scaled = self.data_manager.scaler_X.transform(X_test)

        with torch.no_grad():
            X_tensor = torch.tensor(X_test_scaled, dtype=torch.float32).to(self.config.device)
            y_pred_scaled = self.model(X_tensor).cpu().numpy().ravel()
            y_pred = self.data_manager.scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).ravel()

        # Calculate metrics
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        return {
            'rmse': rmse,
            'mae': mae,
            'r2': r2,
            'predictions': y_pred,
            'targets': y_test
        }


# ============================================================================
# Visualization
# ============================================================================

def plot_results(trainer, evaluator, synthetic_results, real_results=None):
    """Create comprehensive visualization."""

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    # Training history
    ax = axes[0, 0]
    ax.plot(trainer.history['epoch'], trainer.history['train_loss'],
            label='Train', alpha=0.8)
    ax.plot(trainer.history['epoch'], trainer.history['val_loss'],
            label='Validation', alpha=0.8)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('Training History')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')

    # Physics vs Data loss
    ax = axes[0, 1]
    ax.plot(trainer.history['epoch'], trainer.history['data_loss'],
            label='Data Loss', alpha=0.8)
    ax.plot(trainer.history['epoch'], trainer.history['physics_loss'],
            label='Physics Loss', alpha=0.8)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('Loss Components')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')

    # Learning rate schedule
    ax = axes[0, 2]
    ax.plot(trainer.history['epoch'], trainer.history['learning_rate'], 'g-', alpha=0.8)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Learning Rate')
    ax.set_title('Learning Rate Schedule')
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')

    # Predictions vs Tree
    ax = axes[1, 0]
    ax.scatter(synthetic_results['targets'], synthetic_results['predictions'],
               alpha=0.5, s=10)
    min_val = min(synthetic_results['targets'].min(), synthetic_results['predictions'].min())
    max_val = max(synthetic_results['targets'].max(), synthetic_results['predictions'].max())
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', label='Perfect')
    ax.set_xlabel('Tree Predictions')
    ax.set_ylabel('NN Predictions')
    ax.set_title("vs Tree (R2=%.4f)" % synthetic_results["r2"])
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Predictions vs Real (if available)
    if real_results:
        ax = axes[1, 1]
        ax.scatter(real_results['targets'], real_results['predictions'],
                   alpha=0.5, s=10, color='green')
        min_val = min(real_results['targets'].min(), real_results['predictions'].min())
        max_val = max(real_results['targets'].max(), real_results['predictions'].max())
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', label='Perfect')
        ax.set_xlabel('Real Springback')
        ax.set_ylabel('NN Predictions')
        ax.set_title("vs Real Data (R2=%.4f)" % real_results["r2"])
        ax.legend()
        ax.grid(True, alpha=0.3)
    else:
        axes[1, 1].axis('off')

    # Error distribution
    ax = axes[1, 2]
    errors = synthetic_results['predictions'] - synthetic_results['targets']
    ax.hist(errors, bins=30, alpha=0.7, edgecolor='black', density=True)
    ax.axvline(x=0, color='r', linestyle='--', label='Zero Error')
    ax.set_xlabel('Prediction Error')
    ax.set_ylabel('Density')
    ax.set_title("Error Distribution (RMSE=%.4f)" % synthetic_results["rmse"])
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


# ============================================================================
# Save results (compatible with MET-Net assembly)
# ============================================================================

def save_results(model, data_manager, trainer, synthetic_results, real_results,
                 config, ckpt_name):
    """
    Save all results. The primary checkpoint (ckpt_name) bundles the model
    state dict together with the input/output scalers, which is the artifact
    consumed by MET-Net assembly (METNet.load_pretrained).
    """
    # Create save directory
    os.makedirs(config.save_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Lightweight checkpoint for MET-Net integration (state dict + scalers)
    integration_dict = {
        'model_state_dict': model.state_dict(),
        'model_architecture': model.get_architecture_info(),
        'scaler_X': data_manager.scaler_X,
        'scaler_y': data_manager.scaler_y,
        'input_dim': config.input_dim,
        'output_dim': config.output_dim,
        'model_type': 'SLSP-Net'
    }

    integration_path = os.path.join(config.save_dir, ckpt_name)
    torch.save(integration_dict, integration_path)
    print(f"\n[OK] SLSP-Net checkpoint saved to: {integration_path}")

    # Full checkpoint with training history and evaluation results
    save_dict = {
        # Model components
        'model_state_dict': model.state_dict(),
        'model_architecture': model.get_architecture_info(),

        # Scalers (critical for MET-Net integration)
        'scaler_X': data_manager.scaler_X,
        'scaler_y': data_manager.scaler_y,

        # Training configuration
        'config': {
            'device': str(config.device),
            'n_epochs': config.n_epochs,
            'batch_size': config.batch_size,
            'learning_rate': config.learning_rate,
            'lambda_physics': config.lambda_physics,
            'n_synthetic': config.n_synthetic,
            'hidden_dims': config.hidden_dims,
            'activation': config.activation,
            'dropout_rate': config.dropout_rate
        },

        # Training history
        'training_history': trainer.history,

        # Evaluation results
        'synthetic_results': synthetic_results,
        'real_results': real_results if real_results else None,

        # Metadata
        'metadata': {
            'timestamp': timestamp,
            'model_type': 'SLSP-Net',
            'training_strategy': 'physics/tree-guided pretraining',
            'compatible_with': 'MET-Net assembly',
            'pytorch_version': torch.__version__,
            'numpy_version': np.__version__,
        }
    }

    checkpoint_path = os.path.join(config.save_dir, f'slsp_net_full_{timestamp}.pth')
    torch.save(save_dict, checkpoint_path)
    print(f"[OK] Full checkpoint saved to: {checkpoint_path}")

    # Save human-readable summary
    summary = {
        'Model': 'SLSP-Net (Single-Layer Springback Prediction Network)',
        'Training Strategy': 'physics/tree-guided pretraining (lambda_physics=0.1)',
        'Architecture': "Input: %d -> Hidden: %s -> Output: %d" % (
            config.input_dim, config.hidden_dims, config.output_dim),
        'Performance on Synthetic Data': {
            'RMSE': float(synthetic_results['rmse']),
            'MAE': float(synthetic_results['mae']),
            'R2': float(synthetic_results['r2'])
        }
    }

    if real_results:
        summary['Performance on Real Data'] = {
            'RMSE': float(real_results['rmse']),
            'MAE': float(real_results['mae']),
            'R2': float(real_results['r2'])
        }

    summary_path = os.path.join(config.save_dir, f'slsp_net_summary_{timestamp}.json')
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=4)
    print(f"[OK] Summary saved to: {summary_path}")

    return integration_path, checkpoint_path


# ============================================================================
# Main execution
# ============================================================================

def set_seed(seed):
    """Set random seeds for reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)


def main():
    ap = argparse.ArgumentParser(
        description="Stage-1 SLSP-Net physics/tree-guided pretraining."
    )
    ap.add_argument("--data", default="single_layer_source.xlsx",
                    help="Single-layer SOURCE dataset (13 features + springback). NOT "
                         "redistributed with this release; supply your own path to rerun.")
    ap.add_argument("--tree-model", default="models/tree_ensemble_simple.pkl",
                    help="Pickled SimpleTreePredictor used for pseudo-labels/collocation.")
    ap.add_argument("--out-dir", default="./slsp_net_results")
    ap.add_argument("--n-epochs", type=int, default=400)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--learning-rate", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-5)
    ap.add_argument("--n-synthetic", type=int, default=20000)
    ap.add_argument("--lambda-physics", type=float, default=0.1)
    ap.add_argument("--n-collocation", type=int, default=500)
    ap.add_argument("--patience", type=int, default=40)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--ckpt-name", default="slsp_net_pretrained.pth",
                    help="Output checkpoint (state dict + scalers) for MET-Net assembly.")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    set_seed(args.seed)

    print("=" * 70)
    print("SLSP-Net pretraining")
    print("Single-Layer Springback Prediction with physics/tree-guided strategy")
    print("=" * 70)

    # Initialize configuration from CLI
    config = Config()
    config.device = torch.device(args.device)
    config.n_epochs = args.n_epochs
    config.batch_size = args.batch_size
    config.learning_rate = args.learning_rate
    config.weight_decay = args.weight_decay
    config.n_synthetic = args.n_synthetic
    config.lambda_physics = args.lambda_physics
    config.n_collocation = args.n_collocation
    config.patience = args.patience
    config.save_dir = args.out_dir
    print(f"\nUsing device: {config.device}")

    # Initialize data manager
    data_manager = DataManager(config, args.tree_model, args.data)

    # Load tree model (required for the physics/tree-guided constraint)
    if not data_manager.load_tree_model():
        print("Error: Tree model is required for training!")
        return

    # Load real data (optional but recommended)
    data_manager.load_real_data()

    # Prepare training data
    print("\n" + "=" * 70)
    print("Preparing Training Data")
    print("=" * 70)
    data = data_manager.prepare_training_data()

    # Create model
    print("\n" + "=" * 70)
    print("Creating SLSP-Net Model")
    print("=" * 70)
    model = SLSPNet(config)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Architecture: {config.input_dim} -> {config.hidden_dims} -> {config.output_dim}")

    # Train model
    trainer = Trainer(model, data_manager, config)
    best_val_loss = trainer.train(data)

    # Evaluate model
    print("\n" + "=" * 70)
    print("Model Evaluation")
    print("=" * 70)

    evaluator = Evaluator(model, data_manager, config)

    # Evaluate on synthetic data (vs tree)
    print("\n--- Evaluation vs Tree (Synthetic Data) ---")
    synthetic_results = evaluator.evaluate_on_synthetic()
    print(f"RMSE: {synthetic_results['rmse']:.6f}")
    print(f"MAE:  {synthetic_results['mae']:.6f}")
    print(f"R2:   {synthetic_results['r2']:.6f}")

    # Evaluate on real data if available
    real_results = None
    if data_manager.X_real is not None:
        print("\n--- Evaluation vs Real Data ---")
        real_results = evaluator.evaluate_on_real()
        print(f"RMSE: {real_results['rmse']:.6f}")
        print(f"MAE:  {real_results['mae']:.6f}")
        print(f"R2:   {real_results['r2']:.6f}")

    # Create visualizations
    print("\n" + "=" * 70)
    print("Creating Visualizations")
    print("=" * 70)
    fig = plot_results(trainer, evaluator, synthetic_results, real_results)

    # Save figure
    os.makedirs(config.save_dir, exist_ok=True)
    fig_path = os.path.join(config.save_dir, 'slsp_net_results.png')
    fig.savefig(fig_path, dpi=150, bbox_inches='tight')
    print(f"[OK] Results plot saved to: {fig_path}")
    plt.close(fig)

    # Save all results
    print("\n" + "=" * 70)
    print("Saving Results")
    print("=" * 70)
    integration_path, checkpoint_path = save_results(
        model, data_manager, trainer, synthetic_results, real_results,
        config, args.ckpt_name
    )

    print("\n" + "=" * 70)
    print("TRAINING COMPLETE")
    print("=" * 70)
    print("\nFinal Performance Summary:")
    print(f"  vs Tree: R2={synthetic_results['r2']:.4f}, RMSE={synthetic_results['rmse']:.4f}")
    if real_results:
        print(f"  vs Real: R2={real_results['r2']:.4f}, RMSE={real_results['rmse']:.4f}")
    print(f"\nModel saved for MET-Net integration: {integration_path}")

    return model, synthetic_results, real_results


if __name__ == "__main__":
    main()
