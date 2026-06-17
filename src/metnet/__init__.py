"""MET-Net: a mechanism-guided explainable transfer-learning network for
bilayer tube springback prediction under data scarcity and incomplete physics.

Public API re-exports the core model components and training entry points.
"""
from .met_net import (
    METNet,
    SENet,
    SLSPNet,
    TheoryLayer,
    CorrectionHead,
    OutputCalibrator,
    MonitorCalibrator,
    METNetEnsemble,
    set_seed,
    physics_residual_and_gamma_normalized,
    physics_hinge_loss,
    train_single_fold_advanced_with_metrics,
    train_kfold_ensemble_with_metrics,
)

__all__ = [
    "METNet", "SENet", "SLSPNet", "TheoryLayer", "CorrectionHead",
    "OutputCalibrator", "MonitorCalibrator", "METNetEnsemble", "set_seed",
    "physics_residual_and_gamma_normalized", "physics_hinge_loss",
    "train_single_fold_advanced_with_metrics", "train_kfold_ensemble_with_metrics",
]
__version__ = "1.0.0"
