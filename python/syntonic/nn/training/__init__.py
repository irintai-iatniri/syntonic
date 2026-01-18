"""
Syntonic Training - Training utilities for syntonic networks.

Provides training loop, callbacks, and metrics for
syntony-aware neural network training.

Uses pure Python + ResonantTensor (no PyTorch dependencies).
"""

# Pure Python implementations (no PyTorch)
from .trainer import (
    RetrocausalTrainer,
    RESTrainingConfig,
    SyntonyTracker,
)
from .callbacks import (
    Callback,
    SyntonyCallback,
    ArchonicEarlyStop,
    SyntonyCheckpoint,
    MetricsLogger,
    FitnessPlateauCallback,
    default_callbacks,
)
from .metrics import (
    TrainingMetrics,
    SyntonyMetrics,
    MetricsAggregator,
    SyntonyTracker as SyntonyMetricTracker,
    compute_syntony_from_weights,
    compute_accuracy,
    compute_mse,
    check_archonic_pattern,
    compute_syntony_gap,
)

__all__ = [
    # Trainer
    'RetrocausalTrainer',
    'RESTrainingConfig',
    'SyntonyTracker',
    # Callbacks
    'Callback',
    'SyntonyCallback',
    'ArchonicEarlyStop',
    'SyntonyCheckpoint',
    'MetricsLogger',
    'FitnessPlateauCallback',
    'default_callbacks',
    # Metrics
    'TrainingMetrics',
    'SyntonyMetrics',
    'MetricsAggregator',
    'SyntonyMetricTracker',
    'compute_syntony_from_weights',
    'compute_accuracy',
    'compute_mse',
    'check_archonic_pattern',
    'compute_syntony_gap',
]
