"""
Syntonic Training - Training utilities for syntonic networks.

Provides training loop, callbacks, and metrics for
syntony-aware neural network training.
"""

from syntonic.nn.training.trainer import (
    SyntonicTrainer,
    TrainingConfig,
)
from syntonic.nn.training.callbacks import (
    SyntonyCallback,
    ArchonicEarlyStop,
    SyntonyCheckpoint,
    MetricsLogger,
)
from syntonic.nn.training.metrics import (
    TrainingMetrics,
    SyntonyMetrics,
    compute_epoch_metrics,
)

__all__ = [
    'SyntonicTrainer',
    'TrainingConfig',
    'SyntonyCallback',
    'ArchonicEarlyStop',
    'SyntonyCheckpoint',
    'MetricsLogger',
    'TrainingMetrics',
    'SyntonyMetrics',
    'compute_epoch_metrics',
]
