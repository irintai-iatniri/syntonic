"""
Syntonic Loss Functions - Task + Syntony optimization.

This module provides loss functions that optimize for both task performance
AND syntony, creating naturally regularized, coherent representations.

L_total = L_task + λ(1 - S_model) + μC_{iπ}
"""

from syntonic.nn.loss.syntony_metrics import (
    compute_activation_syntony,
    compute_network_syntony,
    SyntonyTracker,
)
from syntonic.nn.loss.syntonic_loss import (
    SyntonicLoss,
    LayerwiseSyntonicLoss,
)
from syntonic.nn.loss.phase_alignment import (
    PhaseAlignmentLoss,
    compute_phase_alignment,
    IPiConstraint,
    GoldenPhaseScheduler,
)
from syntonic.nn.loss.regularization import (
    SyntonicRegularizer,
    GoldenDecay,
    SyntonyConstraint,
    ArchonicPenalty,
)

__all__ = [
    'compute_activation_syntony',
    'compute_network_syntony',
    'SyntonyTracker',
    'SyntonicLoss',
    'LayerwiseSyntonicLoss',
    'PhaseAlignmentLoss',
    'compute_phase_alignment',
    'IPiConstraint',
    'GoldenPhaseScheduler',
    'SyntonicRegularizer',
    'GoldenDecay',
    'SyntonyConstraint',
    'ArchonicPenalty',
]
