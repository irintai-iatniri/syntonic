"""
Pure Syntonic Loss Functions - Task + Syntony optimization.

PURE IMPLEMENTATION: Uses ResonantTensor, no PyTorch dependencies.

L_total = L_task + λ(1 - S_model) + μC_{iπ}
"""

from syntonic.nn.loss.syntony_metrics import (
    compute_activation_syntony,
    aggregate_syntonies,
    SyntonyTracker,
)
from syntonic.nn.loss.syntonic_loss import (
    SyntonicLoss,
    LayerwiseSyntonicLoss,
    mse_loss,
    cross_entropy_loss,
)
from syntonic.nn.loss.phase_alignment import (
    PhaseAlignmentLoss,
    compute_phase_alignment,
    IPiConstraint,
    GoldenPhaseScheduler,
)
from syntonic.nn.loss.regularization import (
    SyntonicRegularizer,
    SyntonyConstraint,
    ArchonicPenalty,
    compute_weight_decay,
    compute_sparsity_penalty,
)

__all__ = [
    # Syntony metrics
    'compute_activation_syntony',
    'aggregate_syntonies',
    'SyntonyTracker',
    
    # Loss functions
    'SyntonicLoss',
    'LayerwiseSyntonicLoss',
    'mse_loss',
    'cross_entropy_loss',
    
    # Phase alignment
    'PhaseAlignmentLoss',
    'compute_phase_alignment',
    'IPiConstraint',
    'GoldenPhaseScheduler',
    
    # Regularization
    'SyntonicRegularizer',
    'SyntonyConstraint',
    'ArchonicPenalty',
    'compute_weight_decay',
    'compute_sparsity_penalty',
]
