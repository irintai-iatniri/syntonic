"""
Syntonic Neural Networks - Pure CRT-Native Deep Learning.

BREAKING CHANGE: This module has been purified to use ResonantTensor
and Retrocausal RES instead of PyTorch dependencies.

Key Features:
- DHSR-structured layers (DifferentiationLayer, HarmonizationLayer, RecursionBlock)
- Pure syntonic loss functions (SyntonicLoss with syntony penalty)
- Retrocausal RES training (replaces gradient-based optimizers)
- Complete architectures (PureSyntonicMLP, PureResonantTransformer)
- Archonic pattern detection and escape mechanisms

Core Insight:
    L_total = L_task + λ(1 - S_model) + μC_{iπ}

Where S_model is the model syntony, optimizing for both
task performance AND representational coherence.

Source: CRT.md §12.2
"""

# Layers (still torch-based for backwards compatibility)
from syntonic.nn.layers import (
    DifferentiationLayer,
    HarmonizationLayer,
    SyntonicGate,
    AdaptiveGate,
    RecursionBlock,
    DeepRecursionNet,
    SyntonicNorm,
    GoldenNorm,
)

# Pure Loss Functions (ResonantTensor-based)
from syntonic.nn.loss.syntonic_loss import (
    SyntonicLoss,
    LayerwiseSyntonicLoss,
    mse_loss,
    cross_entropy_loss,
)
from syntonic.nn.loss.syntony_metrics import (
    SyntonyTracker,
    aggregate_syntonies,
    compute_activation_syntony,
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

# Retrocausal RES Optimization (replaces gradient-based optimizers)
from syntonic.nn.optim import (
    RetrocausalConfig,
    create_retrocausal_evolver,
    create_standard_evolver,
    compare_convergence,
    ResonantEvolver,
    RESConfig,
    RESResult,
)

# Pure Training
from syntonic.nn.training.trainer import (
    RetrocausalTrainer,
    RESTrainingConfig,
    train_with_retrocausal_res,
)

# Constants
import math
PHI = (1 + math.sqrt(5)) / 2
Q_DEFICIT = 0.027395146920
S_TARGET = PHI - Q_DEFICIT

__all__ = [
    # Layers
    'DifferentiationLayer',
    'HarmonizationLayer',
    'SyntonicGate',
    'AdaptiveGate',
    'RecursionBlock',
    'DeepRecursionNet',
    'SyntonicNorm',
    'GoldenNorm',

    # Pure Loss
    'SyntonicLoss',
    'LayerwiseSyntonicLoss',
    'mse_loss',
    'cross_entropy_loss',
    'SyntonyTracker',
    'aggregate_syntonies',
    'compute_activation_syntony',
    'PhaseAlignmentLoss',
    'compute_phase_alignment',
    'IPiConstraint',
    'GoldenPhaseScheduler',
    'SyntonicRegularizer',
    'SyntonyConstraint',
    'ArchonicPenalty',
    'compute_weight_decay',
    'compute_sparsity_penalty',

    # Retrocausal RES (replaces gradient optimizers)
    'RetrocausalConfig',
    'create_retrocausal_evolver',
    'create_standard_evolver',
    'compare_convergence',
    'ResonantEvolver',
    'RESConfig',
    'RESResult',

    # Pure Training
    'RetrocausalTrainer',
    'RESTrainingConfig',
    'train_with_retrocausal_res',

    # Constants
    'PHI',
    'Q_DEFICIT',
    'S_TARGET',
]
