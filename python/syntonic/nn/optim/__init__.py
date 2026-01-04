"""
Syntonic Optimizers - Syntony-aware training algorithms.

These optimizers modulate learning rates based on model syntony,
enabling adaptive training that respects syntonic structure.

Key insight: lr_eff = lr × (1 + α(S - S_target))
"""

from syntonic.nn.optim.syntonic_adam import (
    SyntonicAdam,
    AdaptiveSyntonicAdam,
)
from syntonic.nn.optim.syntonic_sgd import (
    SyntonicSGD,
    SyntonicMomentum,
)
from syntonic.nn.optim.schedulers import (
    GoldenScheduler,
    SyntonyCyclicScheduler,
    WarmupGoldenScheduler,
)
from syntonic.nn.optim.gradient_mod import (
    SyntonyGradientModifier,
    GoldenClipping,
    ArchonicGradientEscape,
)

__all__ = [
    'SyntonicAdam',
    'AdaptiveSyntonicAdam',
    'SyntonicSGD',
    'SyntonicMomentum',
    'GoldenScheduler',
    'SyntonyCyclicScheduler',
    'WarmupGoldenScheduler',
    'SyntonyGradientModifier',
    'GoldenClipping',
    'ArchonicGradientEscape',
]
