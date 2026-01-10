"""
Syntonic Optimization - Retrocausal RES Training.

BREAKING CHANGE: Gradient-based optimizers have been removed.
Use Retrocausal RES (Resonant Evolutionary Search) instead.

Key insight: Syntony as fitness, not loss gradients.

Example:
    >>> from syntonic.resonant.retrocausal import create_retrocausal_evolver
    >>> evolver = create_retrocausal_evolver(tensor, population_size=32)
    >>> result = evolver.run()
"""

# Re-export Retrocausal RES as the primary optimizer
from syntonic.resonant.retrocausal import (
    RetrocausalConfig,
    create_retrocausal_evolver,
    create_standard_evolver,
    compare_convergence,
)

# Re-export core RES types
from syntonic._core import (
    ResonantEvolver,
    RESConfig,
    RESResult,
)

__all__ = [
    # Retrocausal RES (recommended)
    'RetrocausalConfig',
    'create_retrocausal_evolver',
    'create_standard_evolver',
    'compare_convergence',
    # Core RES types
    'ResonantEvolver',
    'RESConfig',
    'RESResult',
]
