"""
Syntonic Optimization - Retrocausal RES and Golden Momentum Training.

Provides two optimization strategies:

1. Retrocausal RES (Resonant Evolutionary Search):
   - Syntony as fitness, not loss gradients
   - Attractor-guided evolution

2. Golden Momentum:
   - Gradient-based with phi-derived momentum (beta = 1/φ)
   - Resistant to noise, responsive to persistent patterns

Example (RES):
    >>> from syntonic.nn.optim import create_retrocausal_evolver
    >>> evolver = create_retrocausal_evolver(tensor, population_size=32)
    >>> result = evolver.run()

Example (Golden Momentum):
    >>> from syntonic.nn.optim import GoldenMomentumOptimizer
    >>> optimizer = GoldenMomentumOptimizer(model.parameters(), lr=0.027395)
    >>> optimizer.step()
"""

# Re-export Retrocausal RES
from syntonic._core import (
    GoldenMomentum,
    RESConfig,
    ResonantEvolver,
    RESResult,
)
from syntonic.resonant.retrocausal import (
    RetrocausalConfig,
    compare_convergence,
    create_retrocausal_evolver,
    create_standard_evolver,
)

# Golden Momentum optimizer
from syntonic.nn.optim.golden_momentum import (
    GoldenMomentumOptimizer,
    create_golden_optimizer,
)

__all__ = [
    # Golden Momentum (gradient-based)
    "GoldenMomentumOptimizer",
    "GoldenMomentum",
    "create_golden_optimizer",
    # Retrocausal RES (evolutionary)
    "RetrocausalConfig",
    "create_retrocausal_evolver",
    "create_standard_evolver",
    "compare_convergence",
    # Core RES types
    "ResonantEvolver",
    "RESConfig",
    "RESResult",
]
