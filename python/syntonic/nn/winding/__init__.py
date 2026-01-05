"""
Winding Neural Networks - Number-theoretic deep learning with SRT structure.

This module provides winding-aware neural network architectures that integrate:
- T^4 torus winding states |n₇, n₈, n₉, n₁₀⟩
- Fibonacci hierarchy (golden ratio scaling)
- Prime selection (Möbius filtering)
- DHSR dynamics (differentiation-harmonization cycles)
- Temporal blockchain (immutable state recording)
- Syntony consensus (ΔS > threshold validation)
- Resonant Engine (exact Q(φ) lattice arithmetic)

Example:
    >>> from syntonic.nn.winding import WindingNet
    >>> from syntonic.physics.fermions.windings import *
    >>>
    >>> # Standard WindingNet (float-based DHSR)
    >>> model = WindingNet(max_winding=5, base_dim=64, num_blocks=3, output_dim=2)
    >>> windings = [ELECTRON_WINDING, MUON_WINDING, UP_WINDING]
    >>> y = model(windings)
    >>>
    >>> # After training, crystallize to Q(φ) and use exact inference
    >>> model.crystallize_weights(precision=100)
    >>> y_exact = model.forward_exact(windings)
"""

from syntonic.nn.winding.embedding import WindingStateEmbedding
from syntonic.nn.winding.fibonacci_hierarchy import FibonacciHierarchy
from syntonic.nn.winding.prime_selection import PrimeSelectionLayer
from syntonic.nn.winding.syntony import WindingSyntonyComputer
from syntonic.nn.winding.dhsr_block import WindingDHSRBlock
from syntonic.nn.winding.resonant_dhsr_block import ResonantWindingDHSRBlock
from syntonic.nn.winding.winding_net import WindingNet

__all__ = [
    "WindingStateEmbedding",
    "FibonacciHierarchy",
    "PrimeSelectionLayer",
    "WindingSyntonyComputer",
    "WindingDHSRBlock",
    "ResonantWindingDHSRBlock",
    "WindingNet",
]
