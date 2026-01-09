"""
Syntonic Neural Network Layers - DHSR-structured layers.

This module provides the fundamental building blocks for syntonic neural networks:
- DifferentiationLayer: D̂ operator expanding complexity
- HarmonizationLayer: Ĥ operator building coherence
- RecursionBlock: Complete R̂ = Ĥ ∘ D̂ cycle
- SyntonicGate: Adaptive mixing based on syntony
- SyntonicNorm: Golden-ratio aware normalization
"""

from syntonic.nn.layers.differentiation import DifferentiationLayer
from syntonic.nn.layers.harmonization import HarmonizationLayer
from syntonic.nn.layers.syntonic_gate import (
    SyntonicGate,
    AdaptiveGate,
)
from syntonic.nn.layers.recursion import (
    RecursionBlock,
    DeepRecursionNet,
)
from syntonic.nn.layers.normalization import (
    SyntonicNorm,
    GoldenNorm,
)

# NOTE: DifferentiationModule and HarmonizationModule remain BLOCKED
# (multi-head architectures require additional API development)

__all__ = [
    'DifferentiationLayer',
    'HarmonizationLayer',
    'SyntonicGate',
    'AdaptiveGate',
    'RecursionBlock',
    'DeepRecursionNet',
    'SyntonicNorm',
    'GoldenNorm',
]
