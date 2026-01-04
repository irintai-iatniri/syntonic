"""
Syntonic Neural Network Layers - DHSR-structured layers.

This module provides the fundamental building blocks for syntonic neural networks:
- DifferentiationLayer: D̂ operator expanding complexity
- HarmonizationLayer: Ĥ operator building coherence
- RecursionBlock: Complete R̂ = Ĥ ∘ D̂ cycle
- SyntonicGate: Adaptive mixing based on syntony
- SyntonicNorm: Golden-ratio aware normalization
"""

from syntonic.nn.layers.differentiation import (
    DifferentiationLayer,
    DifferentiationModule,
)
from syntonic.nn.layers.harmonization import (
    HarmonizationLayer,
    HarmonizationModule,
)
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

__all__ = [
    'DifferentiationLayer',
    'DifferentiationModule',
    'HarmonizationLayer',
    'HarmonizationModule',
    'SyntonicGate',
    'AdaptiveGate',
    'RecursionBlock',
    'DeepRecursionNet',
    'SyntonicNorm',
    'GoldenNorm',
]
