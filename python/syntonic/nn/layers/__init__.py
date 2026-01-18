"""
Syntonic Neural Network Layers - DHSR-structured layers.

This module provides the fundamental building blocks for syntonic neural networks:
- DifferentiationLayer: D̂ operator expanding complexity
- HarmonizationLayer: Ĥ operator building coherence
- RecursionBlock: Complete R̂ = Ĥ ∘ D̂ cycle
- SyntonicGate: Adaptive mixing based on syntony
- SyntonicNorm: Golden-ratio aware normalization
- ResonantLinear: Linear layer in Q(φ)
"""

from syntonic.nn.layers.differentiation import (
    DifferentiationLayer,
    DifferentiationModule,
)
from syntonic.nn.layers.harmonization import HarmonizationLayer
from syntonic.nn.layers.syntonic_gate import (
    SyntonicGate,
    AdaptiveGate,
)
from syntonic.nn.layers.recursion import (
    RecursionBlock,
    DeepRecursionNet,
)
from syntonic.nn.layers.gnosis import GnosisLayer
from syntonic.nn.layers.normalization import (
    SyntonicNorm,
    GoldenNorm,
)
from syntonic.nn.layers.resonant_linear import ResonantLinear
from syntonic.nn.layers.prime_syntony_gate import (
    PrimeSyntonyGate,
    WindingAttention,
    SRTTransformerBlock,
    get_stable_dimensions,
    suggest_network_dimensions,
)
from syntonic.nn.activations.gnosis_gelu import GnosisGELU


__all__ = [
    "DifferentiationLayer",
    "DifferentiationModule",
    "HarmonizationLayer",
    "SyntonicGate",
    "AdaptiveGate",
    "RecursionBlock",
    "DeepRecursionNet",
    "SyntonicNorm",
    "GoldenNorm",
    "ResonantLinear",
    "GnosisLayer",
    "GnosisGELU",
    "PrimeSyntonyGate",
    "WindingAttention",
    "SRTTransformerBlock",
    "get_stable_dimensions",
    "suggest_network_dimensions",
]
