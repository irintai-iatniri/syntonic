"""
Resonant Linear Layer - Projection layer backed by Resonant Engine.

This module implements ResonantLinear, a pure Rust-backed linear layer
using ResonantTensor for weights and bias. All parameters natively
inhabit the golden field Q(φ).

NO PYTORCH OR NUMPY DEPENDENCIES.
"""

from __future__ import annotations
import math
import random
from typing import Optional, List

from syntonic._core import ResonantTensor

PHI = (1 + math.sqrt(5)) / 2


class ResonantLinear:
    """
    Linear transformation layer using Resonant Engine.
    
    Y = XW^T + b
    
    Weights and biases are stored as ResonantTensors, meaning they 
    natively inhabit the Q(φ) lattice with exact arithmetic.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        precision: int = 100,
    ):
        """
        Initialize resonant linear layer.

        Args:
            in_features: Size of each input sample
            out_features: Size of each output sample
            bias: If set to False, the layer will not learn an additive bias
            precision: Bit precision for exact lattice arithmetic
        """
        self.in_features = in_features
        self.out_features = out_features
        self.precision = precision
        self._has_bias = bias

        # 1. Initialize weight matrix
        # Mode norm: |n|² = (i + h) where i is input index and h is hidden/output index
        weight_norms = []
        weight_data = []
        stdv = 1.0 / math.sqrt(in_features)
        
        for o in range(out_features):
            for i in range(in_features):
                mode_norm = float(i + o)
                weight_norms.append(mode_norm)
                
                # Standard initialization with golden attenuation
                val = random.uniform(-stdv, stdv)
                scale = math.exp(-mode_norm / (2 * PHI))
                weight_data.append(val * scale)

        self.weight = ResonantTensor(
            weight_data,
            [out_features, in_features],
            weight_norms,
            precision
        )

        # 2. Initialize bias
        if bias:
            bias_norms = [float(o + in_features) for o in range(out_features)]
            bias_data = []
            for o in range(out_features):
                val = random.uniform(-stdv, stdv)
                scale = math.exp(-bias_norms[o] / (2 * PHI))
                bias_data.append(val * scale)
            
            self.bias = ResonantTensor(
                bias_data,
                [out_features],
                bias_norms,
                precision
            )
        else:
            self.bias = None

    def forward(self, x: ResonantTensor) -> ResonantTensor:
        """
        Forward pass using native Q(φ) matrix multiplication.
        
        Args:
            x: Input tensor of shape [batch, in_features]
            
        Returns:
            Output tensor of shape [batch, out_features]
        """
        # Y = X @ W^T
        out = x.matmul(self.weight)
        
        # Add bias if present
        if self.bias is not None:
            out.add_bias(self.bias)
        
        return out

    def parameters(self) -> List[ResonantTensor]:
        """Return list of learnable parameters."""
        if self.bias is not None:
            return [self.weight, self.bias]
        return [self.weight]

    def __repr__(self) -> str:
        return (
            f"ResonantLinear(in_features={self.in_features}, "
            f"out_features={self.out_features}, bias={self._has_bias})"
        )


if __name__ == "__main__":
    # Test the pure ResonantLinear
    print("Testing ResonantLinear...")
    
    layer = ResonantLinear(4, 8, bias=True)
    print(f"Layer: {layer}")
    print(f"Weight shape: {layer.weight.shape}")
    print(f"Bias shape: {layer.bias.shape if layer.bias else None}")
    
    # Create input
    x_data = [0.5, 0.3, -0.2, 0.8] * 2  # batch of 2
    x = ResonantTensor(x_data, [2, 4])
    
    # Forward pass
    y = layer.forward(x)
    print(f"Output shape: {y.shape}")
    print(f"Output syntony: {y.syntony:.4f}")
    print("SUCCESS")
