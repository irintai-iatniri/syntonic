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
import syntonic.sn as sn

PHI = (1 + math.sqrt(5)) / 2


class ResonantLinear(sn.Module):
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
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.precision = precision
        self.bias_enabled = bias

        # 1. Initialize weight parameter
        # We use 'golden' initialization which respects the lattice structure
        self.weight = sn.Parameter(
            shape=[out_features, in_features],
            init='golden',
            requires_grad=True,
            precision=precision
        )

        # 2. Initialize bias parameter
        if bias:
            self.bias = sn.Parameter(
                shape=[out_features],
                init='golden',
                requires_grad=True,
                precision=precision
            )
        else:
            self.register_buffer('bias', None)

    def forward(self, x: ResonantTensor) -> ResonantTensor:
        """
        Forward pass using native Q(φ) matrix multiplication.

        Args:
            x: Input tensor of shape [..., in_features]

        Returns:
            Output tensor of shape [..., out_features]
        """
        # Handle arbitrary input dimensions by flattening
        original_shape = x.shape
        if len(original_shape) > 2:
            # Flatten: [batch, seq, features] -> [batch*seq, features]
            # Use -1 to infer batch*seq size
            x_flat = x.view([-1, original_shape[-1]])

            # Y = X_flat @ W^T using matmul method
            out = x_flat.matmul(self.weight.tensor)

            # Add bias
            if self.bias is not None:
                out.add_bias(self.bias.tensor)

            # Restoring shape: [batch*seq, out_features] -> [batch, seq, out_features]
            new_shape = list(original_shape[:-1]) + [self.out_features]
            return out.view(new_shape)
        else:
            # Standard 2D case using matmul method
            out = x.matmul(self.weight.tensor)
            if self.bias is not None:
                out.add_bias(self.bias.tensor)
            return out

    def extra_repr(self) -> str:
        return f"in_features={self.in_features}, out_features={self.out_features}, bias={self.bias_enabled}"


if __name__ == "__main__":
    # Test the pure ResonantLinear
    print("Testing ResonantLinear...")
    
    layer = ResonantLinear(4, 8, bias=True)
    print(f"Layer: {layer}")
    print(f"Weight parameter: {layer.weight}")
    print(f"Weight tensor shape: {layer.weight.tensor.shape}")
    
    # Create input
    x_data = [0.5, 0.3, -0.2, 0.8] * 2  # batch of 2
    x = ResonantTensor(x_data, [2, 4])
    
    # Forward pass
    y = layer.forward(x)
    print(f"Output shape: {y.shape}")
    print(f"Output syntony: {y.syntony:.4f}")
    print("SUCCESS")
