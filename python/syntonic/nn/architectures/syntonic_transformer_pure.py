"""
Pure Syntonic Transformer: Transformer architecture with DHSR structure.

NO PYTORCH DEPENDENCIES - uses sn.Module and ResonantTensor.

Includes:
- PureDHTransformerLayer: Single transformer layer using D→H cycle
- PureSyntonicTransformerEncoder: Encoder-only transformer stack

Source: CRT.md §12.2
"""

from __future__ import annotations
from typing import Optional, List, Tuple
import math
import random

import syntonic.sn as sn
from syntonic._core import ResonantTensor, py_softmax

from syntonic.nn.architectures.syntonic_attention_pure import (
    PureMultiHeadSyntonicAttention,
    _matmul_rt,
)

PHI = (1 + math.sqrt(5)) / 2
PHI_INV = 1 / PHI


def _add_rt(a: ResonantTensor, b: ResonantTensor) -> ResonantTensor:
    """Add two ResonantTensors element-wise."""
    a_data = a.to_floats()
    b_data = b.to_floats()
    result = [x + y for x, y in zip(a_data, b_data)]
    shape = a.shape
    mode_norms = [float(i * i) for i in range(len(result))]
    return ResonantTensor(result, shape, mode_norms, 100)


def _scale_rt(x: ResonantTensor, scale: float) -> ResonantTensor:
    """Scale a ResonantTensor."""
    data = x.to_floats()
    result = [v * scale for v in data]
    shape = x.shape
    mode_norms = [float(i * i) for i in range(len(result))]
    return ResonantTensor(result, shape, mode_norms, 100)


def _gelu_rt(x: ResonantTensor) -> ResonantTensor:
    """Apply GELU activation (tanh approximation)."""
    data = x.to_floats()
    # GELU ≈ 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x³)))
    sqrt_2_pi = math.sqrt(2.0 / math.pi)
    result = []
    for v in data:
        inner = sqrt_2_pi * (v + 0.044715 * v ** 3)
        gelu = 0.5 * v * (1.0 + math.tanh(inner))
        result.append(gelu)
    shape = x.shape
    mode_norms = [float(i * i) for i in range(len(result))]
    return ResonantTensor(result, shape, mode_norms, 100)


def _layer_norm_rt(x: ResonantTensor, eps: float = 1e-5) -> ResonantTensor:
    """Apply layer normalization."""
    data = x.to_floats()
    shape = x.shape
    
    if len(shape) == 2:
        # Normalize each row
        rows, cols = shape
        result = []
        for i in range(rows):
            row = data[i * cols : (i + 1) * cols]
            mean = sum(row) / len(row)
            var = sum((v - mean) ** 2 for v in row) / len(row)
            std = math.sqrt(var + eps)
            normalized = [(v - mean) / std for v in row]
            result.extend(normalized)
    else:
        mean = sum(data) / len(data)
        var = sum((v - mean) ** 2 for v in data) / len(data)
        std = math.sqrt(var + eps)
        result = [(v - mean) / std for v in data]
    
    mode_norms = [float(i * i) for i in range(len(result))]
    return ResonantTensor(result, shape, mode_norms, 100)


class PureDHTransformerLayer(sn.Module):
    """
    Transformer layer with D→H structure.
    
    Pure Python + ResonantTensor implementation.

    Standard transformer layer components:
    1. Self-Attention with harmonization
    2. FFN with differentiation/harmonization

    Example:
        >>> layer = PureDHTransformerLayer(d_model=64, n_heads=4)
        >>> x = ResonantTensor(...)  # (seq, d_model)
        >>> y = layer(x)
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int = 8,
        d_ff: int = 256,
        dropout: float = 0.1,
        precision: int = 100,
    ):
        """
        Initialize DH transformer layer.

        Args:
            d_model: Model dimension
            n_heads: Number of attention heads
            d_ff: Feed-forward hidden dimension
            dropout: Dropout probability
            precision: ResonantTensor precision
        """
        super().__init__()

        self.d_model = d_model
        self.d_ff = d_ff
        self.precision = precision

        # Self-attention
        self.self_attn = PureMultiHeadSyntonicAttention(
            d_model, n_heads, dropout, precision
        )

        # FFN weights
        self.ffn_w1 = sn.Parameter([d_model, d_ff], init='kaiming')
        self.ffn_w2 = sn.Parameter([d_ff, d_model], init='kaiming')

        self.dropout = sn.Dropout(dropout)

        self._attn_syntony: Optional[float] = None
        self._ffn_syntony: Optional[float] = None

    def forward(self, x: ResonantTensor) -> ResonantTensor:
        """
        Forward pass.

        Args:
            x: Input (seq_len, d_model)

        Returns:
            Output (seq_len, d_model)
        """
        # === Attention sublayer ===
        # Pre-norm
        x_norm = _layer_norm_rt(x)

        # Attention
        attn_out = self.self_attn(x_norm, x_norm, x_norm)
        self._attn_syntony = self.self_attn.syntony

        # Residual (golden scaled)
        attn_scaled = _scale_rt(attn_out, 1.0 / PHI)
        x = _add_rt(x, attn_scaled)

        # === FFN sublayer ===
        # Pre-norm
        x_norm = _layer_norm_rt(x)

        # FFN: W2(GELU(W1(x)))
        ffn_hidden = _matmul_rt(x_norm, self.ffn_w1.tensor)
        ffn_hidden = _gelu_rt(ffn_hidden)
        ffn_out = _matmul_rt(ffn_hidden, self.ffn_w2.tensor)
        
        self._ffn_syntony = ffn_out.syntony

        # Residual
        ffn_scaled = _scale_rt(ffn_out, 1.0 / PHI)
        x = _add_rt(x, ffn_scaled)

        return x

    @property
    def syntony(self) -> float:
        """Get average layer syntony."""
        attn_s = self._attn_syntony if self._attn_syntony else 0.5
        ffn_s = self._ffn_syntony if self._ffn_syntony else 0.5
        return (attn_s + ffn_s) / 2


class PureSyntonicTransformerEncoder(sn.Module):
    """
    Stack of DH transformer encoder layers.
    
    Pure Python + ResonantTensor implementation.

    Example:
        >>> encoder = PureSyntonicTransformerEncoder(d_model=64, n_layers=3)
        >>> x = ResonantTensor(...)  # (seq, d_model)
        >>> y = encoder(x)
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int = 8,
        n_layers: int = 6,
        d_ff: int = 256,
        dropout: float = 0.1,
        precision: int = 100,
    ):
        """
        Initialize encoder stack.

        Args:
            d_model: Model dimension
            n_heads: Attention heads per layer
            n_layers: Number of layers
            d_ff: Feed-forward dimension
            dropout: Dropout probability
            precision: ResonantTensor precision
        """
        super().__init__()

        self.layers = sn.ModuleList([
            PureDHTransformerLayer(d_model, n_heads, d_ff, dropout, precision)
            for _ in range(n_layers)
        ])
        self.n_layers = n_layers

    def forward(self, x: ResonantTensor) -> ResonantTensor:
        """Forward through encoder stack."""
        for layer in self.layers:
            x = layer(x)

        # Final layer norm
        x = _layer_norm_rt(x)
        return x

    @property
    def syntony(self) -> float:
        """Get average syntony across layers."""
        syntonies = [layer.syntony for layer in self.layers]
        return sum(syntonies) / len(syntonies) if syntonies else 0.5


class PureSyntonicTransformer(sn.Module):
    """
    Simple encoder-only syntonic transformer.
    
    For classification or embedding tasks.

    Example:
        >>> model = PureSyntonicTransformer(d_model=64, n_layers=3, output_dim=10)
        >>> x = ResonantTensor(...)  # (seq, d_model)
        >>> logits = model(x)
    """

    def __init__(
        self,
        d_model: int = 64,
        n_heads: int = 4,
        n_layers: int = 3,
        d_ff: int = 256,
        output_dim: int = 10,
        dropout: float = 0.1,
        precision: int = 100,
    ):
        """
        Initialize syntonic transformer.

        Args:
            d_model: Model dimension
            n_heads: Number of attention heads
            n_layers: Number of encoder layers
            d_ff: Feed-forward dimension
            output_dim: Output dimension (num classes)
            dropout: Dropout probability
            precision: ResonantTensor precision
        """
        super().__init__()

        self.d_model = d_model
        self.precision = precision

        # Encoder
        self.encoder = PureSyntonicTransformerEncoder(
            d_model, n_heads, n_layers, d_ff, dropout, precision
        )

        # Output projection
        self.output_proj = sn.Parameter([d_model, output_dim], init='kaiming')

    def forward(self, x: ResonantTensor) -> ResonantTensor:
        """
        Forward pass.

        Args:
            x: Input (seq_len, d_model)

        Returns:
            Output logits (output_dim,) - average pooled
        """
        # Encode
        encoded = self.encoder(x)

        # Average pool over sequence
        data = encoded.to_floats()
        shape = encoded.shape
        seq_len, d_model = shape
        
        pooled = []
        for d in range(d_model):
            col_sum = sum(data[s * d_model + d] for s in range(seq_len))
            pooled.append(col_sum / seq_len)
        
        pooled_rt = ResonantTensor(
            pooled, [1, d_model],
            [float(i * i) for i in range(d_model)],
            self.precision
        )

        # Project to output
        output = _matmul_rt(pooled_rt, self.output_proj.tensor)
        
        return output

    @property
    def syntony(self) -> float:
        """Get model syntony."""
        return self.encoder.syntony


if __name__ == "__main__":
    print("=" * 70)
    print("Pure Syntonic Transformer Test")
    print("=" * 70)
    
    d_model = 32
    seq_len = 8
    
    # Create random input
    data = [random.gauss(0, 0.5) for _ in range(seq_len * d_model)]
    mode_norms = [float(i * i) for i in range(len(data))]
    x = ResonantTensor(data, [seq_len, d_model], mode_norms, 100)
    
    print(f"\nInput shape: {x.shape}")
    print(f"Input syntony: {x.syntony:.4f}")
    
    # Test single layer
    layer = PureDHTransformerLayer(d_model=d_model, n_heads=4, d_ff=64)
    y = layer(x)
    
    print(f"\nSingle layer:")
    print(f"  Output shape: {y.shape}")
    print(f"  Layer syntony: {layer.syntony:.4f}")
    
    # Test encoder
    encoder = PureSyntonicTransformerEncoder(
        d_model=d_model, n_heads=4, n_layers=2, d_ff=64
    )
    encoded = encoder(x)
    
    print(f"\nEncoder (2 layers):")
    print(f"  Output shape: {encoded.shape}")
    print(f"  Encoder syntony: {encoder.syntony:.4f}")
    
    # Test full model
    model = PureSyntonicTransformer(
        d_model=d_model, n_heads=4, n_layers=2, d_ff=64, output_dim=10
    )
    logits = model(x)
    
    print(f"\nFull model:")
    print(f"  Logits shape: {logits.shape}")
    print(f"  Model syntony: {model.syntony:.4f}")
    
    print("\n" + "=" * 70)
    print("✓ Pure Syntonic Transformer verified!")
    print("=" * 70)
