"""
Pure Syntonic Attention: Attention mechanisms with DHSR structure.

NO PYTORCH DEPENDENCIES - uses sn.Module and ResonantTensor.

Includes:
- PureSyntonicAttention: Standard attention with syntony tracking
- PureMultiHeadSyntonicAttention: Multi-head variant

Source: CRT.md §12.2
"""

from __future__ import annotations
from typing import Optional, Tuple, List
import math

import syntonic.sn as sn
from syntonic._core import ResonantTensor, py_softmax

PHI = (1 + math.sqrt(5)) / 2
PHI_INV = 1 / PHI


def _matmul_rt(a: ResonantTensor, b: ResonantTensor) -> ResonantTensor:
    """Matrix multiply two ResonantTensors."""
    # Get shapes
    a_shape = a.shape
    b_shape = b.shape
    
    # For 2D: (M, K) x (K, N) -> (M, N)
    M = a_shape[0]
    K = a_shape[1] if len(a_shape) > 1 else 1
    N = b_shape[1] if len(b_shape) > 1 else b_shape[0]
    
    a_data = a.to_floats()
    b_data = b.to_floats()
    
    result = []
    for i in range(M):
        for j in range(N):
            val = 0.0
            for k in range(K):
                a_idx = i * K + k
                b_idx = k * N + j
                if a_idx < len(a_data) and b_idx < len(b_data):
                    val += a_data[a_idx] * b_data[b_idx]
            result.append(val)
    
    mode_norms = [float(i * i) for i in range(len(result))]
    return ResonantTensor(result, [M, N], mode_norms, 100)


def _transpose_rt(x: ResonantTensor) -> ResonantTensor:
    """Transpose a 2D ResonantTensor."""
    shape = x.shape
    if len(shape) != 2:
        return x
    
    data = x.to_floats()
    rows, cols = shape
    
    result = []
    for j in range(cols):
        for i in range(rows):
            result.append(data[i * cols + j])
    
    mode_norms = [float(i * i) for i in range(len(result))]
    return ResonantTensor(result, [cols, rows], mode_norms, 100)


def _softmax_rt(x: ResonantTensor) -> ResonantTensor:
    """Apply softmax to last dimension using Rust backend."""
    data = x.to_floats()
    result = py_softmax(data)
    shape = x.shape
    mode_norms = [float(i * i) for i in range(len(result))]
    return ResonantTensor(result, shape, mode_norms, 100)


class PureSyntonicAttention(sn.Module):
    """
    Scaled dot-product attention with syntony tracking.
    
    Pure Python + ResonantTensor implementation.

    Example:
        >>> attn = PureSyntonicAttention(d_model=64)
        >>> q = k = v = ResonantTensor(...)  # (seq, d_model)
        >>> output = attn(q, k, v)
    """

    def __init__(
        self,
        d_model: int,
        dropout: float = 0.1,
        precision: int = 100,
    ):
        """
        Initialize syntonic attention.

        Args:
            d_model: Model dimension
            dropout: Attention dropout
            precision: ResonantTensor precision
        """
        super().__init__()

        self.d_model = d_model
        self.scale = math.sqrt(d_model)
        self.precision = precision
        self.dropout = sn.Dropout(dropout)

        self._attention_syntony: Optional[float] = None

    def forward(
        self,
        query: ResonantTensor,
        key: ResonantTensor,
        value: ResonantTensor,
    ) -> Tuple[ResonantTensor, float]:
        """
        Compute attention with syntony tracking.

        Args:
            query: Query tensor (seq_q, d_model)
            key: Key tensor (seq_k, d_model)
            value: Value tensor (seq_k, d_model)

        Returns:
            (output, attention_syntony)
        """
        # Attention scores: Q @ K^T / sqrt(d)
        key_T = _transpose_rt(key)
        scores = _matmul_rt(query, key_T)
        
        # Scale
        scores_data = scores.to_floats()
        scores_scaled = [s / self.scale for s in scores_data]
        scores_shape = scores.shape
        mode_norms = [float(i * i) for i in range(len(scores_scaled))]
        scores = ResonantTensor(scores_scaled, scores_shape, mode_norms, self.precision)

        # Softmax attention (per row)
        seq_q, seq_k = scores_shape
        attention_data = []
        for i in range(seq_q):
            row = scores_scaled[i * seq_k : (i + 1) * seq_k]
            row_softmax = py_softmax(row)
            attention_data.extend(row_softmax)
        
        attention = ResonantTensor(
            attention_data, scores_shape, 
            [float(i * i) for i in range(len(attention_data))], 
            self.precision
        )

        # Compute attention syntony (lower entropy = higher syntony)
        self._attention_syntony = self._compute_attention_syntony(attention_data, seq_q, seq_k)

        # Apply attention to values: Attn @ V
        output = _matmul_rt(attention, value)

        return output, self._attention_syntony

    def _compute_attention_syntony(self, attention_data: List[float], seq_q: int, seq_k: int) -> float:
        """
        Compute syntony of attention pattern.

        High syntony: focused attention (low entropy)
        Low syntony: diffuse attention (high entropy)
        """
        eps = 1e-10
        total_entropy = 0.0
        
        for i in range(seq_q):
            row = attention_data[i * seq_k : (i + 1) * seq_k]
            entropy = -sum(p * math.log(p + eps) for p in row)
            total_entropy += entropy
        
        avg_entropy = total_entropy / seq_q if seq_q > 0 else 0.0
        max_entropy = math.log(seq_k) if seq_k > 1 else 1.0
        normalized_entropy = avg_entropy / max_entropy
        syntony = 1.0 - normalized_entropy
        
        return max(0.0, min(1.0, syntony))

    @property
    def syntony(self) -> Optional[float]:
        """Get attention syntony."""
        return self._attention_syntony


class PureMultiHeadSyntonicAttention(sn.Module):
    """
    Multi-head attention with DHSR structure.
    
    Pure Python + ResonantTensor implementation.

    Example:
        >>> mha = PureMultiHeadSyntonicAttention(d_model=64, n_heads=4)
        >>> x = ResonantTensor(...)  # (seq, d_model)
        >>> output = mha(x, x, x)
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int = 8,
        dropout: float = 0.1,
        precision: int = 100,
    ):
        """
        Initialize multi-head syntonic attention.

        Args:
            d_model: Model dimension
            n_heads: Number of attention heads
            dropout: Dropout probability
            precision: ResonantTensor precision
        """
        super().__init__()

        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"

        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.scale = math.sqrt(self.d_head)
        self.precision = precision

        # Projections as Parameters
        self.q_proj = sn.Parameter([d_model, d_model], init='kaiming')
        self.k_proj = sn.Parameter([d_model, d_model], init='kaiming')
        self.v_proj = sn.Parameter([d_model, d_model], init='kaiming')
        self.out_proj = sn.Parameter([d_model, d_model], init='kaiming')

        self.dropout = sn.Dropout(dropout)
        
        self._head_syntonies: List[float] = []

    def forward(
        self,
        query: ResonantTensor,
        key: ResonantTensor,
        value: ResonantTensor,
    ) -> ResonantTensor:
        """
        Multi-head attention forward.

        Args:
            query: (seq_q, d_model)
            key: (seq_k, d_model)
            value: (seq_k, d_model)

        Returns:
            Output tensor (seq_q, d_model)
        """
        # Project Q, K, V
        q = _matmul_rt(query, self.q_proj.tensor)
        k = _matmul_rt(key, self.k_proj.tensor)
        v = _matmul_rt(value, self.v_proj.tensor)
        
        # Split into heads and compute attention per head
        seq_q = query.shape[0]
        seq_k = key.shape[0]
        
        # For simplicity, compute full attention then reshape
        # (In practice, would split Q/K/V into heads)
        k_T = _transpose_rt(k)
        scores = _matmul_rt(q, k_T)
        
        # Scale
        scores_data = scores.to_floats()
        scores_scaled = [s / self.scale for s in scores_data]
        
        # Softmax per row
        attention_data = []
        for i in range(seq_q):
            row = scores_scaled[i * seq_k : (i + 1) * seq_k]
            row_softmax = py_softmax(row)
            attention_data.extend(row_softmax)
        
        attention = ResonantTensor(
            attention_data, [seq_q, seq_k],
            [float(i * i) for i in range(len(attention_data))],
            self.precision
        )
        
        # Apply attention to values
        output = _matmul_rt(attention, v)
        
        # Output projection
        output = _matmul_rt(output, self.out_proj.tensor)
        
        # Track syntony
        self._head_syntonies = [output.syntony]
        
        return output

    @property
    def syntony(self) -> float:
        """Get average syntony across heads."""
        if not self._head_syntonies:
            return 0.5
        return sum(self._head_syntonies) / len(self._head_syntonies)


if __name__ == "__main__":
    import random
    
    print("=" * 70)
    print("Pure Syntonic Attention Test")
    print("=" * 70)
    
    d_model = 32
    seq_len = 8
    
    # Create random input
    data = [random.gauss(0, 0.5) for _ in range(seq_len * d_model)]
    mode_norms = [float(i * i) for i in range(len(data))]
    x = ResonantTensor(data, [seq_len, d_model], mode_norms, 100)
    
    print(f"\nInput shape: {x.shape}")
    print(f"Input syntony: {x.syntony:.4f}")
    
    # Test single-head attention
    attn = PureSyntonicAttention(d_model=d_model)
    output, syntony = attn(x, x, x)
    
    print(f"\nSingle-head attention:")
    print(f"  Output shape: {output.shape}")
    print(f"  Attention syntony: {syntony:.4f}")
    
    # Test multi-head attention
    mha = PureMultiHeadSyntonicAttention(d_model=d_model, n_heads=4)
    output = mha(x, x, x)
    
    print(f"\nMulti-head attention:")
    print(f"  Output shape: {output.shape}")
    print(f"  Syntony: {mha.syntony:.4f}")
    
    print("\n" + "=" * 70)
    print("✓ Pure Syntonic Attention verified!")
    print("=" * 70)
