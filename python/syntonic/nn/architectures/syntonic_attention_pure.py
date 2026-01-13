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
from syntonic.nn.resonant_tensor import ResonantTensor

PHI = (1 + math.sqrt(5)) / 2
PHI_INV = 1 / PHI


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
        # ResonantTensor.matmul(B) computes A @ B^T
        # query: [..., S, D], key: [..., S, D]
        # query.matmul(key) -> query @ key^T -> [..., S, S]
        scores = query.matmul(key)
        
        # Scale
        scores = scores.scalar_mul(1.0 / self.scale)

        # Softmax attention (per row)
        # Using Rust-backed softmax
        scores.softmax(precision=self.precision)
        attention = scores

        # Compute attention syntony (lower entropy = higher syntony)
        # We need values for entropy calculation.
        attention_data = attention.to_floats()
        seq_q = attention.shape[-2]
        seq_k = attention.shape[-1]
        
        self._attention_syntony = self._compute_attention_syntony(attention_data, seq_q, seq_k)

        # Apply attention to values: Attn @ V
        # attention: [..., S, S], value: [..., S, D]
        # We want result [..., S, D]
        # attention.matmul(X) -> attention @ X^T
        # So we need X^T = value => X = value^T
        output = attention.matmul(value.transpose(-2, -1))

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
        # Uses sn.Parameter which wraps ResonantTensor
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
        batch_size = query.shape[0] if len(query.shape) > 2 else 1
        seq_q = query.shape[0] if len(query.shape) == 2 else query.shape[1]
        seq_k = key.shape[0] if len(key.shape) == 2 else key.shape[1]
        
        # Project Q, K, V
        # If input is 3D [Batch, Seq, Dim], we need to handle it.
        # ResonantTensor matmul: X @ W^T
        # query: [..., Dim], W_q: [Dim, Dim] (parameter kept as [Out, In])
        
        q = query.matmul(self.q_proj.tensor)
        k = key.matmul(self.k_proj.tensor)
        v = value.matmul(self.v_proj.tensor)
        
        # Reshape for heads: [Batch, Seq, Heads, HeadDim]
        # Since we use 2D/3D flattening often, let's assume we reshape to split last dim
        
        # New shape: [Batch, Seq, Heads, HeadDim]
        # Or if 2D: [Seq, Heads, HeadDim]
        base_shape = q.shape[:-1]
        new_shape = list(base_shape) + [self.n_heads, self.d_head]
        
        q = q.view(new_shape)
        k = k.view(new_shape)
        v = v.view(new_shape)
        
        # Transpose for attention: [Batch, Heads, Seq, HeadDim]
        # Permute: (0, 2, 1, 3) for 4D
        is_batched = (len(new_shape) == 4)
        
        if is_batched:
            q = q.permute([0, 2, 1, 3])
            k = k.permute([0, 2, 1, 3])
            v = v.permute([0, 2, 1, 3])
            # Shape: [Batch, Heads, Seq, HeadDim]
        else:
            # 3D case: [Seq, Heads, HeadDim] -> [Heads, Seq, HeadDim]
            q = q.permute([1, 0, 2])
            k = k.permute([1, 0, 2])
            v = v.permute([1, 0, 2])
            # Shape: [Heads, Seq, HeadDim]

        # Process heads using BMM (Rust-backend accelerated)
        # Q, K, V are permuted to [Batch, Heads, Seq, HeadDim] (or 3D equivalent)
        
        # 1. Attention Scores: Q @ K^T
        # ResonantTensor.matmul(B) computes A @ B^T
        # So q.matmul(k) computes per-head, per-batch attention scores
        # Shape: [..., Heads, Seq, Seq]
        scores = q.matmul(k)
        
        scores = scores.scalar_mul(1.0 / self.scale)
        scores.softmax(precision=self.precision)
        attn = scores
        
        # 2. Output Projection: Attn @ V
        # We need Result = Attn @ V
        # matmul(B) does A @ B^T
        # So providing B = V^T results in A @ (V^T)^T = A @ V
        output_heads = attn.matmul(v.transpose(-2, -1))
        
        # Shape is now [..., Heads, Seq, HeadDim]

        # Permute back to [..., Seq, Heads, HeadDim]
        if is_batched:
            output_heads = output_heads.permute([0, 2, 1, 3])
        else:
            output_heads = output_heads.permute([1, 0, 2])

        # Contiguous/Reshape back to [..., d_model]
        # output_heads is now [..., Seq, Heads, HeadDim]
        # view -> [..., Seq, Heads*HeadDim] = [..., Seq, d_model]
        
        final_shape = list(base_shape) + [self.d_model]
        output = output_heads.view(final_shape)
        
        # Final projection
        output = output.matmul(self.out_proj.tensor)
        
        # Syntony tracking
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
    # Shape [seq_len, d_model]
    data = [random.gauss(0, 0.5) for _ in range(seq_len * d_model)]
    x = ResonantTensor(data, [seq_len, d_model])
    
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

    # Test Slicing (New Feature)
    print("\nTesting Slicing:")
    print(f"  x[0] shape: {x[0].shape}")
    print(f"  x[0:2] shape: {x[0:2].shape}")
    
    # Verify slicing integrity
    x_slice = x[0:2]
    assert x_slice.shape == [2, d_model]
    print("  Slicing verification passed")
    
    print("\n" + "=" * 70)
    print("✓ Pure Syntonic Attention and Tensor Features verified!")
    print("=" * 70)
