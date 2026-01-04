"""
Syntonic Embeddings: Embedding layers with DHSR structure.

Includes:
- SyntonicEmbedding: Token embeddings with harmonization
- WindingEmbedding: Embeddings using winding numbers
- PositionalEncoding: Golden ratio-based positional encodings

Source: CRT.md §12.2
"""

from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
import math

from syntonic.nn.layers import HarmonizationLayer, SyntonicNorm

PHI = (1 + math.sqrt(5)) / 2
PHI_INV = 1 / PHI


class SyntonicEmbedding(nn.Module):
    """
    Embedding layer with harmonization.

    Applies harmonization to embeddings to promote
    coherent representations from the start.

    Example:
        >>> embed = SyntonicEmbedding(vocab_size=10000, embed_dim=512)
        >>> tokens = torch.randint(0, 10000, (32, 128))
        >>> embeddings = embed(tokens)
    """

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        padding_idx: Optional[int] = None,
        harmonize: bool = True,
        scale_by_sqrt_dim: bool = True,
    ):
        """
        Initialize syntonic embedding.

        Args:
            num_embeddings: Vocabulary size
            embedding_dim: Embedding dimension
            padding_idx: Padding token index
            harmonize: Apply harmonization to embeddings
            scale_by_sqrt_dim: Scale embeddings by sqrt(dim)
        """
        super().__init__()

        self.embedding = nn.Embedding(
            num_embeddings,
            embedding_dim,
            padding_idx=padding_idx,
        )

        self.harmonize = harmonize
        self.scale_by_sqrt_dim = scale_by_sqrt_dim
        self.embedding_dim = embedding_dim

        if harmonize:
            self.harm = HarmonizationLayer(embedding_dim, embedding_dim)
            self.norm = SyntonicNorm(embedding_dim)

        # Golden initialization
        self._init_golden()

    def _init_golden(self):
        """Initialize embeddings with golden-scaled variance."""
        std = PHI_INV / math.sqrt(self.embedding_dim)
        nn.init.normal_(self.embedding.weight, mean=0, std=std)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Get embeddings for token indices.

        Args:
            x: Token indices (batch, seq_len)

        Returns:
            Embeddings (batch, seq_len, embed_dim)
        """
        embeddings = self.embedding(x)

        # Scale by sqrt(dim) for stable attention
        if self.scale_by_sqrt_dim:
            embeddings = embeddings * math.sqrt(self.embedding_dim)

        # Harmonize embeddings
        if self.harmonize:
            batch, seq_len, dim = embeddings.shape
            flat = embeddings.reshape(-1, dim)
            flat = self.harm(flat)
            flat = self.norm(flat)
            embeddings = flat.reshape(batch, seq_len, dim)

        return embeddings


class WindingEmbedding(nn.Module):
    """
    Embedding using winding number structure.

    Maps discrete tokens to continuous positions on
    a torus, using winding numbers for rich structure.

    The embedding is: e(t) = [cos(2πw₁t/V), sin(2πw₁t/V), ...]

    Where w_i are coprime winding numbers.

    Example:
        >>> embed = WindingEmbedding(vocab_size=10000, embed_dim=512)
        >>> tokens = torch.randint(0, 10000, (32, 128))
        >>> embeddings = embed(tokens)
    """

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        num_windings: int = 8,
        learnable: bool = True,
    ):
        """
        Initialize winding embedding.

        Args:
            num_embeddings: Vocabulary size
            embedding_dim: Output embedding dimension
            num_windings: Number of winding frequencies
            learnable: Make winding numbers learnable
        """
        super().__init__()

        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.num_windings = num_windings

        # Generate coprime winding numbers
        windings = self._generate_coprimes(num_windings)
        if learnable:
            self.windings = nn.Parameter(torch.tensor(windings, dtype=torch.float32))
        else:
            self.register_buffer('windings', torch.tensor(windings, dtype=torch.float32))

        # Project winding features to embedding dim
        winding_dim = 2 * num_windings  # cos and sin for each
        self.projection = nn.Linear(winding_dim, embedding_dim)
        self.norm = SyntonicNorm(embedding_dim)

    def _generate_coprimes(self, n: int) -> list:
        """Generate n coprime numbers using Fibonacci-like sequence."""
        coprimes = [1, 2]
        while len(coprimes) < n:
            # Use golden ratio for spacing
            next_val = int(coprimes[-1] * PHI) + 1
            # Ensure coprime with all previous
            while any(math.gcd(next_val, c) > 1 for c in coprimes):
                next_val += 1
            coprimes.append(next_val)
        return coprimes[:n]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute winding embeddings.

        Args:
            x: Token indices (batch, seq_len)

        Returns:
            Embeddings (batch, seq_len, embed_dim)
        """
        # Normalize token indices to [0, 1)
        t = x.float() / self.num_embeddings

        # Compute winding features
        features = []
        for w in self.windings:
            angle = 2 * math.pi * w * t
            features.append(torch.cos(angle))
            features.append(torch.sin(angle))

        # Stack: (batch, seq_len, 2*num_windings)
        winding_features = torch.stack(features, dim=-1)

        # Project to embedding dimension
        embeddings = self.projection(winding_features)
        embeddings = self.norm(embeddings)

        return embeddings


class PositionalEncoding(nn.Module):
    """
    Positional encoding with golden ratio frequencies.

    Uses golden ratio-spaced frequencies for richer
    positional information without aliasing.

    PE(pos, 2i) = sin(pos / φ^(2i/d))
    PE(pos, 2i+1) = cos(pos / φ^(2i/d))

    Example:
        >>> pe = PositionalEncoding(d_model=512, max_len=5000)
        >>> x = torch.randn(32, 100, 512)
        >>> x = pe(x)
    """

    def __init__(
        self,
        d_model: int,
        max_len: int = 5000,
        dropout: float = 0.1,
        use_golden: bool = True,
    ):
        """
        Initialize positional encoding.

        Args:
            d_model: Model dimension
            max_len: Maximum sequence length
            dropout: Dropout probability
            use_golden: Use golden ratio frequencies (vs standard)
        """
        super().__init__()

        self.dropout = nn.Dropout(p=dropout)
        self.use_golden = use_golden

        # Compute positional encodings
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)

        if use_golden:
            # Golden ratio frequencies
            div_term = PHI ** (torch.arange(0, d_model, 2).float() / d_model)
        else:
            # Standard sinusoidal
            div_term = torch.exp(
                torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
            )

        pe[:, 0::2] = torch.sin(position / div_term)
        pe[:, 1::2] = torch.cos(position / div_term)

        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Add positional encoding to input.

        Args:
            x: Input tensor (batch, seq_len, d_model)

        Returns:
            Tensor with positional encoding added
        """
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class RotaryEmbedding(nn.Module):
    """
    Rotary Position Embedding (RoPE) with golden frequencies.

    Applies rotation to query/key vectors for
    relative position encoding.

    Example:
        >>> rope = RotaryEmbedding(dim=64)
        >>> q = torch.randn(32, 8, 100, 64)  # batch, heads, seq, dim
        >>> k = torch.randn(32, 8, 100, 64)
        >>> q_rot, k_rot = rope(q, k)
    """

    def __init__(
        self,
        dim: int,
        max_len: int = 5000,
        base: float = PHI * 10000,
    ):
        """
        Initialize rotary embedding.

        Args:
            dim: Embedding dimension (must be even)
            max_len: Maximum sequence length
            base: Base for frequency computation
        """
        super().__init__()

        assert dim % 2 == 0, "Dimension must be even"

        self.dim = dim
        self.max_len = max_len
        self.base = base

        # Compute frequencies
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer('inv_freq', inv_freq)

        # Precompute sin/cos for efficiency
        self._build_cache(max_len)

    def _build_cache(self, seq_len: int):
        """Build sin/cos cache."""
        t = torch.arange(seq_len, device=self.inv_freq.device)
        freqs = torch.einsum('i,j->ij', t, self.inv_freq)
        emb = torch.cat([freqs, freqs], dim=-1)
        self.register_buffer('cos_cache', emb.cos())
        self.register_buffer('sin_cache', emb.sin())

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        seq_len: Optional[int] = None,
    ) -> tuple:
        """
        Apply rotary embedding to query and key.

        Args:
            q: Query tensor (..., seq_len, dim)
            k: Key tensor (..., seq_len, dim)
            seq_len: Sequence length (if not inferrable)

        Returns:
            (rotated_q, rotated_k)
        """
        seq_len = seq_len or q.shape[-2]

        if seq_len > self.max_len:
            self._build_cache(seq_len)

        cos = self.cos_cache[:seq_len]
        sin = self.sin_cache[:seq_len]

        q_rot = self._rotate(q, cos, sin)
        k_rot = self._rotate(k, cos, sin)

        return q_rot, k_rot

    def _rotate(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
    ) -> torch.Tensor:
        """Apply rotation."""
        # Split into even/odd
        x1, x2 = x[..., ::2], x[..., 1::2]

        # Rotate
        x_rotated = torch.cat([
            x1 * cos[..., ::2] - x2 * sin[..., ::2],
            x1 * sin[..., 1::2] + x2 * cos[..., 1::2],
        ], dim=-1)

        return x_rotated
