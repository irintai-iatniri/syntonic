"""
Syntonic Attention: Attention mechanisms with DHSR structure.

Includes:
- SyntonicAttention: Standard attention with syntony tracking
- GnosisAttention: Attention weighted by syntony ("gnosis")
- MultiHeadSyntonicAttention: Multi-head variant

Source: CRT.md §12.2
"""

from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
import math

from syntonic.nn.layers import (
    DifferentiationLayer,
    HarmonizationLayer,
    SyntonicNorm,
)

PHI = (1 + math.sqrt(5)) / 2
PHI_INV = 1 / PHI


class SyntonicAttention(nn.Module):
    """
    Scaled dot-product attention with syntony tracking.

    Tracks syntony of attention patterns and applies
    harmonization to output.

    Example:
        >>> attn = SyntonicAttention(d_model=512)
        >>> q = k = v = torch.randn(32, 100, 512)
        >>> output, weights = attn(q, k, v)
    """

    def __init__(
        self,
        d_model: int,
        dropout: float = 0.1,
        harmonize_output: bool = True,
    ):
        """
        Initialize syntonic attention.

        Args:
            d_model: Model dimension
            dropout: Attention dropout
            harmonize_output: Apply harmonization to output
        """
        super().__init__()

        self.d_model = d_model
        self.scale = math.sqrt(d_model)
        self.dropout = nn.Dropout(dropout)

        self.harmonize_output = harmonize_output
        if harmonize_output:
            self.harm = HarmonizationLayer(d_model, d_model)
            self.norm = SyntonicNorm(d_model)

        self._attention_syntony: Optional[float] = None

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        return_attention: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Compute attention with syntony tracking.

        Args:
            query: Query tensor (batch, seq_q, d_model)
            key: Key tensor (batch, seq_k, d_model)
            value: Value tensor (batch, seq_k, d_model)
            mask: Optional attention mask
            return_attention: Return attention weights

        Returns:
            (output, attention_weights or None)
        """
        # Attention scores
        scores = torch.matmul(query, key.transpose(-2, -1)) / self.scale

        # Apply mask
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        # Softmax attention
        attention = F.softmax(scores, dim=-1)
        attention = self.dropout(attention)

        # Compute attention syntony
        self._attention_syntony = self._compute_attention_syntony(attention)

        # Apply attention to values
        output = torch.matmul(attention, value)

        # Harmonize output
        if self.harmonize_output:
            batch, seq_len, dim = output.shape
            flat = output.reshape(-1, dim)
            flat = self.harm(flat)
            flat = self.norm(flat)
            output = flat.reshape(batch, seq_len, dim)

        if return_attention:
            return output, attention
        return output, None

    def _compute_attention_syntony(self, attention: torch.Tensor) -> float:
        """
        Compute syntony of attention pattern.

        High syntony: focused attention (low entropy)
        Low syntony: diffuse attention (high entropy)
        """
        with torch.no_grad():
            # Compute entropy of attention distribution
            # H = -sum(p * log(p))
            eps = 1e-10
            entropy = -(attention * torch.log(attention + eps)).sum(dim=-1)

            # Max entropy is log(seq_len)
            max_entropy = math.log(attention.shape[-1])

            # Syntony is inverse of normalized entropy
            normalized_entropy = entropy.mean().item() / max_entropy
            syntony = 1.0 - normalized_entropy

            return max(0.0, min(1.0, syntony))

    @property
    def syntony(self) -> Optional[float]:
        """Get attention syntony."""
        return self._attention_syntony


class GnosisAttention(nn.Module):
    """
    Gnosis-weighted attention.

    Attention is modulated by a learned "gnosis" (knowledge)
    score that represents the syntony of each position.

    Higher gnosis positions have more influence.

    Example:
        >>> attn = GnosisAttention(d_model=512)
        >>> q = k = v = torch.randn(32, 100, 512)
        >>> output = attn(q, k, v)
    """

    def __init__(
        self,
        d_model: int,
        dropout: float = 0.1,
        gnosis_dim: int = 64,
    ):
        """
        Initialize gnosis attention.

        Args:
            d_model: Model dimension
            dropout: Attention dropout
            gnosis_dim: Dimension for gnosis computation
        """
        super().__init__()

        self.d_model = d_model
        self.scale = math.sqrt(d_model)
        self.dropout = nn.Dropout(dropout)

        # Gnosis (syntony) scoring
        self.gnosis_query = nn.Linear(d_model, gnosis_dim)
        self.gnosis_key = nn.Linear(d_model, gnosis_dim)
        self.gnosis_gate = nn.Linear(gnosis_dim, 1)

        # Output processing
        self.harm = HarmonizationLayer(d_model, d_model)

        self._syntony: Optional[float] = None

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute gnosis-weighted attention.

        Args:
            query: Query tensor (batch, seq_q, d_model)
            key: Key tensor (batch, seq_k, d_model)
            value: Value tensor (batch, seq_k, d_model)
            mask: Optional attention mask

        Returns:
            Output tensor
        """
        # Standard attention scores
        scores = torch.matmul(query, key.transpose(-2, -1)) / self.scale

        # Compute gnosis scores for keys
        gnosis_q = torch.tanh(self.gnosis_query(query))
        gnosis_k = torch.tanh(self.gnosis_key(key))

        # Gnosis interaction: (batch, seq_q, seq_k)
        gnosis_scores = torch.matmul(gnosis_q, gnosis_k.transpose(-2, -1))
        gnosis_weights = torch.sigmoid(self.gnosis_gate(gnosis_scores.unsqueeze(-1))).squeeze(-1)

        # Modulate attention by gnosis
        scores = scores * (1.0 + gnosis_weights)

        # Mask
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        # Attention weights
        attention = F.softmax(scores, dim=-1)
        attention = self.dropout(attention)

        # Apply attention
        output = torch.matmul(attention, value)

        # Harmonize
        batch, seq_len, dim = output.shape
        output = self.harm(output.reshape(-1, dim)).reshape(batch, seq_len, dim)

        # Track syntony
        self._syntony = gnosis_weights.mean().item()

        return output

    @property
    def syntony(self) -> Optional[float]:
        """Get gnosis-based syntony."""
        return self._syntony


class MultiHeadSyntonicAttention(nn.Module):
    """
    Multi-head attention with DHSR structure.

    Each head applies syntonic attention, with
    harmonization on the combined output.

    Example:
        >>> mha = MultiHeadSyntonicAttention(d_model=512, n_heads=8)
        >>> x = torch.randn(32, 100, 512)
        >>> output = mha(x, x, x)
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int = 8,
        dropout: float = 0.1,
        use_gnosis: bool = False,
    ):
        """
        Initialize multi-head syntonic attention.

        Args:
            d_model: Model dimension
            n_heads: Number of attention heads
            dropout: Dropout probability
            use_gnosis: Use gnosis attention per head
        """
        super().__init__()

        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"

        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.scale = math.sqrt(self.d_head)

        # Projections
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)

        # DHSR for output
        self.diff = DifferentiationLayer(d_model, d_model)
        self.harm = HarmonizationLayer(d_model, d_model)

        # Gnosis scoring if enabled
        self.use_gnosis = use_gnosis
        if use_gnosis:
            self.gnosis = nn.Linear(self.d_head, 1)

        self._head_syntonies: list = []

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Multi-head attention forward.

        Args:
            query: (batch, seq_q, d_model)
            key: (batch, seq_k, d_model)
            value: (batch, seq_k, d_model)
            mask: Optional attention mask

        Returns:
            Output tensor (batch, seq_q, d_model)
        """
        batch_size, seq_len, _ = query.shape

        # Project and reshape to (batch, n_heads, seq, d_head)
        q = self.q_proj(query).view(batch_size, -1, self.n_heads, self.d_head).transpose(1, 2)
        k = self.k_proj(key).view(batch_size, -1, self.n_heads, self.d_head).transpose(1, 2)
        v = self.v_proj(value).view(batch_size, -1, self.n_heads, self.d_head).transpose(1, 2)

        # Attention scores: (batch, n_heads, seq_q, seq_k)
        scores = torch.matmul(q, k.transpose(-2, -1)) / self.scale

        # Gnosis modulation
        if self.use_gnosis:
            gnosis = torch.sigmoid(self.gnosis(k)).squeeze(-1)  # (batch, n_heads, seq_k)
            scores = scores * (1.0 + gnosis.unsqueeze(-2))

        # Mask
        if mask is not None:
            if mask.dim() == 2:
                mask = mask.unsqueeze(0).unsqueeze(0)
            scores = scores.masked_fill(mask == 0, -1e9)

        # Attention weights
        attention = F.softmax(scores, dim=-1)
        attention = self.dropout(attention)

        # Track per-head syntony
        self._head_syntonies = self._compute_head_syntonies(attention)

        # Apply attention: (batch, n_heads, seq_q, d_head)
        output = torch.matmul(attention, v)

        # Reshape: (batch, seq_q, d_model)
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)

        # Output projection with DHSR
        output = self.out_proj(output)

        # Flatten for DHSR
        flat = output.reshape(-1, self.d_model)
        x_diff = self.diff(flat)
        x_harm = self.harm(x_diff)
        output = x_harm.reshape(batch_size, seq_len, self.d_model)

        return output

    def _compute_head_syntonies(self, attention: torch.Tensor) -> list:
        """Compute syntony per attention head."""
        with torch.no_grad():
            syntonies = []
            for h in range(self.n_heads):
                head_attn = attention[:, h]
                eps = 1e-10
                entropy = -(head_attn * torch.log(head_attn + eps)).sum(dim=-1)
                max_entropy = math.log(head_attn.shape[-1])
                syntony = 1.0 - entropy.mean().item() / max_entropy
                syntonies.append(max(0.0, min(1.0, syntony)))
            return syntonies

    @property
    def syntony(self) -> float:
        """Get average syntony across heads."""
        if not self._head_syntonies:
            return 0.5
        return sum(self._head_syntonies) / len(self._head_syntonies)

    @property
    def head_syntonies(self) -> list:
        """Get per-head syntony values."""
        return self._head_syntonies
