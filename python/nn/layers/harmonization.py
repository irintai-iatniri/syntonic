"""
Harmonization Layer: Ĥ operator in neural network form.

Ĥ[x] = x - σ(W_H·x + b_H) + tanh(W_S·x + b_S)

- Sigmoid (σ) damps excessive complexity
- Tanh stabilizes toward syntony projection
- Creates coherence and integration (Whispers)

Source: CRT.md §12.2
"""

from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class HarmonizationLayer(nn.Module):
    """
    Neural layer implementing harmonization Ĥ.

    Ĥ[x] = x - σ(W_H·x + b_H) + tanh(W_S·x + b_S)

    Two terms:
    1. -σ(W_H·x): Damping term (reduces excessive complexity)
    2. +tanh(W_S·x): Syntony projection (stabilizes toward coherence)

    Properties:
    - Reduces dissonance (damping)
    - Enhances coherence (syntony projection)
    - Creates stable, integrated representations (Whispers)

    Example:
        >>> layer = HarmonizationLayer(256)
        >>> x = torch.randn(32, 256)
        >>> y = layer(x)
        >>> y.shape
        torch.Size([32, 256])
    """

    def __init__(
        self,
        in_features: int,
        out_features: Optional[int] = None,
        bias: bool = True,
        beta_scale: float = 1.0,
        gamma_scale: float = 1.0,
    ):
        """
        Initialize harmonization layer.

        Args:
            in_features: Input dimension
            out_features: Output dimension
            bias: Include bias terms
            beta_scale: Damping strength (β in Ĥ formula)
            gamma_scale: Syntony projection strength (γ)
        """
        super().__init__()
        out_features = out_features or in_features

        self.in_features = in_features
        self.out_features = out_features

        # Damping pathway: -β·σ(W_H·x + b_H)
        self.damping = nn.Linear(in_features, out_features, bias=bias)
        self.beta_scale = beta_scale

        # Syntony projection: +γ·tanh(W_S·x + b_S)
        self.syntony_proj = nn.Linear(in_features, out_features, bias=bias)
        self.gamma_scale = gamma_scale

        # Initialize
        nn.init.xavier_uniform_(self.damping.weight, gain=0.1)
        nn.init.xavier_uniform_(self.syntony_proj.weight, gain=0.5)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply Ĥ: x → x - β·σ(W_H·x) + γ·tanh(W_S·x)
        """
        # Damping: reduce complexity
        damp = self.beta_scale * torch.sigmoid(self.damping(x))

        # Syntony projection: stabilize toward coherence
        syntony = self.gamma_scale * torch.tanh(self.syntony_proj(x))

        # Harmonization: x - damping + syntony
        if damp.shape == x.shape:
            return x - damp + syntony
        else:
            return syntony - damp

    def coherence_gain(self, x: torch.Tensor) -> float:
        """Measure coherence increase from harmonization."""
        with torch.no_grad():
            h_x = self(x)
            # Coherence as reduction in variance
            var_before = torch.var(x).item()
            var_after = torch.var(h_x).item()
            return (var_before - var_after) / (var_before + 1e-8)

    def extra_repr(self) -> str:
        return f'in_features={self.in_features}, out_features={self.out_features}, beta={self.beta_scale}, gamma={self.gamma_scale}'


class HarmonizationModule(nn.Module):
    """
    Multi-component harmonization for transformer architectures.

    Combines multiple harmonization pathways analogous to
    Σᵢ βᵢ Q̂ᵢ[Ψ] + γ Ŝ_op[Ψ] in continuous Ĥ.

    Example:
        >>> module = HarmonizationModule(512, n_heads=8)
        >>> x = torch.randn(32, 100, 512)
        >>> y = module(x)
        >>> y.shape
        torch.Size([32, 100, 512])
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        dropout: float = 0.1,
    ):
        """
        Initialize multi-component harmonization.

        Args:
            d_model: Model dimension
            n_heads: Number of harmonization components
            dropout: Dropout rate
        """
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads

        # Harmonization components
        self.damping_heads = nn.ModuleList([
            nn.Linear(d_model, d_model // n_heads)
            for _ in range(n_heads)
        ])

        self.syntony_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Multi-component harmonization."""
        # Damping from each head
        damps = [torch.sigmoid(head(x)) for head in self.damping_heads]
        total_damp = torch.cat(damps, dim=-1)

        # Syntony projection
        syntony = torch.tanh(self.syntony_proj(x))

        # Combine
        out = self.out_proj(syntony - 0.5 * total_damp)
        out = self.dropout(out)

        return self.norm(x + out)

    def extra_repr(self) -> str:
        return f'd_model={self.d_model}, n_heads={self.n_heads}'
