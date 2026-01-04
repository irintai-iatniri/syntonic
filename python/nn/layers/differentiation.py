"""
Differentiation Layer: D̂ operator in neural network form.

D̂[x] = x + ReLU(W_D·x + b_D)

- ReLU introduces non-linearity for complexity generation
- W_D weights serve as αᵢ coupling analogs
- Increases representational complexity (Fire/novelty)

Source: CRT.md §12.2
"""

from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class DifferentiationLayer(nn.Module):
    """
    Neural layer implementing differentiation D̂.

    D̂[x] = x + ReLU(W_D·x + b_D)

    The ReLU introduces non-linearity that generates complexity,
    analogous to D̂ exploring possibility spaces.

    Properties:
    - Increases dimensionality of representational manifold
    - Generates distinctions (Fire)
    - W_D weights control coupling strength to possibility projectors

    Example:
        >>> layer = DifferentiationLayer(256)
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
        alpha_scale: float = 1.0,
    ):
        """
        Initialize differentiation layer.

        Args:
            in_features: Input dimension
            out_features: Output dimension (defaults to in_features)
            bias: Include bias term
            alpha_scale: Scaling factor for differentiation strength
        """
        super().__init__()
        out_features = out_features or in_features

        self.in_features = in_features
        self.out_features = out_features
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        self.alpha_scale = alpha_scale

        # Initialize with small weights (gentle differentiation initially)
        nn.init.xavier_uniform_(self.linear.weight, gain=0.1)
        if bias:
            nn.init.zeros_(self.linear.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply D̂: x → x + α·ReLU(W_D·x + b_D)

        The residual connection preserves input while adding complexity.
        """
        # Differentiation: add complexity via nonlinearity
        d_x = self.alpha_scale * F.relu(self.linear(x))

        # Residual: D̂[x] = x + differentiation
        if d_x.shape == x.shape:
            return x + d_x
        else:
            # If dimensions change, use projection
            return d_x

    def complexity_increase(self, x: torch.Tensor) -> float:
        """Measure how much complexity D̂ added."""
        with torch.no_grad():
            d_x = self(x)
            return torch.norm(d_x - x).item() / (torch.norm(x).item() + 1e-8)

    def extra_repr(self) -> str:
        return f'in_features={self.in_features}, out_features={self.out_features}, alpha_scale={self.alpha_scale}'


class DifferentiationModule(nn.Module):
    """
    Multi-head differentiation for transformer architectures.

    Applies differentiation with multiple "possibility projectors",
    analogous to Σᵢ αᵢ P̂ᵢ[Ψ] in the continuous D̂ operator.

    Example:
        >>> module = DifferentiationModule(512, n_heads=8)
        >>> x = torch.randn(32, 100, 512)  # [batch, seq, d_model]
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
        Initialize multi-head differentiation.

        Args:
            d_model: Model dimension
            n_heads: Number of differentiation heads
            dropout: Dropout rate
        """
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads

        # Multi-head projections (possibility spaces)
        self.projectors = nn.ModuleList([
            nn.Linear(d_model, self.head_dim)
            for _ in range(n_heads)
        ])

        # Recombine
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Multi-head differentiation.

        Each head projects onto a different possibility subspace,
        then recombines with ReLU nonlinearity.
        """
        # Apply each projector
        heads = [F.relu(proj(x)) for proj in self.projectors]

        # Concatenate and project back
        concat = torch.cat(heads, dim=-1)
        out = self.out_proj(concat)
        out = self.dropout(out)

        # Residual + norm
        return self.norm(x + out)

    def extra_repr(self) -> str:
        return f'd_model={self.d_model}, n_heads={self.n_heads}'
