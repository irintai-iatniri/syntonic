"""
Syntonic MLP: Multi-layer perceptron with DHSR structure.

Each layer follows the DHSR cycle:
1. Differentiation (complexity expansion)
2. Harmonization (coherence building)
3. Recursion (integration)

Source: CRT.md §12.2
"""

from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List, Tuple, Union
import math

from syntonic.nn.layers import (
    DifferentiationLayer,
    HarmonizationLayer,
    RecursionBlock,
    SyntonicNorm,
)

PHI = (1 + math.sqrt(5)) / 2
Q_DEFICIT = 0.027395146920


class SyntonicLinear(nn.Module):
    """
    Linear layer with DHSR structure.

    Combines linear transformation with differentiation
    and harmonization for syntonic processing.

    Example:
        >>> layer = SyntonicLinear(256, 128)
        >>> x = torch.randn(32, 256)
        >>> y = layer(x)
        >>> print(f"Syntony: {layer.syntony:.4f}")
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        use_recursion: bool = True,
        dropout: float = 0.0,
        bias: bool = True,
    ):
        """
        Initialize syntonic linear layer.

        Args:
            in_features: Input dimension
            out_features: Output dimension
            use_recursion: Use full recursion block
            dropout: Dropout probability
            bias: Include bias terms
        """
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.use_recursion = use_recursion

        # Main linear transformation
        self.linear = nn.Linear(in_features, out_features, bias=bias)

        # DHSR structure
        if use_recursion:
            self.recursion = RecursionBlock(out_features)
        else:
            self.diff = DifferentiationLayer(out_features, out_features)
            self.harm = HarmonizationLayer(out_features, out_features)

        self.dropout = nn.Dropout(dropout) if dropout > 0 else None
        self.norm = SyntonicNorm(out_features)

        self._syntony: Optional[float] = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with DHSR processing.

        Args:
            x: Input tensor (batch, in_features)

        Returns:
            Output tensor (batch, out_features)
        """
        # Linear transformation
        x = self.linear(x)

        # DHSR processing
        if self.use_recursion:
            x = self.recursion(x)
            self._syntony = self.recursion.syntony
        else:
            x_diff = self.diff(x)
            x_harm = self.harm(x_diff)
            self._syntony = self._compute_syntony(x, x_diff, x_harm)
            x = x_harm

        # Normalization and dropout
        x = self.norm(x)
        if self.dropout is not None:
            x = self.dropout(x)

        return x

    def _compute_syntony(
        self,
        x: torch.Tensor,
        x_diff: torch.Tensor,
        x_harm: torch.Tensor,
    ) -> float:
        """Compute block syntony."""
        with torch.no_grad():
            diff_norm = torch.norm(x_diff - x).item()
            harm_diff_norm = torch.norm(x_diff - x_harm).item()
            S = 1.0 - diff_norm / (harm_diff_norm + 1e-8)
            return max(0.0, min(1.0, S))

    @property
    def syntony(self) -> Optional[float]:
        """Get layer syntony."""
        return self._syntony


class SyntonicMLP(nn.Module):
    """
    Multi-layer perceptron with syntonic structure.

    Each hidden layer uses RecursionBlocks for DHSR processing.
    Tracks syntony throughout the network.

    Example:
        >>> model = SyntonicMLP(784, [512, 256, 128], 10)
        >>> x = torch.randn(32, 784)
        >>> y = model(x)
        >>> print(f"Model syntony: {model.syntony:.4f}")
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int],
        output_dim: int,
        dropout: float = 0.1,
        use_recursion: bool = True,
        output_activation: Optional[str] = None,
    ):
        """
        Initialize syntonic MLP.

        Args:
            input_dim: Input feature dimension
            hidden_dims: List of hidden layer dimensions
            output_dim: Output dimension
            dropout: Dropout probability
            use_recursion: Use RecursionBlocks in hidden layers
            output_activation: Final activation ('softmax', 'sigmoid', None)
        """
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.output_dim = output_dim
        self.output_activation = output_activation

        # Build layers
        layers = []
        prev_dim = input_dim

        for i, hidden_dim in enumerate(hidden_dims):
            layers.append(SyntonicLinear(
                prev_dim,
                hidden_dim,
                use_recursion=use_recursion,
                dropout=dropout,
            ))
            prev_dim = hidden_dim

        self.hidden_layers = nn.ModuleList(layers)

        # Output layer (no recursion - just linear)
        self.output_layer = nn.Linear(prev_dim, output_dim)

        # Track syntony history
        self._layer_syntonies: List[float] = []

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through syntonic MLP.

        Args:
            x: Input tensor (batch, input_dim)

        Returns:
            Output tensor (batch, output_dim)
        """
        self._layer_syntonies = []

        # Hidden layers with DHSR
        for layer in self.hidden_layers:
            x = layer(x)
            if layer.syntony is not None:
                self._layer_syntonies.append(layer.syntony)

        # Output layer
        x = self.output_layer(x)

        # Output activation
        if self.output_activation == 'softmax':
            x = F.softmax(x, dim=-1)
        elif self.output_activation == 'sigmoid':
            x = torch.sigmoid(x)

        return x

    @property
    def syntony(self) -> float:
        """Get average model syntony."""
        if not self._layer_syntonies:
            return 0.5
        return sum(self._layer_syntonies) / len(self._layer_syntonies)

    @property
    def layer_syntonies(self) -> List[float]:
        """Get per-layer syntony values."""
        return self._layer_syntonies

    def get_intermediate_outputs(
        self,
        x: torch.Tensor,
    ) -> List[torch.Tensor]:
        """
        Get outputs from each layer for analysis.

        Args:
            x: Input tensor

        Returns:
            List of intermediate activations
        """
        outputs = [x]

        for layer in self.hidden_layers:
            x = layer(x)
            outputs.append(x)

        x = self.output_layer(x)
        outputs.append(x)

        return outputs


class DeepSyntonicMLP(nn.Module):
    """
    Deep syntonic MLP with skip connections.

    Uses residual connections scaled by golden ratio
    for stable deep training.

    Example:
        >>> model = DeepSyntonicMLP(784, 256, 10, depth=12)
        >>> x = torch.randn(32, 784)
        >>> y = model(x)
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        depth: int = 6,
        dropout: float = 0.1,
    ):
        """
        Initialize deep syntonic MLP.

        Args:
            input_dim: Input dimension
            hidden_dim: Hidden layer dimension (constant)
            output_dim: Output dimension
            depth: Number of hidden layers
            dropout: Dropout probability
        """
        super().__init__()

        self.input_proj = nn.Linear(input_dim, hidden_dim)

        # Stack of recursion blocks with residuals
        self.blocks = nn.ModuleList([
            RecursionBlock(hidden_dim, dropout=dropout)
            for _ in range(depth)
        ])

        self.output_proj = nn.Linear(hidden_dim, output_dim)
        self.norm = SyntonicNorm(hidden_dim)

        # Golden scaling for residuals
        self._residual_scale = 1.0 / PHI

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with golden-scaled residuals."""
        # Project to hidden dimension
        x = self.input_proj(x)

        # Stack of recursion blocks
        for block in self.blocks:
            residual = x
            x = block(x)
            # Golden-scaled residual
            x = x + self._residual_scale * residual

        # Output
        x = self.norm(x)
        x = self.output_proj(x)

        return x

    @property
    def syntony(self) -> float:
        """Get average syntony across blocks."""
        syntonies = [b.syntony for b in self.blocks if b.syntony is not None]
        return sum(syntonies) / len(syntonies) if syntonies else 0.5
