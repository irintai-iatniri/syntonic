"""
Recursion Block: R̂ = Ĥ ∘ D̂ in neural network form.

R_layer(x) = H_layer(D_layer(x))

Complete DHSR cycle as a single neural block.

Source: CRT.md §12.2
"""

from __future__ import annotations
import torch
import torch.nn as nn
from typing import Optional, Tuple, List, Union

from syntonic.nn.layers.differentiation import DifferentiationLayer
from syntonic.nn.layers.harmonization import HarmonizationLayer
from syntonic.nn.layers.syntonic_gate import SyntonicGate


class RecursionBlock(nn.Module):
    """
    Complete DHSR recursion block.

    R̂[x] = Ĥ[D̂[x]]

    Implements one full cycle of:
    1. Differentiation (expand complexity)
    2. Harmonization (build coherence)
    3. Syntonic gating (adaptive mixing)

    This is the fundamental building block of syntonic networks.

    Example:
        >>> block = RecursionBlock(256)
        >>> x = torch.randn(32, 256)
        >>> y, syntony = block(x, return_syntony=True)
        >>> print(f"Block syntony: {syntony:.4f}")
    """

    def __init__(
        self,
        in_features: int,
        hidden_features: Optional[int] = None,
        out_features: Optional[int] = None,
        use_gate: bool = True,
        dropout: float = 0.1,
        alpha_scale: float = 1.0,
        beta_scale: float = 1.0,
        gamma_scale: float = 1.0,
    ):
        """
        Initialize recursion block.

        Args:
            in_features: Input dimension
            hidden_features: Hidden dimension for D/H layers
            out_features: Output dimension
            use_gate: Whether to use syntonic gating
            dropout: Dropout rate
            alpha_scale: Differentiation strength
            beta_scale: Damping strength
            gamma_scale: Syntony projection strength
        """
        super().__init__()
        hidden_features = hidden_features or in_features
        out_features = out_features or in_features

        # D̂ operator
        self.differentiate = DifferentiationLayer(
            in_features, hidden_features, alpha_scale=alpha_scale
        )

        # Ĥ operator
        self.harmonize = HarmonizationLayer(
            hidden_features, out_features,
            beta_scale=beta_scale, gamma_scale=gamma_scale
        )

        # Optional syntonic gate
        self.use_gate = use_gate
        if use_gate:
            self.gate = SyntonicGate(out_features)

        self.dropout = nn.Dropout(dropout)

        # Track syntony for this block
        self._last_syntony = None

    def forward(
        self,
        x: torch.Tensor,
        return_syntony: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, float]]:
        """
        Apply R̂ = Ĥ ∘ D̂.

        Args:
            x: Input tensor
            return_syntony: Whether to compute and return block syntony

        Returns:
            Output tensor (and optionally syntony value)
        """
        # D̂: Differentiate (expand complexity)
        x_diff = self.differentiate(x)

        # Ĥ: Harmonize (build coherence)
        x_harm = self.harmonize(x_diff)
        x_harm = self.dropout(x_harm)

        # Syntonic gating (adaptive mixing of input and output)
        if self.use_gate:
            x_out = self.gate(x, x_harm)
        else:
            x_out = x_harm

        # Compute syntony if requested
        syntony = None
        if return_syntony:
            syntony = self._compute_block_syntony(x, x_diff, x_harm)
            self._last_syntony = syntony

        if return_syntony:
            return x_out, syntony
        return x_out

    def _compute_block_syntony(
        self,
        x: torch.Tensor,
        x_diff: torch.Tensor,
        x_harm: torch.Tensor,
    ) -> float:
        """
        Compute block-level syntony.

        S_block = 1 - ||D(x) - x|| / (||D(x) - H(D(x))|| + ε)

        Derivation (from CRT.md §12.2):
        - Numerator ||D(x) - x||: measures differentiation magnitude
        - Denominator ||D(x) - H(D(x))||: measures harmonization effect
        - S → 1 when H perfectly integrates D's output
        - S → 0 when H has no effect (D output = H(D) output)

        Physical meaning:
        - High S: Network representations are coherent
        - Low S: Network representations are fragmented
        """
        with torch.no_grad():
            # ||D(x) - x||
            diff_norm = torch.norm(x_diff - x).item()

            # ||D(x) - H(D(x))||
            harm_diff_norm = torch.norm(x_diff - x_harm).item()

            # S = 1 - numerator / (denominator + ε)
            epsilon = 1e-8
            syntony = 1.0 - diff_norm / (harm_diff_norm + epsilon)

            return max(0.0, min(1.0, syntony))

    @property
    def syntony(self) -> Optional[float]:
        """Last computed block syntony."""
        return self._last_syntony


class DeepRecursionNet(nn.Module):
    """
    Deep network built from stacked RecursionBlocks.

    Implements n iterations of R̂, tracking syntony through layers.

    Example:
        >>> net = DeepRecursionNet(784, [512, 256, 128], 10)
        >>> x = torch.randn(32, 784)
        >>> y, syntonies = net(x, return_syntonies=True)
        >>> print(f"Mean syntony: {net.mean_syntony:.4f}")
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int],
        output_dim: int,
        use_gates: bool = True,
        dropout: float = 0.1,
    ):
        """
        Initialize deep recursion network.

        Args:
            input_dim: Input dimension
            hidden_dims: List of hidden dimensions
            output_dim: Output dimension
            use_gates: Use syntonic gating in blocks
            dropout: Dropout rate
        """
        super().__init__()

        dims = [input_dim] + hidden_dims + [output_dim]
        self.blocks = nn.ModuleList([
            RecursionBlock(dims[i], dims[i+1], dims[i+1], use_gate=use_gates, dropout=dropout)
            for i in range(len(dims) - 1)
        ])

        self._layer_syntonies = []

    def forward(
        self,
        x: torch.Tensor,
        return_syntonies: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, List[float]]]:
        """Forward through all recursion blocks."""
        self._layer_syntonies = []

        for block in self.blocks:
            x, syntony = block(x, return_syntony=True)
            self._layer_syntonies.append(syntony)

        if return_syntonies:
            return x, self._layer_syntonies
        return x

    @property
    def mean_syntony(self) -> float:
        """Mean syntony across all blocks."""
        if not self._layer_syntonies:
            return 0.0
        return sum(self._layer_syntonies) / len(self._layer_syntonies)

    @property
    def layer_syntonies(self) -> List[float]:
        """Syntony values for each layer."""
        return self._layer_syntonies
