"""
Syntonic Gate: Adaptive mixing based on local syntony.

Gate = σ(W_g·[x, H(D(x))])
Output = Gate · H(D(x)) + (1 - Gate) · x

Source: CRT.md §7.1
"""

from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
import math

PHI = (1 + math.sqrt(5)) / 2


class SyntonicGate(nn.Module):
    """
    Syntonic gating mechanism.

    Adaptively mixes input x with processed output x_harm based on
    how well the processing preserves/enhances syntony.

    Gate = σ(W_g · [x || x_harm])
    Output = Gate · x_harm + (1 - Gate) · x

    High gate → trust the processing (good syntony)
    Low gate → preserve input (processing degraded syntony)

    Example:
        >>> gate = SyntonicGate(256)
        >>> x = torch.randn(32, 256)
        >>> x_processed = torch.randn(32, 256)
        >>> y = gate(x, x_processed)
        >>> y.shape
        torch.Size([32, 256])
    """

    def __init__(self, d_model: int, hidden_dim: Optional[int] = None):
        """
        Initialize syntonic gate.

        Args:
            d_model: Model dimension
            hidden_dim: Hidden dimension for gate network (default: d_model)
        """
        super().__init__()
        hidden_dim = hidden_dim or d_model

        self.d_model = d_model
        self.gate_net = nn.Sequential(
            nn.Linear(d_model * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, d_model),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor, x_processed: torch.Tensor) -> torch.Tensor:
        """
        Apply syntonic gating.

        Args:
            x: Original input
            x_processed: Processed output (e.g., H(D(x)))

        Returns:
            Gated output
        """
        # Compute gate from concatenation
        concat = torch.cat([x, x_processed], dim=-1)
        gate = self.gate_net(concat)

        # Adaptive mixing
        return gate * x_processed + (1 - gate) * x

    def get_gate_values(self, x: torch.Tensor, x_processed: torch.Tensor) -> torch.Tensor:
        """Return gate values for analysis."""
        with torch.no_grad():
            concat = torch.cat([x, x_processed], dim=-1)
            return self.gate_net(concat)

    def extra_repr(self) -> str:
        return f'd_model={self.d_model}'


class AdaptiveGate(nn.Module):
    """
    Adaptive syntonic gate with learned syntony estimation.

    Estimates local syntony and uses it to modulate gating.
    More sophisticated than SyntonicGate.

    Example:
        >>> gate = AdaptiveGate(256)
        >>> x = torch.randn(32, 256)
        >>> x_diff = torch.randn(32, 256)
        >>> x_harm = torch.randn(32, 256)
        >>> y, syntony = gate(x, x_diff, x_harm, return_syntony=True)
    """

    def __init__(
        self,
        d_model: int,
        syntony_temp: float = 1.0,
        min_gate: float = 0.1,
        max_gate: float = 0.9,
    ):
        """
        Initialize adaptive gate.

        Args:
            d_model: Model dimension
            syntony_temp: Temperature for syntony-based modulation
            min_gate: Minimum gate value (always some processing)
            max_gate: Maximum gate value (always some preservation)
        """
        super().__init__()
        self.d_model = d_model
        self.syntony_temp = syntony_temp
        self.min_gate = min_gate
        self.max_gate = max_gate

        # Gate network
        self.gate_net = nn.Sequential(
            nn.Linear(d_model * 3, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )

        # Syntony estimator
        self.syntony_estimator = nn.Sequential(
            nn.Linear(d_model * 3, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, 1),
            nn.Sigmoid(),
        )

    def forward(
        self,
        x: torch.Tensor,
        x_diff: torch.Tensor,
        x_harm: torch.Tensor,
        return_syntony: bool = False,
    ):
        """
        Apply adaptive gating.

        Args:
            x: Original input
            x_diff: After differentiation D(x)
            x_harm: After harmonization H(D(x))
            return_syntony: Return estimated syntony

        Returns:
            Gated output (and optionally syntony estimate)
        """
        # Concatenate all three stages
        concat = torch.cat([x, x_diff, x_harm], dim=-1)

        # Estimate local syntony
        syntony = self.syntony_estimator(concat).squeeze(-1)

        # Compute base gate
        base_gate = torch.sigmoid(self.gate_net(concat))

        # Modulate by syntony: high syntony → higher gate
        syntony_mod = (syntony.unsqueeze(-1) - 0.5) * self.syntony_temp
        gate = base_gate * torch.sigmoid(syntony_mod + base_gate)

        # Clamp to [min, max]
        gate = self.min_gate + (self.max_gate - self.min_gate) * gate

        # Gated output
        output = gate * x_harm + (1 - gate) * x

        if return_syntony:
            return output, syntony.mean().item()
        return output

    def extra_repr(self) -> str:
        return f'd_model={self.d_model}, temp={self.syntony_temp}, gate_range=[{self.min_gate}, {self.max_gate}]'
