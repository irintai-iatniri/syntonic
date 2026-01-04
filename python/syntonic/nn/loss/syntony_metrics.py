"""
Syntony computation for neural networks.

S_model = 1 - ||D(x) - x|| / (||D(x) - H(D(x))|| + ε)

Source: CRT.md §12.2, Syntonic Phase 3 Specification
"""

from __future__ import annotations
import torch
import torch.nn as nn
from typing import Optional, Tuple, List
import math

PHI = (1 + math.sqrt(5)) / 2
Q_DEFICIT = 0.027395146920


def compute_activation_syntony(
    x_input: torch.Tensor,
    x_diff: torch.Tensor,
    x_harm: torch.Tensor,
    epsilon: float = 1e-8,
) -> torch.Tensor:
    """
    Compute syntony from D/H activations.

    S = 1 - ||D(x) - x|| / (||D(x) - H(D(x))|| + ε)

    Args:
        x_input: Original input
        x_diff: After differentiation D(x)
        x_harm: After harmonization H(D(x))
        epsilon: Regularization constant

    Returns:
        Syntony value(s)
    """
    # Numerator: how much D changed x
    diff_change = torch.norm(x_diff - x_input, dim=-1)

    # Denominator: how much H corrected D
    harm_correction = torch.norm(x_diff - x_harm, dim=-1)

    # Syntony
    S = 1.0 - diff_change / (harm_correction + epsilon)

    return torch.clamp(S, 0.0, 1.0)


def compute_network_syntony(
    model: nn.Module,
    x: torch.Tensor,
    aggregation: str = 'mean',
) -> Tuple[float, List[float]]:
    """
    Compute syntony across entire network.

    Args:
        model: Neural network with RecursionBlocks
        x: Input tensor
        aggregation: How to aggregate ('mean', 'min', 'product')

    Returns:
        (global_syntony, layer_syntonies)
    """
    layer_syntonies = []

    # Forward pass collecting syntonies
    with torch.no_grad():
        for name, module in model.named_modules():
            if hasattr(module, 'syntony') and module.syntony is not None:
                layer_syntonies.append(module.syntony)

    if not layer_syntonies:
        return 0.5, []  # Default mid-syntony

    # Aggregate
    if aggregation == 'mean':
        global_S = sum(layer_syntonies) / len(layer_syntonies)
    elif aggregation == 'min':
        global_S = min(layer_syntonies)
    elif aggregation == 'product':
        import numpy as np
        global_S = float(np.prod(layer_syntonies) ** (1 / len(layer_syntonies)))
    else:
        global_S = sum(layer_syntonies) / len(layer_syntonies)

    return global_S, layer_syntonies


class SyntonyTracker:
    """
    Track syntony evolution during training.

    Monitors syntony trends and detects archonic patterns.

    Example:
        >>> tracker = SyntonyTracker()
        >>> for epoch in range(100):
        ...     tracker.update(model_syntony)
        >>> print(f"Mean: {tracker.mean_syntony:.4f}")
        >>> print(f"Trend: {tracker.syntony_trend:.4f}")
    """

    def __init__(self, window_size: int = 100):
        """
        Initialize tracker.

        Args:
            window_size: Size of moving window for statistics
        """
        self.window_size = window_size
        self.history: List[float] = []
        self.layer_histories: dict = {}

    def update(self, global_syntony: float, layer_syntonies: Optional[List[float]] = None):
        """
        Record syntony values.

        Args:
            global_syntony: Global model syntony
            layer_syntonies: Optional per-layer syntonies
        """
        self.history.append(global_syntony)
        if len(self.history) > self.window_size * 10:
            self.history = self.history[-self.window_size * 10:]

        if layer_syntonies:
            for i, s in enumerate(layer_syntonies):
                if i not in self.layer_histories:
                    self.layer_histories[i] = []
                self.layer_histories[i].append(s)
                if len(self.layer_histories[i]) > self.window_size * 10:
                    self.layer_histories[i] = self.layer_histories[i][-self.window_size * 10:]

    @property
    def mean_syntony(self) -> float:
        """Mean syntony over window."""
        if not self.history:
            return 0.0
        window = self.history[-self.window_size:]
        return sum(window) / len(window)

    @property
    def syntony_trend(self) -> float:
        """Syntony trend (positive = improving)."""
        if len(self.history) < 2:
            return 0.0

        half_window = self.window_size // 2
        if len(self.history) > self.window_size:
            recent = self.history[-half_window:]
            earlier = self.history[-self.window_size:-half_window]
        else:
            mid = len(self.history) // 2
            recent = self.history[mid:]
            earlier = self.history[:mid]

        if not earlier:
            return 0.0

        return sum(recent) / len(recent) - sum(earlier) / len(earlier)

    @property
    def variance(self) -> float:
        """Syntony variance over window."""
        if len(self.history) < 2:
            return 0.0
        window = self.history[-self.window_size:]
        mean = sum(window) / len(window)
        return sum((s - mean) ** 2 for s in window) / len(window)

    def is_archonic(self, threshold: float = 0.01, min_samples: int = 50) -> bool:
        """
        Detect if network is in Archonic (stuck) pattern.

        Archonic = cycling without syntony improvement

        Args:
            threshold: Variance threshold
            min_samples: Minimum samples needed

        Returns:
            True if archonic pattern detected
        """
        if len(self.history) < min_samples:
            return False

        # Check if syntony is oscillating without improvement
        mean_S = self.mean_syntony
        variance_S = self.variance
        trend = self.syntony_trend

        # Archonic: high variance, no trend, below target
        target = PHI - Q_DEFICIT - 0.1
        return (
            variance_S > threshold and
            abs(trend) < threshold / 10 and
            mean_S < target
        )

    def get_layer_stats(self, layer_idx: int) -> dict:
        """Get statistics for a specific layer."""
        if layer_idx not in self.layer_histories:
            return {'mean': 0.0, 'variance': 0.0, 'trend': 0.0}

        history = self.layer_histories[layer_idx]
        if not history:
            return {'mean': 0.0, 'variance': 0.0, 'trend': 0.0}

        window = history[-self.window_size:]
        mean = sum(window) / len(window)
        variance = sum((s - mean) ** 2 for s in window) / len(window)

        # Trend
        if len(window) > 1:
            mid = len(window) // 2
            recent = window[mid:]
            earlier = window[:mid]
            trend = sum(recent) / len(recent) - sum(earlier) / len(earlier) if earlier else 0.0
        else:
            trend = 0.0

        return {'mean': mean, 'variance': variance, 'trend': trend}

    def reset(self):
        """Reset tracker."""
        self.history = []
        self.layer_histories = {}
