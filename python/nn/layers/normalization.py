"""
Syntonic Normalization: Golden-ratio aware normalization layers.

These normalization layers incorporate golden ratio structure
for more natural regularization.

Source: CRT.md §7.1
"""

from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List
import math

PHI = (1 + math.sqrt(5)) / 2
PHI_INV = 1 / PHI
Q_DEFICIT = 0.027395146920


class SyntonicNorm(nn.Module):
    """
    Syntonic normalization layer.

    Similar to LayerNorm but with golden-ratio based scaling
    that promotes syntonic representations.

    The normalization target is φ - q (the syntony attractor)
    rather than zero-mean unit-variance.

    Example:
        >>> norm = SyntonicNorm(256)
        >>> x = torch.randn(32, 256)
        >>> y = norm(x)
        >>> y.shape
        torch.Size([32, 256])
    """

    def __init__(
        self,
        normalized_shape: int,
        eps: float = 1e-5,
        elementwise_affine: bool = True,
        golden_target: bool = True,
    ):
        """
        Initialize syntonic normalization.

        Args:
            normalized_shape: Shape to normalize over
            eps: Epsilon for numerical stability
            elementwise_affine: Learn affine parameters
            golden_target: Use golden-ratio based target
        """
        super().__init__()
        self.normalized_shape = normalized_shape
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        self.golden_target = golden_target

        if elementwise_affine:
            self.weight = nn.Parameter(torch.ones(normalized_shape))
            self.bias = nn.Parameter(torch.zeros(normalized_shape))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)

        # Golden ratio scale factor
        self.golden_scale = PHI_INV

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply syntonic normalization.

        Uses golden-ratio based scaling for more natural regularization.
        """
        # Compute mean and variance
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)

        # Normalize
        x_norm = (x - mean) / torch.sqrt(var + self.eps)

        # Golden-ratio scaling (optional)
        if self.golden_target:
            x_norm = x_norm * self.golden_scale

        # Affine transform
        if self.elementwise_affine:
            x_norm = x_norm * self.weight + self.bias

        return x_norm

    def extra_repr(self) -> str:
        return f'{self.normalized_shape}, eps={self.eps}, golden_target={self.golden_target}'


class GoldenNorm(nn.Module):
    """
    Golden ratio normalization.

    Normalizes activations using golden-ratio based statistics:
    - Target mean: 0
    - Target variance: 1/φ

    This creates representations that naturally align with
    syntonic theory predictions.

    Example:
        >>> norm = GoldenNorm(256)
        >>> x = torch.randn(32, 256)
        >>> y = norm(x)
        >>> # Variance should be approximately 1/φ ≈ 0.618
    """

    def __init__(
        self,
        num_features: int,
        momentum: float = 0.1,
        eps: float = 1e-5,
        affine: bool = True,
        track_running_stats: bool = True,
    ):
        """
        Initialize golden normalization.

        Args:
            num_features: Number of features
            momentum: Momentum for running stats
            eps: Epsilon for stability
            affine: Learn affine parameters
            track_running_stats: Track running mean/var
        """
        super().__init__()
        self.num_features = num_features
        self.momentum = momentum
        self.eps = eps
        self.affine = affine
        self.track_running_stats = track_running_stats

        # Target variance is 1/φ
        self.target_var = PHI_INV

        if affine:
            self.weight = nn.Parameter(torch.ones(num_features) * math.sqrt(self.target_var))
            self.bias = nn.Parameter(torch.zeros(num_features))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)

        if track_running_stats:
            self.register_buffer('running_mean', torch.zeros(num_features))
            self.register_buffer('running_var', torch.ones(num_features))
            self.register_buffer('num_batches_tracked', torch.tensor(0, dtype=torch.long))
        else:
            self.register_buffer('running_mean', None)
            self.register_buffer('running_var', None)
            self.register_buffer('num_batches_tracked', None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply golden normalization.
        """
        if self.training or not self.track_running_stats:
            # Use batch statistics
            mean = x.mean(dim=0)
            var = x.var(dim=0, unbiased=False)

            if self.track_running_stats:
                with torch.no_grad():
                    self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * mean
                    self.running_var = (1 - self.momentum) * self.running_var + self.momentum * var
                    self.num_batches_tracked += 1
        else:
            mean = self.running_mean
            var = self.running_var

        # Normalize to zero mean, unit variance
        x_norm = (x - mean) / torch.sqrt(var + self.eps)

        # Scale to target variance (1/φ) via affine
        if self.affine:
            x_norm = x_norm * self.weight + self.bias

        return x_norm

    def extra_repr(self) -> str:
        return f'{self.num_features}, target_var=1/φ≈{self.target_var:.4f}, momentum={self.momentum}'


class RecursionLayerNorm(nn.Module):
    """
    Layer normalization with recursion-aware statistics.

    Tracks how statistics evolve through recursion blocks
    and normalizes accordingly.

    Example:
        >>> norm = RecursionLayerNorm(256, n_recursions=4)
        >>> x = torch.randn(32, 256)
        >>> y = norm(x, recursion_depth=2)
    """

    def __init__(
        self,
        num_features: int,
        n_recursions: int = 4,
        eps: float = 1e-5,
    ):
        """
        Initialize recursion-aware layer norm.

        Args:
            num_features: Number of features
            n_recursions: Number of recursion depths to track
            eps: Epsilon for stability
        """
        super().__init__()
        self.num_features = num_features
        self.n_recursions = n_recursions
        self.eps = eps

        # Separate parameters for each recursion depth
        self.weights = nn.ParameterList([
            nn.Parameter(torch.ones(num_features))
            for _ in range(n_recursions)
        ])
        self.biases = nn.ParameterList([
            nn.Parameter(torch.zeros(num_features))
            for _ in range(n_recursions)
        ])

        # Golden decay factors for each depth
        self.decay_factors = [PHI ** (-i) for i in range(n_recursions)]

    def forward(self, x: torch.Tensor, recursion_depth: int = 0) -> torch.Tensor:
        """
        Apply recursion-aware normalization.

        Args:
            x: Input tensor
            recursion_depth: Current recursion depth (0-indexed)
        """
        depth = min(recursion_depth, self.n_recursions - 1)

        # Standard layer norm
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        x_norm = (x - mean) / torch.sqrt(var + self.eps)

        # Apply depth-specific parameters with golden decay
        decay = self.decay_factors[depth]
        x_norm = x_norm * self.weights[depth] * decay + self.biases[depth]

        return x_norm

    def extra_repr(self) -> str:
        return f'{self.num_features}, n_recursions={self.n_recursions}'
