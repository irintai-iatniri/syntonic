"""
Winding Syntony Computer - Compute syntony with winding-aware mode norms.

This module computes syntony S(Ψ) using the winding state structure:

    S(Ψ) = Σᵢ |ψᵢ|² × exp(-|nᵢ|²/φ) / Σᵢ |ψᵢ|²

where |nᵢ|² is the mode norm squared for each feature.
"""

from __future__ import annotations
import torch
import torch.nn as nn
import math

PHI = (1 + math.sqrt(5)) / 2  # Golden ratio ~ 1.618


class WindingSyntonyComputer(nn.Module):
    """
    Computes syntony with winding-aware mode structure.

    The syntony formula weights energy by golden measure:
        S(Ψ) = Σ |ψ_i|² w(nᵢ) / Σ |ψ_i|²

    where:
        w(n) = exp(-|n|²/φ)  # Golden weight

    High syntony (S → 1): Energy concentrated in low-norm modes
    Low syntony (S → 0): Energy scattered across high-norm modes

    Example:
        >>> computer = WindingSyntonyComputer(dim=64)
        >>> x = torch.randn(32, 64)  # Batch of activations
        >>> mode_norms = torch.arange(64).pow(2).float()
        >>> S = computer(x, mode_norms)
        >>> print(f"Syntony: {S:.4f}")
    """

    def __init__(self, dim: int):
        """
        Initialize syntony computer.

        Args:
            dim: Feature dimension
        """
        super().__init__()
        self.dim = dim

    def forward(
        self, x: torch.Tensor, mode_norms: torch.Tensor
    ) -> float:
        """
        Compute syntony over batch.

        Args:
            x: Activations tensor (batch, dim) or (dim,)
            mode_norms: Mode norm squared |n|² for each feature (dim,)

        Returns:
            Scalar syntony S ∈ [0, 1]
        """
        # Ensure x is 2D
        if x.dim() == 1:
            x = x.unsqueeze(0)

        # Energy per feature: |ψᵢ|²
        energy = x.pow(2)  # (batch, dim)

        # Golden weights: w(n) = exp(-|n|²/φ)
        weights = torch.exp(-mode_norms / PHI)  # (dim,)

        # Syntony formula
        weighted_energy = (energy * weights).sum()
        total_energy = energy.sum() + 1e-8  # Avoid division by zero

        syntony = (weighted_energy / total_energy).item()

        # Clamp to [0, 1]
        return max(0.0, min(1.0, syntony))

    def batch_syntony(
        self, x: torch.Tensor, mode_norms: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute syntony for each sample in batch separately.

        Args:
            x: Activations tensor (batch, dim)
            mode_norms: Mode norm squared |n|² for each feature (dim,)

        Returns:
            Syntony values for each sample (batch,)
        """
        # Energy per feature
        energy = x.pow(2)  # (batch, dim)

        # Golden weights
        weights = torch.exp(-mode_norms / PHI)  # (dim,)

        # Syntony per sample
        weighted_energy = (energy * weights).sum(dim=1)  # (batch,)
        total_energy = energy.sum(dim=1) + 1e-8  # (batch,)

        syntony = weighted_energy / total_energy  # (batch,)

        return syntony.clamp(0.0, 1.0)

    def extra_repr(self) -> str:
        return f"dim={self.dim}"


if __name__ == "__main__":
    # Example usage
    computer = WindingSyntonyComputer(dim=64)
    print(f"Syntony computer: {computer}")

    # Create sample data
    x = torch.randn(32, 64)  # Batch of 32
    mode_norms = torch.arange(64).pow(2).float()  # |n|² = 0, 1, 4, 9, 16, ...

    # Compute syntony
    S = computer(x, mode_norms)
    print(f"Batch syntony: {S:.4f}")

    # Per-sample syntony
    S_per_sample = computer.batch_syntony(x, mode_norms)
    print(f"Per-sample syntony (first 5): {S_per_sample[:5]}")
    print(f"Mean: {S_per_sample.mean():.4f}, Std: {S_per_sample.std():.4f}")

    # Test with concentrated energy (low modes)
    x_concentrated = torch.zeros(1, 64)
    x_concentrated[0, :8] = 1.0  # Energy in first 8 modes
    S_high = computer(x_concentrated, mode_norms)
    print(f"\nConcentrated energy syntony: {S_high:.4f}")

    # Test with scattered energy (high modes)
    x_scattered = torch.zeros(1, 64)
    x_scattered[0, -8:] = 1.0  # Energy in last 8 modes
    S_low = computer(x_scattered, mode_norms)
    print(f"Scattered energy syntony: {S_low:.4f}")
