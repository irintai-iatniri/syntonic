"""
Prime Selection Layer - Möbius filtering for the hadron/matter channel.

This module implements filtering based on the Möbius function μ(n), which
encodes prime factorization structure. Only square-free indices (|μ(n)| = 1)
carry stable hadronic information.

From SRT theory: "Hadrons follow prime or prime-composite structure."
"""

from __future__ import annotations
import torch
import torch.nn as nn
from typing import List


class PrimeSelectionLayer(nn.Module):
    """
    Filters activations based on prime selection (Möbius function).

    The Möbius function μ(n) is:
    - μ(n) = 1 if n is square-free with even number of prime factors
    - μ(n) = -1 if n is square-free with odd number of prime factors
    - μ(n) = 0 if n has a squared prime factor

    Only indices with |μ(n)| = 1 (square-free numbers) are preserved.
    This implements the "hadron channel" filtering from SRT.

    Example:
        >>> layer = PrimeSelectionLayer(dim=100)
        >>> x = torch.randn(32, 100)
        >>> y = layer(x)  # Non-prime indices attenuated
    """

    def __init__(self, dim: int):
        """
        Initialize prime selection layer.

        Args:
            dim: Feature dimension (will compute μ(k) for k=1..dim)
        """
        super().__init__()

        self.dim = dim

        # Compute Möbius function values
        self.mobius_values = self._compute_mobius(dim)

        # Create prime mask: |μ(k)| = 1
        # This preserves square-free numbers (primes and their products)
        prime_mask = torch.tensor(
            [abs(mu) == 1 for mu in self.mobius_values], dtype=torch.float32
        )

        # Register as buffer (not a trainable parameter)
        self.register_buffer("prime_mask", prime_mask)

    def _compute_mobius(self, n: int) -> List[int]:
        """
        Compute Möbius function μ(k) for k = 1, 2, ..., n.

        Uses sieve-based algorithm for efficiency.

        Args:
            n: Maximum index

        Returns:
            List of Möbius values [μ(1), μ(2), ..., μ(n)]
        """
        if n <= 0:
            return []

        # Initialize: μ(k) = 0 for all k
        mu = [0] * (n + 1)
        mu[1] = 1

        # Sieve algorithm
        # For each i, update all multiples of i
        for i in range(1, n + 1):
            if mu[i] != 0:  # Only update if i is relevant
                for j in range(2 * i, n + 1, i):
                    mu[j] -= mu[i]

        # Return μ(1) through μ(n)
        return mu[1:]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply prime selection filtering.

        Args:
            x: Input tensor (..., dim)

        Returns:
            Filtered tensor (..., dim) with non-square-free indices attenuated
        """
        return x * self.prime_mask

    def get_prime_indices(self) -> List[int]:
        """
        Get indices where the prime mask is active (|μ| = 1).

        Returns:
            List of indices (0-indexed) that pass the filter
        """
        return [i for i in range(self.dim) if self.prime_mask[i].item() == 1.0]

    def get_composite_indices(self) -> List[int]:
        """
        Get indices where the prime mask is inactive (μ = 0).

        Returns:
            List of indices (0-indexed) that are attenuated
        """
        return [i for i in range(self.dim) if self.prime_mask[i].item() == 0.0]

    def extra_repr(self) -> str:
        n_primes = sum(self.prime_mask).item()
        return f"dim={self.dim}, active_indices={int(n_primes)}/{self.dim}"


if __name__ == "__main__":
    # Example usage
    layer = PrimeSelectionLayer(dim=20)

    print(f"Prime selection layer: {layer}")
    print(f"Möbius values (1-20): {layer.mobius_values}")
    print(f"Prime mask: {layer.prime_mask}")
    print(f"Active (square-free) indices: {layer.get_prime_indices()}")
    print(f"Inactive (composite) indices: {layer.get_composite_indices()}")

    # Test filtering
    x = torch.ones(1, 20)
    y = layer(x)
    print(f"\nInput:  {x}")
    print(f"Output: {y}")
