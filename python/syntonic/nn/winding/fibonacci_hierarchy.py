"""
Fibonacci Hierarchy - Network depth scaling following Fibonacci sequence.

The Fibonacci hierarchy implements golden ratio scaling for network depth,
where layer dimensions grow according to Fibonacci numbers: 1, 1, 2, 3, 5, 8, 13, ...

This creates a natural hierarchy with ratio F_{n+1}/F_n → φ (golden ratio).
"""

from __future__ import annotations
from typing import List


class FibonacciHierarchy:
    """
    Manages network depth following Fibonacci sequence.

    Layer widths scale as Fibonacci numbers, respecting golden ratio growth.
    Provides expansion factors for D-phase differentiation.

    Example:
        >>> fib = FibonacciHierarchy(max_depth=5)
        >>> layer_dims = fib.get_layer_dims(base_dim=64)
        >>> # [64, 64, 128, 192, 320, 512, 832] for depths 0-6
    """

    def __init__(self, max_depth: int = 5):
        """
        Initialize Fibonacci hierarchy.

        Args:
            max_depth: Maximum hierarchy depth (number of layers)
        """
        self.max_depth = max_depth
        # Generate Fibonacci sequence with extra elements for expansion factors
        self.fib_dims = self._fibonacci(max_depth + 2)

    def _fibonacci(self, n: int) -> List[int]:
        """
        Generate first n Fibonacci numbers.

        Args:
            n: Number of Fibonacci numbers to generate

        Returns:
            List of Fibonacci numbers [F_0, F_1, F_2, ...]
        """
        if n <= 0:
            return []
        elif n == 1:
            return [1]
        elif n == 2:
            return [1, 1]

        fib = [1, 1]
        for i in range(2, n):
            fib.append(fib[-1] + fib[-2])
        return fib

    def get_layer_dims(self, base_dim: int) -> List[int]:
        """
        Get layer dimensions following Fibonacci scaling.

        Args:
            base_dim: Base dimension (will be scaled by Fibonacci numbers)

        Returns:
            List of layer dimensions [base*F_0, base*F_1, base*F_2, ...]
        """
        return [base_dim * f for f in self.fib_dims[: self.max_depth + 1]]

    def get_expansion_factor(self, level: int) -> int:
        """
        Get expansion factor for D-phase at given level.

        The expansion factor is the next Fibonacci number in the sequence.

        Args:
            level: Hierarchy level (0-indexed)

        Returns:
            Expansion factor F_{level+1}
        """
        if level + 1 < len(self.fib_dims):
            return self.fib_dims[level + 1]
        else:
            # Beyond pre-computed range, compute on-the-fly
            return self.fib_dims[-1] + self.fib_dims[-2]

    @property
    def depth(self) -> int:
        """Get maximum depth."""
        return self.max_depth

    def __repr__(self) -> str:
        return f"FibonacciHierarchy(max_depth={self.max_depth}, fib={self.fib_dims[:self.max_depth+1]})"


if __name__ == "__main__":
    # Example usage
    fib = FibonacciHierarchy(max_depth=5)
    print(fib)
    print(f"Layer dims (base=64): {fib.get_layer_dims(64)}")
    print(f"Expansion factors: {[fib.get_expansion_factor(i) for i in range(5)]}")
