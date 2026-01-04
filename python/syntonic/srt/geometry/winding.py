"""
Winding State - Winding numbers on the T^4 torus.

A winding state |n> = |n_7, n_8, n_9, n_10> represents a configuration
of winding numbers on the internal 4-torus. These are the fundamental
quantum numbers in SRT from which all charges derive.

Uses exact golden ratio arithmetic from syntonic.exact.GoldenExact.

Example:
    >>> from syntonic.srt.geometry import winding_state
    >>> n = winding_state(1, 2, 0, -1)
    >>> n.norm_squared
    6
    >>> n.generation
    2
"""

from __future__ import annotations
import math
from typing import Tuple, Iterator, Optional, TYPE_CHECKING
from dataclasses import dataclass

from syntonic.exact import PHI, PHI_NUMERIC, GoldenExact

if TYPE_CHECKING:
    from syntonic.exact import GoldenExact as GoldenExactType


@dataclass(frozen=True, slots=True)
class WindingState:
    """
    Winding state |n> = |n_7, n_8, n_9, n_10> on T^4.

    Represents a configuration of integer winding numbers on the
    four internal circles of the torus. These quantum numbers
    determine charge, mass generation, and golden recursion properties.

    Attributes:
        n7: Winding number on S^1_7
        n8: Winding number on S^1_8
        n9: Winding number on S^1_9
        n10: Winding number on S^1_10

    Properties:
        n: Tuple of all four winding numbers
        norm_squared: |n|^2 = n_7^2 + n_8^2 + n_9^2 + n_10^2
        norm: |n| = sqrt(|n|^2)
        generation: Recursion depth k = 1 + floor(log_phi(max|n_i|))
        max_component: Maximum absolute value of components

    Example:
        >>> n = WindingState(1, 2, 0, -1)
        >>> n.n
        (1, 2, 0, -1)
        >>> n.norm_squared
        6
    """

    n7: int
    n8: int
    n9: int
    n10: int

    def __post_init__(self):
        """Validate that winding numbers are integers."""
        if not all(isinstance(x, int) for x in (self.n7, self.n8, self.n9, self.n10)):
            raise TypeError("Winding numbers must be integers")

    @property
    def n(self) -> Tuple[int, int, int, int]:
        """Tuple of all winding numbers (n_7, n_8, n_9, n_10)."""
        return (self.n7, self.n8, self.n9, self.n10)

    @property
    def norm_squared(self) -> int:
        """Squared norm |n|^2 = n_7^2 + n_8^2 + n_9^2 + n_10^2."""
        return self.n7 ** 2 + self.n8 ** 2 + self.n9 ** 2 + self.n10 ** 2

    @property
    def norm(self) -> float:
        """Euclidean norm |n|."""
        return math.sqrt(self.norm_squared)

    @property
    def max_component(self) -> int:
        """Maximum absolute value of winding components."""
        return max(abs(self.n7), abs(self.n8), abs(self.n9), abs(self.n10))

    @property
    def generation(self) -> int:
        """
        Recursion generation depth k.

        The generation determines the mass hierarchy: m ~ e^(-phi^k).
        For winding n, generation k = 1 + floor(log_phi(max|n_i|)) for |n| > 0,
        and k = 0 for the vacuum |n| = 0.

        Uses exact golden ratio φ from syntonic.exact.PHI.

        Returns:
            Generation number k >= 0
        """
        max_n = self.max_component
        if max_n == 0:
            return 0
        # k = 1 + floor(log_phi(max|n_i|))
        # Use exact phi value via PHI.eval()
        phi_val = PHI.eval()
        return 1 + int(math.log(max_n) / math.log(phi_val))

    def golden_weight(self, phi: Optional[float] = None) -> float:
        """
        Golden measure weight w(n) = exp(-|n|^2 / phi).

        Uses exact golden ratio φ from syntonic.exact.PHI by default.

        Args:
            phi: Golden ratio value (default: PHI.eval() from exact arithmetic)

        Returns:
            The golden weight for this winding state
        """
        if phi is None:
            phi = PHI.eval()  # Use exact golden ratio
        return math.exp(-self.norm_squared / phi)

    def inner_product(self, other: WindingState) -> int:
        """
        Inner product n . m = sum_i n_i * m_i.

        Args:
            other: Another WindingState

        Returns:
            Integer inner product
        """
        return (
            self.n7 * other.n7
            + self.n8 * other.n8
            + self.n9 * other.n9
            + self.n10 * other.n10
        )

    def __add__(self, other: WindingState) -> WindingState:
        """Vector addition of winding states."""
        if not isinstance(other, WindingState):
            return NotImplemented
        return WindingState(
            self.n7 + other.n7,
            self.n8 + other.n8,
            self.n9 + other.n9,
            self.n10 + other.n10,
        )

    def __sub__(self, other: WindingState) -> WindingState:
        """Vector subtraction of winding states."""
        if not isinstance(other, WindingState):
            return NotImplemented
        return WindingState(
            self.n7 - other.n7,
            self.n8 - other.n8,
            self.n9 - other.n9,
            self.n10 - other.n10,
        )

    def __neg__(self) -> WindingState:
        """Negation of winding state."""
        return WindingState(-self.n7, -self.n8, -self.n9, -self.n10)

    def __mul__(self, scalar: int) -> WindingState:
        """Scalar multiplication."""
        if not isinstance(scalar, int):
            return NotImplemented
        return WindingState(
            self.n7 * scalar,
            self.n8 * scalar,
            self.n9 * scalar,
            self.n10 * scalar,
        )

    def __rmul__(self, scalar: int) -> WindingState:
        """Right scalar multiplication."""
        return self.__mul__(scalar)

    def __iter__(self) -> Iterator[int]:
        """Iterate over winding components."""
        return iter(self.n)

    def __len__(self) -> int:
        """Number of components (always 4)."""
        return 4

    def __getitem__(self, index: int) -> int:
        """Index access to components."""
        return self.n[index]

    def is_zero(self) -> bool:
        """Check if this is the vacuum state |0,0,0,0>."""
        return self.n7 == 0 and self.n8 == 0 and self.n9 == 0 and self.n10 == 0

    @property
    def is_vacuum(self) -> bool:
        """True if this is the vacuum state |0,0,0,0>."""
        return self.is_zero()

    @classmethod
    def zero(cls) -> WindingState:
        """Create the vacuum state |0,0,0,0>."""
        return cls(0, 0, 0, 0)

    @classmethod
    def unit(cls, index: int) -> WindingState:
        """
        Create unit winding state along given axis.

        Args:
            index: 0-3 for n_7, n_8, n_9, n_10 respectively

        Returns:
            Unit winding state with 1 in the specified position

        Example:
            >>> WindingState.unit(0)  # |1,0,0,0>
            WindingState(n7=1, n8=0, n9=0, n10=0)
        """
        components = [0, 0, 0, 0]
        components[index] = 1
        return cls(*components)

    @classmethod
    def from_tuple(cls, t: Tuple[int, int, int, int]) -> WindingState:
        """Create from tuple."""
        return cls(*t)

    def to_tuple(self) -> Tuple[int, int, int, int]:
        """Convert to tuple."""
        return self.n

    def __repr__(self) -> str:
        return f"WindingState(n7={self.n7}, n8={self.n8}, n9={self.n9}, n10={self.n10})"

    def __str__(self) -> str:
        return f"|{self.n7},{self.n8},{self.n9},{self.n10}>"


def winding_state(
    n7: int, n8: int = 0, n9: int = 0, n10: int = 0
) -> WindingState:
    """
    Create a winding state |n_7, n_8, n_9, n_10>.

    Factory function for WindingState with optional default zeros.

    Args:
        n7: Winding number on S^1_7
        n8: Winding number on S^1_8 (default: 0)
        n9: Winding number on S^1_9 (default: 0)
        n10: Winding number on S^1_10 (default: 0)

    Returns:
        WindingState instance

    Example:
        >>> n = winding_state(1, 2)  # |1,2,0,0>
        >>> n.norm_squared
        5
    """
    return WindingState(n7, n8, n9, n10)


def enumerate_windings(max_norm: int = 10) -> Iterator[WindingState]:
    """
    Enumerate all winding states with |n| <= max_norm.

    Generates winding states in order of increasing norm,
    useful for lattice sums.

    Args:
        max_norm: Maximum norm to enumerate

    Yields:
        WindingState instances with norm <= max_norm

    Example:
        >>> list(enumerate_windings(1))
        [WindingState(0,0,0,0), WindingState(1,0,0,0), ...]
    """
    max_sq = max_norm * max_norm
    for n7 in range(-max_norm, max_norm + 1):
        for n8 in range(-max_norm, max_norm + 1):
            for n9 in range(-max_norm, max_norm + 1):
                for n10 in range(-max_norm, max_norm + 1):
                    n_sq = n7 * n7 + n8 * n8 + n9 * n9 + n10 * n10
                    if n_sq <= max_sq:
                        yield WindingState(n7, n8, n9, n10)
