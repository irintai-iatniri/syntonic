"""
E8 Lattice - The E8 root lattice with 240 roots.

E8 is the largest exceptional simple Lie group and plays a central
role in SRT. Its 240 roots organize into 120 positive roots, with
36 in the golden cone (= Phi+(E6)) defining the physical particle spectrum.

Uses exact rational arithmetic from syntonic.exact.Rational.

Example:
    >>> from syntonic.srt.lattice import e8_lattice
    >>> E8 = e8_lattice()
    >>> len(E8.roots)
    240
    >>> len(E8.positive_roots)
    120
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Tuple, List, Iterator, Optional, TYPE_CHECKING
from fractions import Fraction
import math

from syntonic.exact import Rational

if TYPE_CHECKING:
    from syntonic.exact import Rational as RationalType


@dataclass(frozen=True, slots=True)
class E8Root:
    """
    A root of the E8 lattice with exact rational coordinates.

    E8 roots come in two types:
    - Type A: (+/-1, +/-1, 0, 0, 0, 0, 0, 0) and permutations (112 roots)
    - Type B: (+/-1/2, ..., +/-1/2) with even number of minus signs (128 roots)

    All roots have |lambda|^2 = 2.

    Uses Python's fractions.Fraction for exact rational arithmetic,
    which is compatible with syntonic.exact.Rational for conversions.

    Attributes:
        coords: 8-tuple of Fraction coordinates

    Properties:
        norm_squared: Always 2 for roots
        root_type: 'A' or 'B'
        dimension: Always 8
    """

    coords: Tuple[Fraction, ...]

    def __post_init__(self):
        """Validate this is a valid E8 root."""
        if len(self.coords) != 8:
            raise ValueError("E8 root must have 8 coordinates")

    @property
    def dimension(self) -> int:
        """Dimension of the space (always 8)."""
        return 8

    @property
    def norm_squared(self) -> Fraction:
        """Squared norm |lambda|^2 (always 2 for roots)."""
        return sum(x * x for x in self.coords)

    @property
    def norm(self) -> float:
        """Euclidean norm |lambda|."""
        return math.sqrt(float(self.norm_squared))

    @property
    def root_type(self) -> str:
        """
        Root type: 'A' (integer coords) or 'B' (half-integer coords).
        """
        # Check if coordinates are integers or half-integers
        first_coord = self.coords[0]
        if first_coord.denominator == 1:
            return 'A'
        return 'B'

    def to_float(self) -> Tuple[float, ...]:
        """Convert to float tuple."""
        return tuple(float(x) for x in self.coords)

    def to_list(self) -> List[float]:
        """Convert to float list."""
        return [float(x) for x in self.coords]

    def to_rational_tuple(self) -> Tuple[Rational, ...]:
        """
        Convert coordinates to syntonic.exact.Rational tuple.

        Returns:
            Tuple of Rational values from syntonic.exact
        """
        return tuple(
            Rational(int(x.numerator), int(x.denominator))
            for x in self.coords
        )

    def norm_squared_rational(self) -> Rational:
        """
        Compute squared norm as syntonic.exact.Rational.

        Returns:
            |lambda|^2 as Rational (always Rational(2, 1) for roots)
        """
        result = Rational(0, 1)
        for x in self.coords:
            r = Rational(int(x.numerator), int(x.denominator))
            result = result + r * r
        return result

    def inner_product(self, other: E8Root) -> Fraction:
        """
        Inner product <lambda, mu>.

        Args:
            other: Another E8 root

        Returns:
            Rational inner product
        """
        return sum(a * b for a, b in zip(self.coords, other.coords))

    def inner_product_float(self, other: E8Root) -> float:
        """Float version of inner product."""
        return float(self.inner_product(other))

    def __neg__(self) -> E8Root:
        """Negation -lambda."""
        return E8Root(tuple(-x for x in self.coords))

    def __add__(self, other: E8Root) -> Tuple[Fraction, ...]:
        """Vector addition (returns tuple, may not be a root)."""
        return tuple(a + b for a, b in zip(self.coords, other.coords))

    def __sub__(self, other: E8Root) -> Tuple[Fraction, ...]:
        """Vector subtraction."""
        return tuple(a - b for a, b in zip(self.coords, other.coords))

    def __iter__(self) -> Iterator[Fraction]:
        """Iterate over coordinates."""
        return iter(self.coords)

    def __getitem__(self, index: int) -> Fraction:
        """Index access to coordinates."""
        return self.coords[index]

    def __str__(self) -> str:
        # Format nicely for half-integers
        def fmt(x):
            if x.denominator == 1:
                return str(x.numerator)
            return f"{x.numerator}/{x.denominator}"
        formatted = ", ".join(fmt(x) for x in self.coords)
        return f"E8({formatted})"

    def __repr__(self) -> str:
        return self.__str__()


class E8Lattice:
    """
    The E8 root lattice with 240 roots.

    E8 = {(x1,...,x8) : all xi in Z or all xi in Z+1/2, sum(xi) even}

    Key properties:
        - 240 roots (kissing number K = 240)
        - 120 positive roots
        - 36 roots in golden cone (= |Phi+(E6)|)
        - Rank 8
        - Dimension 248 (adjoint)
        - Coxeter number 30
        - Self-dual, even, unimodular

    Example:
        >>> E8 = E8Lattice()
        >>> len(E8.roots)
        240
        >>> all(r.norm_squared == 2 for r in E8.roots)
        True
    """

    def __init__(self):
        """Initialize E8 lattice by computing all roots."""
        self._roots = self._generate_roots()
        self._positive_roots = [r for r in self._roots if self._is_positive(r)]
        self._type_a_roots = [r for r in self._roots if r.root_type == 'A']
        self._type_b_roots = [r for r in self._roots if r.root_type == 'B']

    def _generate_roots(self) -> List[E8Root]:
        """Generate all 240 E8 roots."""
        roots = []

        # Type A: (+/-1, +/-1, 0, 0, 0, 0, 0, 0) and permutations
        # Choose 2 positions from 8, then 4 sign combinations
        # C(8,2) * 4 = 28 * 4 = 112 roots
        from itertools import combinations, product

        for positions in combinations(range(8), 2):
            for signs in product([-1, 1], repeat=2):
                coords = [Fraction(0)] * 8
                coords[positions[0]] = Fraction(signs[0])
                coords[positions[1]] = Fraction(signs[1])
                roots.append(E8Root(tuple(coords)))

        # Type B: (+/-1/2, ..., +/-1/2) with even number of minus signs
        # 2^8 = 256 sign combinations, half have even # of minuses = 128 roots
        half = Fraction(1, 2)
        for signs in product([-1, 1], repeat=8):
            # Count number of -1 signs
            num_minus = sum(1 for s in signs if s == -1)
            if num_minus % 2 == 0:  # Even number of minus signs
                coords = tuple(Fraction(s) * half for s in signs)
                roots.append(E8Root(coords))

        return roots

    def _is_positive(self, root: E8Root) -> bool:
        """
        Check if root is positive using lexicographic order.

        A root is positive if the first nonzero coordinate is positive.
        """
        for x in root.coords:
            if x > 0:
                return True
            if x < 0:
                return False
        return False

    @property
    def roots(self) -> List[E8Root]:
        """All 240 roots of E8."""
        return self._roots.copy()

    @property
    def positive_roots(self) -> List[E8Root]:
        """120 positive roots of E8."""
        return self._positive_roots.copy()

    @property
    def type_a_roots(self) -> List[E8Root]:
        """112 Type A roots (integer coordinates)."""
        return self._type_a_roots.copy()

    @property
    def type_b_roots(self) -> List[E8Root]:
        """128 Type B roots (half-integer coordinates)."""
        return self._type_b_roots.copy()

    @property
    def num_roots(self) -> int:
        """Number of roots (240)."""
        return len(self._roots)

    @property
    def kissing_number(self) -> int:
        """Kissing number K(E8) = 240."""
        return 240

    @property
    def dimension(self) -> int:
        """Dimension of the adjoint representation (248)."""
        return 248

    @property
    def rank(self) -> int:
        """Rank of E8 (8)."""
        return 8

    @property
    def coxeter_number(self) -> int:
        """Coxeter number of E8 (30)."""
        return 30

    def simple_roots(self) -> List[E8Root]:
        """
        Return 8 simple roots (basis for positive roots).

        Standard choice (Bourbaki convention):
            alpha_1 = (1, -1, 0, 0, 0, 0, 0, 0)
            alpha_2 = (0, 1, -1, 0, 0, 0, 0, 0)
            ...
            alpha_7 = (0, 0, 0, 0, 0, 1, -1, 0)
            alpha_8 = (-1/2, -1/2, -1/2, -1/2, -1/2, 1/2, 1/2, 1/2)
        """
        half = Fraction(1, 2)
        simple = []

        # First 7 simple roots: (0,...,0,1,-1,0,...,0)
        for i in range(7):
            coords = [Fraction(0)] * 8
            coords[i] = Fraction(1)
            coords[i + 1] = Fraction(-1)
            simple.append(E8Root(tuple(coords)))

        # 8th simple root
        coords = [-half, -half, -half, -half, -half, half, half, half]
        simple.append(E8Root(tuple(coords)))

        return simple

    def cartan_matrix(self) -> List[List[int]]:
        """
        Return 8x8 Cartan matrix of E8.

        A_ij = 2 * <alpha_i, alpha_j> / <alpha_j, alpha_j>
        """
        simple = self.simple_roots()
        n = len(simple)
        A = [[0] * n for _ in range(n)]
        for i in range(n):
            for j in range(n):
                numer = 2 * simple[i].inner_product(simple[j])
                denom = simple[j].norm_squared
                A[i][j] = int(numer / denom)
        return A

    def highest_root(self) -> E8Root:
        """
        Return the highest root of E8.

        theta = (1, 1, 0, 0, 0, 0, 0, 0) in standard conventions.
        """
        coords = [Fraction(1), Fraction(1)] + [Fraction(0)] * 6
        return E8Root(tuple(coords))

    def roots_by_shell(self) -> dict:
        """
        Group roots by squared norm.

        For E8, all roots have norm^2 = 2.

        Returns:
            Dictionary mapping squared norm to list of roots
        """
        shells = {}
        for root in self._roots:
            n_sq = root.norm_squared
            if n_sq not in shells:
                shells[n_sq] = []
            shells[n_sq].append(root)
        return shells

    def count_roots_by_type(self) -> dict:
        """
        Count roots by type.

        Returns:
            Dictionary with counts for each type
        """
        return {
            'total': len(self._roots),
            'positive': len(self._positive_roots),
            'type_A': len(self._type_a_roots),
            'type_B': len(self._type_b_roots),
        }

    def dynkin_labels(self, root: E8Root) -> List[int]:
        """
        Compute Dynkin labels for a root.

        Expresses root as linear combination of simple roots.

        Args:
            root: An E8 root

        Returns:
            List of 8 integer Dynkin labels
        """
        # For roots, we can use the Cartan matrix inverse
        # This is a simplified version - full implementation would
        # solve the linear system
        simple = self.simple_roots()
        labels = []
        for alpha in simple:
            coeff = 2 * root.inner_product(alpha) / alpha.norm_squared
            labels.append(int(coeff))
        return labels

    def __len__(self) -> int:
        """Number of roots."""
        return len(self._roots)

    def __iter__(self) -> Iterator[E8Root]:
        """Iterate over roots."""
        return iter(self._roots)

    def __repr__(self) -> str:
        return f"E8Lattice(roots=240, positive=120, rank=8)"


def e8_lattice() -> E8Lattice:
    """
    Create an E8Lattice instance.

    Factory function for E8Lattice.

    Returns:
        E8Lattice instance

    Example:
        >>> E8 = e8_lattice()
        >>> len(E8.roots)
        240
    """
    return E8Lattice()
