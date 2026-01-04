"""
Knot Laplacian - L²_knot operator on winding states.

The knot Laplacian is the central differential operator in SRT,
acting on winding configurations on T⁴. Its eigenvalues determine
particle masses through the golden recursion.

The knot Laplacian incorporates:
- Standard Laplacian on T⁴
- Golden measure weighting
- Knot topology corrections

    L²_knot = -Δ + V_knot(n)

where V_knot encodes the knot structure.

Example:
    >>> from syntonic.srt.spectral import knot_laplacian
    >>> L = knot_laplacian()
    >>> L.eigenvalue(winding_state(1, 0, 0, 0))
    ...
"""

from __future__ import annotations
from typing import Tuple, Optional, List, Iterator, Dict
import math

from syntonic.exact import PHI, PHI_NUMERIC, GoldenExact
from syntonic.srt.geometry.winding import WindingState, winding_state


class KnotLaplacian:
    """
    Knot Laplacian L²_knot on winding states.

    The knot Laplacian extends the standard Laplacian with
    golden-weighted corrections:

        L²_knot|n⟩ = (4π²|n|² + V_knot(n))|n⟩

    where V_knot(n) encodes topology through the golden ratio.

    The eigenvalues satisfy:
        λ_n = 4π²|n|² · (1 + φ^(-|n|²))

    This gives mass hierarchies through the exponential φ suppression.

    Attributes:
        phi: Golden ratio
        max_norm: Maximum |n|² for computations

    Example:
        >>> L = KnotLaplacian()
        >>> L.eigenvalue(winding_state(1, 0, 0, 0))
        ...
        >>> L.spectrum(max_norm=4)
        [...]
    """

    def __init__(
        self,
        phi: Optional[float] = None,
        max_norm: int = 20,
    ):
        """
        Initialize knot Laplacian.

        Uses exact golden ratio φ from syntonic.exact.PHI by default.

        Args:
            phi: Golden ratio float value. If None, uses PHI.eval().
            max_norm: Maximum |n|² for winding states.
        """
        self._phi_exact = PHI  # Store exact golden ratio
        self._phi = phi if phi is not None else PHI.eval()
        self._max_norm = max_norm
        self._windings = list(self._enumerate_windings())

    def _enumerate_windings(self) -> Iterator[WindingState]:
        """Generate all winding states with |n|² <= max_norm."""
        max_single = int(math.sqrt(self._max_norm)) + 1
        for n7 in range(-max_single, max_single + 1):
            for n8 in range(-max_single, max_single + 1):
                for n9 in range(-max_single, max_single + 1):
                    for n10 in range(-max_single, max_single + 1):
                        n = winding_state(n7, n8, n9, n10)
                        if n.norm_squared <= self._max_norm:
                            yield n

    @property
    def phi(self) -> float:
        """Golden ratio value."""
        return self._phi

    @property
    def max_norm(self) -> int:
        """Maximum norm squared for winding states."""
        return self._max_norm

    def base_eigenvalue(self, n: WindingState) -> float:
        """
        Compute base Laplacian eigenvalue λ⁰_n = 4π²|n|².

        Args:
            n: Winding state

        Returns:
            Base eigenvalue
        """
        return 4 * math.pi * math.pi * n.norm_squared

    def knot_potential(self, n: WindingState) -> float:
        """
        Compute knot potential V_knot(n).

        The knot potential provides golden-weighted corrections:
            V_knot(n) = 4π²|n|² · φ^(-|n|²)

        This creates the mass hierarchy through φ suppression.

        Args:
            n: Winding state

        Returns:
            Knot potential value
        """
        if n.norm_squared == 0:
            return 0.0

        base = 4 * math.pi * math.pi * n.norm_squared
        correction = self._phi ** (-n.norm_squared)
        return base * correction

    def eigenvalue(self, n: WindingState) -> float:
        """
        Compute full knot Laplacian eigenvalue.

            λ_n = λ⁰_n + V_knot(n) = 4π²|n|² · (1 + φ^(-|n|²))

        Args:
            n: Winding state

        Returns:
            Full eigenvalue
        """
        return self.base_eigenvalue(n) + self.knot_potential(n)

    def mass_squared(self, n: WindingState) -> float:
        """
        Compute effective mass squared from eigenvalue.

        m²(n) = λ_n / 4π²

        Args:
            n: Winding state

        Returns:
            Mass squared in natural units
        """
        return self.eigenvalue(n) / (4 * math.pi * math.pi)

    def spectrum(self, max_norm: Optional[int] = None) -> List[Tuple[WindingState, float]]:
        """
        Compute full spectrum up to max_norm.

        Args:
            max_norm: Maximum |n|² (defaults to self._max_norm)

        Returns:
            List of (winding_state, eigenvalue) sorted by eigenvalue
        """
        if max_norm is None:
            max_norm = self._max_norm

        results = []
        for n in self._windings:
            if n.norm_squared <= max_norm:
                results.append((n, self.eigenvalue(n)))

        return sorted(results, key=lambda x: x[1])

    def degeneracy(self, n_sq: int) -> int:
        """
        Count degeneracy of states with given |n|².

        Args:
            n_sq: Value of |n|²

        Returns:
            Number of states with this norm squared
        """
        count = 0
        for n in self._windings:
            if n.norm_squared == n_sq:
                count += 1
        return count

    def level_spacing(self, max_levels: int = 10) -> List[float]:
        """
        Compute spacing between consecutive eigenvalue levels.

        Args:
            max_levels: Number of spacings to compute

        Returns:
            List of level spacings Δλ_k = λ_{k+1} - λ_k
        """
        eigenvalues = sorted(set(self.eigenvalue(n) for n in self._windings))
        spacings = []

        for i in range(min(max_levels, len(eigenvalues) - 1)):
            spacings.append(eigenvalues[i + 1] - eigenvalues[i])

        return spacings

    def spectral_gap(self) -> float:
        """
        Compute spectral gap (first non-zero eigenvalue).

        Returns:
            First positive eigenvalue
        """
        eigenvalues = sorted(self.eigenvalue(n) for n in self._windings)
        for ev in eigenvalues:
            if ev > 1e-10:
                return ev
        return 0.0

    def apply(self, coefficients: Dict[WindingState, complex]) -> Dict[WindingState, complex]:
        """
        Apply L²_knot to a wavefunction given by coefficients.

        If ψ = Σ_n c_n |n⟩, then L²_knot ψ = Σ_n λ_n c_n |n⟩

        Args:
            coefficients: Dict mapping winding states to complex coefficients

        Returns:
            Result as Dict of coefficients
        """
        result = {}
        for n, c in coefficients.items():
            result[n] = self.eigenvalue(n) * c
        return result

    def expectation_value(self, coefficients: Dict[WindingState, complex]) -> float:
        """
        Compute ⟨ψ|L²_knot|ψ⟩ for normalized ψ.

        Args:
            coefficients: Dict mapping winding states to complex coefficients

        Returns:
            Expectation value (real)
        """
        # Compute norm for normalization
        norm_sq = sum(abs(c) ** 2 for c in coefficients.values())
        if norm_sq < 1e-15:
            return 0.0

        # Compute expectation
        total = 0.0
        for n, c in coefficients.items():
            total += (abs(c) ** 2) * self.eigenvalue(n)

        return total / norm_sq

    def heat_kernel_trace(self, t: float) -> float:
        """
        Compute Tr[exp(-t·L²_knot)] using eigenvalues.

        Args:
            t: Time parameter (t > 0)

        Returns:
            Heat kernel trace
        """
        if t <= 0:
            raise ValueError(f"t must be positive, got {t}")

        total = 0.0
        for n in self._windings:
            lambda_n = self.eigenvalue(n)
            total += math.exp(-t * lambda_n)

        return total

    def zeta_function(self, s: complex) -> complex:
        """
        Compute spectral zeta ζ_L(s) = Σ_{λ>0} λ^(-s).

        Args:
            s: Complex parameter

        Returns:
            Spectral zeta value
        """
        total = complex(0.0, 0.0)

        for n in self._windings:
            lambda_n = self.eigenvalue(n)
            if lambda_n > 1e-10:  # Exclude zero mode
                total += lambda_n ** (-s)

        return total

    def golden_suppression_factor(self, n: WindingState) -> float:
        """
        Compute the golden suppression factor φ^(-|n|²).

        This factor creates the mass hierarchy.

        Args:
            n: Winding state

        Returns:
            Suppression factor
        """
        return self._phi ** (-n.norm_squared)

    def __repr__(self) -> str:
        return f"KnotLaplacian(phi={self._phi:.6f}, max_norm={self._max_norm})"


def knot_laplacian(
    phi: Optional[float] = None,
    max_norm: int = 20,
) -> KnotLaplacian:
    """
    Create a KnotLaplacian instance.

    Factory function for KnotLaplacian.

    Args:
        phi: Golden ratio value. If None, uses PHI_NUMERIC.
        max_norm: Maximum |n|² for winding states.

    Returns:
        KnotLaplacian instance

    Example:
        >>> L = knot_laplacian()
        >>> L.eigenvalue(winding_state(1, 0, 0, 0))
        ...
    """
    return KnotLaplacian(phi=phi, max_norm=max_norm)
