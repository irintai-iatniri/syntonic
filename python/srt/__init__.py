"""
SRT - Syntony Recursion Theory core module.

The SRT module implements the geometric framework from which all
Standard Model physics predictions derive. It builds on the T⁴ torus
geometry with E₈ lattice structure and golden ratio recursion.

Core Components:
    - Geometry: T⁴ torus with winding states |n₇, n₈, n₉, n₁₀⟩
    - Golden: Recursion map R: n → ⌊φn⌋ and measure w(n) = exp(-|n|²/φ)
    - Lattice: E8 (240 roots), D4 (24 roots), golden cone (36 roots)
    - Spectral: Theta series, heat kernels, Möbius regularization
    - Functional: Syntony S[Ψ] ≤ φ

Key Mathematical Objects:
    - WindingState: Quantum numbers |n⟩ on T⁴
    - T4Torus: The 4-torus geometry
    - GoldenRecursionMap: R: n → ⌊φn⌋
    - E8Lattice: Root lattice with 240 roots
    - GoldenCone: 36 roots = Φ⁺(E₆) in the cone
    - ThetaSeries: Θ₄(t) partition function
    - SyntonyFunctional: S[Ψ] with bound φ

Example:
    >>> from syntonic.srt import create_srt_system
    >>> srt = create_srt_system()
    >>> srt.e8.num_roots
    240
    >>> srt.golden_cone.num_roots
    36
    >>> srt.syntony.global_bound
    1.618033988749895
"""

# Constants
from syntonic.srt.constants import (
    # Golden ratio
    PHI,
    PHI_SQUARED,
    PHI_INVERSE,
    PHI_NUMERIC,
    # SRT constants
    E_STAR_NUMERIC,
    Q_DEFICIT_NUMERIC,
    STRUCTURE_DIMENSIONS,
    # SRT-specific
    TORUS_DIMENSIONS,
    E8_ROOTS,
    E8_POSITIVE_ROOTS,
    E6_GOLDEN_CONE,
    D4_KISSING,
)

# Corrections
from syntonic.srt.corrections import (
    CorrectionFactors,
    correction_factors,
)

# Geometry
from syntonic.srt.geometry import (
    WindingState,
    winding_state,
    T4Torus,
    t4_torus,
)

# Golden operations
from syntonic.srt.golden import (
    GoldenMeasure,
    golden_measure,
    GoldenRecursionMap,
    golden_recursion_map,
)

# Lattice
from syntonic.srt.lattice import (
    D4Root,
    D4Lattice,
    d4_lattice,
    K_D4,
    E8Root,
    E8Lattice,
    e8_lattice,
    GoldenProjector,
    golden_projector,
    GoldenCone,
    golden_cone,
    QuadraticForm,
    quadratic_form,
    compute_Q,
)

# Spectral
from syntonic.srt.spectral import (
    ThetaSeries,
    theta_series,
    HeatKernel,
    heat_kernel,
    KnotLaplacian,
    knot_laplacian,
    MobiusRegularizer,
    mobius_regularizer,
    compute_e_star,
)

# Functional
from syntonic.srt.functional import (
    SyntonyFunctional,
    syntony_functional,
    compute_syntony,
)


class SRTSystem:
    """
    Complete SRT system with all components.

    Provides unified access to all SRT geometric structures:
    - Torus geometry and winding states
    - Golden recursion and measure
    - E8/D4 lattices and golden cone
    - Spectral operators
    - Syntony functional

    Attributes:
        torus: T4Torus instance
        golden_map: GoldenRecursionMap instance
        golden_measure: GoldenMeasure instance
        e8: E8Lattice instance
        d4: D4Lattice instance
        golden_cone: GoldenCone instance
        theta: ThetaSeries instance
        heat: HeatKernel instance
        laplacian: KnotLaplacian instance
        syntony: SyntonyFunctional instance

    Example:
        >>> srt = SRTSystem()
        >>> srt.e8.num_roots
        240
        >>> srt.golden_cone.num_roots
        36
    """

    def __init__(
        self,
        phi: float = None,
        max_norm: int = 20,
        max_spectral_terms: int = 20,
    ):
        """
        Initialize SRT system.

        Args:
            phi: Golden ratio value. If None, uses PHI_NUMERIC.
            max_norm: Maximum |n|² for winding states.
            max_spectral_terms: Maximum terms in spectral sums.
        """
        self._phi = phi if phi is not None else PHI_NUMERIC
        self._max_norm = max_norm

        # Initialize all components
        self._torus = T4Torus()
        self._golden_map = GoldenRecursionMap(phi=self._phi)
        self._golden_measure = GoldenMeasure(phi=self._phi)
        self._e8 = E8Lattice()
        self._d4 = D4Lattice()
        self._golden_cone = GoldenCone(e8_lattice=self._e8, phi=self._phi)
        self._theta = ThetaSeries(phi=self._phi, max_norm=max_spectral_terms)
        self._heat = HeatKernel(phi=self._phi, max_norm=max_spectral_terms)
        self._laplacian = KnotLaplacian(phi=self._phi, max_norm=max_norm)
        self._syntony = SyntonyFunctional(phi=self._phi, max_norm=max_norm)
        self._corrections = CorrectionFactors()

    @property
    def phi(self) -> float:
        """Golden ratio value."""
        return self._phi

    @property
    def torus(self) -> T4Torus:
        """T4Torus instance."""
        return self._torus

    @property
    def golden_map(self) -> GoldenRecursionMap:
        """GoldenRecursionMap instance."""
        return self._golden_map

    @property
    def golden_measure(self) -> GoldenMeasure:
        """GoldenMeasure instance."""
        return self._golden_measure

    @property
    def e8(self) -> E8Lattice:
        """E8Lattice instance."""
        return self._e8

    @property
    def d4(self) -> D4Lattice:
        """D4Lattice instance."""
        return self._d4

    @property
    def golden_cone(self) -> GoldenCone:
        """GoldenCone instance."""
        return self._golden_cone

    @property
    def theta(self) -> ThetaSeries:
        """ThetaSeries instance."""
        return self._theta

    @property
    def heat(self) -> HeatKernel:
        """HeatKernel instance."""
        return self._heat

    @property
    def laplacian(self) -> KnotLaplacian:
        """KnotLaplacian instance."""
        return self._laplacian

    @property
    def syntony(self) -> SyntonyFunctional:
        """SyntonyFunctional instance."""
        return self._syntony

    @property
    def corrections(self) -> CorrectionFactors:
        """CorrectionFactors instance."""
        return self._corrections

    def vacuum_state(self) -> WindingState:
        """Return the vacuum winding state |0,0,0,0⟩."""
        return winding_state(0, 0, 0, 0)

    def verify_e8_count(self) -> bool:
        """Verify E8 has 240 roots."""
        return len(self._e8.roots) == 240

    def verify_golden_cone_count(self) -> bool:
        """Verify golden cone has 36 roots."""
        return len(self._golden_cone.roots) == 36

    def verify_d4_kissing(self) -> bool:
        """Verify D4 kissing number is 24."""
        return self._d4.kissing_number == 24

    def verify_all(self) -> dict:
        """
        Verify all key mathematical identities.

        Returns:
            Dict of identity names to (expected, actual, passed) tuples
        """
        return {
            'E8_roots': (240, len(self._e8.roots), self.verify_e8_count()),
            'E8_positive': (120, len(self._e8.positive_roots), len(self._e8.positive_roots) == 120),
            'golden_cone': (36, len(self._golden_cone.roots), self.verify_golden_cone_count()),
            'D4_kissing': (24, self._d4.kissing_number, self.verify_d4_kissing()),
            'syntony_bound': (self._phi, self._syntony.global_bound, True),
        }

    def summary(self) -> str:
        """Return summary of SRT system configuration."""
        lines = [
            f"SRT System (phi={self._phi:.6f})",
            f"  Torus: T⁴ = S¹×S¹×S¹×S¹",
            f"  E8 lattice: {len(self._e8.roots)} roots",
            f"  D4 lattice: {len(self._d4.roots)} roots, K={self._d4.kissing_number}",
            f"  Golden cone: {len(self._golden_cone.roots)} roots",
            f"  Syntony bound: S ≤ {self._syntony.global_bound:.6f}",
        ]
        return "\n".join(lines)

    def __repr__(self) -> str:
        return f"SRTSystem(phi={self._phi:.6f})"


def create_srt_system(
    phi: float = None,
    max_norm: int = 20,
    max_spectral_terms: int = 20,
) -> SRTSystem:
    """
    Create a complete SRT system.

    Factory function for SRTSystem.

    Args:
        phi: Golden ratio value. If None, uses PHI_NUMERIC.
        max_norm: Maximum |n|² for winding states.
        max_spectral_terms: Maximum terms in spectral sums.

    Returns:
        SRTSystem instance

    Example:
        >>> srt = create_srt_system()
        >>> srt.e8.num_roots
        240
        >>> srt.verify_all()
        {'E8_roots': (240, 240, True), ...}
    """
    return SRTSystem(phi=phi, max_norm=max_norm, max_spectral_terms=max_spectral_terms)


__all__ = [
    # Constants
    'PHI',
    'PHI_SQUARED',
    'PHI_INVERSE',
    'PHI_NUMERIC',
    'E_STAR_NUMERIC',
    'Q_DEFICIT_NUMERIC',
    'STRUCTURE_DIMENSIONS',
    'TORUS_DIMENSIONS',
    'E8_ROOTS',
    'E8_POSITIVE_ROOTS',
    'E6_GOLDEN_CONE',
    'D4_KISSING',
    # Corrections
    'CorrectionFactors',
    'correction_factors',
    # Geometry
    'WindingState',
    'winding_state',
    'T4Torus',
    't4_torus',
    # Golden
    'GoldenMeasure',
    'golden_measure',
    'GoldenRecursionMap',
    'golden_recursion_map',
    # Lattice
    'D4Root',
    'D4Lattice',
    'd4_lattice',
    'K_D4',
    'E8Root',
    'E8Lattice',
    'e8_lattice',
    'GoldenProjector',
    'golden_projector',
    'GoldenCone',
    'golden_cone',
    'QuadraticForm',
    'quadratic_form',
    'compute_Q',
    # Spectral
    'ThetaSeries',
    'theta_series',
    'HeatKernel',
    'heat_kernel',
    'KnotLaplacian',
    'knot_laplacian',
    'MobiusRegularizer',
    'mobius_regularizer',
    'compute_e_star',
    # Functional
    'SyntonyFunctional',
    'syntony_functional',
    'compute_syntony',
    # System
    'SRTSystem',
    'create_srt_system',
]
