"""
Exact arithmetic for Syntonic.

This module provides exact arithmetic over:
- Q: Rational numbers (arbitrary precision)
- Q(φ): Golden field extension (a + b·φ where φ = (1+√5)/2)

The golden ratio φ satisfies the fundamental identity:
    φ² = φ + 1

All computations in this module are exact (no floating-point errors).
Use .eval() to convert to float when needed.

SRT Constants:
- PHI: Golden ratio φ ≈ 1.618033988749895
- PHI_SQUARED: φ² = 1 + φ ≈ 2.618033988749895
- PHI_INVERSE: 1/φ = φ - 1 ≈ 0.618033988749895
- E_STAR_NUMERIC: e^π - π ≈ 19.999099979189474
- Q_DEFICIT_NUMERIC: Syntony deficit q ≈ 0.027395146920

Usage:
    >>> import syntonic as syn
    >>> phi = syn.PHI
    >>> phi_squared = phi * phi
    >>> phi_squared.eval() == phi.eval() + 1  # φ² = φ + 1
    True
    >>> syn.fibonacci(10)
    55
"""

import math
from syntonic.core import (
    Rational,
    GoldenExact,
    FundamentalConstant,
    srt_phi,
    srt_phi_inv,
    srt_q_deficit,
    srt_structure_dimension,
    srt_correction_factor,
)

# Try to import PySymExpr if available
try:
    from syntonic.core import PySymExpr as SymExpr
except ImportError:
    SymExpr = None

# =============================================================================
# Exact Golden Constants
# =============================================================================

# φ = (1 + √5) / 2 ≈ 1.618033988749895
PHI = GoldenExact.golden_ratio()

# φ² = φ + 1 ≈ 2.618033988749895
PHI_SQUARED = GoldenExact.golden_squared()

# φ̂ = 1/φ = φ - 1 ≈ 0.618033988749895
PHI_INVERSE = GoldenExact.coherence_parameter()

# =============================================================================
# Numeric Constants (float approximations)
# =============================================================================

# Golden ratio as float
PHI_NUMERIC = srt_phi()  # 1.618033988749895

# Spectral constant E* = e^π - π
E_STAR_NUMERIC = math.exp(math.pi) - math.pi  # 19.999099979189474

# Universal syntony deficit q
Q_DEFICIT_NUMERIC = srt_q_deficit()  # 0.027395146920

# =============================================================================
# Structure Dimensions (for correction factors)
# =============================================================================

STRUCTURE_DIMENSIONS = {
    'E8_dim': 248,        # dim(E₈) adjoint representation
    'E8_roots': 240,      # |Φ(E₈)| all roots
    'E8_positive': 120,   # |Φ⁺(E₈)| positive roots (chiral)
    'E6_dim': 78,         # dim(E₆) adjoint
    'E6_positive': 36,    # |Φ⁺(E₆)| = Golden Cone roots
    'E6_fundamental': 27, # dim(27_E₆) fundamental representation
    'D4_kissing': 24,     # K(D₄) kissing number (consciousness threshold)
    'G2_dim': 14,         # dim(G₂) = Aut(O) automorphisms of octonions
}

# Index mapping for srt_correction_factor
_STRUCTURE_INDICES = {
    'E8_dim': 0,
    'E8_roots': 1,
    'E8_positive': 2,
    'E6_dim': 3,
    'E6_positive': 4,
    'E6_fundamental': 5,
    'D4_kissing': 6,
    'G2_dim': 7,
}

# =============================================================================
# Helper Functions
# =============================================================================


def fibonacci(n: int) -> int:
    """
    Compute Fibonacci number F_n using exact golden ratio arithmetic.

    Uses the identity: φ^n = F_{n-1} + F_n·φ

    Args:
        n: Non-negative integer

    Returns:
        F_n (Fibonacci number)

    Examples:
        >>> fibonacci(0)
        0
        >>> fibonacci(1)
        1
        >>> fibonacci(10)
        55
        >>> fibonacci(50)
        12586269025
    """
    if n < 0:
        raise ValueError(f"n must be non-negative, got {n}")
    if n == 0:
        return 0
    if n == 1:
        return 1

    # φ^n = F_{n-1} + F_n·φ
    # So F_n is the coefficient of φ
    phi_n = PHI.phi_to_power(n)
    # phi_coefficient returns (numerator, denominator) tuple
    num, denom = phi_n.phi_coefficient
    return num // denom


def lucas(n: int) -> int:
    """
    Compute Lucas number L_n using exact golden ratio arithmetic.

    Uses the identity: L_n = φ^n + (φ')^n = 2·a_n + b_n
    where φ^n = a_n + b_n·φ

    Args:
        n: Non-negative integer

    Returns:
        L_n (Lucas number)

    Examples:
        >>> lucas(0)
        2
        >>> lucas(1)
        1
        >>> lucas(10)
        123
    """
    if n < 0:
        raise ValueError(f"n must be non-negative, got {n}")
    if n == 0:
        return 2
    if n == 1:
        return 1

    # φ^n = a + b·φ
    # (φ')^n = a - b·(φ - 1) = (a + b) - b·φ  [since φ' = -1/φ = 1 - φ]
    # L_n = φ^n + (φ')^n = 2a + b
    phi_n = PHI.phi_to_power(n)
    # rational_coefficient and phi_coefficient return (num, denom) tuples
    a_num, a_denom = phi_n.rational_coefficient
    b_num, b_denom = phi_n.phi_coefficient
    a = a_num // a_denom
    b = b_num // b_denom
    return 2 * a + b


def correction_factor(structure: str, sign: int = 1) -> float:
    """
    Compute SRT correction factor (1 ± q/N).

    These factors arise from heat kernel regularization in SRT.

    Args:
        structure: One of 'E8_dim', 'E8_roots', 'E8_positive',
                  'E6_dim', 'E6_positive', 'E6_fundamental',
                  'D4_kissing', 'G2_dim'
        sign: +1 for enhancement (1 + q/N), -1 for suppression (1 - q/N)

    Returns:
        Correction factor value

    Examples:
        >>> correction_factor('E8_positive', -1)  # Chiral suppression
        0.9997716...
        >>> correction_factor('E6_positive', +1)  # Intra-generation
        1.0007609...
    """
    if structure not in _STRUCTURE_INDICES:
        raise ValueError(
            f"Unknown structure: {structure}. "
            f"Valid options: {list(_STRUCTURE_INDICES.keys())}"
        )
    index = _STRUCTURE_INDICES[structure]
    return srt_correction_factor(index, sign)


def golden_number(a: int, b: int) -> GoldenExact:
    """
    Create an exact golden number a + b·φ.

    Args:
        a: Rational part coefficient
        b: Golden part coefficient (coefficient of φ)

    Returns:
        GoldenExact representing a + b·φ

    Examples:
        >>> g = golden_number(1, 2)  # 1 + 2φ
        >>> g.eval()
        4.23606797749979
    """
    return GoldenExact.from_integers(a, b)


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    # Classes
    'Rational',
    'GoldenExact',
    'FundamentalConstant',
    'SymExpr',
    # Exact constants
    'PHI',
    'PHI_SQUARED',
    'PHI_INVERSE',
    # Numeric constants
    'PHI_NUMERIC',
    'E_STAR_NUMERIC',
    'Q_DEFICIT_NUMERIC',
    # Structure data
    'STRUCTURE_DIMENSIONS',
    # Functions
    'fibonacci',
    'lucas',
    'correction_factor',
    'golden_number',
]
