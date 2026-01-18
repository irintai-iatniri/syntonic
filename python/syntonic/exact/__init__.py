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
    srt_pi,
    srt_e,
    srt_structure_dimension,
    srt_correction_factor,
)

# Try to import PySymExpr if available
from syntonic.core import SymExpr


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

# π constant
PI_NUMERIC = srt_pi()  # 3.141592653589793

# Euler's number e
E_NUMERIC = srt_e()  # 2.718281828459045


# =============================================================================
# Prime Sequence Constants
# =============================================================================

# Fermat primes - Gauge force selection rules
FERMAT_PRIMES = [3, 5, 17, 257, 65537]  # F_0 through F_4
FERMAT_INDICES = [0, 1, 2, 3, 4]  # Physics stops at n=5 (composite)

# Mersenne primes - Matter stability rules
MERSENNE_PRIMES = [3, 7, 31, 127]  # M_2, M_3, M_5, M_7
MERSENNE_EXPONENTS = [2, 3, 5, 7]  # p values where 2^p - 1 is prime
M11_BARRIER = 2047  # 23 × 89 - Why 4th generation fails

# Lucas primes - Shadow/dark sector
LUCAS_SEQUENCE = [
    2,
    1,
    3,
    4,
    7,
    11,
    18,
    29,
    47,
    76,
    123,
    199,
    322,
    521,
    843,
    1364,
    2207,
    3571,
]
LUCAS_PRIMES = [2, 3, 7, 11, 29, 47, 199, 521, 2207, 3571]

# Kissing number (Collapse threshold)
D4_KISSING = 24  # K(D₄) - Wave function collapse threshold


# =============================================================================
# Structure Dimensions (for correction factors)
# =============================================================================
# Structure Dimensions (for correction factors)
# =============================================================================

STRUCTURE_DIMENSIONS = {
    "E8_dim": 248,  # dim(E₈) adjoint representation
    "E8_roots": 240,  # |Φ(E₈)| all roots
    "E8_positive": 120,  # |Φ⁺(E₈)| positive roots (chiral)
    "E6_dim": 78,  # dim(E₆) adjoint
    "E6_positive": 36,  # |Φ⁺(E₆)| = Golden Cone roots
    "E6_fundamental": 27,  # dim(27_E₆) fundamental representation
    "D4_kissing": 24,  # K(D₄) kissing number (consciousness threshold)
    "G2_dim": 14,  # dim(G₂) = Aut(O) automorphisms of octonions
    "E7_dim": 133,  # dim(E₇) adjoint
    "E7_roots": 126,  # |Φ(E₇)| all roots
    "E7_positive": 63,  # |Φ⁺(E₇)| positive roots
    "E7_fundamental": 56,  # dim(E₇ fundamental)
    "F4_dim": 52,  # dim(F₄)
    "D4_adjoint": 28,  # dim(SO(8)) = 28
}

# Index mapping for srt_correction_factor
_STRUCTURE_INDICES = {
    "E8_dim": 0,
    "E8_roots": 1,
    "E8_positive": 2,
    "E6_dim": 3,
    "E6_positive": 4,
    "E6_fundamental": 5,
    "D4_kissing": 6,
    "G2_dim": 7,
}

# =============================================================================
# Universal Syntony Correction Hierarchy
# =============================================================================

# Complete 25-level hierarchy (currently used levels marked ✓ USED in documentation)
# Extended to include all 60+ geometric structures from T⁴ × E₈ framework
CORRECTION_HIERARCHY = {
    0: 1.0,  # Exact tree-level
    1: Q_DEFICIT_NUMERIC**3,  # Third-order vacuum
    2: Q_DEFICIT_NUMERIC / 1000,  # Fixed-point stability (h(E₈)³/27)
    3: Q_DEFICIT_NUMERIC / 720,  # Coxeter-Kissing product (h(E₈)×K(D₄))
    4: Q_DEFICIT_NUMERIC / 360,  # Complete cone cycles (10×36)
    5: Q_DEFICIT_NUMERIC / 248,  # Full E₈ adjoint representation
    6: Q_DEFICIT_NUMERIC / 240,  # Full E₈ root system (both signs)
    7: Q_DEFICIT_NUMERIC / 133,  # Full E₇ adjoint representation
    8: Q_DEFICIT_NUMERIC / 126,  # Full E₇ root system
    9: Q_DEFICIT_NUMERIC / 120,  # Complete E₈ positive roots
    10: Q_DEFICIT_NUMERIC**2 / PHI_NUMERIC**2,  # Second-order/double golden
    11: Q_DEFICIT_NUMERIC / 78,  # Full E₆ gauge structure
    12: Q_DEFICIT_NUMERIC / 72,  # Full E₆ root system (both signs)
    13: Q_DEFICIT_NUMERIC / 63,  # E₇ positive roots
    14: Q_DEFICIT_NUMERIC**2 / PHI_NUMERIC,  # Second-order massless
    15: Q_DEFICIT_NUMERIC / 56,  # E₇ fundamental representation
    16: Q_DEFICIT_NUMERIC / 52,  # F₄ gauge structure
    17: Q_DEFICIT_NUMERIC**2,  # Second-order vacuum
    18: Q_DEFICIT_NUMERIC / 36,  # E₆ positive roots (Golden Cone)
    19: Q_DEFICIT_NUMERIC / 32,  # Five-fold binary structure
    20: Q_DEFICIT_NUMERIC / 30,  # E₈ Coxeter number
    21: Q_DEFICIT_NUMERIC / 28,  # D₄ adjoint representation
    22: Q_DEFICIT_NUMERIC / 27,  # E₆ fundamental representation
    23: Q_DEFICIT_NUMERIC / 24,  # D₄ kissing number
    24: Q_DEFICIT_NUMERIC**2 * PHI_NUMERIC,  # Quadratic + golden enhancement
    25: Q_DEFICIT_NUMERIC / (6 * math.pi),  # Six-flavor QCD loop
    26: Q_DEFICIT_NUMERIC / 18,  # E₇ Coxeter number
    27: Q_DEFICIT_NUMERIC / 16,  # Four-fold binary/spinor dimension
    28: Q_DEFICIT_NUMERIC / (5 * math.pi),  # Five-flavor QCD loop
    29: Q_DEFICIT_NUMERIC / 14,  # G₂ octonion automorphisms
    30: Q_DEFICIT_NUMERIC / (4 * math.pi),  # One-loop radiative (4D)
    31: Q_DEFICIT_NUMERIC / 12,  # Topology × generations (T⁴ × N_gen)
    32: Q_DEFICIT_NUMERIC / PHI_NUMERIC**5,  # Fifth golden power
    33: Q_DEFICIT_NUMERIC / (3 * math.pi),  # Three-flavor QCD loop
    34: Q_DEFICIT_NUMERIC / 9,  # Generation-squared structure
    35: Q_DEFICIT_NUMERIC / 8,  # Cartan subalgebra (rank E₈)
    36: Q_DEFICIT_NUMERIC / 7,  # E₇ Cartan subalgebra
    37: Q_DEFICIT_NUMERIC / PHI_NUMERIC**4,  # Fourth golden power
    38: Q_DEFICIT_NUMERIC / (2 * math.pi),  # Half-loop integral
    39: Q_DEFICIT_NUMERIC / 6,  # Sub-generation structure (2×3)
    40: Q_DEFICIT_NUMERIC / PHI_NUMERIC**3,  # Third golden power
    41: Q_DEFICIT_NUMERIC / 4,  # Quarter layer (sphaleron)
    42: Q_DEFICIT_NUMERIC / math.pi,  # Circular loop structure
    43: Q_DEFICIT_NUMERIC / 3,  # Single generation
    44: Q_DEFICIT_NUMERIC / PHI_NUMERIC**2,  # Second golden power
    45: Q_DEFICIT_NUMERIC / 2,  # Half layer
    46: Q_DEFICIT_NUMERIC / PHI_NUMERIC,  # Scale running (one layer)
    47: Q_DEFICIT_NUMERIC,  # Universal vacuum
    48: Q_DEFICIT_NUMERIC * PHI_NUMERIC,  # Double layer transitions
    49: Q_DEFICIT_NUMERIC * PHI_NUMERIC**2,  # Fixed point (φ²=φ+1)
    50: 3 * Q_DEFICIT_NUMERIC,  # Triple generation
    51: math.pi * Q_DEFICIT_NUMERIC,  # Circular enhancement
    52: 4 * Q_DEFICIT_NUMERIC,  # Full T⁴ topology
    53: Q_DEFICIT_NUMERIC * PHI_NUMERIC**3,  # Triple golden transitions
    54: 6 * Q_DEFICIT_NUMERIC,  # Full E₆ Cartan enhancement
    55: Q_DEFICIT_NUMERIC * PHI_NUMERIC**4,  # Fourth golden transitions
    56: 8 * Q_DEFICIT_NUMERIC,  # Full E₈ Cartan enhancement
    57: Q_DEFICIT_NUMERIC * PHI_NUMERIC**5,  # Fifth golden transitions
}

# Multiplicative suppression factors
SUPPRESSION_FACTORS = {
    "inverse_double_recursion": 1 / (1 + Q_DEFICIT_NUMERIC / PHI_NUMERIC**2),
    "inverse_recursion": 1 / (1 + Q_DEFICIT_NUMERIC / PHI_NUMERIC),
    "base_suppression": 1 / (1 + Q_DEFICIT_NUMERIC),
    "recursion_penalty": 1 / (1 + Q_DEFICIT_NUMERIC * PHI_NUMERIC),
    "fixed_point_penalty": 1 / (1 + Q_DEFICIT_NUMERIC * PHI_NUMERIC**2),
    "deep_recursion_penalty": 1 / (1 + Q_DEFICIT_NUMERIC * PHI_NUMERIC**3),
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


def get_correction_factor(level: int) -> float:
    """
    Get the correction factor for a given hierarchy level.

    Args:
        level: Hierarchy level (0-57)

    Returns:
        Correction factor value (1 ± correction)

    Examples:
        >>> get_correction_factor(35)  # q/8
        0.996577...
        >>> get_correction_factor(47)  # q
        0.027395...
    """
    return CORRECTION_HIERARCHY.get(level, 1.0)


def get_suppression_factor(name: str) -> float:
    """
    Get the suppression factor for a given name.

    Args:
        name: One of 'inverse_double_recursion', 'inverse_recursion', 'base_suppression',
              'recursion_penalty', 'fixed_point_penalty', 'deep_recursion_penalty'

    Returns:
        Suppression factor value (< 1)

    Examples:
        >>> get_suppression_factor('inverse_recursion')  # 1/(1+q/φ)
        0.9830...
        >>> get_suppression_factor('recursion_penalty')  # 1/(1+qφ)
        0.9578...
    """
    return SUPPRESSION_FACTORS.get(name, 1.0)


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
    "Rational",
    "GoldenExact",
    "FundamentalConstant",
    "SymExpr",
    # Exact constants
    "PHI",
    "PHI_SQUARED",
    "PHI_INVERSE",
    # Numeric constants
    "PHI_NUMERIC",
    "E_STAR_NUMERIC",
    "Q_DEFICIT_NUMERIC",
    "PI_NUMERIC",
    "E_NUMERIC",
    # Prime sequence constants
    "FERMAT_PRIMES",
    "FERMAT_INDICES",
    "MERSENNE_PRIMES",
    "MERSENNE_EXPONENTS",
    "M11_BARRIER",
    "LUCAS_SEQUENCE",
    "LUCAS_PRIMES",
    "D4_KISSING",
    # Structure data
    "STRUCTURE_DIMENSIONS",
    "CORRECTION_HIERARCHY",
    "SUPPRESSION_FACTORS",
    # Functions
    "fibonacci",
    "lucas",
    "correction_factor",
    "get_correction_factor",
    "get_suppression_factor",
    "golden_number",
]
