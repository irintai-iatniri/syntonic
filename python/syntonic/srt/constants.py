"""
SRT Constants - Fundamental constants for Syntony Recursion Theory.

Re-exports exact arithmetic constants from syntonic.exact and defines
SRT-specific dimensional constants.

Constants:
    PHI, PHI_SQUARED, PHI_INVERSE - Exact golden ratio forms
    PHI_NUMERIC - Float approximation of phi
    E_STAR_NUMERIC - e^pi - pi (spectral constant)
    Q_DEFICIT_NUMERIC - Universal syntony deficit q
    STRUCTURE_DIMENSIONS - Dimension dictionary for lattices

SRT-Specific:
    TORUS_DIMENSIONS - Dimension of T^4 torus (4)
    E8_ROOTS - Number of E8 roots (240)
    E8_POSITIVE_ROOTS - Number of positive E8 roots (120)
    E6_GOLDEN_CONE - Roots in golden cone (36)
    D4_KISSING - D4 kissing number / consciousness threshold (24)
    WINDING_INDICES - Names for winding coordinates (7, 8, 9, 10)
"""

from syntonic.exact import (
    # Exact golden ratio constants
    PHI,
    PHI_SQUARED,
    PHI_INVERSE,
    # Numeric constants
    PHI_NUMERIC,
    E_STAR_NUMERIC,
    Q_DEFICIT_NUMERIC,
    # Structure dimensions
    STRUCTURE_DIMENSIONS,
    # Functions
    fibonacci,
    lucas,
    correction_factor,
    golden_number,
    # Types
    GoldenExact,
    Rational,
)

# Import hierarchy constants from Rust backend
from syntonic._core import (
    hierarchy_e8_roots,
    hierarchy_e8_positive_roots,
    hierarchy_e8_rank,
    hierarchy_e8_coxeter,
    hierarchy_e7_roots,
    hierarchy_e7_positive_roots,
    hierarchy_e7_fundamental,
    hierarchy_e7_rank,
    hierarchy_e7_coxeter,
    hierarchy_e7_dim,
    hierarchy_e6_roots,
    hierarchy_e6_positive_roots,
    hierarchy_e6_fundamental,
    hierarchy_e6_rank,
    hierarchy_e6_coxeter,
    hierarchy_d4_rank,
    hierarchy_d4_coxeter,
    hierarchy_g2_rank,
    hierarchy_f4_rank,
    get_fibonacci_primes,
)

# T^4 Torus dimensions
TORUS_DIMENSIONS: int = 4

# E8 Lattice - Values from Rust backend
E8_ROOTS: int = hierarchy_e8_roots()
E8_POSITIVE_ROOTS: int = hierarchy_e8_positive_roots()
E8_RANK: int = hierarchy_e8_rank()
E8_DIMENSION: int = 248  # Adjoint representation (not exposed yet)
E8_COXETER_NUMBER: int = hierarchy_e8_coxeter()

# E7 Lattice - Newly exposed
E7_ROOTS: int = hierarchy_e7_roots()
E7_POSITIVE_ROOTS: int = hierarchy_e7_positive_roots()
E7_FUNDAMENTAL: int = hierarchy_e7_fundamental()
E7_RANK: int = hierarchy_e7_rank()
E7_COXETER: int = hierarchy_e7_coxeter()
E7_DIMENSION: int = hierarchy_e7_dim()

# E6 (Golden Cone) - Values from Rust backend
E6_ROOTS: int = hierarchy_e6_roots()
E6_POSITIVE_ROOTS: int = hierarchy_e6_positive_roots()  # Golden Cone
E6_FUNDAMENTAL: int = hierarchy_e6_fundamental()
E6_RANK: int = hierarchy_e6_rank()
E6_COXETER: int = hierarchy_e6_coxeter()
E6_DIMENSION: int = 78  # Not exposed yet
E6_GOLDEN_CONE: int = E6_POSITIVE_ROOTS  # |Phi+(E6)| = roots in golden cone

# D4 Lattice - Values from Rust backend
D4_KISSING: int = 24  # Consciousness threshold K(D4) (not exposed yet)
D4_RANK: int = hierarchy_d4_rank()
D4_COXETER: int = hierarchy_d4_coxeter()
D4_DIMENSION: int = 28  # Adjoint representation (not exposed yet)

# G2 (Octonions) - Values from Rust backend
G2_RANK: int = hierarchy_g2_rank()
G2_DIMENSION: int = 14  # Not exposed yet

# F4 (Jordan Algebra) - Values from Rust backend
F4_RANK: int = hierarchy_f4_rank()
F4_DIMENSION: int = 52  # Not exposed yet

# Fibonacci Primes - Newly exposed
FIBONACCI_PRIMES: list[int] = get_fibonacci_primes()

# Winding coordinate indices (for documentation)
WINDING_7: int = 0  # First internal dimension
WINDING_8: int = 1  # Second internal dimension
WINDING_9: int = 2  # Third internal dimension
WINDING_10: int = 3  # Fourth internal dimension
WINDING_INDICES = (7, 8, 9, 10)  # Physical dimension labels

# Root norm (all E8 roots have |lambda|^2 = 2)
E8_ROOT_NORM_SQUARED: int = 2

__all__ = [
    # From exact module
    "PHI",
    "PHI_SQUARED",
    "PHI_INVERSE",
    "PHI_NUMERIC",
    "E_STAR_NUMERIC",
    "Q_DEFICIT_NUMERIC",
    "STRUCTURE_DIMENSIONS",
    "fibonacci",
    "lucas",
    "correction_factor",
    "golden_number",
    "GoldenExact",
    "Rational",
    # SRT-specific constants
    "TORUS_DIMENSIONS",
    "E8_ROOTS",
    "E8_POSITIVE_ROOTS",
    "E8_RANK",
    "E8_DIMENSION",
    "E8_COXETER_NUMBER",
    # E7 constants (newly exposed)
    "E7_ROOTS",
    "E7_POSITIVE_ROOTS",
    "E7_FUNDAMENTAL",
    "E7_RANK",
    "E7_COXETER",
    "E7_DIMENSION",
    # E6 constants (updated)
    "E6_ROOTS",
    "E6_POSITIVE_ROOTS",
    "E6_FUNDAMENTAL",
    "E6_RANK",
    "E6_COXETER",
    "E6_DIMENSION",
    "E6_GOLDEN_CONE",
    # D4 constants (updated)
    "D4_KISSING",
    "D4_RANK",
    "D4_COXETER",
    "D4_DIMENSION",
    # G2 constants (updated)
    "G2_RANK",
    "G2_DIMENSION",
    # F4 constants (updated)
    "F4_RANK",
    "F4_DIMENSION",
    # Fibonacci primes (newly exposed)
    "FIBONACCI_PRIMES",
    # Other SRT constants
    "WINDING_7",
    "WINDING_8",
    "WINDING_9",
    "WINDING_10",
    "WINDING_INDICES",
    "E8_ROOT_NORM_SQUARED",
]
