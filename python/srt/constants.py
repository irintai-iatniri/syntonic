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

# T^4 Torus dimensions
TORUS_DIMENSIONS: int = 4

# E8 Lattice
E8_ROOTS: int = 240
E8_POSITIVE_ROOTS: int = 120
E8_RANK: int = 8
E8_DIMENSION: int = 248  # Adjoint representation
E8_COXETER_NUMBER: int = 30

# E6 (Golden Cone)
E6_GOLDEN_CONE: int = 36  # |Phi+(E6)| = roots in golden cone
E6_RANK: int = 6
E6_DIMENSION: int = 78

# D4 Lattice
D4_KISSING: int = 24  # Consciousness threshold K(D4)
D4_RANK: int = 4
D4_DIMENSION: int = 28  # Adjoint representation

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
    'PHI',
    'PHI_SQUARED',
    'PHI_INVERSE',
    'PHI_NUMERIC',
    'E_STAR_NUMERIC',
    'Q_DEFICIT_NUMERIC',
    'STRUCTURE_DIMENSIONS',
    'fibonacci',
    'lucas',
    'correction_factor',
    'golden_number',
    'GoldenExact',
    'Rational',
    # SRT-specific constants
    'TORUS_DIMENSIONS',
    'E8_ROOTS',
    'E8_POSITIVE_ROOTS',
    'E8_RANK',
    'E8_DIMENSION',
    'E8_COXETER_NUMBER',
    'E6_GOLDEN_CONE',
    'E6_RANK',
    'E6_DIMENSION',
    'D4_KISSING',
    'D4_RANK',
    'D4_DIMENSION',
    'WINDING_7',
    'WINDING_8',
    'WINDING_9',
    'WINDING_10',
    'WINDING_INDICES',
    'E8_ROOT_NORM_SQUARED',
]
