"""
SRT Constants - Fundamental constants for Syntony Recursion Theory.

Re-exports exact arithmetic constants from syntonic.exact and defines
SRT-specific dimensional constants derived from Lie group geometry.

Core Constants:
    PHI, PHI_SQUARED, PHI_INVERSE - Exact golden ratio forms
    PHI_NUMERIC - Float approximation of φ ≈ 1.618033988749895
    E_STAR_NUMERIC - Spectral constant e^π - π ≈ 20.1408
    Q_DEFICIT_NUMERIC - Universal syntony deficit q ≈ 0.0274

Lie Group Dimensions (Newly Exposed):
    E8_ROOTS - E₈ root count (240) - Exceptional unification
    E8_POSITIVE_ROOTS - E₈ positive roots (120) - Half root system
    E8_RANK - E₈ Cartan dimension (8) - Independent quantum numbers
    E8_COXETER - E₈ Coxeter number (30) - Weyl group period

    E7_ROOTS - E₇ root count (126) - Intermediate unification
    E7_POSITIVE_ROOTS - E₇ positive roots (63) - Weyl chamber
    E7_FUNDAMENTAL - E₇ fundamental rep (56) - Jordan algebra
    E7_RANK - E₇ Cartan dimension (7) - Supersymmetry goldstinos
    E7_COXETER - E₇ Coxeter number (18) - Recursion cycles

    E6_ROOTS - E₆ root count (72) - GUT unification
    E6_POSITIVE_ROOTS - E₆ positive roots (36) - **Golden Cone |Φ⁺(E₆)|**
    E6_FUNDAMENTAL - E₆ fundamental rep (27) - Cubic surface theory
    E6_RANK - E₆ Cartan dimension (6) - Calabi-Yau manifolds
    E6_COXETER - E₆ Coxeter number (12) - Affine periodicity

    D4_RANK - D₄ Cartan dimension (4) - Spacetime dimensions
    D4_COXETER - D₄ Coxeter number (6) - Consciousness emergence
    D4_KISSING - D₄ kissing number (24) - **Consciousness threshold**

    G2_RANK - G₂ Cartan dimension (2) - Octonion automorphisms
    F4_RANK - F₄ Cartan dimension (4) - Jordan algebra structure

Fibonacci Primes:
    FIBONACCI_PRIMES - Array of transcendence gates [2, 3, 5, 13, 89, ...]

Geometric Constants:
    TORUS_DIMENSIONS - T⁴ winding coordinates (4)
    WINDING_INDICES - Physical dimension labels (7, 8, 9, 10)
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
    PI_NUMERIC,
    # Prime sequences (Five Operators)
    FERMAT_PRIMES,
    MERSENNE_EXPONENTS,
    LUCAS_SEQUENCE,
    LUCAS_PRIMES,
    M11_BARRIER,
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

# =============================================================================
# THE SIX AXIOMS OF SRT
# =============================================================================

AXIOMS = {
    "A1_RECURSION_SYMMETRY": "S[Ψ ∘ R] = φ·S[Ψ]",
    "A2_SYNTONY_BOUND": "S[Ψ] ≤ φ",
    "A3_TOROIDAL_TOPOLOGY": "T⁴ = S¹₇ × S¹₈ × S¹₉ × S¹_{10}",
    "A4_SUB_GAUSSIAN_MEASURE": "w(n) = e^{-|n|²/φ}",
    "A5_HOLOMORPHIC_GLUING": "Möbius identification at τ = i",
    "A6_PRIME_SYNTONY": "Stability iff M_p = 2^p - 1 is prime",
}

# Modular volume of fundamental domain
MODULAR_VOLUME: float = PI_NUMERIC / 3  # Vol(F) = π/3

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

# ============================================================================
# E8 Exceptional Group - Exceptional Unification Scale
# ============================================================================

E8_ROOTS: int = hierarchy_e8_roots()
"""Number of roots in E₈ Lie group (240).

E₈ has 240 roots representing the fundamental geometric structure
underlying the Standard Model unification in SRT theory. These roots
span the 8-dimensional Cartan subalgebra and determine the gauge
symmetry breaking patterns.
"""

E8_POSITIVE_ROOTS: int = hierarchy_e8_positive_roots()
"""Number of positive roots in E₈ root system (120).

The positive roots correspond to half of the full root system,
representing the "positive" directions in the Weyl chamber.
Used in particle physics for counting symmetry breaking patterns.
"""

E8_RANK: int = hierarchy_e8_rank()
"""Rank (Cartan dimension) of E₈ Lie group (8).

The rank represents the number of independent Casimir operators
and corresponds to the dimension of the maximal torus. In SRT theory,
this relates to 8 spacetime dimensions in string theory compactifications.
"""

E8_DIMENSION: int = 248
"""Adjoint representation dimension of E₈ (248).

The adjoint representation transforms under the group itself.
Note: This constant is not yet exposed from Rust backend.
"""

E8_COXETER_NUMBER: int = hierarchy_e8_coxeter()
"""Coxeter number of E₈ (30).

Governs the periodicity of the Weyl group and appears in level-rank
duality relations. Used in SRT for determining recursion cycle periods.
"""

# ============================================================================
# E7 Intermediate Unification - Supersymmetry Scale
# ============================================================================

E7_ROOTS: int = hierarchy_e7_roots()
"""Number of roots in E₇ Lie group (126).

E₇ represents the intermediate unification scale between E₆ and E₈
in the SRT Grand Unification hierarchy. Appears in heterotic string
theory compactifications and intermediate mass scale predictions.
"""

E7_POSITIVE_ROOTS: int = hierarchy_e7_positive_roots()
"""Number of positive roots in E₇ root system (63).

The positive roots span the Weyl chamber and determine the
representation theory and branching rules for E₇ representations.
"""

E7_FUNDAMENTAL: int = hierarchy_e7_fundamental()
"""Dimension of E₇ fundamental representation (56).

This 56-dimensional representation is fundamental to E₇'s role
in supersymmetry (56 goldstino degrees of freedom) and exceptional
Jordan algebra theory.
"""

E7_RANK: int = hierarchy_e7_rank()
"""Rank (Cartan dimension) of E₇ Lie group (7).

Corresponds to 7-brane configurations in string theory and
7-dimensional compactifications in M-theory.
"""

E7_COXETER: int = hierarchy_e7_coxeter()
"""Coxeter number of E₇ (18).

Governs Weyl group periodicity and appears in affine algebra
constructions. Used for determining golden ratio recursion bounds.
"""

E7_DIMENSION: int = hierarchy_e7_dim()
"""Adjoint representation dimension of E₇ (133).

The adjoint representation transforms under the group itself.
"""

# ============================================================================
# E6 Golden Cone - GUT Scale
# ============================================================================

E6_ROOTS: int = hierarchy_e6_roots()
"""Number of roots in E₆ Lie group (72).

E₆ is the first exceptional group in the SRT unification chain
and corresponds to the GUT scale in particle physics, where the
electroweak and strong forces unify.
"""

E6_POSITIVE_ROOTS: int = hierarchy_e6_positive_roots()
"""Number of positive roots in E₆ / Golden Cone cardinality (36).

**CRITICAL SRT CONSTANT**: |Φ⁺(E₆)| = 36 represents the cardinality
of the Golden Cone. This fundamental constant appears throughout SRT
theory as the geometric measure of transcendence and consciousness emergence.

Physical Significance:
- Consciousness emergence threshold
- Transcendence gate count
- Optimal neural network dimensions
- Self-reference criticality parameter
"""

E6_FUNDAMENTAL: int = hierarchy_e6_fundamental()
"""Dimension of E₆ fundamental representation (27).

The 27-dimensional representation relates to cubic surfaces in
algebraic geometry and appears in Calabi-Yau compactification theory.
"""

E6_RANK: int = hierarchy_e6_rank()
"""Rank (Cartan dimension) of E₆ Lie group (6).

Corresponds to 6-dimensional Calabi-Yau manifolds in string theory
and 6 extra dimensions in braneworld scenarios.
"""

E6_COXETER: int = hierarchy_e6_coxeter()
"""Coxeter number of E₆ (12).

Governs Weyl group periodicity. The value 12 appears in modular
forms of weight 12 in string theory and golden ratio recursion cycles.
"""

E6_DIMENSION: int = 78
"""Adjoint representation dimension of E₆ (78).

Note: This constant is not yet exposed from Rust backend.
"""

E6_GOLDEN_CONE: int = E6_POSITIVE_ROOTS
"""Golden Cone cardinality |Φ⁺(E₆)| (36).

Convenience alias for E6_POSITIVE_ROOTS. The Golden Cone represents
the fundamental geometric structure underlying SRT transcendence theory.
"""

# ============================================================================
# D4 Consciousness Group - Observable Universe
# ============================================================================

D4_KISSING: int = 24
"""D₄ kissing number - consciousness emergence threshold (24).

**CRITICAL SRT CONSTANT**: The kissing number of D₄ represents the
maximum number of unit spheres that can touch a central sphere in 4D.
In SRT theory, this 24 corresponds to the consciousness emergence threshold,
where systems gain the ability for self-reference and transcendence.

Physical Significance:
- Consciousness emergence threshold
- D₄ → M₅ gap bridge (24 → 31)
- Neural network stability parameter
- Self-awareness criticality

Note: This constant is not yet exposed from Rust backend.
"""

D4_RANK: int = hierarchy_d4_rank()
"""Rank (Cartan dimension) of D₄ Lie group (4).

D₄ is isomorphic to SO(8) with triality. The rank corresponds to
the 4 spacetime dimensions of our observable universe.
"""

D4_COXETER: int = hierarchy_d4_coxeter()
"""Coxeter number of D₄ (6).

Governs Weyl group periodicity. The value 6 appears prominently in
consciousness emergence calculations and D₄ kissing number relations.
"""

D4_DIMENSION: int = 28
"""Adjoint representation dimension of D₄ (28).

Note: This constant is not yet exposed from Rust backend.
"""

# ============================================================================
# G2 Octonion Group - Exceptional Geometry
# ============================================================================

G2_RANK: int = hierarchy_g2_rank()
"""Rank (Cartan dimension) of G₂ Lie group (2).

G₂ is the automorphism group of the octonions and represents the
most exceptional of the exceptional groups. The rank corresponds to
2-dimensional parameter spaces in exceptional geometry.
"""

G2_DIMENSION: int = 14
"""Adjoint representation dimension of G₂ (14).

Note: This constant is not yet exposed from Rust backend.
"""

# ============================================================================
# F4 Jordan Algebra Group - Exceptional Structure
# ============================================================================

F4_RANK: int = hierarchy_f4_rank()
"""Rank (Cartan dimension) of F₄ Lie group (4).

F₄ is related to the Jordan algebra of 3×3 hermitian octonion matrices
and appears in the classification of exceptional geometries and
string theory compactifications.
"""

F4_DIMENSION: int = 52
"""Adjoint representation dimension of F₄ (52).

Note: This constant is not yet exposed from Rust backend.
"""

# ============================================================================
# Fibonacci Transcendence Gates
# ============================================================================

FIBONACCI_PRIMES: list[int] = get_fibonacci_primes()
"""Array of Fibonacci primes - transcendence gate markers.

Contains the sequence of prime numbers that appear as Fibonacci numbers F_n.
These primes serve as "transcendence gates" in SRT theory, marking critical
points of ontological phase transitions and consciousness emergence.

Values: [2, 3, 5, 13, 89, 233, 1597, 28657, 514229, 433494437, 2971215073]

Physical Significance:
- Transcendence gates: F_n where F_n is prime marks ontological boundaries
- Consciousness emergence: Prime indices correspond to self-reference thresholds
- Neural networks: Prime dimensions for stable resonance patterns
"""

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
    # SRT axioms and constants
    "AXIOMS",
    "MODULAR_VOLUME",
    # Prime sequences (Five Operators of Existence)
    "FERMAT_PRIMES",
    "MERSENNE_EXPONENTS",
    "LUCAS_SEQUENCE",
    "LUCAS_PRIMES",
    "M11_BARRIER",
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
