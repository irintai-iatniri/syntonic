"""
SRT-Zero Kernel: Geometric Library
===================================
Source: Universal_Syntony_Correction_Hierarchy.md

The Complete Object Library of topological invariants (Levels 0-40).
This file defines the 'Geometric DNA' of the universe.

Re-exports key constants from hierarchy.py and provides additional
geometric utilities.
"""

from __future__ import annotations

from typing import Dict

# Import from hierarchy module
from .hierarchy import (
    PHI, PHI_INV, PI, E_STAR, Q,
    H_E8, DIM_E8, ROOTS_E8, ROOTS_E8_POS, RANK_E8,
    DIM_E6, DIM_E6_FUND, ROOTS_E6_POS,
    K_D4, DIM_T4, N_GEN,
    GEOMETRIC_DIVISORS, CORRECTION_HIERARCHY,
)


class GeometricInvariants:
    """The definitive catalog of SRT geometric invariants.

    Provides access to all topological and group-theoretic constants used
    in Syntony Recursion Theory derivations. These invariants arise from
    the E₈ → E₇ → E₆ → SM symmetry breaking chain.

    Structure:
        - Fundamental Constants: φ (golden ratio), π
        - Group Theory: E₈, E₆, D₄ dimensions/roots
        - The 25-Level Correction Hierarchy

    Attributes:
        phi: Golden ratio ≈ 1.618
        pi: Circle constant ≈ 3.14159
        E8_dim: dim(E₈) = 248
        E8_roots: |E₈ root system| = 240
        E6_dim: dim(E₆) = 78
        D4_kissing: K(D₄) = 24 (collapse threshold)

    Examples:
        >>> geom = GeometricInvariants()
        >>> print(geom.E8_dim)
        248
        >>> print(geom.coxeter_kissing)  # h(E₈) × K(D₄)
        720
    """

    def __init__(self) -> None:
        """Initialize the geometric invariants catalog."""
        # Fundamental Seeds (Local copy for composite calcs)
        self.phi: float = PHI
        self.pi: float = PI
        
        # ═══════════════════════════════════════════════════════════════════
        # SECTION A: GROUP THEORETIC PRIMITIVES
        # ═══════════════════════════════════════════════════════════════════
        # E8 Lattice (The Vacuum)
        self.E8_dim: int = DIM_E8          # 248
        self.E8_roots: int = ROOTS_E8       # 240
        self.E8_positive_roots: int = ROOTS_E8_POS  # 120
        self.E8_rank: int = RANK_E8         # 8
        self.h_E8: int = H_E8               # Coxeter number = 30
        
        # E6 Lattice (The Gauge Sector)
        self.E6_dim: int = DIM_E6           # 78
        self.E6_roots: int = 72             # Total roots
        self.E6_positive_roots: int = ROOTS_E6_POS  # 36
        self.E6_fundamental: int = DIM_E6_FUND     # 27
        self.E6_rank: int = 6
        self.E6_cone_roots: int = 36        # Golden Cone
        
        # D4 Lattice (The Spacetime Projection)
        self.D4_kissing: int = K_D4         # 24
        self.D4_dim: int = 28               # dim(SO(8))
        self.D4_rank: int = 4
        
        # T4 Torus (The Winding Space)
        self.torus_dim: int = DIM_T4        # 4
        self.N_gen: int = N_GEN             # 3
        self.topology_gen: int = 12         # dim(T4) * N_gen

        # ═══════════════════════════════════════════════════════════════════
        # SECTION B: KEY COMPOSITE FACTORS
        # ═══════════════════════════════════════════════════════════════════
        
        # Level 1: Fixed-point stability
        self.coxeter_cubed_27: int = 1000   # h(E8)³/27 = 30³/27
        
        # Level 2: Coxeter-Kissing product
        self.coxeter_kissing: int = 720     # h(E8) * K(D4) = 30 * 24 = 6!
        
        # Level 3: Cone cycles
        self.cone_cycles: int = 360         # 10 * 36
        
        # Level 31: Hierarchy exponent
        self.L31_hierarchy: int = (self.h_E8 * self.D4_kissing) - 1  # 719

        # Level 33: Strange Baryon Factor
        self.L33_strange: int = (2 * self.h_E8) + 6  # 66

    def get_full_hierarchy(self) -> Dict[int, float]:
        """Return the complete 25+ level geometric correction hierarchy.

        Maps hierarchy level indices to their corresponding geometric
        divisors. Used by the MassMiner for Deep Resonance Search.

        Returns:
            Dictionary mapping level numbers to divisor values.

        Examples:
            >>> geom = GeometricInvariants()
            >>> hierarchy = geom.get_full_hierarchy()
            >>> print(hierarchy[2])  # Coxeter-Kissing
            720
        """
        return {
            # TIER 1: Stability & Fundamental Forces
            1: 1000,        # Fixed-point
            2: 720,         # Coxeter-Kissing
            3: 360,         # Cone cycles
            4: 248,         # E8 dim
            5: 120,         # E8 positive roots
            6: 78,          # E6 dim
            8: 36,          # Golden Cone
            9: 27,          # E6 fundamental
            
            # TIER 2: QCD Loop Factors
            10: float(6 * PI),       # 6π
            11: float(5 * PI),       # 5π
            12: float(4 * PI),       # 4π
            13: 12,                   # Topology-generation
            14: float(3 * PI),       # 3π
            15: 8,                    # Cartan rank
            
            # TIER 3: Topological
            16: 6,          # Sub-generation
            18: 4,          # Quarter layer
            19: 2,          # Half layer
            
            # TIER 4: Golden
            20: float(self.phi),     # φ
            22: float(self.phi**2),  # φ²
            
            # TIER 5: Exotic
            31: 719,        # Hierarchy exponent
            33: 66,         # Strange baryon
            36: 314,        # Bc meson
            
            # Combinatorial
            100: 248 * 30,  # 7440
            101: 719 * 2,   # 1438
            102: 137.036,   # Fine structure
        }
    
    def get_correction_factor(self, level: int) -> float:
        """Get the divisor for a specific hierarchy level.

        Args:
            level: Hierarchy level number (1-102).

        Returns:
            The geometric divisor for that level, or 1.0 if not defined.

        Examples:
            >>> geom = GeometricInvariants()
            >>> geom.get_correction_factor(2)  # Coxeter-Kissing
            720
        """
        hierarchy = self.get_full_hierarchy()
        return hierarchy.get(level, 1.0)

    def __repr__(self) -> str:
        return f"""GeometricInvariants:
  Levels: 0-24 Defined (25-Level Hierarchy)
  Magic Numbers: {{31: 719, 33: 66, 36: 314}}
  Groups: E8({self.E8_dim}), E6({self.E6_dim}), D4({self.D4_kissing})
  Coxeter: h(E8) = {self.h_E8}
  """


# Named constants for direct import
E8_DIM = DIM_E8
E8_ROOTS = ROOTS_E8
E8_POSITIVE_ROOTS = ROOTS_E8_POS
E8_RANK = RANK_E8
E8_COXETER = H_E8

E6_DIM = DIM_E6
E6_FUNDAMENTAL = DIM_E6_FUND
E6_POSITIVE_ROOTS = ROOTS_E6_POS

D4_KISSING = K_D4

HIERARCHY_EXP = 719
COXETER_KISSING = 720


__all__ = [
    "GeometricInvariants",
    # E8
    "E8_DIM", "E8_ROOTS", "E8_POSITIVE_ROOTS", "E8_RANK", "E8_COXETER",
    # E6
    "E6_DIM", "E6_FUNDAMENTAL", "E6_POSITIVE_ROOTS",
    # D4
    "D4_KISSING",
    # Magic numbers
    "HIERARCHY_EXP", "COXETER_KISSING",
]


if __name__ == "__main__":
    geom = GeometricInvariants()
    print(geom)
    print(f"\nFull hierarchy levels:")
    for level, value in sorted(geom.get_full_hierarchy().items()):
        print(f"  L{level}: {value}")
