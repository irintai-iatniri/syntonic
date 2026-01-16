"""
SRT-Zero Kernel: Core Constants Module
=======================================
Source: SRT_Equations.md + Universal_Syntony_Correction_Hierarchy.md

Initialize the geometric seeds of the universe to high precision.
Re-exports key constants from hierarchy.py for backward compatibility.
"""

from __future__ import annotations


# Import core constants from hierarchy module (use package-relative import)
from .hierarchy import (
    PHI, PHI_INV, PI, E, E_STAR, Q,
    H_E8, DIM_E8, ROOTS_E8, ROOTS_E8_POS, RANK_E8,
    DIM_E6, DIM_E6_FUND, ROOTS_E6_POS,
    K_D4, DIM_T4, N_GEN,
    FIBONACCI,
)


class UniverseSeeds:
    """
    The four geometric constants {φ, π, e, 1} from which all physics emerges.
    """
    
    def __init__(self):
        # The Four Seeds
        self.phi: float = PHI
        self.pi: float = PI
        self.e: float = E
        self.one: float = 1.0
        
        # Derived Constants
        self.E_star: float = E_STAR
        self.q: float = Q
        
        # Three-Term Decomposition of E*
        self.E_bulk: float = self._calculate_E_bulk()
        self.E_torsion: float = self._calculate_E_torsion()
        self.E_cone: float = self._calculate_E_cone()
        self.Delta: float = self._calculate_residual()

    def _calculate_E_bulk(self) -> float:
        """Bulk term: Γ(1/4)² ≈ 13.14504720659687"""
        from scipy.special import gamma
        return float(gamma(0.25)**2)

    def _calculate_E_torsion(self) -> float:
        """Torsion term: π(π - 1) ≈ 6.72801174749952"""
        return self.pi * (self.pi - 1)

    def _calculate_E_cone(self) -> float:
        """Cone term: (35/12)e^(-π) ≈ 0.12604059493600"""
        return (35.0 / 12.0) * (self.e ** (-self.pi))

    def _calculate_residual(self) -> float:
        """
        Residual Δ = E* - E_bulk - E_torsion - E_cone
        Expected: Δ ≈ 4.30 × 10⁻⁷
        
        Physical meaning: The 0.02% that doesn't crystallize—
        the "engine of time" driving cosmic evolution.
        """
        return self.E_star - self.E_bulk - self.E_torsion - self.E_cone

    def validate(self) -> dict:
        """
        Validation checks against theoretical values.
        Returns dict of {constant: (computed, expected, match)}.
        """
        results = {}
        
        # E* validation
        E_star_expected = 19.999099979189475767
        E_star_match = abs(self.E_star - E_star_expected) < 1e-15
        results['E_star'] = (self.E_star, E_star_expected, E_star_match)
        
        # q validation
        q_expected = 0.0273951469201761
        q_match = abs(self.q - q_expected) < 1e-12
        results['q'] = (self.q, q_expected, q_match)
        
        # Δ validation (should be ~4.30 × 10⁻⁷)
        Delta_expected = 4.30e-7
        Delta_match = abs(self.Delta - Delta_expected) / Delta_expected < 0.01
        results['Delta'] = (self.Delta, Delta_expected, Delta_match)
        
        # Three-term decomposition validation
        decomposition_sum = self.E_bulk + self.E_torsion + self.E_cone + self.Delta
        decomposition_match = abs(decomposition_sum - self.E_star) < 1e-50
        results['decomposition'] = (decomposition_sum, self.E_star, decomposition_match)
        
        return results

    def __repr__(self) -> str:
        return f"""UniverseSeeds:
  φ  = {self.phi:.20f}
  π  = {self.pi:.20f}
  e  = {self.e:.20f}
  E* = {self.E_star:.20f}
  q  = {self.q:.20f}
  Δ  = {self.Delta:.10e}"""


# Module-level constants for direct import (backward compatibility)
phi = PHI
phi_inv = PHI_INV
pi = PI
e = E
E_star = E_STAR
q = Q

# Group constants
h_E8 = H_E8
K_D4 = K_D4
dim_E8 = DIM_E8
rank_E8 = RANK_E8
roots_E8 = ROOTS_E8
dim_E6 = DIM_E6
dim_E6_fund = DIM_E6_FUND
roots_E6 = ROOTS_E6_POS
dim_T4 = DIM_T4
N_gen = N_GEN


__all__ = [
    "UniverseSeeds",
    # Seeds
    "phi", "phi_inv", "pi", "e", "E_star", "q",
    # Group constants
    "h_E8", "K_D4", "dim_E8", "rank_E8", "roots_E8",
    "dim_E6", "dim_E6_fund", "roots_E6",
    "dim_T4", "N_gen",
    # Uppercase versions
    "PHI", "PHI_INV", "PI", "E", "E_STAR", "Q",
    "H_E8", "DIM_E8", "ROOTS_E8", "ROOTS_E8_POS", "RANK_E8",
    "DIM_E6", "DIM_E6_FUND", "ROOTS_E6_POS",
    "DIM_T4", "N_GEN",
    "FIBONACCI",
]


if __name__ == "__main__":
    # Self-test
    seeds = UniverseSeeds()
    print(seeds)
    print("\nValidation:")
    for name, (computed, expected, match) in seeds.validate().items():
        status = "✓" if match else "✗"
        print(f"  {name}: {status}")
