"""
Physics Constants - Reference scales and experimental values.

This module provides the reference scales used in SRT physics calculations.
Only the electroweak scale v is used as input; everything else is derived.

Constants:
    V_EW: Higgs VEV (246.22 GeV) - the single input scale
    M_Z: Z boson mass (91.1876 GeV) - derived from V_EW
    M_W_PDG: W boson mass PDG value (80.377 GeV) - for comparison
    M_H_PDG: Higgs mass PDG value (125.25 GeV) - for comparison
    ALPHA_EM_0: Fine structure constant at q=0 (1/137.036)

Note: The PDG values are provided for validation only.
      All SRT predictions are derived, not fitted.
"""

import math
from syntonic.exact import PHI, E_STAR_NUMERIC, Q_DEFICIT_NUMERIC

# =============================================================================
# Input Scale (the only input in SRT)
# =============================================================================

V_EW = 246.22  # GeV, Higgs vacuum expectation value

# =============================================================================
# PDG Reference Values (for validation, not used in derivations)
# =============================================================================

M_Z = 91.1876  # GeV, Z boson mass
M_W_PDG = 80.377  # GeV, W boson mass
M_H_PDG = 125.25  # GeV, Higgs mass
ALPHA_EM_0 = 1 / 137.035999084  # Fine structure constant at q=0
ALPHA_S_MZ = 0.1179  # Strong coupling at M_Z

# =============================================================================
# Derived Scales from SRT
# =============================================================================


def gut_scale() -> float:
    """
    GUT unification scale.

    μ_GUT = v × e^(φ⁷) ≈ 1.0 × 10¹⁵ GeV

    Returns:
        GUT scale in GeV
    """
    phi = PHI.eval()
    return V_EW * math.exp(phi ** 7)


def planck_scale_reduced() -> float:
    """
    Reduced Planck mass scale.

    M_Pl / √(8π) ≈ 2.4 × 10¹⁸ GeV

    Returns:
        Reduced Planck mass in GeV
    """
    return 2.435e18  # GeV


def electroweak_symmetry_breaking_scale() -> float:
    """
    Electroweak symmetry breaking scale.

    Returns V_EW (the Higgs VEV).

    Returns:
        EWSB scale in GeV
    """
    return V_EW


def qcd_scale() -> float:
    """
    QCD confinement scale Λ_QCD.

    Derived from SRT via dimensional transmutation.

    Returns:
        QCD scale in MeV
    """
    # Λ_QCD ≈ 217 MeV from SRT
    phi = PHI.eval()
    q = Q_DEFICIT_NUMERIC
    return E_STAR_NUMERIC * 11 * (1 - q / 120)  # ≈ 217 MeV


# =============================================================================
# Cosmological Constants (for neutrino sector)
# =============================================================================

RHO_LAMBDA_QUARTER = 2.3e-3  # eV, (ρ_Λ)^{1/4} dark energy density


# =============================================================================
# Structure Dimensions (for correction factors in physics)
# =============================================================================

# These map physical processes to algebraic structures
PHYSICS_STRUCTURE_MAP = {
    'chiral_suppression': 'E8_positive',     # 120 - chiral fermions
    'generation_crossing': 'E6_positive',     # 36 - golden cone
    'fundamental_rep': 'E6_fundamental',      # 27 - 27 of E6
    'consciousness': 'D4_kissing',            # 24 - D4 kissing number
    'cartan': 'G2_dim',                       # 8 - rank(E8)
}


__all__ = [
    # Input scale
    'V_EW',
    # PDG reference values
    'M_Z',
    'M_W_PDG',
    'M_H_PDG',
    'ALPHA_EM_0',
    'ALPHA_S_MZ',
    # Scale functions
    'gut_scale',
    'planck_scale_reduced',
    'electroweak_symmetry_breaking_scale',
    'qcd_scale',
    # Cosmological
    'RHO_LAMBDA_QUARTER',
    # Structure map
    'PHYSICS_STRUCTURE_MAP',
]
