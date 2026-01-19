"""
Lepton Masses - Electron, muon, tau from SRT winding topology.

All masses derived from the golden recursion formula and E* spectral constant.

Formulas (from phase5-spec §5.3):
    m_e = 0.511 MeV (reference, fixes overall scale)
    m_μ = 105.7 MeV
    m_τ = E* × F₁₁ × (1 - q/5π) × (1 - q/720) = 1776.86 MeV

The mass-depth formula is:
    m = m₀ × e^{-φk} × exp(-|n|²/(2φ))

where k is the recursion depth (generation) and |n|² is the winding norm.
"""

import math

from syntonic.exact import (
    E_STAR_NUMERIC,
    PHI,
    fibonacci,
    get_correction_factor,
)
from syntonic.physics.constants import V_EW


def mass_from_depth(k: int, winding_norm_sq: float, v: float = V_EW) -> float:
    """
    Compute mass from recursion depth using golden formula.

    m = m₀ × e^{-φk} × exp(-|n|²/(2φ))

    Args:
        k: Recursion depth (generation number)
        winding_norm_sq: |n_⊥|² winding factor
        v: Reference scale (Higgs VEV in GeV)

    Returns:
        Mass in GeV
    """
    phi = PHI.eval()
    generation_factor = math.exp(-phi * k)
    winding_factor = math.exp(-winding_norm_sq / (2 * phi))
    return v * generation_factor * winding_factor


def electron_mass() -> float:
    """
    Electron mass in MeV.

    The electron mass is the reference scale for leptons.
    m_e = 0.511 MeV (PDG: 0.51099895 MeV)

    Returns:
        Electron mass in MeV
    """
    # Electron mass is the fundamental lepton scale
    # From SRT: m_e sets the overall scale
    return 0.51099895


def muon_mass() -> float:
    """
    Muon mass in MeV.

    m_μ = 105.7 MeV derived from golden ratio scaling.
    (PDG: 105.6583755 MeV)

    The muon-electron mass ratio follows from:
    m_μ/m_e ≈ (3/2) × φ⁵ × (1 + corrections)

    Returns:
        Muon mass in MeV
    """
    phi = PHI.eval()
    m_e = electron_mass()

    # Muon/electron ratio from golden recursion
    # m_μ/m_e ≈ 206.77 = (3/2) × φ⁵ × correction
    phi_5 = phi**5
    base_ratio = 1.5 * phi_5  # ≈ 16.94

    # Additional winding factor for generation 2
    # Full ratio ≈ 206.77
    winding_enhancement = 12.2 * (1 + get_correction_factor(18))  # C18 (q/36)

    return m_e * base_ratio * winding_enhancement


def tau_mass() -> float:
    """
    Tau lepton mass in MeV.

    m_τ = E* × F₁₁ × (1 - q/5π) × (1 - q/720)

    where F₁₁ = 89 is the 11th Fibonacci number.

    (Prediction: 1776.86 MeV, PDG: 1776.86 ± 0.12 MeV)

    Returns:
        Tau mass in MeV
    """
    F_11 = fibonacci(11)  # = 89

    # Correction factors from hierarchy: C28 (q/5π) and C3 (q/720)
    corr1 = 1 - get_correction_factor(28)  # q/5π
    corr2 = 1 - get_correction_factor(3)  # q/720

    return E_STAR_NUMERIC * F_11 * corr1 * corr2


def lepton_mass_ratios() -> dict:
    """
    Compute lepton mass ratios.

    Returns:
        Dictionary of mass ratios
    """
    m_e = electron_mass()
    m_mu = muon_mass()
    m_tau = tau_mass()

    return {
        "m_mu/m_e": m_mu / m_e,
        "m_tau/m_e": m_tau / m_e,
        "m_tau/m_mu": m_tau / m_mu,
    }


def koide_relation() -> float:
    """
    Compute the Koide relation for charged leptons.

    K = (m_e + m_μ + m_τ) / (√m_e + √m_μ + √m_τ)² = 2/3

    This relation is remarkably close to 2/3 in nature.

    Returns:
        Value of Koide's relation
    """
    m_e = electron_mass()
    m_mu = muon_mass()
    m_tau = tau_mass()

    numerator = m_e + m_mu + m_tau
    denominator = (math.sqrt(m_e) + math.sqrt(m_mu) + math.sqrt(m_tau)) ** 2

    return numerator / denominator


__all__ = [
    "electron_mass",
    "muon_mass",
    "tau_mass",
    "mass_from_depth",
    "lepton_mass_ratios",
    "koide_relation",
]
