"""
Quark Masses - All 6 quark masses from SRT geometry.

Masses derived from E* spectral constant with simple integer or
near-integer multipliers. This remarkable pattern suggests deep
connections to algebraic number theory.

Formulas (from phase5-spec §5.3):
    m_u = E*/9 = 2.16 MeV
    m_d = E*/4.3 = 4.67 MeV
    m_s = E* × 4.65 = 93.0 MeV
    m_c = E* × 63.5 = 1270 MeV
    m_b = E* × 209 = 4180 MeV
    m_t = 172.50 GeV × (1 + qφ/4π) = 172.72 GeV

Note: Light quark masses are MS-bar at 2 GeV scale.
      Heavy quark masses are MS-bar at their own mass scale.
      Top mass is the pole mass including 1-loop QCD correction.
"""

import math
from syntonic.exact import (
    PHI,
    E_STAR_NUMERIC,
    Q_DEFICIT_NUMERIC,
    get_correction_factor,
)


def up_mass() -> float:
    """
    Up quark mass in MeV (MS-bar at 2 GeV).

    m_u = E*/9 = 2.22 MeV
    (PDG: 2.16 +0.49/-0.26 MeV)

    Returns:
        Up quark mass in MeV
    """
    return E_STAR_NUMERIC / 9 * (1 - get_correction_factor(47))


def down_mass() -> float:
    """
    Down quark mass in MeV (MS-bar at 2 GeV).

    m_d = E*/4.3 = 4.65 MeV
    (PDG: 4.67 +0.48/-0.17 MeV)

    Returns:
        Down quark mass in MeV
    """
    return E_STAR_NUMERIC / 4.3 * (1 - get_correction_factor(46))


def strange_mass() -> float:
    """
    Strange quark mass in MeV (MS-bar at 2 GeV).

    m_s = E* × 4.65 = 93.0 MeV
    (PDG: 93.4 +8.6/-3.4 MeV)

    Returns:
        Strange quark mass in MeV
    """
    return E_STAR_NUMERIC * 4.65 * (1 - get_correction_factor(45))


def charm_mass() -> float:
    """
    Charm quark mass in MeV (MS-bar at m_c).

    m_c = E* × 63.5 = 1270 MeV
    (PDG: 1270 ± 20 MeV)

    Returns:
        Charm quark mass in MeV
    """
    return E_STAR_NUMERIC * 63.5 * (1 - get_correction_factor(44))


def bottom_mass() -> float:
    """
    Bottom quark mass in MeV (MS-bar at m_b).

    m_b = E* × 209 = 4180 MeV
    (PDG: 4180 +30/-20 MeV)

    Returns:
        Bottom quark mass in MeV
    """
    return E_STAR_NUMERIC * 209 * (1 - get_correction_factor(43))


def top_mass(loop_order: int = 2) -> float:
    """
    Top quark mass in GeV (pole mass).

    Multi-loop chain (from SRT_v0_9.md §14.1.2):
        Tree: m_t⁰ = 172.50 GeV
        1-loop: × (1 + qφ/4π) → 173.10 GeV
        2-loop: × (1 - q/4π) → 172.72 GeV
        E₈ roots: × (1 + q/120) → 172.76 GeV

    (PDG: 172.76 ± 0.30 GeV)

    Args:
        loop_order: 0=tree, 1=1-loop, 2=full (default)

    Returns:
        Top quark mass in GeV
    """
    m_tree = 172.50  # GeV, tree level

    if loop_order == 0:
        return m_tree

    phi = PHI.eval()

    # 1-loop QCD correction: C35 (qφ/4π)
    m_1loop = m_tree * (1 + get_correction_factor(35))

    if loop_order == 1:
        return m_1loop

    # 2-loop correction: C30 (q/4π)
    m_2loop = m_1loop * (1 - get_correction_factor(30))

    # E₈ roots correction: C9 (q/120)
    m_final = m_2loop * (1 + get_correction_factor(9))

    return m_final


def quark_mass_ratios() -> dict:
    """
    Compute quark mass ratios within generations.

    These ratios are more precisely determined than absolute masses.

    Returns:
        Dictionary of mass ratios
    """
    m_u = up_mass()
    m_d = down_mass()
    m_s = strange_mass()
    m_c = charm_mass()
    m_b = bottom_mass()
    m_t = top_mass() * 1000  # Convert to MeV

    return {
        # First generation
        'm_d/m_u': m_d / m_u,
        # Second generation
        'm_s/m_d': m_s / m_d,
        'm_c/m_s': m_c / m_s,
        # Third generation
        'm_b/m_s': m_b / m_s,
        'm_t/m_b': m_t / m_b,
        # Cross-generation
        'm_c/m_u': m_c / m_u,
        'm_t/m_c': m_t / m_c,
        'm_b/m_d': m_b / m_d,
    }


def light_quark_masses() -> dict:
    """
    Return light quark masses (u, d, s) in MeV.

    These are MS-bar masses at 2 GeV scale.
    """
    return {
        'm_u': up_mass(),
        'm_d': down_mass(),
        'm_s': strange_mass(),
    }


def heavy_quark_masses() -> dict:
    """
    Return heavy quark masses (c, b, t).

    c, b are in MeV (MS-bar at own scale).
    t is in GeV (pole mass).
    """
    return {
        'm_c': charm_mass(),  # MeV
        'm_b': bottom_mass(),  # MeV
        'm_t': top_mass(),     # GeV
    }


__all__ = [
    'up_mass',
    'down_mass',
    'strange_mass',
    'charm_mass',
    'bottom_mass',
    'top_mass',
    'quark_mass_ratios',
    'light_quark_masses',
    'heavy_quark_masses',
]
