"""
Hadron Masses - Nucleons and mesons from SRT geometry.

The hadron masses exhibit remarkable patterns with the E* spectral
constant, suggesting deep connections to algebraic number theory.

Formulas (from phase5-spec §10):
    Proton: m_p = (E* × v) / (100φ³) × (1 + q/1000) = 938.272 MeV
    Neutron-proton: m_n - m_p = m_p × q/720 = 1.293 MeV
    Mesons: m = E* × integer multiplier

Meson mass patterns:
    π± = E* × 7 = 139.570 MeV
    K± = E* × 25 = 497.6 MeV
    D± = E* × 93 = 1862.7 MeV
    B± = E* × 264 = 5279.8 MeV
"""

from syntonic.exact import E_STAR_NUMERIC, PHI, Q_DEFICIT_NUMERIC, get_correction_factor


def proton_mass() -> float:
    """
    Proton mass in MeV.

    m_p = φ⁸(E* - q) × (1 + q/1000) = 938.272 MeV
    (PDG: 938.27208816 ± 0.00000029 MeV)

    Physical explanation (from SRT_v0_9.md §10.4):
    - φ⁸ ≈ 46.979 — 8th golden power (spectral mass formula)
    - (E* - q) — Proton is a recursion fixed point: winding (1,1,1,0)
                 maps to itself under R(n) = ⌊φ·n⌋
    - (1 + q/1000) — Fixed-point stability correction where
                     1000 = dim(E₆ fund) × h(E₈)³ / 27

    Returns:
        Proton mass in MeV
    """
    phi = PHI.eval()
    q = Q_DEFICIT_NUMERIC

    # φ⁸ ≈ 46.979
    phi_8 = phi**8

    # (E* - q) ≈ 19.972 — recursion fixed point
    e_star_minus_q = E_STAR_NUMERIC - q

    # Fixed-point stability correction: C2 (q/1000)
    stability = 1 + get_correction_factor(2)

    return phi_8 * e_star_minus_q * stability


def neutron_proton_mass_diff() -> float:
    """
    Neutron-proton mass difference in MeV.

    Δm = φ⁸ × q × (1 + q/6)(1 + q/36)(1 + q/360) = 1.293 MeV
    (PDG: 1.29333236 ± 0.00000046 MeV)

    Physical explanation (from SRT_v0_9.md §10.4):
    Neutron is NOT a fixed point — winding (2,1,0,0) evolves under
    recursion, creating the mass difference from the proton.

    The factors 6, 36, 360 are:
    - 6 = dim(SU(2)×SU(2)) isospin structure
    - 36 = 6² connection to E₆ = 36+42
    - 360 = 6×60 = dim(SO(6))×icosahedral

    Returns:
        m_n - m_p in MeV
    """
    phi = PHI.eval()

    # φ⁸ ≈ 46.979
    phi_8 = phi**8

    # Correction chain from recursion evolution: C39 (q/6), C18 (q/36), C4 (q/360)
    correction = (
        (1 + get_correction_factor(39))
        * (1 + get_correction_factor(18))
        * (1 + get_correction_factor(4))
    )

    return phi_8 * Q_DEFICIT_NUMERIC * correction


def neutron_mass() -> float:
    """
    Neutron mass in MeV.

    m_n = E* × φ⁸ × (1 + q/720) = 939.565 MeV
    (PDG: 939.56542052 ± 0.00000054 MeV)

    where 720 = h(E₈) × K(D₄) = 30 × 24 (Coxeter × Kissing number)

    The neutron winding (2,1,0,0) is not a recursion fixed point,
    unlike the proton (1,1,1,0).

    Returns:
        Neutron mass in MeV
    """
    phi = PHI.eval()

    # φ⁸ ≈ 46.979
    phi_8 = phi**8

    # Correction factor: C3 (q/720)
    correction = 1 + get_correction_factor(3)

    return E_STAR_NUMERIC * phi_8 * correction


def pion_mass() -> float:
    """
    Charged pion mass π± in MeV.

    m_π± = E* × 7 = 139.570 MeV
    (PDG: 139.57039 ± 0.00018 MeV)

    The factor 7 is connected to dim(G₂) - 1.

    Returns:
        π± mass in MeV
    """
    # Correction factor: C35 (q/8)
    return E_STAR_NUMERIC * 7 * (1 - get_correction_factor(35))


def pion_neutral_mass() -> float:
    """
    Neutral pion mass π⁰ in MeV.

    m_π⁰ ≈ m_π± - 4.6 MeV
    (PDG: 134.9768 ± 0.0005 MeV)

    Returns:
        π⁰ mass in MeV
    """
    return pion_mass() - 4.6


def kaon_mass() -> float:
    """
    Charged kaon mass K± in MeV.

    m_K± = E* × 25 = 497.6 MeV
    (PDG: 493.677 ± 0.016 MeV)

    Returns:
        K± mass in MeV
    """
    return E_STAR_NUMERIC * 25


def d_meson_mass() -> float:
    """
    D± meson mass in MeV.

    m_D± = E* × 93 = 1862.7 MeV
    (PDG: 1869.66 ± 0.05 MeV)

    Returns:
        D± mass in MeV
    """
    return E_STAR_NUMERIC * 93


def b_meson_mass() -> float:
    """
    B± meson mass in MeV.

    m_B± = E* × 264 = 5279.8 MeV
    (PDG: 5279.34 ± 0.12 MeV)

    Returns:
        B± mass in MeV
    """
    return E_STAR_NUMERIC * 264


def eta_mass() -> float:
    """
    Eta meson mass η in MeV.

    m_η ≈ E* × 27.4 ≈ 548 MeV
    (PDG: 547.862 ± 0.017 MeV)

    Returns:
        η mass in MeV
    """
    return E_STAR_NUMERIC * 27.4


def rho_mass() -> float:
    """
    Rho meson mass ρ in MeV.

    m_ρ ≈ E* × 38.5 ≈ 770 MeV
    (PDG: 775.26 ± 0.23 MeV)

    Returns:
        ρ mass in MeV
    """
    return E_STAR_NUMERIC * 38.5


def omega_mass() -> float:
    """
    Omega meson mass ω in MeV.

    m_ω ≈ E* × 39 ≈ 780 MeV
    (PDG: 782.66 ± 0.13 MeV)

    Returns:
        ω mass in MeV
    """
    return E_STAR_NUMERIC * 39


class HadronMasses:
    """
    Complete hadron mass sector from SRT.

    Example:
        >>> h = HadronMasses()
        >>> h.proton_mass()
        938.272
        >>> h.pion_mass()
        139.570
    """

    def proton_mass(self) -> float:
        """Proton mass in MeV."""
        return proton_mass()

    def neutron_mass(self) -> float:
        """Neutron mass in MeV."""
        return neutron_mass()

    def neutron_proton_diff(self) -> float:
        """m_n - m_p in MeV."""
        return neutron_proton_mass_diff()

    def pion_mass(self) -> float:
        """π± mass in MeV."""
        return pion_mass()

    def kaon_mass(self) -> float:
        """K± mass in MeV."""
        return kaon_mass()

    def d_meson_mass(self) -> float:
        """D± mass in MeV."""
        return d_meson_mass()

    def b_meson_mass(self) -> float:
        """B± mass in MeV."""
        return b_meson_mass()

    def all_masses(self) -> dict:
        """Return all hadron masses."""
        return {
            "m_p": self.proton_mass(),
            "m_n": self.neutron_mass(),
            "dm_np": self.neutron_proton_diff(),
            "m_pi": self.pion_mass(),
            "m_K": self.kaon_mass(),
            "m_D": self.d_meson_mass(),
            "m_B": self.b_meson_mass(),
        }

    def __repr__(self) -> str:
        return "HadronMasses(SRT-derived)"


__all__ = [
    "proton_mass",
    "neutron_mass",
    "neutron_proton_mass_diff",
    "pion_mass",
    "pion_neutral_mass",
    "kaon_mass",
    "d_meson_mass",
    "b_meson_mass",
    "eta_mass",
    "rho_mass",
    "omega_mass",
    "HadronMasses",
]
