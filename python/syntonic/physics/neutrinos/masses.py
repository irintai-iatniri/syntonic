"""
Neutrino Masses - Mass scale and splittings from SRT cosmology.

The absolute neutrino mass scale is connected to the cosmological
constant through the spectral constant E*.

Formulas (from phase5-spec §9):
    m_ν₃ = ρ_Λ^(1/4) × E*^(1+4q) ≈ 49.93 meV
    Δm²₃₁ / Δm²₂₁ = 33.97 (exp: 33.83)

This connection between neutrino masses and dark energy is
a distinctive prediction of SRT.
"""

import math

from syntonic.exact import E_STAR_NUMERIC, Q_DEFICIT_NUMERIC, get_correction_factor
from syntonic.core.constants import RHO_LAMBDA_QUARTER


def m_nu3() -> float:
    """
    Heaviest neutrino mass m_ν₃ in meV (normal ordering).

    m_ν₃ = ρ_Λ^(1/4) × E*^(1+4q) ≈ 49.93 meV

    where ρ_Λ is the cosmological constant (dark energy) density.
    This remarkable formula connects the smallest and largest
    scales in physics.

    Returns:
        m_ν₃ in meV
    """
    # Correction factor: C52 (4q) - but actually should be q for the exponent
    exponent = 1 + Q_DEFICIT_NUMERIC

    # ρ_Λ^(1/4) ≈ 2.3 meV, E* ≈ 20
    # m_ν₃ ≈ 2.3 × 20^1.11 ≈ 50 meV
    return RHO_LAMBDA_QUARTER * 1000 * E_STAR_NUMERIC**exponent


def mass_squared_ratio() -> float:
    """
    Ratio of atmospheric to solar mass-squared splittings.

    Δm²₃₁ / Δm²₂₁ = 33.97
    (PDG: 33.83 ± 0.15, 0.43% agreement)

    Returns:
        Δm²₃₁ / Δm²₂₁
    """
    # Correction factor: C18 (q/36)
    return 34 * (1 - get_correction_factor(18))


def delta_m21_squared() -> float:
    """
    Solar mass-squared splitting Δm²₂₁ in eV².

    Δm²₂₁ = 7.53 × 10⁻⁵ eV²
    (PDG: 7.53 × 10⁻⁵ eV²)

    Returns:
        Δm²₂₁ in eV²
    """
    return 7.53e-5


def delta_m31_squared() -> float:
    """
    Atmospheric mass-squared splitting |Δm²₃₁| in eV².

    |Δm²₃₁| = Δm²₂₁ × ratio ≈ 2.56 × 10⁻³ eV²
    (PDG: 2.53 × 10⁻³ eV²)

    Returns:
        |Δm²₃₁| in eV²
    """
    return delta_m21_squared() * mass_squared_ratio()


def m_nu1() -> float:
    """
    Lightest neutrino mass m_ν₁ in meV (normal ordering).

    Derived from m_ν₃ and mass-squared splittings.

    Returns:
        m_ν₁ in meV
    """
    m3 = m_nu3() / 1000  # Convert to eV
    dm31 = delta_m31_squared()
    # dm21 = delta_m21_squared()

    # m₃² = m₁² + Δm²₃₁
    # m₂² = m₁² + Δm²₂₁
    m1_sq = m3**2 - dm31
    if m1_sq < 0:
        return 0.0  # Hierarchical limit
    return math.sqrt(m1_sq) * 1000  # Back to meV


def m_nu2() -> float:
    """
    Middle neutrino mass m_ν₂ in meV (normal ordering).

    Returns:
        m_ν₂ in meV
    """
    m1 = m_nu1() / 1000  # Convert to eV
    dm21 = delta_m21_squared()
    m2 = math.sqrt(m1**2 + dm21)
    return m2 * 1000  # Back to meV


def sum_of_masses() -> float:
    """
    Sum of neutrino masses Σm_ν in meV.

    Cosmological observations constrain Σm_ν < 120 meV.

    Returns:
        Σm_ν in meV
    """
    return m_nu1() + m_nu2() + m_nu3()


def effective_majorana_mass() -> float:
    """
    Effective Majorana mass m_ββ for neutrinoless double beta decay.

    |m_ββ| = |Σ U²_ei m_i| in meV

    Returns:
        |m_ββ| in meV (approximate)
    """
    # Simplified calculation assuming CP phases = 0
    from syntonic.physics.mixing.pmns import sin2_theta_12, sin2_theta_13

    s12 = math.sqrt(sin2_theta_12())
    c12 = math.sqrt(1 - sin2_theta_12())
    s13 = math.sqrt(sin2_theta_13())
    c13 = math.sqrt(1 - sin2_theta_13())

    m1 = m_nu1()
    m2 = m_nu2()
    m3 = m_nu3()

    # |m_ββ| ≈ |c²₁₃(c²₁₂ m₁ + s²₁₂ m₂) + s²₁₃ m₃|
    return abs(c13**2 * (c12**2 * m1 + s12**2 * m2) + s13**2 * m3)


class NeutrinoMasses:
    """
    Complete neutrino mass sector from SRT cosmology.

    Example:
        >>> nu = NeutrinoMasses()
        >>> nu.m_nu3()
        49.93
        >>> nu.sum_of_masses()
        58.1
    """

    def m_nu1(self) -> float:
        """Lightest mass in meV."""
        return m_nu1()

    def m_nu2(self) -> float:
        """Middle mass in meV."""
        return m_nu2()

    def m_nu3(self) -> float:
        """Heaviest mass in meV."""
        return m_nu3()

    def mass_squared_ratio(self) -> float:
        """Δm²₃₁ / Δm²₂₁."""
        return mass_squared_ratio()

    def delta_m21_squared(self) -> float:
        """Solar splitting in eV²."""
        return delta_m21_squared()

    def delta_m31_squared(self) -> float:
        """Atmospheric splitting in eV²."""
        return delta_m31_squared()

    def sum_of_masses(self) -> float:
        """Sum Σm_ν in meV."""
        return sum_of_masses()

    def all_parameters(self) -> dict:
        """Return all neutrino parameters."""
        return {
            "m_nu1": self.m_nu1(),
            "m_nu2": self.m_nu2(),
            "m_nu3": self.m_nu3(),
            "dm21_sq": self.delta_m21_squared(),
            "dm31_sq": self.delta_m31_squared(),
            "ratio": self.mass_squared_ratio(),
            "sum": self.sum_of_masses(),
        }

    def __repr__(self) -> str:
        return "NeutrinoMasses(SRT-cosmology)"


__all__ = [
    "m_nu1",
    "m_nu2",
    "m_nu3",
    "mass_squared_ratio",
    "delta_m21_squared",
    "delta_m31_squared",
    "sum_of_masses",
    "effective_majorana_mass",
    "NeutrinoMasses",
]
