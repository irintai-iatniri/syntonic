"""
Running Couplings - Golden RG equations from SRT.

The renormalization group equations are modified by the universal
syntony deficit q, leading to Golden RG flow.

Formulas (from phase5-spec §11):
    d(α_i^-1)/d(ln μ) = -b_i(1+q)/(2π)

    Solution: α^-1(μ) = α^-1(μ₀) - b(1+q)/(2π) × ln(μ/μ₀)

Beta coefficients (SM):
    b₁ = 41/10 (U(1)_Y)
    b₂ = -19/6 (SU(2)_L)
    b₃ = -7 (SU(3)_c)

GUT scale:
    μ_GUT = v × e^(φ⁷) ≈ 1.0 × 10¹⁵ GeV
"""

import math

from syntonic.exact import PHI, Q_DEFICIT_NUMERIC
from syntonic.physics.constants import M_Z, V_EW

# =============================================================================
# Beta Coefficients (Standard Model)
# =============================================================================

B1 = 41 / 10  # U(1)_Y
B2 = -19 / 6  # SU(2)_L
B3 = -7  # SU(3)_c

# =============================================================================
# Running Functions
# =============================================================================


def alpha_running(
    alpha_0: float,
    b: float,
    mu: float,
    mu_0: float = M_Z,
) -> float:
    """
    Run coupling from μ₀ to μ using Golden RG equation.

    d(α^-1)/d(ln μ) = -b(1+q)/(2π)

    Solution: α^-1(μ) = α^-1(μ₀) - b(1+q)/(2π) × ln(μ/μ₀)

    Args:
        alpha_0: Coupling at reference scale
        b: Beta function coefficient
        mu: Target scale in GeV
        mu_0: Reference scale in GeV (default: M_Z)

    Returns:
        Coupling at scale μ
    """
    q = Q_DEFICIT_NUMERIC

    # RG evolution with Golden correction
    log_ratio = math.log(mu / mu_0)
    delta = b * (1 + q) / (2 * math.pi) * log_ratio

    alpha_inv = 1 / alpha_0 - delta

    if alpha_inv <= 0:
        return float("inf")  # Coupling diverges (Landau pole)
    return 1 / alpha_inv


def alpha_em_at_scale(mu: float) -> float:
    """
    Electromagnetic coupling α_EM at scale μ.

    Uses U(1)_Y running with appropriate normalization.

    Args:
        mu: Energy scale in GeV

    Returns:
        α_EM(μ)
    """
    # α_EM at M_Z
    alpha_em_mz = 1 / 127.9  # Running from q=0 to M_Z

    # Run to scale mu
    return alpha_running(alpha_em_mz, B1 * 3 / 5, mu)  # GUT normalization


def alpha_s_at_scale(mu: float) -> float:
    """
    Strong coupling α_s at scale μ.

    Args:
        mu: Energy scale in GeV

    Returns:
        α_s(μ)
    """
    alpha_s_mz = 0.1179
    return alpha_running(alpha_s_mz, B3, mu)


def gut_scale(v: float = V_EW) -> float:
    """
    GUT unification scale.

    μ_GUT = v × e^(φ⁷) ≈ 1.0 × 10¹⁵ GeV

    This is the scale where gauge couplings approximately unify.

    Args:
        v: Higgs VEV in GeV

    Returns:
        GUT scale in GeV
    """
    phi = PHI.eval()
    return v * math.exp(phi**7)


def coupling_unification_check() -> dict:
    """
    Check gauge coupling unification at GUT scale.

    Returns the three SM gauge couplings evaluated at μ_GUT.

    Returns:
        Dictionary with α₁, α₂, α₃ at GUT scale
    """
    mu_gut = gut_scale()

    # Reference values at M_Z
    alpha_1_mz = (5 / 3) / 127.9  # GUT normalized U(1)_Y
    alpha_2_mz = 0.0338  # SU(2)_L
    alpha_3_mz = 0.1179  # SU(3)_c

    # Run to GUT scale
    alpha_1_gut = alpha_running(alpha_1_mz, B1, mu_gut)
    alpha_2_gut = alpha_running(alpha_2_mz, B2, mu_gut)
    alpha_3_gut = alpha_running(alpha_3_mz, B3, mu_gut)

    return {
        "alpha_1": alpha_1_gut,
        "alpha_2": alpha_2_gut,
        "alpha_3": alpha_3_gut,
        "mu_GUT": mu_gut,
        "approximate_unification": abs(alpha_1_gut - alpha_2_gut) / alpha_1_gut < 0.1,
    }


def planck_scale() -> float:
    """
    Reduced Planck mass M_Pl / √(8π).

    Returns:
        Reduced Planck mass in GeV
    """
    return 2.435e18


def hierarchy_ratio() -> float:
    """
    Ratio of Planck to electroweak scale.

    M_Pl / v ≈ 10^16

    Returns:
        M_Pl / v
    """
    return planck_scale() / V_EW


class GoldenRG:
    """
    Golden Renormalization Group from SRT.

    The RG equations are modified by the syntony deficit q,
    leading to slightly different unification behavior.

    Example:
        >>> rg = GoldenRG()
        >>> rg.gut_scale()
        1.0e15
        >>> rg.alpha_s_at_scale(1000)  # α_s at 1 TeV
        0.0885
    """

    def alpha_running(
        self,
        alpha_0: float,
        b: float,
        mu: float,
        mu_0: float = M_Z,
    ) -> float:
        """Run coupling from μ₀ to μ."""
        return alpha_running(alpha_0, b, mu, mu_0)

    def alpha_em_at_scale(self, mu: float) -> float:
        """α_EM at scale μ."""
        return alpha_em_at_scale(mu)

    def alpha_s_at_scale(self, mu: float) -> float:
        """α_s at scale μ."""
        return alpha_s_at_scale(mu)

    def gut_scale(self) -> float:
        """GUT unification scale in GeV."""
        return gut_scale()

    def planck_scale(self) -> float:
        """Reduced Planck mass in GeV."""
        return planck_scale()

    def unification_check(self) -> dict:
        """Check coupling unification at GUT scale."""
        return coupling_unification_check()

    def all_parameters(self) -> dict:
        """Return key RG parameters."""
        return {
            "mu_GUT": self.gut_scale(),
            "M_Planck": self.planck_scale(),
            "alpha_s_1TeV": self.alpha_s_at_scale(1000),
            **self.unification_check(),
        }

    def __repr__(self) -> str:
        return "GoldenRG(SRT-modified)"


__all__ = [
    "B1",
    "B2",
    "B3",
    "alpha_running",
    "alpha_em_at_scale",
    "alpha_s_at_scale",
    "gut_scale",
    "planck_scale",
    "hierarchy_ratio",
    "coupling_unification_check",
    "GoldenRG",
]
