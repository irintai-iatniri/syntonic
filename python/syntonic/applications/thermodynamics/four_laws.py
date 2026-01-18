"""
Four Laws of Syntony Thermodynamics.

These are not analogs - they ARE thermodynamics at its geometric core.

Zeroth Law: Universal Connection (transitivity of equilibrium)
First Law: Conservation of Winding Energy [Ĥ, N̂_total] = 0
Second Law: The Syntonic Imperative dF/dt ≤ 0
Third Law: Vacuum Saturation lim_{T→0} S = φ - q
"""

from __future__ import annotations
from typing import TYPE_CHECKING

from syntonic.exact import PHI_NUMERIC, Q_DEFICIT_NUMERIC
from syntonic.srt.geometry import WindingState


class SyntonicLaws:
    """
    The Four Laws of Syntony Thermodynamics.

    These are not analogs - they ARE thermodynamics at its geometric core.

    Example:
        >>> laws = SyntonicLaws()
        >>> print(laws.zeroth_law())
        'If A ~ B and B ~ C (hooking), then T_A = T_B = T_C = φ'
        >>> laws.third_law_limit()
        1.5906...  # φ - q
    """

    @staticmethod
    def zeroth_law() -> str:
        """
        Universal Connection: All systems in stable hooking tend toward T = φ.

        [Ĥ, N̂_total] = transitivity of thermodynamic equilibrium

        Explains why physical constants are uniform throughout the universe.

        Returns:
            Statement of the Zeroth Law
        """
        return "If A ~ B and B ~ C (hooking), then T_A = T_B = T_C = φ"

    @staticmethod
    def first_law() -> str:
        """
        Conservation of Winding Energy: [Ĥ, N̂_total] = 0

        The Hooking Operator commutes with the global winding Hamiltonian.
        In any DHSR cycle: ΔU_total = 0

        The 0.618 that integrates + 0.382 that recycles = 1.0 that entered.

        Returns:
            Statement of the First Law
        """
        return "[Ĥ, N̂_total] = 0 (Conservation of Winding Energy)"

    @staticmethod
    def second_law() -> str:
        """
        The Syntonic Imperative: dF/dt ≤ 0

        Information flows to minimize Free Energy, driving toward
        Golden Measure equilibrium.

        Fokker-Planck: ∂ρ/∂t = D∇²ρ + (1/φ)∇·(nρ)
        - Diffusion term: pushes toward entropy
        - Drift term: pulls toward syntony

        Net flow is always inward → arrow of time.

        Returns:
            Statement of the Second Law
        """
        return "dF/dt ≤ 0 (Free Energy minimization toward Golden Measure)"

    @staticmethod
    def third_law() -> str:
        """
        Vacuum Saturation: lim_{T→0} S_syntonic = φ - q ≈ 1.591

        Perfect syntony (S = φ) is unreachable.
        The deficit q ≈ 0.027395 is the "zero-point entropy" ensuring
        existence never fully resolves into static nothingness.

        q is the breath between reaching and arriving.

        Returns:
            Statement of the Third Law
        """
        return f"lim_{{T→0}} S = φ - q ≈ {PHI_NUMERIC - Q_DEFICIT_NUMERIC:.4f}"

    @staticmethod
    def third_law_limit() -> float:
        """
        Compute the Third Law limit: φ - q.

        Returns:
            The vacuum saturation syntony ≈ 1.591
        """
        return PHI_NUMERIC - Q_DEFICIT_NUMERIC

    @staticmethod
    def describe_all() -> str:
        """
        Return description of all four laws.

        Returns:
            Multi-line description of all laws
        """
        return """
The Four Laws of Syntony Thermodynamics:

ZEROTH LAW (Universal Connection):
    If A ~ B and B ~ C (hooking), then T_A = T_B = T_C = φ
    All systems in stable hooking tend toward temperature T = φ.

FIRST LAW (Conservation):
    [Ĥ, N̂_total] = 0
    The Hooking Operator commutes with the global winding Hamiltonian.
    In any DHSR cycle: ΔU_total = 0

SECOND LAW (Syntonic Imperative):
    dF/dt ≤ 0
    Information flows to minimize Free Energy, driving toward
    Golden Measure equilibrium. This is the arrow of time.

THIRD LAW (Vacuum Saturation):
    lim_{T→0} S = φ - q ≈ 1.591
    Perfect syntony is unreachable. The deficit q ensures
    existence never fully resolves into static nothingness.
"""


class InformationPressure:
    """
    P = 1/φ ≈ 0.618 - constant throughout winding space.

    This constant pressure drives time's arrow and gravity.

    The pressure arises from the Golden Measure gradient:
    P(n) = -∂ln μ(n)/∂|n|² = 1/φ

    Every point in T⁴ experiences the same push toward lower |n|.

    Attributes:
        VALUE: The constant pressure 1/φ ≈ 0.618

    Example:
        >>> p = InformationPressure()
        >>> p.VALUE
        0.6180339887498949
        >>> p.compute(winding_state(1, 0, 0, 0))
        0.6180339887498949
    """

    VALUE = 1 / PHI_NUMERIC  # P ≈ 0.618

    @staticmethod
    def compute(winding_state: 'WindingState' = None) -> float:
        """
        Compute information pressure at a winding state.

        P(n) = -∂ln μ(n)/∂|n|² = 1/φ

        Every point in T⁴ experiences the same push toward lower |n|.
        The pressure is constant (independent of position).

        Args:
            winding_state: Winding state (optional, result is always 1/φ)

        Returns:
            Information pressure = 1/φ
        """
        return 1 / PHI_NUMERIC

    @staticmethod
    def gradient_force(winding_state: 'WindingState') -> float:
        """
        Force from pressure gradient on a winding state.

        F = -P × ∂|n|²/∂n = -2P × |n|

        The force points toward lower |n| (toward vacuum).

        Args:
            winding_state: Winding state

        Returns:
            Force magnitude (negative = toward vacuum)
        """
        P = 1 / PHI_NUMERIC
        return -2 * P * winding_state.norm

    @staticmethod
    def work_against_pressure(delta_norm_sq: float) -> float:
        """
        Work done against information pressure.

        W = P × Δ|n|² = (1/φ) × Δ|n|²

        Args:
            delta_norm_sq: Change in |n|²

        Returns:
            Work done (positive if |n|² increases)
        """
        P = 1 / PHI_NUMERIC
        return P * delta_norm_sq

    def __repr__(self) -> str:
        return f"InformationPressure(P={self.VALUE:.6f})"
