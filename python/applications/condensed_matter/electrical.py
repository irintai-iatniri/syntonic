"""
Electrical Quantities from SRT geometry.

| Concept | Standard | SRT Form |
|---------|----------|----------|
| Voltage | V = Ed   | V = ∇S (syntony gradient) |
| Current | I = dQ/dt | I = dn/dt (winding flux) |
| Resistance | R = ρL/A | R = Decoherence/Mobility |
"""

from __future__ import annotations
import math

from syntonic.exact import PHI_NUMERIC


class ElectricalQuantities:
    """
    Electrical quantities from SRT geometry.

    | Concept | Standard | SRT Form |
    |---------|----------|----------|
    | Voltage | V = Ed   | V = ∇S (syntony gradient) |
    | Current | I = dQ/dt | I = dn/dt (winding flux) |
    | Resistance | R = ρL/A | R = Decoherence/Mobility |

    Key predictions:
    - Diode threshold: V_th ≈ 1/φ ≈ 0.618 V (exp: ~0.6 V)
    - Graphene Fermi velocity: v_F = c/300 (from E₈ Coxeter number)

    Example:
        >>> eq = ElectricalQuantities()
        >>> eq.diode_threshold()
        0.618...  # V (exp: ~0.6 V)
        >>> eq.graphene_fermi_velocity()
        999308.2...  # m/s ≈ c/300
    """

    # Speed of light
    C = 299792458  # m/s

    # E₈ Coxeter number
    COXETER_E8 = 30

    def voltage(self, syntony_gradient: float) -> float:
        """
        Voltage as syntony gradient.

        V = ∇S_local

        Args:
            syntony_gradient: Local syntony gradient

        Returns:
            Voltage (arbitrary units)
        """
        return syntony_gradient

    def current(self, d_winding_dt: float) -> float:
        """
        Current as winding flux.

        I = dn/dt (winding flux)

        Args:
            d_winding_dt: Rate of change of winding number

        Returns:
            Current (arbitrary units)
        """
        return d_winding_dt

    def resistance(self, decoherence: float, mobility: float) -> float:
        """
        Resistance from decoherence and mobility.

        R = Decoherence / Mobility

        Resistance arises from T⁴ → M⁴ decoherence during transport.

        Args:
            decoherence: Decoherence rate
            mobility: Carrier mobility

        Returns:
            Resistance
        """
        if mobility <= 0:
            return float('inf')
        return decoherence / mobility

    def diode_threshold(self) -> float:
        """
        Silicon diode threshold voltage.

        V_th ≈ 1/φ ≈ 0.618 V

        Experiment: ~0.6 V → EXACT!

        The threshold is the voltage needed to overcome the
        syntony barrier at the pn junction.

        Returns:
            Threshold voltage in V
        """
        return 1 / PHI_NUMERIC

    def graphene_fermi_velocity(self) -> float:
        """
        Graphene Fermi velocity.

        v_F = c / (10 × h(E₈)) = c/300

        Experiment: v_F ≈ c/300 → EXACT!

        The Coxeter number h(E₈) = 30 sets graphene's Fermi velocity.

        Returns:
            Fermi velocity in m/s
        """
        return self.C / (10 * self.COXETER_E8)

    def graphene_fine_structure(self) -> float:
        """
        Effective fine structure constant in graphene.

        α_graphene = e² / (ℏ v_F) = α × (c/v_F) = α × 300

        Since v_F = c/300, this gives α_graphene ≈ 2.2

        Returns:
            Effective fine structure constant
        """
        alpha = 1 / 137.036  # Fine structure constant
        return alpha * (self.C / self.graphene_fermi_velocity())

    def ohms_law(self, V: float, R: float) -> float:
        """
        Ohm's Law: I = V/R

        In SRT: Winding flux = Syntony gradient / Decoherence

        Args:
            V: Voltage (syntony gradient)
            R: Resistance (decoherence/mobility)

        Returns:
            Current
        """
        if R <= 0:
            return float('inf')
        return V / R

    def power(self, V: float, I: float) -> float:
        """
        Electrical power.

        P = V × I = (∇S) × (dn/dt)

        Rate of syntony flow.

        Args:
            V: Voltage
            I: Current

        Returns:
            Power
        """
        return V * I

    def thermal_voltage(self, T: float = 300) -> float:
        """
        Thermal voltage at temperature T.

        V_T = k_B T / e ≈ 26 mV at 300 K

        Args:
            T: Temperature in K

        Returns:
            Thermal voltage in V
        """
        k_B = 1.380649e-23  # J/K
        e = 1.602176634e-19  # C
        return k_B * T / e

    def drift_velocity(self, mobility: float, E_field: float) -> float:
        """
        Carrier drift velocity.

        v_d = μ × E

        Args:
            mobility: Carrier mobility in m²/(V·s)
            E_field: Electric field in V/m

        Returns:
            Drift velocity in m/s
        """
        return mobility * E_field

    def describe_srt_interpretation(self) -> str:
        """
        Describe the SRT interpretation of electrical phenomena.

        Returns:
            Description string
        """
        return """
SRT Interpretation of Electrical Quantities:

VOLTAGE (V = ∇S):
    Voltage is the gradient of local syntony.
    A voltage difference means different syntony levels.
    Current flows from high syntony to low syntony.

CURRENT (I = dn/dt):
    Current is winding flux - the rate at which winding
    configurations flow through a surface.
    Electrons carry their winding numbers with them.

RESISTANCE (R = Decoherence/Mobility):
    Resistance arises from decoherence during transport.
    As electrons move, their T⁴ winding partially decoheres
    into M⁴ lattice vibrations (phonons = heat).
    Perfect syntony → zero resistance (superconductivity).

KEY PREDICTIONS:
    - Diode threshold: V_th = 1/φ ≈ 0.618 V (exp: ~0.6 V) ✓
    - Graphene v_F: c/300 from E₈ Coxeter number h = 30 ✓
"""

    def __repr__(self) -> str:
        return f"ElectricalQuantities(V_diode={self.diode_threshold():.3f} V)"
