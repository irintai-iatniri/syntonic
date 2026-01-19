"""
Condensed Matter - Band theory, superconductivity, and quantum Hall effects.

Electricity is winding flux; resistance is metric drag;
superconductivity is perfect syntony.

Classes:
    BandStructure: Band theory from T⁴ → M⁴ projection
    Superconductivity: Perfect syntony with BCS ratio π + 1/φ²
    CooperPair: Winding knots forming superconducting pairs
    QuantumHallEffect: T⁴ winding made visible
    ElectricalQuantities: Voltage, current, resistance from SRT

Key Formulas:
    Band gap: E_g = E* × N × q ≈ 0.548N eV
    BCS ratio: 2Δ/k_B T_c = π + 1/φ² ≈ 3.524
    Hall conductance: σ_xy = (e²/h) × n₇
    FQHE fractions: ν = F_n / F_{n+2} (Fibonacci ratios)

Example:
    >>> from syntonic.applications.condensed_matter import BandStructure
    >>> band = BandStructure()
    >>> band.band_gap(N=2)  # Silicon
    1.096...  # eV
"""

from syntonic.applications.condensed_matter.band_theory import (
    BandStructure,
)
from syntonic.applications.condensed_matter.electrical import (
    ElectricalQuantities,
)
from syntonic.applications.condensed_matter.quantum_hall import (
    QuantumHallEffect,
)
from syntonic.applications.condensed_matter.superconductivity import (
    CooperPair,
    Superconductivity,
)

__all__ = [
    "BandStructure",
    "Superconductivity",
    "CooperPair",
    "QuantumHallEffect",
    "ElectricalQuantities",
]
