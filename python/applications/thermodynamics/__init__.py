"""
Thermodynamics - DHSR cycle as the fundamental thermodynamic engine.

The DHSR (Differentiation-Harmonization-Syntony-Recursion) cycle is the
universal thermodynamic engine of reality with fixed efficiency η = 1/φ ≈ 61.8%.

Classes:
    DHSRThermodynamicCycle: The DHSR cycle as thermodynamic engine
    SyntonicLaws: The Four Laws of Syntony Thermodynamics
    InformationPressure: P = 1/φ constant throughout winding space
    SyntonicEntropy: Entropy from winding distribution
    TemporalCrystallization: Birth of time's arrow
    GnosisTransition: Phase transitions between Gnosis layers

Key Constants:
    EFFICIENCY = 1/φ ≈ 0.618 (maximum thermodynamic efficiency)
    PRESSURE = 1/φ ≈ 0.618 (constant information pressure)
    THIRD_LAW_LIMIT = φ - q ≈ 1.591 (vacuum saturation)

Example:
    >>> from syntonic.applications.thermodynamics import DHSRThermodynamicCycle
    >>> engine = DHSRThermodynamicCycle()
    >>> engine.EFFICIENCY
    0.6180339887498949
"""

from syntonic.applications.thermodynamics.dhsr_engine import (
    DHSRThermodynamicCycle,
    CycleResult,
)
from syntonic.applications.thermodynamics.four_laws import (
    SyntonicLaws,
    InformationPressure,
)
from syntonic.applications.thermodynamics.entropy import (
    SyntonicEntropy,
)
from syntonic.applications.thermodynamics.phase_transitions import (
    TemporalCrystallization,
    GnosisTransition,
)

__all__ = [
    'DHSRThermodynamicCycle',
    'CycleResult',
    'SyntonicLaws',
    'InformationPressure',
    'SyntonicEntropy',
    'TemporalCrystallization',
    'GnosisTransition',
]
