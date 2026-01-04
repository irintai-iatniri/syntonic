"""
Syntonic Applications - Applied Sciences Extension of SRT.

This module implements Phase 6 of Syntonic: the extension of Syntony Recursion
Theory to thermodynamics, chemistry, biology, consciousness, and ecology.

All phenomena emerge from the same geometric principles:
- φ (golden ratio) governs efficiency and hierarchy
- q ≈ 0.027395 (syntony deficit) sets correction scales
- E* ≈ 19.999 eV (spectral constant) anchors mass/energy scales
- K = 24 (kissing number) defines the consciousness threshold

Submodules:
    thermodynamics: DHSR cycle as thermodynamic engine, Four Laws
    condensed_matter: Band theory, superconductivity, quantum Hall
    chemistry: Electronegativity, bonding, periodic table
    biology: Life topology, abiogenesis, metabolism, evolution
    consciousness: K=24 threshold, neural systems, qualia
    ecology: Ecosystem syntony, Gaia, trophic dynamics

Example:
    >>> from syntonic.applications import thermodynamics
    >>> engine = thermodynamics.DHSRThermodynamicCycle()
    >>> engine.EFFICIENCY  # η = 1/φ ≈ 0.618
    0.6180339887498949
"""

from syntonic.applications import thermodynamics
from syntonic.applications import condensed_matter
from syntonic.applications import chemistry
from syntonic.applications import biology
from syntonic.applications import consciousness
from syntonic.applications import ecology

__all__ = [
    'thermodynamics',
    'condensed_matter',
    'chemistry',
    'biology',
    'consciousness',
    'ecology',
]
