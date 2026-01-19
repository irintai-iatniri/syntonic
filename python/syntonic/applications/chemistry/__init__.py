"""
Chemistry - Electronegativity, bonding, and molecular geometry from SRT.

Electronegativity is not a force but topological pressure—the vacuum geometry
closing open loops. The bond threshold ΔS = 1/φ separates ionic from covalent.

Classes:
    SRTElectronegativity: χ = |∇S_local| from syntony gradient
    BondCharacter: Ionic/covalent classification
    PeriodicTable: Shell structure from T⁴ topology
    MolecularGeometry: VSEPR as syntony optimization

Key Formulas:
    Electronegativity: χ = Z_eff × (8-V) / (φ^k × n)
    Ionic threshold: ΔS > 1/φ ≈ 0.618
    Shell capacity: 2n² electrons
    Tetrahedral angle: arccos(-1/3) = 109.47°

Example:
    >>> from syntonic.applications.chemistry import SRTElectronegativity
    >>> en = SRTElectronegativity()
    >>> en.compute(Z_eff=7, valence=5, k=2, n=2)  # Nitrogen
    3.0...  # Close to Pauling value 3.04
"""

from syntonic.applications.chemistry.bonding import (
    BondCharacter,
)
from syntonic.applications.chemistry.electronegativity import (
    SRTElectronegativity,
)
from syntonic.applications.chemistry.molecular import (
    MolecularGeometry,
)
from syntonic.applications.chemistry.periodic_table import (
    PeriodicTable,
)

__all__ = [
    "SRTElectronegativity",
    "BondCharacter",
    "PeriodicTable",
    "MolecularGeometry",
]
