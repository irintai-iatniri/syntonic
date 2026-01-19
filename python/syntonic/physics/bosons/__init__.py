"""
Bosons Module - Gauge and Higgs sector from SRT geometry.

Submodules:
    gauge: W, Z masses, Weinberg angle, couplings
    higgs: Higgs mass (tree + loop), self-coupling

Example:
    >>> from syntonic.physics.bosons import GaugeSector, HiggsSector
    >>> gauge = GaugeSector()
    >>> gauge.w_mass()
    80.357  # GeV
    >>> higgs = HiggsSector()
    >>> higgs.mass()
    125.25  # GeV
"""

from syntonic.physics.bosons.gauge import (
    GaugeSector,
    fine_structure_constant,
    strong_coupling,
    w_mass,
    weinberg_angle,
    z_mass,
    z_width,
)
from syntonic.physics.bosons.higgs import (
    HiggsSector,
    higgs_mass,
    higgs_mass_loop,
    higgs_mass_tree,
    higgs_self_coupling_ratio,
)

__all__ = [
    # Gauge functions
    "weinberg_angle",
    "fine_structure_constant",
    "strong_coupling",
    "w_mass",
    "z_mass",
    "z_width",
    "GaugeSector",
    # Higgs functions
    "higgs_mass_tree",
    "higgs_mass_loop",
    "higgs_mass",
    "higgs_self_coupling_ratio",
    "HiggsSector",
]
