"""
Hadrons Module - Nucleon and meson masses from SRT.

The hadron masses are computed using the E* spectral constant
with simple integer or near-integer multipliers.

Example:
    >>> from syntonic.physics.hadrons import HadronMasses
    >>> h = HadronMasses()
    >>> h.proton_mass()
    938.272  # MeV
"""

from syntonic.physics.hadrons.masses import (
    HadronMasses,
    b_meson_mass,
    d_meson_mass,
    kaon_mass,
    neutron_mass,
    neutron_proton_mass_diff,
    pion_mass,
    proton_mass,
)

__all__ = [
    "proton_mass",
    "neutron_mass",
    "neutron_proton_mass_diff",
    "pion_mass",
    "kaon_mass",
    "d_meson_mass",
    "b_meson_mass",
    "HadronMasses",
]
