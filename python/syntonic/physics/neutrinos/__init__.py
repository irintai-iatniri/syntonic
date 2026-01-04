"""
Neutrinos Module - Neutrino masses from SRT cosmology.

The absolute neutrino mass scale is connected to the cosmological
constant through the spectral constant E*.

Example:
    >>> from syntonic.physics.neutrinos import NeutrinoMasses
    >>> nu = NeutrinoMasses()
    >>> nu.m_nu3()
    49.93  # meV
"""

from syntonic.physics.neutrinos.masses import (
    m_nu3,
    mass_squared_ratio,
    delta_m21_squared,
    delta_m31_squared,
    sum_of_masses,
    NeutrinoMasses,
)

__all__ = [
    'm_nu3',
    'mass_squared_ratio',
    'delta_m21_squared',
    'delta_m31_squared',
    'sum_of_masses',
    'NeutrinoMasses',
]
