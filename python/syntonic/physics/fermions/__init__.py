"""
Fermions Module - Lepton and quark masses from winding topology.

This module derives all 9 charged fermion masses from T⁴ winding
configurations and the golden recursion formula.

Submodules:
    windings: Flavor-winding matrices for all generations
    leptons: Electron, muon, tau masses
    quarks: Up, down, strange, charm, bottom, top masses

Example:
    >>> from syntonic.physics.fermions import FermionMasses
    >>> fm = FermionMasses()
    >>> fm.electron_mass()
    0.511  # MeV
    >>> fm.top_mass()
    172.72  # GeV
"""

from syntonic.physics.fermions.windings import (
    ELECTRON_WINDING,
    MUON_WINDING,
    TAU_WINDING,
    UP_WINDING,
    DOWN_WINDING,
    STRANGE_WINDING,
    CHARM_WINDING,
    BOTTOM_WINDING,
    TOP_WINDING,
    LEPTON_WINDINGS,
    UP_TYPE_WINDINGS,
    DOWN_TYPE_WINDINGS,
)

from syntonic.physics.fermions.leptons import (
    electron_mass,
    muon_mass,
    tau_mass,
    mass_from_depth,
)

from syntonic.physics.fermions.quarks import (
    up_mass,
    down_mass,
    strange_mass,
    charm_mass,
    bottom_mass,
    top_mass,
)


class FermionMasses:
    """
    Complete fermion mass derivation from SRT winding topology.

    All masses derived from {φ, E*, q} with zero free parameters.

    Methods:
        electron_mass(): Electron mass in MeV
        muon_mass(): Muon mass in MeV
        tau_mass(): Tau mass in MeV
        up_mass(): Up quark mass in MeV
        down_mass(): Down quark mass in MeV
        strange_mass(): Strange quark mass in MeV
        charm_mass(): Charm quark mass in MeV
        bottom_mass(): Bottom quark mass in MeV
        top_mass(): Top quark mass in GeV

    Example:
        >>> fm = FermionMasses()
        >>> fm.tau_mass()
        1776.86  # MeV
    """

    def electron_mass(self) -> float:
        """Electron mass in MeV."""
        return electron_mass()

    def muon_mass(self) -> float:
        """Muon mass in MeV."""
        return muon_mass()

    def tau_mass(self) -> float:
        """Tau mass in MeV."""
        return tau_mass()

    def up_mass(self) -> float:
        """Up quark mass in MeV (MS-bar at 2 GeV)."""
        return up_mass()

    def down_mass(self) -> float:
        """Down quark mass in MeV (MS-bar at 2 GeV)."""
        return down_mass()

    def strange_mass(self) -> float:
        """Strange quark mass in MeV (MS-bar at 2 GeV)."""
        return strange_mass()

    def charm_mass(self) -> float:
        """Charm quark mass in MeV (MS-bar at m_c)."""
        return charm_mass()

    def bottom_mass(self) -> float:
        """Bottom quark mass in MeV (MS-bar at m_b)."""
        return bottom_mass()

    def top_mass(self, loop_order: int = 1) -> float:
        """Top quark mass in GeV (pole mass)."""
        return top_mass(loop_order=loop_order)

    def all_masses(self) -> dict:
        """Return all fermion masses as a dictionary."""
        return {
            'm_e': self.electron_mass(),
            'm_mu': self.muon_mass(),
            'm_tau': self.tau_mass(),
            'm_u': self.up_mass(),
            'm_d': self.down_mass(),
            'm_s': self.strange_mass(),
            'm_c': self.charm_mass(),
            'm_b': self.bottom_mass(),
            'm_t': self.top_mass(),
        }

    def __repr__(self) -> str:
        return "FermionMasses(SRT-derived)"


__all__ = [
    # Windings
    'ELECTRON_WINDING',
    'MUON_WINDING',
    'TAU_WINDING',
    'UP_WINDING',
    'DOWN_WINDING',
    'STRANGE_WINDING',
    'CHARM_WINDING',
    'BOTTOM_WINDING',
    'TOP_WINDING',
    'LEPTON_WINDINGS',
    'UP_TYPE_WINDINGS',
    'DOWN_TYPE_WINDINGS',
    # Functions
    'electron_mass',
    'muon_mass',
    'tau_mass',
    'up_mass',
    'down_mass',
    'strange_mass',
    'charm_mass',
    'bottom_mass',
    'top_mass',
    'mass_from_depth',
    # Class
    'FermionMasses',
]
