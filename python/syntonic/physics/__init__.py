"""
Physics Module - Standard Model parameter derivation from SRT geometry.

This module derives all 25+ Standard Model observables from the universal
syntony deficit q ≈ 0.027395 and geometric constants {φ, π, e, E*}.

Core principle: Every observable emerges from winding topology on T⁴.
Zero free parameters - everything derived from topology.

Submodules:
    fermions: Lepton and quark masses from winding topology
    bosons: W, Z, Higgs masses and gauge couplings
    mixing: CKM and PMNS mixing matrices
    neutrinos: Neutrino masses and splittings
    hadrons: Proton, neutron, meson masses
    running: Renormalization group evolution
    validation: PDG comparison utilities

Example:
    >>> import syntonic as syn
    >>> sm = syn.physics.StandardModel()
    >>> sm.higgs_mass()
    125.25  # GeV
    >>> sm.validate()  # Compare all predictions to PDG
"""

from syntonic.physics.constants import (
    V_EW,
    M_Z,
    M_W_PDG,
    M_H_PDG,
    ALPHA_EM_0,
)

# Import submodules
from syntonic.physics import fermions
from syntonic.physics import bosons
from syntonic.physics import mixing
from syntonic.physics import neutrinos
from syntonic.physics import hadrons
from syntonic.physics import running
from syntonic.physics import validation
from syntonic.physics import sm_verification
from syntonic.physics import gravity

# Import key classes
from syntonic.physics.fermions import FermionMasses
from syntonic.physics.bosons import GaugeSector, HiggsSector
from syntonic.physics.mixing import CKMMatrix, PMNSMatrix
from syntonic.physics.neutrinos import NeutrinoMasses
from syntonic.physics.hadrons import HadronMasses
from syntonic.physics.running import GoldenRG
from syntonic.physics.validation import validate_all, PDG_VALUES
from syntonic.physics.sm_verification import verify_all_observables
from syntonic.physics.gravity import BimetricGravity

# Topology functions for neural network architectures
from syntonic.physics.topology import (
    hooking_coefficient,
    golden_resonance,
    e8_root_alignment,
    compute_tensor_norm,
)


class StandardModel:
    """
    Complete Standard Model parameter derivation from SRT geometry.

    All parameters computed from {φ, π, e, E*, q}.
    Zero free parameters - everything derived from winding topology.

    Attributes:
        fermions: FermionMasses instance for leptons and quarks
        bosons: GaugeSector instance for W, Z, couplings
        higgs: HiggsSector instance for Higgs mass
        ckm: CKMMatrix instance for quark mixing
        pmns: PMNSMatrix instance for neutrino mixing
        neutrinos: NeutrinoMasses instance
        hadrons: HadronMasses instance for nucleons and mesons
        running: GoldenRG instance for RG evolution

    Example:
        >>> sm = StandardModel()
        >>> sm.all_parameters()
        {'m_e': 0.511, 'm_mu': 105.7, ..., 'm_H': 125.25}
        >>> sm.validate()
        {'m_e': {'predicted': 0.511, 'pdg': 0.511, 'sigma': 0.0}, ...}
    """

    def __init__(self):
        """Initialize all physics sectors."""
        self.fermions = FermionMasses()
        self.bosons = GaugeSector()
        self.higgs = HiggsSector()
        self.ckm = CKMMatrix()
        self.pmns = PMNSMatrix()
        self.neutrinos = NeutrinoMasses()
        self.hadrons = HadronMasses()
        self.running = GoldenRG()

    def all_parameters(self) -> dict:
        """
        Return all 25+ SM parameters derived from SRT.

        Returns:
            Dictionary mapping parameter names to predicted values
        """
        return {
            # Fermion masses (MeV for leptons/light quarks, GeV for heavy)
            "m_e": self.fermions.electron_mass(),
            "m_mu": self.fermions.muon_mass(),
            "m_tau": self.fermions.tau_mass(),
            "m_u": self.fermions.up_mass(),
            "m_d": self.fermions.down_mass(),
            "m_s": self.fermions.strange_mass(),
            "m_c": self.fermions.charm_mass(),
            "m_b": self.fermions.bottom_mass(),
            "m_t": self.fermions.top_mass(),
            # Gauge bosons (GeV)
            "m_W": self.bosons.w_mass(),
            "m_Z": self.bosons.z_mass(),
            "Gamma_Z": self.bosons.z_width(),
            # Higgs (GeV)
            "m_H": self.higgs.mass(),
            # Couplings
            "alpha_em": self.bosons.fine_structure_constant(),
            "alpha_s": self.bosons.strong_coupling(),
            "sin2_theta_W": self.bosons.weinberg_angle(),
            # CKM elements
            "V_us": self.ckm.V_us(),
            "V_cb": self.ckm.V_cb(),
            "V_ub": self.ckm.V_ub(),
            "J_CP": self.ckm.jarlskog_invariant(),
            # PMNS angles (degrees)
            "theta_12": self.pmns.theta_12(),
            "theta_23": self.pmns.theta_23(),
            "theta_13": self.pmns.theta_13(),
            "delta_CP": self.pmns.delta_CP(),
            # Neutrinos
            "m_nu3": self.neutrinos.m_nu3(),
            "Delta_m2_ratio": self.neutrinos.mass_squared_ratio(),
            # Hadrons (MeV)
            "m_p": self.hadrons.proton_mass(),
            "m_n": self.hadrons.neutron_mass(),
            "m_pi": self.hadrons.pion_mass(),
            "m_K": self.hadrons.kaon_mass(),
        }

    def validate(self) -> dict:
        """
        Compare all predictions to PDG experimental values.

        Returns:
            Dictionary with prediction, PDG value, and sigma deviation
            for each parameter
        """
        return validate_all(self.all_parameters())

    def higgs_mass(self) -> float:
        """Higgs mass in GeV."""
        return self.higgs.mass()

    def proton_mass(self) -> float:
        """Proton mass in MeV."""
        return self.hadrons.proton_mass()

    def verify_hierarchy(self) -> bool:
        """
        Run comprehensive verification of the Universal Syntony Correction Hierarchy.

        Tests all observables against experimental data and reports agreement.

        Returns:
            True if all observables within 2% of experimental values
        """
        from syntonic.physics import sm_verification

        return sm_verification.verify_all_observables()


__all__ = [
    # Constants
    "V_EW",
    "M_Z",
    "M_W_PDG",
    "M_H_PDG",
    "ALPHA_EM_0",
    # Submodules
    "fermions",
    "bosons",
    "mixing",
    "neutrinos",
    "hadrons",
    "running",
    "validation",
    "sm_verification",
    "gravity",
    # Classes
    "StandardModel",
    "FermionMasses",
    "GaugeSector",
    "HiggsSector",
    "CKMMatrix",
    "PMNSMatrix",
    "NeutrinoMasses",
    "HadronMasses",
    "GoldenRG",
    "BimetricGravity",
    # Functions
    "validate_all",
    "PDG_VALUES",
    "verify_all_observables",
    # Topology functions
    "hooking_coefficient",
    "golden_resonance",
    "e8_root_alignment",
    "compute_tensor_norm",
]
