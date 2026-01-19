"""
SRT-Zero: The Geometric Bootloader
==================================

A computational implementation of Syntony Recursion Theory that derives
the Standard Model particle spectrum from zero physical constants,
using only geometric axioms.

Modules:
- hierarchy: Complete 25-level correction system
- constants: The four seeds {φ, π, e, 1} and derived E*, q
- geometry: Topological invariants (E₈, E₆, D₄ structures)
- catalog: Particle configurations (~40 particles)
- engine: Mass derivation engine
- validate: Test harness

Usage:
    from srt_zero import DerivationEngine

    engine = DerivationEngine()
    result = engine.derive("proton")
    print(f"Proton mass: {result.final_value} MeV")
"""

from __future__ import annotations

# Core modules
from syntonic.core.constants import (
    # Constants
    PHI,
    PHI_INV,
    PI,
    E,
    E_STAR,
    Q,
    H_E8,
    DIM_E8,
    ROOTS_E8,
    ROOTS_E8_POS,
    RANK_E8,
    H_E7,
    DIM_E7,
    DIM_E7_FUND,
    ROOTS_E7,
    ROOTS_E7_POS,
    RANK_E7,
    H_E6,
    DIM_E6,
    DIM_E6_FUND,
    ROOTS_E6_POS,
    RANK_E6,
    K_D4,
    DIM_D4,
    RANK_D4,
    DIM_F4,
    DIM_G2,
    DIM_T4,
    N_GEN,
    FERMAT_PRIMES,
    FERMAT_COMPOSITE_5,
    MERSENNE_EXPONENTS,
    M11_BARRIER,
    LUCAS_SEQUENCE,
    LUCAS_PRIMES_INDICES,
    FIBONACCI_PRIME_GATES,
    GEOMETRIC_DIVISORS,
    FIBONACCI,
)

from .hierarchy import (
    # Enums
    CorrectionLevel,
    CorrectionCategory,
    # Dataclasses
    CorrectionInfo,
    CorrectionRecord,
    DerivationResult,
    # Tables
    CORRECTION_HIERARCHY,
    # Functions
    apply_correction,
    apply_special,
    apply_corrections,
    apply_winding_instability,
    apply_recursion_penalty,
    apply_double_inverse,
    apply_fixed_point_penalty,
    compute_E_star_N,
    compute_proton_mass,
    compute_neutron_mass,
    compute_pion_mass,
    compute_kaon_mass,
)

from syntonic.core.constants import UniverseSeeds

from .geometry import GeometricInvariants

from .operators import (
    recursion_map,
    is_recursion_fixed_point,
    GAUGE_FORCES,
    is_stable_generation,
    get_generation,
    lucas_number,
    dark_matter_mass_prediction,
    apply_five_operators,
    OperatorResult,
)

from .dhsr import (
    DHSRState,
    compute_syntony,
    compute_gnosis,
    dhsr_cycle_step,
    map_to_particle_physics,
)

from .catalog import (
    FormulaType,
    ParticleType,
    ParticleConfig,
    CATALOG,
    get_particle,
    list_particles,
    get_all_configs,
)

from .engine import DerivationEngine, MassMiner

from .config import SRTConfig, get_config, set_config

# Import backend module (optional)
try:
    from . import backend as _backend_module
except ImportError:
    _backend_module = None


__version__ = "2.1.0"
__author__ = "SRT-Zero Kernel"

__all__ = [
    # Version
    "__version__",
    # Core Classes
    "UniverseSeeds",
    "GeometricInvariants",
    "DerivationEngine",
    "MassMiner",
    # Constants
    "PHI",
    "PHI_INV",
    "PI",
    "E",
    "E_STAR",
    "Q",
    # Hierarchy
    "CorrectionLevel",
    "CorrectionCategory",
    "CorrectionInfo",
    "CorrectionRecord",
    "DerivationResult",
    "GEOMETRIC_DIVISORS",
    "CORRECTION_HIERARCHY",
    # Catalog
    "FormulaType",
    "ParticleType",
    "ParticleConfig",
    "CATALOG",
    "get_particle",
    "list_particles",
    "get_all_configs",
    # Five Operators
    "recursion_map",
    "is_recursion_fixed_point",
    "GAUGE_FORCES",
    "is_stable_generation",
    "get_generation",
    "lucas_number",
    "dark_matter_mass_prediction",
    "apply_five_operators",
    "OperatorResult",
    # DHSR Framework
    "DHSRState",
    "compute_syntony",
    "compute_gnosis",
    "dhsr_cycle_step",
    "map_to_particle_physics",
    # Functions
    "apply_correction",
    "apply_special",
    "apply_corrections",
]

# Lazy export backend functions
if _backend_module is not None:
    __all__.extend(
        [
            "is_cuda_available",
            "batch_apply_correction",
            "batch_apply_special_correction",
            "batch_apply_suppression",
            "batch_compute_e_star_n",
            "SpecialCorrectionType",
            "SuppressionType",
        ]
    )
