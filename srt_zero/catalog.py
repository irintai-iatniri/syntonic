"""
SRT-Zero: Particle Catalog
==========================
Source: Universal_Syntony_Correction_Hierarchy.md + SRT_Equations.md

Complete catalog of ~80 particles with exact formulas.
Each particle configuration specifies:
- PDG experimental value
- Formula type
- Base integer N (for E* × N formulas)
- Corrections list
- Special corrections
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import List, Tuple, Union, Optional

# Import from hierarchy - handle both relative and absolute imports
try:
    from .hierarchy import PI
except (ImportError, ValueError):
    try:
        from hierarchy import PI
    except ImportError as e:
        raise ImportError(f"Could not import PI from hierarchy module: {e}")


# =============================================================================
# FORMULA TYPE ENUMERATION
# =============================================================================


class FormulaType(Enum):
    """Types of mass/observable formulas in SRT."""

    TREE_EXACT = auto()  # m = E* × N (no corrections)
    E_STAR_N = auto()  # m = E* × N × corrections
    PROTON_SPECIAL = auto()  # m_p = φ⁸(E*-q)(1+q/1000)
    NEUTRON_SPECIAL = auto()  # m_n = E* × φ⁸ × (1+q/720)
    PROTON_RATIO = auto()  # m = m_p × (1 + k·q)
    PROTON_PLUS = auto()  # m = m_p + E* × N × corrections
    MASS_RATIO = auto()  # m = m_base × ratio_formula
    TOP_SPECIAL = auto()  # Special top quark formula
    HIGGS_SPECIAL = auto()  # Higgs with loop corrections
    HOOKING_MECHANISM = auto()  # Electron/muon hooking
    WIDTH_RATIO = auto()  # Γ = m × q × corrections
    CKM_ELEMENT = auto()  # V_ij = φ̂^n × corrections
    PMNS_ANGLE = auto()  # θ_ij = base × corrections
    NEUTRINO_COSMOLOGICAL = auto()  # Neutrino from ρ_Λ
    NUCLEAR_BINDING = auto()  # Nuclear binding energies
    COSMOLOGY_SPECIAL = auto()  # Cosmological observables


class ParticleType(Enum):
    """Classification of particles."""

    NUCLEON = auto()
    BARYON = auto()
    MESON = auto()
    QUARK = auto()
    LEPTON = auto()
    GAUGE_BOSON = auto()
    HIGGS = auto()
    NEUTRINO = auto()
    MIXING = auto()
    NUCLEAR = auto()
    COSMOLOGY = auto()
    OTHER = auto()


# =============================================================================
# PARTICLE CONFIGURATION DATACLASS
# =============================================================================


@dataclass
class ParticleConfig:
    """
    Complete specification for deriving a particle mass/observable.

    Attributes:
        name: Human-readable particle name
        symbol: LaTeX-style symbol (e.g., "m_p", "θ₁₂")
        particle_type: Classification category
        formula_type: How to compute the value
        pdg_value: Experimental value from PDG
        pdg_unit: Unit ("MeV", "GeV", "degrees", etc.)
        pdg_uncertainty: Experimental uncertainty
        base_integer_N: Integer multiplier for E* × N formulas
        corrections: List of (divisor, sign) for (1 ± q/divisor)
        special_corrections: List of special correction type strings
        suppression: List of suppression factors ("winding", "recursion")
        ratio_factor: For PROTON_RATIO type
        base_particle: For MASS_RATIO type
        ratio_formula: String formula for MASS_RATIO
        notes: Geometric/physical explanation
    """

    name: str
    symbol: str
    particle_type: ParticleType
    formula_type: FormulaType
    pdg_value: float
    pdg_unit: str = "MeV"
    pdg_uncertainty: float = 0.0
    base_integer_N: Optional[float] = None
    corrections: List[Tuple[float, int]] = field(default_factory=list)
    special_corrections: List[str] = field(default_factory=list)
    suppression: List[str] = field(default_factory=list)
    ratio_factor: Optional[float] = None
    base_particle: Optional[str] = None
    ratio_formula: Optional[str] = None
    notes: str = ""


# =============================================================================
# PARTICLE CATALOG
# =============================================================================

CATALOG: dict[str, ParticleConfig] = {}

# Backwards-compatible alias expected by other modules
PARTICLE_CATALOG = CATALOG


def _register(config: ParticleConfig) -> ParticleConfig:
    """Register a particle configuration in the catalog with multiple keys."""
    # Primary key
    key = config.name.lower().replace(" ", "_")
    CATALOG[key] = config
    # Also register without special chars for easier lookup
    clean_key = key.replace("-", "").replace("+", "").replace("_", "")
    if clean_key != key:
        CATALOG[clean_key] = config
    return config


def get_particle(name: str) -> ParticleConfig:
    """
    Retrieve a particle configuration by name.

    Args:
        name: Particle name (case-insensitive, spaces/underscores handled)

    Returns:
        ParticleConfig object

    Raises:
        KeyError: If particle not found
    """
    key = name.lower().replace(" ", "_")

    # Try direct lookup
    if key in CATALOG:
        return CATALOG[key]

    # Try clean lookup
    clean_key = key.replace("-", "").replace("+", "").replace("_", "")
    if clean_key in CATALOG:
        return CATALOG[clean_key]

    raise KeyError(f"Particle '{name}' not found in catalog")


def list_particles(ptype: Optional[ParticleType] = None) -> List[str]:
    """
    List available particles, optionally filtered by type.

    Args:
        ptype: Optional ParticleType to filter by

    Returns:
        List of particle names
    """
    particles = []
    seen = set()

    for config in CATALOG.values():
        if config.name in seen:
            continue

        if ptype is None or config.particle_type == ptype:
            particles.append(config.name)
            seen.add(config.name)

    return sorted(particles)


def get_all_configs() -> List[ParticleConfig]:
    """Return all registered ParticleConfig objects.

    Added for API compatibility with package exports.
    """
    return list(CATALOG.values())


# -----------------------------------------------------------------------------
# NUCLEONS
# -----------------------------------------------------------------------------

PROTON = _register(
    ParticleConfig(
        name="Proton",
        symbol="m_p",
        particle_type=ParticleType.NUCLEON,
        formula_type=FormulaType.PROTON_SPECIAL,
        pdg_value=938.27208816,
        pdg_uncertainty=0.00000029,
        corrections=[(1000, +1)],
        notes="m_p = φ⁸(E*-q)(1+q/1000). Fixed-point: 1000 = h(E₈)³/27",
    )
)

NEUTRON = _register(
    ParticleConfig(
        name="Neutron",
        symbol="m_n",
        particle_type=ParticleType.NUCLEON,
        formula_type=FormulaType.NEUTRON_SPECIAL,
        pdg_value=939.56542052,
        pdg_uncertainty=0.00000054,
        corrections=[(720, +1)],
        notes="m_n = E*×φ⁸×(1+q/720). Coxeter-Kissing: 720 = 30×24 = 6!",
    )
)


# -----------------------------------------------------------------------------
# BARYONS
# -----------------------------------------------------------------------------

LAMBDA = _register(
    ParticleConfig(
        name="Lambda",
        symbol="m_Λ",
        particle_type=ParticleType.BARYON,
        formula_type=FormulaType.PROTON_RATIO,
        pdg_value=1115.683,
        pdg_uncertainty=0.006,
        ratio_factor=6.9,
        notes="m_Λ = m_p × (1 + 6.9q)",
    )
)

DELTA = _register(
    ParticleConfig(
        name="Delta",
        symbol="m_Δ",
        particle_type=ParticleType.BARYON,
        formula_type=FormulaType.PROTON_PLUS,
        pdg_value=1232.0,
        pdg_uncertainty=2.0,
        base_integer_N=15,
        corrections=[(1, -1)],
        notes="m_Δ = m_p + E*×15×(1-q). 15 = dim(SU(4)/SU(2))",
    )
)

XI_MINUS = _register(
    ParticleConfig(
        name="Xi-",
        symbol="m_Ξ⁻",
        particle_type=ParticleType.BARYON,
        formula_type=FormulaType.E_STAR_N,
        pdg_value=1321.71,
        pdg_uncertainty=0.07,
        base_integer_N=66,
        corrections=[(36, +1)],
        notes="m_Ξ = E*×66×(1+q/36). 66 = 2×h(E₈)+6",
    )
)

OMEGA_MINUS = _register(
    ParticleConfig(
        name="Omega-",
        symbol="m_Ω⁻",
        particle_type=ParticleType.BARYON,
        formula_type=FormulaType.E_STAR_N,
        pdg_value=1672.45,
        pdg_uncertainty=0.29,
        base_integer_N=84,
        corrections=[(8, -1), (27, -1)],
        notes="m_Ω = E*×84×(1-q/8)(1-q/27) = 1672.47 MeV. M₃×12 base, E₈ rank & E₆ fund corrections.",
    )
)

SIGMA_PLUS = _register(
    ParticleConfig(
        name="Sigma_plus",
        symbol="Σ⁺",
        particle_type=ParticleType.BARYON,
        formula_type=FormulaType.PROTON_PLUS,
        pdg_value=1189.37,
        pdg_unit="MeV",
        pdg_uncertainty=0.07,
        base_integer_N=12.5,  # 25/2
        corrections=[(6, +1)],  # (1 + q/6) sub-generation structure
        notes="m_Σ⁺ = m_p + E*×25/2×(1+q/6) = 1189.4 MeV",
    )
)


# -----------------------------------------------------------------------------
# LIGHT MESONS
# -----------------------------------------------------------------------------

PION = _register(
    ParticleConfig(
        name="Pion",
        symbol="m_π",
        particle_type=ParticleType.MESON,
        formula_type=FormulaType.E_STAR_N,
        pdg_value=139.57039,
        pdg_uncertainty=0.00018,
        base_integer_N=7,
        corrections=[(8, -1)],
        special_corrections=["q_squared_phi_plus"],
        notes="m_π = E*×7×(1-q/8)(1+q²/φ). N=7 first non-Fibonacci prime",
    )
)

KAON = _register(
    ParticleConfig(
        name="Kaon",
        symbol="m_K",
        particle_type=ParticleType.MESON,
        formula_type=FormulaType.E_STAR_N,
        pdg_value=497.611,
        pdg_uncertainty=0.013,
        base_integer_N=25,
        corrections=[(6, -1), (120, -1)],
        notes="m_K = E*×25×(1-q/6)(1-q/120). 25 = F₅² = 5²",
    )
)

ETA = _register(
    ParticleConfig(
        name="Eta",
        symbol="m_η",
        particle_type=ParticleType.MESON,
        formula_type=FormulaType.E_STAR_N,
        pdg_value=547.862,
        pdg_uncertainty=0.017,
        base_integer_N=27,
        corrections=[(2, +1), (36, +1)],
        notes="m_η = E*×27×(1+q/2)(1+q/36). 27 = dim(E₆ fund)",
    )
)

RHO = _register(
    ParticleConfig(
        name="Rho",
        symbol="m_ρ",
        particle_type=ParticleType.MESON,
        formula_type=FormulaType.E_STAR_N,
        pdg_value=775.26,
        pdg_uncertainty=0.23,
        base_integer_N=39,
        corrections=[(4, -1), (27, +1), (1000, +1)],
        notes="m_ρ = E*×39×(1-q/4)(1+q/27)(1+q/1000). 39 = 3×13",
    )
)


# -----------------------------------------------------------------------------
# HEAVY MESONS
# -----------------------------------------------------------------------------

D_MESON = _register(
    ParticleConfig(
        name="D",
        symbol="m_D",
        particle_type=ParticleType.MESON,
        formula_type=FormulaType.E_STAR_N,
        pdg_value=1864.84,
        pdg_uncertainty=0.05,
        base_integer_N=93,
        corrections=[(27, +1), (78, +1), (248, +1)],
        notes="m_D = E*×93×(1+q/27)(1+q/78)(1+q/248). 93 = 3×31",
    )
)

B_MESON = _register(
    ParticleConfig(
        name="B",
        symbol="m_B",
        particle_type=ParticleType.MESON,
        formula_type=FormulaType.TREE_EXACT,
        pdg_value=5279.66,
        pdg_uncertainty=0.12,
        base_integer_N=264,
        notes="m_B = E*×264 TREE-EXACT! 264 = K(D₄)×11 = 24×11",
    )
)

BC_MESON = _register(
    ParticleConfig(
        name="Bc",
        symbol="m_Bc",
        particle_type=ParticleType.MESON,
        formula_type=FormulaType.E_STAR_N,
        pdg_value=6274.9,
        pdg_uncertainty=0.8,
        base_integer_N=314,
        corrections=[(36, -1)],
        notes="m_Bc = E*×314×(1-q/36). 314 ≈ 100π",
    )
)


# -----------------------------------------------------------------------------
# QUARKONIUM
# -----------------------------------------------------------------------------

JPSI = _register(
    ParticleConfig(
        name="Jpsi",
        symbol="m_J/ψ",
        particle_type=ParticleType.MESON,
        formula_type=FormulaType.E_STAR_N,
        pdg_value=3096.9,
        pdg_uncertainty=0.006,
        base_integer_N=155,
        corrections=[(27, -1)],
        notes="m_J/ψ = E*×155×(1-q/27). 155 = 5×31",
    )
)

PSI_2S = _register(
    ParticleConfig(
        name="Psi_2S",
        symbol="ψ(2S)",
        particle_type=ParticleType.MESON,
        formula_type=FormulaType.COSMOLOGY_SPECIAL,
        pdg_value=3686.1,
        pdg_unit="MeV",
        pdg_uncertainty=0.1,
        notes="m_ψ(2S) = m_J/ψ + E*×59/2 = 3686.9 MeV",
    )
)

UPSILON_1S = _register(
    ParticleConfig(
        name="Upsilon_1S",
        symbol="m_Υ(1S)",
        particle_type=ParticleType.MESON,
        formula_type=FormulaType.TREE_EXACT,
        pdg_value=9460.30,
        pdg_uncertainty=0.26,
        base_integer_N=473,
        notes="m_Υ = E*×473 tree-level. 473 = 11×43",
    )
)

UPSILON_2S = _register(
    ParticleConfig(
        name="Upsilon_2S",
        symbol="Υ(2S)",
        particle_type=ParticleType.MESON,
        formula_type=FormulaType.TREE_EXACT,
        pdg_value=10023.3,
        pdg_unit="MeV",
        pdg_uncertainty=0.3,
        base_integer_N=501,
        notes="m_Υ(2S) = E* × 501 = 10019.6 MeV",
    )
)

UPSILON_3S = _register(
    ParticleConfig(
        name="Upsilon_3S",
        symbol="Υ(3S)",
        particle_type=ParticleType.MESON,
        formula_type=FormulaType.TREE_EXACT,
        pdg_value=10355.2,
        pdg_unit="MeV",
        pdg_uncertainty=0.3,
        base_integer_N=518,
        notes="m_Υ(3S) = E* × 518 = 10359.5 MeV",
    )
)


# -----------------------------------------------------------------------------
# QUARKS
# -----------------------------------------------------------------------------

UP = _register(
    ParticleConfig(
        name="Up",
        symbol="m_u",
        particle_type=ParticleType.QUARK,
        formula_type=FormulaType.E_STAR_N,
        pdg_value=2.16,
        pdg_uncertainty=0.49,
        base_integer_N=1 / 9,
        corrections=[(1, -1)],
        notes="m_u = (E*/9)×(1-q). Same E*/9 as deuteron binding",
    )
)

DOWN = _register(
    ParticleConfig(
        name="Down",
        symbol="m_d",
        particle_type=ParticleType.QUARK,
        formula_type=FormulaType.MASS_RATIO,
        pdg_value=4.67,
        pdg_uncertainty=0.48,
        base_particle="Up",
        ratio_formula="2 + 6*q",
        special_corrections=["2_plus_6q"],
        notes="m_d = m_u × (2 + 6q)",
    )
)

STRANGE = _register(
    ParticleConfig(
        name="Strange",
        symbol="m_s",
        particle_type=ParticleType.QUARK,
        formula_type=FormulaType.E_STAR_N,
        pdg_value=93.4,
        pdg_uncertainty=8.6,
        base_integer_N=5,
        corrections=[(1, -1), (120, +1), (8, +1)],
        special_corrections=["q_phi_minus"],
        notes="m_s = E*×5×(1-qφ)(1-q)(1+q/120)(1+q/8) = 93.3 Mev",
    )
)

CHARM = _register(
    ParticleConfig(
        name="Charm",
        symbol="m_c",
        particle_type=ParticleType.QUARK,
        formula_type=FormulaType.E_STAR_N,
        pdg_value=1270,
        pdg_uncertainty=30,
        base_integer_N=63.5,
        corrections=[(120, +1)],
        notes="m_c = E*×63.5×(1+q/120). 63.5 = 127/2, Mersenne prime",
    )
)

BOTTOM = _register(
    ParticleConfig(
        name="Bottom",
        symbol="m_b",
        particle_type=ParticleType.QUARK,
        formula_type=FormulaType.E_STAR_N,
        pdg_value=4180,
        pdg_uncertainty=30,
        base_integer_N=209,
        corrections=[(248, +1)],
        notes="m_b = E*×209×(1+q/248). 209 = 11×19",
    )
)

TOP = _register(
    ParticleConfig(
        name="Top",
        symbol="m_t",
        particle_type=ParticleType.QUARK,
        formula_type=FormulaType.TOP_SPECIAL,
        pdg_value=172760,
        pdg_unit="MeV",
        pdg_uncertainty=300,
        corrections=[(120, +1)],
        notes="m_t = 172.50 GeV × (1+qφ/4π)(1-q/4π)(1+q/120)",
    )
)


# -----------------------------------------------------------------------------
# CHARGED LEPTONS
# -----------------------------------------------------------------------------

TAU = _register(
    ParticleConfig(
        name="Tau",
        symbol="m_τ",
        particle_type=ParticleType.LEPTON,
        formula_type=FormulaType.E_STAR_N,
        pdg_value=1776.86,
        pdg_uncertainty=0.12,
        base_integer_N=89,  # F₁₁
        corrections=[(5 * PI, -1), (720, -1)],
        notes="m_τ = E*×F₁₁×(1-q/5π)(1-q/720). F₁₁ = 89",
    )
)

MUON = _register(
    ParticleConfig(
        name="Muon",
        symbol="m_μ",
        particle_type=ParticleType.LEPTON,
        formula_type=FormulaType.E_STAR_N,
        pdg_value=105.6583745,
        pdg_uncertainty=0.0000024,
        base_integer_N=5.28,  # Approximate mined value
        corrections=[(27, -1)],
        notes="Uses hooking mechanism with C_Higgs = e^(1/φ)",
    )
)

ELECTRON = _register(
    ParticleConfig(
        name="Electron",
        symbol="m_e",
        particle_type=ParticleType.LEPTON,
        formula_type=FormulaType.HOOKING_MECHANISM,
        pdg_value=0.5109989461,
        pdg_uncertainty=0.0000000031,
        notes="C_Higgs = 1 (n₁₀ = 0, no Higgs hooking)",
    )
)


# -----------------------------------------------------------------------------
# NEUTRINOS (NEW EXACT DERIVATIONS)
# -----------------------------------------------------------------------------

NEUTRINO_3 = _register(
    ParticleConfig(
        name="Neutrino_3",
        symbol="m_ν3",
        particle_type=ParticleType.NEUTRINO,
        formula_type=FormulaType.NEUTRINO_COSMOLOGICAL,
        pdg_value=50.1,  # Est from Δm^2
        pdg_unit="meV",
        pdg_uncertainty=1.0,
        notes="m_ν3 = ρ_Λ^(1/4) × E* × (1+4q) = 49.93 meV",
    )
)

NEUTRINO_2 = _register(
    ParticleConfig(
        name="Neutrino_2",
        symbol="m_ν2",
        particle_type=ParticleType.NEUTRINO,
        formula_type=FormulaType.NEUTRINO_COSMOLOGICAL,
        pdg_value=8.61,  # Est from Δm^2
        pdg_unit="meV",
        pdg_uncertainty=0.1,
        notes="m_ν2 = m_ν3 / √[34(1-q/36)] = 8.57 meV",
    )
)

NEUTRINO_1 = _register(
    ParticleConfig(
        name="Neutrino_1",
        symbol="m_ν1",
        particle_type=ParticleType.NEUTRINO,
        formula_type=FormulaType.NEUTRINO_COSMOLOGICAL,
        pdg_value=2.02,  # Prediction
        pdg_unit="meV",
        pdg_uncertainty=0.0,
        notes="m_ν1 = m_ν2 / φ³ = 2.02 meV",
    )
)


# -----------------------------------------------------------------------------
# GAUGE BOSONS
# -----------------------------------------------------------------------------

W_BOSON = _register(
    ParticleConfig(
        name="W",
        symbol="m_W",
        particle_type=ParticleType.GAUGE_BOSON,
        formula_type=FormulaType.E_STAR_N,
        pdg_value=80377.9,
        pdg_unit="MeV",
        pdg_uncertainty=12,
        base_integer_N=4019,
        corrections=[(4 * PI, +1), (248, +1)],
        notes="m_W = E*×4019×(1+q/4π)(1+q/248)",
    )
)

Z_BOSON = _register(
    ParticleConfig(
        name="Z",
        symbol="m_Z",
        particle_type=ParticleType.GAUGE_BOSON,
        formula_type=FormulaType.E_STAR_N,
        pdg_value=91187.6,
        pdg_unit="MeV",
        pdg_uncertainty=2.1,
        base_integer_N=4559,
        corrections=[(4 * PI, +1), (248, +1)],
        notes="m_Z = E*×4559×(1+q/4π)(1+q/248)",
    )
)

HIGGS = _register(
    ParticleConfig(
        name="Higgs",
        symbol="m_H",
        particle_type=ParticleType.HIGGS,
        formula_type=FormulaType.HIGGS_SPECIAL,
        pdg_value=125250,
        pdg_unit="MeV",
        pdg_uncertainty=170,
        notes="Tree = 93 GeV, loop-corrected to 125.25 GeV",
    )
)


# -----------------------------------------------------------------------------
# ELECTROWEAK OBSERVABLES
# -----------------------------------------------------------------------------

Z_WIDTH = _register(
    ParticleConfig(
        name="Gamma_Z",
        symbol="Γ_Z",
        particle_type=ParticleType.GAUGE_BOSON,
        formula_type=FormulaType.WIDTH_RATIO,
        pdg_value=2.4952,
        pdg_unit="GeV",
        pdg_uncertainty=0.0023,
        base_particle="Z",
        corrections=[(24, -1)],
        notes="Γ_Z = m_Z × q × (1 - q/24) = 2.4952 GeV",
    )
)

W_WIDTH = _register(
    ParticleConfig(
        name="Gamma_W",
        symbol="Γ_W",
        particle_type=ParticleType.GAUGE_BOSON,
        formula_type=FormulaType.WIDTH_RATIO,
        pdg_value=2.085,
        pdg_unit="GeV",
        pdg_uncertainty=0.042,
        base_particle="W",
        special_corrections=["2q_minus"],
        notes="Γ_W = m_W × q × (1 - 2q) = 2.081 GeV",
    )
)

R_B = _register(
    ParticleConfig(
        name="R_b",
        symbol="R_b",
        particle_type=ParticleType.MIXING,
        formula_type=FormulaType.COSMOLOGY_SPECIAL,
        pdg_value=0.21629,
        pdg_unit="ratio",
        pdg_uncertainty=0.00066,
        notes="R_b = (1/5) × (1 + 3q) = 0.2164",
    )
)


# -----------------------------------------------------------------------------
# MIXING ANGLES (CKM)
# -----------------------------------------------------------------------------

V_US = _register(
    ParticleConfig(
        name="V_us",
        symbol="|V_us|",
        particle_type=ParticleType.MIXING,
        formula_type=FormulaType.CKM_ELEMENT,
        pdg_value=0.2243,
        pdg_unit="",
        pdg_uncertainty=0.0005,
        corrections=[(4, -1), (120, +1)],
        special_corrections=["q_phi_minus"],
        notes="|V_us| = φ̂³(1-qφ)(1-q/4)(1+q/120) = 0.2253",
    )
)

V_CB = _register(
    ParticleConfig(
        name="V_cb",
        symbol="|V_cb|",
        particle_type=ParticleType.MIXING,
        formula_type=FormulaType.CKM_ELEMENT,
        pdg_value=0.0415,
        pdg_unit="",
        pdg_uncertainty=0.0008,
        corrections=[(3, +1)],
        notes="|V_cb| = Q × 3/2 × (1+q/3) = 0.0415. q/3 = q/N_gen = single generation correction",
    )
)

V_UB = _register(
    ParticleConfig(
        name="V_ub",
        symbol="|V_ub|",
        particle_type=ParticleType.MIXING,
        formula_type=FormulaType.CKM_ELEMENT,
        pdg_value=0.00361,
        pdg_unit="dimensionless",
        pdg_uncertainty=0.00011,
        corrections=[(1, +1)],  # (1 + q)
        special_corrections=["q_phi_minus"],
        notes="|V_ub| = Q² × K(D₄)/F₁ = Q² × 24/5 = 0.00360. Collapse threshold / weak isospin",
    )
)


# -----------------------------------------------------------------------------
# MIXING ANGLES (PMNS)
# -----------------------------------------------------------------------------

THETA_12 = _register(
    ParticleConfig(
        name="theta_12",
        symbol="θ₁₂",
        particle_type=ParticleType.MIXING,
        formula_type=FormulaType.PMNS_ANGLE,
        pdg_value=33.44,
        pdg_unit="degrees",
        pdg_uncertainty=0.77,
        corrections=[(2, +1), (27, +1)],
        notes="θ₁₂ = φ̂²(1+q/2)(1+q/27) → 33.44°",
    )
)

THETA_23 = _register(
    ParticleConfig(
        name="theta_23",
        symbol="θ₂₃",
        particle_type=ParticleType.MIXING,
        formula_type=FormulaType.PMNS_ANGLE,
        pdg_value=49.2,
        pdg_unit="degrees",
        pdg_uncertainty=0.9,
        corrections=[(8, +1), (36, +1), (120, -1)],
        notes="θ₂₃ = 49.0°×(1+q/8)(1+q/36)(1-q/120) = 49.19°. rank(E₈)=8, |Φ⁺(E₆)|=36, |Φ⁺(E₈)|=120",
    )
)

THETA_13 = _register(
    ParticleConfig(
        name="theta_13",
        symbol="θ₁₃",
        particle_type=ParticleType.MIXING,
        formula_type=FormulaType.PMNS_ANGLE,
        pdg_value=8.57,
        pdg_unit="degrees",
        pdg_uncertainty=0.12,
        corrections=[(8, +1), (12, +1)],
        suppression=["recursion"],
        notes="θ₁₃ = φ̂³/(1+qφ)×(1+q/8)(1+q/12) → 8.57°",
    )
)


# -----------------------------------------------------------------------------
# NUCLEAR BINDING
# -----------------------------------------------------------------------------

DEUTERON = _register(
    ParticleConfig(
        name="Deuteron",
        symbol="B_d",
        particle_type=ParticleType.NUCLEAR,
        formula_type=FormulaType.E_STAR_N,
        pdg_value=2.22457,
        pdg_unit="MeV",
        pdg_uncertainty=0.00002,
        base_integer_N=1 / 9,
        corrections=[(27, +1)],
        notes="B_d = E*/9 × (1+q/27). Uses N_gen² = 9",
    )
)

ALPHA = _register(
    ParticleConfig(
        name="Alpha",
        symbol="B_α",
        particle_type=ParticleType.NUCLEAR,
        formula_type=FormulaType.E_STAR_N,
        pdg_value=28.2961,
        pdg_unit="MeV",
        pdg_uncertainty=0.0001,
        base_integer_N=1.414,  # √2
        corrections=[(78, +1), (248, +1)],
        notes="B_α = E*×√2×(1+q/78)(1+q/248)",
    )
)


# -----------------------------------------------------------------------------
# GLUEBALLS
# -----------------------------------------------------------------------------

GLUEBALL_0PP = _register(
    ParticleConfig(
        name="Glueball_0pp",
        symbol="G(0⁺⁺)",
        particle_type=ParticleType.MESON,
        formula_type=FormulaType.COSMOLOGY_SPECIAL,
        pdg_value=1517.0,
        pdg_unit="MeV",
        pdg_uncertainty=50.0,
        notes="m(0⁺⁺) = Λ_QCD×8×(1-4q), where Λ_QCD = 217×(1-q/φ)×(1+q/6π)",
    )
)

GLUEBALL_2PP = _register(
    ParticleConfig(
        name="Glueball_2pp",
        symbol="G(2⁺⁺)",
        particle_type=ParticleType.MESON,
        formula_type=FormulaType.COSMOLOGY_SPECIAL,
        pdg_value=2220.0,
        pdg_unit="MeV",
        pdg_uncertainty=100.0,
        notes="m(2⁺⁺) = E* × 111 = 2219.9 MeV",
    )
)

GLUEBALL_0MP = _register(
    ParticleConfig(
        name="Glueball_0mp",
        symbol="G(0⁻⁺)",
        particle_type=ParticleType.MESON,
        formula_type=FormulaType.COSMOLOGY_SPECIAL,
        pdg_value=2500.0,
        pdg_unit="MeV",
        pdg_uncertainty=100.0,
        notes="m(0⁻⁺) = m(0⁺⁺) × φ = 2455 (lattice: 2500±100)",
    )
)


# -----------------------------------------------------------------------------
# COSMOLOGICAL OBSERVABLES
# -----------------------------------------------------------------------------

DARK_MATTER_BARYON_RATIO = _register(
    ParticleConfig(
        name="DM_Baryon_Ratio",
        symbol="Ω_DM/Ω_b",
        particle_type=ParticleType.COSMOLOGY,
        formula_type=FormulaType.COSMOLOGY_SPECIAL,
        pdg_value=5.36,
        pdg_unit="ratio",
        pdg_uncertainty=0.10,
        notes="Ω_DM/Ω_b = φ³ + 1 + 5q = 5.373",
    )
)

Z_MATTER_RADIATION = _register(
    ParticleConfig(
        name="z_eq",
        symbol="z_eq",
        particle_type=ParticleType.COSMOLOGY,
        formula_type=FormulaType.E_STAR_N,
        pdg_value=3400.0,
        pdg_unit="redshift",
        pdg_uncertainty=100.0,
        base_integer_N=170.0,
        corrections=[],
        notes="z_eq = E* × 170 = 3400",
    )
)

Z_RECOMBINATION = _register(
    ParticleConfig(
        name="z_rec",
        symbol="z_rec",
        particle_type=ParticleType.COSMOLOGY,
        formula_type=FormulaType.E_STAR_N,
        pdg_value=1100.0,
        pdg_unit="redshift",
        pdg_uncertainty=10.0,
        base_integer_N=55.0,
        corrections=[],
        notes="z_rec = E* × F₁₀ = E* × 55 = 1100",
    )
)

HUBBLE_CONSTANT = _register(
    ParticleConfig(
        name="H0",
        symbol="H₀",
        particle_type=ParticleType.COSMOLOGY,
        formula_type=FormulaType.COSMOLOGY_SPECIAL,
        pdg_value=67.4,
        pdg_unit="km/s/Mpc",
        pdg_uncertainty=0.5,
        notes="H₀ = q × M_Pl × c = 67.4 km/s/Mpc",
    )
)

DARK_ENERGY_DENSITY = _register(
    ParticleConfig(
        name="rho_lambda",
        symbol="ρ_Λ",
        particle_type=ParticleType.COSMOLOGY,
        formula_type=FormulaType.COSMOLOGY_SPECIAL,
        pdg_value=2.25,
        pdg_unit="meV",
        pdg_uncertainty=0.01,
        notes="ρ_Λ = (3q² M_Pl⁴ / 8π)^(1/4) × corrections",
    )
)

BARYON_ASYMMETRY = _register(
    ParticleConfig(
        name="eta_B",
        symbol="η_B",
        particle_type=ParticleType.COSMOLOGY,
        formula_type=FormulaType.COSMOLOGY_SPECIAL,
        pdg_value=6.10e-10,
        pdg_unit="dimensionless",
        pdg_uncertainty=0.4e-10,
        notes="η_B = φ·q⁶(1-4q)(1+q/4)",
    )
)

SPECTRAL_INDEX = _register(
    ParticleConfig(
        name="n_s",
        symbol="n_s",
        particle_type=ParticleType.COSMOLOGY,
        formula_type=FormulaType.COSMOLOGY_SPECIAL,
        pdg_value=0.9649,
        pdg_unit="dimensionless",
        pdg_uncertainty=0.0042,
        notes="n_s = 1 - 2/N = 0.9649 (N=60)",
    )
)

TENSOR_TO_SCALAR = _register(
    ParticleConfig(
        name="r_tensor",
        symbol="r",
        particle_type=ParticleType.COSMOLOGY,
        formula_type=FormulaType.COSMOLOGY_SPECIAL,
        pdg_value=0.00328,
        pdg_unit="dimensionless",
        pdg_uncertainty=0.001,
        notes="r = 12/N² × (1-q/φ)",
    )
)

DARK_ENERGY_EOS = _register(
    ParticleConfig(
        name="w_eos",
        symbol="w",
        particle_type=ParticleType.COSMOLOGY,
        formula_type=FormulaType.COSMOLOGY_SPECIAL,
        pdg_value=-1.03,
        pdg_unit="dimensionless",
        pdg_uncertainty=0.03,
        notes="w = -1 - 2.5e-4 × ρ_m/ρ_Λ",
    )
)

STERILE_MIXING = _register(
    ParticleConfig(
        name="sterile_mixing",
        symbol="sin²(2θ)",
        particle_type=ParticleType.COSMOLOGY,
        formula_type=FormulaType.COSMOLOGY_SPECIAL,
        pdg_value=1.14e-11,
        pdg_unit="dimensionless",
        pdg_uncertainty=1e-12,
        notes="sin²(2θ) = q⁷(1-q/φ)",
    )
)

HELIUM_4_ABUNDANCE = _register(
    ParticleConfig(
        name="Y_p",
        symbol="Y_p",
        particle_type=ParticleType.COSMOLOGY,
        formula_type=FormulaType.COSMOLOGY_SPECIAL,
        pdg_value=0.245,
        pdg_unit="dimensionless",
        pdg_uncertainty=0.003,
        notes="BBN Prediction",
    )
)

DEUTERIUM_HYDROGEN = _register(
    ParticleConfig(
        name="D_H",
        symbol="D/H",
        particle_type=ParticleType.COSMOLOGY,
        formula_type=FormulaType.COSMOLOGY_SPECIAL,
        pdg_value=2.53e-5,
        pdg_unit="dimensionless",
        pdg_uncertainty=0.04e-5,
        notes="BBN Prediction",
    )
)

LITHIUM_7_HYDROGEN = _register(
    ParticleConfig(
        name="Li7_H",
        symbol="⁷Li/H",
        particle_type=ParticleType.COSMOLOGY,
        formula_type=FormulaType.COSMOLOGY_SPECIAL,
        pdg_value=1.60e-10,
        pdg_unit="dimensionless",
        pdg_uncertainty=0.3e-10,
        notes="BBN Prediction",
    )
)

EFFECTIVE_NEUTRINO_NUMBER = _register(
    ParticleConfig(
        name="N_eff",
        symbol="N_eff",
        particle_type=ParticleType.COSMOLOGY,
        formula_type=FormulaType.COSMOLOGY_SPECIAL,
        pdg_value=2.99,
        pdg_unit="dimensionless",
        pdg_uncertainty=0.17,
        notes="N_eff = 3(1-q/5) approx",
    )
)

CMB_PEAK_1 = _register(
    ParticleConfig(
        name="cmb_peak_1",
        symbol="ℓ₁",
        particle_type=ParticleType.COSMOLOGY,
        formula_type=FormulaType.COSMOLOGY_SPECIAL,
        pdg_value=220.0,
        pdg_unit="multipole",
        pdg_uncertainty=0.5,
        notes="ℓ₁ = π/θ_s",
    )
)

CMB_PEAK_2 = _register(
    ParticleConfig(
        name="cmb_peak_2",
        symbol="ℓ₂",
        particle_type=ParticleType.COSMOLOGY,
        formula_type=FormulaType.COSMOLOGY_SPECIAL,
        pdg_value=537.5,
        pdg_unit="multipole",
        pdg_uncertainty=0.7,
        notes="ℓ₂ = ℓ₁×2×(1-q²/12φ)",
    )
)

CMB_PEAK_3 = _register(
    ParticleConfig(
        name="cmb_peak_3",
        symbol="ℓ₃",
        particle_type=ParticleType.COSMOLOGY,
        formula_type=FormulaType.COSMOLOGY_SPECIAL,
        pdg_value=810.8,
        pdg_unit="multipole",
        pdg_uncertainty=0.7,
        notes="ℓ₃ = ℓ₁×3×(1-q²/6φ)",
    )
)

CMB_PEAK_4 = _register(
    ParticleConfig(
        name="cmb_peak_4",
        symbol="ℓ₄",
        particle_type=ParticleType.COSMOLOGY,
        formula_type=FormulaType.COSMOLOGY_SPECIAL,
        pdg_value=1120.9,
        pdg_unit="multipole",
        pdg_uncertainty=1.0,
        notes="ℓ₄ = ℓ₁×4×(1-q²/4φ)",
    )
)

CMB_PEAK_5 = _register(
    ParticleConfig(
        name="cmb_peak_5",
        symbol="ℓ₅",
        particle_type=ParticleType.COSMOLOGY,
        formula_type=FormulaType.COSMOLOGY_SPECIAL,
        pdg_value=1444.2,
        pdg_unit="multipole",
        pdg_uncertainty=2.0,
        notes="ℓ₅ = ℓ₁×5×(1-q²/3φ)(1+q/248)",
    )
)

PEAK_RATIO_21 = _register(
    ParticleConfig(
        name="peak_ratio_21",
        symbol="H₂/H₁",
        particle_type=ParticleType.COSMOLOGY,
        formula_type=FormulaType.COSMOLOGY_SPECIAL,
        pdg_value=0.458,
        pdg_unit="ratio",
        pdg_uncertainty=0.01,
        notes="H₂/H₁ = φ⁻¹(1+qφ)",
    )
)

PEAK_RATIO_31 = _register(
    ParticleConfig(
        name="peak_ratio_31",
        symbol="H₃/H₁",
        particle_type=ParticleType.COSMOLOGY,
        formula_type=FormulaType.COSMOLOGY_SPECIAL,
        pdg_value=0.37,
        pdg_unit="ratio",
        pdg_uncertainty=0.01,
        notes="H₃/H₁ = φ⁻²(1-q/4)",
    )
)

# -----------------------------------------------------------------------------
# ATOMIC PHYSICS OBSERVABLES
# -----------------------------------------------------------------------------

FINE_STRUCTURE_INV = _register(
    ParticleConfig(
        name="alpha_inv",
        symbol="α⁻¹",
        particle_type=ParticleType.OTHER,
        formula_type=FormulaType.COSMOLOGY_SPECIAL,
        pdg_value=137.036,
        pdg_unit="dimensionless",
        pdg_uncertainty=0.001,
        notes="α⁻¹(0) = 137.036",
    )
)

RYDBERG_CONSTANT = _register(
    ParticleConfig(
        name="Rydberg",
        symbol="Ry",
        particle_type=ParticleType.OTHER,
        formula_type=FormulaType.COSMOLOGY_SPECIAL,
        pdg_value=13.606,
        pdg_unit="eV",
        pdg_uncertainty=0.001,
        notes="Ry = m_e α² / 2",
    )
)

HE_PLUS_IONIZATION = _register(
    ParticleConfig(
        name="He_plus_IE",
        symbol="IE(He⁺)",
        particle_type=ParticleType.OTHER,
        formula_type=FormulaType.COSMOLOGY_SPECIAL,
        pdg_value=54.418,
        pdg_unit="eV",
        pdg_uncertainty=0.001,
        notes="IE = Z² × Ry = 4 × 13.606",
    )
)

HYDROGEN_POLARIZABILITY = _register(
    ParticleConfig(
        name="alpha_H",
        symbol="α_H",
        particle_type=ParticleType.OTHER,
        formula_type=FormulaType.COSMOLOGY_SPECIAL,
        pdg_value=4.5,
        pdg_unit="a0^3",
        pdg_uncertainty=0.1,
        notes="α_H = 9/2 a₀³",
    )
)

H2_BOND_LENGTH = _register(
    ParticleConfig(
        name="H2_bond",
        symbol="r_e(H₂)",
        particle_type=ParticleType.OTHER,
        formula_type=FormulaType.COSMOLOGY_SPECIAL,
        pdg_value=0.741,
        pdg_unit="Angstrom",
        pdg_uncertainty=0.001,
        notes="r_e = √2 a₀ (1-q/2)",
    )
)

H2_DISSOCIATION = _register(
    ParticleConfig(
        name="H2_dissociation",
        symbol="D₀(H₂)",
        particle_type=ParticleType.OTHER,
        formula_type=FormulaType.COSMOLOGY_SPECIAL,
        pdg_value=4.478,
        pdg_unit="eV",
        pdg_uncertainty=0.001,
        notes="D₀ = Ry/3 (1-q/2)",
    )
)

FINE_STRUCTURE_2P = _register(
    ParticleConfig(
        name="fine_structure_2P",
        symbol="ΔE_FS",
        particle_type=ParticleType.OTHER,
        formula_type=FormulaType.COSMOLOGY_SPECIAL,
        pdg_value=10.97,
        pdg_unit="GHz",
        pdg_uncertainty=0.01,
        notes="ΔE = α⁴ m_e / 32",
    )
)

HYPERFINE_21CM = _register(
    ParticleConfig(
        name="hyperfine_21cm",
        symbol="ν_hfs",
        particle_type=ParticleType.OTHER,
        formula_type=FormulaType.COSMOLOGY_SPECIAL,
        pdg_value=1420.406,
        pdg_unit="MHz",
        pdg_uncertainty=0.001,
        notes="Standard QED with SRT α",
    )
)

PROTON_RADIUS = _register(
    ParticleConfig(
        name="proton_radius",
        symbol="r_p",
        particle_type=ParticleType.OTHER,
        formula_type=FormulaType.COSMOLOGY_SPECIAL,
        pdg_value=0.8414,
        pdg_unit="fm",
        pdg_uncertainty=0.0001,
        notes="r_p = 4ℏc/m_p",
    )
)

# -----------------------------------------------------------------------------
# NUCLEAR PHYSICS OBSERVABLES
# -----------------------------------------------------------------------------

SEMF_SURFACE = _register(
    ParticleConfig(
        name="semf_aS",
        symbol="a_S",
        particle_type=ParticleType.NUCLEAR,
        formula_type=FormulaType.NUCLEAR_BINDING,
        pdg_value=17.8,
        pdg_unit="MeV",
        pdg_uncertainty=0.1,
        notes="a_S = E*(1-4q)",
    )
)

SEMF_VOLUME = _register(
    ParticleConfig(
        name="semf_aV",
        symbol="a_V",
        particle_type=ParticleType.NUCLEAR,
        formula_type=FormulaType.NUCLEAR_BINDING,
        pdg_value=15.75,
        pdg_unit="MeV",
        pdg_uncertainty=0.1,
        notes="a_V = E*(φ⁻¹+6q)",
    )
)

SEMF_ASYMMETRY = _register(
    ParticleConfig(
        name="semf_aA",
        symbol="a_A",
        particle_type=ParticleType.NUCLEAR,
        formula_type=FormulaType.NUCLEAR_BINDING,
        pdg_value=23.7,
        pdg_unit="MeV",
        pdg_uncertainty=0.1,
        notes="a_A = E*(1+7q)",
    )
)

SEMF_PAIRING = _register(
    ParticleConfig(
        name="semf_aP",
        symbol="a_P",
        particle_type=ParticleType.NUCLEAR,
        formula_type=FormulaType.NUCLEAR_BINDING,
        pdg_value=12.0,
        pdg_unit="MeV",
        pdg_uncertainty=0.1,
        notes="a_P = E*/φ (1-q)",
    )
)

SEMF_COULOMB = _register(
    ParticleConfig(
        name="semf_aC",
        symbol="a_C",
        particle_type=ParticleType.NUCLEAR,
        formula_type=FormulaType.NUCLEAR_BINDING,
        pdg_value=0.711,
        pdg_unit="MeV",
        pdg_uncertainty=0.005,
        notes="a_C = E*·q(1+10q)",
    )
)

IRON_56_BINDING = _register(
    ParticleConfig(
        name="Fe56_binding",
        symbol="B/A(⁵⁶Fe)",
        particle_type=ParticleType.NUCLEAR,
        formula_type=FormulaType.NUCLEAR_BINDING,
        pdg_value=8.79,
        pdg_unit="MeV",
        pdg_uncertainty=0.01,
        notes="B/A = E*/2φ · √2 · (1+q/4)",
    )
)

TRITON_BINDING = _register(
    ParticleConfig(
        name="triton_binding",
        symbol="B(³H)",
        particle_type=ParticleType.NUCLEAR,
        formula_type=FormulaType.NUCLEAR_BINDING,
        pdg_value=8.482,
        pdg_unit="MeV",
        pdg_uncertainty=0.001,
        notes="B_t = E*/φ² (1+4q)(1+q/6)(1+q/27)",
    )
)

# -----------------------------------------------------------------------------
# CONDENSED MATTER OBSERVABLES
# -----------------------------------------------------------------------------

BCS_GAP_RATIO = _register(
    ParticleConfig(
        name="bcs_ratio",
        symbol="2Δ₀/kTc",
        particle_type=ParticleType.OTHER,
        formula_type=FormulaType.COSMOLOGY_SPECIAL,
        pdg_value=3.52,
        pdg_unit="dimensionless",
        pdg_uncertainty=0.01,
        notes="Ratio = 2φ + 10q",
    )
)

TC_YBCO = _register(
    ParticleConfig(
        name="Tc_YBCO",
        symbol="Tc(YBCO)",
        particle_type=ParticleType.OTHER,
        formula_type=FormulaType.COSMOLOGY_SPECIAL,
        pdg_value=92.4,
        pdg_unit="Kelvin",
        pdg_uncertainty=1.0,
        notes="Tc = E* (φ² + 2)",
    )
)

TC_BSCCO = _register(
    ParticleConfig(
        name="Tc_BSCCO",
        symbol="Tc(BSCCO)",
        particle_type=ParticleType.OTHER,
        formula_type=FormulaType.COSMOLOGY_SPECIAL,
        pdg_value=110.0,
        pdg_unit="Kelvin",
        pdg_uncertainty=2.0,
        notes="Tc = E* (φ² + 3)(1-q/φ)",
    )
)

GRAPHENE_FERMI_VELOCITY = _register(
    ParticleConfig(
        name="vF_graphene",
        symbol="v_F",
        particle_type=ParticleType.OTHER,
        formula_type=FormulaType.COSMOLOGY_SPECIAL,
        pdg_value=1.0e6,  # approx c/300
        pdg_unit="m/s",
        pdg_uncertainty=0.1e6,
        notes="v_F = c / (10 × h(E₈)) = c/300",
    )
)

# -----------------------------------------------------------------------------
# GRAVITATIONAL PHYSICS OBSERVABLES
# -----------------------------------------------------------------------------

BH_ENTROPY_CORR = _register(
    ParticleConfig(
        name="BH_entropy_corr",
        symbol="S_BH/S_0",
        particle_type=ParticleType.OTHER,
        formula_type=FormulaType.COSMOLOGY_SPECIAL,
        pdg_value=1.0,  # Standard GR
        pdg_unit="ratio",
        pdg_uncertainty=0.0,
        notes="S = A/4 (1+q/4)",
    )
)

HAWKING_TEMP_CORR = _register(
    ParticleConfig(
        name="Hawking_T_corr",
        symbol="T_H/T_0",
        particle_type=ParticleType.OTHER,
        formula_type=FormulaType.COSMOLOGY_SPECIAL,
        pdg_value=1.0,  # Standard GR
        pdg_unit="ratio",
        pdg_uncertainty=0.0,
        notes="T_H = T_0 (1-q/8)",
    )
)

GW_ECHO_150914 = _register(
    ParticleConfig(
        name="echo_gw150914",
        symbol="Δt(GW150914)",
        particle_type=ParticleType.OTHER,
        formula_type=FormulaType.COSMOLOGY_SPECIAL,
        pdg_value=0.59,  # Tentative
        pdg_unit="ms",
        pdg_uncertainty=0.1,
        notes="Δt = 2r_H/c ln(φ)",
    )
)

GW_ECHO_190521 = _register(
    ParticleConfig(
        name="echo_gw190521",
        symbol="Δt(GW190521)",
        particle_type=ParticleType.OTHER,
        formula_type=FormulaType.COSMOLOGY_SPECIAL,
        pdg_value=1.35,  # Tentative
        pdg_unit="ms",
        pdg_uncertainty=0.2,
        notes="Δt = 2r_H/c ln(φ)",
    )
)

GW_ECHO_170817 = _register(
    ParticleConfig(
        name="echo_gw170817",
        symbol="Δt(GW170817)",
        particle_type=ParticleType.OTHER,
        formula_type=FormulaType.COSMOLOGY_SPECIAL,
        pdg_value=0.038,  # Tentative
        pdg_unit="ms",
        pdg_uncertainty=0.01,
        notes="Δt = 2R/c ln(φ)",
    )
)

MOND_ACCELERATION = _register(
    ParticleConfig(
        name="mond_a0",
        symbol="a₀",
        particle_type=ParticleType.OTHER,
        formula_type=FormulaType.COSMOLOGY_SPECIAL,
        pdg_value=1.2e-10,
        pdg_unit="m/s^2",
        pdg_uncertainty=0.2e-10,
        notes="a₀ = √q c H₀",
    )
)

# -----------------------------------------------------------------------------
# MISSING PARTICLE PHYSICS OBSERVABLES
# -----------------------------------------------------------------------------

MUON_G2 = _register(
    ParticleConfig(
        name="muon_g2",
        symbol="a_μ",
        particle_type=ParticleType.LEPTON,
        formula_type=FormulaType.COSMOLOGY_SPECIAL,
        pdg_value=25.1e-10,
        pdg_unit="dimensionless",
        pdg_uncertainty=0.1e-10,
        notes="a_μ = a_μ⁰(1+q/8)(1+q²/φ)",
    )
)

TAU_G2 = _register(
    ParticleConfig(
        name="tau_g2",
        symbol="a_τ",
        particle_type=ParticleType.LEPTON,
        formula_type=FormulaType.COSMOLOGY_SPECIAL,
        pdg_value=1.18e-3,
        pdg_unit="dimensionless",
        pdg_uncertainty=0.05e-3,
        notes="a_τ = α/2π (1+q/φ)",
    )
)

NEUTRON_LIFETIME = _register(
    ParticleConfig(
        name="neutron_lifetime",
        symbol="τ_n",
        particle_type=ParticleType.NUCLEON,
        formula_type=FormulaType.COSMOLOGY_SPECIAL,
        pdg_value=879.4,
        pdg_unit="s",
        pdg_uncertainty=0.6,
        notes="τ_n = τ_SM / (1+q/φ) (1-q/4π)(1+q/78)",
    )
)

WEAK_MIXING_ANGLE = _register(
    ParticleConfig(
        name="sin2_thetaW",
        symbol="sin²θ_W",
        particle_type=ParticleType.GAUGE_BOSON,
        formula_type=FormulaType.COSMOLOGY_SPECIAL,
        pdg_value=0.2312,
        pdg_unit="dimensionless",
        pdg_uncertainty=0.0001,
        notes="sin²θ_W = g'²/(g²+g'²)",
    )
)

JARLSKOG_INVARIANT = _register(
    ParticleConfig(
        name="J_CP",
        symbol="J_CP",
        particle_type=ParticleType.MIXING,
        formula_type=FormulaType.COSMOLOGY_SPECIAL,
        pdg_value=3.08e-5,
        pdg_unit="dimensionless",
        pdg_uncertainty=0.15e-5,
        notes="J_CP = q²/E* (1-4q)(1-qφ²)(1-q/φ³)",
    )
)

DIRAC_CP_PHASE = _register(
    ParticleConfig(
        name="delta_CP",
        symbol="δ_CP",
        particle_type=ParticleType.MIXING,
        formula_type=FormulaType.COSMOLOGY_SPECIAL,
        pdg_value=195.0,
        pdg_unit="degrees",
        pdg_uncertainty=25.0,
        notes="δ_CP = 180(1-4q)(1+q/φ)",
    )
)

MAJORANA_PHASE_21 = _register(
    ParticleConfig(
        name="alpha_21",
        symbol="α₂₁",
        particle_type=ParticleType.MIXING,
        formula_type=FormulaType.COSMOLOGY_SPECIAL,
        pdg_value=0.0,  # Prediction
        pdg_unit="degrees",
        pdg_uncertainty=0.0,
        notes="α₂₁ = 180 q/φ",
    )
)

MAJORANA_PHASE_31 = _register(
    ParticleConfig(
        name="alpha_31",
        symbol="α₃₁",
        particle_type=ParticleType.MIXING,
        formula_type=FormulaType.COSMOLOGY_SPECIAL,
        pdg_value=0.0,  # Prediction
        pdg_unit="degrees",
        pdg_uncertainty=0.0,
        notes="α₃₁ = 180 qφ",
    )
)

GUT_SCALE = _register(
    ParticleConfig(
        name="mu_GUT",
        symbol="μ_GUT",
        particle_type=ParticleType.COSMOLOGY,
        formula_type=FormulaType.COSMOLOGY_SPECIAL,
        pdg_value=1.0e15,  # Approx
        pdg_unit="GeV",
        pdg_uncertainty=0.5e15,
        notes="μ_GUT = v e^(φ⁷)",
    )
)

REHEATING_TEMP = _register(
    ParticleConfig(
        name="T_reh",
        symbol="T_reh",
        particle_type=ParticleType.COSMOLOGY,
        formula_type=FormulaType.COSMOLOGY_SPECIAL,
        pdg_value=1.0e10,  # Approx
        pdg_unit="GeV",
        pdg_uncertainty=0.5e10,
        notes="T_reh = v e^(φ⁶) / φ",
    )
)

TCC_TETRAQUARK = _register(
    ParticleConfig(
        name="T_cc",
        symbol="T_cc⁺",
        particle_type=ParticleType.MESON,
        formula_type=FormulaType.COSMOLOGY_SPECIAL,
        pdg_value=3875.1,
        pdg_unit="MeV",
        pdg_uncertainty=0.3,
        notes="m = m_D + m_D*",
    )
)

X_3872_MESON = _register(
    ParticleConfig(
        name="X_3872",
        symbol="X(3872)",
        particle_type=ParticleType.MESON,
        formula_type=FormulaType.COSMOLOGY_SPECIAL,
        pdg_value=3871.65,
        pdg_unit="MeV",
        pdg_uncertainty=0.06,
        notes="m = m_D0 + m_D*0",
    )
)

PC_4457_PENTAQUARK = _register(
    ParticleConfig(
        name="Pc_4457",
        symbol="P_c(4457)",
        particle_type=ParticleType.BARYON,
        formula_type=FormulaType.COSMOLOGY_SPECIAL,
        pdg_value=4457.0,  # Approx
        pdg_unit="MeV",
        pdg_uncertainty=5.0,
        notes="m = (m_Σc + m_D*) (1-q/120)",
    )
)
