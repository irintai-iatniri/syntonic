"""
Universal Syntony Correction Hierarchy (v1.2)
==============================================

Complete system of 60+ correction factors for SRT predictions.
Each factor corresponds to specific geometric, topological, or group-theoretic structures.

The fundamental principle:
  "Every correction factor corresponds to a specific geometric structure that the observable samples.
   The denominator of the factor equals the dimension, rank, or count of that structure."

HIERARCHY STRUCTURE:
  - Levels 0-59: Primary correction factors (additive/multiplicative)
  - Levels 100-106: Multiplicative suppression factors (1/(1+...))

CATEGORIES:
  - Fixed-point: q/1000, q/720 (Coxeter-Kissing products)
  - E₈ structures: dim=248, rank=8, roots=240, positive=120, Coxeter=30
  - E₇ structures: dim=133, rank=7, roots=126, positive=63, Coxeter=18, fund=56
  - E₆ structures: dim=78, rank=6, roots=72, positive=36, fund=27, Coxeter=12
  - D₄ structures: kissing=24, adjoint=28, Coxeter=6
  - F₄ structure: dim=52, Coxeter=12
  - G₂ structure: dim=14, Coxeter=6
  - QCD loops: 3π, 5π, 6π (flavor counts)
  - Loop integrals: 4π, 2π, π (integration measures)
  - Topology: T⁴=4, N_gen=3, 6=2×3, 9=3², 12=4×3
  - Golden powers: φ, φ², φ³, φ⁴, φ⁵ (recursion eigenvalues)
  - q powers: q, q², q³ (multi-loop orders)
  - Enhanced: qφ, qφ², qφ³, qφ⁴, qφ⁵ (layer transitions)
  - Multiple q: 3q, 4q, 6q, 8q, πq (collective enhancements)
  - Suppressions: 1/(1+qφ⁻²), 1/(1+qφ⁻¹), 1/(1+q), 1/(1+qφ), 1/(1+qφ²), 1/(1+qφ³)

STATUS CODES:
  - USED: Currently applied in validated predictions
  - NEW: Recently added, ready for application
  - RESERVED: Available for future physics

Reference: Universal_Syntony_Correction_Hierarchy.md (v1.2)
"""

from enum import Enum
from dataclasses import dataclass
from typing import Tuple

# Import constants from syntonic library
from .hierarchy import PHI, E_STAR, Q, PI

# ============================================================================
# CORE CONSTANTS
# ============================================================================

# Constants imported from hierarchy module (which uses syntonic.exact)

# ============================================================================
# LIE GROUP INVARIANTS
# ============================================================================

class LieGroupStructure:
    """Dimensions and root counts for exceptional Lie algebras"""
    
    # E₈ (largest exceptional group)
    E8_DIM = 248
    E8_RANK = 8
    E8_ROOTS = 240
    E8_POS_ROOTS = 120
    E8_COXETER = 30
    
    # E₇ (intermediate unification)
    E7_DIM = 133
    E7_RANK = 7
    E7_ROOTS = 126
    E7_POS_ROOTS = 63
    E7_COXETER = 18
    
    # E₆ (GUT scale)
    E6_DIM = 78
    E6_RANK = 6
    E6_ROOTS = 72
    E6_POS_ROOTS = 36
    E6_COXETER = 12
    E6_FUNDAMENTAL = 27  # Quark triplet representation
    
    # Other structures
    F4_DIM = 52
    F4_COXETER = 12
    G2_DIM = 14
    G2_COXETER = 6
    D4_DIM = 28
    D4_COXETER = 6
    D4_KISSING = 24  # Kissing number (collapse threshold)
    
    # Standard Model
    TOPOLOGY_DIM = 4  # T⁴ topology
    GENERATIONS = 3
    FLAVORS_LIGHT = 3  # u, d, s
    FLAVORS_5 = 5  # u, d, s, c, b
    FLAVORS_6 = 6  # all quarks


# ============================================================================
# CORRECTION FACTORS
# ============================================================================

@dataclass
class CorrectionFactor:
    """Represents a single correction factor in the hierarchy"""
    
    level: int
    name: str
    magnitude: float  # Approximate percentage correction
    formula: str  # Mathematical formula
    origin: str  # Geometric/physical origin
    divisor: float  # The denominator in q/divisor or base term
    base: str = "q"  # Base term: "q", "q²", "qφ", "φ", etc.
    applications: list = None  # Physical applications
    status: str = "USED"  # USED, NEW, RESERVED
    
    def __post_init__(self):
        if self.applications is None:
            self.applications = []
    
    def compute(self, q=Q, phi=PHI) -> float:
        """Compute the numerical value of this correction factor"""
        if self.base == "1":
            return 1.0
        elif self.base == "q":
            return q / self.divisor
        elif self.base == "q²":
            return (q ** 2) / self.divisor
        elif self.base == "q³":
            return (q ** 3) / self.divisor
        elif self.base == "q/φ²":
            return q / (self.divisor * (phi ** 2))
        elif self.base == "q/φ":
            return q / (self.divisor * phi)
        elif self.base == "qφ":
            return (q * phi) / self.divisor
        elif self.base == "qφ²":
            return (q * (phi ** 2)) / self.divisor
        elif self.base == "qφ³":
            return (q * (phi ** 3)) / self.divisor
        elif self.base == "qφ⁴":
            return (q * (phi ** 4)) / self.divisor
        elif self.base == "qφ⁵":
            return (q * (phi ** 5)) / self.divisor
        elif self.base == "q²φ":
            return ((q ** 2) * phi) / self.divisor
        elif self.base == "3q":
            return 3 * q / self.divisor
        elif self.base == "4q":
            return 4 * q / self.divisor
        elif self.base == "6q":
            return 6 * q / self.divisor
        elif self.base == "8q":
            return 8 * q / self.divisor
        elif self.base == "πq":
            return PI * q / self.divisor
        elif self.base == "1/(1+...)":
            # Divisor already contains the full denominator (1 + some_term)
            return 1.0 / self.divisor
        else:
            raise ValueError(f"Unknown base: {self.base}")


class CorrectionHierarchy:
    """Complete hierarchy of 60+ correction factors"""
    
    def __init__(self):
        self.factors = {}
        self._initialize_hierarchy()
    
    def _initialize_hierarchy(self):
        """Initialize all correction factors from Level 0-57"""
        
        # Level 0: Tree-level (exact)
        self.factors[0] = CorrectionFactor(
            level=0, name="Tree-level (Exact)",
            magnitude=0, formula="1",
            origin="Classical limit, no corrections",
            divisor=1, base="1",
            applications=["E*, spectral constant", "3-generation structure"],
            status="USED"
        )
        
        # Level 1: Fixed-point stability
        self.factors[1] = CorrectionFactor(
            level=1, name="Fixed-Point Stability",
            magnitude=0.0027, formula="q/1000",
            origin=f"E₈³/dim(E₆ fund) = 30³/27 = 27000",
            divisor=1000, base="q",
            applications=["Proton mass (EXACT)", "Fixed-point particles"],
            status="USED"
        )
        
        # Level 2: Coxeter-Kissing product
        self.factors[2] = CorrectionFactor(
            level=2, name="Coxeter-Kissing Product",
            magnitude=0.0038, formula="q/720",
            origin=f"h(E₈) × K(D₄) = 30 × 24 = 720 = 6!",
            divisor=720, base="q",
            applications=["Neutron mass", "Tau mass", "CMB peaks"],
            status="USED"
        )
        
        # Level 3: q/360
        self.factors[3] = CorrectionFactor(
            level=3, name="Cone Periodicity",
            magnitude=0.0076, formula="q/360",
            origin="10 × 36 = complete cone cycles",
            divisor=360, base="q",
            applications=["Kaon mixing", "Periodic corrections"],
            status="USED"
        )
        
        # Level 4: Full E₈ adjoint
        self.factors[4] = CorrectionFactor(
            level=4, name="Full E₈ Adjoint",
            magnitude=0.011, formula=f"q/{LieGroupStructure.E8_DIM}",
            origin=f"dim(E₈) = {LieGroupStructure.E8_DIM}",
            divisor=LieGroupStructure.E8_DIM, base="q",
            applications=["GUT-scale observables", "E₈ representation"],
            status="USED"
        )
        
        # Level 5: Full E₈ root system
        self.factors[5] = CorrectionFactor(
            level=5, name="Full E₈ Root System",
            magnitude=0.0114, formula=f"q/{LieGroupStructure.E8_ROOTS}",
            origin=f"|Φ(E₈)| = {LieGroupStructure.E8_ROOTS}",
            divisor=LieGroupStructure.E8_ROOTS, base="q",
            applications=["Complete gauge structure", "Root system corrections"],
            status="USED"
        )
        
        # Level 6: Full E₇ adjoint
        self.factors[6] = CorrectionFactor(
            level=6, name="Full E₇ Adjoint",
            magnitude=0.0206, formula=f"q/{LieGroupStructure.E7_DIM}",
            origin=f"dim(E₇) = {LieGroupStructure.E7_DIM}",
            divisor=LieGroupStructure.E7_DIM, base="q",
            applications=["E₇ breaking corrections", "Intermediate scale"],
            status="USED"
        )
        
        # Level 7: Full E₇ root system
        self.factors[7] = CorrectionFactor(
            level=7, name="Full E₇ Root System",
            magnitude=0.0217, formula=f"q/{LieGroupStructure.E7_ROOTS}",
            origin=f"|Φ(E₇)| = {LieGroupStructure.E7_ROOTS}",
            divisor=LieGroupStructure.E7_ROOTS, base="q",
            applications=["E₇ threshold corrections"],
            status="USED"
        )
        
        # Level 8: E₈ positive roots
        self.factors[8] = CorrectionFactor(
            level=8, name="E₈ Positive Roots",
            magnitude=0.023, formula=f"q/{LieGroupStructure.E8_POS_ROOTS}",
            origin=f"|Φ⁺(E₈)| = {LieGroupStructure.E8_POS_ROOTS}",
            divisor=LieGroupStructure.E8_POS_ROOTS, base="q",
            applications=["E₈ root corrections", "Gauge structure"],
            status="USED"
        )
        
        # Level 9: Second-order massless (q²/φ)
        self.factors[9] = CorrectionFactor(
            level=9, name="Second-Order Massless",
            magnitude=0.046, formula="q²/φ",
            origin="Massless particle propagation through T⁴",
            divisor=PHI, base="q²",
            applications=["CMB peaks ℓ₁-ℓ₅", "Neutrino species N_eff", "Photon corrections"],
            status="USED"
        )
        
        # Level 10: Full E₆ adjoint
        self.factors[10] = CorrectionFactor(
            level=10, name="Full E₆ Adjoint",
            magnitude=0.035, formula=f"q/{LieGroupStructure.E6_DIM}",
            origin=f"dim(E₆) = {LieGroupStructure.E6_DIM}",
            divisor=LieGroupStructure.E6_DIM, base="q",
            applications=["E₆ gauge corrections"],
            status="USED"
        )
        
        # Level 11: Full E₆ root system
        self.factors[11] = CorrectionFactor(
            level=11, name="Full E₆ Root System",
            magnitude=0.0380, formula=f"q/{LieGroupStructure.E6_ROOTS}",
            origin=f"|Φ(E₆)| = {LieGroupStructure.E6_ROOTS}",
            divisor=LieGroupStructure.E6_ROOTS, base="q",
            applications=["E₆ structure corrections"],
            status="USED"
        )
        
        # Level 12: E₇ positive roots
        self.factors[12] = CorrectionFactor(
            level=12, name="E₇ Positive Roots",
            magnitude=0.0435, formula=f"q/{LieGroupStructure.E7_POS_ROOTS}",
            origin=f"|Φ⁺(E₇)| = {LieGroupStructure.E7_POS_ROOTS}",
            divisor=LieGroupStructure.E7_POS_ROOTS, base="q",
            applications=["E₇ positive root corrections"],
            status="USED"
        )
        
        # Level 13: F₄ structure
        self.factors[13] = CorrectionFactor(
            level=13, name="F₄ Structure",
            magnitude=0.0527, formula=f"q/{LieGroupStructure.F4_DIM}",
            origin=f"dim(F₄) = {LieGroupStructure.F4_DIM}",
            divisor=LieGroupStructure.F4_DIM, base="q",
            applications=["Jordan algebra corrections"],
            status="USED"
        )
        
        # Level 14: Second-order vacuum (q²)
        self.factors[14] = CorrectionFactor(
            level=14, name="Second-Order Vacuum",
            magnitude=0.075, formula="q²",
            origin="Two-loop universal corrections",
            divisor=1, base="q²",
            applications=["Two-loop processes", "Higher precision"],
            status="USED"
        )
        
        # Level 15: E₆ positive roots
        self.factors[15] = CorrectionFactor(
            level=15, name="E₆ Positive Roots",
            magnitude=0.076, formula=f"q/{LieGroupStructure.E6_POS_ROOTS}",
            origin=f"|Φ⁺(E₆)| = {LieGroupStructure.E6_POS_ROOTS}",
            divisor=LieGroupStructure.E6_POS_ROOTS, base="q",
            applications=["E₆ cone geometry"],
            status="USED"
        )
        
        # Level 16: E₆ fundamental representation
        self.factors[16] = CorrectionFactor(
            level=16, name="E₆ Fundamental Representation",
            magnitude=0.101, formula=f"q/{LieGroupStructure.E6_FUNDAMENTAL}",
            origin=f"dim(E₆ fund) = {LieGroupStructure.E6_FUNDAMENTAL} = 3³",
            divisor=LieGroupStructure.E6_FUNDAMENTAL, base="q",
            applications=["Quark triplet", "E₆ fundamental structure"],
            status="USED"
        )
        
        # Level 17: Six-flavor QCD
        self.factors[17] = CorrectionFactor(
            level=17, name="Six-Flavor QCD",
            magnitude=0.145, formula="q/(6π)",
            origin="6 active quarks above top threshold",
            divisor=6 * PI, base="q",
            applications=["Ultra-high energy processes"],
            status="USED"
        )
        
        # Level 18: Five-flavor QCD
        self.factors[18] = CorrectionFactor(
            level=18, name="Five-Flavor QCD",
            magnitude=0.174, formula="q/(5π)",
            origin="5 active quarks at M_Z scale",
            divisor=5 * PI, base="q",
            applications=["Tau mass (EXACT)", "α_s(M_Z) precision"],
            status="USED"
        )
        
        # Level 19: One-loop radiative
        self.factors[19] = CorrectionFactor(
            level=19, name="One-Loop Radiative",
            magnitude=0.218, formula="q/(4π)",
            origin="Standard 4D loop integration",
            divisor=4 * PI, base="q",
            applications=["W/Z mass precision", "Top quark mass", "Neutron lifetime"],
            status="USED"
        )
        
        # Level 20: Topology × Generations
        self.factors[20] = CorrectionFactor(
            level=20, name="Topology × Generations",
            magnitude=0.228, formula=f"q/{LieGroupStructure.TOPOLOGY_DIM * LieGroupStructure.GENERATIONS}",
            origin=f"dim(T⁴) × N_gen = {LieGroupStructure.TOPOLOGY_DIM} × {LieGroupStructure.GENERATIONS}",
            divisor=12, base="q",
            applications=["θ₁₃ reactor angle"],
            status="USED"
        )
        
        # Level 21: Three-flavor QCD
        self.factors[21] = CorrectionFactor(
            level=21, name="Three-Flavor QCD",
            magnitude=0.290, formula="q/(3π)",
            origin="3 light active quarks below charm",
            divisor=3 * PI, base="q",
            applications=["Low-energy QCD"],
            status="USED"
        )
        
        # Level 22: E₈ Cartan subalgebra
        self.factors[22] = CorrectionFactor(
            level=22, name="E₈ Cartan Subalgebra (Rank)",
            magnitude=0.342, formula=f"q/{LieGroupStructure.E8_RANK}",
            origin=f"rank(E₈) = {LieGroupStructure.E8_RANK}",
            divisor=LieGroupStructure.E8_RANK, base="q",
            applications=["θ₂₃ atmospheric angle", "a_μ (g-2)", "Full exceptional structure"],
            status="USED"
        )
        
        # Level 23: D₄ kissing number
        self.factors[23] = CorrectionFactor(
            level=23, name="D₄ Kissing Number",
            magnitude=0.114, formula=f"q/{LieGroupStructure.D4_KISSING}",
            origin=f"K(D₄) = {LieGroupStructure.D4_KISSING} (collapse threshold)",
            divisor=LieGroupStructure.D4_KISSING, base="q",
            applications=["Collapse corrections", "Lattice geometry"],
            status="USED"
        )
        
        # Level 24: Sub-Generation Structure
        self.factors[24] = CorrectionFactor(
            level=24, name="Sub-Generation Structure",
            magnitude=0.457, formula=f"q/{LieGroupStructure.GENERATIONS * 2}",
            origin="2 × 3 = chirality × generations",
            divisor=6, base="q",
            applications=["Neutron-proton mass difference", "Kaon mass", "Chirality"],
            status="USED"
        )
        
        # Level 25: Third golden power
        self.factors[25] = CorrectionFactor(
            level=25, name="Third Golden Power",
            magnitude=0.65, formula="q/φ³",
            origin="φ³ ≈ 4.236 (third recursion layer)",
            divisor=PHI ** 3, base="q",
            applications=["Third-generation effects", "Deep recursion"],
            status="USED"
        )
        
        # Level 26: Quarter layer
        self.factors[26] = CorrectionFactor(
            level=26, name="Quarter Layer",
            magnitude=0.685, formula="q/4",
            origin="1/4 recursion layer crossing",
            divisor=4, base="q",
            applications=["Cabibbo angle", "V_cb CKM", "Sphaleron"],
            status="USED"
        )
        
        # Level 27: SO(8) adjoint
        self.factors[27] = CorrectionFactor(
            level=27, name="SO(8) Adjoint (D₄ Triality)",
            magnitude=0.0978, formula=f"q/{LieGroupStructure.D4_DIM}",
            origin=f"dim(SO(8)) = {LieGroupStructure.D4_DIM} (triality)",
            divisor=LieGroupStructure.D4_DIM, base="q",
            applications=["Triality corrections"],
            status="USED"
        )
        
        # Level 28: Half layer
        self.factors[28] = CorrectionFactor(
            level=28, name="Half Layer",
            magnitude=1.37, formula="q/2",
            origin="1/2 recursion layer crossing",
            divisor=2, base="q",
            applications=["Solar angle θ₁₂", "Flavor mixing"],
            status="USED"
        )
        
        # Level 29: Scale running (golden eigenvalue)
        self.factors[29] = CorrectionFactor(
            level=29, name="Scale Running",
            magnitude=1.69, formula="q/φ",
            origin="Golden eigenvalue of recursion operator",
            divisor=PHI, base="q",
            applications=["Λ_QCD running", "Scale effects"],
            status="USED"
        )
        
        # Level 30: Universal vacuum
        self.factors[30] = CorrectionFactor(
            level=30, name="Universal Vacuum",
            magnitude=2.74, formula="q",
            origin="Fundamental syntony deficit",
            divisor=1, base="q",
            applications=["Base corrections"],
            status="USED"
        )
        
        # Level 31: Double layer
        self.factors[31] = CorrectionFactor(
            level=31, name="Double Layer",
            magnitude=4.43, formula="qφ",
            origin="Two recursion layer transitions",
            divisor=1, base="qφ",
            applications=["Double-generation transitions"],
            status="USED"
        )
        
        # Level 32: Fixed point (φ² = φ + 1)
        self.factors[32] = CorrectionFactor(
            level=32, name="Fixed Point Eigenvalue",
            magnitude=7.17, formula="qφ²",
            origin="φ² = φ + 1 (recursion fixed point)",
            divisor=1, base="qφ²",
            applications=["Fixed-point stability"],
            status="USED"
        )
        
        # Level 33: Full T⁴ topology
        self.factors[33] = CorrectionFactor(
            level=33, name="Full T⁴ Topology",
            magnitude=10.96, formula="4q",
            origin="dim(T⁴) = 4 CP-violation coupling",
            divisor=1, base="4q",
            applications=["δ_CP phase", "Baryon asymmetry", "CP violation"],
            status="USED"
        )
        
        # Level 34: E₇ Coxeter
        self.factors[34] = CorrectionFactor(
            level=34, name="E₇ Coxeter Number",
            magnitude=0.152, formula=f"q/{LieGroupStructure.E7_COXETER}",
            origin=f"h(E₇) = {LieGroupStructure.E7_COXETER}",
            divisor=LieGroupStructure.E7_COXETER, base="q",
            applications=["E₇ breaking scale"],
            status="USED"
        )
        
        # Level 35: E₆ Coxeter
        self.factors[35] = CorrectionFactor(
            level=35, name="E₆ Coxeter Number",
            magnitude=0.228, formula=f"q/{LieGroupStructure.E6_COXETER}",
            origin=f"h(E₆) = {LieGroupStructure.E6_COXETER}",
            divisor=LieGroupStructure.E6_COXETER, base="q",
            applications=["E₆ breaking scale"],
            status="USED"
        )
        
        # Level 36: E₇ Cartan
        self.factors[36] = CorrectionFactor(
            level=36, name="E₇ Cartan Subalgebra",
            magnitude=0.391, formula=f"q/{LieGroupStructure.E7_RANK}",
            origin=f"rank(E₇) = {LieGroupStructure.E7_RANK}",
            divisor=LieGroupStructure.E7_RANK, base="q",
            applications=["E₇ rank corrections"],
            status="USED"
        )
        
        # Level 37: E₆ Cartan
        self.factors[37] = CorrectionFactor(
            level=37, name="E₆ Cartan Subalgebra",
            magnitude=0.457, formula=f"q/{LieGroupStructure.E6_RANK}",
            origin=f"rank(E₆) = {LieGroupStructure.E6_RANK}",
            divisor=LieGroupStructure.E6_RANK, base="q",
            applications=["E₆ rank corrections"],
            status="USED"
        )
        
        # Level 38: G₂ dimension
        self.factors[38] = CorrectionFactor(
            level=38, name="G₂ Octonion Automorphisms",
            magnitude=0.196, formula=f"q/{LieGroupStructure.G2_DIM}",
            origin=f"dim(G₂) = {LieGroupStructure.G2_DIM}",
            divisor=LieGroupStructure.G2_DIM, base="q",
            applications=["Octonion structure"],
            status="USED"
        )
        
        # Additional higher-order factors
        self.factors[39] = CorrectionFactor(
            level=39, name="Fourth Golden Power",
            magnitude=0.4, formula="q/φ⁴",
            origin="φ⁴ (fourth recursion layer)",
            divisor=PHI ** 4, base="q",
            applications=["Fourth-order recursion"],
            status="NEW"
        )
        
        self.factors[40] = CorrectionFactor(
            level=40, name="Fifth Golden Power",
            magnitude=0.247, formula="q/φ⁵",
            origin="φ⁵ (fifth recursion layer)",
            divisor=PHI ** 5, base="q",
            applications=["Fifth-order recursion"],
            status="NEW"
        )
        
        # ====== ADDITIONAL FACTORS FROM v1.2 (Levels 41-60) ======
        
        # Level 41: q/32 (2⁵)
        self.factors[41] = CorrectionFactor(
            level=41, name="Binary Fifth Power",
            magnitude=0.0856, formula="q/32",
            origin="2⁵ = 32 (five-fold binary structure)",
            divisor=32, base="q",
            applications=["Higher spinor structures", "Five-dimensional effects"],
            status="RESERVED"
        )
        
        # Level 42: q/30 (E₈ Coxeter alone)
        self.factors[42] = CorrectionFactor(
            level=42, name="E₈ Coxeter Alone",
            magnitude=0.0913, formula=f"q/{LieGroupStructure.E8_COXETER}",
            origin=f"h(E₈) = {LieGroupStructure.E8_COXETER} (standalone)",
            divisor=LieGroupStructure.E8_COXETER, base="q",
            applications=["Pure Coxeter corrections", "E₈ periodicity"],
            status="RESERVED"
        )
        
        # Level 43: q/16 (2⁴)
        self.factors[43] = CorrectionFactor(
            level=43, name="Binary Fourth Power",
            magnitude=0.171, formula="q/16",
            origin="2⁴ = 16 (SO(10) spinor, four-fold binary)",
            divisor=16, base="q",
            applications=["Spinor representation corrections", "SO(10) GUT", "Binary lattice"],
            status="RESERVED"
        )
        
        # Level 44: q/14 (G₂)
        self.factors[44] = CorrectionFactor(
            level=44, name="G₂ Octonion Automorphisms",
            magnitude=0.196, formula=f"q/{LieGroupStructure.G2_DIM}",
            origin=f"dim(G₂) = {LieGroupStructure.G2_DIM} (octonion automorphisms)",
            divisor=LieGroupStructure.G2_DIM, base="q",
            applications=["G₂ holonomy", "Octonion structure", "Non-associative corrections"],
            status="RESERVED"
        )
        
        # Level 45: q/9 (generations squared)
        self.factors[45] = CorrectionFactor(
            level=45, name="Generation Squared",
            magnitude=0.304, formula="q/9",
            origin="N_gen² = 3² = 9 (generation-mixing)",
            divisor=9, base="q",
            applications=["Inter-generation mixing", "Flavor sector", "CKM denominators"],
            status="RESERVED"
        )
        
        # Level 46: q/2π (half-loop)
        self.factors[46] = CorrectionFactor(
            level=46, name="Half-Loop Integral",
            magnitude=0.436, formula="q/(2π)",
            origin="Semi-circular integral (2π)",
            divisor=2 * PI, base="q",
            applications=["Half-loop contributions", "Incomplete radiative corrections"],
            status="RESERVED"
        )
        
        # Level 47: q/3 (single generation)
        self.factors[47] = CorrectionFactor(
            level=47, name="Single Generation",
            magnitude=0.913, formula="q/3",
            origin="N_gen = 3 (generation number)",
            divisor=3, base="q",
            applications=["Single-generation observables", "Per-generation corrections"],
            status="RESERVED"
        )
        
        # Level 48: q/φ³ (third golden power)
        self.factors[48] = CorrectionFactor(
            level=48, name="Third Golden Power",
            magnitude=0.65, formula="q/φ³",
            origin="φ³ ≈ 4.236 (third recursion layer)",
            divisor=PHI ** 3, base="q",
            applications=["Third-generation effects", "Deep recursion", "J_CP correction"],
            status="USED"
        )
        
        # Level 49: q/π (fundamental loop)
        self.factors[49] = CorrectionFactor(
            level=49, name="Fundamental Loop",
            magnitude=0.872, formula="q/π",
            origin="Single circular integral (π)",
            divisor=PI, base="q",
            applications=["Fundamental loop geometry", "Basic radiative structure"],
            status="RESERVED"
        )
        
        # Level 50: q²φ (quadratic + golden)
        self.factors[50] = CorrectionFactor(
            level=50, name="Mixed Second-Order Enhancement",
            magnitude=0.121, formula="q²φ",
            origin="Quadratic syntony + single golden enhancement",
            divisor=1, base="q²φ",
            applications=["Two-loop with recursion layer", "Enhanced radiative"],
            status="RESERVED"
        )
        
        # Level 51: q³ (third-order vacuum)
        self.factors[51] = CorrectionFactor(
            level=51, name="Third-Order Vacuum",
            magnitude=0.00206, formula="q³",
            origin="Three-loop universal correction",
            divisor=1, base="q³",
            applications=["Three-loop QCD", "Ultra-precision", "α_s high-order"],
            status="RESERVED"
        )
        
        # Level 52: q²/φ² (deep massless second-order)
        self.factors[52] = CorrectionFactor(
            level=52, name="Deep Massless Second-Order",
            magnitude=0.0287, formula="q²/φ²",
            origin="Quadratic syntony with double golden suppression",
            divisor=PHI ** 2, base="q²",
            applications=["Ultra-light particles", "Deep massless sector precision"],
            status="RESERVED"
        )
        
        # Level 53: 3q (triple generation)
        self.factors[53] = CorrectionFactor(
            level=53, name="Triple Generation Enhancement",
            magnitude=8.22, formula="3q",
            origin="N_gen × q = 3q (all-generation collective)",
            divisor=1, base="3q",
            applications=["All-generation effects", "Triple-flavor sum rules"],
            status="RESERVED"
        )
        
        # Level 54: πq (circular enhancement)
        self.factors[54] = CorrectionFactor(
            level=54, name="Circular Enhancement",
            magnitude=8.61, formula="πq",
            origin="π × q (complete loop enhancement)",
            divisor=1, base="πq",
            applications=["Complete loop enhancement", "Circular topology"],
            status="RESERVED"
        )
        
        # Level 55: qφ³ (triple golden)
        self.factors[55] = CorrectionFactor(
            level=55, name="Triple Golden Enhancement",
            magnitude=11.6, formula="qφ³",
            origin="φ³ ≈ 4.236 (three-layer transitions)",
            divisor=1, base="qφ³",
            applications=["Three-layer transitions", "Triple-generation cumulative", "J_CP"],
            status="USED"
        )
        
        # Level 56: 6q (E₆ Cartan enhancement)
        self.factors[56] = CorrectionFactor(
            level=56, name="E₆ Cartan Enhancement",
            magnitude=16.4, formula="6q",
            origin="rank(E₆) × q = 6q (full E₆ Cartan structure)",
            divisor=1, base="6q",
            applications=["Full E₆ Cartan structure", "Six-fold enhancement"],
            status="RESERVED"
        )
        
        # Level 57: qφ⁴ (fourth golden enhancement)
        self.factors[57] = CorrectionFactor(
            level=57, name="Fourth Golden Enhancement",
            magnitude=18.8, formula="qφ⁴",
            origin="φ⁴ ≈ 6.854 (four-layer transitions)",
            divisor=1, base="qφ⁴",
            applications=["Four-layer transitions", "Fourth-generation physics"],
            status="NEW"
        )
        
        # Level 58: 8q (E₈ Cartan enhancement)
        self.factors[58] = CorrectionFactor(
            level=58, name="E₈ Cartan Enhancement",
            magnitude=21.9, formula="8q",
            origin="rank(E₈) × q = 8q (full E₈ Cartan structure)",
            divisor=1, base="8q",
            applications=["Full E₈ Cartan structure", "Eight-fold enhancement"],
            status="RESERVED"
        )
        
        # Level 59: qφ⁵ (fifth golden enhancement)
        self.factors[59] = CorrectionFactor(
            level=59, name="Fifth Golden Enhancement",
            magnitude=30.4, formula="qφ⁵",
            origin="φ⁵ ≈ 11.09 (five-layer transitions)",
            divisor=1, base="qφ⁵",
            applications=["Five-layer transitions", "Deep recursion"],
            status="NEW"
        )
        
        # ====== MULTIPLICATIVE SUPPRESSION FACTORS (Levels 100+) ======
        
        # Level 100: 1/(1+qφ⁻²)
        self.factors[100] = CorrectionFactor(
            level=100, name="Double Inverse Recursion Suppression",
            magnitude=1.05, formula="1/(1+qφ⁻²)",
            origin="Double backward recursion penalty (φ⁻² ≈ 0.382)",
            divisor=1 + Q * (PHI ** -2), base="1/(1+...)",
            applications=["Deep winding instability", "Doubly-backward transitions"],
            status="RESERVED"
        )
        
        # Level 101: 1/(1+qφ⁻¹)
        self.factors[101] = CorrectionFactor(
            level=101, name="Inverse Recursion Suppression",
            magnitude=1.7, formula="1/(1+qφ⁻¹)",
            origin="Winding instability penalty (φ⁻¹ ≈ 0.618)",
            divisor=1 + Q / PHI, base="1/(1+...)",
            applications=["Neutron lifetime τ_n", "Unstable hadron decays"],
            status="USED"
        )
        
        # Level 102: 1/(1+q)
        self.factors[102] = CorrectionFactor(
            level=102, name="Base Suppression",
            magnitude=2.7, formula="1/(1+q)",
            origin="Universal vacuum penalty",
            divisor=1 + Q, base="1/(1+...)",
            applications=["Universal suppression baseline", "Vacuum structure penalty"],
            status="RESERVED"
        )
        
        # Level 103: 1/(1+qφ)
        self.factors[103] = CorrectionFactor(
            level=103, name="Recursion Penalty",
            magnitude=4.2, formula="1/(1+qφ)",
            origin="Double recursion-layer crossing penalty",
            divisor=1 + Q * PHI, base="1/(1+...)",
            applications=["θ₁₃ reactor angle", "Double-generation transitions"],
            status="USED"
        )
        
        # Level 104: 1/(1+qφ²)
        self.factors[104] = CorrectionFactor(
            level=104, name="Fixed Point Penalty",
            magnitude=6.7, formula="1/(1+qφ²)",
            origin="Triple layer crossing with fixed-point structure",
            divisor=1 + Q * (PHI ** 2), base="1/(1+...)",
            applications=["Triple layer crossings", "Fixed-point transition penalties", "ρ_Λ"],
            status="USED"
        )
        
        # Level 105: 1/(1+qφ³)
        self.factors[105] = CorrectionFactor(
            level=105, name="Deep Recursion Penalty",
            magnitude=10.4, formula="1/(1+qφ³)",
            origin="Four-layer crossing penalty",
            divisor=1 + Q * (PHI ** 3), base="1/(1+...)",
            applications=["Four-layer crossings", "Ultra-heavy transitions"],
            status="RESERVED"
        )
        
        # Level 106: E₇ fundamental (56)
        self.factors[106] = CorrectionFactor(
            level=106, name="E₇ Fundamental Representation",
            magnitude=0.0489, formula="q/56",
            origin="dim(E₇ fund) = 56 (matter multiplet)",
            divisor=56, base="q",
            applications=["E₇ fundamental representation", "Matter multiplet corrections"],
            status="RESERVED"
        )
    
    def get_factor(self, level: int) -> CorrectionFactor:
        """Get correction factor by level"""
        return self.factors.get(level)
    
    def get_factor_by_divisor(self, divisor: float, tolerance: float = 0.01) -> CorrectionFactor:
        """Find correction factor by approximate divisor"""
        for factor in self.factors.values():
            if abs(factor.divisor - divisor) / divisor < tolerance:
                return factor
        return None
    
    def list_factors(self, status: str = None) -> list:
        """List all factors, optionally filtered by status"""
        factors = list(self.factors.values())
        if status:
            factors = [f for f in factors if f.status == status]
        return sorted(factors, key=lambda x: x.level)
    
    def get_applicable_factors(self, observable_type: str) -> list:
        """Get correction factors applicable to a specific observable type"""
        applicable = []
        for factor in self.factors.values():
            if observable_type in factor.applications or not factor.applications:
                applicable.append(factor)
        return sorted(applicable, key=lambda x: x.level)


# ============================================================================
# CORRECTION SELECTORS BY OBSERVABLE TYPE
# ============================================================================

class CorrectionSelector:
    """Smart selection of appropriate corrections for different observable types
    
    Based on validation table from Universal_Syntony_Correction_Hierarchy.md (v1.2)
    Maps observables to their documented correction factors achieving EXACT agreement.
    """
    
    def __init__(self, hierarchy: CorrectionHierarchy = None):
        self.hierarchy = hierarchy or CorrectionHierarchy()
    
    def get_corrections_for_particle_mass(self, particle_name: str) -> list:
        """Select appropriate corrections for a particle mass prediction
        
        Returns list of (factor, sign) tuples where:
        - factor: CorrectionFactor object
        - sign: +1 for (1+correction), -1 for (1-correction), 0 for divisor (1/(1+...))
        """
        
        # Map particle names to their correction factors with signs
        # Format: [(level, sign), ...] where sign: +1=add, -1=subtract, 0=divide
        corrections_map = {
            # ===== NUCLEONS =====
            "Proton": [(1, +1)],  # q/1000 (fixed-point)
            "m_p": [(1, +1)],
            "Neutron": [(2, +1)],  # q/720 (Coxeter-Kissing)
            "m_n": [(2, +1)],
            "Δm_np": [(24, +1), (15, +1), (3, +1)],  # q/6, q/36, q/360
            
            # ===== MESONS =====
            "Pion": [(22, +1), (9, +1)],  # q/8, q²/φ
            "m_π": [(22, +1), (9, +1)],
            "Kaon": [(24, -1), (8, +1)],  # q/6, q/120
            "m_K": [(24, -1), (8, +1)],
            "m_η": [(28, -1), (15, +1)],  # q/2, q/36
            "Eta": [(28, -1), (15, +1)],
            "m_ρ": [(26, +1), (16, +1), (1, +1)],  # q/4, q/27, q/1000
            "m_ω": [(26, +1), (16, +1), (1, +1)],
            "Rho": [(26, +1), (16, +1), (1, +1)],
            "Omega_meson": [(26, +1), (16, +1), (1, +1)],
            
            # ===== ELECTROWEAK BOSONS =====
            "W Boson": [(19, +1), (4, +1)],  # q/4π, q/248
            "m_W": [(19, +1), (4, +1)],
            "Z Boson": [(19, -1), (4, +1)],  # q/4π, q/248
            "m_Z": [(19, -1), (4, +1)],
            "sin²θ_W": [(4, +1)],  # q/248
            
            # ===== QUARKS =====
            "Top": [(19, +1), (8, +1)],  # q/4π, q/120
            "m_t": [(19, +1), (8, +1)],
            "Charm": [(8, +1)],  # q/120
            "m_c": [(8, +1)],
            "Bottom": [(4, +1)],  # q/248
            "m_b": [(4, +1)],
            
            # ===== LEPTONS =====
            "Tau": [(18, -1), (2, -1)],  # q/5π, q/720
            "m_τ": [(18, -1), (2, -1)],
            "Muon": [(18, -1), (22, +1)],  # q/5π, q/8 (for g-2)
            "Electron": [(22, +1)],  # q/8
            
            # ===== PRECISION OBSERVABLES =====
            "a_μ": [(22, +1), (9, +1)],  # q/8, q²/φ (muon g-2)
            "α⁻¹": [(9, +1)],  # q²/φ (fine structure constant)
            
            # ===== QCD =====
            "α_s": [],  # Tree-level EXACT
            "Λ_QCD": [(29, -1), (17, -1)],  # q/φ, q/6π
            
            # ===== DEFAULT =====
            "default": [(30, +1)],  # q (universal vacuum)
        }
        
        # Get factor specs for this particle
        factor_specs = corrections_map.get(particle_name, corrections_map.get("default"))
        
        # Convert to (factor, sign) tuples
        result = []
        for level, sign in factor_specs:
            factor = self.hierarchy.get_factor(level)
            if factor:
                result.append((factor, sign))
        
        return result
    
    def get_corrections_for_mixing_angle(self, angle_name: str) -> list:
        """Select corrections for neutrino/quark mixing angles
        
        Returns list of (factor, sign) tuples.
        """
        
        angle_map = {
            # ===== CKM MIXING =====
            "sin θ_C": [(26, +1), (8, +1)],  # q/4, q/120 (Cabibbo)
            "θ_C": [(26, +1), (8, +1)],
            "Cabibbo": [(26, +1), (8, +1)],
            
            # ===== PMNS MIXING =====
            "θ₁₂": [(28, +1), (16, +1)],  # q/2, q/27 (solar)
            "θ₂₃": [(22, +1), (15, +1), (8, +1)],  # q/8, q/36, q/120 (atmospheric)
            "θ₁₃": [(103, 0), (22, +1), (20, +1)],  # 1/(1+qφ), q/8, q/12 (reactor)
            
            # ===== CP VIOLATION =====
            "δ_CP": [(33, -1), (29, +1)],  # (1-4q), q/φ (Dirac phase)
            "J_CP": [(33, -1), (32, -1), (48, -1)],  # (1-4q), qφ², q/φ³ (Jarlskog)
            "η_B": [(33, -1), (26, +1)],  # (1-4q), q/4 (baryon asymmetry)
        }
        
        factor_specs = angle_map.get(angle_name, [(30, +1)])
        
        result = []
        for level, sign in factor_specs:
            factor = self.hierarchy.get_factor(level)
            if factor:
                result.append((factor, sign))
        
        return result
    
    def get_corrections_for_cosmology(self, observable_name: str) -> list:
        """Select corrections for cosmological observables
        
        Returns list of (factor, sign) tuples.
        """
        
        cosmo_map = {
            # ===== COSMOLOGY =====
            "ρ_Λ": [(104, 0), (28, 0)],  # (1-qφ²), (1-q/2) - COSMOLOGICAL CONSTANT!
            "H₀": [],  # Tree-level EXACT
            "n_s": [],  # Tree-level EXACT
            "N_eff": [(9, -1)],  # q²/φ (neutrino species)
            "Y_p": [],  # Tree-level EXACT
            "D/H": [],  # Tree-level EXACT
            "⁷Li/H": [(31, -1), (30, -1), (29, -1)],  # Special: 7/E*, qφ, q, q/φ
            
            # ===== CMB PEAKS =====
            "ℓ₁": [],  # Tree-level EXACT (220.0)
            "ℓ₂": [(9, +1), (4, +1)],  # q²/φ, q/248
            "ℓ₃": [(9, +1), (4, +1)],  # q²/φ, q/248
            "ℓ₄": [(9, +1), (4, +1)],  # q²/φ, q/248
            "ℓ₅": [(9, +1), (4, +1)],  # q²/φ, q/248
            "CMB Peak": [(9, +1), (4, +1)],  # Generic CMB peak
            
            # ===== REDSHIFTS =====
            "z_eq": [(30, +1)],  # Matter-radiation equality
            "z_rec": [],  # Recombination (tree-level from E*×F₁₀)
        }
        
        factor_specs = cosmo_map.get(observable_name, [(30, +1)])
        
        result = []
        for level, sign in factor_specs:
            factor = self.hierarchy.get_factor(level)
            if factor:
                result.append((factor, sign))
        
        return result
    
    def get_corrections_for_decay(self, process_name: str) -> list:
        """Select corrections for decay processes
        
        Returns list of (factor, sign) tuples.
        """
        
        decay_map = {
            # ===== LIFETIMES =====
            "τ_n": [(101, 0), (10, +1)],  # 1/(1+qφ⁻¹), q/78 (neutron lifetime)
            "Neutron lifetime": [(101, 0), (10, +1)],
            
            # ===== DECAY WIDTHS =====
            "Γ_Z": [(30, +1), (16, -1)],  # m_Z×q×(1-q/27)
            "Γ_W": [(30, +1), (30, -1)],  # m_W×q×(1-2q)
        }
        
        factor_specs = decay_map.get(process_name, [(30, +1)])
        
        result = []
        for level, sign in factor_specs:
            factor = self.hierarchy.get_factor(level)
            if factor:
                result.append((factor, sign))
        
        return result
    
    def get_all_corrections(self, observable_name: str, category: str = "auto") -> list:
        """Get corrections for any observable, auto-detecting category if needed
        
        Args:
            observable_name: Name of the observable
            category: One of "mass", "angle", "cosmology", "decay", or "auto"
        
        Returns:
            List of (factor, sign) tuples
        """
        
        if category == "auto":
            # Auto-detect category from observable name
            if any(x in observable_name.lower() for x in ["m_", "mass", "mev", "gev"]):
                category = "mass"
            elif any(x in observable_name.lower() for x in ["θ", "theta", "angle", "δ_cp", "j_cp"]):
                category = "angle"
            elif any(x in observable_name.lower() for x in ["h₀", "ℓ", "z_", "ρ_λ", "n_eff"]):
                category = "cosmology"
            elif any(x in observable_name.lower() for x in ["τ", "tau", "lifetime", "γ", "gamma", "width"]):
                category = "decay"
            else:
                category = "mass"  # Default
        
        if category == "mass":
            return self.get_corrections_for_particle_mass(observable_name)
        elif category == "angle":
            return self.get_corrections_for_mixing_angle(observable_name)
        elif category == "cosmology":
            return self.get_corrections_for_cosmology(observable_name)
        elif category == "decay":
            return self.get_corrections_for_decay(observable_name)
        else:
            return [(self.hierarchy.get_factor(30), +1)]  # Default: q


# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================

def get_hierarchy() -> CorrectionHierarchy:
    """Get the global correction hierarchy instance"""
    global _HIERARCHY
    if '_HIERARCHY' not in globals():
        _HIERARCHY = CorrectionHierarchy()
    return _HIERARCHY


def get_correction_value(level: int, q: float = Q, phi: float = PHI) -> float:
    """Get numeric value of a correction factor"""
    hierarchy = get_hierarchy()
    factor = hierarchy.get_factor(level)
    return factor.compute(q, phi) if factor else None


def print_hierarchy_summary(status: str = None):
    """Print a summary of the correction hierarchy"""
    hierarchy = get_hierarchy()
    factors = hierarchy.list_factors(status=status)
    
    print("\n" + "=" * 100)
    print(f"SYNTONY CORRECTION HIERARCHY ({len(factors)} factors)")
    print("=" * 100)
    print(f"{'Level':<6} {'Name':<30} {'Formula':<20} {'Magnitude':<12} {'Origin':<25}")
    print("-" * 100)
    
    for factor in factors:
        mag_str = f"{factor.magnitude:.4f}%" if factor.magnitude > 0 else "Exact"
        print(f"{factor.level:<6} {factor.name:<30} {factor.formula:<20} {mag_str:<12} {factor.origin:<25}")
    
    print("=" * 100 + "\n")


if __name__ == "__main__":
    # Example: Print full hierarchy
    print_hierarchy_summary()
    
    # Example: Get specific correction
    hierarchy = get_hierarchy()
    factor = hierarchy.get_factor(22)
    print(f"\nFactor {factor.level}: {factor.name}")
    print(f"  Formula: {factor.formula}")
    print(f"  Magnitude: {factor.magnitude:.4f}%")
    print(f"  Computed: {factor.compute():.10f}")
    print(f"  Origin: {factor.origin}")
    print(f"  Applications: {', '.join(factor.applications)}")
