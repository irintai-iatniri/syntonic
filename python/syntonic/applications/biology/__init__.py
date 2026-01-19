"""
Biology - Life topology, abiogenesis, metabolism, and evolution from SRT.

Life is bidirectional M⁴ ↔ T⁴ information flow—recording AND steering.
The π threshold marks the transition from chemistry to life.

Classes:
    LifeTopology: M⁴ ↔ T⁴ bidirectionality definition of life
    TranscendenceThreshold: π, 2π, 3π phase thresholds
    GnosisLayers: Hierarchy from matter to consciousness
    KleiberLaw: BMR ∝ M^(3/4) from interface dimension
    ATPCycle: ATP as DHSR engine with η = 1/φ
    EvolutionaryDirectionality: dk/dt ≥ 0
    ProteinFolding: Levinthal resolution via φ-contraction
    DNAStructure: DNA as crystallized Tv history
    GeneticCode: 64 codons from T⁴ topology

Key Formulas:
    Life threshold: Σ Tv = π
    Kleiber exponent: 3/4 (interface/bulk dimension)
    ATP efficiency: η = 1/φ ≈ 61.8%
    Trophic levels: N_gen + 1 = 4

Example:
    >>> from syntonic.applications.biology import TranscendenceThreshold
    >>> tt = TranscendenceThreshold()
    >>> tt.check_threshold(3.5)
    {'layer': 1, 'status': 'alive', ...}
"""

from syntonic.applications.biology.abiogenesis import (
    GnosisLayers,
    TranscendenceThreshold,
)
from syntonic.applications.biology.evolution import (
    EvolutionaryDirectionality,
    ProteinFolding,
)
from syntonic.applications.biology.genetics import (
    DNAStructure,
    GeneticCode,
)
from syntonic.applications.biology.life_topology import (
    LifeTopology,
)
from syntonic.applications.biology.metabolism import (
    ATPCycle,
    KleiberLaw,
)

__all__ = [
    "LifeTopology",
    "TranscendenceThreshold",
    "GnosisLayers",
    "KleiberLaw",
    "ATPCycle",
    "EvolutionaryDirectionality",
    "ProteinFolding",
    "DNAStructure",
    "GeneticCode",
]
