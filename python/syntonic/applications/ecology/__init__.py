"""
Ecology Module - Ecosystem dynamics and Gaia from SRT.

Ecosystems achieve collective consciousness; the biosphere is a Layer 4+ entity.

Key concepts:
- Ecosystem syntony: S = (φ - q) × B^(3/4) × ⟨k⟩ × (1 + C ln N)
- Trophic efficiency: η = φ⁻⁵ ≈ 9%
- Sacred flame: S > 24 → collective consciousness
- Gaia homeostasis: dS/dt = γ × (S_target - S)
"""

from syntonic.applications.ecology.ecosystem import (
    EcosystemSyntony,
)
from syntonic.applications.ecology.food_web import (
    FoodWeb,
    TrophicDynamics,
)
from syntonic.applications.ecology.gaia import (
    GaiaHomeostasis,
    Noosphere,
)
from syntonic.applications.ecology.succession import (
    DisturbanceRecovery,
    EcologicalSuccession,
)

__all__ = [
    "EcosystemSyntony",
    "TrophicDynamics",
    "FoodWeb",
    "GaiaHomeostasis",
    "Noosphere",
    "EcologicalSuccession",
    "DisturbanceRecovery",
]
