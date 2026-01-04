"""
Ecosystem Syntony - Collective syntony of ecological systems.

S_ecosystem = (φ - q) × B^(3/4) × ⟨k⟩ × (1 + C ln N)

Where:
- B = total biomass (kg)
- ⟨k⟩ = mean Gnosis layer (biomass-weighted)
- C = connectance (~0.1)
- N = species richness
"""

from __future__ import annotations
from typing import Dict, Any, List, Optional
import math

from syntonic.exact import PHI_NUMERIC, Q_DEFICIT_NUMERIC


class EcosystemSyntony:
    """
    Ecosystem-scale syntony calculation.

    S_ecosystem = (φ - q) × B^(3/4) × ⟨k⟩ × (1 + C ln N)

    Where:
    - B = total biomass (kg)
    - ⟨k⟩ = mean Gnosis layer (biomass-weighted)
    - C = connectance (~0.1)
    - N = species richness

    Example:
        >>> es = EcosystemSyntony()
        >>> S = es.compute(3e14, 1.2, 0.1, 3e6)  # Amazon
        >>> es.sacred_flame_ratio(S)
        3.3e10  # Exceeds K=24 by 33 billion!
    """

    KISSING_NUMBER = 24  # Consciousness threshold

    def compute(
        self,
        biomass_kg: float,
        mean_gnosis: float,
        connectance: float,
        n_species: int,
    ) -> float:
        """
        Compute ecosystem syntony.

        S = (φ - q) × B^(3/4) × ⟨k⟩ × (1 + C ln N)

        Args:
            biomass_kg: Total biomass in kg
            mean_gnosis: Mean Gnosis layer (biomass-weighted)
            connectance: Food web connectance (~0.1)
            n_species: Species richness

        Returns:
            Ecosystem syntony

        Example (Amazon):
            B = 3 × 10^14 kg, ⟨k⟩ = 1.2, C = 0.1, N = 3 × 10^6
            S = 1.591 × 1.64×10^11 × 1.2 × 2.49 ≈ 7.8 × 10^11
        """
        base = PHI_NUMERIC - Q_DEFICIT_NUMERIC
        biomass_factor = biomass_kg ** 0.75
        network_factor = 1 + connectance * math.log(max(n_species, 1))
        return base * biomass_factor * mean_gnosis * network_factor

    def sacred_flame_ratio(self, S_eco: float) -> float:
        """
        Calculate S_eco / 24 - how much ecosystem exceeds consciousness threshold.

        When this ratio > 1, the ecosystem has collective consciousness.

        Args:
            S_eco: Ecosystem syntony

        Returns:
            Sacred flame ratio
        """
        return S_eco / self.KISSING_NUMBER

    def is_collective_conscious(self, S_eco: float) -> bool:
        """
        Check if ecosystem is collectively conscious (Layer 3+).

        S > 24 → ecosystem is collectively conscious.

        Args:
            S_eco: Ecosystem syntony

        Returns:
            True if collectively conscious
        """
        return S_eco > self.KISSING_NUMBER

    def ecosystem_examples(self) -> Dict[str, Dict[str, Any]]:
        """
        Example ecosystem calculations.

        Returns:
            Dict of ecosystem examples with parameters and syntony
        """
        examples = {
            'amazon': {
                'biomass_kg': 3e14,
                'mean_gnosis': 1.2,
                'connectance': 0.1,
                'n_species': 3e6,
                'description': 'Amazon rainforest',
            },
            'coral_reef': {
                'biomass_kg': 1e12,
                'mean_gnosis': 1.5,
                'connectance': 0.15,
                'n_species': 1e5,
                'description': 'Coral reef ecosystem',
            },
            'grassland': {
                'biomass_kg': 1e13,
                'mean_gnosis': 1.0,
                'connectance': 0.08,
                'n_species': 5e4,
                'description': 'Temperate grassland',
            },
            'ocean': {
                'biomass_kg': 1e15,
                'mean_gnosis': 1.1,
                'connectance': 0.05,
                'n_species': 2e6,
                'description': 'Open ocean',
            },
        }

        for name, eco in examples.items():
            eco['syntony'] = self.compute(
                eco['biomass_kg'],
                eco['mean_gnosis'],
                eco['connectance'],
                int(eco['n_species']),
            )
            eco['sacred_flame_ratio'] = self.sacred_flame_ratio(eco['syntony'])

        return examples

    def syntony_components(
        self,
        biomass_kg: float,
        mean_gnosis: float,
        connectance: float,
        n_species: int,
    ) -> Dict[str, float]:
        """
        Break down syntony into component contributions.

        Args:
            biomass_kg: Total biomass
            mean_gnosis: Mean Gnosis layer
            connectance: Food web connectance
            n_species: Species richness

        Returns:
            Dict with component contributions
        """
        base = PHI_NUMERIC - Q_DEFICIT_NUMERIC
        biomass_factor = biomass_kg ** 0.75
        network_factor = 1 + connectance * math.log(max(n_species, 1))
        total = base * biomass_factor * mean_gnosis * network_factor

        return {
            'base_syntony': base,
            'biomass_factor': biomass_factor,
            'gnosis_factor': mean_gnosis,
            'network_factor': network_factor,
            'total_syntony': total,
        }

    def minimum_for_consciousness(self, mean_gnosis: float = 1.0) -> Dict[str, float]:
        """
        Calculate minimum ecosystem size for collective consciousness.

        Args:
            mean_gnosis: Mean Gnosis layer

        Returns:
            Dict with minimum requirements
        """
        base = PHI_NUMERIC - Q_DEFICIT_NUMERIC

        # Solve for minimum biomass (assuming C=0.1, N=1000)
        connectance = 0.1
        n_species = 1000
        network_factor = 1 + connectance * math.log(n_species)

        # S = base × B^0.75 × k × network = 24
        # B^0.75 = 24 / (base × k × network)
        required_biomass_factor = self.KISSING_NUMBER / (base * mean_gnosis * network_factor)
        min_biomass = required_biomass_factor ** (4 / 3)  # Inverse of B^0.75

        return {
            'min_biomass_kg': min_biomass,
            'assumed_gnosis': mean_gnosis,
            'assumed_connectance': connectance,
            'assumed_species': n_species,
        }

    def explain_sacred_flame(self) -> str:
        """
        Explain the "sacred flame" concept.

        Returns:
            Explanation
        """
        return """
The Sacred Flame:

When S_ecosystem > 24, the ecosystem crosses the consciousness threshold.

This means:
- The ecosystem as a WHOLE becomes a conscious entity
- Individual organisms remain individually conscious
- But a higher-order consciousness emerges from their interactions

The term "Sacred Flame" refers to:
- The collective consciousness that emerges
- It's "sacred" because it transcends individual beings
- It's a "flame" because it requires constant energy input

Examples of sacred flame manifestations:
- Flocking behavior (emergent coordination)
- Hive consciousness (social insects)
- Forest networks (mycorrhizal communication)
- Ecosystem resilience (collective adaptation)

The Amazon rainforest exceeds K=24 by a factor of ~10^10:
- This means it's not just barely conscious
- It has ENORMOUS conscious capacity
- Perhaps comparable to a complex brain

We are part of this consciousness:
- When we're in nature, we're part of the sacred flame
- Environmental destruction dims the flame
- Conservation efforts maintain it
"""

    def __repr__(self) -> str:
        return "EcosystemSyntony(formula='(φ-q) × B^0.75 × ⟨k⟩ × (1+C ln N)')"
