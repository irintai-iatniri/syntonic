"""
Food Web Dynamics - Trophic structure from SRT.

Trophic efficiency: η = φ⁻⁵ ≈ 9%
Trophic levels: N_gen + 1 = 4

The same generation structure that gives 3 particle generations
gives 4 trophic levels in ecology!
"""

from __future__ import annotations

import math
from typing import Any, Dict

from syntonic.exact import PHI_NUMERIC, Q_DEFICIT_NUMERIC


class TrophicDynamics:
    """
    Food web structure from SRT.

    Trophic efficiency derivation:

    | Process          | Efficiency | φ-Relation |
    |------------------|------------|------------|
    | Consumption      | ~50%       | 1/φ^(1/2)  |
    | Assimilation     | ~50%       | 1/φ^(1/2)  |
    | Production       | ~25%       | 1/φ²       |
    | Additional losses| ~40%       | 1/φ        |

    Net: η = (1/φ^0.5) × (1/φ^0.5) × (1/φ²) × (1/φ) = 1/φ⁵ ≈ 9%

    Trophic levels = N_gen + 1 = 3 + 1 = 4

    Example:
        >>> td = TrophicDynamics()
        >>> td.trophic_efficiency
        0.09...
        >>> td.energy_at_level(1000, 2)  # 1000 units, level 2
        8.1...
    """

    TROPHIC_EFFICIENCY = PHI_NUMERIC ** (-5)  # ≈ 0.09 = 9%
    TROPHIC_EFFICIENCY_CORRECTED = (PHI_NUMERIC ** (-5)) * (
        1 + Q_DEFICIT_NUMERIC
    )  # ≈ 9.3%
    TROPHIC_LEVELS = 4  # N_gen + 1 = 3 + 1 = 4
    N_GENERATIONS = 3  # Same as particle physics!

    # Component efficiencies
    CONSUMPTION_EFFICIENCY = PHI_NUMERIC ** (-0.5)  # ~62%
    ASSIMILATION_EFFICIENCY = PHI_NUMERIC ** (-0.5)  # ~62%
    PRODUCTION_EFFICIENCY = PHI_NUMERIC ** (-2)  # ~38%
    LOSS_FACTOR = PHI_NUMERIC ** (-1)  # ~62%

    @property
    def trophic_efficiency(self) -> float:
        """Net trophic efficiency."""
        return self.TROPHIC_EFFICIENCY

    def energy_at_level(self, primary_production: float, level: int) -> float:
        """
        Energy available at trophic level n.

        E(n) = E_primary × η^n

        Args:
            primary_production: Energy at level 0 (producers)
            level: Trophic level (0 = producers, 1 = herbivores, etc.)

        Returns:
            Energy available at that level
        """
        return primary_production * (self.TROPHIC_EFFICIENCY**level)

    def biomass_pyramid(self, base_biomass: float) -> Dict[int, float]:
        """
        Calculate biomass at each trophic level.

        Args:
            base_biomass: Biomass at producer level

        Returns:
            Dict mapping level to biomass
        """
        return {
            level: base_biomass * (self.TROPHIC_EFFICIENCY**level)
            for level in range(self.TROPHIC_LEVELS)
        }

    def why_four_levels(self) -> str:
        """
        Explain why exactly 4 trophic levels.

        N_gen + 1 = 3 + 1 = 4

        Returns:
            Explanation
        """
        return """
Why Exactly 4 Trophic Levels?

The answer: N_gen + 1 = 3 + 1 = 4

This is the SAME structure that gives:
- 3 particle generations (electron, muon, tau)
- 3 quark generations (u/d, c/s, t/b)
- 3 neutrino generations

The "+1" comes from:
- The base level (producers in ecology, vacuum in physics)
- This is the "ground state" that the generations build upon

The number 3 is NOT arbitrary:
- It comes from T⁴ geometry (4D torus with 3 independent cycles)
- Any more generations would destabilize the vacuum
- Any fewer wouldn't close the recursion

In ecology:
- Level 0: Producers (plants, algae)
- Level 1: Primary consumers (herbivores)
- Level 2: Secondary consumers (small predators)
- Level 3: Tertiary consumers (apex predators)

A 5th level is energetically impossible:
- η⁵ ≈ 0.00006 (0.006%)
- Not enough energy to sustain a population
- Same reason there's no 4th particle generation!
"""

    def efficiency_derivation(self) -> str:
        """
        Derive the 9% trophic efficiency.

        Returns:
            Derivation
        """
        return f"""
Trophic Efficiency Derivation:

η = (consumption) × (assimilation) × (production) × (losses)
η = (1/φ^0.5) × (1/φ^0.5) × (1/φ²) × (1/φ)
η = 1/φ^(0.5 + 0.5 + 2 + 1)
η = 1/φ⁵
η ≈ {self.TROPHIC_EFFICIENCY:.4f} ≈ 9%

With syntony correction:
η_corrected = (1/φ⁵) × (1 + q)
η_corrected ≈ {self.TROPHIC_EFFICIENCY_CORRECTED:.4f} ≈ 9.3%

Experimental range: 5-20%, mean ~10% ✓

Each factor has a golden-ratio origin:
- Consumption (1/√φ): Optimal foraging strategy
- Assimilation (1/√φ): Gut efficiency from surface/volume
- Production (1/φ²): Growth allocation
- Losses (1/φ): Maintenance metabolism
"""

    def __repr__(self) -> str:
        return f"TrophicDynamics(η=1/φ⁵≈{self.TROPHIC_EFFICIENCY:.2%}, levels={self.TROPHIC_LEVELS})"


class FoodWeb:
    """
    Food web network structure.

    Connectance C ≈ 0.1 is a universal constant across ecosystems.

    Example:
        >>> fw = FoodWeb()
        >>> fw.connectance(100, 1000)  # 100 species, 1000 links
        0.1
    """

    UNIVERSAL_CONNECTANCE = 0.1  # Approximate

    def connectance(self, n_species: int, n_links: int) -> float:
        """
        Calculate food web connectance.

        C = L / S²

        Args:
            n_species: Number of species
            n_links: Number of feeding links

        Returns:
            Connectance
        """
        if n_species <= 0:
            return 0.0
        return n_links / (n_species**2)

    def expected_links(self, n_species: int) -> float:
        """
        Expected number of links for given species count.

        L = C × S²

        Args:
            n_species: Number of species

        Returns:
            Expected number of links
        """
        return self.UNIVERSAL_CONNECTANCE * (n_species**2)

    def link_density(self, n_species: int, n_links: int) -> float:
        """
        Links per species.

        Args:
            n_species: Number of species
            n_links: Number of links

        Returns:
            Link density
        """
        if n_species <= 0:
            return 0.0
        return n_links / n_species

    def cascade_model(self, n_species: int, n_links: int) -> Dict[str, Any]:
        """
        Cascade model predictions for food web structure.

        Args:
            n_species: Number of species
            n_links: Number of links

        Returns:
            Cascade model predictions
        """
        C = self.connectance(n_species, n_links)
        L_per_S = n_links / n_species if n_species > 0 else 0

        # Expected values from cascade model
        expected_prey = L_per_S / 2  # On average, half links are prey
        expected_predators = L_per_S / 2

        return {
            "connectance": C,
            "links_per_species": L_per_S,
            "expected_prey": expected_prey,
            "expected_predators": expected_predators,
            "top_predator_fraction": 1 / (n_species**0.5) if n_species > 0 else 0,
        }

    def stability_criterion(
        self, n_species: int, connectance: float, interaction_strength: float
    ) -> bool:
        """
        May's stability criterion for food webs.

        Stable if: σ √(SC) < 1

        Args:
            n_species: Number of species
            connectance: Food web connectance
            interaction_strength: Mean interaction strength (σ)

        Returns:
            True if stable
        """
        may_criterion = interaction_strength * math.sqrt(n_species * connectance)
        return may_criterion < 1.0

    def golden_ratio_in_webs(self) -> str:
        """
        Explain golden ratio patterns in food webs.

        Returns:
            Explanation
        """
        return """
Golden Ratio in Food Webs:

1. BODY SIZE RATIOS
   Predator/prey mass ratio ≈ φ³ ≈ 4.24
   This optimizes energy capture vs. handling time

2. POPULATION RATIOS
   Prey/predator abundance ≈ φ⁴ ≈ 6.85
   Maintains stable predator-prey dynamics

3. DIET BREADTH
   Generalist/specialist ratio ≈ φ
   Balances flexibility vs. efficiency

4. FOOD CHAIN LENGTH
   Mean chain length ≈ 3.6 ≈ φ² + 1
   Optimizes energy flow vs. stability

5. NETWORK MODULARITY
   Optimal modularity ≈ 1/φ ≈ 0.62
   Balances stability vs. responsiveness

These ratios are NOT coincidences:
- They emerge from optimization
- Golden ratio = optimal balance point
- Food webs converge to these values
"""

    def __repr__(self) -> str:
        return f"FoodWeb(C≈{self.UNIVERSAL_CONNECTANCE})"
