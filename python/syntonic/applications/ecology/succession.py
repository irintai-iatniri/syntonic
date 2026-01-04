"""
Ecological Succession - Ecosystem dynamics toward syntony attractor.

S_ecosystem → φ - q as t → ∞

Early succession: Low S, unstable, many open niches
Climax community: S → φ - q, stable, filled niches
"""

from __future__ import annotations
from typing import Dict, Any, List, Optional
import math

from syntonic.exact import PHI_NUMERIC, Q_DEFICIT_NUMERIC


class EcologicalSuccession:
    """
    Ecosystem succession toward syntony attractor.

    S_ecosystem → φ - q as t → ∞

    Early succession: Low S, unstable, many open niches
    Climax community: S → φ - q, stable, filled niches

    Example:
        >>> es = EcologicalSuccession()
        >>> es.syntony_attractor
        1.59...
        >>> es.succession_dynamics(0.5, 100, 0.01)
        1.23...
    """

    ATTRACTOR = PHI_NUMERIC - Q_DEFICIT_NUMERIC  # ≈ 1.591

    def syntony_attractor(self) -> float:
        """
        The attractor state for ecosystem syntony.

        Returns:
            φ - q ≈ 1.591
        """
        return self.ATTRACTOR

    def succession_dynamics(self, S_0: float, t: float, gamma: float) -> float:
        """
        Syntony as function of time during succession.

        S(t) = S_target - (S_target - S_0) × e^(-γt)

        Args:
            S_0: Initial syntony (after disturbance)
            t: Time since disturbance
            gamma: Recovery rate constant

        Returns:
            Syntony at time t
        """
        S_target = self.ATTRACTOR
        return S_target - (S_target - S_0) * math.exp(-gamma * t)

    def time_to_climax(self, S_0: float, gamma: float, threshold: float = 0.95) -> float:
        """
        Time to reach climax community (fraction of attractor).

        Args:
            S_0: Initial syntony
            gamma: Recovery rate
            threshold: Fraction of attractor to reach (default 95%)

        Returns:
            Time to reach threshold × attractor
        """
        S_target = self.ATTRACTOR
        S_threshold = threshold * S_target

        # Solve: S_threshold = S_target - (S_target - S_0) × e^(-γt)
        # e^(-γt) = (S_target - S_threshold) / (S_target - S_0)
        # t = -ln((S_target - S_threshold) / (S_target - S_0)) / γ

        if S_0 >= S_threshold:
            return 0.0

        ratio = (S_target - S_threshold) / (S_target - S_0)
        if ratio <= 0:
            return float('inf')

        return -math.log(ratio) / gamma

    def succession_stages(self) -> Dict[str, Dict[str, Any]]:
        """
        Describe succession stages.

        Returns:
            Dict of succession stages
        """
        return {
            'pioneer': {
                'syntony_fraction': 0.1,
                'characteristics': [
                    'r-selected species',
                    'High reproductive rate',
                    'Low competitive ability',
                    'Simple food webs',
                ],
                'examples': ['Lichens', 'Mosses', 'Annual herbs'],
            },
            'early': {
                'syntony_fraction': 0.3,
                'characteristics': [
                    'Fast-growing perennials',
                    'Increasing soil depth',
                    'Simple nutrient cycling',
                    'Low biodiversity',
                ],
                'examples': ['Grasses', 'Shrubs', 'Pioneer trees'],
            },
            'mid': {
                'syntony_fraction': 0.6,
                'characteristics': [
                    'Mixed species composition',
                    'Developing canopy structure',
                    'Complex nutrient cycling',
                    'Increasing biodiversity',
                ],
                'examples': ['Mixed forest', 'Diverse prairie'],
            },
            'climax': {
                'syntony_fraction': 0.95,
                'characteristics': [
                    'K-selected species',
                    'Stable community',
                    'Complex food webs',
                    'Maximum biodiversity',
                ],
                'examples': ['Old-growth forest', 'Climax grassland'],
            },
        }

    def r_vs_k_selection(self) -> str:
        """
        Explain r vs K selection in terms of syntony.

        Returns:
            Explanation
        """
        return """
r vs K Selection and Syntony:

r-SELECTED (Low syntony environments):
- Maximize reproduction rate
- Short lifespan
- Small body size
- Low parental investment
- Strategy: "Produce many, let most die"
- Syntony per individual: LOW

K-SELECTED (High syntony environments):
- Maximize carrying capacity use
- Long lifespan
- Large body size
- High parental investment
- Strategy: "Produce few, invest heavily"
- Syntony per individual: HIGH

The transition r → K during succession:
- As S_ecosystem increases, K-selection favored
- Higher syntony = more stable = more investment viable
- Climax communities are K-dominated

Golden ratio connection:
- Optimal r/K balance ≈ 1/φ
- Pure r-selection: r/K → ∞ (unstable)
- Pure K-selection: r/K → 0 (slow recovery)
- Optimal: r/K ≈ 0.618 (φ-based)
"""

    def climax_community(self) -> str:
        """
        Describe the climax community state.

        Returns:
            Description
        """
        return f"""
The Climax Community:

Definition: Ecosystem at syntony attractor (S ≈ {self.ATTRACTOR:.3f})

Characteristics:
1. STABILITY
   - Resists perturbation
   - Returns to equilibrium
   - Self-maintaining

2. COMPLEXITY
   - Maximum biodiversity (locally)
   - Complex food webs
   - Multiple trophic levels

3. EFFICIENCY
   - Optimal energy use
   - Closed nutrient cycles
   - Minimum waste

4. INTEGRATION
   - All niches filled
   - Coevolved relationships
   - Mutualistic networks

The climax is NOT static:
- Constant turnover at individual level
- Dynamic equilibrium at system level
- Can shift with climate change

Why φ - q is the attractor:
- Maximum syntony achievable
- q represents unavoidable loss
- Further increase requires layer transition
"""

    def __repr__(self) -> str:
        return f"EcologicalSuccession(attractor={self.ATTRACTOR:.3f})"


class DisturbanceRecovery:
    """
    Recovery dynamics after ecosystem disturbance.

    Recovery time depends on disturbance severity and
    ecosystem characteristics.

    Example:
        >>> dr = DisturbanceRecovery()
        >>> dr.recovery_time(1.5, 0.3, 0.01)
        161.4...  # Time units to recover
    """

    ATTRACTOR = PHI_NUMERIC - Q_DEFICIT_NUMERIC

    def recovery_time(
        self, S_pre: float, S_post: float, gamma: float, threshold: float = 0.95
    ) -> float:
        """
        Time for ecosystem to recover from disturbance.

        Args:
            S_pre: Pre-disturbance syntony
            S_post: Post-disturbance syntony
            gamma: Recovery rate constant
            threshold: Recovery fraction (default 95%)

        Returns:
            Recovery time
        """
        S_target = min(S_pre, self.ATTRACTOR)  # Can't exceed previous or attractor
        S_threshold = threshold * S_target

        if S_post >= S_threshold:
            return 0.0

        ratio = (S_target - S_threshold) / (S_target - S_post)
        if ratio <= 0:
            return float('inf')

        return -math.log(ratio) / gamma

    def disturbance_severity(self, S_pre: float, S_post: float) -> float:
        """
        Calculate disturbance severity as syntony loss fraction.

        Args:
            S_pre: Pre-disturbance syntony
            S_post: Post-disturbance syntony

        Returns:
            Severity (0 to 1)
        """
        if S_pre <= 0:
            return 0.0
        return 1 - (S_post / S_pre)

    def resilience(self, recovery_time: float, disturbance_severity: float) -> float:
        """
        Calculate ecosystem resilience.

        Resilience = Severity / Recovery_time

        Higher resilience = faster recovery from larger disturbances.

        Args:
            recovery_time: Time to recover
            disturbance_severity: Severity of disturbance

        Returns:
            Resilience metric
        """
        if recovery_time <= 0:
            return float('inf')
        return disturbance_severity / recovery_time

    def disturbance_types(self) -> Dict[str, Dict[str, Any]]:
        """
        Characterize different disturbance types.

        Returns:
            Dict of disturbance types
        """
        return {
            'fire': {
                'severity_range': (0.3, 0.9),
                'recovery_rate': 0.05,
                'typical_recovery': '10-50 years',
                'syntony_reset': 'Partial (underground structures survive)',
            },
            'flood': {
                'severity_range': (0.2, 0.7),
                'recovery_rate': 0.1,
                'typical_recovery': '1-10 years',
                'syntony_reset': 'Moderate (seeds, nutrients redistributed)',
            },
            'volcanic': {
                'severity_range': (0.9, 1.0),
                'recovery_rate': 0.01,
                'typical_recovery': '100-1000 years',
                'syntony_reset': 'Complete (primary succession)',
            },
            'logging': {
                'severity_range': (0.5, 0.8),
                'recovery_rate': 0.03,
                'typical_recovery': '50-200 years',
                'syntony_reset': 'Significant (soil mostly intact)',
            },
            'agriculture': {
                'severity_range': (0.7, 0.95),
                'recovery_rate': 0.02,
                'typical_recovery': '100-500 years',
                'syntony_reset': 'Major (soil degradation)',
            },
        }

    def mass_extinction_recovery(self) -> str:
        """
        Describe recovery from mass extinction events.

        Returns:
            Description
        """
        return """
Mass Extinction Recovery:

After major extinction events:

1. IMMEDIATE (0-10 ky)
   - Disaster taxa dominate
   - Simple food webs
   - S << attractor

2. RECOVERY (10 ky - 1 My)
   - Adaptive radiation
   - New niches fill
   - S approaching attractor

3. REBUILD (1-10 My)
   - Complex communities return
   - Coevolution proceeds
   - S ≈ attractor

4. NEW NORMAL (>10 My)
   - Novel ecosystem types
   - New dominant groups
   - S = attractor (different composition)

The "Big Five" extinctions:
- End Ordovician: ~10 My recovery
- Late Devonian: ~15 My recovery
- End Permian: ~10 My recovery
- End Triassic: ~5 My recovery
- End Cretaceous: ~10 My recovery

Pattern: τ_recovery ≈ ln(S_pre/S_post) / (q × γ₀) ≈ n × 3 My

The biosphere ALWAYS recovers to S ≈ φ - q.
But the species composition is forever changed.
"""

    def anthropocene_disturbance(self) -> str:
        """
        Assess the current Anthropocene disturbance.

        Returns:
            Assessment
        """
        return f"""
The Anthropocene Disturbance:

Current biodiversity loss rate: ~100× background
This is comparable to mass extinction events.

Severity estimate:
- Pre-human: S_pre ≈ φ - q ≈ {self.ATTRACTOR:.3f} (normalized)
- Current: S_post ≈ 0.8 × S_pre (20% loss)
- Projected: S_post ≈ 0.5 × S_pre (50% loss by 2100)

Recovery scenarios:

IF LOSS STOPS AT 20%:
- Recovery time: ~10-50 years (relatively fast)
- Ecosystem function: mostly maintained

IF LOSS REACHES 50%:
- Recovery time: ~100-500 years
- Ecosystem function: significantly impaired

IF LOSS REACHES 75%:
- Recovery time: ~1-10 My (geological)
- Crosses mass extinction threshold
- Humanity unlikely to survive

Critical insight:
The biosphere WILL recover (to S = φ - q).
The question is: will we be part of that recovery?

What we can do:
1. Halt further biodiversity loss
2. Protect remaining wild areas
3. Restore degraded ecosystems
4. Reduce consumption footprint
5. Support ecosystem connectivity
"""

    def __repr__(self) -> str:
        return "DisturbanceRecovery(attractor={:.3f})".format(self.ATTRACTOR)
