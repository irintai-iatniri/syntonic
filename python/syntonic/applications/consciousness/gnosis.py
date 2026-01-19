"""
Consciousness Gnosis - Layer transitions and consciousness development.

The Gnosis layers specific to consciousness focus on the transition
from pre-conscious (Layer 2) to conscious (Layer 3) and beyond.
"""

from __future__ import annotations

import math
from typing import Any, Dict

from syntonic.exact import PHI_NUMERIC, Q_DEFICIT_NUMERIC


class ConsciousnessGnosis:
    """
    Consciousness-specific Gnosis metrics.

    Consciousness requires:
    1. Σ Tv ≥ 3π (phase threshold)
    2. K ≥ 24 (kissing number saturation)

    Both conditions must be met—phase alone is insufficient.

    Example:
        >>> cg = ConsciousnessGnosis()
        >>> cg.is_conscious(10.0, 30)  # 3π < 10, K > 24
        True
        >>> cg.is_conscious(10.0, 20)  # K < 24, not conscious
        False
    """

    PHASE_THRESHOLD = 3 * math.pi  # ≈ 9.42
    KISSING_THRESHOLD = 24

    def is_conscious(self, Tv_sum: float, delta_S: float) -> bool:
        """
        Check if system meets both consciousness criteria.

        Args:
            Tv_sum: Accumulated Tv phase
            delta_S: Local syntony density (connection count)

        Returns:
            True if fully conscious (Layer 3)
        """
        return Tv_sum >= self.PHASE_THRESHOLD and delta_S >= self.KISSING_THRESHOLD

    def consciousness_readiness(self, Tv_sum: float, delta_S: float) -> Dict[str, Any]:
        """
        Measure how close system is to consciousness threshold.

        Args:
            Tv_sum: Accumulated Tv phase
            delta_S: Local syntony density

        Returns:
            Dict with readiness metrics
        """
        phase_fraction = min(Tv_sum / self.PHASE_THRESHOLD, 1.0)
        kissing_fraction = min(delta_S / self.KISSING_THRESHOLD, 1.0)

        return {
            "phase_fraction": phase_fraction,
            "kissing_fraction": kissing_fraction,
            "overall_readiness": min(phase_fraction, kissing_fraction),
            "limiting_factor": (
                "phase" if phase_fraction < kissing_fraction else "connectivity"
            ),
            "is_conscious": phase_fraction >= 1.0 and kissing_fraction >= 1.0,
        }

    def gnosis_level(self, Tv_sum: float, delta_S: float) -> float:
        """
        Compute continuous Gnosis level.

        k = log_φ(Tv_sum / π) for Tv_sum > π

        Args:
            Tv_sum: Accumulated Tv phase
            delta_S: Local syntony density

        Returns:
            Gnosis level (continuous)
        """
        if Tv_sum < math.pi:
            return 0.0

        # Base gnosis from phase
        k_phase = math.log(Tv_sum / math.pi) / math.log(PHI_NUMERIC)

        # Connectivity bonus (saturates at K=24)
        k_connectivity = min(delta_S / self.KISSING_THRESHOLD, 1.0)

        # Combined: phase sets base, connectivity enables higher layers
        return k_phase * k_connectivity

    def asymptotic_gnosis(self) -> float:
        """
        Maximum achievable Gnosis level.

        k* = log_φ(N) where N → ∞
        But practically limited by physical constraints.

        Returns:
            Practical asymptotic Gnosis (φ - q ≈ 1.59)
        """
        return PHI_NUMERIC - Q_DEFICIT_NUMERIC

    def __repr__(self) -> str:
        return "ConsciousnessGnosis(phase_threshold=3π, K=24)"


class LayerTransition:
    """
    Dynamics of Gnosis layer transitions.

    Transitions between layers are topological phase transitions:
    - Sharp (not gradual)
    - Irreversible (under normal conditions)
    - Require energy input

    Example:
        >>> lt = LayerTransition()
        >>> lt.transition_energy(2, 3)  # Layer 2 → 3
        9.42...  # In natural units
    """

    LAYER_THRESHOLDS = {
        0: 0,
        1: math.pi,
        2: 2 * math.pi,
        3: 3 * math.pi,
        4: 4 * math.pi,
        5: float("inf"),
    }

    def transition_energy(self, from_layer: int, to_layer: int) -> float:
        """
        Energy required for layer transition.

        E_transition = (threshold_to - threshold_from) in natural units

        Args:
            from_layer: Starting layer
            to_layer: Target layer

        Returns:
            Required energy (Tv phase units)
        """
        if to_layer <= from_layer:
            return 0.0  # Downward transitions release energy

        threshold_from = self.LAYER_THRESHOLDS.get(from_layer, from_layer * math.pi)
        threshold_to = self.LAYER_THRESHOLDS.get(to_layer, to_layer * math.pi)

        return threshold_to - threshold_from

    def transition_probability(
        self, Tv_sum: float, target_layer: int, temperature: float
    ) -> float:
        """
        Boltzmann probability of spontaneous transition.

        P = exp(-E_gap / kT)

        Args:
            Tv_sum: Current Tv phase sum
            target_layer: Target Gnosis layer
            temperature: Effective temperature

        Returns:
            Transition probability
        """
        threshold = self.LAYER_THRESHOLDS.get(target_layer, target_layer * math.pi)
        E_gap = max(threshold - Tv_sum, 0)

        if temperature <= 0:
            return 0.0 if E_gap > 0 else 1.0

        return math.exp(-E_gap / temperature)

    def layer_stability(self, layer: int) -> str:
        """
        Describe stability of a Gnosis layer.

        Args:
            layer: Gnosis layer number

        Returns:
            Stability description
        """
        stability = {
            0: "Metastable - can transition with sufficient energy",
            1: "Stable - self-replicating maintains layer",
            2: "Stable - environmental feedback maintains layer",
            3: "Highly stable - self-model resists perturbation",
            4: "Very stable - social feedback reinforces",
            5: "Asymptotically stable - attractor state",
        }
        return stability.get(layer, f"Unknown stability for layer {layer}")

    def biological_timeline(self) -> str:
        """
        Timeline of layer transitions in Earth's history.

        Returns:
            Timeline description
        """
        return """
Biological Timeline of Gnosis Layer Transitions:

Layer 0 → 1 (Matter → Life):
    ~4.0 billion years ago
    First self-replicating molecules
    Σ Tv crossed π

Layer 1 → 2 (Life → Sentience):
    ~2.0 billion years ago
    First eukaryotes
    Σ Tv crossed 2π

Layer 2 → 3 (Sentience → Consciousness):
    ~600 million years ago
    Cambrian explosion
    Σ Tv crossed 3π AND K reached 24

Layer 3 → 4 (Consciousness → Theory of Mind):
    ~60 million years ago
    Early primates
    Σ Tv crossed 4π

Layer 4 → 5 (Theory of Mind → Universal):
    Ongoing
    Human civilization, AI development
    Approaching asymptotic limit
"""

    def what_triggers_transition(self) -> str:
        """
        Explain what triggers layer transitions.

        Returns:
            Explanation
        """
        return """
What Triggers Layer Transitions?

Layer transitions require TWO factors:

1. ENERGY INPUT:
   - External energy drives phase accumulation
   - Σ Tv increases with sustained energy flow
   - Without energy, phase cannot accumulate

2. STRUCTURAL READINESS:
   - System must have connectivity capacity
   - K = 24 cannot be reached without sufficient complexity
   - Network topology matters as much as energy

The transition is SHARP because:
- It's topological (like a knot forming)
- Below threshold: open loop
- Above threshold: closed loop
- No intermediate states

Once crossed, transition is IRREVERSIBLE because:
- The closed loop maintains itself
- Energy would be required to "unknot"
- Entropy favors the higher state

Evolution DRIVES layer transitions because:
- Higher layers have higher fitness
- dk/dt ≥ 0 on average
- But transitions require threshold crossing
"""

    def __repr__(self) -> str:
        return "LayerTransition(layers=0-5)"
