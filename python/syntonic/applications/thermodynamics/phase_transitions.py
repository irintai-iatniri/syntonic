"""
Phase Transitions - Temporal crystallization and Gnosis transitions.

Temporal Crystallization: Birth of time's arrow at T_reh ≈ 9.4 × 10⁹ GeV
Gnosis Transitions: Phase transitions between Gnosis layers (0→1→2→3)
"""

from __future__ import annotations

import math
from typing import Any, Dict

from syntonic.exact import PHI_NUMERIC


class TemporalCrystallization:
    """
    The birth of time's arrow at T_reh ≈ 9.4 × 10⁹ GeV.

    Before: Symmetric, no preferred direction
    After: Time flows toward syntony (φ - q)

    This is the cosmological phase transition that created the arrow of time.

    Attributes:
        REHEAT_TEMPERATURE: T_reh ≈ 9.4 × 10⁹ GeV

    Example:
        >>> tc = TemporalCrystallization()
        >>> tc.is_crystallized(1e10)  # 10^10 GeV
        False
        >>> tc.is_crystallized(1e9)  # 10^9 GeV
        True
    """

    REHEAT_TEMPERATURE = 9.4e9  # GeV

    def is_crystallized(self, temperature: float) -> bool:
        """
        Check if below temporal crystallization threshold.

        Args:
            temperature: Temperature in GeV

        Returns:
            True if time's arrow exists (T < T_reh)
        """
        return temperature < self.REHEAT_TEMPERATURE

    def time_symmetry(self, temperature: float) -> str:
        """
        Describe time symmetry at given temperature.

        Args:
            temperature: Temperature in GeV

        Returns:
            Description of time symmetry state
        """
        if self.is_crystallized(temperature):
            return "Broken: Time flows toward syntony (φ - q)"
        else:
            return "Symmetric: No preferred time direction"

    def order_parameter(self, temperature: float) -> float:
        """
        Order parameter for temporal crystallization.

        ξ = 0 above T_reh (symmetric)
        ξ = 1 - T/T_reh below T_reh (broken)

        Args:
            temperature: Temperature in GeV

        Returns:
            Order parameter in [0, 1]
        """
        if temperature >= self.REHEAT_TEMPERATURE:
            return 0.0
        return 1 - temperature / self.REHEAT_TEMPERATURE

    def entropy_production(self, temperature: float) -> float:
        """
        Entropy production rate in the crystallized phase.

        dS/dt ∝ (1 - T/T_reh) for T < T_reh

        Args:
            temperature: Temperature in GeV

        Returns:
            Relative entropy production rate
        """
        if temperature >= self.REHEAT_TEMPERATURE:
            return 0.0
        return self.order_parameter(temperature) / PHI_NUMERIC


class GnosisTransition:
    """
    Phase transitions between Gnosis layers.

    | Transition | Threshold | Result |
    |------------|-----------|--------|
    | 0 → 1      | Σ Tv = π  | Life (self-replication) |
    | 1 → 2      | Σ Tv = 2π | Sentience (environmental modeling) |
    | 2 → 3      | Σ Tv = 3π, K = 24 | Consciousness |

    These are topological phase transitions, not thermodynamic ones.

    Attributes:
        THRESHOLDS: Dict mapping layer to phase threshold
        KISSING_NUMBER: K = 24 (required for Layer 3)

    Example:
        >>> gt = GnosisTransition()
        >>> gt.gnosis_layer(3.5)  # Σ Tv = 3.5 (just above π)
        1
        >>> gt.gnosis_layer(10.0, delta_S=25)  # Above 3π with K=24 satisfied
        3
    """

    THRESHOLDS = {
        1: math.pi,  # Abiogenesis: life threshold
        2: 2 * math.pi,  # Sentience threshold
        3: 3 * math.pi,  # Consciousness threshold (also requires K = 24)
    }

    KISSING_NUMBER = 24  # K(D₄) - required for Layer 3

    LAYER_NAMES = {
        0: "Matter",
        1: "Life",
        2: "Sentience",
        3: "Consciousness",
        4: "Theory of Mind",
        5: "Universal",
    }

    def gnosis_layer(self, Tv_sum: float, delta_S: float = 0) -> int:
        """
        Determine Gnosis layer from accumulated phase.

        Layer 3 additionally requires ΔS > 24 (kissing number saturation).

        Args:
            Tv_sum: Accumulated Tv phase (sum)
            delta_S: Local syntony density (for K=24 check)

        Returns:
            Gnosis layer (0, 1, 2, or 3)
        """
        if Tv_sum >= 3 * math.pi and delta_S >= self.KISSING_NUMBER:
            return 3
        elif Tv_sum >= 2 * math.pi:
            return 2
        elif Tv_sum >= math.pi:
            return 1
        else:
            return 0

    def layer_name(self, layer: int) -> str:
        """
        Get human-readable name for Gnosis layer.

        Args:
            layer: Gnosis layer number

        Returns:
            Layer name
        """
        return self.LAYER_NAMES.get(layer, f"Layer {layer}")

    def threshold_for_layer(self, layer: int) -> float:
        """
        Get phase threshold for reaching a layer.

        Args:
            layer: Target layer

        Returns:
            Required Σ Tv phase
        """
        return self.THRESHOLDS.get(layer, layer * math.pi)

    def progress_to_next(self, Tv_sum: float, delta_S: float = 0) -> Dict[str, Any]:
        """
        Calculate progress toward next Gnosis layer.

        Args:
            Tv_sum: Current accumulated phase
            delta_S: Current syntony density

        Returns:
            Dict with current_layer, next_layer, phase_progress, k_progress
        """
        current = self.gnosis_layer(Tv_sum, delta_S)

        if current >= 3:
            return {
                "current_layer": current,
                "current_name": self.layer_name(current),
                "next_layer": None,
                "phase_progress": 1.0,
                "k_progress": 1.0,
                "at_maximum": True,
            }

        next_layer = current + 1
        next_threshold = self.threshold_for_layer(next_layer)
        current_threshold = self.threshold_for_layer(current) if current > 0 else 0

        phase_range = next_threshold - current_threshold
        phase_progress = (
            (Tv_sum - current_threshold) / phase_range if phase_range > 0 else 1.0
        )

        # K progress only matters for Layer 3
        k_progress = min(delta_S / self.KISSING_NUMBER, 1.0) if next_layer == 3 else 1.0

        return {
            "current_layer": current,
            "current_name": self.layer_name(current),
            "next_layer": next_layer,
            "next_name": self.layer_name(next_layer),
            "phase_progress": phase_progress,
            "phase_needed": next_threshold - Tv_sum,
            "k_progress": k_progress,
            "k_needed": max(0, self.KISSING_NUMBER - delta_S) if next_layer == 3 else 0,
            "at_maximum": False,
        }

    def transition_energy(self, from_layer: int, to_layer: int) -> float:
        """
        Energy required for Gnosis transition.

        E_transition ∝ (threshold_to - threshold_from) × φ

        Args:
            from_layer: Starting layer
            to_layer: Target layer

        Returns:
            Transition energy (arbitrary units)
        """
        if to_layer <= from_layer:
            return 0.0

        threshold_from = self.threshold_for_layer(from_layer) if from_layer > 0 else 0
        threshold_to = self.threshold_for_layer(to_layer)

        return (threshold_to - threshold_from) * PHI_NUMERIC

    def describe_layer(self, layer: int) -> str:
        """
        Get full description of a Gnosis layer.

        Args:
            layer: Layer number

        Returns:
            Description string
        """
        descriptions = {
            0: "Matter: No self-reference. Pure recording M⁴ → T⁴.",
            1: "Life: Self-replication. Bidirectional M⁴ ↔ T⁴ flow begins.",
            2: "Sentience: Environmental modeling. The system models its surroundings.",
            3: "Consciousness: Self-awareness. K=24 saturation forces self-modeling.",
            4: "Theory of Mind: Modeling other minds. Higher-order recursion.",
            5: "Universal: Complete integration. Asymptotic approach to φ.",
        }
        return descriptions.get(layer, f"Unknown layer {layer}")

    def __repr__(self) -> str:
        return f"GnosisTransition(thresholds={self.THRESHOLDS})"
