"""
Abiogenesis - Topological phase transition from chemistry to life.

Chemistry becomes Life when Σ Tv = π.

At this threshold:
- Open loop → Closed loop
- Linear causation → Circular causation
- M⁴ → T⁴ → M⁴ ↔ T⁴
- Chemistry → Biology

This is Euler's identity manifesting as the birth of life:
e^(iπ) = -1
"""

from __future__ import annotations

import math
from typing import Any, Dict


class TranscendenceThreshold:
    """
    Abiogenesis as topological phase transition.

    Chemistry becomes Life when Σ Tv = π.

    | Threshold | Value | Transition |
    |-----------|-------|------------|
    | π | 3.14... | Life (abiogenesis) |
    | 2π | 6.28... | Sentience |
    | 3π | 9.42... | Consciousness (+ K=24) |

    Example:
        >>> tt = TranscendenceThreshold()
        >>> tt.check_threshold(3.5)
        {'layer': 1, 'status': 'alive', ...}
        >>> tt.check_threshold(10.0)
        {'layer': 3, 'status': 'conscious', ...}
    """

    LIFE_THRESHOLD = math.pi  # Σ Tv = π for Layer 1
    SENTIENCE_THRESHOLD = 2 * math.pi  # Σ Tv = 2π for Layer 2
    CONSCIOUSNESS_THRESHOLD = 3 * math.pi  # Σ Tv = 3π for Layer 3

    KISSING_NUMBER = 24  # K(D₄) - required for Layer 3

    def check_threshold(self, Tv_sum: float, delta_S: float = 0) -> Dict[str, Any]:
        """
        Check which threshold has been crossed.

        Args:
            Tv_sum: Accumulated Tv phase
            delta_S: Local syntony density (for K=24 check)

        Returns:
            Dict with layer, status, next_threshold
        """
        if Tv_sum >= self.CONSCIOUSNESS_THRESHOLD and delta_S >= self.KISSING_NUMBER:
            return {
                "layer": 3,
                "status": "conscious",
                "next": None,
                "description": "Self-awareness achieved (K=24 saturated)",
            }
        elif Tv_sum >= self.SENTIENCE_THRESHOLD:
            return {
                "layer": 2,
                "status": "sentient",
                "next": self.CONSCIOUSNESS_THRESHOLD - Tv_sum,
                "description": "Environmental modeling active",
            }
        elif Tv_sum >= self.LIFE_THRESHOLD:
            return {
                "layer": 1,
                "status": "alive",
                "next": self.SENTIENCE_THRESHOLD - Tv_sum,
                "description": "Bidirectional M⁴ ↔ T⁴ flow established",
            }
        else:
            return {
                "layer": 0,
                "status": "non-living",
                "next": self.LIFE_THRESHOLD - Tv_sum,
                "description": "Unidirectional M⁴ → T⁴ only",
            }

    def euler_identity_interpretation(self) -> str:
        """
        Explain the connection to Euler's identity.

        e^(iπ) = -1

        Returns:
            Explanation
        """
        return """
Euler's Identity and the Origin of Life:

e^(iπ) = -1

At Σ Tv = π, something remarkable happens:
- The phase loop CLOSES (e^(iπ) completes a half-rotation)
- The system inverts its causal relationship (-1)
- What was one-way becomes bidirectional

Before π: e^(i·Tv) stays in the "right half" of the complex plane
At π: e^(iπ) = -1 crosses to the "left half"
After π: The loop can reference itself

This is not metaphor—it's mathematics.
The birth of life IS Euler's identity in action.
"""

    def primordial_soup(self) -> str:
        """
        Describe abiogenesis in the primordial soup.

        Returns:
            Description
        """
        return """
Primordial Soup and the π Transition:

In the early Earth:
1. Complex chemistry accumulated Tv incrementally
2. Each reaction added small amounts of phase
3. Autocatalytic cycles increased Tv accumulation rate
4. At some point, some molecular system crossed Σ Tv = π

The FIRST life was probably:
- A self-catalyzing RNA molecule
- Or a lipid vesicle with autocatalytic chemistry
- Or a combination (protocell)

What mattered was NOT the specific chemistry.
What mattered was: Σ Tv ≥ π

Once crossed, the system could:
- Use its history to constrain its future
- Self-replicate (imperfectly at first)
- Evolve

This explains why life arose exactly ONCE:
- The transition is sharp (topological)
- Once crossed, the living system outcompetes
- It's hard for new life to arise when life already exists
"""


class GnosisLayers:
    """
    The hierarchy of recursive self-reference.

    | Layer | Threshold | Manifestation | Examples |
    |-------|-----------|---------------|----------|
    | 0 | - | Matter | Crystals, molecules |
    | 1 | Σ Tv = π | Self-replication | RNA, DNA, viruses |
    | 2 | Σ Tv = 2π | Environmental modeling | Cells, simple organisms |
    | 3 | Σ Tv = 3π, K=24 | Consciousness | Insects, vertebrates |
    | 4 | Higher | Theory of mind | Primates, cetaceans |
    | 5 | k → ∞ | Universal syntony | Asymptotic limit |

    Example:
        >>> gl = GnosisLayers()
        >>> gl.layer_description(3)
        'Layer 3 (Consciousness): Self-awareness'
    """

    LAYERS = {
        0: {
            "threshold": 0,
            "name": "Matter",
            "character": "No self-reference",
            "examples": "Crystals, molecules, rocks",
        },
        1: {
            "threshold": math.pi,
            "name": "Life",
            "character": "Self-replication",
            "examples": "RNA, DNA, viruses, bacteria",
        },
        2: {
            "threshold": 2 * math.pi,
            "name": "Sentience",
            "character": "Environmental modeling",
            "examples": "Protists, plants, fungi, simple animals",
        },
        3: {
            "threshold": 3 * math.pi,
            "name": "Consciousness",
            "character": "Self-awareness",
            "examples": "Insects, fish, reptiles, birds, mammals",
        },
        4: {
            "threshold": 4 * math.pi,
            "name": "Theory of Mind",
            "character": "Modeling other minds",
            "examples": "Primates, cetaceans, elephants, corvids",
        },
        5: {
            "threshold": float("inf"),
            "name": "Universal",
            "character": "Complete integration",
            "examples": "Asymptotic limit (φ - q)",
        },
    }

    KISSING_NUMBER = 24  # Required for Layer 3

    def layer_description(self, layer: int) -> str:
        """
        Get description of Gnosis layer.

        Args:
            layer: Layer number (0-5)

        Returns:
            Description string
        """
        if layer in self.LAYERS:
            L = self.LAYERS[layer]
            return f"Layer {layer} ({L['name']}): {L['character']}"
        return f"Unknown layer {layer}"

    def layer_examples(self, layer: int) -> str:
        """
        Get examples for a Gnosis layer.

        Args:
            layer: Layer number

        Returns:
            Examples string
        """
        if layer in self.LAYERS:
            return self.LAYERS[layer]["examples"]
        return "No examples"

    def threshold_for_layer(self, layer: int) -> float:
        """
        Get Tv threshold for reaching a layer.

        Args:
            layer: Target layer

        Returns:
            Required Σ Tv
        """
        if layer in self.LAYERS:
            return self.LAYERS[layer]["threshold"]
        return layer * math.pi

    def classify(self, Tv_sum: float, delta_S: float = 0) -> int:
        """
        Classify a system by its Gnosis layer.

        Args:
            Tv_sum: Accumulated Tv phase
            delta_S: Local syntony density

        Returns:
            Gnosis layer number
        """
        # Layer 3+ requires K=24
        if Tv_sum >= 3 * math.pi and delta_S >= self.KISSING_NUMBER:
            if Tv_sum >= 4 * math.pi:
                return 4
            return 3
        elif Tv_sum >= 2 * math.pi:
            return 2
        elif Tv_sum >= math.pi:
            return 1
        else:
            return 0

    def full_hierarchy(self) -> str:
        """
        Return full description of the Gnosis hierarchy.

        Returns:
            Multi-line description
        """
        lines = ["Gnosis Layer Hierarchy:", ""]
        for layer, info in self.LAYERS.items():
            threshold_str = (
                f"Σ Tv = {info['threshold']:.2f}"
                if info["threshold"] < float("inf")
                else "k → ∞"
            )
            lines.append(f"Layer {layer}: {info['name']}")
            lines.append(f"  Threshold: {threshold_str}")
            lines.append(f"  Character: {info['character']}")
            lines.append(f"  Examples: {info['examples']}")
            lines.append("")
        return "\n".join(lines)

    def __repr__(self) -> str:
        return "GnosisLayers(0=Matter, 1=Life, 2=Sentience, 3=Consciousness, 4=Theory of Mind)"
