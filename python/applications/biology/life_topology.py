"""
Life Topology - Life defined by information flow topology.

NON-LIFE: M⁴ → T⁴ (unidirectional)
    Events are recorded but do not steer.

LIFE: M⁴ ↔ T⁴ (bidirectional)
    The accumulated Tv history "hooks back" to constrain
    and shape future M⁴ expression.

This is not a difference of degree—it is a topological distinction.
"""

from __future__ import annotations
from typing import Any, List
import math


class LifeTopology:
    """
    Life defined by information flow topology.

    NON-LIFE: M⁴ → T⁴ (unidirectional)
        Events are recorded but do not steer.

    LIFE: M⁴ ↔ T⁴ (bidirectional)
        The accumulated Tv history "hooks back" to constrain
        and shape future M⁴ expression.

    This is not a difference of degree—it is a topological distinction.

    Attributes:
        LIFE_THRESHOLD: π (minimum Tv sum for life)

    Example:
        >>> lt = LifeTopology()
        >>> lt.is_alive(None, 3.5)  # Σ Tv > π
        True
        >>> lt.is_alive(None, 2.0)  # Σ Tv < π
        False
    """

    LIFE_THRESHOLD = math.pi  # Σ Tv = π for bidirectionality

    def is_alive(self, system: Any, Tv_sum: float) -> bool:
        """
        Check for bidirectional M⁴ ↔ T⁴ flow.

        Requires Σ Tv ≥ π (phase closure).

        Args:
            system: System object (can be None, used for future extensions)
            Tv_sum: Accumulated Tv phase

        Returns:
            True if system is alive (bidirectional)
        """
        return Tv_sum >= self.LIFE_THRESHOLD

    def tv_hook_strength(self, Tv_history: List[float]) -> float:
        """
        Measure topological constraint on future M⁴.

        Stronger hooks = more deterministic future.
        This is the "accumulated history" that constrains possibilities.

        Args:
            Tv_history: List of Tv phase values

        Returns:
            Total hook strength
        """
        return sum(abs(tv) for tv in Tv_history)

    def information_flow(self, system_type: str) -> str:
        """
        Describe information flow for different system types.

        | System | Flow | Character |
        |--------|------|-----------|
        | Crystal | M⁴ → T⁴ | Recording without steering |
        | Cell | M⁴ ↔ T⁴ | Recording AND steering |

        Args:
            system_type: Type of system

        Returns:
            Information flow description
        """
        flows = {
            'crystal': 'M⁴ → T⁴ (unidirectional, recording only)',
            'rock': 'M⁴ → T⁴ (unidirectional, recording only)',
            'virus': 'M⁴ ↔ T⁴ (weak bidirectional, parasitic)',
            'cell': 'M⁴ ↔ T⁴ (strong bidirectional)',
            'bacterium': 'M⁴ ↔ T⁴ (strong bidirectional)',
            'plant': 'M⁴ ↔ T⁴ (hierarchical bidirectional)',
            'animal': 'M⁴ ↔ T⁴ (hierarchical bidirectional)',
            'organism': 'M⁴ ↔ T⁴ (hierarchical bidirectional)',
        }
        return flows.get(system_type.lower(), f'Unknown flow for {system_type}')

    def flow_direction(self, Tv_sum: float) -> str:
        """
        Determine flow direction from Tv sum.

        Args:
            Tv_sum: Accumulated Tv phase

        Returns:
            Flow direction description
        """
        if Tv_sum < self.LIFE_THRESHOLD:
            return 'unidirectional (M⁴ → T⁴)'
        else:
            return 'bidirectional (M⁴ ↔ T⁴)'

    def topological_definition(self) -> str:
        """
        Explain the topological definition of life.

        Returns:
            Explanation
        """
        return """
Topological Definition of Life:

LIFE is NOT defined by:
- Chemical composition (carbon, water, etc.)
- Self-replication alone
- Metabolism alone
- Response to stimuli alone

LIFE IS defined by:
- Bidirectional information flow: M⁴ ↔ T⁴
- The accumulated Tv history (in T⁴) reaches back to constrain
  and shape future events in M⁴

This creates a CLOSED LOOP:
    M⁴ events → recorded as Tv in T⁴ → hooks constrain → future M⁴ events

The threshold Σ Tv = π marks where this loop CLOSES.
Below π: Open loop, one-way recording (chemistry)
At/above π: Closed loop, bidirectional steering (life)

This is a TOPOLOGICAL distinction, not a quantitative one.
It's like the difference between a line and a circle.
"""

    def crystal_vs_cell(self) -> str:
        """
        Compare crystals and cells in terms of information flow.

        Returns:
            Comparison
        """
        return """
Crystal vs Cell - The Topological Difference:

CRYSTAL (not alive):
- Grows by adding atoms according to lattice template
- Each layer records the previous configuration
- BUT: The recorded history does not steer future growth
- Information flow: M⁴ → T⁴ only
- The crystal doesn't "choose" how to grow

CELL (alive):
- DNA stores accumulated Tv history
- This history actively constrains protein synthesis
- Which constrains metabolism
- Which constrains reproduction
- Which constrains DNA maintenance
- Information flow: M⁴ ↔ T⁴ (bidirectional)
- The cell's history shapes its future

The difference is NOT complexity—it's topology.
A simple cell with just a few genes crosses the π threshold.
A complex crystal never crosses it, no matter how large.
"""

    def virus_paradox(self) -> str:
        """
        Address the "are viruses alive?" question.

        Returns:
            Explanation
        """
        return """
The Virus Question:

Viruses are WEAKLY bidirectional:
- They have Tv history (genetic material)
- This history does constrain future events
- BUT: They cannot close the loop independently
- They require host machinery to complete the cycle

Σ Tv(virus alone) < π
Σ Tv(virus + host) > π

Viruses are PARASITICALLY alive:
- They borrow part of the host's M⁴ ↔ T⁴ cycle
- Without a host, they're just chemicals
- With a host, they participate in life

This resolves the paradox: Viruses are conditionally alive,
depending on whether they're actively parasitizing a host.
"""

    def __repr__(self) -> str:
        return f"LifeTopology(threshold=π≈{self.LIFE_THRESHOLD:.4f})"
