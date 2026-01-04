"""
DHSR Thermodynamic Cycle - The fundamental engine of reality.

The DHSR cycle operates with fixed efficiency η = 1/φ ≈ 61.8%.

Per-Cycle Throughput:
- 0.618 (= 1/φ) passes through aperture → integrates as Gnosis (product)
- 0.382 (= 1/φ²) recycles as potential → fuel for next cycle

The universe is a heat engine with this fixed efficiency.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, TYPE_CHECKING
import math

from syntonic.exact import PHI, PHI_NUMERIC, Q_DEFICIT_NUMERIC

if TYPE_CHECKING:
    from syntonic.core import State


@dataclass
class CycleResult:
    """Result of a DHSR thermodynamic cycle."""

    work_output: float
    heat_input: float
    heat_rejected: float
    efficiency: float
    syntony_change: float
    gnosis_change: float

    @property
    def is_carnot_limited(self) -> bool:
        """Check if efficiency is at the Carnot limit (1/φ)."""
        return abs(self.efficiency - 1/PHI_NUMERIC) < 1e-6


class DHSRThermodynamicCycle:
    """
    The DHSR cycle as thermodynamic engine.

    Per-Cycle Throughput (Theorem 3.1):
    - 0.618 (= 1/φ) passes through aperture → integrates as Gnosis (product)
    - 0.382 (= 1/φ²) recycles as potential → fuel for next cycle

    The universe is a heat engine with fixed efficiency η = 1/φ ≈ 61.8%

    Attributes:
        EFFICIENCY: The Carnot limit of reality (1/φ)
        WORK_FRACTION: Fraction converted to work (1/φ)
        RECYCLE_FRACTION: Fraction recycled (1/φ²)

    Example:
        >>> engine = DHSRThermodynamicCycle()
        >>> engine.EFFICIENCY
        0.6180339887498949
        >>> engine.carnot_efficiency_from_syntony(1.0, 0.382)
        0.618
    """

    # Fundamental efficiency - the Carnot limit of reality
    EFFICIENCY = 1 / PHI_NUMERIC  # η = 1/φ ≈ 0.618

    # DHSR partition: D + H = 1 where D = 1/φ² ≈ 0.382, H = 1/φ ≈ 0.618
    WORK_FRACTION = 1 / PHI_NUMERIC  # 0.618 - converted to work/gnosis
    RECYCLE_FRACTION = 1 / PHI_NUMERIC**2  # 0.382 - recycled as potential

    def __init__(self, working_medium: Optional['State'] = None):
        """
        Initialize with optional working medium state.

        Args:
            working_medium: Initial state (optional)
        """
        self._medium = working_medium
        self._accumulated_gnosis = 0.0
        self._cycle_count = 0

    @property
    def medium(self) -> Optional['State']:
        """Current working medium state."""
        return self._medium

    @property
    def accumulated_gnosis(self) -> float:
        """Total gnosis accumulated over all cycles."""
        return self._accumulated_gnosis

    @property
    def cycle_count(self) -> int:
        """Number of cycles completed."""
        return self._cycle_count

    def differentiation_step(self, state: 'State') -> 'State':
        """
        D̂: WU → {WU₁, WU₂, ..., WUₙ}

        Thermodynamic role: Entropy production, novelty creation.
        Analogous to expansion/heat absorption.

        Args:
            state: Input state

        Returns:
            Differentiated state
        """
        return state.differentiate()

    def harmonization_step(self, state: 'State') -> 'State':
        """
        Ĥ: Recombination into ratio pairs.

        Thermodynamic role: Integration, coherence building.
        Analogous to compression/work output.

        Args:
            state: Input state

        Returns:
            Harmonized state
        """
        return state.harmonize()

    def syntonization_step(self, state: 'State') -> tuple['State', str]:
        """
        Ŝ: Oscillation between Mv and Tv pairs.

        The filter - not a value but the oscillation itself.
        - If DH = φ → proceed to R
        - If DH < φ → accumulate (gain new WU)
        - If DH > φ → split (return to D)

        Args:
            state: Input state

        Returns:
            Tuple of (state, decision) where decision is 'proceed', 'accumulate', or 'split'
        """
        syntony = state.syntony

        if abs(syntony - PHI_NUMERIC) < 0.01:
            return state, 'proceed'
        elif syntony < PHI_NUMERIC:
            return state, 'accumulate'
        else:
            return state, 'split'

    def recursion_step(self, state: 'State') -> 'State':
        """
        R̂: Filtering and recycling.

        1. Subtract WU from total Mv → remainder = 0.618
        2. SiU moves to torus center
        3. WU splits: Mv₁ = 1 (continues), Mv₂ = 0.618 (new WU)
        4. Tv phase becomes permanent in T⁴

        Args:
            state: Input state

        Returns:
            Recursed state
        """
        return state.recurse()

    def run_cycle(
        self,
        heat_input: float,
        T_hot: Optional[float] = None,
        T_cold: Optional[float] = None,
    ) -> CycleResult:
        """
        Run complete thermodynamic cycle.

        Args:
            heat_input: Heat energy input (Q_in)
            T_hot: Hot reservoir temperature (optional)
            T_cold: Cold reservoir temperature (optional)

        Returns:
            CycleResult with efficiency, work, heat, syntony changes.
        """
        # Work output at maximum efficiency
        work = heat_input * self.EFFICIENCY
        heat_rejected = heat_input - work

        # Syntony and gnosis changes
        syntony_change = self.EFFICIENCY  # Approaches φ - q asymptotically
        gnosis_change = self.WORK_FRACTION * heat_input

        self._accumulated_gnosis += gnosis_change
        self._cycle_count += 1

        return CycleResult(
            work_output=work,
            heat_input=heat_input,
            heat_rejected=heat_rejected,
            efficiency=self.EFFICIENCY,
            syntony_change=syntony_change,
            gnosis_change=gnosis_change,
        )

    def carnot_efficiency_from_syntony(self, S_hot: float, S_cold: float) -> float:
        """
        Syntonic Carnot efficiency.

        η = 1 - S_cold/S_hot

        Note: Maximum possible = 1/φ ≈ 61.8%

        Args:
            S_hot: Syntony of hot reservoir
            S_cold: Syntony of cold reservoir

        Returns:
            Carnot efficiency (capped at 1/φ)
        """
        if S_hot <= 0:
            return 0.0
        eta = 1 - S_cold / S_hot
        return min(eta, self.EFFICIENCY)

    def work_from_syntony_gradient(self, delta_S: float, scale: float = 1.0) -> float:
        """
        Work extracted from syntony gradient.

        W = scale × ΔS × (1/φ)

        Args:
            delta_S: Syntony difference
            scale: Energy scale factor

        Returns:
            Work output
        """
        return scale * delta_S * self.EFFICIENCY

    def partition_check(self) -> dict:
        """
        Verify DHSR partition: D + H = S → 0.382 + 0.618 = 1.

        Returns:
            Dict with D, H, sum, and verification status
        """
        D = self.RECYCLE_FRACTION
        H = self.WORK_FRACTION
        total = D + H

        return {
            'D': D,
            'H': H,
            'sum': total,
            'verified': abs(total - 1.0) < 1e-10,
        }

    def __repr__(self) -> str:
        return f"DHSRThermodynamicCycle(efficiency={self.EFFICIENCY:.4f}, cycles={self._cycle_count})"
