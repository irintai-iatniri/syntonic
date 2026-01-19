"""
Harmonization Operator Ĥ for CRT.

Implements the full harmonization formula:
Ĥ[Ψ] = Ψ - Σᵢ βᵢ(S,Δ_D) Q̂ᵢ[Ψ] + γ(S) Ŝ_op[Ψ] + Δ_NL[Ψ]

Where:
- Q̂ᵢ: High-frequency damping projectors
- βᵢ(S,Δ_D) = β₀ × S × decay_factor  (syntony-dependent damping)
- γ(S): Syntony projection strength
- Ŝ_op: Syntony-promoting operator (projects toward golden measure)
- Δ_NL: Nonlinear correction term
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Callable, List, Optional

from syntonic.crt.operators.base import OperatorBase
from syntonic.crt.operators.projectors import (
    DampingProjector,
    create_damping_cascade,
)

if TYPE_CHECKING:
    from syntonic.core.state import State

# Golden ratio constants
PHI = (1 + math.sqrt(5)) / 2
PHI_INV = PHI - 1  # ≈ 0.618


class HarmonizationOperator(OperatorBase):
    """
    Harmonization Operator Ĥ.

    Decreases complexity/differentiation by:
    1. Damping high-frequency components (via Q̂ᵢ projectors)
    2. Projecting toward a syntony-promoting target
    3. Applying nonlinear corrections

    The damping strengths are syntony-dependent:
    - High syntony (S ≈ 1): Strong harmonization (maintain equilibrium)
    - Low syntony (S ≈ 0): Weak harmonization (allow differentiation)

    This creates complementary dynamics with D̂:
    D + H = S → 0.382 + 0.618 = 1 (golden partition)

    Example:
        >>> H_op = HarmonizationOperator(beta_0=0.618)
        >>> evolved = H_op.apply(state)
    """

    def __init__(
        self,
        beta_0: float = 0.618,  # 1/φ by default
        gamma_0: float = 0.1,
        num_dampers: int = 3,
        damping_projectors: Optional[List[DampingProjector]] = None,
        target_generator: Optional[Callable[["State"], "State"]] = None,
        nonlinear_strength: float = 0.01,
    ):
        """
        Create a harmonization operator.

        Args:
            beta_0: Base damping strength (default: 1/φ ≈ 0.618)
            gamma_0: Syntony projection strength
            num_dampers: Number of damping levels (if damping_projectors not given)
            damping_projectors: Custom damping projectors (optional)
            target_generator: Function to generate target state (optional)
            nonlinear_strength: Strength of nonlinear correction term
        """
        self.beta_0 = beta_0
        self.gamma_0 = gamma_0
        self.num_dampers = num_dampers
        self.nonlinear_strength = nonlinear_strength

        # Damping projectors (cascade with golden decay)
        if damping_projectors is not None:
            self._dampers = damping_projectors
        else:
            self._dampers = create_damping_cascade(
                num_levels=num_dampers,
                base_cutoff=0.7,
                decay=PHI_INV,
            )

        # Target generator for syntony projection
        self._target_generator = target_generator or self._default_target

    @property
    def dampers(self) -> List[DampingProjector]:
        """Get damping projectors."""
        return self._dampers

    @staticmethod
    def _default_target(state: "State") -> "State":
        """
        Default target generator: normalized mean projection.

        This projects toward a uniform state with the same total magnitude,
        representing the "most harmonized" configuration.
        """
        from syntonic.core.state import State

        flat = state.to_list()
        N = len(flat)

        # Compute mean value
        if isinstance(flat[0], complex):
            mean_val = sum(flat) / N
        else:
            mean_val = sum(flat) / N

        # Target is uniform state with mean value
        target_flat = [mean_val] * N

        return State(
            target_flat, dtype=state.dtype, device=state.device, shape=state.shape
        )

    def _decay_factor(self, level: int, delta_d: Optional[float]) -> float:
        """
        Compute decay factor for damping level.

        Higher levels have stronger decay, modulated by differentiation magnitude.

        Args:
            level: Damping level index
            delta_d: Differentiation magnitude (optional)

        Returns:
            Decay factor in [0, 1]
        """
        # Base decay follows golden ratio
        base_decay = PHI_INV**level

        # Modulate by differentiation magnitude if provided
        if delta_d is not None:
            # Stronger damping when more differentiated
            modulation = 1.0 + delta_d
        else:
            modulation = 1.0

        return base_decay * min(modulation, 2.0)  # Cap at 2x

    def _syntony_projection(self, state: "State", S: float) -> "State":
        """
        Compute syntony-promoting projection Ŝ_op[Ψ].

        Projects state toward target, weighted by syntony.

        Args:
            state: Input state
            S: Current syntony

        Returns:
            Syntony projection contribution
        """
        target = self._target_generator(state)

        # Direction toward target
        direction = target - state

        # Weight by γ(S) = γ₀ × S
        gamma = self.gamma_0 * S

        return direction * gamma

    def _nonlinear_correction(self, state: "State", S: float) -> "State":
        """
        Compute nonlinear correction Δ_NL[Ψ].

        Adds small corrections based on local curvature.

        Args:
            state: Input state
            S: Current syntony

        Returns:
            Nonlinear correction term
        """
        from syntonic.core.state import State

        if self.nonlinear_strength < 1e-12:
            # Return zero state
            flat = [0.0] * state.size
            return State(
                flat, dtype=state.dtype, device=state.device, shape=state.shape
            )

        flat = state.to_list()
        N = len(flat)

        # Simple nonlinear correction: cubic damping
        # Δ_NL[Ψ]ᵢ = -η × Ψᵢ³ / (1 + |Ψᵢ|²)
        correction = []
        for i in range(N):
            x = flat[i]
            if isinstance(x, complex):
                x_abs_sq = abs(x) ** 2
                nl = -self.nonlinear_strength * x * x_abs_sq / (1 + x_abs_sq)
            else:
                x_sq = x * x
                nl = -self.nonlinear_strength * x * x_sq / (1 + x_sq)
            correction.append(nl)

        # Scale by syntony (stronger correction at higher syntony)
        correction = [c * S for c in correction]

        return State(
            correction, dtype=state.dtype, device=state.device, shape=state.shape
        )

    def apply(
        self,
        state: "State",
        syntony: Optional[float] = None,
        delta_d: Optional[float] = None,
        **kwargs,
    ) -> "State":
        """
        Apply harmonization operator Ĥ.

        Ĥ[Ψ] = Ψ - Σᵢ βᵢ(S,Δ_D) Q̂ᵢ[Ψ] + γ(S) Ŝ_op[Ψ] + Δ_NL[Ψ]

        Args:
            state: Input state Ψ
            syntony: Current syntony S (if None, computed from state)
            delta_d: Differentiation magnitude (optional, for adaptive damping)

        Returns:
            Harmonized state Ĥ[Ψ]
        """
        # Get syntony
        S = syntony if syntony is not None else state.syntony

        # Start with identity: result = Ψ
        result = state

        # Subtract damping contributions: -Σᵢ βᵢ(S,Δ_D) Q̂ᵢ[Ψ]
        for i, damper in enumerate(self._dampers):
            # βᵢ(S,Δ_D) = β₀ × S × decay_factor
            decay = self._decay_factor(i, delta_d)
            beta_i = self.beta_0 * S * decay

            # Subtract contribution: result -= βᵢ × (Ψ - Q̂ᵢ[Ψ])
            # This removes the high-frequency part
            if beta_i > 1e-12:
                damped = damper.project(state)
                high_freq = state - damped
                result = result - high_freq * beta_i

        # Add syntony projection: γ(S) Ŝ_op[Ψ]
        syntony_proj = self._syntony_projection(state, S)
        result = result + syntony_proj

        # Add nonlinear correction: Δ_NL[Ψ]
        nl_correction = self._nonlinear_correction(state, S)
        result = result + nl_correction

        return result

    def harmonization_magnitude(self, state: "State") -> float:
        """
        Compute magnitude of harmonization Δ_H = ||Ĥ[Ψ] - Ψ||.

        Args:
            state: Input state

        Returns:
            Harmonization magnitude
        """
        h_state = self.apply(state)
        diff = h_state - state
        return diff.norm()

    def __repr__(self) -> str:
        return (
            f"HarmonizationOperator(beta_0={self.beta_0}, "
            f"gamma_0={self.gamma_0}, num_dampers={len(self._dampers)})"
        )


def default_harmonization_operator() -> HarmonizationOperator:
    """
    Create a harmonization operator with default SRT parameters.

    Returns:
        HarmonizationOperator with standard settings
    """
    return HarmonizationOperator(
        beta_0=PHI_INV,  # 1/φ ≈ 0.618
        gamma_0=0.1,
        num_dampers=3,
    )
