"""
Differentiation Operator D̂ for CRT.

Implements the full differentiation formula:
D̂[Ψ] = Ψ + Σᵢ αᵢ(S) P̂ᵢ[Ψ] + ζ(S) ∇²[Ψ]

Where:
- P̂ᵢ: Fourier mode projectors (orthogonal)
- αᵢ(S) = α₀ × (1 - S) × wᵢ  (syntony-dependent coupling)
- ζ(S) = ζ₀ × (1 - S)  (Laplacian diffusion coefficient)
- ∇²: Discrete Laplacian operator
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, List, Optional

from syntonic.crt.operators.base import OperatorBase
from syntonic.crt.operators.projectors import (
    FourierProjector,
    LaplacianOperator,
    create_mode_projectors,
)

if TYPE_CHECKING:
    from syntonic.core.state import State

# Golden ratio for weight computation
PHI = (1 + math.sqrt(5)) / 2
PHI_INV = PHI - 1  # ≈ 0.618


class DifferentiationOperator(OperatorBase):
    """
    Differentiation Operator D̂.

    Increases complexity/differentiation in a state by:
    1. Amplifying selected Fourier modes (via P̂ᵢ projectors)
    2. Adding Laplacian diffusion (∇²)

    The coupling strengths are syntony-dependent:
    - High syntony (S ≈ 1): Weak differentiation
    - Low syntony (S ≈ 0): Strong differentiation

    This creates the fundamental asymmetry: differentiation is stronger
    when the system is far from equilibrium.

    Example:
        >>> D_op = DifferentiationOperator(alpha_0=0.1, zeta_0=0.01)
        >>> evolved = D_op.apply(state)
    """

    def __init__(
        self,
        alpha_0: float = 0.1,
        zeta_0: float = 0.01,
        num_modes: int = 8,
        projectors: Optional[List[FourierProjector]] = None,
        weights: Optional[List[float]] = None,
        laplacian: Optional[LaplacianOperator] = None,
    ):
        """
        Create a differentiation operator.

        Args:
            alpha_0: Base coupling strength for Fourier modes
            zeta_0: Base Laplacian diffusion coefficient
            num_modes: Number of Fourier modes to use (if projectors not given)
            projectors: Custom Fourier projectors (optional)
            weights: Custom mode weights wᵢ (optional, default: golden decay)
            laplacian: Custom Laplacian operator (optional)
        """
        self.alpha_0 = alpha_0
        self.zeta_0 = zeta_0
        self.num_modes = num_modes

        # Projectors will be initialized on first use if not provided
        self._projectors = projectors
        self._laplacian = laplacian or LaplacianOperator(boundary="periodic")

        # Mode weights (golden ratio decay by default)
        if weights is not None:
            self._weights = weights
        else:
            # wᵢ = φ^(-i) gives golden decay
            self._weights = [PHI_INV**i for i in range(num_modes)]

    @property
    def projectors(self) -> List[FourierProjector]:
        """Get Fourier projectors (may be lazily initialized)."""
        return self._projectors or []

    @property
    def weights(self) -> List[float]:
        """Get mode weights."""
        return self._weights

    @property
    def laplacian(self) -> LaplacianOperator:
        """Get Laplacian operator."""
        return self._laplacian

    def _ensure_projectors(self, size: int) -> List[FourierProjector]:
        """Ensure projectors exist for given size."""
        if self._projectors is None:
            self._projectors = create_mode_projectors(
                size=size,
                num_modes=self.num_modes,
                include_dc=False,  # Don't include DC for differentiation
            )
        return self._projectors

    def apply(
        self,
        state: "State",
        syntony: Optional[float] = None,
        **kwargs,
    ) -> "State":
        """
        Apply differentiation operator D̂.

        D̂[Ψ] = Ψ + Σᵢ αᵢ(S) P̂ᵢ[Ψ] + ζ(S) ∇²[Ψ]

        Args:
            state: Input state Ψ
            syntony: Current syntony S (if None, computed from state)

        Returns:
            Differentiated state D̂[Ψ]
        """
        # Get syntony
        S = syntony if syntony is not None else state.syntony

        # Ensure projectors are initialized
        projectors = self._ensure_projectors(state.size)

        # Start with identity: result = Ψ
        result = state

        # Add Fourier mode contributions: Σᵢ αᵢ(S) P̂ᵢ[Ψ]
        for i, projector in enumerate(projectors):
            if i >= len(self._weights):
                break

            # αᵢ(S) = α₀ × (1 - S) × wᵢ
            alpha_i = self.alpha_0 * (1 - S) * self._weights[i]

            # Add contribution: result += αᵢ(S) × P̂ᵢ[Ψ]
            if alpha_i > 1e-12:  # Skip negligible contributions
                projected = projector.project(state)
                result = result + projected * alpha_i

        # Add Laplacian diffusion: ζ(S) ∇²[Ψ]
        zeta = self.zeta_0 * (1 - S)
        if zeta > 1e-12:
            laplacian_term = self._laplacian.apply(state)
            result = result + laplacian_term * zeta

        return result

    def differentiation_magnitude(self, state: "State") -> float:
        """
        Compute magnitude of differentiation Δ_D = ||D̂[Ψ] - Ψ||.

        Args:
            state: Input state

        Returns:
            Differentiation magnitude
        """
        d_state = self.apply(state)
        diff = d_state - state
        return diff.norm()

    def __repr__(self) -> str:
        return (
            f"DifferentiationOperator(alpha_0={self.alpha_0}, "
            f"zeta_0={self.zeta_0}, num_modes={self.num_modes})"
        )


def default_differentiation_operator() -> DifferentiationOperator:
    """
    Create a differentiation operator with default SRT parameters.

    Returns:
        DifferentiationOperator with standard settings
    """
    return DifferentiationOperator(
        alpha_0=0.1,
        zeta_0=0.01,
        num_modes=8,
    )
