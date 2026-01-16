"""
Rust/CUDA backend for hierarchy.py module

This module provides GPU-accelerated operations for the SRT-Zero hierarchy system.
All operations have CPU fallback for systems without CUDA.

Usage:
    from srt_zero.backend import (
        apply_correction,
        apply_correction_uniform,
        apply_special,
        apply_suppression,
        compute_e_star_n,
        apply_chain,
        init_divisors,
    )
"""

from __future__ import annotations

from typing import List, Tuple, Union, Optional

# Try to import Rust backend
_CUDA_AVAILABLE = False
try:
    from syntonic._core import (
        hierarchy_apply_correction as apply_correction,
        hierarchy_apply_correction_uniform as apply_correction_uniform,
        hierarchy_apply_special as apply_special,
        hierarchy_apply_suppression as apply_suppression,
        hierarchy_compute_e_star_n as compute_e_star_n,
        hierarchy_apply_chain as apply_chain,
        hierarchy_init_divisors as init_divisors,
    )

    _CUDA_AVAILABLE = True
except ImportError:
    # Define stubs for when CUDA is not available
    # These will never be called since we check _CUDA_AVAILABLE
    def apply_correction(*args, **kwargs):
        raise RuntimeError("CUDA not available")

    def apply_correction_uniform(*args, **kwargs):
        raise RuntimeError("CUDA not available")

    def apply_special(*args, **kwargs):
        raise RuntimeError("CUDA not available")

    def apply_suppression(*args, **kwargs):
        raise RuntimeError("CUDA not available")

    def compute_e_star_n(*args, **kwargs):
        raise RuntimeError("CUDA not available")

    def apply_chain(*args, **kwargs):
        raise RuntimeError("CUDA not available")

    def init_divisors(*args, **kwargs):
        raise RuntimeError("CUDA not available")


# Import only constants from hierarchy.py to avoid circular imports
from .hierarchy import (
    Q,
    PHI,
    PI,
    E_STAR,
    GEOMETRIC_DIVISORS,
)

# Compute derived phi powers locally (not in hierarchy.py exports)
PHI_SQUARED = PHI * PHI
PHI_CUBED = PHI_SQUARED * PHI

# Define dataclasses locally to avoid circular imports
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Union


@dataclass
class CorrectionRecord:
    """Record of a single correction step."""

    step_type: str
    description: str
    factor: float
    before: float
    after: float


@dataclass
class DerivationResult:
    """Complete result of a derivation with correction trace."""

    tree_value: float
    final_value: float
    steps: List[CorrectionRecord] = field(default_factory=list)
    deviation_percent: float = 0.0

    def set_experimental(self, exp_value: float) -> None:
        """Calculate deviation from experimental value."""
        if abs(exp_value) < 1e-12:
            self.deviation_percent = (
                0.0 if abs(float(self.final_value)) < 1e-9 else 100.0
            )
        else:
            self.deviation_percent = (
                100 * abs(float(self.final_value) - exp_value) / exp_value
            )


# =============================================================================
# TYPE ENUMS FOR SPECIAL CORRECTIONS
# =============================================================================


class SpecialCorrectionType:
    """Enum for special correction types matching Rust backend."""

    Q_PHI_PLUS = 0
    Q_PHI_MINUS = 1
    Q_PHI_SQUARED_PLUS = 2
    Q_PHI_SQUARED_MINUS = 3
    Q_PHI_CUBED_PLUS = 4
    Q_PHI_CUBED_MINUS = 5
    Q_PHI_FOURTH_PLUS = 6
    Q_PHI_FOURTH_MINUS = 7
    Q_PHI_FIFTH_PLUS = 8
    Q_PHI_FIFTH_MINUS = 9
    Q_SQUARED_PLUS = 10
    Q_SQUARED_MINUS = 11
    Q_SQUARED_PHI_PLUS = 12
    Q_SQUARED_PHI_MINUS = 13
    Q_SQ_PHI_SQ_PLUS = 14
    Q_SQ_PHI_SQ_MINUS = 15
    Q_SQ_PHI_PLUS = 16
    FOUR_Q_PLUS = 17
    FOUR_Q_MINUS = 18
    THREE_Q_PLUS = 19
    THREE_Q_MINUS = 20
    SIX_Q_PLUS = 21
    EIGHT_Q_PLUS = 22
    PI_Q_PLUS = 23
    Q_CUBED = 24
    Q_PHI_DIV_4PI_PLUS = 25
    EIGHT_Q_INV_PLUS = 26
    Q_SQUARED_HALF_PLUS = 27
    Q_6PI_PLUS = 28
    Q_PHI_INV_PLUS = 29


class SuppressionType:
    """Enum for suppression factor types."""

    WINDING_INSTABILITY = 0
    RECURSION_PENALTY = 1
    DOUBLE_INVERSE = 2
    FIXED_POINT_PENALTY = 3


# =============================================================================
# HIGH-LEVEL WRAPPERS
# =============================================================================


def is_cuda_available() -> bool:
    """Check if CUDA backend is available."""
    return _CUDA_AVAILABLE


def batch_apply_correction(
    values: List[float],
    divisor: float,
    sign: int,
) -> List[float]:
    """
    Apply a single correction to a batch of values.

    Args:
        values: List of values to correct
        divisor: Single divisor for all values
        sign: +1 for enhancement, -1 for suppression

    Returns:
        Corrected values
    """
    if _CUDA_AVAILABLE:
        return apply_correction_uniform(values, divisor, sign)

    # CPU fallback
    factor = 1.0 + sign * Q / divisor if divisor != 0 else 1.0
    return [v * factor for v in values]


def batch_apply_special_correction(
    values: List[float],
    correction_names: List[str],
) -> List[float]:
    """
    Apply special corrections by name.

    Args:
        values: List of values to correct
        correction_names: List of correction type names (e.g., 'q_phi_plus', 'q_squared')

    Returns:
        Corrected values
    """
    # Map correction names to integer types
    type_map = {
        "q_phi_plus": SpecialCorrectionType.Q_PHI_PLUS,
        "q_phi_minus": SpecialCorrectionType.Q_PHI_MINUS,
        "q_phi_squared_plus": SpecialCorrectionType.Q_PHI_SQUARED_PLUS,
        "q_phi_squared_minus": SpecialCorrectionType.Q_PHI_SQUARED_MINUS,
        "q_phi_cubed_plus": SpecialCorrectionType.Q_PHI_CUBED_PLUS,
        "q_phi_cubed_minus": SpecialCorrectionType.Q_PHI_CUBED_MINUS,
        "q_phi_fourth_plus": SpecialCorrectionType.Q_PHI_FOURTH_PLUS,
        "q_phi_fourth_minus": SpecialCorrectionType.Q_PHI_FOURTH_MINUS,
        "q_phi_fifth_plus": SpecialCorrectionType.Q_PHI_FIFTH_PLUS,
        "q_phi_fifth_minus": SpecialCorrectionType.Q_PHI_FIFTH_MINUS,
        "q_squared_plus": SpecialCorrectionType.Q_SQUARED_PLUS,
        "q_squared_minus": SpecialCorrectionType.Q_SQUARED_MINUS,
        "q_squared_phi_plus": SpecialCorrectionType.Q_SQUARED_PHI_PLUS,
        "q_squared_phi_minus": SpecialCorrectionType.Q_SQUARED_PHI_MINUS,
        "q_sq_phi_sq_plus": SpecialCorrectionType.Q_SQ_PHI_SQ_PLUS,
        "q_sq_phi_sq_minus": SpecialCorrectionType.Q_SQ_PHI_SQ_MINUS,
        "q_sq_phi_plus": SpecialCorrectionType.Q_SQ_PHI_PLUS,
        "4q_plus": SpecialCorrectionType.FOUR_Q_PLUS,
        "4q_minus": SpecialCorrectionType.FOUR_Q_MINUS,
        "3q_plus": SpecialCorrectionType.THREE_Q_PLUS,
        "3q_minus": SpecialCorrectionType.THREE_Q_MINUS,
        "6q_plus": SpecialCorrectionType.SIX_Q_PLUS,
        "8q_plus": SpecialCorrectionType.EIGHT_Q_PLUS,
        "pi_q_plus": SpecialCorrectionType.PI_Q_PLUS,
        "q_cubed": SpecialCorrectionType.Q_CUBED,
        "q_phi_div_4pi_plus": SpecialCorrectionType.Q_PHI_DIV_4PI_PLUS,
        "8q_inv_plus": SpecialCorrectionType.EIGHT_Q_INV_PLUS,
        "q_squared_half_plus": SpecialCorrectionType.Q_SQUARED_HALF_PLUS,
        "q_6pi_plus": SpecialCorrectionType.Q_6PI_PLUS,
        "q_phi_inv_plus": SpecialCorrectionType.Q_PHI_INV_PLUS,
    }

    types = [type_map.get(name, 0) for name in correction_names]

    if _CUDA_AVAILABLE:
        return apply_special(values, types)

    # CPU fallback
    outputs = []
    for value, type_ in zip(values, types):
        if type_ == SpecialCorrectionType.Q_PHI_PLUS:
            factor = 1.0 + Q * PHI
        elif type_ == SpecialCorrectionType.Q_PHI_MINUS:
            factor = 1.0 - Q * PHI
        elif type_ == SpecialCorrectionType.Q_PHI_SQUARED_PLUS:
            factor = 1.0 + Q * PHI_SQUARED
        elif type_ == SpecialCorrectionType.Q_PHI_SQUARED_MINUS:
            factor = 1.0 - Q * PHI_SQUARED
        elif type_ == SpecialCorrectionType.Q_PHI_CUBED_PLUS:
            factor = 1.0 + Q * PHI_CUBED
        elif type_ == SpecialCorrectionType.Q_PHI_CUBED_MINUS:
            factor = 1.0 - Q * PHI_CUBED
        elif type_ == SpecialCorrectionType.Q_PHI_FOURTH_PLUS:
            factor = 1.0 + Q * (PHI_SQUARED**2)
        elif type_ == SpecialCorrectionType.Q_PHI_FOURTH_MINUS:
            factor = 1.0 - Q * (PHI_SQUARED**2)
        elif type_ == SpecialCorrectionType.Q_PHI_FIFTH_PLUS:
            factor = 1.0 + Q * (PHI_SQUARED**2) * PHI
        elif type_ == SpecialCorrectionType.Q_PHI_FIFTH_MINUS:
            factor = 1.0 - Q * (PHI_SQUARED**2) * PHI
        elif type_ == SpecialCorrectionType.Q_SQUARED_PLUS:
            factor = 1.0 + Q * Q
        elif type_ == SpecialCorrectionType.Q_SQUARED_MINUS:
            factor = 1.0 - Q * Q
        elif type_ == SpecialCorrectionType.Q_SQUARED_PHI_PLUS:
            factor = 1.0 + Q * Q / PHI
        elif type_ == SpecialCorrectionType.Q_SQUARED_PHI_MINUS:
            factor = 1.0 - Q * Q / PHI
        elif type_ == SpecialCorrectionType.Q_SQ_PHI_SQ_PLUS:
            factor = 1.0 + Q * Q / PHI_SQUARED
        elif type_ == SpecialCorrectionType.Q_SQ_PHI_SQ_MINUS:
            factor = 1.0 - Q * Q / PHI_SQUARED
        elif type_ == SpecialCorrectionType.Q_SQ_PHI_PLUS:
            factor = 1.0 + Q * Q * PHI
        elif type_ == SpecialCorrectionType.FOUR_Q_PLUS:
            factor = 1.0 + 4 * Q
        elif type_ == SpecialCorrectionType.FOUR_Q_MINUS:
            factor = 1.0 - 4 * Q
        elif type_ == SpecialCorrectionType.THREE_Q_PLUS:
            factor = 1.0 + 3 * Q
        elif type_ == SpecialCorrectionType.THREE_Q_MINUS:
            factor = 1.0 - 3 * Q
        elif type_ == SpecialCorrectionType.SIX_Q_PLUS:
            factor = 1.0 + 6 * Q
        elif type_ == SpecialCorrectionType.EIGHT_Q_PLUS:
            factor = 1.0 + 8 * Q
        elif type_ == SpecialCorrectionType.PI_Q_PLUS:
            factor = 1.0 + PI * Q
        elif type_ == SpecialCorrectionType.Q_CUBED:
            factor = 1.0 + Q * Q * Q
        elif type_ == SpecialCorrectionType.Q_PHI_DIV_4PI_PLUS:
            factor = 1.0 + Q * PHI / (4 * PI)
        elif type_ == SpecialCorrectionType.EIGHT_Q_INV_PLUS:
            factor = 1.0 + Q / 8.0
        elif type_ == SpecialCorrectionType.Q_SQUARED_HALF_PLUS:
            factor = 1.0 + Q * Q / 2.0
        elif type_ == SpecialCorrectionType.Q_6PI_PLUS:
            factor = 1.0 + Q / (6 * PI)
        elif type_ == SpecialCorrectionType.Q_PHI_INV_PLUS:
            factor = 1.0 + Q / PHI
        else:
            factor = 1.0

        outputs.append(value * factor)

    return outputs


def batch_apply_suppression(
    values: List[float],
    suppression_name: str,
) -> Tuple[List[float], float]:
    """
    Apply suppression factor to a batch of values.

    Args:
        values: List of values to suppress
        suppression_name: Type of suppression ('winding', 'recursion', 'double_inverse', 'fixed_point')

    Returns:
        (suppressed_values, factor)
    """
    type_map = {
        "winding": SuppressionType.WINDING_INSTABILITY,
        "recursion": SuppressionType.RECURSION_PENALTY,
        "double_inverse": SuppressionType.DOUBLE_INVERSE,
        "fixed_point": SuppressionType.FIXED_POINT_PENALTY,
    }

    sup_type = type_map.get(suppression_name, 0)

    if _CUDA_AVAILABLE:
        outputs = apply_suppression(values, sup_type)
    else:
        # CPU fallback
        if sup_type == SuppressionType.WINDING_INSTABILITY:
            factor = 1.0 / (1.0 + Q / PHI)
        elif sup_type == SuppressionType.RECURSION_PENALTY:
            factor = 1.0 / (1.0 + Q * PHI)
        elif sup_type == SuppressionType.DOUBLE_INVERSE:
            factor = 1.0 / (1.0 + Q / PHI_SQUARED)
        elif sup_type == SuppressionType.FIXED_POINT_PENALTY:
            factor = 1.0 / (1.0 + Q * PHI_SQUARED)
        else:
            factor = 1.0

        outputs = [v * factor for v in values]

    # Compute factor for return
    if sup_type == SuppressionType.WINDING_INSTABILITY:
        factor = 1.0 / (1.0 + Q / PHI)
    elif sup_type == SuppressionType.RECURSION_PENALTY:
        factor = 1.0 / (1.0 + Q * PHI)
    elif sup_type == SuppressionType.DOUBLE_INVERSE:
        factor = 1.0 / (1.0 + Q / PHI_SQUARED)
    elif sup_type == SuppressionType.FIXED_POINT_PENALTY:
        factor = 1.0 / (1.0 + Q * PHI_SQUARED)
    else:
        factor = 1.0

    return outputs, float(factor)


def batch_compute_e_star_n(
    N: List[float],
    corrections: List[Tuple[float, int]],
) -> List[float]:
    """
    Compute E* × N × ∏(1 ± q/divisor) for a batch.

    Args:
        N: List of N multipliers
        corrections: List of (divisor, sign) tuples to apply to all values

    Returns:
        Computed values
    """
    if not corrections:
        return [E_STAR * n for n in N]

    # Flatten corrections for all values
    divisors = [div for (div, _) in corrections for _ in N]
    signs = [sign for (_, sign) in corrections for _ in N]

    n_corrections = len(corrections)

    if _CUDA_AVAILABLE:
        return compute_e_star_n(N, divisors, signs, n_corrections)

    # CPU fallback
    outputs = []
    for n_val in N:
        value = E_STAR * n_val
        for divisor, sign in corrections:
            if divisor != 0:
                factor = 1.0 + sign * Q / divisor
                value *= factor
        outputs.append(value)

    return outputs


def batch_apply_corrections(
    tree_values: List[float],
    correction_chains: List[List[Tuple[float, int]]],
    suppression: Optional[str] = None,
) -> DerivationResult:
    """
    Apply correction chains to a batch of tree-level values.

    Args:
        tree_values: List of tree-level values
        correction_chains: List of correction chains for each value
        suppression: Optional suppression type to apply first

    Returns:
        DerivationResult with batched outputs
    """
    current = tree_values.copy()

    # Apply suppression first
    factor = 1.0
    if suppression:
        current, factor = batch_apply_suppression(current, suppression)

    # Apply correction chains
    all_divisors = []
    all_signs = []
    chain_lengths = []
    chain_starts = []

    total_corrections = 0
    for chain in correction_chains:
        chain_lengths.append(len(chain))
        chain_starts.append(total_corrections)
        for divisor, sign in chain:
            all_divisors.append(divisor)
            all_signs.append(sign)
        total_corrections += len(chain)

    if _CUDA_AVAILABLE and total_corrections > 0:
        final = apply_chain(
            current, all_divisors, all_signs, chain_lengths, chain_starts
        )
    else:
        # CPU fallback
        final = []
        for i, value in enumerate(current):
            val = value
            chain_len = chain_lengths[i]
            chain_start = chain_starts[i]
            for j in range(chain_len):
                corr_idx = chain_start + j
                divisor = all_divisors[corr_idx]
                sign = all_signs[corr_idx]
                if divisor != 0:
                    val *= 1.0 + sign * Q / divisor
            final.append(val)

    # Build result
    result = DerivationResult(
        tree_value=0.0,  # Not meaningful for batch
        final_value=final[0] if len(final) == 1 else 0.0,  # Not meaningful for batch
    )
    result.steps = []
    result.steps.append(
        CorrectionRecord(
            step_type="suppression" if suppression else "none",
            description=f"{suppression}" if suppression else "No suppression",
            factor=factor,
            before=float(tree_values[0]) if len(tree_values) == 1 else 0.0,
            after=float(final[0]) if len(final) == 1 else 0.0,
        )
    )

    return result


# =============================================================================
# INITIALIZATION
# =============================================================================


def initialize_backend():
    """Initialize the Rust/CUDA backend."""
    if not _CUDA_AVAILABLE:
        return

    # Initialize geometric divisors in constant memory
    divisors_list = list(GEOMETRIC_DIVISORS.values())
    if len(divisors_list) >= 84:
        try:
            init_divisors(divisors_list[:84])
        except Exception as e:
            print(f"Warning: Failed to initialize divisors: {e}")


# Auto-initialize on import
initialize_backend()


__all__ = [
    "is_cuda_available",
    "batch_apply_correction",
    "batch_apply_special_correction",
    "batch_apply_suppression",
    "batch_compute_e_star_n",
    "batch_apply_corrections",
    "SpecialCorrectionType",
    "SuppressionType",
    "initialize_backend",
]
