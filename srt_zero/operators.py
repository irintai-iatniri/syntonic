"""
SRT-Zero: The Five Operators of Existence
=========================================
Source: Foundations.md Section 2.4

The five operators work in concert:
1. φ (Recursion) - generates time and complexity
2. π (Topology) - constrains to finite volume
3. Fermat Primes - differentiate into interaction layers
4. Mersenne Primes - stabilize energy into matter
5. Lucas Primes - balance with dark sector and enable evolution

This module re-exports canonical implementations from syntonic.exact and
syntonic.srt to ensure consistency across the library.
"""

from __future__ import annotations
from typing import List, Optional, Union
from dataclasses import dataclass

# Import WindingState from syntonic - the Rust-backed winding vector type
from syntonic.srt.geometry.winding import WindingState, winding_state

# Import constants from hierarchy (which gets them from syntonic.exact)
from .hierarchy import (
    PHI,
    PHI_INV,
    PI,
    Q,
    FERMAT_PRIMES,
    MERSENNE_EXPONENTS,
    LUCAS_SEQUENCE,
    FIBONACCI_PRIME_GATES,
    M11_BARRIER,
    LUCAS_PRIMES_INDICES,
)

# Import canonical implementations from syntonic.exact
from syntonic.exact import (
    lucas as lucas_number,  # Canonical Lucas number computation
    fibonacci as fibonacci_number,  # Canonical Fibonacci number computation
    LUCAS_PRIMES,
    MERSENNE_PRIMES,
)

# Import prime selection functions from syntonic.srt
from syntonic.srt import (
    # Fermat primes (Operator 3: Differentiation)
    fermat_number,
    is_fermat_prime,
    get_force_spectrum,
    validate_force_existence,
    # Mersenne primes (Operator 4: Harmonization)
    mersenne_number,
    is_mersenne_prime,
    get_generation_spectrum,
    validate_generation_stability,
    generation_barrier_explanation,
    # Lucas primes (Operator 5: Balance)
    shadow_phase,
    is_lucas_prime,
    dark_matter_mass_prediction as _srt_dark_matter_prediction,
    get_shadow_spectrum,
)


# =============================================================================
# OPERATOR 1: RECURSION (φ) - The Engine
# =============================================================================
def recursion_map(winding: WindingState) -> WindingState:
    """
    The Golden Recursion: R: n → ⌊φn⌋

    The primary driver of time evolution and complexity generation.

    Args:
        winding: WindingState n ∈ Z⁴

    Returns:
        Transformed WindingState
    """
    # WindingState supports scalar multiplication
    return winding_state(
        int(PHI * winding.n7),
        int(PHI * winding.n8),
        int(PHI * winding.n9),
        int(PHI * winding.n10),
    )


def is_recursion_fixed_point(winding: WindingState) -> bool:
    """
    Check if winding is a fixed point of the recursion map.

    Fixed points satisfy: n_i ∈ {0, ±1, ±2, ±3} for all i
    These correspond to stable particle states.
    """
    return winding.max_component <= 3


def get_recursion_orbit(winding: WindingState, max_steps: int = 20) -> List[WindingState]:
    """
    Compute the recursion orbit of a winding vector.

    Orbits terminate at fixed points (stable particles) or grow
    indefinitely (unstable/virtual states).
    """
    orbit = [winding]
    current = winding

    for _ in range(max_steps):
        next_winding = recursion_map(current)
        if next_winding == current:
            break  # Fixed point reached
        orbit.append(next_winding)
        current = next_winding

    return orbit


def winding_norm(winding: WindingState) -> float:
    """
    Compute the Euclidean norm of a winding vector.

    |n| = √(n₇² + n₈² + n₉² + n₁₀²)

    Uses the Rust WindingState.norm property.
    """
    return winding.norm


# =============================================================================
# OPERATOR 2: TOPOLOGY (π) - The Boundary
# =============================================================================
MODULAR_VOLUME = PI / 3  # Vol(F) = π/3


def topological_constraint(syntony: float) -> float:
    """
    Apply the topological boundary condition.

    The modular volume π/3 constrains the infinite recursion
    to fold back on itself.
    """
    return syntony * (1 - 1 / (3 * PI))


def compute_gravitational_coupling(length_scale: float) -> float:
    """
    Derive Newton's constant from topology.

    G = ℓ² / (12πq) - ratio of length scale to information capacity
    """
    return length_scale**2 / (12 * PI * Q)


# =============================================================================
# OPERATOR 3: DIFFERENTIATION (Fermat Primes) - The Architect
# These functions are re-exported from syntonic.srt.fermat_forces
# =============================================================================

# Build GAUGE_FORCES list from the canonical force spectrum
_force_spectrum = get_force_spectrum()
GAUGE_FORCES = [
    {"n": n, "fermat_number": f, "name": name, "gauge_group": group, "is_prime": n < 5}
    for n, (name, f, group) in _force_spectrum.items()
]

# Alias for backward compatibility
is_valid_force_index = validate_force_existence



# =============================================================================
# OPERATOR 4: HARMONIZATION (Mersenne Primes) - The Builder
# mersenne_number and is_mersenne_prime are imported from syntonic.srt
# =============================================================================


def is_stable_generation(p: int) -> bool:
    """
    Check if recursion index p produces a stable generation.

    Stability requires M_p = 2^p - 1 to be prime.
    Valid values: p ∈ {2, 3, 5, 7} (Generations 1-3 + heavy anchor)

    Args:
        p: Mersenne exponent (not generation index)

    Returns:
        True if M_p is Mersenne prime
    """
    return p in MERSENNE_EXPONENTS


# Alias generation barrier explanation
why_no_fourth_generation = generation_barrier_explanation


def get_generation(p: int) -> Optional[int]:
    """
    Get generation number for a given winding index.

    Returns:
        Generation 1-3 for stable matter, 4 for heavy anchor,
        None if unstable (like p=11).
    """
    gen_map = {2: 1, 3: 2, 5: 3, 7: 4}  # p -> generation
    return gen_map.get(p)


# =============================================================================
# OPERATOR 5: BALANCE (Lucas Primes) - The Shadow
# These functions are re-exported from syntonic.srt.lucas_shadow
# =============================================================================

# lucas_number is imported from syntonic.exact as the canonical implementation
# shadow_phase, is_lucas_prime, get_shadow_spectrum are imported from syntonic.srt


def dark_matter_mass_prediction(m_top: float = 173000.0) -> float:
    """
    Predict dark matter mass using Lucas stability.

    m_DM ≈ m_top × (L₁₇/L₁₃) ≈ 173 GeV × (3571/521) ≈ 1.18 TeV

    Args:
        m_top: Top quark mass in MeV (default 173 GeV)

    Returns:
        Dark matter particle mass in MeV

    Note:
        This is a wrapper that calls the canonical syntonic.srt implementation
        when m_top is the default, or computes directly for custom values.
    """
    if m_top == 173000.0:
        # Use canonical implementation (returns TeV, convert to MeV)
        prediction_tev, _ = _srt_dark_matter_prediction()
        return prediction_tev * 1000000  # Convert TeV to MeV
    else:
        # Custom calculation
        L17 = lucas_number(17)  # 3571
        L13 = lucas_number(13)  # 521
        return m_top * (L17 / L13)


def is_shadow_stable(n: int) -> bool:
    """
    Check if index n produces a Lucas-stable shadow state.

    Used for dark sector particle predictions.
    """
    return n in LUCAS_PRIMES_INDICES


# =============================================================================
# UNIFIED OPERATOR APPLICATION
# =============================================================================
@dataclass
class OperatorResult:
    """Result of applying the five operators to a state."""

    recursion_depth: int
    is_fixed_point: bool
    generation: Optional[int]
    forces_active: List[str]
    shadow_stable: bool
    syntony: float


def apply_five_operators(winding: WindingState, recursion_index: int) -> OperatorResult:
    """
    Apply all five operators to determine state properties.

    This is the unified entry point for classifying physical states.

    Args:
        winding: WindingState representing the T⁴ winding configuration
        recursion_index: Mersenne exponent for generation assignment
    """
    # Operator 1: Recursion
    orbit = get_recursion_orbit(winding)
    is_fixed = is_recursion_fixed_point(winding)
    depth = len(orbit)

    # Operator 2: Topology (syntony computation)
    norm = winding_norm(winding)
    syntony = PHI - Q * (1 + norm / 3)

    # Operator 3: Differentiation (active forces)
    forces = [f["name"] for f in GAUGE_FORCES if f["is_prime"] and f["n"] <= 4]

    # Operator 4: Harmonization (generation)
    gen = get_generation(recursion_index)

    # Operator 5: Balance (shadow stability)
    shadow = is_shadow_stable(recursion_index)

    return OperatorResult(
        recursion_depth=depth,
        is_fixed_point=is_fixed,
        generation=gen,
        forces_active=forces,
        shadow_stable=shadow,
        syntony=syntony,
    )


# =============================================================================
# EXPORTS
# =============================================================================
__all__ = [
    # WindingState from syntonic (re-exported for convenience)
    "WindingState",
    "winding_state",
    # Operator 1: Recursion
    "recursion_map",
    "is_recursion_fixed_point",
    "get_recursion_orbit",
    "winding_norm",
    # Operator 2: Topology
    "MODULAR_VOLUME",
    "topological_constraint",
    "compute_gravitational_coupling",
    # Operator 3: Differentiation
    "GAUGE_FORCES",
    "is_valid_force_index",
    "fermat_number",
    # Operator 4: Harmonization
    "mersenne_number",
    "is_stable_generation",
    "get_generation",
    "why_no_fourth_generation",
    # Operator 5: Balance
    "lucas_number",
    "dark_matter_mass_prediction",
    "is_shadow_stable",
    # Unified
    "OperatorResult",
    "apply_five_operators",
]
