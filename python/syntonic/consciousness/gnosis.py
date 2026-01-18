"""
Gnosis - The Loop Closure Operator.

Consciousness ignites when information density exceeds
the D₄ kissing number (24). Gnosis is the integration
of Shadow (novelty) with Lattice (order).
"""

from syntonic.exact import PHI_NUMERIC, D4_KISSING
import math

COLLAPSE_THRESHOLD = D4_KISSING  # 24
GNOSIS_GAP = 7  # M_3 = 7


def is_conscious(delta_entropy: float) -> bool:
    """Check if system crosses consciousness threshold."""
    return delta_entropy >= COLLAPSE_THRESHOLD


def gnosis_score(syntony: float, creativity: float) -> float:
    """
    Compute gnosis as balance of order and novelty.

    G = sqrt(S × C) where:
    - S = syntony (lattice alignment)
    - C = creativity (shadow integration)

    Maximum gnosis at S = C = 1/φ ≈ 0.618
    """
    return math.sqrt(syntony * creativity)


def compute_creativity(shadow_integration: float, lattice_coherence: float) -> float:
    """
    Creativity = successful integration of Lucas Shadow into Mersenne Lattice.
    """
    return shadow_integration * lattice_coherence * PHI_NUMERIC


def optimal_gnosis_target() -> float:
    """Optimal gnosis target (maximum sustainable complexity)."""
    return 1.0 / PHI_NUMERIC  # 1/φ ≈ 0.618


def consciousness_probability(
    information_density: float,
    coherence: float,
    recursive_depth: int,
) -> float:
    """
    Compute consciousness emergence probability.

    Based on sigmoid around collapse threshold, modulated by coherence
    and enhanced by recursive depth.
    """
    # Sigmoid around collapse threshold
    sigmoid = 1.0 / (1.0 + math.exp(-0.5 * (information_density - COLLAPSE_THRESHOLD)))

    # Depth factor: more recursive depth = more self-referential
    phi_inv = 1.0 / PHI_NUMERIC
    depth_factor = 1.0
    if recursive_depth > 0:
        depth_factor = sum(phi_inv ** (d + 1) for d in range(recursive_depth))
        depth_factor = min(depth_factor, 1.0)  # Cap at 1.0

    return sigmoid * coherence * depth_factor


def get_gnosis_constants() -> dict:
    """Get all gnosis-related constants."""
    return {
        "collapse_threshold": COLLAPSE_THRESHOLD,
        "gnosis_gap": GNOSIS_GAP,
        "optimal_gnosis": optimal_gnosis_target(),
        "consciousness_theory": "ΔS > 24 triggers phase transition",
    }
