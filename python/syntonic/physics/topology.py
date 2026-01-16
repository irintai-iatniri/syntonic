"""
Topological Physics Functions for GnosticOuroboros and related architectures.

Implements SRT-based physics metrics for neural network architectures:
- hooking_coefficient: Topological linkage between winding states
- golden_resonance: Alignment with golden ratio structure
- e8_root_alignment: Projection onto E8 root lattice

These functions bridge the geometric SRT framework with neural network training.

Source: Theory/SRT_Altruxa_Bridge.md, Theory/Lepton_Entropy.md
"""

from __future__ import annotations
import math
from typing import Union, List, TYPE_CHECKING

from syntonic.srt.constants import PHI_NUMERIC

if TYPE_CHECKING:
    from syntonic.nn.resonant_tensor import ResonantTensor


def hooking_coefficient(
    winding1: Union['ResonantTensor', List[float]],
    winding2: Union['ResonantTensor', List[float]],
) -> float:
    """
    Compute topological hooking coefficient between two winding states.

    The hooking coefficient measures the topological linkage between two
    winding configurations on T^4. From SRT, particles that "hook" have
    aligned winding vectors.

    Formula: H(w1, w2) = |w1 · w2| / (|w1| * |w2| + eps) * φ

    Args:
        winding1: First winding vector (ResonantTensor or list of floats)
        winding2: Second winding vector (ResonantTensor or list of floats)

    Returns:
        Hooking coefficient in [0, φ]. Higher values indicate stronger
        topological linkage.

    Example:
        >>> from syntonic.nn.resonant_tensor import ResonantTensor
        >>> w1 = ResonantTensor([1., 0., 0., 0., 0., 0., 0., 1.], [8])
        >>> w2 = ResonantTensor([1., 0., 0., 0., 0., 0., 0., 1.], [8])
        >>> hooking_coefficient(w1, w2)  # Same winding -> high hooking
        1.618...

    Source: Theory/SRT_Altruxa_Bridge.md §475
    """
    # Extract float values
    if hasattr(winding1, 'to_floats'):
        v1 = winding1.to_floats()
    else:
        v1 = list(winding1)

    if hasattr(winding2, 'to_floats'):
        v2 = winding2.to_floats()
    else:
        v2 = list(winding2)

    # Ensure same length (pad shorter with zeros)
    max_len = max(len(v1), len(v2))
    v1 = v1 + [0.0] * (max_len - len(v1))
    v2 = v2 + [0.0] * (max_len - len(v2))

    # Compute dot product
    dot = sum(a * b for a, b in zip(v1, v2))

    # Compute norms
    norm1 = math.sqrt(sum(x * x for x in v1))
    norm2 = math.sqrt(sum(x * x for x in v2))

    eps = 1e-8
    # Normalized hooking scaled by PHI
    return abs(dot) / (norm1 * norm2 + eps) * PHI_NUMERIC


def golden_resonance(tensor: 'ResonantTensor') -> float:
    """
    Compute golden resonance metric for a tensor.

    Golden resonance measures how well a tensor's spectral structure aligns
    with the golden ratio hierarchy. Uses the golden measure w(n) = exp(-|n|²/φ)
    to weight each mode.

    Formula: R = Σ w(n) * |ψ_n|² where w(n) = exp(-|n|²/(2φ))

    Args:
        tensor: ResonantTensor to evaluate

    Returns:
        Golden resonance value. Target: R > 24.0 for transcendence
        (related to D4 kissing number = 24).

    Example:
        >>> from syntonic.nn.resonant_tensor import ResonantTensor
        >>> t = ResonantTensor([1.0, 2.0, 3.0, 4.0], [4])
        >>> golden_resonance(t)
        12.5...  # Depends on mode structure

    Source: Theory/Lepton_Entropy.md, SRT golden measure
    """
    values = tensor.to_floats()

    # Get mode norms if available, otherwise use sequential
    if hasattr(tensor, 'get_mode_norms'):
        try:
            mode_norms = tensor.get_mode_norms()
        except Exception:
            # Fallback to sequential mode norms
            mode_norms = [float(i * i) for i in range(len(values))]
    else:
        mode_norms = [float(i * i) for i in range(len(values))]

    # Compute weighted sum using golden measure
    # w(n) = exp(-|n|² / (2*φ))
    resonance = 0.0
    for val, norm_sq in zip(values, mode_norms):
        weight = math.exp(-norm_sq / (2 * PHI_NUMERIC))
        resonance += weight * val * val

    # Scale to useful range (D4 kissing number = 24 is transcendence target)
    # Multiply by dimension factor to get values in 0-30 range typically
    return resonance * len(values) / 10.0


def e8_root_alignment(tensor: 'ResonantTensor') -> float:
    """
    Compute E8 root alignment metric.

    Measures how closely a tensor's structure aligns with E8 root vectors.
    Projects the first 8 dimensions onto the 240 E8 roots and returns
    the maximum alignment (cosine similarity).

    Args:
        tensor: ResonantTensor to evaluate (uses first 8 elements)

    Returns:
        Alignment score in [0, 1]. Target: > 0.987 for transcendence.

    Example:
        >>> from syntonic.nn.resonant_tensor import ResonantTensor
        >>> # A tensor aligned with an E8 root
        >>> t = ResonantTensor([1., 1., 0., 0., 0., 0., 0., 0.], [8])
        >>> e8_root_alignment(t)
        1.0  # Perfectly aligned with type-A root

    Source: SRT E8 lattice structure, 240 roots
    """
    from syntonic.srt.lattice import e8_lattice

    values = tensor.to_floats()

    # Pad or truncate to 8 dimensions for E8 projection
    if len(values) >= 8:
        tensor_8d = values[:8]
    else:
        tensor_8d = values + [0.0] * (8 - len(values))

    # Compute tensor norm
    tensor_norm = math.sqrt(sum(x * x for x in tensor_8d))
    if tensor_norm < 1e-10:
        return 0.0

    # Get E8 lattice
    e8 = e8_lattice()

    # Find maximum alignment with any E8 root
    max_alignment = 0.0
    for root in e8.roots:
        # Get root coordinates as floats
        root_coords = root.to_list()

        # Compute dot product
        dot = sum(a * b for a, b in zip(tensor_8d, root_coords))

        # Root norm is always sqrt(2) for E8 roots
        root_norm = root.norm

        # Cosine similarity
        alignment = abs(dot) / (tensor_norm * root_norm + 1e-10)
        max_alignment = max(max_alignment, alignment)

    return max_alignment


def compute_tensor_norm(tensor: 'ResonantTensor') -> float:
    """
    Compute Frobenius norm of a tensor.

    Helper function replacing torch.norm().

    Args:
        tensor: ResonantTensor to compute norm of

    Returns:
        Frobenius norm (sqrt of sum of squared elements)
    """
    values = tensor.to_floats()
    return math.sqrt(sum(x * x for x in values))


__all__ = [
    'hooking_coefficient',
    'golden_resonance',
    'e8_root_alignment',
    'compute_tensor_norm',
]
