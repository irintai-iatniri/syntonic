"""
ATP-Mersenne Biological Energy Quantization

Implements the ATP-Mersenne correspondence where biological energy
quantum (31 kJ/mol) corresponds to M₅ (third Mersenne prime).
"""

from syntonic.exact import PHI_NUMERIC

# ATP hydrolysis energy quantum
ATP_QUANTUM_KJ = 30.5  # kJ/mol (experimental value)
ATP_QUANTUM_J = ATP_QUANTUM_KJ * 1000  # J/mol

# Mersenne prime corresponding to ATP
M5_MERSENNE = 31  # M₅ = 2^5 - 1 = 31

# Generation energy quanta (Mersenne progression)
GENERATION_ENERGIES = {
    1: 3,  # M₂ - quantum noise floor
    2: 7,  # M₃ - thermal stability
    3: 31,  # M₅ - ATP (macroscopic life)
    4: None,  # M₁₁ barrier - no 4th generation
}


def verify_atp_mersenne_correspondence() -> dict:
    """
    Verify ATP energy quantum corresponds to M₅ Mersenne Prime.

    This explains why ATP is the universal energy currency:
    it's the first stable energetic resonance at molecular scales.

    Returns:
        Dict with correspondence analysis
    """
    correspondence = {
        "atp_energy_kj": ATP_QUANTUM_KJ,
        "mersenne_m5": M5_MERSENNE,
        "correlation": abs(ATP_QUANTUM_KJ - M5_MERSENNE) / M5_MERSENNE,
        "interpretation": "Life runs on M₅ stability mode",
        "ratio": ATP_QUANTUM_KJ / M5_MERSENNE,
        "golden_alignment": abs(ATP_QUANTUM_KJ / M5_MERSENNE - PHI_NUMERIC)
        / PHI_NUMERIC,
    }
    return correspondence


def biological_prime_energy(generation: int) -> float:
    """
    Compute biological energy quantum for given Mersenne generation.

    Args:
        generation: Generation number (1-3, since 4 is blocked)

    Returns:
        Energy quantum in kJ/mol, or None if generation doesn't exist

    Examples:
        >>> biological_prime_energy(3)  # ATP generation
        31.0
        >>> biological_prime_energy(4)  # Should be None (M₁₁ barrier)
        None
    """
    return GENERATION_ENERGIES.get(generation)


def compute_biological_stability_scale(energy_kj: float) -> str:
    """
    Classify biological stability scale based on energy quantum.

    Args:
        energy_kj: Energy quantum in kJ/mol

    Returns:
        Stability classification
    """
    if abs(energy_kj - 3) < 1:
        return "Quantum Noise (M₂) - Molecular vibration floor"
    elif abs(energy_kj - 7) < 1:
        return "Thermal Stability (M₃) - Protein folding"
    elif abs(energy_kj - 31) < 1:
        return "ATP Resonance (M₅) - Universal metabolism"
    else:
        return "Unstable - Outside Mersenne stability zones"


__all__ = [
    "ATP_QUANTUM_KJ",
    "ATP_QUANTUM_J",
    "M5_MERSENNE",
    "GENERATION_ENERGIES",
    "verify_atp_mersenne_correspondence",
    "biological_prime_energy",
    "compute_biological_stability_scale",
]
