"""
Prime Selection Rules - Python Wrapper

Re-exports the Python implementations for Fermat, Mersenne, and Lucas
prime-selection utilities so the rest of the codebase can import
these functions from `syntonic.srt.prime_selection`.
"""

# Fermat primes / force spectrum
from syntonic.srt.fermat_forces import (
    fermat_number,
    is_fermat_prime,
    get_force_spectrum,
    validate_force_existence,
)

# Mersenne primes / generation spectrum
from syntonic.srt.mersenne_matter import (
    mersenne_number,
    is_mersenne_prime,
    get_generation_spectrum,
    generation_barrier_explanation,
    validate_generation_stability,
)

# Lucas numbers / dark sector
from syntonic.srt.lucas_shadow import (
    lucas_number,
    shadow_phase,
    is_lucas_prime,
    dark_matter_mass_prediction,
    get_shadow_spectrum,
)

__all__ = [
    # Fermat
    "fermat_number",
    "is_fermat_prime",
    "get_force_spectrum",
    "validate_force_existence",
    # Mersenne
    "mersenne_number",
    "is_mersenne_prime",
    "get_generation_spectrum",
    "generation_barrier_explanation",
    "validate_generation_stability",
    # Lucas
    "lucas_number",
    "shadow_phase",
    "is_lucas_prime",
    "dark_matter_mass_prediction",
    "get_shadow_spectrum",
]
