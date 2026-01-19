"""
Fermat Force Spectrum

Maps Fermat primes to fundamental forces.
"""

from syntonic.srt.prime_selection import (
    FERMAT_PRIMES,
    get_force_spectrum,
    is_fermat_prime,
)

FORCE_NAMES = ["Strong", "Electroweak", "Dark Boundary", "Gravity", "Versal"]


def get_force_prime(force_index: int) -> int:
    """Get Fermat prime for force index (0-4)."""
    return FERMAT_PRIMES[force_index]


def is_valid_gauge_force(n: int) -> bool:
    """Check if n corresponds to a valid gauge force."""
    return is_fermat_prime(n)


def force_spectrum_table() -> list:
    """Get full force spectrum table."""
    return get_force_spectrum()


__all__ = [
    "FORCE_NAMES",
    "get_force_prime",
    "is_valid_gauge_force",
    "force_spectrum_table",
]
