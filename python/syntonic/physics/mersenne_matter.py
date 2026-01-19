"""
Mersenne Matter Stability

Maps Mersenne primes to particle generations.
"""

from syntonic.srt.prime_selection import (
    generation_barrier_explanation,
    mersenne_number,
)


def get_generation_prime(gen: int) -> int:
    """Get Mersenne prime for generation (1-3)."""
    gen_map = {1: 2, 2: 3, 3: 5}
    p = gen_map.get(gen)
    return mersenne_number(p) if p else None


def explain_no_4th_generation() -> str:
    """Explain why 4th generation doesn't exist."""
    return generation_barrier_explanation()


__all__ = [
    "get_generation_prime",
    "explain_no_4th_generation",
]
