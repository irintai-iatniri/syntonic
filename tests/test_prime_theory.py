#!/usr/bin/env python3
"""
Test Prime Theory Functions

Tests for core SRT/CRT prime theory functions and predictions.
"""

import pytest
import sys
import os

# Add syntonic package to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'python'))

try:
    import syntonic._core as core
    HAS_RUST_CORE = True
except ImportError:
    HAS_RUST_CORE = False
    print("Warning: syntonic._core not available, some tests will be skipped")


class TestPrimeTheory:
    """Test SRT/CRT prime theory functions."""

    @pytest.mark.skipif(not HAS_RUST_CORE, reason="Rust core not available")
    def test_mersenne_primes(self):
        """Test Mersenne prime detection."""
        # Known Mersenne primes
        mersenne_primes = [3, 7, 31, 127, 8191]

        for p in mersenne_primes[:3]:  # Test first few
            assert core.py_is_mersenne_prime(p), f"{p} should be Mersenne prime"

        # Non-Mersenne primes
        non_mersenne = [5, 11, 13, 17, 19]
        for p in non_mersenne:
            assert not core.py_is_mersenne_prime(p), f"{p} should not be Mersenne prime"

    @pytest.mark.skipif(not HAS_RUST_CORE, reason="Rust core not available")
    def test_fermat_primes(self):
        """Test Fermat prime detection."""
        # Known Fermat primes
        fermat_primes = [3, 5, 17, 257, 65537]

        for p in fermat_primes[:3]:  # Test first few
            assert core.py_is_fermat_prime(p), f"{p} should be Fermat prime"

        # Non-Fermat primes
        non_fermat = [7, 11, 13, 19, 23]
        for p in non_fermat:
            assert not core.py_is_fermat_prime(p), f"{p} should not be Fermat prime"

    @pytest.mark.skipif(not HAS_RUST_CORE, reason="Rust core not available")
    def test_lucas_numbers(self):
        """Test Lucas number computation."""
        # Known Lucas numbers: L(n) = F(n-1) + F(n+1)
        expected = [2, 1, 3, 4, 7, 11, 18, 29, 47, 76]

        for n, exp in enumerate(expected, 1):
            assert core.py_lucas_number(n) == exp, f"L({n}) should be {exp}"

    @pytest.mark.skipif(not HAS_RUST_CORE, reason="Rust core not available")
    def test_pisano_periods(self):
        """Test Pisano period computation."""
        # Known Pisano periods
        test_cases = [
            (2, 3),   # mod 2: period 3
            (3, 8),   # mod 3: period 8
            (5, 20),  # mod 5: period 20
            (7, 16),  # mod 7: period 16
        ]

        for mod, expected_period in test_cases:
            period = core.py_pisano_period(mod)
            assert period == expected_period, f"Pisano period mod {mod} should be {expected_period}"

    @pytest.mark.skipif(not HAS_RUST_CORE, reason="Rust core not available")
    def test_stability_checks(self):
        """Test winding stability checks."""
        # Stable primes (should return True)
        stable_primes = [2, 3, 5, 7]

        for p in stable_primes:
            assert core.py_is_stable_winding(p), f"Prime {p} should be stable"

        # Unstable primes (should return False)
        unstable_primes = [11, 13, 17, 19, 23]

        for p in unstable_primes:
            assert not core.py_is_stable_winding(p), f"Prime {p} should be unstable"

    @pytest.mark.skipif(not HAS_RUST_CORE, reason="Rust core not available")
    def test_transcendence_gates(self):
        """Test Fibonacci transcendence gates."""
        # Test first 20 Fibonacci numbers for transcendence gates
        gates_found = 0

        for n in range(20):
            if core.py_is_transcendence_gate(n):
                gates_found += 1

        # Should find at least 7 transcendence gates
        assert gates_found >= 7, f"Should find at least 7 transcendence gates, found {gates_found}"

    @pytest.mark.skipif(not HAS_RUST_CORE, reason="Rust core not available")
    def test_stability_barrier(self):
        """Test stability barrier prediction."""
        barrier = core.py_get_stability_barrier()

        # SRT predicts stability barrier at p=11
        assert barrier == 11, f"Stability barrier should be 11, got {barrier}"

    @pytest.mark.skipif(not HAS_RUST_CORE, reason="Rust core not available")
    def test_dark_matter_prediction(self):
        """Test dark matter mass prediction."""
        # Using standard Higgs mass
        higgs_mass = 125.1  # GeV
        dm_mass = core.py_predict_dark_matter_mass(higgs_mass)

        # SRT predicts ~1.18 TeV dark matter
        expected_min = 1100  # GeV
        expected_max = 1500  # GeV

        assert expected_min <= dm_mass <= expected_max, \
            f"Dark matter mass {dm_mass} GeV should be between {expected_min}-{expected_max} GeV"

    @pytest.mark.skipif(not HAS_RUST_CORE, reason="Rust core not available")
    def test_lucas_dark_boost(self):
        """Test Lucas dark boost factor."""
        boost = core.py_lucas_dark_boost()

        # Expected value from SRT theory
        expected = 6.854
        tolerance = 0.001

        assert abs(boost - expected) < tolerance, \
            f"Lucas boost {boost} should be close to {expected}"

    @pytest.mark.skipif(not HAS_RUST_CORE, reason="Rust core not available")
    def test_fermat_sequence(self):
        """Test Fermat prime sequence generation."""
        sequence = core.py_fermat_sequence(10)

        # Should have exactly 5 Fermat primes
        assert len(sequence) == 5, f"Should have exactly 5 Fermat primes, got {len(sequence)}"

        # First few should be correct
        expected_first = [3, 5, 17, 257, 65537]
        assert sequence[:5] == expected_first, f"First 5 Fermat primes should be {expected_first}"

    @pytest.mark.skipif(not HAS_RUST_CORE, reason="Rust core not available")
    def test_force_count_prediction(self):
        """Test that Fermat sequence predicts exactly 5 forces."""
        fermat_primes = core.py_fermat_sequence(10)

        # SRT predicts exactly 5 fundamental forces
        assert len(fermat_primes) == 5, f"SRT predicts exactly 5 forces, got {len(fermat_primes)}"

    @pytest.mark.skipif(not HAS_RUST_CORE, reason="Rust core not available")
    def test_matter_generations_prediction(self):
        """Test that stable windings predict exactly 4 matter generations."""
        stable_count = sum(1 for p in range(2, 12) if core.py_is_stable_winding(p))

        # SRT predicts exactly 4 matter generations
        assert stable_count == 4, f"SRT predicts exactly 4 matter generations, got {stable_count}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])