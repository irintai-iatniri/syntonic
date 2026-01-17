"""
Tests for SRT/CRT Prime Theory Functions

Tests the core number-theoretic functions that implement the SRT/CRT axioms
from The_Grand_Synthesis.md.
"""

import pytest
import math
import syntonic._core as core


class TestPrimeTheory:
    """Test the fundamental prime theory functions."""

    def test_mersenne_primes(self):
        """Test Mersenne prime detection (Axiom 6: Matter Stability)."""
        # Known Mersenne primes (stable matter generations)
        assert core.py_is_mersenne_prime(2) == True   # M2 = 3 (electron generation)
        assert core.py_is_mersenne_prime(3) == True   # M3 = 7 (muon generation)
        assert core.py_is_mersenne_prime(5) == True   # M5 = 31 (tau generation)
        assert core.py_is_mersenne_prime(7) == True   # M7 = 127 (top quark)

        # The barrier - M11 is composite (prevents 4th generation)
        assert core.py_is_mersenne_prime(11) == False  # M11 = 2047 = 23 × 89

        # Non-prime exponents
        assert core.py_is_mersenne_prime(1) == False
        assert core.py_is_mersenne_prime(4) == False

    def test_fermat_primes(self):
        """Test Fermat prime detection (Force Differentiation)."""
        # Known Fermat primes (fundamental forces)
        assert core.py_is_fermat_prime(0) == True   # F0 = 3 (Strong force)
        assert core.py_is_fermat_prime(1) == True   # F1 = 5 (Weak force)
        assert core.py_is_fermat_prime(2) == True   # F2 = 17 (EM/Dark boundary)
        assert core.py_is_fermat_prime(3) == True   # F3 = 257 (Gravity)
        assert core.py_is_fermat_prime(4) == True   # F4 = 65537 (Versal force)

        # F5 is composite (Euler's theorem) - terminates force hierarchy
        assert core.py_is_fermat_prime(5) == False

    def test_lucas_primes(self):
        """Test Lucas prime detection (Dark Sector Stability)."""
        # Known Lucas primes
        assert core.py_is_lucas_prime(2) == True   # L2 = 3
        assert core.py_is_lucas_prime(4) == True   # L4 = 7
        assert core.py_is_lucas_prime(5) == True   # L5 = 11
        assert core.py_is_lucas_prime(7) == True   # L7 = 29
        assert core.py_is_lucas_prime(13) == True  # L13 = 521
        assert core.py_is_lucas_prime(17) == True  # L17 = 3571 (dark matter anchor)

    def test_lucas_numbers(self):
        """Test Lucas number computation."""
        # Basic Lucas numbers
        assert core.py_lucas_number(0) == 2
        assert core.py_lucas_number(1) == 1
        assert core.py_lucas_number(2) == 3
        assert core.py_lucas_number(3) == 4
        assert core.py_lucas_number(4) == 7
        assert core.py_lucas_number(5) == 11

        # Dark matter anchor
        assert core.py_lucas_number(17) == 3571

    def test_pisano_periods(self):
        """Test Pisano period computation (Hooking Cycles)."""
        # Known Pisano periods
        assert core.py_pisano_period(2) == 3
        assert core.py_pisano_period(3) == 8
        assert core.py_pisano_period(5) == 20
        assert core.py_pisano_period(7) == 16

        # Pisano periods should be positive
        assert core.py_pisano_period(11) > 0
        assert core.py_pisano_period(13) > 0

    def test_stable_windings(self):
        """Test winding stability (Axiom 6)."""
        # Stable generations
        assert core.py_is_stable_winding(2) == True   # Electron
        assert core.py_is_stable_winding(3) == True   # Muon
        assert core.py_is_stable_winding(5) == True   # Tau
        assert core.py_is_stable_winding(7) == True   # Top

        # The barrier
        assert core.py_is_stable_winding(11) == False  # No 4th generation

    def test_stability_barrier(self):
        """Test the stability barrier constant."""
        assert core.py_get_stability_barrier() == 11

    def test_transcendence_gates(self):
        """Test Fibonacci prime transcendence gates."""
        # Known transcendence gates
        assert core.py_is_transcendence_gate(3) == True   # F3 = 2
        assert core.py_is_transcendence_gate(4) == True   # F4 = 3 (material anomaly)
        assert core.py_is_transcendence_gate(5) == True   # F5 = 5
        assert core.py_is_transcendence_gate(7) == True   # F7 = 13
        assert core.py_is_transcendence_gate(11) == True  # F11 = 89
        assert core.py_is_transcendence_gate(13) == True  # F13 = 233
        assert core.py_is_transcendence_gate(17) == True  # F17 = 1597 (consciousness)

        # Non-gates
        assert core.py_is_transcendence_gate(1) == False
        assert core.py_is_transcendence_gate(2) == False
        assert core.py_is_transcendence_gate(6) == False

    def test_versal_grip_strength(self):
        """Test versal grip strength computation."""
        # Should return positive values for primes
        strength_5 = core.py_versal_grip_strength(5)
        assert strength_5 >= 0.0

        strength_7 = core.py_versal_grip_strength(7)
        assert strength_7 >= 0.0

    def test_prime_sequences(self):
        """Test prime sequence generation."""
        # Mersenne sequence
        mersenne = core.py_mersenne_sequence(10)
        assert 2 in mersenne  # M2
        assert 3 in mersenne  # M3
        assert 5 in mersenne  # M5
        assert 7 in mersenne  # M7
        assert 11 not in mersenne  # M11 (composite)

        # Fermat sequence
        fermat = core.py_fermat_sequence(4)
        assert 0 in fermat
        assert 1 in fermat
        assert 2 in fermat
        assert 3 in fermat
        assert 4 in fermat
        assert 5 not in fermat  # F5 composite

        # Lucas primes
        lucas = core.py_lucas_primes(20)
        assert 2 in lucas
        assert 4 in lucas
        assert 5 in lucas
        assert 13 in lucas

    def test_lucas_dark_boost(self):
        """Test the Lucas dark matter boost factor."""
        boost = core.py_lucas_dark_boost()
        # L17/L13 ≈ 3571/521 ≈ 6.854
        assert abs(boost - 6.854) < 0.001

    def test_dark_matter_prediction(self):
        """Test dark matter mass prediction."""
        # Using top quark mass as anchor
        top_mass = 172.7  # GeV
        dm_mass = core.py_predict_dark_matter_mass(top_mass)

        # Should be around 1.18 TeV = 1180 GeV
        expected = top_mass * core.py_lucas_dark_boost()
        assert abs(dm_mass - expected) < 1.0


class TestSRTCRTPredictions:
    """Test SRT/CRT theoretical predictions."""

    def test_force_count(self):
        """Test that exactly 5 forces exist."""
        fermat_primes = core.py_fermat_sequence(10)
        assert len(fermat_primes) == 5  # F0 through F4

    def test_matter_generations(self):
        """Test that exactly 4 matter generations exist."""
        stable_windings = []
        for p in range(2, 12):
            if core.py_is_stable_winding(p):
                stable_windings.append(p)

        assert len(stable_windings) == 4  # 2, 3, 5, 7

    def test_stability_barrier_position(self):
        """Test that the barrier prevents 4th generation."""
        barrier = core.py_get_stability_barrier()
        assert barrier == 11

        # Generations before barrier are stable
        for p in [2, 3, 5, 7]:
            assert core.py_is_stable_winding(p)

        # Barrier itself is unstable
        assert not core.py_is_stable_winding(barrier)

    def test_dark_matter_mass_range(self):
        """Test dark matter mass prediction is in expected range."""
        dm_mass = core.py_predict_dark_matter_mass(172.7)  # Top quark mass

        # Should be around 1.1-1.5 TeV as predicted
        assert 1100 <= dm_mass <= 1500

    def test_fibonacci_transcendence_gates(self):
        """Test that Fibonacci primes define transcendence gates."""
        gates = []
        for n in range(20):
            if core.py_is_transcendence_gate(n):
                gates.append(n)

        # Should have the expected gates
        expected_gates = [3, 4, 5, 7, 11, 13, 17]
        assert gates == expected_gates

    def test_pisano_self_hooking(self):
        """Test that primes with π(p) divisible by p have strong grip."""
        # Test a few known cases
        for p in [2, 3, 5, 7, 11]:
            pi = core.py_pisano_period(p)
            grip = core.py_versal_grip_strength(p)

            if pi % p == 0:
                assert grip > 1.0  # Strong grip
            else:
                assert grip == 0.0  # Weak grip


if __name__ == "__main__":
    pytest.main([__file__])