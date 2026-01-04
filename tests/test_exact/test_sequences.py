"""Tests for Fibonacci, Lucas sequences and SRT correction factors."""

import math
import pytest
import syntonic as syn
from syntonic.exact import (
    fibonacci,
    lucas,
    correction_factor,
    E_STAR_NUMERIC,
    Q_DEFICIT_NUMERIC,
    STRUCTURE_DIMENSIONS,
)


class TestFibonacci:
    """Tests for Fibonacci sequence computation."""

    def test_fibonacci_base_case_0(self):
        """Test F(0) = 0."""
        assert fibonacci(0) == 0

    def test_fibonacci_base_case_1(self):
        """Test F(1) = 1."""
        assert fibonacci(1) == 1

    def test_fibonacci_small_values(self):
        """Test first several Fibonacci numbers."""
        expected = [0, 1, 1, 2, 3, 5, 8, 13, 21, 34, 55]
        for n, fib_n in enumerate(expected):
            assert fibonacci(n) == fib_n, f"F({n}) should be {fib_n}"

    def test_fibonacci_10(self):
        """Test F(10) = 55."""
        assert fibonacci(10) == 55

    def test_fibonacci_20(self):
        """Test F(20) = 6765."""
        assert fibonacci(20) == 6765

    def test_fibonacci_large(self):
        """Test large Fibonacci numbers."""
        # F(50) = 12586269025
        assert fibonacci(50) == 12586269025

    def test_fibonacci_recurrence(self):
        """Test F(n) = F(n-1) + F(n-2)."""
        for n in range(2, 20):
            assert fibonacci(n) == fibonacci(n - 1) + fibonacci(n - 2)

    def test_fibonacci_negative_raises(self):
        """Test that negative input raises ValueError."""
        with pytest.raises(ValueError):
            fibonacci(-1)


class TestLucas:
    """Tests for Lucas sequence computation."""

    def test_lucas_base_case_0(self):
        """Test L(0) = 2."""
        assert lucas(0) == 2

    def test_lucas_base_case_1(self):
        """Test L(1) = 1."""
        assert lucas(1) == 1

    def test_lucas_small_values(self):
        """Test first several Lucas numbers."""
        expected = [2, 1, 3, 4, 7, 11, 18, 29, 47, 76, 123]
        for n, lucas_n in enumerate(expected):
            assert lucas(n) == lucas_n, f"L({n}) should be {lucas_n}"

    def test_lucas_10(self):
        """Test L(10) = 123."""
        assert lucas(10) == 123

    def test_lucas_20(self):
        """Test L(20) = 15127."""
        assert lucas(20) == 15127

    def test_lucas_recurrence(self):
        """Test L(n) = L(n-1) + L(n-2)."""
        for n in range(2, 20):
            assert lucas(n) == lucas(n - 1) + lucas(n - 2)

    def test_lucas_negative_raises(self):
        """Test that negative input raises ValueError."""
        with pytest.raises(ValueError):
            lucas(-1)


class TestFibonacciLucasRelations:
    """Tests for relationships between Fibonacci and Lucas numbers."""

    def test_lucas_equals_fib_sum(self):
        """Test L(n) = F(n-1) + F(n+1) for n >= 1."""
        for n in range(1, 15):
            assert lucas(n) == fibonacci(n - 1) + fibonacci(n + 1)

    def test_fib_lucas_identity(self):
        """Test F(2n) = F(n) × L(n)."""
        for n in range(1, 10):
            assert fibonacci(2 * n) == fibonacci(n) * lucas(n)


class TestCorrectionFactor:
    """Tests for SRT correction factors (1 ± q/N)."""

    def test_correction_factor_e8_positive(self):
        """Test correction factor for E8 positive roots (chiral suppression)."""
        factor = correction_factor('E8_positive', -1)
        # (1 - q/120) where q ≈ 0.027395
        expected = 1 - Q_DEFICIT_NUMERIC / 120
        assert factor == pytest.approx(expected, rel=1e-10)

    def test_correction_factor_e6_positive(self):
        """Test correction factor for E6 positive roots (intra-generation)."""
        factor = correction_factor('E6_positive', +1)
        # (1 + q/36)
        expected = 1 + Q_DEFICIT_NUMERIC / 36
        assert factor == pytest.approx(expected, rel=1e-10)

    def test_correction_factor_d4_kissing(self):
        """Test correction factor for D4 kissing number."""
        factor = correction_factor('D4_kissing', +1)
        # (1 + q/24)
        expected = 1 + Q_DEFICIT_NUMERIC / 24
        assert factor == pytest.approx(expected, rel=1e-10)

    def test_correction_factor_all_structures(self):
        """Test that all valid structures work."""
        structures = [
            'E8_dim', 'E8_roots', 'E8_positive',
            'E6_dim', 'E6_positive', 'E6_fundamental',
            'D4_kissing', 'G2_dim'
        ]
        for struct in structures:
            # Should not raise
            plus = correction_factor(struct, +1)
            minus = correction_factor(struct, -1)
            # Plus should be > 1, minus should be < 1
            assert plus > 1.0
            assert minus < 1.0

    def test_correction_factor_invalid_structure(self):
        """Test that invalid structure raises ValueError."""
        with pytest.raises(ValueError):
            correction_factor('invalid_structure', +1)


class TestSRTConstants:
    """Tests for SRT-specific constants."""

    def test_e_star_numeric(self):
        """Test E* = e^π - π ≈ 20."""
        expected = math.exp(math.pi) - math.pi
        assert E_STAR_NUMERIC == pytest.approx(expected, rel=1e-14)
        # Also verify it's close to 20
        assert E_STAR_NUMERIC == pytest.approx(20.0, rel=5e-5)

    def test_q_deficit_numeric(self):
        """Test syntony deficit q ≈ 0.027395."""
        # The exact value from SRT
        assert Q_DEFICIT_NUMERIC == pytest.approx(0.027395146920, rel=1e-8)

    def test_structure_dimensions(self):
        """Test that STRUCTURE_DIMENSIONS contains correct values."""
        assert STRUCTURE_DIMENSIONS['E8_dim'] == 248
        assert STRUCTURE_DIMENSIONS['E8_roots'] == 240
        assert STRUCTURE_DIMENSIONS['E8_positive'] == 120
        assert STRUCTURE_DIMENSIONS['E6_dim'] == 78
        assert STRUCTURE_DIMENSIONS['E6_positive'] == 36
        assert STRUCTURE_DIMENSIONS['E6_fundamental'] == 27
        assert STRUCTURE_DIMENSIONS['D4_kissing'] == 24
        assert STRUCTURE_DIMENSIONS['G2_dim'] == 14


class TestGoldenPowerFibonacciConnection:
    """Tests connecting golden powers to Fibonacci/Lucas."""

    def test_phi_power_fibonacci_relation(self):
        """Test that φⁿ = F(n-1) + F(n)·φ (Binet form)."""
        phi = syn.PHI
        for n in range(2, 15):
            phi_n = phi.phi_to_power(n)

            # F(n-1) is rational part, F(n) is phi part
            a_num, a_denom = phi_n.rational_coefficient
            b_num, b_denom = phi_n.phi_coefficient

            # Check coefficients match Fibonacci numbers
            assert a_num // a_denom == fibonacci(n - 1), f"n={n}: rational part mismatch"
            assert b_num // b_denom == fibonacci(n), f"n={n}: phi part mismatch"

    def test_phi_power_lucas_relation(self):
        """Test that φⁿ + (φ')ⁿ = L(n) (Lucas number)."""
        phi = syn.PHI
        phi_conjugate = (1 - math.sqrt(5)) / 2  # φ'

        for n in range(0, 15):
            # Using numeric approximation
            phi_n = phi.phi_to_power(n).eval()
            phi_conj_n = phi_conjugate ** n

            # Sum should equal Lucas number
            sum_val = phi_n + phi_conj_n
            assert sum_val == pytest.approx(lucas(n), rel=1e-10)
