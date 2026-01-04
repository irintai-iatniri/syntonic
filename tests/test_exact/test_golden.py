"""Tests for GoldenExact exact arithmetic over Q(φ)."""

import math
import pytest
import syntonic as syn
from syntonic.exact import (
    GoldenExact,
    PHI,
    PHI_SQUARED,
    PHI_INVERSE,
    PHI_NUMERIC,
    golden_number,
)


class TestGoldenExactConstruction:
    """Tests for GoldenExact creation."""

    def test_phi_constructor(self):
        """Test creating golden ratio φ."""
        phi = GoldenExact.golden_ratio()
        # φ ≈ 1.618033988749895
        assert phi.eval() == pytest.approx(1.618033988749895, rel=1e-14)

    def test_phi_squared_constructor(self):
        """Test creating φ²."""
        phi_sq = GoldenExact.golden_squared()
        # φ² ≈ 2.618033988749895
        assert phi_sq.eval() == pytest.approx(2.618033988749895, rel=1e-14)

    def test_phi_inverse_constructor(self):
        """Test creating 1/φ = φ - 1."""
        phi_inv = GoldenExact.coherence_parameter()
        # 1/φ ≈ 0.618033988749895
        assert phi_inv.eval() == pytest.approx(0.618033988749895, rel=1e-14)

    def test_from_integers(self):
        """Test creating a + b·φ from integers."""
        g = GoldenExact.from_integers(3, 2)  # 3 + 2φ
        expected = 3 + 2 * 1.618033988749895
        assert g.eval() == pytest.approx(expected, rel=1e-14)

    def test_golden_number_helper(self):
        """Test golden_number() helper function."""
        g = golden_number(1, 2)  # 1 + 2φ
        expected = 1 + 2 * 1.618033988749895
        assert g.eval() == pytest.approx(expected, rel=1e-14)


class TestGoldenExactIdentities:
    """Tests for fundamental golden ratio identities."""

    def test_phi_squared_equals_phi_plus_one(self):
        """Test the defining identity: φ² = φ + 1."""
        phi_sq = PHI * PHI
        one = GoldenExact.from_integers(1, 0)
        phi_plus_one = PHI + one

        # Both should evaluate to same value
        assert phi_sq.eval() == pytest.approx(phi_plus_one.eval(), rel=1e-14)

        # And both should equal 2.618...
        assert phi_sq.eval() == pytest.approx(2.618033988749895, rel=1e-14)

    def test_phi_inverse_times_phi_equals_one(self):
        """Test that φ × (1/φ) = 1."""
        product = PHI * PHI_INVERSE
        assert product.eval() == pytest.approx(1.0, rel=1e-14)

    def test_phi_minus_phi_inverse_equals_one(self):
        """Test that φ - 1/φ = 1."""
        diff = PHI - PHI_INVERSE
        assert diff.eval() == pytest.approx(1.0, rel=1e-14)

    def test_phi_inverse_equals_phi_minus_one(self):
        """Test that 1/φ = φ - 1."""
        one = GoldenExact.from_integers(1, 0)
        phi_minus_one = PHI - one
        assert PHI_INVERSE.eval() == pytest.approx(phi_minus_one.eval(), rel=1e-14)


class TestGoldenExactArithmetic:
    """Tests for GoldenExact arithmetic operations."""

    def test_addition(self):
        """Test (a + bφ) + (c + dφ) = (a+c) + (b+d)φ."""
        g1 = golden_number(1, 2)  # 1 + 2φ
        g2 = golden_number(3, 4)  # 3 + 4φ
        result = g1 + g2          # 4 + 6φ
        expected = golden_number(4, 6)
        assert result.eval() == pytest.approx(expected.eval(), rel=1e-14)

    def test_subtraction(self):
        """Test (a + bφ) - (c + dφ) = (a-c) + (b-d)φ."""
        g1 = golden_number(5, 6)  # 5 + 6φ
        g2 = golden_number(2, 3)  # 2 + 3φ
        result = g1 - g2          # 3 + 3φ
        expected = golden_number(3, 3)
        assert result.eval() == pytest.approx(expected.eval(), rel=1e-14)

    def test_multiplication(self):
        """Test multiplication using φ² = φ + 1."""
        # (1 + φ) × (1 + φ) = 1 + 2φ + φ² = 1 + 2φ + φ + 1 = 2 + 3φ
        g = golden_number(1, 1)  # 1 + φ
        result = g * g
        expected = golden_number(2, 3)  # 2 + 3φ
        assert result.eval() == pytest.approx(expected.eval(), rel=1e-14)

    def test_negation(self):
        """Test negation -(a + bφ) = -a - bφ."""
        g = golden_number(3, 5)
        neg_g = -g
        assert neg_g.eval() == pytest.approx(-g.eval(), rel=1e-14)


class TestGoldenExactPowers:
    """Tests for golden ratio powers."""

    def test_phi_to_power_positive(self):
        """Test φⁿ for positive n."""
        # φ¹ = φ
        phi_1 = PHI.phi_to_power(1)
        assert phi_1.eval() == pytest.approx(PHI.eval(), rel=1e-14)

        # φ² = φ + 1
        phi_2 = PHI.phi_to_power(2)
        assert phi_2.eval() == pytest.approx(2.618033988749895, rel=1e-14)

        # φ³ = φ² × φ = (φ+1) × φ = φ² + φ = 2φ + 1
        phi_3 = PHI.phi_to_power(3)
        expected = 2 * 1.618033988749895 + 1
        assert phi_3.eval() == pytest.approx(expected, rel=1e-14)

    def test_phi_to_power_zero(self):
        """Test φ⁰ = 1."""
        phi_0 = PHI.phi_to_power(0)
        assert phi_0.eval() == pytest.approx(1.0, rel=1e-14)

    def test_phi_to_power_negative(self):
        """Test φ⁻¹ = 1/φ = φ - 1."""
        phi_neg1 = PHI.phi_to_power(-1)
        assert phi_neg1.eval() == pytest.approx(0.618033988749895, rel=1e-14)

    def test_phi_power_recurrence(self):
        """Test φⁿ = φⁿ⁻¹ + φⁿ⁻² (Fibonacci recurrence)."""
        for n in range(2, 10):
            phi_n = PHI.phi_to_power(n)
            phi_n_1 = PHI.phi_to_power(n - 1)
            phi_n_2 = PHI.phi_to_power(n - 2)

            expected = phi_n_1.eval() + phi_n_2.eval()
            assert phi_n.eval() == pytest.approx(expected, rel=1e-12)


class TestGoldenExactCoefficients:
    """Tests for extracting rational and phi coefficients."""

    def test_phi_coefficient_for_phi(self):
        """Test that φ = 0 + 1·φ has phi coefficient 1."""
        num, denom = PHI.phi_coefficient
        assert num / denom == 1

    def test_rational_coefficient_for_phi(self):
        """Test that φ = 0 + 1·φ has rational coefficient 0."""
        num, denom = PHI.rational_coefficient
        assert num / denom == 0

    def test_coefficients_for_golden_number(self):
        """Test coefficients for 3 + 5φ."""
        g = golden_number(3, 5)

        a_num, a_denom = g.rational_coefficient
        b_num, b_denom = g.phi_coefficient

        assert a_num / a_denom == 3
        assert b_num / b_denom == 5


class TestGoldenExactConversion:
    """Tests for GoldenExact conversion and comparison."""

    def test_to_f64_accuracy(self):
        """Test that eval() gives accurate float approximation."""
        # φ should be accurate to ~15 decimal places
        phi_approx = (1 + math.sqrt(5)) / 2
        assert PHI.eval() == pytest.approx(phi_approx, rel=1e-15)

    def test_galois_conjugate(self):
        """Test Galois conjugate: (a + bφ)' = a + bφ' where φ' = 1 - φ."""
        if hasattr(PHI, 'galois_conjugate'):
            phi_conj = PHI.galois_conjugate()
            # φ' = (1 - √5)/2 ≈ -0.618...
            expected = (1 - math.sqrt(5)) / 2
            assert phi_conj.eval() == pytest.approx(expected, rel=1e-14)
        else:
            pytest.skip("galois_conjugate not implemented")

    def test_field_norm(self):
        """Test field norm N(a + bφ) = (a + bφ)(a + bφ') = a² + ab - b²."""
        if hasattr(PHI, 'field_norm'):
            # N(φ) = φ × φ' = -1
            # field_norm returns (numerator, denominator) tuple
            num, denom = PHI.field_norm()
            norm = num / denom
            assert norm == pytest.approx(-1.0, rel=1e-14)
        else:
            pytest.skip("field_norm not implemented")


class TestModuleConstants:
    """Tests for module-level constants."""

    def test_PHI_matches_numeric(self):
        """Test that PHI.eval() matches PHI_NUMERIC."""
        assert PHI.eval() == pytest.approx(PHI_NUMERIC, rel=1e-15)

    def test_PHI_SQUARED_value(self):
        """Test PHI_SQUARED module constant."""
        assert PHI_SQUARED.eval() == pytest.approx(2.618033988749895, rel=1e-14)

    def test_PHI_INVERSE_value(self):
        """Test PHI_INVERSE module constant."""
        assert PHI_INVERSE.eval() == pytest.approx(0.618033988749895, rel=1e-14)
