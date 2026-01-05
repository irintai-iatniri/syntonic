"""Tests for LLL-based crystallization and golden lattice snapping."""

import pytest
import math

# Import from syntonic core module
from syntonic._core import GoldenExact

# Golden ratio constant
PHI = 1.6180339887498949


class TestGoldenExactNearest:
    """Tests for GoldenExact.nearest() LLL approximation."""

    def test_nearest_exact_phi(self):
        """φ should snap to exactly (0, 1)."""
        result = GoldenExact.nearest(PHI, 100)
        a_num, a_denom = result.rational_coefficient
        b_num, b_denom = result.phi_coefficient

        assert a_num == 0, f"Expected a=0, got {a_num}/{a_denom}"
        assert b_num / b_denom == 1, f"Expected b=1, got {b_num}/{b_denom}"

    def test_nearest_exact_phi_squared(self):
        """φ² should snap to (1, 1)."""
        phi_sq = PHI * PHI
        result = GoldenExact.nearest(phi_sq, 100)

        # Should be 1 + φ
        expected = GoldenExact.from_integers(1, 1)
        assert result.eval() == pytest.approx(expected.eval(), rel=1e-10)

    def test_nearest_exact_phi_cubed(self):
        """φ³ should snap to (1, 2)."""
        phi_cubed = PHI ** 3
        result = GoldenExact.nearest(phi_cubed, 100)

        # Should be 1 + 2φ (from Fibonacci recurrence)
        expected = GoldenExact.from_integers(1, 2)
        assert result.eval() == pytest.approx(expected.eval(), rel=1e-10)

    def test_nearest_integers(self):
        """Integers should snap to themselves."""
        for n in [0, 1, 2, 5, 10, -3]:
            result = GoldenExact.nearest(float(n), 100)
            a_num, a_denom = result.rational_coefficient
            b_num, b_denom = result.phi_coefficient

            assert a_num / a_denom == pytest.approx(n, abs=0.01)
            assert b_num == 0, f"Expected b=0 for integer {n}"

    def test_nearest_error_bound(self):
        """Approximation error should be small for reasonable max_coeff."""
        test_values = [0.5, 1.23, 2.5, 3.14159, 10.0, -2.7]

        for x in test_values:
            result = GoldenExact.nearest(x, 1000)
            error = result.error_from(x)
            assert error < 0.01, f"Error {error} too large for x={x}"

    def test_nearest_fibonacci_ratios(self):
        """Fibonacci ratios should be well-approximated."""
        fib_ratios = [
            (2, 1),   # 2.0
            (3, 2),   # 1.5
            (5, 3),   # 1.667
            (8, 5),   # 1.6
            (13, 8),  # 1.625
            (21, 13), # 1.615
        ]

        for num, denom in fib_ratios:
            ratio = num / denom
            result = GoldenExact.nearest(ratio, 100)
            error = result.error_from(ratio)
            assert error < 0.01, f"Error {error} for ratio {num}/{denom}"


class TestGoldenExactSnap:
    """Tests for GoldenExact.snap() batch operation."""

    def test_snap_basic(self):
        """Test basic snap operation."""
        values = [1.0, PHI, PHI * PHI, 3.0]
        lattice, residuals = GoldenExact.snap(values, 100)

        assert len(lattice) == 4
        assert len(residuals) == 4

    def test_snap_residuals_small(self):
        """Residuals should be small for well-approximated values."""
        values = [1.0, PHI, PHI * PHI, 3.0]
        lattice, residuals = GoldenExact.snap(values, 100)

        for i, r in enumerate(residuals):
            assert abs(r) < 0.01, f"Residual {r} too large at index {i}"

    def test_snap_golden_values(self):
        """Golden ratio multiples should snap exactly."""
        values = [PHI, 2 * PHI, PHI * PHI]
        lattice, residuals = GoldenExact.snap(values, 100)

        # Check phi snaps to phi
        assert lattice[0].eval() == pytest.approx(PHI, rel=1e-10)

        # Check 2*phi snaps to 2*phi = 0 + 2*phi
        assert lattice[1].eval() == pytest.approx(2 * PHI, rel=1e-10)

        # Check phi^2 snaps to 1 + phi
        assert lattice[2].eval() == pytest.approx(PHI * PHI, rel=1e-10)

    def test_snap_preserves_order(self):
        """Snap should preserve relative order when possible."""
        values = [1.0, 2.0, 3.0, 4.0]
        lattice, _ = GoldenExact.snap(values, 100)

        # Converted values should be in ascending order
        converted = [g.eval() for g in lattice]
        for i in range(len(converted) - 1):
            assert converted[i] < converted[i + 1]


class TestGoldenExactRational:
    """Tests for GoldenExact.nearest_rational() higher precision."""

    def test_nearest_rational_higher_precision(self):
        """Rational version should give better precision."""
        x = math.sqrt(2)  # Irrational, not in Q(phi)

        # Integer coefficients
        result_int = GoldenExact.nearest(x, 100)
        error_int = result_int.error_from(x)

        # Rational coefficients
        result_rat = GoldenExact.nearest_rational(x, 100)
        error_rat = result_rat.error_from(x)

        # Rational should be at least as good
        assert error_rat <= error_int + 1e-10


class TestGoldenExactError:
    """Tests for GoldenExact.error_from() method."""

    def test_error_from_exact(self):
        """Error should be zero for exact representations."""
        g = GoldenExact.golden_ratio()
        error = g.error_from(PHI)
        assert error < 1e-14

    def test_error_from_approx(self):
        """Error should be positive for approximations."""
        g = GoldenExact.from_integers(1, 0)  # Just 1.0
        error = g.error_from(PHI)
        assert error == pytest.approx(PHI - 1.0, rel=1e-10)
