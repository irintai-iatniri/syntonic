"""Tests for Octonion implementation."""

import math
import pytest
import syntonic as syn
from syntonic.hypercomplex import (
    Octonion, octonion,
    E0, E1, E2, E3, E4, E5, E6, E7,
)


def ocomponents(o):
    """Helper to get octonion components as list [e0, e1, ..., e7]."""
    return [o.e0, o.e1, o.e2, o.e3, o.e4, o.e5, o.e6, o.e7]


class TestOctonionConstruction:
    """Tests for octonion creation."""

    def test_basic_construction(self):
        """Test creating an octonion directly."""
        o = Octonion(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0)
        components = ocomponents(o)
        assert len(components) == 8
        assert components == pytest.approx([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])

    def test_from_factory_function(self):
        """Test creating octonion via factory function."""
        o = octonion(1, 2, 3, 4, 5, 6, 7, 8)
        assert ocomponents(o) == pytest.approx([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])

    def test_unit_octonion_E0(self):
        """Test unit octonion E0 = 1."""
        expected = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        assert ocomponents(E0) == pytest.approx(expected)

    def test_unit_octonions_basis(self):
        """Test all unit octonion basis elements."""
        basis = [E0, E1, E2, E3, E4, E5, E6, E7]
        for i, ei in enumerate(basis):
            expected = [0.0] * 8
            expected[i] = 1.0
            assert ocomponents(ei) == pytest.approx(expected), f"E{i} incorrect"


class TestOctonionProperties:
    """Tests for octonion properties."""

    def test_real_part(self):
        """Test octonion real part extraction."""
        o = octonion(3, 1, 2, 3, 4, 5, 6, 7)
        assert o.real == pytest.approx(3.0)

    def test_norm(self):
        """Test octonion norm |o| = sqrt(sum of squares)."""
        o = octonion(1, 2, 3, 4, 5, 6, 7, 8)
        # sqrt(1 + 4 + 9 + 16 + 25 + 36 + 49 + 64) = sqrt(204)
        expected = math.sqrt(1 + 4 + 9 + 16 + 25 + 36 + 49 + 64)
        assert o.norm() == pytest.approx(expected)

    def test_norm_unit(self):
        """Test that unit octonions have norm 1."""
        for ei in [E0, E1, E2, E3, E4, E5, E6, E7]:
            assert ei.norm() == pytest.approx(1.0)

    def test_conjugate(self):
        """Test octonion conjugate o* = e0 - sum(ei*ei)."""
        o = octonion(1, 2, 3, 4, 5, 6, 7, 8)
        o_conj = o.conjugate()
        expected = [1.0, -2.0, -3.0, -4.0, -5.0, -6.0, -7.0, -8.0]
        assert ocomponents(o_conj) == pytest.approx(expected)

    def test_inverse(self):
        """Test octonion inverse o⁻¹ = o* / |o|²."""
        o = octonion(1, 2, 0, 0, 0, 0, 0, 0)
        o_inv = o.inverse()
        # o * o_inv should be close to 1
        product = o * o_inv
        assert product.real == pytest.approx(1.0, abs=1e-10)
        # Imaginary parts should be close to zero
        components = ocomponents(product)
        for i in range(1, 8):
            assert components[i] == pytest.approx(0.0, abs=1e-10)


class TestOctonionArithmetic:
    """Tests for octonion arithmetic operations."""

    def test_addition(self):
        """Test octonion addition."""
        o1 = octonion(1, 2, 3, 4, 5, 6, 7, 8)
        o2 = octonion(8, 7, 6, 5, 4, 3, 2, 1)
        result = o1 + o2
        assert ocomponents(result) == pytest.approx([9.0] * 8)

    def test_subtraction(self):
        """Test octonion subtraction."""
        o1 = octonion(10, 10, 10, 10, 10, 10, 10, 10)
        o2 = octonion(1, 2, 3, 4, 5, 6, 7, 8)
        result = o1 - o2
        expected = [9.0, 8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0]
        assert ocomponents(result) == pytest.approx(expected)

    def test_negation(self):
        """Test octonion negation."""
        o = octonion(1, 2, 3, 4, 5, 6, 7, 8)
        neg_o = -o
        expected = [-1.0, -2.0, -3.0, -4.0, -5.0, -6.0, -7.0, -8.0]
        assert ocomponents(neg_o) == pytest.approx(expected)

    def test_scalar_multiplication(self):
        """Test octonion scalar multiplication."""
        o = octonion(1, 2, 3, 4, 5, 6, 7, 8)
        # Multiply by creating a scalar octonion
        scalar = octonion(2.0, 0, 0, 0, 0, 0, 0, 0)
        result = scalar * o
        expected = [2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0]
        assert ocomponents(result) == pytest.approx(expected)


class TestOctonionNonAssociativity:
    """Tests verifying octonions are non-associative."""

    def test_non_associativity(self):
        """Test that (a*b)*c != a*(b*c) in general for octonions."""
        # Using basis elements that are known to violate associativity
        a = E1
        b = E2
        c = E4

        left = (a * b) * c
        right = a * (b * c)

        left_components = ocomponents(left)
        right_components = ocomponents(right)

        # They should NOT be equal
        is_different = any(
            abs(l - r) > 1e-10 for l, r in zip(left_components, right_components)
        )
        assert is_different, "Octonions should be non-associative"

    def test_associator_nonzero(self):
        """Test that the associator [a,b,c] = (ab)c - a(bc) is nonzero."""
        a = E1
        b = E2
        c = E4
        left = (a * b) * c
        right = a * (b * c)
        diff = left - right
        assert diff.norm() > 1e-10

    def test_alternativity_left(self):
        """Test left alternativity: (a*a)*b = a*(a*b)."""
        a = octonion(1, 2, 0, 0, 0, 0, 0, 0)
        b = octonion(0, 0, 1, 2, 0, 0, 0, 0)

        left = (a * a) * b
        right = a * (a * b)

        assert ocomponents(left) == pytest.approx(ocomponents(right), abs=1e-10)

    def test_alternativity_right(self):
        """Test right alternativity: (a*b)*b = a*(b*b)."""
        a = octonion(1, 2, 0, 0, 0, 0, 0, 0)
        b = octonion(0, 0, 1, 2, 0, 0, 0, 0)

        left = (a * b) * b
        right = a * (b * b)

        assert ocomponents(left) == pytest.approx(ocomponents(right), abs=1e-10)


class TestOctonionCayleyDickson:
    """Tests related to Cayley-Dickson construction from quaternions."""

    def test_quaternion_subalgebra(self):
        """Test that first 4 basis elements form a quaternion subalgebra."""
        # E1 * E1 = -1
        result = E1 * E1
        assert result.real == pytest.approx(-1.0)
        components = ocomponents(result)
        for i in range(1, 8):
            assert components[i] == pytest.approx(0.0, abs=1e-10)

    def test_e1_times_e2_equals_e3(self):
        """Test e1 * e2 = e3 (quaternion-like behavior)."""
        result = E1 * E2
        assert ocomponents(result) == pytest.approx(ocomponents(E3))

    def test_norm_multiplicative(self):
        """Test that |a*b| = |a|*|b| (norm is multiplicative)."""
        a = octonion(1, 2, 3, 4, 0, 0, 0, 0)
        b = octonion(0, 0, 0, 0, 1, 2, 3, 4)

        product = a * b
        expected_norm = a.norm() * b.norm()
        assert product.norm() == pytest.approx(expected_norm)


class TestOctonionSpecialProperties:
    """Tests for special octonion properties relevant to SRT."""

    def test_seven_imaginaries_anticommute(self):
        """Test that distinct imaginary units anticommute: ei*ej = -ej*ei for i≠j."""
        # Test a few pairs
        pairs = [(E1, E2), (E1, E4), (E2, E5)]
        for ei, ej in pairs:
            result1 = ei * ej
            result2 = ej * ei
            neg_result2 = -result2
            assert ocomponents(result1) == pytest.approx(ocomponents(neg_result2))

    def test_octonion_from_quaternion_pair(self):
        """Test constructing octonion from pair of quaternion-like parts."""
        o = octonion(1, 2, 3, 4, 5, 6, 7, 8)
        components = ocomponents(o)
        assert len(components) == 8

    def test_imag_property(self):
        """Test the imag property returns non-real components."""
        o = octonion(1, 2, 3, 4, 5, 6, 7, 8)
        imag = o.imag
        # Should be [e1, e2, ..., e7]
        assert len(imag) == 7
        assert imag == pytest.approx([2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
