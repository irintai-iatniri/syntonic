"""Tests for Quaternion implementation."""

import math
import pytest
import syntonic as syn
from syntonic.hypercomplex import Quaternion, quaternion, I, J, K


def qcomponents(q):
    """Helper to get quaternion components as list [a, b, c, d]."""
    return [q.a, q.b, q.c, q.d]


class TestQuaternionConstruction:
    """Tests for quaternion creation."""

    def test_basic_construction(self):
        """Test creating a quaternion directly."""
        q = Quaternion(1.0, 2.0, 3.0, 4.0)
        assert q.real == pytest.approx(1.0)
        assert qcomponents(q) == pytest.approx([1.0, 2.0, 3.0, 4.0])

    def test_from_factory_function(self):
        """Test creating quaternion via factory function."""
        q = quaternion(1, 2, 3, 4)
        assert q.real == pytest.approx(1.0)
        assert qcomponents(q) == pytest.approx([1.0, 2.0, 3.0, 4.0])

    def test_unit_quaternion_I(self):
        """Test unit quaternion I = 0 + 1i + 0j + 0k."""
        assert I.real == pytest.approx(0.0)
        assert qcomponents(I) == pytest.approx([0.0, 1.0, 0.0, 0.0])

    def test_unit_quaternion_J(self):
        """Test unit quaternion J = 0 + 0i + 1j + 0k."""
        assert J.real == pytest.approx(0.0)
        assert qcomponents(J) == pytest.approx([0.0, 0.0, 1.0, 0.0])

    def test_unit_quaternion_K(self):
        """Test unit quaternion K = 0 + 0i + 0j + 1k."""
        assert K.real == pytest.approx(0.0)
        assert qcomponents(K) == pytest.approx([0.0, 0.0, 0.0, 1.0])


class TestQuaternionProperties:
    """Tests for quaternion properties."""

    def test_norm(self):
        """Test quaternion norm |q| = sqrt(a² + b² + c² + d²)."""
        q = quaternion(1, 2, 3, 4)
        expected = math.sqrt(1 + 4 + 9 + 16)  # sqrt(30)
        assert q.norm() == pytest.approx(expected)

    def test_norm_unit(self):
        """Test that unit quaternions have norm 1."""
        assert I.norm() == pytest.approx(1.0)
        assert J.norm() == pytest.approx(1.0)
        assert K.norm() == pytest.approx(1.0)

    def test_conjugate(self):
        """Test quaternion conjugate q* = a - bi - cj - dk."""
        q = quaternion(1, 2, 3, 4)
        q_conj = q.conjugate()
        assert q_conj.real == pytest.approx(1.0)
        assert qcomponents(q_conj) == pytest.approx([1.0, -2.0, -3.0, -4.0])

    def test_inverse(self):
        """Test quaternion inverse q⁻¹ = q* / |q|²."""
        q = quaternion(1, 2, 3, 4)
        q_inv = q.inverse()
        # q * q_inv should equal 1
        product = q * q_inv
        assert product.real == pytest.approx(1.0)
        # Imaginary parts should be ~0
        assert product.imag == pytest.approx([0.0, 0.0, 0.0], abs=1e-10)


class TestQuaternionArithmetic:
    """Tests for quaternion arithmetic operations."""

    def test_addition(self):
        """Test quaternion addition."""
        q1 = quaternion(1, 2, 3, 4)
        q2 = quaternion(5, 6, 7, 8)
        result = q1 + q2
        assert qcomponents(result) == pytest.approx([6.0, 8.0, 10.0, 12.0])

    def test_subtraction(self):
        """Test quaternion subtraction."""
        q1 = quaternion(5, 6, 7, 8)
        q2 = quaternion(1, 2, 3, 4)
        result = q1 - q2
        assert qcomponents(result) == pytest.approx([4.0, 4.0, 4.0, 4.0])

    def test_negation(self):
        """Test quaternion negation."""
        q = quaternion(1, 2, 3, 4)
        neg_q = -q
        assert qcomponents(neg_q) == pytest.approx([-1.0, -2.0, -3.0, -4.0])

    def test_scalar_multiplication(self):
        """Test quaternion scalar multiplication."""
        q = quaternion(1, 2, 3, 4)
        # Multiply by creating a scalar quaternion
        scalar = quaternion(2.0, 0, 0, 0)
        result = scalar * q
        assert qcomponents(result) == pytest.approx([2.0, 4.0, 6.0, 8.0])

    def test_quaternion_multiplication_associative(self):
        """Test that quaternion multiplication is associative: (q1*q2)*q3 = q1*(q2*q3)."""
        q1 = quaternion(1, 2, 0, 0)
        q2 = quaternion(0, 1, 1, 0)
        q3 = quaternion(0, 0, 1, 1)

        left = (q1 * q2) * q3
        right = q1 * (q2 * q3)

        assert qcomponents(left) == pytest.approx(qcomponents(right))


class TestHamiltonProduct:
    """Tests for Hamilton product identities: i² = j² = k² = ijk = -1."""

    def test_I_squared_equals_minus_one(self):
        """Test i² = -1."""
        result = I * I
        assert result.real == pytest.approx(-1.0)
        assert result.imag == pytest.approx([0.0, 0.0, 0.0], abs=1e-10)

    def test_J_squared_equals_minus_one(self):
        """Test j² = -1."""
        result = J * J
        assert result.real == pytest.approx(-1.0)
        assert result.imag == pytest.approx([0.0, 0.0, 0.0], abs=1e-10)

    def test_K_squared_equals_minus_one(self):
        """Test k² = -1."""
        result = K * K
        assert result.real == pytest.approx(-1.0)
        assert result.imag == pytest.approx([0.0, 0.0, 0.0], abs=1e-10)

    def test_IJ_equals_K(self):
        """Test i*j = k."""
        result = I * J
        assert qcomponents(result) == pytest.approx(qcomponents(K))

    def test_JK_equals_I(self):
        """Test j*k = i."""
        result = J * K
        assert qcomponents(result) == pytest.approx(qcomponents(I))

    def test_KI_equals_J(self):
        """Test k*i = j."""
        result = K * I
        assert qcomponents(result) == pytest.approx(qcomponents(J))

    def test_JI_equals_minus_K(self):
        """Test j*i = -k (non-commutativity)."""
        result = J * I
        neg_K = -K
        assert qcomponents(result) == pytest.approx(qcomponents(neg_K))

    def test_non_commutativity(self):
        """Test that quaternion multiplication is non-commutative: q1*q2 != q2*q1."""
        q1 = quaternion(1, 2, 3, 4)
        q2 = quaternion(5, 6, 7, 8)

        result1 = q1 * q2
        result2 = q2 * q1

        # They should NOT be equal in general
        comp1 = qcomponents(result1)
        comp2 = qcomponents(result2)
        assert any(abs(c1 - c2) > 1e-10 for c1, c2 in zip(comp1, comp2))


class TestQuaternionRotation:
    """Tests for quaternion rotation operations."""

    def test_rotation_matrix_identity(self):
        """Test that identity quaternion gives identity rotation matrix."""
        q = quaternion(1, 0, 0, 0)  # Identity quaternion
        if hasattr(q, 'to_rotation_matrix'):
            mat = q.to_rotation_matrix()
            assert mat[0] == pytest.approx([1.0, 0.0, 0.0])
            assert mat[1] == pytest.approx([0.0, 1.0, 0.0])
            assert mat[2] == pytest.approx([0.0, 0.0, 1.0])
        else:
            pytest.skip("to_rotation_matrix not implemented")

    def test_from_axis_angle(self):
        """Test creating quaternion from axis-angle representation."""
        if hasattr(Quaternion, 'from_axis_angle'):
            q = Quaternion.from_axis_angle([0, 0, 1], math.pi / 2)
            assert q.norm() == pytest.approx(1.0)
        else:
            pytest.skip("from_axis_angle not implemented")

    def test_normalize(self):
        """Test quaternion normalization."""
        q = quaternion(1, 2, 3, 4)
        if hasattr(q, 'normalize'):
            q_norm = q.normalize()
            assert q_norm.norm() == pytest.approx(1.0)
        else:
            norm = q.norm()
            q_norm = q * (1.0 / norm)
            assert q_norm.norm() == pytest.approx(1.0)


class TestQuaternionEquality:
    """Tests for quaternion equality comparison."""

    def test_equal_quaternions(self):
        """Test that equal quaternions compare as equal."""
        q1 = quaternion(1, 2, 3, 4)
        q2 = quaternion(1, 2, 3, 4)
        assert qcomponents(q1) == pytest.approx(qcomponents(q2))

    def test_different_quaternions(self):
        """Test that different quaternions don't compare as equal."""
        q1 = quaternion(1, 2, 3, 4)
        q2 = quaternion(1, 2, 3, 5)
        # Should differ in the d component
        assert q1.d != pytest.approx(q2.d)
