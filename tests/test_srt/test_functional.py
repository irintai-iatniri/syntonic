"""Tests for SRT functional module (SyntonyFunctional)."""

import pytest
import math


class TestSyntonyFunctional:
    """Test SyntonyFunctional class."""

    def test_creation(self):
        """Test SyntonyFunctional creation."""
        from syntonic.srt import syntony_functional
        S = syntony_functional()
        assert S is not None

    def test_global_bound_is_phi(self):
        """Test global bound is φ."""
        from syntonic.srt import syntony_functional, PHI_NUMERIC
        S = syntony_functional()
        assert abs(S.global_bound - PHI_NUMERIC) < 1e-15

    def test_vacuum_partition_positive(self):
        """Test vacuum partition function is positive."""
        from syntonic.srt import syntony_functional
        S = syntony_functional()
        assert S.vacuum_partition > 0

    def test_evaluate_positive(self):
        """Test syntony is positive for non-zero states."""
        from syntonic.srt import syntony_functional, winding_state
        S = syntony_functional()
        n = winding_state(0, 0, 0, 0)
        coefficients = {n: complex(1.0)}
        value = S.evaluate(coefficients)
        assert value > 0

    def test_evaluate_bounded_by_phi(self):
        """Test S[Ψ] ≤ φ."""
        from syntonic.srt import syntony_functional, winding_state, PHI_NUMERIC
        S = syntony_functional()

        # Test various states
        for n7, n8 in [(0, 0), (1, 0), (0, 1), (1, 1)]:
            n = winding_state(n7, n8, 0, 0)
            coefficients = {n: complex(1.0)}
            value, bounded = S.verify_bound(coefficients)
            assert bounded, f"S[|{n7},{n8},0,0⟩] = {value} > φ = {PHI_NUMERIC}"

    def test_ground_state(self):
        """Test ground state is vacuum."""
        from syntonic.srt import syntony_functional, winding_state
        S = syntony_functional()
        coeffs, value = S.ground_state()

        vacuum = winding_state(0, 0, 0, 0)
        assert vacuum in coeffs

    def test_superposition_bounded(self):
        """Test superposition states are bounded."""
        from syntonic.srt import syntony_functional, winding_state
        S = syntony_functional()

        states = [
            winding_state(0, 0, 0, 0),
            winding_state(1, 0, 0, 0),
            winding_state(0, 1, 0, 0),
        ]
        coeffs, value = S.coherent_superposition(states)
        _, bounded = S.verify_bound(coeffs)
        assert bounded

    def test_thermal_state_bounded(self):
        """Test thermal state is bounded."""
        from syntonic.srt import syntony_functional
        S = syntony_functional(max_norm=5)

        for beta in [0.1, 1.0, 10.0]:
            coeffs, value = S.thermal_state(beta)
            _, bounded = S.verify_bound(coeffs)
            assert bounded, f"Thermal state at β={beta} exceeds bound"

    def test_gradient_returns_dict(self):
        """Test gradient returns dictionary."""
        from syntonic.srt import syntony_functional, winding_state
        S = syntony_functional()
        n = winding_state(0, 0, 0, 0)
        coefficients = {n: complex(1.0)}
        grad = S.gradient(coefficients)
        assert isinstance(grad, dict)
        assert n in grad

    def test_optimize_step(self):
        """Test optimization step updates coefficients."""
        from syntonic.srt import syntony_functional, winding_state
        S = syntony_functional()
        n = winding_state(1, 0, 0, 0)
        coefficients = {n: complex(1.0)}
        updated = S.optimize_step(coefficients, step_size=0.01)
        assert n in updated


class TestComputeSyntony:
    """Test compute_syntony convenience function."""

    def test_compute_syntony(self):
        """Test compute_syntony function."""
        from syntonic.srt import compute_syntony, winding_state
        n = winding_state(0, 0, 0, 0)
        coefficients = {n: complex(1.0)}
        value = compute_syntony(coefficients)
        assert value > 0

    def test_compute_syntony_custom_phi(self):
        """Test compute_syntony with custom phi."""
        from syntonic.srt import compute_syntony, winding_state
        n = winding_state(0, 0, 0, 0)
        coefficients = {n: complex(1.0)}
        value = compute_syntony(coefficients, phi=1.5)
        assert value > 0


class TestFunctionalIntegration:
    """Integration tests for functional module."""

    def test_syntony_with_laplacian(self):
        """Test syntony uses the correct Laplacian."""
        from syntonic.srt import syntony_functional, knot_laplacian
        S = syntony_functional()
        L = knot_laplacian()

        # They should use the same phi
        assert abs(S.phi - L.phi) < 1e-15

    def test_vacuum_maximizes_syntony(self):
        """Test vacuum state has highest syntony among pure states."""
        from syntonic.srt import syntony_functional, winding_state
        S = syntony_functional(max_norm=5)

        vacuum = winding_state(0, 0, 0, 0)
        _, S_vacuum = S.excited_state(vacuum)

        # Compare with some excited states
        for n7, n8, n9, n10 in [(1, 0, 0, 0), (0, 1, 0, 0), (1, 1, 0, 0)]:
            n = winding_state(n7, n8, n9, n10)
            _, S_n = S.excited_state(n)
            # Vacuum should have higher syntony due to lower eigenvalue
            assert S_vacuum >= S_n
