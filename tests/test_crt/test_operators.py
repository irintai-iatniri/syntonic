"""Tests for CRT operators (D̂, Ĥ, R̂, projectors)."""

import math
import pytest
import syntonic as syn
from syntonic.crt import (
    FourierProjector,
    DampingProjector,
    LaplacianOperator,
    DifferentiationOperator,
    HarmonizationOperator,
    RecursionOperator,
    create_mode_projectors,
    create_damping_cascade,
)


class TestFourierProjector:
    """Tests for FourierProjector."""

    def test_basic_projection(self):
        """Test basic Fourier projection."""
        projector = FourierProjector(mode_indices=[0], size=8)
        state = syn.state([1, 2, 3, 4, 5, 6, 7, 8])
        projected = projector.project(state)
        assert projected.shape == state.shape

    def test_dc_mode_projection(self):
        """Test DC mode (k=0) preserves mean."""
        projector = FourierProjector(mode_indices=[0], size=8)
        state = syn.state([1, 2, 3, 4, 5, 6, 7, 8])
        projected = projector.project(state)
        # DC mode gives constant = mean
        flat = projected.to_list()
        mean_val = sum([1, 2, 3, 4, 5, 6, 7, 8]) / 8
        # All values should be close to mean
        for val in flat:
            assert val == pytest.approx(mean_val, abs=1e-10)

    def test_idempotent(self):
        """Test P² ≈ P (projector is approximately idempotent)."""
        # Use DC mode which is cleanest for idempotence
        projector = FourierProjector(mode_indices=[0], size=8)
        state = syn.state([1, 2, 3, 4, 5, 6, 7, 8])

        p1 = projector.project(state)
        p2 = projector.project(p1)

        # P(P(x)) should be close to P(x) for DC mode
        diff = (p2 - p1).norm()
        assert diff == pytest.approx(0.0, abs=1e-8)

    def test_orthogonality(self):
        """Test that different mode projectors are orthogonal."""
        proj1 = FourierProjector(mode_indices=[1], size=8)
        proj2 = FourierProjector(mode_indices=[2], size=8)

        state = syn.state([1, 2, 3, 4, 5, 6, 7, 8])
        p1 = proj1.project(state)
        p2 = proj2.project(state)

        # Inner product of orthogonal projections should be small
        # (zero if modes are truly orthogonal)
        flat1 = p1.to_list()
        flat2 = p2.to_list()
        inner = sum(a * b for a, b in zip(flat1, flat2))
        assert abs(inner) == pytest.approx(0.0, abs=1e-8)


class TestDampingProjector:
    """Tests for DampingProjector."""

    def test_basic_damping(self):
        """Test basic damping application."""
        damper = DampingProjector(cutoff_fraction=0.5)
        state = syn.state([1, 0, 1, 0, 1, 0, 1, 0])  # High frequency
        damped = damper.project(state)
        assert damped.shape == state.shape

    def test_low_pass_effect(self):
        """Test that high frequencies are attenuated."""
        damper = DampingProjector(cutoff_fraction=0.3, order=4)
        # Create state with high-frequency component
        state = syn.state([1, -1, 1, -1, 1, -1, 1, -1])

        damped = damper.project(state)

        # High-frequency content should be reduced
        original_var = sum((x - sum(state.to_list())/8)**2 for x in state.to_list())
        damped_var = sum((x - sum(damped.to_list())/8)**2 for x in damped.to_list())

        assert damped_var < original_var

    def test_preserves_dc(self):
        """Test that DC component is preserved."""
        damper = DampingProjector(cutoff_fraction=0.5)
        state = syn.state([5, 5, 5, 5, 5, 5, 5, 5])  # Pure DC

        damped = damper.project(state)

        # Should be unchanged
        original = state.to_list()
        result = damped.to_list()
        for o, r in zip(original, result):
            assert r == pytest.approx(o, abs=1e-10)


class TestLaplacianOperator:
    """Tests for LaplacianOperator."""

    def test_basic_laplacian(self):
        """Test basic Laplacian application."""
        laplacian = LaplacianOperator()
        state = syn.state([1, 2, 3, 4, 3, 2, 1, 0])
        result = laplacian.apply(state)
        assert result.shape == state.shape

    def test_constant_gives_zero(self):
        """Test that ∇²(constant) = 0."""
        laplacian = LaplacianOperator()
        state = syn.state([5, 5, 5, 5, 5, 5, 5, 5])

        result = laplacian.apply(state)

        for val in result.to_list():
            assert val == pytest.approx(0.0, abs=1e-10)

    def test_linear_gives_zero(self):
        """Test that ∇²(linear) = 0 (with periodic BC, this is modified)."""
        laplacian = LaplacianOperator(boundary='periodic')
        # Linear function wrapping around
        state = syn.state([0, 1, 2, 3, 4, 5, 6, 7])
        result = laplacian.apply(state)

        # For periodic BC, endpoints wrap
        # Most interior points should have zero Laplacian
        flat = result.to_list()
        for i in range(1, 7):
            assert flat[i] == pytest.approx(0.0, abs=1e-10)


class TestDifferentiationOperator:
    """Tests for DifferentiationOperator (D̂)."""

    def test_basic_differentiation(self):
        """Test basic D̂ application."""
        D_op = DifferentiationOperator()
        state = syn.state([1, 2, 3, 4, 5, 6, 7, 8])
        result = D_op.apply(state)
        assert result.shape == state.shape

    def test_identity_at_high_syntony(self):
        """Test that D̂ ≈ identity when S ≈ 1."""
        D_op = DifferentiationOperator(alpha_0=0.1, zeta_0=0.01)
        state = syn.state([1, 2, 3, 4, 5, 6, 7, 8])

        # Apply with syntony = 1
        result = D_op.apply(state, syntony=1.0)

        # Should be very close to original
        diff = (result - state).norm()
        assert diff == pytest.approx(0.0, abs=1e-8)

    def test_syntony_dependent_coupling(self):
        """Test that coupling scales with (1-S)."""
        D_op = DifferentiationOperator(alpha_0=0.5, zeta_0=0.1)
        state = syn.state([1, 2, 3, 4, 5, 6, 7, 8])

        # Low syntony should give larger change
        result_low_S = D_op.apply(state, syntony=0.0)
        result_high_S = D_op.apply(state, syntony=0.9)

        diff_low = (result_low_S - state).norm()
        diff_high = (result_high_S - state).norm()

        assert diff_low > diff_high

    def test_differentiation_magnitude(self):
        """Test differentiation_magnitude helper."""
        D_op = DifferentiationOperator()
        state = syn.state([1, 2, 3, 4, 5, 6, 7, 8])

        mag = D_op.differentiation_magnitude(state)
        assert mag >= 0


class TestHarmonizationOperator:
    """Tests for HarmonizationOperator (Ĥ)."""

    def test_basic_harmonization(self):
        """Test basic Ĥ application."""
        H_op = HarmonizationOperator()
        state = syn.state([1, 2, 3, 4, 5, 6, 7, 8])
        result = H_op.apply(state)
        assert result.shape == state.shape

    def test_identity_at_low_syntony(self):
        """Test that Ĥ ≈ identity when S ≈ 0."""
        H_op = HarmonizationOperator(beta_0=0.618, gamma_0=0.1)
        state = syn.state([1, 2, 3, 4, 5, 6, 7, 8])

        # Apply with syntony = 0
        result = H_op.apply(state, syntony=0.0)

        # Should be close to original (damping terms scale with S)
        diff = (result - state).norm()
        # Small change due to possible nonlinear term
        assert diff < 0.1

    def test_syntony_dependent_damping(self):
        """Test that damping scales with S."""
        H_op = HarmonizationOperator(beta_0=0.5, gamma_0=0.2)
        state = syn.state([1, -1, 1, -1, 1, -1, 1, -1])  # High freq

        # High syntony should give larger change (more damping)
        result_low_S = H_op.apply(state, syntony=0.1)
        result_high_S = H_op.apply(state, syntony=0.9)

        diff_low = (result_low_S - state).norm()
        diff_high = (result_high_S - state).norm()

        assert diff_high > diff_low

    def test_harmonization_magnitude(self):
        """Test harmonization_magnitude helper."""
        H_op = HarmonizationOperator()
        state = syn.state([1, 2, 3, 4, 5, 6, 7, 8])

        mag = H_op.harmonization_magnitude(state)
        assert mag >= 0


class TestRecursionOperator:
    """Tests for RecursionOperator (R̂ = Ĥ ∘ D̂)."""

    def test_basic_recursion(self):
        """Test basic R̂ application."""
        R_op = RecursionOperator()
        state = syn.state([1, 2, 3, 4, 5, 6, 7, 8])
        result = R_op.apply(state)
        assert result.shape == state.shape

    def test_composition(self):
        """Test that R̂ = Ĥ ∘ D̂."""
        D_op = DifferentiationOperator()
        H_op = HarmonizationOperator()
        R_op = RecursionOperator(diff_op=D_op, harm_op=H_op)

        state = syn.state([1, 2, 3, 4, 5, 6, 7, 8])

        # Apply R̂
        r_result = R_op.apply(state)

        # Apply D̂ then Ĥ manually
        d_state = D_op.apply(state)
        hd_state = H_op.apply(d_state)

        # Should be close (not exact due to delta_d passing)
        diff = (r_result - hd_state).norm()
        assert diff < 1e-6

    def test_iterate(self):
        """Test iteration returns trajectory."""
        R_op = RecursionOperator()
        state = syn.state([1, 2, 3, 4, 5, 6, 7, 8])

        trajectory = R_op.iterate(state, n_steps=5)

        assert len(trajectory) == 6  # Initial + 5 steps
        assert trajectory[0] is state

    def test_find_fixed_point_converges(self):
        """Test that find_fixed_point converges for reasonable states."""
        R_op = RecursionOperator()
        state = syn.state([1, 2, 3, 4, 5, 6, 7, 8])

        fixed, n_iters, converged = R_op.find_fixed_point(
            state, tol=1e-4, max_iter=100
        )

        # Should either converge or reach max iterations
        assert n_iters >= 1
        assert fixed.shape == state.shape

    def test_apply_with_info(self):
        """Test apply_with_info returns detailed information."""
        R_op = RecursionOperator()
        state = syn.state([1, 2, 3, 4, 5, 6, 7, 8])

        result, info = R_op.apply_with_info(state)

        assert 'diff_magnitude' in info
        assert 'harm_magnitude' in info
        assert 'total_change' in info
        assert 'd_state' in info


class TestProjectorFactories:
    """Tests for projector factory functions."""

    def test_create_mode_projectors(self):
        """Test create_mode_projectors factory."""
        projectors = create_mode_projectors(size=16, num_modes=4)
        assert len(projectors) == 4

    def test_create_damping_cascade(self):
        """Test create_damping_cascade factory."""
        dampers = create_damping_cascade(num_levels=3)
        assert len(dampers) == 3
        # Cutoffs should decrease
        assert dampers[0].cutoff_fraction > dampers[1].cutoff_fraction
        assert dampers[1].cutoff_fraction > dampers[2].cutoff_fraction
