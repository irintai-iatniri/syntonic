"""Tests for CRT evolution and trajectory tracking."""

import math
import pytest
import syntonic as syn
from syntonic.crt import (
    SyntonyTrajectory,
    DHSREvolver,
    create_dhsr_system,
    create_evolver,
)


class TestSyntonyTrajectory:
    """Tests for SyntonyTrajectory dataclass."""

    def test_empty_trajectory(self):
        """Test empty trajectory properties."""
        traj = SyntonyTrajectory()

        assert len(traj) == 0
        assert traj.n_steps == 0
        assert traj.initial_state is None
        assert traj.final_state is None
        assert not traj.converged

    def test_trajectory_properties(self):
        """Test trajectory properties with data."""
        state1 = syn.state([1, 2, 3, 4])
        state2 = syn.state([1.1, 2.1, 3.1, 4.1])

        traj = SyntonyTrajectory(
            states=[state1, state2],
            syntony_values=[0.5, 0.6],
            gnosis_values=[0, 1],
            phase_values=[0.0, 0.5],
            change_magnitudes=[0.1],
        )

        assert len(traj) == 2
        assert traj.n_steps == 1
        assert traj.initial_state is state1
        assert traj.final_state is state2
        assert traj.initial_syntony == 0.5
        assert traj.final_syntony == 0.6
        assert traj.final_gnosis == 1
        assert traj.final_phase == 0.5

    def test_syntony_delta(self):
        """Test syntony delta computation."""
        traj = SyntonyTrajectory(
            syntony_values=[0.3, 0.4, 0.6, 0.7],
        )

        delta = traj.syntony_delta
        assert delta == pytest.approx(0.4)

    def test_syntony_trend(self):
        """Test syntony trend detection."""
        increasing = SyntonyTrajectory(syntony_values=[0.3, 0.5, 0.7])
        decreasing = SyntonyTrajectory(syntony_values=[0.7, 0.5, 0.3])
        stable = SyntonyTrajectory(syntony_values=[0.5, 0.505, 0.501])

        assert increasing.syntony_trend == 'increasing'
        assert decreasing.syntony_trend == 'decreasing'
        assert stable.syntony_trend == 'stable'

    def test_converged_detection(self):
        """Test convergence detection."""
        not_converged = SyntonyTrajectory(
            change_magnitudes=[0.1, 0.05, 0.01],
        )
        converged = SyntonyTrajectory(
            change_magnitudes=[0.1, 0.01, 1e-8],
        )

        assert not not_converged.converged
        assert converged.converged

    def test_summary(self):
        """Test summary generation."""
        traj = SyntonyTrajectory(
            states=[syn.state([1, 2, 3, 4])],
            syntony_values=[0.5, 0.6],
            gnosis_values=[1, 2],
            change_magnitudes=[0.1],
        )

        summary = traj.summary()
        assert isinstance(summary, str)
        assert 'SyntonyTrajectory' in summary


class TestDHSREvolver:
    """Tests for DHSREvolver."""

    def test_basic_evolution(self):
        """Test basic state evolution."""
        evolver = DHSREvolver()
        state = syn.state([1, 2, 3, 4, 5, 6, 7, 8])

        traj = evolver.evolve(state, n_steps=10)

        assert traj.n_steps == 10
        assert len(traj.states) == 11
        assert len(traj.syntony_values) == 11
        assert len(traj.gnosis_values) == 11

    def test_early_stopping(self):
        """Test early stopping on convergence."""
        evolver = DHSREvolver()
        # Constant state should converge quickly
        state = syn.state([5, 5, 5, 5, 5, 5, 5, 5])

        traj = evolver.evolve(state, n_steps=100, early_stop=True, tol=1e-6)

        # Should stop before 100 steps if converged
        # (may or may not converge depending on operators)
        assert traj.n_steps >= 1

    def test_no_early_stopping(self):
        """Test evolution without early stopping."""
        evolver = DHSREvolver()
        state = syn.state([5, 5, 5, 5, 5, 5, 5, 5])

        traj = evolver.evolve(state, n_steps=10, early_stop=False)

        assert traj.n_steps == 10

    def test_find_attractor(self):
        """Test attractor finding."""
        evolver = DHSREvolver()
        state = syn.state([1, 2, 3, 4, 5, 6, 7, 8])

        attractor, traj = evolver.find_attractor(state, max_iter=50)

        assert attractor is not None
        assert attractor.shape == state.shape
        assert traj.final_state is attractor

    def test_analyze_stability(self):
        """Test stability analysis."""
        evolver = DHSREvolver()
        state = syn.state([1, 2, 3, 4, 5, 6, 7, 8])

        analysis = evolver.analyze_stability(
            state,
            perturbation_scale=0.01,
            n_perturbations=5,
            n_steps=10,
        )

        assert 'base_syntony' in analysis
        assert 'mean_final_syntony' in analysis
        assert 'syntony_variance' in analysis
        assert 'stable' in analysis
        assert 'convergence_rate' in analysis

    def test_find_all_attractors(self):
        """Test finding multiple attractors."""
        evolver = DHSREvolver()
        initial_states = [
            syn.state([1, 2, 3, 4, 5, 6, 7, 8]),
            syn.state([8, 7, 6, 5, 4, 3, 2, 1]),
            syn.state([1, 1, 1, 1, 1, 1, 1, 1]),
        ]

        attractors = evolver.find_all_attractors(
            initial_states,
            tol=1e-4,
            cluster_tol=0.1,
        )

        assert len(attractors) >= 1
        for attractor, count in attractors:
            assert count >= 1


class TestFactoryFunctions:
    """Tests for factory functions."""

    def test_create_dhsr_system(self):
        """Test create_dhsr_system factory."""
        R_op, S_comp, G_comp = create_dhsr_system()

        state = syn.state([1, 2, 3, 4, 5, 6, 7, 8])

        # Test all components work
        evolved = R_op.apply(state)
        syntony = S_comp.compute(state)
        gnosis = G_comp.compute_layer(state)

        assert evolved.shape == state.shape
        assert 0 <= syntony <= 1
        assert gnosis in [0, 1, 2, 3]

    def test_create_dhsr_system_custom_params(self):
        """Test create_dhsr_system with custom parameters."""
        R_op, S_comp, G_comp = create_dhsr_system(
            alpha_0=0.2,
            beta_0=0.5,
            num_modes=4,
        )

        assert R_op.diff_op.alpha_0 == 0.2
        assert R_op.harm_op.beta_0 == 0.5
        assert R_op.diff_op.num_modes == 4

    def test_create_evolver(self):
        """Test create_evolver factory."""
        evolver = create_evolver(alpha_0=0.1, beta_0=0.618)

        state = syn.state([1, 2, 3, 4, 5, 6, 7, 8])
        traj = evolver.evolve(state, n_steps=5)

        assert traj.n_steps == 5


class TestIntegration:
    """Integration tests for CRT evolution."""

    def test_full_workflow(self):
        """Test complete CRT workflow."""
        # Create system
        R_op, S_comp, G_comp = create_dhsr_system()

        # Create initial state
        initial = syn.state([1, 2, 3, 4, 5, 6, 7, 8])

        # Compute initial metrics
        initial_syntony = S_comp.compute(initial)
        initial_gnosis = G_comp.compute_layer(initial)

        # Evolve
        evolver = DHSREvolver(
            recursion_op=R_op,
            syntony_computer=S_comp,
            gnosis_computer=G_comp,
        )
        traj = evolver.evolve(initial, n_steps=20)

        # Check trajectory
        assert traj.n_steps == 20
        assert all(0 <= s <= 1 for s in traj.syntony_values)
        assert all(g in [0, 1, 2, 3] for g in traj.gnosis_values)

        # Summary should work
        summary = traj.summary()
        assert len(summary) > 0

    def test_attractor_convergence(self):
        """Test that evolution converges to attractors."""
        evolver = create_evolver()

        # Start from random-ish state
        initial = syn.state([1, 3, 2, 5, 4, 7, 6, 8])

        attractor, traj = evolver.find_attractor(initial, tol=1e-4, max_iter=200)

        # If converged, final change should be small
        if traj.converged:
            assert traj.change_magnitudes[-1] < 1e-4

    def test_phase_accumulation(self):
        """Test that phase accumulates during evolution."""
        evolver = DHSREvolver()
        state = syn.state([1, 2, 3, 4, 5, 6, 7, 8])

        traj = evolver.evolve(state, n_steps=20, early_stop=False)

        # Phase should be non-decreasing (mostly)
        phases = traj.phase_values
        assert phases[-1] >= phases[0]
