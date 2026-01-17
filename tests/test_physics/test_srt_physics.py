"""
Tests for SRT/CRT Physics Simulation Engine

Tests the physics simulation components that validate SRT/CRT predictions
against known physical phenomena.
"""

import pytest
import math
from python.syntonic.physics.srt_physics import (
    SRTPhysicsEngine,
    ForceSimulator,
    MatterSimulator,
    DarkSectorSimulator,
    ConsciousnessSimulator
)


class TestSRTPhysicsEngine:
    """Test the main SRT physics engine."""

    def test_engine_creation(self):
        """Test physics engine initialization."""
        engine = SRTPhysicsEngine()
        assert engine is not None

    def test_prediction_validation(self):
        """Test SRT/CRT prediction validation."""
        engine = SRTPhysicsEngine()
        validations = engine.validate_srt_predictions()

        # Should validate all core predictions
        assert 'force_count' in validations
        assert 'matter_generations' in validations
        assert 'stability_barrier' in validations
        assert 'dark_matter_mass' in validations

        # All should be true for correct implementation
        for prediction, valid in validations.items():
            assert valid, f"Prediction {prediction} failed validation"

    def test_force_count_validation(self):
        """Test that exactly 5 forces are predicted."""
        engine = SRTPhysicsEngine()
        validations = engine.validate_srt_predictions()

        assert validations['force_count'] == True

        # Check the actual count
        forces = engine.get_fundamental_forces()
        assert len(forces) == 5

    def test_matter_generation_validation(self):
        """Test matter generation predictions."""
        engine = SRTPhysicsEngine()
        validations = engine.validate_srt_predictions()

        assert validations['matter_generations'] == True

        # Check stable generations
        generations = engine.get_stable_matter_generations()
        assert len(generations) == 4  # 3 + top quark

    def test_dark_matter_validation(self):
        """Test dark matter mass prediction."""
        engine = SRTPhysicsEngine()
        validations = engine.validate_srt_predictions()

        assert validations['dark_matter_mass'] == True

        # Check mass range
        dm_mass = engine.predict_dark_matter_mass()
        assert 1100 <= dm_mass <= 1500  # 1.1-1.5 TeV range


class TestForceSimulator:
    """Test the force differentiation simulator."""

    def test_force_simulator_creation(self):
        """Test force simulator initialization."""
        simulator = ForceSimulator()
        assert simulator is not None

    def test_fermat_force_hierarchy(self):
        """Test Fermat prime force hierarchy."""
        simulator = ForceSimulator()

        forces = simulator.get_forces()
        assert len(forces) == 5

        # Check force names and Fermat indices
        expected_forces = [
            ('Strong', 0),      # F0 = 3
            ('Weak', 1),        # F1 = 5
            ('EM', 2),          # F2 = 17
            ('Gravity', 3),     # F3 = 257
            ('Versal', 4)       # F4 = 65537
        ]

        for i, (name, fermat_idx) in enumerate(expected_forces):
            assert forces[i]['name'] == name
            assert forces[i]['fermat_index'] == fermat_idx

    def test_force_termination(self):
        """Test that force hierarchy terminates at F5."""
        simulator = ForceSimulator()

        # F5 should not exist (composite)
        assert not simulator.is_valid_force(5)

        # F0-F4 should exist
        for i in range(5):
            assert simulator.is_valid_force(i)

    def test_force_interactions(self):
        """Test force interaction patterns."""
        simulator = ForceSimulator()

        # Strong force should couple to color
        strong = simulator.get_force_by_name('Strong')
        assert 'color' in strong['couplings']

        # Weak force should couple to weak isospin
        weak = simulator.get_force_by_name('Weak')
        assert 'weak_isospin' in weak['couplings']

        # Gravity should couple universally
        gravity = simulator.get_force_by_name('Gravity')
        assert 'universal' in gravity['couplings']


class TestMatterSimulator:
    """Test the matter generation simulator."""

    def test_matter_simulator_creation(self):
        """Test matter simulator initialization."""
        simulator = MatterSimulator()
        assert simulator is not None

    def test_mersenne_matter_generations(self):
        """Test Mersenne prime matter generations."""
        simulator = MatterSimulator()

        generations = simulator.get_generations()
        assert len(generations) == 4

        # Check generation details
        expected_generations = [
            {'index': 1, 'mersenne_prime': 2, 'mass_scale': 'electron'},
            {'index': 2, 'mersenne_prime': 3, 'mass_scale': 'muon'},
            {'index': 3, 'mersenne_prime': 5, 'mass_scale': 'tau'},
            {'index': 4, 'mersenne_prime': 7, 'mass_scale': 'top'}
        ]

        for i, expected in enumerate(expected_generations):
            gen = generations[i]
            assert gen['index'] == expected['index']
            assert gen['mersenne_prime'] == expected['mersenne_prime']
            assert expected['mass_scale'] in gen['mass_scale'].lower()

    def test_stability_barrier(self):
        """Test the M11 stability barrier."""
        simulator = MatterSimulator()

        # Generations 1-4 should be stable
        for gen_idx in [1, 2, 3, 4]:
            assert simulator.is_generation_stable(gen_idx)

        # Generation 5 should be unstable (beyond barrier)
        assert not simulator.is_generation_stable(5)

    def test_barrier_position(self):
        """Test that barrier is at correct position."""
        simulator = MatterSimulator()

        barrier = simulator.get_stability_barrier()
        assert barrier == 11  # M11

        # Check that M11 is indeed composite
        assert not simulator.is_mersenne_prime(barrier)

    def test_particle_spectra(self):
        """Test particle spectra for each generation."""
        simulator = MatterSimulator()

        for gen_idx in [1, 2, 3, 4]:
            particles = simulator.get_generation_particles(gen_idx)
            assert len(particles) > 0

            # Each generation should have quarks and leptons
            has_quarks = any('quark' in p['type'] for p in particles)
            has_leptons = any('lepton' in p['type'] for p in particles)

            assert has_quarks
            assert has_leptons


class TestDarkSectorSimulator:
    """Test the dark sector simulator."""

    def test_dark_sector_creation(self):
        """Test dark sector simulator initialization."""
        simulator = DarkSectorSimulator()
        assert simulator is not None

    def test_lucas_dark_matter(self):
        """Test Lucas prime dark matter predictions."""
        simulator = DarkSectorSimulator()

        dm_candidates = simulator.get_dark_matter_candidates()
        assert len(dm_candidates) > 0

        # Should include L17 boosted candidate
        l17_candidate = None
        for candidate in dm_candidates:
            if candidate['lucas_index'] == 17:
                l17_candidate = candidate
                break

        assert l17_candidate is not None
        assert 'mass_gev' in l17_candidate

    def test_dark_matter_mass_prediction(self):
        """Test dark matter mass calculation."""
        simulator = DarkSectorSimulator()

        dm_mass = simulator.predict_dark_matter_mass()
        assert 1100 <= dm_mass <= 1500  # 1.1-1.5 TeV

        # Should match Lucas boost calculation
        boost = simulator.get_lucas_boost_factor()
        assert abs(boost - 6.854) < 0.01  # L17/L13

    def test_dark_energy_mechanism(self):
        """Test dark energy gap pressure mechanism."""
        simulator = DarkSectorSimulator()

        # Should identify Lucas gaps
        gaps = simulator.get_lucas_gaps(50)
        assert len(gaps) > 0

        # Gaps should correspond to non-prime Lucas indices
        for gap_start, gap_end in gaps:
            for n in range(gap_start, gap_end + 1):
                assert not simulator.is_lucas_prime(n)

    def test_dark_sector_stability(self):
        """Test dark sector stability properties."""
        simulator = DarkSectorSimulator()

        # Dark particles should be stable (long-lived)
        dm_particle = simulator.get_primary_dark_matter()
        assert dm_particle['lifetime'] == 'stable' or dm_particle['lifetime'] > 1e30

        # Should be weakly interacting
        assert dm_particle['interaction_strength'] == 'weak'


class TestConsciousnessSimulator:
    """Test the consciousness emergence simulator."""

    def test_consciousness_simulator_creation(self):
        """Test consciousness simulator initialization."""
        simulator = ConsciousnessSimulator()
        assert simulator is not None

    def test_fibonacci_transcendence_gates(self):
        """Test Fibonacci prime transcendence gates."""
        simulator = ConsciousnessSimulator()

        gates = simulator.get_transcendence_gates()
        assert len(gates) > 0

        # Should include key gates
        gate_indices = [g['fib_index'] for g in gates]
        assert 17 in gate_indices  # Consciousness emergence

    def test_gamma_synchrony_alignment(self):
        """Test gamma wave synchrony with Fibonacci ratios."""
        simulator = ConsciousnessSimulator()

        gamma_freq = 40.0  # Hz
        alignment = simulator.compute_gamma_alignment(gamma_freq)

        # Should be close to Fibonacci ratio
        phi = (1 + math.sqrt(5)) / 2
        fib_ratio = phi ** 2  # Approximately 2.618

        assert abs(alignment - fib_ratio) < 0.1

    def test_consciousness_threshold(self):
        """Test consciousness emergence threshold."""
        simulator = ConsciousnessSimulator()

        threshold = simulator.get_consciousness_threshold()

        # Should be related to D4 kissing number (24) and M5 (31)
        assert 24 <= threshold <= 31

    def test_neural_synchrony_model(self):
        """Test neural synchrony modeling."""
        simulator = ConsciousnessSimulator()

        # Test different synchrony states
        states = ['alpha', 'beta', 'gamma', 'theta']

        for state in states:
            sync_measure = simulator.compute_synchrony_measure(state)
            assert sync_measure >= 0.0
            assert sync_measure <= 1.0

        # Gamma should have highest syntony
        gamma_sync = simulator.compute_synchrony_measure('gamma')
        alpha_sync = simulator.compute_synchrony_measure('alpha')
        assert gamma_sync > alpha_sync


class TestPhysicsIntegration:
    """Integration tests for physics simulations."""

    def test_unified_physics_engine(self):
        """Test integrated physics engine."""
        engine = SRTPhysicsEngine()

        # Should be able to run complete simulation
        results = engine.run_universe_simulation(generations=10)

        assert 'forces' in results
        assert 'matter' in results
        assert 'dark_sector' in results
        assert 'consciousness' in results

        # Check force count
        assert len(results['forces']) == 5

        # Check matter generations
        assert len(results['matter']) == 4

    def test_prediction_consistency(self):
        """Test that all predictions are mutually consistent."""
        engine = SRTPhysicsEngine()

        # Get all predictions
        forces = engine.get_fundamental_forces()
        matter = engine.get_stable_matter_generations()
        dm_mass = engine.predict_dark_matter_mass()

        # Check internal consistency
        assert len(forces) == 5
        assert len(matter) == 4
        assert 1100 <= dm_mass <= 1500

        # Check that force hierarchy terminates
        assert not any(f['valid'] for f in forces if f.get('fermat_index', 0) >= 5)

    def test_srt_crt_coherence(self):
        """Test SRT/CRT theoretical coherence."""
        engine = SRTPhysicsEngine()

        coherence_metrics = engine.compute_theoretical_coherence()

        # All coherence metrics should be high (>0.8)
        for metric, value in coherence_metrics.items():
            assert value > 0.8, f"Low coherence in {metric}: {value}"

    def test_numerical_stability(self):
        """Test numerical stability of simulations."""
        engine = SRTPhysicsEngine()

        # Run multiple simulations
        for _ in range(10):
            results = engine.run_universe_simulation(generations=5)

            # Results should be consistent
            assert len(results['forces']) == 5
            assert len(results['matter']) == 4

            dm_mass = results['dark_sector']['primary_candidate']['mass_gev']
            assert 1100 <= dm_mass <= 1500


if __name__ == "__main__":
    pytest.main([__file__])