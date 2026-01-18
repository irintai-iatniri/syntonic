#!/usr/bin/env python3
"""
Test SRT Physics Engine

Tests for SRT physics simulators and theoretical predictions.
"""

import pytest
import sys
import os
import numpy as np

# Add syntonic package to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'python'))

try:
    from syntonic.physics.srt_physics import SRTPhysicsEngine
    HAS_PHYSICS_ENGINE = True
except ImportError:
    HAS_PHYSICS_ENGINE = False
    print("Warning: Physics engine not available, tests will be skipped")


class TestForceDifferentiation:
    """Test force differentiation simulator."""

    @pytest.mark.skipif(not HAS_PHYSICS_ENGINE, reason="Physics engine not available")
    def test_force_hierarchy_creation(self):
        """Test force hierarchy simulator creation."""
        engine = SRTPhysicsEngine()
        hierarchy = engine.create_force_hierarchy()

        assert hierarchy is not None
        assert isinstance(hierarchy, dict)

        # Should have force information
        assert 'forces' in hierarchy
        assert 'couplings' in hierarchy

    @pytest.mark.skipif(not HAS_PHYSICS_ENGINE, reason="Physics engine not available")
    def test_force_count_validation(self):
        """Test that exactly 5 forces are predicted."""
        engine = SRTPhysicsEngine()
        hierarchy = engine.create_force_hierarchy()

        forces = hierarchy['forces']
        assert len(forces) == 5, f"SRT predicts exactly 5 forces, got {len(forces)}"

        # Check force names
        expected_forces = ['gravity', 'electromagnetic', 'weak', 'strong', 'dark']
        actual_names = [f['name'] for f in forces]

        for expected in expected_forces:
            assert expected in actual_names, f"Missing force: {expected}"

    @pytest.mark.skipif(not HAS_PHYSICS_ENGINE, reason="Physics engine not available")
    def test_force_coupling_constants(self):
        """Test force coupling constants are reasonable."""
        engine = SRTPhysicsEngine()
        hierarchy = engine.create_force_hierarchy()

        couplings = hierarchy['couplings']

        # Gravity should be weakest
        assert couplings['gravity'] < couplings['electromagnetic']
        assert couplings['gravity'] < couplings['weak']
        assert couplings['gravity'] < couplings['strong']

        # Strong should be strongest
        assert couplings['strong'] > couplings['electromagnetic']
        assert couplings['strong'] > couplings['weak']

        # All couplings should be positive and reasonable
        for name, coupling in couplings.items():
            assert coupling > 0, f"Coupling for {name} should be positive"
            assert coupling < 1e40, f"Coupling for {name} should be reasonable"


class TestMatterGenerations:
    """Test matter generation simulator."""

    @pytest.mark.skipif(not HAS_PHYSICS_ENGINE, reason="Physics engine not available")
    def test_matter_generation_creation(self):
        """Test matter generation simulator."""
        engine = SRTPhysicsEngine()
        generations = engine.simulate_matter_generations()

        assert generations is not None
        assert isinstance(generations, list)

    @pytest.mark.skipif(not HAS_PHYSICS_ENGINE, reason="Physics engine not available")
    def test_matter_generation_count(self):
        """Test exactly 4 matter generations predicted."""
        engine = SRTPhysicsEngine()
        generations = engine.simulate_matter_generations()

        # SRT predicts exactly 4 matter generations
        assert len(generations) == 4, f"SRT predicts exactly 4 matter generations, got {len(generations)}"

    @pytest.mark.skipif(not HAS_PHYSICS_ENGINE, reason="Physics engine not available")
    def test_generation_properties(self):
        """Test matter generation properties."""
        engine = SRTPhysicsEngine()
        generations = engine.simulate_matter_generations()

        for i, gen in enumerate(generations):
            # Each generation should have mass and lifetime
            assert 'mass' in gen
            assert 'lifetime' in gen
            assert 'particles' in gen

            # Mass should increase with generation number
            if i > 0:
                assert gen['mass'] > generations[i-1]['mass']

            # Lifetime should decrease with generation number
            if i > 0:
                assert gen['lifetime'] < generations[i-1]['lifetime']


class TestDarkSector:
    """Test dark sector simulator."""

    @pytest.mark.skipif(not HAS_PHYSICS_ENGINE, reason="Physics engine not available")
    def test_dark_sector_creation(self):
        """Test dark sector simulator."""
        engine = SRTPhysicsEngine()
        dark_sector = engine.simulate_dark_sector()

        assert dark_sector is not None
        assert isinstance(dark_sector, dict)

    @pytest.mark.skipif(not HAS_PHYSICS_ENGINE, reason="Physics engine not available")
    def test_dark_matter_mass(self):
        """Test dark matter mass prediction."""
        engine = SRTPhysicsEngine()
        dark_sector = engine.simulate_dark_sector()

        assert 'dark_matter' in dark_sector
        dm = dark_sector['dark_matter']

        assert 'mass' in dm
        mass_gev = dm['mass']

        # SRT predicts ~1.18 TeV dark matter
        assert 1100 <= mass_gev <= 1500, f"Dark matter mass {mass_gev} GeV should be 1100-1500 GeV"

    @pytest.mark.skipif(not HAS_PHYSICS_ENGINE, reason="Physics engine not available")
    def test_dark_energy_density(self):
        """Test dark energy density prediction."""
        engine = SRTPhysicsEngine()
        dark_sector = engine.simulate_dark_sector()

        assert 'dark_energy' in dark_sector
        de = dark_sector['dark_energy']

        assert 'density' in de
        density = de['density']

        # Dark energy density should be positive and reasonable
        assert density > 0
        assert density < 1e-8  # Current observed value ~1e-9 erg/cm³


class TestConsciousnessEmergence:
    """Test consciousness emergence simulator."""

    @pytest.mark.skipif(not HAS_PHYSICS_ENGINE, reason="Physics engine not available")
    def test_consciousness_simulation(self):
        """Test consciousness emergence simulation."""
        engine = SRTPhysicsEngine()
        consciousness = engine.simulate_consciousness_emergence()

        assert consciousness is not None
        assert isinstance(consciousness, dict)

    @pytest.mark.skipif(not HAS_PHYSICS_ENGINE, reason="Physics engine not available")
    def test_consciousness_threshold(self):
        """Test consciousness emergence threshold."""
        engine = SRTPhysicsEngine()
        consciousness = engine.simulate_consciousness_emergence()

        assert 'threshold' in consciousness
        threshold = consciousness['threshold']

        # Threshold should be reasonable (around Planck scale or so)
        assert 1e18 < threshold < 1e22  # GeV range

    @pytest.mark.skipif(not HAS_PHYSICS_ENGINE, reason="Physics engine not available")
    def test_consciousness_phases(self):
        """Test consciousness emergence phases."""
        engine = SRTPhysicsEngine()
        consciousness = engine.simulate_consciousness_emergence()

        assert 'phases' in consciousness
        phases = consciousness['phases']

        # Should have multiple phases
        assert len(phases) >= 3

        # Phases should be ordered by complexity
        for i in range(1, len(phases)):
            assert phases[i]['complexity'] > phases[i-1]['complexity']


class TestPhysicsIntegration:
    """Test integrated physics predictions."""

    @pytest.mark.skipif(not HAS_PHYSICS_ENGINE, reason="Physics engine not available")
    def test_universe_simulation(self):
        """Test complete universe simulation."""
        engine = SRTPhysicsEngine()
        universe = engine.simulate_universe()

        assert universe is not None
        assert isinstance(universe, dict)

        # Should contain all components
        required_keys = [
            'forces', 'matter_generations', 'dark_sector',
            'consciousness', 'geometry', 'timeline'
        ]

        for key in required_keys:
            assert key in universe, f"Universe simulation missing {key}"

    @pytest.mark.skipif(not HAS_PHYSICS_ENGINE, reason="Physics engine not available")
    def test_prediction_validation(self):
        """Test that all SRT predictions are validated."""
        engine = SRTPhysicsEngine()
        validation = engine.validate_predictions()

        assert validation is not None
        assert isinstance(validation, dict)

        # Should validate all key predictions
        required_predictions = [
            'force_count', 'matter_generations', 'dark_matter_mass',
            'stability_barrier', 'fibonacci_gates'
        ]

        for pred in required_predictions:
            assert pred in validation
            assert validation[pred] is True, f"Prediction {pred} failed validation"

    @pytest.mark.skipif(not HAS_PHYSICS_ENGINE, reason="Physics engine not available")
    def test_coherence_score(self):
        """Test theoretical coherence scoring."""
        engine = SRTPhysicsEngine()
        coherence = engine.compute_coherence_score()

        # Coherence should be high (>80%)
        assert coherence > 0.8, f"Theoretical coherence {coherence:.1%} should be >80%"

        # Should be a reasonable value
        assert 0 <= coherence <= 1.0

    @pytest.mark.skipif(not HAS_PHYSICS_ENGINE, reason="Physics engine not available")
    def test_against_known_physics(self):
        """Test predictions against known physics."""
        engine = SRTPhysicsEngine()
        comparison = engine.compare_to_standard_model()

        assert comparison is not None
        assert isinstance(comparison, dict)

        # Should have comparison metrics
        assert 'matches' in comparison
        assert 'discrepancies' in comparison

        # Should match at least some known physics
        matches = comparison['matches']
        assert len(matches) > 0, "Should match at least some known physics"


class TestPhysicsBenchmarks:
    """Test physics simulation performance."""

    @pytest.mark.skipif(not HAS_PHYSICS_ENGINE, reason="Physics engine not available")
    def test_simulation_speed(self):
        """Test that simulations run in reasonable time."""
        import time

        engine = SRTPhysicsEngine()

        # Time universe simulation
        start = time.time()
        universe = engine.simulate_universe()
        duration = time.time() - start

        # Should complete in reasonable time (< 10 seconds)
        assert duration < 10.0, f"Universe simulation took {duration:.2f}s, should be <10s"

        # Should still produce valid results
        assert universe is not None

    @pytest.mark.skipif(not HAS_PHYSICS_ENGINE, reason="Physics engine not available")
    def test_memory_usage(self):
        """Test memory usage is reasonable."""
        import psutil
        import os

        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB

        engine = SRTPhysicsEngine()
        universe = engine.simulate_universe()

        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_used = final_memory - initial_memory

        # Should use reasonable memory (< 500 MB)
        assert memory_used < 500, f"Memory usage {memory_used:.1f} MB should be <500 MB"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])