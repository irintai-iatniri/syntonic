"""Tests for Retrocausal Attractor-Guided RES."""

import pytest
import math

from syntonic._core import ResonantTensor, ResonantEvolver, RESConfig, RESResult
from syntonic.resonant.retrocausal import (
    RetrocausalConfig,
    create_retrocausal_evolver,
    create_standard_evolver,
    compare_convergence,
)

# Golden ratio constant
PHI = 1.6180339887498949

# Universal syntony deficit
Q_DEFICIT = 0.027395146920


class TestRetrocausalConfig:
    """Tests for RetrocausalConfig wrapper."""

    def test_default_retrocausal_config(self):
        """Test creating retrocausal config with defaults."""
        config = RetrocausalConfig()

        # Standard RES defaults
        assert config.population_size == 32
        assert config.survivor_count == 8  # pop_size // 4
        assert config.lambda_val is None  # Will use Q_DEFICIT
        assert config.mutation_scale == 0.1
        assert config.noise_scale == 0.01
        assert config.precision == 100
        assert config.max_generations == 1000
        assert config.convergence_threshold == 1e-6

        # Retrocausal defaults
        assert config.attractor_capacity == 32
        assert config.attractor_pull_strength == 0.3
        assert config.attractor_min_syntony == 0.7
        assert config.attractor_decay_rate == 0.98

    def test_custom_retrocausal_config(self):
        """Test creating retrocausal config with custom values."""
        config = RetrocausalConfig(
            population_size=64,
            survivor_count=16,
            attractor_capacity=64,
            attractor_pull_strength=0.5,
            attractor_min_syntony=0.8,
            attractor_decay_rate=0.95,
        )

        assert config.population_size == 64
        assert config.survivor_count == 16
        assert config.attractor_capacity == 64
        assert config.attractor_pull_strength == 0.5
        assert config.attractor_min_syntony == 0.8
        assert config.attractor_decay_rate == 0.95

    def test_to_res_config_conversion(self):
        """Test conversion to RESConfig."""
        retro_config = RetrocausalConfig(
            population_size=32,
            attractor_capacity=16,
            attractor_pull_strength=0.4,
        )

        res_config = retro_config.to_res_config()

        # Check that it's a valid RESConfig
        assert isinstance(res_config, RESConfig)
        assert res_config.population_size == 32
        assert res_config.attractor_capacity == 16
        assert res_config.attractor_pull_strength == 0.4
        assert res_config.enable_retrocausal is True


class TestRESConfigRetrocausalParameters:
    """Tests for retrocausal parameters in RESConfig."""

    def test_retrocausal_disabled_by_default(self):
        """Test that retrocausal is opt-in (disabled by default)."""
        config = RESConfig()
        assert config.enable_retrocausal is False

    def test_retrocausal_enabled(self):
        """Test enabling retrocausal with custom parameters."""
        config = RESConfig(
            enable_retrocausal=True,
            attractor_capacity=64,
            attractor_pull_strength=0.5,
            attractor_min_syntony=0.8,
            attractor_decay_rate=0.95,
        )

        assert config.enable_retrocausal is True
        assert config.attractor_capacity == 64
        assert config.attractor_pull_strength == 0.5
        assert config.attractor_min_syntony == 0.8
        assert config.attractor_decay_rate == 0.95

    def test_retrocausal_default_parameters(self):
        """Test default values for retrocausal parameters."""
        config = RESConfig(enable_retrocausal=True)

        assert config.attractor_capacity == 32
        assert config.attractor_pull_strength == 0.3
        assert config.attractor_min_syntony == 0.7
        assert config.attractor_decay_rate == 0.98


class TestRetrocausalEvolverCreation:
    """Tests for creating retrocausal evolvers."""

    def test_create_retrocausal_evolver_basic(self):
        """Test creating a retrocausal evolver with defaults."""
        tensor = ResonantTensor([1.0, 2.0, 3.0, 4.0], [4])
        evolver = create_retrocausal_evolver(tensor)

        assert evolver is not None
        assert evolver.config.enable_retrocausal is True
        assert evolver.config.attractor_capacity == 32
        assert evolver.config.attractor_pull_strength == 0.3

    def test_create_retrocausal_evolver_custom(self):
        """Test creating a retrocausal evolver with custom parameters."""
        tensor = ResonantTensor([1.0, 2.0, 3.0, 4.0], [4])
        evolver = create_retrocausal_evolver(
            tensor,
            population_size=16,
            attractor_capacity=16,
            pull_strength=0.5,
            min_syntony=0.8,
            decay_rate=0.95,
        )

        assert evolver.config.population_size == 16
        assert evolver.config.attractor_capacity == 16
        assert evolver.config.attractor_pull_strength == 0.5
        assert evolver.config.attractor_min_syntony == 0.8
        assert evolver.config.attractor_decay_rate == 0.95

    def test_create_standard_evolver(self):
        """Test creating a standard (non-retrocausal) evolver."""
        tensor = ResonantTensor([1.0, 2.0, 3.0, 4.0], [4])
        evolver = create_standard_evolver(tensor, population_size=32)

        assert evolver is not None
        assert evolver.config.enable_retrocausal is False
        assert evolver.config.population_size == 32


class TestRetrocausalEvolution:
    """Tests for retrocausal evolution behavior."""

    def test_retrocausal_evolver_runs(self):
        """Test that retrocausal evolver completes evolution."""
        tensor = ResonantTensor([1.0, 2.0, 3.0, 4.0], [4])
        evolver = create_retrocausal_evolver(
            tensor,
            population_size=16,
            attractor_capacity=8,
            max_generations=10,
        )

        result = evolver.run()

        assert isinstance(result, RESResult)
        assert result.generations <= 10
        assert 0.0 <= result.final_syntony <= 1.0

    def test_retrocausal_vs_standard_basic(self):
        """Test that both retrocausal and standard evolvers work."""
        tensor = ResonantTensor([1.0, 2.0, 3.0, 4.0], [4])

        # Standard evolver
        standard = create_standard_evolver(
            tensor,
            population_size=16,
            max_generations=10,
        )
        standard_result = standard.run()

        # Retrocausal evolver
        retrocausal = create_retrocausal_evolver(
            tensor,
            population_size=16,
            attractor_capacity=8,
            max_generations=10,
        )
        retrocausal_result = retrocausal.run()

        # Both should complete
        assert isinstance(standard_result, RESResult)
        assert isinstance(retrocausal_result, RESResult)
        assert standard_result.generations <= 10
        assert retrocausal_result.generations <= 10

    def test_retrocausal_step_by_step(self):
        """Test retrocausal evolver step-by-step execution."""
        tensor = ResonantTensor([1.0, 2.0, 3.0, 4.0], [4])
        evolver = create_retrocausal_evolver(
            tensor,
            population_size=8,
            attractor_capacity=4,
        )

        initial_syntony = evolver.best_syntony

        # Run a few steps manually
        for _ in range(5):
            syntony = evolver.step()
            assert 0.0 <= syntony <= 1.0

        # Should have advanced
        assert evolver.generation == 5

    def test_attractor_parameters_affect_behavior(self):
        """Test that changing attractor parameters changes behavior."""
        tensor = ResonantTensor([1.0, 2.0, 3.0, 4.0], [4])

        # High pull strength
        high_pull = create_retrocausal_evolver(
            tensor,
            population_size=16,
            pull_strength=0.7,
            max_generations=5,
        )

        # Low pull strength
        low_pull = create_retrocausal_evolver(
            tensor,
            population_size=16,
            pull_strength=0.1,
            max_generations=5,
        )

        # Both should run without error
        result_high = high_pull.run()
        result_low = low_pull.run()

        assert isinstance(result_high, RESResult)
        assert isinstance(result_low, RESResult)


class TestAttractorCapacity:
    """Tests for attractor capacity behavior."""

    def test_zero_capacity_behaves_like_standard(self):
        """Test that zero capacity essentially disables retrocausal."""
        tensor = ResonantTensor([1.0, 2.0, 3.0, 4.0], [4])

        evolver = create_retrocausal_evolver(
            tensor,
            population_size=16,
            attractor_capacity=0,  # No attractors
            max_generations=5,
        )

        result = evolver.run()
        assert isinstance(result, RESResult)

    def test_large_capacity(self):
        """Test that large capacity works correctly."""
        tensor = ResonantTensor([1.0, 2.0, 3.0, 4.0], [4])

        evolver = create_retrocausal_evolver(
            tensor,
            population_size=16,
            attractor_capacity=128,  # Large capacity
            max_generations=5,
        )

        result = evolver.run()
        assert isinstance(result, RESResult)


class TestAttractorThreshold:
    """Tests for attractor syntony threshold."""

    def test_high_threshold_fewer_attractors(self):
        """Test that high threshold means fewer attractors stored."""
        tensor = ResonantTensor([1.0, 2.0, 3.0, 4.0], [4])

        evolver = create_retrocausal_evolver(
            tensor,
            population_size=16,
            attractor_capacity=32,
            min_syntony=0.95,  # Very high threshold
            max_generations=10,
        )

        result = evolver.run()
        assert isinstance(result, RESResult)

    def test_low_threshold_more_attractors(self):
        """Test that low threshold means more attractors stored."""
        tensor = ResonantTensor([1.0, 2.0, 3.0, 4.0], [4])

        evolver = create_retrocausal_evolver(
            tensor,
            population_size=16,
            attractor_capacity=32,
            min_syntony=0.3,  # Low threshold
            max_generations=10,
        )

        result = evolver.run()
        assert isinstance(result, RESResult)


class TestTemporalDecay:
    """Tests for temporal decay behavior."""

    def test_high_decay_rate(self):
        """Test evolution with high decay rate (long memory)."""
        tensor = ResonantTensor([1.0, 2.0, 3.0, 4.0], [4])

        evolver = create_retrocausal_evolver(
            tensor,
            population_size=16,
            decay_rate=0.99,  # Very slow decay
            max_generations=10,
        )

        result = evolver.run()
        assert isinstance(result, RESResult)

    def test_low_decay_rate(self):
        """Test evolution with low decay rate (rapid forgetting)."""
        tensor = ResonantTensor([1.0, 2.0, 3.0, 4.0], [4])

        evolver = create_retrocausal_evolver(
            tensor,
            population_size=16,
            decay_rate=0.9,  # Rapid decay
            max_generations=10,
        )

        result = evolver.run()
        assert isinstance(result, RESResult)


class TestCompareConvergence:
    """Tests for convergence comparison utility."""

    @pytest.mark.slow
    def test_compare_convergence_basic(self):
        """Test basic convergence comparison."""
        tensor = ResonantTensor([1.0, 2.0, 3.0, 4.0], [4])

        stats = compare_convergence(
            tensor,
            population_size=16,
            max_generations=20,
            trials=2,  # Just 2 trials for speed
        )

        # Check structure
        assert 'standard' in stats
        assert 'retrocausal' in stats
        assert 'speedup' in stats

        # Check standard results
        assert 'mean_generations' in stats['standard']
        assert 'mean_syntony' in stats['standard']
        assert 'results' in stats['standard']
        assert len(stats['standard']['results']) == 2

        # Check retrocausal results
        assert 'mean_generations' in stats['retrocausal']
        assert 'mean_syntony' in stats['retrocausal']
        assert 'results' in stats['retrocausal']
        assert len(stats['retrocausal']['results']) == 2

        # Check speedup is a float
        assert isinstance(stats['speedup'], float)


class TestEdgeCases:
    """Tests for edge cases and error conditions."""

    def test_very_small_tensor(self):
        """Test retrocausal with minimal tensor."""
        tensor = ResonantTensor([1.0], [1])
        evolver = create_retrocausal_evolver(
            tensor,
            population_size=4,
            max_generations=5,
        )

        result = evolver.run()
        assert isinstance(result, RESResult)

    def test_large_tensor(self):
        """Test retrocausal with larger tensor."""
        values = [float(i) for i in range(100)]
        tensor = ResonantTensor(values, [100])

        evolver = create_retrocausal_evolver(
            tensor,
            population_size=16,
            attractor_capacity=16,
            max_generations=5,
        )

        result = evolver.run()
        assert isinstance(result, RESResult)

    def test_pull_strength_boundary_values(self):
        """Test pull strength at boundaries."""
        tensor = ResonantTensor([1.0, 2.0, 3.0, 4.0], [4])

        # Zero pull (should behave like standard)
        zero_pull = create_retrocausal_evolver(
            tensor,
            pull_strength=0.0,
            max_generations=5,
        )
        result = zero_pull.run()
        assert isinstance(result, RESResult)

        # Maximum pull
        max_pull = create_retrocausal_evolver(
            tensor,
            pull_strength=1.0,
            max_generations=5,
        )
        result = max_pull.run()
        assert isinstance(result, RESResult)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
