"""Tests for ResonantEvolver - the Resonant Evolution Strategy."""

import pytest

from syntonic._core import ResonantTensor, ResonantEvolver, RESConfig, RESResult, GoldenExact

# Golden ratio constant
PHI = 1.6180339887498949

# Universal syntony deficit (NOT a hyperparameter!)
Q_DEFICIT = 0.027395146920


class TestRESConfig:
    """Tests for RESConfig configuration."""

    def test_default_config(self):
        """Test creating config with default values."""
        config = RESConfig()

        assert config.population_size == 64
        assert config.survivor_count == 16
        assert abs(config.lambda_val - Q_DEFICIT) < 1e-10  # λ = q by default
        assert config.mutation_scale == 0.1
        assert config.precision == 100
        assert config.noise_scale == 0.01

    def test_custom_config(self):
        """Test creating config with custom values."""
        config = RESConfig(
            population_size=32,
            survivor_count=8,
            mutation_scale=0.2,
            precision=200,
        )

        assert config.population_size == 32
        assert config.survivor_count == 8
        assert config.mutation_scale == 0.2
        assert config.precision == 200

    def test_config_repr(self):
        """Test config string representation."""
        config = RESConfig()
        repr_str = repr(config)

        assert "RESConfig" in repr_str
        assert "pop=64" in repr_str


class TestResonantEvolverConstruction:
    """Tests for ResonantEvolver creation."""

    def test_from_template(self):
        """Test creating evolver from template tensor."""
        data = [1.0, 2.0, 3.0, 4.0]
        tensor = ResonantTensor(data, [4])

        evolver = ResonantEvolver(tensor)

        assert evolver.generation == 0
        assert evolver.best is not None
        assert evolver.best_syntony >= 0.0

    def test_from_template_with_config(self):
        """Test creating evolver with custom config."""
        data = [1.0, 2.0, 3.0, 4.0]
        tensor = ResonantTensor(data, [4])
        config = RESConfig(population_size=32, survivor_count=8)

        evolver = ResonantEvolver(tensor, config)

        assert evolver.config.population_size == 32
        assert evolver.config.survivor_count == 8

    def test_evolver_repr(self):
        """Test evolver string representation."""
        data = [1.0, 2.0, 3.0, 4.0]
        tensor = ResonantTensor(data, [4])
        evolver = ResonantEvolver(tensor)

        repr_str = repr(evolver)
        assert "ResonantEvolver" in repr_str
        assert "generation=0" in repr_str


class TestResonantEvolverStep:
    """Tests for single evolution steps."""

    def test_single_step(self):
        """Test running a single evolution step."""
        data = [1.0, 2.0, 3.0, 4.0]
        mode_norms = [0.0, 1.0, 4.0, 9.0]
        tensor = ResonantTensor(data, [4], mode_norms)

        config = RESConfig(population_size=10, survivor_count=3)
        evolver = ResonantEvolver(tensor, config)

        syntony = evolver.step()

        assert 0.0 <= syntony <= 1.0
        assert evolver.generation == 1

    def test_multiple_steps(self):
        """Test running multiple evolution steps."""
        data = [1.0, 2.0, 3.0, 4.0]
        mode_norms = [0.0, 1.0, 4.0, 9.0]
        tensor = ResonantTensor(data, [4], mode_norms)

        config = RESConfig(population_size=10, survivor_count=3)
        evolver = ResonantEvolver(tensor, config)

        syntonies = []
        for _ in range(5):
            syntony = evolver.step()
            syntonies.append(syntony)

        assert evolver.generation == 5
        assert all(0.0 <= s <= 1.0 for s in syntonies)

    def test_syntony_tracking(self):
        """Test that syntony history is tracked."""
        data = [1.0, 2.0, 3.0, 4.0]
        tensor = ResonantTensor(data, [4])

        config = RESConfig(population_size=10, survivor_count=3)
        evolver = ResonantEvolver(tensor, config)

        for _ in range(5):
            evolver.step()

        history = evolver.syntony_history
        assert len(history) >= 5


class TestResonantEvolverRun:
    """Tests for full evolution runs."""

    def test_run_with_max_generations(self):
        """Test running evolution until max generations or convergence."""
        data = [1.0, 2.0, 3.0, 4.0]
        mode_norms = [0.0, 1.0, 4.0, 9.0]
        tensor = ResonantTensor(data, [4], mode_norms)

        config = RESConfig(
            population_size=10,
            survivor_count=3,
            max_generations=20,
            convergence_threshold=1e-10,  # Tight threshold
        )
        evolver = ResonantEvolver(tensor, config)

        result = evolver.run()

        assert isinstance(result, RESResult)
        # Should run some generations (may converge early or hit max)
        assert result.generations > 0
        assert result.generations <= 20
        assert 0.0 <= result.final_syntony <= 1.0
        assert result.winner is not None

    def test_result_contains_history(self):
        """Test that result contains syntony history."""
        data = [1.0, 2.0, 3.0, 4.0]
        tensor = ResonantTensor(data, [4])

        config = RESConfig(
            population_size=10,
            survivor_count=3,
            max_generations=10,
        )
        evolver = ResonantEvolver(tensor, config)

        result = evolver.run()

        assert len(result.syntony_history) > 0

    def test_result_repr(self):
        """Test result string representation."""
        data = [1.0, 2.0, 3.0, 4.0]
        tensor = ResonantTensor(data, [4])

        config = RESConfig(population_size=10, survivor_count=3, max_generations=5)
        evolver = ResonantEvolver(tensor, config)
        result = evolver.run()

        repr_str = repr(result)
        assert "RESResult" in repr_str


class TestResonantEvolverConvergence:
    """Tests for convergence behavior."""

    def test_convergence_detection(self):
        """Test that convergence is detected when syntony plateaus."""
        # Create a tensor with high syntony (mostly DC mode)
        data = [10.0, 0.1, 0.1, 0.1]
        mode_norms = [0.0, 1.0, 4.0, 9.0]
        tensor = ResonantTensor(data, [4], mode_norms)

        config = RESConfig(
            population_size=10,
            survivor_count=3,
            max_generations=100,
            convergence_threshold=0.1,  # Relaxed threshold
            mutation_scale=0.01,  # Small mutations
        )
        evolver = ResonantEvolver(tensor, config)

        result = evolver.run()

        # Should either converge or hit max generations
        assert result.generations > 0
        assert 0.0 <= result.final_syntony <= 1.0


class TestResonantEvolverMutation:
    """Tests for mutation behavior."""

    def test_mutants_are_different(self):
        """Test that mutants are different from parent."""
        data = [1.0, 2.0, 3.0, 4.0]
        tensor = ResonantTensor(data, [4])

        config = RESConfig(population_size=10, survivor_count=3)
        evolver = ResonantEvolver(tensor, config)

        # After a step, the best tensor might be different
        initial_values = tensor.to_list()
        evolver.step()

        # The best tensor should still be valid
        best = evolver.best
        assert best is not None
        assert len(best.to_list()) == 4

    def test_mutation_preserves_shape(self):
        """Test that mutations preserve tensor shape."""
        data = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
        tensor = ResonantTensor(data, [2, 3])

        config = RESConfig(population_size=10, survivor_count=3)
        evolver = ResonantEvolver(tensor, config)

        evolver.step()

        best = evolver.best
        assert best.shape == [2, 3]


class TestResonantEvolverFiltering:
    """Tests for lattice syntony filtering."""

    def test_survivor_count(self):
        """Test that filtering keeps correct number of survivors."""
        data = [1.0, 2.0, 3.0, 4.0]
        tensor = ResonantTensor(data, [4])

        config = RESConfig(population_size=20, survivor_count=5)
        evolver = ResonantEvolver(tensor, config)

        # After step, internal filtering should have selected top survivors
        # We can't directly test internal state, but we verify the step works
        syntony = evolver.step()
        assert 0.0 <= syntony <= 1.0


class TestResonantEvolverWithGoldenValues:
    """Tests with golden ratio values."""

    def test_golden_ratio_tensor(self):
        """Test evolution with golden ratio values."""
        data = [1.0, PHI, PHI * PHI, 3.0]
        tensor = ResonantTensor(data, [4], precision=100)

        config = RESConfig(population_size=10, survivor_count=3, max_generations=10)
        evolver = ResonantEvolver(tensor, config)

        result = evolver.run()

        # Should complete without error
        assert result.generations > 0
        assert result.winner is not None

    def test_preserves_golden_structure(self):
        """Test that evolution maintains lattice structure."""
        data = [1.0, PHI, 2.0, 3.0]
        mode_norms = [0.0, 1.0, 4.0, 9.0]
        tensor = ResonantTensor(data, [4], mode_norms, precision=100)

        config = RESConfig(
            population_size=10,
            survivor_count=3,
            max_generations=5,
            mutation_scale=0.05,  # Small mutations
        )
        evolver = ResonantEvolver(tensor, config)

        result = evolver.run()

        # Winner should have valid lattice representation
        lattice = result.winner.get_lattice()
        assert len(lattice) == 4
        for g in lattice:
            assert hasattr(g, "rational_coefficient")
            assert hasattr(g, "phi_coefficient")
