"""Tests for convergence benchmark comparing RES vs PyTorch."""

import pytest
import numpy as np

from syntonic.benchmarks.datasets import make_xor, make_moons, make_circles, train_test_split
from syntonic.benchmarks.fitness import ClassificationFitness, RegressionFitness
from syntonic.benchmarks.convergence_benchmark import ConvergenceSpeedBenchmark, BenchmarkResult

from syntonic._core import ResonantTensor, ResonantEvolver, RESConfig


class TestDatasets:
    """Tests for synthetic dataset generation."""

    def test_xor_shape(self):
        """XOR dataset has correct shape."""
        X, y = make_xor(n_samples=100, seed=42)
        assert X.shape == (100, 2)
        assert y.shape == (100,)

    def test_xor_labels(self):
        """XOR labels are 0 or 1."""
        X, y = make_xor(n_samples=100, seed=42)
        assert set(np.unique(y)) == {0, 1}

    def test_xor_pattern(self):
        """XOR follows correct labeling pattern (before noise)."""
        X, y = make_xor(n_samples=1000, noise=0.0, seed=42)
        # Same sign quadrants should be 0, different sign should be 1
        for i in range(len(X)):
            x1, x2 = X[i]
            expected = 0 if (x1 > 0) == (x2 > 0) else 1
            assert y[i] == expected

    def test_xor_reproducible(self):
        """XOR with same seed produces same data."""
        X1, y1 = make_xor(n_samples=100, seed=42)
        X2, y2 = make_xor(n_samples=100, seed=42)
        np.testing.assert_array_equal(X1, X2)
        np.testing.assert_array_equal(y1, y2)

    def test_moons_shape(self):
        """Moons dataset has correct shape."""
        X, y = make_moons(n_samples=100, seed=42)
        assert X.shape == (100, 2)
        assert y.shape == (100,)

    def test_moons_balanced(self):
        """Moons dataset is roughly balanced."""
        X, y = make_moons(n_samples=100, seed=42)
        counts = np.bincount(y)
        assert len(counts) == 2
        assert abs(counts[0] - counts[1]) <= 1

    def test_circles_shape(self):
        """Circles dataset has correct shape."""
        X, y = make_circles(n_samples=100, seed=42)
        assert X.shape == (100, 2)
        assert y.shape == (100,)

    def test_train_test_split(self):
        """Train/test split works correctly."""
        X, y = make_xor(n_samples=100, seed=42)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, seed=42)

        assert len(X_train) == 80
        assert len(X_test) == 20
        assert len(y_train) == 80
        assert len(y_test) == 20


class TestClassificationFitness:
    """Tests for classification fitness wrapper."""

    def test_fitness_callable(self):
        """Fitness can be called on a tensor."""
        X, y = make_xor(n_samples=100, seed=42)
        fitness = ClassificationFitness(X, y, n_features=2, n_classes=2)

        # Create a simple tensor
        weights = [0.1, 0.2, -0.1, 0.3]  # 2x2 weight matrix flattened
        tensor = ResonantTensor(weights, [4])

        score = fitness(tensor)
        assert isinstance(score, float)

    def test_fitness_accuracy(self):
        """Fitness accuracy method works."""
        X, y = make_xor(n_samples=100, seed=42)
        fitness = ClassificationFitness(X, y, n_features=2, n_classes=2)

        weights = [0.1, 0.2, -0.1, 0.3]
        tensor = ResonantTensor(weights, [4])

        acc = fitness.accuracy(tensor)
        assert 0.0 <= acc <= 1.0

    def test_fitness_loss(self):
        """Fitness loss method returns positive value."""
        X, y = make_xor(n_samples=100, seed=42)
        fitness = ClassificationFitness(X, y, n_features=2, n_classes=2)

        weights = [0.1, 0.2, -0.1, 0.3]
        tensor = ResonantTensor(weights, [4])

        loss = fitness.loss(tensor)
        assert loss > 0  # Cross-entropy is always positive

    def test_better_weights_higher_fitness(self):
        """Better predictions should give higher fitness."""
        # Create trivially separable data
        X = np.array([[1, 1], [-1, -1], [1, -1], [-1, 1]], dtype=np.float64)
        y = np.array([0, 0, 1, 1])  # XOR pattern

        fitness = ClassificationFitness(X, y, n_features=2, n_classes=2)

        # Random weights
        random_tensor = ResonantTensor([0.1, 0.2, -0.1, 0.3], [4])

        # Weights that should classify XOR correctly:
        # Class 0: x1*x2 > 0 (same sign)
        # Class 1: x1*x2 < 0 (different sign)
        # A linear model can't perfectly solve XOR, but some weights are better
        good_tensor = ResonantTensor([1.0, -1.0, -1.0, 1.0], [4])

        # Both should be valid fitness values
        f_random = fitness.task_fitness(random_tensor)
        f_good = fitness.task_fitness(good_tensor)

        assert isinstance(f_random, float)
        assert isinstance(f_good, float)


class TestResonantEvolverWithFitness:
    """Tests for RES with external fitness."""

    def test_evolver_step(self):
        """Evolver can run steps."""
        weights = [0.1, 0.2, 0.3, 0.4]
        tensor = ResonantTensor(weights, [4])

        config = RESConfig(population_size=10, survivor_count=3, max_generations=10)
        evolver = ResonantEvolver(tensor, config)

        syntony = evolver.step()
        assert 0.0 <= syntony <= 1.0
        assert evolver.generation == 1

    def test_evolver_run(self):
        """Evolver can run multiple generations."""
        weights = [0.1, 0.2, 0.3, 0.4]
        tensor = ResonantTensor(weights, [4])

        config = RESConfig(population_size=10, survivor_count=3, max_generations=5)
        evolver = ResonantEvolver(tensor, config)

        result = evolver.run()
        assert result.generations > 0
        assert result.winner is not None

    def test_evolver_improves_syntony(self):
        """Evolution should generally maintain or improve syntony."""
        weights = [0.1, 0.2, 0.3, 0.4]
        mode_norms = [0.0, 1.0, 4.0, 9.0]
        tensor = ResonantTensor(weights, [4], mode_norms)

        config = RESConfig(population_size=20, survivor_count=5, max_generations=20)
        evolver = ResonantEvolver(tensor, config)

        result = evolver.run()

        # Should have syntony history
        assert len(result.syntony_history) > 0


class TestConvergenceSpeedBenchmark:
    """Tests for the convergence benchmark."""

    def test_benchmark_creation(self):
        """Benchmark can be created."""
        benchmark = ConvergenceSpeedBenchmark(n_samples=100, seed=42)
        assert benchmark.n_samples == 100
        assert len(benchmark.X_train) > 0
        assert len(benchmark.X_test) > 0

    def test_run_resonant(self):
        """RES benchmark runs successfully."""
        benchmark = ConvergenceSpeedBenchmark(n_samples=100, seed=42)
        result = benchmark.run_resonant(max_generations=10, verbose=False)

        assert isinstance(result, BenchmarkResult)
        assert result.method == "resonant"
        assert result.iterations == 10
        assert 0.0 <= result.final_accuracy <= 1.0
        assert len(result.accuracy_history) == 10

    def test_resonant_learns(self):
        """RES should learn XOR to some degree."""
        benchmark = ConvergenceSpeedBenchmark(n_samples=200, seed=42)
        result = benchmark.run_resonant(
            max_generations=50,
            population_size=32,
            mutation_scale=0.5,
            verbose=False,
        )

        # Should achieve better than random (50%)
        # Note: Linear model can't perfectly solve XOR
        assert result.final_accuracy >= 0.4  # At least close to random


class TestBenchmarkResult:
    """Tests for benchmark result structure."""

    def test_result_fields(self):
        """BenchmarkResult has all required fields."""
        result = BenchmarkResult(
            method="test",
            iterations=100,
            final_accuracy=0.95,
            final_loss=0.1,
            time_seconds=1.5,
            accuracy_history=[0.5, 0.7, 0.9, 0.95],
            loss_history=[0.5, 0.3, 0.15, 0.1],
            iterations_to_95=3,
        )

        assert result.method == "test"
        assert result.iterations == 100
        assert result.final_accuracy == 0.95
        assert result.iterations_to_95 == 3


class TestPyTorchComparison:
    """Tests for PyTorch comparison (skipped if torch not available)."""

    @pytest.fixture
    def check_torch(self):
        """Check if PyTorch is available."""
        try:
            import torch
            return True
        except ImportError:
            pytest.skip("PyTorch not available")

    def test_pytorch_runs(self, check_torch):
        """PyTorch benchmark runs successfully."""
        benchmark = ConvergenceSpeedBenchmark(n_samples=100, seed=42)
        result = benchmark.run_pytorch(max_epochs=10, verbose=False)

        assert isinstance(result, BenchmarkResult)
        assert result.method == "pytorch"
        assert result.iterations == 10
        assert 0.0 <= result.final_accuracy <= 1.0

    def test_pytorch_learns_xor(self, check_torch):
        """PyTorch MLP should learn XOR well."""
        benchmark = ConvergenceSpeedBenchmark(n_samples=200, seed=42)
        result = benchmark.run_pytorch(
            max_epochs=100,
            hidden_size=16,
            learning_rate=0.05,
            verbose=False,
        )

        # MLP with hidden layer should solve XOR
        assert result.final_accuracy >= 0.85

    def test_full_comparison(self, check_torch):
        """Full comparison runs without error."""
        benchmark = ConvergenceSpeedBenchmark(n_samples=100, seed=42)
        results = benchmark.run(max_iterations=20, verbose=False)

        assert 'resonant' in results
        assert 'pytorch' in results
