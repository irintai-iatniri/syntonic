"""
Benchmarks for comparing Resonant Engine vs Standard ML.

This module provides comparison benchmarks that demonstrate where the
Resonant Engine excels over standard ML frameworks (PyTorch/NumPy):

1. **Noise Robustness** - Syntony filter rejects noisy mutants
2. **Convergence Stability** - H-phase snap prevents gradient explosion
3. **Geometric Fidelity** - Exact lattice preserves golden structure

Example:
    >>> from syntonic.benchmarks import ConvergenceSpeedBenchmark
    >>> benchmark = ConvergenceSpeedBenchmark()
    >>> results = benchmark.run()
    >>> print(f"RES generations: {results['resonant']['generations']}")
    >>> print(f"PyTorch epochs: {results['pytorch']['epochs']}")
"""

from .datasets import make_xor, make_moons, make_circles, make_spiral, train_test_split
from .fitness import ClassificationFitness, RegressionFitness, WavefunctionFitness, evolve_with_fitness
from .convergence_benchmark import ConvergenceSpeedBenchmark, BenchmarkResult

__all__ = [
    # Datasets
    'make_xor',
    'make_moons',
    'make_circles',
    'make_spiral',
    'train_test_split',
    # Fitness wrappers
    'ClassificationFitness',
    'RegressionFitness',
    'WavefunctionFitness',
    'evolve_with_fitness',
    # Benchmarks
    'ConvergenceSpeedBenchmark',
    'BenchmarkResult',
]
