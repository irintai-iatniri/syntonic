"""
Syntonic Benchmarks - Tools for benchmarking syntonic networks.

Provides:
- Standard benchmark suite
- Convergence rate analysis
- Ablation studies
"""

from syntonic.nn.benchmarks.standard import (
    BenchmarkSuite,
    BenchmarkResult,
    run_mnist_benchmark,
    run_cifar_benchmark,
)
from syntonic.nn.benchmarks.convergence import (
    ConvergenceAnalyzer,
    ConvergenceMetrics,
    compare_convergence,
)
from syntonic.nn.benchmarks.ablation import (
    AblationStudy,
    AblationResult,
    run_component_ablation,
)

__all__ = [
    'BenchmarkSuite',
    'BenchmarkResult',
    'run_mnist_benchmark',
    'run_cifar_benchmark',
    'ConvergenceAnalyzer',
    'ConvergenceMetrics',
    'compare_convergence',
    'AblationStudy',
    'AblationResult',
    'run_component_ablation',
]
