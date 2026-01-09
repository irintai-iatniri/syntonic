"""
Simple Retrocausal Convergence Benchmark

Measures convergence speed of standard vs retrocausal RES on a simple
optimization problem: finding high-syntony configurations in Q(φ) space.

This is a pure test of the retrocausal mechanism without confounding factors
from external fitness functions.
"""

from __future__ import annotations
import time
import numpy as np
from dataclasses import dataclass
from typing import List, Tuple

try:
    from syntonic._core import ResonantTensor
    from syntonic.resonant.retrocausal import create_retrocausal_evolver, create_standard_evolver
    RESONANT_AVAILABLE = True
except ImportError:
    RESONANT_AVAILABLE = False


@dataclass
class BenchmarkResult:
    """Results from a single trial."""
    approach: str
    generations_to_target: int
    final_syntony: float
    wall_time: float
    syntony_history: List[float]


def run_trial(
    approach: str,
    tensor_size: int = 100,
    target_syntony: float = 0.8,
    population_size: int = 32,
    max_generations: int = 1000,
) -> BenchmarkResult:
    """
    Run a single convergence trial.

    Args:
        approach: 'standard' or 'retrocausal'
        tensor_size: Size of tensor to optimize
        target_syntony: Target syntony to reach
        population_size: RES population size
        max_generations: Maximum generations to run

    Returns:
        BenchmarkResult
    """
    # Create random starting tensor (low initial syntony)
    values = np.random.randn(tensor_size) * 0.5
    template = ResonantTensor(values.tolist(), [tensor_size])

    start_time = time.time()

    if approach == 'standard':
        evolver = create_standard_evolver(
            template,
            population_size=population_size,
            max_generations=max_generations,
            convergence_threshold=1e-8,  # Very tight threshold
        )
    else:  # retrocausal
        evolver = create_retrocausal_evolver(
            template,
            population_size=population_size,
            attractor_capacity=32,
            pull_strength=0.3,
            min_syntony=0.6,
            max_generations=max_generations,
            convergence_threshold=1e-8,
        )

    # Run until we hit target syntony or max generations
    syntony_history = [evolver.best_syntony]
    generations_to_target = max_generations

    for gen in range(max_generations):
        syntony = evolver.step()
        syntony_history.append(syntony)

        if syntony >= target_syntony:
            generations_to_target = gen + 1
            break

    wall_time = time.time() - start_time

    return BenchmarkResult(
        approach=approach,
        generations_to_target=generations_to_target,
        final_syntony=evolver.best_syntony,
        wall_time=wall_time,
        syntony_history=syntony_history,
    )


def run_benchmark(
    trials: int = 10,
    tensor_size: int = 100,
    target_syntony: float = 0.8,
    population_size: int = 32,
    max_generations: int = 1000,
    verbose: bool = True,
) -> Tuple[dict, dict, float]:
    """
    Run full benchmark comparing standard vs retrocausal RES.

    Returns:
        (standard_stats, retrocausal_stats, speedup_percentage)
    """
    if not RESONANT_AVAILABLE:
        raise ImportError("Resonant engine not available")

    if verbose:
        print("=" * 70)
        print("SIMPLE RETROCAUSAL CONVERGENCE BENCHMARK")
        print("=" * 70)
        print(f"Configuration:")
        print(f"  Trials: {trials}")
        print(f"  Tensor size: {tensor_size}")
        print(f"  Target syntony: {target_syntony}")
        print(f"  Population size: {population_size}")
        print(f"  Max generations: {max_generations}")
        print()

    # Run standard RES
    if verbose:
        print("Running STANDARD RES trials...")
    standard_results = []
    for i in range(trials):
        result = run_trial(
            'standard',
            tensor_size=tensor_size,
            target_syntony=target_syntony,
            population_size=population_size,
            max_generations=max_generations,
        )
        standard_results.append(result)
        if verbose:
            status = "✓ reached" if result.final_syntony >= target_syntony else "✗ timeout"
            print(f"  Trial {i+1}/{trials}: {result.generations_to_target} gens, "
                  f"syntony={result.final_syntony:.4f} {status}")

    # Run retrocausal RES
    if verbose:
        print()
        print("Running RETROCAUSAL RES trials...")
    retrocausal_results = []
    for i in range(trials):
        result = run_trial(
            'retrocausal',
            tensor_size=tensor_size,
            target_syntony=target_syntony,
            population_size=population_size,
            max_generations=max_generations,
        )
        retrocausal_results.append(result)
        if verbose:
            status = "✓ reached" if result.final_syntony >= target_syntony else "✗ timeout"
            print(f"  Trial {i+1}/{trials}: {result.generations_to_target} gens, "
                  f"syntony={result.final_syntony:.4f} {status}")

    # Filter out trials that didn't reach target (for fair comparison)
    standard_successful = [r for r in standard_results if r.final_syntony >= target_syntony]
    retrocausal_successful = [r for r in retrocausal_results if r.final_syntony >= target_syntony]

    # Compute statistics
    standard_stats = {
        'mean_generations': np.mean([r.generations_to_target for r in standard_successful]) if standard_successful else float('inf'),
        'std_generations': np.std([r.generations_to_target for r in standard_successful]) if standard_successful else 0,
        'mean_syntony': np.mean([r.final_syntony for r in standard_results]),
        'success_rate': len(standard_successful) / trials,
        'mean_time': np.mean([r.wall_time for r in standard_results]),
    }

    retrocausal_stats = {
        'mean_generations': np.mean([r.generations_to_target for r in retrocausal_successful]) if retrocausal_successful else float('inf'),
        'std_generations': np.std([r.generations_to_target for r in retrocausal_successful]) if retrocausal_successful else 0,
        'mean_syntony': np.mean([r.final_syntony for r in retrocausal_results]),
        'success_rate': len(retrocausal_successful) / trials,
        'mean_time': np.mean([r.wall_time for r in retrocausal_results]),
    }

    # Compute speedup (only for successful trials)
    if standard_successful and retrocausal_successful:
        speedup = (standard_stats['mean_generations'] - retrocausal_stats['mean_generations']) / standard_stats['mean_generations'] * 100
    else:
        speedup = 0.0

    if verbose:
        print()
        print("=" * 70)
        print("RESULTS")
        print("=" * 70)
        print()
        print(f"Standard RES:")
        print(f"  Success rate: {standard_stats['success_rate'] * 100:.0f}% ({len(standard_successful)}/{trials})")
        if standard_successful:
            print(f"  Mean generations to target: {standard_stats['mean_generations']:.1f} ± {standard_stats['std_generations']:.1f}")
        print(f"  Mean final syntony: {standard_stats['mean_syntony']:.4f}")
        print(f"  Mean time: {standard_stats['mean_time']:.2f}s")
        print()
        print(f"Retrocausal RES:")
        print(f"  Success rate: {retrocausal_stats['success_rate'] * 100:.0f}% ({len(retrocausal_successful)}/{trials})")
        if retrocausal_successful:
            print(f"  Mean generations to target: {retrocausal_stats['mean_generations']:.1f} ± {retrocausal_stats['std_generations']:.1f}")
        print(f"  Mean final syntony: {retrocausal_stats['mean_syntony']:.4f}")
        print(f"  Mean time: {retrocausal_stats['mean_time']:.2f}s")
        print()
        print("=" * 70)
        if speedup > 0:
            print(f"SPEEDUP: {speedup:.1f}% faster convergence with retrocausal RES")
            if speedup >= 15:
                print("✓ MEETS EXPECTED 15-30% SPEEDUP CLAIM")
            elif speedup >= 10:
                print("~ APPROACHING expected 15-30% speedup")
            else:
                print("⚠ Below expected 15-30% speedup")
        elif speedup < 0:
            print(f"WARNING: Retrocausal was {-speedup:.1f}% slower")
        else:
            print("INCONCLUSIVE: Not enough successful trials")
        print("=" * 70)

    return standard_stats, retrocausal_stats, speedup


def main():
    """Run the simple convergence benchmark."""
    if not RESONANT_AVAILABLE:
        print("ERROR: Resonant engine not available.")
        print("Build with: maturin develop")
        return 1

    # Run benchmark with reasonable parameters
    print("Running convergence benchmark (10 trials)...")
    print()
    standard, retrocausal, speedup = run_benchmark(
        trials=10,
        tensor_size=50,  # Smaller for faster testing
        target_syntony=0.7,  # Achievable target
        population_size=32,
        max_generations=500,
        verbose=True,
    )

    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
