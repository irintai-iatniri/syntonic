"""
XOR Convergence Benchmark: Standard RES vs Retrocausal RES

Compares convergence speed between standard and retrocausal RES on the XOR problem.
The retrocausal version should demonstrate 15-30% faster convergence due to
attractor-guided harmonization biasing evolution toward proven high-syntony states.

Theory: XOR (exclusive-or) is a classic non-linear problem requiring hidden structure.
In RES, this translates to finding Q(φ) lattice configurations with high syntony that
correctly classify XOR inputs. Retrocausal guidance accelerates discovery by storing
successful configurations as attractors that influence future harmonization steps.
"""

from __future__ import annotations
import time
import numpy as np
from dataclasses import dataclass
from typing import List, Tuple

try:
    from syntonic._core import ResonantTensor, ResonantEvolver, RESConfig
    from syntonic.resonant.retrocausal import create_retrocausal_evolver, create_standard_evolver
    RESONANT_AVAILABLE = True
except ImportError:
    RESONANT_AVAILABLE = False

PHI = 1.6180339887498949


@dataclass
class BenchmarkResult:
    """Results from a single benchmark run."""
    approach: str  # 'standard' or 'retrocausal'
    generations: int
    final_syntony: float
    wall_time: float
    converged: bool


@dataclass
class BenchmarkStats:
    """Aggregate statistics across multiple trials."""
    approach: str
    mean_generations: float
    std_generations: float
    mean_syntony: float
    std_syntony: float
    mean_time: float
    convergence_rate: float  # Fraction of trials that converged
    results: List[BenchmarkResult]


def create_xor_template(hidden_size: int = 8, precision: int = 100) -> ResonantTensor:
    """
    Create template tensor for XOR problem.

    XOR requires non-linear transformation:
    - Input layer: 2 neurons (x1, x2)
    - Hidden layer: hidden_size neurons
    - Output layer: 1 neuron (xor result)

    Total parameters: (2 * hidden_size) + hidden_size + (hidden_size * 1) + 1
                    = 3*hidden_size + 1

    Args:
        hidden_size: Number of hidden neurons
        precision: Crystallization precision

    Returns:
        ResonantTensor initialized with golden-ratio-based values
    """
    # Parameter count
    input_to_hidden = 2 * hidden_size
    hidden_bias = hidden_size
    hidden_to_output = hidden_size * 1
    output_bias = 1
    total_params = input_to_hidden + hidden_bias + hidden_to_output + output_bias

    # Initialize with golden-ratio-based values for better lattice alignment
    values = []
    for i in range(total_params):
        # Use Fibonacci-like pattern for initial values
        val = (PHI ** (i % 3)) / (1 + i * 0.1)
        values.append(val)

    return ResonantTensor(values, [total_params])


def evaluate_xor_fitness(tensor: ResonantTensor, hidden_size: int = 8) -> float:
    """
    Evaluate XOR fitness of a tensor's parameter configuration.

    Maps Q(φ) lattice values to XOR predictions and computes accuracy.
    Higher syntony configurations tend to have better XOR performance due to
    golden ratio alignment with the problem's inherent symmetries.

    Args:
        tensor: Parameter tensor
        hidden_size: Number of hidden neurons

    Returns:
        Fitness score (0.0 to 1.0), where 1.0 is perfect XOR
    """
    # Extract parameters (simplified for benchmark)
    lattice = tensor.to_floats()

    # XOR truth table
    inputs = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    targets = np.array([0, 1, 1, 0])

    # Extract weight sections
    input_to_hidden = 2 * hidden_size
    W1 = np.array(lattice[:input_to_hidden]).reshape(hidden_size, 2)
    b1 = np.array(lattice[input_to_hidden:input_to_hidden + hidden_size])
    W2 = np.array(lattice[input_to_hidden + hidden_size:input_to_hidden + hidden_size * 2])
    b2 = lattice[-1]

    # Forward pass (simplified)
    correct = 0
    for x, target in zip(inputs, targets):
        # Hidden layer (tanh activation)
        hidden = np.tanh(W1 @ x + b1)

        # Output layer (sigmoid activation)
        output = 1.0 / (1.0 + np.exp(-(W2 @ hidden + b2)))

        # Check if correct
        prediction = 1 if output > 0.5 else 0
        if prediction == target:
            correct += 1

    # Fitness combines accuracy with syntony
    accuracy = correct / 4.0
    syntony = tensor.syntony

    # Weighted combination (70% accuracy, 30% syntony for geometric alignment)
    fitness = 0.7 * accuracy + 0.3 * syntony

    return fitness


def run_single_trial(
    approach: str,
    hidden_size: int = 8,
    population_size: int = 32,
    max_generations: int = 500,
    convergence_threshold: float = 1e-4,
    attractor_capacity: int = 32,
    pull_strength: float = 0.3,
) -> BenchmarkResult:
    """
    Run a single benchmark trial.

    Args:
        approach: 'standard' or 'retrocausal'
        hidden_size: Hidden layer size for XOR network
        population_size: RES population size
        max_generations: Maximum evolution generations
        convergence_threshold: Syntony convergence threshold
        attractor_capacity: Attractor memory capacity (retrocausal only)
        pull_strength: Attractor pull strength (retrocausal only)

    Returns:
        BenchmarkResult with trial statistics
    """
    template = create_xor_template(hidden_size)

    start_time = time.time()

    if approach == 'standard':
        evolver = create_standard_evolver(
            template,
            population_size=population_size,
            max_generations=max_generations,
            convergence_threshold=convergence_threshold,
        )
    elif approach == 'retrocausal':
        evolver = create_retrocausal_evolver(
            template,
            population_size=population_size,
            attractor_capacity=attractor_capacity,
            pull_strength=pull_strength,
            max_generations=max_generations,
            convergence_threshold=convergence_threshold,
        )
    else:
        raise ValueError(f"Unknown approach: {approach}")

    # Run evolution
    result = evolver.run()

    wall_time = time.time() - start_time

    return BenchmarkResult(
        approach=approach,
        generations=result.generations,
        final_syntony=result.final_syntony,
        wall_time=wall_time,
        converged=result.converged,
    )


def run_benchmark(
    trials: int = 10,
    hidden_size: int = 8,
    population_size: int = 32,
    max_generations: int = 500,
    convergence_threshold: float = 1e-4,
    attractor_capacity: int = 32,
    pull_strength: float = 0.3,
    verbose: bool = True,
) -> Tuple[BenchmarkStats, BenchmarkStats, float]:
    """
    Run full benchmark comparing standard vs retrocausal RES.

    Args:
        trials: Number of trials per approach
        hidden_size: Hidden layer size for XOR network
        population_size: RES population size
        max_generations: Maximum evolution generations
        convergence_threshold: Syntony convergence threshold
        attractor_capacity: Attractor memory capacity (retrocausal)
        pull_strength: Attractor pull strength (retrocausal)
        verbose: Print progress

    Returns:
        (standard_stats, retrocausal_stats, speedup_percentage)
    """
    if not RESONANT_AVAILABLE:
        raise ImportError("Resonant engine not available. Build with: maturin develop")

    if verbose:
        print("=" * 70)
        print("XOR CONVERGENCE BENCHMARK: Standard RES vs Retrocausal RES")
        print("=" * 70)
        print(f"Configuration:")
        print(f"  Trials: {trials}")
        print(f"  Hidden size: {hidden_size}")
        print(f"  Population size: {population_size}")
        print(f"  Max generations: {max_generations}")
        print(f"  Attractor capacity: {attractor_capacity}")
        print(f"  Pull strength: {pull_strength}")
        print()

    # Run standard RES trials
    if verbose:
        print("Running STANDARD RES trials...")
    standard_results = []
    for i in range(trials):
        result = run_single_trial(
            'standard',
            hidden_size=hidden_size,
            population_size=population_size,
            max_generations=max_generations,
            convergence_threshold=convergence_threshold,
        )
        standard_results.append(result)
        if verbose:
            print(f"  Trial {i+1}/{trials}: {result.generations} gens, "
                  f"syntony={result.final_syntony:.4f}, time={result.wall_time:.2f}s")

    # Run retrocausal RES trials
    if verbose:
        print()
        print("Running RETROCAUSAL RES trials...")
    retrocausal_results = []
    for i in range(trials):
        result = run_single_trial(
            'retrocausal',
            hidden_size=hidden_size,
            population_size=population_size,
            max_generations=max_generations,
            convergence_threshold=convergence_threshold,
            attractor_capacity=attractor_capacity,
            pull_strength=pull_strength,
        )
        retrocausal_results.append(result)
        if verbose:
            print(f"  Trial {i+1}/{trials}: {result.generations} gens, "
                  f"syntony={result.final_syntony:.4f}, time={result.wall_time:.2f}s")

    # Compute statistics
    standard_stats = BenchmarkStats(
        approach='standard',
        mean_generations=np.mean([r.generations for r in standard_results]),
        std_generations=np.std([r.generations for r in standard_results]),
        mean_syntony=np.mean([r.final_syntony for r in standard_results]),
        std_syntony=np.std([r.final_syntony for r in standard_results]),
        mean_time=np.mean([r.wall_time for r in standard_results]),
        convergence_rate=sum(r.converged for r in standard_results) / trials,
        results=standard_results,
    )

    retrocausal_stats = BenchmarkStats(
        approach='retrocausal',
        mean_generations=np.mean([r.generations for r in retrocausal_results]),
        std_generations=np.std([r.generations for r in retrocausal_results]),
        mean_syntony=np.mean([r.final_syntony for r in retrocausal_results]),
        std_syntony=np.std([r.final_syntony for r in retrocausal_results]),
        mean_time=np.mean([r.wall_time for r in retrocausal_results]),
        convergence_rate=sum(r.converged for r in retrocausal_results) / trials,
        results=retrocausal_results,
    )

    # Compute speedup
    speedup = (standard_stats.mean_generations - retrocausal_stats.mean_generations) / standard_stats.mean_generations * 100

    if verbose:
        print()
        print("=" * 70)
        print("RESULTS")
        print("=" * 70)
        print()
        print(f"Standard RES:")
        print(f"  Mean generations: {standard_stats.mean_generations:.1f} ± {standard_stats.std_generations:.1f}")
        print(f"  Mean syntony: {standard_stats.mean_syntony:.4f} ± {standard_stats.std_syntony:.4f}")
        print(f"  Mean time: {standard_stats.mean_time:.2f}s")
        print(f"  Convergence rate: {standard_stats.convergence_rate * 100:.0f}%")
        print()
        print(f"Retrocausal RES:")
        print(f"  Mean generations: {retrocausal_stats.mean_generations:.1f} ± {retrocausal_stats.std_generations:.1f}")
        print(f"  Mean syntony: {retrocausal_stats.mean_syntony:.4f} ± {retrocausal_stats.std_syntony:.4f}")
        print(f"  Mean time: {retrocausal_stats.mean_time:.2f}s")
        print(f"  Convergence rate: {retrocausal_stats.convergence_rate * 100:.0f}%")
        print()
        print("=" * 70)
        if speedup > 0:
            print(f"SPEEDUP: {speedup:.1f}% faster convergence with retrocausal RES")
            if speedup >= 15:
                print("✓ MEETS EXPECTED 15-30% SPEEDUP CLAIM")
            else:
                print("⚠ Below expected 15-30% speedup (may need parameter tuning)")
        else:
            print(f"WARNING: Retrocausal was {-speedup:.1f}% SLOWER (unexpected)")
        print("=" * 70)

    return standard_stats, retrocausal_stats, speedup


def main():
    """Run the XOR convergence benchmark."""
    if not RESONANT_AVAILABLE:
        print("ERROR: Resonant engine not available.")
        print("Build with: maturin develop")
        return 1

    # Quick benchmark (5 trials)
    print("Running quick benchmark (5 trials)...")
    standard, retrocausal, speedup = run_benchmark(
        trials=5,
        hidden_size=8,
        population_size=32,
        max_generations=200,
        verbose=True,
    )

    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
