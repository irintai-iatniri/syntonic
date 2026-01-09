"""
Quick Retrocausal Demo: Measure syntony improvement in fixed generations.

Instead of targeting absolute syntony, this measures how much improvement
each approach achieves in a fixed number of generations.
"""

import time
import numpy as np
from typing import Tuple

from syntonic._core import ResonantTensor
from syntonic.resonant.retrocausal import create_retrocausal_evolver, create_standard_evolver


def run_fixed_generations_trial(
    approach: str,
    tensor_size: int = 50,
    generations: int = 100,
    population_size: int = 32,
) -> Tuple[float, float, float]:
    """
    Run evolution for fixed generations and measure improvement.

    Returns:
        (initial_syntony, final_syntony, improvement)
    """
    # Create random starting tensor
    values = np.random.randn(tensor_size) * 0.5
    template = ResonantTensor(values.tolist(), [tensor_size])

    initial_syntony = template.syntony

    if approach == 'standard':
        evolver = create_standard_evolver(
            template,
            population_size=population_size,
            max_generations=generations,
        )
    else:  # retrocausal
        evolver = create_retrocausal_evolver(
            template,
            population_size=population_size,
            attractor_capacity=32,
            pull_strength=0.3,
            min_syntony=0.2,  # Low threshold to capture early improvements
            decay_rate=0.98,
            max_generations=generations,
        )

    # Run for exactly N generations
    for _ in range(generations):
        evolver.step()

    final_syntony = evolver.best_syntony
    improvement = final_syntony - initial_syntony

    return initial_syntony, final_syntony, improvement


def main():
    print("=" * 70)
    print("QUICK RETROCAUSAL DEMO: Syntony Improvement in 100 Generations")
    print("=" * 70)
    print()

    trials = 5
    generations = 100
    tensor_size = 50

    print(f"Running {trials} trials of {generations} generations each...")
    print(f"Tensor size: {tensor_size}")
    print()

    # Standard RES trials
    print("Standard RES:")
    standard_improvements = []
    for i in range(trials):
        initial, final, improvement = run_fixed_generations_trial(
            'standard',
            tensor_size=tensor_size,
            generations=generations,
        )
        standard_improvements.append(improvement)
        print(f"  Trial {i+1}: {initial:.4f} → {final:.4f} (Δ = {improvement:+.4f})")

    print()

    # Retrocausal RES trials
    print("Retrocausal RES:")
    retrocausal_improvements = []
    for i in range(trials):
        initial, final, improvement = run_fixed_generations_trial(
            'retrocausal',
            tensor_size=tensor_size,
            generations=generations,
        )
        retrocausal_improvements.append(improvement)
        print(f"  Trial {i+1}: {initial:.4f} → {final:.4f} (Δ = {improvement:+.4f})")

    print()
    print("=" * 70)
    print("RESULTS")
    print("=" * 70)

    std_mean = np.mean(standard_improvements)
    std_std = np.std(standard_improvements)
    retro_mean = np.mean(retrocausal_improvements)
    retro_std = np.std(retrocausal_improvements)

    print(f"Standard RES mean improvement:     {std_mean:+.4f} ± {std_std:.4f}")
    print(f"Retrocausal RES mean improvement:  {retro_mean:+.4f} ± {retro_std:.4f}")
    print()

    if std_mean > 0:
        relative_improvement = (retro_mean - std_mean) / std_mean * 100
        print(f"Relative improvement: {relative_improvement:+.1f}%")
        print()

        if relative_improvement >= 15:
            print("✓ MEETS EXPECTED 15-30% IMPROVEMENT TARGET")
        elif relative_improvement >= 10:
            print("~ APPROACHING expected improvement (10-15%)")
        elif relative_improvement > 0:
            print(f"✓ Retrocausal shows {relative_improvement:.1f}% improvement")
        else:
            print("⚠ Retrocausal underperformed")
    else:
        print("⚠ Standard RES showed no improvement (may need more generations)")

    print("=" * 70)


if __name__ == "__main__":
    main()
