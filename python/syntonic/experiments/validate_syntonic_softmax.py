"""
Syntonic Softmax Validation Suite

Runs validation experiments to test theoretical predictions of E8-based
syntonic softmax with Golden Cone filtering.

Usage:
    # Run all experiments
    python -m syntonic.experiments.validate_syntonic_softmax --all

    # Run specific experiment
    python -m syntonic.experiments.validate_syntonic_softmax --experiment xor

    # Run with custom configuration
    python -m syntonic.experiments.validate_syntonic_softmax --experiment xor --trials 5
"""

import argparse
import json
import time
from pathlib import Path
from typing import Dict, List

from syntonic.experiments.datasets.winding_datasets import (
    load_winding_xor_dataset,
    load_particle_dataset,
    load_csv_dataset,
    train_test_split,
)
from syntonic.experiments.runner import ExperimentRunner
from syntonic.experiments.metrics import ComparisonMetrics, MetricsTracker


def _average_fold_comparisons(comparisons: List[ComparisonMetrics]) -> ComparisonMetrics:
    """
    Average metrics across multiple cross-validation folds.
    
    Creates a new ComparisonMetrics with averaged results for each variant.
    
    Args:
        comparisons: List of ComparisonMetrics, one per fold
        
    Returns:
        ComparisonMetrics with averaged results
    """
    if not comparisons:
        return ComparisonMetrics()
    
    # Get all variant names from first comparison
    variant_names = list(comparisons[0].variants.keys())
    
    # Create result comparison
    result = ComparisonMetrics()
    
    for variant_name in variant_names:
        # Collect all trackers for this variant across folds
        trackers = []
        for comparison in comparisons:
            if variant_name in comparison.variants:
                trackers.append(comparison.variants[variant_name])
        
        if not trackers:
            continue
        
        # Create averaged tracker
        avg_tracker = MetricsTracker()
        
        # Average the key metrics across folds
        avg_best_accuracy = sum(t.best_accuracy for t in trackers) / len(trackers)
        avg_best_gen = sum(t.best_generation for t in trackers) / len(trackers)
        
        # Create synthetic history entries for averaging
        # Use the final accuracy from each fold as a data point
        for i, tracker in enumerate(trackers):
            summary = tracker.get_summary()
            final_acc = summary["final_accuracy"]
            
            # Add to averaged tracker's history
            avg_tracker.update(
                generation=i,  # Use fold index as "generation"
                accuracy=final_acc,
                syntony_dict={
                    "network": summary.get("syntony_correlation_network", 0.0),
                    "layers": [],
                    "softmax": summary.get("syntony_correlation_softmax", 0.0)
                }
            )
        
        # Override with averaged values
        avg_tracker.best_accuracy = avg_best_accuracy
        avg_tracker.best_generation = int(avg_best_gen)
        
        result.add_variant(variant_name, avg_tracker)
    
    return result



def run_experiment_xor(
    runner: ExperimentRunner,
    n_samples: int = 200,
    test_size: float = 0.2,
    variants: List[str] = None
) -> ComparisonMetrics:
    """
    Experiment 1A: XOR via Winding States

    Args:
        runner: Experiment runner
        n_samples: Number of samples to generate
        test_size: Fraction for test set
        variants: List of variants to test

    Returns:
        ComparisonMetrics with results
    """
    print("\n" + "="*70)
    print("EXPERIMENT 1A: XOR VIA WINDING STATES")
    print("="*70)

    # Load dataset
    X, y = load_winding_xor_dataset(n_samples=n_samples, noise=0.1, seed=42)
    X_train, y_train, X_test, y_test = train_test_split(X, y, test_size=test_size, seed=42)

    print(f"Dataset: {n_samples} samples")
    print(f"  Train: {len(X_train)} samples")
    print(f"  Test:  {len(X_test)} samples")
    print(f"  Classes: 2 (XOR)")

    # Run experiment
    return runner.run_experiment(
        "XOR Winding Classification",
        X_train, y_train,
        X_test, y_test,
        num_classes=2,
        variants=variants,
    )


def run_experiment_particles(
    runner: ExperimentRunner,
    variants: List[str] = None
) -> ComparisonMetrics:
    """
    Experiment 1B: Particle Classification (Leptons vs Quarks)

    Uses leave-one-out cross-validation due to small dataset size (9 samples).

    Args:
        runner: Experiment runner
        variants: List of variants to test

    Returns:
        ComparisonMetrics with results
    """
    print("\n" + "="*70)
    print("EXPERIMENT 1B: PARTICLE CLASSIFICATION (LEPTONS VS QUARKS)")
    print("="*70)

    # Load dataset
    X, y, names = load_particle_dataset()

    print(f"Dataset: {len(X)} particles")
    print(f"  Leptons (Class 0): {sum(1 for label in y if label == 0)}")
    print(f"  Quarks (Class 1):  {sum(1 for label in y if label == 1)}")

    # Leave-one-out cross-validation
    print("\nUsing leave-one-out cross-validation...")

    all_comparisons = []
    for i in range(len(X)):
        print(f"\nFold {i+1}/{len(X)}: Holding out {names[i]}")

        # Create train/test split
        X_test = [X[i]]
        y_test = [y[i]]
        X_train = [X[j] for j in range(len(X)) if j != i]
        y_train = [y[j] for j in range(len(y)) if j != i]

        # Run experiment
        comparison = runner.run_experiment(
            f"Particle Classification (Fold {i+1})",
            X_train, y_train,
            X_test, y_test,
            num_classes=2,
            variants=variants,
        )

        all_comparisons.append(comparison)

    # Average results across folds
    avg_comparison = _average_fold_comparisons(all_comparisons)
    
    print("\nLeave-one-out results (averaged across folds):")
    avg_comparison.print_comparison_table()

    return avg_comparison


def run_experiment_csv(
    runner: ExperimentRunner,
    csv_path: str = None,
    train_split: float = 0.8,
    variants: List[str] = None
) -> ComparisonMetrics:
    """
    Experiment 1C: CSV Winding Dataset

    Args:
        runner: Experiment runner
        csv_path: Path to CSV file
        train_split: Fraction for training
        variants: List of variants to test

    Returns:
        ComparisonMetrics with results
    """
    print("\n" + "="*70)
    print("EXPERIMENT 1C: CSV WINDING DATASET")
    print("="*70)

    # Load dataset
    X_train, y_train, X_test, y_test = load_csv_dataset(
        csv_path=csv_path,
        train_split=train_split,
        seed=42
    )

    # Determine number of classes
    num_classes = max(max(y_train), max(y_test)) + 1

    print(f"Dataset loaded from CSV")
    print(f"  Train: {len(X_train)} samples")
    print(f"  Test:  {len(X_test)} samples")
    print(f"  Classes: {num_classes}")

    # Run experiment
    return runner.run_experiment(
        "CSV Winding Classification",
        X_train, y_train,
        X_test, y_test,
        num_classes=num_classes,
        variants=variants,
    )


def save_results(results: Dict, output_path: Path):
    """Save experiment results to JSON file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Syntonic Softmax Validation Experiments"
    )

    parser.add_argument(
        "--experiment",
        type=str,
        choices=["xor", "particles", "csv", "all"],
        default="xor",
        help="Which experiment to run"
    )

    parser.add_argument(
        "--variants",
        type=str,
        nargs="+",
        default=["pytorch", "identity", "syntonic_e6"],
        help="Which variants to test"
    )

    parser.add_argument(
        "--trials",
        type=int,
        default=1,
        help="Number of trials to run (with different seeds)"
    )

    parser.add_argument(
        "--max-generations",
        type=int,
        default=50,
        help="Maximum training generations"
    )

    parser.add_argument(
        "--base-dim",
        type=int,
        default=64,
        help="Base embedding dimension"
    )

    parser.add_argument(
        "--num-blocks",
        type=int,
        default=3,
        help="Number of DHSR blocks"
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default="python/syntonic/experiments/results",
        help="Output directory for results"
    )

    parser.add_argument(
        "--csv-path",
        type=str,
        default=None,
        help="Path to CSV dataset (for csv experiment)"
    )

    args = parser.parse_args()

    print("="*70)
    print("SYNTONIC SOFTMAX VALIDATION SUITE")
    print("="*70)
    print(f"Configuration:")
    print(f"  Experiment: {args.experiment}")
    print(f"  Variants: {', '.join(args.variants)}")
    print(f"  Trials: {args.trials}")
    print(f"  Max Generations: {args.max_generations}")
    print(f"  Base Dim: {args.base_dim}")
    print(f"  Num Blocks: {args.num_blocks}")

    start_time = time.time()

    # Run experiments with appropriate max_winding per experiment
    results = {}

    if args.experiment == "xor" or args.experiment == "all":
        runner = ExperimentRunner(
            max_winding=5,  # XOR uses small windings
            base_dim=args.base_dim,
            num_blocks=args.num_blocks,
            precision=100,
            max_generations=args.max_generations,
            population_size=16,
            batch_size=16,
        )
        comparison = run_experiment_xor(runner, variants=args.variants)
        results["xor"] = comparison.get_comparison_table()

    if args.experiment == "particles" or args.experiment == "all":
        runner = ExperimentRunner(
            max_winding=3,  # Particles use small windings (max |2,1,1,0|)
            base_dim=args.base_dim,
            num_blocks=args.num_blocks,
            precision=100,
            max_generations=args.max_generations,
            population_size=16,
            batch_size=16,
        )
        comparison = run_experiment_particles(runner, variants=args.variants)
        results["particles"] = comparison.get_comparison_table()

    if args.experiment == "csv" or args.experiment == "all":
        runner = ExperimentRunner(
            max_winding=8,  # CSV needs 8 to include (5,5,0,0)
            base_dim=args.base_dim,
            num_blocks=args.num_blocks,
            precision=100,
            max_generations=args.max_generations,
            population_size=16,
            batch_size=16,
        )
        comparison = run_experiment_csv(
            runner,
            csv_path=args.csv_path,
            variants=args.variants
        )
        results["csv"] = comparison.get_comparison_table()

    total_time = time.time() - start_time

    print("\n" + "="*70)
    print("VALIDATION COMPLETE")
    print("="*70)
    print(f"Total time: {total_time:.2f}s")

    # Save results
    output_dir = Path(args.output_dir)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    output_file = output_dir / f"results_{args.experiment}_{timestamp}.json"
    save_results(results, output_file)


if __name__ == "__main__":
    main()