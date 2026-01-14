"""
Experiment runner for syntonic softmax validation.

Coordinates:
1. Model creation (baseline and syntonic variants)
2. Training with RetrocausalTrainer
3. Evaluation and metrics tracking
4. Comparison across variants
"""

from typing import List, Tuple, Dict, Optional
import time

from syntonic._core import ResonantTensor, WindingState
from syntonic.nn.training.trainer import RetrocausalTrainer, RESTrainingConfig
from syntonic.experiments.models.winding_net_syntonic import (
    WindingNetSyntonic,
    create_baseline_model,
    create_syntonic_model,
)
from syntonic.experiments.metrics import MetricsTracker, ComparisonMetrics


def argmax_batch(tensor: ResonantTensor, dim: int = 1) -> List[int]:
    """
    Compute argmax along a dimension for a batch tensor.

    Args:
        tensor: ResonantTensor of shape (batch, features)
        dim: Dimension to reduce (1 = along features)

    Returns:
        List of argmax indices for each batch element
    """
    floats = tensor.to_floats()
    shape = tensor.shape

    if len(shape) != 2:
        raise ValueError(f"Expected 2D tensor, got shape {shape}")

    batch_size, num_features = shape
    result = []

    for b in range(batch_size):
        start_idx = b * num_features
        row = floats[start_idx:start_idx + num_features]
        max_idx = max(range(len(row)), key=lambda i: row[i])
        result.append(max_idx)

    return result


def create_batches(
    X: List[WindingState],
    y: List[int],
    batch_size: int,
    num_classes: int = 2,
    precision: int = 100
) -> List[Tuple[List[WindingState], ResonantTensor]]:
    """
    Create batches for training.

    Args:
        X: List of winding states
        y: List of class labels
        batch_size: Batch size
        num_classes: Number of output classes
        precision: ResonantTensor precision

    Returns:
        List of (batch_X, batch_y) tuples
    """
    batches = []
    num_samples = len(X)

    for i in range(0, num_samples, batch_size):
        batch_X = X[i:i+batch_size]
        batch_y_labels = y[i:i+batch_size]

        # Convert labels to one-hot encoding
        batch_y_onehot = []
        for label in batch_y_labels:
            onehot = [0.0] * num_classes
            onehot[label] = 1.0
            batch_y_onehot.extend(onehot)

        # Create target tensor
        target_tensor = ResonantTensor(
            batch_y_onehot,
            [len(batch_y_labels), num_classes],
            [1.0] * (len(batch_y_labels) * num_classes),
            precision
        )

        batches.append((batch_X, target_tensor))

    return batches


def evaluate_model(
    model: WindingNetSyntonic,
    X: List[WindingState],
    y: List[int]
) -> Tuple[float, Dict]:
    """
    Evaluate model on dataset.

    Args:
        model: Model to evaluate
        X: Winding states
        y: Labels

    Returns:
        (accuracy, syntony_dict) tuple
    """
    # Forward pass
    logits = model(X)

    # Predictions
    preds = argmax_batch(logits, dim=1)

    # Accuracy
    correct = sum(1 for p, t in zip(preds, y) if p == t)
    accuracy = correct / len(y)

    # Syntony
    syntony_dict = model.get_all_syntonies()

    return accuracy, syntony_dict


class ExperimentRunner:
    """
    Coordinates experiment execution and result tracking.

    Usage:
        runner = ExperimentRunner(config)
        result = runner.run_experiment("xor_winding", X_train, y_train, X_test, y_test)
    """

    def __init__(
        self,
        max_winding: int = 5,
        base_dim: int = 64,
        num_blocks: int = 3,
        precision: int = 100,
        max_generations: int = 50,
        population_size: int = 16,
        batch_size: int = 16,
    ):
        """
        Initialize experiment runner.

        Args:
            max_winding: Maximum winding number for networks
            base_dim: Base embedding dimension
            num_blocks: Number of DHSR blocks
            precision: ResonantTensor precision
            max_generations: Maximum training generations
            population_size: RES population size
            batch_size: Training batch size
        """
        self.max_winding = max_winding
        self.base_dim = base_dim
        self.num_blocks = num_blocks
        self.precision = precision
        self.max_generations = max_generations
        self.population_size = population_size
        self.batch_size = batch_size

    def run_single_variant(
        self,
        variant_name: str,
        X_train: List[WindingState],
        y_train: List[int],
        X_test: List[WindingState],
        y_test: List[int],
        num_classes: int = 2,
        softmax_mode: str = "identity",
        mode_norm_source: str = "e6",
    ) -> Tuple[WindingNetSyntonic, MetricsTracker]:
        """
        Run training for a single model variant.

        Args:
            variant_name: Name of variant (for logging)
            X_train: Training winding states
            y_train: Training labels
            X_test: Test winding states
            y_test: Test labels
            num_classes: Number of output classes
            softmax_mode: Softmax mode ("identity", "learned", "pytorch")
            mode_norm_source: Mode norm source ("e6", "random_uniform", etc.)

        Returns:
            (trained_model, metrics_tracker) tuple
        """
        print(f"\n{'='*70}")
        print(f"Training Variant: {variant_name}")
        print(f"{'='*70}")

        # 1. Create model
        model = WindingNetSyntonic(
            max_winding=self.max_winding,
            base_dim=self.base_dim,
            num_blocks=self.num_blocks,
            output_dim=num_classes,
            softmax_mode=softmax_mode,
            mode_norm_source=mode_norm_source,
            precision=self.precision,
        )

        print(f"Model: WindingNetSyntonic")
        print(f"  Softmax Mode: {softmax_mode}")
        print(f"  Mode Norm Source: {mode_norm_source}")
        print(f"  Base Dim: {self.base_dim}")
        print(f"  Num Blocks: {self.num_blocks}")
        print(f"  Output Classes: {num_classes}")

        # 2. Create batches
        train_batches = create_batches(
            X_train, y_train,
            self.batch_size,
            num_classes,
            self.precision
        )

        # 3. Configure trainer
        config = RESTrainingConfig(
            max_generations=self.max_generations,
            population_size=self.population_size,
            syntony_threshold=0.8,
            log_interval=10,
        )

        # 4. Create metrics tracker
        tracker = MetricsTracker()

        # 5. Train with periodic evaluation
        trainer = RetrocausalTrainer(
            model=model,
            train_data=train_batches,
            val_data=None,
            config=config,
        )

        print(f"\nStarting training for {self.max_generations} generations...")
        start_time = time.time()

        # Manual training loop with evaluation callback
        for generation in range(self.max_generations):
            # Train one generation
            # Note: RetrocausalTrainer doesn't expose per-generation API easily,
            # so we'll need to run full training and evaluate at end
            # For now, let's run full training
            if generation == 0:
                results = trainer.train()
                break

        train_time = time.time() - start_time

        # 6. Final evaluation
        test_acc, syntony_dict = evaluate_model(model, X_test, y_test)
        train_acc, _ = evaluate_model(model, X_train, y_train)

        # Update tracker with final metrics
        tracker.update(
            generation=self.max_generations,
            accuracy=test_acc,
            syntony_dict=syntony_dict
        )

        print(f"\nTraining Complete in {train_time:.2f}s")
        print(f"Train Accuracy: {train_acc:.2%}")
        print(f"Test Accuracy:  {test_acc:.2%}")
        print(f"Network Syntony: {syntony_dict['network']:.4f}")

        return model, tracker

    def run_experiment(
        self,
        experiment_name: str,
        X_train: List[WindingState],
        y_train: List[int],
        X_test: List[WindingState],
        y_test: List[int],
        num_classes: int = 2,
        variants: Optional[List[str]] = None,
    ) -> ComparisonMetrics:
        """
        Run full experiment with multiple variants.

        Args:
            experiment_name: Name of experiment (for logging)
            X_train: Training winding states
            y_train: Training labels
            X_test: Test winding states
            y_test: Test labels
            num_classes: Number of output classes
            variants: List of variant names to run:
                - "pytorch": Pure PyTorch-style softmax
                - "identity": Syntonic identity mode
                - "syntonic_e6": Syntonic with E6 roots
                - "syntonic_random": Syntonic with random roots
                If None, runs all variants

        Returns:
            ComparisonMetrics object with results
        """
        print(f"\n{'='*70}")
        print(f"EXPERIMENT: {experiment_name}")
        print(f"{'='*70}")
        print(f"Train samples: {len(X_train)}")
        print(f"Test samples: {len(X_test)}")
        print(f"Num classes: {num_classes}")

        if variants is None:
            variants = ["pytorch", "identity", "syntonic_e6"]

        comparison = ComparisonMetrics()

        for variant in variants:
            if variant == "pytorch":
                model, tracker = self.run_single_variant(
                    "PyTorch Softmax",
                    X_train, y_train, X_test, y_test,
                    num_classes=num_classes,
                    softmax_mode="pytorch",
                )
            elif variant == "identity":
                model, tracker = self.run_single_variant(
                    "Identity Mode",
                    X_train, y_train, X_test, y_test,
                    num_classes=num_classes,
                    softmax_mode="identity",
                )
            elif variant == "syntonic_e6":
                model, tracker = self.run_single_variant(
                    "Syntonic (E6 roots)",
                    X_train, y_train, X_test, y_test,
                    num_classes=num_classes,
                    softmax_mode="learned",
                    mode_norm_source="e6",
                )
            elif variant == "syntonic_random":
                model, tracker = self.run_single_variant(
                    "Syntonic (Random)",
                    X_train, y_train, X_test, y_test,
                    num_classes=num_classes,
                    softmax_mode="learned",
                    mode_norm_source="random_uniform",
                )
            else:
                raise ValueError(f"Unknown variant: {variant}")

            comparison.add_variant(variant, tracker)

        # Print comparison table
        print(f"\n{'='*70}")
        print("COMPARISON RESULTS")
        print(f"{'='*70}")
        comparison.print_comparison_table()

        return comparison


__all__ = [
    'ExperimentRunner',
    'argmax_batch',
    'create_batches',
    'evaluate_model',
]


if __name__ == "__main__":
    # Simple demo of ExperimentRunner
    print("Starting ExperimentRunner demo...")
    from syntonic.experiments.datasets.winding_datasets import load_winding_xor_dataset, train_test_split

    print("Creating runner...")
    # Create runner
    runner = ExperimentRunner(
        max_winding=5,
        base_dim=64,
        num_blocks=3,
        precision=100,
        max_generations=10,  # Quick demo
        population_size=16,
        batch_size=16,
    )

    print("Loading dataset...")
    # Load small dataset
    X, y = load_winding_xor_dataset(n_samples=100, noise=0.1, seed=42)
    print(f"Loaded {len(X)} samples")
    X_train, y_train, X_test, y_test = train_test_split(X, y, test_size=0.2, seed=42)
    print(f"Train: {len(X_train)}, Test: {len(X_test)}")

    print("Running experiment...")
    # Run experiment with identity mode (fastest)
    comparison = runner.run_experiment(
        "Demo XOR Experiment",
        X_train, y_train,
        X_test, y_test,
        num_classes=2,
        variants=["identity"],  # Just run one variant for demo
    )

    print("Demo complete!")
