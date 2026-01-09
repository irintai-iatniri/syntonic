"""
Pure Resonant Transformer Benchmark: Parity Task

Tests the Pure Resonant Transformer on the classic parity prediction task
(XOR of N binary inputs). Compares against a simple MLP baseline.
"""

import math
import time
import random

from syntonic._core import ResonantTensor, GoldenExact
from syntonic.pure.resonant_transformer import PureResonantTransformer

# Constants
PHI = (1 + math.sqrt(5)) / 2
Q_SYNTONY = 0.027395146920


def generate_parity_data(n_samples: int, n_bits: int = 4):
    """Generate parity dataset: predict XOR of N bits."""
    X = []
    y = []
    for _ in range(n_samples):
        bits = [random.randint(0, 1) for _ in range(n_bits)]
        label = sum(bits) % 2  # Parity (XOR)
        X.append([float(b) for b in bits])
        y.append(label)
    return X, y


def accuracy(preds, labels):
    """Compute accuracy."""
    correct = sum(1 for p, l in zip(preds, labels) if p == l)
    return correct / len(labels)


class SimpleRESOptimizer:
    """Simple Resonant Evolution Strategy for training."""

    def __init__(self, model: PureResonantTransformer, pop_size: int = 20):
        self.model = model
        self.pop_size = pop_size

    def train(self, X, y, generations: int = 50):
        """Train the model using evolution strategy."""
        best_acc = 0.0
        best_params = None

        # Get initial parameters
        params = [
            self.model.input_proj,
            self.model.output_proj,
            self.model.ffn.weights
        ]

        for gen in range(generations):
            candidates = []

            for _ in range(self.pop_size):
                # Mutate parameters
                mutated = []
                for p in params:
                    lattice = p.to_lattice_list()
                    new_lattice = []
                    for g in lattice:
                        if random.random() < 0.1:
                            da = random.randint(-1, 1)
                            db = random.randint(-1, 1)
                            new_lattice.append(g + GoldenExact.from_integers(da, db))
                        else:
                            new_lattice.append(g)
                    mutated.append(ResonantTensor.from_golden_exact(new_lattice, p.shape))

                # Evaluate
                acc = self._evaluate(mutated, X, y)
                candidates.append((mutated, acc))

            # Select best
            candidates.sort(key=lambda x: x[1], reverse=True)
            if candidates[0][1] > best_acc:
                best_acc = candidates[0][1]
                best_params = candidates[0][0]
                params = best_params

            if gen % 10 == 0:
                print(f"  Gen {gen}: Best Accuracy = {best_acc:.2%}")

            if best_acc >= 0.99:
                break

        # Apply best params
        if best_params:
            self.model.input_proj = best_params[0]
            self.model.output_proj = best_params[1]
            self.model.ffn.weights = best_params[2]

        return best_acc

    def _evaluate(self, params, X, y):
        """Evaluate a parameter set."""
        self.model.input_proj = params[0]
        self.model.output_proj = params[1]
        self.model.ffn.weights = params[2]

        preds = []
        for x_sample in X:
            x_tensor = ResonantTensor(x_sample, [1, len(x_sample)])
            out = self.model.forward(x_tensor)
            out_floats = out.to_floats()
            pred = 1 if out_floats[1] > out_floats[0] else 0
            preds.append(pred)

        return accuracy(preds, y)


def benchmark_parity():
    """Run the Parity benchmark."""
    print("=" * 60)
    print("PURE RESONANT TRANSFORMER BENCHMARK: PARITY TASK")
    print("=" * 60)

    n_bits = 4
    n_train = 100
    n_test = 50
    generations = 100

    # Generate data
    print(f"\nGenerating data: {n_bits}-bit parity...")
    X_train, y_train = generate_parity_data(n_train, n_bits)
    X_test, y_test = generate_parity_data(n_test, n_bits)

    # Create model
    print("Creating PureResonantTransformer...")
    model = PureResonantTransformer(
        input_dim=n_bits,
        hidden_dim=16,
        output_dim=2,
        num_layers=2
    )
    print(f"  Model: {model}")

    # Train
    print(f"\nTraining with RES for {generations} generations...")
    optimizer = SimpleRESOptimizer(model, pop_size=30)
    start_time = time.time()
    train_acc = optimizer.train(X_train, y_train, generations=generations)
    train_time = time.time() - start_time

    # Test
    print("\nEvaluating on test set...")
    preds = []
    for x_sample in X_test:
        x_tensor = ResonantTensor(x_sample, [1, len(x_sample)])
        out = model.forward(x_tensor)
        out_floats = out.to_floats()
        pred = 1 if out_floats[1] > out_floats[0] else 0
        preds.append(pred)

    test_acc = accuracy(preds, y_test)

    # Results
    print("\n" + "=" * 60)
    print("BENCHMARK RESULTS")
    print("-" * 60)
    print(f"Task:           {n_bits}-bit Parity (XOR)")
    print(f"Architecture:   PureResonantTransformer (Golden Cone Attention)")
    print(f"Train Samples:  {n_train}")
    print(f"Test Samples:   {n_test}")
    print(f"Generations:    {generations}")
    print("-" * 60)
    print(f"Train Accuracy: {train_acc:.2%}")
    print(f"Test Accuracy:  {test_acc:.2%}")
    print(f"Training Time:  {train_time:.2f}s")
    print("=" * 60)

    return {
        "train_acc": train_acc,
        "test_acc": test_acc,
        "train_time": train_time
    }


if __name__ == "__main__":
    benchmark_parity()
