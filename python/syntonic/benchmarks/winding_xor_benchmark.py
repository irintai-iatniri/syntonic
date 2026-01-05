"""
WindingNet XOR Benchmark - Compare WindingNet vs PyTorch MLP on XOR classification.

This benchmark replicates the convergence_benchmark.py test but with WindingNet,
allowing direct comparison with RES and PyTorch baselines.
"""

from __future__ import annotations
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import List, Tuple, Dict
import time

from syntonic.nn.winding import WindingNet
from syntonic.srt.geometry.winding import WindingState, winding_state


def make_xor_dataset(n_samples: int = 500, noise: float = 0.1, seed: int = 42) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate XOR dataset with noise.

    Args:
        n_samples: Number of samples
        noise: Gaussian noise level
        seed: Random seed

    Returns:
        X: Features (n_samples, 2)
        y: Labels (n_samples,)
    """
    rng = np.random.RandomState(seed)

    # Generate balanced dataset
    n_per_class = n_samples // 4
    X = []
    y = []

    # Class 0: (0,0) and (1,1)
    X.append(rng.randn(n_per_class, 2) * noise + np.array([0, 0]))
    y.extend([0] * n_per_class)

    X.append(rng.randn(n_per_class, 2) * noise + np.array([1, 1]))
    y.extend([0] * n_per_class)

    # Class 1: (0,1) and (1,0)
    X.append(rng.randn(n_per_class, 2) * noise + np.array([0, 1]))
    y.extend([1] * n_per_class)

    X.append(rng.randn(n_per_class, 2) * noise + np.array([1, 0]))
    y.extend([1] * n_per_class)

    X = np.vstack(X)
    y = np.array(y)

    # Shuffle
    perm = rng.permutation(len(y))
    return X[perm], y[perm]


def features_to_windings(X: np.ndarray) -> List[WindingState]:
    """
    Map XOR features to winding states.

    Simple mapping: discretize continuous features to integer winding numbers.
    Features in [0, 1] range map to {0, 1, 2}.

    Args:
        X: Features (n_samples, 2)

    Returns:
        List of WindingState objects
    """
    windings = []
    for x in X:
        # Map [0, 1] → {0, 1, 2} with rounding
        # This keeps winding norms small
        n1 = int(np.clip(np.round(x[0] * 2), 0, 2))
        n2 = int(np.clip(np.round(x[1] * 2), 0, 2))
        # Use n7, n8 for the two features, keep n9, n10 as 0
        windings.append(winding_state(n1, n2, 0, 0))
    return windings


class PyTorchMLP(nn.Module):
    """Baseline PyTorch MLP for comparison."""

    def __init__(self, input_dim: int = 2, hidden_dim: int = 16, output_dim: int = 2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def train_windingnet(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    max_winding: int = 3,
    base_dim: int = 64,
    num_blocks: int = 3,
    num_epochs: int = 100,
    lr: float = 0.01,
    verbose: bool = False,
) -> Dict:
    """Train WindingNet on XOR."""

    # Convert features to windings
    train_windings = features_to_windings(X_train)
    test_windings = features_to_windings(X_test)

    y_train_t = torch.tensor(y_train, dtype=torch.long)
    y_test_t = torch.tensor(y_test, dtype=torch.long)

    # Create model
    model = WindingNet(
        max_winding=max_winding,
        base_dim=base_dim,
        num_blocks=num_blocks,
        output_dim=2,
    )

    optimizer = optim.Adam(model.parameters(), lr=lr)

    history = {"train_acc": [], "test_acc": [], "loss": [], "syntony": []}

    start = time.time()

    for epoch in range(num_epochs):
        # Training
        model.train()
        optimizer.zero_grad()

        y_pred = model(train_windings)
        total_loss, task_loss, syntony_loss = model.compute_loss(y_pred, y_train_t)

        total_loss.backward()
        optimizer.step()

        # Evaluation
        model.eval()
        with torch.no_grad():
            # Train accuracy
            train_pred = y_pred.argmax(dim=1)
            train_acc = (train_pred == y_train_t).float().mean().item()

            # Test accuracy
            y_test_pred = model(test_windings)
            test_pred = y_test_pred.argmax(dim=1)
            test_acc = (test_pred == y_test_t).float().mean().item()

        stats = model.get_blockchain_stats()

        history["train_acc"].append(train_acc)
        history["test_acc"].append(test_acc)
        history["loss"].append(total_loss.item())
        history["syntony"].append(stats["network_syntony"])

        if verbose and (epoch % 20 == 0 or epoch == num_epochs - 1):
            print(
                f"Epoch {epoch:3d}: "
                f"Loss={total_loss.item():.4f}, "
                f"Train={train_acc:.2%}, "
                f"Test={test_acc:.2%}, "
                f"Syntony={stats['network_syntony']:.4f}"
            )

    elapsed = time.time() - start

    # Crystallize weights to Q(φ) lattice
    if verbose:
        print(f"\n{'='*70}")
        print("Crystallizing weights to Q(φ) lattice...")
        print(f"{'='*70}")

    try:
        model.crystallize_weights(precision=100)
        crystallized = True

        # Evaluate with exact DHSR
        if verbose:
            print("\nEvaluating with exact ResonantTensor...")

        model.eval()
        with torch.no_grad():
            y_exact = model.forward_exact(test_windings)
            test_pred_exact = y_exact.argmax(dim=1)
            exact_test_acc = (test_pred_exact == y_test_t).float().mean().item()

        if verbose:
            print(f"Float accuracy:  {history['test_acc'][-1]:.2%}")
            print(f"Exact accuracy:  {exact_test_acc:.2%}")

    except Exception as e:
        if verbose:
            print(f"Warning: Could not use exact mode - {e}")
        crystallized = False
        exact_test_acc = None

    return {
        "history": history,
        "final_train_acc": history["train_acc"][-1],
        "final_test_acc": history["test_acc"][-1],
        "exact_test_acc": exact_test_acc,
        "final_syntony": history["syntony"][-1],
        "time": elapsed,
        "crystallized": crystallized,
        "model": model,
    }


def train_pytorch_mlp(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    hidden_dim: int = 16,
    num_epochs: int = 100,
    lr: float = 0.01,
    verbose: bool = False,
) -> Dict:
    """Train PyTorch MLP baseline."""

    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.long)
    X_test_t = torch.tensor(X_test, dtype=torch.float32)
    y_test_t = torch.tensor(y_test, dtype=torch.long)

    model = PyTorchMLP(input_dim=2, hidden_dim=hidden_dim, output_dim=2)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    history = {"train_acc": [], "test_acc": [], "loss": []}

    start = time.time()

    for epoch in range(num_epochs):
        # Training
        model.train()
        optimizer.zero_grad()

        y_pred = model(X_train_t)
        loss = criterion(y_pred, y_train_t)

        loss.backward()
        optimizer.step()

        # Evaluation
        model.eval()
        with torch.no_grad():
            # Train accuracy
            train_pred = y_pred.argmax(dim=1)
            train_acc = (train_pred == y_train_t).float().mean().item()

            # Test accuracy
            y_test_pred = model(X_test_t)
            test_pred = y_test_pred.argmax(dim=1)
            test_acc = (test_pred == y_test_t).float().mean().item()

        history["train_acc"].append(train_acc)
        history["test_acc"].append(test_acc)
        history["loss"].append(loss.item())

        if verbose and (epoch % 20 == 0 or epoch == num_epochs - 1):
            print(
                f"Epoch {epoch:3d}: "
                f"Loss={loss.item():.4f}, "
                f"Train={train_acc:.2%}, "
                f"Test={test_acc:.2%}"
            )

    elapsed = time.time() - start

    return {
        "history": history,
        "final_train_acc": history["train_acc"][-1],
        "final_test_acc": history["test_acc"][-1],
        "time": elapsed,
        "model": model,
    }


def run_xor_benchmark(
    n_samples: int = 500,
    noise: float = 0.1,
    test_split: float = 0.2,
    num_epochs: int = 100,
    seed: int = 42,
    verbose: bool = True,
) -> Dict:
    """
    Run complete XOR benchmark comparing WindingNet vs PyTorch MLP.

    Args:
        n_samples: Number of samples
        noise: Noise level
        test_split: Test set fraction
        num_epochs: Training epochs
        seed: Random seed
        verbose: Print progress

    Returns:
        Results dictionary
    """
    if verbose:
        print("=" * 70)
        print("WindingNet XOR Benchmark")
        print("=" * 70)

    # Generate dataset
    X, y = make_xor_dataset(n_samples=n_samples, noise=noise, seed=seed)

    # Train/test split
    n_train = int(n_samples * (1 - test_split))
    X_train, X_test = X[:n_train], X[n_train:]
    y_train, y_test = y[:n_train], y[n_train:]

    if verbose:
        print(f"\nDataset: {n_samples} samples, noise={noise}")
        print(f"Train: {len(y_train)}, Test: {len(y_test)}")

    # Train WindingNet
    if verbose:
        print(f"\n{'='*70}")
        print("Training WindingNet...")
        print(f"{'='*70}")

    winding_results = train_windingnet(
        X_train, y_train, X_test, y_test,
        max_winding=3,
        base_dim=64,
        num_blocks=3,
        num_epochs=num_epochs,
        lr=0.01,
        verbose=verbose,
    )

    if verbose:
        print(f"\nWindingNet: {winding_results['final_test_acc']:.2%} accuracy in {winding_results['time']:.2f}s")

    # Train PyTorch MLP
    if verbose:
        print(f"\n{'='*70}")
        print("Training PyTorch MLP...")
        print(f"{'='*70}")

    pytorch_results = train_pytorch_mlp(
        X_train, y_train, X_test, y_test,
        hidden_dim=16,
        num_epochs=num_epochs,
        lr=0.01,
        verbose=verbose,
    )

    if verbose:
        print(f"\nPyTorch MLP: {pytorch_results['final_test_acc']:.2%} accuracy in {pytorch_results['time']:.2f}s")

    # Summary
    if verbose:
        print(f"\n{'='*70}")
        print("Summary")
        print(f"{'='*70}")
        print(f"{'Model':<20} {'Test Acc':<12} {'Syntony':<12} {'Time':<10}")
        print(f"{'-'*70}")

        # WindingNet float accuracy
        float_acc_str = f"{winding_results['final_test_acc']:>10.2%}"
        print(
            f"{'WindingNet (float)':<20} "
            f"{float_acc_str}  "
            f"{winding_results['final_syntony']:>10.4f}  "
            f"{winding_results['time']:>8.2f}s"
        )

        # WindingNet exact accuracy (if available)
        if winding_results.get('crystallized', False) and winding_results.get('exact_test_acc') is not None:
            exact_acc_str = f"{winding_results['exact_test_acc']:>10.2%}"
            print(
                f"{'WindingNet (exact)':<20} "
                f"{exact_acc_str}  "
                f"{'Q(φ) lattice':>10}  "
                f"{'crystallized':>8}"
            )

        # PyTorch MLP baseline
        print(
            f"{'PyTorch MLP':<20} "
            f"{pytorch_results['final_test_acc']:>10.2%}  "
            f"{'N/A':>10}  "
            f"{pytorch_results['time']:>8.2f}s"
        )
        print(f"{'='*70}")

    return {
        "windingnet": winding_results,
        "pytorch": pytorch_results,
        "dataset": {"X_train": X_train, "y_train": y_train, "X_test": X_test, "y_test": y_test},
    }


if __name__ == "__main__":
    results = run_xor_benchmark(
        n_samples=500,
        noise=0.1,
        test_split=0.2,
        num_epochs=100,
        seed=42,
        verbose=True,
    )

    # Plot comparison if matplotlib available
    try:
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(2, 2, figsize=(12, 8))

        # Test accuracy
        axes[0, 0].plot(results["windingnet"]["history"]["test_acc"], label="WindingNet")
        axes[0, 0].plot(results["pytorch"]["history"]["test_acc"], label="PyTorch MLP")
        axes[0, 0].set_title("Test Accuracy")
        axes[0, 0].set_xlabel("Epoch")
        axes[0, 0].set_ylabel("Accuracy")
        axes[0, 0].legend()
        axes[0, 0].grid(True)

        # Loss
        axes[0, 1].plot(results["windingnet"]["history"]["loss"], label="WindingNet")
        axes[0, 1].plot(results["pytorch"]["history"]["loss"], label="PyTorch MLP")
        axes[0, 1].set_title("Loss")
        axes[0, 1].set_xlabel("Epoch")
        axes[0, 1].set_ylabel("Loss")
        axes[0, 1].legend()
        axes[0, 1].grid(True)

        # Syntony (WindingNet only)
        axes[1, 0].plot(results["windingnet"]["history"]["syntony"])
        axes[1, 0].set_title("WindingNet Syntony")
        axes[1, 0].set_xlabel("Epoch")
        axes[1, 0].set_ylabel("Syntony")
        axes[1, 0].grid(True)

        # Train vs Test (WindingNet)
        axes[1, 1].plot(results["windingnet"]["history"]["train_acc"], label="Train", alpha=0.7)
        axes[1, 1].plot(results["windingnet"]["history"]["test_acc"], label="Test")
        axes[1, 1].set_title("WindingNet Train vs Test")
        axes[1, 1].set_xlabel("Epoch")
        axes[1, 1].set_ylabel("Accuracy")
        axes[1, 1].legend()
        axes[1, 1].grid(True)

        plt.tight_layout()
        plt.savefig("winding_xor_benchmark.png", dpi=150)
        print("\nPlot saved to winding_xor_benchmark.png")

    except ImportError:
        print("\nMatplotlib not available, skipping plots")
