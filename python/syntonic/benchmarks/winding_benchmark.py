"""
Winding Network Benchmark - Particle classification with WindingNet.

This benchmark trains WindingNet on particle type classification:
- Leptons (electron, muon, tau) → class 0
- Quarks (up, down, charm, strange, top, bottom) → class 1

Demonstrates that WindingNet can learn from winding state structure.
"""

from __future__ import annotations
import torch
import torch.nn as nn
import torch.optim as optim
from typing import List, Tuple, Dict
import time

from syntonic.nn.winding import WindingNet
from syntonic.physics.fermions.windings import (
    ELECTRON_WINDING,
    MUON_WINDING,
    TAU_WINDING,
    UP_WINDING,
    DOWN_WINDING,
    CHARM_WINDING,
    STRANGE_WINDING,
    TOP_WINDING,
    BOTTOM_WINDING,
    WindingState,
)


class WindingDataset:
    """
    Simple dataset for particle type classification.

    Leptons → 0
    Quarks → 1
    """

    def __init__(self):
        self.data = [
            # Leptons
            (ELECTRON_WINDING, 0),
            (MUON_WINDING, 0),
            (TAU_WINDING, 0),
            # Up-type quarks
            (UP_WINDING, 1),
            (CHARM_WINDING, 1),
            (TOP_WINDING, 1),
            # Down-type quarks
            (DOWN_WINDING, 1),
            (STRANGE_WINDING, 1),
            (BOTTOM_WINDING, 1),
        ]

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Tuple[WindingState, int]:
        return self.data[idx]

    def get_batch(self) -> Tuple[List[WindingState], torch.Tensor]:
        """Get all data as a batch."""
        windings, labels = zip(*self.data)
        return list(windings), torch.tensor(labels, dtype=torch.long)


def train_winding_net(
    model: WindingNet,
    dataset: WindingDataset,
    num_epochs: int = 100,
    learning_rate: float = 0.01,
    verbose: bool = True,
) -> Dict:
    """
    Train WindingNet on particle classification.

    Args:
        model: WindingNet model
        dataset: Training dataset
        num_epochs: Number of training epochs
        learning_rate: Learning rate
        verbose: Print progress

    Returns:
        Training history dictionary
    """
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    history = {
        "loss": [],
        "task_loss": [],
        "syntony_loss": [],
        "accuracy": [],
        "syntony": [],
        "validation_rate": [],
    }

    start_time = time.time()

    for epoch in range(num_epochs):
        # Get batch
        windings, labels = dataset.get_batch()

        # Forward pass
        optimizer.zero_grad()
        y_pred = model(windings)

        # Compute loss
        total_loss, task_loss, syntony_loss = model.compute_loss(y_pred, labels)

        # Backward pass
        total_loss.backward()
        optimizer.step()

        # Compute accuracy
        predictions = y_pred.argmax(dim=1)
        accuracy = (predictions == labels).float().mean().item()

        # Get blockchain stats
        stats = model.get_blockchain_stats()

        # Record history
        history["loss"].append(total_loss.item())
        history["task_loss"].append(task_loss.item())
        history["syntony_loss"].append(syntony_loss.item())
        history["accuracy"].append(accuracy)
        history["syntony"].append(stats["network_syntony"])
        history["validation_rate"].append(stats["validation_rate"])

        # Print progress
        if verbose and (epoch % 10 == 0 or epoch == num_epochs - 1):
            print(
                f"Epoch {epoch:3d}: "
                f"Loss={total_loss.item():.4f}, "
                f"Acc={accuracy:.2%}, "
                f"Syntony={stats['network_syntony']:.4f}, "
                f"ValRate={stats['validation_rate']:.2%}"
            )

    elapsed = time.time() - start_time
    history["time"] = elapsed

    if verbose:
        print(f"\nTraining complete in {elapsed:.2f}s")

    return history


def run_benchmark(
    max_winding: int = 5,
    base_dim: int = 64,
    num_blocks: int = 3,
    num_epochs: int = 100,
    learning_rate: float = 0.01,
    verbose: bool = True,
) -> Dict:
    """
    Run complete WindingNet benchmark.

    Args:
        max_winding: Maximum winding number
        base_dim: Base embedding dimension
        num_blocks: Number of DHSR blocks
        num_epochs: Training epochs
        learning_rate: Learning rate
        verbose: Print progress

    Returns:
        Results dictionary with history and final metrics
    """
    if verbose:
        print("=" * 60)
        print("WindingNet Benchmark: Particle Classification")
        print("=" * 60)

    # Create dataset
    dataset = WindingDataset()
    if verbose:
        print(f"\nDataset: {len(dataset)} particles (3 leptons, 6 quarks)")

    # Create model
    model = WindingNet(
        max_winding=max_winding,
        base_dim=base_dim,
        num_blocks=num_blocks,
        output_dim=2,
        use_prime_filter=True,
        consensus_threshold=0.024,
    )

    if verbose:
        num_params = sum(p.numel() for p in model.parameters())
        print(f"Model: WindingNet({base_dim=}, {num_blocks=})")
        print(f"Parameters: {num_params:,}")

    # Train
    if verbose:
        print(f"\nTraining for {num_epochs} epochs (lr={learning_rate})...")
        print()

    history = train_winding_net(
        model=model,
        dataset=dataset,
        num_epochs=num_epochs,
        learning_rate=learning_rate,
        verbose=verbose,
    )

    # Final evaluation
    windings, labels = dataset.get_batch()
    with torch.no_grad():
        y_pred = model(windings)
        predictions = y_pred.argmax(dim=1)
        accuracy = (predictions == labels).float().mean().item()

    stats = model.get_blockchain_stats()

    results = {
        "history": history,
        "final_accuracy": accuracy,
        "final_syntony": stats["network_syntony"],
        "validation_rate": stats["validation_rate"],
        "blockchain_length": stats["blockchain_length"],
        "model": model,
    }

    if verbose:
        print(f"\n{'='*60}")
        print("Final Results:")
        print(f"  Accuracy: {accuracy:.2%}")
        print(f"  Network syntony: {stats['network_syntony']:.4f}")
        print(f"  Validation rate: {stats['validation_rate']:.2%}")
        print(f"  Blockchain length: {stats['blockchain_length']}")
        print(f"  Training time: {history['time']:.2f}s")
        print(f"{'='*60}")

    return results


if __name__ == "__main__":
    # Run benchmark
    results = run_benchmark(
        max_winding=5,
        base_dim=64,
        num_blocks=3,
        num_epochs=100,
        learning_rate=0.01,
        verbose=True,
    )

    # Plot history if matplotlib available
    try:
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(2, 2, figsize=(12, 8))

        # Accuracy
        axes[0, 0].plot(results["history"]["accuracy"])
        axes[0, 0].set_title("Accuracy")
        axes[0, 0].set_xlabel("Epoch")
        axes[0, 0].set_ylabel("Accuracy")
        axes[0, 0].grid(True)

        # Loss
        axes[0, 1].plot(results["history"]["loss"], label="Total")
        axes[0, 1].plot(results["history"]["task_loss"], label="Task", alpha=0.7)
        axes[0, 1].set_title("Loss")
        axes[0, 1].set_xlabel("Epoch")
        axes[0, 1].set_ylabel("Loss")
        axes[0, 1].legend()
        axes[0, 1].grid(True)

        # Syntony
        axes[1, 0].plot(results["history"]["syntony"])
        axes[1, 0].set_title("Network Syntony")
        axes[1, 0].set_xlabel("Epoch")
        axes[1, 0].set_ylabel("Syntony")
        axes[1, 0].grid(True)

        # Validation rate
        axes[1, 1].plot(results["history"]["validation_rate"])
        axes[1, 1].set_title("Blockchain Validation Rate")
        axes[1, 1].set_xlabel("Epoch")
        axes[1, 1].set_ylabel("Validation Rate")
        axes[1, 1].grid(True)

        plt.tight_layout()
        plt.savefig("winding_net_benchmark.png", dpi=150)
        print("\nPlot saved to winding_net_benchmark.png")

    except ImportError:
        print("\nMatplotlib not available, skipping plots")
