"""
Standard Benchmarks for Syntonic Networks.

Compare syntonic networks against standard baselines
on common benchmarks (MNIST, CIFAR, etc.).

Source: CRT.md §12.2
"""

from __future__ import annotations
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Callable
import time
import math

PHI = (1 + math.sqrt(5)) / 2


@dataclass
class BenchmarkResult:
    """
    Result of a benchmark run.

    Attributes:
        name: Benchmark name
        model_type: Type of model tested
        accuracy: Final test accuracy
        train_loss: Final training loss
        test_loss: Final test loss
        syntony: Final model syntony (if applicable)
        epochs_to_95: Epochs to reach 95% accuracy
        total_time: Total training time
        peak_memory: Peak GPU memory usage
        metrics_history: Full metrics history
    """

    name: str
    model_type: str
    accuracy: float
    train_loss: float
    test_loss: float
    syntony: Optional[float] = None
    epochs_to_95: Optional[int] = None
    total_time: float = 0.0
    peak_memory: float = 0.0
    metrics_history: Dict[str, List[float]] = field(default_factory=dict)

    def __str__(self) -> str:
        s = f"Benchmark: {self.name} ({self.model_type})\n"
        s += f"  Accuracy: {self.accuracy:.2%}\n"
        s += f"  Train Loss: {self.train_loss:.4f}\n"
        s += f"  Test Loss: {self.test_loss:.4f}\n"
        if self.syntony is not None:
            s += f"  Syntony: {self.syntony:.4f}\n"
        if self.epochs_to_95 is not None:
            s += f"  Epochs to 95%: {self.epochs_to_95}\n"
        s += f"  Time: {self.total_time:.1f}s\n"
        return s


class BenchmarkSuite:
    """
    Suite for running standardized benchmarks.

    Example:
        >>> suite = BenchmarkSuite()
        >>> results = suite.run_all(syntonic_model, baseline_model)
        >>> suite.print_comparison(results)
    """

    def __init__(
        self,
        device: str = 'cpu',
        default_epochs: int = 50,
        default_lr: float = 0.001,
        seed: int = 42,
    ):
        """
        Initialize benchmark suite.

        Args:
            device: Compute device
            default_epochs: Default training epochs
            default_lr: Default learning rate
            seed: Random seed
        """
        self.device = device
        self.default_epochs = default_epochs
        self.default_lr = default_lr
        self.seed = seed

        # Set seed
        torch.manual_seed(seed)

    def run_benchmark(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        test_loader: DataLoader,
        name: str = "Custom",
        epochs: Optional[int] = None,
        lr: Optional[float] = None,
        use_syntonic_optimizer: bool = False,
    ) -> BenchmarkResult:
        """
        Run a single benchmark.

        Args:
            model: Model to benchmark
            train_loader: Training data loader
            test_loader: Test data loader
            name: Benchmark name
            epochs: Training epochs
            lr: Learning rate
            use_syntonic_optimizer: Use SyntonicAdam

        Returns:
            BenchmarkResult
        """
        epochs = epochs or self.default_epochs
        lr = lr or self.default_lr

        model = model.to(self.device)

        # Setup optimizer
        if use_syntonic_optimizer:
            from syntonic.nn.optim import SyntonicAdam
            optimizer = SyntonicAdam(model.parameters(), lr=lr)
        else:
            optimizer = torch.optim.Adam(model.parameters(), lr=lr)

        criterion = nn.CrossEntropyLoss()

        # Training history
        history = {
            'train_loss': [],
            'test_loss': [],
            'accuracy': [],
            'syntony': [],
        }

        epochs_to_95 = None
        start_time = time.time()

        for epoch in range(epochs):
            # Train
            model.train()
            train_loss = 0.0
            n_batches = 0

            for inputs, targets in train_loader:
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)

                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                loss.backward()

                # Syntonic optimizer step
                if use_syntonic_optimizer and hasattr(model, 'syntony'):
                    optimizer.step(syntony=model.syntony)
                else:
                    optimizer.step()

                train_loss += loss.item()
                n_batches += 1

            train_loss /= n_batches

            # Evaluate
            model.eval()
            test_loss = 0.0
            correct = 0
            total = 0

            with torch.no_grad():
                for inputs, targets in test_loader:
                    inputs = inputs.to(self.device)
                    targets = targets.to(self.device)

                    outputs = model(inputs)
                    loss = criterion(outputs, targets)

                    test_loss += loss.item()
                    _, predicted = outputs.max(1)
                    total += targets.size(0)
                    correct += predicted.eq(targets).sum().item()

            test_loss /= len(test_loader)
            accuracy = correct / total

            # Record history
            history['train_loss'].append(train_loss)
            history['test_loss'].append(test_loss)
            history['accuracy'].append(accuracy)

            if hasattr(model, 'syntony'):
                history['syntony'].append(model.syntony)

            # Check epochs to 95%
            if epochs_to_95 is None and accuracy >= 0.95:
                epochs_to_95 = epoch + 1

        total_time = time.time() - start_time

        # Get model type
        model_type = type(model).__name__
        if hasattr(model, 'syntony'):
            model_type += " (Syntonic)"

        return BenchmarkResult(
            name=name,
            model_type=model_type,
            accuracy=accuracy,
            train_loss=train_loss,
            test_loss=test_loss,
            syntony=model.syntony if hasattr(model, 'syntony') else None,
            epochs_to_95=epochs_to_95,
            total_time=total_time,
            metrics_history=history,
        )

    def run_comparison(
        self,
        syntonic_model: nn.Module,
        baseline_model: nn.Module,
        train_loader: DataLoader,
        test_loader: DataLoader,
        name: str = "Comparison",
        epochs: Optional[int] = None,
    ) -> Dict[str, BenchmarkResult]:
        """
        Run comparison between syntonic and baseline models.

        Args:
            syntonic_model: Syntonic network
            baseline_model: Baseline network
            train_loader: Training data
            test_loader: Test data
            name: Benchmark name
            epochs: Training epochs

        Returns:
            Dict with 'syntonic' and 'baseline' results
        """
        results = {}

        # Run baseline
        results['baseline'] = self.run_benchmark(
            baseline_model, train_loader, test_loader,
            name=f"{name} (Baseline)",
            epochs=epochs,
            use_syntonic_optimizer=False,
        )

        # Reset seed for fair comparison
        torch.manual_seed(self.seed)

        # Run syntonic
        results['syntonic'] = self.run_benchmark(
            syntonic_model, train_loader, test_loader,
            name=f"{name} (Syntonic)",
            epochs=epochs,
            use_syntonic_optimizer=True,
        )

        return results

    @staticmethod
    def print_comparison(results: Dict[str, BenchmarkResult]):
        """Print comparison table."""
        print("\n" + "=" * 60)
        print("BENCHMARK COMPARISON")
        print("=" * 60)

        baseline = results.get('baseline')
        syntonic = results.get('syntonic')

        if baseline and syntonic:
            print(f"\n{'Metric':<25} {'Baseline':<15} {'Syntonic':<15} {'Diff':<15}")
            print("-" * 70)

            # Accuracy
            acc_diff = syntonic.accuracy - baseline.accuracy
            print(f"{'Accuracy':<25} {baseline.accuracy:.2%:<15} {syntonic.accuracy:.2%:<15} {acc_diff:+.2%}")

            # Train loss
            loss_diff = syntonic.train_loss - baseline.train_loss
            print(f"{'Train Loss':<25} {baseline.train_loss:.4f:<15} {syntonic.train_loss:.4f:<15} {loss_diff:+.4f}")

            # Time
            time_diff = syntonic.total_time - baseline.total_time
            print(f"{'Time (s)':<25} {baseline.total_time:.1f:<15} {syntonic.total_time:.1f:<15} {time_diff:+.1f}")

            # Epochs to 95%
            if baseline.epochs_to_95 and syntonic.epochs_to_95:
                e_diff = syntonic.epochs_to_95 - baseline.epochs_to_95
                print(f"{'Epochs to 95%':<25} {baseline.epochs_to_95:<15} {syntonic.epochs_to_95:<15} {e_diff:+d}")

            # Syntony
            if syntonic.syntony:
                print(f"{'Final Syntony':<25} {'N/A':<15} {syntonic.syntony:.4f}")

            # Speedup
            if baseline.epochs_to_95 and syntonic.epochs_to_95:
                speedup = (baseline.epochs_to_95 - syntonic.epochs_to_95) / baseline.epochs_to_95 * 100
                print(f"\nConvergence speedup: {speedup:.1f}%")

        print("=" * 60)


def run_mnist_benchmark(
    syntonic_model: nn.Module,
    baseline_model: Optional[nn.Module] = None,
    epochs: int = 20,
    batch_size: int = 64,
    device: str = 'cpu',
) -> Dict[str, BenchmarkResult]:
    """
    Run MNIST benchmark.

    Args:
        syntonic_model: Syntonic model to test
        baseline_model: Optional baseline for comparison
        epochs: Training epochs
        batch_size: Batch size
        device: Compute device

    Returns:
        Benchmark results
    """
    try:
        from torchvision import datasets, transforms
    except ImportError:
        print("torchvision required for MNIST benchmark")
        return {}

    # Data loading
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST('./data', train=False, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    suite = BenchmarkSuite(device=device, default_epochs=epochs)

    if baseline_model is not None:
        return suite.run_comparison(
            syntonic_model, baseline_model,
            train_loader, test_loader,
            name="MNIST",
            epochs=epochs,
        )
    else:
        return {
            'syntonic': suite.run_benchmark(
                syntonic_model, train_loader, test_loader,
                name="MNIST",
                epochs=epochs,
                use_syntonic_optimizer=True,
            )
        }


def run_cifar_benchmark(
    syntonic_model: nn.Module,
    baseline_model: Optional[nn.Module] = None,
    epochs: int = 50,
    batch_size: int = 64,
    device: str = 'cpu',
) -> Dict[str, BenchmarkResult]:
    """
    Run CIFAR-10 benchmark.

    Args:
        syntonic_model: Syntonic model to test
        baseline_model: Optional baseline for comparison
        epochs: Training epochs
        batch_size: Batch size
        device: Compute device

    Returns:
        Benchmark results
    """
    try:
        from torchvision import datasets, transforms
    except ImportError:
        print("torchvision required for CIFAR benchmark")
        return {}

    # Data loading with augmentation
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    train_dataset = datasets.CIFAR10('./data', train=True, download=True, transform=transform_train)
    test_dataset = datasets.CIFAR10('./data', train=False, transform=transform_test)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    suite = BenchmarkSuite(device=device, default_epochs=epochs)

    if baseline_model is not None:
        return suite.run_comparison(
            syntonic_model, baseline_model,
            train_loader, test_loader,
            name="CIFAR-10",
            epochs=epochs,
        )
    else:
        return {
            'syntonic': suite.run_benchmark(
                syntonic_model, train_loader, test_loader,
                name="CIFAR-10",
                epochs=epochs,
                use_syntonic_optimizer=True,
            )
        }
