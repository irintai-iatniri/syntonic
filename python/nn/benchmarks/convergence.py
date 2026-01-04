"""
Convergence Analysis for Syntonic Networks.

Tools for analyzing and comparing convergence rates
of syntonic vs standard networks.

Expected: ~35% faster convergence for high-S networks.

Source: CRT.md §12.2
"""

from __future__ import annotations
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import math

PHI = (1 + math.sqrt(5)) / 2


@dataclass
class ConvergenceMetrics:
    """
    Metrics for convergence analysis.

    Attributes:
        epochs_to_threshold: Epochs to reach each threshold
        convergence_rate: Rate of loss decrease
        stability: Training stability score
        final_loss: Final loss value
        syntony_correlation: Correlation between syntony and convergence
    """

    epochs_to_threshold: Dict[float, int] = field(default_factory=dict)
    convergence_rate: float = 0.0
    stability: float = 0.0
    final_loss: float = 0.0
    syntony_correlation: Optional[float] = None

    def __str__(self) -> str:
        s = "Convergence Metrics:\n"
        s += f"  Final Loss: {self.final_loss:.6f}\n"
        s += f"  Convergence Rate: {self.convergence_rate:.4f}\n"
        s += f"  Stability: {self.stability:.4f}\n"

        if self.epochs_to_threshold:
            s += "  Epochs to threshold:\n"
            for threshold, epochs in sorted(self.epochs_to_threshold.items(), reverse=True):
                s += f"    {threshold:.1%} accuracy: {epochs} epochs\n"

        if self.syntony_correlation is not None:
            s += f"  Syntony-Convergence Correlation: {self.syntony_correlation:.4f}\n"

        return s


class ConvergenceAnalyzer:
    """
    Analyze convergence behavior of models.

    Example:
        >>> analyzer = ConvergenceAnalyzer()
        >>> metrics = analyzer.analyze(model, train_loader, test_loader)
        >>> print(metrics)
    """

    def __init__(
        self,
        accuracy_thresholds: List[float] = [0.90, 0.95, 0.99],
        loss_thresholds: List[float] = [0.1, 0.05, 0.01],
    ):
        """
        Initialize analyzer.

        Args:
            accuracy_thresholds: Accuracy levels to track
            loss_thresholds: Loss levels to track
        """
        self.accuracy_thresholds = accuracy_thresholds
        self.loss_thresholds = loss_thresholds

    def analyze(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        test_loader: DataLoader,
        epochs: int = 100,
        lr: float = 0.001,
        device: str = 'cpu',
    ) -> ConvergenceMetrics:
        """
        Analyze convergence of a model.

        Args:
            model: Model to analyze
            train_loader: Training data
            test_loader: Test data
            epochs: Maximum epochs
            lr: Learning rate
            device: Compute device

        Returns:
            ConvergenceMetrics
        """
        model = model.to(device)

        # Check if syntonic
        is_syntonic = hasattr(model, 'syntony')
        if is_syntonic:
            from syntonic.nn.optim import SyntonicAdam
            optimizer = SyntonicAdam(model.parameters(), lr=lr)
        else:
            optimizer = torch.optim.Adam(model.parameters(), lr=lr)

        criterion = nn.CrossEntropyLoss()

        # Tracking
        loss_history = []
        accuracy_history = []
        syntony_history = []
        epochs_to_accuracy = {}
        epochs_to_loss = {}

        for epoch in range(epochs):
            # Train
            model.train()
            epoch_loss = 0.0
            n_batches = 0

            for inputs, targets in train_loader:
                inputs = inputs.to(device)
                targets = targets.to(device)

                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                loss.backward()

                if is_syntonic:
                    optimizer.step(syntony=model.syntony)
                else:
                    optimizer.step()

                epoch_loss += loss.item()
                n_batches += 1

            epoch_loss /= n_batches
            loss_history.append(epoch_loss)

            # Evaluate
            accuracy = self._evaluate(model, test_loader, device)
            accuracy_history.append(accuracy)

            if is_syntonic:
                syntony_history.append(model.syntony)

            # Check thresholds
            for threshold in self.accuracy_thresholds:
                if threshold not in epochs_to_accuracy and accuracy >= threshold:
                    epochs_to_accuracy[threshold] = epoch + 1

            for threshold in self.loss_thresholds:
                if threshold not in epochs_to_loss and epoch_loss <= threshold:
                    epochs_to_loss[threshold] = epoch + 1

        # Compute metrics
        convergence_rate = self._compute_convergence_rate(loss_history)
        stability = self._compute_stability(loss_history)

        syntony_correlation = None
        if syntony_history:
            # Correlation between syntony and loss decrease
            loss_decrease = [loss_history[i] - loss_history[i+1]
                           for i in range(len(loss_history)-1)]
            if len(loss_decrease) == len(syntony_history) - 1:
                syntony_correlation = self._compute_correlation(
                    syntony_history[:-1], loss_decrease
                )

        return ConvergenceMetrics(
            epochs_to_threshold=epochs_to_accuracy,
            convergence_rate=convergence_rate,
            stability=stability,
            final_loss=loss_history[-1],
            syntony_correlation=syntony_correlation,
        )

    def _evaluate(
        self,
        model: nn.Module,
        test_loader: DataLoader,
        device: str,
    ) -> float:
        """Evaluate model accuracy."""
        model.eval()
        correct = 0
        total = 0

        with torch.no_grad():
            for inputs, targets in test_loader:
                inputs = inputs.to(device)
                targets = targets.to(device)

                outputs = model(inputs)
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

        return correct / total

    def _compute_convergence_rate(self, loss_history: List[float]) -> float:
        """
        Compute convergence rate.

        Higher is faster convergence.
        """
        if len(loss_history) < 2:
            return 0.0

        # Fit exponential decay: L(t) = L0 * exp(-r*t)
        # ln(L(t)) = ln(L0) - r*t
        # Linear regression on log losses

        import math
        log_losses = [math.log(max(l, 1e-10)) for l in loss_history]

        n = len(log_losses)
        t = list(range(n))

        # Linear regression
        mean_t = sum(t) / n
        mean_log = sum(log_losses) / n

        numerator = sum((t[i] - mean_t) * (log_losses[i] - mean_log) for i in range(n))
        denominator = sum((t[i] - mean_t) ** 2 for i in range(n))

        if denominator == 0:
            return 0.0

        slope = numerator / denominator

        # Convergence rate is negative of slope (positive = converging)
        return -slope

    def _compute_stability(self, loss_history: List[float]) -> float:
        """
        Compute training stability.

        1.0 = perfectly stable, 0.0 = very unstable
        """
        if len(loss_history) < 10:
            return 1.0

        # Count how many times loss increases
        increases = sum(1 for i in range(1, len(loss_history))
                       if loss_history[i] > loss_history[i-1])

        stability = 1.0 - increases / (len(loss_history) - 1)
        return max(0.0, stability)

    def _compute_correlation(self, x: List[float], y: List[float]) -> float:
        """Compute Pearson correlation."""
        n = min(len(x), len(y))
        if n < 2:
            return 0.0

        mean_x = sum(x[:n]) / n
        mean_y = sum(y[:n]) / n

        numerator = sum((x[i] - mean_x) * (y[i] - mean_y) for i in range(n))
        denom_x = math.sqrt(sum((x[i] - mean_x) ** 2 for i in range(n)))
        denom_y = math.sqrt(sum((y[i] - mean_y) ** 2 for i in range(n)))

        if denom_x == 0 or denom_y == 0:
            return 0.0

        return numerator / (denom_x * denom_y)


def compare_convergence(
    syntonic_model: nn.Module,
    baseline_model: nn.Module,
    train_loader: DataLoader,
    test_loader: DataLoader,
    epochs: int = 100,
    device: str = 'cpu',
) -> Dict[str, ConvergenceMetrics]:
    """
    Compare convergence of syntonic vs baseline models.

    Args:
        syntonic_model: Syntonic model
        baseline_model: Baseline model
        train_loader: Training data
        test_loader: Test data
        epochs: Maximum epochs
        device: Compute device

    Returns:
        Dict with 'syntonic' and 'baseline' metrics
    """
    analyzer = ConvergenceAnalyzer()

    print("Analyzing baseline model...")
    baseline_metrics = analyzer.analyze(
        baseline_model, train_loader, test_loader,
        epochs=epochs, device=device,
    )

    print("Analyzing syntonic model...")
    syntonic_metrics = analyzer.analyze(
        syntonic_model, train_loader, test_loader,
        epochs=epochs, device=device,
    )

    # Print comparison
    print("\n" + "=" * 50)
    print("CONVERGENCE COMPARISON")
    print("=" * 50)

    print("\nBaseline:")
    print(baseline_metrics)

    print("\nSyntonic:")
    print(syntonic_metrics)

    # Compute speedup
    for threshold in [0.90, 0.95]:
        baseline_epochs = baseline_metrics.epochs_to_threshold.get(threshold)
        syntonic_epochs = syntonic_metrics.epochs_to_threshold.get(threshold)

        if baseline_epochs and syntonic_epochs:
            speedup = (baseline_epochs - syntonic_epochs) / baseline_epochs * 100
            print(f"\nSpeedup to {threshold:.0%}: {speedup:.1f}%")

    return {
        'syntonic': syntonic_metrics,
        'baseline': baseline_metrics,
    }
