"""
Metrics tracking for syntonic softmax validation experiments.

Tracks three key metrics:
1. Convergence speed (generations to target accuracy)
2. Final accuracy/loss (best performance achieved)
3. Syntony correlation (relationship between syntony and accuracy)
"""

from typing import List, Dict, Optional, Tuple
import math


class MetricsTracker:
    """
    Tracks convergence, accuracy, and syntony metrics during training.

    Usage:
        tracker = MetricsTracker()
        for generation in range(max_generations):
            # ... training ...
            tracker.update(generation, accuracy, syntony_dict)

        # Get results
        conv_speed = tracker.compute_convergence_speed(target_accuracy=0.95)
        correlation = tracker.compute_syntony_correlation()
    """

    def __init__(self):
        """Initialize empty metrics tracker."""
        self.convergence_history = []  # List[(gen, accuracy)]
        self.syntony_history = []      # List[(gen, syntony_dict)]
        self.final_metrics = {}        # Final summary dict

        self.best_accuracy = 0.0
        self.best_generation = 0

    def update(
        self,
        generation: int,
        accuracy: float,
        syntony_dict: Dict[str, any],
        loss: Optional[float] = None
    ):
        """
        Update metrics for current generation.

        Args:
            generation: Current generation number
            accuracy: Accuracy on validation/test set
            syntony_dict: Dictionary with keys:
                - "network": Overall network syntony
                - "layers": List of per-layer syntonies
                - "softmax": Softmax-specific syntony
            loss: Optional loss value
        """
        self.convergence_history.append((generation, accuracy))
        self.syntony_history.append((generation, syntony_dict))

        # Track best accuracy
        if accuracy > self.best_accuracy:
            self.best_accuracy = accuracy
            self.best_generation = generation

    def compute_convergence_speed(
        self,
        target_accuracy: float = 0.95
    ) -> Optional[int]:
        """
        Compute generation number when target accuracy was first reached.

        Args:
            target_accuracy: Target accuracy threshold (default 0.95)

        Returns:
            Generation number when target reached, or None if never reached
        """
        for gen, acc in self.convergence_history:
            if acc >= target_accuracy:
                return gen
        return None

    def compute_syntony_correlation(
        self,
        level: str = "network"
    ) -> float:
        """
        Compute Pearson correlation between syntony and accuracy.

        Args:
            level: Which syntony level to correlate:
                - "network": Overall network syntony
                - "softmax": Softmax-specific syntony
                - "layer_0", "layer_1", etc.: Specific layer

        Returns:
            Pearson correlation coefficient r ∈ [-1, 1]
        """
        if len(self.convergence_history) < 2:
            return 0.0

        # Extract accuracy and syntony values
        accuracies = [acc for _, acc in self.convergence_history]
        syntonies = []

        for _, syntony_dict in self.syntony_history:
            if level == "network":
                syntonies.append(syntony_dict["network"])
            elif level == "softmax":
                syntonies.append(syntony_dict["softmax"])
            elif level.startswith("layer_"):
                layer_idx = int(level.split("_")[1])
                layers = syntony_dict["layers"]
                if layer_idx < len(layers):
                    syntonies.append(layers[layer_idx])
                else:
                    syntonies.append(0.0)
            else:
                raise ValueError(f"Unknown syntony level: {level}")

        # Compute Pearson correlation
        return pearson_correlation(accuracies, syntonies)

    def get_summary(self) -> Dict[str, any]:
        """
        Get summary of all metrics.

        Returns:
            Dictionary with:
            - "convergence_speed_95": Generations to 95% accuracy (or None)
            - "convergence_speed_90": Generations to 90% accuracy (or None)
            - "best_accuracy": Best accuracy achieved
            - "best_generation": Generation with best accuracy
            - "final_accuracy": Accuracy at last generation
            - "syntony_correlation_network": Correlation (network level)
            - "syntony_correlation_softmax": Correlation (softmax level)
            - "total_generations": Total number of generations
        """
        conv_95 = self.compute_convergence_speed(0.95)
        conv_90 = self.compute_convergence_speed(0.90)
        corr_network = self.compute_syntony_correlation("network")
        corr_softmax = self.compute_syntony_correlation("softmax")

        final_acc = self.convergence_history[-1][1] if self.convergence_history else 0.0

        return {
            "convergence_speed_95": conv_95,
            "convergence_speed_90": conv_90,
            "best_accuracy": self.best_accuracy,
            "best_generation": self.best_generation,
            "final_accuracy": final_acc,
            "syntony_correlation_network": corr_network,
            "syntony_correlation_softmax": corr_softmax,
            "total_generations": len(self.convergence_history),
        }

    def get_convergence_curve(self) -> Tuple[List[int], List[float]]:
        """
        Get convergence curve (generation, accuracy) for plotting.

        Returns:
            (generations, accuracies) tuple
        """
        gens = [g for g, _ in self.convergence_history]
        accs = [a for _, a in self.convergence_history]
        return gens, accs

    def get_syntony_curve(
        self,
        level: str = "network"
    ) -> Tuple[List[int], List[float]]:
        """
        Get syntony curve (generation, syntony) for plotting.

        Args:
            level: Which syntony level to extract

        Returns:
            (generations, syntonies) tuple
        """
        gens = [g for g, _ in self.syntony_history]
        syntonies = []

        for _, syntony_dict in self.syntony_history:
            if level == "network":
                syntonies.append(syntony_dict["network"])
            elif level == "softmax":
                syntonies.append(syntony_dict["softmax"])
            elif level.startswith("layer_"):
                layer_idx = int(level.split("_")[1])
                layers = syntony_dict["layers"]
                if layer_idx < len(layers):
                    syntonies.append(layers[layer_idx])
                else:
                    syntonies.append(0.0)

        return gens, syntonies

    def get_syntony_vs_accuracy(
        self,
        level: str = "network"
    ) -> Tuple[List[float], List[float]]:
        """
        Get (syntony, accuracy) pairs for scatter plot.

        Args:
            level: Which syntony level to extract

        Returns:
            (syntonies, accuracies) tuple
        """
        accuracies = [acc for _, acc in self.convergence_history]
        syntonies = []

        for _, syntony_dict in self.syntony_history:
            if level == "network":
                syntonies.append(syntony_dict["network"])
            elif level == "softmax":
                syntonies.append(syntony_dict["softmax"])
            elif level.startswith("layer_"):
                layer_idx = int(level.split("_")[1])
                layers = syntony_dict["layers"]
                if layer_idx < len(layers):
                    syntonies.append(layers[layer_idx])
                else:
                    syntonies.append(0.0)

        return syntonies, accuracies


def pearson_correlation(x: List[float], y: List[float]) -> float:
    """
    Compute Pearson correlation coefficient between two lists.

    Args:
        x: First variable
        y: Second variable

    Returns:
        Correlation coefficient r ∈ [-1, 1]
    """
    if len(x) != len(y) or len(x) < 2:
        return 0.0

    n = len(x)
    mean_x = sum(x) / n
    mean_y = sum(y) / n

    # Compute covariance and standard deviations
    cov = sum((x[i] - mean_x) * (y[i] - mean_y) for i in range(n))
    std_x = math.sqrt(sum((x[i] - mean_x) ** 2 for i in range(n)))
    std_y = math.sqrt(sum((y[i] - mean_y) ** 2 for i in range(n)))

    if std_x == 0 or std_y == 0:
        return 0.0

    return cov / (std_x * std_y)


class ComparisonMetrics:
    """
    Compare metrics across multiple variants.

    Usage:
        comparison = ComparisonMetrics()
        comparison.add_variant("identity", tracker_identity)
        comparison.add_variant("syntonic", tracker_syntonic)
        comparison.print_comparison_table()
    """

    def __init__(self):
        """Initialize empty comparison."""
        self.variants = {}  # Dict[str, MetricsTracker]

    def add_variant(self, name: str, tracker: MetricsTracker):
        """
        Add a variant to comparison.

        Args:
            name: Variant name (e.g., "identity", "syntonic_e6")
            tracker: MetricsTracker for this variant
        """
        self.variants[name] = tracker

    def get_comparison_table(self) -> str:
        """
        Generate comparison table as formatted string.

        Returns:
            Multi-line string with comparison table
        """
        if not self.variants:
            return "No variants to compare"

        # Header
        lines = []
        lines.append("Variant              | Conv 95 | Conv 90 | Final Acc | Best Acc | Corr (Net) | Corr (Soft)")
        lines.append("--------------------|---------|---------|-----------|----------|------------|------------")

        # Rows
        for name, tracker in self.variants.items():
            summary = tracker.get_summary()
            conv_95 = summary["convergence_speed_95"]
            conv_90 = summary["convergence_speed_90"]
            final_acc = summary["final_accuracy"]
            best_acc = summary["best_accuracy"]
            corr_net = summary["syntony_correlation_network"]
            corr_soft = summary["syntony_correlation_softmax"]

            conv_95_str = str(conv_95) if conv_95 is not None else "N/A"
            conv_90_str = str(conv_90) if conv_90 is not None else "N/A"

            line = f"{name:20s} | {conv_95_str:>7s} | {conv_90_str:>7s} | {final_acc:>9.4f} | {best_acc:>8.4f} | {corr_net:>10.4f} | {corr_soft:>11.4f}"
            lines.append(line)

        return "\n".join(lines)

    def print_comparison_table(self):
        """Print comparison table to stdout."""
        print(self.get_comparison_table())


__all__ = [
    'MetricsTracker',
    'ComparisonMetrics',
    'pearson_correlation',
]
