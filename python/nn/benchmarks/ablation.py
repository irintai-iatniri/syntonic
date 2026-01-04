"""
Ablation Studies for Syntonic Networks.

Tools for analyzing which components of syntonic
networks contribute most to performance.

Source: CRT.md §12.2
"""

from __future__ import annotations
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Callable, Any
import copy
import math

PHI = (1 + math.sqrt(5)) / 2


@dataclass
class AblationResult:
    """
    Result of a single ablation experiment.

    Attributes:
        name: Name of ablated component
        accuracy: Test accuracy
        accuracy_delta: Change from baseline
        loss: Final loss
        loss_delta: Change from baseline
        syntony: Final syntony
        training_time: Training time
    """

    name: str
    accuracy: float
    accuracy_delta: float = 0.0
    loss: float = 0.0
    loss_delta: float = 0.0
    syntony: Optional[float] = None
    training_time: float = 0.0

    def __str__(self) -> str:
        delta_str = f"{self.accuracy_delta:+.2%}"
        s = f"{self.name}: {self.accuracy:.2%} ({delta_str})"
        if self.syntony is not None:
            s += f" [S={self.syntony:.3f}]"
        return s


class AblationStudy:
    """
    Run ablation studies on syntonic networks.

    Systematically disable components to measure
    their contribution.

    Example:
        >>> study = AblationStudy(model)
        >>> results = study.run(train_loader, test_loader)
        >>> study.print_report(results)
    """

    def __init__(
        self,
        model: nn.Module,
        device: str = 'cpu',
        epochs: int = 20,
        lr: float = 0.001,
    ):
        """
        Initialize ablation study.

        Args:
            model: Full syntonic model
            device: Compute device
            epochs: Training epochs per experiment
            lr: Learning rate
        """
        self.model = model
        self.device = device
        self.epochs = epochs
        self.lr = lr

        self._baseline_accuracy: Optional[float] = None

    def run(
        self,
        train_loader: DataLoader,
        test_loader: DataLoader,
        ablations: Optional[Dict[str, Callable]] = None,
    ) -> List[AblationResult]:
        """
        Run ablation study.

        Args:
            train_loader: Training data
            test_loader: Test data
            ablations: Dict of {name: ablation_fn}

        Returns:
            List of AblationResults
        """
        results = []

        # First run baseline (full model)
        print("Running baseline (full model)...")
        baseline_result = self._run_single(
            self.model, train_loader, test_loader, "Full Model"
        )
        results.append(baseline_result)
        self._baseline_accuracy = baseline_result.accuracy
        baseline_loss = baseline_result.loss

        # Default ablations if not provided
        if ablations is None:
            ablations = self._get_default_ablations()

        # Run each ablation
        for name, ablation_fn in ablations.items():
            print(f"Running ablation: {name}...")
            ablated_model = ablation_fn(copy.deepcopy(self.model))
            result = self._run_single(
                ablated_model, train_loader, test_loader, name
            )
            result.accuracy_delta = result.accuracy - self._baseline_accuracy
            result.loss_delta = result.loss - baseline_loss
            results.append(result)

        return results

    def _run_single(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        test_loader: DataLoader,
        name: str,
    ) -> AblationResult:
        """Run single experiment."""
        import time

        model = model.to(self.device)

        # Check if syntonic
        is_syntonic = hasattr(model, 'syntony')
        if is_syntonic:
            from syntonic.nn.optim import SyntonicAdam
            optimizer = SyntonicAdam(model.parameters(), lr=self.lr)
        else:
            optimizer = torch.optim.Adam(model.parameters(), lr=self.lr)

        criterion = nn.CrossEntropyLoss()

        start_time = time.time()

        for epoch in range(self.epochs):
            model.train()
            for inputs, targets in train_loader:
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)

                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                loss.backward()

                if is_syntonic and hasattr(model, 'syntony'):
                    optimizer.step(syntony=model.syntony)
                else:
                    optimizer.step()

        training_time = time.time() - start_time

        # Evaluate
        model.eval()
        correct = 0
        total = 0
        total_loss = 0.0

        with torch.no_grad():
            for inputs, targets in test_loader:
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)

                outputs = model(inputs)
                loss = criterion(outputs, targets)

                total_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

        accuracy = correct / total
        avg_loss = total_loss / len(test_loader)

        syntony = None
        if hasattr(model, 'syntony'):
            syntony = model.syntony

        return AblationResult(
            name=name,
            accuracy=accuracy,
            loss=avg_loss,
            syntony=syntony,
            training_time=training_time,
        )

    def _get_default_ablations(self) -> Dict[str, Callable]:
        """Get default ablation functions."""
        return {
            "No Differentiation": self._ablate_differentiation,
            "No Harmonization": self._ablate_harmonization,
            "No Recursion Block": self._ablate_recursion,
            "No Syntonic Norm": self._ablate_norm,
            "No Syntonic Gate": self._ablate_gate,
            "Standard Adam (no syntony-aware lr)": lambda m: m,  # Will use standard Adam
        }

    def _ablate_differentiation(self, model: nn.Module) -> nn.Module:
        """Replace differentiation layers with identity."""
        for name, module in model.named_modules():
            if 'diff' in name.lower() or 'differentiation' in name.lower():
                if hasattr(module, 'forward'):
                    module.forward = lambda x: x
        return model

    def _ablate_harmonization(self, model: nn.Module) -> nn.Module:
        """Replace harmonization layers with identity."""
        for name, module in model.named_modules():
            if 'harm' in name.lower() or 'harmonization' in name.lower():
                if hasattr(module, 'forward'):
                    module.forward = lambda x: x
        return model

    def _ablate_recursion(self, model: nn.Module) -> nn.Module:
        """Replace recursion blocks with pass-through."""
        for name, module in model.named_modules():
            if 'recursion' in name.lower():
                if hasattr(module, 'forward'):
                    # Just use differentiation without harmonization
                    if hasattr(module, 'diff') and hasattr(module, 'harm'):
                        original_forward = module.forward
                        module.forward = lambda x, m=module: m.diff(x)
        return model

    def _ablate_norm(self, model: nn.Module) -> nn.Module:
        """Replace syntonic norm with standard layer norm."""
        for name, module in model.named_modules():
            if 'syntonic' in name.lower() and 'norm' in name.lower():
                if hasattr(module, 'dim'):
                    parent = model
                    for part in name.split('.')[:-1]:
                        parent = getattr(parent, part)
                    setattr(parent, name.split('.')[-1], nn.LayerNorm(module.dim))
        return model

    def _ablate_gate(self, model: nn.Module) -> nn.Module:
        """Replace syntonic gate with fixed 0.5 mixing."""
        for name, module in model.named_modules():
            if 'gate' in name.lower():
                if hasattr(module, 'forward'):
                    module.forward = lambda d, h: 0.5 * d + 0.5 * h
        return model

    @staticmethod
    def print_report(results: List[AblationResult]):
        """Print ablation report."""
        print("\n" + "=" * 60)
        print("ABLATION STUDY RESULTS")
        print("=" * 60)

        # Sort by accuracy delta
        sorted_results = sorted(results[1:], key=lambda r: r.accuracy_delta)

        print(f"\nBaseline (Full Model): {results[0].accuracy:.2%}")
        if results[0].syntony is not None:
            print(f"Baseline Syntony: {results[0].syntony:.4f}")

        print("\nAblation Impact (sorted by importance):")
        print("-" * 60)
        print(f"{'Component':<35} {'Accuracy':<12} {'Delta':<12}")
        print("-" * 60)

        for result in sorted_results:
            delta_str = f"{result.accuracy_delta:+.2%}"
            print(f"{result.name:<35} {result.accuracy:.2%:<12} {delta_str:<12}")

        print("-" * 60)

        # Most important component
        most_important = min(sorted_results, key=lambda r: r.accuracy_delta)
        print(f"\nMost important component: {most_important.name}")
        print(f"  Removing it causes {most_important.accuracy_delta:.2%} accuracy drop")

        print("=" * 60)


def run_component_ablation(
    model: nn.Module,
    train_loader: DataLoader,
    test_loader: DataLoader,
    components: List[str],
    device: str = 'cpu',
    epochs: int = 20,
) -> Dict[str, AblationResult]:
    """
    Run ablation on specific components.

    Args:
        model: Model to ablate
        train_loader: Training data
        test_loader: Test data
        components: List of component names to ablate
        device: Compute device
        epochs: Training epochs

    Returns:
        Dict mapping component name to result
    """
    study = AblationStudy(model, device=device, epochs=epochs)

    # Build ablation dict for requested components
    ablations = {}
    for comp in components:
        if comp.lower() == 'differentiation':
            ablations[comp] = study._ablate_differentiation
        elif comp.lower() == 'harmonization':
            ablations[comp] = study._ablate_harmonization
        elif comp.lower() == 'recursion':
            ablations[comp] = study._ablate_recursion
        elif comp.lower() == 'norm':
            ablations[comp] = study._ablate_norm
        elif comp.lower() == 'gate':
            ablations[comp] = study._ablate_gate

    results = study.run(train_loader, test_loader, ablations)

    return {r.name: r for r in results}
