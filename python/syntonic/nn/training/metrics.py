"""
Training Metrics for Syntonic Networks.

Metrics computation and aggregation for syntony-aware training.

Source: CRT.md §12.2
"""

from __future__ import annotations
import torch
import torch.nn as nn
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
import math

PHI = (1 + math.sqrt(5)) / 2
Q_DEFICIT = 0.027395146920
S_TARGET = PHI - Q_DEFICIT


@dataclass
class SyntonyMetrics:
    """
    Metrics for tracking syntony.

    Attributes:
        global_syntony: Overall model syntony
        layer_syntonies: Per-layer syntony values
        syntony_trend: Recent syntony trend
        is_archonic: Whether archonic pattern detected
    """

    global_syntony: float = 0.0
    layer_syntonies: List[float] = field(default_factory=list)
    syntony_trend: float = 0.0
    is_archonic: bool = False
    phase_alignment: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'global_syntony': self.global_syntony,
            'layer_syntonies': self.layer_syntonies,
            'syntony_trend': self.syntony_trend,
            'is_archonic': self.is_archonic,
            'phase_alignment': self.phase_alignment,
        }


@dataclass
class TrainingMetrics:
    """
    Comprehensive training metrics.

    Attributes:
        loss: Total loss
        task_loss: Task-specific loss
        syntony_loss: Syntony regularization loss
        phase_loss: Phase alignment loss
        accuracy: Classification accuracy (if applicable)
        syntony: Syntony metrics
    """

    loss: float = 0.0
    task_loss: float = 0.0
    syntony_loss: float = 0.0
    phase_loss: float = 0.0
    accuracy: Optional[float] = None
    syntony: SyntonyMetrics = field(default_factory=SyntonyMetrics)

    # Additional metrics
    gradient_norm: float = 0.0
    weight_norm: float = 0.0
    learning_rate: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        result = {
            'loss': self.loss,
            'task_loss': self.task_loss,
            'syntony_loss': self.syntony_loss,
            'phase_loss': self.phase_loss,
            'gradient_norm': self.gradient_norm,
            'weight_norm': self.weight_norm,
            'learning_rate': self.learning_rate,
        }
        if self.accuracy is not None:
            result['accuracy'] = self.accuracy
        result.update(self.syntony.to_dict())
        return result


def compute_epoch_metrics(
    model: nn.Module,
    data_loader: torch.utils.data.DataLoader,
    loss_fn: nn.Module,
    device: str = 'cpu',
) -> TrainingMetrics:
    """
    Compute comprehensive metrics for one epoch.

    Args:
        model: Neural network
        data_loader: Data loader
        loss_fn: Loss function
        device: Compute device

    Returns:
        TrainingMetrics object
    """
    model.eval()

    total_loss = 0.0
    total_task_loss = 0.0
    total_syntony_loss = 0.0
    total_phase_loss = 0.0
    total_correct = 0
    total_samples = 0

    syntonies = []

    with torch.no_grad():
        for inputs, targets in data_loader:
            inputs = inputs.to(device)
            targets = targets.to(device)

            outputs = model(inputs)

            # Compute loss
            if hasattr(loss_fn, 'forward') and 'model' in loss_fn.forward.__code__.co_varnames:
                loss, metrics_dict = loss_fn(outputs, targets, model, inputs)
                total_task_loss += metrics_dict.get('loss_task', 0.0)
                total_syntony_loss += metrics_dict.get('loss_syntony', 0.0)
                total_phase_loss += metrics_dict.get('loss_phase', 0.0)
                syntonies.append(metrics_dict.get('syntony', 0.5))
            else:
                loss = loss_fn(outputs, targets)
                total_task_loss += loss.item()
                syntonies.append(_compute_model_syntony(model))

            total_loss += loss.item()

            # Accuracy for classification
            if outputs.dim() == 2 and outputs.shape[1] > 1:
                preds = outputs.argmax(dim=1)
                total_correct += (preds == targets).sum().item()
                total_samples += targets.size(0)

    n_batches = len(data_loader)

    # Compute layer syntonies
    layer_syntonies = []
    for module in model.modules():
        if hasattr(module, 'syntony') and module.syntony is not None:
            layer_syntonies.append(module.syntony)

    # Compute trend
    syntony_trend = 0.0
    if len(syntonies) >= 4:
        mid = len(syntonies) // 2
        syntony_trend = sum(syntonies[mid:]) / len(syntonies[mid:]) - \
                       sum(syntonies[:mid]) / len(syntonies[:mid])

    # Check archonic
    global_syntony = sum(syntonies) / len(syntonies) if syntonies else 0.5
    is_archonic = _check_archonic(syntonies)

    return TrainingMetrics(
        loss=total_loss / n_batches,
        task_loss=total_task_loss / n_batches,
        syntony_loss=total_syntony_loss / n_batches,
        phase_loss=total_phase_loss / n_batches,
        accuracy=total_correct / total_samples if total_samples > 0 else None,
        syntony=SyntonyMetrics(
            global_syntony=global_syntony,
            layer_syntonies=layer_syntonies,
            syntony_trend=syntony_trend,
            is_archonic=is_archonic,
        ),
    )


def _compute_model_syntony(model: nn.Module) -> float:
    """Compute model syntony from modules."""
    syntonies = []
    for module in model.modules():
        if hasattr(module, 'syntony') and module.syntony is not None:
            syntonies.append(module.syntony)
    return sum(syntonies) / len(syntonies) if syntonies else 0.5


def _check_archonic(syntonies: List[float], threshold: float = 0.01) -> bool:
    """Check for archonic pattern in syntony sequence."""
    if len(syntonies) < 10:
        return False

    mean_S = sum(syntonies) / len(syntonies)
    var_S = sum((s - mean_S) ** 2 for s in syntonies) / len(syntonies)

    mid = len(syntonies) // 2
    trend = sum(syntonies[mid:]) / len(syntonies[mid:]) - \
            sum(syntonies[:mid]) / len(syntonies[:mid])

    return var_S > threshold and abs(trend) < threshold / 5 and mean_S < S_TARGET - 0.1


def compute_gradient_metrics(model: nn.Module) -> Dict[str, float]:
    """
    Compute gradient statistics.

    Args:
        model: Neural network (after backward pass)

    Returns:
        Dictionary with gradient metrics
    """
    grad_norms = []
    grad_means = []
    grad_stds = []

    for param in model.parameters():
        if param.grad is not None:
            grad_norms.append(param.grad.norm().item())
            grad_means.append(param.grad.mean().item())
            grad_stds.append(param.grad.std().item())

    if not grad_norms:
        return {'grad_norm': 0.0, 'grad_mean': 0.0, 'grad_std': 0.0}

    return {
        'grad_norm': sum(grad_norms) / len(grad_norms),
        'grad_norm_max': max(grad_norms),
        'grad_mean': sum(grad_means) / len(grad_means),
        'grad_std': sum(grad_stds) / len(grad_stds),
    }


def compute_weight_metrics(model: nn.Module) -> Dict[str, float]:
    """
    Compute weight statistics.

    Args:
        model: Neural network

    Returns:
        Dictionary with weight metrics
    """
    weight_norms = []
    weight_means = []
    weight_stds = []

    for param in model.parameters():
        weight_norms.append(param.norm().item())
        weight_means.append(param.mean().item())
        weight_stds.append(param.std().item())

    if not weight_norms:
        return {'weight_norm': 0.0, 'weight_mean': 0.0, 'weight_std': 0.0}

    return {
        'weight_norm': sum(weight_norms) / len(weight_norms),
        'weight_norm_max': max(weight_norms),
        'weight_mean': sum(weight_means) / len(weight_means),
        'weight_std': sum(weight_stds) / len(weight_stds),
    }


def compute_syntony_gap(
    current_syntony: float,
    target: Optional[float] = None,
) -> Dict[str, float]:
    """
    Compute gap between current and target syntony.

    Args:
        current_syntony: Current model syntony
        target: Target syntony (default: φ - q)

    Returns:
        Dictionary with gap metrics
    """
    target = target if target is not None else S_TARGET

    gap = target - current_syntony
    relative_gap = gap / target if target != 0 else 0.0

    return {
        'syntony_gap': gap,
        'syntony_relative_gap': relative_gap,
        'syntony_ratio': current_syntony / target if target != 0 else 0.0,
        'target': target,
    }


class MetricsAggregator:
    """
    Aggregate metrics over multiple batches/epochs.

    Example:
        >>> aggregator = MetricsAggregator()
        >>> for batch in dataloader:
        ...     metrics = compute_batch_metrics(batch)
        ...     aggregator.update(metrics)
        >>> epoch_metrics = aggregator.compute()
    """

    def __init__(self):
        """Initialize aggregator."""
        self._metrics: Dict[str, List[float]] = {}
        self._counts: Dict[str, int] = {}

    def update(self, metrics: Dict[str, float], count: int = 1):
        """
        Add batch metrics.

        Args:
            metrics: Dictionary of metric values
            count: Number of samples in batch
        """
        for key, value in metrics.items():
            if key not in self._metrics:
                self._metrics[key] = []
                self._counts[key] = 0
            self._metrics[key].append(value * count)
            self._counts[key] += count

    def compute(self) -> Dict[str, float]:
        """
        Compute aggregated metrics.

        Returns:
            Dictionary of mean metric values
        """
        result = {}
        for key, values in self._metrics.items():
            total = sum(values)
            count = self._counts[key]
            result[key] = total / count if count > 0 else 0.0
        return result

    def reset(self):
        """Reset aggregator."""
        self._metrics = {}
        self._counts = {}
