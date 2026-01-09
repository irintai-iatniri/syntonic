"""
Syntonic Trainer: Training loop with syntony tracking.

Provides a complete training loop that:
1. Tracks syntony throughout training
2. Detects archonic patterns
3. Applies syntony-based learning rate modulation

Source: CRT.md §12.2
"""

from __future__ import annotations
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Optional, List, Dict, Any, Callable, Tuple
from dataclasses import dataclass, field
import math
import time

PHI = (1 + math.sqrt(5)) / 2
Q_DEFICIT = 0.027395146920
S_TARGET = PHI - Q_DEFICIT


@dataclass
class TrainingConfig:
    """Configuration for syntonic training."""

    epochs: int = 100
    lr: float = 0.001
    weight_decay: float = 0.01

    # Syntony parameters
    lambda_syntony: float = 0.1
    mu_phase: float = 0.01
    syntony_target: float = S_TARGET

    # Training options
    gradient_clip: float = 1.0
    use_syntonic_optimizer: bool = True
    use_archonic_detection: bool = True

    # Logging
    log_interval: int = 10
    eval_interval: int = 1

    # Device
    device: str = 'cpu'


class SyntonicTrainer:
    """
    Complete training loop with syntony awareness.

    Handles:
    - Forward/backward passes
    - Syntony computation and tracking
    - Archonic pattern detection
    - Learning rate modulation

    Example:
        >>> model = SyntonicMLP(784, [256, 128], 10)
        >>> trainer = SyntonicTrainer(model, train_loader, val_loader)
        >>> history = trainer.train(epochs=100)
        >>> print(f"Final syntony: {trainer.current_syntony:.4f}")
    """

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        config: Optional[TrainingConfig] = None,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[Any] = None,
        loss_fn: Optional[nn.Module] = None,
        callbacks: Optional[List[Any]] = None,
    ):
        """
        Initialize trainer.

        Args:
            model: Neural network to train
            train_loader: Training data loader
            val_loader: Validation data loader
            config: Training configuration
            optimizer: Custom optimizer (default: SyntonicAdam)
            scheduler: Learning rate scheduler
            loss_fn: Loss function (default: SyntonicLoss)
            callbacks: List of callbacks
        """
        self.config = config or TrainingConfig()
        self.model = model.to(self.config.device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.callbacks = callbacks or []

        # Setup optimizer
        if optimizer is None:
            from syntonic.nn.optim import SyntonicAdam
            self.optimizer = SyntonicAdam(
                model.parameters(),
                lr=self.config.lr,
                weight_decay=self.config.weight_decay,
            )
        else:
            self.optimizer = optimizer

        self.scheduler = scheduler

        # Setup loss function
        if loss_fn is None:
            from syntonic.nn.loss import SyntonicLoss
            self.loss_fn = SyntonicLoss(
                nn.CrossEntropyLoss(),
                lambda_syntony=self.config.lambda_syntony,
                mu_phase=self.config.mu_phase,
            )
        else:
            self.loss_fn = loss_fn

        # Tracking
        self._syntony_history: List[float] = []
        self._loss_history: List[float] = []
        self._current_syntony = 0.5
        self._current_epoch = 0
        self._global_step = 0

        # Archonic detection
        self._archonic_count = 0

    def train(
        self,
        epochs: Optional[int] = None,
    ) -> Dict[str, List[float]]:
        """
        Run training loop.

        Args:
            epochs: Number of epochs (overrides config)

        Returns:
            Training history dictionary
        """
        epochs = epochs or self.config.epochs

        history = {
            'train_loss': [],
            'train_syntony': [],
            'val_loss': [],
            'val_syntony': [],
            'lr': [],
        }

        # Call callbacks
        self._on_train_begin()

        for epoch in range(epochs):
            self._current_epoch = epoch

            # Train one epoch
            train_metrics = self._train_epoch()
            history['train_loss'].append(train_metrics['loss'])
            history['train_syntony'].append(train_metrics['syntony'])

            # Validation
            if self.val_loader is not None and (epoch + 1) % self.config.eval_interval == 0:
                val_metrics = self._validate()
                history['val_loss'].append(val_metrics['loss'])
                history['val_syntony'].append(val_metrics['syntony'])
            else:
                history['val_loss'].append(None)
                history['val_syntony'].append(None)

            # Learning rate
            current_lr = self.optimizer.param_groups[0]['lr']
            history['lr'].append(current_lr)

            # Scheduler step
            if self.scheduler is not None:
                if hasattr(self.scheduler, 'step'):
                    # Some schedulers need syntony
                    if hasattr(self.scheduler, '_syntony_history'):
                        self.scheduler.step(syntony=train_metrics['syntony'])
                    else:
                        self.scheduler.step()

            # Callbacks
            self._on_epoch_end(epoch, train_metrics, val_metrics if self.val_loader else None)

            # Early stopping check
            if self._should_stop():
                break

        self._on_train_end()
        return history

    def _train_epoch(self) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()

        total_loss = 0.0
        total_syntony = 0.0
        n_batches = 0

        for batch_idx, (inputs, targets) in enumerate(self.train_loader):
            inputs = inputs.to(self.config.device)
            targets = targets.to(self.config.device)

            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(inputs)

            # Compute loss
            if hasattr(self.loss_fn, 'forward') and 'model' in self.loss_fn.forward.__code__.co_varnames:
                loss, metrics = self.loss_fn(outputs, targets, self.model, inputs)
                batch_syntony = metrics.get('syntony', 0.5)
            else:
                loss = self.loss_fn(outputs, targets)
                batch_syntony = self._compute_model_syntony()
                metrics = {'loss': loss.item(), 'syntony': batch_syntony}

            # Backward pass
            loss.backward()

            # Gradient clipping
            if self.config.gradient_clip > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config.gradient_clip
                )

            # Optimizer step with syntony
            if hasattr(self.optimizer, 'step') and 'syntony' in self.optimizer.step.__code__.co_varnames:
                self.optimizer.step(syntony=batch_syntony)
            else:
                self.optimizer.step()

            # INTEGRATION: Crystallize ResonantParameters after weight update
            from syntonic.nn.layers.resonant_parameter import crystallize_all_resonant_
            crystallize_all_resonant_(self.model)

            # Track
            total_loss += loss.item()
            total_syntony += batch_syntony
            n_batches += 1
            self._global_step += 1

            # Logging
            if batch_idx % self.config.log_interval == 0:
                self._on_batch_end(batch_idx, metrics)

        avg_loss = total_loss / max(1, n_batches)
        avg_syntony = total_syntony / max(1, n_batches)

        self._current_syntony = avg_syntony
        self._syntony_history.append(avg_syntony)
        self._loss_history.append(avg_loss)

        return {'loss': avg_loss, 'syntony': avg_syntony}

    def _validate(self) -> Dict[str, float]:
        """Run validation."""
        self.model.eval()

        total_loss = 0.0
        total_syntony = 0.0
        n_batches = 0

        with torch.no_grad():
            for inputs, targets in self.val_loader:
                inputs = inputs.to(self.config.device)
                targets = targets.to(self.config.device)

                outputs = self.model(inputs)

                if hasattr(self.loss_fn, 'forward') and 'model' in self.loss_fn.forward.__code__.co_varnames:
                    loss, metrics = self.loss_fn(outputs, targets, self.model, inputs)
                    batch_syntony = metrics.get('syntony', 0.5)
                else:
                    loss = self.loss_fn(outputs, targets)
                    batch_syntony = self._compute_model_syntony()

                total_loss += loss.item()
                total_syntony += batch_syntony
                n_batches += 1

        return {
            'loss': total_loss / max(1, n_batches),
            'syntony': total_syntony / max(1, n_batches),
        }

    def _compute_model_syntony(self) -> float:
        """Compute current model syntony."""
        syntonies = []
        for module in self.model.modules():
            if hasattr(module, 'syntony') and module.syntony is not None:
                syntonies.append(module.syntony)

        if syntonies:
            return sum(syntonies) / len(syntonies)
        return 0.5

    def _should_stop(self) -> bool:
        """Check if training should stop early."""
        for callback in self.callbacks:
            if hasattr(callback, 'should_stop') and callback.should_stop:
                return True
        return False

    def _on_train_begin(self):
        """Call at start of training."""
        for callback in self.callbacks:
            if hasattr(callback, 'on_train_begin'):
                callback.on_train_begin(self)

    def _on_train_end(self):
        """Call at end of training."""
        for callback in self.callbacks:
            if hasattr(callback, 'on_train_end'):
                callback.on_train_end(self)

    def _on_epoch_end(
        self,
        epoch: int,
        train_metrics: Dict[str, float],
        val_metrics: Optional[Dict[str, float]],
    ):
        """Call at end of epoch."""
        for callback in self.callbacks:
            if hasattr(callback, 'on_epoch_end'):
                callback.on_epoch_end(self, epoch, train_metrics, val_metrics)

    def _on_batch_end(self, batch_idx: int, metrics: Dict[str, Any]):
        """Call at end of batch."""
        for callback in self.callbacks:
            if hasattr(callback, 'on_batch_end'):
                callback.on_batch_end(self, batch_idx, metrics)

    @property
    def current_syntony(self) -> float:
        """Get current model syntony."""
        return self._current_syntony

    @property
    def syntony_history(self) -> List[float]:
        """Get syntony history."""
        return self._syntony_history

    @property
    def is_archonic(self) -> bool:
        """Check if model is in archonic pattern."""
        if len(self._syntony_history) < 50:
            return False

        recent = self._syntony_history[-50:]
        mean_S = sum(recent) / len(recent)
        var_S = sum((s - mean_S) ** 2 for s in recent) / len(recent)

        mid = len(recent) // 2
        trend = sum(recent[mid:]) / len(recent[mid:]) - sum(recent[:mid]) / len(recent[:mid])

        return (
            var_S > 0.01 and
            abs(trend) < 0.001 and
            mean_S < S_TARGET - 0.1
        )
