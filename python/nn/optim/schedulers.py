"""
Syntonic Learning Rate Schedulers.

Golden ratio-based scheduling for more natural learning dynamics.

Source: CRT.md §12.2
"""

from __future__ import annotations
import torch
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from typing import Optional, List
import math

PHI = (1 + math.sqrt(5)) / 2
PHI_INV = 1 / PHI
Q_DEFICIT = 0.027395146920


class GoldenScheduler(LRScheduler):
    """
    Golden ratio-based learning rate scheduler.

    lr(t) = lr_0 × φ^{-t/T}

    Decays learning rate following the golden ratio,
    providing natural annealing that respects syntonic structure.

    Example:
        >>> scheduler = GoldenScheduler(optimizer, T_max=100)
        >>> for epoch in range(100):
        ...     train_one_epoch()
        ...     scheduler.step()
    """

    def __init__(
        self,
        optimizer: Optimizer,
        T_max: int,
        eta_min: float = 0.0,
        last_epoch: int = -1,
        verbose: bool = False,
    ):
        """
        Initialize golden scheduler.

        Args:
            optimizer: Wrapped optimizer
            T_max: Maximum number of epochs
            eta_min: Minimum learning rate
            last_epoch: Last epoch index
            verbose: Print lr updates
        """
        self.T_max = T_max
        self.eta_min = eta_min
        super().__init__(optimizer, last_epoch, verbose)

    def get_lr(self) -> List[float]:
        """Compute learning rate for current epoch."""
        if self.last_epoch == 0:
            return [group['lr'] for group in self.optimizer.param_groups]

        # Golden decay: lr × φ^{-t/T}
        decay_factor = PHI ** (-self.last_epoch / self.T_max)

        return [
            max(self.eta_min, base_lr * decay_factor)
            for base_lr in self.base_lrs
        ]


class SyntonyCyclicScheduler(LRScheduler):
    """
    Cyclic learning rate with syntony-aware cycle length.

    Uses Fibonacci sequence for cycle lengths, creating
    golden ratio relationships between cycles.

    Example:
        >>> scheduler = SyntonyCyclicScheduler(optimizer, base_lr=0.001, max_lr=0.01)
        >>> for step in range(10000):
        ...     train_one_step()
        ...     scheduler.step()
    """

    def __init__(
        self,
        optimizer: Optimizer,
        base_lr: float,
        max_lr: float,
        step_size_up: int = 2000,
        mode: str = 'triangular',
        gamma: float = 1.0,
        scale_fn: Optional[callable] = None,
        last_epoch: int = -1,
        verbose: bool = False,
    ):
        """
        Initialize cyclic scheduler.

        Args:
            optimizer: Wrapped optimizer
            base_lr: Minimum learning rate
            max_lr: Maximum learning rate
            step_size_up: Steps in ascending phase
            mode: Cycle mode ('triangular', 'triangular2', 'golden')
            gamma: Decay factor for 'exp_range' mode
            scale_fn: Custom scaling function
            last_epoch: Last epoch
            verbose: Print updates
        """
        self.base_lr_value = base_lr
        self.max_lr = max_lr
        self.step_size_up = step_size_up
        self.step_size_down = int(step_size_up * PHI)  # Golden ratio relationship
        self.mode = mode
        self.gamma = gamma

        if scale_fn is None:
            if mode == 'triangular':
                self.scale_fn = lambda x: 1.0
            elif mode == 'triangular2':
                self.scale_fn = lambda x: 1 / (2.0 ** (x - 1))
            elif mode == 'golden':
                self.scale_fn = lambda x: PHI ** (-x)
            else:
                self.scale_fn = lambda x: 1.0
        else:
            self.scale_fn = scale_fn

        self._cycle_count = 0
        self._step_in_cycle = 0

        super().__init__(optimizer, last_epoch, verbose)

    def get_lr(self) -> List[float]:
        """Compute learning rate for current step."""
        cycle_length = self.step_size_up + self.step_size_down

        # Position in current cycle
        step_in_cycle = self.last_epoch % cycle_length

        # Cycle count
        cycle = self.last_epoch // cycle_length

        # Position ratio
        if step_in_cycle < self.step_size_up:
            # Ascending phase
            x = step_in_cycle / self.step_size_up
        else:
            # Descending phase
            x = (cycle_length - step_in_cycle) / self.step_size_down

        # Scale by cycle count
        scale = self.scale_fn(cycle + 1)

        # Interpolate between base and max
        lr_range = self.max_lr - self.base_lr_value

        return [
            self.base_lr_value + lr_range * x * scale
            for _ in self.base_lrs
        ]

    @property
    def cycle_position(self) -> float:
        """Get position within current cycle [0, 1]."""
        cycle_length = self.step_size_up + self.step_size_down
        return (self.last_epoch % cycle_length) / cycle_length


class WarmupGoldenScheduler(LRScheduler):
    """
    Golden scheduler with linear warmup.

    First warmup_steps: linear increase from 0 to lr
    Then: golden decay

    Example:
        >>> scheduler = WarmupGoldenScheduler(optimizer, warmup_steps=1000, T_max=10000)
        >>> for step in range(10000):
        ...     train_one_step()
        ...     scheduler.step()
    """

    def __init__(
        self,
        optimizer: Optimizer,
        warmup_steps: int,
        T_max: int,
        eta_min: float = 0.0,
        last_epoch: int = -1,
        verbose: bool = False,
    ):
        """
        Initialize warmup golden scheduler.

        Args:
            optimizer: Wrapped optimizer
            warmup_steps: Number of warmup steps
            T_max: Maximum number of steps (after warmup)
            eta_min: Minimum learning rate
            last_epoch: Last epoch
            verbose: Print updates
        """
        self.warmup_steps = warmup_steps
        self.T_max = T_max
        self.eta_min = eta_min
        super().__init__(optimizer, last_epoch, verbose)

    def get_lr(self) -> List[float]:
        """Compute learning rate for current step."""
        if self.last_epoch < self.warmup_steps:
            # Linear warmup
            warmup_factor = self.last_epoch / self.warmup_steps
            return [base_lr * warmup_factor for base_lr in self.base_lrs]
        else:
            # Golden decay
            steps_after_warmup = self.last_epoch - self.warmup_steps
            decay_factor = PHI ** (-steps_after_warmup / self.T_max)

            return [
                max(self.eta_min, base_lr * decay_factor)
                for base_lr in self.base_lrs
            ]


class FibonacciScheduler(LRScheduler):
    """
    Learning rate scheduler using Fibonacci sequence.

    Changes learning rate at Fibonacci step counts,
    creating natural rhythm in training.

    Example:
        >>> scheduler = FibonacciScheduler(optimizer)
        >>> # LR changes at steps 1, 1, 2, 3, 5, 8, 13, 21, ...
    """

    def __init__(
        self,
        optimizer: Optimizer,
        decay_factor: float = 0.9,
        max_steps: int = 10000,
        last_epoch: int = -1,
        verbose: bool = False,
    ):
        """
        Initialize Fibonacci scheduler.

        Args:
            optimizer: Wrapped optimizer
            decay_factor: LR multiplier at each Fibonacci point
            max_steps: Maximum training steps
            last_epoch: Last epoch
            verbose: Print updates
        """
        self.decay_factor = decay_factor
        self.max_steps = max_steps

        # Precompute Fibonacci sequence up to max_steps
        self._fib_steps = self._compute_fibonacci(max_steps)
        self._decay_count = 0

        super().__init__(optimizer, last_epoch, verbose)

    def _compute_fibonacci(self, max_val: int) -> List[int]:
        """Compute Fibonacci numbers up to max_val."""
        fibs = [1, 1]
        while fibs[-1] < max_val:
            fibs.append(fibs[-1] + fibs[-2])
        return fibs

    def get_lr(self) -> List[float]:
        """Compute learning rate for current step."""
        # Count how many Fibonacci points we've passed
        decay_count = sum(1 for f in self._fib_steps if f <= self.last_epoch)

        return [
            base_lr * (self.decay_factor ** decay_count)
            for base_lr in self.base_lrs
        ]

    @property
    def next_fibonacci_step(self) -> Optional[int]:
        """Get next Fibonacci step count."""
        for f in self._fib_steps:
            if f > self.last_epoch:
                return f
        return None


class SyntonyAdaptiveScheduler(LRScheduler):
    """
    Scheduler that adapts based on syntony trajectory.

    - Rising syntony → decrease lr (consolidation phase)
    - Falling syntony → increase lr (exploration phase)
    - Stable syntony → maintain lr

    Example:
        >>> scheduler = SyntonyAdaptiveScheduler(optimizer)
        >>> for epoch in range(100):
        ...     train_one_epoch()
        ...     scheduler.step(syntony=model.syntony)
    """

    def __init__(
        self,
        optimizer: Optimizer,
        lr_min: float = 1e-6,
        lr_max: float = 0.1,
        adapt_rate: float = 0.1,
        syntony_window: int = 10,
        last_epoch: int = -1,
        verbose: bool = False,
    ):
        """
        Initialize syntony-adaptive scheduler.

        Args:
            optimizer: Wrapped optimizer
            lr_min: Minimum learning rate
            lr_max: Maximum learning rate
            adapt_rate: Rate of lr adaptation
            syntony_window: Window for syntony trend computation
            last_epoch: Last epoch
            verbose: Print updates
        """
        self.lr_min = lr_min
        self.lr_max = lr_max
        self.adapt_rate = adapt_rate
        self.syntony_window = syntony_window

        self._syntony_history: List[float] = []
        self._current_lr_multiplier = 1.0

        super().__init__(optimizer, last_epoch, verbose)

    def step(self, syntony: Optional[float] = None, epoch: Optional[int] = None):
        """
        Step scheduler with syntony value.

        Args:
            syntony: Current model syntony
            epoch: Current epoch (optional)
        """
        if syntony is not None:
            self._syntony_history.append(syntony)
            if len(self._syntony_history) > self.syntony_window * 2:
                self._syntony_history = self._syntony_history[-self.syntony_window * 2:]

            # Compute trend
            if len(self._syntony_history) >= self.syntony_window:
                recent = self._syntony_history[-self.syntony_window // 2:]
                earlier = self._syntony_history[-self.syntony_window:-self.syntony_window // 2]
                trend = sum(recent) / len(recent) - sum(earlier) / len(earlier)

                # Adapt lr multiplier
                # Rising syntony → decrease lr (multiply by < 1)
                # Falling syntony → increase lr (multiply by > 1)
                adaptation = 1.0 - self.adapt_rate * trend
                self._current_lr_multiplier *= adaptation

                # Clamp
                self._current_lr_multiplier = max(
                    self.lr_min / self.base_lrs[0],
                    min(self.lr_max / self.base_lrs[0], self._current_lr_multiplier)
                )

        super().step(epoch)

    def get_lr(self) -> List[float]:
        """Compute learning rate for current epoch."""
        return [
            max(self.lr_min, min(self.lr_max, base_lr * self._current_lr_multiplier))
            for base_lr in self.base_lrs
        ]
