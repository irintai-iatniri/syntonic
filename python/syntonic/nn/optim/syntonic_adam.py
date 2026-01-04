"""
Syntonic Adam: Adam optimizer with syntony-aware learning rate.

lr_eff = lr × (1 + α(S - S_target))

When syntony is high, learning rate increases (efficient learning).
When syntony is low, learning rate decreases (careful exploration).

Source: CRT.md §12.2
"""

from __future__ import annotations
import torch
from torch.optim import Optimizer
from typing import Optional, Callable, Iterable, Tuple, Dict, Any
import math

PHI = (1 + math.sqrt(5)) / 2
Q_DEFICIT = 0.027395146920
S_TARGET = PHI - Q_DEFICIT  # ≈ 1.591


class SyntonicAdam(Optimizer):
    """
    Adam optimizer with syntony-modulated learning rate.

    lr_eff = lr × (1 + α(S - S_target))

    This allows the optimizer to:
    - Speed up when representations are coherent (high S)
    - Slow down when representations are fragmented (low S)

    Example:
        >>> optimizer = SyntonicAdam(model.parameters(), lr=0.001)
        >>> for batch in dataloader:
        ...     loss = model(batch)
        ...     optimizer.zero_grad()
        ...     loss.backward()
        ...     optimizer.step(syntony=model.syntony)
    """

    def __init__(
        self,
        params: Iterable[torch.Tensor],
        lr: float = 1e-3,
        betas: Tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0,
        alpha_syntony: float = 0.1,
        syntony_target: Optional[float] = None,
        amsgrad: bool = False,
    ):
        """
        Initialize SyntonicAdam.

        Args:
            params: Model parameters
            lr: Base learning rate
            betas: Adam momentum coefficients
            eps: Epsilon for numerical stability
            weight_decay: L2 regularization
            alpha_syntony: Syntony modulation strength
            syntony_target: Target syntony (default: φ - q)
            amsgrad: Use AMSGrad variant
        """
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if eps < 0.0:
            raise ValueError(f"Invalid epsilon: {eps}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta[0]: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta[1]: {betas[1]}")

        self.alpha_syntony = alpha_syntony
        self.syntony_target = syntony_target if syntony_target is not None else S_TARGET
        self._last_syntony = None
        self._effective_lr = lr

        defaults = dict(
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            amsgrad=amsgrad,
        )
        super().__init__(params, defaults)

    def step(
        self,
        closure: Optional[Callable[[], float]] = None,
        syntony: Optional[float] = None,
    ) -> Optional[float]:
        """
        Perform a single optimization step.

        Args:
            closure: Optional closure for loss computation
            syntony: Current model syntony for lr modulation

        Returns:
            Loss value if closure provided
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        # Compute effective learning rate based on syntony
        lr_multiplier = 1.0
        if syntony is not None:
            self._last_syntony = syntony
            # Modulate: higher syntony → higher lr
            lr_multiplier = 1.0 + self.alpha_syntony * (syntony - self.syntony_target)
            # Clamp to prevent extreme values
            lr_multiplier = max(0.1, min(2.0, lr_multiplier))

        for group in self.param_groups:
            effective_lr = group['lr'] * lr_multiplier
            self._effective_lr = effective_lr

            beta1, beta2 = group['betas']
            amsgrad = group['amsgrad']

            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad
                if grad.is_sparse:
                    raise RuntimeError('SyntonicAdam does not support sparse gradients')

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p)
                    state['exp_avg_sq'] = torch.zeros_like(p)
                    if amsgrad:
                        state['max_exp_avg_sq'] = torch.zeros_like(p)

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                if amsgrad:
                    max_exp_avg_sq = state['max_exp_avg_sq']

                state['step'] += 1

                # Weight decay
                if group['weight_decay'] != 0:
                    grad = grad.add(p, alpha=group['weight_decay'])

                # Update biased first moment estimate
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)

                # Update biased second raw moment estimate
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                if amsgrad:
                    # Maintains max of all 2nd moment running avg
                    torch.max(max_exp_avg_sq, exp_avg_sq, out=max_exp_avg_sq)
                    denom = max_exp_avg_sq.sqrt().add_(group['eps'])
                else:
                    denom = exp_avg_sq.sqrt().add_(group['eps'])

                # Bias correction
                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']
                step_size = effective_lr * math.sqrt(bias_correction2) / bias_correction1

                # Update parameters
                p.addcdiv_(exp_avg, denom, value=-step_size)

        return loss

    @property
    def effective_lr(self) -> float:
        """Get current effective learning rate."""
        return self._effective_lr

    @property
    def last_syntony(self) -> Optional[float]:
        """Get last syntony value used."""
        return self._last_syntony


class AdaptiveSyntonicAdam(SyntonicAdam):
    """
    Adaptive Syntonic Adam with self-adjusting α.

    Automatically adjusts alpha_syntony based on training dynamics:
    - Stable training → increase α (more syntony influence)
    - Unstable training → decrease α (conservative updates)

    Example:
        >>> optimizer = AdaptiveSyntonicAdam(model.parameters())
        >>> # α adjusts automatically during training
    """

    def __init__(
        self,
        params: Iterable[torch.Tensor],
        lr: float = 1e-3,
        betas: Tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0,
        alpha_syntony: float = 0.1,
        alpha_min: float = 0.01,
        alpha_max: float = 0.5,
        adapt_rate: float = 0.01,
        stability_window: int = 100,
    ):
        """
        Initialize AdaptiveSyntonicAdam.

        Args:
            params: Model parameters
            lr: Base learning rate
            betas: Adam momentum coefficients
            eps: Epsilon for numerical stability
            weight_decay: L2 regularization
            alpha_syntony: Initial syntony modulation strength
            alpha_min: Minimum alpha value
            alpha_max: Maximum alpha value
            adapt_rate: Rate of alpha adaptation
            stability_window: Window for stability computation
        """
        super().__init__(
            params, lr, betas, eps, weight_decay, alpha_syntony
        )
        self.alpha_min = alpha_min
        self.alpha_max = alpha_max
        self.adapt_rate = adapt_rate
        self.stability_window = stability_window

        self._loss_history: list = []
        self._syntony_history: list = []

    def step(
        self,
        closure: Optional[Callable[[], float]] = None,
        syntony: Optional[float] = None,
        loss: Optional[float] = None,
    ) -> Optional[float]:
        """
        Perform optimization step with adaptive α.

        Args:
            closure: Optional closure for loss computation
            syntony: Current model syntony
            loss: Current loss value (for stability tracking)

        Returns:
            Loss value if closure provided
        """
        # Track history for stability computation
        if loss is not None:
            self._loss_history.append(loss)
            if len(self._loss_history) > self.stability_window:
                self._loss_history = self._loss_history[-self.stability_window:]

        if syntony is not None:
            self._syntony_history.append(syntony)
            if len(self._syntony_history) > self.stability_window:
                self._syntony_history = self._syntony_history[-self.stability_window:]

        # Adapt alpha based on stability
        if len(self._loss_history) >= 10:
            stability = self._compute_stability()
            self._adapt_alpha(stability)

        return super().step(closure, syntony)

    def _compute_stability(self) -> float:
        """
        Compute training stability score.

        Returns value in [0, 1]: 1 = stable, 0 = unstable
        """
        if len(self._loss_history) < 2:
            return 0.5

        # Loss variance (normalized)
        losses = self._loss_history[-50:]
        mean_loss = sum(losses) / len(losses)
        var_loss = sum((l - mean_loss) ** 2 for l in losses) / len(losses)
        normalized_var = var_loss / (mean_loss ** 2 + 1e-8)

        # Syntony trend
        if len(self._syntony_history) >= 10:
            syntonies = self._syntony_history[-50:]
            mid = len(syntonies) // 2
            trend = sum(syntonies[mid:]) / len(syntonies[mid:]) - \
                    sum(syntonies[:mid]) / len(syntonies[:mid])
            trend_score = 0.5 + trend  # Positive trend → higher score
        else:
            trend_score = 0.5

        # Combine: low variance + positive trend = stable
        variance_score = 1.0 / (1.0 + normalized_var)
        stability = 0.7 * variance_score + 0.3 * trend_score

        return max(0.0, min(1.0, stability))

    def _adapt_alpha(self, stability: float):
        """Adapt alpha_syntony based on stability."""
        target_alpha = self.alpha_min + (self.alpha_max - self.alpha_min) * stability

        # Smooth adaptation
        self.alpha_syntony = (
            (1 - self.adapt_rate) * self.alpha_syntony +
            self.adapt_rate * target_alpha
        )

    @property
    def current_alpha(self) -> float:
        """Get current alpha_syntony value."""
        return self.alpha_syntony
