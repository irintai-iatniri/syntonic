"""
Syntonic SGD: SGD with syntony-modulated momentum.

Momentum is adjusted based on syntony:
- High syntony → higher momentum (confident trajectory)
- Low syntony → lower momentum (more exploration)

Source: CRT.md §12.2
"""

from __future__ import annotations
import torch
from torch.optim import Optimizer
from typing import Optional, Callable, Iterable
import math

PHI = (1 + math.sqrt(5)) / 2
PHI_INV = 1 / PHI
Q_DEFICIT = 0.027395146920
S_TARGET = PHI - Q_DEFICIT


class SyntonicSGD(Optimizer):
    """
    SGD with syntony-modulated learning rate.

    lr_eff = lr × (1 + α(S - S_target))

    Example:
        >>> optimizer = SyntonicSGD(model.parameters(), lr=0.01)
        >>> optimizer.step(syntony=model.syntony)
    """

    def __init__(
        self,
        params: Iterable[torch.Tensor],
        lr: float = 0.01,
        momentum: float = 0.0,
        dampening: float = 0.0,
        weight_decay: float = 0.0,
        nesterov: bool = False,
        alpha_syntony: float = 0.1,
        syntony_target: Optional[float] = None,
    ):
        """
        Initialize SyntonicSGD.

        Args:
            params: Model parameters
            lr: Base learning rate
            momentum: Momentum factor
            dampening: Dampening for momentum
            weight_decay: Weight decay (L2 penalty)
            nesterov: Use Nesterov momentum
            alpha_syntony: Syntony modulation strength
            syntony_target: Target syntony (default: φ - q)
        """
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if momentum < 0.0:
            raise ValueError(f"Invalid momentum: {momentum}")
        if nesterov and (momentum <= 0 or dampening != 0):
            raise ValueError("Nesterov requires momentum > 0 and dampening = 0")

        self.alpha_syntony = alpha_syntony
        self.syntony_target = syntony_target if syntony_target is not None else S_TARGET
        self._last_syntony = None
        self._effective_lr = lr

        defaults = dict(
            lr=lr,
            momentum=momentum,
            dampening=dampening,
            weight_decay=weight_decay,
            nesterov=nesterov,
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
            syntony: Current model syntony

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
            lr_multiplier = 1.0 + self.alpha_syntony * (syntony - self.syntony_target)
            lr_multiplier = max(0.1, min(2.0, lr_multiplier))

        for group in self.param_groups:
            effective_lr = group['lr'] * lr_multiplier
            self._effective_lr = effective_lr

            weight_decay = group['weight_decay']
            momentum = group['momentum']
            dampening = group['dampening']
            nesterov = group['nesterov']

            for p in group['params']:
                if p.grad is None:
                    continue

                d_p = p.grad

                if weight_decay != 0:
                    d_p = d_p.add(p, alpha=weight_decay)

                if momentum != 0:
                    state = self.state[p]
                    if 'momentum_buffer' not in state:
                        buf = state['momentum_buffer'] = torch.clone(d_p).detach()
                    else:
                        buf = state['momentum_buffer']
                        buf.mul_(momentum).add_(d_p, alpha=1 - dampening)

                    if nesterov:
                        d_p = d_p.add(buf, alpha=momentum)
                    else:
                        d_p = buf

                p.add_(d_p, alpha=-effective_lr)

        return loss

    @property
    def effective_lr(self) -> float:
        """Get current effective learning rate."""
        return self._effective_lr


class SyntonicMomentum(Optimizer):
    """
    SGD with syntony-modulated momentum.

    Momentum coefficient varies with syntony:
    μ_eff = μ × (φ^S / φ^S_target)

    High syntony → momentum approaches 1 (confident trajectory)
    Low syntony → momentum decreases (more exploration)

    Example:
        >>> optimizer = SyntonicMomentum(model.parameters(), lr=0.01, momentum=0.9)
        >>> optimizer.step(syntony=model.syntony)
    """

    def __init__(
        self,
        params: Iterable[torch.Tensor],
        lr: float = 0.01,
        momentum: float = 0.9,
        weight_decay: float = 0.0,
        nesterov: bool = True,
        syntony_target: Optional[float] = None,
        momentum_modulation: float = 0.2,
    ):
        """
        Initialize SyntonicMomentum.

        Args:
            params: Model parameters
            lr: Learning rate
            momentum: Base momentum coefficient
            weight_decay: Weight decay
            nesterov: Use Nesterov momentum
            syntony_target: Target syntony (default: φ - q)
            momentum_modulation: Strength of momentum modulation
        """
        if momentum < 0.0 or momentum > 1.0:
            raise ValueError(f"Invalid momentum: {momentum}")

        self.syntony_target = syntony_target if syntony_target is not None else S_TARGET
        self.momentum_modulation = momentum_modulation
        self._last_syntony = None
        self._effective_momentum = momentum

        defaults = dict(
            lr=lr,
            momentum=momentum,
            weight_decay=weight_decay,
            nesterov=nesterov,
        )
        super().__init__(params, defaults)

    def step(
        self,
        closure: Optional[Callable[[], float]] = None,
        syntony: Optional[float] = None,
    ) -> Optional[float]:
        """
        Perform optimization step with syntony-modulated momentum.

        Args:
            closure: Optional closure
            syntony: Current model syntony

        Returns:
            Loss value if closure provided
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            # Compute effective momentum based on syntony
            base_momentum = group['momentum']

            if syntony is not None:
                self._last_syntony = syntony
                # Golden ratio scaling of momentum
                # S > S_target → momentum increases
                # S < S_target → momentum decreases
                ratio = PHI ** (syntony - self.syntony_target)
                momentum_delta = self.momentum_modulation * (ratio - 1)
                effective_momentum = base_momentum + momentum_delta
                effective_momentum = max(0.0, min(0.99, effective_momentum))
            else:
                effective_momentum = base_momentum

            self._effective_momentum = effective_momentum

            weight_decay = group['weight_decay']
            nesterov = group['nesterov']
            lr = group['lr']

            for p in group['params']:
                if p.grad is None:
                    continue

                d_p = p.grad

                if weight_decay != 0:
                    d_p = d_p.add(p, alpha=weight_decay)

                state = self.state[p]
                if 'momentum_buffer' not in state:
                    buf = state['momentum_buffer'] = torch.clone(d_p).detach()
                else:
                    buf = state['momentum_buffer']
                    buf.mul_(effective_momentum).add_(d_p)

                if nesterov:
                    d_p = d_p.add(buf, alpha=effective_momentum)
                else:
                    d_p = buf

                p.add_(d_p, alpha=-lr)

        return loss

    @property
    def effective_momentum(self) -> float:
        """Get current effective momentum."""
        return self._effective_momentum
