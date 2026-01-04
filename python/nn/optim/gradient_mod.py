"""
Syntony-aware Gradient Modifications.

Gradient clipping and modifications that respect syntonic structure.

Source: CRT.md §12.2
"""

from __future__ import annotations
import torch
import torch.nn as nn
from typing import Optional, List, Dict, Iterator
import math

PHI = (1 + math.sqrt(5)) / 2
PHI_INV = 1 / PHI
Q_DEFICIT = 0.027395146920
S_TARGET = PHI - Q_DEFICIT


class GoldenClipping:
    """
    Golden ratio-based gradient clipping.

    Clips gradients using golden ratio thresholds:
    - Layer-wise thresholds follow φ^{-l}
    - Global threshold is φ

    Example:
        >>> clipper = GoldenClipping(model)
        >>> loss.backward()
        >>> clipper.clip_gradients()
        >>> optimizer.step()
    """

    def __init__(
        self,
        model: nn.Module,
        max_norm: float = PHI,
        norm_type: float = 2.0,
        layer_scaling: bool = True,
    ):
        """
        Initialize golden clipping.

        Args:
            model: Neural network
            max_norm: Maximum gradient norm (default: φ)
            norm_type: Type of norm (default: L2)
            layer_scaling: Scale clip threshold by layer depth
        """
        self.model = model
        self.max_norm = max_norm
        self.norm_type = norm_type
        self.layer_scaling = layer_scaling

        # Compute layer indices
        self._param_layer_idx: Dict[nn.Parameter, int] = {}
        layer_idx = 0
        for module in model.modules():
            for param in module.parameters(recurse=False):
                self._param_layer_idx[param] = layer_idx
            if list(module.parameters(recurse=False)):
                layer_idx += 1

        self._n_layers = layer_idx
        self._last_grad_norm = 0.0

    def clip_gradients(self) -> float:
        """
        Clip gradients using golden ratio thresholds.

        Returns:
            Total gradient norm before clipping
        """
        parameters = [p for p in self.model.parameters() if p.grad is not None]
        if not parameters:
            return 0.0

        # Compute total gradient norm
        total_norm = torch.norm(
            torch.stack([
                torch.norm(p.grad.detach(), self.norm_type)
                for p in parameters
            ]),
            self.norm_type
        ).item()
        self._last_grad_norm = total_norm

        # Global clipping
        if total_norm > self.max_norm:
            clip_coef = self.max_norm / (total_norm + 1e-8)

            for p in parameters:
                if self.layer_scaling:
                    # Layer-specific scaling
                    layer_idx = self._param_layer_idx.get(p, 0)
                    layer_scale = PHI ** (-layer_idx / max(1, self._n_layers - 1))
                    p.grad.mul_(clip_coef * layer_scale)
                else:
                    p.grad.mul_(clip_coef)

        return total_norm

    @property
    def last_grad_norm(self) -> float:
        """Get last computed gradient norm."""
        return self._last_grad_norm


class SyntonyGradientModifier:
    """
    Modify gradients based on syntony.

    High syntony → allow larger gradients (confident updates)
    Low syntony → constrain gradients (careful updates)

    Example:
        >>> modifier = SyntonyGradientModifier(model)
        >>> loss.backward()
        >>> modifier.modify(syntony=model.syntony)
        >>> optimizer.step()
    """

    def __init__(
        self,
        model: nn.Module,
        base_max_norm: float = 1.0,
        syntony_modulation: float = 0.5,
        syntony_target: Optional[float] = None,
    ):
        """
        Initialize gradient modifier.

        Args:
            model: Neural network
            base_max_norm: Base maximum gradient norm
            syntony_modulation: Strength of syntony modulation
            syntony_target: Target syntony (default: φ - q)
        """
        self.model = model
        self.base_max_norm = base_max_norm
        self.syntony_modulation = syntony_modulation
        self.syntony_target = syntony_target if syntony_target is not None else S_TARGET

        self._effective_max_norm = base_max_norm
        self._last_modification = 1.0

    def modify(
        self,
        syntony: Optional[float] = None,
        clip: bool = True,
    ) -> float:
        """
        Modify gradients based on syntony.

        Args:
            syntony: Current model syntony
            clip: Whether to clip after modification

        Returns:
            Effective max norm used
        """
        # Compute effective max norm
        if syntony is not None:
            # High syntony → higher max norm (more confident)
            ratio = 1.0 + self.syntony_modulation * (syntony - self.syntony_target)
            self._effective_max_norm = self.base_max_norm * max(0.5, min(2.0, ratio))
        else:
            self._effective_max_norm = self.base_max_norm

        if clip:
            # Clip gradients
            parameters = [p for p in self.model.parameters() if p.grad is not None]
            if parameters:
                total_norm = torch.nn.utils.clip_grad_norm_(
                    parameters, self._effective_max_norm
                )
                self._last_modification = min(1.0, self._effective_max_norm / (total_norm + 1e-8))

        return self._effective_max_norm

    @property
    def effective_max_norm(self) -> float:
        """Get current effective max norm."""
        return self._effective_max_norm


class ArchonicGradientEscape:
    """
    Gradient modification for escaping archonic patterns.

    When archonic cycling is detected, injects noise into
    gradients to escape the stuck configuration.

    Example:
        >>> escape = ArchonicGradientEscape(model)
        >>> for batch in dataloader:
        ...     loss.backward()
        ...     escape.step(syntony=model.syntony)
        ...     optimizer.step()
    """

    def __init__(
        self,
        model: nn.Module,
        noise_scale: float = 0.01,
        detection_window: int = 100,
        variance_threshold: float = 0.01,
        cooldown: int = 50,
    ):
        """
        Initialize archonic escape.

        Args:
            model: Neural network
            noise_scale: Scale of escape noise
            detection_window: Window for archonic detection
            variance_threshold: Threshold for detecting cycling
            cooldown: Steps between escape injections
        """
        self.model = model
        self.noise_scale = noise_scale
        self.detection_window = detection_window
        self.variance_threshold = variance_threshold
        self.cooldown = cooldown

        self._syntony_history: List[float] = []
        self._steps_since_escape = 0
        self._escape_count = 0
        self._is_archonic = False

    def step(self, syntony: float) -> bool:
        """
        Check for archonic pattern and inject escape if needed.

        Args:
            syntony: Current model syntony

        Returns:
            True if escape was injected
        """
        # Update history
        self._syntony_history.append(syntony)
        if len(self._syntony_history) > self.detection_window:
            self._syntony_history = self._syntony_history[-self.detection_window:]

        self._steps_since_escape += 1

        # Check for archonic pattern
        if len(self._syntony_history) >= self.detection_window // 2:
            self._is_archonic = self._detect_archonic()

            if self._is_archonic and self._steps_since_escape >= self.cooldown:
                self._inject_escape_noise()
                self._steps_since_escape = 0
                self._escape_count += 1
                return True

        return False

    def _detect_archonic(self) -> bool:
        """Detect archonic cycling pattern."""
        recent = self._syntony_history[-self.detection_window // 2:]

        mean_S = sum(recent) / len(recent)
        var_S = sum((s - mean_S) ** 2 for s in recent) / len(recent)

        # Trend
        mid = len(recent) // 2
        first_half = recent[:mid]
        second_half = recent[mid:]
        trend = sum(second_half) / len(second_half) - sum(first_half) / len(first_half)

        # Archonic: cycling (high variance) without improvement
        target_S = S_TARGET - 0.1
        return (
            var_S > self.variance_threshold and
            abs(trend) < self.variance_threshold / 5 and
            mean_S < target_S
        )

    def _inject_escape_noise(self):
        """Inject noise into gradients to escape archonic pattern."""
        for param in self.model.parameters():
            if param.grad is not None:
                # Add golden-scaled noise
                noise = torch.randn_like(param.grad) * self.noise_scale
                # Scale by gradient magnitude
                grad_scale = param.grad.abs().mean() + 1e-8
                param.grad.add_(noise * grad_scale)

    @property
    def is_archonic(self) -> bool:
        """Check if currently in archonic pattern."""
        return self._is_archonic

    @property
    def escape_count(self) -> int:
        """Number of escape injections performed."""
        return self._escape_count

    def reset(self):
        """Reset escape mechanism."""
        self._syntony_history = []
        self._steps_since_escape = 0
        self._escape_count = 0
        self._is_archonic = False


class LayerwiseGradientScaling:
    """
    Scale gradients per layer using golden ratio.

    Later layers get larger gradients (more adaptation),
    earlier layers get smaller gradients (more stability).

    Example:
        >>> scaler = LayerwiseGradientScaling(model)
        >>> loss.backward()
        >>> scaler.scale()
        >>> optimizer.step()
    """

    def __init__(
        self,
        model: nn.Module,
        base_scale: float = 1.0,
        reverse: bool = False,
    ):
        """
        Initialize layerwise scaling.

        Args:
            model: Neural network
            base_scale: Base gradient scale
            reverse: If True, earlier layers get larger gradients
        """
        self.model = model
        self.base_scale = base_scale
        self.reverse = reverse

        # Compute layer structure
        self._param_scales: Dict[nn.Parameter, float] = {}
        params_list = list(model.parameters())
        n_layers = len(params_list)

        for i, param in enumerate(params_list):
            if self.reverse:
                layer_idx = n_layers - 1 - i
            else:
                layer_idx = i

            # Golden ratio scaling
            scale = base_scale * (PHI ** (layer_idx / max(1, n_layers - 1)))
            self._param_scales[param] = scale

    def scale(self):
        """Apply layerwise gradient scaling."""
        for param, scale in self._param_scales.items():
            if param.grad is not None:
                param.grad.mul_(scale)
