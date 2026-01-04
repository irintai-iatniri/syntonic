"""
Syntonic Regularization: Golden-ratio based weight decay and constraints.

Regularization that promotes syntonic structure in neural networks:
- GoldenDecay: Weight decay with φ-based scaling
- SyntonicRegularizer: Combined regularization for syntony

Source: CRT.md §12.2
"""

from __future__ import annotations
import torch
import torch.nn as nn
from typing import Optional, List, Dict, Iterator, Tuple
import math

PHI = (1 + math.sqrt(5)) / 2
PHI_INV = 1 / PHI
Q_DEFICIT = 0.027395146920


class GoldenDecay(nn.Module):
    """
    Golden ratio-based weight decay.

    Instead of uniform L2 decay, applies φ-scaled decay
    that preserves syntonic weight structure.

    L_decay = λ Σ_l (φ^{-l}) ||W_l||²

    Earlier layers decay faster (more regularization),
    later layers decay slower (more expressivity).

    Example:
        >>> decay = GoldenDecay(model, lambda_decay=0.01)
        >>> loss = task_loss + decay()
    """

    def __init__(
        self,
        model: nn.Module,
        lambda_decay: float = 0.01,
        reverse: bool = False,
        min_decay: float = 0.001,
    ):
        """
        Initialize golden decay.

        Args:
            model: Neural network
            lambda_decay: Base decay rate
            reverse: If True, later layers decay faster
            min_decay: Minimum decay rate
        """
        super().__init__()
        self.model = model
        self.lambda_decay = lambda_decay
        self.reverse = reverse
        self.min_decay = min_decay

        # Count layers with parameters
        self.n_layers = sum(1 for p in model.parameters() if p.requires_grad)

    def forward(self) -> torch.Tensor:
        """Compute golden decay loss."""
        decay_loss = torch.tensor(0.0)
        device = None

        for i, param in enumerate(self.model.parameters()):
            if not param.requires_grad:
                continue

            if device is None:
                device = param.device
                decay_loss = decay_loss.to(device)

            # Golden ratio scaling
            if self.reverse:
                layer_idx = self.n_layers - 1 - i
            else:
                layer_idx = i

            # φ^{-l} scaling
            scale = max(PHI ** (-layer_idx), self.min_decay / self.lambda_decay)

            # L2 norm
            decay_loss = decay_loss + scale * param.pow(2).sum()

        return self.lambda_decay * decay_loss

    def extra_repr(self) -> str:
        return f'lambda_decay={self.lambda_decay}, n_layers={self.n_layers}, reverse={self.reverse}'


class SyntonicRegularizer(nn.Module):
    """
    Combined syntonic regularization.

    Applies multiple regularization terms that promote syntony:
    1. Golden weight decay
    2. Activation sparsity (promotes differentiation)
    3. Weight coherence (promotes harmonization)
    4. Spectral norm constraint

    Example:
        >>> reg = SyntonicRegularizer(model)
        >>> loss = task_loss + reg(activations)
    """

    def __init__(
        self,
        model: nn.Module,
        lambda_decay: float = 0.01,
        lambda_sparsity: float = 0.001,
        lambda_coherence: float = 0.001,
        lambda_spectral: float = 0.0,
    ):
        """
        Initialize syntonic regularizer.

        Args:
            model: Neural network
            lambda_decay: Weight decay strength
            lambda_sparsity: Activation sparsity strength
            lambda_coherence: Weight coherence strength
            lambda_spectral: Spectral norm constraint strength
        """
        super().__init__()
        self.model = model
        self.lambda_decay = lambda_decay
        self.lambda_sparsity = lambda_sparsity
        self.lambda_coherence = lambda_coherence
        self.lambda_spectral = lambda_spectral

        self.golden_decay = GoldenDecay(model, lambda_decay)

    def forward(
        self,
        activations: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute total regularization loss.

        Args:
            activations: Optional intermediate activations for sparsity

        Returns:
            (total_reg_loss, metrics_dict)
        """
        device = next(self.model.parameters()).device
        total_loss = torch.tensor(0.0, device=device)
        metrics = {}

        # 1. Golden weight decay
        if self.lambda_decay > 0:
            decay_loss = self.golden_decay()
            total_loss = total_loss + decay_loss
            metrics['reg_decay'] = decay_loss.item()

        # 2. Activation sparsity
        if self.lambda_sparsity > 0 and activations is not None:
            sparsity_loss = self._compute_sparsity(activations)
            total_loss = total_loss + self.lambda_sparsity * sparsity_loss
            metrics['reg_sparsity'] = (self.lambda_sparsity * sparsity_loss).item()

        # 3. Weight coherence
        if self.lambda_coherence > 0:
            coherence_loss = self._compute_coherence()
            total_loss = total_loss + self.lambda_coherence * coherence_loss
            metrics['reg_coherence'] = (self.lambda_coherence * coherence_loss).item()

        # 4. Spectral norm constraint
        if self.lambda_spectral > 0:
            spectral_loss = self._compute_spectral()
            total_loss = total_loss + self.lambda_spectral * spectral_loss
            metrics['reg_spectral'] = (self.lambda_spectral * spectral_loss).item()

        metrics['reg_total'] = total_loss.item()
        return total_loss, metrics

    def _compute_sparsity(self, activations: torch.Tensor) -> torch.Tensor:
        """
        Compute activation sparsity regularization.

        Promotes golden-ratio sparsity: ~61.8% zeros.
        """
        # L1 norm encourages sparsity
        l1_norm = activations.abs().mean()

        # Target sparsity is 1/φ ≈ 0.618
        target_sparsity = PHI_INV

        # Current sparsity (fraction of near-zero activations)
        current_sparsity = (activations.abs() < 0.01).float().mean()

        # Penalize deviation from golden sparsity
        sparsity_penalty = (current_sparsity - target_sparsity).pow(2)

        return l1_norm + sparsity_penalty

    def _compute_coherence(self) -> torch.Tensor:
        """
        Compute weight coherence regularization.

        Encourages weight matrices to have coherent structure
        (low condition number, balanced singular values).
        """
        coherence_loss = torch.tensor(0.0)
        device = None
        n_matrices = 0

        for name, param in self.model.named_parameters():
            if param.dim() < 2:
                continue

            if device is None:
                device = param.device
                coherence_loss = coherence_loss.to(device)

            # Reshape to 2D for SVD
            weight = param.view(param.shape[0], -1)

            # Use Frobenius norm ratio as coherence proxy
            # (SVD is expensive, this approximates condition number)
            fro_norm = weight.norm('fro')
            max_norm = weight.abs().max()

            # Coherent weights: Frobenius ~ √(m*n) * max
            # Incoherent: Frobenius >> √(m*n) * max
            expected_fro = max_norm * math.sqrt(weight.numel())
            coherence_loss = coherence_loss + (fro_norm / (expected_fro + 1e-8) - 1).pow(2)
            n_matrices += 1

        if n_matrices > 0:
            coherence_loss = coherence_loss / n_matrices

        return coherence_loss

    def _compute_spectral(self) -> torch.Tensor:
        """
        Compute spectral norm constraint.

        Constrains largest singular value to be bounded,
        promoting stable gradients.
        """
        spectral_loss = torch.tensor(0.0)
        device = None
        n_matrices = 0

        for name, param in self.model.named_parameters():
            if param.dim() < 2:
                continue

            if device is None:
                device = param.device
                spectral_loss = spectral_loss.to(device)

            # Reshape to 2D
            weight = param.view(param.shape[0], -1)

            # Power iteration approximation of spectral norm
            # (faster than full SVD)
            with torch.no_grad():
                u = torch.randn(weight.shape[0], device=device)
                u = u / u.norm()

                for _ in range(3):  # Few iterations suffice
                    v = weight.T @ u
                    v = v / (v.norm() + 1e-8)
                    u = weight @ v
                    u = u / (u.norm() + 1e-8)

            sigma_max = (u @ weight @ v).abs()

            # Penalize if spectral norm exceeds golden ratio
            if sigma_max > PHI:
                spectral_loss = spectral_loss + (sigma_max - PHI).pow(2)

            n_matrices += 1

        if n_matrices > 0:
            spectral_loss = spectral_loss / n_matrices

        return spectral_loss


class SyntonyConstraint(nn.Module):
    """
    Soft constraint maintaining syntony above threshold.

    Penalizes syntony values below the target S* = φ - q.

    Example:
        >>> constraint = SyntonyConstraint(target=1.59)
        >>> syntony = model.compute_syntony()
        >>> loss = task_loss + constraint(syntony)
    """

    def __init__(
        self,
        target: Optional[float] = None,
        margin: float = 0.1,
        weight: float = 1.0,
    ):
        """
        Initialize syntony constraint.

        Args:
            target: Target syntony (default: φ - q ≈ 1.591)
            margin: Soft margin around target
            weight: Constraint weight
        """
        super().__init__()
        self.target = target if target is not None else (PHI - Q_DEFICIT)
        self.margin = margin
        self.weight = weight

    def forward(self, syntony: float) -> torch.Tensor:
        """
        Compute syntony constraint violation.

        Args:
            syntony: Current model syntony

        Returns:
            Constraint violation loss
        """
        if syntony >= self.target - self.margin:
            return torch.tensor(0.0)

        # Quadratic penalty for syntony below target
        violation = self.target - self.margin - syntony
        return self.weight * (violation ** 2)


class ArchonicPenalty(nn.Module):
    """
    Penalty for archonic (stuck) patterns.

    Detects and penalizes representations that exhibit
    archonic cycling (high variance, no syntony improvement).

    Example:
        >>> penalty = ArchonicPenalty()
        >>> for batch in dataloader:
        ...     outputs = model(batch)
        ...     arch_loss = penalty(outputs, model.syntony)
    """

    def __init__(
        self,
        weight: float = 0.1,
        history_size: int = 100,
        variance_threshold: float = 0.01,
    ):
        """
        Initialize archonic penalty.

        Args:
            weight: Penalty weight
            history_size: Size of syntony history window
            variance_threshold: Threshold for detecting cycling
        """
        super().__init__()
        self.weight = weight
        self.history_size = history_size
        self.variance_threshold = variance_threshold
        self.syntony_history: List[float] = []

    def forward(
        self,
        outputs: torch.Tensor,
        current_syntony: float,
    ) -> torch.Tensor:
        """
        Compute archonic penalty.

        Args:
            outputs: Network outputs
            current_syntony: Current model syntony

        Returns:
            Archonic penalty loss
        """
        # Update history
        self.syntony_history.append(current_syntony)
        if len(self.syntony_history) > self.history_size:
            self.syntony_history = self.syntony_history[-self.history_size:]

        if len(self.syntony_history) < 10:
            return torch.tensor(0.0, device=outputs.device)

        # Check for archonic pattern
        recent = self.syntony_history[-50:] if len(self.syntony_history) >= 50 else self.syntony_history
        mean_S = sum(recent) / len(recent)
        var_S = sum((s - mean_S) ** 2 for s in recent) / len(recent)

        # Trend (last half vs first half)
        mid = len(recent) // 2
        first_half = recent[:mid]
        second_half = recent[mid:]
        trend = sum(second_half) / len(second_half) - sum(first_half) / len(first_half)

        # Archonic: high variance, no trend, below target
        target_S = PHI - Q_DEFICIT - 0.1
        is_archonic = (
            var_S > self.variance_threshold and
            abs(trend) < self.variance_threshold / 10 and
            mean_S < target_S
        )

        if is_archonic:
            # Penalize based on how archonic
            archonic_score = var_S * (target_S - mean_S) / (abs(trend) + 1e-8)
            return self.weight * torch.tensor(archonic_score, device=outputs.device)

        return torch.tensor(0.0, device=outputs.device)

    def reset(self):
        """Reset history."""
        self.syntony_history = []
