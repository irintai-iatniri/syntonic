"""
Syntonic Loss Functions.

L_total = L_task + λ_syntony(1 - S_model) + μ_{iπ}·C_{iπ}

Where:
- L_task: Standard task loss (CrossEntropy, MSE, etc.)
- S_model: Model syntony (coherence measure)
- C_{iπ}: Phase-cycle alignment (i ≃ π constraint)

Source: CRT.md §12.2
"""

from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Any, Tuple
import math

PHI = (1 + math.sqrt(5)) / 2
Q_DEFICIT = 0.027395146920


class SyntonicLoss(nn.Module):
    """
    Syntonic loss function combining task and syntony objectives.

    L_total = L_task + λ_syntony·(1 - S_model) + μ_{iπ}·C_{iπ}

    This loss encourages networks to:
    1. Perform the task well (L_task)
    2. Maintain high syntony representations (L_syntony)
    3. Align with i ≃ π phase structure (L_phase)

    Example:
        >>> loss_fn = SyntonicLoss(nn.CrossEntropyLoss(), lambda_syntony=0.1)
        >>> outputs = model(inputs)
        >>> loss, metrics = loss_fn(outputs, targets, model, inputs)
    """

    def __init__(
        self,
        task_loss: nn.Module,
        lambda_syntony: float = 0.1,
        mu_phase: float = 0.01,
        syntony_target: Optional[float] = None,
    ):
        """
        Initialize syntonic loss.

        Args:
            task_loss: Base task loss function (e.g., CrossEntropyLoss)
            lambda_syntony: Weight for syntony term
            mu_phase: Weight for phase alignment term
            syntony_target: Target syntony (default: φ - q ≈ 1.591)
        """
        super().__init__()
        self.task_loss = task_loss
        self.lambda_syntony = lambda_syntony
        self.mu_phase = mu_phase
        self.syntony_target = syntony_target if syntony_target is not None else (PHI - Q_DEFICIT)

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        model: nn.Module,
        inputs: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute total syntonic loss.

        Args:
            pred: Model predictions
            target: Ground truth targets
            model: The model (for syntony computation)
            inputs: Original inputs (for S_model computation)

        Returns:
            (total_loss, metrics_dict)
        """
        # Task loss
        L_task = self.task_loss(pred, target)

        # Syntony loss
        S_model = self._compute_model_syntony(model, inputs, pred)
        L_syntony = self.lambda_syntony * (1.0 - S_model)

        # Phase alignment loss
        C_phase = self._compute_phase_alignment(model, pred)
        L_phase = self.mu_phase * C_phase

        # Total
        L_total = L_task + L_syntony + L_phase

        metrics = {
            'loss_task': L_task.item(),
            'loss_syntony': L_syntony if isinstance(L_syntony, float) else L_syntony.item() if hasattr(L_syntony, 'item') else L_syntony,
            'loss_phase': L_phase if isinstance(L_phase, float) else L_phase.item() if hasattr(L_phase, 'item') else L_phase,
            'loss_total': L_total.item(),
            'syntony': S_model,
            'phase_alignment': 1.0 - C_phase,
        }

        return L_total, metrics

    def _compute_model_syntony(
        self,
        model: nn.Module,
        inputs: Optional[torch.Tensor],
        outputs: torch.Tensor,
    ) -> float:
        """
        Compute model syntony S_model.

        S_model ≈ 1 - |D(x) - x| / |D(x) - H(D(x))|

        Aggregated across all RecursionBlocks in the model.
        """
        syntonies = []

        # Collect syntony from RecursionBlocks
        for module in model.modules():
            if hasattr(module, 'syntony') and module.syntony is not None:
                syntonies.append(module.syntony)

        if not syntonies:
            # Fallback: estimate from output statistics
            return self._estimate_syntony_from_output(outputs)

        return sum(syntonies) / len(syntonies)

    def _estimate_syntony_from_output(self, outputs: torch.Tensor) -> float:
        """
        Estimate syntony from output statistics.

        High syntony → well-structured, coherent outputs
        Low syntony → chaotic, fragmented outputs
        """
        with torch.no_grad():
            # Use entropy as inverse syntony proxy
            if outputs.dim() > 1 and outputs.shape[-1] > 1:
                probs = F.softmax(outputs, dim=-1)
                entropy = -torch.sum(probs * torch.log(probs + 1e-8), dim=-1)
                max_entropy = math.log(outputs.shape[-1])
                normalized_entropy = entropy.mean().item() / max_entropy
                return 1.0 - normalized_entropy
            else:
                # Use variance for regression
                var = torch.var(outputs).item()
                return 1.0 / (1.0 + var)

    def _compute_phase_alignment(
        self,
        model: nn.Module,
        outputs: torch.Tensor,
    ) -> float:
        """
        Compute phase-cycle alignment C_{iπ}.

        C_{iπ} = |Arg Tr[e^{iπρ_model}] - π/2|²

        Measures how well the model's density matrix aligns with i ≃ π.
        """
        with torch.no_grad():
            # Approximate: use output correlation structure
            if outputs.dim() == 2 and outputs.shape[0] > 1 and outputs.shape[1] > 1:
                # Compute pseudo-density matrix from outputs
                outputs_norm = F.normalize(outputs, dim=-1)
                rho = torch.mm(outputs_norm.T, outputs_norm) / outputs.shape[0]

                # Make symmetric for eigenvalue computation
                rho = (rho + rho.T) / 2

                # Eigenvalues for phase analysis
                try:
                    eigvals = torch.linalg.eigvalsh(rho)
                    # Ensure positive
                    eigvals = torch.clamp(eigvals, min=1e-10)
                    eigvals = eigvals / eigvals.sum()

                    # Phase alignment: deviation from balanced spectrum
                    spectral_entropy = -torch.sum(
                        eigvals * torch.log(eigvals + 1e-8)
                    ).item()
                    target_entropy = math.log(rho.shape[0])
                    alignment = abs(spectral_entropy - target_entropy / 2)
                    return min(1.0, alignment / target_entropy)
                except Exception:
                    return 0.0
            return 0.0


class LayerwiseSyntonicLoss(SyntonicLoss):
    """
    Syntonic loss with layer-wise syntony tracking.

    Applies syntony regularization at each layer, not just globally.
    Uses golden-ratio weighting (later layers matter more).

    Example:
        >>> loss_fn = LayerwiseSyntonicLoss(nn.CrossEntropyLoss())
        >>> loss, metrics = loss_fn(outputs, targets, model)
    """

    def __init__(
        self,
        task_loss: nn.Module,
        lambda_syntony: float = 0.1,
        mu_phase: float = 0.01,
        layer_weights: Optional[list] = None,
    ):
        """
        Initialize layerwise syntonic loss.

        Args:
            task_loss: Base task loss
            lambda_syntony: Syntony weight
            mu_phase: Phase alignment weight
            layer_weights: Optional custom weights per layer
        """
        super().__init__(task_loss, lambda_syntony, mu_phase)
        self.layer_weights = layer_weights

    def _compute_model_syntony(
        self,
        model: nn.Module,
        inputs: Optional[torch.Tensor],
        outputs: torch.Tensor,
    ) -> float:
        """Compute weighted layer-wise syntony."""
        syntonies = []

        for module in model.modules():
            if hasattr(module, 'syntony') and module.syntony is not None:
                syntonies.append(module.syntony)

        if not syntonies:
            return self._estimate_syntony_from_output(outputs)

        # Apply weights (default: golden ratio weighting)
        if self.layer_weights is None:
            # Golden ratio weighting: later layers matter more
            weights = [PHI ** i for i in range(len(syntonies))]
        else:
            weights = self.layer_weights[:len(syntonies)]

        total_weight = sum(weights)
        weighted_syntony = sum(w * s for w, s in zip(weights, syntonies))

        return weighted_syntony / total_weight
