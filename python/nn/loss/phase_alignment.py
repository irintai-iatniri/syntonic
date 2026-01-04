"""
Phase Alignment Loss: i ≃ π constraint in neural networks.

C_{iπ} = |Arg Tr[e^{iπρ}] - π/2|²

The i ≃ π isomorphism creates the phase structure that enables
coherent information processing.

Source: CRT.md §12.2, §5.3
"""

from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
import math

PHI = (1 + math.sqrt(5)) / 2
Q_DEFICIT = 0.027395146920


def compute_phase_alignment(
    outputs: torch.Tensor,
    target_phase: float = math.pi / 2,
    method: str = 'spectral',
) -> torch.Tensor:
    """
    Compute phase alignment C_{iπ}.

    C_{iπ} = |Arg Tr[e^{iπρ}] - π/2|²

    Args:
        outputs: Network outputs (batch_size, features)
        target_phase: Target phase (default: π/2)
        method: Computation method ('spectral', 'correlation', 'fft')

    Returns:
        Phase alignment cost (lower is better)
    """
    if method == 'spectral':
        return _phase_alignment_spectral(outputs, target_phase)
    elif method == 'correlation':
        return _phase_alignment_correlation(outputs, target_phase)
    elif method == 'fft':
        return _phase_alignment_fft(outputs, target_phase)
    else:
        return _phase_alignment_spectral(outputs, target_phase)


def _phase_alignment_spectral(
    outputs: torch.Tensor,
    target_phase: float,
) -> torch.Tensor:
    """
    Spectral method for phase alignment.

    Constructs pseudo-density matrix from outputs and
    computes phase deviation from target.
    """
    if outputs.dim() < 2 or outputs.shape[0] < 2 or outputs.shape[1] < 2:
        return torch.tensor(0.0, device=outputs.device)

    # Normalize outputs
    outputs_norm = F.normalize(outputs, dim=-1)

    # Pseudo-density matrix: ρ = X^T X / N
    rho = torch.mm(outputs_norm.T, outputs_norm) / outputs.shape[0]

    # Make Hermitian (symmetric for real case)
    rho = (rho + rho.T) / 2

    try:
        # Eigenvalue decomposition
        eigvals = torch.linalg.eigvalsh(rho)

        # Ensure positive (for valid density matrix interpretation)
        eigvals = torch.clamp(eigvals, min=1e-10)
        eigvals = eigvals / eigvals.sum()

        # Phase from eigenvalue spectrum
        # For balanced spectrum: phase = π/2
        # For concentrated spectrum: phase → 0 or π

        # Spectral entropy as phase proxy
        spectral_entropy = -torch.sum(eigvals * torch.log(eigvals + 1e-10))
        max_entropy = math.log(rho.shape[0])

        # Map entropy to phase: S=0 → θ=0, S=max → θ=π/2
        estimated_phase = (spectral_entropy / max_entropy) * (math.pi / 2)

        # Phase deviation squared
        phase_deviation = (estimated_phase - target_phase) ** 2

        return phase_deviation

    except Exception:
        return torch.tensor(0.0, device=outputs.device)


def _phase_alignment_correlation(
    outputs: torch.Tensor,
    target_phase: float,
) -> torch.Tensor:
    """
    Correlation-based phase alignment.

    Uses cross-correlation structure to estimate phase.
    """
    if outputs.dim() < 2 or outputs.shape[0] < 2:
        return torch.tensor(0.0, device=outputs.device)

    # Compute correlation matrix
    mean = outputs.mean(dim=0, keepdim=True)
    centered = outputs - mean
    std = centered.std(dim=0, keepdim=True) + 1e-8
    normalized = centered / std

    corr = torch.mm(normalized.T, normalized) / outputs.shape[0]

    # Off-diagonal elements indicate phase coherence
    n = corr.shape[0]
    off_diag_mask = ~torch.eye(n, dtype=torch.bool, device=corr.device)
    off_diag = corr[off_diag_mask]

    # Mean absolute correlation → phase
    # High correlation → low phase deviation
    mean_corr = off_diag.abs().mean()

    # Map: corr=0 → phase=π/2 (random), corr=1 → phase=0 (aligned)
    estimated_phase = (1 - mean_corr) * (math.pi / 2)
    phase_deviation = (estimated_phase - target_phase) ** 2

    return phase_deviation


def _phase_alignment_fft(
    outputs: torch.Tensor,
    target_phase: float,
) -> torch.Tensor:
    """
    FFT-based phase alignment.

    Analyzes phase structure in frequency domain.
    """
    if outputs.dim() < 2 or outputs.shape[1] < 4:
        return torch.tensor(0.0, device=outputs.device)

    # FFT along feature dimension
    fft = torch.fft.fft(outputs, dim=-1)

    # Phase angles
    phases = torch.angle(fft)

    # Mean phase (excluding DC component)
    mean_phase = phases[:, 1:].mean()

    # Wrap to [0, π]
    mean_phase = torch.abs(mean_phase % math.pi)

    phase_deviation = (mean_phase - target_phase) ** 2

    return phase_deviation


class PhaseAlignmentLoss(nn.Module):
    """
    Loss function for i ≃ π phase alignment.

    Encourages network representations to maintain the
    phase structure required by the i ≃ π isomorphism.

    L_phase = μ · C_{iπ}

    Where C_{iπ} measures deviation from target phase structure.

    Example:
        >>> phase_loss = PhaseAlignmentLoss(mu=0.01)
        >>> outputs = model(inputs)
        >>> loss = phase_loss(outputs)
    """

    def __init__(
        self,
        mu: float = 0.01,
        target_phase: float = math.pi / 2,
        method: str = 'spectral',
    ):
        """
        Initialize phase alignment loss.

        Args:
            mu: Loss weight
            target_phase: Target phase (default: π/2)
            method: Computation method ('spectral', 'correlation', 'fft')
        """
        super().__init__()
        self.mu = mu
        self.target_phase = target_phase
        self.method = method

    def forward(self, outputs: torch.Tensor) -> torch.Tensor:
        """
        Compute phase alignment loss.

        Args:
            outputs: Network outputs

        Returns:
            Weighted phase alignment loss
        """
        C_phase = compute_phase_alignment(
            outputs, self.target_phase, self.method
        )
        return self.mu * C_phase

    def extra_repr(self) -> str:
        return f'mu={self.mu}, target_phase={self.target_phase:.4f}, method={self.method}'


class IPiConstraint(nn.Module):
    """
    Soft constraint enforcing i ≃ π in representations.

    The i ≃ π isomorphism (CRT.md §5.3) states that:
    - Imaginary unit i emerges from π-rotation
    - e^{iπ} = -1 (Euler's identity) is structural

    This constraint encourages representations where
    half-period rotations map to sign flips.

    Example:
        >>> constraint = IPiConstraint()
        >>> outputs = model(inputs)
        >>> loss = constraint(outputs)
    """

    def __init__(
        self,
        weight: float = 0.01,
        temperature: float = 1.0,
    ):
        """
        Initialize i ≃ π constraint.

        Args:
            weight: Constraint weight in loss
            temperature: Softness of constraint
        """
        super().__init__()
        self.weight = weight
        self.temperature = temperature

    def forward(self, outputs: torch.Tensor) -> torch.Tensor:
        """
        Compute i ≃ π constraint violation.

        Measures how well the representation satisfies
        the half-period rotation property.
        """
        if outputs.dim() < 2 or outputs.shape[1] < 2:
            return torch.tensor(0.0, device=outputs.device)

        # Split features into even/odd pairs (representing real/imaginary)
        n_features = outputs.shape[1]
        n_pairs = n_features // 2

        # "Real" and "imaginary" parts
        real_part = outputs[:, :n_pairs]
        imag_part = outputs[:, n_pairs:2*n_pairs]

        # i ≃ π means: rotation by π flips sign
        # For complex z = x + iy: z·e^{iπ} = -z = -x - iy
        # Check if structure supports this

        # Compute pseudo-rotation
        # If i acts like π-rotation, then applying i twice gives -1
        # |i²z + z|² should be small

        # Use correlation between real and imag as proxy
        # Perfect i ≃ π: real and imag are orthogonal with specific phase

        # Normalize
        real_norm = F.normalize(real_part, dim=-1)
        imag_norm = F.normalize(imag_part, dim=-1)

        # Dot product should be 0 (orthogonality)
        dot = (real_norm * imag_norm).sum(dim=-1)

        # Loss is deviation from orthogonality
        constraint_violation = dot.pow(2).mean()

        return self.weight * constraint_violation / self.temperature

    def extra_repr(self) -> str:
        return f'weight={self.weight}, temperature={self.temperature}'


class GoldenPhaseScheduler:
    """
    Schedule target phase using golden ratio.

    Gradually shifts target phase following golden spiral
    for stable training dynamics.

    Example:
        >>> scheduler = GoldenPhaseScheduler(phase_loss, n_epochs=100)
        >>> for epoch in range(100):
        ...     train_one_epoch()
        ...     scheduler.step()
    """

    def __init__(
        self,
        phase_loss: PhaseAlignmentLoss,
        n_epochs: int = 100,
        initial_phase: float = 0.0,
        final_phase: float = math.pi / 2,
    ):
        """
        Initialize scheduler.

        Args:
            phase_loss: PhaseAlignmentLoss to schedule
            n_epochs: Total training epochs
            initial_phase: Starting target phase
            final_phase: Ending target phase
        """
        self.phase_loss = phase_loss
        self.n_epochs = n_epochs
        self.initial_phase = initial_phase
        self.final_phase = final_phase
        self.current_epoch = 0

    def step(self):
        """Update target phase for next epoch."""
        self.current_epoch += 1

        # Golden ratio interpolation
        t = self.current_epoch / self.n_epochs
        # Use golden spiral: faster at start, slower at end
        t_golden = 1 - (1 - t) ** PHI

        new_phase = (
            self.initial_phase +
            (self.final_phase - self.initial_phase) * t_golden
        )
        self.phase_loss.target_phase = new_phase

    def get_phase(self) -> float:
        """Get current target phase."""
        return self.phase_loss.target_phase
