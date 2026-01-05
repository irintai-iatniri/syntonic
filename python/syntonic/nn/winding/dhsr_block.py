"""
Winding DHSR Block - DHSR cycle with winding structure + blockchain recording.

This module implements a complete DHSR cycle that:
1. D-phase: Fibonacci expansion (differentiation)
2. Prime filter: Möbius filtering (matter channel)
3. H-phase: Harmonization (using existing RecursionBlock)
4. Blockchain: Temporal state recording
5. Consensus: ΔS > threshold validation
"""

from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional
import math

try:
    from syntonic.nn.layers import RecursionBlock
except ImportError:
    # Fallback for testing
    from syntonic.nn.layers.recursion import RecursionBlock

from syntonic.nn.winding.prime_selection import PrimeSelectionLayer
from syntonic.nn.winding.syntony import WindingSyntonyComputer

PHI = (1 + math.sqrt(5)) / 2  # Golden ratio ~ 1.618
PHI_INV = 1 / PHI  # ~ 0.618
PHI_INV_SQ = PHI_INV - PHI_INV * PHI_INV  # ~ 0.382


class WindingDHSRBlock(nn.Module):
    """
    DHSR block with winding structure and blockchain recording.

    Architecture:
    1. D-phase: Fibonacci expansion via linear layers
    2. Prime filter: Möbius selection (optional)
    3. H-phase: RecursionBlock for DHSR processing
    4. Syntony: WindingSyntonyComputer
    5. Blockchain: Temporal ledger of accepted states

    Example:
        >>> block = WindingDHSRBlock(dim=64, fib_expand_factor=3)
        >>> x = torch.randn(32, 64)
        >>> mode_norms = torch.arange(64).pow(2).float()
        >>> prev_syntony = 0.5
        >>> x_new, syntony_new, accepted = block(x, mode_norms, prev_syntony)
    """

    def __init__(
        self,
        dim: int,
        fib_expand_factor: int = 3,
        use_prime_filter: bool = True,
        consensus_threshold: float = 0.024,
        dropout: float = 0.1,
    ):
        """
        Initialize winding DHSR block.

        Args:
            dim: Feature dimension
            fib_expand_factor: Fibonacci expansion factor for D-phase
            use_prime_filter: Whether to use prime selection filtering
            consensus_threshold: ΔS threshold for block validation
            dropout: Dropout rate
        """
        super().__init__()

        self.dim = dim
        self.fib_expand_factor = fib_expand_factor
        self.consensus_threshold = consensus_threshold

        # Fibonacci expansion (D-phase)
        expanded_dim = dim * fib_expand_factor
        self.d_expand = nn.Linear(dim, expanded_dim)
        self.d_project = nn.Linear(expanded_dim, dim)

        # Prime filter (matter channel)
        if use_prime_filter:
            self.prime_filter = PrimeSelectionLayer(dim)
        else:
            self.prime_filter = nn.Identity()

        # H-phase: Use existing RecursionBlock for DHSR processing
        self.recursion = RecursionBlock(
            in_features=dim,
            use_gate=True,
            dropout=dropout,
        )

        # Syntony computer
        self.syntony_computer = WindingSyntonyComputer(dim)

        # Temporal blockchain (immutable ledger)
        self.register_buffer("temporal_record", torch.zeros(0, dim))
        self.register_buffer("syntony_record", torch.zeros(0))

        # Statistics
        self.total_blocks_validated = 0
        self.total_blocks_rejected = 0

    def forward(
        self,
        x: torch.Tensor,
        mode_norms: torch.Tensor,
        prev_syntony: float,
    ) -> Tuple[torch.Tensor, float, bool]:
        """
        Execute one DHSR cycle with blockchain recording.

        Args:
            x: Input activations (batch, dim)
            mode_norms: Mode norm squared |n|² for each feature (dim,)
            prev_syntony: Syntony from previous block

        Returns:
            x_new: Updated state (batch, dim)
            syntony_new: New syntony value
            accepted: Whether ΔS > threshold (block validated)
        """
        # === D-PHASE: Fibonacci expansion ===
        # Amplification inversely proportional to syntony
        alpha = PHI_INV_SQ * (1.0 - prev_syntony)

        # Expand via Fibonacci factor
        h = F.relu(self.d_expand(x))
        delta = self.d_project(h)

        # Apply differentiation: x' = x + α·Δ
        x = x + alpha * delta

        # === PRIME FILTER: Matter selection ===
        x = self.prime_filter(x)

        # === H-PHASE: Harmonization via RecursionBlock ===
        # The RecursionBlock handles D̂ ∘ Ĥ internally
        x, block_syntony = self.recursion(x, return_syntony=True)

        # === SYNTONY COMPUTATION: Winding-aware ===
        # Use winding syntony computer with mode norms
        syntony_new = self.syntony_computer(x, mode_norms)

        # Combine block syntony and winding syntony
        # Use weighted average: 70% winding, 30% block
        syntony_new = 0.7 * syntony_new + 0.3 * block_syntony

        # === CONSENSUS CHECK: ΔS > threshold ===
        delta_s = abs(syntony_new - prev_syntony)
        accepted = delta_s > self.consensus_threshold

        # === TEMPORAL RECORDING: Blockchain ===
        if accepted:
            self._record_block(x.detach(), syntony_new)
            self.total_blocks_validated += 1
        else:
            self.total_blocks_rejected += 1

        return x, syntony_new, accepted

    def _record_block(self, state: torch.Tensor, syntony: float):
        """
        Append block to temporal blockchain.

        Immutable, append-only ledger recording state evolution.

        Args:
            state: State tensor (batch, dim)
            syntony: Syntony value
        """
        # Average over batch for single state record
        state_avg = state.mean(dim=0, keepdim=True)  # (1, dim)

        # Concatenate to blockchain
        self.temporal_record = torch.cat([self.temporal_record, state_avg], dim=0)

        # Record syntony
        self.syntony_record = torch.cat([
            self.syntony_record,
            torch.tensor([syntony], device=self.syntony_record.device)
        ], dim=0)

    def get_blockchain_length(self) -> int:
        """
        Get length of temporal blockchain.

        Returns:
            Number of recorded blocks
        """
        return len(self.syntony_record)

    def get_blockchain(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get complete blockchain history.

        Returns:
            states: Temporal state record (blockchain_length, dim)
            syntonies: Syntony record (blockchain_length,)
        """
        return self.temporal_record, self.syntony_record

    def get_validation_rate(self) -> float:
        """
        Get consensus validation rate.

        Returns:
            Fraction of blocks validated (accepted / total)
        """
        total = self.total_blocks_validated + self.total_blocks_rejected
        if total == 0:
            return 0.0
        return self.total_blocks_validated / total

    def extra_repr(self) -> str:
        return (
            f"dim={self.dim}, expand_factor={self.fib_expand_factor}, "
            f"threshold={self.consensus_threshold:.4f}"
        )


if __name__ == "__main__":
    # Example usage
    block = WindingDHSRBlock(dim=64, fib_expand_factor=3)
    print(f"DHSR Block: {block}")

    # Create sample data
    x = torch.randn(32, 64)
    mode_norms = torch.arange(64).pow(2).float()
    prev_syntony = 0.5

    # Forward pass
    x_new, syntony_new, accepted = block(x, mode_norms, prev_syntony)

    print(f"\nInput shape: {x.shape}")
    print(f"Output shape: {x_new.shape}")
    print(f"Previous syntony: {prev_syntony:.4f}")
    print(f"New syntony: {syntony_new:.4f}")
    print(f"ΔS: {abs(syntony_new - prev_syntony):.4f}")
    print(f"Block accepted: {accepted}")
    print(f"Blockchain length: {block.get_blockchain_length()}")
    print(f"Validation rate: {block.get_validation_rate():.2%}")

    # Run a few more cycles
    for i in range(5):
        x_new, syntony_new, accepted = block(x_new, mode_norms, syntony_new)
        print(f"Cycle {i+2}: S={syntony_new:.4f}, accepted={accepted}")

    print(f"\nFinal blockchain length: {block.get_blockchain_length()}")
    print(f"Final validation rate: {block.get_validation_rate():.2%}")
