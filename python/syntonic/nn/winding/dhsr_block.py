"""
Winding DHSR Block - DHSR cycle with winding structure + blockchain recording.

This module implements a complete DHSR cycle using the Rust backend:
1. D-phase: Fibonacci expansion (differentiation)
2. H-phase: Harmonization (lattice crystallization)
3. S-phase: Syntony computation
4. R-phase: Recursion (golden scaling)
5. Blockchain: Temporal state recording

NO PYTORCH OR NUMPY DEPENDENCIES.
"""

from __future__ import annotations
import math
import random
from typing import Tuple, List, Optional

from syntonic._core import ResonantTensor
from syntonic.nn.layers.resonant_linear import ResonantLinear

PHI = (1 + math.sqrt(5)) / 2
PHI_INV = 1 / PHI
PHI_INV_SQ = PHI_INV - PHI_INV * PHI_INV


class WindingDHSRBlock:
    """
    DHSR block with winding structure and blockchain recording.

    Architecture:
    1. D-phase: Fibonacci expansion via ResonantLinear
    2. H-phase: Crystallization via cpu_cycle
    3. S-phase: Syntony extracted from ResonantTensor
    4. R-phase: Golden recursion scaling
    5. Blockchain: Temporal ledger of accepted states
    """

    def __init__(
        self,
        dim: int,
        fib_expand_factor: int = 3,
        consensus_threshold: float = 0.024,
        precision: int = 100,
    ):
        """
        Initialize winding DHSR block.

        Args:
            dim: Feature dimension
            fib_expand_factor: Fibonacci expansion factor for D-phase
            consensus_threshold: ΔS threshold for block validation
            precision: Lattice precision for exact arithmetic
        """
        self.dim = dim
        self.fib_expand_factor = fib_expand_factor
        self.consensus_threshold = consensus_threshold
        self.precision = precision

        # Fibonacci expansion (D-phase)
        expanded_dim = dim * fib_expand_factor
        self.d_expand = ResonantLinear(dim, expanded_dim, precision=precision)
        self.d_project = ResonantLinear(expanded_dim, dim, precision=precision)

        # Temporal blockchain (immutable ledger)
        self.temporal_record: List[List[float]] = []
        self.syntony_record: List[float] = []

        # Statistics
        self.total_blocks_validated = 0
        self.total_blocks_rejected = 0

    def forward(
        self,
        x: ResonantTensor,
        prev_syntony: float,
    ) -> Tuple[ResonantTensor, float, bool]:
        """
        Execute one DHSR cycle with blockchain recording.

        Args:
            x: Input activations (batch, dim)
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
        h = self.d_expand.forward(x)
        h.relu()  # Activation
        
        delta = self.d_project.forward(h)

        # Apply differentiation: x' = x + α·Δ
        # Combine x and delta (simplified for now - add floats and reconstruct)
        x_floats = x.to_floats()
        delta_floats = delta.to_floats()
        combined = [x_floats[i] + alpha * delta_floats[i] for i in range(len(x_floats))]
        
        x = ResonantTensor(combined, list(x.shape), precision=self.precision)

        # === H-PHASE: Harmonization via DHSR cycle ===
        syntony_new = x.cpu_cycle(noise_scale=0.01, precision=self.precision)

        # === R-PHASE: Golden recursion ===
        x.apply_recursion()
        x.apply_inverse_recursion()  # Undo to keep scale, but phases are mixed

        # === CONSENSUS CHECK: ΔS > threshold ===
        delta_s = abs(syntony_new - prev_syntony)
        accepted = delta_s > self.consensus_threshold

        # === TEMPORAL RECORDING: Blockchain ===
        if accepted:
            self._record_block(x.to_floats(), syntony_new)
            self.total_blocks_validated += 1
        else:
            self.total_blocks_rejected += 1

        return x, syntony_new, accepted

    def _record_block(self, state: List[float], syntony: float):
        """
        Append block to temporal blockchain.

        Args:
            state: State as flattened float list
            syntony: Syntony value
        """
        # Average over batch for single state record
        dim = self.dim
        batch_size = len(state) // dim
        state_avg = [0.0] * dim
        for b in range(batch_size):
            for d in range(dim):
                state_avg[d] += state[b * dim + d]
        state_avg = [s / batch_size for s in state_avg]

        self.temporal_record.append(state_avg)
        self.syntony_record.append(syntony)

    def get_blockchain_length(self) -> int:
        """Get length of temporal blockchain."""
        return len(self.syntony_record)

    def get_blockchain(self) -> Tuple[List[List[float]], List[float]]:
        """Get complete blockchain history."""
        return self.temporal_record, self.syntony_record

    def get_validation_rate(self) -> float:
        """Get consensus validation rate."""
        total = self.total_blocks_validated + self.total_blocks_rejected
        if total == 0:
            return 0.0
        return self.total_blocks_validated / total

    def parameters(self) -> List[ResonantTensor]:
        """Return all learnable parameters."""
        return self.d_expand.parameters() + self.d_project.parameters()

    def __repr__(self) -> str:
        return (
            f"WindingDHSRBlock(dim={self.dim}, expand_factor={self.fib_expand_factor}, "
            f"threshold={self.consensus_threshold:.4f})"
        )


if __name__ == "__main__":
    # Test the pure WindingDHSRBlock
    print("Testing WindingDHSRBlock...")
    
    block = WindingDHSRBlock(dim=16, fib_expand_factor=2)
    print(f"Block: {block}")

    # Create sample data
    x_data = [random.gauss(0, 1) for _ in range(32 * 16)]  # batch of 32
    x = ResonantTensor(x_data, [32, 16])
    prev_syntony = 0.5

    # Forward pass
    x_new, syntony_new, accepted = block.forward(x, prev_syntony)

    print(f"\nInput shape: {x.shape}")
    print(f"Output shape: {x_new.shape}")
    print(f"Previous syntony: {prev_syntony:.4f}")
    print(f"New syntony: {syntony_new:.4f}")
    print(f"ΔS: {abs(syntony_new - prev_syntony):.4f}")
    print(f"Block accepted: {accepted}")
    print(f"Blockchain length: {block.get_blockchain_length()}")

    # Run a few more cycles
    for i in range(5):
        x_new, syntony_new, accepted = block.forward(x_new, syntony_new)
        print(f"Cycle {i+2}: S={syntony_new:.4f}, accepted={accepted}")

    print(f"\nFinal blockchain length: {block.get_blockchain_length()}")
    print(f"Final validation rate: {block.get_validation_rate():.2%}")
    print("SUCCESS")
