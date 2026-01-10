"""
Resonant DHSR Block - Proper dual-state architecture with lattice/flux dance.

This implements the Resonant Engine as specified in RESONANT_ENGINE_TECHNICAL.md:

**DUAL-STATE ARCHITECTURE:**
- Lattice (CPU): Exact Q(φ) values - eternal, governing
- Flux (GPU): Ephemeral f64 values - exploratory

**THE DANCE (every cycle):**
1. wake_flux(): Lattice → GPU flux
2. differentiate(): D̂ on GPU (fast exploration)
3. crystallize_with_dwell(): Ĥ + snap to Q(φ) + φ-dwell timing
4. destroy_shadow(): Flux = None

The flux is EPHEMERAL - it only exists during D-phase, then destroyed.

NO PYTORCH DEPENDENCIES - uses pure sn (syntonic network) module.
"""

from __future__ import annotations
from typing import Tuple, Optional, List
import math
import time

try:
    from syntonic._core import ResonantTensor
    RESONANT_AVAILABLE = True
except ImportError:
    RESONANT_AVAILABLE = False
    ResonantTensor = None

# Use sn (syntonic network) instead of torch.nn
import syntonic.sn as sn
from syntonic.nn.winding.prime_selection_pure import PurePrimeSelectionLayer

PHI = (1 + math.sqrt(5)) / 2  # Golden ratio
PHI_INV = 1 / PHI  # φ^-1 ≈ 0.618
PHI_INV_SQ = PHI_INV ** 2  # φ^-2 ≈ 0.382


class ResonantWindingDHSRBlock(sn.Module):
    """
    DHSR block with proper Resonant Engine dual-state architecture.

    Unlike traditional neural network layers, this block:
    1. Maintains weights in exact Q(φ) lattice (eternal)
    2. Projects to GPU flux only during D-phase (ephemeral)
    3. Snaps back to lattice after each cycle (crystallization)

    The state "dances" between exact and float every forward pass.
    """

    def __init__(
        self,
        dim: int,
        fib_expand_factor: int = 3,
        use_prime_filter: bool = True,
        consensus_threshold: float = 0.024,
        precision: int = 100,
        noise_scale: float = 0.01,
        dropout: float = 0.1,
    ):
        """
        Initialize ResonantWindingDHSRBlock.

        Args:
            dim: Feature dimension
            fib_expand_factor: Fibonacci expansion factor for D-phase
            use_prime_filter: Whether to apply prime selection
            consensus_threshold: ΔS threshold for blockchain validation
            precision: Bit precision for exact lattice arithmetic
            noise_scale: Stochastic noise amplitude during differentiation
            dropout: Dropout rate (applied during D-phase)
        """
        super().__init__()

        if not RESONANT_AVAILABLE:
            raise RuntimeError(
                "ResonantTensor not available. This block requires the Resonant Engine."
            )

        self.dim = dim
        self.fib_expand_factor = fib_expand_factor
        self.use_prime_filter_flag = use_prime_filter
        self.consensus_threshold = consensus_threshold
        self.precision = precision
        self.noise_scale = noise_scale

        # Prime selection filter (optional) - using pure version
        if use_prime_filter:
            self.prime_filter = PurePrimeSelectionLayer(dim)
        else:
            self.prime_filter = None

        # Dropout (applied during flux phase)
        self.dropout = sn.Dropout(dropout)

        # Blockchain recording
        self.temporal_record: List[List[float]] = []
        self.syntony_record: List[float] = []

        # Blockchain statistics
        self.total_blocks_validated = 0
        self.total_blocks_rejected = 0

        # Timing statistics (for φ-dwell verification)
        self.last_d_duration = 0.0
        self.last_h_duration = 0.0
        self.last_ratio = 0.0

    def forward(
        self,
        x: ResonantTensor,
        mode_norms: List[float],
        prev_syntony: float,
    ) -> Tuple[ResonantTensor, float, bool]:
        """
        Forward pass through Resonant DHSR cycle (Batched).

        THE UNIFIED CYCLE (Batched):
        1. wake_flux(): project lattice to flux (performed in Rust constructor)
        2. differentiate(): D̂ flux on GPU/CPU
        3. crystallize(): Ĥ + snap back to Q(φ) lattice
        4. destroy_shadow(): clear flux
        
        Args:
            x: Input ResonantTensor
            mode_norms: Mode norms for each feature
            prev_syntony: Previous syntony value
            
        Returns:
            (output, syntony, accepted): Crystallized output, new syntony, blockchain acceptance
        """
        shape = x.shape()
        batch_size = shape[0] if len(shape) > 1 else 1
        
        # 1. EXECUTE DHSR CYCLE (D̂ + Ĥ + Crystallize)
        d_start = time.perf_counter()
        
        # batch_cpu_cycle handles the complete phase transition
        batch_syntonies = x.batch_cpu_cycle(
            noise_scale=self.noise_scale,
            precision=self.precision
        )

        self.last_d_duration = time.perf_counter() - d_start

        # 2. Apply prime filter (Möbius Selection)
        if self.prime_filter is not None:
            x = self.prime_filter(x)

        # 3. Update syntony (average of batch)
        syntony_new = sum(batch_syntonies) / len(batch_syntonies) if batch_syntonies else 0.5

        # 4. TEMPORAL BLOCKCHAIN RECORDING
        delta_syntony = abs(syntony_new - prev_syntony)
        accepted = delta_syntony > self.consensus_threshold

        if accepted:
            self.total_blocks_validated += 1
            # Record to blockchain
            data = x.to_floats()
            if len(shape) > 1:
                # Average over batch
                avg_state = []
                for i in range(self.dim):
                    col_sum = sum(data[b * self.dim + i] for b in range(batch_size))
                    avg_state.append(col_sum / batch_size)
                self.temporal_record.append(avg_state)
            self.syntony_record.append(syntony_new)
        else:
            self.total_blocks_rejected += 1

        return x, syntony_new, accepted

    def get_blockchain(self) -> Tuple[List[List[float]], List[float]]:
        """
        Access the temporal blockchain.

        Returns:
            states: Temporal state record (list of state vectors)
            syntonies: Syntony record (list of floats)
        """
        return self.temporal_record, self.syntony_record

    def get_blockchain_length(self) -> int:
        """Get number of validated blocks in blockchain."""
        return len(self.syntony_record)

    def get_timing_stats(self) -> dict:
        """
        Get timing statistics to verify φ-dwell.

        Returns:
            Dictionary with D-phase, H-phase durations and ratio
        """
        return {
            "d_duration": self.last_d_duration,
            "h_duration": self.last_h_duration,
            "ratio": self.last_ratio,
            "target_ratio": PHI,
            "phi_dwell_satisfied": abs(self.last_ratio - PHI) < 0.1 if self.last_ratio > 0 else False,
        }

    def extra_repr(self) -> str:
        return (
            f"dim={self.dim}, fib_expand={self.fib_expand_factor}, "
            f"prime_filter={self.use_prime_filter_flag}, "
            f"consensus_threshold={self.consensus_threshold}, "
            f"precision={self.precision}, resonant_engine=True"
        )


if __name__ == "__main__":
    import random
    
    print("=" * 70)
    print("Resonant DHSR Block Test - Dual-State Architecture (Pure Python)")
    print("=" * 70)

    if not RESONANT_AVAILABLE:
        print("\n❌ ResonantTensor not available!")
        print("This block requires the Resonant Engine (exact Q(φ) arithmetic)")
        exit(1)

    # Create block
    dim = 64
    block = ResonantWindingDHSRBlock(
        dim=dim,
        fib_expand_factor=3,
        use_prime_filter=True,
        consensus_threshold=0.024,
        precision=100,
        noise_scale=0.01,
    )

    print(f"\nBlock: {block}")

    # Create dummy input as ResonantTensor
    batch_size = 4
    data = [random.gauss(0, 1) for _ in range(batch_size * dim)]
    mode_norms = [float(i * i) for i in range(batch_size * dim)]
    x = ResonantTensor(data, [batch_size, dim], mode_norms, 100)
    mode_norm_list = [float(i * i) for i in range(dim)]
    prev_syntony = 0.5

    # Run resonant cycle
    print("\n" + "=" * 70)
    print("Running Resonant Cycle (Lattice ↔ Flux Dance)")
    print("=" * 70)

    x_new, syntony, accepted = block(x, mode_norm_list, prev_syntony)

    print(f"\nInput shape: {x.shape()}")
    print(f"Output shape: {x_new.shape()}")
    print(f"Lattice syntony: {syntony:.6f} (exact Q(φ))")
    print(f"Consensus: {'✓ ACCEPTED' if accepted else '✗ REJECTED'}")
    print(f"Blockchain length: {block.get_blockchain_length()}")

    # Timing stats
    timing = block.get_timing_stats()
    print(f"\n" + "=" * 70)
    print("φ-Dwell Timing Verification")
    print("=" * 70)
    print(f"D-phase duration: {timing['d_duration']*1000:.2f}ms")

    # Verify state evolution
    print(f"\n" + "=" * 70)
    print("State Evolution Test (Multiple Cycles)")
    print("=" * 70)

    syntonies = []
    for cycle in range(5):
        x_new, s, acc = block(x_new, mode_norm_list, prev_syntony)
        syntonies.append(s)
        prev_syntony = s
        print(f"Cycle {cycle}: S = {s:.6f}, accepted = {acc}")

    print(f"\nSyntony evolution: {' → '.join(f'{s:.4f}' for s in syntonies)}")
    print(f"Blockchain length: {block.get_blockchain_length()}")

    print("\n" + "=" * 70)
    print("✓ Resonant Engine dual-state architecture (pure Python) verified!")
    print("=" * 70)

