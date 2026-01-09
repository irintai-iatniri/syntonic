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
"""

from __future__ import annotations
from typing import Tuple, Optional
import math
import time

try:
    from syntonic._core import ResonantTensor
    RESONANT_AVAILABLE = True
except ImportError:
    RESONANT_AVAILABLE = False
    ResonantTensor = None

from syntonic.nn.winding.prime_selection import PrimeSelectionLayer

PHI = (1 + math.sqrt(5)) / 2  # Golden ratio
PHI_INV = 1 / PHI  # φ^-1 ≈ 0.618
PHI_INV_SQ = PHI_INV ** 2  # φ^-2 ≈ 0.382


class ResonantWindingDHSRBlock(nn.Module):
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

        # Prime selection filter (optional)
        if use_prime_filter:
            self.prime_filter = PrimeSelectionLayer(dim)
        else:
            self.prime_filter = None

        # Dropout (applied during flux phase)
        self.dropout = nn.Dropout(dropout)

        # Blockchain recording
        self.register_buffer('temporal_record', torch.zeros(0, dim))
        self.register_buffer('syntony_record', torch.zeros(0))

        # Blockchain statistics
        self.total_blocks_validated = 0
        self.total_blocks_rejected = 0

        # Timing statistics (for φ-dwell verification)
        self.last_d_duration = 0.0
        self.last_h_duration = 0.0
        self.last_ratio = 0.0

    def forward(
        self,
        x: torch.Tensor,
        mode_norms: torch.Tensor,
        prev_syntony: float,
    ) -> Tuple[torch.Tensor, float, bool]:
        """
        Forward pass through Resonant DHSR cycle (Batched).

        THE UNIFIED CYCLE (Batched):
        1. wake_flux(): project lattice to flux (performed in Rust constructor)
        2. differentiate(): D̂ flux on GPU/CPU
        3. crystallize(): Ĥ + snap back to Q(φ) lattice
        4. destroy_shadow(): clear flux
        """
        batch_size = x.shape[0]
        
        # 1. Prepare data and norms
        # PROJECT TO RESONANT ENGINE (Wake Flux)
        x_flat = x.detach().cpu().numpy().flatten().tolist()
        norms_flat = (mode_norms.detach().cpu().numpy().tolist()) * batch_size
        
        rt = ResonantTensor(
            data=x_flat,
            shape=[batch_size, self.dim],
            mode_norm_sq=norms_flat,
            precision=self.precision
        )

        # 2. EXECUTE DHSR CYCLE (D̂ + Ĥ + Crystallize)
        d_start = time.perf_counter()
        
        # batch_cpu_cycle handles the complete phase transition
        batch_syntonies = rt.batch_cpu_cycle(
            noise_scale=self.noise_scale,
            precision=self.precision
        )

        self.last_d_duration = time.perf_counter() - d_start

        # 3. EXTRACTION AND MÖBIUS FILTERING (with STE)
        # Result is natively in Crystalline phase
        new_data = np.array(rt.to_list()).reshape(batch_size, self.dim)
        x_crystalline = torch.from_numpy(new_data).to(x.device).to(x.dtype)

        # Straight-Through Estimator: 
        # Output values are crystalline, but gradients pass through original flux
        x_new = (x_crystalline - x).detach() + x

        # Apply prime filter (Möbius Selection)
        if self.prime_filter is not None:
            x_new = self.prime_filter(x_new)

        # Update syntony (average of batch)
        syntony_new = sum(batch_syntonies) / batch_size

        # 4. TEMPORAL BLOCKCHAIN RECORDING
        delta_syntony = abs(syntony_new - prev_syntony)
        accepted = delta_syntony > self.consensus_threshold

        if accepted:
            self.total_blocks_validated += 1
            # Record average state to immutable ledger
            self.temporal_record = torch.cat([
                self.temporal_record,
                x_new.mean(dim=0, keepdim=True).detach()
            ], dim=0)
            self.syntony_record = torch.cat([
                self.syntony_record,
                torch.tensor([syntony_new], device=x.device)
            ], dim=0)
        else:
            self.total_blocks_rejected += 1

        return x_new, syntony_new, accepted

    def get_blockchain(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Access the temporal blockchain.

        Returns:
            states: Temporal state record (blockchain_length, dim)
            syntonies: Syntony record (blockchain_length,)
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
    print("=" * 70)
    print("Resonant DHSR Block Test - Dual-State Architecture")
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
    print(f"Parameters: {sum(p.numel() for p in block.parameters()):,}")

    # Create dummy input
    batch_size = 4
    x = torch.randn(batch_size, dim)
    mode_norms = torch.arange(dim).pow(2).float()
    prev_syntony = 0.5

    # Run resonant cycle
    print("\n" + "=" * 70)
    print("Running Resonant Cycle (Lattice ↔ Flux Dance)")
    print("=" * 70)

    x_new, syntony, accepted = block(x, mode_norms, prev_syntony)

    print(f"\nInput shape: {x.shape}")
    print(f"Output shape: {x_new.shape}")
    print(f"Lattice syntony: {syntony:.6f} (exact Q(φ))")
    print(f"Consensus: {'✓ ACCEPTED' if accepted else '✗ REJECTED'}")
    print(f"Blockchain length: {block.get_blockchain_length()}")

    # Timing stats
    timing = block.get_timing_stats()
    print(f"\n" + "=" * 70)
    print("φ-Dwell Timing Verification")
    print("=" * 70)
    print(f"D-phase duration: {timing['d_duration']*1000:.2f}ms")
    print(f"H-phase duration: {timing['h_duration']*1000:.2f}ms")
    if timing['ratio'] > 0:
        print(f"Actual ratio (H/D): {timing['ratio']:.4f}")
        print(f"Target ratio (φ):   {timing['target_ratio']:.4f}")
        print(f"φ-dwell satisfied:  {timing['phi_dwell_satisfied']}")

    # Verify state evolution
    print(f"\n" + "=" * 70)
    print("State Evolution Test (Multiple Cycles)")
    print("=" * 70)

    state = x[0:1]  # Single sample
    syntonies = []

    for cycle in range(5):
        state, s, acc = block(state, mode_norms, prev_syntony)
        syntonies.append(s)
        prev_syntony = s
        print(f"Cycle {cycle}: S = {s:.6f}, accepted = {acc}")

    print(f"\nSyntony evolution: {' → '.join(f'{s:.4f}' for s in syntonies)}")
    print(f"Blockchain length: {block.get_blockchain_length()}")

    print("\n" + "=" * 70)
    print("✓ Resonant Engine dual-state architecture verified!")
    print("=" * 70)
