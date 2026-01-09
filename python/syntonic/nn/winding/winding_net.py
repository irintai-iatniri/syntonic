"""
WindingNet - Complete winding-aware neural network architecture.

This module implements the full WindingNet architecture that integrates:
1. Winding state embeddings (T^4 torus structure)
2. Fibonacci hierarchy (golden ratio depth scaling)
3. Prime selection (Möbius filtering)
4. DHSR dynamics (differentiation-harmonization cycles)
5. Temporal blockchain (immutable state recording)
6. Syntony consensus (ΔS > threshold validation)
"""

from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Tuple, Optional
import math
import inspect

try:
    from syntonic.srt.geometry.winding import WindingState
except ImportError:
    from syntonic._core import WindingState

try:
    from syntonic._core import ResonantTensor
    RESONANT_AVAILABLE = True
except ImportError:
    RESONANT_AVAILABLE = False
    ResonantTensor = None

from syntonic.nn.winding.resonant_embedding import ResonantWindingEmbedding
from syntonic.nn.winding.fibonacci_hierarchy import FibonacciHierarchy
from syntonic.nn.winding.resonant_dhsr_block import ResonantWindingDHSRBlock
from syntonic.nn.layers.resonant_linear import ResonantLinear

PHI = (1 + math.sqrt(5)) / 2  # Golden ratio ~ 1.618
Q_DEFICIT = 0.027395146920  # Universal syntony deficit


class WindingNet(nn.Module):
    """
    Neural network operating on winding space.

    Architecture:
    - Winding embedding layer (maps |n₇,n₈,n₉,n₁₀⟩ to embeddings)
    - Fibonacci hierarchy (layer dims grow as F_k)
    - WindingDHSRBlocks (one per Fibonacci level)
    - Prime selection filtering (Möbius function)
    - Temporal blockchain recording
    - Syntony consensus mechanism

    Example:
        >>> from syntonic.physics.fermions.windings import *
        >>> model = WindingNet(max_winding=5, base_dim=64, num_blocks=3, output_dim=2)
        >>> windings = [ELECTRON_WINDING, MUON_WINDING, UP_WINDING]
        >>> y = model(windings)
        >>> y.shape
        torch.Size([3, 2])
    """

    def __init__(
        self,
        max_winding: int = 5,
        base_dim: int = 64,
        num_blocks: int = 3,
        output_dim: int = 2,
        use_prime_filter: bool = True,
        consensus_threshold: float = 0.024,
        dropout: float = 0.1,
    ):
        """
        Initialize WindingNet.

        Args:
            max_winding: Maximum winding number for enumeration
            base_dim: Base embedding dimension
            num_blocks: Number of DHSR blocks (Fibonacci levels)
            output_dim: Output dimension (number of classes)
            use_prime_filter: Whether to use prime selection in blocks
            consensus_threshold: ΔS threshold for block validation
            dropout: Dropout rate
        """
        super().__init__()

        self.max_winding = max_winding
        self.base_dim = base_dim
        self.num_blocks = num_blocks
        self.output_dim = output_dim

        # 1. Resonant Winding embedding layer
        self.winding_embed = ResonantWindingEmbedding(
            max_n=max_winding,
            embed_dim=base_dim,
        )

        # 2. Fibonacci hierarchy
        self.fib_hierarchy = FibonacciHierarchy(num_blocks)
        layer_dims = self.fib_hierarchy.get_layer_dims(base_dim)
        self.layer_dims = layer_dims

        # 3. Resonant DHSR blocks (one per Fibonacci level)
        self.blocks = nn.ModuleList()
        for i in range(num_blocks):
            block = ResonantWindingDHSRBlock(
                dim=layer_dims[i],
                fib_expand_factor=self.fib_hierarchy.get_expansion_factor(i),
                use_prime_filter=use_prime_filter,
                consensus_threshold=consensus_threshold,
                dropout=dropout,
            )
            self.blocks.append(block)

        # 4. Transitions between Fibonacci levels (Resonant)
        self.transitions = nn.ModuleList()
        for i in range(num_blocks - 1):
            self.transitions.append(
                ResonantLinear(layer_dims[i], layer_dims[i + 1], precision=64)
            )

        # 5. Output projection (Resonant)
        final_dim = layer_dims[num_blocks - 1]
        self.output_proj = ResonantLinear(final_dim, output_dim, precision=64)

        # 6. Mode norms per layer
        self.mode_norms = nn.ParameterList([
            nn.Parameter(
                torch.arange(dim).pow(2).float(), requires_grad=False
            )
            for dim in layer_dims
        ])

        # Network-level metrics
        self.network_syntony = 0.0
        self.layer_syntonies = []

    def forward(self, winding_states: List[WindingState]) -> torch.Tensor:
        """
        Forward pass through WindingNet.

        Args:
            winding_states: List of input winding configurations

        Returns:
            predictions: (batch, output_dim) logits
        """
        # 1. Embed windings
        x = self.winding_embed.batch_forward(winding_states)
        
        # Initial syntony
        syntony = 0.5
        syntonies = []

        # 2. Pass through DHSR blocks
        for i, block in enumerate(self.blocks):
            # DHSR cycle (Unified Lattice ↔ Flux)
            x, syntony_new, accepted = block(
                x,
                self.mode_norms[i],
                syntony,
            )

            syntonies.append(syntony_new)
            syntony = syntony_new

            # Transition to next Fibonacci level
            if i < len(self.transitions):
                x = F.relu(self.transitions[i](x))

        # 3. Final metrics and Output
        self.network_syntony = sum(syntonies) / len(syntonies)
        self.layer_syntonies = syntonies

        return self.output_proj(x)

    def compute_loss(
        self,
        y_pred: torch.Tensor,
        y_true: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute loss with syntony regularization.

        Loss = L_task + q × (1 - S_network)

        where q = 0.027395... (universal syntony deficit)

        Args:
            y_pred: Predictions (batch, output_dim)
            y_true: Ground truth labels (batch,)

        Returns:
            total_loss: Combined loss
            task_loss: Task-specific loss
            syntony_loss: Syntony regularization term
        """
        # Task loss
        task_loss = F.cross_entropy(y_pred, y_true)

        # Syntony loss: penalize low syntony
        syntony_loss_value = 1.0 - self.network_syntony
        syntony_loss = torch.tensor(syntony_loss_value, device=y_pred.device)

        # Combined loss
        total_loss = task_loss + Q_DEFICIT * syntony_loss

        return total_loss, task_loss, syntony_loss

    def get_blockchain_stats(self) -> Dict[str, float]:
        """
        Get blockchain and consensus statistics.

        Returns:
            Dictionary with validation metrics
        """
        total_validated = sum(b.total_blocks_validated for b in self.blocks)
        total_rejected = sum(b.total_blocks_rejected for b in self.blocks)
        total_cycles = total_validated + total_rejected

        blockchain_length = sum(b.get_blockchain_length() for b in self.blocks)

        return {
            "total_cycles": total_cycles,
            "validated_blocks": total_validated,
            "rejected_blocks": total_rejected,
            "validation_rate": total_validated / max(total_cycles, 1),
            "network_syntony": self.network_syntony,
            "blockchain_length": blockchain_length,
            "layer_syntonies": self.layer_syntonies,
        }

    def get_temporal_blockchain(self, block_idx: int = 0) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Access the temporal ledger for a given block.

        Args:
            block_idx: Index of the DHSR block

        Returns:
            states: Temporal state record
            syntonies: Syntony record
        """
        if block_idx >= len(self.blocks):
            raise ValueError(f"Block index {block_idx} out of range (have {len(self.blocks)} blocks)")
        return self.blocks[block_idx].get_blockchain()

    # Removed crystallize_weights and forward_exact
    # The unified DHSR cycle handles these phases natively in forward()

    def extra_repr(self) -> str:
        return (
            f"max_winding={self.max_winding}, base_dim={self.base_dim}, "
            f"num_blocks={self.num_blocks}, output_dim={self.output_dim}, "
            f"layer_dims={self.layer_dims}"
        )


if __name__ == "__main__":
    # Example usage
    print("="*60)
    print("WindingNet Example")
    print("="*60)

    from syntonic.physics.fermions.windings import (
        ELECTRON_WINDING,
        MUON_WINDING,
        TAU_WINDING,
        UP_WINDING,
        DOWN_WINDING,
        CHARM_WINDING,
    )

    # Create model
    model = WindingNet(
        max_winding=5,
        base_dim=64,
        num_blocks=3,
        output_dim=2,
    )
    print(f"\nModel: {model}")
    print(f"\nTotal parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Create dataset: leptons (0) vs quarks (1)
    windings = [
        ELECTRON_WINDING, MUON_WINDING, TAU_WINDING,  # Leptons
        UP_WINDING, DOWN_WINDING, CHARM_WINDING,      # Quarks
    ]
    labels = torch.tensor([0, 0, 0, 1, 1, 1])

    # Forward pass
    y_pred = model(windings)
    print(f"\nPredictions shape: {y_pred.shape}")
    print(f"Predictions:\n{y_pred}")

    # Compute loss
    total_loss, task_loss, syntony_loss = model.compute_loss(y_pred, labels)
    print(f"\nLosses:")
    print(f"  Task loss: {task_loss.item():.4f}")
    print(f"  Syntony loss: {syntony_loss.item():.4f}")
    print(f"  Total loss: {total_loss.item():.4f}")

    # Blockchain stats
    stats = model.get_blockchain_stats()
    print(f"\nBlockchain stats:")
    print(f"  Network syntony: {stats['network_syntony']:.4f}")
    print(f"  Layer syntonies: {[f'{s:.4f}' for s in stats['layer_syntonies']]}")
    print(f"  Validation rate: {stats['validation_rate']:.2%}")
    print(f"  Blockchain length: {stats['blockchain_length']}")

    # Test gradients
    print(f"\nGradient test:")
    total_loss.backward()
    grad_norms = [p.grad.norm().item() for p in model.parameters() if p.grad is not None]
    print(f"  Gradient norms (first 5): {grad_norms[:5]}")
    print(f"  Max gradient norm: {max(grad_norms):.4f}")
    print(f"  Mean gradient norm: {sum(grad_norms)/len(grad_norms):.4f}")

    print("\n" + "="*60)
    print("WindingNet test complete!")
    print("="*60)
