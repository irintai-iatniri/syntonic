"""
Resonant Winding Embedding - Maps winding states to resonant embeddings.

This module provides ResonantWindingEmbedding, which utilizes ResonantParameter
to store and evolve embeddings directly on the golden lattice Q(φ).
"""

from __future__ import annotations
import torch
import torch.nn as nn
from typing import List, Optional, Dict
import math

try:
    from syntonic.srt.geometry.winding import enumerate_windings, WindingState
except ImportError:
    from syntonic._core import enumerate_windings, WindingState

from syntonic.nn.layers.resonant_parameter import ResonantParameter

PHI = (1 + math.sqrt(5)) / 2

class ResonantWindingEmbedding(nn.Module):
    """
    Embedding layer for winding states using Resonant Engine.
    
    Weights are stored in a ResonantParameter, allowing exact lattice
    evolution and crystallization.
    """

    def __init__(
        self,
        max_n: int = 5,
        embed_dim: int = 64,
        precision: int = 100,
    ):
        """
        Initialize resonant winding embedding.

        Args:
            max_n: Maximum winding number for enumeration
            embed_dim: Embedding dimension
            precision: Exact arithmetic precision
        """
        super().__init__()

        self.max_n = max_n
        self.embed_dim = embed_dim

        # 1. Enumerate windings and map to indices
        self.windings = enumerate_windings(max_n)
        self.num_windings = len(self.windings)
        self.winding_to_idx = {
            self._winding_key(w): i for i, w in enumerate(self.windings)
        }

        # 2. Precompute mode norms for each index
        mode_norms = torch.tensor([w.norm_squared for w in self.windings], dtype=torch.float32)
        # Expansion: norms for each element in (num_windings, embed_dim)
        # Actually, the ResonantTensor expects a mode_norm for EVERY element.
        # But in an embedding, each 'token' (winding) has a norm.
        # We broadcast the winding norm to all elements of its embedding.
        full_mode_norms = mode_norms.unsqueeze(1).expand(-1, embed_dim).contiguous()

        # 3. Initialize data with Golden Measure: exp(-|n|²/2φ)
        # This ensures higher-energy windings start with smaller representations
        data = torch.randn(self.num_windings, embed_dim)
        scales = torch.exp(-mode_norms / (2 * PHI)).unsqueeze(1)
        data = data * (scales / math.sqrt(embed_dim))

        # 4. Create ResonantParameter
        self.weight = ResonantParameter(
            data=data,
            mode_norm_sq=full_mode_norms,
            precision=precision
        )

    def _winding_key(self, w: WindingState) -> str:
        return f"{w.n7},{w.n8},{w.n9},{w.n10}"

    def batch_forward(self, winding_states: List[WindingState]) -> torch.Tensor:
        """
        Get embeddings for a list of winding states.
        
        Args:
            winding_states: List of WindingState objects
            
        Returns:
            Tensor of shape (batch_size, embed_dim)
        """
        indices = []
        for w in winding_states:
            key = self._winding_key(w)
            if key not in self.winding_to_idx:
                raise ValueError(f"Winding {w} not in vocabulary (max_n={self.max_n})")
            indices.append(self.winding_to_idx[key])
            
        idx_tensor = torch.tensor(indices, device=self.weight.data.device)
        return torch.embedding(self.weight, idx_tensor)

    def crystallize(self):
        """Snap entire embedding matrix to lattice."""
        self.weight.crystallize()

    def wake_flux(self):
        """Prepare for GPU/Float evolution."""
        self.weight.wake_flux()

    def extra_repr(self) -> str:
        return f"num_windings={self.num_windings}, embed_dim={self.embed_dim}, max_n={self.max_n}"
