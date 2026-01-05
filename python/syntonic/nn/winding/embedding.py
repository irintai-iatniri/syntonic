"""
Winding State Embedding - Maps winding states to neural embeddings.

This module provides WindingStateEmbedding, which maps discrete T^4 winding
states |n₇, n₈, n₉, n₁₀⟩ to continuous learned embeddings in neural space.

NOTE: This is different from the existing WindingEmbedding in
architectures/embeddings.py, which maps token indices to torus positions.
Here we map WindingState objects (physical states) to learned embeddings.
"""

from __future__ import annotations
import torch
import torch.nn as nn
from typing import List, Dict, Optional
import math

try:
    from syntonic.srt.geometry.winding import enumerate_windings, WindingState
except ImportError:
    from syntonic._core import enumerate_windings, WindingState


class WindingStateEmbedding(nn.Module):
    """
    Embedding layer for winding states.

    Maps discrete winding states |n₇, n₈, n₉, n₁₀⟩ to learned continuous
    embeddings. Each unique winding state gets its own learnable parameter vector.

    Example:
        >>> from syntonic.physics.fermions.windings import ELECTRON_WINDING
        >>> embed = WindingStateEmbedding(max_n=5, embed_dim=64)
        >>> x = embed(ELECTRON_WINDING)
        >>> x.shape
        torch.Size([64])
    """

    def __init__(
        self,
        max_n: int = 5,
        embed_dim: int = 64,
        init_scale: Optional[float] = None,
    ):
        """
        Initialize winding state embedding.

        Args:
            max_n: Maximum winding number (enumerates [-max_n, max_n]^4)
            embed_dim: Embedding dimension
            init_scale: Initialization scale (defaults to 1/sqrt(embed_dim))
        """
        super().__init__()

        self.max_n = max_n
        self.embed_dim = embed_dim

        # Enumerate all winding states in the range
        self.windings = enumerate_windings(max_n)
        self.num_windings = len(self.windings)

        # Create learnable embeddings for each winding
        # Using ParameterDict allows string keys and avoids tensor indexing issues
        self.embeddings = nn.ParameterDict()
        for w in self.windings:
            key = self._winding_key(w)
            param = nn.Parameter(torch.randn(embed_dim))
            self.embeddings[key] = param

        # Store mode norms |n|² for each winding
        self.mode_norms: Dict[str, int] = {}
        for w in self.windings:
            key = self._winding_key(w)
            self.mode_norms[key] = w.norm_squared

        # Initialize embeddings
        self._initialize_embeddings(init_scale)

    def _winding_key(self, w: WindingState) -> str:
        """
        Generate string key for winding state.

        Args:
            w: WindingState object

        Returns:
            String key "n7,n8,n9,n10"
        """
        return f"{w.n7},{w.n8},{w.n9},{w.n10}"

    def _initialize_embeddings(self, init_scale: Optional[float] = None):
        """
        Initialize embeddings with appropriate scaling.

        Uses Xavier-style initialization scaled by mode norm.

        Args:
            init_scale: Custom initialization scale
        """
        if init_scale is None:
            init_scale = 1.0 / math.sqrt(self.embed_dim)

        for key, param in self.embeddings.items():
            # Golden initialization: scale by exp(-|n|²/(2φ))
            # This gives higher-norm windings smaller initial embeddings
            PHI = 1.6180339887498949
            mode_norm_sq = self.mode_norms[key]
            golden_scale = math.exp(-mode_norm_sq / (2 * PHI))

            nn.init.normal_(param, mean=0.0, std=init_scale * golden_scale)

    def forward(self, winding: WindingState) -> torch.Tensor:
        """
        Get embedding for a single winding state.

        Args:
            winding: WindingState object

        Returns:
            Embedding tensor of shape (embed_dim,)
        """
        key = self._winding_key(winding)
        if key not in self.embeddings:
            raise ValueError(
                f"Winding state {winding} not in embedding vocabulary. "
                f"max_n={self.max_n} may be too small."
            )
        return self.embeddings[key]

    def batch_forward(self, windings: List[WindingState]) -> torch.Tensor:
        """
        Get embeddings for a batch of winding states.

        Args:
            windings: List of WindingState objects

        Returns:
            Embeddings tensor of shape (batch_size, embed_dim)
        """
        embeddings = [self.forward(w) for w in windings]
        return torch.stack(embeddings, dim=0)

    def get_mode_norm(self, winding: WindingState) -> int:
        """
        Get mode norm squared |n|² for a winding state.

        Args:
            winding: WindingState object

        Returns:
            Mode norm squared
        """
        key = self._winding_key(winding)
        return self.mode_norms[key]

    def get_all_mode_norms(self, windings: List[WindingState]) -> torch.Tensor:
        """
        Get mode norms for a batch of windings.

        Args:
            windings: List of WindingState objects

        Returns:
            Tensor of mode norms, shape (batch_size,)
        """
        norms = [self.get_mode_norm(w) for w in windings]
        return torch.tensor(norms, dtype=torch.float32)

    def extra_repr(self) -> str:
        return (
            f"max_n={self.max_n}, embed_dim={self.embed_dim}, "
            f"num_windings={self.num_windings}"
        )


if __name__ == "__main__":
    # Example usage
    from syntonic.physics.fermions.windings import (
        ELECTRON_WINDING,
        MUON_WINDING,
        UP_WINDING,
    )

    embed = WindingStateEmbedding(max_n=5, embed_dim=64)
    print(f"Embedding: {embed}")

    # Single winding
    x = embed(ELECTRON_WINDING)
    print(f"Electron embedding shape: {x.shape}")
    print(f"Electron mode norm: {embed.get_mode_norm(ELECTRON_WINDING)}")

    # Batch
    windings = [ELECTRON_WINDING, MUON_WINDING, UP_WINDING]
    X = embed.batch_forward(windings)
    print(f"Batch embeddings shape: {X.shape}")

    norms = embed.get_all_mode_norms(windings)
    print(f"Batch mode norms: {norms}")
