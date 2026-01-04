"""
Syntonic Transformer: Transformer architecture with DHSR structure.

Includes:
- DHTransformerLayer: Transformer layer using D→H cycle
- CRTTransformer: Complete transformer with syntonic structure
- SyntonicTransformerEncoder/Decoder: Encoder/Decoder stacks

Source: CRT.md §12.2
"""

from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List, Tuple
import math

from syntonic.nn.layers import (
    DifferentiationLayer,
    HarmonizationLayer,
    RecursionBlock,
    SyntonicNorm,
)
from syntonic.nn.architectures.syntonic_attention import (
    MultiHeadSyntonicAttention,
)
from syntonic.nn.architectures.embeddings import (
    SyntonicEmbedding,
    PositionalEncoding,
)

PHI = (1 + math.sqrt(5)) / 2


class DHTransformerLayer(nn.Module):
    """
    Transformer layer with D→H structure.

    Standard transformer layer components are wrapped with
    differentiation and harmonization:
    1. D[Attention(H[x])]
    2. D[FFN(H[x])]

    Example:
        >>> layer = DHTransformerLayer(d_model=512, n_heads=8)
        >>> x = torch.randn(32, 100, 512)
        >>> y = layer(x)
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int = 8,
        d_ff: int = 2048,
        dropout: float = 0.1,
        use_gnosis: bool = False,
    ):
        """
        Initialize DH transformer layer.

        Args:
            d_model: Model dimension
            n_heads: Number of attention heads
            d_ff: Feed-forward hidden dimension
            dropout: Dropout probability
            use_gnosis: Use gnosis attention
        """
        super().__init__()

        self.d_model = d_model

        # Self-attention with DHSR
        self.self_attn = MultiHeadSyntonicAttention(
            d_model, n_heads, dropout, use_gnosis
        )

        # Feed-forward network
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )

        # DHSR for attention sublayer
        self.attn_diff = DifferentiationLayer(d_model, d_model)
        self.attn_harm = HarmonizationLayer(d_model, d_model)

        # DHSR for FFN sublayer
        self.ffn_diff = DifferentiationLayer(d_model, d_model)
        self.ffn_harm = HarmonizationLayer(d_model, d_model)

        # Norms
        self.norm1 = SyntonicNorm(d_model)
        self.norm2 = SyntonicNorm(d_model)

        self.dropout = nn.Dropout(dropout)

        self._attn_syntony: Optional[float] = None
        self._ffn_syntony: Optional[float] = None

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input (batch, seq_len, d_model)
            mask: Optional attention mask

        Returns:
            Output (batch, seq_len, d_model)
        """
        batch, seq_len, dim = x.shape

        # === Attention sublayer ===
        # Pre-norm with harmonization
        x_norm = self.norm1(x.reshape(-1, dim)).reshape(batch, seq_len, dim)
        x_harm = self._apply_harm(x_norm, self.attn_harm)

        # Attention
        attn_out = self.self_attn(x_harm, x_harm, x_harm, mask)

        # Differentiate
        attn_diff = self._apply_diff(attn_out, self.attn_diff)
        self._attn_syntony = self._compute_syntony(attn_out, attn_diff)

        # Residual (golden scaled)
        x = x + attn_diff / PHI

        # === FFN sublayer ===
        # Pre-norm with harmonization
        x_norm = self.norm2(x.reshape(-1, dim)).reshape(batch, seq_len, dim)
        x_harm = self._apply_harm(x_norm, self.ffn_harm)

        # FFN
        ffn_out = self.ffn(x_harm)

        # Differentiate
        ffn_diff = self._apply_diff(ffn_out, self.ffn_diff)
        self._ffn_syntony = self._compute_syntony(ffn_out, ffn_diff)

        # Residual
        x = x + ffn_diff / PHI

        return x

    def _apply_diff(self, x: torch.Tensor, diff: DifferentiationLayer) -> torch.Tensor:
        """Apply differentiation layer."""
        batch, seq_len, dim = x.shape
        flat = x.reshape(-1, dim)
        flat = diff(flat)
        return flat.reshape(batch, seq_len, dim)

    def _apply_harm(self, x: torch.Tensor, harm: HarmonizationLayer) -> torch.Tensor:
        """Apply harmonization layer."""
        batch, seq_len, dim = x.shape
        flat = x.reshape(-1, dim)
        flat = harm(flat)
        return flat.reshape(batch, seq_len, dim)

    def _compute_syntony(self, x: torch.Tensor, x_diff: torch.Tensor) -> float:
        """Compute sublayer syntony."""
        with torch.no_grad():
            diff_norm = torch.norm(x_diff - x).item()
            return 1.0 / (1.0 + diff_norm)

    @property
    def syntony(self) -> float:
        """Get average layer syntony."""
        attn_s = self._attn_syntony if self._attn_syntony else 0.5
        ffn_s = self._ffn_syntony if self._ffn_syntony else 0.5
        return (attn_s + ffn_s) / 2


class SyntonicTransformerEncoder(nn.Module):
    """
    Stack of DH transformer encoder layers.

    Example:
        >>> encoder = SyntonicTransformerEncoder(d_model=512, n_layers=6)
        >>> x = torch.randn(32, 100, 512)
        >>> y = encoder(x)
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int = 8,
        n_layers: int = 6,
        d_ff: int = 2048,
        dropout: float = 0.1,
        use_gnosis: bool = False,
    ):
        """
        Initialize encoder stack.

        Args:
            d_model: Model dimension
            n_heads: Attention heads per layer
            n_layers: Number of layers
            d_ff: Feed-forward dimension
            dropout: Dropout probability
            use_gnosis: Use gnosis attention
        """
        super().__init__()

        self.layers = nn.ModuleList([
            DHTransformerLayer(d_model, n_heads, d_ff, dropout, use_gnosis)
            for _ in range(n_layers)
        ])

        self.norm = SyntonicNorm(d_model)

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward through encoder stack."""
        for layer in self.layers:
            x = layer(x, mask)

        batch, seq_len, dim = x.shape
        x = self.norm(x.reshape(-1, dim)).reshape(batch, seq_len, dim)
        return x

    @property
    def syntony(self) -> float:
        """Get average syntony across layers."""
        syntonies = [layer.syntony for layer in self.layers]
        return sum(syntonies) / len(syntonies) if syntonies else 0.5


class SyntonicTransformerDecoder(nn.Module):
    """
    Stack of DH transformer decoder layers.

    Includes cross-attention to encoder outputs.

    Example:
        >>> decoder = SyntonicTransformerDecoder(d_model=512, n_layers=6)
        >>> tgt = torch.randn(32, 50, 512)
        >>> memory = torch.randn(32, 100, 512)
        >>> output = decoder(tgt, memory)
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int = 8,
        n_layers: int = 6,
        d_ff: int = 2048,
        dropout: float = 0.1,
    ):
        """
        Initialize decoder stack.

        Args:
            d_model: Model dimension
            n_heads: Attention heads per layer
            n_layers: Number of layers
            d_ff: Feed-forward dimension
            dropout: Dropout probability
        """
        super().__init__()

        self.layers = nn.ModuleList([
            DHTransformerDecoderLayer(d_model, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        ])

        self.norm = SyntonicNorm(d_model)

    def forward(
        self,
        tgt: torch.Tensor,
        memory: torch.Tensor,
        tgt_mask: Optional[torch.Tensor] = None,
        memory_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward through decoder stack.

        Args:
            tgt: Target sequence (batch, tgt_len, d_model)
            memory: Encoder output (batch, src_len, d_model)
            tgt_mask: Target attention mask
            memory_mask: Cross-attention mask

        Returns:
            Decoder output (batch, tgt_len, d_model)
        """
        for layer in self.layers:
            tgt = layer(tgt, memory, tgt_mask, memory_mask)

        batch, seq_len, dim = tgt.shape
        tgt = self.norm(tgt.reshape(-1, dim)).reshape(batch, seq_len, dim)
        return tgt

    @property
    def syntony(self) -> float:
        """Get average syntony across layers."""
        syntonies = [layer.syntony for layer in self.layers]
        return sum(syntonies) / len(syntonies) if syntonies else 0.5


class DHTransformerDecoderLayer(nn.Module):
    """Single decoder layer with cross-attention."""

    def __init__(
        self,
        d_model: int,
        n_heads: int = 8,
        d_ff: int = 2048,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.d_model = d_model

        # Self-attention
        self.self_attn = MultiHeadSyntonicAttention(d_model, n_heads, dropout)

        # Cross-attention
        self.cross_attn = MultiHeadSyntonicAttention(d_model, n_heads, dropout)

        # FFN
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )

        # DHSR
        self.diff = DifferentiationLayer(d_model, d_model)
        self.harm = HarmonizationLayer(d_model, d_model)

        # Norms
        self.norm1 = SyntonicNorm(d_model)
        self.norm2 = SyntonicNorm(d_model)
        self.norm3 = SyntonicNorm(d_model)

        self._syntony: Optional[float] = None

    def forward(
        self,
        tgt: torch.Tensor,
        memory: torch.Tensor,
        tgt_mask: Optional[torch.Tensor] = None,
        memory_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass."""
        batch, seq_len, dim = tgt.shape

        # Self-attention
        x_norm = self.norm1(tgt.reshape(-1, dim)).reshape(batch, seq_len, dim)
        self_attn_out = self.self_attn(x_norm, x_norm, x_norm, tgt_mask)
        tgt = tgt + self_attn_out / PHI

        # Cross-attention
        x_norm = self.norm2(tgt.reshape(-1, dim)).reshape(batch, seq_len, dim)
        cross_attn_out = self.cross_attn(x_norm, memory, memory, memory_mask)
        tgt = tgt + cross_attn_out / PHI

        # FFN with DHSR
        x_norm = self.norm3(tgt.reshape(-1, dim)).reshape(batch, seq_len, dim)

        # Harmonize before FFN
        x_harm = self.harm(x_norm.reshape(-1, dim)).reshape(batch, seq_len, dim)
        ffn_out = self.ffn(x_harm)

        # Differentiate after FFN
        ffn_diff = self.diff(ffn_out.reshape(-1, dim)).reshape(batch, seq_len, dim)

        tgt = tgt + ffn_diff / PHI

        self._syntony = (self.self_attn.syntony + self.cross_attn.syntony) / 2
        return tgt

    @property
    def syntony(self) -> Optional[float]:
        return self._syntony


class CRTTransformer(nn.Module):
    """
    Complete CRT Transformer for sequence-to-sequence tasks.

    Encoder-decoder architecture with:
    - Syntonic embeddings
    - DH transformer layers
    - Golden ratio residuals
    - Syntony tracking throughout

    Example:
        >>> model = CRTTransformer(
        ...     src_vocab=10000,
        ...     tgt_vocab=10000,
        ...     d_model=512,
        ... )
        >>> src = torch.randint(0, 10000, (32, 100))
        >>> tgt = torch.randint(0, 10000, (32, 50))
        >>> output = model(src, tgt)
        >>> print(f"Model syntony: {model.syntony:.4f}")
    """

    def __init__(
        self,
        src_vocab: int,
        tgt_vocab: int,
        d_model: int = 512,
        n_heads: int = 8,
        n_encoder_layers: int = 6,
        n_decoder_layers: int = 6,
        d_ff: int = 2048,
        dropout: float = 0.1,
        max_len: int = 5000,
        share_embeddings: bool = False,
    ):
        """
        Initialize CRT Transformer.

        Args:
            src_vocab: Source vocabulary size
            tgt_vocab: Target vocabulary size
            d_model: Model dimension
            n_heads: Number of attention heads
            n_encoder_layers: Encoder layer count
            n_decoder_layers: Decoder layer count
            d_ff: Feed-forward dimension
            dropout: Dropout probability
            max_len: Maximum sequence length
            share_embeddings: Share src/tgt embeddings
        """
        super().__init__()

        self.d_model = d_model

        # Embeddings
        self.src_embed = SyntonicEmbedding(src_vocab, d_model)
        if share_embeddings:
            self.tgt_embed = self.src_embed
        else:
            self.tgt_embed = SyntonicEmbedding(tgt_vocab, d_model)

        self.pos_encoding = PositionalEncoding(d_model, max_len, dropout)

        # Encoder
        self.encoder = SyntonicTransformerEncoder(
            d_model, n_heads, n_encoder_layers, d_ff, dropout
        )

        # Decoder
        self.decoder = SyntonicTransformerDecoder(
            d_model, n_heads, n_decoder_layers, d_ff, dropout
        )

        # Output projection
        self.output_proj = nn.Linear(d_model, tgt_vocab)

        # Tie output weights to embeddings if shared
        if share_embeddings:
            self.output_proj.weight = self.src_embed.embedding.weight

    def forward(
        self,
        src: torch.Tensor,
        tgt: torch.Tensor,
        src_mask: Optional[torch.Tensor] = None,
        tgt_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass.

        Args:
            src: Source tokens (batch, src_len)
            tgt: Target tokens (batch, tgt_len)
            src_mask: Source padding mask
            tgt_mask: Target causal mask

        Returns:
            Logits (batch, tgt_len, tgt_vocab)
        """
        # Generate causal mask for target if not provided
        if tgt_mask is None:
            tgt_mask = self._generate_causal_mask(tgt.size(1), tgt.device)

        # Encode
        src_emb = self.pos_encoding(self.src_embed(src))
        memory = self.encoder(src_emb, src_mask)

        # Decode
        tgt_emb = self.pos_encoding(self.tgt_embed(tgt))
        decoder_out = self.decoder(tgt_emb, memory, tgt_mask, src_mask)

        # Project to vocabulary
        logits = self.output_proj(decoder_out)

        return logits

    def _generate_causal_mask(self, size: int, device: torch.device) -> torch.Tensor:
        """Generate causal attention mask."""
        mask = torch.triu(torch.ones(size, size, device=device), diagonal=1)
        return mask == 0

    @property
    def syntony(self) -> float:
        """Get average model syntony."""
        return (self.encoder.syntony + self.decoder.syntony) / 2

    def encode(
        self,
        src: torch.Tensor,
        src_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Encode source sequence."""
        src_emb = self.pos_encoding(self.src_embed(src))
        return self.encoder(src_emb, src_mask)

    def decode(
        self,
        tgt: torch.Tensor,
        memory: torch.Tensor,
        tgt_mask: Optional[torch.Tensor] = None,
        memory_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Decode with encoder memory."""
        if tgt_mask is None:
            tgt_mask = self._generate_causal_mask(tgt.size(1), tgt.device)

        tgt_emb = self.pos_encoding(self.tgt_embed(tgt))
        decoder_out = self.decoder(tgt_emb, memory, tgt_mask, memory_mask)
        return self.output_proj(decoder_out)
