"""
Syntonic Architectures - Complete neural network architectures.

Pre-built architectures that embed the DHSR cycle throughout:
- SyntonicMLP: Fully-connected with RecursionBlocks
- SyntonicConv: Convolutional with recursion
- CRTTransformer: Transformer with syntonic attention
"""

from python.syntonic.nn.architectures.syntonic_mlp import (
    PureSyntonicMLP as SyntonicMLP,
    PureSyntonicLinear as SyntonicLinear,
)
from python.syntonic.nn.architectures.syntonic_mlp import (
    PureSyntonicMLP,
    PureSyntonicLinear,
    PureDeepSyntonicMLP,
)
from python.syntonic.nn.architectures.syntonic_cnn import (
    PureSyntonicConv2d as SyntonicConv2d,
    PureSyntonicCNN1d as RecursionConvBlock,
    PureSyntonicCNN1d as SyntonicCNN,
)
from python.syntonic.nn.architectures.syntonic_cnn import (
    PureSyntonicConv1d,
    PureSyntonicConv2d,
    PureSyntonicCNN1d,
)
from python.syntonic.nn.architectures.embeddings import (
    PureSyntonicEmbedding as SyntonicEmbedding,
    PureWindingEmbedding as WindingEmbedding,
    PurePositionalEncoding as PositionalEncoding,
)
from python.syntonic.nn.architectures.embeddings import (
    PureSyntonicEmbedding,
    PureWindingEmbedding,
    PurePositionalEncoding,
)
from python.syntonic.nn.architectures.syntonic_attention import (
    PureSyntonicAttention as SyntonicAttention,
    PureSyntonicAttention as GnosisAttention,
    PureMultiHeadSyntonicAttention as MultiHeadSyntonicAttention,
)
from python.syntonic.nn.architectures.syntonic_attention import (
    PureSyntonicAttention,
    PureMultiHeadSyntonicAttention,
)
from python.syntonic.nn.architectures.syntonic_transformer import (
    PureDHTransformerLayer,
    PureSyntonicTransformerEncoder,
    PureSyntonicTransformer,
)
from syntonic.nn.layers.prime_syntony_gate import (
    PrimeSyntonyGate,
    WindingAttention,
    SRTTransformerBlock,
    get_stable_dimensions,
    suggest_network_dimensions,
)

__all__ = [
    # MLP
    "SyntonicMLP",
    "SyntonicLinear",
    "PureSyntonicMLP",
    "PureSyntonicLinear",
    "PureDeepSyntonicMLP",
    # CNN
    "SyntonicConv2d",
    "RecursionConvBlock",
    "SyntonicCNN",
    "PureSyntonicConv1d",
    "PureSyntonicCNN1d",
    # Embeddings
    "SyntonicEmbedding",
    "WindingEmbedding",
    "PositionalEncoding",
    "PureSyntonicEmbedding",
    "PureWindingEmbedding",
    "PurePositionalEncoding",
    # Attention
    "SyntonicAttention",
    "GnosisAttention",
    "MultiHeadSyntonicAttention",
    "PureSyntonicAttention",
    "PureMultiHeadSyntonicAttention",
    # Transformer (Pure only - PyTorch version removed)
    "PureDHTransformerLayer",
    "PureSyntonicTransformerEncoder",
    "PureSyntonicTransformer",
    # Prime Syntony Gates
    "PrimeSyntonyGate",
    "WindingAttention",
    "SRTTransformerBlock",
    "get_stable_dimensions",
    "suggest_network_dimensions",
]
