"""
Syntonic Architectures - Complete neural network architectures.

Pre-built architectures that embed the DHSR cycle throughout:
- SyntonicMLP: Fully-connected with RecursionBlocks
- SyntonicConv: Convolutional with recursion
- CRTTransformer: Transformer with syntonic attention
"""

from syntonic.nn.architectures.embeddings import (
    PurePositionalEncoding,
    PureSyntonicEmbedding,
    PureWindingEmbedding,
)
from syntonic.nn.architectures.embeddings import (
    PurePositionalEncoding as PositionalEncoding,
)
from syntonic.nn.architectures.embeddings import (
    PureSyntonicEmbedding as SyntonicEmbedding,
)
from syntonic.nn.architectures.embeddings import (
    PureWindingEmbedding as WindingEmbedding,
)
from syntonic.nn.architectures.syntonic_attention import (
    PureMultiHeadSyntonicAttention,
    PureSyntonicAttention,
)
from syntonic.nn.architectures.syntonic_attention import (
    PureMultiHeadSyntonicAttention as MultiHeadSyntonicAttention,
)
from syntonic.nn.architectures.syntonic_attention import (
    PureSyntonicAttention as GnosisAttention,
)
from syntonic.nn.architectures.syntonic_attention import (
    PureSyntonicAttention as SyntonicAttention,
)
from syntonic.nn.architectures.syntonic_cnn import (
    PureSyntonicCNN1d,
    PureSyntonicCNN2d,
    PureSyntonicConv1d,
    PureSyntonicConv2d,
)
from syntonic.nn.architectures.syntonic_cnn import (
    PureSyntonicCNN1d as RecursionConvBlock,
)
from syntonic.nn.architectures.syntonic_cnn import (
    PureSyntonicCNN1d as SyntonicCNN,
)
from syntonic.nn.architectures.syntonic_cnn import (
    PureSyntonicCNN2d as SyntonicCNN2d,
)
from syntonic.nn.architectures.syntonic_cnn import (
    PureSyntonicConv2d as SyntonicConv2d,
)
from syntonic.nn.architectures.syntonic_mlp import (
    PureDeepSyntonicMLP,
    PureSyntonicLinear,
    PureSyntonicMLP,
)
from syntonic.nn.architectures.syntonic_mlp import (
    PureSyntonicLinear as SyntonicLinear,
)
from syntonic.nn.architectures.syntonic_mlp import (
    PureSyntonicMLP as SyntonicMLP,
)
from syntonic.nn.architectures.syntonic_transformer import (
    PureDHTransformerLayer,
    PureSyntonicTransformer,
    PureSyntonicTransformerEncoder,
)
from syntonic.nn.layers.prime_syntony_gate import (
    PrimeSyntonyGate,
    SRTTransformerBlock,
    WindingAttention,
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
    "SyntonicCNN2d",
    "PureSyntonicConv1d",
    "PureSyntonicConv2d",
    "PureSyntonicCNN1d",
    "PureSyntonicCNN2d",
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
