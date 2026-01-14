"""
Syntonic Architectures - Complete neural network architectures.

Pre-built architectures that embed the DHSR cycle throughout:
- SyntonicMLP: Fully-connected with RecursionBlocks
- SyntonicConv: Convolutional with recursion
- CRTTransformer: Transformer with syntonic attention
"""

from syntonic.nn.architectures.syntonic_mlp_pure import (
    PureSyntonicMLP as SyntonicMLP,
    PureSyntonicLinear as SyntonicLinear,
)
from syntonic.nn.architectures.syntonic_mlp_pure import (
    PureSyntonicMLP,
    PureSyntonicLinear,
    PureDeepSyntonicMLP,
)
from syntonic.nn.architectures.syntonic_cnn_pure import (
    PureSyntonicConv2d as SyntonicConv2d,
    PureSyntonicCNN1d as RecursionConvBlock,
    PureSyntonicCNN1d as SyntonicCNN,
)
from syntonic.nn.architectures.syntonic_cnn_pure import (
    PureSyntonicConv1d,
    PureSyntonicConv2d,
    PureSyntonicCNN1d,
)
from syntonic.nn.architectures.embeddings_pure import (
    PureSyntonicEmbedding as SyntonicEmbedding,
    PureWindingEmbedding as WindingEmbedding,
    PurePositionalEncoding as PositionalEncoding,
)
from syntonic.nn.architectures.embeddings_pure import (
    PureSyntonicEmbedding,
    PureWindingEmbedding,
    PurePositionalEncoding,
)
from syntonic.nn.architectures.syntonic_attention_pure import (
    PureSyntonicAttention as SyntonicAttention,
    PureSyntonicAttention as GnosisAttention,
    PureMultiHeadSyntonicAttention as MultiHeadSyntonicAttention,
)
from syntonic.nn.architectures.syntonic_attention_pure import (
    PureSyntonicAttention,
    PureMultiHeadSyntonicAttention,
)
from syntonic.nn.architectures.syntonic_transformer_pure import (
    PureDHTransformerLayer,
    PureSyntonicTransformerEncoder,
    PureSyntonicTransformer,
)

__all__ = [
    # MLP
    'SyntonicMLP',
    'SyntonicLinear',
    'PureSyntonicMLP',
    'PureSyntonicLinear',
    'PureDeepSyntonicMLP',
    # CNN
    'SyntonicConv2d',
    'RecursionConvBlock',
    'SyntonicCNN',
    'PureSyntonicConv1d',
    'PureSyntonicCNN1d',
    # Embeddings
    'SyntonicEmbedding',
    'WindingEmbedding',
    'PositionalEncoding',
    'PureSyntonicEmbedding',
    'PureWindingEmbedding',
    'PurePositionalEncoding',
    # Attention
    'SyntonicAttention',
    'GnosisAttention',
    'MultiHeadSyntonicAttention',
    'PureSyntonicAttention',
    'PureMultiHeadSyntonicAttention',
    # Transformer (Pure only - PyTorch version removed)
    'PureDHTransformerLayer',
    'PureSyntonicTransformerEncoder',
    'PureSyntonicTransformer',
]

