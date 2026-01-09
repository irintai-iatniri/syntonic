"""
Syntonic Architectures - Complete neural network architectures.

Pre-built architectures that embed the DHSR cycle throughout:
- SyntonicMLP: Fully-connected with RecursionBlocks
- SyntonicConv: Convolutional with recursion
- CRTTransformer: Transformer with syntonic attention
"""

from syntonic.nn.architectures.syntonic_mlp import (
    SyntonicMLP,
    SyntonicLinear,
)
from syntonic.nn.architectures.syntonic_mlp_pure import (
    PureSyntonicMLP,
    PureSyntonicLinear,
    PureDeepSyntonicMLP,
)
from syntonic.nn.architectures.syntonic_cnn import (
    SyntonicConv2d,
    RecursionConvBlock,
    SyntonicCNN,
)
from syntonic.nn.architectures.embeddings import (
    SyntonicEmbedding,
    WindingEmbedding,
    PositionalEncoding,
)
from syntonic.nn.architectures.embeddings_pure import (
    PureSyntonicEmbedding,
    PureWindingEmbedding,
    PurePositionalEncoding,
)
from syntonic.nn.architectures.syntonic_attention import (
    SyntonicAttention,
    GnosisAttention,
    MultiHeadSyntonicAttention,
)
from syntonic.nn.architectures.syntonic_transformer import (
    CRTTransformer,
    DHTransformerLayer,
    SyntonicTransformerEncoder,
    SyntonicTransformerDecoder,
)

__all__ = [
    'SyntonicMLP',
    'SyntonicLinear',
    # Pure versions
    'PureSyntonicMLP',
    'PureSyntonicLinear',
    'PureDeepSyntonicMLP',
    'SyntonicConv2d',
    'RecursionConvBlock',
    'SyntonicCNN',
    'SyntonicEmbedding',
    'WindingEmbedding',
    'PositionalEncoding',
    # Pure embedding versions
    'PureSyntonicEmbedding',
    'PureWindingEmbedding',
    'PurePositionalEncoding',
    'SyntonicAttention',
    'GnosisAttention',
    'MultiHeadSyntonicAttention',
    'CRTTransformer',
    'DHTransformerLayer',
    'SyntonicTransformerEncoder',
    'SyntonicTransformerDecoder',
]
