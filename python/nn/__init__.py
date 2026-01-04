"""
Syntonic Neural Networks - CRT-Native Deep Learning.

This module provides neural network architectures that embed the
DHSR (Differentiation-Harmonization-Syntony-Recursion) cycle
directly into deep learning.

Key Features:
- DHSR-structured layers (DifferentiationLayer, HarmonizationLayer, RecursionBlock)
- Syntonic loss functions (SyntonicLoss with syntony penalty)
- Syntony-aware optimizers (SyntonicAdam with lr modulation)
- Complete architectures (SyntonicMLP, SyntonicCNN, CRTTransformer)
- Archonic pattern detection and escape mechanisms
- Benchmarking and analysis tools

Core Insight:
    L_total = L_task + λ(1 - S_model) + μC_{iπ}

Where S_model is the model syntony, optimizing for both
task performance AND representational coherence.

Example:
    >>> import syntonic.nn as snn
    >>> model = snn.SyntonicMLP(784, [512, 256], 10)
    >>> optimizer = snn.SyntonicAdam(model.parameters(), lr=0.001)
    >>> criterion = snn.SyntonicLoss(nn.CrossEntropyLoss())
    >>>
    >>> for inputs, targets in dataloader:
    ...     outputs = model(inputs)
    ...     loss, metrics = criterion(outputs, targets, model)
    ...     optimizer.zero_grad()
    ...     loss.backward()
    ...     optimizer.step(syntony=model.syntony)
    >>>
    >>> print(f"Final syntony: {model.syntony:.4f}")

Source: CRT.md §12.2
"""

# Layers
from syntonic.nn.layers import (
    DifferentiationLayer,
    DifferentiationModule,
    HarmonizationLayer,
    HarmonizationModule,
    SyntonicGate,
    AdaptiveGate,
    RecursionBlock,
    DeepRecursionNet,
    SyntonicNorm,
    GoldenNorm,
)

# Loss Functions
from syntonic.nn.loss import (
    SyntonicLoss,
    LayerwiseSyntonicLoss,
    compute_activation_syntony,
    compute_network_syntony,
    SyntonyTracker,
    PhaseAlignmentLoss,
    compute_phase_alignment,
    IPiConstraint,
    GoldenPhaseScheduler,
    SyntonicRegularizer,
    GoldenDecay,
    SyntonyConstraint,
    ArchonicPenalty,
)

# Optimizers
from syntonic.nn.optim import (
    SyntonicAdam,
    AdaptiveSyntonicAdam,
    SyntonicSGD,
    SyntonicMomentum,
    GoldenScheduler,
    SyntonyCyclicScheduler,
    WarmupGoldenScheduler,
    SyntonyGradientModifier,
    GoldenClipping,
    ArchonicGradientEscape,
)

# Training
from syntonic.nn.training import (
    SyntonicTrainer,
    TrainingConfig,
    SyntonyCallback,
    ArchonicEarlyStop,
    SyntonyCheckpoint,
    MetricsLogger,
    TrainingMetrics,
    SyntonyMetrics,
    compute_epoch_metrics,
)

# Architectures
from syntonic.nn.architectures import (
    SyntonicMLP,
    SyntonicLinear,
    SyntonicConv2d,
    RecursionConvBlock,
    SyntonicCNN,
    SyntonicEmbedding,
    WindingEmbedding,
    PositionalEncoding,
    SyntonicAttention,
    GnosisAttention,
    MultiHeadSyntonicAttention,
    CRTTransformer,
    DHTransformerLayer,
    SyntonicTransformerEncoder,
    SyntonicTransformerDecoder,
)

# Analysis
from syntonic.nn.analysis import (
    ArchonicDetector,
    ArchonicReport,
    detect_archonic_pattern,
    EscapeMechanism,
    NoiseInjection,
    LearningRateShock,
    NetworkHealth,
    SyntonyMonitor,
    HealthReport,
    SyntonyViz,
    plot_syntony_history,
    plot_layer_syntonies,
    plot_archonic_regions,
)

# Benchmarks
from syntonic.nn.benchmarks import (
    BenchmarkSuite,
    BenchmarkResult,
    run_mnist_benchmark,
    run_cifar_benchmark,
    ConvergenceAnalyzer,
    ConvergenceMetrics,
    compare_convergence,
    AblationStudy,
    AblationResult,
    run_component_ablation,
)

# Constants
import math
PHI = (1 + math.sqrt(5)) / 2
Q_DEFICIT = 0.027395146920
S_TARGET = PHI - Q_DEFICIT

__all__ = [
    # Layers
    'DifferentiationLayer',
    'DifferentiationModule',
    'HarmonizationLayer',
    'HarmonizationModule',
    'SyntonicGate',
    'AdaptiveGate',
    'RecursionBlock',
    'DeepRecursionNet',
    'SyntonicNorm',
    'GoldenNorm',

    # Loss
    'SyntonicLoss',
    'LayerwiseSyntonicLoss',
    'compute_activation_syntony',
    'compute_network_syntony',
    'SyntonyTracker',
    'PhaseAlignmentLoss',
    'compute_phase_alignment',
    'IPiConstraint',
    'GoldenPhaseScheduler',
    'SyntonicRegularizer',
    'GoldenDecay',
    'SyntonyConstraint',
    'ArchonicPenalty',

    # Optimizers
    'SyntonicAdam',
    'AdaptiveSyntonicAdam',
    'SyntonicSGD',
    'SyntonicMomentum',
    'GoldenScheduler',
    'SyntonyCyclicScheduler',
    'WarmupGoldenScheduler',
    'SyntonyGradientModifier',
    'GoldenClipping',
    'ArchonicGradientEscape',

    # Training
    'SyntonicTrainer',
    'TrainingConfig',
    'SyntonyCallback',
    'ArchonicEarlyStop',
    'SyntonyCheckpoint',
    'MetricsLogger',
    'TrainingMetrics',
    'SyntonyMetrics',
    'compute_epoch_metrics',

    # Architectures
    'SyntonicMLP',
    'SyntonicLinear',
    'SyntonicConv2d',
    'RecursionConvBlock',
    'SyntonicCNN',
    'SyntonicEmbedding',
    'WindingEmbedding',
    'PositionalEncoding',
    'SyntonicAttention',
    'GnosisAttention',
    'MultiHeadSyntonicAttention',
    'CRTTransformer',
    'DHTransformerLayer',
    'SyntonicTransformerEncoder',
    'SyntonicTransformerDecoder',

    # Analysis
    'ArchonicDetector',
    'ArchonicReport',
    'detect_archonic_pattern',
    'EscapeMechanism',
    'NoiseInjection',
    'LearningRateShock',
    'NetworkHealth',
    'SyntonyMonitor',
    'HealthReport',
    'SyntonyViz',
    'plot_syntony_history',
    'plot_layer_syntonies',
    'plot_archonic_regions',

    # Benchmarks
    'BenchmarkSuite',
    'BenchmarkResult',
    'run_mnist_benchmark',
    'run_cifar_benchmark',
    'ConvergenceAnalyzer',
    'ConvergenceMetrics',
    'compare_convergence',
    'AblationStudy',
    'AblationResult',
    'run_component_ablation',

    # Constants
    'PHI',
    'Q_DEFICIT',
    'S_TARGET',
]
