"""
Syntonic Analysis - Tools for analyzing syntonic networks.

Provides:
- Archonic pattern detection
- Network health monitoring
- Escape mechanisms
- Visualization tools
"""

from syntonic.nn.analysis.archonic_detector import (
    ArchonicDetector,
    ArchonicReport,
    detect_archonic_pattern,
)
from syntonic.nn.analysis.escape import (
    EscapeMechanism,
    NoiseInjection,
    LearningRateShock,
)
from syntonic.nn.analysis.health import (
    NetworkHealth,
    SyntonyMonitor,
    HealthReport,
)
from syntonic.nn.analysis.visualization import (
    SyntonyViz,
    plot_syntony_history,
    plot_layer_syntonies,
    plot_archonic_regions,
)

__all__ = [
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
]
