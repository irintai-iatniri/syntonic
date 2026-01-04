"""
CRT Metrics - Syntony and Gnosis computation.

Provides:
- SyntonyComputer: Full S(Ψ) computation using D̂/Ĥ operators
- GnosisComputer: Gnosis layer (0-3) classification
- Quick variants: syntony_entropy, syntony_spectral
"""

from syntonic.crt.metrics.syntony import (
    SyntonyComputer,
    syntony_entropy,
    syntony_spectral,
    syntony_quick,
)

from syntonic.crt.metrics.gnosis import (
    GnosisComputer,
    default_gnosis_computer,
    K_D4,
)

__all__ = [
    'SyntonyComputer',
    'syntony_entropy',
    'syntony_spectral',
    'syntony_quick',
    'GnosisComputer',
    'default_gnosis_computer',
    'K_D4',
]
