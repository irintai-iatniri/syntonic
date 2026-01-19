"""
CRT Operators - DHSR operator implementations.

Provides:
- OperatorBase: Abstract base class for all operators
- FourierProjector: Fourier mode projection
- DampingProjector: High-frequency damping
- LaplacianOperator: Discrete Laplacian ∇²
- DifferentiationOperator: D̂ operator
- HarmonizationOperator: Ĥ operator
- RecursionOperator: R̂ = Ĥ ∘ D̂
"""

from syntonic.crt.operators.base import OperatorBase
from syntonic.crt.operators.differentiation import (
    DifferentiationOperator,
    default_differentiation_operator,
)
from syntonic.crt.operators.gnosis import (
    K_D4,
    GnosisComputer,
    default_gnosis_computer,
)
from syntonic.crt.operators.harmonization import (
    HarmonizationOperator,
    default_harmonization_operator,
)
from syntonic.crt.operators.projectors import (
    DampingProjector,
    FourierProjector,
    LaplacianOperator,
    create_damping_cascade,
    create_mode_projectors,
)
from syntonic.crt.operators.recursion import (
    RecursionOperator,
    default_recursion_operator,
)
from syntonic.crt.operators.syntony import (
    SyntonyComputer,
    syntony_entropy,
    syntony_quick,
    syntony_spectral,
)

__all__ = [
    "OperatorBase",
    "FourierProjector",
    "DampingProjector",
    "LaplacianOperator",
    "create_mode_projectors",
    "create_damping_cascade",
    "DifferentiationOperator",
    "default_differentiation_operator",
    "HarmonizationOperator",
    "default_harmonization_operator",
    "RecursionOperator",
    "default_recursion_operator",
    "SyntonyComputer",
    "syntony_entropy",
    "syntony_spectral",
    "syntony_quick",
    "GnosisComputer",
    "default_gnosis_computer",
    "K_D4",
]
