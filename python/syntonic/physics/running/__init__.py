"""
Running Couplings Module - Renormalization group evolution.

Golden RG equations with the universal syntony deficit q.

Example:
    >>> from syntonic.physics.running import GoldenRG
    >>> rg = GoldenRG()
    >>> rg.gut_scale()
    1.0e15  # GeV
"""

from syntonic.physics.running.rg import (
    B1,
    B2,
    B3,
    GoldenRG,
    alpha_em_at_scale,
    alpha_running,
    alpha_s_at_scale,
    coupling_unification_check,
    gut_scale,
)

__all__ = [
    "B1",
    "B2",
    "B3",
    "alpha_running",
    "gut_scale",
    "alpha_em_at_scale",
    "alpha_s_at_scale",
    "coupling_unification_check",
    "GoldenRG",
]
