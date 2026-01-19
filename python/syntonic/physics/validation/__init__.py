"""
Validation Module - Compare SRT predictions to PDG values.

Provides utilities for validating all Standard Model predictions
against experimental data from the Particle Data Group (PDG).

Example:
    >>> from syntonic.physics.validation import validate_all, PDG_VALUES
    >>> predictions = {'m_e': 0.511, 'm_H': 125.25}
    >>> results = validate_all(predictions)
    >>> results['m_e']['status']
    'PASS'
"""

from syntonic.physics.validation.pdg import (
    PDG_VALUES,
    summary_report,
    validate_all,
    validate_prediction,
)

__all__ = [
    "PDG_VALUES",
    "validate_prediction",
    "validate_all",
    "summary_report",
]
