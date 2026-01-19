import math

import torch

from syntonic.srt.constants import (
    E_STAR_NUMERIC,
    correction_factor,
)
from syntonic.srt.constants import (
    PHI_NUMERIC as PHI,
)
from syntonic.srt.constants import (
    Q_DEFICIT_NUMERIC as Q,
)


def get_resonance_value(base_scale="particle", n_winding=1, corrections=None):
    """
    Calculates the EXACT geometric value based on Part IV formulas.
    Ex: Top Mass = v * e^(2*phi) * corrections
    """
    if corrections is None:
        corrections = []

    # Base Scalar (The "v" or "E*")
    if base_scale == "spectral":
        val = E_STAR_NUMERIC
    elif base_scale == "higgs":
        val = 246.0  # Vacuum expectation
    else:
        val = 1.0

    # The N-winding (Geometric scaling)
    val *= n_winding

    # Apply Nested Corrections (The "(1 +/- q/N)" factors)
    for struct_name, sign in corrections:
        # Uses your existing correction_factor function from exact/__init__.py
        factor = correction_factor(struct_name, sign)
        val *= factor

    return val


def syntonic_uniform_(tensor, mode="golden"):
    """
    Replaces torch.nn.init.xavier_uniform_
    Initializes weights bounded by the Golden Ratio rather than arbitrary variance.
    """
    if mode == "golden":
        # Bound weights by [-phi, phi] normalized by root(dim)
        limit = PHI / math.sqrt(tensor.shape[1])
        with torch.no_grad():
            tensor.uniform_(-limit, limit)
    elif mode == "syntony_deficit":
        # Initialize small weights to Q (the 'vacuum noise' level)
        with torch.no_grad():
            tensor.fill_(Q)
