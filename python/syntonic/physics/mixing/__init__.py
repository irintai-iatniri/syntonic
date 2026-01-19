"""
Mixing Matrices Module - CKM and PMNS from golden ratio geometry.

Submodules:
    ckm: CKM quark mixing matrix elements
    pmns: PMNS neutrino mixing matrix elements

The mixing angles emerge from powers of the golden conjugate φ̂ = 1/φ
with correction factors from the syntony deficit q.

Example:
    >>> from syntonic.physics.mixing import CKMMatrix, PMNSMatrix
    >>> ckm = CKMMatrix()
    >>> ckm.V_us()
    0.2242
    >>> pmns = PMNSMatrix()
    >>> pmns.theta_12()
    33.44  # degrees
"""

from syntonic.physics.mixing.ckm import (
    CKMMatrix,
    V_cb,
    V_tb,
    V_td,
    V_ts,
    V_ub,
    V_us,
    jarlskog_invariant,
)
from syntonic.physics.mixing.pmns import (
    PMNSMatrix,
    delta_CP,
    theta_12,
    theta_13,
    theta_23,
)

__all__ = [
    # CKM elements
    "V_us",
    "V_cb",
    "V_ub",
    "V_td",
    "V_ts",
    "V_tb",
    "jarlskog_invariant",
    "CKMMatrix",
    # PMNS elements
    "theta_12",
    "theta_23",
    "theta_13",
    "delta_CP",
    "PMNSMatrix",
]
