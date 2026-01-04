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
    V_us,
    V_cb,
    V_ub,
    V_td,
    V_ts,
    V_tb,
    jarlskog_invariant,
    CKMMatrix,
)

from syntonic.physics.mixing.pmns import (
    theta_12,
    theta_23,
    theta_13,
    delta_CP,
    PMNSMatrix,
)

__all__ = [
    # CKM elements
    'V_us',
    'V_cb',
    'V_ub',
    'V_td',
    'V_ts',
    'V_tb',
    'jarlskog_invariant',
    'CKMMatrix',
    # PMNS elements
    'theta_12',
    'theta_23',
    'theta_13',
    'delta_CP',
    'PMNSMatrix',
]
