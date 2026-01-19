"""
CKM Matrix - Quark mixing from golden ratio powers.

The CKM matrix elements emerge from powers of φ̂ = 1/φ with
correction factors involving the syntony deficit q.

Formulas (from phase5-spec §8.1):
    |V_us| = φ̂³(1-qφ)(1-q/4) = 0.2242 (exp: 0.2243)
    |V_cb| = q × 3/2 = 0.04109 (exp: 0.0412)
    |V_ub| = q × φ̂⁴ × (1-4q)(1+q/2) = 0.00361 (exp: 0.00361)

Jarlskog invariant:
    J_CP = q²/E* × (1-4q)(1-qφ²)(1-q/φ³) = 3.08 × 10⁻⁵

The Cabibbo angle θ_C ≈ φ̂² ≈ 0.38 rad ≈ 22° sets the scale
for inter-generation mixing.
"""

import math

from syntonic.exact import (
    E_STAR_NUMERIC,
    PHI,
    PHI_INVERSE,
    Q_DEFICIT_NUMERIC,
    get_correction_factor,
)


def V_us() -> float:
    """
    CKM element |V_us| (Cabibbo mixing).

    |V_us| = φ̂³(1-qφ)(1-q/4) = 0.2242
    (PDG: 0.2243 ± 0.0005)

    This is the most precisely measured CKM element and
    provides an excellent test of the SRT prediction.

    Returns:
        |V_us|
    """
    phi_hat = PHI_INVERSE.eval()  # φ̂ = 1/φ

    # Correction factors from hierarchy: C48 (qφ), C41 (q/4)
    return (
        phi_hat**3 * (1 - get_correction_factor(48)) * (1 - get_correction_factor(41))
    )


def V_cb() -> float:
    """
    CKM element |V_cb|.

    |V_cb| = q × 3/2 = 0.04109
    (PDG: 0.0412 ± 0.0008)

    The simple form q × 3/2 is remarkable - the CKM element
    is directly proportional to the syntony deficit.

    Returns:
        |V_cb|
    """
    return Q_DEFICIT_NUMERIC * 1.5


def V_ub() -> float:
    """
    CKM element |V_ub|.

    |V_ub| = q × φ̂⁴ × (1-4q)(1+q/2) = 0.00361
    (PDG: 0.00361 ± 0.00011)

    Returns:
        |V_ub|
    """
    phi_hat = PHI_INVERSE.eval()

    # Correction factors from hierarchy: C47 (q), C52 (4q), C45 (q/2)
    return (
        get_correction_factor(47)
        * phi_hat**4
        * (1 - get_correction_factor(52))
        * (1 + get_correction_factor(45))
    )


def V_cd() -> float:
    """
    CKM element |V_cd|.

    |V_cd| ≈ |V_us| from unitarity.

    Returns:
        |V_cd|
    """
    # From unitarity: |V_cd| ≈ |V_us|
    return V_us()


def V_cs() -> float:
    """
    CKM element |V_cs|.

    |V_cs| ≈ 1 - |V_us|²/2 from unitarity.

    Returns:
        |V_cs|
    """
    v_us = V_us()
    return 1 - v_us**2 / 2


def V_td() -> float:
    """
    CKM element |V_td|.

    |V_td| ≈ |V_ub| × (1 + corrections)

    Returns:
        |V_td|
    """
    phi_hat = PHI_INVERSE.eval()

    # V_td is slightly larger than V_ub
    # Correction factors from hierarchy: C52 (4q), C45 (q/2)
    return (
        get_correction_factor(52)
        * phi_hat**3
        * (1 - 3 * get_correction_factor(52))
        * (1 + get_correction_factor(45))
    )


def V_ts() -> float:
    """
    CKM element |V_ts|.

    |V_ts| ≈ |V_cb| from unitarity.

    Returns:
        |V_ts|
    """
    return V_cb()


def V_tb() -> float:
    """
    CKM element |V_tb|.

    |V_tb| ≈ 1 - |V_cb|²/2 from unitarity.

    Returns:
        |V_tb|
    """
    v_cb = V_cb()
    return 1 - v_cb**2 / 2


def V_ud() -> float:
    """
    CKM element |V_ud|.

    |V_ud| ≈ 1 - |V_us|²/2 from unitarity.

    Returns:
        |V_ud|
    """
    v_us = V_us()
    return math.sqrt(1 - v_us**2 - V_ub() ** 2)


def jarlskog_invariant() -> float:
    """
    Jarlskog invariant J_CP (measure of CP violation).

    J_CP = q²/E* × (1-4q)(1-qφ²)(1-q/φ³) = 3.08 × 10⁻⁵
    (PDG: (3.08 ± 0.15) × 10⁻⁵)

    The Jarlskog invariant is the unique measure of CP violation
    in the quark sector, appearing in all CP-violating observables.

    Returns:
        J_CP
    """
    phi = PHI.eval()

    # Correction factors from hierarchy: C52 (4q), C49 (qφ²), C40 (q/φ³)
    return (
        (Q_DEFICIT_NUMERIC**2 / E_STAR_NUMERIC)
        * (1 - get_correction_factor(52))
        * (1 - get_correction_factor(49))
        * (1 - get_correction_factor(40))
    )


def cabibbo_angle() -> float:
    """
    Cabibbo angle θ_C in radians.

    sin(θ_C) = |V_us| ≈ 0.2242
    θ_C ≈ 12.96° ≈ 0.226 rad

    Returns:
        θ_C in radians
    """
    return math.asin(V_us())


def cabibbo_angle_degrees() -> float:
    """
    Cabibbo angle θ_C in degrees.

    Returns:
        θ_C in degrees
    """
    return math.degrees(cabibbo_angle())


class CKMMatrix:
    """
    Complete CKM quark mixing matrix from SRT.

    All elements derived from {φ, q} with zero free parameters.

    Example:
        >>> ckm = CKMMatrix()
        >>> ckm.V_us()
        0.2242
        >>> ckm.matrix()
        [[0.974, 0.224, 0.004], [0.224, 0.973, 0.041], [0.009, 0.040, 0.999]]
    """

    def V_ud(self) -> float:
        return V_ud()

    def V_us(self) -> float:
        return V_us()

    def V_ub(self) -> float:
        return V_ub()

    def V_cd(self) -> float:
        return V_cd()

    def V_cs(self) -> float:
        return V_cs()

    def V_cb(self) -> float:
        return V_cb()

    def V_td(self) -> float:
        return V_td()

    def V_ts(self) -> float:
        return V_ts()

    def V_tb(self) -> float:
        return V_tb()

    def jarlskog_invariant(self) -> float:
        return jarlskog_invariant()

    def cabibbo_angle(self) -> float:
        """Cabibbo angle in radians."""
        return cabibbo_angle()

    def matrix(self) -> list:
        """
        Return full CKM matrix as 3x3 list of magnitudes.

        Returns:
            [[|V_ud|, |V_us|, |V_ub|],
             [|V_cd|, |V_cs|, |V_cb|],
             [|V_td|, |V_ts|, |V_tb|]]
        """
        return [
            [self.V_ud(), self.V_us(), self.V_ub()],
            [self.V_cd(), self.V_cs(), self.V_cb()],
            [self.V_td(), self.V_ts(), self.V_tb()],
        ]

    def all_parameters(self) -> dict:
        """Return all CKM parameters."""
        return {
            "V_ud": self.V_ud(),
            "V_us": self.V_us(),
            "V_ub": self.V_ub(),
            "V_cd": self.V_cd(),
            "V_cs": self.V_cs(),
            "V_cb": self.V_cb(),
            "V_td": self.V_td(),
            "V_ts": self.V_ts(),
            "V_tb": self.V_tb(),
            "J_CP": self.jarlskog_invariant(),
            "theta_C": self.cabibbo_angle(),
        }

    def __repr__(self) -> str:
        return "CKMMatrix(SRT-derived)"


__all__ = [
    "V_ud",
    "V_us",
    "V_ub",
    "V_cd",
    "V_cs",
    "V_cb",
    "V_td",
    "V_ts",
    "V_tb",
    "jarlskog_invariant",
    "cabibbo_angle",
    "cabibbo_angle_degrees",
    "CKMMatrix",
]
