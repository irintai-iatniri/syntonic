"""
Hypercomplex number systems for Syntonic.

This module provides Quaternions and Octonions for representing
rotations, orientations, and higher-dimensional algebraic structures.

Quaternions (4D):
    - Non-commutative: q1 * q2 != q2 * q1
    - Associative: (q1 * q2) * q3 == q1 * (q2 * q3)
    - Hamilton product with i² = j² = k² = ijk = -1

Octonions (8D):
    - Non-commutative AND non-associative
    - Cayley-Dickson construction from quaternions
    - Related to exceptional Lie groups (G₂, E₈)

Usage:
    >>> import syntonic as syn
    >>> q = syn.quaternion(1, 2, 3, 4)  # 1 + 2i + 3j + 4k
    >>> q.norm()
    5.477225575051661
    >>> syn.hypercomplex.I * syn.hypercomplex.J == syn.hypercomplex.K
    True
"""

from syntonic.core import Octonion, Quaternion

# Unit quaternions (basis elements)
I = Quaternion(0.0, 1.0, 0.0, 0.0)  # i
J = Quaternion(0.0, 0.0, 1.0, 0.0)  # j
K = Quaternion(0.0, 0.0, 0.0, 1.0)  # k

# Unit octonion basis elements
E0 = Octonion(1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)  # 1
E1 = Octonion(0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)  # e₁
E2 = Octonion(0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0)  # e₂
E3 = Octonion(0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0)  # e₃
E4 = Octonion(0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0)  # e₄
E5 = Octonion(0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0)  # e₅
E6 = Octonion(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0)  # e₆
E7 = Octonion(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0)  # e₇


def quaternion(a, b=0.0, c=0.0, d=0.0):
    """
    Create a quaternion a + bi + cj + dk.

    Args:
        a: Real (scalar) part
        b: i component
        c: j component
        d: k component

    Returns:
        Quaternion object

    Examples:
        >>> q = quaternion(1, 2, 3, 4)  # 1 + 2i + 3j + 4k
        >>> q.real  # property, not method
        1.0
        >>> q.norm()
        5.477225575051661
    """
    return Quaternion(float(a), float(b), float(c), float(d))


def octonion(e0, e1=0.0, e2=0.0, e3=0.0, e4=0.0, e5=0.0, e6=0.0, e7=0.0):
    """
    Create an octonion e0 + e1·e₁ + e2·e₂ + ... + e7·e₇.

    Args:
        e0: Real (scalar) part
        e1-e7: Imaginary basis coefficients

    Returns:
        Octonion object

    Note:
        Octonions are non-associative: (a*b)*c != a*(b*c) in general.
        Use Octonion.associator(a, b, c) to measure non-associativity.

    Examples:
        >>> o = octonion(1, 2, 3, 4, 5, 6, 7, 8)
        >>> o.real()
        1.0
        >>> o.norm()
        14.2828568570857
    """
    return Octonion(
        float(e0),
        float(e1),
        float(e2),
        float(e3),
        float(e4),
        float(e5),
        float(e6),
        float(e7),
    )


__all__ = [
    # Classes
    "Quaternion",
    "Octonion",
    # Factory functions
    "quaternion",
    "octonion",
    # Quaternion basis
    "I",
    "J",
    "K",
    # Octonion basis
    "E0",
    "E1",
    "E2",
    "E3",
    "E4",
    "E5",
    "E6",
    "E7",
]
