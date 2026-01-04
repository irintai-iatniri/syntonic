"""
Quantum Hall Effect - T⁴ winding made visible.

The integer n in σ_xy = (e²/h) × n is literally the winding number n₇.
Conductance is quantized because windings are quantized.

FQHE fractions are Fibonacci ratios: ν = F_n / F_{n+2}
"""

from __future__ import annotations
from typing import List, Tuple
import math

from syntonic.exact import PHI_NUMERIC, fibonacci


class QuantumHallEffect:
    """
    Quantum Hall as T⁴ winding made visible.

    The integer n in σ_xy = (e²/h) × n is literally the winding number n₇.
    Conductance is quantized because windings are quantized.

    For FQHE, the filling fractions follow Fibonacci ratios:
        ν = F_n / F_{n+2}

    | n | Fraction | Value |
    |---|----------|-------|
    | 1 | 1/3      | 0.333 |
    | 2 | 2/5      | 0.400 |
    | 3 | 3/8      | 0.375 |
    | 4 | 5/13     | 0.385 |
    | ∞ | → φ⁻²   | 0.382 |

    Attributes:
        E_CHARGE: Electron charge in C
        H_PLANCK: Planck constant in J·s
        CONDUCTANCE_QUANTUM: e²/h in S

    Example:
        >>> qhe = QuantumHallEffect()
        >>> qhe.hall_conductance(1)
        3.874e-5  # Siemens (1 × e²/h)
        >>> qhe.fqhe_fractions(5)
        [(1, 3, 0.333...), (2, 5, 0.4), ...]
    """

    E_CHARGE = 1.602176634e-19  # C (exact)
    H_PLANCK = 6.62607015e-34  # J·s (exact)
    CONDUCTANCE_QUANTUM = E_CHARGE**2 / H_PLANCK  # e²/h ≈ 3.874e-5 S

    def hall_conductance(self, n7: int) -> float:
        """
        Integer Quantum Hall conductance.

        σ_xy = (e²/h) × n₇

        The winding index n₇ directly determines conductance.

        Args:
            n7: Winding number (Landau level index)

        Returns:
            Hall conductance in Siemens
        """
        return self.CONDUCTANCE_QUANTUM * n7

    def hall_resistance(self, n7: int) -> float:
        """
        Integer Quantum Hall resistance.

        R_H = h / (e² × n₇)

        Args:
            n7: Winding number

        Returns:
            Hall resistance in Ohms
        """
        if n7 == 0:
            return float('inf')
        return self.H_PLANCK / (self.E_CHARGE**2 * n7)

    def fqhe_fractions(self, max_n: int = 10) -> List[Tuple[int, int, float]]:
        """
        Fractional QHE: Filling fractions ARE Fibonacci ratios!

        ν = F_n / F_{n+2}

        | n | Fraction | Value |
        |---|----------|-------|
        | 1 | 1/3      | 0.333 |
        | 2 | 2/5      | 0.400 |
        | 3 | 3/8      | 0.375 |
        | 4 | 5/13     | 0.385 |
        | ∞ | → φ⁻²   | 0.382 |

        The principal FQHE fractions follow golden ratio recursion!

        Args:
            max_n: Maximum Fibonacci index

        Returns:
            List of (numerator, denominator, value) tuples
        """
        fractions = []
        for n in range(1, max_n):
            F_n = fibonacci(n)
            F_n2 = fibonacci(n + 2)
            fractions.append((F_n, F_n2, F_n / F_n2))
        return fractions

    def fqhe_conductance(self, F_n: int, F_n2: int) -> float:
        """
        Fractional Quantum Hall conductance.

        σ_xy = (e²/h) × (F_n / F_{n+2})

        Args:
            F_n: Fibonacci numerator
            F_n2: Fibonacci denominator (F_{n+2})

        Returns:
            Hall conductance in Siemens
        """
        return self.CONDUCTANCE_QUANTUM * F_n / F_n2

    def laughlin_fraction(self, m: int) -> Tuple[int, int, float]:
        """
        Laughlin state filling fraction.

        ν = 1/(2m + 1) for odd denominators

        These are the most robust FQHE states.

        Args:
            m: Laughlin index (0, 1, 2, ...)

        Returns:
            (numerator, denominator, value)
        """
        denominator = 2 * m + 1
        return (1, denominator, 1 / denominator)

    def jain_sequence(self, p: int, sign: int = 1) -> List[Tuple[int, int, float]]:
        """
        Jain sequence of composite fermion fractions.

        ν = p / (2p ± 1)

        Args:
            p: Maximum p value
            sign: +1 or -1

        Returns:
            List of (numerator, denominator, value)
        """
        fractions = []
        for n in range(1, p + 1):
            denominator = 2 * n + sign
            if denominator > 0:
                fractions.append((n, denominator, n / denominator))
        return fractions

    def asymptotic_fraction(self) -> float:
        """
        Asymptotic FQHE fraction as n → ∞.

        lim_{n→∞} F_n / F_{n+2} = 1/φ² ≈ 0.382

        Returns:
            Asymptotic fraction
        """
        return 1 / PHI_NUMERIC**2

    def topological_origin(self) -> str:
        """
        Explain the topological origin of quantization.

        Returns:
            Explanation string
        """
        return """
Topological Origin of Quantum Hall Quantization:

The Hall conductance σ_xy = (e²/h) × n is quantized because:

1. The electron winding around the magnetic flux is topological
2. The winding number n₇ ∈ ℤ (integer by definition)
3. Each Landau level corresponds to a specific winding number
4. The conductance per winding is exactly e²/h (universal constant)

For FQHE, the composite fermions have fractional effective charge,
but the fractions follow Fibonacci ratios because:
- Fibonacci numbers are the attractor of the golden recursion
- The golden ratio φ is the fixed point of the DHSR cycle
- Physical fractions approach 1/φ² ≈ 0.382 asymptotically
"""

    def __repr__(self) -> str:
        return f"QuantumHallEffect(G_quantum={self.CONDUCTANCE_QUANTUM:.4e} S)"
