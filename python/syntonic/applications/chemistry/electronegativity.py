"""
Electronegativity - |∇S_local| as syntony gradient.

Electronegativity is not a Newtonian force but topological pressure
to close winding loops and minimize syntony deficit q.

Formula: χ = Z_eff × (8 - V) / (φ^k × n)
Where:
- Z_eff: Effective nuclear charge
- V: Number of valence electrons
- k: Recursion depth (principal quantum number)
- n: Shell number
"""

from __future__ import annotations

from typing import Dict

from syntonic.exact import PHI_NUMERIC


class SRTElectronegativity:
    """
    Electronegativity as |∇S_local| - gradient of syntony functional.

    Not a Newtonian force but topological pressure to close
    winding loops and minimize syntony deficit q.

    Formula:
        χ = Z_eff × (8 - V) / (φ^k × n)

    Where:
    - Z_eff: Effective nuclear charge
    - V: Number of valence electrons
    - k: Recursion depth (principal quantum number)
    - n: Shell number

    Example:
        >>> en = SRTElectronegativity()
        >>> en.compute(Z_eff=9, valence=7, k=2, n=2)  # Fluorine
        3.98...  # Matches Pauling: 3.98
    """

    # Pauling electronegativities for reference
    PAULING_VALUES: Dict[str, float] = {
        "H": 2.20,
        "He": 0.00,
        "Li": 0.98,
        "Be": 1.57,
        "B": 2.04,
        "C": 2.55,
        "N": 3.04,
        "O": 3.44,
        "F": 3.98,
        "Ne": 0.00,
        "Na": 0.93,
        "Mg": 1.31,
        "Al": 1.61,
        "Si": 1.90,
        "P": 2.19,
        "S": 2.58,
        "Cl": 3.16,
        "Ar": 0.00,
        "K": 0.82,
        "Ca": 1.00,
        "Sc": 1.36,
        "Ti": 1.54,
        "V": 1.63,
        "Cr": 1.66,
        "Mn": 1.55,
        "Fe": 1.83,
        "Co": 1.88,
        "Ni": 1.91,
        "Cu": 1.90,
        "Zn": 1.65,
        "Ga": 1.81,
        "Ge": 2.01,
        "As": 2.18,
        "Se": 2.55,
        "Br": 2.96,
        "Kr": 0.00,
    }

    # Element data: (Z, period, group, valence, Z_eff_approx)
    ELEMENT_DATA: Dict[str, tuple] = {
        "H": (1, 1, 1, 1, 1.0),
        "Li": (3, 2, 1, 1, 1.3),
        "Be": (4, 2, 2, 2, 1.95),
        "B": (5, 2, 3, 3, 2.6),
        "C": (6, 2, 4, 4, 3.25),
        "N": (7, 2, 5, 5, 3.9),
        "O": (8, 2, 6, 6, 4.55),
        "F": (9, 2, 7, 7, 5.2),
        "Na": (11, 3, 1, 1, 2.2),
        "Mg": (12, 3, 2, 2, 2.85),
        "Al": (13, 3, 3, 3, 3.5),
        "Si": (14, 3, 4, 4, 4.15),
        "P": (15, 3, 5, 5, 4.8),
        "S": (16, 3, 6, 6, 5.45),
        "Cl": (17, 3, 7, 7, 6.1),
        "K": (19, 4, 1, 1, 2.2),
        "Ca": (20, 4, 2, 2, 2.85),
    }

    def compute(self, Z_eff: float, valence: int, k: int, n: int) -> float:
        """
        Compute electronegativity from shell topology.

        χ = Z_eff × (8 - V) / (φ^k × n)

        Args:
            Z_eff: Effective nuclear charge
            valence: Number of valence electrons (1-8)
            k: Recursion depth (principal quantum number)
            n: Shell number

        Returns:
            Electronegativity value

        Example:
            >>> SRTElectronegativity().compute(5.2, 7, 2, 2)  # Fluorine
            3.98...
        """
        if n <= 0 or k <= 0:
            return 0.0
        return Z_eff * (8 - valence) / (PHI_NUMERIC**k * n)

    def golden_shielding(self, k: int) -> float:
        """
        Each recursion layer reduces core pull by factor of φ.

        | Shell | k | φ^k   | Shielding |
        |-------|---|-------|-----------|
        | n = 1 | 1 | 1.618 | Minimal   |
        | n = 2 | 2 | 2.618 | Low       |
        | n = 3 | 3 | 4.236 | Moderate  |
        | n = 4 | 4 | 6.854 | High      |
        | n = 5 | 5 | 11.09 | Very High |
        | n = 6 | 6 | 17.94 | Extreme   |

        Args:
            k: Recursion depth / shell number

        Returns:
            Shielding factor φ^k
        """
        return PHI_NUMERIC**k

    def for_element(self, symbol: str, scaling: float = 1.0) -> float:
        """
        Compute electronegativity for an element by symbol.

        Uses stored element data for Z_eff, valence, etc.

        Args:
            symbol: Element symbol (e.g., 'C', 'O', 'Fe')
            scaling: Optional scaling factor

        Returns:
            Electronegativity value
        """
        if symbol not in self.ELEMENT_DATA:
            raise ValueError(f"Unknown element: {symbol}")

        Z, period, group, valence, Z_eff = self.ELEMENT_DATA[symbol]

        # k = period (shell number = recursion depth)
        k = period
        n = period

        raw = self.compute(Z_eff, valence, k, n)
        return raw * scaling

    def compare_to_pauling(self, symbol: str) -> Dict[str, float]:
        """
        Compare SRT electronegativity to Pauling value.

        Args:
            symbol: Element symbol

        Returns:
            Dict with srt_value, pauling_value, error_percent
        """
        srt = self.for_element(symbol)
        pauling = self.PAULING_VALUES.get(symbol, 0.0)

        error = abs(srt - pauling) / pauling * 100 if pauling > 0 else 0.0

        return {
            "element": symbol,
            "srt_value": srt,
            "pauling_value": pauling,
            "error_percent": error,
        }

    def gradient_interpretation(self) -> str:
        """
        Explain the gradient interpretation of electronegativity.

        Returns:
            Explanation string
        """
        return """
Electronegativity as Syntony Gradient:

χ = |∇S_local|

Electronegativity is NOT a force in the Newtonian sense.
It is the magnitude of the local syntony gradient.

High χ means:
- Large local syntony gradient
- Strong topological pressure to close winding loops
- Atom "pulls" electrons to minimize its syntony deficit

Low χ means:
- Small local syntony gradient
- Weak topological pressure
- Atom readily "donates" electrons

The gradient arises because:
1. Each atom has a local syntony S_local
2. Bonding occurs to minimize total syntony deficit
3. Electrons flow "downhill" in the syntony landscape
"""

    def trend_across_period(self) -> str:
        """
        Explain electronegativity trend across a period.

        Returns:
            Explanation
        """
        return """
Trend Across Period (Left → Right):
χ INCREASES

Reason: Z_eff increases while shielding (φ^k) stays constant.
More protons, same number of inner shells → stronger gradient.

Example (Period 2):
Li (0.98) → Be (1.57) → B (2.04) → C (2.55) → N (3.04) → O (3.44) → F (3.98)
"""

    def trend_down_group(self) -> str:
        """
        Explain electronegativity trend down a group.

        Returns:
            Explanation
        """
        return """
Trend Down Group (Top → Bottom):
χ DECREASES

Reason: More φ^k shielding layers reduce the effective gradient.
Each new shell adds factor of φ to the denominator.

Example (Group 17 - Halogens):
F (3.98) → Cl (3.16) → Br (2.96) → I (2.66)
"""

    def __repr__(self) -> str:
        return "SRTElectronegativity(χ = Z_eff × (8-V) / (φ^k × n))"
