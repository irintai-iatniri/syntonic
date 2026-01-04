"""
Periodic Table - Structure from T⁴ winding topology.

Shell capacity = 2n² emerges from winding state counting:
- |n|² = 0: 1 state × 2 spin = 2 (s orbital)
- |n|² = 1: 3 states × 2 spin = 6 (p orbitals)
- |n|² = 2: 5 states × 2 spin = 10 (d orbitals)
- |n|² = 3: 7 states × 2 spin = 14 (f orbitals)
"""

from __future__ import annotations
from typing import List, Dict, Any, Optional

from syntonic.exact import PHI_NUMERIC


class PeriodicTable:
    """
    Periodic table structure from T⁴ winding topology.

    Shell capacity = 2n² emerges from winding state counting:
    - |n|² = 0: 1 state × 2 spin = 2 (s orbital)
    - |n|² = 1: 3 states × 2 spin = 6 (p orbitals)
    - |n|² = 2: 5 states × 2 spin = 10 (d orbitals)
    - |n|² = 3: 7 states × 2 spin = 14 (f orbitals)

    The 2n² rule is NOT empirical—it follows from counting
    winding states on T⁴ with given |n|² = n.

    Example:
        >>> pt = PeriodicTable()
        >>> pt.shell_capacity(3)
        18
        >>> pt.period_lengths()
        [2, 8, 8, 18, 18, 32, 32]
    """

    # Orbital capacities from winding counting
    ORBITAL_CAPACITIES = {
        's': 2,   # 1 × 2 = 2
        'p': 6,   # 3 × 2 = 6
        'd': 10,  # 5 × 2 = 10
        'f': 14,  # 7 × 2 = 14
    }

    # Period structure
    PERIODS = {
        1: ['1s'],
        2: ['2s', '2p'],
        3: ['3s', '3p'],
        4: ['4s', '3d', '4p'],
        5: ['5s', '4d', '5p'],
        6: ['6s', '4f', '5d', '6p'],
        7: ['7s', '5f', '6d', '7p'],
    }

    def shell_capacity(self, n: int) -> int:
        """
        2n² electrons per shell.

        Args:
            n: Principal quantum number

        Returns:
            Maximum electrons in shell n
        """
        return 2 * n**2

    def subshell_capacity(self, l: int) -> int:
        """
        2(2l + 1) electrons per subshell.

        l = 0 (s): 2
        l = 1 (p): 6
        l = 2 (d): 10
        l = 3 (f): 14

        Args:
            l: Angular momentum quantum number

        Returns:
            Maximum electrons in subshell
        """
        return 2 * (2 * l + 1)

    def period_lengths(self) -> List[int]:
        """
        Period lengths 2, 8, 8, 18, 18, 32, 32 from T⁴ geometry.

        The pattern arises from the filling order with d and f blocks.

        Returns:
            List of period lengths
        """
        return [2, 8, 8, 18, 18, 32, 32]

    def aufbau_order(self) -> List[str]:
        """
        Electron filling order (Aufbau principle).

        Order follows (n + l) rule, then n for ties.
        This order emerges from energy minimization in the
        syntony landscape.

        Returns:
            List of orbitals in filling order
        """
        return [
            '1s',
            '2s', '2p',
            '3s', '3p',
            '4s', '3d', '4p',
            '5s', '4d', '5p',
            '6s', '4f', '5d', '6p',
            '7s', '5f', '6d', '7p',
        ]

    def electron_configuration(self, Z: int) -> str:
        """
        Generate electron configuration for atomic number Z.

        Args:
            Z: Atomic number

        Returns:
            Electron configuration string
        """
        order = self.aufbau_order()
        capacities = {'s': 2, 'p': 6, 'd': 10, 'f': 14}

        config = []
        remaining = Z

        for orbital in order:
            if remaining <= 0:
                break
            n = int(orbital[0])
            l_letter = orbital[1]
            capacity = capacities[l_letter]

            electrons = min(remaining, capacity)
            if electrons > 0:
                config.append(f"{orbital}{electrons}")
            remaining -= electrons

        return ' '.join(config)

    def valence_electrons(self, Z: int) -> int:
        """
        Count valence electrons for element with atomic number Z.

        Simplified: counts electrons in outermost s and p orbitals.

        Args:
            Z: Atomic number

        Returns:
            Number of valence electrons
        """
        config = self.electron_configuration(Z)
        orbitals = config.split()

        # Find highest n
        max_n = 0
        for orb in orbitals:
            n = int(orb[0])
            if n > max_n:
                max_n = n

        # Sum electrons in orbitals with n = max_n
        valence = 0
        for orb in orbitals:
            n = int(orb[0])
            if n == max_n and orb[1] in ['s', 'p']:
                # Extract electron count
                count = int(orb[2:]) if len(orb) > 2 else 0
                valence += count

        return valence

    def electronegativity_trend(self, direction: str) -> str:
        """
        Explain electronegativity trends.

        | Trend | Direction | Winding Explanation |
        |-------|-----------|---------------------|
        | Across period → | χ increases | Z_eff increases, same shielding |
        | Down group ↓ | χ decreases | More φ^k shielding layers |
        | Noble gases | χ ≈ 0 | Closed shells, no deficit |

        Args:
            direction: 'across', 'down', or 'noble'

        Returns:
            Explanation of trend
        """
        trends = {
            'across': "χ increases (Z_eff increases, same shielding)",
            'down': "χ decreases (more φ^k shielding layers)",
            'noble': "χ ≈ 0 (closed shells, no syntony deficit)"
        }
        return trends.get(direction, "Unknown direction")

    def atomic_radius_trend(self) -> str:
        """
        Explain atomic radius trends.

        Returns:
            Explanation
        """
        return """
Atomic Radius Trends:

Across period (→): DECREASES
- Z increases but same shell
- Greater nuclear pull contracts orbitals
- φ^k shielding constant

Down group (↓): INCREASES
- New shells added
- Outer electrons farther from nucleus
- More shielding from inner shells
"""

    def ionization_energy_trend(self) -> str:
        """
        Explain ionization energy trends.

        Returns:
            Explanation
        """
        return """
Ionization Energy Trends:

Across period (→): INCREASES
- Higher Z_eff holds electrons tighter
- Removing electron costs more energy

Down group (↓): DECREASES
- Outer electrons farther from nucleus
- More shielding → easier to remove

Exception: Noble gas configuration is especially stable
→ Following element (alkali metal) has very low IE
"""

    def block_classification(self, orbital: str) -> str:
        """
        Classify element block by last filled orbital type.

        Args:
            orbital: Last filled orbital (e.g., '4s', '3d', '4p', '4f')

        Returns:
            Block name ('s', 'p', 'd', or 'f')
        """
        if len(orbital) >= 2:
            return orbital[1]
        return 'unknown'

    def winding_interpretation(self) -> str:
        """
        Explain periodic table in terms of T⁴ winding.

        Returns:
            Explanation
        """
        return """
T⁴ Winding Interpretation of the Periodic Table:

Each orbital corresponds to a winding mode on T⁴:
- s orbitals: |n|² = 0 (spherically symmetric, 1 state)
- p orbitals: |n|² = 1 (3 oriented states)
- d orbitals: |n|² = 2 (5 oriented states)
- f orbitals: |n|² = 3 (7 oriented states)

The factor of 2 (spin) doubles each orbital's capacity.

Shell capacity 2n² counts ALL winding states with:
- Principal quantum number n
- All possible (l, m_l) combinations
- Both spin states

This is NOT an empirical rule—it's winding state enumeration.
"""

    def __repr__(self) -> str:
        return "PeriodicTable(shell_capacity=2n²)"
