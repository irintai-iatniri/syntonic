"""
Bond Character - Ionic vs covalent from syntony gap.

ΔS < 1/φ → covalent (delocalized hybrid windings)
ΔS > 1/φ → ionic (localized winding transfer)

The threshold 1/φ ≈ 0.618 emerges from DHSR partition.
"""

from __future__ import annotations
from typing import Dict, Any
import math

from syntonic.exact import PHI_NUMERIC


class BondCharacter:
    """
    Bond character from syntony gap.

    ΔS < 1/φ → covalent (delocalized hybrid windings)
    ΔS > 1/φ → ionic (localized winding transfer)

    The threshold 1/φ ≈ 0.618 is not arbitrary—it emerges from
    the DHSR partition where D = 1/φ² and H = 1/φ.

    Attributes:
        IONIC_THRESHOLD: 1/φ ≈ 0.618

    Example:
        >>> bc = BondCharacter()
        >>> bc.analyze(3.98, 0.93)  # F-Na (NaF)
        {'delta_S': 3.05, 'character': 'ionic', ...}
        >>> bc.analyze(2.55, 2.20)  # C-H
        {'delta_S': 0.35, 'character': 'covalent', ...}
    """

    IONIC_THRESHOLD = 1 / PHI_NUMERIC  # ≈ 0.618

    def analyze(self, chi_1: float, chi_2: float) -> Dict[str, Any]:
        """
        Analyze bond between two elements.

        Args:
            chi_1: Electronegativity of first element
            chi_2: Electronegativity of second element

        Returns:
            Dict with delta_S, character, ionic_fraction, dipole info
        """
        delta_S = abs(chi_1 - chi_2)

        if delta_S > self.IONIC_THRESHOLD:
            character = 'ionic'
        else:
            character = 'covalent'

        # Ionic fraction: continuous measure from 0 (pure covalent) to 1 (pure ionic)
        # Using: 1 - exp(-ΔS/φ)
        ionic_fraction = 1 - math.exp(-delta_S / PHI_NUMERIC)

        # Dipole moment prediction (qualitative)
        if delta_S < 0.4:
            polarity = 'nonpolar'
        elif delta_S < self.IONIC_THRESHOLD:
            polarity = 'polar covalent'
        else:
            polarity = 'ionic'

        return {
            'delta_S': delta_S,
            'character': character,
            'ionic_fraction': ionic_fraction,
            'covalent_fraction': 1 - ionic_fraction,
            'polarity': polarity,
            'threshold': self.IONIC_THRESHOLD,
        }

    def dipole_moment(self, delta_S: float, bond_length: float) -> float:
        """
        Estimate dipole moment from syntony gap.

        μ = d × (1 - e^(-ΔS/φ))

        This gives the partial charge separation.

        Args:
            delta_S: Electronegativity difference
            bond_length: Bond length in Angstroms

        Returns:
            Dipole moment in Debye (approximate)
        """
        # Partial charge from ionic fraction
        ionic_frac = 1 - math.exp(-delta_S / PHI_NUMERIC)

        # Dipole = charge × distance
        # 1 Debye ≈ 0.2082 e·Å
        # So μ ≈ ionic_frac × e × d / 0.2082
        return ionic_frac * bond_length / 0.2082

    def bond_order(self, shared_electrons: int) -> int:
        """
        Bond order from shared electron pairs.

        Single bond: 2 electrons → order 1
        Double bond: 4 electrons → order 2
        Triple bond: 6 electrons → order 3

        Args:
            shared_electrons: Number of shared electrons

        Returns:
            Bond order
        """
        return shared_electrons // 2

    def bond_energy_trend(self, bond_order: int) -> str:
        """
        Bond energy increases with bond order.

        Higher bond order = more winding overlap = stronger bond.

        Args:
            bond_order: 1, 2, or 3

        Returns:
            Qualitative description
        """
        energies = {
            1: "Single bond: ~350 kJ/mol (C-C)",
            2: "Double bond: ~610 kJ/mol (C=C)",
            3: "Triple bond: ~840 kJ/mol (C≡C)",
        }
        return energies.get(bond_order, f"Bond order {bond_order}")

    def resonance_stabilization(self) -> str:
        """
        Explain resonance as winding delocalization.

        Returns:
            Explanation
        """
        return """
Resonance as Winding Delocalization:

In resonance structures, the winding configuration is delocalized
across multiple atomic centers rather than localized on specific bonds.

Example: Benzene C₆H₆
- Not alternating single/double bonds
- All C-C bonds equivalent (bond order 1.5)
- Winding spreads over entire ring
- Stabilization energy ≈ 150 kJ/mol

SRT interpretation:
Delocalization minimizes syntony deficit by spreading the
winding across more space → lower gradient → lower energy.
"""

    def hydrogen_bonding(self) -> str:
        """
        Explain hydrogen bonding in SRT.

        Returns:
            Explanation
        """
        return """
Hydrogen Bonding in SRT:

H-bond is NOT just electrostatic attraction.
It is partial winding sharing between:
- Donor: X-H where X is electronegative (O, N, F)
- Acceptor: Lone pair on another electronegative atom

Strength: 10-40 kJ/mol (intermediate between covalent and van der Waals)

H-bond is strong because:
1. H has no inner shell shielding (k=1, minimal φ^k)
2. Small atomic radius → close approach
3. Partial winding overlap with acceptor lone pair
"""

    def classify_bond_type(self, delta_S: float) -> str:
        """
        Classify bond type by electronegativity difference.

        | ΔS Range | Type | Example |
        |----------|------|---------|
        | 0.0-0.4  | Nonpolar covalent | H-H, C-C |
        | 0.4-1/φ  | Polar covalent | C-O, N-H |
        | 1/φ-2.0  | Ionic | Na-Cl |
        | > 2.0    | Strongly ionic | Cs-F |

        Args:
            delta_S: Electronegativity difference

        Returns:
            Bond type classification
        """
        if delta_S < 0.4:
            return 'nonpolar covalent'
        elif delta_S < self.IONIC_THRESHOLD:
            return 'polar covalent'
        elif delta_S < 2.0:
            return 'ionic'
        else:
            return 'strongly ionic'

    def __repr__(self) -> str:
        return f"BondCharacter(threshold=1/φ≈{self.IONIC_THRESHOLD:.3f})"
