"""
Molecular Geometry - VSEPR as syntony optimization.

Geometry = argmin_θ Σᵢ ΔSᵢ(θ)

Atoms arrange to minimize total syntony deficit across all bonds.
The tetrahedral angle 109.47° = arccos(-1/3) emerges from
optimal 4-winding packing in 3D.
"""

from __future__ import annotations
from typing import Dict, Any, Tuple, List
import math


class MolecularGeometry:
    """
    VSEPR as syntony optimization.

    Geometry = argmin_θ Σᵢ ΔSᵢ(θ)

    Atoms arrange to minimize total syntony deficit across all bonds.

    The optimal geometries emerge from minimizing mutual syntony
    interference between bonding orbitals.

    Example:
        >>> mg = MolecularGeometry()
        >>> mg.tetrahedral_angle()
        109.47...
        >>> mg.optimal_geometry(4)
        {'geometry': 'tetrahedral', 'angle': 109.47}
    """

    # Standard bond angles
    ANGLES = {
        'linear': 180.0,
        'trigonal_planar': 120.0,
        'tetrahedral': 109.47,
        'trigonal_bipyramidal_eq': 120.0,
        'trigonal_bipyramidal_ax': 90.0,
        'octahedral': 90.0,
    }

    def tetrahedral_angle(self) -> float:
        """
        Tetrahedral angle 109.47° from optimal 4-winding packing.

        arccos(-1/3) = 109.47°

        Emerges from minimizing mutual syntony interference
        for 4 equivalent bonds.

        Returns:
            Tetrahedral angle in degrees
        """
        return math.degrees(math.acos(-1/3))

    def optimal_geometry(self, n_bonds: int, n_lone_pairs: int = 0) -> Dict[str, Any]:
        """
        Optimal geometry for n bonds and lone pairs.

        VSEPR domains = bonds + lone pairs
        Geometry minimizes inter-domain repulsion (syntony interference).

        | Domains | Geometry | Angle |
        |---------|----------|-------|
        | 2 | Linear | 180° |
        | 3 | Trigonal planar | 120° |
        | 4 | Tetrahedral | 109.5° |
        | 5 | Trigonal bipyramidal | 120°/90° |
        | 6 | Octahedral | 90° |

        Args:
            n_bonds: Number of bonding pairs
            n_lone_pairs: Number of lone pairs

        Returns:
            Dict with geometry name, angles, and description
        """
        total_domains = n_bonds + n_lone_pairs

        # Base geometries by total electron domains
        base_geometries = {
            2: ('linear', 180.0),
            3: ('trigonal_planar', 120.0),
            4: ('tetrahedral', self.tetrahedral_angle()),
            5: ('trigonal_bipyramidal', (120.0, 90.0)),
            6: ('octahedral', 90.0),
        }

        if total_domains not in base_geometries:
            return {
                'geometry': 'unknown',
                'angle': 0.0,
                'n_bonds': n_bonds,
                'n_lone_pairs': n_lone_pairs,
            }

        name, angle = base_geometries[total_domains]

        # Adjust molecular geometry based on lone pairs
        molecular_geometry = self._adjust_for_lone_pairs(
            total_domains, n_lone_pairs, name
        )

        return {
            'electron_geometry': name,
            'molecular_geometry': molecular_geometry,
            'ideal_angle': angle,
            'n_bonds': n_bonds,
            'n_lone_pairs': n_lone_pairs,
            'total_domains': total_domains,
        }

    def _adjust_for_lone_pairs(
        self, total_domains: int, n_lone_pairs: int, base_geometry: str
    ) -> str:
        """
        Adjust geometry name for lone pairs.

        Lone pairs compress bond angles and change molecular shape.
        """
        if n_lone_pairs == 0:
            return base_geometry

        # Common modifications
        adjustments = {
            (4, 1): 'trigonal_pyramidal',  # NH3
            (4, 2): 'bent',  # H2O
            (5, 1): 'seesaw',
            (5, 2): 'T-shaped',
            (5, 3): 'linear',
            (6, 1): 'square_pyramidal',
            (6, 2): 'square_planar',
        }

        return adjustments.get((total_domains, n_lone_pairs), base_geometry)

    def bond_angle_compression(self, ideal_angle: float, n_lone_pairs: int) -> float:
        """
        Estimate bond angle compression from lone pairs.

        Lone pairs occupy more angular space than bonding pairs
        → compress bond angles by ~2-3° per lone pair.

        Args:
            ideal_angle: Ideal angle without lone pairs
            n_lone_pairs: Number of lone pairs

        Returns:
            Estimated actual bond angle
        """
        compression_per_lp = 2.5  # degrees
        return ideal_angle - n_lone_pairs * compression_per_lp

    def water_geometry(self) -> Dict[str, Any]:
        """
        Water (H₂O) molecular geometry.

        Returns:
            Geometry details for water
        """
        return {
            'molecule': 'H2O',
            'electron_geometry': 'tetrahedral',
            'molecular_geometry': 'bent',
            'ideal_angle': self.tetrahedral_angle(),
            'actual_angle': 104.5,  # experimental
            'compression': 'Two lone pairs compress H-O-H angle',
            'n_bonds': 2,
            'n_lone_pairs': 2,
        }

    def ammonia_geometry(self) -> Dict[str, Any]:
        """
        Ammonia (NH₃) molecular geometry.

        Returns:
            Geometry details for ammonia
        """
        return {
            'molecule': 'NH3',
            'electron_geometry': 'tetrahedral',
            'molecular_geometry': 'trigonal_pyramidal',
            'ideal_angle': self.tetrahedral_angle(),
            'actual_angle': 107.0,  # experimental
            'compression': 'One lone pair compresses H-N-H angles',
            'n_bonds': 3,
            'n_lone_pairs': 1,
        }

    def methane_geometry(self) -> Dict[str, Any]:
        """
        Methane (CH₄) molecular geometry.

        Returns:
            Geometry details for methane
        """
        return {
            'molecule': 'CH4',
            'electron_geometry': 'tetrahedral',
            'molecular_geometry': 'tetrahedral',
            'ideal_angle': self.tetrahedral_angle(),
            'actual_angle': self.tetrahedral_angle(),  # exact
            'compression': 'None - all equivalent bonds',
            'n_bonds': 4,
            'n_lone_pairs': 0,
        }

    def hybridization(self, n_bonds: int, n_lone_pairs: int = 0) -> str:
        """
        Determine hybridization from total electron domains.

        | Domains | Hybridization | Geometry |
        |---------|---------------|----------|
        | 2 | sp | Linear |
        | 3 | sp² | Trigonal planar |
        | 4 | sp³ | Tetrahedral |
        | 5 | sp³d | Trigonal bipyramidal |
        | 6 | sp³d² | Octahedral |

        Args:
            n_bonds: Number of bonding pairs
            n_lone_pairs: Number of lone pairs

        Returns:
            Hybridization label
        """
        total = n_bonds + n_lone_pairs
        hybridizations = {
            2: 'sp',
            3: 'sp²',
            4: 'sp³',
            5: 'sp³d',
            6: 'sp³d²',
        }
        return hybridizations.get(total, f'unknown (domains={total})')

    def syntony_optimization_principle(self) -> str:
        """
        Explain VSEPR as syntony optimization.

        Returns:
            Explanation
        """
        return """
VSEPR as Syntony Optimization:

Molecular geometry minimizes total syntony deficit:
    θ_optimal = argmin_θ Σᵢ ΔSᵢ(θ)

Why does this work?
1. Each bonding orbital has a winding configuration
2. Overlapping windings interfere (increase deficit)
3. Geometry adjusts to minimize interference
4. Optimal angles maximize winding separation

Lone pairs:
- Occupy more angular space than bonding pairs
- Their winding "spreads out" without constraint
- Push bonding pairs closer together
- Result: Compressed bond angles

The tetrahedral angle arccos(-1/3) = 109.47° is
the optimal configuration for 4 equivalent domains
in 3D space - pure geometry, no empirical fitting.
"""

    def __repr__(self) -> str:
        return f"MolecularGeometry(tetrahedral={self.tetrahedral_angle():.2f}°)"
