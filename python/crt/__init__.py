"""
CRT Core - Cosmological Recursion Theory operators and metrics.

Provides the DHSR (Differentiation-Harmonization-Syntony-Recursion) framework
for state evolution and analysis.

Components:
- **Operators**: D̂ (differentiation), Ĥ (harmonization), R̂ (recursion)
- **Metrics**: Syntony S(Ψ), Gnosis layers (0-3)
- **Evolution**: Trajectory tracking, attractor finding

Basic Usage:
    >>> import syntonic as syn
    >>> from syntonic.crt import create_dhsr_system

    >>> # Create DHSR system
    >>> R_op, S_comp, G_comp = create_dhsr_system()

    >>> # Create a state
    >>> psi = syn.state([1, 2, 3, 4, 5, 6, 7, 8])

    >>> # Apply recursion
    >>> evolved = R_op.apply(psi)

    >>> # Compute metrics
    >>> syntony = S_comp.compute(psi)
    >>> gnosis = G_comp.compute_layer(psi)

    >>> # Or use the evolver for full trajectory
    >>> from syntonic.crt import DHSREvolver
    >>> evolver = DHSREvolver()
    >>> trajectory = evolver.evolve(psi, n_steps=100)
    >>> print(trajectory.summary())
"""

from typing import Tuple

# Operators
from syntonic.crt.operators import (
    OperatorBase,
    FourierProjector,
    DampingProjector,
    LaplacianOperator,
    create_mode_projectors,
    create_damping_cascade,
    DifferentiationOperator,
    default_differentiation_operator,
    HarmonizationOperator,
    default_harmonization_operator,
    RecursionOperator,
    default_recursion_operator,
)

# Metrics
from syntonic.crt.metrics import (
    SyntonyComputer,
    syntony_entropy,
    syntony_spectral,
    syntony_quick,
    GnosisComputer,
    default_gnosis_computer,
    K_D4,
)

# Evolution
from syntonic.crt.evolution import (
    SyntonyTrajectory,
    DHSREvolver,
    default_evolver,
)


def create_dhsr_system(
    alpha_0: float = 0.1,
    beta_0: float = 0.618,
    zeta_0: float = 0.01,
    gamma_0: float = 0.1,
    num_modes: int = 8,
    num_dampers: int = 3,
) -> Tuple[RecursionOperator, SyntonyComputer, GnosisComputer]:
    """
    Create a complete DHSR system with specified parameters.

    This is the recommended way to create a configured DHSR system
    for evolution and analysis.

    Args:
        alpha_0: Base differentiation coupling strength
        beta_0: Base harmonization damping strength (default: 1/φ)
        zeta_0: Base Laplacian diffusion coefficient
        gamma_0: Syntony projection strength
        num_modes: Number of Fourier modes for differentiation
        num_dampers: Number of damping levels for harmonization

    Returns:
        (RecursionOperator, SyntonyComputer, GnosisComputer) tuple

    Example:
        >>> R_op, S_comp, G_comp = create_dhsr_system()
        >>> evolved = R_op.apply(state)
        >>> syntony = S_comp.compute(state)
        >>> layer = G_comp.compute_layer(state)
    """
    # Create operators
    D_op = DifferentiationOperator(
        alpha_0=alpha_0,
        zeta_0=zeta_0,
        num_modes=num_modes,
    )

    H_op = HarmonizationOperator(
        beta_0=beta_0,
        gamma_0=gamma_0,
        num_dampers=num_dampers,
    )

    R_op = RecursionOperator(diff_op=D_op, harm_op=H_op)

    # Create metrics
    S_comp = SyntonyComputer(diff_op=D_op, harm_op=H_op)
    G_comp = GnosisComputer(recursion_op=R_op, syntony_computer=S_comp)

    return R_op, S_comp, G_comp


def create_evolver(
    alpha_0: float = 0.1,
    beta_0: float = 0.618,
    **kwargs,
) -> DHSREvolver:
    """
    Create a configured DHSR evolver.

    Args:
        alpha_0: Base differentiation strength
        beta_0: Base harmonization strength
        **kwargs: Additional parameters for create_dhsr_system

    Returns:
        DHSREvolver configured with specified parameters
    """
    R_op, S_comp, G_comp = create_dhsr_system(
        alpha_0=alpha_0,
        beta_0=beta_0,
        **kwargs,
    )

    return DHSREvolver(
        recursion_op=R_op,
        syntony_computer=S_comp,
        gnosis_computer=G_comp,
    )


__all__ = [
    # Operators
    'OperatorBase',
    'FourierProjector',
    'DampingProjector',
    'LaplacianOperator',
    'create_mode_projectors',
    'create_damping_cascade',
    'DifferentiationOperator',
    'default_differentiation_operator',
    'HarmonizationOperator',
    'default_harmonization_operator',
    'RecursionOperator',
    'default_recursion_operator',
    # Metrics
    'SyntonyComputer',
    'syntony_entropy',
    'syntony_spectral',
    'syntony_quick',
    'GnosisComputer',
    'default_gnosis_computer',
    'K_D4',
    # Evolution
    'SyntonyTrajectory',
    'DHSREvolver',
    'default_evolver',
    # Factories
    'create_dhsr_system',
    'create_evolver',
]
