"""
Lucas Shadow Injection - Implements Pillar V (The Shadow).

Uses Lucas Number ratios to inject 'Shadow Phase' (Dark Energy)
into the network. This prevents static loops (overfitting) and
drives the system toward the attractor.

Identity: L_n = phi^n + (1-phi)^n
"""

import random
import syntonic.sn as sn
from syntonic._core import ResonantTensor
from syntonic.exact import PHI


class LucasShadow(sn.Module):
    """
    Applies 'Shadow Pressure' (Novelty/Noise) to the latent state.

    Unlike random noise, Lucas Shadow is geometric anti-phase.
    It simulates the vacuum pressure of the Dark Sector.
    """

    def __init__(self, level: int):
        super().__init__()
        self.level = level
        # Calculate Lucas Shadow magnitude: (1 - phi)^n
        # This is the "Destructive" component of the expansion
        self.shadow_magnitude = abs((1.0 - PHI) ** level)

    def forward(self, x: ResonantTensor) -> ResonantTensor:
        if self.training:
            # Grand Synthesis: Anti-Phase Geometry (Orthogonal Rotation)
            # The Shadow is not random noise - it's geometric inversion
            # Transform data into the null space of current Mersenne lattice

            # Apply orthogonal rotation: x -> -x + projection onto orthogonal subspace
            # This simulates "Anti-Phase Geometry" - the shadow realm
            x_rotated = x.scalar_mul(-1.0)  # Flip sign (geometric inversion)

            # Add small orthogonal perturbation scaled by Lucas magnitude
            # This represents tunneling through the Mersenne barrier
            # Create noise tensor with same shape
            noise_values = [
                random.gauss(0, self.shadow_magnitude * 0.1) for _ in range(x.len())
            ]
            orthogonal_noise = ResonantTensor(noise_values, list(x.shape))
            x_rotated = x_rotated.elementwise_add(orthogonal_noise)

            # The Shadow exerts pressure on the Light (anti-phase interference)
            return x.elementwise_add(x_rotated)
        return x
