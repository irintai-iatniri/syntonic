"""
Syntonic Entropy - Entropy from winding distribution.

S_thermo = -k_B Σ p(n) ln p(n)

Where p(n) follows the Golden Measure: μ(n) ∝ exp(-|n|²/φ)
"""

from __future__ import annotations

import math
from typing import Dict, Tuple

from syntonic.exact import PHI_NUMERIC


class SyntonicEntropy:
    """
    Entropy from winding distribution.

    S_thermo = -k_B Σ p(n) ln p(n)

    Where p(n) follows the Golden Measure: μ(n) ∝ exp(-|n|²/φ)

    The syntony-entropy relationship:
    S_thermo = S_thermo,0 × (1 - S)^η + S₀ × S × ln(S)

    First term: amplified disorder at low syntony
    Second term: information content of syntony

    Example:
        >>> entropy = SyntonicEntropy()
        >>> dist = {(0,0,0,0): 0.5, (1,0,0,0): 0.3, (0,1,0,0): 0.2}
        >>> entropy.winding_entropy(dist)
        1.0296...  # Shannon entropy
    """

    def __init__(self, k_B: float = 1.0):
        """
        Initialize entropy calculator.

        Args:
            k_B: Boltzmann constant (default 1.0 for natural units)
        """
        self._k_B = k_B

    def winding_entropy(self, distribution: Dict[Tuple, float]) -> float:
        """
        Shannon entropy of winding distribution.

        S = -Σ p(n) ln p(n)

        Args:
            distribution: Dict mapping winding tuples to probabilities

        Returns:
            Shannon entropy in natural units
        """
        total = 0.0
        for p in distribution.values():
            if p > 0:
                total -= p * math.log(p)
        return self._k_B * total

    def golden_measure_entropy(self, max_norm_sq: int = 10) -> float:
        """
        Entropy of the Golden Measure distribution.

        For μ(n) ∝ exp(-|n|²/φ), compute the entropy.

        Args:
            max_norm_sq: Maximum |n|² to include

        Returns:
            Entropy of Golden Measure
        """
        # Build distribution
        weights = {}
        total_weight = 0.0

        # Enumerate winding states (simplified for |n|² values)
        for norm_sq in range(max_norm_sq + 1):
            # Degeneracy of states with |n|² = norm_sq (approximate)
            # For 4D: roughly proportional to norm_sq^1.5
            degeneracy = max(1, int(4 * (norm_sq**1.5))) if norm_sq > 0 else 1
            weight = math.exp(-norm_sq / PHI_NUMERIC)
            weights[norm_sq] = weight * degeneracy
            total_weight += weight * degeneracy

        # Normalize and compute entropy
        entropy = 0.0
        for norm_sq, weight in weights.items():
            p = weight / total_weight
            if p > 0:
                entropy -= p * math.log(p)

        return self._k_B * entropy

    def syntony_entropy_relation(self, S: float, eta: float = 1.0) -> float:
        """
        Relate thermodynamic entropy to syntony.

        S_thermo = S_thermo,0 × (1 - S)^η + S₀ × S × ln(S)

        First term: amplified disorder at low syntony
        Second term: information content of syntony

        Args:
            S: Syntony value in [0, 1]
            eta: Exponent (default 1.0)

        Returns:
            Thermodynamic entropy
        """
        S_0 = 1.0  # Reference entropy

        if S <= 0:
            return float("inf")
        if S >= 1:
            return 0.0

        term1 = S_0 * (1 - S) ** eta
        term2 = S_0 * S * math.log(S)

        return term1 + term2

    def maximum_entropy(self, n_states: int) -> float:
        """
        Maximum entropy for n_states (uniform distribution).

        S_max = k_B × ln(n_states)

        Args:
            n_states: Number of accessible states

        Returns:
            Maximum entropy
        """
        if n_states <= 0:
            return 0.0
        return self._k_B * math.log(n_states)

    def relative_entropy(
        self,
        p: Dict[Tuple, float],
        q: Dict[Tuple, float],
    ) -> float:
        """
        Kullback-Leibler divergence D_KL(P || Q).

        D_KL = Σ p(n) ln(p(n)/q(n))

        Measures how much P differs from Q.

        Args:
            p: Distribution P
            q: Distribution Q (reference)

        Returns:
            KL divergence (non-negative)
        """
        kl = 0.0
        for key, p_val in p.items():
            if p_val > 0:
                q_val = q.get(key, 1e-10)
                kl += p_val * math.log(p_val / q_val)
        return self._k_B * kl

    def entropy_production_rate(
        self,
        distribution: Dict[Tuple, float],
        flow_rates: Dict[Tuple, float],
    ) -> float:
        """
        Entropy production rate from probability currents.

        dS/dt = -Σ (∂p/∂t) ln p

        Args:
            distribution: Current probability distribution
            flow_rates: Rate of change dp/dt for each state

        Returns:
            Entropy production rate
        """
        rate = 0.0
        for key, dp_dt in flow_rates.items():
            p = distribution.get(key, 1e-10)
            if p > 0:
                rate -= dp_dt * math.log(p)
        return self._k_B * rate

    def free_energy(
        self,
        internal_energy: float,
        entropy: float,
        temperature: float = PHI_NUMERIC,
    ) -> float:
        """
        Helmholtz free energy F = U - TS.

        In syntonic thermodynamics, T = φ is the natural temperature.

        Args:
            internal_energy: U
            entropy: S
            temperature: T (default φ)

        Returns:
            Free energy F
        """
        return internal_energy - temperature * entropy

    def __repr__(self) -> str:
        return f"SyntonicEntropy(k_B={self._k_B})"
