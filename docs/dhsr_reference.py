"""
DHSR Operators - Reference Implementation
Based on CRT/SRT DHSR Cycle Methodology v1.0

This module implements the corrected DHSR operators with spatial position-based
Golden Measure projection as specified in the methodology document.

Key Formula:
    S(Ψ) = ||Ĥ[D̂[Ψ]] - D̂[Ψ]|| / (||D̂[Ψ] - Ψ|| + ε)
    Target equilibrium: S* ≈ 1/φ ≈ 0.618
"""

import numpy as np
from typing import Tuple, Optional, Callable, List
from dataclasses import dataclass, field


# =============================================================================
# CONSTANTS
# =============================================================================

PHI = (1 + np.sqrt(5)) / 2          # Golden ratio φ ≈ 1.618034
PHI_INV = 1 / PHI                   # φ⁻¹ ≈ 0.618034
PHI_INV_SQ = 1 / PHI**2             # φ⁻² ≈ 0.381966
Q_DEFICIT = 0.027395146920          # Universal syntony deficit

# Verify golden partition: D + H = 1
assert abs(PHI_INV + PHI_INV_SQ - 1.0) < 1e-10, "Golden partition violated!"


# =============================================================================
# GOLDEN MEASURE
# =============================================================================

def golden_distribution(N: int) -> np.ndarray:
    """
    Normalized Golden Measure distribution for N modes.
    
    Returns weights where weights[n] = exp(-n²/φ) / Z
    
    CRITICAL: Index n is SPATIAL POSITION, not magnitude rank.
    """
    n_values = np.arange(N)
    weights = np.exp(-n_values**2 / PHI)
    return weights / np.sum(weights)


# =============================================================================
# HARMONIZATION OPERATOR Ĥ
# =============================================================================

def harmonize(
    psi: np.ndarray,
    strength: float = PHI_INV,
    preserve_phase: bool = True
) -> np.ndarray:
    """
    Apply Harmonization Operator Ĥ[Ψ].
    
    Projects toward Golden Measure equilibrium: ρ(n) ∝ exp(-n²/φ)
    
    ⚠️ CRITICAL: Weight assignment based on SPATIAL POSITION (mode index n),
    NOT on current magnitude.
    
    Args:
        psi: State vector (index = spatial position)
        strength: Projection strength γ ∈ [0, 1], default φ⁻¹
        preserve_phase: If True, keep phase from original
        
    Returns:
        Ĥ[Ψ] - harmonized state
    """
    N = len(psi)
    total_energy = np.sum(np.abs(psi)**2)
    
    # Golden weights based on POSITION n
    golden_weights = golden_distribution(N)
    target_amplitudes = np.sqrt(total_energy * golden_weights)
    
    # Phase handling
    if preserve_phase:
        phases = np.angle(psi)
        zero_mask = np.abs(psi) < 1e-15
        phases[zero_mask] = np.random.uniform(0, 2*np.pi, np.sum(zero_mask))
    else:
        phases = np.zeros(N)
    
    golden_target = target_amplitudes * np.exp(1j * phases)
    
    # Interpolate: Ĥ[Ψ] = (1 - γ)Ψ + γ·target
    result = (1 - strength) * psi + strength * golden_target
    
    return result


# =============================================================================
# DIFFERENTIATION OPERATOR D̂
# =============================================================================

def differentiate(
    psi: np.ndarray,
    syntony: float = 0.5,
    alpha_0: float = 0.1,
    seed: Optional[int] = None
) -> np.ndarray:
    """
    Apply Differentiation Operator D̂[Ψ].
    
    Increases complexity with syntony-dependent coupling: α(S) = α₀(1 - S)
    
    Args:
        psi: State vector
        syntony: Current syntony estimate S ∈ [0, 1]
        alpha_0: Base coupling strength
        seed: Random seed for reproducibility
        
    Returns:
        D̂[Ψ] - differentiated state
    """
    if seed is not None:
        np.random.seed(seed)
    
    N = len(psi)
    
    # State-dependent coupling: weaker differentiation at high syntony
    alpha = alpha_0 * (1 - syntony)
    
    # Work in Fourier space
    fft_psi = np.fft.fft(psi)
    
    # Excite higher modes (complexity increase)
    mode_indices = np.arange(N)
    excitation_weight = mode_indices / N
    
    # Add structured noise
    noise_real = np.random.randn(N) * excitation_weight
    noise_imag = np.random.randn(N) * excitation_weight
    noise = (noise_real + 1j * noise_imag) * np.abs(fft_psi).mean()
    
    fft_result = fft_psi + alpha * noise
    result = np.fft.ifft(fft_result)
    
    # Energy conservation
    original_energy = np.sum(np.abs(psi)**2)
    result_energy = np.sum(np.abs(result)**2)
    if result_energy > 0:
        result *= np.sqrt(original_energy / result_energy)
    
    return result


# =============================================================================
# SYNTONY COMPUTATION
# =============================================================================

def compute_syntony(
    psi: np.ndarray,
    D_op: Callable = None,
    H_op: Callable = None,
    epsilon: float = 1e-10
) -> float:
    """
    Compute Syntony Index S(Ψ).
    
    S(Ψ) = ||Ĥ[D̂[Ψ]] - D̂[Ψ]|| / (||D̂[Ψ] - Ψ|| + ε)
    
    Measures the harmonization contribution to the DHSR cycle.
    Target equilibrium: S* ≈ 1/φ ≈ 0.618
    
    Note: This is the CORRECTED formula (no "1 -" inversion).
    The complement 1 - S ≈ 1/φ² ≈ 0.382 measures differentiation contribution.
    """
    if D_op is None:
        D_op = lambda x: differentiate(x, syntony=0.5)
    if H_op is None:
        H_op = harmonize
    
    D_psi = D_op(psi)
    H_D_psi = H_op(D_psi)
    
    # Harmonization contribution (how much Ĥ changes D̂[Ψ])
    numerator = np.linalg.norm(H_D_psi - D_psi)
    
    # Differentiation contribution (how much D̂ changes Ψ)
    denominator = np.linalg.norm(D_psi - psi) + epsilon
    
    # S = H/(D+ε) → should converge to 1/φ ≈ 0.618
    S = numerator / denominator
    
    return float(np.clip(S, 0.0, 2.0))  # Allow slightly above 1 during transients


# =============================================================================
# RECURSION OPERATOR R̂ = Ĥ ∘ D̂
# =============================================================================

def recurse(
    psi: np.ndarray,
    syntony: Optional[float] = None,
    alpha_0: float = 0.1,
    strength: float = PHI_INV,
    seed: Optional[int] = None
) -> np.ndarray:
    """
    Apply Recursion Operator R̂ = Ĥ ∘ D̂.
    
    One complete DHSR cycle.
    
    Args:
        psi: Input state
        syntony: Current syntony (computed if None)
        alpha_0: Differentiation strength
        strength: Harmonization strength
        seed: Random seed
        
    Returns:
        R̂[Ψ] = Ĥ[D̂[Ψ]]
    """
    if syntony is None:
        syntony = compute_syntony(psi)
    
    D_psi = differentiate(psi, syntony=syntony, alpha_0=alpha_0, seed=seed)
    H_D_psi = harmonize(D_psi, strength=strength)
    
    return H_D_psi


# =============================================================================
# TRAJECTORY TRACKING
# =============================================================================

@dataclass
class DHSRTrajectory:
    """Records DHSR evolution."""
    syntony_values: List[float] = field(default_factory=list)
    golden_distances: List[float] = field(default_factory=list)
    states: List[np.ndarray] = field(default_factory=list)
    
    @property
    def converged(self) -> bool:
        if len(self.syntony_values) < 20:
            return False
        recent = self.syntony_values[-10:]
        return np.std(recent) < 1e-4
    
    @property 
    def final_syntony(self) -> float:
        return self.syntony_values[-1] if self.syntony_values else 0.0


def evolve(
    psi: np.ndarray,
    n_steps: int = 1000,
    alpha_0: float = 0.1,
    strength: float = PHI_INV,
    track_states: bool = False,
    verbose: bool = False
) -> Tuple[np.ndarray, DHSRTrajectory]:
    """
    Evolve state through multiple DHSR cycles.
    
    Args:
        psi: Initial state
        n_steps: Number of R̂ applications
        alpha_0: Differentiation strength
        strength: Harmonization strength
        track_states: Store intermediate states
        verbose: Print progress
        
    Returns:
        (final_state, trajectory)
    """
    trajectory = DHSRTrajectory()
    current = psi.copy()
    
    for step in range(n_steps):
        S = compute_syntony(current)
        trajectory.syntony_values.append(S)
        
        # Golden measure distance
        N = len(current)
        golden_target = np.sqrt(np.sum(np.abs(current)**2) * golden_distribution(N))
        dist = np.linalg.norm(np.abs(current) - golden_target)
        trajectory.golden_distances.append(dist)
        
        if track_states:
            trajectory.states.append(current.copy())
        
        if verbose and step % 10 == 0:
            print(f"Step {step:4d}: S = {S:.6f}, target = {PHI_INV:.6f}, "
                  f"Δ = {abs(S - PHI_INV):.6f}")
        
        # Check convergence
        if trajectory.converged:
            if verbose:
                print(f"Converged at step {step}")
            break
        
        # Apply DHSR cycle
        current = recurse(current, syntony=S, alpha_0=alpha_0, strength=strength)
    
    # Final measurement
    S_final = compute_syntony(current)
    trajectory.syntony_values.append(S_final)
    
    return current, trajectory


# =============================================================================
# VERIFICATION
# =============================================================================

def verify_implementation():
    """Run verification tests."""
    print("DHSR Implementation Verification")
    print("=" * 60)
    print(f"\nTarget syntony: S* = 1/φ = {PHI_INV:.6f}")
    print(f"Golden partition: {PHI_INV_SQ:.6f} + {PHI_INV:.6f} = {PHI_INV_SQ + PHI_INV:.6f}")
    
    # Test 1: Single cycle
    print("\n--- Test 1: Single DHSR Cycle ---")
    np.random.seed(42)
    N = 50
    psi = np.random.randn(N) + 1j * np.random.randn(N)
    psi /= np.linalg.norm(psi)
    
    S_before = compute_syntony(psi)
    psi_after = recurse(psi)
    S_after = compute_syntony(psi_after)
    
    print(f"Before R̂: S = {S_before:.6f}")
    print(f"After R̂:  S = {S_after:.6f}")
    
    # Test 2: Golden measure convergence
    print("\n--- Test 2: Golden Measure Projection ---")
    H_psi = harmonize(psi, strength=PHI_INV)
    golden_target = np.sqrt(np.sum(np.abs(psi)**2) * golden_distribution(N))
    
    dist_before = np.linalg.norm(np.abs(psi) - golden_target)
    dist_after = np.linalg.norm(np.abs(H_psi) - golden_target)
    
    print(f"Distance before Ĥ: {dist_before:.6f}")
    print(f"Distance after Ĥ:  {dist_after:.6f}")
    print(f"Improvement: {(1 - dist_after/dist_before)*100:.1f}%")
    
    # Test 3: Evolution
    print("\n--- Test 3: DHSR Evolution ---")
    np.random.seed(42)
    psi = np.random.randn(N) + 1j * np.random.randn(N)
    psi /= np.linalg.norm(psi)
    
    final, traj = evolve(psi, n_steps=100, verbose=False)
    
    print(f"Initial syntony: {traj.syntony_values[0]:.6f}")
    print(f"Final syntony:   {traj.syntony_values[-1]:.6f}")
    print(f"Target (1/φ):    {PHI_INV:.6f}")
    print(f"Deviation:       {abs(traj.syntony_values[-1] - PHI_INV):.6f}")
    print(f"Converged:       {traj.converged}")
    
    # Test 4: Multiple runs
    print("\n--- Test 4: Consistency Across Seeds ---")
    final_syntonies = []
    for seed in [1, 42, 123, 456, 789]:
        np.random.seed(seed)
        psi = np.random.randn(N) + 1j * np.random.randn(N)
        psi /= np.linalg.norm(psi)
        _, traj = evolve(psi, n_steps=200, verbose=False)
        final_syntonies.append(traj.syntony_values[-1])
        print(f"  Seed {seed:3d}: S_final = {traj.syntony_values[-1]:.6f}")
    
    mean_S = np.mean(final_syntonies)
    std_S = np.std(final_syntonies)
    print(f"\nMean: {mean_S:.6f} ± {std_S:.6f}")
    print(f"Target: {PHI_INV:.6f}")
    print(f"Agreement: {(1 - abs(mean_S - PHI_INV)/PHI_INV)*100:.1f}%")
    
    print("\n" + "=" * 60)


if __name__ == "__main__":
    verify_implementation()
    
    # Additional detailed evolution display
    print("\n\nDetailed Evolution:")
    print("=" * 60)
    
    np.random.seed(42)
    N = 50
    psi = np.random.randn(N) + 1j * np.random.randn(N)
    psi /= np.linalg.norm(psi)
    
    print(f"\nDHSR Evolution (target S* = {PHI_INV:.6f}):")
    print("-" * 50)
    
    current = psi
    for i in range(50):
        S = compute_syntony(current)
        if i < 15 or i % 10 == 0:
            delta = S - PHI_INV
            print(f"Step {i:3d}: S = {S:.6f}  (Δ from target: {delta:+.6f})")
        current = recurse(current, syntony=S)
    
    S_final = compute_syntony(current)
    print(f"\nFinal: S = {S_final:.6f}")
    print(f"Target:   {PHI_INV:.6f}")
    print(f"Match:    {(1 - abs(S_final - PHI_INV)/PHI_INV)*100:.2f}%")