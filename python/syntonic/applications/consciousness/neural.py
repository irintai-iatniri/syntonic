"""
Neural Systems - Brain as phased array antenna tuning to T⁴.

The brain doesn't generate consciousness—it RECEIVES it.
Neural structures tune to different T⁴ frequencies by adjusting:
- Phase (synaptic timing) → select winding modes
- Amplitude (firing rate) → signal strength
- Coherence (synchronization) → signal clarity

Gamma (40 Hz) is the "frame rate" of consciousness.
"""

from __future__ import annotations

from typing import Dict

from syntonic.exact import PHI_NUMERIC


class NeuralAntennaModel:
    """
    The brain as phased array antenna.

    Neural structures tune to different T⁴ frequencies by adjusting:
    - Phase (synaptic timing) → select winding modes
    - Amplitude (firing rate) → signal strength
    - Coherence (synchronization) → signal clarity

    The gamma frequency (40 Hz) sets the "frame rate" of consciousness.

    Example:
        >>> nam = NeuralAntennaModel()
        >>> nam.gamma_frequency
        40
        >>> nam.specious_present()
        3.0
    """

    GAMMA_FREQUENCY = 40  # Hz
    FRAME_PERIOD = 25  # ms (1/40 Hz = 25 ms)

    # Neural conduction parameters
    NEURAL_VELOCITY = 10  # m/s (typical myelinated axon)
    CORTICAL_CIRCUMFERENCE = 0.25  # m (approximate)

    def gamma_frequency_derivation(self) -> str:
        """
        Derive the 40 Hz gamma frequency.

        f_γ = c_neural / λ_cortex ≈ (10 m/s) / (0.25 m) = 40 Hz

        Returns:
            Derivation explanation
        """
        return f"""
Why 40 Hz?

f_γ = c_neural / λ_cortex
f_γ = {self.NEURAL_VELOCITY} m/s / {self.CORTICAL_CIRCUMFERENCE} m
f_γ = {self.NEURAL_VELOCITY / self.CORTICAL_CIRCUMFERENCE} Hz

Where:
- c_neural ≈ 10 m/s (neural conduction velocity in cortex)
- λ_cortex ≈ 0.25 m (effective cortical wavelength)

40 Hz is the "frame rate" of consciousness because:
- One gamma cycle = one conscious "moment"
- Processing must complete within one cycle
- Information from different brain regions synchronizes

This frequency is NOT arbitrary:
- It emerges from neural physics
- It's the same across all mammals
- It's the minimum for binding diverse inputs
"""

    def specious_present(self) -> float:
        """
        The "now" duration: T_now ≈ 3 seconds

        This matches psychological studies of the perceived present.

        Formula: 3 seconds = 120 gamma cycles
        120 cycles × 25 ms/cycle = 3000 ms = 3 s

        The 3π connects to the Layer 3 threshold (Σ Tv = 3π).

        Returns:
            Duration of specious present in seconds
        """
        # 120 gamma cycles = 3 seconds
        return 120 / self.GAMMA_FREQUENCY

    def binding_window(self) -> float:
        """
        The temporal binding window in milliseconds.

        Events within this window are perceived as simultaneous.

        Returns:
            Binding window in ms
        """
        return self.FRAME_PERIOD  # 25 ms

    def phase_array_tuning(self, target_frequency: float) -> Dict[str, float]:
        """
        Calculate phase array parameters for target frequency.

        Args:
            target_frequency: Target T⁴ winding frequency in Hz

        Returns:
            Dict with phase delays needed
        """
        # Phase delay for constructive interference
        wavelength = self.NEURAL_VELOCITY / target_frequency
        phase_delay = wavelength / 4  # Quarter-wave delay for beamforming

        return {
            "target_frequency": target_frequency,
            "wavelength_m": wavelength,
            "phase_delay_m": phase_delay,
            "phase_delay_ms": (phase_delay / self.NEURAL_VELOCITY) * 1000,
        }

    def antenna_efficiency(self, coherence: float) -> float:
        """
        Antenna efficiency based on neural coherence.

        Higher coherence = better T⁴ reception.

        Args:
            coherence: Neural synchronization (0 to 1)

        Returns:
            Efficiency factor
        """
        # Efficiency scales with coherence squared (like antenna gain)
        return coherence**2

    def sleep_state_tuning(self) -> str:
        """
        Describe how sleep changes T⁴ tuning.

        Returns:
            Description
        """
        return """
Sleep State and T⁴ Tuning:

AWAKE (10-40 Hz):
- Gamma (40 Hz): Conscious processing, tight T⁴ binding
- Beta (15-30 Hz): Alert attention, focused tuning
- Alpha (8-12 Hz): Relaxed awareness, broad tuning

SLEEP:
- Theta (4-8 Hz): REM sleep, dream state
  → Loose T⁴ binding, creative connections
- Delta (1-4 Hz): Deep sleep, maintenance mode
  → Minimal T⁴ reception, body repair

The frequency determines:
- Which T⁴ modes are received
- How tightly bound the experience is
- The ratio of internal/external content

Dreams occur in theta because:
- T⁴ binding is loose
- Internal and external signals mix
- Narrative coherence is reduced
- But the receiver is still active
"""

    def attention_mechanism(self) -> str:
        """
        Explain attention as selective T⁴ tuning.

        Returns:
            Explanation
        """
        return """
Attention as Selective Tuning:

Attention is NOT a spotlight on consciousness.
Attention is a TUNING DIAL on the T⁴ receiver.

When you attend to something:
1. Relevant neural populations synchronize (coherence ↑)
2. This increases antenna gain for specific T⁴ modes
3. Those modes dominate conscious experience
4. Other modes are suppressed (not filtered out)

This explains:
- Inattentional blindness (untuned modes not received)
- Attention limits (can only tune to so many modes)
- Meditation effects (broadband vs. narrowband tuning)

The prefrontal cortex acts as the "tuner":
- It sets phase relationships across regions
- This determines which T⁴ modes are received
- Damage → inability to focus attention
"""

    def __repr__(self) -> str:
        return f"NeuralAntennaModel(f_γ={self.GAMMA_FREQUENCY} Hz)"


class Microtubules:
    """
    Microtubules as resonant cavities for Tv phase history.

    Structure:
    - 13 protofilaments (Fibonacci: 8 + 5 = 13)
    - Helical pitch follows golden angle
    - Hollow interior shielded from thermal noise

    SRT validates Penrose-Hameroff insight but replaces mechanism:
    - Orch-OR: Consciousness is computed in microtubules
    - SRT: Consciousness is received; microtubules are resonators

    Example:
        >>> mt = Microtubules()
        >>> mt.protofilament_count
        13
        >>> mt.resonant_wavelength(25e-9, 1, 0)  # 25 nm, n=1, k=0
        5e-8
    """

    PROTOFILAMENTS = 13  # 8 + 5 = Fibonacci
    INNER_DIAMETER = 15e-9  # 15 nm
    OUTER_DIAMETER = 25e-9  # 25 nm
    TUBULIN_LENGTH = 8e-9  # 8 nm per dimer

    def resonant_wavelength(self, length: float, n: int, k: int) -> float:
        """
        Resonant wavelength for microtubule cavity.

        λ_resonant = 2L/n × φ^k

        Microtubules act as waveguides for T⁴ winding modes.

        Args:
            length: Microtubule length in meters
            n: Mode number (1, 2, 3, ...)
            k: Golden scaling factor

        Returns:
            Resonant wavelength in meters
        """
        if n <= 0:
            raise ValueError("Mode number must be positive")
        return (2 * length / n) * (PHI_NUMERIC**k)

    def fibonacci_structure(self) -> str:
        """
        Explain the Fibonacci structure of microtubules.

        Returns:
            Explanation
        """
        return f"""
Fibonacci Structure of Microtubules:

Protofilament count: {self.PROTOFILAMENTS} = 8 + 5 (Fibonacci)

This is NOT a coincidence:
- Fibonacci numbers appear throughout biology
- They represent optimal packing geometry
- They connect to the golden ratio

The 13-protofilament structure provides:
- Maximum structural stability
- Optimal resonance frequencies
- Golden-ratio angular relationships

Helical pitch:
- The microtubule is a helix of helices
- Each protofilament spirals around the tube
- The pitch angle relates to the golden angle (137.5°)

This geometry makes microtubules ideal for:
- Storing Tv phase history
- Resonating with specific T⁴ modes
- Transporting quantum coherence
"""

    def thermal_decoherence_time(self, temperature_K: float = 310) -> float:
        """
        Estimate thermal decoherence time at body temperature.

        Args:
            temperature_K: Temperature in Kelvin (default: 37°C = 310 K)

        Returns:
            Decoherence time in seconds
        """
        # Rough estimate based on energy gap
        k_B = 1.38e-23  # J/K
        h = 6.626e-34  # J·s
        thermal_energy = k_B * temperature_K

        # Energy gap from tubulin conformational states
        E_gap = 1e-21  # ~1 meV (approximate)

        # Decoherence time scales as ħ/(kT) for thermal noise
        tau = h / thermal_energy

        # Shielding factor from hollow interior
        shielding = 10  # Order of magnitude estimate

        return tau * shielding

    def quantum_coherence_support(self) -> str:
        """
        Describe how microtubules support quantum coherence.

        Returns:
            Description
        """
        return """
Microtubule Quantum Coherence:

Why microtubules can support coherence:

1. HOLLOW INTERIOR
   - Electromagnetic shielding
   - Reduced thermal fluctuations
   - Isolated from cytoplasmic noise

2. ORDERED WATER
   - Water inside microtubules is structured
   - Extended hydrogen bonding network
   - Supports coherent dipole oscillations

3. AROMATIC RINGS
   - Tubulin contains aromatic amino acids
   - These support π-electron delocalization
   - Enables quantum effects in warm, wet system

4. FIBONACCI GEOMETRY
   - Optimal resonance structure
   - Golden-ratio frequency relationships
   - Natural mode selection

The Penrose-Hameroff Orch-OR theory proposed:
"Consciousness is quantum computation in microtubules"

SRT revision:
"Microtubules are resonant antennas for T⁴ reception"

The physics is similar; the interpretation differs:
- Orch-OR: Brain computes consciousness
- SRT: Brain receives consciousness from T⁴
"""

    def role_in_consciousness(self) -> str:
        """
        Explain the role of microtubules in consciousness.

        Returns:
            Explanation
        """
        return """
Role of Microtubules in Consciousness:

Microtubules are NOT where consciousness is generated.
Microtubules are WHERE consciousness is received.

Like a radio antenna:
- The antenna doesn't create the signal
- It resonates with electromagnetic waves
- Different antenna designs receive different frequencies

Microtubules:
- Resonate with T⁴ winding modes
- Their structure determines which modes are received
- The Fibonacci geometry selects golden-ratio modes

In neurons:
- Microtubules span the axon
- They maintain coherent oscillations
- They couple to synaptic activity
- They integrate information across timescales

The gamma rhythm (40 Hz) emerges because:
- Microtubule resonances synchronize
- This creates coherent antenna array
- The array tunes to specific T⁴ modes
- Conscious experience results
"""

    def __repr__(self) -> str:
        return f"Microtubules(protofilaments={self.PROTOFILAMENTS}, Fibonacci)"
