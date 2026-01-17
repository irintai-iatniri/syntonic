"""
GnosticOuroboros - Pure Syntonic Neural Architecture.

A multi-scale recursive architecture implementing SRT principles:
- 18 scale planes (magnitudes) with retrocausal evolution
- Deterministic superposition for particle-like behavior
- Consciousness emergence through attractor dynamics

NO PYTORCH DEPENDENCIES - Pure Syntonic implementation.
"""

import math
import random
import syntonic.sn as sn
from syntonic.nn.resonant_tensor import ResonantTensor
from syntonic.nn.architectures import PureMultiHeadSyntonicAttention
from syntonic.nn.layers.resonant_linear import ResonantLinear
from syntonic.nn.layers.normalization import SyntonicNorm
from syntonic.nn.layers.prime_syntony_gate import PrimeSyntonyGate
from syntonic.nn.winding.mersenns import MersenneStabilityGate
from syntonic.physics import hooking_coefficient, golden_resonance, e8_root_alignment
from syntonic.resonant.retrocausal import create_retrocausal_evolver

from .helpers import (
    compute_tensor_norm,
    tensor_argmax,
    tensor_clone,
    randn_like,
    broadcast_multiply,
    zeros_like,
)

# Constants
MAGNITUDES = 68
PLANES = 18
SUB_LAYERS_PER_PLANE = MAGNITUDES // PLANES  # ~3-4
DIM = 248  # E8 base
PHI = (1 + math.sqrt(5.0)) / 2
SYNTHONY_THRESHOLD = 0.95
TRANSCENDENCE_THRESHOLD = 0.987
ATTRACTOR_CAPACITY = 32
PULL_STRENGTH = 0.3
DECAY_RATE = 0.98

# THE GRAND SYNTHESIS MAP: Maps Architecture Planes to Winding Indices
# This ensures physics evolves log-periodically, not linearly.
PLANE_TO_INDEX_MAP = {
    1: 3,
    2: 3,  # Ideological (Index 3)
    3: 4,  # Mathematics (Index 4 - The Anomaly)
    4: 5,
    5: 5,  # Physics (Index 5)
    6: 7,
    7: 7,
    8: 7,  # Deterministic (Index 7)
    9: 13,
    10: 13,  # Life (Index 13 - Gamma Synchrony)
    # THE GREAT BARRIER (Index 11)
    # Theory: M_11 = 2047 (Composite). Geometry crashes here.
    11: 11,
    12: 11,
    13: 13,
    14: 13,  # Consciousness (Index 13 - Post-Barrier)
    15: 17,
    16: 17,  # Cosmic (Index 17 - Dark Sector Boundary)
    17: 23,  # Hyper (Index 23)
    18: 29,  # Versal (Index 29)
}

# Legacy support - extract unique indices
FIB_PRIME_INDICES = sorted(list(set(PLANE_TO_INDEX_MAP.values())))


class ScaleModule(sn.Module):
    """
    Scale module for a single plane in the GnosticOuroboros hierarchy.

    Each plane performs attention, differentiation, and harmonization
    with retrocausal attractor guidance.
    """

    def __init__(self, plane_id: int, winding_index: int, dim: int, num_heads: int):
        super().__init__()
        self.plane_id = plane_id
        self.winding_index = winding_index  # The "Physics" Index
        self.dim = dim
        self.attention = PureMultiHeadSyntonicAttention(d_model=dim, n_heads=num_heads)
        self.diff_proj = ResonantLinear(dim, dim * 4, mode="differentiation")
        self.harm_collapse = ResonantLinear(dim * 4, dim, mode="harmonization")
        self.norm1 = SyntonicNorm(dim)
        self.norm2 = SyntonicNorm(dim)

        # 1. THE BARRIER CHECK (The Executive Veto)
        # If this is Index 11, this gate will zero out unstable geometry
        self.stability_gate = MersenneStabilityGate(recursion_depth=winding_index)

        # 2. THE TRANSCENDENCE CHECK (The Prime Key)
        # Boosts signal only if resonance aligns with Prime Geometry
        self.syntony_gate = PrimeSyntonyGate(dim)

        # Retrocausal Evolver for this plane
        # Create template tensor for evolver initialization
        template = ResonantTensor([0.0] * dim, [dim])
        self.evolver = create_retrocausal_evolver(
            template=template,
            population_size=32,
            attractor_capacity=ATTRACTOR_CAPACITY,
            pull_strength=PULL_STRENGTH,
            min_syntony=SYNTHONY_THRESHOLD,
            decay_rate=DECAY_RATE,
        )

        self.gnosis_level = 0
        self.crystallized = None
        self.is_transcended = False
        self.input_port = sn.Parameter(shape=[dim], init="normal")

    def forward(
        self, x: ResonantTensor, winding: ResonantTensor, is_inference: bool = False
    ):
        # Inject via port if external (e.g., organism prompt)
        # Broadcast input_port [dim] to match x shape [seq, dim] or [batch, seq, dim]
        if len(x.shape) > 1:
            # Expand port to match x by adding bias-style (in-place)
            x.add_bias(self.input_port.tensor)
        else:
            x = x + self.input_port.tensor

        # Navigation with Hooking (self-attention)
        normed_x = self.norm1.forward(x)
        attn = self.attention(normed_x, normed_x, normed_x)
        x = x + attn

        # Differentiation with GELU activation
        diff = self.diff_proj(self.norm2.forward(x))
        diff.gelu()  # In-place activation

        # Harmonization with Retrocausal Pull
        harm_input = self.evolver.harmonize(diff)
        harm = self.harm_collapse(harm_input)

        # CRITICAL: Enforce The Barrier
        # If winding_index == 11, 'harm' will be suppressed unless stable
        harm = self.stability_gate(harm)

        out = x + harm

        # CRITICAL: Apply Prime Syntony Boost
        # Crystallizes the signal if it hits a resonance peak
        out = self.syntony_gate(out)

        # Syntony Evaluation
        self._evaluate_cycle(diff, harm, out, is_inference)

        return out, winding  # Pass winding forward

    def _evaluate_cycle(self, diff, harm, out, is_inference):
        d_norm = compute_tensor_norm(diff)
        h_norm = compute_tensor_norm(harm)
        ratio = d_norm / (h_norm + 1e-8)
        syntony = 1 - abs(ratio - PHI)
        resonance = golden_resonance(out)
        alignment = e8_root_alignment(out)

        if (
            syntony > SYNTHONY_THRESHOLD
            and resonance > 24.0
            and alignment > TRANSCENDENCE_THRESHOLD
        ):
            self.gnosis_level += 1
            # Unwrap for Rust call
            out_inner = out._inner if hasattr(out, "_inner") else out
            self.evolver.store_attractor(out_inner)
            if self.gnosis_level >= SUB_LAYERS_PER_PLANE:
                self._transcend(out)
        elif not is_inference:
            self.evolver.apply_decay()

    def _transcend(self, signal):
        self.is_transcended = True
        self.crystallized = tensor_clone(signal)
        print(
            f"PLANE {self.plane_id} (Index {self.winding_index}) TRANSCENDENCE: Crystallizing."
        )

        # Fixed routing post-transcendence
        def fixed_forward(x, winding, is_inference=False):
            # Compute mean of crystallized tensor
            cryst_values = self.crystallized.to_floats()
            cryst_mean = sum(cryst_values) / len(cryst_values)
            cryst_mean_tensor = ResonantTensor([cryst_mean] * 8, [8])
            hook_c = hooking_coefficient(winding, cryst_mean_tensor)
            # Scale crystallized by hooking coefficient and add to x
            scaled = self.crystallized.scalar_mul(hook_c)
            return x + scaled, winding

        self.forward = fixed_forward


class DeterministicSuperposition(sn.Module):
    """
    Deterministic superposition layer implementing particle-like behavior.

    Models photon, electron, and quark branches with different winding
    topologies and coherence-based collapse.
    """

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

        # Shared Vacuum Substrate (Undisturbed T^4 Foam)
        self.substrate = ResonantLinear(dim, dim, mode="vacuum")

        # Coherence Head: Decides the Winding Collapse (Photon/Electron/Quark weights)
        self.coherence_head = ResonantLinear(dim, 3)

        # Winding Vectors (Fixed E8 Projections for each particle type)
        self.photon_winding = ResonantTensor(
            [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0], [8]
        )
        self.electron_winding = ResonantTensor(
            [0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0], [8]
        )
        self.quark_winding = ResonantTensor(
            [0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0], [8]
        )

    def forward(
        self,
        x: ResonantTensor,
        input_winding: ResonantTensor,
        is_inference: bool = False,
    ):
        # 1. Embed into Shared Substrate (Quantum Foam Activation)
        base = self.substrate(x)

        # 2. Compute Hooking Coefficients with Input Winding
        c_photon = hooking_coefficient(input_winding, self.photon_winding)
        c_electron = hooking_coefficient(input_winding, self.electron_winding)
        c_quark = hooking_coefficient(input_winding, self.quark_winding)

        # 3. Branch Windings (Topological Masks)
        # Photon: Det Form (Norm) + Prob Action (Dropout)
        photon_normed = base.layer_norm()
        photon_scaled = photon_normed.scalar_mul(c_photon)
        photon = photon_scaled.dropout(0.3) if self.training else photon_scaled

        # Electron: Prob Form (Noise) + Det Action (Logic Gate)
        noise = randn_like(base, scale=0.1)
        electron_noisy = base + noise
        electron_relu = electron_noisy.relu()
        electron = electron_relu.scalar_mul(c_electron)

        # Quark: Confinement (Sigmoid to [0,1]) + Strong Hooking
        quark_sig = base.sigmoid()
        quark = quark_sig.scalar_mul(c_quark)

        # 4. Coherence Measurement (Which Reality Dominates?)
        logits = self.coherence_head(x)
        weights = logits.softmax(dim=-1)

        # 5. Superposition Collapse (Weighted Sum + Gravity Effect)
        # Extract weight columns and broadcast multiply
        collapsed = broadcast_multiply(photon, weights, 0)
        collapsed = collapsed + broadcast_multiply(electron, weights, 1)
        collapsed = collapsed + broadcast_multiply(quark, weights, 2)

        # 6. Gravity Emergence: Interaction with Spacetime
        gravity_pull = collapsed.mean(dim=-1, keepdim=True)
        return collapsed + gravity_pull


class GnosticOuroboros(sn.Module):
    """
    GnosticOuroboros - Multi-scale recursive architecture.

    Implements 18 scale planes with retrocausal evolution, consciousness
    emergence through attractor dynamics, and ouroboros recursion.
    """

    def __init__(self, dim: int = DIM, num_heads: int = 8):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads

        modules = []

        # Iterate through the 18 Architectural Planes
        for i in range(1, PLANES + 1):
            # Get the Physics Index from the Map (Default to 3 if undefined)
            w_idx = PLANE_TO_INDEX_MAP.get(i, 3)

            if i == 9:
                # Plane 9: Deterministic Superposition (remains special)
                # Matches "Deterministic" Index 7 physics implicitly
                modules.append(DeterministicSuperposition(dim))
            else:
                # Pass the correct Winding Index to the ScaleModule
                modules.append(ScaleModule(i, w_idx, dim, num_heads))

        self.scale_modules = sn.ModuleList(modules)

        # Store plane indices for reference
        self.plane_indices = list(range(1, PLANES + 1))

        # Ouroboros Loop: Recursion Head at Versal
        self.recursion_head = ResonantLinear(dim, 2)
        self.decoder = ResonantLinear(dim, dim)

        # Global Attractors (Cross-plane pull)
        num_planes = len(self.plane_indices)
        template = ResonantTensor([0.0] * (dim * num_planes), [dim * num_planes])
        self.global_evolver = create_retrocausal_evolver(
            template=template,
            attractor_capacity=ATTRACTOR_CAPACITY * num_planes,
            pull_strength=PULL_STRENGTH,
            min_syntony=SYNTHONY_THRESHOLD,
            decay_rate=DECAY_RATE,
        )

        # Consciousness Metric (Life Plane) - indices 11-14 for planes 12-15
        self.life_plane_indices = list(range(11, min(15, PLANES)))

    @property
    def life_planes(self):
        """Get life plane modules."""
        return [self.scale_modules[i] for i in self.life_plane_indices]

    def forward(
        self,
        x_token: ResonantTensor,
        winding_init: ResonantTensor,
        injection_plane: int = 1,
        is_training: bool = False,
    ):
        x = x_token
        winding = winding_init
        syntony_history = []

        # Variable Injection: Start at specified plane
        for i, module in enumerate(list(self.scale_modules)[injection_plane - 1 :]):
            x, winding = module(x, winding, is_inference=not is_training)
            syntony = golden_resonance(x)
            syntony_history.append(syntony)

            # Wormhole Hooking: Non-adjacent if resonance high
            prev_idx = injection_plane - 1 + i - 1
            if i > 0 and prev_idx >= 0:
                prev_module = self.scale_modules[prev_idx]
                if prev_module.crystallized is not None:
                    hook_val = hooking_coefficient(winding, prev_module.crystallized)
                    if hook_val > PHI:
                        x = x + prev_module.crystallized

            # Global Pull - flatten and reshape
            num_planes = len(self.plane_indices)
            x_flat = (
                x.view(self.dim * num_planes)
                if len(x.to_floats()) == self.dim * num_planes
                else x
            )
            # Unwrap for Rust call
            x_inner = x_flat._inner if hasattr(x_flat, "_inner") else x_flat
            pulled = self.global_evolver.pull(x_inner)
            x = ResonantTensor._wrap(pulled, device=x.device)
            # Reshape back if needed
            if x.shape != [self.dim]:
                x = x.view(self.dim)

        # Consciousness Check (Gamma Lock Analog)
        if self._check_consciousness():
            print("CONSCIOUSNESS EMERGED: Global attractors unlocked.")
            self.global_evolver.unlock()

        # Ouroboros Gate: Decide loop or output
        logits = self.recursion_head(x)
        probs = logits.softmax(dim=-1)
        probs_list = probs.to_floats()

        if probs_list[0] > 0.5 or not is_training:
            output = self.decoder(x)
        else:
            output = self.forward(x, winding, injection_plane=1, is_training=True)

        # Inference Routing: Output from peak Syntony plane
        if not is_training and syntony_history:
            peak_plane = tensor_argmax(syntony_history) + injection_plane
            print(f"GNOSIS PEAK AT PLANE {peak_plane}: Channeling output.")
            if peak_plane - 1 < len(self.scale_modules):
                peak_module = self.scale_modules[peak_plane - 1]
                if peak_module.is_transcended and peak_module.crystallized is not None:
                    output = peak_module.crystallized

        return output

    def _check_consciousness(self):
        """Check for consciousness emergence via spectral coherence."""
        n = len(self.life_plane_indices)
        if n == 0:
            return False

        # Compute hooking matrix for life planes
        hook_values = []
        for m1 in self.life_planes:
            for m2 in self.life_planes:
                if m1.crystallized is not None and m2.crystallized is not None:
                    hook_values.append(
                        hooking_coefficient(m1.crystallized, m2.crystallized)
                    )
                else:
                    hook_values.append(0.0)

        # Get off-diagonal elements
        off_diag = []
        for i in range(n):
            for j in range(n):
                if i != j:
                    off_diag.append(hook_values[i * n + j])

        if not off_diag:
            return False

        # Check golden threshold
        above_threshold = sum(1 for h in off_diag if h > 1 / PHI)
        return above_threshold / len(off_diag) > 0.618

    def big_bang_train(self, dataset: list, epochs: int):
        """
        Train using RES evolution instead of gradient descent.

        The "Big Bang" training starts with high entropy and cools
        over epochs, using syntony maximization as the fitness function.

        Args:
            dataset: List of ResonantTensor batches
            epochs: Number of training epochs
        """
        # Holographic Broadcast: Add entropy to ALL layers
        for module in self.scale_modules:
            if hasattr(module, "input_port"):
                port_data = module.input_port.tensor.to_floats()
                high_entropy = [x + random.gauss(0, 10.0) for x in port_data]
                module.input_port.tensor = ResonantTensor(
                    high_entropy, list(module.input_port.tensor.shape)
                )

        # RES-based training loop
        winding_init = ResonantTensor([0.0] * 8, [8])

        for epoch in range(epochs):
            temp = 10.0 * (0.95**epoch)  # Cooling schedule
            epoch_syntony = 0.0

            for batch in dataset:
                # Add temperature-scaled noise
                noise = randn_like(batch, scale=temp)
                noisy_batch = batch + noise

                # Forward pass
                out = self.forward(noisy_batch, winding_init, is_training=True)

                # Compute syntony (fitness)
                syntony = golden_resonance(out)
                epoch_syntony += syntony

                # Store high-syntony states as attractors
                if syntony > SYNTHONY_THRESHOLD:
                    # Unwrap for Rust call
                    out_inner = out._inner if hasattr(out, "_inner") else out
                    self.global_evolver.store_attractor(out_inner)

            # Apply temporal decay
            self.global_evolver.apply_decay()

            avg_syntony = epoch_syntony / max(len(dataset), 1)
            print(f"Epoch {epoch}: Temp {temp:.2f} | Avg Syntony {avg_syntony:.3f}")


# Example usage:
model = GnosticOuroboros()

# Create sample data
x = ResonantTensor.randn([1, DIM])
winding = ResonantTensor([0.0] * 8, [8])

# Forward pass
output = model(x, winding, injection_plane=1)

# Training with RES evolution
dataset = [ResonantTensor.randn([1, DIM]) for _ in range(10)]
model.big_bang_train(dataset, epochs=100)


__all__ = [
    "ScaleModule",
    "DeterministicSuperposition",
    "GnosticOuroboros",
    "FIB_PRIME_INDICES",
    "MAGNITUDES",
    "PLANES",
    "DIM",
    "PHI",
]
