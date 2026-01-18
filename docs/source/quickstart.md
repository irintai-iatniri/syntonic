# Quick Start Guide

This guide walks you through the core concepts and basic usage of Syntonic.

## Core Concepts

### The Golden Ratio (φ)

Syntonic is built around the golden ratio:

```python
from syntonic.core import PHI, PHI_INV

print(f"φ = {PHI}")        # 1.6180339887498948
print(f"1/φ = {PHI_INV}")  # 0.6180339887498948
```

The golden ratio satisfies the fundamental identity: φ² = φ + 1

### Syntony

**Syntony** is a measure of resonance quality, computed as:

$$\mathcal{S}[\psi] = 1 - H_\varphi[\psi]$$

Where $H_\varphi$ is the golden-base entropy. The target equilibrium is:

$$\mathcal{S}^* = \frac{1}{\varphi} \approx 0.618$$

### The DHSR Cycle

The Differentiation-Harmonization-Syntony-Recursion (DHSR) cycle is the fundamental
operation that drives systems toward the golden fixed point:

1. **Differentiation (D̂)**: Introduces complexity through Fourier perturbations
2. **Harmonization (Ĥ)**: Damps non-golden modes, increases coherence
3. **Syntony (Ŝ)**: Measures and reports resonance quality
4. **Recursion (R̂)**: Records state and prepares for next cycle

## Basic Usage

### Creating Resonant States

```python
from syntonic.core import ResonantState

# From Python list
state = ResonantState.from_floats([1.0, 2.0, 3.0, 4.0])

# Check syntony
print(f"Initial syntony: {state.syntony():.6f}")
```

### Applying the DHSR Cycle

```python
# Apply full DHSR cycle
for i in range(10):
    state.differentiate()
    state.harmonize()
    syntony = state.compute_syntony()
    state.recurse()
    print(f"Cycle {i+1}: S = {syntony:.6f}")
```

### Using Neural Network Layers

```python
from syntonic.nn import GoldenGELU, PhiResidualConnection

# Golden GELU activation: x * sigmoid(φ * x)
gelu = GoldenGELU()

# Phi-residual connection: out = φ * residual + (1/φ) * x
residual = PhiResidualConnection()
```

## GPU Acceleration

```python
import syntonic

if syntonic.cuda_available():
    # Move state to GPU
    state = state.to_device(0)
    
    # Operations automatically use CUDA kernels
    state.differentiate()
```

## Next Steps

- Read the [Theory documentation](theory/dhsr.md) for mathematical foundations
- Explore the [API Reference](api/core.md) for detailed documentation
- Check out [Examples](https://github.com/irintai-iatniri/syntonic/tree/main/examples)
