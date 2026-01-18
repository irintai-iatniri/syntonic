# Core Concepts

This page explains the fundamental concepts underlying Syntonic and Syntony Recursion Theory (SRT).

## The Golden Ratio

The golden ratio φ = (1 + √5)/2 ≈ 1.618 is the mathematical foundation of SRT:

| Constant | Value | Description |
|----------|-------|-------------|
| φ | 1.6180339887498948 | Golden ratio |
| 1/φ | 0.6180339887498948 | Inverse golden ratio |
| φ² | 2.6180339887498948 | φ + 1 |
| q | 0.027395146920 | Universal syntony deficit |

## Q(φ) Lattice

The **Q(φ) lattice** consists of numbers of the form:

$$a + b\varphi, \quad a, b \in \mathbb{Q}$$

This ring is closed under addition, subtraction, multiplication, and division,
enabling exact arithmetic without floating-point errors.

## Resonant Tensors

Syntonic uses a dual representation for tensor data:

1. **Lattice (exact)**: Values on the Q(φ) lattice, lossless
2. **Ephemeral (approximate)**: Floating-point flux, for efficiency

```python
from syntonic.resonant import ResonantTensor

# Create with automatic crystallization
tensor = ResonantTensor.from_floats([1.0, 1.618, 2.618])

# Access both representations
lattice = tensor.lattice()      # Exact Q(φ) values
ephemeral = tensor.ephemeral()  # Floating-point approximation
```

## Syntony Measure

**Syntony** quantifies resonance quality as the complement of golden-base entropy:

$$\mathcal{S}[\psi] = 1 - H_\varphi[\psi] = 1 + \sum_i p_i \log_\varphi p_i$$

Where:
- $p_i = |\psi_i|^2 / \sum_j |\psi_j|^2$ are normalized probabilities
- $\log_\varphi$ is the logarithm base φ

**Interpretation:**
- S = 0: Maximum entropy (uniform distribution)
- S = 1: Minimum entropy (single state)
- S* = 1/φ ≈ 0.618: Golden equilibrium target

## DHSR Operators

### Differentiation (D̂)

Introduces complexity to explore solution space:

$$\hat{D}[\psi]_n = \mathcal{F}^{-1}\left[\mathcal{F}[\psi]_k \cdot (1 + \alpha(1-S) \cdot \eta_k)\right]_n$$

### Harmonization (Ĥ)

Damps non-golden modes to increase coherence:

$$\hat{H}[\psi]_n = \psi_n \cdot \left(1 - \beta(S) \cdot (1 - w(n))\right)$$

Where $w(n) = e^{-|n|^2/\varphi}$ is the golden weight.

### Syntony (Ŝ)

Measures current resonance state:

$$\hat{S}[\psi] = 1 - H_\varphi[\psi]$$

### Recursion (R̂)

Records history and updates integration:

$$\hat{R}[\psi]_{t+1} = \varphi \cdot \psi_t + (1/\varphi) \cdot \psi_{t-1}$$

## E₈ Lattice

The E₈ root lattice provides the geometric structure for high-dimensional operations:

- **240 roots**: Unit vectors in 8D satisfying specific constraints
- **120 positive roots**: Half of the roots (in golden cone)
- **36 cone roots**: Roots satisfying the golden cone condition

The golden cone is defined by:

$$Q(\lambda) = \|\lambda_\parallel\|^2 - \varphi \cdot \|\lambda_\perp\|^2 \geq 0$$

Where $\lambda_\parallel$ and $\lambda_\perp$ are projections onto parallel and perpendicular subspaces.
