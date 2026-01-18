# The Golden Ratio in SRT

The golden ratio φ = (1 + √5)/2 ≈ 1.618 is the mathematical foundation of 
Syntony Recursion Theory. This page explores its properties and role in SRT.

## Fundamental Properties

### Definition

$$\varphi = \frac{1 + \sqrt{5}}{2} = 1.6180339887498948...$$

### Key Identities

| Identity | Expression | Value |
|----------|------------|-------|
| Self-similarity | $\varphi^2 = \varphi + 1$ | 2.618... |
| Reciprocal | $1/\varphi = \varphi - 1$ | 0.618... |
| Golden power | $\varphi^n = F_n \varphi + F_{n-1}$ | — |
| Continued fraction | $\varphi = 1 + 1/(1 + 1/(1 + ...))$ | — |

### Fibonacci Connection

$$\lim_{n \to \infty} \frac{F_{n+1}}{F_n} = \varphi$$

Where $F_n$ is the $n$-th Fibonacci number.

## The Q(φ) Ring

The **Q(φ) ring** is the set of algebraic integers of the form:

$$\mathbb{Q}[\varphi] = \{a + b\varphi : a, b \in \mathbb{Q}\}$$

### Properties

1. **Closed under arithmetic**: $+, -, \times, \div$ all stay in Q(φ)
2. **Norm form**: $N(a + b\varphi) = a^2 + ab - b^2$
3. **Conjugate**: $(a + b\varphi)' = a + b(1 - \varphi) = a - b/\varphi$

### Why Q(φ)?

Q(φ) arithmetic is exact—no floating-point errors. This is essential for:
- Preserving theory-correct relationships
- Ensuring reproducibility
- Validating against analytic predictions

## Golden Gaussian

The **golden gaussian weight** governs mode selection:

$$w(\lambda) = e^{-|\lambda|^2/\varphi}$$

This gives exponentially decaying weights for high-frequency modes, with the
decay rate determined by φ.

## Syntony Target

The fundamental fixed point of SRT is:

$$S^* = \frac{1}{\varphi} \approx 0.618$$

This arises from the balance between differentiation (entropy increase) and
harmonization (entropy decrease) in the DHSR cycle.

## Implementation

```python
from syntonic.core import PHI, PHI_INV

# Basic constants
print(f"φ = {PHI}")         # 1.6180339887498948
print(f"1/φ = {PHI_INV}")   # 0.6180339887498948
print(f"φ² = {PHI**2}")     # 2.6180339887498948

# Golden identity: φ² = φ + 1
assert abs(PHI**2 - (PHI + 1)) < 1e-15

# Reciprocal identity: 1/φ = φ - 1
assert abs(PHI_INV - (PHI - 1)) < 1e-15
```
