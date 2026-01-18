# Hierarchy Module

SRT correction hierarchy for particle physics predictions.

```{eval-rst}
.. automodule:: syntonic.hierarchy
   :members:
   :undoc-members:
   :show-inheritance:
```

## Correction Functions

```{eval-rst}
.. autofunction:: syntonic.hierarchy.apply_correction
.. autofunction:: syntonic.hierarchy.apply_special_correction
.. autofunction:: syntonic.hierarchy.apply_suppression
```

## Geometric Divisors

The hierarchy module implements SRT-Zero correction factors using geometric divisors
derived from E₈ lattice structure:

| Divisor | Value | Description |
|---------|-------|-------------|
| 248 | E₈ dimension | Root system dimension |
| 240 | E₈ roots | Number of roots |
| 120 | Positive roots | Half of roots |
| 78 | E₆ dimension | Subgroup dimension |
| 36 | Cone roots | Golden cone roots |
| 27 | E₆(27) | Exceptional representation |
| 24 | D₄ kissing | D₄ kissing number |
| 14 | G₂ dimension | G₂ dimension |

## Correction Types

### Standard Corrections

$$\text{corrected} = \text{value} \times \left(1 \pm \frac{q}{N}\right)$$

### Special Corrections

- `q·φ`: Golden ratio scaling
- `q²/φ`: Squared deficit inverse scaling
- `4q`, `6q`, `8q`: Multiple deficit corrections

### Suppression Factors

- Winding instability: $1/(1+q/\varphi)$# For 
- Recursion penalty: $1/(1+q \cdot \varphi)$
- Double inverse: $1/(1+q/\varphi^2)$

## Example Usage

```python
from syntonic.hierarchy import apply_correction, apply_suppression

# Apply standard correction
m_electron = 0.511  # MeV
m_corrected = apply_correction(m_electron, divisor=248, sign=+1)

# Apply suppression
m_suppressed = apply_suppression(m_corrected, "winding_instability")
```
