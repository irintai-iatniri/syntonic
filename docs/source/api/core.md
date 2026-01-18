# Core Module

```{eval-rst}
.. automodule:: syntonic.core
   :members:
   :undoc-members:
   :show-inheritance:
```

## State Management

```{eval-rst}
.. automodule:: syntonic.core.state
   :members:
   :undoc-members:
   :show-inheritance:
```

## Constants

The core module exports fundamental SRT constants:

| Constant | Value | Description |
|----------|-------|-------------|
| `PHI` | 1.6180339887498948 | Golden ratio φ |
| `PHI_INV` | 0.6180339887498948 | Inverse φ⁻¹ |
| `Q_DEFICIT` | 0.027395146920 | Universal syntony deficit |

## Example Usage

```python
from syntonic.core import ResonantState, PHI

# Create a resonant state
state = ResonantState.from_floats([1.0, PHI, PHI**2])

# Compute syntony
print(f"Syntony: {state.syntony():.6f}")

# Apply DHSR operators
state.differentiate()
state.harmonize()
```
