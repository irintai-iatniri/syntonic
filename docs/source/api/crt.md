# CRT Module

Cosmological Recursion Theory (CRT) reference implementations.

```{eval-rst}
.. automodule:: syntonic.crt
   :members:
   :undoc-members:
   :show-inheritance:
```

## DHSR Reference

```{eval-rst}
.. automodule:: syntonic.crt.dhsr_reference
   :members:
   :undoc-members:
   :show-inheritance:
```

## Key Functions

### Syntony Computation

```{eval-rst}
.. autofunction:: syntonic.crt.compute_syntony
```

### DHSR Operators

```{eval-rst}
.. autofunction:: syntonic.crt.differentiate
.. autofunction:: syntonic.crt.harmonize
.. autofunction:: syntonic.crt.recurse
```

## Example Usage

```python
from syntonic.crt import (
    compute_syntony,
    differentiate,
    harmonize,
    recurse
)

import numpy as np

# Initial state
psi = np.random.randn(64) + 1j * np.random.randn(64)

# Apply DHSR cycle
psi = differentiate(psi)
psi = harmonize(psi)
syntony = compute_syntony(psi)
psi = recurse(psi)

print(f"Syntony after cycle: {syntony:.6f}")
```
