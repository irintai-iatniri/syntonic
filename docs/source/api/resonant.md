# Resonant Module

```{eval-rst}
.. automodule:: syntonic.resonant
   :members:
   :undoc-members:
   :show-inheritance:
```

## ResonantTensor

```{eval-rst}
.. autoclass:: syntonic.resonant.ResonantTensor
   :members:
   :special-members: __init__
   :undoc-members:
```

## AttractorMemory

```{eval-rst}
.. autoclass:: syntonic.resonant.AttractorMemory
   :members:
   :special-members: __init__
```

## Retrocausal Harmonization

```{eval-rst}
.. autofunction:: syntonic.resonant.harmonize_with_attractor_pull
```

## Example Usage

```python
from syntonic.resonant import ResonantTensor, AttractorMemory

# Create a resonant tensor
tensor = ResonantTensor.from_floats([1.0, 1.618, 2.618, 4.236])

# Access dual representations
lattice = tensor.lattice()      # Exact Q(φ) values
ephemeral = tensor.ephemeral()  # Floating-point

# Compute syntony
print(f"Syntony: {tensor.syntony():.6f}")

# Attractor-guided harmonization
memory = AttractorMemory(capacity=32, threshold=0.7, decay=0.98)
```
