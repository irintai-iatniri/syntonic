# Neural Network Module

```{eval-rst}
.. automodule:: syntonic.nn
   :members:
   :undoc-members:
   :show-inheritance:
```

## Layers

### GoldenGELU

```{eval-rst}
.. autoclass:: syntonic.nn.GoldenGELU
   :members:
   :special-members: __init__, __call__
```

**Mathematical Definition:**

$$\text{GeLU}_\varphi(x) = x \cdot \sigma(\varphi \cdot x)$$

Where σ is the sigmoid function.

### PhiResidualConnection

```{eval-rst}
.. autoclass:: syntonic.nn.PhiResidualConnection
   :members:
   :special-members: __init__, __call__
```

**Mathematical Definition:**

$$\text{out} = \varphi \cdot \text{residual} + \frac{1}{\varphi} \cdot x$$

### GoldenBatchNorm

```{eval-rst}
.. autoclass:: syntonic.nn.GoldenBatchNorm1d
   :members:
   :special-members: __init__, __call__
```

### SyntonicSoftmax

```{eval-rst}
.. autoclass:: syntonic.nn.SyntonicSoftmax
   :members:
   :special-members: __init__, __call__
```

## Training

```{eval-rst}
.. automodule:: syntonic.nn.training
   :members:
   :undoc-members:
```

## Example Usage

```python
from syntonic.nn import (
    GoldenGELU,
    PhiResidualConnection,
    GoldenBatchNorm1d,
    SyntonicSoftmax
)

# Create layers
gelu = GoldenGELU()
residual = PhiResidualConnection()
bn = GoldenBatchNorm1d(features=64)
softmax = SyntonicSoftmax()
```
