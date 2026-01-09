# New ResonantTensor APIs (Phase 1 Purification)

This document describes the new APIs added to `ResonantTensor` to support the library purification effort.

## Overview

Added 6 new methods to enable PyTorch-free neural network implementations:
1. `sigmoid()` - Sigmoid activation
2. `tanh()` - Tanh activation
3. `elementwise_mul()` - Element-wise multiplication (Hadamard product)
4. `elementwise_add()` - Element-wise addition
5. `layer_norm()` - Layer normalization with golden target variance
6. `concat()` - Concatenate tensors along a dimension

All methods operate on the exact Q(φ) golden lattice.

## API Reference

### Activations

#### `sigmoid(precision=100)`
Apply sigmoid activation: σ(x) = 1 / (1 + e^(-x))

```python
from syntonic._core import ResonantTensor

x = ResonantTensor([1.0, 2.0, -1.0, -2.0], [4])
x.sigmoid(precision=100)  # In-place operation
# Result: [0.729, 0.875, 0.271, 0.125]
```

**Parameters:**
- `precision` (int): Lattice precision for crystallization (default: 100)

**Note:** In-place operation. Converts to floats, applies sigmoid, snaps back to Q(φ).

---

#### `tanh(precision=100)`
Apply tanh activation: tanh(x) = (e^(2x) - 1) / (e^(2x) + 1)

```python
x = ResonantTensor([0.0, 1.0, -1.0], [3])
x.tanh(precision=100)  # In-place operation
# Result: [0.0, 0.762, -0.762]
```

**Parameters:**
- `precision` (int): Lattice precision for crystallization (default: 100)

---

### Element-wise Operations

#### `elementwise_mul(other) -> ResonantTensor`
Element-wise multiplication (Hadamard product): self ⊙ other

```python
a = ResonantTensor([1.0, 2.0, 3.0, 4.0], [4])
b = ResonantTensor([2.0, 3.0, 4.0, 5.0], [4])
c = a.elementwise_mul(b)
# Result: [2.0, 6.0, 12.0, 20.0]
```

**Parameters:**
- `other` (ResonantTensor): Tensor with same shape as self

**Returns:** New ResonantTensor with element-wise product

**Note:** Exact Q(φ) arithmetic - no precision loss!

---

#### `elementwise_add(other) -> ResonantTensor`
Element-wise addition: self + other

```python
a = ResonantTensor([1.0, 2.0, 3.0, 4.0], [4])
b = ResonantTensor([2.0, 3.0, 4.0, 5.0], [4])
d = a.elementwise_add(b)
# Result: [3.0, 5.0, 7.0, 9.0]
```

**Parameters:**
- `other` (ResonantTensor): Tensor with same shape as self

**Returns:** New ResonantTensor with element-wise sum

**Note:** Exact Q(φ) arithmetic - no precision loss!

---

### Normalization

#### `layer_norm(gamma=None, beta=None, eps=1e-8, golden_target=False) -> ResonantTensor`
Layer normalization with optional golden target variance.

Normalizes across the last dimension (features). For 2D tensors [batch, features], normalizes each batch sample independently.

```python
# Basic usage
x = ResonantTensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], [2, 3])
normalized = x.layer_norm()
# Each row normalized to mean=0, var=1

# With golden target variance (1/φ ≈ 0.618)
normalized = x.layer_norm(golden_target=True)

# With learnable affine parameters
gamma = ResonantTensor([1.0, 1.0, 1.0], [3])  # scale
beta = ResonantTensor([0.0, 0.0, 0.0], [3])   # shift
normalized = x.layer_norm(gamma=gamma, beta=beta)
```

**Parameters:**
- `gamma` (ResonantTensor, optional): Scale parameter of shape [features]
- `beta` (ResonantTensor, optional): Shift parameter of shape [features]
- `eps` (float): Small constant for numerical stability (default: 1e-8)
- `golden_target` (bool): If True, scale to target variance = 1/φ (default: False)

**Returns:** New ResonantTensor with normalized values

**Formula:**
```
y = (x - mean) / sqrt(var + eps) * [sqrt(1/φ) if golden_target] * gamma + beta
```

---

### Concatenation

#### `ResonantTensor.concat(tensors, dim=-1) -> ResonantTensor` (static method)
Concatenate tensors along a specified dimension.

Preserves exact Q(φ) lattice arithmetic by directly combining GoldenExact values.
All tensors must have compatible shapes (same on all dimensions except the concatenation dimension).

```python
from syntonic._core import ResonantTensor

# Concatenate along first dimension (rows)
a = ResonantTensor([1.0, 2.0], [2])
b = ResonantTensor([3.0, 4.0], [2])
c = ResonantTensor.concat([a, b], dim=0)
# Result: [1.0, 2.0, 3.0, 4.0], shape=[4]

# Concatenate along last dimension (columns) for 2D
x = ResonantTensor([1.0, 2.0, 3.0, 4.0], [2, 2])
y = ResonantTensor([5.0, 6.0, 7.0, 8.0], [2, 2])
z = ResonantTensor.concat([x, y], dim=-1)
# Result: shape=[2, 4], concatenates columns
```

**Parameters:**
- `tensors` (List[ResonantTensor]): List of tensors to concatenate
- `dim` (int): Dimension along which to concatenate (default: -1, last dimension)
  - Supports negative indexing (e.g., -1 = last dimension)

**Returns:** New ResonantTensor with tensors concatenated

**Note:** This is an **exact operation** in Q(φ) - no floating-point error!

**SRT Compatibility:**
- ✅ Preserves exact Q(φ) lattice structure
- ✅ Concatenates mode_norm_sq appropriately
- ✅ Recomputes syntony on the result
- ✅ No loss of precision - pure lattice arithmetic

---

## Implementation Details

### CUDA Kernels
Added to `rust/kernels/core_ops.cu`:
- `sigmoid_f64` / `sigmoid_f32`
- `tanh_f64` / `tanh_f32`

Existing kernels reused:
- `add_f64` / `mul_f64` (from `elementwise.cu`)
- `layer_norm_f64` (already in `core_ops.cu`)

### CPU Fallbacks
All methods have CPU-only implementations:
- Activations: Convert to floats, apply operation, snap to lattice
- Element-wise ops: Exact Q(φ) arithmetic using `GoldenExact` multiplication/addition
- Layer norm: Pure Rust implementation with mean/var computation

### Syntony Preservation
- Activations automatically recompute syntony after snapping to lattice
- Element-wise ops preserve mode structure and recompute syntony
- Layer norm maintains syntony through careful normalization

## Usage Examples

### Building a Pure MLP Layer

```python
from syntonic._core import ResonantTensor
from syntonic.nn.layers import ResonantLinear

class PureMLPLayer:
    def __init__(self, in_features, out_features):
        self.linear = ResonantLinear(in_features, out_features)

    def forward(self, x: ResonantTensor) -> ResonantTensor:
        # Linear transformation
        x = self.linear.forward(x)

        # Layer normalization with golden target
        x = x.layer_norm(golden_target=True)

        # ReLU activation (already existed)
        x.relu()

        return x
```

### Syntonic Gate Implementation

```python
class PureSyntonicGate:
    def __init__(self, features):
        self.gate_linear = ResonantLinear(features, features)

    def forward(self, x: ResonantTensor) -> ResonantTensor:
        # Compute gate values
        gate = self.gate_linear.forward(x)
        gate.sigmoid(precision=100)

        # Apply gating: x * gate
        return x.elementwise_mul(gate)
```

## Testing

All new APIs have been tested and verified:

```bash
python3 -c "
from syntonic._core import ResonantTensor
x = ResonantTensor([1.0, 2.0, -1.0, -2.0], [4])
x.sigmoid()
print(x.to_floats())  # [0.729, 0.875, 0.271, 0.125]
"
```

See `verify_integration.py` for comprehensive integration tests.

## Migration Guide

### PyTorch → Pure Replacements

| PyTorch | Pure Rust Backend |
|---------|-------------------|
| `torch.sigmoid(x)` | `x.sigmoid()` |
| `torch.tanh(x)` | `x.tanh()` |
| `x * y` | `x.elementwise_mul(y)` |
| `x + y` | `x.elementwise_add(y)` |
| `F.layer_norm(x, ...)` | `x.layer_norm(...)` |
| `torch.cat([x, y], dim=0)` | `ResonantTensor.concat([x, y], dim=0)` |

### Key Differences
1. **No autograd**: These are forward-only operations
2. **Exact arithmetic**: Element-wise ops use exact Q(φ) multiplication/addition
3. **Syntony tracking**: All ops maintain/recompute syntony automatically
4. **Precision parameter**: Activations require specifying lattice precision

## Performance Notes

- **CPU mode**: All operations available without CUDA
- **CUDA kernels**: Available but not yet fully integrated (future work)
- **Exact arithmetic**: Element-wise ops are exact in Q(φ) - no floating-point error!
- **Memory**: Operations create new tensors (functional style)

## Next Steps

These APIs enable:
1. ✅ Phase 1.2 - Core Layer Primitives (normalization, differentiation, harmonization, gates)
2. Phase 2 - Winding Components
3. Phase 3 - Network Architectures (MLP, attention, transformer)

See `/home/Andrew/.claude/plans/vast-stirring-puffin.md` for the full purification plan.
