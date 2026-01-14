# Syntonic Softmax Implementation Plan

## Overview

This plan addresses the issues identified in the syntonic_softmax review. The implementation has solid CUDA kernels and Rust infrastructure, but requires bug fixes, feature completions, and testing before production use.

**Current Status**:
- ✅ CUDA kernels (f64 learned/provided modes, contiguous + strided)
- ✅ Rust state management
- ✅ Python bindings (partial)
- ❌ Mode norm initialization bug (blocks proper RES evolution)
- ❌ No Python integration in neural networks
- ⚠️ Missing F32 support and GPU identity mode

**Goal**: Fix critical bugs, complete missing features, and validate correctness.

---

## Implementation Priorities

### Phase 1: Critical Fixes (MUST FIX)
1. **Fix mode norm initialization bug** - Blocks proper RES evolution
2. **Add basic tests** - Validate correctness before further development

### Phase 2: Feature Completion (HIGH PRIORITY)
3. **Implement F32 strided kernels** - Performance improvement and correct dtype dispatch for 8D weights
4. **Add GPU identity mode** - Fair benchmarking and efficient GPU-only softmax
5. **Expose all modes to Python** - API completeness and 8D-aware inputs
6. **Export per-feature weights from Rust for plotting** - small helper to avoid recomputing projections in Python

### Phase 3: Robustness (MEDIUM PRIORITY)
6. **Comprehensive error handling** - Production readiness and 8D input validation
7. **Extended test suite** - Numerical stability, Golden Cone logic, dtype/device consistency

### Phase 4: Integration (OPTIONAL - User Dependent)
8. **Python neural network integration** - update layers and examples for 8D weights
9. **Documentation, plotting helpers and examples** - notebook, Rust helper, and README

---

## Temporal plotting adjustments for 8D softmax

With the move from scalar mode-norms to 8-dimensional E8 lattice vectors, the temporal plotting guidance must be updated. The new weights are computed from 8D E8 roots via the Golden Projector and Golden Cone filter, so plotting should surface the physically meaningful projections and the Golden Cone mask rather than raw lattice coordinates.

Key points:
- **Mode tensor shape**: learned `mode_norms` are now stored as an `[num_features, 8]` ResonantTensor (8 components per feature). Treat the second axis as the E8 vector components.
- **Golden projection**: compute the parallel projection `P_parallel(v) ∈ R^4` using the GoldenProjector (phi-based projector). For visualization prefer the physical magnitude `|P_parallel(v)|` (or squared) rather than raw 8D coordinates.
- **Weight-to-plot mapping**: the syntonic weight used in the softmax is
    $w(v) = \exp(-\|P_{\parallel}(v)\|^2 / \phi)$. Plot either `w` (linear) or `log10(w)` / `-\|P_\parallel\|^2` for better dynamic-range visibility.
- **Golden Cone mask**: compute `Q(v) = \|P_{\parallel}(v)\|^2 - \|P_{\perp}(v)\|^2`. Features with `Q < 0` are outside the Golden Cone and heavily suppressed; overlay or highlight them in plots.
- **Temporal aggregation**: for per-step visualization (training steps / RES iterations) show either:
    - top-K features by weight per step (sparse view)
    - heatmap of weights across features × steps (log-scaled colormap)
    - summary statistics: mean(|P_parallel|), fraction inside Golden Cone, top-1 weight evolution
- **Spatial/time decomposition**: `P_parallel` encodes 1 time + 3 spatial components (by convention). You can plot the time-like component independently to track temporal alignment across modes.

Example (Python, vectorized):

```python
import numpy as np
from math import sqrt

phi = 1.6180339887498949
norm = 1.0 / sqrt(1.0 + phi * phi)

# `arr` is shape [N,8] numeric array from `mode_norms.to_floats()` reshaped
P_par = np.empty((arr.shape[0], 4))
P_perp = np.empty((arr.shape[0], 4))
for i in range(4):
        P_par[:, i] = (arr[:, i] + phi * arr[:, i + 4]) * norm
        P_perp[:, i] = (phi * arr[:, i] - arr[:, i + 4]) * norm

par_sq = (P_par ** 2).sum(axis=1)
perp_sq = (P_perp ** 2).sum(axis=1)
Q = par_sq - perp_sq
weights = np.exp(-par_sq / phi)

# Plot weights over features/steps using matplotlib (log-scale recommended)
```

Visualization recommendations:
- Use a diverging colormap for `Q` (highlight cone boundary at 0).
- Use `imshow(..., norm=LogNorm())` or `np.log10(weights)` for heatmaps.
- Smooth short-term noise with a small rolling window (e.g., 3-5 steps) when plotting evolution.
- Increase `precision` in crystallization (`precision=1000`) when you need finer continuous plots (reduces quantization due to GoldenExact snapping).

Performance notes:
- Computing projections and weights on the CPU for very large `N` and many steps may be costly; vectorized NumPy is efficient but consider moving the projection into Rust for batched computation if plotting becomes a bottleneck.
- The default softmax pipeline already computes `w` in Rust; prefer exporting `w` directly from the Rust side (via a small helper) to avoid recomputing the projector in Python.


## Phase 1: Critical Fixes

### Task 1.1: Fix Mode Norm Initialization Bug

**File**: `rust/src/resonant/syntonic_softmax.rs:85-91`

**Current code**:
```rust
Some(
    ResonantTensor::from_floats_default_modes(&vec![1.0; n], vec![n], 100)
        .map_err(|e| PyErr::from(e))?
)
```

**Issue**: All mode norms initialized to 1.0 → constant weight `exp(-1/φ) ≈ 0.527` for all features → loses hierarchical structure

**Fix**:
```rust
// Generate sequential mode norms: [0, 1, 4, 9, 16, ...]
let mode_norms_vec: Vec<f64> = (0..n)
    .map(|i| (i * i) as f64)
    .collect();

Some(
    ResonantTensor::from_floats_default_modes(&mode_norms_vec, vec![n], 100)
        .map_err(|e| PyErr::from(e))?
)
```

**Verification**:
- Check that mode_norms tensor contains `[0.0, 1.0, 4.0, 9.0, ...]`
- Verify weights computed as `w(i) = exp(-i²/φ)` show proper decay
- Test that `w(0) = 1.0`, `w(1) ≈ 0.527`, `w(2) ≈ 0.063`

**Impact**: Enables proper hierarchical golden measure weighting during RES evolution

---

### Task 1.2: Add Basic Correctness Tests

**New file**: `tests/test_core/test_syntonic_softmax.py`

**Test cases**:

1. **Test mode norm initialization**:
   ```python
   def test_mode_norm_initialization():
       state = SyntonicSoftmaxState.new(
           SyntonicSoftmaxMode.Learned,
           dim=-1,
           num_features=10,
           syntony_scale=1.0
       )
       mode_norms = state.mode_norms.to_floats()
       expected = [float(i*i) for i in range(10)]
       np.testing.assert_array_almost_equal(mode_norms, expected)
   ```

2. **Test forward pass shape preservation**:
   ```python
   def test_forward_pass_shapes():
       # Test various input shapes [batch, features]
       for batch_size in [1, 4, 16]:
           for num_features in [10, 100]:
               x = ResonantTensor.from_floats(...)
               state = SyntonicSoftmaxState.new(...)
               y = state.forward(x)
               assert y.shape() == x.shape()
   ```

3. **Test probability normalization**:
   ```python
   def test_probability_sums():
       # Softmax outputs should sum to 1.0 per row
       x = ResonantTensor.from_floats([[1.0, 2.0, 3.0]], [1, 3], ...)
       state = SyntonicSoftmaxState.new(SyntonicSoftmaxMode.Identity, ...)
       y = state.forward(x)
       probs = y.to_floats()
       assert np.isclose(sum(probs), 1.0)
   ```

4. **Test golden weighting**:
   ```python
   def test_golden_measure_weighting():
       # Learned mode should apply exp(-i²/φ) weighting
       x = ResonantTensor.zeros([1, 5])  # Equal logits
       state = SyntonicSoftmaxState.new(SyntonicSoftmaxMode.Learned, ...)
       y = state.forward(x)
       probs = y.to_floats()

       # First feature (i=0, mode_norm=0) should have highest weight
       # Last feature (i=4, mode_norm=16) should have lowest weight
       assert probs[0] > probs[1] > probs[2] > probs[3] > probs[4]
   ```

5. **Test CPU vs GPU consistency** (if CUDA available):
   ```python
   def test_cpu_gpu_consistency():
       x_cpu = ResonantTensor.from_floats(...)
       x_gpu = x_cpu.to_device(0)

       state = SyntonicSoftmaxState.new(SyntonicSoftmaxMode.Learned, ...)

       y_cpu = state.forward(x_cpu)
       y_gpu = state.forward(x_gpu)

       np.testing.assert_allclose(
           y_cpu.to_floats(),
           y_gpu.to_cpu().to_floats(),
           rtol=1e-5
       )
   ```

**Run tests**:
```bash
pytest tests/test_core/test_syntonic_softmax.py -v
```

---

## Phase 2: Feature Completion

### Task 2.1: Implement F32 Strided Kernels

**File**: `rust/kernels/syntonic_softmax.cu` (lines 298-319 exist but incomplete)

**Missing kernels**:
1. `syntonic_softmax_learned_strided_f32`
2. `syntonic_softmax_provided_strided_f32`

**Implementation approach**:
- Copy `syntonic_softmax_learned_strided_f64` structure
- Replace `double` → `float`, `DBL_MAX` → `FLT_MAX`
- Use `__expf()` instead of `exp()`
- Update PHI constant to `PHI_INV_F32`

**Example** (learned strided f32):
```cuda
__global__ void syntonic_softmax_learned_strided_f32(
    float* __restrict__ out,
    const float* __restrict__ x,
    const float* __restrict__ mode_norms,
    const float syntony_scale,
    const int outer_size,
    const int dim_size,
    const int inner_size
) {
    // Similar structure to f64 version
    // See lines 225-272 for reference

    __shared__ float shared_max[256];
    __shared__ float shared_sum[256];

    // ... implementation matching f64 logic
}
```

**Rust wrappers** (`rust/src/tensor/srt_kernels.rs`):
- Add `cuda_syntonic_softmax_learned_strided_f32()`
- Add `cuda_syntonic_softmax_provided_strided_f32()`
- Follow existing f64 wrapper pattern (lines 2285-2362)

**Update state management** (`rust/src/resonant/syntonic_softmax.rs`):
- Detect tensor dtype in `forward_cuda()`
- Dispatch to f32 or f64 kernels based on dtype

**Verification**:
- Compare f32 vs f64 outputs (should match within float precision)
- Benchmark performance improvement (expect ~2x speedup)

---

### Task 2.2: Add GPU Identity Mode

**Current issue**: Identity mode forces GPU→CPU→GPU transfer (lines 324-335)

**Solution**: Implement standard softmax CUDA kernel

**New kernel** (`rust/kernels/syntonic_softmax.cu`):
```cuda
__global__ void softmax_f64(
    double* __restrict__ out,
    const double* __restrict__ x,
    const int batch_size,
    const int num_classes
) {
    // Standard numerically stable softmax
    // No golden measure weighting

    int batch_idx = blockIdx.x;
    if (batch_idx >= batch_size) return;

    int tid = threadIdx.x;
    __shared__ double shared_max[256];
    __shared__ double shared_sum[256];

    // Step 1: Find max
    double local_max = -DBL_MAX;
    for (int i = tid; i < num_classes; i += blockDim.x) {
        local_max = fmax(local_max, x[batch_idx * num_classes + i]);
    }
    shared_max[tid] = local_max;
    __syncthreads();

    // Reduce max
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shared_max[tid] = fmax(shared_max[tid], shared_max[tid + stride]);
        }
        __syncthreads();
    }
    double max_val = shared_max[0];
    __syncthreads();

    // Step 2: Compute exp and sum
    double local_sum = 0.0;
    for (int i = tid; i < num_classes; i += blockDim.x) {
        local_sum += exp(x[batch_idx * num_classes + i] - max_val);
    }
    shared_sum[tid] = local_sum;
    __syncthreads();

    // Reduce sum
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shared_sum[tid] += shared_sum[tid + stride];
        }
        __syncthreads();
    }
    double sum = shared_sum[0];

    // Step 3: Normalize
    for (int i = tid; i < num_classes; i += blockDim.x) {
        out[batch_idx * num_classes + i] =
            exp(x[batch_idx * num_classes + i] - max_val) / sum;
    }
}
```

**Add strided version** for arbitrary dimensions

**Update Rust** (`rust/src/resonant/syntonic_softmax.rs:324-335`):
```rust
SyntonicSoftmaxMode::Identity => {
    #[cfg(feature = "cuda")]
    {
        if x.device_idx().is_some() && x.phase() == ResonantPhase::Flux {
            return self.forward_cuda_identity(x);  // New function
        }
    }

    // CPU fallback
    let mut output = x.clone();
    output.softmax_core(Some(self.dim), 32)?;
    Ok(output)
}
```

**Verification**:
- Compare GPU identity mode vs CPU softmax (should match exactly)
- No GPU↔CPU transfers in GPU path
- Benchmark: identity mode should be fast baseline

---

### Task 2.3: Expose All Modes to Python

**Current issue**: Python API only supports Learned (if mode_norms) or Identity (else)

**File**: `rust/src/lib.rs:362-382`

**New signature**:
```rust
#[pyfunction]
#[pyo3(name = "syntonic_softmax")]
pub fn syntonic_softmax_py(
    x: &ResonantTensor,
    dim: Option<isize>,
    mode: Option<String>,  // NEW: "learned", "provided", "identity"
    mode_norms: Option<&ResonantTensor>,
    syntony: Option<&ResonantTensor>,  // NEW: for provided mode
    syntony_scale: Option<f64>,
) -> PyResult<ResonantTensor> {
    // Parse mode
    let mode_enum = match mode.as_deref() {
        Some("learned") | None if mode_norms.is_some() => SyntonicSoftmaxMode::Learned,
        Some("provided") => {
            if syntony.is_none() {
                return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                    "syntony tensor required for provided mode"
                ));
            }
            SyntonicSoftmaxMode::Provided
        },
        Some("identity") | None => SyntonicSoftmaxMode::Identity,
        _ => return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "mode must be 'learned', 'provided', or 'identity'"
        )),
    };

    let num_features = match mode_enum {
        SyntonicSoftmaxMode::Learned => {
            mode_norms.map(|t| t.shape()[0])
        },
        _ => None,
    };

    let mut state = SyntonicSoftmaxState::new(
        mode_enum,
        dim,
        num_features,
        syntony_scale,
    )?;

    // Set mode_norms or syntony based on mode
    if mode_enum == SyntonicSoftmaxMode::Learned && mode_norms.is_some() {
        state.mode_norms = Some(mode_norms.unwrap().clone());
    }

    state.forward(x, syntony)
}
```

**Python usage**:
```python
# Learned mode (evolve mode norms)
mode_norms = ResonantTensor.from_floats([0, 1, 4, 9, 16], [5], 100)
out = syntonic_softmax(x, mode='learned', mode_norms=mode_norms)

# Provided mode (use pre-computed weights)
syntony = ResonantTensor.from_floats(precomputed_weights, [batch, features], 100)
out = syntonic_softmax(x, mode='provided', syntony=syntony)

# Identity mode (standard softmax)
out = syntonic_softmax(x, mode='identity')
```

**Verification**:
- Test all three modes from Python
- Verify error messages for invalid mode combinations

---

## Phase 3: Robustness

### Task 3.1: Add Comprehensive Error Handling

**Locations to update**:

1. **Python API validation** (`rust/src/lib.rs`):
```rust
// Validate shapes
if x.ndim() < 1 {
    return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
        "input must have at least 1 dimension"
    ));
}

if let Some(norms) = mode_norms {
    if norms.ndim() != 1 {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "mode_norms must be 1D tensor"
        ));
    }
    // Validate size matches input dimension
}

if let Some(syn) = syntony {
    if syn.shape() != x.shape() {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "syntony shape must match input shape"
        ));
    }
}
```

2. **GPU memory allocation checks** (`rust/src/resonant/syntonic_softmax.rs:241-248`):
```rust
let flux_ref = device
    .flux_ref()
    .ok_or_else(|| {
        ResonantError::DeviceError(
            "GPU flux storage not available".to_string()
        )
    })?;

let mut out_flux = flux_ref
    .try_alloc(out_len)?  // Use try_alloc instead of alloc
    .ok_or_else(|| {
        ResonantError::DeviceError(
            "Failed to allocate GPU memory for output".to_string()
        )
    })?;
```

3. **CUDA kernel bounds checking** (optional, debug builds):
```cuda
__global__ void syntonic_softmax_learned_f64(...) {
    int batch_idx = blockIdx.x;

    #ifdef DEBUG
    if (batch_idx >= batch_size) return;
    if (threadIdx.x >= 256) return;
    #endif

    // ... kernel logic
}
```

4. **Numerical stability guards** (`rust/src/resonant/syntonic_softmax.rs`):
```rust
// Check for NaN/Inf in input
if x.to_floats().iter().any(|v| !v.is_finite()) {
    return Err(ResonantError::NumericalError(
        "Input contains NaN or Inf values".to_string()
    ));
}

// Check for degenerate mode norms (all very large)
if let Some(norms) = &self.mode_norms {
    let max_norm = norms.to_floats().iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    if max_norm > 1e6 {
        warn!("Very large mode norms detected: max = {}", max_norm);
    }
}
```

**Verification**:
- Test invalid inputs (wrong shapes, NaN, Inf)
- Test GPU memory exhaustion scenarios
- Verify graceful error messages

---

### Task 3.2: Extended Test Suite

**New test file**: `tests/test_core/test_syntonic_softmax_extended.py`

**Test cases**:

1. **Numerical stability**:
   - Very large logits (1e6)
   - Very small logits (-1e6)
   - Mixed large/small logits
   - Zero logits
   - Nearly equal logits

2. **Edge cases**:
   - Single feature (num_classes=1)
   - Large number of features (10,000+)
   - Different batch sizes (1, 2, 16, 256)
   - Different dimensions (last, first, middle)

3. **Mode comparisons**:
   - Learned vs Identity difference should decrease as mode_norms → 0
   - Provided mode should match learned when syntony = weights
   - Identity mode should match standard softmax exactly

4. **RES evolution simulation** (if time permits):
   - Initialize with random mode_norms
   - Evolve via RES for several generations
   - Verify syntony increases over time
   - Check convergence to stable state

5. **Performance benchmarks**:
   - CPU vs GPU speedup
   - F64 vs F32 speedup
   - Contiguous vs strided performance
   - Learned vs Identity overhead

**Run extended tests**:
```bash
pytest tests/test_core/test_syntonic_softmax_extended.py -v --benchmark
```

---

## Phase 4: Integration (Optional)

### Task 4.1: Python Neural Network Integration

**Create** `python/syntonic/nn/layers/syntonic_classifier.py`:
```python
class SyntonicClassifier(nn.Module):
    """
    Classification head using syntonic softmax with golden measure weighting.

    Args:
        in_features: Input dimension
        num_classes: Number of output classes
        mode: 'learned', 'provided', or 'identity'
        syntony_scale: Weight strength (default 1.0)
    """

    def __init__(self, in_features, num_classes, mode='learned', syntony_scale=1.0):
        super().__init__()
        self.linear = nn.Linear(in_features, num_classes)
        self.mode = mode
        self.syntony_scale = syntony_scale

        if mode == 'learned':
            # Initialize mode norms as learnable ResonantTensor
            mode_norms_data = [float(i*i) for i in range(num_classes)]
            self.mode_norms = ResonantTensor.from_floats(
                mode_norms_data,
                [num_classes],
                precision=100
            )

    def forward(self, x):
        logits = self.linear(x)

        # Convert to ResonantTensor if needed
        if not isinstance(logits, ResonantTensor):
            logits_data = logits.detach().cpu().numpy().flatten().tolist()
            logits = ResonantTensor.from_floats(
                logits_data,
                list(logits.shape),
                precision=100
            )

        # Apply syntonic softmax
        if self.mode == 'learned':
            return syntonic_softmax(
                logits,
                mode='learned',
                mode_norms=self.mode_norms,
                syntony_scale=self.syntony_scale
            )
        else:
            return syntonic_softmax(logits, mode=self.mode)
```

**Example usage**:
```python
# In a transformer or CNN
classifier = SyntonicClassifier(
    in_features=768,
    num_classes=10,
    mode='learned'
)

# Train with RES
trainer = RetrocausalTrainer(classifier, ...)
trainer.train()
```

**Create test**: `tests/test_nn/test_syntonic_classifier.py`

---

### Task 4.2: Documentation and Examples

**Create** `docs/syntonic_softmax_guide.md`:
- Mathematical background
- Usage examples for all three modes
- Performance characteristics
- Integration with RES training
- Comparison with standard softmax

**Add docstrings** to Python bindings

**Create** `examples/syntonic_softmax_demo.py`:
- Simple classification example
- Visualization of golden measure weighting
- RES evolution demonstration

---

## Verification Plan

### After Phase 1 (Critical Fixes):
```bash
# 1. Fix mode norm bug
# Verify: Check rust code shows [0, 1, 4, 9, ...] initialization

# 2. Run basic tests
pytest tests/test_core/test_syntonic_softmax.py -v

# Expected: All tests pass
# - Mode norms are [0, 1, 4, 9, ...]
# - Forward pass preserves shapes
# - Probabilities sum to 1.0
# - Golden weighting shows proper decay
# - CPU/GPU results match
```

### After Phase 2 (Feature Completion):
```bash
# 1. Rebuild with new kernels
cd rust
cargo build --release --features cuda
cd ..
maturin develop --release

# 2. Test F32 support
pytest tests/test_core/test_syntonic_softmax.py::test_f32_support -v

# 3. Test GPU identity mode
pytest tests/test_core/test_syntonic_softmax.py::test_gpu_identity -v

# 4. Test all Python modes
pytest tests/test_core/test_syntonic_softmax.py::test_all_modes -v
```

### After Phase 3 (Robustness):
```bash
# Run extended test suite
pytest tests/test_core/test_syntonic_softmax_extended.py -v

# Check error handling
pytest tests/test_core/test_syntonic_softmax_extended.py::test_error_handling -v

# Run benchmarks
pytest tests/test_core/test_syntonic_softmax_extended.py --benchmark-only
```

### After Phase 4 (Integration):
```bash
# Test neural network integration
pytest tests/test_nn/test_syntonic_classifier.py -v

# Run demo
python examples/syntonic_softmax_demo.py
```

---

## Critical Files to Modify

### Phase 1:
- `rust/src/resonant/syntonic_softmax.rs` (lines 85-91) - Fix init bug
- `tests/test_core/test_syntonic_softmax.py` (new file) - Basic tests

### Phase 2:
- `rust/kernels/syntonic_softmax.cu` (lines 298-end) - F32 strided kernels + identity kernels
- `rust/src/tensor/srt_kernels.rs` (add ~200 lines) - F32 wrappers + identity wrappers
- `rust/src/resonant/syntonic_softmax.rs` (lines 191-341, 324-335) - F32/identity dispatch
- `rust/src/lib.rs` (lines 362-382) - Expose all modes

### Phase 3:
- `rust/src/lib.rs` (lines 362-382) - Input validation
- `rust/src/resonant/syntonic_softmax.rs` (lines 241-248) - GPU error handling
- `tests/test_core/test_syntonic_softmax_extended.py` (new file) - Extended tests

### Phase 4 (Optional):
- `python/syntonic/nn/layers/syntonic_classifier.py` (new file)
- `tests/test_nn/test_syntonic_classifier.py` (new file)
- `docs/syntonic_softmax_guide.md` (new file)
- `examples/syntonic_softmax_demo.py` (new file)

---

## Estimated Effort

- **Phase 1** (Critical): 2-3 hours
  - Fix: 15 minutes
  - Tests: 2-3 hours

- **Phase 2** (High Priority): 6-8 hours
  - F32 kernels: 3-4 hours
  - Identity mode: 2-3 hours
  - Python API: 1 hour

- **Phase 3** (Medium Priority): 4-6 hours
  - Error handling: 2-3 hours
  - Extended tests: 2-3 hours

- **Phase 4** (Optional): 4-6 hours
  - Integration: 2-3 hours
  - Documentation: 2-3 hours

**Total**: 16-23 hours (without Phase 4), 20-29 hours (with Phase 4)

---

## Dependencies

- Rust toolchain (1.70+)
- CUDA toolkit (for kernel compilation)
- Python 3.10+
- pytest (for testing)
- numpy (for test assertions)

---

## Success Criteria

### Minimum (Phase 1 + 2):
- ✅ Mode norm bug fixed
- ✅ Basic tests pass (CPU/GPU correctness)
- ✅ F32 support working
- ✅ GPU identity mode functional
- ✅ All three modes callable from Python

### Complete (Phase 1 + 2 + 3):
- ✅ All minimum criteria
- ✅ Comprehensive error handling
- ✅ Extended test suite passes
- ✅ Numerical stability validated
- ✅ Performance benchmarks documented

### Production-Ready (All Phases):
- ✅ All complete criteria
- ✅ Neural network integration examples
- ✅ Documentation complete
- ✅ Usage examples provided
