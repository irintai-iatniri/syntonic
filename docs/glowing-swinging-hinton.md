# Plan: Wire Up Unused Functions and Methods in Syntonic

## Overview
Wire up the 48 warnings about unused code by either:
1. **Exposing to Python** - Add PyO3 bindings for useful diagnostic/monitoring methods
2. **Implementing** - Complete TODO stubs
3. **Integrating** - Connect internal helpers to their call sites
4. **Implement** - Implement dead code

## Critical Files
- `rust/src/tensor/srt_kernels.rs` - Kernel name constants
- `rust/src/resonant/attractor.rs` - AttractorMemory methods
- `rust/src/resonant/evolver.rs` - ResonantEvolver PyO3 bindings
- `rust/src/tensor/cuda/srt_memory_protocol.rs` - SRT transfer protocol
- `rust/src/tensor/cuda/async_transfer.rs` - Async transfer handles
- `rust/src/tensor/broadcast.rs` - Broadcasting helpers
- `rust/src/exact/golden.rs` - Golden ratio utilities
- `rust/src/tensor/storage.rs` - Tensor storage fallbacks
- `rust/src/lib.rs` - Python module exports

---

## Task 1: Expose AttractorMemory Methods to Python

**Status**: These methods exist and work but aren't Python-accessible

### Methods to Wire Up
In `rust/src/resonant/evolver.rs` `#[pymethods]` block (after line 849):

```rust
/// Get the top k attractors by effective weight.
fn get_top_attractors(&self, k: usize) -> Vec<ResonantTensor> {
    self.attractor_memory.get_top_attractors(k)
        .into_iter()
        .cloned()
        .collect()
}

/// Get the syntony values for all attractors.
#[getter]
fn get_attractor_syntony_values(&self) -> Vec<f64> {
    self.attractor_memory.get_syntony_values().to_vec()
}

/// Get the generations when attractors were added.
#[getter]
fn get_attractor_generations(&self) -> Vec<usize> {
    self.attractor_memory.get_generations().to_vec()
}

/// Clear all attractors from memory.
fn clear_attractors(&mut self) {
    self.attractor_memory.clear();
}
```

**Result**: These are already implemented (I added them in the previous session)! Just need to verify warnings are gone.

---

## Task 2: Expose AsyncTransfer Query Methods to Python

**Status**: Useful diagnostic methods not yet exposed

### Expose AsyncTensorTransfer Methods

Add PyO3 bindings in `rust/src/tensor/cuda/async_transfer.rs`:

```rust
#[pymethods]
impl AsyncTensorTransfer {
    /// Check if transfer is ready
    fn is_ready(&self) -> bool {
        self.transfer.is_ready()
    }

    /// Get device index
    #[getter]
    fn device_idx(&self) -> usize {
        self.device_idx
    }

    /// Get tensor shape
    #[getter]
    fn shape(&self) -> Vec<usize> {
        self.shape.clone()
    }

    /// Get tensor dtype string
    #[getter]
    fn dtype(&self) -> String {
        self.dtype.clone()
    }
}
```

**Note**: Check if `TransferComputeOverlap` already has `#[pyclass]` decorator. If not, add it.

---

## Task 3: Fix SRTMemoryTransferProtocol::alloc_srt_pinned

**Status**: TODO stub returning error

### Location
`rust/src/tensor/cuda/srt_memory_protocol.rs:679`

### Implementation

**Implement it properly**
```rust
pub fn alloc_srt_pinned(&self, size: usize) -> Result<Vec<u8>, CudaError> {
    let pool = self.pinned_pool.write().unwrap();
    // Use existing pinned pool infrastructure
    pool.alloc_pinned(size, &self.device)
}
```

---

## Task 4: Add Python Bindings for Inplace Operations

**Status**: Functions exist but Python wrappers missing

### Add to `rust/src/tensor/broadcast.rs`

After the existing `py_inplace_*` functions, add:

```rust
#[pyfunction]
fn py_inplace_sub_scalar(mut data: Vec<f64>, scalar: f64) -> Vec<f64> {
    inplace_sub_scalar(&mut data, scalar);
    data
}

#[pyfunction]
fn py_inplace_div_scalar(mut data: Vec<f64>, scalar: f64) -> Vec<f64> {
    inplace_div_scalar(&mut data, scalar);
    data
}
```

### Add to `rust/src/lib.rs`

In the `_core` module definition (around line 253), add:

```rust
m.add_function(wrap_pyfunction!(py_inplace_sub_scalar, m)?)?;
m.add_function(wrap_pyfunction!(py_inplace_div_scalar, m)?)?;
```

---

## Task 5: Handle SRT Kernel Constants

**Status**: Documentation arrays not validated

### Implementation

**Implement it properly**

Add to `rust/src/tensor/srt_kernels.rs`:

```rust
/// Validate that all listed kernel functions exist in PTX module
#[cfg(feature = "cuda")]
pub fn validate_kernels(device: &Arc<CudaDevice>) -> PyResult<Vec<String>> {
    let (major, minor) = get_compute_capability(device);
    let mut missing = Vec::new();

    // Validate GOLDEN_FUNCS
    let ptx = select_golden_ptx(major, minor);
    let module = device.load_module(cudarc::nvrtc::Ptx::from_src(ptx))
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("Failed to load golden PTX: {}", e)))?;

    for &func_name in GOLDEN_FUNCS {
        if module.load_function(func_name).is_err() {
            missing.push(format!("golden: {}", func_name));
        }
    }

    // Repeat for E8_FUNCS, HEAT_FUNCS, etc...

    Ok(missing)
}
```

**Export to Python** in `lib.rs`:
```rust
#[cfg(feature = "cuda")]
m.add_function(wrap_pyfunction!(srt_kernels::validate_kernels, m)?)?;
```

---

## Task 6: Implement Dead Code

**Status**: Unused helper functions

### Implementation

Implement the helper functions properly.

### Decision on `linear_index` in `broadcast.rs`

**Option A**: Keep and add Python binding
```rust
#[pyfunction]
fn py_linear_index(indices: Vec<usize>, strides: Vec<usize>) -> usize {
    linear_index(&indices, &strides)
}
```

---

## Task 7: Handle CPU Fallback Methods

**Status**: Private methods never called

### Location
- `rust/src/tensor/storage.rs:3024` - `unary_cpu_fallback`
- `rust/src/tensor/storage.rs:3097` - `binary_cpu_fallback`

### Implementation

**Implement it properly**

Implement the CPU fallback methods properly.
---

## Task 8: Implement Remaining Unused Variables

**Status**: Variables that need implementing in alignment with theoretical necessity.

### Files to Update

**`rust/src/resonant/evolver.rs:263`**
```rust
// Implement:
template: Option<ResonantTensor>,

```

**`rust/src/tensor/cuda/srt_memory_protocol.rs:85`**
```rust
// Implement:
current_idx: usize,

```

---

## Implementation Order

1. ✅ **AttractorMemory methods** - Already done in previous session
2. **Implement kernel constants** 
3. **Implement CPU fallbacks** 
4. **Implement unused field names** 
5. **Properly Implement `add` function in golden.rs** 
6. **Implement `linear_index`** 
7. **AsyncTransfer Python exposure** 
8. **Add inplace_sub/div Python bindings** 
9. **Implement alloc_srt_pinned** 

**Total estimated time**: **60 mintutes**

---

## Verification

### Build Test
```bash
cd /home/Andrew/Documents/SRT\ Complete/implementation/syntonic
cargo build --release 2>&1 | grep "warning:" | wc -l
```

**Expected**: Warnings reduced from 48 to ~5-10

### Python API Test
```python
import syntonic as syn

# Test attractor methods
evolver = syn.ResonantEvolver(template, config)
print(evolver.attractor_count)
print(evolver.attractor_syntony_values)
print(evolver.attractor_generations)
evolver.clear_attractors()

# When exposing AsyncTransfer:
transfer = async_transfer.AsyncTensorTransfer.new(...)
print(transfer.is_ready())
print(transfer.device_idx)
print(transfer.shape)

# When adding inplace operations:
from syntonic._core import py_inplace_sub_scalar, py_inplace_div_scalar
data = [1.0, 2.0, 3.0]
result = py_inplace_sub_scalar(data, 0.5)  # [0.5, 1.5, 2.5]
```

### Functional Test
```bash
# Run existing tests
pytest tests/test_resonant/test_evolver.py -v

# Check that new methods work
pytest tests/test_resonant/test_attractor.py -v
```

---

## Notes

- **AttractorMemory methods already implemented** in previous session
- AsyncTransfer and inplace operations are not optional.
- Priority: Implement the plan respecting the theory and existent files.
- All changes are non-breaking (additions only)

---

## Success Criteria

1. ✅ Build completes successfully
2. ✅ Warnings reduced from 48 to <10
3. ✅ No functionality broken
4. ✅ New Python methods are tested as fully functional