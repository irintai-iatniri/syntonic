# Syntonic CUDA Implementation

## Overview

Syntonic includes full CUDA GPU acceleration via the `cudarc` Rust crate. This enables transparent tensor operations on NVIDIA GPUs with automatic CPU-GPU data transfer.

## Architecture

### Backend Stack

```
Python (syntonic.core.state.State)
         │
         ▼
    PyO3 Bindings
         │
         ▼
Rust (TensorStorage) ─────► cudarc (CUDA bindings)
         │                        │
         ▼                        ▼
   ndarray (CPU)            CUDA Driver API
```

### Key Components

1. **`rust/src/tensor/storage.rs`** - Core tensor storage with CPU and CUDA backends
2. **`python/syntonic/core/state.py`** - Python State class with `.cuda()` and `.cpu()` methods
3. **`python/syntonic/core/device.py`** - Device management and CUDA availability checks

## Supported Data Types on CUDA

| DType | CPU Storage | CUDA Storage | Notes |
|-------|-------------|--------------|-------|
| float32 | `ArrayD<f32>` | `CudaSlice<f32>` | Direct transfer |
| float64 | `ArrayD<f64>` | `CudaSlice<f64>` | Direct transfer |
| complex128 | `ArrayD<Complex64>` | `CudaSlice<f64>` | Interleaved pairs |
| int64 | `ArrayD<i64>` | Not supported | Raises `NotImplementedError` |

## Complex Number Handling

CUDA does not natively support complex numbers in the same way as CPU. Syntonic handles this by storing complex numbers as **interleaved f64 pairs**:

### Memory Layout

```
CPU Complex Array:     [1+2j, 3+4j, 5+6j]
                            │
                            ▼
GPU f64 Array:         [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
                        re₀  im₀  re₁  im₁  re₂  im₂
```

### Implementation (Rust)

**CPU to CUDA transfer:**
```rust
CpuData::Complex128(arr) => {
    // Convert complex to interleaved f64 pairs [re0, im0, re1, im1, ...]
    let interleaved: Vec<f64> = arr.iter()
        .flat_map(|c| vec![c.re, c.im])
        .collect();
    let slice = device.htod_sync_copy(&interleaved)?;
    (CudaData::Complex128(slice), "complex128".to_string())
}
```

**CUDA to CPU transfer:**
```rust
CudaData::Complex128(slice) => {
    let mut host_data = vec![0f64; slice.len()];
    device.dtoh_sync_copy_into(slice, &mut host_data)?;
    // Convert interleaved pairs back to Complex64
    let complex_data: Vec<Complex64> = host_data.chunks(2)
        .map(|c| Complex64::new(c[0], c[1]))
        .collect();
    Ok(CpuData::Complex128(ArrayD::from_shape_vec(dim, complex_data)?))
}
```

## API Usage

### Basic CUDA Operations

```python
import syntonic as syn

# Check CUDA availability
if syn.cuda_is_available():
    print(f"CUDA devices: {syn.cuda_device_count()}")

# Create a state on CPU
psi = syn.state([1.0, 2.0, 3.0, 4.0])
print(psi.device)  # cpu

# Transfer to CUDA
psi_cuda = psi.cuda()
print(psi_cuda.device)  # cuda:0

# Transfer back to CPU
psi_cpu = psi_cuda.cpu()

# Use .to() for device-agnostic code
device = syn.device('cuda:0')
psi_gpu = psi.to(device)
```

### Complex Numbers on CUDA

```python
# Complex tensors work transparently
psi_complex = syn.state([1+2j, 3+4j, 5+6j])
psi_cuda = psi_complex.cuda()  # Stored as interleaved f64 pairs
psi_back = psi_cuda.cpu()      # Reconstructed as complex
assert psi_back.to_list() == [1+2j, 3+4j, 5+6j]
```

### Device Specification

```python
# Multiple ways to specify CUDA device
psi.cuda()        # Default device 0
psi.cuda(0)       # Explicit device 0
psi.cuda(1)       # Device 1 (if available)

psi.to(syn.cpu)              # CPU
psi.to(syn.device('cuda:0')) # CUDA device 0
```

## Build Configuration

### Cargo.toml

```toml
[features]
default = ["cpu"]
cpu = []
cuda = ["cudarc"]

[dependencies]
cudarc = { version = "0.12", optional = true, features = ["cuda-12060"] }
```

### Build Commands

```bash
# CPU only (default)
maturin develop

# With CUDA support
PATH="/usr/local/cuda-13.1/bin:$PATH" \
CUDA_PATH="/usr/local/cuda-13.1" \
maturin develop --features cuda
```

## Rust Data Structures

### CudaData Enum

```rust
#[cfg(feature = "cuda")]
pub enum CudaData {
    Float32(CudaSlice<f32>),
    Float64(CudaSlice<f64>),
    /// Complex128 stored as interleaved f64 pairs [re0, im0, re1, im1, ...]
    Complex128(CudaSlice<f64>),
}
```

### TensorData Enum

```rust
pub enum TensorData {
    Cpu(CpuData),
    #[cfg(feature = "cuda")]
    Cuda {
        data: Arc<CudaData>,
        device: Arc<CudaDevice>,
        shape: Vec<usize>,
        dtype: String,
    },
}
```

## Operations on CUDA Tensors

Currently, CUDA tensors are transferred back to CPU for operations via `ensure_cpu()`. This is a design decision that prioritizes correctness over performance for the initial implementation.

```rust
pub fn add(&self, other: &TensorStorage) -> PyResult<TensorStorage> {
    let a = self.ensure_cpu()?;  // Transfers from CUDA if needed
    let b = other.ensure_cpu()?;
    // ... perform operation on CPU ...
}
```

Future optimization would implement CUDA kernels for arithmetic operations directly on GPU.

## Error Handling

```python
# RuntimeError if CUDA not available
try:
    psi.cuda()
except RuntimeError as e:
    print("CUDA not available")

# NotImplementedError for unsupported types
try:
    int_state = syn.state([1, 2, 3], dtype=syn.int64)
    int_state.cuda()  # Raises NotImplementedError
except NotImplementedError:
    print("Int64 not supported on CUDA")
```

## Testing

The test suite includes comprehensive CUDA tests:

- `test_cuda_is_available` - Verifies CUDA detection
- `test_cuda_device_count` - Verifies device enumeration
- `test_cuda_transfer` - Basic CPU ↔ CUDA transfer
- `test_cuda_device_id` - Device index handling
- `test_cuda_already_on_cuda` - Idempotent transfer
- `test_cuda_large_tensor` - Performance with 1M elements
- `test_cuda_complex` - Complex number round-trip
- `test_cuda_2d` - Multi-dimensional tensor handling

Run CUDA tests:
```bash
pytest tests/test_core/test_state.py::TestStateDeviceOperations -v
pytest tests/test_core/test_device.py -v
```

## Dependencies

- **cudarc 0.12** - Rust CUDA bindings (safe wrapper around CUDA driver API)
- **CUDA Toolkit 12.x** - NVIDIA CUDA development tools
- **NVIDIA Driver 525+** - GPU driver with CUDA 12 support

## Known Limitations

1. **Int64 not supported on CUDA** - Would require custom kernels
2. **Operations execute on CPU** - Data is transferred back for computation
3. **Single GPU only** - Multi-GPU support would require additional coordination
4. **Synchronous transfers** - Uses `htod_sync_copy` / `dtoh_sync_copy_into`

## Future Enhancements

1. CUDA kernels for element-wise operations
2. Async data transfers
3. Multi-GPU support
4. cuBLAS integration for matrix operations
5. Memory pooling for reduced allocation overhead
