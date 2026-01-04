# Syntonic CUDA Optimization Status

## Completed Phases

### Phase 1: Complex Conversion to Rust (COMPLETE)
- **Result:** 35-60x speedup for complex128 CUDA transfers
- **Change:** Complex numbers now passed directly from Python to Rust via PyO3, eliminating Python-side list conversion
- **File:** `rust/src/tensor/storage.rs` - `from_list()` method handles complex128 natively

### Phase 2: CUDA Element-wise Kernels (COMPLETE)
- **Result:** GPU-native add, sub, mul, div, neg, abs operations
- **Supports:** float32, float64, complex128

#### Key Files Created/Modified:

1. **`rust/kernels/elementwise.cu`** - CUDA kernel source
2. **`rust/kernels/compile_kernels.sh`** - Offline compilation script
3. **`rust/kernels/ptx/elementwise_sm75.ptx`** - PTX for compute_75+ (Turing)
4. **`rust/kernels/ptx/elementwise_sm80.ptx`** - PTX for compute_80+ (Ampere)
5. **`rust/kernels/ptx/elementwise_sm86.ptx`** - PTX for compute_86+ (RTX 30xx)
6. **`rust/kernels/ptx/elementwise_sm90.ptx`** - PTX for compute_90+ (Hopper)
7. **`rust/src/tensor/storage.rs`** - Main implementation

#### Critical Implementation Details:

**PTX Version Fix:** PTX files were compiled with CUDA 13.1 (PTX ISA 9.1) but driver only supports 8.3. Fixed by patching `.version 9.1` to `.version 8.3` in all PTX files. This works because kernels only use basic instructions (add, mul, ld, st, etc.) available since PTX 1.0.

**Kernel Loading:** Kernels are loaded per `Arc<CudaDevice>` instance, NOT per device index. The fix checks if kernels exist via `device.get_func()` before loading:

```rust
fn ensure_kernels_loaded(device: &Arc<CudaDevice>, _device_idx: usize) -> PyResult<()> {
    if device.get_func("syntonic", "add_f64").is_some() {
        return Ok(());
    }
    // Load PTX...
}
```

**PTX Selection:** Based on compute capability:
```rust
fn select_ptx(major: i32, minor: i32) -> &'static str {
    let cc = major * 10 + minor;
    if cc >= 90 { PTX_SM90 }
    else if cc >= 86 { PTX_SM86 }
    else if cc >= 80 { PTX_SM80 }
    else { PTX_SM75 }
}
```

#### Kernel Functions Available:
- `add_f64`, `add_f32`, `sub_f64`, `sub_f32`
- `mul_f64`, `mul_f32`, `div_f64`, `div_f32`
- `neg_f64`, `neg_f32`, `abs_f64`, `abs_f32`
- `scalar_add_f64`, `scalar_mul_f64`
- `add_c128`, `sub_c128`, `mul_c128`, `neg_c128`

---

## Remaining Phases

### Phase 3: cuBLAS Integration
**Goal:** GPU-accelerated matrix operations via cuBLAS GEMM.
**Note:** User wants specialized version tailored for SRT theory (golden ratio recursion, E8 lattice ops, etc.)

**cudarc API:**
```rust
use cudarc::cublas::{CudaBlas, Gemm};

let blas = CudaBlas::new(device.clone())?;
unsafe {
    blas.gemm(
        cublas::sys::cublasOperation_t::CUBLAS_OP_N,
        cublas::sys::cublasOperation_t::CUBLAS_OP_N,
        m, n, k,
        &alpha,
        a_slice, lda,
        b_slice, ldb,
        &beta,
        c_slice, ldc,
    )?;
}
```

**Cargo.toml change needed:**
```toml
cudarc = { version = "0.12", optional = true, features = ["cuda-12060", "nvrtc", "cublas"] }
```

### Phase 4: Memory Pooling
**Goal:** Reduce cudaMalloc overhead by reusing allocations.

**Implementation approach:**
```rust
pub struct CudaMemoryPool {
    device: Arc<CudaDevice>,
    f64_pools: Mutex<HashMap<usize, Vec<CudaSlice<f64>>>>,
    // Round allocations to power-of-2 buckets
}
```

**Where to add:** Create `rust/src/tensor/cuda_pool.rs` or add to `storage.rs`

### Phase 5: Async Data Transfers
**Goal:** Overlap computation with data transfer using CUDA streams.

**cudarc API:**
```rust
// Fork a stream from device
let stream = device.fork_default_stream()?;
// Use stream for async operations
```

### Phase 6: Multi-GPU Support
**Goal:** Support operations across multiple GPUs.

**cudarc API:**
```rust
let device_count = CudaDevice::count()?;
let device = CudaDevice::new(device_idx)?;
```

---

## Key APIs (cudarc 0.12)

### Device Management
```rust
use cudarc::driver::{CudaDevice, CudaSlice, LaunchConfig, LaunchAsync};

let device = CudaDevice::new(0)?;  // Get device 0
let ordinal = device.ordinal();     // Get device index
```

### Memory Operations
```rust
// Allocate on GPU
let slice: CudaSlice<f64> = device.alloc_zeros(n)?;

// Host to device
let gpu_data = device.htod_copy(vec![1.0, 2.0, 3.0])?;

// Device to host
let cpu_data = device.dtoh_sync_copy(&gpu_slice)?;
```

### PTX Loading
```rust
use cudarc::nvrtc::Ptx;

device.load_ptx(
    Ptx::from_src(ptx_string),  // Pre-compiled PTX source
    "module_name",
    &["func1", "func2"],        // Function names to register
)?;
```

### Kernel Launch
```rust
let func = device.get_func("module_name", "kernel_name").unwrap();
let cfg = LaunchConfig {
    block_dim: (256, 1, 1),
    grid_dim: ((n as u32 + 255) / 256, 1, 1),
    shared_mem_bytes: 0,
};
unsafe { func.launch(cfg, (&mut out, &input, n as i32))? };
```

### Compute Capability
```rust
use cudarc::driver::sys::CUdevice_attribute_enum;
use cudarc::driver::result;

let major = unsafe {
    result::device::get_attribute(
        ordinal,
        CUdevice_attribute_enum::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR
    ).unwrap_or(7)
};
```

---

## File Locations

| File | Purpose |
|------|---------|
| `rust/src/tensor/storage.rs` | Main tensor implementation, CUDA ops |
| `rust/kernels/elementwise.cu` | CUDA kernel source |
| `rust/kernels/ptx/*.ptx` | Pre-compiled PTX (4 versions) |
| `rust/kernels/compile_kernels.sh` | PTX compilation script |
| `rust/Cargo.toml` | Dependencies (cudarc with cuda-12060, nvrtc) |
| `python/syntonic/core/state.py` | Python State class wrapping TensorStorage |

---

## Build Commands

```bash
# Build with CUDA
cd /home/Andrew/Documents/SRT\ Complete/implementation/syntonic
PATH="/home/Andrew/.cargo/bin:/usr/local/cuda-13.1/bin:/usr/bin:/bin:$PATH" \
CUDA_PATH="/usr/local/cuda-13.1" \
/home/Andrew/miniforge3/bin/python -m maturin develop --features cuda

# Recompile PTX (if kernels change)
cd rust/kernels && ./compile_kernels.sh
# Then patch version: sed -i 's/^\.version 9\.1$/.version 8.3/' ptx/*.ptx
```

---

## Test Commands

```python
from python.syntonic import State

a = State([1.0, 2.0, 3.0], dtype='float64')
b = State([4.0, 5.0, 6.0], dtype='float64')

a_cuda = a.cuda()
b_cuda = b.cuda()

c = (a_cuda + b_cuda).cpu()  # [5.0, 7.0, 9.0]
```

---

## Known Issues / Gotchas

1. **PTX Version:** Must be 8.3 or lower for CUDA 13.0 driver compatibility. Patch after compiling.

2. **Kernel Loading:** Each `Arc<CudaDevice>` needs its own kernel load. Check with `get_func()` not a HashMap cache.

3. **Device Matching:** Binary ops compare device INDEX not Arc pointer:
   ```rust
   if let (DeviceType::Cuda(idx_a), DeviceType::Cuda(idx_b)) = (&self.device, &other.device) {
       if idx_a == idx_b { /* use CUDA */ }
   }
   ```

4. **Complex128:** Stored as interleaved f64 pairs on GPU. Kernel receives element count (not byte count).

---

## Current Cargo.toml CUDA Config

```toml
[dependencies]
cudarc = { version = "0.12", optional = true, features = ["cuda-12060", "nvrtc"] }
lazy_static = "1.4"

[features]
cuda = ["cudarc"]
```
