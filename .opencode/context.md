# Project Context

## Environment
- Language: Rust (backend) + Python (bindings via PyO3/maturin)
- Runtime: Python 3.10+
- Build: `maturin develop` (development), `cargo build` (Rust only)
- Test: `pytest tests/`
- Package Manager: cargo (Rust), pip/maturin (Python)

## Project Type
- [x] Library/Package (Syntonic library implementing SRT mathematics)
- [ ] Application (CLI/Web/Mobile/Desktop)
- [ ] Microservice
- [ ] Monorepo
- [ ] Other: [describe]

## Infrastructure
- Container: None
- Orchestration: None
- CI/CD: GitHub Actions (found .github/workflows/)
- Cloud: None

## Structure
- Source: rust/src/
  - lib.rs: Main module, Python bindings registration
  - tensor/ subdirectory: Tensor operations and CUDA kernels
    - srt_kernels.rs (4505 lines): PTX kernel wrapper functions
    - py_srt_cuda_ops.rs (529 lines): Python wrappers for CUDA operations
    - data_loading.rs: Native data loading (CSV/binary)
    - py_data_loading.rs: Python wrappers for data loading
    - mod.rs: Tensor module exports
- Tests: tests/
- Docs: docs/
- Entry: rust/src/lib.rs (PyO3 module registration)

## Conventions (OBSERVED from existing code)

### Rust Code Style
- Functions: `snake_case` (e.g., `cuda_backward_add_f64`)
- Types/Structs: `PascalCase` (e.g., `CudaDevice`, `PyDataBatch`)
- Constants: `UPPER_CASE` (e.g., `PTX_AUTOGRAD_SM75`, `AUTOGRAD_FUNCS`)
- Modules: `snake_case` (e.g., `tensor`, `srt_kernels`)

### PTX Kernel Wrapper Pattern
```rust
pub fn cuda_<kernel_name>(device: &Arc<CudaDevice>, ...args...) -> Result<(), String> {
    let (major, minor) = get_compute_capability(device);
    let module = device.load_module(cudarc::nvrtc::Ptx::from_src(select_<category>_ptx(major, minor)))
        .map_err(|e| format!("Failed to load <category> kernels: {}", e))?;
    let func = module.load_function("<kernel_name>")
        .map_err(|_| "Kernel <kernel_name> not found".to_string())?;
    let cfg = LaunchConfig::for_num_elems(n);
    unsafe {
        device.default_stream()
            .launch_builder(&func)
            .arg(<args...>)
            .launch(cfg)
    }
    .map(|_| ()).map_err(|e| e.to_string())
}
```

### Python Wrapper Pattern
```rust
#[pyfunction]
pub fn py_<op_name>(...) -> PyResult<Vec<f64>> {
    // CPU fallback or CUDA dispatch
}
```

### Kernel Loader Pattern
```rust
#[cfg(feature = "cuda")]
pub fn load_<category>_kernels(device: &Arc<CudaDevice>) -> Result<HashMap<String, CudaFunction>, String> {
    let (major, minor) = get_compute_capability(device);
    let module = device.load_module(Ptx::from_src(select_<category>_ptx(major, minor)))?;
    let mut functions = HashMap::new();
    for &func_name in <CATEGORY>_FUNCS {
        let func = module.load_function(func_name)?;
        functions.insert(func_name.to_string(), func);
    }
    Ok(functions)
}
```

## Current Implementation Status

### Already Implemented (Partial)
- AUTOGRAD: 1 wrapper (cuda_backward_add_f32)
- ATTRACTOR: 1 wrapper (cuda_attractor_memory_update_f64)
- REDUCTION: 1 wrapper (cuda_reduce_sum_mersenne_stable_f64)
- Function lists: AUTOGRAD_FUNCS (12), ATTRACTOR_FUNCS (5)
- Selection functions: select_autograd_ptx, select_attractor_ptx, select_reduction_ptx,
                     select_trilinear_ptx, select_complex_ops_ptx, select_attention_ptx,
                     select_wmma_syntonic_ptx
- PTX constants included for all categories

### Missing Implementation
- Function lists: SCATTER_GATHER_FUNCS, TRILINEAR_FUNCS, COMPLEX_OPS_FUNCS,
                 ATTENTION_FUNCS, WMMA_SYNTONIC_FUNCS
- Individual wrappers: ~95 kernel wrappers missing
- Loader pyfunctions: Only load_autograd_kernels and load_attractor_kernels exposed
- Python bindings: None for the new kernels

### Existing Warnings
- unused PTX constants (WMMA_SYNTONIC, SCATTER_GATHER_SRT, REDUCTION, TRILINEAR)
- unused fields in data_loading: precision (PyGoldenExactConverter), phi_inv (some struct)
- unused pool imports in py_srt_cuda_ops.rs (but pool is actually used - false positive)

## PTX Kernel Entry Points (from PTX files)

### AUTOGRAD (already defined, incomplete list)
backward_add_f64, backward_add_f32, backward_mul_f64, backward_mul_f32,
backward_matmul_f64, backward_matmul_f32, backward_softmax_f64, backward_softmax_f32,
backward_layernorm_f64, backward_layernorm_f32, backward_elementwise_f64,
backward_elementwise_f32

### Missing from AUTOGRAD_FUNCS (found in PTX)
backward_exp_f64, backward_exp_f32, backward_log_f64, backward_log_f32,
backward_sqrt_f64, backward_reciprocal_f64, backward_relu_f64, backward_relu_f32,
backward_sigmoid_f64, backward_tanh_f64, backward_gelu_f64, backward_phi_residual_f64

### ATTRACTOR (already defined, incomplete list)
attractor_memory_update_f64, hooking_coefficient_f64, retrocausal_harmonize_f64,
attractor_distance_f64, attractor_centroid_f64

### Missing from ATTRACTOR_FUNCS (found in PTX)
attractor_memory_decay_f64, hooking_coefficient_batch_f64, attractor_centroid_batch_f64,
attractor_distance_per_feature_f64, retrocausal_harmonize_full_f64,
syntony_gradient_f64, hybrid_backward_dispatch_f64

### WMMA_SYNTONIC
wmma_golden_weighted_fp16, wmma_syntonic_fp16 (template)

### SCATTER_GATHER_SRT
gather_f64, gather_f32, scatter_f64, scatter_f32, scatter_add_f64, scatter_add_f32,
gather_phi_weighted_f64, scatter_golden_f64, scatter_mersenne_stable_f64,
gather_lucas_shadow_f64, gather_pisano_hooked_f64, gather_e8_roots_f64,
scatter_golden_cone_f64, gather_transcendence_gate_f64, scatter_consciousness_threshold_f64

### REDUCTION
reduce_sum_f64, reduce_sum_f32, reduce_mean_f64, reduce_mean_f32,
reduce_max_f64, reduce_max_f32, reduce_min_f64, reduce_min_f32,
reduce_norm_l2_f64, reduce_norm_l2_f32, reduce_sum_golden_weighted_f64,
reduce_syntony_f64, reduce_sum_rows_f64, reduce_sum_cols_f64,
reduce_sum_phi_scaled_f64, reduce_variance_golden_target_f64,
reduce_sum_c128, reduce_norm_c128, reduce_sum_mersenne_stable_f64 (already wrapped),
reduce_sum_lucas_shadow_f64, reduce_syntony_deviation_f64,
reduce_consciousness_count_f64, reduce_sum_q_corrected_f64, reduce_e8_norm_f64

### TRILINEAR
trilinear_f64, trilinear_toroidal_f64, trilinear_phi_weighted_f64,
trilinear_golden_decay_f64, trilinear_causal_f64, trilinear_retrocausal_f64,
trilinear_symmetric_f64, trilinear_acausal_f64, bilinear_f64

### COMPLEX_OPS
arg_c128, arg_c64, normalize_phase_c128, normalize_phase_c64,
rotate_phase_c128, rotate_phase_c64, quantize_phase_pi_c128,
conj_c128, conj_c64, hermitian_inner_c128, phase_syntony_c128,
berry_phase_c128, probability_c128, probability_c64, normalize_wavefunction_c128,
golden_rotate_c128, phi_weighted_sum_c128

### ATTENTION
flash_attention_f32, flash_attention_syntony_f32, flash_attention_golden_f32,
flash_attention_mersenne_127_f32, flash_attention_causal_f32,
flash_attention_retrocausal_f32

## Notes
- All PTX files are pre-compiled in rust/kernels/ptx/
- Compute capability selection pattern is consistent across all categories
- Launch config uses `LaunchConfig::for_num_elems()` for 1D, custom for 2D/3D
- Error handling uses `Result<(), String>` for internal functions, `PyResult` for Python wrappers
- All wrappers must preserve the existing pattern and error handling style


## Recent Changes (2026-01-21)
- Added ~100 new kernel wrapper functions across 7 PTX categories
- Created 6 kernel loader functions that return HashMap<String, CudaFunction>
- All PTX constants now properly used (no "never used" warnings)
- Fixed unused field warnings in data_loading (precision, phi_inv)
- Kernel loaders exposed to Python via py_load_* wrappers in py_srt_cuda_ops.rs

## Recent Changes (2026-01-21)
- Added ~100 new kernel wrapper functions across 7 PTX categories
- Created 6 kernel loader functions that return HashMap<String, CudaFunction>
- All PTX constants now properly used (no "never used" warnings)
- Fixed unused field warnings in data_loading (precision, phi_inv)
- Kernel loaders exposed to Python via py_load_* wrappers in py_srt_cuda_ops.rs
