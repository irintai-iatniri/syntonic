# Rust Unused Kernel Full Implementation Plan

## Overview

Fully implement wrappers for ALL unused PTX kernel constants. No code removal. No silent fallbacks.

---

## Unused PTX Files and Their Kernels

### 1. AUTOGRAD (`PTX_AUTOGRAD_SM*`)

Entry points:
- `backward_add_f64`, `backward_add_f32`
- `backward_mul_f64`, `backward_mul_f32`
- `backward_exp_f64`, `backward_exp_f32`
- `backward_log_f64`, `backward_log_f32`
- `backward_sqrt_f64`
- `backward_reciprocal_f64`
- `backward_relu_f64`, `backward_relu_f32`
- `backward_sigmoid_f64`
- `backward_tanh_f64`
- `backward_gelu_f64`
- `backward_softmax_f64`, `backward_softmax_f32`
- `backward_layernorm_f64`
- `backward_phi_residual_f64`

### 2. ATTRACTOR (`PTX_ATTRACTOR_SM*`)

Entry points:
- `attractor_memory_update_f64`
- `attractor_memory_decay_f64`
- `hooking_coefficient_f64`, `hooking_coefficient_batch_f64`
- `attractor_centroid_f64`, `attractor_centroid_batch_f64`
- `attractor_distance_f64`, `attractor_distance_per_feature_f64`
- `retrocausal_harmonize_f64`, `retrocausal_harmonize_full_f64`
- `syntony_gradient_f64`
- `hybrid_backward_dispatch_f64`

### 3. WMMA_SYNTONIC (`PTX_WMMA_SYNTONIC_SM*`)

Entry points:
- `wmma_golden_weighted_fp16` (mangled name)
- `wmma_syntonic_fp16` (template, mangled)

Note: `cuda_wmma_fp16_matmul()` exists but is never called. Must expose to Python.

### 4. SCATTER_GATHER_SRT (`PTX_SCATTER_GATHER_SRT_SM*`)

Entry points:
- `gather_f64`, `gather_f32`
- `scatter_f64`, `scatter_f32`
- `scatter_add_f64`, `scatter_add_f32`
- `gather_phi_weighted_f64`
- `scatter_golden_f64`
- `scatter_mersenne_stable_f64`
- `gather_lucas_shadow_f64`
- `gather_pisano_hooked_f64`
- `gather_e8_roots_f64`
- `scatter_golden_cone_f64`
- `gather_transcendence_gate_f64`
- `scatter_consciousness_threshold_f64`

### 5. REDUCTION (`PTX_REDUCTION_SM*`)

Entry points:
- `reduce_sum_f64`, `reduce_sum_f32`
- `reduce_mean_f64`, `reduce_mean_f32`
- `reduce_max_f64`, `reduce_max_f32`
- `reduce_min_f64`, `reduce_min_f32`
- `reduce_norm_l2_f64`, `reduce_norm_l2_f32`
- `reduce_sum_golden_weighted_f64`
- `reduce_syntony_f64`
- `reduce_sum_rows_f64`, `reduce_sum_cols_f64`
- `reduce_sum_phi_scaled_f64`
- `reduce_variance_golden_target_f64`
- `reduce_sum_c128`, `reduce_norm_c128`
- `reduce_sum_mersenne_stable_f64`
- `reduce_sum_lucas_shadow_f64`
- `reduce_syntony_deviation_f64`
- `reduce_consciousness_count_f64`
- `reduce_sum_q_corrected_f64`
- `reduce_e8_norm_f64`

### 6. TRILINEAR (`PTX_TRILINEAR_SM*`)

Entry points:
- `trilinear_f64`
- `trilinear_toroidal_f64`
- `trilinear_phi_weighted_f64`
- `trilinear_golden_decay_f64`
- `trilinear_causal_f64`
- `trilinear_retrocausal_f64`
- `trilinear_symmetric_f64`
- `trilinear_acausal_f64`
- `bilinear_f64`

### 7. COMPLEX_OPS (`PTX_COMPLEX_OPS_SM*`)

Entry points:
- `arg_c128`, `arg_c64`
- `normalize_phase_c128`, `normalize_phase_c64`
- `rotate_phase_c128`, `rotate_phase_c64`
- `quantize_phase_pi_c128`
- `conj_c128`, `conj_c64`
- `hermitian_inner_c128`
- `phase_syntony_c128`
- `berry_phase_c128`
- `probability_c128`, `probability_c64`
- `normalize_wavefunction_c128`
- `golden_rotate_c128`
- `phi_weighted_sum_c128`

### 8. ATTENTION (`PTX_ATTENTION_SM*`)

Entry points:
- `flash_attention_f32`
- `flash_attention_syntony_f32`
- `flash_attention_golden_f32`
- `flash_attention_mersenne_127_f32`
- `flash_attention_causal_f32`
- `flash_attention_retrocausal_f32`

---

## Implementation Strategy

### File: `rust/src/tensor/srt_kernels.rs`

For each PTX category, add:

1. **Function list constant** (if not exists):
```rust
#[cfg(feature = "cuda")]
const AUTOGRAD_FUNCS: &[&str] = &[
    "backward_add_f64", "backward_add_f32",
    "backward_mul_f64", "backward_mul_f32",
    // ... all entries
];
```

2. **Kernel loader pyfunction**:
```rust
#[cfg(feature = "cuda")]
#[pyfunction]
pub fn load_autograd_kernels(device_idx: usize) -> PyResult<Vec<String>> {
    let device = get_device(device_idx)?;
    let (major, minor) = get_compute_capability(&device);
    let module = device.load_module(Ptx::from_src(select_autograd_ptx(major, minor)))?;
    Ok(AUTOGRAD_FUNCS.iter()
        .filter(|&&f| module.load_function(f).is_ok())
        .map(|&s| s.to_string())
        .collect())
}
```

3. **Individual kernel wrappers** for each entry point:
```rust
#[cfg(feature = "cuda")]
pub fn cuda_backward_add_f64(
    device: &Arc<CudaDevice>,
    grad_a: &mut CudaSlice<f64>,
    grad_b: &mut CudaSlice<f64>,
    grad_out: &CudaSlice<f64>,
    n: usize,
) -> PyResult<()> {
    let (major, minor) = get_compute_capability(device);
    let module = device.load_module(Ptx::from_src(select_autograd_ptx(major, minor)))?;
    let func = module.load_function("backward_add_f64")?;
    unsafe {
        device.default_stream()
            .launch_builder(&func)
            .arg(grad_a).arg(grad_b).arg(grad_out).arg(&(n as i32))
            .launch(launch_cfg_256(n))
    }?;
    Ok(())
}
```

---

## Detailed Implementation for Each Category

### AUTOGRAD Wrappers (~20 functions)

Each backward kernel follows pattern: `(grad_a, grad_b?, input?, grad_out, n)`

```rust
// backward_add: gradient flows to both inputs
cuda_backward_add_f64(grad_a, grad_b, grad_out, n)
cuda_backward_add_f32(grad_a, grad_b, grad_out, n)

// backward_mul: requires original inputs for gradient
cuda_backward_mul_f64(grad_a, grad_b, a, b, grad_out, n)
cuda_backward_mul_f32(grad_a, grad_b, a, b, grad_out, n)

// backward_exp: exp'(x) = exp(x), needs output
cuda_backward_exp_f64(grad_in, output, grad_out, n)
cuda_backward_exp_f32(grad_in, output, grad_out, n)

// backward_log: log'(x) = 1/x
cuda_backward_log_f64(grad_in, input, grad_out, n)
cuda_backward_log_f32(grad_in, input, grad_out, n)

// backward_sqrt: sqrt'(x) = 1/(2*sqrt(x))
cuda_backward_sqrt_f64(grad_in, output, grad_out, n)

// backward_reciprocal: (1/x)' = -1/x^2
cuda_backward_reciprocal_f64(grad_in, output, grad_out, n)

// backward_relu: relu'(x) = x > 0 ? 1 : 0
cuda_backward_relu_f64(grad_in, input, grad_out, n)
cuda_backward_relu_f32(grad_in, input, grad_out, n)

// backward_sigmoid: sig'(x) = sig(x)*(1-sig(x))
cuda_backward_sigmoid_f64(grad_in, output, grad_out, n)

// backward_tanh: tanh'(x) = 1 - tanh^2(x)
cuda_backward_tanh_f64(grad_in, output, grad_out, n)

// backward_gelu: gelu'(x) = complex formula
cuda_backward_gelu_f64(grad_in, input, grad_out, n)

// backward_softmax: Jacobian computation
cuda_backward_softmax_f64(grad_in, output, grad_out, batch, features)
cuda_backward_softmax_f32(grad_in, output, grad_out, batch, features)

// backward_layernorm: requires mean, variance
cuda_backward_layernorm_f64(grad_in, input, mean, var, gamma, grad_out, n, features)

// backward_phi_residual: phi-scaled residual gradient
cuda_backward_phi_residual_f64(grad_in, grad_out, n)
```

### ATTRACTOR Wrappers (~12 functions)

```rust
cuda_attractor_memory_update_f64(state, gradients, attractor_strength, n)
cuda_attractor_memory_decay_f64(state, decay_rate, n)
cuda_hooking_coefficient_f64(coefficients, state, target, n)
cuda_hooking_coefficient_batch_f64(coefficients, states, targets, batch, n)
cuda_attractor_centroid_f64(centroid, states, n)
cuda_attractor_centroid_batch_f64(centroids, states, batch, n)
cuda_attractor_distance_f64(distances, state, centroids, n_centroids, dim)
cuda_attractor_distance_per_feature_f64(distances, state, centroid, n)
cuda_retrocausal_harmonize_f64(output, current, history, weight, n)
cuda_retrocausal_harmonize_full_f64(output, current, history, weights, history_len, n)
cuda_syntony_gradient_f64(gradient, state, target_syntony, n)
cuda_hybrid_backward_dispatch_f64(grad, forward_grad, backward_grad, ratio, n)
```

### SCATTER_GATHER Wrappers (~15 functions)

```rust
cuda_gather_f64(output, input, indices, n)
cuda_gather_f32(output, input, indices, n)
cuda_scatter_f64(output, input, indices, n)
cuda_scatter_f32(output, input, indices, n)
cuda_scatter_add_f64(output, input, indices, n)
cuda_scatter_add_f32(output, input, indices, n)
cuda_gather_phi_weighted_f64(output, input, indices, weights, n)
cuda_scatter_golden_f64(output, input, indices, n)
cuda_scatter_mersenne_stable_f64(output, input, indices, n)
cuda_gather_lucas_shadow_f64(output, input, indices, n)
cuda_gather_pisano_hooked_f64(output, input, indices, n)
cuda_gather_e8_roots_f64(output, input, indices, n)
cuda_scatter_golden_cone_f64(output, input, indices, n)
cuda_gather_transcendence_gate_f64(output, input, indices, n)
cuda_scatter_consciousness_threshold_f64(output, input, indices, threshold, n)
```

### REDUCTION Wrappers (~24 functions)

```rust
cuda_reduce_sum_f64(output, input, n) -> f64
cuda_reduce_sum_f32(output, input, n) -> f32
cuda_reduce_mean_f64(output, input, n) -> f64
cuda_reduce_mean_f32(output, input, n) -> f32
cuda_reduce_max_f64(output, input, n) -> f64
cuda_reduce_max_f32(output, input, n) -> f32
cuda_reduce_min_f64(output, input, n) -> f64
cuda_reduce_min_f32(output, input, n) -> f32
cuda_reduce_norm_l2_f64(output, input, n) -> f64
cuda_reduce_norm_l2_f32(output, input, n) -> f32
cuda_reduce_sum_golden_weighted_f64(output, input, n) -> f64
cuda_reduce_syntony_f64(output, input, n) -> f64
cuda_reduce_sum_rows_f64(output, input, rows, cols)
cuda_reduce_sum_cols_f64(output, input, rows, cols)
cuda_reduce_sum_phi_scaled_f64(output, input, n) -> f64
cuda_reduce_variance_golden_target_f64(output, input, target, n) -> f64
cuda_reduce_sum_c128(output, input, n) -> Complex128
cuda_reduce_norm_c128(output, input, n) -> f64
cuda_reduce_sum_mersenne_stable_f64(output, input, n) -> f64
cuda_reduce_sum_lucas_shadow_f64(output, input, n) -> f64
cuda_reduce_syntony_deviation_f64(output, input, target_syntony, n) -> f64
cuda_reduce_consciousness_count_f64(output, input, threshold, n) -> i64
cuda_reduce_sum_q_corrected_f64(output, input, structure_idx, n) -> f64
cuda_reduce_e8_norm_f64(output, input, n) -> f64
```

### TRILINEAR Wrappers (~9 functions)

```rust
cuda_trilinear_f64(output, input, x, y, z, nx, ny, nz)
cuda_trilinear_toroidal_f64(output, input, x, y, z, nx, ny, nz)
cuda_trilinear_phi_weighted_f64(output, input, x, y, z, nx, ny, nz)
cuda_trilinear_golden_decay_f64(output, input, x, y, z, nx, ny, nz, decay)
cuda_trilinear_causal_f64(output, input, x, y, z, t, nx, ny, nz, nt)
cuda_trilinear_retrocausal_f64(output, input, x, y, z, t, nx, ny, nz, nt)
cuda_trilinear_symmetric_f64(output, input, x, y, z, nx, ny, nz)
cuda_trilinear_acausal_f64(output, input, x, y, z, t, nx, ny, nz, nt)
cuda_bilinear_f64(output, input, x, y, nx, ny)
```

### COMPLEX_OPS Wrappers (~17 functions)

```rust
cuda_arg_c128(output, input, n)
cuda_arg_c64(output, input, n)
cuda_normalize_phase_c128(output, input, n)
cuda_normalize_phase_c64(output, input, n)
cuda_rotate_phase_c128(output, input, angle, n)
cuda_rotate_phase_c64(output, input, angle, n)
cuda_quantize_phase_pi_c128(output, input, n)
cuda_conj_c128(output, input, n)
cuda_conj_c64(output, input, n)
cuda_hermitian_inner_c128(output, a, b, n) -> Complex128
cuda_phase_syntony_c128(output, input, n) -> f64
cuda_berry_phase_c128(output, path, n) -> Complex128
cuda_probability_c128(output, input, n)
cuda_probability_c64(output, input, n)
cuda_normalize_wavefunction_c128(output, input, n)
cuda_golden_rotate_c128(output, input, n)
cuda_phi_weighted_sum_c128(output, input, n) -> Complex128
```

### ATTENTION Wrappers (~6 functions)

```rust
cuda_flash_attention_f32(output, q, k, v, batch, heads, seq_len, dim)
cuda_flash_attention_syntony_f32(output, q, k, v, batch, heads, seq_len, dim, syntony_threshold)
cuda_flash_attention_golden_f32(output, q, k, v, batch, heads, seq_len, dim)
cuda_flash_attention_mersenne_127_f32(output, q, k, v, batch, heads, seq_len, dim)
cuda_flash_attention_causal_f32(output, q, k, v, batch, heads, seq_len, dim)
cuda_flash_attention_retrocausal_f32(output, q, k, v, batch, heads, seq_len, dim)
```

---

## Files to Modify

| File | Changes |
|------|---------|
| `rust/src/tensor/srt_kernels.rs` | Add ~100 wrapper functions |
| `rust/src/lib.rs` | Register all new functions |
| `rust/src/tensor/data_loading.rs` | Use all variants/fields |
| `rust/src/tensor/py_data_loading.rs` | Use precision field |
| `rust/src/tensor/py_srt_cuda_ops.rs` | Fix unused pool warning |

---

## Verification

```bash
cargo build 2>&1 | grep -c "^warning"
# Target: 0 warnings

# Test kernel availability
python -c "
from syntonic._core import (
    load_autograd_kernels,
    load_attractor_kernels,
    load_scatter_gather_kernels,
    load_reduction_kernels,
    load_trilinear_kernels,
    load_complex_ops_kernels,
    load_attention_kernels,
)
print('Autograd:', load_autograd_kernels(0))
print('Attractor:', load_attractor_kernels(0))
print('ScatterGather:', load_scatter_gather_kernels(0))
print('Reduction:', load_reduction_kernels(0))
print('Trilinear:', load_trilinear_kernels(0))
print('ComplexOps:', load_complex_ops_kernels(0))
print('Attention:', load_attention_kernels(0))
"
```

---

## Estimated LOC

- ~100 wrapper functions × ~20 lines avg = ~2000 lines in srt_kernels.rs
- ~100 function registrations in lib.rs
- Total: ~2100 lines of new code
