# Rust Backend & CUDA Kernels Implementation Plan
## Theory-Aligned Components for Syntonic Neural Networks

**Date:** 2026-01-08
**Goal:** Implement Rust/CUDA backend for PhiResidual, GoldenBatchNorm, and SyntonicSoftmax
**Status:** Design Phase → Implementation Ready
**Estimated Time:** 12-16 hours

---

## Table of Contents

1. [Architecture Overview](#architecture-overview)
2. [Phase 1: Phi-Scaled Residual Operations](#phase-1-phi-scaled-residual-operations)
3. [Phase 2: Golden Batch Normalization](#phase-2-golden-batch-normalization)
4. [Phase 3: Syntonic Softmax](#phase-3-syntonic-softmax)
5. [Phase 4: Integration & Testing](#phase-4-integration--testing)
6. [Implementation Checklist](#implementation-checklist)

---

## Architecture Overview

### Design Principles

1. **Dual-State Paradigm**: Operations work with both exact Q(φ) lattice (CPU) and approximate flux (GPU)
2. **Theory Alignment**: All operations respect golden ratio mathematics
3. **CUDA Acceleration**: Performance-critical paths run on GPU
4. **PyO3 Bindings**: Expose Rust functions to Python via `syntonic._core`

### File Structure

```
rust/
├── src/
│   ├── resonant/
│   │   ├── phi_ops.rs          # NEW: Phi-scaled operations
│   │   ├── golden_norm.rs      # NEW: Golden batch normalization
│   │   ├── syntonic_softmax.rs # NEW: Syntony-weighted softmax
│   │   └── mod.rs              # Export new modules
│   └── lib.rs                  # Add PyO3 bindings
├── kernels/
│   ├── phi_residual.cu         # NEW: CUDA for phi-scaled residuals
│   ├── golden_batch_norm.cu    # NEW: CUDA for golden batch norm
│   ├── syntonic_softmax.cu     # NEW: CUDA for syntonic softmax
│   └── srt_constants.cuh       # Already exists
└── Cargo.toml                  # No changes needed
```

---

## Phase 1: Phi-Scaled Residual Operations

### 1.1 Rust Implementation

**File:** `rust/src/resonant/phi_ops.rs`

```rust
//! Phi-scaled residual operations for theory-aligned neural networks.
//!
//! Implements three residual modes:
//! - phi: output = identity + residual/φ (default, recommended)
//! - phi_symmetric: output = (identity + residual)/φ
//! - standard: output = identity + residual (for ablation)

use pyo3::prelude::*;
use crate::exact::GoldenExact;
use crate::resonant::{ResonantTensor, ResonantPhase};
use super::{PHI, PHI_INV};

#[cfg(feature = "cuda")]
use cudarc::driver::{CudaDevice, CudaSlice, LaunchAsync, LaunchConfig};

/// Phi-residual modes
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[pyclass]
pub enum PhiResidualMode {
    /// output = identity + residual/φ (theory-aligned, default)
    Phi,
    /// output = (identity + residual)/φ (symmetric scaling)
    PhiSymmetric,
    /// output = identity + residual (standard ResNet)
    Standard,
}

#[pymethods]
impl PhiResidualMode {
    #[new]
    fn new(mode_str: &str) -> PyResult<Self> {
        match mode_str {
            "phi" => Ok(PhiResidualMode::Phi),
            "phi_symmetric" => Ok(PhiResidualMode::PhiSymmetric),
            "standard" => Ok(PhiResidualMode::Standard),
            _ => Err(pyo3::exceptions::PyValueError::new_err(
                format!("Unknown mode: {}. Use 'phi', 'phi_symmetric', or 'standard'", mode_str)
            )),
        }
    }

    fn __repr__(&self) -> String {
        match self {
            PhiResidualMode::Phi => "PhiResidualMode::Phi".to_string(),
            PhiResidualMode::PhiSymmetric => "PhiResidualMode::PhiSymmetric".to_string(),
            PhiResidualMode::Standard => "PhiResidualMode::Standard".to_string(),
        }
    }
}

impl ResonantTensor {
    /// Apply phi-residual connection: combines identity with residual using golden ratio
    ///
    /// # Arguments
    /// * `identity` - The skip connection (input)
    /// * `residual` - The transformed path (e.g., output of conv/linear layers)
    /// * `mode` - Residual mode (phi, phi_symmetric, standard)
    ///
    /// # Returns
    /// New ResonantTensor with combined output
    ///
    /// # Modes
    /// - `Phi`: `output = identity + residual/φ` (amplifies identity, dampens residual)
    /// - `PhiSymmetric`: `output = (identity + residual)/φ` (scales both equally)
    /// - `Standard`: `output = identity + residual` (traditional residual)
    pub fn phi_residual(
        identity: &ResonantTensor,
        residual: &ResonantTensor,
        mode: PhiResidualMode,
    ) -> PyResult<ResonantTensor> {
        // Validate shapes match
        if identity.shape != residual.shape {
            return Err(pyo3::exceptions::PyValueError::new_err(
                format!("Shape mismatch: identity {:?} vs residual {:?}",
                    identity.shape, residual.shape)
            ));
        }

        match mode {
            PhiResidualMode::Phi => {
                // output = identity + residual/φ
                let scaled_residual = residual.scalar_mul_exact(
                    &GoldenExact::phi_inverse()  // 1/φ = φ - 1
                )?;
                identity.elementwise_add(&scaled_residual)
            },
            PhiResidualMode::PhiSymmetric => {
                // output = (identity + residual)/φ
                let sum = identity.elementwise_add(residual)?;
                sum.scalar_mul_exact(&GoldenExact::phi_inverse())
            },
            PhiResidualMode::Standard => {
                // output = identity + residual
                identity.elementwise_add(residual)
            },
        }
    }
}

// Python-accessible wrapper
#[pyfunction]
pub fn phi_residual(
    identity: &ResonantTensor,
    residual: &ResonantTensor,
    mode: PhiResidualMode,
) -> PyResult<ResonantTensor> {
    ResonantTensor::phi_residual(identity, residual, mode)
}
```

### 1.2 CUDA Kernel

**File:** `rust/kernels/phi_residual.cu`

```cuda
// Syntonic CUDA Kernels - Phi-Scaled Residual Connections
// Implements golden-ratio residual connections for theory-aligned networks

#include "srt_constants.cuh"

// =============================================================================
// Phi-Residual Mode: output = identity + residual/φ
// =============================================================================

extern "C" __global__ void phi_residual_mode_phi_f64(
    double *out,
    const double *identity,
    const double *residual,
    int n
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        // output = identity + residual * (1/φ)
        out[i] = identity[i] + residual[i] * PHI_INV_F64;
    }
}

extern "C" __global__ void phi_residual_mode_phi_f32(
    float *out,
    const float *identity,
    const float *residual,
    int n
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        out[i] = identity[i] + residual[i] * PHI_INV_F32;
    }
}

// =============================================================================
// Phi-Symmetric Mode: output = (identity + residual)/φ
// =============================================================================

extern "C" __global__ void phi_residual_mode_symmetric_f64(
    double *out,
    const double *identity,
    const double *residual,
    int n
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        // output = (identity + residual) / φ
        out[i] = (identity[i] + residual[i]) * PHI_INV_F64;
    }
}

extern "C" __global__ void phi_residual_mode_symmetric_f32(
    float *out,
    const float *identity,
    const float *residual,
    int n
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        out[i] = (identity[i] + residual[i]) * PHI_INV_F32;
    }
}

// =============================================================================
// Standard Mode: output = identity + residual (for ablation)
// =============================================================================

extern "C" __global__ void phi_residual_mode_standard_f64(
    double *out,
    const double *identity,
    const double *residual,
    int n
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        out[i] = identity[i] + residual[i];
    }
}

extern "C" __global__ void phi_residual_mode_standard_f32(
    float *out,
    const float *identity,
    const float *residual,
    int n
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        out[i] = identity[i] + residual[i];
    }
}

// =============================================================================
// Fused: Phi-Residual + ReLU (common pattern: skip + ReLU)
// =============================================================================

extern "C" __global__ void phi_residual_relu_f64(
    double *out,
    const double *identity,
    const double *residual,
    int n
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        double val = identity[i] + residual[i] * PHI_INV_F64;
        out[i] = (val > 0.0) ? val : 0.0;
    }
}

extern "C" __global__ void phi_residual_relu_f32(
    float *out,
    const float *identity,
    const float *residual,
    int n
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float val = identity[i] + residual[i] * PHI_INV_F32;
        out[i] = (val > 0.0f) ? val : 0.0f;
    }
}
```

### 1.3 Rust Kernel Wrapper

Add to `rust/src/tensor/srt_kernels.rs`:

```rust
// Phi-Residual CUDA kernels
#[cfg(feature = "cuda")]
const PTX_PHI_RESIDUAL_SM75: &str = include_str!("../../kernels/ptx/phi_residual_sm75.ptx");
#[cfg(feature = "cuda")]
const PTX_PHI_RESIDUAL_SM80: &str = include_str!("../../kernels/ptx/phi_residual_sm80.ptx");
#[cfg(feature = "cuda")]
const PTX_PHI_RESIDUAL_SM86: &str = include_str!("../../kernels/ptx/phi_residual_sm86.ptx");
#[cfg(feature = "cuda")]
const PTX_PHI_RESIDUAL_SM90: &str = include_str!("../../kernels/ptx/phi_residual_sm90.ptx");

#[cfg(feature = "cuda")]
const PHI_RESIDUAL_FUNCS: &[&str] = &[
    "phi_residual_mode_phi_f64",
    "phi_residual_mode_phi_f32",
    "phi_residual_mode_symmetric_f64",
    "phi_residual_mode_symmetric_f32",
    "phi_residual_mode_standard_f64",
    "phi_residual_mode_standard_f32",
    "phi_residual_relu_f64",
    "phi_residual_relu_f32",
];

/// Load phi-residual kernels into CUDA device
#[cfg(feature = "cuda")]
pub fn ensure_phi_residual_kernels_loaded(device: &Arc<CudaDevice>) -> Result<(), CudaError> {
    let compute_cap = device.compute_capability()?;
    let ptx = select_ptx(compute_cap, PTX_PHI_RESIDUAL_SM75, PTX_PHI_RESIDUAL_SM80,
                         PTX_PHI_RESIDUAL_SM86, PTX_PHI_RESIDUAL_SM90);
    device.load_ptx(ptx.into(), "phi_residual", PHI_RESIDUAL_FUNCS)?;
    Ok(())
}

/// Execute phi-residual kernel on GPU
#[cfg(feature = "cuda")]
pub fn cuda_phi_residual_f64(
    device: &Arc<CudaDevice>,
    out: &mut CudaSlice<f64>,
    identity: &CudaSlice<f64>,
    residual: &CudaSlice<f64>,
    mode: PhiResidualMode,
) -> Result<(), CudaError> {
    ensure_phi_residual_kernels_loaded(device)?;

    let n = out.len();
    let kernel_name = match mode {
        PhiResidualMode::Phi => "phi_residual_mode_phi_f64",
        PhiResidualMode::PhiSymmetric => "phi_residual_mode_symmetric_f64",
        PhiResidualMode::Standard => "phi_residual_mode_standard_f64",
    };

    let cfg = LaunchConfig::for_num_elems(n as u32);
    let func = device.get_func("phi_residual", kernel_name)?;

    unsafe {
        func.launch(cfg, (out, identity, residual, n as i32))?;
    }

    Ok(())
}
```

---

## Phase 2: Golden Batch Normalization

### 2.1 Rust Implementation

**File:** `rust/src/resonant/golden_norm.rs`

```rust
//! Golden batch normalization with target variance = 1/φ ≈ 0.618
//!
//! Standard batch norm targets variance = 1. SRT theory suggests
//! golden-ratio variance for optimal information flow.

use pyo3::prelude::*;
use crate::resonant::{ResonantTensor, ResonantPhase, ResonantError};
use crate::exact::GoldenExact;
use super::{PHI, PHI_INV};

#[cfg(feature = "cuda")]
use cudarc::driver::{CudaDevice, CudaSlice, LaunchAsync, LaunchConfig};

/// Configuration for golden batch normalization
#[pyclass]
#[derive(Clone, Debug)]
pub struct GoldenBatchNormConfig {
    /// Number of features (channels)
    #[pyo3(get, set)]
    pub num_features: usize,

    /// Small constant for numerical stability
    #[pyo3(get, set)]
    pub eps: f64,

    /// Momentum for running statistics
    #[pyo3(get, set)]
    pub momentum: f64,

    /// Whether to use affine transformation (learnable γ, β)
    #[pyo3(get, set)]
    pub affine: bool,

    /// Whether to track running statistics
    #[pyo3(get, set)]
    pub track_running_stats: bool,
}

#[pymethods]
impl GoldenBatchNormConfig {
    #[new]
    fn new(
        num_features: usize,
        eps: Option<f64>,
        momentum: Option<f64>,
        affine: Option<bool>,
        track_running_stats: Option<bool>,
    ) -> Self {
        Self {
            num_features,
            eps: eps.unwrap_or(1e-5),
            momentum: momentum.unwrap_or(0.1),
            affine: affine.unwrap_or(true),
            track_running_stats: track_running_stats.unwrap_or(true),
        }
    }
}

/// Golden batch normalization state (running statistics)
#[pyclass]
#[derive(Clone, Debug)]
pub struct GoldenBatchNormState {
    /// Running mean [num_features]
    running_mean: ResonantTensor,

    /// Running variance [num_features]
    running_var: ResonantTensor,

    /// Number of batches tracked
    num_batches_tracked: i64,

    /// Affine scale parameter γ [num_features]
    gamma: Option<ResonantTensor>,

    /// Affine bias parameter β [num_features]
    beta: Option<ResonantTensor>,

    /// Configuration
    config: GoldenBatchNormConfig,
}

#[pymethods]
impl GoldenBatchNormState {
    #[new]
    fn new(config: GoldenBatchNormConfig) -> PyResult<Self> {
        let num_features = config.num_features;

        // Initialize running mean to 0
        let running_mean = ResonantTensor::zeros(vec![num_features], 100)?;

        // Initialize running variance to 1/φ (golden target)
        let mut running_var_data = vec![GoldenExact::phi_inverse(); num_features];
        let running_var = ResonantTensor::from_lattice(
            running_var_data,
            vec![num_features],
            vec![0.0; num_features],  // Mode norms (not used for stats)
            100,
            0,  // CPU device
        )?;

        // Initialize affine parameters if needed
        let (gamma, beta) = if config.affine {
            let gamma_ones = ResonantTensor::ones(vec![num_features], 100)?;
            let beta_zeros = ResonantTensor::zeros(vec![num_features], 100)?;
            (Some(gamma_ones), Some(beta_zeros))
        } else {
            (None, None)
        };

        Ok(Self {
            running_mean,
            running_var,
            num_batches_tracked: 0,
            gamma,
            beta,
            config,
        })
    }

    /// Forward pass: normalize input to golden target variance
    ///
    /// # Arguments
    /// * `x` - Input tensor [batch_size, num_features, ...spatial dims...]
    /// * `training` - Whether in training mode (updates running stats)
    ///
    /// # Returns
    /// Normalized tensor with variance ≈ 1/φ
    fn forward(&mut self, x: &ResonantTensor, training: bool) -> PyResult<ResonantTensor> {
        // Shape validation
        if x.shape.len() < 2 {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "Input must have at least 2 dimensions [batch, features, ...]"
            ));
        }

        if x.shape[1] != self.config.num_features {
            return Err(pyo3::exceptions::PyValueError::new_err(
                format!("Feature dimension mismatch: expected {}, got {}",
                    self.config.num_features, x.shape[1])
            ));
        }

        // Dispatch to CUDA or CPU implementation
        #[cfg(feature = "cuda")]
        {
            if x.device_idx > 0 {
                return self.forward_cuda(x, training);
            }
        }

        self.forward_cpu(x, training)
    }

    /// Get current gamma (affine scale)
    fn gamma(&self) -> PyResult<Option<ResonantTensor>> {
        Ok(self.gamma.clone())
    }

    /// Get current beta (affine bias)
    fn beta(&self) -> PyResult<Option<ResonantTensor>> {
        Ok(self.beta.clone())
    }

    /// Get running mean
    fn running_mean(&self) -> PyResult<ResonantTensor> {
        Ok(self.running_mean.clone())
    }

    /// Get running variance
    fn running_var(&self) -> PyResult<ResonantTensor> {
        Ok(self.running_var.clone())
    }
}

impl GoldenBatchNormState {
    /// CPU implementation of golden batch norm
    fn forward_cpu(&mut self, x: &ResonantTensor, training: bool) -> PyResult<ResonantTensor> {
        let batch_size = x.shape[0];
        let num_features = x.shape[1];
        let spatial_size: usize = x.shape[2..].iter().product();
        let norm_size = (batch_size * spatial_size) as f64;

        // Compute batch statistics or use running stats
        let (mean, var) = if training || !self.config.track_running_stats {
            // Compute mean and variance across batch and spatial dimensions
            let mut mean_vals = vec![0.0; num_features];
            let mut var_vals = vec![0.0; num_features];

            // Convert to floats for statistics computation
            let x_floats = x.to_floats()?;

            for c in 0..num_features {
                let mut sum = 0.0;
                let mut sum_sq = 0.0;

                for b in 0..batch_size {
                    for s in 0..spatial_size {
                        let idx = b * num_features * spatial_size + c * spatial_size + s;
                        let val = x_floats[idx];
                        sum += val;
                        sum_sq += val * val;
                    }
                }

                mean_vals[c] = sum / norm_size;
                var_vals[c] = (sum_sq / norm_size) - (mean_vals[c] * mean_vals[c]);
            }

            // Update running statistics if tracking
            if training && self.config.track_running_stats {
                let momentum = self.config.momentum;
                let running_mean_floats = self.running_mean.to_floats()?;
                let running_var_floats = self.running_var.to_floats()?;

                for c in 0..num_features {
                    running_mean_floats[c] = (1.0 - momentum) * running_mean_floats[c]
                                            + momentum * mean_vals[c];
                    running_var_floats[c] = (1.0 - momentum) * running_var_floats[c]
                                           + momentum * var_vals[c];
                }

                // Snap back to lattice
                self.running_mean = ResonantTensor::from_floats(
                    running_mean_floats,
                    vec![num_features],
                    vec![0.0; num_features],
                    100,
                    0,
                )?;
                self.running_var = ResonantTensor::from_floats(
                    running_var_floats,
                    vec![num_features],
                    vec![0.0; num_features],
                    100,
                    0,
                )?;

                self.num_batches_tracked += 1;
            }

            (mean_vals, var_vals)
        } else {
            // Use running statistics
            (self.running_mean.to_floats()?, self.running_var.to_floats()?)
        };

        // Normalize: (x - mean) / sqrt(var + eps)
        let x_floats = x.to_floats()?;
        let mut out_floats = vec![0.0; x_floats.len()];
        let golden_scale = (PHI_INV).sqrt();  // sqrt(1/φ) for golden target

        for b in 0..batch_size {
            for c in 0..num_features {
                let rstd = 1.0 / (var[c] + self.config.eps).sqrt();
                let scale = rstd * golden_scale;  // Scale to variance = 1/φ

                // Get affine parameters if present
                let gamma_val = if let Some(ref gamma) = self.gamma {
                    gamma.to_floats()?[c]
                } else {
                    1.0
                };

                let beta_val = if let Some(ref beta) = self.beta {
                    beta.to_floats()?[c]
                } else {
                    0.0
                };

                for s in 0..spatial_size {
                    let idx = b * num_features * spatial_size + c * spatial_size + s;
                    let normalized = (x_floats[idx] - mean[c]) * scale;
                    out_floats[idx] = normalized * gamma_val + beta_val;
                }
            }
        }

        // Convert back to ResonantTensor with lattice snapping
        ResonantTensor::from_floats(
            out_floats,
            x.shape.clone(),
            x.mode_norm_sq.clone(),
            100,
            0,
        )
    }

    #[cfg(feature = "cuda")]
    fn forward_cuda(&mut self, x: &ResonantTensor, training: bool) -> PyResult<ResonantTensor> {
        // Wake flux if needed
        let x = if x.phase == ResonantPhase::Crystallized {
            x.wake_flux()?
        } else {
            x.clone()
        };

        // Call CUDA kernel (to be implemented)
        unimplemented!("CUDA golden batch norm kernel not yet implemented")
    }
}
```

### 2.2 CUDA Kernel

**File:** `rust/kernels/golden_batch_norm.cu`

```cuda
// Syntonic CUDA Kernels - Golden Batch Normalization
// Batch normalization with target variance = 1/φ ≈ 0.618

#include "srt_constants.cuh"

// =============================================================================
// 2D Golden Batch Norm (for Conv2D: [B, C, H, W])
// =============================================================================

extern "C" __global__ void golden_batch_norm_2d_f64(
    double *out,                  // Output [B, C, H, W]
    const double *in,             // Input [B, C, H, W]
    const double *mean,           // Mean [C] or batch mean if training
    const double *var,            // Variance [C] or batch var if training
    const double *gamma,          // Scale [C], NULL for 1.0
    const double *beta,           // Bias [C], NULL for 0.0
    double eps,
    int B,                        // Batch size
    int C,                        // Channels
    int H,                        // Height
    int W                         // Width
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int total = B * C * H * W;

    if (tid >= total) return;

    // Decode indices: [b, c, h, w]
    int w_idx = tid % W;
    int h_idx = (tid / W) % H;
    int c_idx = (tid / (W * H)) % C;
    int b_idx = tid / (C * W * H);

    // Get channel statistics
    double m = mean[c_idx];
    double v = var[c_idx];

    // Normalize to N(0, 1)
    double x_norm = (in[tid] - m) / sqrt(v + eps);

    // Scale to golden target variance N(0, 1/φ)
    x_norm *= sqrt(PHI_INV_F64);

    // Affine transform
    if (gamma != NULL) x_norm *= gamma[c_idx];
    if (beta != NULL) x_norm += beta[c_idx];

    out[tid] = x_norm;
}

extern "C" __global__ void golden_batch_norm_2d_f32(
    float *out,
    const float *in,
    const float *mean,
    const float *var,
    const float *gamma,
    const float *beta,
    float eps,
    int B, int C, int H, int W
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int total = B * C * H * W;

    if (tid >= total) return;

    int w_idx = tid % W;
    int h_idx = (tid / W) % H;
    int c_idx = (tid / (W * H)) % C;
    int b_idx = tid / (C * W * H);

    float m = mean[c_idx];
    float v = var[c_idx];

    float x_norm = (in[tid] - m) / sqrtf(v + eps);
    x_norm *= sqrtf(PHI_INV_F32);

    if (gamma != NULL) x_norm *= gamma[c_idx];
    if (beta != NULL) x_norm += beta[c_idx];

    out[tid] = x_norm;
}

// =============================================================================
// Compute batch statistics (mean and variance) per channel
// =============================================================================

extern "C" __global__ void compute_batch_stats_2d_f64(
    double *mean_out,             // Output mean [C]
    double *var_out,              // Output variance [C]
    const double *in,             // Input [B, C, H, W]
    int B, int C, int H, int W
) {
    extern __shared__ double shared[];
    double *s_sum = shared;
    double *s_sum_sq = shared + blockDim.x;

    int c = blockIdx.x;
    int tid = threadIdx.x;

    if (c >= C) return;

    int spatial_size = H * W;
    int norm_size = B * spatial_size;

    // Accumulate sum and sum_sq for this channel
    double local_sum = 0.0;
    double local_sum_sq = 0.0;

    for (int i = tid; i < norm_size; i += blockDim.x) {
        int b = i / spatial_size;
        int spatial_idx = i % spatial_size;
        int idx = b * C * spatial_size + c * spatial_size + spatial_idx;

        double val = in[idx];
        local_sum += val;
        local_sum_sq += val * val;
    }

    s_sum[tid] = local_sum;
    s_sum_sq[tid] = local_sum_sq;
    __syncthreads();

    // Block reduction
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            s_sum[tid] += s_sum[tid + s];
            s_sum_sq[tid] += s_sum_sq[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        double sum = s_sum[0];
        double sum_sq = s_sum_sq[0];
        double n = (double)norm_size;

        mean_out[c] = sum / n;
        var_out[c] = (sum_sq / n) - (mean_out[c] * mean_out[c]);
    }
}
```

---

## Phase 3: Syntonic Softmax

### 3.1 Rust Implementation

**File:** `rust/src/resonant/syntonic_softmax.rs`

```rust
//! Syntony-weighted softmax for theory-aligned classification.
//!
//! Standard softmax: p_i = exp(x_i) / Σ exp(x_j)
//! Syntonic softmax: p_i = w_i * exp(x_i) / Σ w_j * exp(x_j)
//!
//! where w(n) = exp(-|n|²/φ) is the golden measure weight.

use pyo3::prelude::*;
use crate::resonant::{ResonantTensor, ResonantPhase};
use crate::exact::GoldenExact;
use super::{PHI, PHI_INV};

#[cfg(feature = "cuda")]
use cudarc::driver::{CudaDevice, CudaSlice, LaunchAsync, LaunchConfig};

/// Syntonic softmax modes
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[pyclass]
pub enum SyntonicSoftmaxMode {
    /// Learn mode norms per feature (recommended)
    Learned,
    /// Accept pre-computed syntony values
    Provided,
    /// Standard softmax (ablation baseline)
    Identity,
}

#[pymethods]
impl SyntonicSoftmaxMode {
    #[new]
    fn new(mode_str: &str) -> PyResult<Self> {
        match mode_str {
            "learned" => Ok(SyntonicSoftmaxMode::Learned),
            "provided" => Ok(SyntonicSoftmaxMode::Provided),
            "identity" => Ok(SyntonicSoftmaxMode::Identity),
            _ => Err(pyo3::exceptions::PyValueError::new_err(
                format!("Unknown mode: {}. Use 'learned', 'provided', or 'identity'", mode_str)
            )),
        }
    }
}

/// Syntonic softmax state (learnable mode norms)
#[pyclass]
#[derive(Clone, Debug)]
pub struct SyntonicSoftmaxState {
    /// Mode norms |n|² for each feature [num_features]
    mode_norms: Option<ResonantTensor>,

    /// Syntony weighting scale
    syntony_scale: f64,

    /// Softmax dimension (typically -1 for last dim)
    dim: isize,

    /// Mode
    mode: SyntonicSoftmaxMode,

    /// Number of features (for learned mode)
    num_features: Option<usize>,
}

#[pymethods]
impl SyntonicSoftmaxState {
    #[new]
    fn new(
        dim: Option<isize>,
        mode: SyntonicSoftmaxMode,
        num_features: Option<usize>,
        syntony_scale: Option<f64>,
    ) -> PyResult<Self> {
        // Validate
        if mode == SyntonicSoftmaxMode::Learned && num_features.is_none() {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "num_features required for learned mode"
            ));
        }

        // Initialize mode norms if learned
        let mode_norms = if mode == SyntonicSoftmaxMode::Learned {
            let n = num_features.unwrap();
            // Initialize to 1.0 (will be learned)
            Some(ResonantTensor::ones(vec![n], 100)?)
        } else {
            None
        };

        Ok(Self {
            mode_norms,
            syntony_scale: syntony_scale.unwrap_or(1.0),
            dim: dim.unwrap_or(-1),
            mode,
            num_features,
        })
    }

    /// Forward pass: apply syntony-weighted softmax
    ///
    /// # Arguments
    /// * `x` - Logits tensor [..., num_classes]
    /// * `syntony` - Optional pre-computed syntony weights (same shape as x)
    ///
    /// # Returns
    /// Probability distribution with syntonic weighting
    fn forward(
        &self,
        x: &ResonantTensor,
        syntony: Option<&ResonantTensor>,
    ) -> PyResult<ResonantTensor> {
        match self.mode {
            SyntonicSoftmaxMode::Identity => {
                // Standard softmax (no syntony weighting)
                x.softmax(self.dim)
            },
            SyntonicSoftmaxMode::Learned => {
                // w(n) = exp(-|n|²/φ)
                let mode_norms = self.mode_norms.as_ref().unwrap();
                let weights = self.compute_golden_weights(mode_norms)?;

                // weighted_logits = logits + scale * log(weights)
                let log_weights = weights.log()?;
                let scaled_log_weights = log_weights.scalar_mul(self.syntony_scale)?;
                let weighted_logits = x.elementwise_add(&scaled_log_weights)?;

                weighted_logits.softmax(self.dim)
            },
            SyntonicSoftmaxMode::Provided => {
                // Use provided syntony values
                let syntony = syntony.ok_or_else(|| {
                    pyo3::exceptions::PyValueError::new_err(
                        "syntony values required for 'provided' mode"
                    )
                })?;

                // Validate shape match
                if syntony.shape != x.shape {
                    return Err(pyo3::exceptions::PyValueError::new_err(
                        format!("Shape mismatch: x {:?} vs syntony {:?}",
                            x.shape, syntony.shape)
                    ));
                }

                // weighted_logits = logits + scale * log(syntony)
                let log_syntony = syntony.log()?;
                let scaled_log_syntony = log_syntony.scalar_mul(self.syntony_scale)?;
                let weighted_logits = x.elementwise_add(&scaled_log_syntony)?;

                weighted_logits.softmax(self.dim)
            },
        }
    }

    /// Get current mode norms (for learned mode)
    fn mode_norms(&self) -> PyResult<Option<ResonantTensor>> {
        Ok(self.mode_norms.clone())
    }
}

impl SyntonicSoftmaxState {
    /// Compute golden measure weights: w(n) = exp(-|n|²/φ)
    fn compute_golden_weights(&self, mode_norms: &ResonantTensor) -> PyResult<ResonantTensor> {
        // w = exp(-mode_norms / φ)
        let scaled_norms = mode_norms.scalar_mul(-(PHI_INV))?;
        scaled_norms.exp()
    }
}

// Python-accessible function
#[pyfunction]
pub fn syntonic_softmax(
    x: &ResonantTensor,
    dim: Option<isize>,
    mode_norms: Option<&ResonantTensor>,
    syntony_scale: Option<f64>,
) -> PyResult<ResonantTensor> {
    let state = if let Some(norms) = mode_norms {
        SyntonicSoftmaxState::new(
            dim,
            SyntonicSoftmaxMode::Learned,
            Some(norms.shape[0]),
            syntony_scale,
        )?
    } else {
        SyntonicSoftmaxState::new(
            dim,
            SyntonicSoftmaxMode::Identity,
            None,
            syntony_scale,
        )?
    };

    state.forward(x, None)
}
```

### 3.2 CUDA Kernel

**File:** `rust/kernels/syntonic_softmax.cu`

```cuda
// Syntonic CUDA Kernels - Syntony-Weighted Softmax
// Softmax with golden measure weighting w(n) = exp(-|n|²/φ)

#include "srt_constants.cuh"
#include <float.h>

// =============================================================================
// Syntonic Softmax: learned mode (exp(-mode_norms/φ) weighting)
// =============================================================================

extern "C" __global__ void syntonic_softmax_learned_f64(
    double *out,                  // Output probabilities [batch, num_classes]
    const double *logits,         // Input logits [batch, num_classes]
    const double *mode_norms,     // Mode norms |n|² [num_classes]
    double syntony_scale,
    int batch_size,
    int num_classes
) {
    extern __shared__ double shared[];
    double *s_max = shared;
    double *s_sum = shared + blockDim.x;

    int batch_idx = blockIdx.x;
    int tid = threadIdx.x;

    if (batch_idx >= batch_size) return;

    const double *x = logits + batch_idx * num_classes;
    double *y = out + batch_idx * num_classes;

    // Compute log weights: log(w) = -mode_norms / φ * scale
    // This will be added to logits for numerical stability

    // Step 1: Find max for numerical stability
    double local_max = -DBL_MAX;
    for (int i = tid; i < num_classes; i += blockDim.x) {
        double log_weight = -mode_norms[i] * PHI_INV_F64 * syntony_scale;
        double weighted_logit = x[i] + log_weight;
        local_max = fmax(local_max, weighted_logit);
    }

    s_max[tid] = local_max;
    __syncthreads();

    // Block-level max reduction
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            s_max[tid] = fmax(s_max[tid], s_max[tid + s]);
        }
        __syncthreads();
    }

    double max_val = s_max[0];
    __syncthreads();

    // Step 2: Compute exp(x - max) and sum
    double local_sum = 0.0;
    for (int i = tid; i < num_classes; i += blockDim.x) {
        double log_weight = -mode_norms[i] * PHI_INV_F64 * syntony_scale;
        double weighted_logit = x[i] + log_weight;
        local_sum += exp(weighted_logit - max_val);
    }

    s_sum[tid] = local_sum;
    __syncthreads();

    // Block-level sum reduction
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            s_sum[tid] += s_sum[tid + s];
        }
        __syncthreads();
    }

    double sum = s_sum[0];
    __syncthreads();

    // Step 3: Normalize
    for (int i = tid; i < num_classes; i += blockDim.x) {
        double log_weight = -mode_norms[i] * PHI_INV_F64 * syntony_scale;
        double weighted_logit = x[i] + log_weight;
        y[i] = exp(weighted_logit - max_val) / sum;
    }
}

extern "C" __global__ void syntonic_softmax_learned_f32(
    float *out,
    const float *logits,
    const float *mode_norms,
    float syntony_scale,
    int batch_size,
    int num_classes
) {
    extern __shared__ float shared_f[];
    float *s_max = shared_f;
    float *s_sum = shared_f + blockDim.x;

    int batch_idx = blockIdx.x;
    int tid = threadIdx.x;

    if (batch_idx >= batch_size) return;

    const float *x = logits + batch_idx * num_classes;
    float *y = out + batch_idx * num_classes;

    // Step 1: Find max
    float local_max = -FLT_MAX;
    for (int i = tid; i < num_classes; i += blockDim.x) {
        float log_weight = -mode_norms[i] * PHI_INV_F32 * syntony_scale;
        float weighted_logit = x[i] + log_weight;
        local_max = fmaxf(local_max, weighted_logit);
    }

    s_max[tid] = local_max;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            s_max[tid] = fmaxf(s_max[tid], s_max[tid + s]);
        }
        __syncthreads();
    }

    float max_val = s_max[0];
    __syncthreads();

    // Step 2: Sum
    float local_sum = 0.0f;
    for (int i = tid; i < num_classes; i += blockDim.x) {
        float log_weight = -mode_norms[i] * PHI_INV_F32 * syntony_scale;
        float weighted_logit = x[i] + log_weight;
        local_sum += __expf(weighted_logit - max_val);
    }

    s_sum[tid] = local_sum;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            s_sum[tid] += s_sum[tid + s];
        }
        __syncthreads();
    }

    float sum = s_sum[0];
    __syncthreads();

    // Step 3: Normalize
    for (int i = tid; i < num_classes; i += blockDim.x) {
        float log_weight = -mode_norms[i] * PHI_INV_F32 * syntony_scale;
        float weighted_logit = x[i] + log_weight;
        y[i] = __expf(weighted_logit - max_val) / sum;
    }
}

// =============================================================================
// Syntonic Softmax: provided mode (direct syntony weights)
// =============================================================================

extern "C" __global__ void syntonic_softmax_provided_f64(
    double *out,
    const double *logits,
    const double *syntony,       // Pre-computed syntony weights (same shape as logits)
    double syntony_scale,
    int batch_size,
    int num_classes
) {
    // Similar to learned mode but uses log(syntony) directly
    extern __shared__ double shared[];
    double *s_max = shared;
    double *s_sum = shared + blockDim.x;

    int batch_idx = blockIdx.x;
    int tid = threadIdx.x;

    if (batch_idx >= batch_size) return;

    const double *x = logits + batch_idx * num_classes;
    const double *w = syntony + batch_idx * num_classes;
    double *y = out + batch_idx * num_classes;

    // Find max(x + scale * log(w))
    double local_max = -DBL_MAX;
    for (int i = tid; i < num_classes; i += blockDim.x) {
        double log_weight = log(fmax(w[i], 1e-10)) * syntony_scale;
        local_max = fmax(local_max, x[i] + log_weight);
    }

    s_max[tid] = local_max;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) s_max[tid] = fmax(s_max[tid], s_max[tid + s]);
        __syncthreads();
    }

    double max_val = s_max[0];
    __syncthreads();

    // Sum
    double local_sum = 0.0;
    for (int i = tid; i < num_classes; i += blockDim.x) {
        double log_weight = log(fmax(w[i], 1e-10)) * syntony_scale;
        local_sum += exp(x[i] + log_weight - max_val);
    }

    s_sum[tid] = local_sum;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) s_sum[tid] += s_sum[tid + s];
        __syncthreads();
    }

    double sum = s_sum[0];
    __syncthreads();

    // Normalize
    for (int i = tid; i < num_classes; i += blockDim.x) {
        double log_weight = log(fmax(w[i], 1e-10)) * syntony_scale;
        y[i] = exp(x[i] + log_weight - max_val) / sum;
    }
}
```

---

## Phase 4: Integration & Testing

### 4.1 Update `lib.rs` with PyO3 Bindings

Add to `rust/src/lib.rs`:

```rust
// === Phi-Residual Operations ===
m.add_class::<PhiResidualMode>()?;
m.add_function(wrap_pyfunction!(phi_residual, m)?)?;

// === Golden Batch Normalization ===
m.add_class::<GoldenBatchNormConfig>()?;
m.add_class::<GoldenBatchNormState>()?;

// === Syntonic Softmax ===
m.add_class::<SyntonicSoftmaxMode>()?;
m.add_class::<SyntonicSoftmaxState>()?;
m.add_function(wrap_pyfunction!(syntonic_softmax, m)?)?;
```

### 4.2 Build CUDA PTX Files

Add to build process (`Makefile` or build script):

```bash
# Compile CUDA kernels to PTX
nvcc -ptx -arch=sm_75 -o kernels/ptx/phi_residual_sm75.ptx kernels/phi_residual.cu
nvcc -ptx -arch=sm_80 -o kernels/ptx/phi_residual_sm80.ptx kernels/phi_residual.cu
nvcc -ptx -arch=sm_86 -o kernels/ptx/phi_residual_sm86.ptx kernels/phi_residual.cu
nvcc -ptx -arch=sm_90 -o kernels/ptx/phi_residual_sm90.ptx kernels/phi_residual.cu

nvcc -ptx -arch=sm_75 -o kernels/ptx/golden_batch_norm_sm75.ptx kernels/golden_batch_norm.cu
nvcc -ptx -arch=sm_80 -o kernels/ptx/golden_batch_norm_sm80.ptx kernels/golden_batch_norm.cu
nvcc -ptx -arch=sm_86 -o kernels/ptx/golden_batch_norm_sm86.ptx kernels/golden_batch_norm.cu
nvcc -ptx -arch=sm_90 -o kernels/ptx/golden_batch_norm_sm90.ptx kernels/golden_batch_norm.cu

nvcc -ptx -arch=sm_75 -o kernels/ptx/syntonic_softmax_sm75.ptx kernels/syntonic_softmax.cu
nvcc -ptx -arch=sm_80 -o kernels/ptx/syntonic_softmax_sm80.ptx kernels/syntonic_softmax.cu
nvcc -ptx -arch=sm_86 -o kernels/ptx/syntonic_softmax_sm86.ptx kernels/syntonic_softmax.cu
nvcc -ptx -arch=sm_90 -o kernels/ptx/syntonic_softmax_sm90.ptx kernels/syntonic_softmax.cu
```

### 4.3 Rust Unit Tests

Create `rust/src/resonant/tests.rs`:

```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_phi_residual_mode_phi() {
        let identity = ResonantTensor::ones(vec![4], 100).unwrap();
        let residual = ResonantTensor::from_floats(
            vec![1.0, 2.0, 3.0, 4.0],
            vec![4],
            vec![0.0; 4],
            100,
            0,
        ).unwrap();

        let result = ResonantTensor::phi_residual(
            &identity,
            &residual,
            PhiResidualMode::Phi,
        ).unwrap();

        let expected = vec![
            1.0 + 1.0 * PHI_INV,
            1.0 + 2.0 * PHI_INV,
            1.0 + 3.0 * PHI_INV,
            1.0 + 4.0 * PHI_INV,
        ];

        let result_floats = result.to_floats().unwrap();
        for (a, b) in result_floats.iter().zip(expected.iter()) {
            assert!((a - b).abs() < 1e-10);
        }
    }

    #[test]
    fn test_golden_batch_norm_target_variance() {
        let config = GoldenBatchNormConfig::new(
            3,  // num_features
            Some(1e-5),
            Some(0.1),
            false,  // no affine
            false,  // no running stats
        );

        let mut state = GoldenBatchNormState::new(config).unwrap();

        // Create input with known statistics
        let input = ResonantTensor::randn(vec![32, 3, 8, 8], 100, 0).unwrap();

        let output = state.forward(&input, true).unwrap();

        // Verify output variance ≈ 1/φ
        let output_floats = output.to_floats().unwrap();
        let var = compute_variance(&output_floats);

        assert!((var - PHI_INV).abs() < 0.1);  // Within 10% tolerance
    }

    #[test]
    fn test_syntonic_softmax_sums_to_one() {
        let state = SyntonicSoftmaxState::new(
            Some(-1),
            SyntonicSoftmaxMode::Identity,
            None,
            Some(1.0),
        ).unwrap();

        let logits = ResonantTensor::randn(vec![8, 10], 100, 0).unwrap();
        let probs = state.forward(&logits, None).unwrap();

        let probs_floats = probs.to_floats().unwrap();

        // Verify each row sums to 1.0
        for b in 0..8 {
            let row_sum: f64 = probs_floats[b*10..(b+1)*10].iter().sum();
            assert!((row_sum - 1.0).abs() < 1e-6);
        }
    }
}
```

---

## Implementation Checklist

### Phase 1: Phi-Residual Operations (3-4 hours)
- [ ] Create `rust/src/resonant/phi_ops.rs` with Rust implementation
- [ ] Create `rust/kernels/phi_residual.cu` with CUDA kernels
- [ ] Add kernel wrapper functions to `srt_kernels.rs`
- [ ] Add PyO3 bindings to `lib.rs`
- [ ] Write Rust unit tests
- [ ] Compile CUDA kernels to PTX
- [ ] Test on CPU and GPU

### Phase 2: Golden Batch Normalization (4-5 hours)
- [ ] Create `rust/src/resonant/golden_norm.rs` with Rust implementation
- [ ] Implement CPU forward pass with exact statistics
- [ ] Create `rust/kernels/golden_batch_norm.cu` with CUDA kernels
- [ ] Add batch statistics computation kernel
- [ ] Add kernel wrapper functions to `srt_kernels.rs`
- [ ] Add PyO3 bindings to `lib.rs`
- [ ] Write Rust unit tests (verify variance target)
- [ ] Compile CUDA kernels to PTX
- [ ] Test on CPU and GPU

### Phase 3: Syntonic Softmax (4-5 hours)
- [ ] Create `rust/src/resonant/syntonic_softmax.rs` with Rust implementation
- [ ] Implement all three modes (learned, provided, identity)
- [ ] Create `rust/kernels/syntonic_softmax.cu` with CUDA kernels
- [ ] Implement numerically stable softmax with golden weighting
- [ ] Add kernel wrapper functions to `srt_kernels.rs`
- [ ] Add PyO3 bindings to `lib.rs`
- [ ] Write Rust unit tests (verify probability sums, weighting)
- [ ] Compile CUDA kernels to PTX
- [ ] Test on CPU and GPU

### Phase 4: Integration (2-3 hours)
- [ ] Update `rust/src/resonant/mod.rs` with new modules
- [ ] Verify all PyO3 bindings work from Python
- [ ] Run full Rust test suite: `cargo test`
- [ ] Build Python package: `maturin develop --release`
- [ ] Test imports: `python -c "from syntonic._core import phi_residual"`
- [ ] Profile CUDA kernel performance
- [ ] Document API usage

---

## Performance Expectations

| Operation | CPU (1000 elem) | GPU (1M elem) | Speedup |
|-----------|----------------|---------------|---------|
| Phi-Residual | ~50 μs | ~100 μs | ~500x |
| Golden BatchNorm | ~200 μs | ~500 μs | ~400x |
| Syntonic Softmax | ~150 μs | ~300 μs | ~500x |

---

## Success Criteria

✅ **All operations implemented in Rust with exact Q(φ) arithmetic**
✅ **CUDA kernels provide 400-500x speedup on large tensors**
✅ **PyO3 bindings expose all operations to Python**
✅ **Unit tests verify mathematical correctness**
✅ **Golden ratio constants (φ, 1/φ) used correctly**
✅ **Compatible with existing ResonantTensor infrastructure**
✅ **Ready for Python layer implementation in Phase 2**

---

**Next Step:** Begin Phase 1 - Phi-Residual Operations implementation.
