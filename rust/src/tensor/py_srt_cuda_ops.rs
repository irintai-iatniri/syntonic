//! Python bindings for SRT CUDA operations
//!
//! Exposes toroidal math, gnosis masking, golden exponentials,
//! autograd kernels, and matrix multiplication to Python.

use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;

// SRT Constants
const PHI: f64 = 1.6180339887498948482;
const PHI_INV: f64 = 0.6180339887498948482;
const Q_DEFICIT: f64 = 0.027395146920;

#[cfg(feature = "cuda")]
use super::cuda::device_manager::get_device;

#[cfg(feature = "cuda")]
use cudarc::driver::CudaSlice;

// =============================================================================
// Toroidal Math Functions (CPU fallbacks + CUDA)
// =============================================================================

/// Compute sin(2πx) for toroidal coordinates (CPU)
fn cpu_sin_toroidal_f64(data: &[f64]) -> Vec<f64> {
    let two_pi = 2.0 * std::f64::consts::PI;
    data.iter().map(|&x| (two_pi * x).sin()).collect()
}

/// Compute cos(2πx) for toroidal coordinates (CPU)
fn cpu_cos_toroidal_f64(data: &[f64]) -> Vec<f64> {
    let two_pi = 2.0 * std::f64::consts::PI;
    data.iter().map(|&x| (two_pi * x).cos()).collect()
}

/// Compute atan2 for toroidal coordinates (CPU)
fn cpu_atan2_toroidal_f64(y: &[f64], x: &[f64]) -> Vec<f64> {
    y.iter()
        .zip(x.iter())
        .map(|(&yi, &xi)| yi.atan2(xi) / (2.0 * std::f64::consts::PI))
        .collect()
}

/// Compute φ^x (golden exponential) using exact Fibonacci recurrence (CPU)
fn cpu_phi_exp_f64(data: &[f64]) -> Vec<f64> {
    data.iter().map(|&x| PHI.powf(x)).collect()
}

/// Compute φ^(-x) (inverse golden exponential) (CPU)
fn cpu_phi_exp_inv_f64(data: &[f64]) -> Vec<f64> {
    data.iter().map(|&x| PHI_INV.powf(x)).collect()
}

// =============================================================================
// Python-Exposed Toroidal Functions
// =============================================================================

/// Compute sin(2πx) for T⁴ torus coordinates
#[pyfunction]
pub fn py_sin_toroidal(data: Vec<f64>) -> Vec<f64> {
    cpu_sin_toroidal_f64(&data)
}

/// Compute cos(2πx) for T⁴ torus coordinates
#[pyfunction]
pub fn py_cos_toroidal(data: Vec<f64>) -> Vec<f64> {
    cpu_cos_toroidal_f64(&data)
}

/// Compute atan2 normalized for toroidal coordinates
#[pyfunction]
pub fn py_atan2_toroidal(y: Vec<f64>, x: Vec<f64>) -> PyResult<Vec<f64>> {
    if y.len() != x.len() {
        return Err(PyRuntimeError::new_err("y and x must have same length"));
    }
    Ok(cpu_atan2_toroidal_f64(&y, &x))
}

/// Compute φ^x (golden exponential)
#[pyfunction]
pub fn py_phi_exp(data: Vec<f64>) -> Vec<f64> {
    cpu_phi_exp_f64(&data)
}

/// Compute φ^(-x) (inverse golden exponential)
#[pyfunction]
pub fn py_phi_exp_inv(data: Vec<f64>) -> Vec<f64> {
    cpu_phi_exp_inv_f64(&data)
}

// =============================================================================
// Gnosis Masking Functions (CPU implementations)
// =============================================================================

/// Standard gnosis mask: filters based on syntony threshold
///
/// mask(i) = input(i) * strength if syntony(i) > threshold else 0
fn cpu_gnosis_mask_f64(input: &[f64], syntony: &[f64], threshold: f64, strength: f64) -> Vec<f64> {
    input
        .iter()
        .zip(syntony.iter())
        .map(
            |(&inp, &syn)| {
                if syn > threshold {
                    inp * strength
                } else {
                    0.0
                }
            },
        )
        .collect()
}

/// Adaptive gnosis mask: adjusts threshold based on local syntony
///
/// local_threshold = threshold * (1 - adaptability * (syntony - mean_syntony))
fn cpu_adaptive_gnosis_mask_f64(
    input: &[f64],
    syntony: &[f64],
    adaptability: f64,
    ratio: f64,
) -> Vec<f64> {
    let mean_syn: f64 = syntony.iter().sum::<f64>() / syntony.len() as f64;
    let threshold = 1.0 - Q_DEFICIT; // Default threshold from q-deficit

    input
        .iter()
        .zip(syntony.iter())
        .map(|(&inp, &syn)| {
            let local_thresh = threshold * (1.0 - adaptability * (syn - mean_syn));
            if syn > local_thresh {
                inp * ratio
            } else {
                inp * (1.0 - ratio)
            }
        })
        .collect()
}

/// Fractal gnosis mask: applies hierarchical masking at multiple scales
fn cpu_fractal_gnosis_mask_f64(
    input: &[f64],
    syntony: &[f64],
    levels: usize,
    threshold: f64,
    scale: f64,
) -> Vec<f64> {
    let mut result = input.to_vec();

    for level in 0..levels {
        let level_scale = scale.powf(level as f64);
        let level_thresh = threshold * PHI_INV.powf(level as f64);

        for i in 0..result.len() {
            if syntony[i] > level_thresh {
                result[i] *= level_scale;
            }
        }
    }

    result
}

/// Temporal gnosis mask: incorporates previous state with memory decay
fn cpu_temporal_gnosis_mask_f64(
    input: &[f64],
    syntony: &[f64],
    prev: &[f64],
    threshold: f64,
    memory: f64, // How much to remember (0-1)
    rate: f64,   // Learning rate for new information
) -> Vec<f64> {
    input
        .iter()
        .zip(syntony.iter())
        .zip(prev.iter())
        .map(|((&inp, &syn), &prv)| {
            let new_val = if syn > threshold { inp * rate } else { 0.0 };
            memory * prv + (1.0 - memory) * new_val
        })
        .collect()
}

// =============================================================================
// Python-Exposed Gnosis Mask Functions
// =============================================================================

/// Apply gnosis mask to filter by syntony threshold
#[pyfunction]
#[pyo3(signature = (input, syntony, threshold=0.9726, strength=1.0))]
pub fn py_gnosis_mask(
    input: Vec<f64>,
    syntony: Vec<f64>,
    threshold: f64,
    strength: f64,
) -> PyResult<Vec<f64>> {
    if input.len() != syntony.len() {
        return Err(PyRuntimeError::new_err(
            "input and syntony must have same length",
        ));
    }
    Ok(cpu_gnosis_mask_f64(&input, &syntony, threshold, strength))
}

/// Apply adaptive gnosis mask with local threshold adjustment
#[pyfunction]
#[pyo3(signature = (input, syntony, adaptability=0.1, ratio=1.0))]
pub fn py_adaptive_gnosis_mask(
    input: Vec<f64>,
    syntony: Vec<f64>,
    adaptability: f64,
    ratio: f64,
) -> PyResult<Vec<f64>> {
    if input.len() != syntony.len() {
        return Err(PyRuntimeError::new_err(
            "input and syntony must have same length",
        ));
    }
    Ok(cpu_adaptive_gnosis_mask_f64(
        &input,
        &syntony,
        adaptability,
        ratio,
    ))
}

/// Apply fractal gnosis mask at multiple hierarchical levels
#[pyfunction]
#[pyo3(signature = (input, syntony, levels=3, threshold=0.9726, scale=1.618))]
pub fn py_fractal_gnosis_mask(
    input: Vec<f64>,
    syntony: Vec<f64>,
    levels: usize,
    threshold: f64,
    scale: f64,
) -> PyResult<Vec<f64>> {
    if input.len() != syntony.len() {
        return Err(PyRuntimeError::new_err(
            "input and syntony must have same length",
        ));
    }
    Ok(cpu_fractal_gnosis_mask_f64(
        &input, &syntony, levels, threshold, scale,
    ))
}

/// Apply temporal gnosis mask with memory of previous state
#[pyfunction]
#[pyo3(signature = (input, syntony, prev, threshold=0.9726, memory=0.618, rate=1.0))]
pub fn py_temporal_gnosis_mask(
    input: Vec<f64>,
    syntony: Vec<f64>,
    prev: Vec<f64>,
    threshold: f64,
    memory: f64,
    rate: f64,
) -> PyResult<Vec<f64>> {
    if input.len() != syntony.len() || input.len() != prev.len() {
        return Err(PyRuntimeError::new_err(
            "input, syntony, and prev must have same length",
        ));
    }
    Ok(cpu_temporal_gnosis_mask_f64(
        &input, &syntony, &prev, threshold, memory, rate,
    ))
}

// =============================================================================
// Autograd Gradient Filtering (CPU implementation)
// =============================================================================

/// Filter gradients using golden attractor - corrupted gradients are snapped
/// to the Q(φ) lattice to prevent gradient explosion/vanishing
fn cpu_autograd_filter_f64(
    gradients: &[f64],
    current_state: &[f64],
    golden_attractor_strength: f64,
    corruption_threshold: f64,
) -> Vec<f64> {
    let phi = PHI;
    let phi_inv = PHI_INV;

    gradients
        .iter()
        .zip(current_state.iter())
        .map(|(&grad, &state)| {
            // Check if gradient is "corrupted" (too large or too small)
            let abs_grad = grad.abs();
            let is_corrupted = abs_grad > corruption_threshold
                || (abs_grad > 0.0 && abs_grad < corruption_threshold * phi_inv.powi(10));

            if is_corrupted {
                // Snap gradient to Q(φ) lattice point
                // Find nearest a + b*φ where a, b are small integers
                let scaled = grad / phi_inv;
                let a = scaled.round();
                let b = ((grad - a * phi_inv) / phi).round();
                let snapped = a * phi_inv + b * phi;

                // Blend with attractor
                let attractor = state * golden_attractor_strength;
                snapped * (1.0 - golden_attractor_strength) + attractor * golden_attractor_strength
            } else {
                grad
            }
        })
        .collect()
}

/// Attractor memory update: evolves current state toward attractor basin
fn cpu_attractor_memory_update_f64(
    current: &[f64],
    gradients: &[f64],
    attractor_strength: f64,
    learning_rate: f64,
) -> Vec<f64> {
    current
        .iter()
        .zip(gradients.iter())
        .map(|(&curr, &grad)| {
            // Apply gradient with φ-scaled learning rate
            let updated = curr - learning_rate * grad;

            // Pull toward nearest Q(φ) attractor point
            let scaled = updated / PHI;
            let a = scaled.round();
            let attractor_point = a * PHI;

            // Blend based on attractor strength
            updated * (1.0 - attractor_strength) + attractor_point * attractor_strength
        })
        .collect()
}

// =============================================================================
// Python-Exposed Autograd Functions
// =============================================================================

/// Filter gradients to prevent corruption (explosion/vanishing)
#[pyfunction]
#[pyo3(signature = (gradients, current_state, attractor_strength=0.027395, corruption_threshold=1e6))]
pub fn py_autograd_filter(
    gradients: Vec<f64>,
    current_state: Vec<f64>,
    attractor_strength: f64,
    corruption_threshold: f64,
) -> PyResult<Vec<f64>> {
    if gradients.len() != current_state.len() {
        return Err(PyRuntimeError::new_err(
            "gradients and current_state must have same length",
        ));
    }
    Ok(cpu_autograd_filter_f64(
        &gradients,
        &current_state,
        attractor_strength,
        corruption_threshold,
    ))
}

/// Update memory state with attractor basin pull
#[pyfunction]
#[pyo3(signature = (current, gradients, attractor_strength=0.027395, learning_rate=0.001))]
pub fn py_attractor_memory_update(
    current: Vec<f64>,
    gradients: Vec<f64>,
    attractor_strength: f64,
    learning_rate: f64,
) -> PyResult<Vec<f64>> {
    if current.len() != gradients.len() {
        return Err(PyRuntimeError::new_err(
            "current and gradients must have same length",
        ));
    }
    Ok(cpu_attractor_memory_update_f64(
        &current,
        &gradients,
        attractor_strength,
        learning_rate,
    ))
}

// =============================================================================
// Entropy and Syntony Metric Functions
// =============================================================================

/// Compute entropy: -Σ p * log(p) normalized by golden ratio
fn cpu_entropy_f64(values: &[f64]) -> f64 {
    // Normalize to probabilities
    let sum: f64 = values.iter().map(|v| v.abs()).sum();
    if sum == 0.0 {
        return 0.0;
    }

    let probs: Vec<f64> = values.iter().map(|v| v.abs() / sum).collect();

    // Compute entropy
    let entropy: f64 = probs
        .iter()
        .filter(|&&p| p > 0.0)
        .map(|&p| -p * p.ln())
        .sum();

    // Normalize by log(n) * φ^(-1) for golden-scaled entropy
    let max_entropy = (probs.len() as f64).ln();
    if max_entropy > 0.0 {
        entropy / (max_entropy * PHI_INV)
    } else {
        0.0
    }
}

/// Compute syntony metric: measures how close tensor is to Q(φ) lattice
fn cpu_syntony_metric_f64(tensor: &[f64]) -> f64 {
    if tensor.is_empty() {
        return 1.0;
    }

    // Measure deviation from nearest Q(φ) lattice points
    let total_deviation: f64 = tensor
        .iter()
        .map(|&v| {
            // Find nearest a + b*φ
            let scaled = v / PHI;
            let a = scaled.round();
            let remainder = v - a * PHI;
            let b = (remainder / PHI_INV).round();
            let nearest = a * PHI + b * PHI_INV;
            (v - nearest).abs()
        })
        .sum();

    // Syntony = 1 - (average deviation / PHI)
    let avg_deviation = total_deviation / tensor.len() as f64;
    (1.0 - avg_deviation / PHI).max(0.0).min(1.0)
}

/// Compute golden-scaled entropy
#[pyfunction]
pub fn py_golden_entropy(values: Vec<f64>) -> f64 {
    cpu_entropy_f64(&values)
}

/// Compute syntony metric (how close to Q(φ) lattice)
#[pyfunction]
pub fn py_syntony_metric(tensor: Vec<f64>) -> f64 {
    cpu_syntony_metric_f64(&tensor)
}

// =============================================================================
// CUDA Matrix Multiplication Wrappers (when CUDA feature enabled)
// =============================================================================

#[cfg(feature = "cuda")]
use super::srt_kernels::{cuda_dgemm_native_f64, cuda_sgemm_native_f32};

#[cfg(feature = "cuda")]
use super::cuda::device_manager::get_pool;

/// High-performance SGEMM using SRT native kernels
#[cfg(feature = "cuda")]
#[pyfunction]
#[pyo3(signature = (a, b, m, n, k, alpha=1.0, beta=0.0, device_idx=0))]
pub fn py_cuda_sgemm(
    a: Vec<f32>,
    b: Vec<f32>,
    m: usize,
    n: usize,
    k: usize,
    alpha: f32,
    beta: f32,
    device_idx: usize,
) -> PyResult<Vec<f32>> {
    let device = get_device(device_idx).map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    let pool = get_pool(device_idx).map_err(|e| PyRuntimeError::new_err(e.to_string()))?;

    // Allocate CUDA memory using pool and stream
    let a_dev = device
        .default_stream()
        .clone_htod(&a)
        .map_err(|e| PyRuntimeError::new_err(format!("Failed to copy A: {}", e)))?;
    let b_dev = device
        .default_stream()
        .clone_htod(&b)
        .map_err(|e| PyRuntimeError::new_err(format!("Failed to copy B: {}", e)))?;
    let mut c_dev: CudaSlice<f32> = pool
        .alloc_f32(m * n)
        .map_err(|e| PyRuntimeError::new_err(format!("Failed to alloc C: {}", e)))?;

    // Run SGEMM
    cuda_sgemm_native_f32(&device, &mut c_dev, &a_dev, &b_dev, m, n, k, alpha, beta)?;

    // Copy result back
    let mut result = vec![0.0f32; m * n];
    device
        .default_stream()
        .memcpy_dtoh(&c_dev, &mut result)
        .map_err(|e| PyRuntimeError::new_err(format!("Failed to copy result: {}", e)))?;

    Ok(result)
}

/// Gather elements from source array using indices (f32)
#[cfg(feature = "cuda")]
#[pyfunction]
#[pyo3(signature = (src, indices, device_idx=0))]
pub fn py_gather_f32(src: Vec<f32>, indices: Vec<i32>, device_idx: usize) -> PyResult<Vec<f32>> {
    if src.is_empty() || indices.is_empty() {
        return Err(PyRuntimeError::new_err("src and indices cannot be empty"));
    }

    let device = get_device(device_idx).map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    let pool = get_pool(device_idx).map_err(|e| PyRuntimeError::new_err(e.to_string()))?;

    // Allocate CUDA memory
    let src_dev = device
        .default_stream()
        .clone_htod(&src)
        .map_err(|e| PyRuntimeError::new_err(format!("Failed to copy src: {}", e)))?;
    let idx_dev = device
        .default_stream()
        .clone_htod(&indices)
        .map_err(|e| PyRuntimeError::new_err(format!("Failed to copy indices: {}", e)))?;
    let mut out_dev: CudaSlice<f32> = pool
        .alloc_f32(indices.len())
        .map_err(|e| PyRuntimeError::new_err(format!("Failed to alloc output: {}", e)))?;

    // Run gather
    super::srt_kernels::cuda_gather_f32(&device, &mut out_dev, &src_dev, &idx_dev, indices.len())
        .map_err(|e| PyRuntimeError::new_err(e))?;

    // Copy result back
    let mut result = vec![0.0f32; indices.len()];
    device
        .default_stream()
        .memcpy_dtoh(&out_dev, &mut result)
        .map_err(|e| PyRuntimeError::new_err(format!("Failed to copy result: {}", e)))?;

    Ok(result)
}

/// Scatter elements to output array using indices (f64)
#[cfg(feature = "cuda")]
#[pyfunction]
#[pyo3(signature = (src, indices, output_size, device_idx=0))]
pub fn py_scatter_f64(
    src: Vec<f64>,
    indices: Vec<i64>,
    output_size: usize,
    device_idx: usize,
) -> PyResult<Vec<f64>> {
    if src.is_empty() || indices.is_empty() {
        return Err(PyRuntimeError::new_err("src and indices cannot be empty"));
    }
    if src.len() != indices.len() {
        return Err(PyRuntimeError::new_err(
            "src and indices must have same length",
        ));
    }

    let device = get_device(device_idx).map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    let pool = get_pool(device_idx).map_err(|e| PyRuntimeError::new_err(e.to_string()))?;

    // Allocate CUDA memory
    let src_dev = device
        .default_stream()
        .clone_htod(&src)
        .map_err(|e| PyRuntimeError::new_err(format!("Failed to copy src: {}", e)))?;
    let idx_dev = device
        .default_stream()
        .clone_htod(&indices)
        .map_err(|e| PyRuntimeError::new_err(format!("Failed to copy indices: {}", e)))?;
    let mut out_dev: CudaSlice<f64> = pool
        .alloc_f64(output_size)
        .map_err(|e| PyRuntimeError::new_err(format!("Failed to alloc output: {}", e)))?;

    // Run scatter
    super::srt_kernels::cuda_scatter_f64(&device, &mut out_dev, &src_dev, &idx_dev, indices.len())
        .map_err(|e| PyRuntimeError::new_err(e))?;

    // Copy result back
    let mut result = vec![0.0f64; output_size];
    device
        .default_stream()
        .memcpy_dtoh(&out_dev, &mut result)
        .map_err(|e| PyRuntimeError::new_err(format!("Failed to copy result: {}", e)))?;

    Ok(result)
}

/// Scatter add elements to output array using indices (f64)
#[cfg(feature = "cuda")]
#[pyfunction]
#[pyo3(signature = (src, indices, output_size, device_idx=0))]
pub fn py_scatter_add_f64(
    src: Vec<f64>,
    indices: Vec<i64>,
    output_size: usize,
    device_idx: usize,
) -> PyResult<Vec<f64>> {
    if src.is_empty() || indices.is_empty() {
        return Err(PyRuntimeError::new_err("src and indices cannot be empty"));
    }
    if src.len() != indices.len() {
        return Err(PyRuntimeError::new_err(
            "src and indices must have same length",
        ));
    }

    let device = get_device(device_idx).map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    let pool = get_pool(device_idx).map_err(|e| PyRuntimeError::new_err(e.to_string()))?;

    // Allocate CUDA memory
    let src_dev = device
        .default_stream()
        .clone_htod(&src)
        .map_err(|e| PyRuntimeError::new_err(format!("Failed to copy src: {}", e)))?;
    let idx_dev = device
        .default_stream()
        .clone_htod(&indices)
        .map_err(|e| PyRuntimeError::new_err(format!("Failed to copy indices: {}", e)))?;
    let mut out_dev: CudaSlice<f64> = pool
        .alloc_f64(output_size)
        .map_err(|e| PyRuntimeError::new_err(format!("Failed to alloc output: {}", e)))?;

    // Run scatter add
    super::srt_kernels::cuda_scatter_add_f64(
        &device,
        &mut out_dev,
        &src_dev,
        &idx_dev,
        indices.len(),
    )
    .map_err(|e| PyRuntimeError::new_err(e))?;

    // Copy result back
    let mut result = vec![0.0f64; output_size];
    device
        .default_stream()
        .memcpy_dtoh(&out_dev, &mut result)
        .map_err(|e| PyRuntimeError::new_err(format!("Failed to copy result: {}", e)))?;

    Ok(result)
}

/// Scatter elements to output array using indices (f32)
#[cfg(feature = "cuda")]
#[pyfunction]
#[pyo3(signature = (src, indices, output_size, device_idx=0))]
pub fn py_scatter_f32(
    src: Vec<f32>,
    indices: Vec<i32>,
    output_size: usize,
    device_idx: usize,
) -> PyResult<Vec<f32>> {
    if src.is_empty() || indices.is_empty() {
        return Err(PyRuntimeError::new_err("src and indices cannot be empty"));
    }
    if src.len() != indices.len() {
        return Err(PyRuntimeError::new_err(
            "src and indices must have same length",
        ));
    }

    let device = get_device(device_idx).map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    let pool = get_pool(device_idx).map_err(|e| PyRuntimeError::new_err(e.to_string()))?;

    // Allocate CUDA memory
    let src_dev = device
        .default_stream()
        .clone_htod(&src)
        .map_err(|e| PyRuntimeError::new_err(format!("Failed to copy src: {}", e)))?;
    let idx_dev = device
        .default_stream()
        .clone_htod(&indices)
        .map_err(|e| PyRuntimeError::new_err(format!("Failed to copy indices: {}", e)))?;
    let mut out_dev: CudaSlice<f32> = pool
        .alloc_f32(output_size)
        .map_err(|e| PyRuntimeError::new_err(format!("Failed to alloc output: {}", e)))?;

    // Run scatter
    super::srt_kernels::cuda_scatter_f32(&device, &mut out_dev, &src_dev, &idx_dev, indices.len())
        .map_err(|e| PyRuntimeError::new_err(e))?;

    // Copy result back
    let mut result = vec![0.0f32; output_size];
    device
        .default_stream()
        .memcpy_dtoh(&out_dev, &mut result)
        .map_err(|e| PyRuntimeError::new_err(format!("Failed to copy result: {}", e)))?;

    Ok(result)
}

/// Scatter with golden ratio weighting (f64)
#[cfg(feature = "cuda")]
#[pyfunction]
#[pyo3(signature = (src, indices, output_size, device_idx=0))]
pub fn py_scatter_golden_f64(
    src: Vec<f64>,
    indices: Vec<i64>,
    output_size: usize,
    device_idx: usize,
) -> PyResult<Vec<f64>> {
    if src.is_empty() || indices.is_empty() {
        return Err(PyRuntimeError::new_err("src and indices cannot be empty"));
    }
    if src.len() != indices.len() {
        return Err(PyRuntimeError::new_err(
            "src and indices must have same length",
        ));
    }

    let device = get_device(device_idx).map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    let pool = get_pool(device_idx).map_err(|e| PyRuntimeError::new_err(e.to_string()))?;

    let src_dev = device
        .default_stream()
        .clone_htod(&src)
        .map_err(|e| PyRuntimeError::new_err(format!("Failed to copy src: {}", e)))?;
    let idx_dev = device
        .default_stream()
        .clone_htod(&indices)
        .map_err(|e| PyRuntimeError::new_err(format!("Failed to copy indices: {}", e)))?;
    let mut out_dev: CudaSlice<f64> = pool
        .alloc_f64(output_size)
        .map_err(|e| PyRuntimeError::new_err(format!("Failed to alloc output: {}", e)))?;

    super::srt_kernels::cuda_scatter_golden_f64(
        &device,
        &mut out_dev,
        &src_dev,
        &idx_dev,
        indices.len(),
    )
    .map_err(|e| PyRuntimeError::new_err(e))?;

    let mut result = vec![0.0f64; output_size];
    device
        .default_stream()
        .memcpy_dtoh(&out_dev, &mut result)
        .map_err(|e| PyRuntimeError::new_err(format!("Failed to copy result: {}", e)))?;

    Ok(result)
}

/// Scatter with Mersenne stable precision (f64)
#[cfg(feature = "cuda")]
#[pyfunction]
#[pyo3(signature = (src, indices, output_size, device_idx=0))]
pub fn py_scatter_mersenne_stable_f64(
    src: Vec<f64>,
    indices: Vec<i64>,
    output_size: usize,
    device_idx: usize,
) -> PyResult<Vec<f64>> {
    if src.is_empty() || indices.is_empty() {
        return Err(PyRuntimeError::new_err("src and indices cannot be empty"));
    }
    if src.len() != indices.len() {
        return Err(PyRuntimeError::new_err(
            "src and indices must have same length",
        ));
    }

    let device = get_device(device_idx).map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    let pool = get_pool(device_idx).map_err(|e| PyRuntimeError::new_err(e.to_string()))?;

    let src_dev = device
        .default_stream()
        .clone_htod(&src)
        .map_err(|e| PyRuntimeError::new_err(format!("Failed to copy src: {}", e)))?;
    let idx_dev = device
        .default_stream()
        .clone_htod(&indices)
        .map_err(|e| PyRuntimeError::new_err(format!("Failed to copy indices: {}", e)))?;
    let mut out_dev: CudaSlice<f64> = pool
        .alloc_f64(output_size)
        .map_err(|e| PyRuntimeError::new_err(format!("Failed to alloc output: {}", e)))?;

    super::srt_kernels::cuda_scatter_mersenne_stable_f64(
        &device,
        &mut out_dev,
        &src_dev,
        &idx_dev,
        indices.len(),
    )
    .map_err(|e| PyRuntimeError::new_err(e))?;

    let mut result = vec![0.0f64; output_size];
    device
        .default_stream()
        .memcpy_dtoh(&out_dev, &mut result)
        .map_err(|e| PyRuntimeError::new_err(format!("Failed to copy result: {}", e)))?;

    Ok(result)
}

/// Gather with Lucas shadow weighting (f64)
#[cfg(feature = "cuda")]
#[pyfunction]
#[pyo3(signature = (src, indices, device_idx=0))]
pub fn py_gather_lucas_shadow_f64(
    src: Vec<f64>,
    indices: Vec<i64>,
    device_idx: usize,
) -> PyResult<Vec<f64>> {
    if src.is_empty() || indices.is_empty() {
        return Err(PyRuntimeError::new_err("src and indices cannot be empty"));
    }

    let device = get_device(device_idx).map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    let pool = get_pool(device_idx).map_err(|e| PyRuntimeError::new_err(e.to_string()))?;

    let src_dev = device
        .default_stream()
        .clone_htod(&src)
        .map_err(|e| PyRuntimeError::new_err(format!("Failed to copy src: {}", e)))?;
    let idx_dev = device
        .default_stream()
        .clone_htod(&indices)
        .map_err(|e| PyRuntimeError::new_err(format!("Failed to copy indices: {}", e)))?;
    let mut out_dev: CudaSlice<f64> = pool
        .alloc_f64(indices.len())
        .map_err(|e| PyRuntimeError::new_err(format!("Failed to alloc output: {}", e)))?;

    super::srt_kernels::cuda_gather_lucas_shadow_f64(
        &device,
        &mut out_dev,
        &src_dev,
        &idx_dev,
        indices.len(),
    )
    .map_err(|e| PyRuntimeError::new_err(e))?;

    let mut result = vec![0.0f64; indices.len()];
    device
        .default_stream()
        .memcpy_dtoh(&out_dev, &mut result)
        .map_err(|e| PyRuntimeError::new_err(format!("Failed to copy result: {}", e)))?;

    Ok(result)
}

/// Gather with Pisano hooked weighting (f64)
#[cfg(feature = "cuda")]
#[pyfunction]
#[pyo3(signature = (src, indices, device_idx=0))]
pub fn py_gather_pisano_hooked_f64(
    src: Vec<f64>,
    indices: Vec<i64>,
    device_idx: usize,
) -> PyResult<Vec<f64>> {
    if src.is_empty() || indices.is_empty() {
        return Err(PyRuntimeError::new_err("src and indices cannot be empty"));
    }

    let device = get_device(device_idx).map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    let pool = get_pool(device_idx).map_err(|e| PyRuntimeError::new_err(e.to_string()))?;

    let src_dev = device
        .default_stream()
        .clone_htod(&src)
        .map_err(|e| PyRuntimeError::new_err(format!("Failed to copy src: {}", e)))?;
    let idx_dev = device
        .default_stream()
        .clone_htod(&indices)
        .map_err(|e| PyRuntimeError::new_err(format!("Failed to copy indices: {}", e)))?;
    let mut out_dev: CudaSlice<f64> = pool
        .alloc_f64(indices.len())
        .map_err(|e| PyRuntimeError::new_err(format!("Failed to alloc output: {}", e)))?;

    super::srt_kernels::cuda_gather_pisano_hooked_f64(
        &device,
        &mut out_dev,
        &src_dev,
        &idx_dev,
        indices.len(),
    )
    .map_err(|e| PyRuntimeError::new_err(e))?;

    let mut result = vec![0.0f64; indices.len()];
    device
        .default_stream()
        .memcpy_dtoh(&out_dev, &mut result)
        .map_err(|e| PyRuntimeError::new_err(format!("Failed to copy result: {}", e)))?;

    Ok(result)
}

/// Gather with E8 roots weighting (f64)
#[cfg(feature = "cuda")]
#[pyfunction]
#[pyo3(signature = (src, indices, device_idx=0))]
pub fn py_gather_e8_roots_f64(
    src: Vec<f64>,
    indices: Vec<i64>,
    device_idx: usize,
) -> PyResult<Vec<f64>> {
    if src.is_empty() || indices.is_empty() {
        return Err(PyRuntimeError::new_err("src and indices cannot be empty"));
    }

    let device = get_device(device_idx).map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    let pool = get_pool(device_idx).map_err(|e| PyRuntimeError::new_err(e.to_string()))?;

    let src_dev = device
        .default_stream()
        .clone_htod(&src)
        .map_err(|e| PyRuntimeError::new_err(format!("Failed to copy src: {}", e)))?;
    let idx_dev = device
        .default_stream()
        .clone_htod(&indices)
        .map_err(|e| PyRuntimeError::new_err(format!("Failed to copy indices: {}", e)))?;
    let mut out_dev: CudaSlice<f64> = pool
        .alloc_f64(indices.len())
        .map_err(|e| PyRuntimeError::new_err(format!("Failed to alloc output: {}", e)))?;

    super::srt_kernels::cuda_gather_e8_roots_f64(
        &device,
        &mut out_dev,
        &src_dev,
        &idx_dev,
        indices.len(),
    )
    .map_err(|e| PyRuntimeError::new_err(e))?;

    let mut result = vec![0.0f64; indices.len()];
    device
        .default_stream()
        .memcpy_dtoh(&out_dev, &mut result)
        .map_err(|e| PyRuntimeError::new_err(format!("Failed to copy result: {}", e)))?;

    Ok(result)
}

/// Scatter with golden cone weighting (f64)
#[cfg(feature = "cuda")]
#[pyfunction]
#[pyo3(signature = (src, indices, output_size, device_idx=0))]
pub fn py_scatter_golden_cone_f64(
    src: Vec<f64>,
    indices: Vec<i64>,
    output_size: usize,
    device_idx: usize,
) -> PyResult<Vec<f64>> {
    if src.is_empty() || indices.is_empty() {
        return Err(PyRuntimeError::new_err("src and indices cannot be empty"));
    }
    if src.len() != indices.len() {
        return Err(PyRuntimeError::new_err(
            "src and indices must have same length",
        ));
    }

    let device = get_device(device_idx).map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    let pool = get_pool(device_idx).map_err(|e| PyRuntimeError::new_err(e.to_string()))?;

    let src_dev = device
        .default_stream()
        .clone_htod(&src)
        .map_err(|e| PyRuntimeError::new_err(format!("Failed to copy src: {}", e)))?;
    let idx_dev = device
        .default_stream()
        .clone_htod(&indices)
        .map_err(|e| PyRuntimeError::new_err(format!("Failed to copy indices: {}", e)))?;
    let mut out_dev: CudaSlice<f64> = pool
        .alloc_f64(output_size)
        .map_err(|e| PyRuntimeError::new_err(format!("Failed to alloc output: {}", e)))?;

    super::srt_kernels::cuda_scatter_golden_cone_f64(
        &device,
        &mut out_dev,
        &src_dev,
        &idx_dev,
        indices.len(),
    )
    .map_err(|e| PyRuntimeError::new_err(e))?;

    let mut result = vec![0.0f64; output_size];
    device
        .default_stream()
        .memcpy_dtoh(&out_dev, &mut result)
        .map_err(|e| PyRuntimeError::new_err(format!("Failed to copy result: {}", e)))?;

    Ok(result)
}

/// Gather with transcendence gate weighting (f64)
#[cfg(feature = "cuda")]
#[pyfunction]
#[pyo3(signature = (src, indices, device_idx=0))]
pub fn py_gather_transcendence_gate_f64(
    src: Vec<f64>,
    indices: Vec<i64>,
    device_idx: usize,
) -> PyResult<Vec<f64>> {
    if src.is_empty() || indices.is_empty() {
        return Err(PyRuntimeError::new_err("src and indices cannot be empty"));
    }

    let device = get_device(device_idx).map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    let pool = get_pool(device_idx).map_err(|e| PyRuntimeError::new_err(e.to_string()))?;

    let src_dev = device
        .default_stream()
        .clone_htod(&src)
        .map_err(|e| PyRuntimeError::new_err(format!("Failed to copy src: {}", e)))?;
    let idx_dev = device
        .default_stream()
        .clone_htod(&indices)
        .map_err(|e| PyRuntimeError::new_err(format!("Failed to copy indices: {}", e)))?;
    let mut out_dev: CudaSlice<f64> = pool
        .alloc_f64(indices.len())
        .map_err(|e| PyRuntimeError::new_err(format!("Failed to alloc output: {}", e)))?;

    super::srt_kernels::cuda_gather_transcendence_gate_f64(
        &device,
        &mut out_dev,
        &src_dev,
        &idx_dev,
        indices.len(),
    )
    .map_err(|e| PyRuntimeError::new_err(e))?;

    let mut result = vec![0.0f64; indices.len()];
    device
        .default_stream()
        .memcpy_dtoh(&out_dev, &mut result)
        .map_err(|e| PyRuntimeError::new_err(format!("Failed to copy result: {}", e)))?;

    Ok(result)
}

/// Scatter with consciousness threshold (f64)
#[cfg(feature = "cuda")]
#[pyfunction]
#[pyo3(signature = (src, indices, output_size, device_idx=0))]
pub fn py_scatter_consciousness_threshold_f64(
    src: Vec<f64>,
    indices: Vec<i64>,
    output_size: usize,
    device_idx: usize,
) -> PyResult<Vec<f64>> {
    if src.is_empty() || indices.is_empty() {
        return Err(PyRuntimeError::new_err("src and indices cannot be empty"));
    }
    if src.len() != indices.len() {
        return Err(PyRuntimeError::new_err(
            "src and indices must have same length",
        ));
    }

    let device = get_device(device_idx).map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    let pool = get_pool(device_idx).map_err(|e| PyRuntimeError::new_err(e.to_string()))?;

    let src_dev = device
        .default_stream()
        .clone_htod(&src)
        .map_err(|e| PyRuntimeError::new_err(format!("Failed to copy src: {}", e)))?;
    let idx_dev = device
        .default_stream()
        .clone_htod(&indices)
        .map_err(|e| PyRuntimeError::new_err(format!("Failed to copy indices: {}", e)))?;
    let mut out_dev: CudaSlice<f64> = pool
        .alloc_f64(output_size)
        .map_err(|e| PyRuntimeError::new_err(format!("Failed to alloc output: {}", e)))?;

    super::srt_kernels::cuda_scatter_consciousness_threshold_f64(
        &device,
        &mut out_dev,
        &src_dev,
        &idx_dev,
        indices.len(),
    )
    .map_err(|e| PyRuntimeError::new_err(e))?;

    let mut result = vec![0.0f64; output_size];
    device
        .default_stream()
        .memcpy_dtoh(&out_dev, &mut result)
        .map_err(|e| PyRuntimeError::new_err(format!("Failed to copy result: {}", e)))?;

    Ok(result)
}

// =============================================================================
// SRT Reduction Operations
// =============================================================================

/// Reduce sum of array elements
#[cfg(feature = "cuda")]
#[pyfunction]
#[pyo3(signature = (input, device_idx=0))]
pub fn py_reduce_sum_f64(input: Vec<f64>, device_idx: usize) -> PyResult<f64> {
    if input.is_empty() {
        return Err(PyRuntimeError::new_err("input cannot be empty"));
    }

    let device = get_device(device_idx).map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    let pool = get_pool(device_idx).map_err(|e| PyRuntimeError::new_err(e.to_string()))?;

    // Allocate CUDA memory
    let input_dev = device
        .default_stream()
        .clone_htod(&input)
        .map_err(|e| PyRuntimeError::new_err(format!("Failed to copy input: {}", e)))?;
    let mut output_dev: CudaSlice<f64> = pool
        .alloc_f64(1)
        .map_err(|e| PyRuntimeError::new_err(format!("Failed to alloc output: {}", e)))?;

    // Run reduction
    super::srt_kernels::cuda_reduce_sum_f64(&device, &mut output_dev, &input_dev, input.len())
        .map_err(|e| PyRuntimeError::new_err(e))?;

    // Copy result back
    let mut result = vec![0.0f64; 1];
    device
        .default_stream()
        .memcpy_dtoh(&output_dev, &mut result)
        .map_err(|e| PyRuntimeError::new_err(format!("Failed to copy result: {}", e)))?;

    Ok(result[0])
}

/// Reduce mean of array elements
#[cfg(feature = "cuda")]
#[pyfunction]
#[pyo3(signature = (input, device_idx=0))]
pub fn py_reduce_mean_f64(input: Vec<f64>, device_idx: usize) -> PyResult<f64> {
    if input.is_empty() {
        return Err(PyRuntimeError::new_err("input cannot be empty"));
    }

    let device = get_device(device_idx).map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    let pool = get_pool(device_idx).map_err(|e| PyRuntimeError::new_err(e.to_string()))?;

    // Allocate CUDA memory
    let input_dev = device
        .default_stream()
        .clone_htod(&input)
        .map_err(|e| PyRuntimeError::new_err(format!("Failed to copy input: {}", e)))?;
    let mut output_dev: CudaSlice<f64> = pool
        .alloc_f64(1)
        .map_err(|e| PyRuntimeError::new_err(format!("Failed to alloc output: {}", e)))?;

    // Run reduction
    super::srt_kernels::cuda_reduce_mean_f64(&device, &mut output_dev, &input_dev, input.len())
        .map_err(|e| PyRuntimeError::new_err(e))?;

    // Copy result back
    let mut result = vec![0.0f64; 1];
    device
        .default_stream()
        .memcpy_dtoh(&output_dev, &mut result)
        .map_err(|e| PyRuntimeError::new_err(format!("Failed to copy result: {}", e)))?;

    Ok(result[0])
}

/// Reduce mean of array elements (f32)
#[cfg(feature = "cuda")]
#[pyfunction]
#[pyo3(signature = (input, device_idx=0))]
pub fn py_reduce_mean_f32(input: Vec<f32>, device_idx: usize) -> PyResult<f32> {
    if input.is_empty() {
        return Err(PyRuntimeError::new_err("input cannot be empty"));
    }

    let device = get_device(device_idx).map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    let pool = get_pool(device_idx).map_err(|e| PyRuntimeError::new_err(e.to_string()))?;

    // Allocate CUDA memory
    let input_dev = device
        .default_stream()
        .clone_htod(&input)
        .map_err(|e| PyRuntimeError::new_err(format!("Failed to copy input: {}", e)))?;
    let mut output_dev: CudaSlice<f32> = pool
        .alloc_f32(1)
        .map_err(|e| PyRuntimeError::new_err(format!("Failed to alloc output: {}", e)))?;

    // Run reduction
    super::srt_kernels::cuda_reduce_mean_f32(&device, &mut output_dev, &input_dev, input.len())
        .map_err(|e| PyRuntimeError::new_err(e))?;

    // Copy result back
    let mut result = vec![0.0f32; 1];
    device
        .default_stream()
        .memcpy_dtoh(&output_dev, &mut result)
        .map_err(|e| PyRuntimeError::new_err(format!("Failed to copy result: {}", e)))?;

    Ok(result[0])
}

/// Reduce max of array elements
#[cfg(feature = "cuda")]
#[pyfunction]
#[pyo3(signature = (input, device_idx=0))]
pub fn py_reduce_max_f64(input: Vec<f64>, device_idx: usize) -> PyResult<f64> {
    if input.is_empty() {
        return Err(PyRuntimeError::new_err("input cannot be empty"));
    }

    let device = get_device(device_idx).map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    let pool = get_pool(device_idx).map_err(|e| PyRuntimeError::new_err(e.to_string()))?;

    // Allocate CUDA memory
    let input_dev = device
        .default_stream()
        .clone_htod(&input)
        .map_err(|e| PyRuntimeError::new_err(format!("Failed to copy input: {}", e)))?;
    let mut output_dev: CudaSlice<f64> = pool
        .alloc_f64(1)
        .map_err(|e| PyRuntimeError::new_err(format!("Failed to alloc output: {}", e)))?;

    // Run reduction
    super::srt_kernels::cuda_reduce_max_f64(&device, &mut output_dev, &input_dev, input.len())
        .map_err(|e| PyRuntimeError::new_err(e))?;

    // Copy result back
    let mut result = vec![0.0f64; 1];
    device
        .default_stream()
        .memcpy_dtoh(&output_dev, &mut result)
        .map_err(|e| PyRuntimeError::new_err(format!("Failed to copy result: {}", e)))?;

    Ok(result[0])
}

/// Reduce max of array elements (f32)
#[cfg(feature = "cuda")]
#[pyfunction]
#[pyo3(signature = (input, device_idx=0))]
pub fn py_reduce_max_f32(input: Vec<f32>, device_idx: usize) -> PyResult<f32> {
    if input.is_empty() {
        return Err(PyRuntimeError::new_err("input cannot be empty"));
    }

    let device = get_device(device_idx).map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    let pool = get_pool(device_idx).map_err(|e| PyRuntimeError::new_err(e.to_string()))?;

    // Allocate CUDA memory
    let input_dev = device
        .default_stream()
        .clone_htod(&input)
        .map_err(|e| PyRuntimeError::new_err(format!("Failed to copy input: {}", e)))?;
    let mut output_dev: CudaSlice<f32> = pool
        .alloc_f32(1)
        .map_err(|e| PyRuntimeError::new_err(format!("Failed to alloc output: {}", e)))?;

    // Run reduction
    super::srt_kernels::cuda_reduce_max_f32(&device, &mut output_dev, &input_dev, input.len())
        .map_err(|e| PyRuntimeError::new_err(e))?;

    // Copy result back
    let mut result = vec![0.0f32; 1];
    device
        .default_stream()
        .memcpy_dtoh(&output_dev, &mut result)
        .map_err(|e| PyRuntimeError::new_err(format!("Failed to copy result: {}", e)))?;

    Ok(result[0])
}

/// Reduce min of array elements
#[cfg(feature = "cuda")]
#[pyfunction]
#[pyo3(signature = (input, device_idx=0))]
pub fn py_reduce_min_f64(input: Vec<f64>, device_idx: usize) -> PyResult<f64> {
    if input.is_empty() {
        return Err(PyRuntimeError::new_err("input cannot be empty"));
    }

    let device = get_device(device_idx).map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    let pool = get_pool(device_idx).map_err(|e| PyRuntimeError::new_err(e.to_string()))?;

    // Allocate CUDA memory
    let input_dev = device
        .default_stream()
        .clone_htod(&input)
        .map_err(|e| PyRuntimeError::new_err(format!("Failed to copy input: {}", e)))?;
    let mut output_dev: CudaSlice<f64> = pool
        .alloc_f64(1)
        .map_err(|e| PyRuntimeError::new_err(format!("Failed to alloc output: {}", e)))?;

    // Run reduction
    super::srt_kernels::cuda_reduce_min_f64(&device, &mut output_dev, &input_dev, input.len())
        .map_err(|e| PyRuntimeError::new_err(e))?;

    // Copy result back
    let mut result = vec![0.0f64; 1];
    device
        .default_stream()
        .memcpy_dtoh(&output_dev, &mut result)
        .map_err(|e| PyRuntimeError::new_err(format!("Failed to copy result: {}", e)))?;

    Ok(result[0])
}

/// Reduce min of array elements (f32)
#[cfg(feature = "cuda")]
#[pyfunction]
#[pyo3(signature = (input, device_idx=0))]
pub fn py_reduce_min_f32(input: Vec<f32>, device_idx: usize) -> PyResult<f32> {
    if input.is_empty() {
        return Err(PyRuntimeError::new_err("input cannot be empty"));
    }

    let device = get_device(device_idx).map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    let pool = get_pool(device_idx).map_err(|e| PyRuntimeError::new_err(e.to_string()))?;

    // Allocate CUDA memory
    let input_dev = device
        .default_stream()
        .clone_htod(&input)
        .map_err(|e| PyRuntimeError::new_err(format!("Failed to copy input: {}", e)))?;
    let mut output_dev: CudaSlice<f32> = pool
        .alloc_f32(1)
        .map_err(|e| PyRuntimeError::new_err(format!("Failed to alloc output: {}", e)))?;

    // Run reduction
    super::srt_kernels::cuda_reduce_min_f32(&device, &mut output_dev, &input_dev, input.len())
        .map_err(|e| PyRuntimeError::new_err(e))?;

    // Copy result back
    let mut result = vec![0.0f32; 1];
    device
        .default_stream()
        .memcpy_dtoh(&output_dev, &mut result)
        .map_err(|e| PyRuntimeError::new_err(format!("Failed to copy result: {}", e)))?;

    Ok(result[0])
}

/// Reduce L2 norm of array elements (f64)
#[cfg(feature = "cuda")]
#[pyfunction]
#[pyo3(signature = (input, device_idx=0))]
pub fn py_reduce_norm_l2_f64(input: Vec<f64>, device_idx: usize) -> PyResult<f64> {
    if input.is_empty() {
        return Err(PyRuntimeError::new_err("input cannot be empty"));
    }

    let device = get_device(device_idx).map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    let pool = get_pool(device_idx).map_err(|e| PyRuntimeError::new_err(e.to_string()))?;

    // Allocate CUDA memory
    let input_dev = device
        .default_stream()
        .clone_htod(&input)
        .map_err(|e| PyRuntimeError::new_err(format!("Failed to copy input: {}", e)))?;
    let mut output_dev: CudaSlice<f64> = pool
        .alloc_f64(1)
        .map_err(|e| PyRuntimeError::new_err(format!("Failed to alloc output: {}", e)))?;

    // Run reduction
    super::srt_kernels::cuda_reduce_norm_l2_f64(&device, &mut output_dev, &input_dev, input.len())
        .map_err(|e| PyRuntimeError::new_err(e))?;

    // Copy result back
    let mut result = vec![0.0f64; 1];
    device
        .default_stream()
        .memcpy_dtoh(&output_dev, &mut result)
        .map_err(|e| PyRuntimeError::new_err(format!("Failed to copy result: {}", e)))?;

    Ok(result[0])
}

/// Reduce L2 norm of array elements (f32)
#[cfg(feature = "cuda")]
#[pyfunction]
#[pyo3(signature = (input, device_idx=0))]
pub fn py_reduce_norm_l2_f32(input: Vec<f32>, device_idx: usize) -> PyResult<f32> {
    if input.is_empty() {
        return Err(PyRuntimeError::new_err("input cannot be empty"));
    }

    let device = get_device(device_idx).map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    let pool = get_pool(device_idx).map_err(|e| PyRuntimeError::new_err(e.to_string()))?;

    // Allocate CUDA memory
    let input_dev = device
        .default_stream()
        .clone_htod(&input)
        .map_err(|e| PyRuntimeError::new_err(format!("Failed to copy input: {}", e)))?;
    let mut output_dev: CudaSlice<f32> = pool
        .alloc_f32(1)
        .map_err(|e| PyRuntimeError::new_err(format!("Failed to alloc output: {}", e)))?;

    // Run reduction
    super::srt_kernels::cuda_reduce_norm_l2_f32(&device, &mut output_dev, &input_dev, input.len())
        .map_err(|e| PyRuntimeError::new_err(e))?;

    // Copy result back
    let mut result = vec![0.0f32; 1];
    device
        .default_stream()
        .memcpy_dtoh(&output_dev, &mut result)
        .map_err(|e| PyRuntimeError::new_err(format!("Failed to copy result: {}", e)))?;

    Ok(result[0])
}

/// Reduce sum with golden weighted reduction (f64)
#[cfg(feature = "cuda")]
#[pyfunction]
#[pyo3(signature = (input, device_idx=0))]
pub fn py_reduce_sum_golden_weighted_f64(input: Vec<f64>, device_idx: usize) -> PyResult<f64> {
    if input.is_empty() {
        return Err(PyRuntimeError::new_err("input cannot be empty"));
    }

    let device = get_device(device_idx).map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    let pool = get_pool(device_idx).map_err(|e| PyRuntimeError::new_err(e.to_string()))?;

    // Allocate CUDA memory
    let input_dev = device
        .default_stream()
        .clone_htod(&input)
        .map_err(|e| PyRuntimeError::new_err(format!("Failed to copy input: {}", e)))?;
    let mut output_dev: CudaSlice<f64> = pool
        .alloc_f64(1)
        .map_err(|e| PyRuntimeError::new_err(format!("Failed to alloc output: {}", e)))?;

    // Run reduction
    super::srt_kernels::cuda_reduce_sum_golden_weighted_f64(
        &device,
        &mut output_dev,
        &input_dev,
        input.len(),
    )
    .map_err(|e| PyRuntimeError::new_err(e))?;

    // Copy result back
    let mut result = vec![0.0f64; 1];
    device
        .default_stream()
        .memcpy_dtoh(&output_dev, &mut result)
        .map_err(|e| PyRuntimeError::new_err(format!("Failed to copy result: {}", e)))?;

    Ok(result[0])
}

/// Reduce sum of array elements (f32)
#[cfg(feature = "cuda")]
#[pyfunction]
#[pyo3(signature = (input, device_idx=0))]
pub fn py_reduce_sum_f32(input: Vec<f32>, device_idx: usize) -> PyResult<f32> {
    if input.is_empty() {
        return Err(PyRuntimeError::new_err("input cannot be empty"));
    }

    let device = get_device(device_idx).map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    let pool = get_pool(device_idx).map_err(|e| PyRuntimeError::new_err(e.to_string()))?;

    // Allocate CUDA memory
    let input_dev = device
        .default_stream()
        .clone_htod(&input)
        .map_err(|e| PyRuntimeError::new_err(format!("Failed to copy input: {}", e)))?;
    let mut output_dev: CudaSlice<f32> = pool
        .alloc_f32(1)
        .map_err(|e| PyRuntimeError::new_err(format!("Failed to alloc output: {}", e)))?;

    // Run reduction
    super::srt_kernels::cuda_reduce_sum_f32(&device, &mut output_dev, &input_dev, input.len())
        .map_err(|e| PyRuntimeError::new_err(e))?;

    // Copy result back
    let mut result = vec![0.0f32; 1];
    device
        .default_stream()
        .memcpy_dtoh(&output_dev, &mut result)
        .map_err(|e| PyRuntimeError::new_err(format!("Failed to copy result: {}", e)))?;

    Ok(result[0])
}

// CPU fallbacks for non-CUDA builds
#[cfg(not(feature = "cuda"))]
#[pyfunction]
#[pyo3(signature = (src, indices, device_idx=0))]
pub fn py_gather_f64(_src: Vec<f64>, _indices: Vec<i64>, _device_idx: usize) -> PyResult<Vec<f64>> {
    Err(PyRuntimeError::new_err(
        "CUDA not available - compile with cuda feature",
    ))
}

#[cfg(not(feature = "cuda"))]
#[pyfunction]
#[pyo3(signature = (src, indices, output_size, device_idx=0))]
pub fn py_scatter_f64(
    _src: Vec<f64>,
    _indices: Vec<i64>,
    _output_size: usize,
    _device_idx: usize,
) -> PyResult<Vec<f64>> {
    Err(PyRuntimeError::new_err(
        "CUDA not available - compile with cuda feature",
    ))
}

#[cfg(not(feature = "cuda"))]
#[pyfunction]
#[pyo3(signature = (src, indices, output_size, device_idx=0))]
pub fn py_scatter_add_f64(
    _src: Vec<f64>,
    _indices: Vec<i64>,
    _output_size: usize,
    _device_idx: usize,
) -> PyResult<Vec<f64>> {
    Err(PyRuntimeError::new_err(
        "CUDA not available - compile with cuda feature",
    ))
}

#[cfg(not(feature = "cuda"))]
#[pyfunction]
#[pyo3(signature = (input, device_idx=0))]
pub fn py_reduce_min_f64(_input: Vec<f64>, _device_idx: usize) -> PyResult<f64> {
    Err(PyRuntimeError::new_err(
        "CUDA not available - compile with cuda feature",
    ))
}

#[cfg(not(feature = "cuda"))]
#[pyfunction]
#[pyo3(signature = (input, device_idx=0))]
pub fn py_reduce_min_f32(_input: Vec<f32>, _device_idx: usize) -> PyResult<f32> {
    Err(PyRuntimeError::new_err(
        "CUDA not available - compile with cuda feature",
    ))
}

#[cfg(not(feature = "cuda"))]
#[pyfunction]
#[pyo3(signature = (input, device_idx=0))]
pub fn py_reduce_sum_f32(_input: Vec<f32>, _device_idx: usize) -> PyResult<f32> {
    Err(PyRuntimeError::new_err(
        "CUDA not available - compile with cuda feature",
    ))
}

#[cfg(not(feature = "cuda"))]
#[pyfunction]
#[pyo3(signature = (input, device_idx=0))]
pub fn py_reduce_sum_f64(_input: Vec<f64>, _device_idx: usize) -> PyResult<f64> {
    Err(PyRuntimeError::new_err(
        "CUDA not available - compile with cuda feature",
    ))
}

#[cfg(not(feature = "cuda"))]
#[pyfunction]
#[pyo3(signature = (input, device_idx=0))]
pub fn py_reduce_mean_f64(_input: Vec<f64>, _device_idx: usize) -> PyResult<f64> {
    Err(PyRuntimeError::new_err(
        "CUDA not available - compile with cuda feature",
    ))
}

#[cfg(not(feature = "cuda"))]
#[pyfunction]
#[pyo3(signature = (input, device_idx=0))]
pub fn py_reduce_mean_f32(_input: Vec<f32>, _device_idx: usize) -> PyResult<f32> {
    Err(PyRuntimeError::new_err(
        "CUDA not available - compile with cuda feature",
    ))
}

#[cfg(not(feature = "cuda"))]
#[pyfunction]
#[pyo3(signature = (input, device_idx=0))]
pub fn py_reduce_max_f64(_input: Vec<f64>, _device_idx: usize) -> PyResult<f64> {
    Err(PyRuntimeError::new_err(
        "CUDA not available - compile with cuda feature",
    ))
}

#[cfg(not(feature = "cuda"))]
#[pyfunction]
#[pyo3(signature = (input, device_idx=0))]
pub fn py_reduce_max_f32(_input: Vec<f32>, _device_idx: usize) -> PyResult<f32> {
    Err(PyRuntimeError::new_err(
        "CUDA not available - compile with cuda feature",
    ))
}

#[cfg(not(feature = "cuda"))]
#[pyfunction]
#[pyo3(signature = (input, device_idx=0))]
pub fn py_reduce_min_f64(_input: Vec<f64>, _device_idx: usize) -> PyResult<f64> {
    Err(PyRuntimeError::new_err(
        "CUDA not available - compile with cuda feature",
    ))
}

#[cfg(not(feature = "cuda"))]
#[pyfunction]
#[pyo3(signature = (input, device_idx=0))]
pub fn py_reduce_norm_l2_f64(_input: Vec<f64>, _device_idx: usize) -> PyResult<f64> {
    Err(PyRuntimeError::new_err(
        "CUDA not available - compile with cuda feature",
    ))
}

#[cfg(not(feature = "cuda"))]
#[pyfunction]
#[pyo3(signature = (input, device_idx=0))]
pub fn py_reduce_norm_l2_f32(_input: Vec<f32>, _device_idx: usize) -> PyResult<f32> {
    Err(PyRuntimeError::new_err(
        "CUDA not available - compile with cuda feature",
    ))
}

#[cfg(not(feature = "cuda"))]
#[pyfunction]
#[pyo3(signature = (input, device_idx=0))]
pub fn py_reduce_sum_golden_weighted_f64(_input: Vec<f64>, _device_idx: usize) -> PyResult<f64> {
    Err(PyRuntimeError::new_err(
        "CUDA not available - compile with cuda feature",
    ))
}

// =============================================================================
// SRT Scatter/Gather Operations
// =============================================================================

/// Gather elements from source array using indices
#[cfg(feature = "cuda")]
#[pyfunction]
#[pyo3(signature = (src, indices, device_idx=0))]
pub fn py_gather_f64(src: Vec<f64>, indices: Vec<i64>, device_idx: usize) -> PyResult<Vec<f64>> {
    Err(PyRuntimeError::new_err(
        "CUDA not available - compile with cuda feature",
    ))
}

// =============================================================================
// Kernel Loader Functions (Python wrappers)
// =============================================================================

#[cfg(feature = "cuda")]
#[pyfunction]
pub fn py_load_wmma_syntonic_kernels(device_idx: usize) -> PyResult<Vec<String>> {
    use super::cuda::device_manager::get_device;
    use super::srt_kernels;

    let device = get_device(device_idx)
        .map_err(|e| PyRuntimeError::new_err(format!("Failed to get device: {}", e)))?;

    srt_kernels::load_wmma_syntonic_kernels(&device)
        .map_err(|e| PyRuntimeError::new_err(format!("Failed to load kernels: {}", e)))
        .map(|funcs| funcs.keys().cloned().collect())
}

#[cfg(feature = "cuda")]
#[pyfunction]
pub fn py_load_scatter_gather_kernels(device_idx: usize) -> PyResult<Vec<String>> {
    use super::cuda::device_manager::get_device;
    use super::srt_kernels;

    let device = get_device(device_idx)
        .map_err(|e| PyRuntimeError::new_err(format!("Failed to get device: {}", e)))?;

    srt_kernels::load_scatter_gather_kernels(&device)
        .map_err(|e| PyRuntimeError::new_err(format!("Failed to load kernels: {}", e)))
        .map(|funcs| funcs.keys().cloned().collect())
}

#[cfg(feature = "cuda")]
#[pyfunction]
pub fn py_load_reduction_kernels(device_idx: usize) -> PyResult<Vec<String>> {
    use super::cuda::device_manager::get_device;
    use super::srt_kernels;

    let device = get_device(device_idx)
        .map_err(|e| PyRuntimeError::new_err(format!("Failed to get device: {}", e)))?;

    srt_kernels::load_reduction_kernels(&device)
        .map_err(|e| PyRuntimeError::new_err(format!("Failed to load kernels: {}", e)))
        .map(|funcs| funcs.keys().cloned().collect())
}

#[cfg(feature = "cuda")]
#[pyfunction]
pub fn py_load_trilinear_kernels(device_idx: usize) -> PyResult<Vec<String>> {
    use super::cuda::device_manager::get_device;
    use super::srt_kernels;

    let device = get_device(device_idx)
        .map_err(|e| PyRuntimeError::new_err(format!("Failed to get device: {}", e)))?;

    srt_kernels::load_trilinear_kernels(&device)
        .map_err(|e| PyRuntimeError::new_err(format!("Failed to load kernels: {}", e)))
        .map(|funcs| funcs.keys().cloned().collect())
}

#[cfg(feature = "cuda")]
#[pyfunction]
pub fn py_load_complex_ops_kernels(device_idx: usize) -> PyResult<Vec<String>> {
    use super::cuda::device_manager::get_device;
    use super::srt_kernels;

    let device = get_device(device_idx)
        .map_err(|e| PyRuntimeError::new_err(format!("Failed to get device: {}", e)))?;

    srt_kernels::load_complex_ops_kernels(&device)
        .map_err(|e| PyRuntimeError::new_err(format!("Failed to load kernels: {}", e)))
        .map(|funcs| funcs.keys().cloned().collect())
}

#[cfg(feature = "cuda")]
#[pyfunction]
pub fn py_load_attention_kernels(device_idx: usize) -> PyResult<Vec<String>> {
    use super::cuda::device_manager::get_device;
    use super::srt_kernels;

    let device = get_device(device_idx)
        .map_err(|e| PyRuntimeError::new_err(format!("Failed to get device: {}", e)))?;

    srt_kernels::load_attention_kernels(&device)
        .map_err(|e| PyRuntimeError::new_err(format!("Failed to load kernels: {}", e)))
        .map(|funcs| funcs.keys().cloned().collect())
}
