//! SRT-specific CUDA kernels for Syntonic Resonance Theory computations.
//!
//! This module provides GPU-accelerated operations for:
//! - Golden ratio operations (φ scaling, gaussian weights)
//! - E₈ lattice projections (P_φ, P_⊥, quadratic forms)
//! - Heat kernel / theta series summation
//! - DHSR cycle operations (differentiation, harmonization, syntony)
//! - SRT correction factors (1 ± q/N)

#[cfg(feature = "cuda")]
use cudarc::driver::{CudaDevice, CudaSlice, LaunchConfig, LaunchAsync, DevicePtr};
#[cfg(feature = "cuda")]
use std::sync::Arc;
#[cfg(feature = "cuda")]
use pyo3::prelude::*;

// =============================================================================
// PTX Kernel Sources (Pre-compiled)
// =============================================================================

#[cfg(feature = "cuda")]
const PTX_GOLDEN_SM75: &str = include_str!("../../kernels/ptx/golden_ops_sm75.ptx");
#[cfg(feature = "cuda")]
const PTX_GOLDEN_SM80: &str = include_str!("../../kernels/ptx/golden_ops_sm80.ptx");
#[cfg(feature = "cuda")]
const PTX_GOLDEN_SM86: &str = include_str!("../../kernels/ptx/golden_ops_sm86.ptx");
#[cfg(feature = "cuda")]
const PTX_GOLDEN_SM90: &str = include_str!("../../kernels/ptx/golden_ops_sm90.ptx");

#[cfg(feature = "cuda")]
const PTX_E8_SM75: &str = include_str!("../../kernels/ptx/e8_projection_sm75.ptx");
#[cfg(feature = "cuda")]
const PTX_E8_SM80: &str = include_str!("../../kernels/ptx/e8_projection_sm80.ptx");
#[cfg(feature = "cuda")]
const PTX_E8_SM86: &str = include_str!("../../kernels/ptx/e8_projection_sm86.ptx");
#[cfg(feature = "cuda")]
const PTX_E8_SM90: &str = include_str!("../../kernels/ptx/e8_projection_sm90.ptx");

#[cfg(feature = "cuda")]
const PTX_HEAT_SM75: &str = include_str!("../../kernels/ptx/heat_kernel_sm75.ptx");
#[cfg(feature = "cuda")]
const PTX_HEAT_SM80: &str = include_str!("../../kernels/ptx/heat_kernel_sm80.ptx");
#[cfg(feature = "cuda")]
const PTX_HEAT_SM86: &str = include_str!("../../kernels/ptx/heat_kernel_sm86.ptx");
#[cfg(feature = "cuda")]
const PTX_HEAT_SM90: &str = include_str!("../../kernels/ptx/heat_kernel_sm90.ptx");

#[cfg(feature = "cuda")]
const PTX_DHSR_SM75: &str = include_str!("../../kernels/ptx/dhsr_sm75.ptx");
#[cfg(feature = "cuda")]
const PTX_DHSR_SM80: &str = include_str!("../../kernels/ptx/dhsr_sm80.ptx");
#[cfg(feature = "cuda")]
const PTX_DHSR_SM86: &str = include_str!("../../kernels/ptx/dhsr_sm86.ptx");
#[cfg(feature = "cuda")]
const PTX_DHSR_SM90: &str = include_str!("../../kernels/ptx/dhsr_sm90.ptx");

#[cfg(feature = "cuda")]
const PTX_CORR_SM75: &str = include_str!("../../kernels/ptx/corrections_sm75.ptx");
#[cfg(feature = "cuda")]
const PTX_CORR_SM80: &str = include_str!("../../kernels/ptx/corrections_sm80.ptx");
#[cfg(feature = "cuda")]
const PTX_CORR_SM86: &str = include_str!("../../kernels/ptx/corrections_sm86.ptx");
#[cfg(feature = "cuda")]
const PTX_CORR_SM90: &str = include_str!("../../kernels/ptx/corrections_sm90.ptx");

#[cfg(feature = "cuda")]
const PTX_RESONANT_SM75: &str = include_str!("../../kernels/ptx/resonant_d_sm75.ptx");
#[cfg(feature = "cuda")]
const PTX_RESONANT_SM80: &str = include_str!("../../kernels/ptx/resonant_d_sm80.ptx");
#[cfg(feature = "cuda")]
const PTX_RESONANT_SM86: &str = include_str!("../../kernels/ptx/resonant_d_sm86.ptx");
#[cfg(feature = "cuda")]
const PTX_RESONANT_SM90: &str = include_str!("../../kernels/ptx/resonant_d_sm90.ptx");

// =============================================================================
// Kernel Function Lists
// =============================================================================

/// Golden operations kernel functions
#[cfg(feature = "cuda")]
const GOLDEN_FUNCS: &[&str] = &[
    "scale_phi_f64", "scale_phi_f32",
    "scale_phi_inv_f64", "scale_phi_inv_f32",
    "fma_phi_kernel_f64", "fma_phi_kernel_f32",
    "fma_phi_inv_kernel_f64", "fma_phi_inv_kernel_f32",
    "golden_gaussian_weight_scalar_f64", "golden_gaussian_weight_scalar_f32",
    "golden_gaussian_weight_4d_int", "golden_gaussian_weight_4d_f32",
    "golden_gaussian_weight_8d_f32", "golden_gaussian_weight_8d_f64",
    "golden_recursion_4d_int", "golden_recursion_f32", "golden_recursion_f64",
    "golden_recursion_inv_4d_int",
    "fibonacci_binet_f64", "lucas_binet_f64",
    "compute_generation_4d_int",
    "weighted_inner_product_golden_f32",
    "golden_normalize_f32", "golden_norm_factor_f32",
    "scale_phi_c128", "golden_gaussian_weight_c128",
];

/// E₈ projection kernel functions
#[cfg(feature = "cuda")]
const E8_FUNCS: &[&str] = &[
    "project_parallel_f32", "project_parallel_f64",
    "project_perpendicular_f32", "project_perpendicular_f64",
    "quadratic_form_f32", "quadratic_form_f64",
    "golden_cone_test_f32", "golden_cone_test_f64",
    "e8_batch_projection_f32", "e8_batch_projection_f64",
    "count_cone_roots",
    "norm_squared_8d_f32", "norm_squared_8d_f64",
    "weighted_e8_contribution_f32", "weighted_e8_contribution_f64",
];

/// Heat kernel / theta series functions
#[cfg(feature = "cuda")]
const HEAT_FUNCS: &[&str] = &[
    "theta_series_sum_f32", "theta_series_sum_f64",
    "heat_kernel_e8_f32", "heat_kernel_e8_f64",
    "theta_series_shells_f32",
    "golden_weighted_sum_f32", "golden_weighted_sum_f64",
    "modular_inversion_f32", "modular_inversion_f64",
    "spectral_zeta_f64",
    "winding_heat_kernel_f32",
    "knot_contribution_f32",
];

/// DHSR cycle functions
#[cfg(feature = "cuda")]
const DHSR_FUNCS: &[&str] = &[
    "compute_syntony_f32", "compute_syntony_c128",
    "differentiation_f32", "differentiation_c128",
    "harmonization_f32", "harmonization_c128",
    "dhsr_cycle_f32", "dhsr_cycle_c128",
    "dhsr_cycle_inplace_f32", "dhsr_cycle_inplace_c128",
    "compute_gnosis_f32",
    "verify_dh_partition_f32",
    "dhsr_multi_cycle_c128",
];

/// Correction factor functions
#[cfg(feature = "cuda")]
const CORR_FUNCS: &[&str] = &[
    "apply_correction_f64", "apply_correction_f32",
    "apply_corrections_batch_f64", "apply_corrections_batch_f32",
    "compound_correction_f64", "compound_correction_f32",
    "lepton_mass_correction_f64",
    "quark_mass_correction_f64",
    "coupling_correction_f64",
    "custom_correction_f64", "custom_correction_f32",
    "higgs_mass_correction_f64",
    "mixing_matrix_correction_f64",
    "compute_correction_factors_f64", "compute_correction_factors_f32",
    "get_q_deficit", "get_structure_dimension",
];

/// Resonant D-phase functions
#[cfg(feature = "cuda")]
const RESONANT_FUNCS: &[&str] = &[
    "resonant_d_phase_f64", "resonant_d_phase_f32",
    "resonant_d_phase_batch_f64",
    "resonant_compute_syntony_f64", "resonant_compute_syntony_f32",
    "resonant_snap_gradient_f64",
    "resonant_argmax_syntony_f64",
    "resonant_box_muller_f64", "resonant_box_muller_f32",
    "resonant_residual_modulated_noise_f64",
    "resonant_compute_dwell_f64",
];

// =============================================================================
// PTX Selection Based on Compute Capability
// =============================================================================

#[cfg(feature = "cuda")]
fn select_golden_ptx(major: i32, minor: i32) -> &'static str {
    let cc = major * 10 + minor;
    if cc >= 90 { PTX_GOLDEN_SM90 }
    else if cc >= 86 { PTX_GOLDEN_SM86 }
    else if cc >= 80 { PTX_GOLDEN_SM80 }
    else { PTX_GOLDEN_SM75 }
}

#[cfg(feature = "cuda")]
fn select_e8_ptx(major: i32, minor: i32) -> &'static str {
    let cc = major * 10 + minor;
    if cc >= 90 { PTX_E8_SM90 }
    else if cc >= 86 { PTX_E8_SM86 }
    else if cc >= 80 { PTX_E8_SM80 }
    else { PTX_E8_SM75 }
}

#[cfg(feature = "cuda")]
fn select_heat_ptx(major: i32, minor: i32) -> &'static str {
    let cc = major * 10 + minor;
    if cc >= 90 { PTX_HEAT_SM90 }
    else if cc >= 86 { PTX_HEAT_SM86 }
    else if cc >= 80 { PTX_HEAT_SM80 }
    else { PTX_HEAT_SM75 }
}

#[cfg(feature = "cuda")]
fn select_dhsr_ptx(major: i32, minor: i32) -> &'static str {
    let cc = major * 10 + minor;
    if cc >= 90 { PTX_DHSR_SM90 }
    else if cc >= 86 { PTX_DHSR_SM86 }
    else if cc >= 80 { PTX_DHSR_SM80 }
    else { PTX_DHSR_SM75 }
}

#[cfg(feature = "cuda")]
fn select_corr_ptx(major: i32, minor: i32) -> &'static str {
    let cc = major * 10 + minor;
    if cc >= 90 { PTX_CORR_SM90 }
    else if cc >= 86 { PTX_CORR_SM86 }
    else if cc >= 80 { PTX_CORR_SM80 }
    else { PTX_CORR_SM75 }
}

#[cfg(feature = "cuda")]
fn select_resonant_ptx(major: i32, minor: i32) -> &'static str {
    let cc = major * 10 + minor;
    if cc >= 90 { PTX_RESONANT_SM90 }
    else if cc >= 86 { PTX_RESONANT_SM86 }
    else if cc >= 80 { PTX_RESONANT_SM80 }
    else { PTX_RESONANT_SM75 }
}

// =============================================================================
// Kernel Loading
// =============================================================================

/// Get compute capability from device
#[cfg(feature = "cuda")]
fn get_compute_capability(device: &Arc<CudaDevice>) -> (i32, i32) {
    use cudarc::driver::sys::CUdevice_attribute_enum;
    use cudarc::driver::result;

    let ordinal = device.ordinal() as i32;

    let major = unsafe {
        result::device::get_attribute(ordinal, CUdevice_attribute_enum::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR)
            .unwrap_or(7)
    };
    let minor = unsafe {
        result::device::get_attribute(ordinal, CUdevice_attribute_enum::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR)
            .unwrap_or(0)
    };

    (major, minor)
}

/// Ensure all SRT kernels are loaded for the given device
#[cfg(feature = "cuda")]
pub fn ensure_srt_kernels_loaded(device: &Arc<CudaDevice>) -> PyResult<()> {
    // Check if already loaded by testing for a characteristic function
    if device.get_func("srt_golden", "scale_phi_f64").is_some() {
        return Ok(());
    }

    let (major, minor) = get_compute_capability(device);

    // Load golden operations
    device.load_ptx(
        cudarc::nvrtc::Ptx::from_src(select_golden_ptx(major, minor)),
        "srt_golden",
        GOLDEN_FUNCS,
    ).map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(
        format!("Failed to load golden_ops kernels: {}", e)
    ))?;

    // Load E₈ projection operations
    device.load_ptx(
        cudarc::nvrtc::Ptx::from_src(select_e8_ptx(major, minor)),
        "srt_e8",
        E8_FUNCS,
    ).map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(
        format!("Failed to load e8_projection kernels: {}", e)
    ))?;

    // Load heat kernel operations
    device.load_ptx(
        cudarc::nvrtc::Ptx::from_src(select_heat_ptx(major, minor)),
        "srt_heat",
        HEAT_FUNCS,
    ).map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(
        format!("Failed to load heat_kernel kernels: {}", e)
    ))?;

    // Load DHSR cycle operations
    device.load_ptx(
        cudarc::nvrtc::Ptx::from_src(select_dhsr_ptx(major, minor)),
        "srt_dhsr",
        DHSR_FUNCS,
    ).map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(
        format!("Failed to load dhsr kernels: {}", e)
    ))?;

    // Load correction factor operations
    device.load_ptx(
        cudarc::nvrtc::Ptx::from_src(select_corr_ptx(major, minor)),
        "srt_corr",
        CORR_FUNCS,
    ).map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(
        format!("Failed to load corrections kernels: {}", e)
    ))?;

    // Load resonant D-phase operations
    device.load_ptx(
        cudarc::nvrtc::Ptx::from_src(select_resonant_ptx(major, minor)),
        "srt_resonant",
        RESONANT_FUNCS,
    ).map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(
        format!("Failed to load resonant_d kernels: {}", e)
    ))?;

    Ok(())
}

// =============================================================================
// SRT Constants
// =============================================================================

/// Golden ratio φ
pub const PHI: f64 = 1.6180339887498948482;
/// Golden ratio inverse φ⁻¹
pub const PHI_INV: f64 = 0.6180339887498948482;
/// q-deficit value
pub const Q_DEFICIT: f64 = 0.027395146920;

/// Structure indices for correction factors
pub mod structure {
    pub const E8_DIM: i32 = 0;       // 248
    pub const E8_ROOTS: i32 = 1;     // 240
    pub const E8_POS: i32 = 2;       // 120
    pub const E6_DIM: i32 = 3;       // 78
    pub const E6_CONE: i32 = 4;      // 36
    pub const E6_27: i32 = 5;        // 27
    pub const D4_KISSING: i32 = 6;   // 24
    pub const G2_DIM: i32 = 7;       // 14
}

// =============================================================================
// Launch Configuration Helpers
// =============================================================================

/// Standard launch configuration for n elements (block size 256)
#[cfg(feature = "cuda")]
pub fn launch_cfg_256(n: usize) -> LaunchConfig {
    let block_size = 256u32;
    let grid_size = ((n as u32) + block_size - 1) / block_size;
    LaunchConfig {
        block_dim: (block_size, 1, 1),
        grid_dim: (grid_size, 1, 1),
        shared_mem_bytes: 0,
    }
}

/// Launch configuration for E₈ operations (block size 240)
#[cfg(feature = "cuda")]
pub fn launch_cfg_e8(n: usize) -> LaunchConfig {
    let block_size = 240u32;
    let grid_size = ((n as u32) + block_size - 1) / block_size;
    LaunchConfig {
        block_dim: (block_size, 1, 1),
        grid_dim: (grid_size, 1, 1),
        shared_mem_bytes: 0,
    }
}

/// Launch configuration with shared memory for reductions
#[cfg(feature = "cuda")]
pub fn launch_cfg_reduce(n: usize, elem_size: usize) -> LaunchConfig {
    let block_size = 256u32;
    let grid_size = ((n as u32) + block_size - 1) / block_size;
    LaunchConfig {
        block_dim: (block_size, 1, 1),
        grid_dim: (grid_size, 1, 1),
        shared_mem_bytes: (block_size as u32) * (elem_size as u32),
    }
}

// =============================================================================
// High-Level Operation Wrappers
// =============================================================================

/// Scale array by golden ratio φ
#[cfg(feature = "cuda")]
pub fn cuda_scale_phi_f64(
    device: &Arc<CudaDevice>,
    input: &CudaSlice<f64>,
    output: &mut CudaSlice<f64>,
    n: usize,
) -> PyResult<()> {
    ensure_srt_kernels_loaded(device)?;
    let func = device.get_func("srt_golden", "scale_phi_f64")
        .ok_or_else(|| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>("Kernel not found"))?;

    unsafe {
        func.launch(launch_cfg_256(n), (output, input, n as i32))
    }.map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;

    Ok(())
}

/// Compute golden gaussian weights for 8D vectors: w(λ) = exp(-|λ|²/φ)
#[cfg(feature = "cuda")]
pub fn cuda_golden_gaussian_8d_f64(
    device: &Arc<CudaDevice>,
    vectors: &CudaSlice<f64>,  // count × 8 flattened
    weights: &mut CudaSlice<f64>,
    count: usize,
) -> PyResult<()> {
    ensure_srt_kernels_loaded(device)?;
    let func = device.get_func("srt_golden", "golden_gaussian_weight_8d_f64")
        .ok_or_else(|| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>("Kernel not found"))?;

    unsafe {
        func.launch(launch_cfg_256(count), (weights, vectors, count as i32))
    }.map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;

    Ok(())
}

/// Batch E₈ projection with Q values and cone test
#[cfg(feature = "cuda")]
pub fn cuda_e8_batch_projection_f64(
    device: &Arc<CudaDevice>,
    roots: &CudaSlice<f64>,           // count × 8
    proj_parallel: &mut CudaSlice<f64>,  // count × 4
    proj_perp: &mut CudaSlice<f64>,      // count × 4
    q_values: &mut CudaSlice<f64>,       // count
    in_cone: &mut CudaSlice<i32>,        // count
    count: usize,
) -> PyResult<()> {
    ensure_srt_kernels_loaded(device)?;
    let func = device.get_func("srt_e8", "e8_batch_projection_f64")
        .ok_or_else(|| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>("Kernel not found"))?;

    unsafe {
        func.launch(
            launch_cfg_e8(count),
            (proj_parallel, proj_perp, q_values, in_cone, roots, count as i32)
        )
    }.map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;

    Ok(())
}

/// Compute theta series sum: Θ(t) = Σ_λ w(λ) exp(-π Q(λ) / t)
#[cfg(feature = "cuda")]
pub fn cuda_theta_series_f64(
    device: &Arc<CudaDevice>,
    q_values: &CudaSlice<f64>,
    in_cone: &CudaSlice<i32>,
    weights: Option<&CudaSlice<f64>>,
    t: f64,
    count: usize,
) -> PyResult<f64> {
    ensure_srt_kernels_loaded(device)?;

    // Allocate result on device
    let mut result: CudaSlice<f64> = device.alloc_zeros(1)
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;

    let func = device.get_func("srt_heat", "theta_series_sum_f64")
        .ok_or_else(|| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>("Kernel not found"))?;

    let cfg = launch_cfg_reduce(count, std::mem::size_of::<f64>());

    // Handle optional weights pointer
    let weights_ptr: u64 = match weights {
        Some(w) => *w.device_ptr() as u64,
        None => 0u64,  // null pointer
    };

    unsafe {
        func.launch(cfg, (&mut result, q_values, in_cone, weights_ptr, t, count as i32))
    }.map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;

    // Copy result back
    let mut host_result = [0.0f64];
    device.dtoh_sync_copy_into(&result, &mut host_result)
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;

    Ok(host_result[0])
}

/// Compute syntony metric S(ψ)
#[cfg(feature = "cuda")]
pub fn cuda_compute_syntony_c128(
    device: &Arc<CudaDevice>,
    psi: &CudaSlice<f64>,           // Interleaved complex [re, im, ...]
    mode_norm_sq: &CudaSlice<f64>,  // |n|² for each mode
    n: usize,
) -> PyResult<f64> {
    ensure_srt_kernels_loaded(device)?;

    let mut numerator: CudaSlice<f64> = device.alloc_zeros(1)
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
    let mut denominator: CudaSlice<f64> = device.alloc_zeros(1)
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;

    let func = device.get_func("srt_dhsr", "compute_syntony_c128")
        .ok_or_else(|| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>("Kernel not found"))?;

    let cfg = launch_cfg_reduce(n, 2 * std::mem::size_of::<f64>());

    unsafe {
        func.launch(cfg, (&mut numerator, &mut denominator, psi, mode_norm_sq, n as i32))
    }.map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;

    let mut host_num = [0.0f64];
    let mut host_den = [0.0f64];
    device.dtoh_sync_copy_into(&numerator, &mut host_num)
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
    device.dtoh_sync_copy_into(&denominator, &mut host_den)
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;

    if host_den[0] < 1e-15 {
        Ok(0.0)
    } else {
        Ok(host_num[0] / host_den[0])
    }
}

/// Apply DHSR cycle in-place
#[cfg(feature = "cuda")]
pub fn cuda_dhsr_cycle_inplace_c128(
    device: &Arc<CudaDevice>,
    psi: &mut CudaSlice<f64>,       // In/out: interleaved complex
    mode_norm_sq: &CudaSlice<f64>,
    syntony: f64,
    n: usize,
) -> PyResult<f64> {
    ensure_srt_kernels_loaded(device)?;

    let mut new_num: CudaSlice<f64> = device.alloc_zeros(1)
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
    let mut new_den: CudaSlice<f64> = device.alloc_zeros(1)
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;

    let func = device.get_func("srt_dhsr", "dhsr_cycle_inplace_c128")
        .ok_or_else(|| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>("Kernel not found"))?;

    let cfg = launch_cfg_reduce(n, 2 * std::mem::size_of::<f64>());

    unsafe {
        func.launch(cfg, (psi, mode_norm_sq, syntony, &mut new_num, &mut new_den, n as i32))
    }.map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;

    let mut host_num = [0.0f64];
    let mut host_den = [0.0f64];
    device.dtoh_sync_copy_into(&new_num, &mut host_num)
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
    device.dtoh_sync_copy_into(&new_den, &mut host_den)
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;

    if host_den[0] < 1e-15 {
        Ok(syntony)
    } else {
        Ok(host_num[0] / host_den[0])
    }
}

/// Apply SRT correction factor: value × (1 + sign × q / N)
/// structure_idx should be one of the structure:: constants:
///   E8_DIM(0), E8_ROOTS(1), E8_POS(2), E6_DIM(3), E6_CONE(4), E6_27(5), D4_KISSING(6), G2_DIM(7)
#[cfg(feature = "cuda")]
pub fn cuda_apply_correction_f64(
    device: &Arc<CudaDevice>,
    input: &CudaSlice<f64>,
    output: &mut CudaSlice<f64>,
    structure_idx: i32,
    sign: i32,
    n: usize,
) -> PyResult<()> {
    // Validate structure index using structure constants
    let valid_idx = matches!(
        structure_idx,
        structure::E8_DIM | structure::E8_ROOTS | structure::E8_POS |
        structure::E6_DIM | structure::E6_CONE | structure::E6_27 |
        structure::D4_KISSING | structure::G2_DIM
    );
    if !valid_idx {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            format!("Invalid structure index {}. Use structure::E8_DIM, E8_ROOTS, etc.", structure_idx)
        ));
    }

    ensure_srt_kernels_loaded(device)?;

    let func = device.get_func("srt_corr", "apply_correction_f64")
        .ok_or_else(|| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>("Kernel not found"))?;

    unsafe {
        func.launch(launch_cfg_256(n), (output, input, structure_idx, sign, n as i32))
    }.map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;

    Ok(())
}

// =============================================================================
// CPU Fallback Implementations (when CUDA not available)
// =============================================================================

/// CPU implementation of golden gaussian weight
pub fn cpu_golden_gaussian_8d_f64(vectors: &[f64], weights: &mut [f64]) {
    let count = weights.len();
    for i in 0..count {
        let idx = i * 8;
        let mut norm_sq = 0.0;
        for j in 0..8 {
            let v = vectors[idx + j];
            norm_sq += v * v;
        }
        weights[i] = (-norm_sq * PHI_INV).exp();
    }
}

/// CPU implementation of correction factor
pub fn cpu_correction_factor(structure_n: i32, sign: i32) -> f64 {
    1.0 + (sign as f64) * Q_DEFICIT / (structure_n as f64)
}

/// Get structure dimension by index
pub fn get_structure_dimension(idx: i32) -> i32 {
    match idx {
        0 => 248,  // E₈ dim
        1 => 240,  // E₈ roots
        2 => 120,  // E₈ positive roots
        3 => 78,   // E₆ dim
        4 => 36,   // E₆ golden cone
        5 => 27,   // E₆ 27-rep
        6 => 24,   // D₄ kissing
        7 => 14,   // G₂ dim
        _ => 1,
    }
}

// =============================================================================
// Resonant Engine CUDA Operations
// =============================================================================

/// Execute D-phase: lattice → flux with stochastic noise
#[cfg(feature = "cuda")]
pub fn cuda_resonant_d_phase_f64(
    device: &Arc<CudaDevice>,
    flux: &mut CudaSlice<f64>,        // Output: ephemeral floats
    lattice: &CudaSlice<f64>,         // Input: crystallized values
    mode_norm_sq: &CudaSlice<f64>,    // |n|² for each mode
    noise: &CudaSlice<f64>,           // Pre-generated Gaussian noise
    syntony: f64,                      // Current syntony S
    noise_scale: f64,                  // Base noise amplitude
    n: usize,
) -> PyResult<()> {
    ensure_srt_kernels_loaded(device)?;

    let func = device.get_func("srt_resonant", "resonant_d_phase_f64")
        .ok_or_else(|| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(
            "Kernel resonant_d_phase_f64 not found"
        ))?;

    unsafe {
        func.launch(
            launch_cfg_256(n),
            (flux, lattice, mode_norm_sq, noise, syntony, noise_scale, n as i32)
        )
    }.map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;

    Ok(())
}

/// Execute batch D-phase for RES population
#[cfg(feature = "cuda")]
pub fn cuda_resonant_d_phase_batch_f64(
    device: &Arc<CudaDevice>,
    flux_batch: &mut CudaSlice<f64>,
    lattice_batch: &CudaSlice<f64>,
    mode_norm_sq: &CudaSlice<f64>,
    noise_batch: &CudaSlice<f64>,
    syntonies: &CudaSlice<f64>,
    noise_scale: f64,
    n: usize,
    pop_size: usize,
) -> PyResult<()> {
    ensure_srt_kernels_loaded(device)?;

    let func = device.get_func("srt_resonant", "resonant_d_phase_batch_f64")
        .ok_or_else(|| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(
            "Kernel resonant_d_phase_batch_f64 not found"
        ))?;

    let total = n * pop_size;
    unsafe {
        func.launch(
            launch_cfg_256(total),
            (flux_batch, lattice_batch, mode_norm_sq, noise_batch, syntonies,
             noise_scale, n as i32, pop_size as i32)
        )
    }.map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;

    Ok(())
}

/// Compute syntony on flux values
#[cfg(feature = "cuda")]
pub fn cuda_resonant_compute_syntony_f64(
    device: &Arc<CudaDevice>,
    flux: &CudaSlice<f64>,
    mode_norm_sq: &CudaSlice<f64>,
    n: usize,
) -> PyResult<f64> {
    ensure_srt_kernels_loaded(device)?;

    let mut numerator: CudaSlice<f64> = device.alloc_zeros(1)
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
    let mut denominator: CudaSlice<f64> = device.alloc_zeros(1)
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;

    let func = device.get_func("srt_resonant", "resonant_compute_syntony_f64")
        .ok_or_else(|| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(
            "Kernel resonant_compute_syntony_f64 not found"
        ))?;

    let cfg = launch_cfg_reduce(n, 2 * std::mem::size_of::<f64>());

    unsafe {
        func.launch(cfg, (&mut numerator, &mut denominator, flux, mode_norm_sq, n as i32))
    }.map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;

    let mut host_num = [0.0f64];
    let mut host_den = [0.0f64];
    device.dtoh_sync_copy_into(&numerator, &mut host_num)
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
    device.dtoh_sync_copy_into(&denominator, &mut host_den)
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;

    if host_den[0] < 1e-15 {
        Ok(0.0)
    } else {
        Ok(host_num[0] / host_den[0])
    }
}

/// Compute snap gradient for directed exploration
#[cfg(feature = "cuda")]
pub fn cuda_resonant_snap_gradient_f64(
    device: &Arc<CudaDevice>,
    gradient: &mut CudaSlice<f64>,
    flux: &CudaSlice<f64>,
    lattice: &CudaSlice<f64>,
    mode_norm_sq: &CudaSlice<f64>,
    n: usize,
) -> PyResult<()> {
    ensure_srt_kernels_loaded(device)?;

    let func = device.get_func("srt_resonant", "resonant_snap_gradient_f64")
        .ok_or_else(|| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(
            "Kernel resonant_snap_gradient_f64 not found"
        ))?;

    unsafe {
        func.launch(launch_cfg_256(n), (gradient, flux, lattice, mode_norm_sq, n as i32))
    }.map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;

    Ok(())
}

/// Box-Muller transform: uniform → Gaussian noise
#[cfg(feature = "cuda")]
pub fn cuda_resonant_box_muller_f64(
    device: &Arc<CudaDevice>,
    gaussian: &mut CudaSlice<f64>,
    uniform: &CudaSlice<f64>,
    n: usize,
) -> PyResult<()> {
    ensure_srt_kernels_loaded(device)?;

    let func = device.get_func("srt_resonant", "resonant_box_muller_f64")
        .ok_or_else(|| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(
            "Kernel resonant_box_muller_f64 not found"
        ))?;

    unsafe {
        func.launch(launch_cfg_256(n), (gaussian, uniform, n as i32))
    }.map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;

    Ok(())
}
