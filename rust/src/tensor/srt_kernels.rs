//! SRT-specific CUDA kernels for Syntonic Resonance Theory computations.
//!
//! This module provides GPU-accelerated operations for:
//! - Golden ratio operations (φ scaling, gaussian weights)
//! - E₈ lattice projections (P_φ, P_⊥, quadratic forms)
//! - Heat kernel / theta series summation
//! - DHSR cycle operations (differentiation, harmonization, syntony)
//! - SRT correction factors (1 ± q/N)

#[cfg(feature = "cuda")]
use crate::tensor::cuda::device_manager::get_device;
#[cfg(feature = "cuda")]
use crate::tensor::cuda::memory_pool::CudaComplex64;
#[cfg(feature = "cuda")]
use cudarc::driver::safe::CudaContext as CudaDevice;
#[cfg(feature = "cuda")]
use cudarc::driver::PushKernelArg;
#[cfg(feature = "cuda")]
use cudarc::driver::{CudaSlice, DevicePtr, LaunchConfig};
#[cfg(feature = "cuda")]
use pyo3::prelude::*;
#[cfg(feature = "cuda")]
use std::sync::Arc;

// =============================================================================
// PTX Kernel Sources (Pre-compiled)
// =============================================================================

#[cfg(feature = "cuda")]
const PTX_GOLDEN_SM75: &str = include_str!("../../kernels/ptx/golden_ops_sm_75.ptx");
#[cfg(feature = "cuda")]
const PTX_GOLDEN_SM80: &str = include_str!("../../kernels/ptx/golden_ops_sm_80.ptx");
#[cfg(feature = "cuda")]
const PTX_GOLDEN_SM86: &str = include_str!("../../kernels/ptx/golden_ops_sm_86.ptx");
#[cfg(feature = "cuda")]
const PTX_GOLDEN_SM90: &str = include_str!("../../kernels/ptx/golden_ops_sm_90.ptx");

#[cfg(feature = "cuda")]
const PTX_E8_SM75: &str = include_str!("../../kernels/ptx/e8_projection_sm_75.ptx");
#[cfg(feature = "cuda")]
const PTX_E8_SM80: &str = include_str!("../../kernels/ptx/e8_projection_sm_80.ptx");
#[cfg(feature = "cuda")]
const PTX_E8_SM86: &str = include_str!("../../kernels/ptx/e8_projection_sm_86.ptx");
#[cfg(feature = "cuda")]
const PTX_E8_SM90: &str = include_str!("../../kernels/ptx/e8_projection_sm_90.ptx");

#[cfg(feature = "cuda")]
const PTX_HEAT_SM75: &str = include_str!("../../kernels/ptx/heat_kernel_sm_75.ptx");
#[cfg(feature = "cuda")]
const PTX_HEAT_SM80: &str = include_str!("../../kernels/ptx/heat_kernel_sm_80.ptx");
#[cfg(feature = "cuda")]
const PTX_HEAT_SM86: &str = include_str!("../../kernels/ptx/heat_kernel_sm_86.ptx");
#[cfg(feature = "cuda")]
const PTX_HEAT_SM90: &str = include_str!("../../kernels/ptx/heat_kernel_sm_90.ptx");

#[cfg(feature = "cuda")]
const PTX_DHSR_SM75: &str = include_str!("../../kernels/ptx/dhsr_sm_75.ptx");
#[cfg(feature = "cuda")]
const PTX_DHSR_SM80: &str = include_str!("../../kernels/ptx/dhsr_sm_80.ptx");
#[cfg(feature = "cuda")]
const PTX_DHSR_SM86: &str = include_str!("../../kernels/ptx/dhsr_sm_86.ptx");
#[cfg(feature = "cuda")]
const PTX_DHSR_SM90: &str = include_str!("../../kernels/ptx/dhsr_sm_90.ptx");

#[cfg(feature = "cuda")]
const PTX_CORR_SM75: &str = include_str!("../../kernels/ptx/corrections_sm_75.ptx");
#[cfg(feature = "cuda")]
const PTX_CORR_SM80: &str = include_str!("../../kernels/ptx/corrections_sm_80.ptx");
#[cfg(feature = "cuda")]
const PTX_CORR_SM86: &str = include_str!("../../kernels/ptx/corrections_sm_86.ptx");
#[cfg(feature = "cuda")]
const PTX_CORR_SM90: &str = include_str!("../../kernels/ptx/corrections_sm_90.ptx");

#[cfg(feature = "cuda")]
const PTX_RESONANT_SM75: &str = include_str!("../../kernels/ptx/resonant_d_sm_75.ptx");
#[cfg(feature = "cuda")]
const PTX_RESONANT_SM80: &str = include_str!("../../kernels/ptx/resonant_d_sm_80.ptx");
#[cfg(feature = "cuda")]
const PTX_RESONANT_SM86: &str = include_str!("../../kernels/ptx/resonant_d_sm_86.ptx");
#[cfg(feature = "cuda")]
const PTX_RESONANT_SM90: &str = include_str!("../../kernels/ptx/resonant_d_sm_90.ptx");

#[cfg(feature = "cuda")]
const PTX_PHI_RESIDUAL_SM75: &str = include_str!("../../kernels/ptx/phi_residual_sm_75.ptx");
#[cfg(feature = "cuda")]
const PTX_PHI_RESIDUAL_SM80: &str = include_str!("../../kernels/ptx/phi_residual_sm_80.ptx");
#[cfg(feature = "cuda")]
const PTX_PHI_RESIDUAL_SM86: &str = include_str!("../../kernels/ptx/phi_residual_sm_86.ptx");
#[cfg(feature = "cuda")]
const PTX_PHI_RESIDUAL_SM90: &str = include_str!("../../kernels/ptx/phi_residual_sm_90.ptx");

// Matmul PTX (4 compute capabilities)
#[cfg(feature = "cuda")]
const PTX_MATMUL_SM75: &str = include_str!("../../kernels/ptx/matmul_sm_75.ptx");
#[cfg(feature = "cuda")]
const PTX_MATMUL_SM80: &str = include_str!("../../kernels/ptx/matmul_sm_80.ptx");
#[cfg(feature = "cuda")]
const PTX_MATMUL_SM86: &str = include_str!("../../kernels/ptx/matmul_sm_86.ptx");
#[cfg(feature = "cuda")]
const PTX_MATMUL_SM90: &str = include_str!("../../kernels/ptx/matmul_sm_90.ptx");

#[cfg(feature = "cuda")]
const PTX_GOLDEN_BATCH_NORM_SM90: &str =
    include_str!("../../kernels/ptx/golden_batch_norm_sm_90.ptx");

// Syntonic Softmax PTX (4 compute capabilities)
#[cfg(feature = "cuda")]
const PTX_SYNTONIC_SOFTMAX_SM75: &str =
    include_str!("../../kernels/ptx/syntonic_softmax_sm_75.ptx");
#[cfg(feature = "cuda")]
const PTX_SYNTONIC_SOFTMAX_SM80: &str =
    include_str!("../../kernels/ptx/syntonic_softmax_sm_80.ptx");
#[cfg(feature = "cuda")]
const PTX_SYNTONIC_SOFTMAX_SM86: &str =
    include_str!("../../kernels/ptx/syntonic_softmax_sm_86.ptx");
#[cfg(feature = "cuda")]
const PTX_SYNTONIC_SOFTMAX_SM90: &str =
    include_str!("../../kernels/ptx/syntonic_softmax_sm_90.ptx");

// =============================================================================
// Kernel Function Lists
// =============================================================================

/// Golden operations kernel functions
#[cfg(feature = "cuda")]
const GOLDEN_FUNCS: &[&str] = &[
    "scale_phi_f64",
    "scale_phi_f32",
    "scale_phi_inv_f64",
    "scale_phi_inv_f32",
    "fma_phi_kernel_f64",
    "fma_phi_kernel_f32",
    "fma_phi_inv_kernel_f64",
    "fma_phi_inv_kernel_f32",
    "golden_gaussian_weight_scalar_f64",
    "golden_gaussian_weight_scalar_f32",
    "golden_gaussian_weight_4d_int",
    "golden_gaussian_weight_4d_f32",
    "golden_gaussian_weight_8d_f32",
    "golden_gaussian_weight_8d_f64",
    "golden_recursion_4d_int",
    "golden_recursion_f32",
    "golden_recursion_f64",
    "golden_recursion_inv_4d_int",
    "fibonacci_binet_f64",
    "lucas_binet_f64",
    "compute_generation_4d_int",
    "weighted_inner_product_golden_f32",
    "golden_normalize_f32",
    "golden_norm_factor_f32",
    "scale_phi_c128",
    "golden_gaussian_weight_c128",
];

/// E₈ projection kernel functions
#[cfg(feature = "cuda")]
const E8_FUNCS: &[&str] = &[
    "project_parallel_f32",
    "project_parallel_f64",
    "project_perpendicular_f32",
    "project_perpendicular_f64",
    "quadratic_form_f32",
    "quadratic_form_f64",
    "golden_cone_test_f32",
    "golden_cone_test_f64",
    "e8_batch_projection_f32",
    "e8_batch_projection_f64",
    "count_cone_roots",
    "norm_squared_8d_f32",
    "norm_squared_8d_f64",
    "weighted_e8_contribution_f32",
    "weighted_e8_contribution_f64",
];

/// Heat kernel / theta series functions
#[cfg(feature = "cuda")]
const HEAT_FUNCS: &[&str] = &[
    "theta_series_sum_f32",
    "theta_series_sum_f64",
    "heat_kernel_e8_f32",
    "heat_kernel_e8_f64",
    "theta_series_shells_f32",
    "golden_weighted_sum_f32",
    "golden_weighted_sum_f64",
    "modular_inversion_f32",
    "modular_inversion_f64",
    "spectral_zeta_f64",
    "winding_heat_kernel_f32",
    "knot_contribution_f32",
];

/// DHSR cycle functions
#[cfg(feature = "cuda")]
const DHSR_FUNCS: &[&str] = &[
    "compute_syntony_f32",
    "compute_syntony_c128",
    "differentiation_f32",
    "differentiation_c128",
    "harmonization_f32",
    "harmonization_c128",
    "dhsr_cycle_f32",
    "dhsr_cycle_c128",
    "dhsr_cycle_inplace_f32",
    "dhsr_cycle_inplace_c128",
    "compute_gnosis_f32",
    "verify_dh_partition_f32",
    "dhsr_multi_cycle_c128",
];

/// DHSR gravity functions
#[cfg(feature = "cuda")]
const DHSR_GRAVITY_FUNCS: &[&str] = &[
    "apply_geodesic_gravity_f64",
];

/// Correction factor functions
#[cfg(feature = "cuda")]
const CORR_FUNCS: &[&str] = &[
    "apply_correction_f64",
    "apply_correction_f32",
    "apply_corrections_batch_f64",
    "apply_corrections_batch_f32",
    "compound_correction_f64",
    "compound_correction_f32",
    "lepton_mass_correction_f64",
    "quark_mass_correction_f64",
    "coupling_correction_f64",
    "custom_correction_f64",
    "custom_correction_f32",
    "higgs_mass_correction_f64",
    "mixing_matrix_correction_f64",
    "compute_correction_factors_f64",
    "compute_correction_factors_f32",
    "get_q_deficit",
    "get_structure_dimension",
];

/// Resonant D-phase functions
#[cfg(feature = "cuda")]
const RESONANT_FUNCS: &[&str] = &[
    "resonant_d_phase_f64",
    "resonant_d_phase_f32",
    "resonant_d_phase_batch_f64",
    "resonant_compute_syntony_f64",
    "resonant_compute_syntony_f32",
    "resonant_weighted_snap_gradient_f64",
    "resonant_argmax_syntony_f64",
    "resonant_box_muller_f64",
    "resonant_box_muller_f32",
    "resonant_residual_modulated_noise_f64",
    "resonant_compute_dwell_f64",
];

/// Phi-residual connection functions
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
    "phi_residual_gelu_f64",
    "phi_residual_gelu_f32",
    "phi_residual_layernorm_f64",
    "phi_residual_layernorm_f32",
    "phi_residual_mode_phi_vec4_f32",
    "phi_residual_mode_phi_vec2_f64",
    "phi_residual_component_norm_f64",
];

/// Golden batch normalization functions
#[cfg(feature = "cuda")]
const GOLDEN_BATCH_NORM_FUNCS: &[&str] = &[
    "golden_bn_1d_compute_stats_f64",
    "golden_bn_1d_compute_stats_f32",
    "golden_bn_1d_normalize_f64",
    "golden_bn_1d_normalize_f32",
    "golden_bn_2d_compute_stats_f64",
    "golden_bn_2d_compute_stats_f32",
    "golden_bn_2d_normalize_f64",
    "golden_bn_2d_normalize_f32",
    "golden_bn_1d_fused_f64",
    "golden_bn_1d_fused_f32",
    "golden_layer_norm_f64",
    "golden_layer_norm_f32",
    "compute_output_stats_f64",
];

/// Syntonic softmax functions
#[cfg(feature = "cuda")]
const SYNTONIC_SOFTMAX_FUNCS: &[&str] = &[
    "syntonic_softmax_learned_f64",
    "syntonic_softmax_learned_f32",
    "syntonic_softmax_provided_f64",
];

/// Matmul kernel functions
#[cfg(feature = "cuda")]
const MATMUL_FUNCS: &[&str] = &[
    // Standard matrix multiplication
    "matmul_f64",
    "matmul_f32",
    "matmul_c128",
    "matmul_tiled_f64",
    "matmul_tiled_f32",
    // Transposed variants
    "matmul_tn_f64",
    "matmul_tn_f32",
    "matmul_nt_f64",
    "matmul_nt_f32",
    "matmul_tt_f64",
    "matmul_tt_f32",
    // Hermitian variants (complex)
    "matmul_hn_c128",
    "matmul_nh_c128",
    // GEMM operations
    "gemm_nn_f64",
    "gemm_tn_f64",
    "gemm_nt_f64",
    "gemm_tt_f64",
    // Batched operations
    "bmm_f64",
    "bmm_c128",
    // SRT-specific operations
    "matmul_phi_scaled_f64",
    "golden_commutator_f64",
    "golden_anticommutator_f64",
    "matmul_golden_weighted_f64",
    "matmul_golden_weighted_c128",
    // Complex arithmetic
    "complex_div_c128",
    "complex_reciprocal_c128",
];

// =============================================================================
// PTX Selection Based on Compute Capability
// =============================================================================

#[cfg(feature = "cuda")]
fn select_golden_ptx(major: i32, minor: i32) -> &'static str {
    let cc = major * 10 + minor;
    if cc >= 90 {
        PTX_GOLDEN_SM90
    } else if cc >= 86 {
        PTX_GOLDEN_SM86
    } else if cc >= 80 {
        PTX_GOLDEN_SM80
    } else {
        PTX_GOLDEN_SM75
    }
}

#[cfg(feature = "cuda")]
fn select_e8_ptx(major: i32, minor: i32) -> &'static str {
    let cc = major * 10 + minor;
    if cc >= 90 {
        PTX_E8_SM90
    } else if cc >= 86 {
        PTX_E8_SM86
    } else if cc >= 80 {
        PTX_E8_SM80
    } else {
        PTX_E8_SM75
    }
}

#[cfg(feature = "cuda")]
fn select_heat_ptx(major: i32, minor: i32) -> &'static str {
    let cc = major * 10 + minor;
    if cc >= 90 {
        PTX_HEAT_SM90
    } else if cc >= 86 {
        PTX_HEAT_SM86
    } else if cc >= 80 {
        PTX_HEAT_SM80
    } else {
        PTX_HEAT_SM75
    }
}

#[cfg(feature = "cuda")]
fn select_dhsr_ptx(major: i32, minor: i32) -> &'static str {
    let cc = major * 10 + minor;
    if cc >= 90 {
        PTX_DHSR_SM90
    } else if cc >= 86 {
        PTX_DHSR_SM86
    } else if cc >= 80 {
        PTX_DHSR_SM80
    } else {
        PTX_DHSR_SM75
    }
}

#[cfg(feature = "cuda")]
fn select_corr_ptx(major: i32, minor: i32) -> &'static str {
    let cc = major * 10 + minor;
    if cc >= 90 {
        PTX_CORR_SM90
    } else if cc >= 86 {
        PTX_CORR_SM86
    } else if cc >= 80 {
        PTX_CORR_SM80
    } else {
        PTX_CORR_SM75
    }
}

#[cfg(feature = "cuda")]
fn select_resonant_ptx(major: i32, minor: i32) -> &'static str {
    let cc = major * 10 + minor;
    if cc >= 90 {
        PTX_RESONANT_SM90
    } else if cc >= 86 {
        PTX_RESONANT_SM86
    } else if cc >= 80 {
        PTX_RESONANT_SM80
    } else {
        PTX_RESONANT_SM75
    }
}

#[cfg(feature = "cuda")]
fn select_phi_residual_ptx(major: i32, minor: i32) -> &'static str {
    let cc = major * 10 + minor;
    if cc >= 90 {
        PTX_PHI_RESIDUAL_SM90
    } else if cc >= 86 {
        PTX_PHI_RESIDUAL_SM86
    } else if cc >= 80 {
        PTX_PHI_RESIDUAL_SM80
    } else {
        PTX_PHI_RESIDUAL_SM75
    }
}

#[cfg(feature = "cuda")]
fn select_golden_batch_norm_ptx(_major: i32, _minor: i32) -> &'static str {
    PTX_GOLDEN_BATCH_NORM_SM90
}

#[cfg(feature = "cuda")]
fn select_syntonic_softmax_ptx(major: i32, minor: i32) -> &'static str {
    let cc = major * 10 + minor;
    if cc >= 90 {
        PTX_SYNTONIC_SOFTMAX_SM90
    } else if cc >= 86 {
        PTX_SYNTONIC_SOFTMAX_SM86
    } else if cc >= 80 {
        PTX_SYNTONIC_SOFTMAX_SM80
    } else {
        PTX_SYNTONIC_SOFTMAX_SM75
    }
}

#[cfg(feature = "cuda")]
fn select_matmul_ptx(major: i32, minor: i32) -> &'static str {
    let cc = major * 10 + minor;
    if cc >= 90 {
        PTX_MATMUL_SM90
    } else if cc >= 86 {
        PTX_MATMUL_SM86
    } else if cc >= 80 {
        PTX_MATMUL_SM80
    } else {
        PTX_MATMUL_SM75
    }
}

// =============================================================================
// Kernel Loading
// =============================================================================

/// Get compute capability from device
#[cfg(feature = "cuda")]
fn get_compute_capability(device: &Arc<CudaDevice>) -> (i32, i32) {
    use cudarc::driver::result;
    use cudarc::driver::sys::CUdevice_attribute_enum;

    let ordinal = device.ordinal() as i32;

    let major = unsafe {
        result::device::get_attribute(
            ordinal,
            CUdevice_attribute_enum::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR,
        )
        .unwrap_or(7)
    };
    let minor = unsafe {
        result::device::get_attribute(
            ordinal,
            CUdevice_attribute_enum::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR,
        )
        .unwrap_or(0)
    };

    (major, minor)
}

/// Ensure all SRT kernels are loaded for the given device
#[cfg(feature = "cuda")]
pub fn ensure_srt_kernels_loaded(_device: &Arc<CudaDevice>) -> PyResult<()> {
    // In cudarc 0.18.2, modules are loaded on-demand from PTX source
    // No global caching by name exists, so this function is a no-op
    Ok(())
}

/// Validate that listed SRT kernels are present in the PTX modules for a device.
#[cfg(feature = "cuda")]
#[pyfunction]
pub fn validate_kernels(device_idx: usize) -> PyResult<Vec<String>> {
    let device = get_device(device_idx)
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
    validate_kernels_for_device(&device)
}

#[cfg(feature = "cuda")]
fn validate_kernels_for_device(device: &Arc<CudaDevice>) -> PyResult<Vec<String>> {
    use cudarc::nvrtc::Ptx;

    let (major, minor) = get_compute_capability(device);
    let mut missing = Vec::new();

    let mut check_module = |ptx_src: &str, label: &str, funcs: &[&str]| -> PyResult<()> {
        let module = device.load_module(Ptx::from_src(ptx_src)).map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!(
                "Failed to load {} PTX: {}",
                label, e
            ))
        })?;

        for &func_name in funcs {
            if module.load_function(func_name).is_err() {
                missing.push(format!("{}: {}", label, func_name));
            }
        }

        Ok(())
    };

    check_module(select_golden_ptx(major, minor), "golden", GOLDEN_FUNCS)?;
    check_module(select_e8_ptx(major, minor), "e8", E8_FUNCS)?;
    check_module(select_heat_ptx(major, minor), "heat", HEAT_FUNCS)?;
    check_module(select_dhsr_ptx(major, minor), "dhsr", DHSR_FUNCS)?;
    check_module(select_dhsr_ptx(major, minor), "dhsr_gravity", DHSR_GRAVITY_FUNCS)?;
    check_module(select_corr_ptx(major, minor), "correction", CORR_FUNCS)?;
    check_module(
        select_resonant_ptx(major, minor),
        "resonant",
        RESONANT_FUNCS,
    )?;
    check_module(
        select_phi_residual_ptx(major, minor),
        "phi_residual",
        PHI_RESIDUAL_FUNCS,
    )?;
    check_module(
        select_golden_batch_norm_ptx(major, minor),
        "golden_batch_norm",
        GOLDEN_BATCH_NORM_FUNCS,
    )?;
    check_module(
        select_syntonic_softmax_ptx(major, minor),
        "syntonic_softmax",
        SYNTONIC_SOFTMAX_FUNCS,
    )?;
    check_module(select_matmul_ptx(major, minor), "matmul", MATMUL_FUNCS)?;

    Ok(missing)
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
    pub const E8_DIM: i32 = 0; // 248
    pub const E8_ROOTS: i32 = 1; // 240
    pub const E8_POS: i32 = 2; // 120
    pub const E6_DIM: i32 = 3; // 78
    pub const E6_CONE: i32 = 4; // 36
    pub const E6_27: i32 = 5; // 27
    pub const D4_KISSING: i32 = 6; // 24
    pub const G2_DIM: i32 = 7; // 14
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
    let (major, minor) = get_compute_capability(device);
    let module = device
        .load_module(cudarc::nvrtc::Ptx::from_src(select_golden_ptx(
            major, minor,
        )))
        .map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!(
                "Failed to load golden_ops kernels: {}",
                e
            ))
        })?;
    let func = module
        .load_function("scale_phi_f64")
        .map_err(|_| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>("Kernel not found"))?;

    unsafe {
        device
            .default_stream()
            .launch_builder(&func)
            .arg(output)
            .arg(input)
            .arg(&(n as i32))
            .launch(launch_cfg_256(n))
    }
    .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;

    Ok(())
}

/// Compute golden gaussian weights for 8D vectors: w(λ) = exp(-|λ|²/φ)
#[cfg(feature = "cuda")]
pub fn cuda_golden_gaussian_8d_f64(
    device: &Arc<CudaDevice>,
    vectors: &CudaSlice<f64>, // count × 8 flattened
    weights: &mut CudaSlice<f64>,
    count: usize,
) -> PyResult<()> {
    let (major, minor) = get_compute_capability(device);
    let module = device
        .load_module(cudarc::nvrtc::Ptx::from_src(select_golden_ptx(
            major, minor,
        )))
        .map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!(
                "Failed to load golden_ops kernels: {}",
                e
            ))
        })?;
    let func = module
        .load_function("golden_gaussian_weight_8d_f64")
        .map_err(|_| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>("Kernel not found"))?;

    unsafe {
        device
            .default_stream()
            .launch_builder(&func)
            .arg(weights)
            .arg(vectors)
            .arg(&(count as i32))
            .launch(launch_cfg_256(count))
    }
    .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;

    Ok(())
}

/// Batch E₈ projection with Q values and cone test
#[cfg(feature = "cuda")]
pub fn cuda_e8_batch_projection_f64(
    device: &Arc<CudaDevice>,
    roots: &CudaSlice<f64>,             // count × 8
    proj_parallel: &mut CudaSlice<f64>, // count × 4
    proj_perp: &mut CudaSlice<f64>,     // count × 4
    q_values: &mut CudaSlice<f64>,      // count
    in_cone: &mut CudaSlice<i32>,       // count
    count: usize,
) -> PyResult<()> {
    let (major, minor) = get_compute_capability(device);
    let module = device
        .load_module(cudarc::nvrtc::Ptx::from_src(select_e8_ptx(major, minor)))
        .map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!(
                "Failed to load e8_projection kernels: {}",
                e
            ))
        })?;
    let func = module
        .load_function("e8_batch_projection_f64")
        .map_err(|_| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>("Kernel not found"))?;

    unsafe {
        device
            .default_stream()
            .launch_builder(&func)
            .arg(proj_parallel)
            .arg(proj_perp)
            .arg(q_values)
            .arg(in_cone)
            .arg(roots)
            .arg(&(count as i32))
            .launch(launch_cfg_e8(count))
    }
    .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;

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
    let (major, minor) = get_compute_capability(device);
    let module = device
        .load_module(cudarc::nvrtc::Ptx::from_src(select_heat_ptx(major, minor)))
        .map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!(
                "Failed to load heat_kernel kernels: {}",
                e
            ))
        })?;

    // Allocate result on device
    let mut result: CudaSlice<f64> = device
        .default_stream()
        .alloc_zeros(1)
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;

    let func = module
        .load_function("theta_series_sum_f64")
        .map_err(|_| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>("Kernel not found"))?;

    let cfg = launch_cfg_reduce(count, std::mem::size_of::<f64>());

    // Handle optional weights pointer
    let weights_ptr: u64 = match weights {
        Some(w) => w.device_ptr(&device.default_stream()).0,
        None => 0u64, // null pointer
    };

    unsafe {
        device
            .default_stream()
            .launch_builder(&func)
            .arg(&mut result)
            .arg(q_values)
            .arg(in_cone)
            .arg(&weights_ptr)
            .arg(&t)
            .arg(&(count as i32))
            .launch(cfg)
    }
    .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;

    // Copy result back
    let mut host_result = [0.0f64];
    device
        .default_stream()
        .memcpy_dtoh(&result, &mut host_result)
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;

    Ok(host_result[0])
}

/// Compute syntony metric S(ψ)
#[cfg(feature = "cuda")]
pub fn cuda_compute_syntony_c128(
    device: &Arc<CudaDevice>,
    psi: &CudaSlice<CudaComplex64>, // Interleaved complex [re, im, ...]
    mode_norm_sq: &CudaSlice<f64>,  // |n|² for each mode
    n: usize,
) -> PyResult<f64> {
    let (major, minor) = get_compute_capability(device);
    let module = device
        .load_module(cudarc::nvrtc::Ptx::from_src(select_dhsr_ptx(major, minor)))
        .map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!(
                "Failed to load dhsr kernels: {}",
                e
            ))
        })?;

    let mut numerator: CudaSlice<f64> = device
        .default_stream()
        .alloc_zeros(1)
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
    let mut denominator: CudaSlice<f64> = device
        .default_stream()
        .alloc_zeros(1)
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;

    let func = module
        .load_function("compute_syntony_c128")
        .map_err(|_| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>("Kernel not found"))?;

    let cfg = launch_cfg_reduce(n, 2 * std::mem::size_of::<f64>());

    unsafe {
        device
            .default_stream()
            .launch_builder(&func)
            .arg(&mut numerator)
            .arg(&mut denominator)
            .arg(psi)
            .arg(mode_norm_sq)
            .arg(&(n as i32))
            .launch(cfg)
    }
    .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;

    let mut host_num = [0.0f64];
    let mut host_den = [0.0f64];
    device
        .default_stream()
        .memcpy_dtoh(&numerator, &mut host_num)
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
    device
        .default_stream()
        .memcpy_dtoh(&denominator, &mut host_den)
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
    psi: &mut CudaSlice<CudaComplex64>, // In/out: interleaved complex
    mode_norm_sq: &CudaSlice<f64>,
    syntony: f64,
    n: usize,
) -> PyResult<f64> {
    let (major, minor) = get_compute_capability(device);
    let module = device
        .load_module(cudarc::nvrtc::Ptx::from_src(select_dhsr_ptx(major, minor)))
        .map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!(
                "Failed to load dhsr kernels: {}",
                e
            ))
        })?;

    let mut new_num: CudaSlice<f64> = device
        .default_stream()
        .alloc_zeros(1)
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
    let mut new_den: CudaSlice<f64> = device
        .default_stream()
        .alloc_zeros(1)
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;

    let func = module
        .load_function("dhsr_cycle_inplace_c128")
        .map_err(|_| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>("Kernel not found"))?;

    let cfg = launch_cfg_reduce(n, 2 * std::mem::size_of::<f64>());

    unsafe {
        device
            .default_stream()
            .launch_builder(&func)
            .arg(psi)
            .arg(mode_norm_sq)
            .arg(&syntony)
            .arg(&mut new_num)
            .arg(&mut new_den)
            .arg(&(n as i32))
            .launch(cfg)
    }
    .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;

    let mut host_num = [0.0f64];
    let mut host_den = [0.0f64];
    device
        .default_stream()
        .memcpy_dtoh(&new_num, &mut host_num)
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
    device
        .default_stream()
        .memcpy_dtoh(&new_den, &mut host_den)
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
        structure::E8_DIM
            | structure::E8_ROOTS
            | structure::E8_POS
            | structure::E6_DIM
            | structure::E6_CONE
            | structure::E6_27
            | structure::D4_KISSING
            | structure::G2_DIM
    );
    if !valid_idx {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
            "Invalid structure index {}. Use structure::E8_DIM, E8_ROOTS, etc.",
            structure_idx
        )));
    }

    let (major, minor) = get_compute_capability(device);
    let module = device
        .load_module(cudarc::nvrtc::Ptx::from_src(select_corr_ptx(major, minor)))
        .map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!(
                "Failed to load corrections kernels: {}",
                e
            ))
        })?;

    let func = module
        .load_function("apply_correction_f64")
        .map_err(|_| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>("Kernel not found"))?;

    unsafe {
        device
            .default_stream()
            .launch_builder(&func)
            .arg(output)
            .arg(input)
            .arg(&structure_idx)
            .arg(&sign)
            .arg(&(n as i32))
            .launch(launch_cfg_256(n))
    }
    .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;

    Ok(())
}

// =============================================================================
// Matrix Multiplication CUDA Operations
// =============================================================================

/// Standard matrix multiplication: C = A × B (f64)
#[cfg(feature = "cuda")]
pub fn cuda_matmul_f64(
    device: &Arc<CudaDevice>,
    c: &mut CudaSlice<f64>,
    a: &CudaSlice<f64>,
    b: &CudaSlice<f64>,
    m: usize,
    n: usize,
    k: usize,
) -> PyResult<()> {
    let (major, minor) = get_compute_capability(device);
    let module = device
        .load_module(cudarc::nvrtc::Ptx::from_src(select_matmul_ptx(
            major, minor,
        )))
        .map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!(
                "Failed to load matmul kernels: {}",
                e
            ))
        })?;

    let func = module
        .load_function("matmul_f64")
        .map_err(|_| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>("Kernel not found"))?;

    let block_dim = (16, 16, 1);
    let grid_dim = (((n + 15) / 16) as u32, ((m + 15) / 16) as u32, 1);

    unsafe {
        device
            .default_stream()
            .launch_builder(&func)
            .arg(c)
            .arg(a)
            .arg(b)
            .arg(&(m as i32))
            .arg(&(n as i32))
            .arg(&(k as i32))
            .launch(LaunchConfig {
                block_dim,
                grid_dim,
                shared_mem_bytes: 0,
            })
    }
    .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;

    Ok(())
}

/// Tiled matrix multiplication for better performance: C = A × B (f64)
#[cfg(feature = "cuda")]
pub fn cuda_matmul_tiled_f64(
    device: &Arc<CudaDevice>,
    c: &mut CudaSlice<f64>,
    a: &CudaSlice<f64>,
    b: &CudaSlice<f64>,
    m: usize,
    n: usize,
    k: usize,
) -> PyResult<()> {
    println!(
        "DEBUG: cuda_matmul_tiled_f64 called with m={}, n={}, k={}",
        m, n, k
    );

    let (major, minor) = get_compute_capability(device);
    println!("DEBUG: CUDA compute capability: {}.{}", major, minor);

    let ptx_src = select_matmul_ptx(major, minor);
    println!("DEBUG: Selected PTX length: {}", ptx_src.len());

    let module = device
        .load_module(cudarc::nvrtc::Ptx::from_src(ptx_src))
        .map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!(
                "Failed to load matmul kernels: {}",
                e
            ))
        })?;
    println!("DEBUG: Module loaded successfully");

    let func = module.load_function("matmul_tiled_f64").map_err(|_| {
        PyErr::new::<pyo3::exceptions::PyRuntimeError, _>("Kernel matmul_tiled_f64 not found")
    })?;
    println!("DEBUG: Function matmul_tiled_f64 loaded successfully");

    let block_dim = (16, 16, 1);
    let grid_dim = (((n + 15) / 16) as u32, ((m + 15) / 16) as u32, 1);
    println!(
        "DEBUG: Launch config: block_dim={:?}, grid_dim={:?}",
        block_dim, grid_dim
    );

    let launch_result = unsafe {
        device
            .default_stream()
            .launch_builder(&func)
            .arg(c)
            .arg(a)
            .arg(b)
            .arg(&(m as i32))
            .arg(&(n as i32))
            .arg(&(k as i32))
            .launch(LaunchConfig {
                block_dim,
                grid_dim,
                shared_mem_bytes: 0,
            })
    };

    match &launch_result {
        Ok(_) => println!("DEBUG: Kernel launch succeeded"),
        Err(e) => println!("DEBUG: Kernel launch failed: {}", e),
    }

    launch_result.map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;

    // Synchronize to ensure kernel completes
    device.default_stream().synchronize().map_err(|e| {
        PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("Stream sync failed: {}", e))
    })?;
    println!("DEBUG: Stream synchronized successfully");

    Ok(())
}

/// Complex matrix multiplication: C = A × B (complex128)
#[cfg(feature = "cuda")]
pub fn cuda_matmul_c128(
    device: &Arc<CudaDevice>,
    c: &mut CudaSlice<CudaComplex64>,
    a: &CudaSlice<CudaComplex64>,
    b: &CudaSlice<CudaComplex64>,
    m: usize,
    n: usize,
    k: usize,
) -> PyResult<()> {
    let (major, minor) = get_compute_capability(device);
    let module = device
        .load_module(cudarc::nvrtc::Ptx::from_src(select_matmul_ptx(
            major, minor,
        )))
        .map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!(
                "Failed to load matmul kernels: {}",
                e
            ))
        })?;

    let func = module
        .load_function("matmul_c128")
        .map_err(|_| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>("Kernel not found"))?;

    let block_dim = (16, 16, 1);
    let grid_dim = (((n + 15) / 16) as u32, ((m + 15) / 16) as u32, 1);

    unsafe {
        device
            .default_stream()
            .launch_builder(&func)
            .arg(c)
            .arg(a)
            .arg(b)
            .arg(&(m as i32))
            .arg(&(n as i32))
            .arg(&(k as i32))
            .launch(LaunchConfig {
                block_dim,
                grid_dim,
                shared_mem_bytes: 0,
            })
    }
    .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;

    Ok(())
}

/// Transposed matrix multiplication: C = Aᵀ × B (f64)
#[cfg(feature = "cuda")]
pub fn cuda_matmul_tn_f64(
    device: &Arc<CudaDevice>,
    c: &mut CudaSlice<f64>,
    a: &CudaSlice<f64>,
    b: &CudaSlice<f64>,
    m: usize,
    n: usize,
    k: usize,
) -> PyResult<()> {
    let (major, minor) = get_compute_capability(device);
    let module = device
        .load_module(cudarc::nvrtc::Ptx::from_src(select_matmul_ptx(
            major, minor,
        )))
        .map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!(
                "Failed to load matmul kernels: {}",
                e
            ))
        })?;

    let func = module
        .load_function("matmul_tn_f64")
        .map_err(|_| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>("Kernel not found"))?;

    let block_dim = (16, 16, 1);
    let grid_dim = (((n + 15) / 16) as u32, ((m + 15) / 16) as u32, 1);

    unsafe {
        device
            .default_stream()
            .launch_builder(&func)
            .arg(c)
            .arg(a)
            .arg(b)
            .arg(&(m as i32))
            .arg(&(n as i32))
            .arg(&(k as i32))
            .launch(LaunchConfig {
                block_dim,
                grid_dim,
                shared_mem_bytes: 0,
            })
    }
    .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;

    Ok(())
}

/// Transposed matrix multiplication: C = Aᵀ × B (f32)
#[cfg(feature = "cuda")]
pub fn cuda_matmul_tn_f32(
    device: &Arc<CudaDevice>,
    c: &mut CudaSlice<f32>,
    a: &CudaSlice<f32>,
    b: &CudaSlice<f32>,
    m: usize,
    n: usize,
    k: usize,
) -> PyResult<()> {
    let (major, minor) = get_compute_capability(device);
    let module = device
        .load_module(cudarc::nvrtc::Ptx::from_src(select_matmul_ptx(
            major, minor,
        )))
        .map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!(
                "Failed to load matmul kernels: {}",
                e
            ))
        })?;

    let func = module
        .load_function("matmul_tn_f32")
        .map_err(|_| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>("Kernel not found"))?;

    let block_dim = (16, 16, 1);
    let grid_dim = (((n + 15) / 16) as u32, ((m + 15) / 16) as u32, 1);

    unsafe {
        device
            .default_stream()
            .launch_builder(&func)
            .arg(c)
            .arg(a)
            .arg(b)
            .arg(&(m as i32))
            .arg(&(n as i32))
            .arg(&(k as i32))
            .launch(LaunchConfig {
                block_dim,
                grid_dim,
                shared_mem_bytes: 0,
            })
    }
    .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;

    Ok(())
}

/// Golden commutator: [A, B]_φ = AB - φ⁻¹BA
#[cfg(feature = "cuda")]
pub fn cuda_golden_commutator_f64(
    device: &Arc<CudaDevice>,
    c: &mut CudaSlice<f64>,
    a: &CudaSlice<f64>,
    b: &CudaSlice<f64>,
    m: usize,
    n: usize,
    k: usize,
) -> PyResult<()> {
    let (major, minor) = get_compute_capability(device);
    let module = device
        .load_module(cudarc::nvrtc::Ptx::from_src(select_matmul_ptx(
            major, minor,
        )))
        .map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!(
                "Failed to load matmul kernels: {}",
                e
            ))
        })?;

    let func = module
        .load_function("golden_commutator_f64")
        .map_err(|_| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>("Kernel not found"))?;

    let block_dim = (16, 16, 1);
    let grid_dim = (((n + 15) / 16) as u32, ((m + 15) / 16) as u32, 1);

    unsafe {
        device
            .default_stream()
            .launch_builder(&func)
            .arg(c)
            .arg(a)
            .arg(b)
            .arg(&(m as i32))
            .arg(&(n as i32))
            .arg(&(k as i32))
            .launch(LaunchConfig {
                block_dim,
                grid_dim,
                shared_mem_bytes: 0,
            })
    }
    .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;

    Ok(())
}

/// Golden anticommutator: {A, B}_φ = AB + φ⁻¹BA
#[cfg(feature = "cuda")]
pub fn cuda_golden_anticommutator_f64(
    device: &Arc<CudaDevice>,
    c: &mut CudaSlice<f64>,
    a: &CudaSlice<f64>,
    b: &CudaSlice<f64>,
    m: usize,
    n: usize,
    k: usize,
) -> PyResult<()> {
    let (major, minor) = get_compute_capability(device);
    let module = device
        .load_module(cudarc::nvrtc::Ptx::from_src(select_matmul_ptx(
            major, minor,
        )))
        .map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!(
                "Failed to load matmul kernels: {}",
                e
            ))
        })?;

    let func = module
        .load_function("golden_anticommutator_f64")
        .map_err(|_| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>("Kernel not found"))?;

    let block_dim = (16, 16, 1);
    let grid_dim = (((n + 15) / 16) as u32, ((m + 15) / 16) as u32, 1);

    unsafe {
        device
            .default_stream()
            .launch_builder(&func)
            .arg(c)
            .arg(a)
            .arg(b)
            .arg(&(m as i32))
            .arg(&(n as i32))
            .arg(&(k as i32))
            .launch(LaunchConfig {
                block_dim,
                grid_dim,
                shared_mem_bytes: 0,
            })
    }
    .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;

    Ok(())
}

/// φ-scaled matrix multiplication: C = φⁿ × (A × B)
#[cfg(feature = "cuda")]
pub fn cuda_matmul_phi_scaled_f64(
    device: &Arc<CudaDevice>,
    c: &mut CudaSlice<f64>,
    a: &CudaSlice<f64>,
    b: &CudaSlice<f64>,
    n: i32, // Power of phi
    m: usize,
    k: usize,
    p: usize,
) -> PyResult<()> {
    let (major, minor) = get_compute_capability(device);
    let module = device
        .load_module(cudarc::nvrtc::Ptx::from_src(select_matmul_ptx(
            major, minor,
        )))
        .map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!(
                "Failed to load matmul kernels: {}",
                e
            ))
        })?;

    let func = module
        .load_function("matmul_phi_scaled_f64")
        .map_err(|_| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>("Kernel not found"))?;

    let block_dim = (16, 16, 1);
    let grid_dim = (((p + 15) / 16) as u32, ((m + 15) / 16) as u32, 1);

    unsafe {
        device
            .default_stream()
            .launch_builder(&func)
            .arg(c)
            .arg(a)
            .arg(b)
            .arg(&n)
            .arg(&(m as i32))
            .arg(&(p as i32))
            .arg(&(k as i32))
            .launch(LaunchConfig {
                block_dim,
                grid_dim,
                shared_mem_bytes: 0,
            })
    }
    .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;

    Ok(())
}

/// Batched matrix multiplication: C[i] = A[i] × B[i]
#[cfg(feature = "cuda")]
pub fn cuda_bmm_f64(
    device: &Arc<CudaDevice>,
    c: &mut CudaSlice<f64>,
    a: &CudaSlice<f64>,
    b: &CudaSlice<f64>,
    batch_size: usize,
    m: usize,
    n: usize,
    k: usize,
) -> PyResult<()> {
    let (major, minor) = get_compute_capability(device);
    let module = device
        .load_module(cudarc::nvrtc::Ptx::from_src(select_matmul_ptx(
            major, minor,
        )))
        .map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!(
                "Failed to load matmul kernels: {}",
                e
            ))
        })?;

    let func = module
        .load_function("bmm_f64")
        .map_err(|_| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>("Kernel not found"))?;

    let block_dim = (8, 8, 8);
    let grid_dim = (
        ((n + 7) / 8) as u32,
        ((m + 7) / 8) as u32,
        ((batch_size + 7) / 8) as u32,
    );

    unsafe {
        device
            .default_stream()
            .launch_builder(&func)
            .arg(c)
            .arg(a)
            .arg(b)
            .arg(&(batch_size as i32))
            .arg(&(m as i32))
            .arg(&(n as i32))
            .arg(&(k as i32))
            .launch(LaunchConfig {
                block_dim,
                grid_dim,
                shared_mem_bytes: 0,
            })
    }
    .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;

    Ok(())
}

/// Complex division: C = A / B (element-wise)
#[cfg(feature = "cuda")]
pub fn cuda_complex_div_c128(
    device: &Arc<CudaDevice>,
    c: &mut CudaSlice<CudaComplex64>,
    a: &CudaSlice<CudaComplex64>,
    b: &CudaSlice<CudaComplex64>,
    n: usize,
) -> PyResult<()> {
    let (major, minor) = get_compute_capability(device);
    let module = device
        .load_module(cudarc::nvrtc::Ptx::from_src(select_matmul_ptx(
            major, minor,
        )))
        .map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!(
                "Failed to load matmul kernels: {}",
                e
            ))
        })?;

    let func = module
        .load_function("complex_div_c128")
        .map_err(|_| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>("Kernel not found"))?;

    unsafe {
        device
            .default_stream()
            .launch_builder(&func)
            .arg(c)
            .arg(a)
            .arg(b)
            .arg(&(n as i32))
            .launch(launch_cfg_256(n))
    }
    .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;

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
        0 => 248, // E₈ dim
        1 => 240, // E₈ roots
        2 => 120, // E₈ positive roots
        3 => 78,  // E₆ dim
        4 => 36,  // E₆ golden cone
        5 => 27,  // E₆ 27-rep
        6 => 24,  // D₄ kissing
        7 => 14,  // G₂ dim
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
    flux: &mut CudaSlice<f64>,     // Output: ephemeral floats
    lattice: &CudaSlice<f64>,      // Input: crystallized values
    mode_norm_sq: &CudaSlice<f64>, // |n|² for each mode
    noise: &CudaSlice<f64>,        // Pre-generated Gaussian noise
    syntony: f64,                  // Current syntony S
    noise_scale: f64,              // Base noise amplitude
    n: usize,
) -> PyResult<()> {
    let (major, minor) = get_compute_capability(device);
    let module = device
        .load_module(cudarc::nvrtc::Ptx::from_src(select_resonant_ptx(
            major, minor,
        )))
        .map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!(
                "Failed to load resonant_d kernels: {}",
                e
            ))
        })?;

    let func = module
        .load_function("resonant_d_phase_f64")
        .map_err(|_| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>("Kernel not found"))?;

    unsafe {
        device
            .default_stream()
            .launch_builder(&func)
            .arg(flux)
            .arg(lattice)
            .arg(mode_norm_sq)
            .arg(noise)
            .arg(&syntony)
            .arg(&noise_scale)
            .arg(&(n as i32))
            .launch(launch_cfg_256(n))
    }
    .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;

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
    let (major, minor) = get_compute_capability(device);
    let module = device
        .load_module(cudarc::nvrtc::Ptx::from_src(select_resonant_ptx(
            major, minor,
        )))
        .map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!(
                "Failed to load resonant_d kernels: {}",
                e
            ))
        })?;

    let func = module
        .load_function("resonant_d_phase_batch_f64")
        .map_err(|_| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>("Kernel not found"))?;

    let total = n * pop_size;
    unsafe {
        device
            .default_stream()
            .launch_builder(&func)
            .arg(flux_batch)
            .arg(lattice_batch)
            .arg(mode_norm_sq)
            .arg(noise_batch)
            .arg(syntonies)
            .arg(&noise_scale)
            .arg(&(n as i32))
            .arg(&(pop_size as i32))
            .launch(launch_cfg_256(total))
    }
    .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;

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
    let (major, minor) = get_compute_capability(device);
    let module = device
        .load_module(cudarc::nvrtc::Ptx::from_src(select_resonant_ptx(
            major, minor,
        )))
        .map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!(
                "Failed to load resonant_d kernels: {}",
                e
            ))
        })?;

    let mut numerator: CudaSlice<f64> = device
        .default_stream()
        .alloc_zeros(1)
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
    let mut denominator: CudaSlice<f64> = device
        .default_stream()
        .alloc_zeros(1)
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;

    let func = module
        .load_function("resonant_compute_syntony_f64")
        .map_err(|_| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>("Kernel not found"))?;

    let cfg = launch_cfg_reduce(n, 2 * std::mem::size_of::<f64>());

    unsafe {
        device
            .default_stream()
            .launch_builder(&func)
            .arg(&mut numerator)
            .arg(&mut denominator)
            .arg(flux)
            .arg(mode_norm_sq)
            .arg(&(n as i32))
            .launch(cfg)
    }
    .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;

    let mut host_num = [0.0f64];
    let mut host_den = [0.0f64];
    device
        .default_stream()
        .memcpy_dtoh(&numerator, &mut host_num)
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
    device
        .default_stream()
        .memcpy_dtoh(&denominator, &mut host_den)
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;

    if host_den[0] < 1e-15 {
        Ok(0.0)
    } else {
        Ok(host_num[0] / host_den[0])
    }
}

// =============================================================================
// Phi-Residual CUDA Kernels
// =============================================================================

/// Execute phi-residual connection on GPU
#[cfg(feature = "cuda")]
pub fn cuda_phi_residual_f64(
    device: &Arc<CudaDevice>,
    out: &mut CudaSlice<f64>,
    identity: &CudaSlice<f64>,
    residual: &CudaSlice<f64>,
    mode: crate::resonant::phi_ops::PhiResidualMode,
) -> Result<(), String> {
    use crate::resonant::phi_ops::PhiResidualMode;

    let n = out.len();

    let (major, minor) = get_compute_capability(device);
    let module = device
        .load_module(cudarc::nvrtc::Ptx::from_src(select_phi_residual_ptx(
            major, minor,
        )))
        .map_err(|e| format!("Failed to load phi_residual kernels: {}", e))?;

    // Select kernel based on mode
    let kernel_name = match mode {
        PhiResidualMode::Phi => "phi_residual_mode_phi_f64",
        PhiResidualMode::PhiSymmetric => "phi_residual_mode_symmetric_f64",
        PhiResidualMode::Standard => "phi_residual_mode_standard_f64",
    };

    let func = module
        .load_function(kernel_name)
        .map_err(|_| "Kernel not found".to_string())?;

    let cfg = launch_cfg_256(n);

    unsafe {
        device
            .default_stream()
            .launch_builder(&func)
            .arg(out)
            .arg(identity)
            .arg(residual)
            .arg(&(n as i32))
            .launch(cfg)
    }
    .map(|_| ())
    .map_err(|e| e.to_string())
}

/// Execute fused phi-residual + ReLU on GPU
#[cfg(feature = "cuda")]
pub fn cuda_phi_residual_relu_f64(
    device: &Arc<CudaDevice>,
    out: &mut CudaSlice<f64>,
    identity: &CudaSlice<f64>,
    residual: &CudaSlice<f64>,
) -> Result<(), String> {
    let n = out.len();

    let (major, minor) = get_compute_capability(device);
    let module = device
        .load_module(cudarc::nvrtc::Ptx::from_src(select_phi_residual_ptx(
            major, minor,
        )))
        .map_err(|e| format!("Failed to load phi_residual kernels: {}", e))?;

    let func = module
        .load_function("phi_residual_relu_f64")
        .map_err(|_| "Kernel not found".to_string())?;

    let cfg = launch_cfg_256(n);

    unsafe {
        device
            .default_stream()
            .launch_builder(&func)
            .arg(out)
            .arg(identity)
            .arg(residual)
            .arg(&(n as i32))
            .launch(cfg)
    }
    .map(|_| ())
    .map_err(|e| e.to_string())
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
    let (major, minor) = get_compute_capability(device);
    let module = device
        .load_module(cudarc::nvrtc::Ptx::from_src(select_resonant_ptx(
            major, minor,
        )))
        .map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!(
                "Failed to load resonant_d kernels: {}",
                e
            ))
        })?;

    let func = module
        .load_function("resonant_weighted_snap_gradient_f64")
        .map_err(|_| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>("Kernel not found"))?;

    unsafe {
        device
            .default_stream()
            .launch_builder(&func)
            .arg(gradient)
            .arg(flux)
            .arg(lattice)
            .arg(mode_norm_sq)
            .arg(&(n as i32))
            .launch(launch_cfg_256(n))
    }
    .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;

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
    let (major, minor) = get_compute_capability(device);
    let module = device
        .load_module(cudarc::nvrtc::Ptx::from_src(select_resonant_ptx(
            major, minor,
        )))
        .map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!(
                "Failed to load resonant_d kernels: {}",
                e
            ))
        })?;

    let func = module
        .load_function("resonant_box_muller_f64")
        .map_err(|_| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>("Kernel not found"))?;

    unsafe {
        device
            .default_stream()
            .launch_builder(&func)
            .arg(gaussian)
            .arg(uniform)
            .arg(&(n as i32))
            .launch(launch_cfg_256(n))
    }
    .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;

    Ok(())
}

// =============================================================================
// Golden Batch Normalization CUDA Kernels
// =============================================================================

/// Golden batch normalization for 1D tensors (batch, features)
#[cfg(feature = "cuda")]
pub fn cuda_golden_bn_1d_f64(
    device: &Arc<CudaDevice>,
    out: &mut CudaSlice<f64>,
    input: &CudaSlice<f64>,
    gamma: Option<&CudaSlice<f64>>,
    beta: Option<&CudaSlice<f64>>,
    eps: f64,
    batch_size: i32,
    features: i32,
) -> Result<(), String> {
    let n = (batch_size * features) as usize;
    let (major, minor) = get_compute_capability(device);
    let module = device
        .load_module(cudarc::nvrtc::Ptx::from_src(select_golden_batch_norm_ptx(
            major, minor,
        )))
        .map_err(|e| format!("Failed to load golden_batch_norm kernels: {}", e))?;

    let func = module
        .load_function("golden_bn_1d_fused_f64")
        .map_err(|_| "Kernel not found".to_string())?;

    let gamma_ptr = gamma
        .map(|g| g.device_ptr(&device.default_stream()).0)
        .unwrap_or(0);
    let beta_ptr = beta
        .map(|b| b.device_ptr(&device.default_stream()).0)
        .unwrap_or(0);

    unsafe {
        device
            .default_stream()
            .launch_builder(&func)
            .arg(out)
            .arg(input)
            .arg(&gamma_ptr)
            .arg(&beta_ptr)
            .arg(&eps)
            .arg(&batch_size)
            .arg(&features)
            .launch(launch_cfg_256(n))
    }
    .map(|_| ())
    .map_err(|e| e.to_string())
}

#[cfg(feature = "cuda")]
pub fn cuda_golden_bn_2d_f64(
    device: &Arc<CudaDevice>,
    out: &mut CudaSlice<f64>,
    input: &CudaSlice<f64>,
    mean: &CudaSlice<f64>,
    var: &CudaSlice<f64>,
    gamma: Option<&CudaSlice<f64>>,
    beta: Option<&CudaSlice<f64>>,
    eps: f64,
    b: i32,
    c: i32,
    h: i32,
    w: i32,
) -> Result<(), String> {
    let n = (b * c * h * w) as usize;
    let (major, minor) = get_compute_capability(device);
    let module = device
        .load_module(cudarc::nvrtc::Ptx::from_src(select_golden_batch_norm_ptx(
            major, minor,
        )))
        .map_err(|e| format!("Failed to load golden_batch_norm kernels: {}", e))?;

    let func = module
        .load_function("golden_bn_2d_normalize_f64")
        .map_err(|_| "Kernel not found".to_string())?;

    let gamma_ptr = gamma
        .map(|g| g.device_ptr(&device.default_stream()).0)
        .unwrap_or(0);
    let beta_ptr = beta
        .map(|b| b.device_ptr(&device.default_stream()).0)
        .unwrap_or(0);

    unsafe {
        device
            .default_stream()
            .launch_builder(&func)
            .arg(out)
            .arg(input)
            .arg(mean)
            .arg(var)
            .arg(&gamma_ptr)
            .arg(&beta_ptr)
            .arg(&eps)
            .arg(&b)
            .arg(&c)
            .arg(&h)
            .arg(&w)
            .launch(launch_cfg_256(n))
    }
    .map(|_| ())
    .map_err(|e| e.to_string())
}

// =============================================================================
// Syntonic Softmax CUDA Kernels
// =============================================================================

#[cfg(feature = "cuda")]
pub fn cuda_syntonic_softmax_learned_f64(
    device: &Arc<CudaDevice>,
    out: &mut CudaSlice<f64>,
    logits: &CudaSlice<f64>,
    mode_norms: &CudaSlice<f64>,
    syntony_scale: f64,
    batch_size: i32,
    num_classes: i32,
) -> Result<(), String> {
    let (major, minor) = get_compute_capability(device);
    let module = device
        .load_module(cudarc::nvrtc::Ptx::from_src(select_syntonic_softmax_ptx(
            major, minor,
        )))
        .map_err(|e| format!("Failed to load syntonic_softmax kernels: {}", e))?;

    let func = module
        .load_function("syntonic_softmax_learned_f64")
        .map_err(|_| "Kernel not found".to_string())?;

    let cfg = LaunchConfig {
        grid_dim: (batch_size as u32, 1, 1),
        block_dim: (256, 1, 1),
        shared_mem_bytes: 256 * 2 * std::mem::size_of::<f64>() as u32,
    };

    unsafe {
        device
            .default_stream()
            .launch_builder(&func)
            .arg(out)
            .arg(logits)
            .arg(mode_norms)
            .arg(&syntony_scale)
            .arg(&batch_size)
            .arg(&num_classes)
            .launch(cfg)
    }
    .map(|_| ())
    .map_err(|e| e.to_string())
}

#[cfg(feature = "cuda")]
pub fn cuda_syntonic_softmax_provided_f64(
    device: &Arc<CudaDevice>,
    out: &mut CudaSlice<f64>,
    logits: &CudaSlice<f64>,
    syntony: &CudaSlice<f64>,
    syntony_scale: f64,
    batch_size: i32,
    num_classes: i32,
) -> Result<(), String> {
    let (major, minor) = get_compute_capability(device);
    let module = device
        .load_module(cudarc::nvrtc::Ptx::from_src(select_syntonic_softmax_ptx(
            major, minor,
        )))
        .map_err(|e| format!("Failed to load syntonic_softmax kernels: {}", e))?;

    let func = module
        .load_function("syntonic_softmax_provided_f64")
        .map_err(|_| "Kernel not found".to_string())?;

    let cfg = LaunchConfig {
        grid_dim: (batch_size as u32, 1, 1),
        block_dim: (256, 1, 1),
        shared_mem_bytes: 256 * 2 * std::mem::size_of::<f64>() as u32,
    };

    unsafe {
        device
            .default_stream()
            .launch_builder(&func)
            .arg(out)
            .arg(logits)
            .arg(syntony)
            .arg(&syntony_scale)
            .arg(&batch_size)
            .arg(&num_classes)
            .launch(cfg)
    }
    .map(|_| ())
    .map_err(|e| e.to_string())
}

// =============================================================================
// Geodesic Gravity Operations
// =============================================================================

/// Apply Geodesic Gravity Slide to weights (Physical AI update)
#[cfg(feature = "cuda")]
pub fn apply_geodesic_gravity_f64(
    device: &Arc<CudaDevice>,
    weights: &CudaSlice<f64>, // Mutable reference in intent, but Arc shared in Python
    attractor: &CudaSlice<f64>,
    mode_norm_sq: &CudaSlice<f64>,
    gravity: f64,
    temperature: f64,
    n: usize,
) -> Result<(), String> {
    let (major, minor) = get_compute_capability(device);
    let module = device
        .load_module(cudarc::nvrtc::Ptx::from_src(select_dhsr_ptx(major, minor)))
        .map_err(|e| format!("Failed to load dhsr kernels: {}", e))?;

    let func = module
        .load_function("apply_geodesic_gravity_f64")
        .map_err(|_| "Kernel apply_geodesic_gravity_f64 not found".to_string())?;

    // IMPORTANT: The kernel processes chunks of 8 (E8 unit cells)
    // We launch threads for *blocks of 8*, not individual elements
    let num_threads = (n + 7) / 8;
    let cfg = LaunchConfig::for_num_elems(num_threads as u32);

    unsafe {
        device
            .default_stream()
            .launch_builder(&func)
            .arg(weights)
            .arg(attractor)
            .arg(mode_norm_sq)
            .arg(&gravity)
            .arg(&temperature)
            .arg(&(n as i32))
            .launch(cfg)
    }
    .map(|_| ())
    .map_err(|e| e.to_string())
}
