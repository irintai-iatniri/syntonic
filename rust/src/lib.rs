use pyo3::prelude::*;

mod exact;
mod gnosis;
mod golden_gelu;
mod hierarchy;
mod hypercomplex;
mod linalg;
mod prime_selection;
mod resonant;
mod spectral;
mod tensor;
mod transcendence;
mod winding;

use hypercomplex::{Octonion, Quaternion, Sedenion};
#[cfg(feature = "cuda")]
use tensor::cuda::{AsyncTensorTransfer, TransferComputeOverlap};
use tensor::srt_kernels;
use tensor::storage::{
    cuda_device_count, cuda_is_available, srt_apply_correction, srt_compute_syntony,
    srt_dhsr_cycle, srt_e8_batch_projection, srt_golden_gaussian_weights,
    srt_scale_phi, srt_theta_series, TensorStorage,
};

// SRT Inflationary Broadcasting
use tensor::broadcast::{
    py_inflationary_broadcast, py_golden_inflationary_broadcast,
    py_consciousness_inflationary_broadcast,
};

// Causal History DAG for DHSR tracking
use tensor::causal_history::{
    PyCausalHistoryTracker, create_causal_tracker, d4_consciousness_threshold,
};
#[cfg(feature = "cuda")]
use tensor::storage::{
    srt_memory_resonance, srt_pool_stats, srt_reserve_memory, srt_transfer_stats,
    srt_wait_for_resonance,
};

// Winding state and enumeration
use winding::{
    count_windings, enumerate_windings, enumerate_windings_by_norm, enumerate_windings_exact_norm,
    WindingState, WindingStateIterator,
};

// Prime selection, gnosis, and transcendence
use gnosis::register_gnosis;
use prime_selection::register_extended_prime_selection;
use transcendence::register_transcendence;

// Spectral operations
use spectral::{
    compute_eigenvalues,
    compute_golden_weights,
    compute_knot_eigenvalues,
    compute_norm_squared,
    count_by_generation,
    filter_by_generation,
    heat_kernel_derivative,
    heat_kernel_trace,
    heat_kernel_weighted,
    // Knot Laplacian operations
    knot_eigenvalue,
    knot_heat_kernel_trace,
    knot_spectral_zeta,
    knot_spectral_zeta_complex,
    partition_function,
    spectral_zeta,
    spectral_zeta_weighted,
    theta_series_derivative,
    theta_series_evaluate,
    theta_series_weighted,
    theta_sum_combined,
};

// New exact arithmetic types
use exact::{CorrectionLevel, FundamentalConstant, GoldenExact, PySymExpr, Rational, Structure};

// Resonant Engine types
use resonant::{RESConfig, RESResult, ResonantEvolver, ResonantTensor};

// E8 Lattice and Golden Projector wrappers
use resonant::{
    py_compute_8d_weight, py_e8_generate_roots, py_e8_generate_weights, py_golden_project_parallel,
    py_golden_project_perp, py_golden_projector_phi, py_golden_projector_q, py_is_in_golden_cone,
};

// Neural Network E8 Lattice and Golden Projector wrappers
use resonant::{
    py_compute_8d_weight_nn, py_e8_generate_roots_nn, py_e8_generate_weights_nn,
    py_golden_project_parallel_nn, py_golden_project_perp_nn, py_golden_projector_phi_nn,
    py_golden_projector_q_nn, py_is_in_golden_cone_nn,
};

// Number theory and syntony wrappers
use resonant::py_wrappers::{
    py_aggregate_syntony,
    py_are_broadcastable,
    py_avg_pool2d,
    py_batch_cross_entropy_loss,
    py_batch_winding_syntony,
    py_broadcast_add,
    py_broadcast_div,
    py_broadcast_mul,
    // Broadcasting
    py_broadcast_shape,
    py_broadcast_sub,
    py_compute_snap_gradient,
    py_compute_winding_syntony,
    // Convolution
    py_conv2d,
    py_cross_entropy_loss,
    py_crystallize_with_dwell_legacy,
    py_e_star,
    py_estimate_syntony_from_probs,
    py_get_q_deficit,
    py_get_target_syntony,
    py_global_avg_pool2d,
    py_golden_decay_loss,
    py_golden_weight,
    py_golden_weights,
    py_inplace_abs,
    // In-place
    py_inplace_add_scalar,
    py_inplace_clamp,
    py_inplace_div_scalar,
    py_inplace_golden_weight,
    py_inplace_mul_scalar,
    py_inplace_negate,
    py_inplace_sub_scalar,
    py_is_square_free,
    py_linear_index,
    py_max_pool2d,
    py_mertens,
    py_mobius,
    py_mse_loss,
    py_phase_alignment_loss,
    py_snap_distance,
    py_softmax,
    py_standard_mode_norms,
    py_syntonic_loss,
    py_syntony_loss,
    py_syntony_loss_srt,
};

// Linear algebra operations
use linalg::{
    py_bmm,
    py_mm,
    py_mm_add,
    py_mm_corrected,
    // New generalized functions
    py_mm_gemm,
    py_mm_golden_phase,
    py_mm_golden_weighted,
    py_mm_hn,
    py_mm_nh,
    py_mm_nt,
    py_mm_phi,
    py_mm_q_corrected_direct,
    py_mm_tn,
    py_mm_tt,
    py_phi_antibracket,
    py_phi_bracket,
    py_projection_sum,
    py_q_correction_scalar,
};

// =============================================================================
// SRT Constant Functions (Python-accessible)
// =============================================================================

/// Get the golden ratio φ = (1 + √5) / 2
#[pyfunction]
fn srt_phi() -> f64 {
    srt_kernels::PHI
}

/// Get the golden ratio inverse φ⁻¹ = φ - 1
#[pyfunction]
fn srt_phi_inv() -> f64 {
    srt_kernels::PHI_INV
}

/// Get the q-deficit value q = W(∞) - 1 ≈ 0.027395
#[pyfunction]
fn srt_q_deficit() -> f64 {
    srt_kernels::Q_DEFICIT
}

/// Get π (pi) constant
#[pyfunction]
fn srt_pi() -> f64 {
    std::f64::consts::PI
}

/// Get e (Euler's number) constant
#[pyfunction]
fn srt_e() -> f64 {
    std::f64::consts::E
}

/// Get structure dimension by index
/// 0: E₈ dim (248), 1: E₈ roots (240), 2: E₈ pos (120),
/// 3: E₆ dim (78), 4: E₆ cone (36), 5: E₆ 27 (27),
/// 6: D₄ kissing (24), 7: G₂ dim (14)
#[pyfunction]
fn srt_structure_dimension(index: i32) -> i32 {
    srt_kernels::get_structure_dimension(index)
}

/// Compute correction factor (1 + sign * q / N)
#[pyfunction]
fn srt_correction_factor(structure_index: i32, sign: i32) -> f64 {
    let n = srt_kernels::get_structure_dimension(structure_index);
    srt_kernels::cpu_correction_factor(n, sign)
}

/// Apply Geodesic Gravity Slide to weights in-place (Physical AI update)
#[pyfunction]
fn py_apply_geodesic_slide(
    weights: &TensorStorage,
    attractor: &TensorStorage,
    mode_norms: &TensorStorage,
    gravity: f64,
    temperature: f64,
) -> PyResult<()> {
    #[cfg(feature = "cuda")]
    {
        use tensor::storage::{CudaData, TensorData};
        if let (
            TensorData::Cuda {
                data: w_data,
                device: dev,
                ..
            },
            TensorData::Cuda { data: a_data, .. },
            TensorData::Cuda { data: m_data, .. },
        ) = (&weights.data, &attractor.data, &mode_norms.data)
        {
            if let (
                CudaData::Float64(w_slice),
                CudaData::Float64(a_slice),
                CudaData::Float64(m_slice),
            ) = (w_data.as_ref(), a_data.as_ref(), m_data.as_ref())
            {
                let n = w_slice.len();
                srt_kernels::apply_geodesic_gravity_f64(
                    dev,
                    w_slice,
                    a_slice,
                    m_slice,
                    gravity,
                    temperature,
                    n,
                )
                .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e))?;
            }
        }
    }
    Ok(())
}

/// Core module for Syntonic
///
/// This module provides:
/// - Exact arithmetic types (Rational, GoldenExact, SymExpr)
/// - The five fundamental SRT constants (π, e, φ, E*, q)
/// - Tensor storage (legacy, uses floats - to be replaced)
/// - Hypercomplex numbers (Quaternion, Octonion)
#[pymodule]
fn _core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    // === Exact Arithmetic (NEW - preferred) ===
    m.add_class::<Rational>()?;
    m.add_class::<GoldenExact>()?;
    m.add_class::<FundamentalConstant>()?;
    m.add_class::<CorrectionLevel>()?;
    m.add_class::<PySymExpr>()?;

    // === Resonant Engine ===
    m.add_class::<ResonantTensor>()?;
    m.add_class::<ResonantEvolver>()?;
    m.add_class::<RESConfig>()?;
    m.add_class::<RESResult>()?;

    // === Phi-Residual Operations ===
    m.add_class::<resonant::PhiResidualMode>()?;
    m.add_function(wrap_pyfunction!(resonant::phi_residual, m)?)?;
    m.add_function(wrap_pyfunction!(resonant::phi_residual_relu, m)?)?;

    // === Golden Batch Normalization ===
    m.add_class::<resonant::GoldenNormMode>()?;
    m.add_function(wrap_pyfunction!(
        resonant::golden_norm::golden_batch_norm_1d_py,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        resonant::golden_norm::golden_batch_norm_2d_py,
        m
    )?)?;

    // === Syntonic Softmax ===
    m.add_class::<resonant::SyntonicSoftmaxMode>()?;
    m.add_class::<resonant::SyntonicSoftmaxState>()?;
    m.add_function(wrap_pyfunction!(resonant::syntonic_softmax_py, m)?)?;
    m.add_function(wrap_pyfunction!(
        resonant::syntonic_softmax::compute_syntonic_weights_py,
        m
    )?)?;

    // === Core Tensor Operations ===
    m.add_class::<TensorStorage>()?;
    #[cfg(feature = "cuda")]
    m.add_class::<AsyncTensorTransfer>()?;
    #[cfg(feature = "cuda")]
    m.add_class::<TransferComputeOverlap>()?;
    m.add_function(wrap_pyfunction!(cuda_is_available, m)?)?;
    m.add_function(wrap_pyfunction!(cuda_device_count, m)?)?;

    // === Hypercomplex Numbers ===
    m.add_class::<Quaternion>()?;
    m.add_class::<Octonion>()?;
    m.add_class::<Sedenion>()?;

    // === SRT Constants ===
    m.add_function(wrap_pyfunction!(srt_phi, m)?)?;
    m.add_function(wrap_pyfunction!(srt_phi_inv, m)?)?;
    m.add_function(wrap_pyfunction!(srt_q_deficit, m)?)?;
    m.add_function(wrap_pyfunction!(srt_pi, m)?)?;
    m.add_function(wrap_pyfunction!(srt_e, m)?)?;
    m.add_function(wrap_pyfunction!(srt_structure_dimension, m)?)?;
    m.add_function(wrap_pyfunction!(srt_correction_factor, m)?)?;

    // === Hierarchy Correction (SRT-Zero) ===
    m.add_function(wrap_pyfunction!(hierarchy::apply_correction, m)?)?;
    m.add_function(wrap_pyfunction!(hierarchy::apply_correction_uniform, m)?)?;
    m.add_function(wrap_pyfunction!(hierarchy::apply_special, m)?)?;
    m.add_function(wrap_pyfunction!(hierarchy::apply_suppression, m)?)?;
    m.add_function(wrap_pyfunction!(hierarchy::compute_e_star_n, m)?)?;
    m.add_function(wrap_pyfunction!(hierarchy::apply_chain, m)?)?;
    m.add_function(wrap_pyfunction!(hierarchy::init_divisors, m)?)?;
    // Extended hierarchy corrections
    m.add_function(wrap_pyfunction!(hierarchy::apply_e7_correction, m)?)?;
    m.add_function(wrap_pyfunction!(
        hierarchy::apply_collapse_threshold_correction,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        hierarchy::apply_coxeter_kissing_correction,
        m
    )?)?;
    // Extended hierarchy constants
    m.add_function(wrap_pyfunction!(hierarchy::e8_dim, m)?)?;
    m.add_function(wrap_pyfunction!(hierarchy::e7_dim, m)?)?;
    m.add_function(wrap_pyfunction!(hierarchy::e6_dim, m)?)?;
    m.add_function(wrap_pyfunction!(hierarchy::d4_dim, m)?)?;
    m.add_function(wrap_pyfunction!(hierarchy::d4_kissing, m)?)?;
    m.add_function(wrap_pyfunction!(hierarchy::g2_dim, m)?)?;
    m.add_function(wrap_pyfunction!(hierarchy::f4_dim, m)?)?;
    m.add_function(wrap_pyfunction!(hierarchy::coxeter_kissing_720, m)?)?;
    m.add_function(wrap_pyfunction!(hierarchy::hierarchy_exponent, m)?)?;

    // Extended hierarchy constants
    m.add_function(wrap_pyfunction!(hierarchy::e8_roots, m)?)?;
    m.add_function(wrap_pyfunction!(hierarchy::e8_positive_roots, m)?)?;
    m.add_function(wrap_pyfunction!(hierarchy::e8_rank, m)?)?;
    m.add_function(wrap_pyfunction!(hierarchy::e8_coxeter, m)?)?;
    m.add_function(wrap_pyfunction!(hierarchy::e7_roots, m)?)?;
    m.add_function(wrap_pyfunction!(hierarchy::e7_positive_roots, m)?)?;
    m.add_function(wrap_pyfunction!(hierarchy::e7_fundamental, m)?)?;
    m.add_function(wrap_pyfunction!(hierarchy::e7_rank, m)?)?;
    m.add_function(wrap_pyfunction!(hierarchy::e7_coxeter, m)?)?;
    m.add_function(wrap_pyfunction!(hierarchy::e6_roots, m)?)?;
    m.add_function(wrap_pyfunction!(hierarchy::e6_positive_roots, m)?)?;
    m.add_function(wrap_pyfunction!(hierarchy::e6_fundamental, m)?)?;
    m.add_function(wrap_pyfunction!(hierarchy::e6_rank, m)?)?;
    m.add_function(wrap_pyfunction!(hierarchy::e6_coxeter, m)?)?;
    m.add_function(wrap_pyfunction!(hierarchy::d4_rank, m)?)?;
    m.add_function(wrap_pyfunction!(hierarchy::d4_coxeter, m)?)?;
    m.add_function(wrap_pyfunction!(hierarchy::g2_rank, m)?)?;
    m.add_function(wrap_pyfunction!(hierarchy::f4_rank, m)?)?;

    // === GoldenGELU Activation ===
    m.add_function(wrap_pyfunction!(golden_gelu::golden_gelu_forward, m)?)?;
    m.add_function(wrap_pyfunction!(golden_gelu::golden_gelu_backward, m)?)?;
    m.add_function(wrap_pyfunction!(
        golden_gelu::batched_golden_gelu_forward,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(golden_gelu::get_golden_gelu_phi, m)?)?;

    // === SRT Tensor Operations (GPU-accelerated when on CUDA) ===
    m.add_function(wrap_pyfunction!(py_apply_geodesic_slide, m)?)?;
    m.add_function(wrap_pyfunction!(srt_scale_phi, m)?)?;
    m.add_function(wrap_pyfunction!(srt_golden_gaussian_weights, m)?)?;
    m.add_function(wrap_pyfunction!(srt_apply_correction, m)?)?;
    m.add_function(wrap_pyfunction!(srt_e8_batch_projection, m)?)?;
    m.add_function(wrap_pyfunction!(srt_theta_series, m)?)?;
    m.add_function(wrap_pyfunction!(srt_compute_syntony, m)?)?;
    m.add_function(wrap_pyfunction!(srt_dhsr_cycle, m)?)?;

    // === SRT Inflationary Broadcasting ===
    m.add_function(wrap_pyfunction!(py_inflationary_broadcast, m)?)?;
    m.add_function(wrap_pyfunction!(py_golden_inflationary_broadcast, m)?)?;
    m.add_function(wrap_pyfunction!(py_consciousness_inflationary_broadcast, m)?)?;

    // === Causal History DAG (DHSR Tracking) ===
    m.add_class::<PyCausalHistoryTracker>()?;
    m.add_function(wrap_pyfunction!(create_causal_tracker, m)?)?;
    m.add_function(wrap_pyfunction!(d4_consciousness_threshold, m)?)?;

    // === SRT Memory Transfer Statistics ===
    #[cfg(feature = "cuda")]
    m.add_function(wrap_pyfunction!(srt_transfer_stats, m)?)?;
    #[cfg(feature = "cuda")]
    m.add_function(wrap_pyfunction!(srt_reserve_memory, m)?)?;
    #[cfg(feature = "cuda")]
    m.add_function(wrap_pyfunction!(srt_wait_for_resonance, m)?)?;
    #[cfg(feature = "cuda")]
    m.add_function(wrap_pyfunction!(srt_pool_stats, m)?)?;
    #[cfg(feature = "cuda")]
    m.add_function(wrap_pyfunction!(srt_memory_resonance, m)?)?;
    #[cfg(feature = "cuda")]
    m.add_function(wrap_pyfunction!(
        tensor::storage::_debug_stress_pool_take,
        m
    )?)?;
    #[cfg(feature = "cuda")]
    m.add_function(wrap_pyfunction!(srt_kernels::validate_kernels, m)?)?;

    // === Winding State ===
    m.add_class::<WindingState>()?;
    m.add_class::<WindingStateIterator>()?;
    m.add_function(wrap_pyfunction!(enumerate_windings, m)?)?;
    m.add_function(wrap_pyfunction!(enumerate_windings_by_norm, m)?)?;
    m.add_function(wrap_pyfunction!(enumerate_windings_exact_norm, m)?)?;
    m.add_function(wrap_pyfunction!(count_windings, m)?)?;

    // === Spectral Operations ===
    m.add_function(wrap_pyfunction!(theta_series_evaluate, m)?)?;
    m.add_function(wrap_pyfunction!(theta_series_weighted, m)?)?;
    m.add_function(wrap_pyfunction!(theta_series_derivative, m)?)?;
    m.add_function(wrap_pyfunction!(heat_kernel_trace, m)?)?;
    m.add_function(wrap_pyfunction!(heat_kernel_weighted, m)?)?;
    m.add_function(wrap_pyfunction!(heat_kernel_derivative, m)?)?;
    m.add_function(wrap_pyfunction!(compute_eigenvalues, m)?)?;
    m.add_function(wrap_pyfunction!(compute_golden_weights, m)?)?;
    m.add_function(wrap_pyfunction!(compute_norm_squared, m)?)?;
    m.add_function(wrap_pyfunction!(spectral_zeta, m)?)?;
    m.add_function(wrap_pyfunction!(spectral_zeta_weighted, m)?)?;
    m.add_function(wrap_pyfunction!(partition_function, m)?)?;
    m.add_function(wrap_pyfunction!(theta_sum_combined, m)?)?;
    m.add_function(wrap_pyfunction!(count_by_generation, m)?)?;
    m.add_function(wrap_pyfunction!(filter_by_generation, m)?)?;

    // === Knot Laplacian Operations ===
    m.add_function(wrap_pyfunction!(knot_eigenvalue, m)?)?;
    m.add_function(wrap_pyfunction!(compute_knot_eigenvalues, m)?)?;
    m.add_function(wrap_pyfunction!(knot_heat_kernel_trace, m)?)?;
    m.add_function(wrap_pyfunction!(knot_spectral_zeta, m)?)?;
    m.add_function(wrap_pyfunction!(knot_spectral_zeta_complex, m)?)?;

    // === E8 Lattice and Golden Projector ===
    m.add_function(wrap_pyfunction!(py_e8_generate_weights, m)?)?;
    m.add_function(wrap_pyfunction!(py_e8_generate_roots, m)?)?;
    m.add_function(wrap_pyfunction!(py_golden_projector_q, m)?)?;
    m.add_function(wrap_pyfunction!(py_golden_projector_phi, m)?)?;
    m.add_function(wrap_pyfunction!(py_golden_project_parallel, m)?)?;
    m.add_function(wrap_pyfunction!(py_golden_project_perp, m)?)?;
    m.add_function(wrap_pyfunction!(py_is_in_golden_cone, m)?)?;
    m.add_function(wrap_pyfunction!(py_compute_8d_weight, m)?)?;

    // === Neural-network-friendly E8 wrappers ===
    m.add_function(wrap_pyfunction!(py_e8_generate_weights_nn, m)?)?;
    m.add_function(wrap_pyfunction!(py_e8_generate_roots_nn, m)?)?;
    m.add_function(wrap_pyfunction!(py_golden_projector_q_nn, m)?)?;
    m.add_function(wrap_pyfunction!(py_golden_projector_phi_nn, m)?)?;
    m.add_function(wrap_pyfunction!(py_golden_project_parallel_nn, m)?)?;
    m.add_function(wrap_pyfunction!(py_golden_project_perp_nn, m)?)?;
    m.add_function(wrap_pyfunction!(py_is_in_golden_cone_nn, m)?)?;
    m.add_function(wrap_pyfunction!(py_compute_8d_weight_nn, m)?)?;

    // === Number Theory and Syntony (Rust Performance Backend) ===
    m.add_function(wrap_pyfunction!(py_mobius, m)?)?;
    m.add_function(wrap_pyfunction!(py_is_square_free, m)?)?;
    m.add_function(wrap_pyfunction!(py_mertens, m)?)?;
    m.add_function(wrap_pyfunction!(py_golden_weight, m)?)?;
    m.add_function(wrap_pyfunction!(py_golden_weights, m)?)?;
    m.add_function(wrap_pyfunction!(py_e_star, m)?)?;
    m.add_function(wrap_pyfunction!(py_compute_winding_syntony, m)?)?;
    m.add_function(wrap_pyfunction!(py_batch_winding_syntony, m)?)?;
    m.add_function(wrap_pyfunction!(py_aggregate_syntony, m)?)?;
    m.add_function(wrap_pyfunction!(py_standard_mode_norms, m)?)?;

    // === SRT/CRT Prime Theory Functions ===
    m.add_function(wrap_pyfunction!(
        resonant::py_wrappers::py_is_mersenne_prime,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        resonant::py_wrappers::py_is_fermat_prime,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        resonant::py_wrappers::py_is_lucas_prime,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(resonant::py_wrappers::py_lucas_number, m)?)?;
    m.add_function(wrap_pyfunction!(
        resonant::py_wrappers::py_pisano_period,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        resonant::py_wrappers::py_is_stable_winding,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        resonant::py_wrappers::py_get_stability_barrier,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        resonant::py_wrappers::py_is_transcendence_gate,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        resonant::py_wrappers::py_versal_grip_strength,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        resonant::py_wrappers::py_mersenne_sequence,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        resonant::py_wrappers::py_fermat_sequence,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(resonant::py_wrappers::py_lucas_primes, m)?)?;
    m.add_function(wrap_pyfunction!(
        resonant::py_wrappers::py_lucas_dark_boost,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        resonant::py_wrappers::py_predict_dark_matter_mass,
        m
    )?)?;

    // === Crystallization Functions ===
    m.add_function(wrap_pyfunction!(py_crystallize_with_dwell_legacy, m)?)?;
    m.add_function(wrap_pyfunction!(py_snap_distance, m)?)?;
    m.add_function(wrap_pyfunction!(py_compute_snap_gradient, m)?)?;
    m.add_function(wrap_pyfunction!(
        resonant::py_wrappers::py_compute_snap_gradient_cuda,
        m
    )?)?;

    // === Loss Functions (Rust Performance Backend) ===
    m.add_function(wrap_pyfunction!(py_mse_loss, m)?)?;
    m.add_function(wrap_pyfunction!(py_softmax, m)?)?;
    m.add_function(wrap_pyfunction!(py_cross_entropy_loss, m)?)?;
    m.add_function(wrap_pyfunction!(py_batch_cross_entropy_loss, m)?)?;
    m.add_function(wrap_pyfunction!(py_syntony_loss, m)?)?;
    m.add_function(wrap_pyfunction!(py_syntony_loss_srt, m)?)?;
    m.add_function(wrap_pyfunction!(py_get_target_syntony, m)?)?;
    m.add_function(wrap_pyfunction!(py_get_q_deficit, m)?)?;
    m.add_function(wrap_pyfunction!(py_phase_alignment_loss, m)?)?;
    m.add_function(wrap_pyfunction!(py_syntonic_loss, m)?)?;
    m.add_function(wrap_pyfunction!(py_estimate_syntony_from_probs, m)?)?;
    m.add_function(wrap_pyfunction!(py_golden_decay_loss, m)?)?;

    // === Broadcasting Operations ===
    m.add_function(wrap_pyfunction!(py_broadcast_shape, m)?)?;
    m.add_function(wrap_pyfunction!(py_are_broadcastable, m)?)?;
    m.add_function(wrap_pyfunction!(py_broadcast_add, m)?)?;
    m.add_function(wrap_pyfunction!(py_broadcast_mul, m)?)?;
    m.add_function(wrap_pyfunction!(py_broadcast_sub, m)?)?;
    m.add_function(wrap_pyfunction!(py_broadcast_div, m)?)?;
    m.add_function(wrap_pyfunction!(py_linear_index, m)?)?;

    // === In-place Operations ===
    m.add_function(wrap_pyfunction!(py_inplace_add_scalar, m)?)?;
    m.add_function(wrap_pyfunction!(py_inplace_mul_scalar, m)?)?;
    m.add_function(wrap_pyfunction!(py_inplace_sub_scalar, m)?)?;
    m.add_function(wrap_pyfunction!(py_inplace_div_scalar, m)?)?;
    m.add_function(wrap_pyfunction!(py_inplace_negate, m)?)?;
    m.add_function(wrap_pyfunction!(py_inplace_abs, m)?)?;
    m.add_function(wrap_pyfunction!(py_inplace_clamp, m)?)?;
    m.add_function(wrap_pyfunction!(py_inplace_golden_weight, m)?)?;

    // === Convolution Operations ===
    m.add_function(wrap_pyfunction!(py_conv2d, m)?)?;
    m.add_function(wrap_pyfunction!(py_max_pool2d, m)?)?;
    m.add_function(wrap_pyfunction!(py_avg_pool2d, m)?)?;
    m.add_function(wrap_pyfunction!(py_global_avg_pool2d, m)?)?;

    // === Structure enum for correction factors ===
    m.add_class::<Structure>()?;

    // === Linear Algebra Operations ===
    // Core matmul
    m.add_function(wrap_pyfunction!(py_mm, m)?)?;
    m.add_function(wrap_pyfunction!(py_mm_add, m)?)?;
    // Transpose variants
    m.add_function(wrap_pyfunction!(py_mm_tn, m)?)?;
    m.add_function(wrap_pyfunction!(py_mm_nt, m)?)?;
    m.add_function(wrap_pyfunction!(py_mm_tt, m)?)?;
    // Hermitian variants
    m.add_function(wrap_pyfunction!(py_mm_hn, m)?)?;
    m.add_function(wrap_pyfunction!(py_mm_nh, m)?)?;
    // Batched
    m.add_function(wrap_pyfunction!(py_bmm, m)?)?;
    // SRT-specific operations
    m.add_function(wrap_pyfunction!(py_mm_phi, m)?)?;
    m.add_function(wrap_pyfunction!(py_phi_bracket, m)?)?;
    m.add_function(wrap_pyfunction!(py_phi_antibracket, m)?)?;
    m.add_function(wrap_pyfunction!(py_mm_corrected, m)?)?;
    m.add_function(wrap_pyfunction!(py_mm_golden_phase, m)?)?;
    m.add_function(wrap_pyfunction!(py_mm_golden_weighted, m)?)?;
    m.add_function(wrap_pyfunction!(py_projection_sum, m)?)?;
    // Generalized GEMM and q-correction operations
    m.add_function(wrap_pyfunction!(py_mm_gemm, m)?)?;
    m.add_function(wrap_pyfunction!(py_mm_q_corrected_direct, m)?)?;
    m.add_function(wrap_pyfunction!(py_q_correction_scalar, m)?)?;

    // === Prime Selection Rules ===
    register_extended_prime_selection(m)?;

    // === Gnosis/Consciousness Module ===
    register_gnosis(m)?;

    // === Transcendence Module ===
    register_transcendence(m)?;

    Ok(())
}
