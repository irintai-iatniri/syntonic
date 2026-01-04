use pyo3::prelude::*;

mod tensor;
mod hypercomplex;
mod exact;
mod linalg;
mod winding;
mod spectral;

use tensor::storage::{TensorStorage, cuda_is_available, cuda_device_count};
use tensor::srt_kernels;
use hypercomplex::{Quaternion, Octonion};

// Winding state and enumeration
use winding::{
    WindingState, WindingStateIterator,
    enumerate_windings, enumerate_windings_by_norm, enumerate_windings_exact_norm, count_windings,
};

// Spectral operations
use spectral::{
    theta_series_evaluate, theta_series_weighted, theta_series_derivative,
    heat_kernel_trace, heat_kernel_weighted, heat_kernel_derivative,
    compute_eigenvalues, compute_golden_weights, compute_norm_squared,
    spectral_zeta, spectral_zeta_weighted,
    partition_function, theta_sum_combined,
    count_by_generation, filter_by_generation,
    // Knot Laplacian operations
    knot_eigenvalue, compute_knot_eigenvalues,
    knot_heat_kernel_trace, knot_spectral_zeta, knot_spectral_zeta_complex,
};

// New exact arithmetic types
use exact::{
    Rational,
    GoldenExact,
    FundamentalConstant,
    CorrectionLevel,
    PySymExpr,
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

    // === Core Tensor Operations ===
    m.add_class::<TensorStorage>()?;
    m.add_function(wrap_pyfunction!(cuda_is_available, m)?)?;
    m.add_function(wrap_pyfunction!(cuda_device_count, m)?)?;

    // === Hypercomplex Numbers ===
    m.add_class::<Quaternion>()?;
    m.add_class::<Octonion>()?;

    // === SRT Constants ===
    m.add_function(wrap_pyfunction!(srt_phi, m)?)?;
    m.add_function(wrap_pyfunction!(srt_phi_inv, m)?)?;
    m.add_function(wrap_pyfunction!(srt_q_deficit, m)?)?;
    m.add_function(wrap_pyfunction!(srt_structure_dimension, m)?)?;
    m.add_function(wrap_pyfunction!(srt_correction_factor, m)?)?;

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

    Ok(())
}
