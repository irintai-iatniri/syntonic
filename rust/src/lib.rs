use pyo3::prelude::*;

mod tensor;
mod hypercomplex;
mod symbolic;
mod exact;
mod linalg;

use tensor::storage::{TensorStorage, cuda_is_available, cuda_device_count};
use tensor::srt_kernels;
use hypercomplex::{Quaternion, Octonion};
use symbolic::{GoldenNumber, Expr, SRTConstants};

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

    // === Legacy types (to be phased out) ===
    m.add_class::<GoldenNumber>()?;  // Legacy: use GoldenExact instead
    m.add_class::<Expr>()?;           // Legacy: use SymExpr instead
    m.add_class::<SRTConstants>()?;   // Legacy: use FundamentalConstant instead

    // === SRT Constants ===
    m.add_function(wrap_pyfunction!(srt_phi, m)?)?;
    m.add_function(wrap_pyfunction!(srt_phi_inv, m)?)?;
    m.add_function(wrap_pyfunction!(srt_q_deficit, m)?)?;
    m.add_function(wrap_pyfunction!(srt_structure_dimension, m)?)?;
    m.add_function(wrap_pyfunction!(srt_correction_factor, m)?)?;

    Ok(())
}
