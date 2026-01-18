//! Hierarchy correction module for SRT-Zero
//!
//! Provides Rust interface for:
//! - Batched correction application
//! - Special corrections (q²/φ, q·φ, 4q, etc.)
//! - Suppression factors
//! - Nested correction chains
//! - E*×N batch computation
//!
//! Note: CUDA backend pending cudarc API fixes. CPU fallback always active.

use crate::tensor::srt_kernels::{PHI, Q_DEFICIT};

// Compute derived phi powers locally
const PHI_SQUARED: f64 = PHI * PHI;
const PHI_CUBED: f64 = PHI_SQUARED * PHI;

// E_STAR = e^π - π (computed at runtime for now since const f64::pow is not const-fn)
fn e_star() -> f64 {
    std::f64::consts::E.powf(std::f64::consts::PI) - std::f64::consts::PI
}

// ============================================================================
// Extended Structure Dimensions (E₈ → E₇ → E₆ → SM Chain)
// ============================================================================

// E₈ Family
pub const E8_DIM: i32 = 248;
pub const E8_ROOTS: i32 = 240;
pub const E8_POSITIVE_ROOTS: i32 = 120;
pub const E8_RANK: i32 = 8;
pub const E8_COXETER: i32 = 30;

// E₇ Family (Intermediate Unification Scale)
pub const E7_DIM: i32 = 133;
pub const E7_ROOTS: i32 = 126;
pub const E7_POSITIVE_ROOTS: i32 = 63;
pub const E7_FUNDAMENTAL: i32 = 56;
pub const E7_RANK: i32 = 7;
pub const E7_COXETER: i32 = 18;

// E₆ Family
pub const E6_DIM: i32 = 78;
pub const E6_ROOTS: i32 = 72;
pub const E6_POSITIVE_ROOTS: i32 = 36;
pub const E6_FUNDAMENTAL: i32 = 27;
pub const E6_RANK: i32 = 6;
pub const E6_COXETER: i32 = 12;

// D₄ Family (SO(8) with Triality)
pub const D4_DIM: i32 = 28;
pub const D4_KISSING: i32 = 24; // Collapse threshold!
pub const D4_RANK: i32 = 4;
pub const D4_COXETER: i32 = 6;

// G₂ (Octonion Automorphisms)
pub const G2_DIM: i32 = 14;
pub const G2_RANK: i32 = 2;

// F₄ (Jordan Algebra Structure)
pub const F4_DIM: i32 = 52;
pub const F4_RANK: i32 = 4;

// Coxeter-Kissing Products
pub const COXETER_KISSING_720: i32 = E8_COXETER * D4_KISSING; // 30 × 24 = 720
pub const HIERARCHY_EXPONENT: i32 = COXETER_KISSING_720 - 1; // 719

use pyo3::prelude::*;

// =============================================================================
// PYTHON WRAPPERS
// =============================================================================

/// Apply a single standard correction: value * (1 ± q/divisor)
///
/// Args:
///   values: List of float64 values to correct
///   divisors: List of divisors (one per value)
///   signs: List of signs (+1 or -1, one per value)
///
/// Returns:
///   Corrected values
#[pyfunction]
#[pyo3(name = "hierarchy_apply_correction")]
pub fn apply_correction(
    values: Vec<f64>,
    divisors: Vec<f64>,
    signs: Vec<i32>,
) -> PyResult<Vec<f64>> {
    if values.len() != divisors.len() || values.len() != signs.len() {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "values, divisors, and signs must have same length",
        ));
    }

    let mut outputs = Vec::with_capacity(values.len());
    for i in 0..values.len() {
        let value = values[i];
        let divisor = divisors[i];
        let sign = signs[i];

        let factor = if divisor != 0.0 {
            1.0 + sign as f64 * Q_DEFICIT / divisor
        } else {
            1.0
        };

        outputs.push(value * factor);
    }
    Ok(outputs)
}

/// Apply a single standard correction with uniform divisor
///
/// Args:
///   values: List of float64 values to correct
///   divisor: Single divisor to apply to all values
///   sign: +1 or -1
///
/// Returns:
///   Corrected values
#[pyfunction]
#[pyo3(name = "hierarchy_apply_correction_uniform")]
pub fn apply_correction_uniform(values: Vec<f64>, divisor: f64, sign: i32) -> PyResult<Vec<f64>> {
    let factor = if divisor != 0.0 {
        1.0 + sign as f64 * Q_DEFICIT / divisor
    } else {
        1.0
    };
    Ok(values.iter().map(|&v| v * factor).collect())
}

/// Apply special corrections (q²/φ, q·φ, 4q, etc.)
///
/// Types:
///   0: q_phi_plus, 1: q_phi_minus, 2: q_phi_squared_plus, 3: q_phi_squared_minus
///   4: q_phi_cubed_plus, 5: q_phi_cubed_minus, 6: q_phi_fourth_plus, 7: q_phi_fourth_minus
///   8: q_phi_fifth_plus, 9: q_phi_fifth_minus
///   10: q_squared_plus, 11: q_squared_minus
///   12: q_squared_phi_plus, 13: q_squared_phi_minus
///   14: q_sq_phi_sq_plus, 15: q_sq_phi_sq_minus, 16: q_sq_phi_plus
///   17: 4q_plus, 18: 4q_minus, 19: 3q_plus, 20: 3q_minus
///   21: 6q_plus, 22: 8q_plus, 23: pi_q_plus
///   24: q_cubed, 25: q_phi_div_4pi_plus, 26: 8q_inv_plus, 27: q_squared_half_plus
///   28: q_6pi_plus, 29: q_phi_inv_plus
#[pyfunction]
#[pyo3(name = "hierarchy_apply_special")]
pub fn apply_special(values: Vec<f64>, types: Vec<i32>) -> PyResult<Vec<f64>> {
    if values.len() != types.len() {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "values and types must have same length",
        ));
    }

    let q = Q_DEFICIT;
    let phi = PHI;
    let phi_sq = PHI_SQUARED;
    let phi_cubed = PHI_CUBED;
    let phi_fourth = phi_sq * phi_sq;
    let phi_fifth = phi_fourth * phi;
    let pi = std::f64::consts::PI;

    let mut outputs = Vec::with_capacity(values.len());
    for i in 0..values.len() {
        let value = values[i];
        let type_ = types[i];
        let factor = match type_ {
            0 => 1.0 + q * phi,               // q_phi_plus
            1 => 1.0 - q * phi,               // q_phi_minus
            2 => 1.0 + q * phi_sq,            // q_phi_squared_plus
            3 => 1.0 - q * phi_sq,            // q_phi_squared_minus
            4 => 1.0 + q * phi_cubed,         // q_phi_cubed_plus
            5 => 1.0 - q * phi_cubed,         // q_phi_cubed_minus
            6 => 1.0 + q * phi_fourth,        // q_phi_fourth_plus
            7 => 1.0 - q * phi_fourth,        // q_phi_fourth_minus
            8 => 1.0 + q * phi_fifth,         // q_phi_fifth_plus
            9 => 1.0 - q * phi_fifth,         // q_phi_fifth_minus
            10 => 1.0 + q * q,                // q_squared_plus
            11 => 1.0 - q * q,                // q_squared_minus
            12 => 1.0 + q * q / phi,          // q_squared_phi_plus
            13 => 1.0 - q * q / phi,          // q_squared_phi_minus
            14 => 1.0 + q * q / phi_sq,       // q_sq_phi_sq_plus
            15 => 1.0 - q * q / phi_sq,       // q_sq_phi_sq_minus
            16 => 1.0 + q * q * phi,          // q_sq_phi_plus
            17 => 1.0 + 4.0 * q,              // 4q_plus
            18 => 1.0 - 4.0 * q,              // 4q_minus
            19 => 1.0 + 3.0 * q,              // 3q_plus
            20 => 1.0 - 3.0 * q,              // 3q_minus
            21 => 1.0 + 6.0 * q,              // 6q_plus
            22 => 1.0 + 8.0 * q,              // 8q_plus
            23 => 1.0 + pi * q,               // pi_q_plus
            24 => 1.0 + q * q * q,            // q_cubed
            25 => 1.0 + q * phi / (4.0 * pi), // q_phi_div_4pi_plus
            26 => 1.0 + q / 8.0,              // 8q_inv_plus
            27 => 1.0 + q * q / 2.0,          // q_squared_half_plus
            28 => 1.0 + q / (6.0 * pi),       // q_6pi_plus
            29 => 1.0 + q / phi,              // q_phi_inv_plus
            _ => 1.0,
        };
        outputs.push(value * factor);
    }
    Ok(outputs)
}

/// Apply suppression factors
///
/// Types:
///   0: winding_instability (1/(1+q/φ))
///   1: recursion_penalty (1/(1+q·φ))
///   2: double_inverse (1/(1+q/φ²))
///   3: fixed_point_penalty (1/(1+q·φ²))
#[pyfunction]
#[pyo3(name = "hierarchy_apply_suppression")]
pub fn apply_suppression(values: Vec<f64>, suppression_type: i32) -> PyResult<Vec<f64>> {
    let q = Q_DEFICIT;
    let phi = PHI;
    let phi_sq = PHI_SQUARED;

    let factor = match suppression_type {
        0 => 1.0 / (1.0 + q / phi),    // winding_instability
        1 => 1.0 / (1.0 + q * phi),    // recursion_penalty
        2 => 1.0 / (1.0 + q / phi_sq), // double_inverse
        3 => 1.0 / (1.0 + q * phi_sq), // fixed_point_penalty
        _ => 1.0,
    };
    Ok(values.iter().map(|&v| v * factor).collect())
}

/// Compute E*×N with corrections for a batch of values
///
/// Args:
///   N: List of N multipliers
///   divisors: Flat list of divisors for all corrections
///   signs: Flat list of signs for all corrections
///   n_corrections_per_value: Number of corrections to apply to each value
///
/// Returns:
///   Computed values: E* × N × ∏(1 ± q/divisor)
#[pyfunction]
#[pyo3(name = "hierarchy_compute_e_star_n")]
pub fn compute_e_star_n(
    n: Vec<f64>,
    divisors: Vec<f64>,
    signs: Vec<i32>,
    n_corrections_per_value: usize,
) -> PyResult<Vec<f64>> {
    if divisors.len() != signs.len() {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "divisors and signs must have same length",
        ));
    }
    if divisors.len() != n.len() * n_corrections_per_value {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "divisors length must equal N.len() * n_corrections_per_value",
        ));
    }

    let mut outputs = Vec::with_capacity(n.len());
    for (i, &n_val) in n.iter().enumerate() {
        let mut value = e_star() * n_val;

        for j in 0..n_corrections_per_value {
            let corr_idx = i * n_corrections_per_value + j;
            let divisor = divisors[corr_idx];
            let sign = signs[corr_idx];

            if divisor != 0.0 {
                let factor = 1.0 + sign as f64 * Q_DEFICIT / divisor;
                value *= factor;
            }
        }

        outputs.push(value);
    }
    Ok(outputs)
}

/// Apply nested correction chain (supports varying length chains per value)
///
/// Args:
///   values: List of values to correct
///   divisors: Flat list of all divisors for all chains
///   signs: Flat list of all signs for all chains
///   chain_lengths: Length of correction chain for each value
///   chain_starts: Starting index in divisors/signs for each value's chain
///
/// Returns:
///   Corrected values
#[pyfunction]
#[pyo3(name = "hierarchy_apply_chain")]
pub fn apply_chain(
    values: Vec<f64>,
    divisors: Vec<f64>,
    signs: Vec<i32>,
    chain_lengths: Vec<i32>,
    chain_starts: Vec<i32>,
) -> PyResult<Vec<f64>> {
    let q = Q_DEFICIT;

    let mut outputs = Vec::with_capacity(values.len());
    for (i, &value) in values.iter().enumerate() {
        let mut val = value;
        let chain_len = chain_lengths[i] as usize;
        let chain_start = chain_starts[i] as usize;

        for j in 0..chain_len {
            let corr_idx = chain_start + j;
            if corr_idx >= divisors.len() {
                break;
            }

            let divisor = divisors[corr_idx];
            let sign = signs[corr_idx];

            if divisor != 0.0 {
                let factor = 1.0 + sign as f64 * q / divisor;
                val *= factor;
            }
        }

        outputs.push(val);
    }
    Ok(outputs)
}

// ============================================================================
// Extended Correction Factor Functions
// ============================================================================

/// Apply correction with E₇ structure
#[pyfunction]
#[pyo3(name = "hierarchy_apply_e7_correction")]
pub fn apply_e7_correction(value: f64, structure_index: i32) -> f64 {
    let divisor = match structure_index {
        0 => E7_DIM,            // 133
        1 => E7_ROOTS,          // 126
        2 => E7_POSITIVE_ROOTS, // 63
        3 => E7_FUNDAMENTAL,    // 56
        4 => E7_RANK,           // 7
        5 => E7_COXETER,        // 18
        _ => return value,      // No correction
    };

    value * (1.0 + Q_DEFICIT / (divisor as f64))
}

/// Apply D₄ collapse threshold correction
#[pyfunction]
#[pyo3(name = "hierarchy_apply_collapse_threshold_correction")]
pub fn apply_collapse_threshold_correction(value: f64) -> f64 {
    value * (1.0 + Q_DEFICIT / (D4_KISSING as f64))
}

/// Apply Coxeter-Kissing product correction (720)
#[pyfunction]
#[pyo3(name = "hierarchy_apply_coxeter_kissing_correction")]
pub fn apply_coxeter_kissing_correction(value: f64) -> f64 {
    value * (1.0 + Q_DEFICIT / (COXETER_KISSING_720 as f64))
}

// ============================================================================
// Extended Hierarchy Constant Access Functions
// ============================================================================

/// Get E₈ dimension (248)
#[pyfunction]
#[pyo3(name = "hierarchy_e8_dim")]
pub fn e8_dim() -> i32 {
    E8_DIM
}

/// Get E₇ dimension (133)
#[pyfunction]
#[pyo3(name = "hierarchy_e7_dim")]
pub fn e7_dim() -> i32 {
    E7_DIM
}

/// Get E₆ dimension (78)
#[pyfunction]
#[pyo3(name = "hierarchy_e6_dim")]
pub fn e6_dim() -> i32 {
    E6_DIM
}

/// Get D₄ dimension (28)
#[pyfunction]
#[pyo3(name = "hierarchy_d4_dim")]
pub fn d4_dim() -> i32 {
    D4_DIM
}

/// Get D₄ kissing number (24) - consciousness threshold
#[pyfunction]
#[pyo3(name = "hierarchy_d4_kissing")]
pub fn d4_kissing() -> i32 {
    D4_KISSING
}

/// Get G₂ dimension (14)
#[pyfunction]
#[pyo3(name = "hierarchy_g2_dim")]
pub fn g2_dim() -> i32 {
    G2_DIM
}

/// Get F₄ dimension (52)
#[pyfunction]
#[pyo3(name = "hierarchy_f4_dim")]
pub fn f4_dim() -> i32 {
    F4_DIM
}

/// Get Coxeter-Kissing product (720)
#[pyfunction]
#[pyo3(name = "hierarchy_coxeter_kissing_720")]
pub fn coxeter_kissing_720() -> i32 {
    COXETER_KISSING_720
}

/// Get hierarchy exponent (719)
#[pyfunction]
#[pyo3(name = "hierarchy_exponent")]
pub fn hierarchy_exponent() -> i32 {
    HIERARCHY_EXPONENT
}

/// Initialize geometric divisors in constant memory
///
/// Args:
///   divisors: List of 84 divisors matching hierarchy.py GEOMETRIC_DIVISORS
#[pyfunction]
#[pyo3(name = "hierarchy_init_divisors")]
pub fn init_divisors(_divisors: Vec<f64>) -> PyResult<()> {
    // Placeholder for CUDA constant memory initialization
    // CPU implementation doesn't need constant memory
    Ok(())
}
