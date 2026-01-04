//! Spectral operations for SRT theta series and heat kernels.
//!
//! This module provides high-performance implementations of:
//! - Theta series evaluation: Θ(t) = Σ_n w(n) exp(-π|n|²/t)
//! - Heat kernel trace: K(t) = Σ_n exp(-λ_n t)
//! - Spectral zeta function: ζ(s) = Σ_n λ_n^(-s)
//! - Eigenvalue computations

use pyo3::prelude::*;
use std::f64::consts::PI;

use crate::tensor::srt_kernels::PHI;
use crate::winding::WindingState;

// =============================================================================
// Theta Series Operations
// =============================================================================

/// Compute theta series: Θ(t) = Σ_n w(n) exp(-π|n|²/t)
///
/// Args:
///     windings: List of WindingState instances
///     t: Temperature parameter
///
/// Returns:
///     Theta series value at t
#[pyfunction]
pub fn theta_series_evaluate(windings: Vec<WindingState>, t: f64) -> f64 {
    let mut sum = 0.0;

    for w in &windings {
        let norm_sq = w.norm_squared() as f64;
        // Golden weight: exp(-|n|²/φ)
        let weight = (-norm_sq / PHI).exp();
        // Theta term: exp(-π|n|²/t)
        let theta_term = (-PI * norm_sq / t).exp();
        sum += weight * theta_term;
    }

    sum
}

/// Compute theta series with custom weights.
///
/// Args:
///     windings: List of WindingState instances
///     weights: List of weights corresponding to each winding
///     t: Temperature parameter
///
/// Returns:
///     Weighted theta series value at t
#[pyfunction]
pub fn theta_series_weighted(windings: Vec<WindingState>, weights: Vec<f64>, t: f64) -> f64 {
    let mut sum = 0.0;

    for (w, weight) in windings.iter().zip(weights.iter()) {
        let norm_sq = w.norm_squared() as f64;
        let theta_term = (-PI * norm_sq / t).exp();
        sum += weight * theta_term;
    }

    sum
}

/// Compute theta series derivative: dΘ/dt = (π/t²) Σ_n w(n) |n|² exp(-π|n|²/t)
#[pyfunction]
pub fn theta_series_derivative(windings: Vec<WindingState>, t: f64) -> f64 {
    let mut sum = 0.0;
    let t_sq = t * t;

    for w in &windings {
        let norm_sq = w.norm_squared() as f64;
        let weight = (-norm_sq / PHI).exp();
        let theta_term = (-PI * norm_sq / t).exp();
        sum += weight * norm_sq * theta_term;
    }

    (PI / t_sq) * sum
}

// =============================================================================
// Heat Kernel Operations
// =============================================================================

/// Compute heat kernel trace: K(t) = Σ_n exp(-λ_n t)
///
/// Args:
///     windings: List of WindingState instances
///     t: Time parameter
///     base_eigenvalue: Base eigenvalue scale (λ_n = base * |n|²)
///
/// Returns:
///     Heat kernel trace at time t
#[pyfunction]
pub fn heat_kernel_trace(windings: Vec<WindingState>, t: f64, base_eigenvalue: f64) -> f64 {
    let mut sum = 0.0;

    for w in &windings {
        let norm_sq = w.norm_squared() as f64;
        let eigenvalue = base_eigenvalue * norm_sq;
        sum += (-eigenvalue * t).exp();
    }

    sum
}

/// Compute weighted heat kernel trace: K(t) = Σ_n w(n) exp(-λ_n t)
#[pyfunction]
pub fn heat_kernel_weighted(
    windings: Vec<WindingState>,
    t: f64,
    base_eigenvalue: f64,
) -> f64 {
    let mut sum = 0.0;

    for w in &windings {
        let norm_sq = w.norm_squared() as f64;
        let weight = (-norm_sq / PHI).exp();
        let eigenvalue = base_eigenvalue * norm_sq;
        sum += weight * (-eigenvalue * t).exp();
    }

    sum
}

/// Compute heat kernel derivative: dK/dt = -Σ_n λ_n exp(-λ_n t)
#[pyfunction]
pub fn heat_kernel_derivative(
    windings: Vec<WindingState>,
    t: f64,
    base_eigenvalue: f64,
) -> f64 {
    let mut sum = 0.0;

    for w in &windings {
        let norm_sq = w.norm_squared() as f64;
        let eigenvalue = base_eigenvalue * norm_sq;
        sum += eigenvalue * (-eigenvalue * t).exp();
    }

    -sum
}

// =============================================================================
// Eigenvalue Operations
// =============================================================================

/// Batch compute eigenvalues for all windings: λ_n = base * |n|²
///
/// Args:
///     windings: List of WindingState instances
///     base: Base eigenvalue scale
///
/// Returns:
///     List of eigenvalues
#[pyfunction]
pub fn compute_eigenvalues(windings: Vec<WindingState>, base: f64) -> Vec<f64> {
    windings
        .iter()
        .map(|w| base * (w.norm_squared() as f64))
        .collect()
}

/// Batch compute golden weights for all windings: w(n) = exp(-|n|²/φ)
#[pyfunction]
pub fn compute_golden_weights(windings: Vec<WindingState>) -> Vec<f64> {
    windings
        .iter()
        .map(|w| (-(w.norm_squared() as f64) / PHI).exp())
        .collect()
}

/// Batch compute norm squared for all windings
#[pyfunction]
pub fn compute_norm_squared(windings: Vec<WindingState>) -> Vec<i64> {
    windings.iter().map(|w| w.norm_squared()).collect()
}

// =============================================================================
// Spectral Zeta Function
// =============================================================================

/// Compute spectral zeta function: ζ(s) = Σ_{n≠0} λ_n^(-s)
///
/// Args:
///     windings: List of WindingState instances
///     s: Complex exponent (real part)
///     base_eigenvalue: Base eigenvalue scale
///
/// Returns:
///     Spectral zeta function value
#[pyfunction]
pub fn spectral_zeta(windings: Vec<WindingState>, s: f64, base_eigenvalue: f64) -> f64 {
    let mut sum = 0.0;

    for w in &windings {
        let norm_sq = w.norm_squared();
        if norm_sq > 0 {
            // Skip zero mode
            let eigenvalue = base_eigenvalue * (norm_sq as f64);
            sum += eigenvalue.powf(-s);
        }
    }

    sum
}

/// Compute weighted spectral zeta: ζ_w(s) = Σ_{n≠0} w(n) λ_n^(-s)
#[pyfunction]
pub fn spectral_zeta_weighted(
    windings: Vec<WindingState>,
    s: f64,
    base_eigenvalue: f64,
) -> f64 {
    let mut sum = 0.0;

    for w in &windings {
        let norm_sq = w.norm_squared();
        if norm_sq > 0 {
            let weight = (-(norm_sq as f64) / PHI).exp();
            let eigenvalue = base_eigenvalue * (norm_sq as f64);
            sum += weight * eigenvalue.powf(-s);
        }
    }

    sum
}

// =============================================================================
// Partition Function
// =============================================================================

/// Compute partition function: Z = Σ_n exp(-|n|²/φ)
#[pyfunction]
pub fn partition_function(windings: Vec<WindingState>) -> f64 {
    windings
        .iter()
        .map(|w| (-(w.norm_squared() as f64) / PHI).exp())
        .sum()
}

/// Compute combined theta sum: Θ_c(t) = Σ_n exp(-|n|² * (1/φ + π/t))
///
/// This efficiently computes the product of golden measure and theta series.
#[pyfunction]
pub fn theta_sum_combined(windings: Vec<WindingState>, t: f64) -> f64 {
    let combined_factor = 1.0 / PHI + PI / t;

    windings
        .iter()
        .map(|w| (-(w.norm_squared() as f64) * combined_factor).exp())
        .sum()
}

// =============================================================================
// Knot Laplacian Operations
// =============================================================================

/// Compute knot Laplacian eigenvalue: λ_n = base * |n|² * (1 + φ^(-|n|²))
///
/// This includes the golden-weighted knot potential correction.
#[pyfunction]
pub fn knot_eigenvalue(norm_squared: i64, base: f64) -> f64 {
    let n_sq = norm_squared as f64;
    let correction = PHI.powf(-n_sq);
    base * n_sq * (1.0 + correction)
}

/// Batch compute knot eigenvalues for all windings
#[pyfunction]
pub fn compute_knot_eigenvalues(windings: Vec<WindingState>, base: f64) -> Vec<f64> {
    windings
        .iter()
        .map(|w| {
            let n_sq = w.norm_squared() as f64;
            let correction = PHI.powf(-n_sq);
            base * n_sq * (1.0 + correction)
        })
        .collect()
}

/// Compute knot heat kernel trace: K(t) = Σ_n exp(-λ_knot_n * t)
///
/// Uses knot eigenvalues λ_n = base * |n|² * (1 + φ^(-|n|²))
#[pyfunction]
pub fn knot_heat_kernel_trace(windings: Vec<WindingState>, t: f64, base_eigenvalue: f64) -> f64 {
    let mut sum = 0.0;

    for w in &windings {
        let n_sq = w.norm_squared() as f64;
        let correction = PHI.powf(-n_sq);
        let eigenvalue = base_eigenvalue * n_sq * (1.0 + correction);
        sum += (-eigenvalue * t).exp();
    }

    sum
}

/// Compute knot spectral zeta: ζ(s) = Σ_{n≠0} λ_knot_n^(-s)
///
/// Uses knot eigenvalues λ_n = base * |n|² * (1 + φ^(-|n|²))
#[pyfunction]
pub fn knot_spectral_zeta(windings: Vec<WindingState>, s: f64, base_eigenvalue: f64) -> f64 {
    let mut sum = 0.0;

    for w in &windings {
        let n_sq = w.norm_squared();
        if n_sq > 0 {
            let n_sq_f = n_sq as f64;
            let correction = PHI.powf(-n_sq_f);
            let eigenvalue = base_eigenvalue * n_sq_f * (1.0 + correction);
            sum += eigenvalue.powf(-s);
        }
    }

    sum
}

/// Compute knot spectral zeta with complex s parameter
#[pyfunction]
pub fn knot_spectral_zeta_complex(
    windings: Vec<WindingState>,
    s_real: f64,
    s_imag: f64,
    base_eigenvalue: f64,
) -> (f64, f64) {
    let mut sum_real = 0.0;
    let mut sum_imag = 0.0;

    for w in &windings {
        let n_sq = w.norm_squared();
        if n_sq > 0 {
            let n_sq_f = n_sq as f64;
            let correction = PHI.powf(-n_sq_f);
            let eigenvalue = base_eigenvalue * n_sq_f * (1.0 + correction);

            // λ^(-s) = exp(-s * ln(λ)) = exp(-(s_r + i*s_i) * ln(λ))
            let ln_lambda = eigenvalue.ln();
            let exp_arg_real = -s_real * ln_lambda;
            let exp_arg_imag = -s_imag * ln_lambda;

            let magnitude = exp_arg_real.exp();
            sum_real += magnitude * exp_arg_imag.cos();
            sum_imag += magnitude * exp_arg_imag.sin();
        }
    }

    (sum_real, sum_imag)
}

// =============================================================================
// Generation Statistics
// =============================================================================

/// Count windings by generation.
///
/// Returns a HashMap mapping generation number to count.
#[pyfunction]
pub fn count_by_generation(windings: Vec<WindingState>) -> std::collections::HashMap<i64, usize> {
    let mut result = std::collections::HashMap::new();

    for w in &windings {
        let gen = w.generation();
        *result.entry(gen).or_insert(0) += 1;
    }

    result
}

/// Filter windings by generation.
#[pyfunction]
pub fn filter_by_generation(windings: Vec<WindingState>, generation: i64) -> Vec<WindingState> {
    windings
        .into_iter()
        .filter(|w| w.generation() == generation)
        .collect()
}
