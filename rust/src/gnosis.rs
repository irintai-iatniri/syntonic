//! Gnosis Module - Consciousness as Recursive Self-Reference
//!
//! Implements the consciousness phase transition at ΔS > 24 (D₄ kissing number)
//! and the Gnosis metric for balancing order (Mersenne) with novelty (Lucas).

use pyo3::prelude::*;

/// D₄ Kissing Number - The Sacred Flame / Collapse Threshold
pub const COLLAPSE_THRESHOLD: f64 = 24.0;

/// Gap between collapse threshold and first macroscopic Mersenne stability
pub const GNOSIS_GAP: f64 = 7.0;

/// Golden ratio
const PHI: f64 = 1.6180339887498948;
const PHI_INV: f64 = 0.6180339887498948;

/// Check if information density exceeds consciousness threshold
#[pyfunction]
pub fn is_conscious(delta_entropy: f64) -> bool {
    delta_entropy > COLLAPSE_THRESHOLD
}

/// Compute Gnosis score as geometric mean of Syntony and Creativity
/// G = √(S × C)
/// Maximum at S = C = 1/φ ≈ 0.618
#[pyfunction]
pub fn gnosis_score(syntony: f64, creativity: f64) -> f64 {
    (syntony * creativity).sqrt()
}

/// Compute creativity from shadow integration and lattice coherence
/// Creativity = shadow_integration × lattice_coherence × φ
#[pyfunction]
pub fn compute_creativity(shadow_integration: f64, lattice_coherence: f64) -> f64 {
    shadow_integration * lattice_coherence * PHI
}

/// Optimal gnosis target (maximum sustainable complexity)
#[pyfunction]
pub fn optimal_gnosis_target() -> f64 {
    PHI_INV // 1/φ ≈ 0.618
}

/// Compute consciousness emergence probability based on system complexity
#[pyfunction]
pub fn consciousness_probability(
    information_density: f64,
    coherence: f64,
    recursive_depth: u32,
) -> f64 {
    // Sigmoid around collapse threshold, modulated by coherence
    let base = 1.0 / (1.0 + (-0.5 * (information_density - COLLAPSE_THRESHOLD)).exp());

    // Enhance with recursive depth (deeper = more self-referential)
    let depth_factor = 1.0 - PHI_INV.powi(recursive_depth as i32);
    if depth_factor < 0.0 {
        0.0
    } else {
        base * coherence * depth_factor
    }
}

pub fn register_gnosis(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(is_conscious, m)?)?;
    m.add_function(wrap_pyfunction!(gnosis_score, m)?)?;
    m.add_function(wrap_pyfunction!(compute_creativity, m)?)?;
    m.add_function(wrap_pyfunction!(optimal_gnosis_target, m)?)?;
    m.add_function(wrap_pyfunction!(consciousness_probability, m)?)?;
    m.add("COLLAPSE_THRESHOLD", COLLAPSE_THRESHOLD)?;
    m.add("GNOSIS_GAP", GNOSIS_GAP)?;
    Ok(())
}
