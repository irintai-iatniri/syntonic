//! Number-theoretic functions for SRT.
//!
//! Provides Möbius function, golden weights, and related computations
//! used throughout the Syntonic library.

use std::f64::consts::PI;

/// Golden ratio φ = (1 + √5) / 2
pub const PHI: f64 = 1.618033988749895;

/// Compute the Möbius function μ(n).
///
/// μ(n) = 1 if n is a product of an even number of distinct primes
/// μ(n) = -1 if n is a product of an odd number of distinct primes
/// μ(n) = 0 if n has a squared prime factor
///
/// # Arguments
/// * `n` - Positive integer
///
/// # Returns
/// μ(n) ∈ {-1, 0, 1}
///
/// # Example
/// ```
/// use syntonic_core::resonant::number_theory::mobius;
/// assert_eq!(mobius(1), 1);
/// assert_eq!(mobius(2), -1);  // prime
/// assert_eq!(mobius(4), 0);   // 2² has squared factor
/// assert_eq!(mobius(6), 1);   // 2×3 (two distinct primes)
/// ```
pub fn mobius(n: i64) -> i64 {
    if n <= 0 {
        return 0;
    }
    if n == 1 {
        return 1;
    }

    let mut temp = n;
    let mut num_factors = 0i64;
    let mut d = 2i64;

    while d * d <= temp {
        if temp % d == 0 {
            temp /= d;
            if temp % d == 0 {
                // Squared prime factor
                return 0;
            }
            num_factors += 1;
        }
        d += 1;
    }

    if temp > 1 {
        num_factors += 1;
    }

    if num_factors % 2 == 0 { 1 } else { -1 }
}

/// Check if n is square-free (has no squared prime factors).
///
/// # Arguments
/// * `n` - Positive integer
///
/// # Returns
/// true if n has no squared prime factors
pub fn is_square_free(n: i64) -> bool {
    mobius(n) != 0
}

/// Compute the Mertens function M(n) = Σ_{k=1}^n μ(k).
///
/// # Arguments
/// * `n` - Upper limit
///
/// # Returns
/// M(n)
pub fn mertens(n: usize) -> i64 {
    (1..=n as i64).map(mobius).sum()
}

/// Compute golden weight w(n) = exp(-|n|²/φ).
///
/// This is the fundamental weighting function in SRT that
/// preferentially weights low-norm modes.
///
/// # Arguments
/// * `mode_norm_sq` - Mode norm squared |n|²
///
/// # Returns
/// w(n) = exp(-|n|²/φ)
pub fn golden_weight(mode_norm_sq: f64) -> f64 {
    (-mode_norm_sq / PHI).exp()
}

/// Compute E* = e^π - π ≈ 20.1408...
///
/// This is the fundamental constant appearing in SRT spectral theory.
pub fn e_star() -> f64 {
    PI.exp() - PI
}

/// Batch compute golden weights for a vector of mode norms.
///
/// # Arguments
/// * `mode_norms` - Slice of mode norm squared values
///
/// # Returns
/// Vector of golden weights
pub fn golden_weights(mode_norms: &[f64]) -> Vec<f64> {
    mode_norms.iter().map(|&n| golden_weight(n)).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mobius_values() {
        // μ(1) = 1
        assert_eq!(mobius(1), 1);
        
        // Primes have μ(p) = -1
        assert_eq!(mobius(2), -1);
        assert_eq!(mobius(3), -1);
        assert_eq!(mobius(5), -1);
        assert_eq!(mobius(7), -1);
        
        // Perfect squares have μ(n) = 0
        assert_eq!(mobius(4), 0);
        assert_eq!(mobius(9), 0);
        assert_eq!(mobius(25), 0);
        
        // Products of two distinct primes have μ = 1
        assert_eq!(mobius(6), 1);   // 2×3
        assert_eq!(mobius(10), 1);  // 2×5
        assert_eq!(mobius(15), 1);  // 3×5
        
        // Products of three distinct primes have μ = -1
        assert_eq!(mobius(30), -1); // 2×3×5
    }

    #[test]
    fn test_is_square_free() {
        assert!(is_square_free(1));
        assert!(is_square_free(2));
        assert!(is_square_free(6));
        assert!(!is_square_free(4));
        assert!(!is_square_free(12)); // 4×3
    }

    #[test]
    fn test_mertens() {
        // M(1) = 1
        assert_eq!(mertens(1), 1);
        // M(2) = M(1) + μ(2) = 1 + (-1) = 0
        assert_eq!(mertens(2), 0);
        // M(3) = M(2) + μ(3) = 0 + (-1) = -1
        assert_eq!(mertens(3), -1);
    }

    #[test]
    fn test_golden_weight() {
        // w(0) = exp(0) = 1
        assert!((golden_weight(0.0) - 1.0).abs() < 1e-10);
        
        // w(φ) = exp(-1) ≈ 0.368
        let w = golden_weight(PHI);
        assert!((w - (-1.0_f64).exp()).abs() < 1e-10);
        
        // Decreasing with mode norm
        assert!(golden_weight(1.0) > golden_weight(2.0));
        assert!(golden_weight(2.0) > golden_weight(4.0));
    }

    #[test]
    fn test_e_star() {
        let e = e_star();
        // E* ≈ 20.1408...
        assert!((e - 20.140876).abs() < 0.001);
    }
}
