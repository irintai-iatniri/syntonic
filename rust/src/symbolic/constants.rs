use pyo3::prelude::*;
use super::golden::GoldenNumber;

/// SRT fundamental constants expressed symbolically
/// These are the core parameters of Syntony Recursion Theory

/// E* = Spectral Möbius Constant = e^π - π ≈ 19.9990999792 (mass quantum)
pub const E_STAR: f64 = 19.99909997918947;

/// φ = (1 + √5)/2 ≈ 1.618033988749895 (golden ratio)
pub const PHI: f64 = 1.618033988749895;

/// φ² = φ + 1 ≈ 2.618033988749895 (golden square)
pub const PHI_SQ: f64 = 2.618033988749895;

/// φ_hat = 1/φ = φ - 1 ≈ 0.618033988749895 (coherence parameter)
pub const PHI_HAT: f64 = 0.618033988749895;

/// q = Syntony Deficit ≈ 0.027395146920
pub const Q_DEFICIT: f64 = 0.027395146920;

/// Python-accessible SRT constants module
#[pyclass]
pub struct SRTConstants;

#[pymethods]
impl SRTConstants {
    /// E* = Spectral Möbius Constant (≈ 19.99 MeV)
    #[staticmethod]
    pub fn e_star() -> f64 {
        E_STAR
    }

    /// φ_hat = 1/φ = Coherence Parameter (≈ 0.618)
    #[staticmethod]
    pub fn phi_hat() -> f64 {
        PHI_HAT
    }

    /// φ_hat as exact GoldenNumber: -1 + φ
    #[staticmethod]
    pub fn phi_hat_exact() -> GoldenNumber {
        GoldenNumber::new(-1.0, 1.0)
    }

    /// q = Syntony Deficit (≈ 0.027395)
    #[staticmethod]
    pub fn q() -> f64 {
        Q_DEFICIT
    }

    /// φ² = φ + 1 (≈ 2.618)
    #[staticmethod]
    pub fn phi_sq() -> f64 {
        PHI_SQ
    }

    /// φ² as exact GoldenNumber: 1 + φ
    #[staticmethod]
    pub fn phi_sq_exact() -> GoldenNumber {
        GoldenNumber::new(1.0, 1.0)
    }

    /// φ (golden ratio)
    #[staticmethod]
    pub fn phi() -> f64 {
        PHI
    }

    /// φ as exact GoldenNumber: 0 + 1*φ
    #[staticmethod]
    pub fn phi_exact() -> GoldenNumber {
        GoldenNumber::new(0.0, 1.0)
    }

    /// √5 ≈ 2.2360679774997896
    #[staticmethod]
    pub fn sqrt5() -> f64 {
        5.0_f64.sqrt()
    }

    /// Fibonacci(n) using Binet's formula (exact for small n)
    #[staticmethod]
    pub fn fibonacci(n: i32) -> i64 {
        if n <= 0 {
            return 0;
        }
        if n == 1 || n == 2 {
            return 1;
        }
        // Use matrix exponentiation for exact result
        let mut a: i64 = 1;
        let mut b: i64 = 1;
        for _ in 2..n {
            let temp = a + b;
            a = b;
            b = temp;
        }
        b
    }

    /// Lucas(n) sequence: L(n) = φ^n + (1-φ)^n
    #[staticmethod]
    pub fn lucas(n: i32) -> i64 {
        if n == 0 {
            return 2;
        }
        if n == 1 {
            return 1;
        }
        let mut a: i64 = 2;
        let mut b: i64 = 1;
        for _ in 1..n {
            let temp = a + b;
            a = b;
            b = temp;
        }
        b
    }

    /// The SRT winding energy: E(n) = n² * E*
    #[staticmethod]
    pub fn winding_energy(n: i32) -> f64 {
        (n * n) as f64 * E_STAR
    }

    /// The SRT syntony functional for layer l: S(l) = φ^(-l)
    #[staticmethod]
    pub fn syntony(layer: i32) -> f64 {
        PHI.powi(-layer)
    }
}
