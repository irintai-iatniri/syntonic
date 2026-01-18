//! Prime Selection Rules for SRT Physics
//!
//! Implements the number-theoretic selection rules that determine:
//! - Gauge forces (Fermat primes)
//! - Matter stability (Mersenne primes)
//! - Dark sector (Lucas primes)

use pyo3::prelude::*;

// ============================================================================
// Constants
// ============================================================================

/// Fermat primes F_n = 2^(2^n) + 1 for n = 0..4
pub const FERMAT_PRIMES: [u64; 5] = [3, 5, 17, 257, 65537];

/// Mersenne prime exponents where M_p = 2^p - 1 is prime
pub const MERSENNE_EXPONENTS: [u32; 8] = [2, 3, 5, 7, 13, 17, 19, 31];

/// First 20 Lucas numbers L_n = φ^n + (φ')^n
pub const LUCAS_SEQUENCE: [u64; 20] = [
    2, 1, 3, 4, 7, 11, 18, 29, 47, 76, 123, 199, 322, 521, 843, 1364, 2207, 3571, 5778, 9349,
];

/// Lucas primes (L_n where L_n is prime)
pub const LUCAS_PRIMES: [u64; 10] = [2, 3, 7, 11, 29, 47, 199, 521, 2207, 3571];

/// The M_11 barrier: 2^11 - 1 = 2047 = 23 × 89 (composite)
pub const M11_BARRIER: u64 = 2047;

/// Generation barrier factors
pub const M11_FACTOR_1: u64 = 23;
pub const M11_FACTOR_2: u64 = 89;

// ============================================================================
// Fermat Prime Functions (Gauge Forces)
// ============================================================================

/// Compute Fermat number F_n = 2^(2^n) + 1
#[pyfunction]
pub fn fermat_number(n: u32) -> PyResult<u64> {
    if n > 5 {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "Fermat numbers for n > 5 are too large for u64",
        ));
    }
    let exp = 1u64 << n; // 2^n
    Ok((1u64 << exp) + 1) // 2^(2^n) + 1
}

/// Check if Fermat number F_n is prime (valid gauge force)
/// Returns true for n ∈ {0, 1, 2, 3, 4}, false for n ≥ 5
#[pyfunction]
pub fn is_fermat_prime(n: u32) -> bool {
    n <= 4 // F_0 through F_4 are prime; F_5+ are composite
}

/// Get force spectrum information
#[pyfunction]
pub fn get_force_spectrum() -> Vec<(u32, String, u64, String)> {
    vec![
        (
            0,
            "Strong".to_string(),
            3,
            "SU(3) Color - Trinity".to_string(),
        ),
        (
            1,
            "Electroweak".to_string(),
            5,
            "Symmetry Breaking - Pentagon".to_string(),
        ),
        (
            2,
            "Dark Boundary".to_string(),
            17,
            "Topological Firewall".to_string(),
        ),
        (
            3,
            "Gravity".to_string(),
            257,
            "Geometric Container - 2^8 spinor".to_string(),
        ),
        (
            4,
            "Versal".to_string(),
            65537,
            "Syntonic Repulsion".to_string(),
        ),
    ]
}

// ============================================================================
// Mersenne Prime Functions (Matter Stability)
// ============================================================================

/// Compute Mersenne number M_p = 2^p - 1
#[pyfunction]
pub fn mersenne_number(p: u32) -> u64 {
    (1u64 << p) - 1
}

/// Lucas-Lehmer primality test for Mersenne numbers
/// M_p is prime iff s_{p-2} ≡ 0 (mod M_p) where s_0=4, s_i = s_{i-1}^2 - 2
#[pyfunction]
pub fn is_mersenne_prime(p: u32) -> bool {
    if p == 2 {
        return true; // M_2 = 3 is prime
    }
    if p < 2 {
        return false;
    }

    let mp = mersenne_number(p);
    let mut s: u128 = 4;

    for _ in 0..(p - 2) {
        s = (s * s - 2) % (mp as u128);
    }

    s == 0
}

/// Get generation spectrum information
#[pyfunction]
pub fn get_generation_spectrum() -> Vec<(u32, String, u64, Vec<String>)> {
    vec![
        (
            2,
            "Generation 1".to_string(),
            3,
            vec!["Electron".to_string(), "Up".to_string(), "Down".to_string()],
        ),
        (
            3,
            "Generation 2".to_string(),
            7,
            vec![
                "Muon".to_string(),
                "Charm".to_string(),
                "Strange".to_string(),
            ],
        ),
        (
            5,
            "Generation 3".to_string(),
            31,
            vec!["Tau".to_string(), "Bottom".to_string()],
        ),
        (
            7,
            "Heavy Anchor".to_string(),
            127,
            vec!["Top".to_string(), "Higgs VEV".to_string()],
        ),
    ]
}

/// Explain why there's no 4th generation
#[pyfunction]
pub fn generation_barrier_explanation() -> String {
    format!(
        "M_11 = 2^11 - 1 = {} = {} × {} (composite)\n\
         The geometry at winding depth 11 factorizes into modes {} and {}.\n\
         No stable fermion can exist at the 4th generation.\n\
         This is the M_11 Barrier.",
        M11_BARRIER, M11_FACTOR_1, M11_FACTOR_2, M11_FACTOR_1, M11_FACTOR_2
    )
}

// ============================================================================
// Lucas Shadow Functions (Dark Sector)
// ============================================================================

/// Compute Lucas number L_n iteratively
#[pyfunction]
pub fn lucas_number(n: u32) -> u64 {
    if n == 0 {
        return 2;
    }
    if n == 1 {
        return 1;
    }

    let mut a: u64 = 2;
    let mut b: u64 = 1;

    for _ in 1..n {
        let temp = a + b;
        a = b;
        b = temp;
    }

    b
}

/// Compute shadow phase (1-φ)^n
#[pyfunction]
pub fn shadow_phase(n: i32) -> f64 {
    const PHI_CONJUGATE: f64 = -0.6180339887498948; // 1 - φ
    PHI_CONJUGATE.powi(n)
}

/// Check if a Lucas number is prime
#[pyfunction]
pub fn is_lucas_prime(n: u32) -> bool {
    let ln = lucas_number(n);
    is_prime_u64(ln)
}

/// Simple primality test for u64
fn is_prime_u64(n: u64) -> bool {
    if n < 2 {
        return false;
    }
    if n == 2 || n == 3 {
        return true;
    }
    if n % 2 == 0 || n % 3 == 0 {
        return false;
    }

    let mut i = 5u64;
    while i * i <= n {
        if n % i == 0 || n % (i + 2) == 0 {
            return false;
        }
        i += 6;
    }
    true
}

/// Dark matter mass prediction from Lucas boost
#[pyfunction]
pub fn dark_matter_mass_prediction() -> (f64, String) {
    let l17 = lucas_number(17) as f64; // 3571
    let l13 = lucas_number(13) as f64; // 521
    let lucas_boost = l17 / l13; // ≈ 6.85
    let top_mass = 173.0; // GeV
    let prediction = top_mass * lucas_boost / 1000.0; // TeV

    (
        prediction,
        format!(
            "Dark Matter Mass = m_top × (L_17/L_13) = {} GeV × ({}/{}) = {:.2} TeV",
            top_mass, l17 as u64, l13 as u64, prediction
        ),
    )
}

// ============================================================================
// PyO3 Module Registration
// ============================================================================

pub fn register_prime_selection(m: &Bound<'_, PyModule>) -> PyResult<()> {
    // Fermat functions
    m.add_function(wrap_pyfunction!(fermat_number, m)?)?;
    m.add_function(wrap_pyfunction!(is_fermat_prime, m)?)?;
    m.add_function(wrap_pyfunction!(get_force_spectrum, m)?)?;

    // Mersenne functions
    m.add_function(wrap_pyfunction!(mersenne_number, m)?)?;
    m.add_function(wrap_pyfunction!(is_mersenne_prime, m)?)?;
    m.add_function(wrap_pyfunction!(get_generation_spectrum, m)?)?;
    m.add_function(wrap_pyfunction!(generation_barrier_explanation, m)?)?;

    // Lucas functions
    m.add_function(wrap_pyfunction!(lucas_number, m)?)?;
    m.add_function(wrap_pyfunction!(shadow_phase, m)?)?;
    m.add_function(wrap_pyfunction!(is_lucas_prime, m)?)?;
    m.add_function(wrap_pyfunction!(dark_matter_mass_prediction, m)?)?;

    // Constants
    m.add("FERMAT_PRIMES", FERMAT_PRIMES.to_vec())?;
    m.add("MERSENNE_EXPONENTS", MERSENNE_EXPONENTS.to_vec())?;
    m.add("LUCAS_SEQUENCE", LUCAS_SEQUENCE.to_vec())?;
    m.add("LUCAS_PRIMES", LUCAS_PRIMES.to_vec())?;
    m.add("M11_BARRIER", M11_BARRIER)?;

    Ok(())
}
