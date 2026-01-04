//! Exact Golden Field Arithmetic Q(√5) = Q(φ)
//!
//! Elements of the form a + b·φ where a, b ∈ Q (exact rationals).
//! This is the natural number field for SRT since φ is the recursion eigenvalue.
//!
//! # Key Properties
//!
//! - φ = (1 + √5)/2 is a root of x² - x - 1 = 0
//! - φ² = φ + 1 (fundamental identity, used for multiplication)
//! - φ · φ̂ = -1 where φ̂ = (1 - √5)/2 = 1 - φ (Galois conjugate)
//! - 1/φ = φ - 1 = φ̂ (with sign correction)
//!
//! # Norm and Trace
//!
//! For α = a + bφ:
//! - Conjugate: α* = (a + b) - bφ = a + b(1 - φ)
//! - Norm: N(α) = α · α* = a² + ab - b² (always rational)
//! - Trace: Tr(α) = α + α* = 2a + b (always rational)

use pyo3::prelude::*;
use std::fmt;
use std::ops::{Add, Sub, Mul, Div, Neg};

use super::rational::Rational;

/// Exact element of Q(φ): a + b·φ where a, b ∈ Q
#[pyclass]
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub struct GoldenExact {
    /// Rational component (coefficient of 1)
    a: Rational,
    /// Golden component (coefficient of φ)
    b: Rational,
}

impl GoldenExact {
    /// Create a + b·φ
    pub fn new(a: Rational, b: Rational) -> Self {
        GoldenExact { a, b }
    }

    /// Create from two integers: a + b·φ
    pub fn from_ints(a: i64, b: i64) -> Self {
        GoldenExact {
            a: Rational::from_int(a as i128),
            b: Rational::from_int(b as i128),
        }
    }

    /// Zero in Q(φ)
    pub fn zero() -> Self {
        GoldenExact {
            a: Rational::zero(),
            b: Rational::zero(),
        }
    }

    /// One in Q(φ)
    pub fn one() -> Self {
        GoldenExact {
            a: Rational::one(),
            b: Rational::zero(),
        }
    }

    /// φ = 0 + 1·φ
    pub fn phi() -> Self {
        GoldenExact {
            a: Rational::zero(),
            b: Rational::one(),
        }
    }

    /// φ² = 1 + 1·φ (derived from φ² = φ + 1)
    pub fn phi_squared() -> Self {
        GoldenExact {
            a: Rational::one(),
            b: Rational::one(),
        }
    }

    /// φ̂ = 1/φ = φ - 1 = -1 + 1·φ
    /// Note: This is the SRT coherence parameter
    pub fn phi_hat() -> Self {
        GoldenExact {
            a: Rational::from_int(-1),
            b: Rational::one(),
        }
    }

    /// √5 = 2φ - 1 = -1 + 2·φ
    /// Since φ = (1 + √5)/2, we have √5 = 2φ - 1
    pub fn sqrt5() -> Self {
        GoldenExact {
            a: Rational::from_int(-1),
            b: Rational::from_int(2),
        }
    }

    /// Embed a rational into Q(φ)
    pub fn from_rational(r: Rational) -> Self {
        GoldenExact {
            a: r,
            b: Rational::zero(),
        }
    }

    /// Get the rational coefficient
    pub fn rational_part(&self) -> Rational {
        self.a
    }

    /// Get the φ coefficient
    pub fn phi_part(&self) -> Rational {
        self.b
    }

    /// Is this exactly zero?
    pub fn is_zero(&self) -> bool {
        self.a.is_zero() && self.b.is_zero()
    }

    /// Is this exactly one?
    pub fn is_one(&self) -> bool {
        self.a.is_one() && self.b.is_zero()
    }

    /// Is this a pure rational (b = 0)?
    pub fn is_rational(&self) -> bool {
        self.b.is_zero()
    }

    /// Is this a pure φ-multiple (a = 0)?
    pub fn is_pure_phi(&self) -> bool {
        self.a.is_zero()
    }

    /// Galois conjugate: (a + bφ)* = (a + b) - bφ
    ///
    /// Since φ* = 1 - φ (the other root of x² - x - 1 = 0):
    /// (a + bφ)* = a + b(1 - φ) = (a + b) - bφ
    pub fn conjugate(&self) -> Self {
        GoldenExact {
            a: self.a + self.b,
            b: -self.b,
        }
    }

    /// Norm: N(α) = α · α* = a² + ab - b²
    ///
    /// This is always a rational number.
    /// For φ: N(φ) = 0 + 0·1 - 1 = -1
    pub fn norm(&self) -> Rational {
        // (a + bφ)(a + b - bφ) = a(a+b) - abφ + b(a+b)φ - b²φ²
        // = a² + ab + abφ + b²φ - abφ - b²φ²
        // = a² + ab + b²φ - b²(φ+1)
        // = a² + ab + b²φ - b²φ - b²
        // = a² + ab - b²
        self.a * self.a + self.a * self.b - self.b * self.b
    }

    /// Trace: Tr(α) = α + α* = 2a + b
    pub fn trace(&self) -> Rational {
        self.a + self.a + self.b
    }

    /// Multiplicative inverse: 1/α = α* / N(α)
    pub fn inverse(&self) -> Self {
        let n = self.norm();
        if n.is_zero() {
            panic!("GoldenExact: inverse of zero");
        }
        let conj = self.conjugate();
        GoldenExact {
            a: conj.a / n,
            b: conj.b / n,
        }
    }

    /// Integer power using binary exponentiation
    pub fn pow(&self, exp: i32) -> Self {
        if exp == 0 {
            return GoldenExact::one();
        }
        if exp < 0 {
            return self.inverse().pow(-exp);
        }

        let mut result = GoldenExact::one();
        let mut base = *self;
        let mut e = exp as u32;

        while e > 0 {
            if e & 1 == 1 {
                result = result * base;
            }
            base = base * base;
            e >>= 1;
        }
        result
    }

    /// φ^n computed exactly
    /// Uses the identity φ^n = F_n·φ + F_{n-1} where F_n is Fibonacci
    pub fn phi_power(n: i32) -> Self {
        if n == 0 {
            return GoldenExact::one();
        }
        if n < 0 {
            // φ^(-n) = (-1)^n · (F_{n-1} - F_n·φ) / (F_n² - F_n·F_{n-1} - F_{n-1}²)
            // Simpler: just compute and invert
            return GoldenExact::phi_power(-n).inverse();
        }

        // φ^n = F_n·φ + F_{n-1}
        let (f_prev, f_curr) = fibonacci_pair(n as u32);
        GoldenExact::from_ints(f_prev, f_curr)
    }

    /// Evaluate to f64 (for numerical verification ONLY)
    /// φ ≈ 1.6180339887498949
    pub fn to_f64(&self) -> f64 {
        const PHI_F64: f64 = 1.6180339887498949;
        self.a.to_f64() + self.b.to_f64() * PHI_F64
    }
}

// === Arithmetic Operations ===

impl Add for GoldenExact {
    type Output = Self;

    fn add(self, other: Self) -> Self {
        GoldenExact {
            a: self.a + other.a,
            b: self.b + other.b,
        }
    }
}

impl Sub for GoldenExact {
    type Output = Self;

    fn sub(self, other: Self) -> Self {
        GoldenExact {
            a: self.a - other.a,
            b: self.b - other.b,
        }
    }
}

impl Mul for GoldenExact {
    type Output = Self;

    /// Multiplication using φ² = φ + 1
    ///
    /// (a₁ + b₁φ)(a₂ + b₂φ) = a₁a₂ + (a₁b₂ + a₂b₁)φ + b₁b₂φ²
    ///                       = a₁a₂ + (a₁b₂ + a₂b₁)φ + b₁b₂(φ + 1)
    ///                       = (a₁a₂ + b₁b₂) + (a₁b₂ + a₂b₁ + b₁b₂)φ
    fn mul(self, other: Self) -> Self {
        let new_a = self.a * other.a + self.b * other.b;
        let new_b = self.a * other.b + self.b * other.a + self.b * other.b;
        GoldenExact { a: new_a, b: new_b }
    }
}

impl Div for GoldenExact {
    type Output = Self;

    fn div(self, other: Self) -> Self {
        self * other.inverse()
    }
}

impl Neg for GoldenExact {
    type Output = Self;

    fn neg(self) -> Self {
        GoldenExact {
            a: -self.a,
            b: -self.b,
        }
    }
}

// === Scalar operations ===

impl Mul<Rational> for GoldenExact {
    type Output = Self;

    fn mul(self, scalar: Rational) -> Self {
        GoldenExact {
            a: self.a * scalar,
            b: self.b * scalar,
        }
    }
}

impl Div<Rational> for GoldenExact {
    type Output = Self;

    fn div(self, scalar: Rational) -> Self {
        GoldenExact {
            a: self.a / scalar,
            b: self.b / scalar,
        }
    }
}

impl fmt::Display for GoldenExact {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.b.is_zero() {
            write!(f, "{}", self.a)
        } else if self.a.is_zero() {
            if self.b.is_one() {
                write!(f, "φ")
            } else {
                write!(f, "{}·φ", self.b)
            }
        } else if self.b.is_positive() {
            if self.b.is_one() {
                write!(f, "{} + φ", self.a)
            } else {
                write!(f, "{} + {}·φ", self.a, self.b)
            }
        } else {
            let neg_b = -self.b;
            if neg_b.is_one() {
                write!(f, "{} - φ", self.a)
            } else {
                write!(f, "{} - {}·φ", self.a, neg_b)
            }
        }
    }
}

// === PyO3 Methods ===

#[pymethods]
impl GoldenExact {
    #[new]
    fn py_new(a_num: i64, a_denom: i64, b_num: i64, b_denom: i64) -> Self {
        GoldenExact::new(
            Rational::new(a_num as i128, a_denom as i128),
            Rational::new(b_num as i128, b_denom as i128),
        )
    }

    /// Create from integer coefficients: a + b·φ
    #[staticmethod]
    fn from_integers(a: i64, b: i64) -> Self {
        GoldenExact::from_ints(a, b)
    }

    /// The golden ratio φ
    #[staticmethod]
    fn golden_ratio() -> Self {
        GoldenExact::phi()
    }

    /// φ² = φ + 1
    #[staticmethod]
    fn golden_squared() -> Self {
        GoldenExact::phi_squared()
    }

    /// 1/φ = φ - 1 (coherence parameter)
    #[staticmethod]
    fn coherence_parameter() -> Self {
        GoldenExact::phi_hat()
    }

    /// √5 = 2φ - 1
    #[staticmethod]
    fn root_five() -> Self {
        GoldenExact::sqrt5()
    }

    /// φ^n computed exactly
    #[staticmethod]
    fn phi_to_power(n: i32) -> Self {
        GoldenExact::phi_power(n)
    }

    #[getter]
    fn rational_coefficient(&self) -> (i64, i64) {
        (self.a.numerator() as i64, self.a.denominator() as i64)
    }

    #[getter]
    fn phi_coefficient(&self) -> (i64, i64) {
        (self.b.numerator() as i64, self.b.denominator() as i64)
    }

    fn __repr__(&self) -> String {
        format!("GoldenExact({}/{}  + ({}/{})·φ)",
            self.a.numerator(), self.a.denominator(),
            self.b.numerator(), self.b.denominator())
    }

    fn __str__(&self) -> String {
        self.to_string()
    }

    fn __add__(&self, other: &GoldenExact) -> Self {
        *self + *other
    }

    fn __sub__(&self, other: &GoldenExact) -> Self {
        *self - *other
    }

    fn __mul__(&self, other: &GoldenExact) -> Self {
        *self * *other
    }

    fn __truediv__(&self, other: &GoldenExact) -> Self {
        *self / *other
    }

    fn __neg__(&self) -> Self {
        -*self
    }

    fn __eq__(&self, other: &GoldenExact) -> bool {
        self == other
    }

    /// Galois conjugate
    fn galois_conjugate(&self) -> Self {
        self.conjugate()
    }

    /// Field norm (always rational)
    fn field_norm(&self) -> (i64, i64) {
        let n = self.norm();
        (n.numerator() as i64, n.denominator() as i64)
    }

    /// Field trace (always rational)
    fn field_trace(&self) -> (i64, i64) {
        let t = self.trace();
        (t.numerator() as i64, t.denominator() as i64)
    }

    /// Integer power
    fn power(&self, exp: i32) -> Self {
        self.pow(exp)
    }

    /// Multiplicative inverse
    fn reciprocal(&self) -> Self {
        self.inverse()
    }

    /// Evaluate to float (for verification only!)
    fn eval(&self) -> f64 {
        self.to_f64()
    }

    /// Check if this is exactly rational (no φ component)
    fn is_purely_rational(&self) -> bool {
        self.is_rational()
    }

    /// Check if this is exactly a multiple of φ (no rational part)
    fn is_purely_golden(&self) -> bool {
        self.is_pure_phi()
    }
}

/// Compute (F_{n-1}, F_n) - the Fibonacci pair
/// F_0 = 0, F_1 = 1, F_n = F_{n-1} + F_{n-2}
fn fibonacci_pair(n: u32) -> (i64, i64) {
    if n == 0 {
        return (1, 0); // F_{-1} = 1, F_0 = 0 (by convention for φ^0 = F_0·φ + F_{-1})
    }

    let mut prev: i64 = 0;
    let mut curr: i64 = 1;

    for _ in 1..n {
        let next = prev + curr;
        prev = curr;
        curr = next;
    }

    (prev, curr)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_phi_squared() {
        let phi = GoldenExact::phi();
        let phi_sq = phi * phi;
        assert_eq!(phi_sq, GoldenExact::phi_squared());
        // φ² = 1 + φ
        assert_eq!(phi_sq, GoldenExact::from_ints(1, 1));
    }

    #[test]
    fn test_phi_inverse() {
        let phi = GoldenExact::phi();
        let phi_inv = phi.inverse();
        // 1/φ = φ - 1
        assert_eq!(phi_inv, GoldenExact::phi_hat());

        // Verify: φ · (1/φ) = 1
        let product = phi * phi_inv;
        assert_eq!(product, GoldenExact::one());
    }

    #[test]
    fn test_norm() {
        // N(φ) = 0² + 0·1 - 1² = -1
        let phi = GoldenExact::phi();
        assert_eq!(phi.norm(), Rational::from_int(-1));

        // N(1) = 1
        let one = GoldenExact::one();
        assert_eq!(one.norm(), Rational::one());
    }

    #[test]
    fn test_sqrt5() {
        let sqrt5 = GoldenExact::sqrt5();
        let five = sqrt5 * sqrt5;
        // (√5)² = 5
        assert_eq!(five, GoldenExact::from_rational(Rational::from_int(5)));
    }

    #[test]
    fn test_phi_powers() {
        // φ¹ = φ
        assert_eq!(GoldenExact::phi_power(1), GoldenExact::phi());

        // φ² = 1 + φ
        assert_eq!(GoldenExact::phi_power(2), GoldenExact::from_ints(1, 1));

        // φ³ = φ² + φ = (1 + φ) + φ = 1 + 2φ
        assert_eq!(GoldenExact::phi_power(3), GoldenExact::from_ints(1, 2));

        // φ⁴ = φ³ + φ² = (1 + 2φ) + (1 + φ) = 2 + 3φ
        assert_eq!(GoldenExact::phi_power(4), GoldenExact::from_ints(2, 3));

        // φ⁵ = φ⁴ + φ³ = (2 + 3φ) + (1 + 2φ) = 3 + 5φ
        assert_eq!(GoldenExact::phi_power(5), GoldenExact::from_ints(3, 5));
    }

    #[test]
    fn test_conjugate() {
        // φ* = 1 - φ
        let phi = GoldenExact::phi();
        let phi_conj = phi.conjugate();
        assert_eq!(phi_conj, GoldenExact::from_ints(1, -1));

        // φ + φ* = 1 (trace)
        let sum = phi + phi_conj;
        assert_eq!(sum, GoldenExact::one());
    }
}
