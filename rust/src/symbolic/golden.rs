use pyo3::prelude::*;

/// GoldenNumber: Exact representation of a + b*φ where φ = (1 + √5)/2
/// This allows exact arithmetic in the golden ratio field Q(√5)
#[pyclass]
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct GoldenNumber {
    #[pyo3(get, set)]
    pub a: f64,  // Rational part
    #[pyo3(get, set)]
    pub b: f64,  // Coefficient of φ
}

/// The golden ratio φ = (1 + √5)/2 ≈ 1.618033988749895
pub const PHI: f64 = 1.618033988749895;
/// φ² = φ + 1
pub const PHI_SQ: f64 = 2.618033988749895;

#[pymethods]
impl GoldenNumber {
    #[new]
    pub fn new(a: f64, b: f64) -> Self {
        GoldenNumber { a, b }
    }

    /// Create from just rational part (b = 0)
    #[staticmethod]
    pub fn from_rational(a: f64) -> Self {
        GoldenNumber { a, b: 0.0 }
    }

    /// The golden ratio φ itself
    #[staticmethod]
    pub fn phi() -> Self {
        GoldenNumber { a: 0.0, b: 1.0 }
    }

    /// Evaluate to floating point: a + b*φ
    pub fn eval(&self) -> f64 {
        self.a + self.b * PHI
    }

    /// Addition: (a1 + b1*φ) + (a2 + b2*φ) = (a1+a2) + (b1+b2)*φ
    fn __add__(&self, other: &GoldenNumber) -> Self {
        GoldenNumber::new(self.a + other.a, self.b + other.b)
    }

    /// Subtraction
    fn __sub__(&self, other: &GoldenNumber) -> Self {
        GoldenNumber::new(self.a - other.a, self.b - other.b)
    }

    /// Negation
    fn __neg__(&self) -> Self {
        GoldenNumber::new(-self.a, -self.b)
    }

    /// Multiplication using φ² = φ + 1
    /// (a1 + b1*φ)(a2 + b2*φ) = a1*a2 + (a1*b2 + a2*b1)*φ + b1*b2*φ²
    ///                        = a1*a2 + b1*b2 + (a1*b2 + a2*b1 + b1*b2)*φ
    fn __mul__(&self, other: &GoldenNumber) -> Self {
        let new_a = self.a * other.a + self.b * other.b;
        let new_b = self.a * other.b + self.b * other.a + self.b * other.b;
        GoldenNumber::new(new_a, new_b)
    }

    /// Scalar multiplication
    pub fn scale(&self, scalar: f64) -> Self {
        GoldenNumber::new(self.a * scalar, self.b * scalar)
    }

    /// Conjugate in Q(√5): (a + b*φ)* = a + b - b*φ = (a + b) - b*φ
    /// Since φ' = (1 - √5)/2 = 1 - φ, we have a + b*φ' = a + b*(1-φ) = (a+b) - b*φ
    pub fn conjugate(&self) -> Self {
        GoldenNumber::new(self.a + self.b, -self.b)
    }

    /// Norm in Q(√5): N(a + b*φ) = (a + b*φ)(a + b*φ') = a² + ab - b²
    pub fn norm(&self) -> f64 {
        self.a * self.a + self.a * self.b - self.b * self.b
    }

    /// Multiplicative inverse: 1/(a + b*φ) = conjugate / norm
    pub fn inverse(&self) -> Self {
        let n = self.norm();
        let conj = self.conjugate();
        GoldenNumber::new(conj.a / n, conj.b / n)
    }

    /// Division
    fn __truediv__(&self, other: &GoldenNumber) -> Self {
        let inv = other.inverse();
        self.__mul__(&inv)
    }

    /// Power (integer exponent)
    pub fn pow(&self, n: i32) -> Self {
        if n == 0 {
            return GoldenNumber::new(1.0, 0.0);
        }
        if n < 0 {
            return self.inverse().pow(-n);
        }
        let mut result = GoldenNumber::new(1.0, 0.0);
        let mut base = *self;
        let mut exp = n;
        while exp > 0 {
            if exp % 2 == 1 {
                result = result.__mul__(&base);
            }
            base = base.__mul__(&base);
            exp /= 2;
        }
        result
    }

    /// String representation
    fn __repr__(&self) -> String {
        if self.b == 0.0 {
            format!("GoldenNumber({:.6})", self.a)
        } else if self.a == 0.0 {
            format!("GoldenNumber({:.6}φ)", self.b)
        } else if self.b > 0.0 {
            format!("GoldenNumber({:.6} + {:.6}φ)", self.a, self.b)
        } else {
            format!("GoldenNumber({:.6} - {:.6}φ)", self.a, -self.b)
        }
    }

    /// Check if this is exactly an integer multiple of φ
    pub fn is_pure_phi(&self) -> bool {
        self.a.abs() < 1e-10
    }

    /// Check if this is exactly rational (no φ component)
    pub fn is_rational(&self) -> bool {
        self.b.abs() < 1e-10
    }

    /// φ_hat = 1/φ = φ - 1 (the SRT coherence parameter)
    #[staticmethod]
    pub fn phi_hat() -> Self {
        GoldenNumber::new(-1.0, 1.0)  // -1 + φ = 1/φ
    }

    /// φ² = φ + 1 (the golden square)
    #[staticmethod]
    pub fn phi_sq() -> Self {
        GoldenNumber::new(1.0, 1.0)  // 1 + φ = φ²
    }
}
