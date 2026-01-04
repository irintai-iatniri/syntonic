use pyo3::prelude::*;
use std::f64::consts::PI;
use std::f64::consts::E;

/// Symbolic expression tree for exact mathematical expressions
/// Supports lazy evaluation and symbolic manipulation
#[derive(Clone, Debug)]
pub enum Expr {
    /// Rational number a/b
    Rational { num: i64, den: i64 },
    /// Floating point (fallback)
    Float(f64),
    /// π (pi)
    Pi,
    /// e (Euler's number)
    Euler,
    /// φ (golden ratio)
    Phi,
    /// √5
    Sqrt5,
    /// Addition
    Add(Box<Expr>, Box<Expr>),
    /// Subtraction
    Sub(Box<Expr>, Box<Expr>),
    /// Multiplication
    Mul(Box<Expr>, Box<Expr>),
    /// Division
    Div(Box<Expr>, Box<Expr>),
    /// Power
    Pow(Box<Expr>, Box<Expr>),
    /// Square root
    Sqrt(Box<Expr>),
    /// Sine
    Sin(Box<Expr>),
    /// Cosine
    Cos(Box<Expr>),
    /// Natural logarithm
    Ln(Box<Expr>),
    /// Exponential
    Exp(Box<Expr>),
}

impl Expr {
    /// Evaluate the expression to a floating point number
    pub fn eval(&self) -> f64 {
        match self {
            Expr::Rational { num, den } => *num as f64 / *den as f64,
            Expr::Float(x) => *x,
            Expr::Pi => PI,
            Expr::Euler => E,
            Expr::Phi => (1.0 + 5.0_f64.sqrt()) / 2.0,
            Expr::Sqrt5 => 5.0_f64.sqrt(),
            Expr::Add(a, b) => a.eval() + b.eval(),
            Expr::Sub(a, b) => a.eval() - b.eval(),
            Expr::Mul(a, b) => a.eval() * b.eval(),
            Expr::Div(a, b) => a.eval() / b.eval(),
            Expr::Pow(a, b) => a.eval().powf(b.eval()),
            Expr::Sqrt(a) => a.eval().sqrt(),
            Expr::Sin(a) => a.eval().sin(),
            Expr::Cos(a) => a.eval().cos(),
            Expr::Ln(a) => a.eval().ln(),
            Expr::Exp(a) => a.eval().exp(),
        }
    }

    /// Check if expression is a constant (no variables)
    pub fn is_constant(&self) -> bool {
        match self {
            Expr::Rational { .. } | Expr::Float(_) |
            Expr::Pi | Expr::Euler | Expr::Phi | Expr::Sqrt5 => true,
            Expr::Add(a, b) | Expr::Sub(a, b) |
            Expr::Mul(a, b) | Expr::Div(a, b) | Expr::Pow(a, b) => {
                a.is_constant() && b.is_constant()
            }
            Expr::Sqrt(a) | Expr::Sin(a) | Expr::Cos(a) |
            Expr::Ln(a) | Expr::Exp(a) => a.is_constant(),
        }
    }
}

/// Python-accessible wrapper for symbolic expressions
#[pyclass(name = "Expr")]
#[derive(Clone)]
pub struct PyExpr {
    pub inner: Expr,
}

#[pymethods]
impl PyExpr {
    /// Create from rational a/b
    #[staticmethod]
    pub fn rational(num: i64, den: i64) -> Self {
        PyExpr { inner: Expr::Rational { num, den } }
    }

    /// Create from float
    #[staticmethod]
    pub fn float(x: f64) -> Self {
        PyExpr { inner: Expr::Float(x) }
    }

    /// π constant
    #[staticmethod]
    pub fn pi() -> Self {
        PyExpr { inner: Expr::Pi }
    }

    /// e constant
    #[staticmethod]
    pub fn euler() -> Self {
        PyExpr { inner: Expr::Euler }
    }

    /// φ constant
    #[staticmethod]
    pub fn phi() -> Self {
        PyExpr { inner: Expr::Phi }
    }

    /// √5 constant
    #[staticmethod]
    pub fn sqrt5() -> Self {
        PyExpr { inner: Expr::Sqrt5 }
    }

    /// Evaluation: returns floating point
    pub fn eval(&self) -> f64 {
        self.inner.eval()
    }

    /// Addition
    fn __add__(&self, other: Bound<'_, PyExpr>) -> Self {
        let other = other.borrow();
        PyExpr {
            inner: Expr::Add(Box::new(self.inner.clone()), Box::new(other.inner.clone()))
        }
    }

    /// Subtraction
    fn __sub__(&self, other: Bound<'_, PyExpr>) -> Self {
        let other = other.borrow();
        PyExpr {
            inner: Expr::Sub(Box::new(self.inner.clone()), Box::new(other.inner.clone()))
        }
    }

    /// Multiplication
    fn __mul__(&self, other: Bound<'_, PyExpr>) -> Self {
        let other = other.borrow();
        PyExpr {
            inner: Expr::Mul(Box::new(self.inner.clone()), Box::new(other.inner.clone()))
        }
    }

    /// Division
    fn __truediv__(&self, other: Bound<'_, PyExpr>) -> Self {
        let other = other.borrow();
        PyExpr {
            inner: Expr::Div(Box::new(self.inner.clone()), Box::new(other.inner.clone()))
        }
    }

    /// Power
    fn __pow__(&self, exp: Bound<'_, PyExpr>, _modulo: Option<Bound<'_, PyExpr>>) -> Self {
        let exp = exp.borrow();
        PyExpr {
            inner: Expr::Pow(Box::new(self.inner.clone()), Box::new(exp.inner.clone()))
        }
    }

    /// Square root
    pub fn sqrt(&self) -> Self {
        PyExpr { inner: Expr::Sqrt(Box::new(self.inner.clone())) }
    }

    /// Sine
    pub fn sin(&self) -> Self {
        PyExpr { inner: Expr::Sin(Box::new(self.inner.clone())) }
    }

    /// Cosine
    pub fn cos(&self) -> Self {
        PyExpr { inner: Expr::Cos(Box::new(self.inner.clone())) }
    }

    /// Natural log
    pub fn ln(&self) -> Self {
        PyExpr { inner: Expr::Ln(Box::new(self.inner.clone())) }
    }

    /// Exponential
    pub fn exp(&self) -> Self {
        PyExpr { inner: Expr::Exp(Box::new(self.inner.clone())) }
    }

    /// String representation
    fn __repr__(&self) -> String {
        format!("Expr({:.10})", self.inner.eval())
    }
}
