//! Exact Arithmetic Module for Syntony Recursion Theory
//!
//! This module provides exact (non-floating-point) arithmetic types
//! required for rigorous SRT computations. Standard IEEE-754 floats
//! cannot represent algebraic numbers like φ or preserve the exact
//! relationships between transcendental constants.
//!
//! # Type Hierarchy
//!
//! ```text
//! Rational (Q)
//!     ↓
//! GoldenExact (Q(√5) = Q(φ))
//!     ↓
//! SymExpr (symbolic expressions with π, e, E*, q)
//! ```
//!
//! # The Five Fundamental Constants
//!
//! SRT is built on five constants:
//! - π (pi) - circle constant, toroidal topology
//! - e (euler) - natural base, exponential evolution
//! - φ (phi) - golden ratio, recursion symmetry (algebraic: x² - x - 1 = 0)
//! - E* (e_star) - spectral Möbius constant = e^π - π
//! - q - universal syntony deficit, THE fundamental scale of SRT

pub mod rational;
pub mod golden;
pub mod constants;
pub mod symexpr;

pub use rational::Rational;
pub use golden::GoldenExact;
pub use constants::{FundamentalConstant, CorrectionLevel, Structure};
pub use symexpr::{SymExpr, PySymExpr};
