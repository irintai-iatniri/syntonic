//! Linear algebra module for Syntonic tensor operations.
//!
//! This module provides matrix multiplication and related operations,
//! with all constants derived from the exact symbolic infrastructure.
//!
//! # Design Principle
//!
//! Every numerical value traces back to the five fundamental SRT constants:
//! - φ (phi) from `GoldenExact::phi()`
//! - φ⁻¹ (phi inverse) from `GoldenExact::phi_hat()`
//! - q (syntony deficit) from `FundamentalConstant::Q`
//! - π from `FundamentalConstant::Pi`
//! - E* from `FundamentalConstant::EStar`
//!
//! No hardcoded floating-point constants are used.

pub mod matmul;

// Re-export primary types and functions
pub use matmul::{
    // Core operations
    mm,
    mm_add,
    // Transpose variants
    mm_tn,
    mm_nt,
    mm_tt,
    // Hermitian variants
    mm_hn,
    mm_nh,
    // Batched
    bmm,
    // SRT-specific operations
    mm_phi,
    phi_bracket,
    phi_antibracket,
    mm_corrected,
    mm_golden_phase,
    mm_golden_weighted,
    projection_sum,
    // Error type
    MatmulError,
    // Transpose enum
    Transpose,
};
