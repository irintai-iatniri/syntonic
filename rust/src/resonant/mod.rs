//! Resonant Engine Module
//!
//! Hardware-native SRT/CRT architecture where:
//! - GPU = D̂ (Differentiation) - chaotic flux generator
//! - CPU = Ĥ (Harmonization) - exact lattice enforcer via crystallization
//! - PCIe bus = Phase boundary with φ-dwell timing enforcement
//!
//! # Core Components
//!
//! - [`ResonantTensor`]: Dual-state tensor with exact lattice (CPU) and ephemeral flux (GPU)
//! - [`ResonantPhase`]: Phase enumeration (Crystallized, Flux, Transitioning)
//! - [`ResonanceEnforcer`]: φ-dwell timing controller
//! - [`ResonantEvolver`]: Resonant Evolution Strategy (RES) for discrete learning
//!
//! # Phase Transitions
//!
//! ```text
//! ┌─────────────────┐     wake_flux()      ┌─────────────────┐
//! │   CRYSTALLIZED  │ ─────────────────────► │      FLUX       │
//! │  (CPU/Exact)    │                       │   (GPU/Float)   │
//! │  GoldenExact    │                       │   CudaSlice     │
//! └────────┬────────┘                       └────────┬────────┘
//!          │                                         │
//!          │        crystallize()                    │
//!          │◄────────────────────────────────────────┤
//!          │        (snap f64→GoldenExact)           │
//!          │                                         │
//!          │        destroy_shadow()                 │
//!          │        (flux = None)                    │
//!          │                            ─────────────┘
//! ```
//!
//! # Example
//!
//! ```ignore
//! use syntonic::resonant::{ResonantTensor, ResonantEngine};
//!
//! // Create tensor from floats
//! let values = vec![1.0, 2.0, 3.0, 4.0];
//! let mode_norms = vec![0.0, 1.0, 4.0, 9.0];
//! let mut tensor = ResonantTensor::from_floats(&values, vec![4], mode_norms, 100);
//!
//! // D-phase: wake flux (transfer to GPU)
//! tensor.wake_flux(device, 0.01)?;
//!
//! // H-phase: crystallize (snap back to exact lattice)
//! let syntony = tensor.crystallize(1000)?;
//! ```

mod tensor;
mod crystallize;
mod evolver;
mod attractor;
mod retrocausal;
pub mod phi_ops;
pub mod golden_norm;
pub mod syntonic_softmax;
pub mod number_theory;
pub mod syntony;
pub mod py_wrappers;
pub mod loss;

pub use tensor::{ResonantTensor, ResonantPhase, ResonantError};
pub use evolver::{ResonantEvolver, RESConfig, RESResult};
pub use attractor::AttractorMemory;
pub use phi_ops::{PhiResidualMode, phi_residual, phi_residual_relu};
pub use golden_norm::GoldenNormMode;
pub use syntonic_softmax::{SyntonicSoftmaxMode, SyntonicSoftmaxState, syntonic_softmax_py};

// Re-export key constants
pub const PHI: f64 = 1.6180339887498948482;
pub const PHI_INV: f64 = 0.6180339887498948482;
pub const PHI_INV_SQ: f64 = 0.3819660112501051518;
