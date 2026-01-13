//! ResonantTensor: Dual-state tensor for the Resonant Engine
//!
//! The ResonantTensor exists in two complementary representations:
//! - **Lattice** (CPU): Exact GoldenExact values in Q(φ) - the "truth"
//! - **Flux** (GPU): Ephemeral floating-point values - the "shadow"
//!
//! Phase transitions:
//! - `wake_flux()`: Project lattice → GPU floats (enter D-phase)
//! - `crystallize()`: Snap GPU floats → lattice (enter H-phase)
//! - `destroy_shadow()`: Free GPU memory without crystallizing

use pyo3::prelude::*;
use std::fmt;
use std::time::Duration;

use super::crystallize::{
    compute_lattice_syntony, crystallize_with_dwell, harmonize_and_crystallize, snap_to_lattice,
};
use super::{PHI, PHI_INV, PHI_INV_SQ};
use crate::exact::{GoldenExact, Rational};

#[cfg(feature = "cuda")]
use crate::tensor::srt_kernels::{
    cuda_resonant_compute_syntony_f64, cuda_resonant_d_phase_f64, ensure_srt_kernels_loaded,
};
#[cfg(feature = "cuda")]
use cudarc::driver::safe::CudaContext as CudaDevice;
#[cfg(feature = "cuda")]
use cudarc::driver::CudaSlice;
#[cfg(feature = "cuda")]
use std::sync::Arc;

/// Error type for resonant operations
#[derive(Debug, Clone)]
pub enum ResonantError {
    /// Invalid phase transition attempted
    InvalidPhaseTransition(String),
    /// No flux present when expected
    NoFluxPresent,
    /// No device present when expected
    NoDevicePresent,
    /// CUDA error
    #[cfg(feature = "cuda")]
    CudaError(String),
    /// Shape mismatch
    ShapeMismatch(String),
}

impl fmt::Display for ResonantError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ResonantError::InvalidPhaseTransition(msg) => {
                write!(f, "Invalid phase transition: {}", msg)
            }
            ResonantError::NoFluxPresent => write!(f, "No flux present"),
            ResonantError::NoDevicePresent => write!(f, "No CUDA device present"),
            #[cfg(feature = "cuda")]
            ResonantError::CudaError(msg) => write!(f, "CUDA error: {}", msg),
            ResonantError::ShapeMismatch(msg) => write!(f, "Shape mismatch: {}", msg),
        }
    }
}

impl std::error::Error for ResonantError {}

impl From<ResonantError> for PyErr {
    fn from(err: ResonantError) -> PyErr {
        pyo3::exceptions::PyRuntimeError::new_err(err.to_string())
    }
}

/// Phase of the resonant tensor
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ResonantPhase {
    /// Crystallized on CPU lattice (exact arithmetic)
    /// No GPU memory allocated
    Crystallized,

    /// Active flux on GPU (floating point)
    /// GPU memory allocated, lattice is stale
    Flux,

    /// Transitioning between phases
    /// Should not persist
    Transitioning,
}

impl fmt::Display for ResonantPhase {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ResonantPhase::Crystallized => write!(f, "crystallized"),
            ResonantPhase::Flux => write!(f, "flux"),
            ResonantPhase::Transitioning => write!(f, "transitioning"),
        }
    }
}

/// A tensor that exists in two complementary representations.
///
/// The ResonantTensor is the core data structure of the Resonant Engine.
/// It maintains:
/// - An exact lattice representation in Q(φ) (always valid)
/// - An optional GPU flux representation (valid only during D-phase)
///
/// # Invariants
/// - `lattice.len() == shape.product()`
/// - `mode_norm_sq.len() == lattice.len()`
/// - `flux.is_some()` implies `phase == Flux`
/// - `phase == Crystallized` implies `flux.is_none()`
#[pyclass]
#[derive(Debug)]
pub struct ResonantTensor {
    /// Exact lattice representation: a + b·φ for each element
    lattice: Vec<GoldenExact>,

    /// GPU flux representation (active during D-phase only)
    #[cfg(feature = "cuda")]
    flux: Option<CudaSlice<f64>>,

    /// CPU flux representation (active during D-phase fallback or pure CPU mode)
    cpu_flux: Option<Vec<f64>>,

    /// Shape of the tensor
    shape: Vec<usize>,

    /// Current syntony value S ∈ [0, 1]
    syntony: f64,

    /// Precomputed mode norm squared |n|² for each element
    /// Used for syntony computation and D̂/Ĥ operators
    mode_norm_sq: Vec<f64>,

    /// Current phase
    phase: ResonantPhase,

    /// CUDA device reference
    #[cfg(feature = "cuda")]
    device: Option<Arc<CudaDevice>>,

    /// Device index
    device_idx: usize,

    /// Last D-phase duration (nanoseconds)
    last_d_duration_ns: u64,

    /// Precision used for last crystallization
    precision: i64,
}

impl ResonantTensor {
    // =========================================================================
    // Construction
    // =========================================================================

    /// Create a ResonantTensor from an existing GoldenExact lattice.
    pub fn from_lattice(
        lattice: Vec<GoldenExact>,
        shape: Vec<usize>,
        mode_norm_sq: Vec<f64>,
    ) -> Result<Self, ResonantError> {
        let expected_size: usize = shape.iter().product();
        if lattice.len() != expected_size {
            return Err(ResonantError::ShapeMismatch(format!(
                "Lattice size {} doesn't match shape {:?}",
                lattice.len(),
                shape
            )));
        }
        if mode_norm_sq.len() != lattice.len() {
            return Err(ResonantError::ShapeMismatch(format!(
                "mode_norm_sq size {} doesn't match lattice size {}",
                mode_norm_sq.len(),
                lattice.len()
            )));
        }

        let syntony = compute_lattice_syntony(&lattice, &mode_norm_sq);

        Ok(ResonantTensor {
            lattice,
            #[cfg(feature = "cuda")]
            flux: None,
            cpu_flux: None,
            shape,
            syntony,
            mode_norm_sq,
            phase: ResonantPhase::Crystallized,
            #[cfg(feature = "cuda")]
            device: None,
            device_idx: 0,
            last_d_duration_ns: 0,
            precision: 100, // Default precision
        })
    }

    /// Create from f64 data by snapping to nearest golden lattice points.
    pub fn from_floats(
        data: &[f64],
        shape: Vec<usize>,
        mode_norm_sq: Vec<f64>,
        precision: i64,
    ) -> Result<Self, ResonantError> {
        let lattice = snap_to_lattice(data, precision);
        let mut tensor = Self::from_lattice(lattice, shape, mode_norm_sq)?;
        tensor.precision = precision;
        Ok(tensor)
    }

    /// Create with default mode norms (|n|² = index²).
    pub fn from_floats_default_modes(
        data: &[f64],
        shape: Vec<usize>,
        precision: i64,
    ) -> Result<Self, ResonantError> {
        let mode_norm_sq: Vec<f64> = (0..data.len()).map(|i| (i as f64).powi(2)).collect();
        Self::from_floats(data, shape, mode_norm_sq, precision)
    }

    // =========================================================================
    // Phase Transitions
    // =========================================================================

    /// PROJECT: Lattice → Flux (exact → approximate)
    ///
    /// This is the "wake_flux" operation that enters D-phase.
    /// The exact lattice values are projected to f64 and transferred to GPU.
    #[cfg(feature = "cuda")]
    pub fn wake_flux(&mut self, device: Arc<CudaDevice>) -> Result<(), ResonantError> {
        if self.phase != ResonantPhase::Crystallized {
            return Err(ResonantError::InvalidPhaseTransition(
                "wake_flux requires Crystallized phase".to_string(),
            ));
        }

        self.phase = ResonantPhase::Transitioning;

        // Project exact lattice to f64
        let floats: Vec<f64> = self.lattice.iter().map(|g| g.to_f64()).collect();

        // Upload to GPU
        let gpu_slice = device
            .default_stream()
            .clone_htod(&floats)
            .map_err(|e| ResonantError::CudaError(e.to_string()))?;

        self.flux = Some(gpu_slice);
        self.device = Some(device);
        self.phase = ResonantPhase::Flux;

        Ok(())
    }

    /// CPU-only version of wake_flux for testing without GPU
    #[cfg(not(feature = "cuda"))]
    pub fn wake_flux_cpu(&mut self) -> Result<Vec<f64>, ResonantError> {
        if self.phase != ResonantPhase::Crystallized {
            return Err(ResonantError::InvalidPhaseTransition(
                "wake_flux requires Crystallized phase".to_string(),
            ));
        }

        // Project exact lattice to f64
        let floats: Vec<f64> = self.lattice.iter().map(|g| g.to_f64()).collect();

        self.phase = ResonantPhase::Flux;
        Ok(floats)
    }

    /// CRYSTALLIZE: Flux → Lattice (approximate → exact)
    ///
    /// This is the core H-phase operation. GPU floats are snapped to the
    /// nearest golden lattice points using LLL-based approximation.
    #[cfg(feature = "cuda")]
    pub fn crystallize(&mut self, precision: i64) -> Result<f64, ResonantError> {
        if self.phase != ResonantPhase::Flux {
            return Err(ResonantError::InvalidPhaseTransition(
                "crystallize requires Flux phase".to_string(),
            ));
        }

        let flux = self.flux.as_ref().ok_or(ResonantError::NoFluxPresent)?;
        let device = self.device.as_ref().ok_or(ResonantError::NoDevicePresent)?;

        self.phase = ResonantPhase::Transitioning;

        // Download from GPU
        let mut host_data = vec![0.0f64; flux.len()];
        device
            .default_stream()
            .memcpy_dtoh(flux, &mut host_data)
            .map_err(|e| ResonantError::CudaError(e.to_string()))?;

        // Golden snap: find nearest exact lattice point for each element
        self.lattice = snap_to_lattice(&host_data, precision);
        self.precision = precision;

        // Recompute syntony on crystallized lattice
        self.syntony = compute_lattice_syntony(&self.lattice, &self.mode_norm_sq);

        // Destroy the shadow
        self.flux = None;
        self.phase = ResonantPhase::Crystallized;

        Ok(self.syntony)
    }

    /// Crystallize with Ĥ attenuation and φ-dwell timing enforcement.
    ///
    /// This implements the full H-phase:
    /// 1. Download flux from GPU
    /// 2. Apply Ĥ attenuation: Ĥ[ψ]ₙ = ψₙ × (1 - β(S) × (1 - w(n)))
    /// 3. Snap to Q(φ) lattice
    /// 4. If time remains before φ × t_D, deepen precision
    /// 5. Recompute syntony
    ///
    /// # Arguments
    /// * `base_precision` - Initial precision (max coefficient bound)
    /// * `target_duration` - Target duration for H-phase (φ × D-phase duration)
    #[cfg(feature = "cuda")]
    pub fn crystallize_with_dwell_time(
        &mut self,
        base_precision: i64,
        target_duration: Duration,
    ) -> Result<f64, ResonantError> {
        if self.phase != ResonantPhase::Flux {
            return Err(ResonantError::InvalidPhaseTransition(
                "crystallize requires Flux phase".to_string(),
            ));
        }

        let flux = self.flux.as_ref().ok_or(ResonantError::NoFluxPresent)?;
        let device = self.device.as_ref().ok_or(ResonantError::NoDevicePresent)?;

        self.phase = ResonantPhase::Transitioning;

        // Download from GPU
        let mut host_data = vec![0.0f64; flux.len()];
        device
            .default_stream()
            .memcpy_dtoh(flux, &mut host_data)
            .map_err(|e| ResonantError::CudaError(e.to_string()))?;

        // Crystallize with Ĥ attenuation and φ-dwell timing
        // Pass mode_norm_sq and syntony for Ĥ operator
        let (new_lattice, final_precision, _actual_duration) = crystallize_with_dwell(
            &host_data,
            &self.mode_norm_sq,
            self.syntony,
            base_precision,
            target_duration,
        );

        self.lattice = new_lattice;
        self.precision = final_precision;
        self.syntony = compute_lattice_syntony(&self.lattice, &self.mode_norm_sq);

        // Destroy the shadow
        self.flux = None;
        self.phase = ResonantPhase::Crystallized;

        Ok(self.syntony)
    }

    /// CPU-only crystallization with Ĥ attenuation.
    ///
    /// Applies the Ĥ (harmonization) operator then snaps to Q(φ) lattice:
    ///   Ĥ[ψ]ₙ = ψₙ × (1 - β(S) × (1 - w(n)))
    ///
    /// Where β(S) = φ⁻¹ × S and w(n) = exp(-|n|²/φ)
    pub fn crystallize_cpu(
        &mut self,
        values: &[f64],
        precision: i64,
    ) -> Result<f64, ResonantError> {
        if values.len() != self.lattice.len() {
            return Err(ResonantError::ShapeMismatch(format!(
                "Values length {} doesn't match lattice length {}",
                values.len(),
                self.lattice.len()
            )));
        }

        // Apply Ĥ attenuation + snap to lattice
        self.lattice =
            harmonize_and_crystallize(values, &self.mode_norm_sq, self.syntony, precision);
        self.precision = precision;
        self.syntony = compute_lattice_syntony(&self.lattice, &self.mode_norm_sq);
        self.phase = ResonantPhase::Crystallized;

        Ok(self.syntony)
    }

    /// Explicitly destroy flux without crystallizing (discard D-phase work)
    pub fn destroy_shadow(&mut self) {
        #[cfg(feature = "cuda")]
        {
            self.flux = None;
        }
        if self.phase == ResonantPhase::Flux {
            self.phase = ResonantPhase::Crystallized;
        }
    }

    // =========================================================================
    // CUDA-Accelerated D-Phase
    // =========================================================================

    /// Wake flux and apply D-phase transformation with CUDA kernel.
    ///
    /// This applies the D̂ operator with stochastic noise on GPU:
    /// flux[i] = lattice[i] * (1 + α(S) * √|n_i|²) + noise_scale * (1 - S) * noise[i]
    #[cfg(feature = "cuda")]
    pub fn wake_flux_with_d_phase(
        &mut self,
        device: Arc<CudaDevice>,
        noise_scale: f64,
    ) -> Result<(), ResonantError> {
        use rand::Rng;

        if self.phase != ResonantPhase::Crystallized {
            return Err(ResonantError::InvalidPhaseTransition(
                "wake_flux requires Crystallized phase".to_string(),
            ));
        }

        self.phase = ResonantPhase::Transitioning;

        // Ensure kernels are loaded
        ensure_srt_kernels_loaded(&device).map_err(|e| ResonantError::CudaError(e.to_string()))?;

        // Project exact lattice to f64
        let floats: Vec<f64> = self.lattice.iter().map(|g| g.to_f64()).collect();
        let n = floats.len();

        // Upload lattice and mode norms to GPU
        let gpu_lattice = device
            .default_stream()
            .clone_htod(&floats)
            .map_err(|e| ResonantError::CudaError(e.to_string()))?;
        let gpu_mode_norms = device
            .default_stream()
            .clone_htod(&self.mode_norm_sq)
            .map_err(|e| ResonantError::CudaError(e.to_string()))?;

        // Generate Gaussian noise on CPU and upload
        let mut rng = rand::thread_rng();
        let noise: Vec<f64> = (0..n).map(|_| rng.gen::<f64>() * 2.0 - 1.0).collect();
        let gpu_noise = device
            .default_stream()
            .clone_htod(&noise)
            .map_err(|e| ResonantError::CudaError(e.to_string()))?;

        // Allocate output buffer
        let mut gpu_flux: CudaSlice<f64> = device
            .default_stream()
            .alloc_zeros(n)
            .map_err(|e| ResonantError::CudaError(e.to_string()))?;

        // Run D-phase kernel
        let start = std::time::Instant::now();
        cuda_resonant_d_phase_f64(
            &device,
            &mut gpu_flux,
            &gpu_lattice,
            &gpu_mode_norms,
            &gpu_noise,
            self.syntony,
            noise_scale,
            n,
        )
        .map_err(|e| ResonantError::CudaError(e.to_string()))?;
        self.last_d_duration_ns = start.elapsed().as_nanos() as u64;

        self.flux = Some(gpu_flux);
        self.device = Some(device);
        self.phase = ResonantPhase::Flux;

        Ok(())
    }

    /// Complete a full D→H cycle using CUDA for D-phase.
    ///
    /// 1. Wake flux with D-phase kernel (GPU)
    /// 2. Crystallize to lattice (CPU with LLL)
    ///
    /// Returns the new syntony value.
    #[cfg(feature = "cuda")]
    pub fn cuda_cycle(
        &mut self,
        device: Arc<CudaDevice>,
        noise_scale: f64,
        precision: i64,
    ) -> Result<f64, ResonantError> {
        // D-phase: GPU differentiation with noise
        self.wake_flux_with_d_phase(device, noise_scale)?;

        // H-phase: CPU crystallization
        self.crystallize(precision)
    }

    /// Complete a full D→H cycle with φ-dwell timing.
    ///
    /// H-phase duration is scaled by φ^syntony relative to D-phase duration.
    #[cfg(feature = "cuda")]
    pub fn cuda_cycle_with_phi_dwell(
        &mut self,
        device: Arc<CudaDevice>,
        noise_scale: f64,
        base_precision: i64,
    ) -> Result<f64, ResonantError> {
        // D-phase: GPU differentiation with noise
        self.wake_flux_with_d_phase(device.clone(), noise_scale)?;

        // Compute target H-phase duration: φ^S × D_duration
        let phi_factor = PHI.powf(self.syntony);
        let target_h_ns = (self.last_d_duration_ns as f64 * phi_factor) as u64;
        let target_duration = Duration::from_nanos(target_h_ns.max(1000)); // Min 1μs

        // H-phase: CPU crystallization with dwell
        self.crystallize_with_dwell_time(base_precision, target_duration)
    }

    /// Get the duration of the last D-phase in nanoseconds.
    pub fn last_d_duration_ns(&self) -> u64 {
        self.last_d_duration_ns
    }

    /// Compute syntony on current GPU flux (without crystallizing).
    #[cfg(feature = "cuda")]
    pub fn compute_flux_syntony(&self) -> Result<f64, ResonantError> {
        let flux = self.flux.as_ref().ok_or(ResonantError::NoFluxPresent)?;
        let device = self.device.as_ref().ok_or(ResonantError::NoDevicePresent)?;

        // Upload mode norms if needed (they may already be there, but simpler to re-upload)
        let gpu_mode_norms = device
            .default_stream()
            .clone_htod(&self.mode_norm_sq)
            .map_err(|e| ResonantError::CudaError(e.to_string()))?;

        cuda_resonant_compute_syntony_f64(device, flux, &gpu_mode_norms, self.len())
            .map_err(|e| ResonantError::CudaError(e.to_string()))
    }

    // =========================================================================
    // Accessors
    // =========================================================================

    /// Get the current syntony value.
    pub fn syntony(&self) -> f64 {
        self.syntony
    }

    /// Get the current phase.
    pub fn phase(&self) -> ResonantPhase {
        self.phase
    }

    /// Get the tensor shape.
    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    /// Get the lattice values (read-only).
    pub fn lattice(&self) -> &[GoldenExact] {
        &self.lattice
    }

    /// Get the mode norm squared values.
    pub fn mode_norm_sq(&self) -> &[f64] {
        &self.mode_norm_sq
    }

    /// Get the number of elements.
    pub fn len(&self) -> usize {
        self.lattice.len()
    }

    /// Check if empty.
    pub fn is_empty(&self) -> bool {
        self.lattice.is_empty()
    }

    /// Convert lattice to f64 vector (for export/inspection).
    pub fn to_floats_core(&self) -> Vec<f64> {
        self.lattice.iter().map(|g| g.to_f64()).collect()
    }

    /// Get the precision used for last crystallization.
    pub fn precision(&self) -> i64 {
        self.precision
    }

    pub fn set_device_idx(&mut self, idx: usize) {
        self.device_idx = idx;
    }

    pub fn device_idx(&self) -> Option<usize> {
        Some(self.device_idx)
    }

    /// Access the flux (GPU buffer) if present.
    #[cfg(feature = "cuda")]
    pub fn flux_ref(&self) -> Option<&cudarc::driver::CudaSlice<f64>> {
        self.flux.as_ref()
    }

    /// Set the flux (GPU buffer) and update phase.
    #[cfg(feature = "cuda")]
    pub fn set_flux(&mut self, flux: cudarc::driver::CudaSlice<f64>) {
        self.flux = Some(flux);
        self.phase = ResonantPhase::Flux;
    }

    /// Recompute syntony from current lattice state.
    pub fn recompute_syntony(&mut self) {
        self.syntony = compute_lattice_syntony(&self.lattice, &self.mode_norm_sq);
    }

    /// Set lattice values (module-private, for retrocausal harmonization).
    pub(super) fn set_lattice(&mut self, values: &[GoldenExact]) -> Result<(), ResonantError> {
        if self.lattice.len() != values.len() {
            return Err(ResonantError::ShapeMismatch(format!(
                "Lattice size mismatch: expected {}, got {}",
                self.lattice.len(),
                values.len()
            )));
        }
        self.lattice.copy_from_slice(values);
        Ok(())
    }

    /// Complete a full D→H cycle in CPU mode (Rust-accessible).
    ///
    /// This simulates what would happen with GPU differentiation
    /// by applying noise to the values and then crystallizing.
    ///
    /// Returns the new syntony value.
    pub fn run_cpu_cycle(
        &mut self,
        noise_scale: f64,
        precision: i64,
    ) -> Result<f64, ResonantError> {
        use rand::Rng;

        // Wake flux
        let mut values = self.to_floats_core();
        self.phase = ResonantPhase::Flux;

        // Apply simulated D-phase noise
        let mut rng = rand::thread_rng();
        for (i, v) in values.iter_mut().enumerate() {
            let norm_sq = self.mode_norm_sq[i];
            let s = self.syntony;

            // D̂ coefficient: α(S) = φ⁻² × (1 - S)
            let alpha = PHI_INV_SQ * (1.0 - s);
            let scale = 1.0 + alpha * norm_sq.sqrt();

            // Stochastic noise
            let noise: f64 = rng.gen::<f64>() - 0.5;
            *v = *v * scale + noise * noise_scale * (1.0 - s);
        }

        // Crystallize
        self.crystallize_cpu(&values, precision)
    }

    /// Native matrix-vector multiplication for GoldenExact lattice.
    ///
    /// Performs Y = XW^T where self is X (batch, in_features)
    /// and weights is W (out_features, in_features).
    ///
    /// Resulting tensor is in Crystallized phase.
    pub fn matmul_core(&self, weights: &ResonantTensor) -> Result<ResonantTensor, ResonantError> {
        if self.shape.len() != 2 || weights.shape.len() != 2 {
            return Err(ResonantError::ShapeMismatch(
                "matmul currently requires 2D tensors [batch, in] and [out, in]".to_string(),
            ));
        }

        let batch_size = self.shape[0];
        let in_features = self.shape[1];
        let out_features = weights.shape[0];

        if weights.shape[1] != in_features {
            return Err(ResonantError::ShapeMismatch(format!(
                "DIM mismatch: self.in={} vs weights.in={}",
                in_features, weights.shape[1]
            )));
        }

        let mut result_lattice = Vec::with_capacity(batch_size * out_features);

        // Perform exact Q(φ) multiplication and addition
        for b in 0..batch_size {
            for o in 0..out_features {
                let mut sum = GoldenExact::zero();
                for i in 0..in_features {
                    let x_val = self.lattice[b * in_features + i];
                    let w_val = weights.lattice[o * in_features + i];
                    sum = sum + (x_val * w_val);
                }
                result_lattice.push(sum);
            }
        }

        // New mode norms for output (simple index-based for now)
        let result_len = batch_size * out_features;
        let mut result_norms = Vec::with_capacity(result_len);
        for _ in 0..batch_size {
            for o in 0..out_features {
                result_norms.push((o as f64).powi(2));
            }
        }

        Self::from_lattice(result_lattice, vec![batch_size, out_features], result_norms)
    }

    /// Native bias addition for GoldenExact lattice.
    ///
    /// Performs self + bias where self is (batch, out) and bias is (out).
    pub fn add_bias_core(&mut self, bias: &ResonantTensor) -> Result<(), ResonantError> {
        if self.shape.len() != 2 {
            return Err(ResonantError::ShapeMismatch(
                "self must be 2D [batch, out]".to_string(),
            ));
        }

        let batch_size = self.shape[0];
        let out_features = self.shape[1];

        if bias.len() != out_features {
            return Err(ResonantError::ShapeMismatch(format!(
                "Bias dim {} must match layer dim {}",
                bias.len(),
                out_features
            )));
        }

        for b in 0..batch_size {
            for o in 0..out_features {
                self.lattice[b * out_features + o] =
                    self.lattice[b * out_features + o] + bias.lattice[o];
            }
        }

        Ok(())
    }

    /// Native ReLU activation on GoldenExact lattice.
    ///
    /// Snaps all negative lattice points to zero.
    pub fn relu_core(&mut self) {
        for val in self.lattice.iter_mut() {
            if val.to_f64() < 0.0 {
                *val = GoldenExact::zero();
            }
        }
    }

    /// Apply the Golden Recursion Map R(n) = floor(phi * n) to all lattice values.
    ///
    /// This scales all lattice values by phi, implementing one "recursion step".
    /// Used for building recursive layers without depth explosion.
    pub fn apply_recursion_core(&mut self) {
        // GoldenExact multiplication by phi is exact: (a, b) -> (b, a+b)
        // This is the Fibonacci scaling property of the golden lattice.
        for val in self.lattice.iter_mut() {
            *val = val.mul_phi();
        }
    }

    /// Apply the inverse recursion map R^{-1}(n) = floor(n / phi).
    ///
    /// This scales all lattice values by 1/phi.
    pub fn apply_inverse_recursion_core(&mut self) {
        for val in self.lattice.iter_mut() {
            *val = val.div_phi();
        }
    }

    /// Apply hierarchical pruning: snap values below threshold to zero.
    ///
    /// Threshold is specified as a fraction of q (e.g., 248 for q/248).
    /// Values with |v| < q/divisor are snapped to zero.
    pub fn prune_hierarchy_core(&mut self, q: f64, divisor: f64) {
        let threshold = q / divisor;
        for val in self.lattice.iter_mut() {
            if val.to_f64().abs() < threshold {
                *val = GoldenExact::zero();
            }
        }
    }

    /// Apply sigmoid activation: σ(x) = 1 / (1 + e^(-x)) to all lattice values.
    ///
    /// This converts lattice values to floats, applies sigmoid, and snaps back to Q(φ).
    pub fn sigmoid_core(&mut self, precision: i64) {
        let floats: Vec<f64> = self
            .lattice
            .iter()
            .map(|g| {
                let x = g.to_f64();
                1.0 / (1.0 + (-x).exp())
            })
            .collect();

        self.lattice = snap_to_lattice(&floats, precision);
        self.precision = precision;
        self.recompute_syntony();
    }

    /// Apply tanh activation to all lattice values.
    pub fn tanh_core(&mut self, precision: i64) {
        let floats: Vec<f64> = self.lattice.iter().map(|g| g.to_f64().tanh()).collect();

        self.lattice = snap_to_lattice(&floats, precision);
        self.precision = precision;
        self.recompute_syntony();
    }

    /// Apply GELU activation (erf-based) and snap to lattice.
    pub fn gelu_core(&mut self, precision: i64) {
        let floats: Vec<f64> = self
            .lattice
            .iter()
            .map(|g| {
                let x = g.to_f64();
                // GELU approximation: 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x^3)))
                let x3 = x * x * x;
                let tanh_arg = 0.7978845608 * (x + 0.044715 * x3);
                0.5 * x * (1.0 + tanh_arg.tanh())
            })
            .collect();

        self.lattice = snap_to_lattice(&floats, precision);
        self.precision = precision;
        self.recompute_syntony();
    }

    /// Element-wise multiplication: self * other (Hadamard product).
    ///
    /// Both tensors must have the same shape.
    /// Result is in exact Q(φ) arithmetic.
    pub fn elementwise_mul_core(
        &self,
        other: &ResonantTensor,
    ) -> Result<ResonantTensor, ResonantError> {
        if self.shape != other.shape {
            return Err(ResonantError::ShapeMismatch(format!(
                "Shape mismatch: {:?} vs {:?}",
                self.shape, other.shape
            )));
        }

        let result_lattice: Vec<GoldenExact> = self
            .lattice
            .iter()
            .zip(other.lattice.iter())
            .map(|(a, b)| *a * *b)
            .collect();

        // Mode norms: for element-wise ops, preserve the mode structure
        let result_norms = self.mode_norm_sq.clone();

        Self::from_lattice(result_lattice, self.shape.clone(), result_norms)
    }

    /// Element-wise addition: self + other.
    ///
    /// Both tensors must have the same shape.
    /// Result is in exact Q(φ) arithmetic.
    pub fn elementwise_add_core(
        &self,
        other: &ResonantTensor,
    ) -> Result<ResonantTensor, ResonantError> {
        if self.shape != other.shape {
            return Err(ResonantError::ShapeMismatch(format!(
                "Shape mismatch: {:?} vs {:?}",
                self.shape, other.shape
            )));
        }

        let result_lattice: Vec<GoldenExact> = self
            .lattice
            .iter()
            .zip(other.lattice.iter())
            .map(|(a, b)| *a + *b)
            .collect();

        let result_norms = self.mode_norm_sq.clone();

        Self::from_lattice(result_lattice, self.shape.clone(), result_norms)
    }

    /// Scalar multiplication: self * scalar.
    ///
    /// Multiplies every element by a scalar value.
    /// Result is in exact Q(φ) arithmetic.
    pub fn scalar_mul_core(&self, scalar: f64) -> Result<ResonantTensor, ResonantError> {
        // Convert scalar to Q(φ) with precision from self
        let scalar_golden = GoldenExact::find_nearest(scalar, self.precision);

        let result_lattice: Vec<GoldenExact> =
            self.lattice.iter().map(|g| *g * scalar_golden).collect();

        let result_norms = self.mode_norm_sq.clone();

        Self::from_lattice(result_lattice, self.shape.clone(), result_norms)
    }

    /// Scalar addition: self + scalar.
    ///
    /// Adds a scalar value to every element.
    /// Result is in exact Q(φ) arithmetic.
    pub fn scalar_add_core(&self, scalar: f64) -> Result<ResonantTensor, ResonantError> {
        // Convert scalar to Q(φ) with precision from self
        let scalar_golden = GoldenExact::find_nearest(scalar, self.precision);

        let result_lattice: Vec<GoldenExact> =
            self.lattice.iter().map(|g| *g + scalar_golden).collect();

        let result_norms = self.mode_norm_sq.clone();

        Self::from_lattice(result_lattice, self.shape.clone(), result_norms)
    }

    /// Negate: -self.
    ///
    /// Multiplies every element by -1.
    /// Equivalent to scalar_mul(-1.0) but more explicit.
    pub fn negate_core(&self) -> Result<ResonantTensor, ResonantError> {
        // -1 = -1 + 0·φ
        let neg_one = GoldenExact::new(Rational::new(-1, 1), Rational::new(0, 1));

        let result_lattice: Vec<GoldenExact> = self.lattice.iter().map(|g| *g * neg_one).collect();

        let result_norms = self.mode_norm_sq.clone();

        Self::from_lattice(result_lattice, self.shape.clone(), result_norms)
    }

    /// One minus: 1 - self.
    ///
    /// Computes 1 - x for every element.
    /// Common pattern in gating mechanisms.
    pub fn one_minus_core(&self) -> Result<ResonantTensor, ResonantError> {
        // 1 = 1 + 0·φ
        let one = GoldenExact::new(Rational::new(1, 1), Rational::new(0, 1));
        // -1 = -1 + 0·φ
        let neg_one = GoldenExact::new(Rational::new(-1, 1), Rational::new(0, 1));

        let result_lattice: Vec<GoldenExact> =
            self.lattice.iter().map(|g| one + (*g * neg_one)).collect();

        let result_norms = self.mode_norm_sq.clone();

        Self::from_lattice(result_lattice, self.shape.clone(), result_norms)
    }

    /// Natural logarithm: ln(x).
    ///
    /// Converts to floats, applies ln, snaps back to Q(φ) lattice.
    pub fn log_core(&self, precision: i64) -> Result<ResonantTensor, ResonantError> {
        let floats = self.to_floats_core();
        let result_floats: Vec<f64> = floats.iter().map(|&x| x.ln()).collect();
        let result_lattice = snap_to_lattice(&result_floats, precision);
        let result_norms = self.mode_norm_sq.clone();

        Self::from_lattice(result_lattice, self.shape.clone(), result_norms)
    }

    /// Natural exponential: e^x.
    ///
    /// Converts to floats, applies exp, snaps back to Q(φ) lattice.
    pub fn exp_core(&self, precision: i64) -> Result<ResonantTensor, ResonantError> {
        let floats = self.to_floats_core();
        let result_floats: Vec<f64> = floats.iter().map(|&x| x.exp()).collect();
        let result_lattice = snap_to_lattice(&result_floats, precision);
        let result_norms = self.mode_norm_sq.clone();

        Self::from_lattice(result_lattice, self.shape.clone(), result_norms)
    }

    /// Softmax activation along the last dimension.
    ///
    /// Computes softmax(x_i) = exp(x_i) / sum(exp(x_j)) for all j in the last dimension.
    /// Uses numerically stable version: softmax(x_i) = exp(x_i - max(x)) / sum(exp(x_j - max(x)))
    ///
    /// For 2D tensors [batch, features], applies softmax independently to each batch.
    /// For 1D tensors, applies softmax to the entire vector.
    pub fn softmax_core(&mut self, precision: i64) -> Result<(), ResonantError> {
        let floats = self.to_floats();

        let result_floats = match self.shape.len() {
            1 => {
                // 1D: Apply softmax to entire vector
                let n = floats.len();
                self.softmax_1d(&floats, n)
            }
            2 => {
                // 2D: Apply softmax to each row (batch) independently
                let (batch_size, feature_dim) = (self.shape[0], self.shape[1]);
                let mut result = Vec::with_capacity(floats.len());

                for b in 0..batch_size {
                    let start = b * feature_dim;
                    let end = start + feature_dim;
                    let row = &floats[start..end];
                    let softmax_row = self.softmax_1d(row, feature_dim);
                    result.extend(softmax_row);
                }
                result
            }
            _ => {
                return Err(ResonantError::ShapeMismatch(format!(
                    "Softmax only supports 1D and 2D tensors, got shape {:?}",
                    self.shape
                )));
            }
        };

        // Snap back to Q(φ) lattice
        self.lattice = snap_to_lattice(&result_floats, precision);
        self.precision = precision;
        self.recompute_syntony();

        Ok(())
    }

    /// Helper: Softmax for 1D slice (numerically stable).
    fn softmax_1d(&self, x: &[f64], n: usize) -> Vec<f64> {
        if n == 0 {
            return Vec::new();
        }

        // Find max for numerical stability
        let max_val = x.iter().copied().fold(f64::NEG_INFINITY, f64::max);

        // Compute exp(x_i - max) and sum
        let mut exp_vals = Vec::with_capacity(n);
        let mut sum = 0.0;

        for &val in x.iter().take(n) {
            let exp_val = (val - max_val).exp();
            exp_vals.push(exp_val);
            sum += exp_val;
        }

        // Normalize: exp(x_i - max) / sum
        exp_vals.iter().map(|&e| e / sum).collect()
    }

    /// Mean reduction along an optional dimension.
    pub fn mean_core(
        &self,
        dim: Option<usize>,
        keepdim: bool,
        precision: i64,
    ) -> Result<ResonantTensor, ResonantError> {
        if self.is_empty() {
            return Err(ResonantError::ShapeMismatch(
                "Mean on empty tensor".to_string(),
            ));
        }

        let ndim = self.shape.len();
        let axis = match dim {
            Some(d) if d < ndim => Some(d),
            Some(d) => {
                return Err(ResonantError::ShapeMismatch(format!(
                    "Dimension {} out of bounds for {}-D tensor",
                    d, ndim
                )))
            }
            None => None,
        };

        // Global mean
        if axis.is_none() {
            let total: f64 = self.lattice.iter().map(|g| g.to_f64()).sum();
            let mean = total / self.len() as f64;
            let lattice = vec![GoldenExact::find_nearest(mean, precision)];
            let mode_norm_sq = vec![0.0];
            return Self::from_lattice(lattice, vec![1], mode_norm_sq);
        }

        let axis = axis.unwrap();
        let outer: usize = self.shape[..axis].iter().product::<usize>().max(1);
        let inner: usize = self.shape[axis + 1..].iter().product::<usize>().max(1);
        let axis_len = self.shape[axis];

        let mut outputs = Vec::with_capacity(outer * inner);

        for outer_idx in 0..outer {
            for inner_idx in 0..inner {
                let mut acc = 0.0;
                for axis_idx in 0..axis_len {
                    let idx = outer_idx * axis_len * inner + axis_idx * inner + inner_idx;
                    acc += self.lattice[idx].to_f64();
                }
                outputs.push(acc / axis_len as f64);
            }
        }

        let mut result_shape = if keepdim {
            let mut shape = self.shape.clone();
            shape[axis] = 1;
            shape
        } else {
            let mut shape = self.shape.clone();
            shape.remove(axis);
            if shape.is_empty() {
                vec![1]
            } else {
                shape
            }
        };

        let mode_norm_sq: Vec<f64> = (0..outputs.len()).map(|i| (i as f64).powi(2)).collect();

        let lattice = snap_to_lattice(&outputs, precision);
        if result_shape.is_empty() {
            result_shape.push(1);
        }
        Self::from_lattice(lattice, result_shape, mode_norm_sq)
    }

    /// Variance reduction along an optional dimension (population variance).
    pub fn var_core(
        &self,
        dim: Option<usize>,
        keepdim: bool,
        precision: i64,
    ) -> Result<ResonantTensor, ResonantError> {
        if self.is_empty() {
            return Err(ResonantError::ShapeMismatch(
                "Var on empty tensor".to_string(),
            ));
        }

        let ndim = self.shape.len();
        let axis = match dim {
            Some(d) if d < ndim => Some(d),
            Some(d) => {
                return Err(ResonantError::ShapeMismatch(format!(
                    "Dimension {} out of bounds for {}-D tensor",
                    d, ndim
                )))
            }
            None => None,
        };

        // Global variance
        if axis.is_none() {
            let mean: f64 =
                self.lattice.iter().map(|g| g.to_f64()).sum::<f64>() / self.len() as f64;
            let var: f64 = self
                .lattice
                .iter()
                .map(|g| {
                    let v = g.to_f64() - mean;
                    v * v
                })
                .sum::<f64>()
                / self.len() as f64;
            let lattice = vec![GoldenExact::find_nearest(var, precision)];
            let mode_norm_sq = vec![0.0];
            return Self::from_lattice(lattice, vec![1], mode_norm_sq);
        }

        let axis = axis.unwrap();
        let outer: usize = self.shape[..axis].iter().product::<usize>().max(1);
        let inner: usize = self.shape[axis + 1..].iter().product::<usize>().max(1);
        let axis_len = self.shape[axis];

        let mut outputs = Vec::with_capacity(outer * inner);

        for outer_idx in 0..outer {
            for inner_idx in 0..inner {
                let mut acc = 0.0;
                for axis_idx in 0..axis_len {
                    let idx = outer_idx * axis_len * inner + axis_idx * inner + inner_idx;
                    acc += self.lattice[idx].to_f64();
                }
                let mean = acc / axis_len as f64;
                let mut var_acc = 0.0;
                for axis_idx in 0..axis_len {
                    let idx = outer_idx * axis_len * inner + axis_idx * inner + inner_idx;
                    let v = self.lattice[idx].to_f64() - mean;
                    var_acc += v * v;
                }
                outputs.push(var_acc / axis_len as f64);
            }
        }

        let mut result_shape = if keepdim {
            let mut shape = self.shape.clone();
            shape[axis] = 1;
            shape
        } else {
            let mut shape = self.shape.clone();
            shape.remove(axis);
            if shape.is_empty() {
                vec![1]
            } else {
                shape
            }
        };

        let mode_norm_sq: Vec<f64> = (0..outputs.len()).map(|i| (i as f64).powi(2)).collect();

        let lattice = snap_to_lattice(&outputs, precision);
        if result_shape.is_empty() {
            result_shape.push(1);
        }
        Self::from_lattice(lattice, result_shape, mode_norm_sq)
    }

    /// Concatenate tensors along a specified dimension.
    ///
    /// Preserves exact Q(φ) lattice arithmetic by directly combining GoldenExact values.
    /// All tensors must have the same shape except along the concatenation dimension.
    ///
    /// # Arguments
    /// * `tensors` - Slice of ResonantTensor references to concatenate
    /// * `dim` - Dimension to concatenate along (0-indexed)
    ///
    /// # Returns
    /// New ResonantTensor with tensors concatenated along dimension `dim`
    ///
    /// # Example
    /// ```ignore
    /// let a = ResonantTensor::from_floats(&[1.0, 2.0], vec![2], ...);
    /// let b = ResonantTensor::from_floats(&[3.0, 4.0], vec![2], ...);
    /// let c = ResonantTensor::concat(&[&a, &b], 0)?; // Shape: [4]
    /// ```
    pub fn concat_core(
        tensors: &[&ResonantTensor],
        dim: usize,
    ) -> Result<ResonantTensor, ResonantError> {
        if tensors.is_empty() {
            return Err(ResonantError::ShapeMismatch(
                "Cannot concat empty tensor list".to_string(),
            ));
        }

        let first = tensors[0];
        let ndim = first.shape.len();

        if dim >= ndim {
            return Err(ResonantError::ShapeMismatch(format!(
                "Dimension {} out of bounds for {}-dimensional tensor",
                dim, ndim
            )));
        }

        // Validate all tensors have compatible shapes
        for (i, tensor) in tensors.iter().enumerate().skip(1) {
            if tensor.shape.len() != ndim {
                return Err(ResonantError::ShapeMismatch(format!(
                    "Tensor {} has {} dimensions, expected {}",
                    i,
                    tensor.shape.len(),
                    ndim
                )));
            }
            for (d, (&s1, &s2)) in first.shape.iter().zip(tensor.shape.iter()).enumerate() {
                if d != dim && s1 != s2 {
                    return Err(ResonantError::ShapeMismatch(format!(
                        "Tensor {} has incompatible shape {:?}, expected {:?} along dim {}",
                        i, tensor.shape, first.shape, d
                    )));
                }
            }
        }

        // Compute result shape
        let mut result_shape = first.shape.clone();
        result_shape[dim] = tensors.iter().map(|t| t.shape[dim]).sum();

        let outer: usize = first.shape[..dim].iter().product::<usize>().max(1);
        let inner: usize = first.shape[dim + 1..].iter().product::<usize>().max(1);
        let total_size = result_shape.iter().product();

        let mut result_lattice = Vec::with_capacity(total_size);
        let mut result_norms = Vec::with_capacity(total_size);

        // Copy contiguous slices for each outer index, preserving per-element mode norms
        for outer_idx in 0..outer {
            for tensor in tensors.iter() {
                let axis = tensor.shape[dim];
                for axis_idx in 0..axis {
                    let start = outer_idx * axis * inner + axis_idx * inner;
                    let end = start + inner;
                    result_lattice.extend_from_slice(&tensor.lattice[start..end]);
                    result_norms.extend_from_slice(&tensor.mode_norm_sq[start..end]);
                }
            }
        }

        Self::from_lattice(result_lattice, result_shape, result_norms)
    }

    /// Layer normalization with optional golden target variance.
    ///
    /// Normalizes across the last dimension (features).
    /// For 2D tensors [batch, features], normalizes each batch sample independently.
    ///
    /// # Arguments
    /// * `gamma` - Optional scale parameter (broadcast to features)
    /// * `beta` - Optional shift parameter (broadcast to features)
    /// * `eps` - Small constant for numerical stability (default: 1e-8)
    /// * `golden_target` - If true, scale to target variance = 1/φ
    pub fn layer_norm_core(
        &self,
        gamma: Option<&ResonantTensor>,
        beta: Option<&ResonantTensor>,
        eps: f64,
        golden_target: bool,
    ) -> Result<ResonantTensor, ResonantError> {
        if self.shape.len() < 1 {
            return Err(ResonantError::ShapeMismatch(
                "LayerNorm requires at least 1D tensor".to_string(),
            ));
        }

        let feature_dim = self.shape[self.shape.len() - 1];
        let batch_size: usize = self.shape[..self.shape.len() - 1].iter().product();
        let batch_size = if batch_size == 0 { 1 } else { batch_size };

        // Validate gamma/beta shapes if provided
        if let Some(g) = gamma {
            if g.len() != feature_dim {
                return Err(ResonantError::ShapeMismatch(format!(
                    "Gamma length {} must match feature_dim {}",
                    g.len(),
                    feature_dim
                )));
            }
        }
        if let Some(b) = beta {
            if b.len() != feature_dim {
                return Err(ResonantError::ShapeMismatch(format!(
                    "Beta length {} must match feature_dim {}",
                    b.len(),
                    feature_dim
                )));
            }
        }

        let mut result_floats = Vec::with_capacity(self.len());

        // Process each sample
        for b in 0..batch_size {
            let offset = b * feature_dim;
            let sample: Vec<f64> = (0..feature_dim)
                .map(|i| self.lattice[offset + i].to_f64())
                .collect();

            // Compute mean
            let mean: f64 = sample.iter().sum::<f64>() / feature_dim as f64;

            // Compute variance
            let var: f64 =
                sample.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / feature_dim as f64;

            // Compute reciprocal standard deviation
            let mut rstd = 1.0 / (var + eps).sqrt();

            // Golden scaling: scale to variance = 1/φ
            if golden_target {
                rstd *= PHI_INV.sqrt();
            }

            // Normalize and apply affine transform
            for i in 0..feature_dim {
                let mut val = (sample[i] - mean) * rstd;

                // Apply gamma (scale)
                if let Some(g) = gamma {
                    val *= g.lattice[i].to_f64();
                }

                // Apply beta (shift)
                if let Some(b) = beta {
                    val += b.lattice[i].to_f64();
                }

                result_floats.push(val);
            }
        }

        // Snap to lattice and create result tensor
        let result_lattice = snap_to_lattice(&result_floats, self.precision);
        let result_norms = self.mode_norm_sq.clone();

        Self::from_lattice(result_lattice, self.shape.clone(), result_norms)
    }

    /// Complete a full D→H cycle for a batch of samples.
    ///
    /// Assumes the first dimension of the shape is the batch dimension.
    /// Applies the same mode_norm_sq sequence to each sample in the batch.
    ///
    /// Returns a vector of new syntony values for each sample.
    pub fn run_batch_cpu_cycle(
        &mut self,
        noise_scale: f64,
        precision: i64,
    ) -> Result<Vec<f64>, ResonantError> {
        use rand::Rng;

        if self.shape.len() < 2 {
            return Err(ResonantError::ShapeMismatch(
                "batch_cpu_cycle requires at least 2 dimensions [batch, dim, ...]".to_string(),
            ));
        }

        let batch_size = self.shape[0];
        let sample_dim: usize = self.shape[1..].iter().product();

        if self.mode_norm_sq.len() != sample_dim && self.mode_norm_sq.len() != self.len() {
            return Err(ResonantError::ShapeMismatch(format!(
                "mode_norm_sq length {} must match sample dimension {} or total length {}",
                self.mode_norm_sq.len(),
                sample_dim,
                self.len()
            )));
        }

        // Wake flux
        let mut values = self.to_floats_core();
        self.phase = ResonantPhase::Flux;

        let mut rng = rand::thread_rng();
        let mut batch_syntony = Vec::with_capacity(batch_size);

        // Process each sample in the batch
        for b in 0..batch_size {
            let offset = b * sample_dim;
            let sample_slice = &mut values[offset..offset + sample_dim];

            // Current syntony (approximate for the whole batch or per-sample?)
            // For now, use the tensor's global syntony to drive the D-phase
            let s = self.syntony;
            let alpha = PHI_INV_SQ * (1.0 - s);

            for (i, v) in sample_slice.iter_mut().enumerate() {
                let norm_sq = self.mode_norm_sq[i];
                let scale = 1.0 + alpha * norm_sq.sqrt();
                let noise: f64 = rng.gen::<f64>() - 0.5;
                *v = *v * scale + noise * noise_scale * (1.0 - s);
            }
        }

        // Crystallize: returns exact lattice
        self.lattice =
            harmonize_and_crystallize(&values, &vec![0.0; values.len()], self.syntony, precision);
        // Note: harmonize_and_crystallize above uses dummy mode_norms because we already applied DH logic?
        // Wait, harmonize_and_crystallize applies H-phase (attenuation).
        // I should probably implement a batch version of harmonize_and_crystallize too.

        // Actually, let's just use the logic directly here for clarity
        let beta = PHI_INV * self.syntony;
        self.lattice = values
            .iter()
            .enumerate()
            .map(|(idx, &val)| {
                let i = idx % sample_dim;
                let norm_sq = self.mode_norm_sq[i];
                let golden_weight = (-norm_sq / PHI).exp();
                let h_scale = 1.0 - beta * (1.0 - golden_weight);
                GoldenExact::find_nearest(val * h_scale, precision)
            })
            .collect();

        self.precision = precision;
        self.phase = ResonantPhase::Crystallized;

        // Compute per-sample syntony for return values
        // Note: each sample shares the same mode structure, so we use the first sample_dim norms
        let sample_norms = &self.mode_norm_sq[0..sample_dim];
        for b in 0..batch_size {
            let offset = b * sample_dim;
            let sample_lattice = &self.lattice[offset..offset + sample_dim];
            batch_syntony.push(compute_lattice_syntony(sample_lattice, sample_norms));
        }

        // Update global syntony as average
        self.syntony = batch_syntony.iter().sum::<f64>() / batch_size as f64;

        Ok(batch_syntony)
    }
}

// =========================================================================
// PyO3 Methods
// =========================================================================

#[pymethods]
impl ResonantTensor {
    /// Create a ResonantTensor from a list of floats.
    ///
    /// Args:
    ///     data: List of floating-point values
    ///     shape: Shape of the tensor
    ///     mode_norm_sq: Precomputed |n|² for each mode (optional)
    ///     precision: Maximum coefficient for golden lattice (default: 100)
    ///
    /// Returns:
    ///     A new ResonantTensor in Crystallized phase
    #[new]
    #[pyo3(signature = (data, shape, mode_norm_sq=None, precision=100))]
    fn py_new(
        data: Vec<f64>,
        shape: Vec<usize>,
        mode_norm_sq: Option<Vec<f64>>,
        precision: i64,
    ) -> PyResult<Self> {
        let mode_norms =
            mode_norm_sq.unwrap_or_else(|| (0..data.len()).map(|i| (i as f64).powi(2)).collect());
        Self::from_floats(&data, shape, mode_norms, precision).map_err(|e| e.into())
    }

    /// Create from a list of GoldenExact values.
    #[staticmethod]
    #[pyo3(signature = (lattice, shape, mode_norm_sq=None))]
    fn from_golden_exact(
        lattice: Vec<GoldenExact>,
        shape: Vec<usize>,
        mode_norm_sq: Option<Vec<f64>>,
    ) -> PyResult<Self> {
        let mode_norms = mode_norm_sq
            .unwrap_or_else(|| (0..lattice.len()).map(|i| (i as f64).powi(2)).collect());
        Self::from_lattice(lattice, shape, mode_norms).map_err(|e| e.into())
    }

    /// Matrix multiplication: self @ weights.
    ///
    /// Args:
    ///     weights: Another ResonantTensor (W)
    ///
    /// Returns:
    ///     New ResonantTensor representing self * weights^T
    fn matmul(&self, weights: &ResonantTensor) -> PyResult<Self> {
        self.matmul_core(weights).map_err(|e| e.into())
    }

    /// Add bias to the tensor.
    fn add_bias(&mut self, bias: &ResonantTensor) -> PyResult<()> {
        self.add_bias_core(bias).map_err(|e| e.into())
    }

    /// Apply ReLU activation.
    fn relu(&mut self) {
        self.relu_core();
    }

    /// Complete a full D→H cycle for a batch of samples.
    fn batch_cpu_cycle(&mut self, noise_scale: f64, precision: i64) -> PyResult<Vec<f64>> {
        self.run_batch_cpu_cycle(noise_scale, precision)
            .map_err(|e| e.into())
    }

    /// Convert lattice to a list of GoldenExact objects.
    fn to_lattice_list(&self) -> Vec<GoldenExact> {
        self.lattice.clone()
    }

    /// Get the current syntony value.
    #[getter]
    fn get_syntony(&self) -> f64 {
        self.syntony
    }

    /// Get the current phase as a string.
    #[getter]
    fn get_phase(&self) -> String {
        self.phase.to_string()
    }

    /// Get the tensor shape.
    #[getter]
    fn get_shape(&self) -> Vec<usize> {
        self.shape.clone()
    }

    /// Get the number of elements.
    fn __len__(&self) -> usize {
        self.len()
    }

    /// Get the precision used for last crystallization.
    #[getter]
    fn get_precision(&self) -> i64 {
        self.precision()
    }

    /// String representation.
    fn __repr__(&self) -> String {
        format!(
            "ResonantTensor(shape={:?}, phase={}, syntony={:.4}, precision={})",
            self.shape, self.phase, self.syntony, self.precision
        )
    }

    /// Convert to list of floats.
    fn to_list(&self) -> Vec<f64> {
        self.to_floats_core()
    }

    /// Alias for to_list()
    fn to_floats(&self) -> Vec<f64> {
        self.to_floats_core()
    }

    /// Get the lattice as a list of GoldenExact values.
    fn get_lattice(&self) -> Vec<GoldenExact> {
        self.lattice.clone()
    }

    /// Get the mode norm squared values.
    fn get_mode_norm_sq(&self) -> Vec<f64> {
        self.mode_norm_sq.clone()
    }

    /// Crystallize CPU flux values (for testing without GPU).
    fn crystallize_from_values(&mut self, values: Vec<f64>, precision: i64) -> PyResult<f64> {
        self.crystallize_cpu(&values, precision)
            .map_err(|e| e.into())
    }

    /// Wake flux in CPU mode (for testing without GPU).
    /// Returns the projected float values.
    fn wake_flux_values(&mut self) -> PyResult<Vec<f64>> {
        if self.phase != ResonantPhase::Crystallized {
            return Err(ResonantError::InvalidPhaseTransition(
                "wake_flux requires Crystallized phase".to_string(),
            )
            .into());
        }

        let floats = self.to_floats_core();
        self.phase = ResonantPhase::Flux;
        Ok(floats)
    }

    /// Complete a full D→H cycle in CPU mode.
    ///
    /// This simulates what would happen with GPU differentiation
    /// by applying noise to the values and then crystallizing.
    fn cpu_cycle(&mut self, noise_scale: f64, precision: i64) -> PyResult<f64> {
        self.run_cpu_cycle(noise_scale, precision)
            .map_err(|e| e.into())
    }

    /// Apply the Golden Recursion Map R(n) = floor(phi * n).
    ///
    /// Scales all lattice values by phi (exactly).
    fn apply_recursion(&mut self) {
        self.apply_recursion_core();
    }

    /// Apply the inverse Golden Recursion Map R^{-1}(n).
    ///
    /// Scales all lattice values by 1/phi (exactly).
    fn apply_inverse_recursion(&mut self) {
        self.apply_inverse_recursion_core();
    }

    /// Apply hierarchical pruning to the lattice.
    ///
    /// Values with magnitude < q/divisor are snapped to zero.
    #[pyo3(signature = (q, divisor=248.0))]
    fn prune_hierarchy(&mut self, q: f64, divisor: f64) {
        self.prune_hierarchy_core(q, divisor);
    }

    /// Apply sigmoid activation: σ(x) = 1 / (1 + e^(-x)).
    ///
    /// Converts to floats, applies sigmoid, snaps back to Q(φ) lattice.
    ///
    /// Args:
    ///     precision: Lattice precision for crystallization (default: 100)
    #[pyo3(signature = (precision=100))]
    fn sigmoid(&mut self, precision: i64) {
        self.sigmoid_core(precision);
    }

    /// Apply tanh activation.
    ///
    /// Args:
    ///     precision: Lattice precision for crystallization (default: 100)
    #[pyo3(signature = (precision=100))]
    fn tanh(&mut self, precision: i64) {
        self.tanh_core(precision);
    }

    /// Apply GELU activation (erf-based) and snap to Q(φ) lattice.
    #[pyo3(signature = (precision=100))]
    fn gelu(&mut self, precision: i64) {
        self.gelu_core(precision);
    }

    /// Element-wise multiplication (Hadamard product): self * other.
    ///
    /// Args:
    ///     other: Another ResonantTensor with the same shape
    ///
    /// Returns:
    ///     New ResonantTensor with element-wise product
    fn elementwise_mul(&self, other: &ResonantTensor) -> PyResult<Self> {
        self.elementwise_mul_core(other).map_err(|e| e.into())
    }

    /// Element-wise addition: self + other.
    ///
    /// Args:
    ///     other: Another ResonantTensor with the same shape
    ///
    /// Returns:
    ///     New ResonantTensor with element-wise sum
    fn elementwise_add(&self, other: &ResonantTensor) -> PyResult<Self> {
        self.elementwise_add_core(other).map_err(|e| e.into())
    }

    /// Scalar multiplication: self * scalar.
    ///
    /// Multiplies every element by a scalar value.
    ///
    /// Args:
    ///     scalar: Scalar value to multiply by
    ///
    /// Returns:
    ///     New ResonantTensor with all elements multiplied by scalar
    ///
    /// Example:
    ///     >>> x = ResonantTensor([1.0, 2.0, 3.0], [3])
    ///     >>> y = x.scalar_mul(2.0)
    ///     >>> y.to_floats()
    ///     [2.0, 4.0, 6.0]
    fn scalar_mul(&self, scalar: f64) -> PyResult<Self> {
        self.scalar_mul_core(scalar).map_err(|e| e.into())
    }

    /// Scalar addition: self + scalar.
    ///
    /// Adds a scalar value to every element.
    ///
    /// Args:
    ///     scalar: Scalar value to add
    ///
    /// Returns:
    ///     New ResonantTensor with scalar added to all elements
    ///
    /// Example:
    ///     >>> x = ResonantTensor([1.0, 2.0, 3.0], [3])
    ///     >>> y = x.scalar_add(10.0)
    ///     >>> y.to_floats()
    ///     [11.0, 12.0, 13.0]
    fn scalar_add(&self, scalar: f64) -> PyResult<Self> {
        self.scalar_add_core(scalar).map_err(|e| e.into())
    }

    /// Negate: -self.
    ///
    /// Multiplies every element by -1.
    ///
    /// Returns:
    ///     New ResonantTensor with all elements negated
    ///
    /// Example:
    ///     >>> x = ResonantTensor([1.0, -2.0, 3.0], [3])
    ///     >>> y = x.negate()
    ///     >>> y.to_floats()
    ///     [-1.0, 2.0, -3.0]
    fn negate(&self) -> PyResult<Self> {
        self.negate_core().map_err(|e| e.into())
    }

    /// One minus: 1 - self.
    ///
    /// Computes 1 - x for every element.
    /// Common pattern in gating mechanisms.
    ///
    /// Returns:
    ///     New ResonantTensor with 1 - x for each element
    ///
    /// Example:
    ///     >>> x = ResonantTensor([0.2, 0.5, 0.8], [3])
    ///     >>> y = x.one_minus()
    ///     >>> y.to_floats()
    ///     [0.8, 0.5, 0.2]
    fn one_minus(&self) -> PyResult<Self> {
        self.one_minus_core().map_err(|e| e.into())
    }

    /// Natural logarithm.
    #[pyo3(signature = (precision=None))]
    fn log(&self, precision: Option<i64>) -> PyResult<Self> {
        let prec = precision.unwrap_or(self.precision);
        self.log_core(prec).map_err(|e| e.into())
    }

    /// Natural exponential.
    #[pyo3(signature = (precision=None))]
    fn exp(&self, precision: Option<i64>) -> PyResult<Self> {
        let prec = precision.unwrap_or(self.precision);
        self.exp_core(prec).map_err(|e| e.into())
    }

    /// Applies softmax normalization along the last dimension.
    ///
    /// For 1D tensors: softmax(x_i) = exp(x_i) / sum(exp(x_j) for all j)
    /// For 2D tensors: applies softmax independently to each row
    ///
    /// Uses numerically stable computation: exp(x_i - max(x)) / sum(exp(x_j - max(x)))
    /// Snaps result back to Q(φ) lattice with specified precision.
    ///
    /// Args:
    ///     precision: Precision for lattice snapping (default: 32)
    ///
    /// Returns:
    ///     Self (mutated in-place)
    ///
    /// Example:
    ///     >>> x = ResonantTensor([1.0, 2.0, 3.0], [3])
    ///     >>> x.softmax()
    ///     >>> x.to_floats()
    ///     [0.09003057, 0.24472847, 0.66524096]
    #[pyo3(signature = (precision=32))]
    fn softmax(&mut self, precision: i64) -> PyResult<()> {
        self.softmax_core(precision).map_err(|e| e.into())
    }

    /// Concatenate tensors along a specified dimension.
    ///
    /// Preserves exact Q(φ) lattice arithmetic. All tensors must have compatible shapes.
    ///
    /// Args:
    ///     tensors: List of ResonantTensor objects to concatenate
    ///     dim: Dimension along which to concatenate (default: -1, last dimension)
    ///
    /// Returns:
    ///     New ResonantTensor with tensors concatenated
    ///
    /// Example:
    ///     >>> a = ResonantTensor([1.0, 2.0], [2])
    ///     >>> b = ResonantTensor([3.0, 4.0], [2])
    ///     >>> c = ResonantTensor.concat([a, b], dim=0)
    ///     >>> c.shape
    ///     [4]
    #[staticmethod]
    #[pyo3(signature = (tensors, dim=-1))]
    fn concat(py: Python, tensors: Vec<Py<ResonantTensor>>, dim: i32) -> PyResult<Self> {
        if tensors.is_empty() {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "Cannot concatenate empty tensor list",
            ));
        }

        // Borrow all tensors and collect PyRef objects
        let borrowed_tensors: Vec<pyo3::PyRef<ResonantTensor>> =
            tensors.iter().map(|t| t.borrow(py)).collect();

        // Convert to slice of references
        let tensor_refs: Vec<&ResonantTensor> = borrowed_tensors.iter().map(|t| &**t).collect();

        // Handle negative dimension indexing
        let ndim = tensor_refs[0].shape.len() as i32;
        let actual_dim = if dim < 0 {
            (ndim + dim) as usize
        } else {
            dim as usize
        };

        Self::concat_core(&tensor_refs, actual_dim).map_err(|e| e.into())
    }

    /// Layer normalization with optional golden target variance.
    ///
    /// Normalizes across the last dimension (features).
    ///
    /// Args:
    ///     gamma: Optional scale parameter (ResonantTensor of shape [features])
    ///     beta: Optional shift parameter (ResonantTensor of shape [features])
    ///     eps: Small constant for numerical stability (default: 1e-8)
    ///     golden_target: If True, scale to target variance = 1/φ (default: True)
    ///
    /// Returns:
    ///     New ResonantTensor with normalized values
    #[pyo3(signature = (gamma=None, beta=None, eps=1e-8, golden_target=true))]
    fn layer_norm(
        &self,
        gamma: Option<&ResonantTensor>,
        beta: Option<&ResonantTensor>,
        eps: f64,
        golden_target: bool,
    ) -> PyResult<Self> {
        self.layer_norm_core(gamma, beta, eps, golden_target)
            .map_err(|e| e.into())
    }

    /// Mean reduction along an optional dimension.
    #[pyo3(signature = (dim=None, keepdim=false, precision=None))]
    fn mean(&self, dim: Option<i32>, keepdim: bool, precision: Option<i64>) -> PyResult<Self> {
        let ndim = self.shape.len() as i32;
        let axis = match dim {
            Some(d) if d < 0 => {
                let adj = ndim + d;
                if adj < 0 {
                    return Err(ResonantError::ShapeMismatch(format!(
                        "Dimension {} out of bounds for {}-D tensor",
                        d, ndim
                    ))
                    .into());
                }
                Some(adj as usize)
            }
            Some(d) => Some(d as usize),
            None => None,
        };
        let prec = precision.unwrap_or(self.precision);
        self.mean_core(axis, keepdim, prec).map_err(|e| e.into())
    }

    /// Variance reduction along an optional dimension (population variance).
    #[pyo3(signature = (dim=None, keepdim=false, precision=None))]
    fn var(&self, dim: Option<i32>, keepdim: bool, precision: Option<i64>) -> PyResult<Self> {
        let ndim = self.shape.len() as i32;
        let axis = match dim {
            Some(d) if d < 0 => {
                let adj = ndim + d;
                if adj < 0 {
                    return Err(ResonantError::ShapeMismatch(format!(
                        "Dimension {} out of bounds for {}-D tensor",
                        d, ndim
                    ))
                    .into());
                }
                Some(adj as usize)
            }
            Some(d) => Some(d as usize),
            None => None,
        };
        let prec = precision.unwrap_or(self.precision);
        self.var_core(axis, keepdim, prec).map_err(|e| e.into())
    }

    // =========================================================================
    // CUDA-specific Methods (compiled only with cuda feature)
    // =========================================================================

    /// Run a full D→H cycle using CUDA for the D-phase and CPU crystallization for H-phase.
    ///
    /// Args:
    ///     device_idx: CUDA device index (usize)
    ///     noise_scale: stochastic noise scale for D-phase
    ///     precision: precision for crystallization
    #[cfg(feature = "cuda")]
    #[pyo3(signature = (device_idx, noise_scale=0.1, precision=100))]
    fn cuda_cycle_gpu(
        &mut self,
        device_idx: usize,
        noise_scale: f64,
        precision: i64,
    ) -> PyResult<f64> {
        let device = crate::tensor::cuda::device_manager::get_device(device_idx).map_err(|e| {
            pyo3::exceptions::PyRuntimeError::new_err(format!("cuda device error: {}", e))
        })?;

        self.cuda_cycle(device, noise_scale, precision)
            .map_err(|e| e.into())
    }

    /// Wake flux with D-phase on the specified CUDA device (GPU-only helper).
    #[cfg(feature = "cuda")]
    #[pyo3(signature = (device_idx, noise_scale=0.1))]
    fn wake_flux_with_d_phase_py(&mut self, device_idx: usize, noise_scale: f64) -> PyResult<()> {
        let device = crate::tensor::cuda::device_manager::get_device(device_idx).map_err(|e| {
            pyo3::exceptions::PyRuntimeError::new_err(format!("cuda device error: {}", e))
        })?;

        self.wake_flux_with_d_phase(device, noise_scale)
            .map_err(|e| e.into())
    }
}

impl Clone for ResonantTensor {
    fn clone(&self) -> Self {
        ResonantTensor {
            lattice: self.lattice.clone(),
            #[cfg(feature = "cuda")]
            flux: None, // Don't clone GPU memory
            cpu_flux: None,
            shape: self.shape.clone(),
            syntony: self.syntony,
            mode_norm_sq: self.mode_norm_sq.clone(),
            phase: ResonantPhase::Crystallized, // Clone always starts crystallized
            #[cfg(feature = "cuda")]
            device: None,
            device_idx: self.device_idx,
            last_d_duration_ns: 0,
            precision: self.precision,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_from_floats() {
        let data = vec![1.0, 2.0, 3.0, 4.0];
        let tensor = ResonantTensor::from_floats_default_modes(&data, vec![4], 100)
            .expect("Should create tensor");

        assert_eq!(tensor.len(), 4);
        assert_eq!(tensor.shape(), &[4]);
        assert_eq!(tensor.phase(), ResonantPhase::Crystallized);
        assert!(tensor.syntony() >= 0.0 && tensor.syntony() <= 1.0);
    }

    #[test]
    fn test_to_floats() {
        let data = vec![1.0, PHI, 3.0, 4.0];
        let tensor = ResonantTensor::from_floats_default_modes(&data, vec![4], 100)
            .expect("Should create tensor");

        let recovered = tensor.to_floats_core();
        assert_eq!(recovered.len(), 4);

        // Values should be close to originals
        for (orig, recovered) in data.iter().zip(recovered.iter()) {
            assert!((orig - recovered).abs() < 0.1);
        }
    }

    #[test]
    fn test_cpu_cycle() {
        let data = vec![1.0, 2.0, 3.0, 4.0];
        let mut tensor = ResonantTensor::from_floats_default_modes(&data, vec![4], 100)
            .expect("Should create tensor");

        let initial_syntony = tensor.syntony();

        // Run a CPU cycle
        let new_syntony = tensor.cpu_cycle(0.1, 100).expect("Should complete cycle");

        assert!(new_syntony >= 0.0 && new_syntony <= 1.0);
        assert_eq!(tensor.phase(), ResonantPhase::Crystallized);
    }

    #[test]
    fn test_clone() {
        let data = vec![1.0, 2.0, 3.0, 4.0];
        let tensor = ResonantTensor::from_floats_default_modes(&data, vec![4], 100)
            .expect("Should create tensor");

        let cloned = tensor.clone();

        assert_eq!(cloned.len(), tensor.len());
        assert_eq!(cloned.syntony(), tensor.syntony());
        assert_eq!(cloned.phase(), ResonantPhase::Crystallized);
    }

    #[test]
    fn test_concat_semantics() {
        let a = ResonantTensor::from_floats_default_modes(&[1.0, 2.0, 3.0, 4.0], vec![2, 2], 100)
            .expect("create a");
        let b = ResonantTensor::from_floats_default_modes(&[5.0, 6.0, 7.0, 8.0], vec![2, 2], 100)
            .expect("create b");

        let c = ResonantTensor::concat_core(&[&a, &b], 1).expect("concat");
        assert_eq!(c.shape(), &[2, 4]);

        let floats = c.to_floats_core();
        let expected = vec![1.0, 2.0, 5.0, 6.0, 3.0, 4.0, 7.0, 8.0];
        for (x, y) in floats.iter().zip(expected.iter()) {
            assert!((x - y).abs() < 1e-6);
        }
    }

    #[test]
    fn test_mean_var_axis() {
        let tensor =
            ResonantTensor::from_floats_default_modes(&[1.0, 2.0, 3.0, 5.0], vec![2, 2], 200)
                .expect("create tensor");

        let mean = tensor.mean_core(Some(1), false, 200).expect("mean");
        assert_eq!(mean.shape(), &[2]);
        let m = mean.to_floats_core();
        assert!((m[0] - 1.5).abs() < 1e-6);
        assert!((m[1] - 4.0).abs() < 1e-6);

        let var = tensor.var_core(Some(1), false, 200).expect("var");
        assert_eq!(var.shape(), &[2]);
        let v = var.to_floats_core();
        assert!((v[0] - 0.25).abs() < 1e-6);
        assert!((v[1] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_gelu_activation() {
        let mut tensor = ResonantTensor::from_floats_default_modes(&[-1.0, 0.0, 1.0], vec![3], 150)
            .expect("create tensor");
        tensor.gelu_core(200);
        let vals = tensor.to_floats_core();
        assert!(vals[0] < 0.0);
        assert!(vals[1].abs() < 1e-6);
        assert!(vals[2] > 0.0);
    }

    #[test]
    fn test_layer_norm_golden_target() {
        let tensor =
            ResonantTensor::from_floats_default_modes(&[1.0, 2.0, 3.0, 4.0], vec![2, 2], 300)
                .expect("create tensor");
        let out = tensor
            .layer_norm_core(None, None, 1e-8, true)
            .expect("layer norm");
        let vals = out.to_floats_core();

        // Per-sample mean ≈ 0 and variance ≈ 1/φ
        for sample in 0..2 {
            let start = sample * 2;
            let sample_vals = &vals[start..start + 2];
            let mean: f64 = sample_vals.iter().sum::<f64>() / 2.0;
            let var: f64 = sample_vals
                .iter()
                .map(|v| {
                    let d = *v - mean;
                    d * d
                })
                .sum::<f64>()
                / 2.0;

            assert!(mean.abs() < 1e-6);
            assert!((var - PHI_INV).abs() < 1e-3);
        }
    }
}
