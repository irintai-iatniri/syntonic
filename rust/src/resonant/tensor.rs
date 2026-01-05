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

use crate::exact::GoldenExact;
use super::crystallize::{compute_lattice_syntony, crystallize_with_dwell, harmonize_and_crystallize, snap_to_lattice};
use super::{PHI, PHI_INV, PHI_INV_SQ};

#[cfg(feature = "cuda")]
use cudarc::driver::{CudaDevice, CudaSlice, LaunchAsync, LaunchConfig};
#[cfg(feature = "cuda")]
use std::sync::Arc;
#[cfg(feature = "cuda")]
use crate::tensor::srt_kernels::{
    cuda_resonant_d_phase_f64,
    cuda_resonant_compute_syntony_f64,
    ensure_srt_kernels_loaded,
};

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
            ResonantError::InvalidPhaseTransition(msg) => write!(f, "Invalid phase transition: {}", msg),
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
pub struct ResonantTensor {
    /// Exact lattice representation: a + b·φ for each element
    lattice: Vec<GoldenExact>,

    /// GPU flux representation (active during D-phase only)
    #[cfg(feature = "cuda")]
    flux: Option<CudaSlice<f64>>,

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
            return Err(ResonantError::ShapeMismatch(
                format!("Lattice size {} doesn't match shape {:?}", lattice.len(), shape)
            ));
        }
        if mode_norm_sq.len() != lattice.len() {
            return Err(ResonantError::ShapeMismatch(
                format!("mode_norm_sq size {} doesn't match lattice size {}", mode_norm_sq.len(), lattice.len())
            ));
        }

        let syntony = compute_lattice_syntony(&lattice, &mode_norm_sq);

        Ok(ResonantTensor {
            lattice,
            #[cfg(feature = "cuda")]
            flux: None,
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
        let mode_norm_sq: Vec<f64> = (0..data.len())
            .map(|i| (i as f64).powi(2))
            .collect();
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
                "wake_flux requires Crystallized phase".to_string()
            ));
        }

        self.phase = ResonantPhase::Transitioning;

        // Project exact lattice to f64
        let floats: Vec<f64> = self.lattice
            .iter()
            .map(|g| g.to_f64())
            .collect();

        // Upload to GPU
        let gpu_slice = device.htod_sync_copy(&floats)
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
                "wake_flux requires Crystallized phase".to_string()
            ));
        }

        // Project exact lattice to f64
        let floats: Vec<f64> = self.lattice
            .iter()
            .map(|g| g.to_f64())
            .collect();

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
                "crystallize requires Flux phase".to_string()
            ));
        }

        let flux = self.flux.as_ref()
            .ok_or(ResonantError::NoFluxPresent)?;
        let device = self.device.as_ref()
            .ok_or(ResonantError::NoDevicePresent)?;

        self.phase = ResonantPhase::Transitioning;

        // Download from GPU
        let host_data = device.dtoh_sync_copy(flux)
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
                "crystallize requires Flux phase".to_string()
            ));
        }

        let flux = self.flux.as_ref()
            .ok_or(ResonantError::NoFluxPresent)?;
        let device = self.device.as_ref()
            .ok_or(ResonantError::NoDevicePresent)?;

        self.phase = ResonantPhase::Transitioning;

        // Download from GPU
        let host_data = device.dtoh_sync_copy(flux)
            .map_err(|e| ResonantError::CudaError(e.to_string()))?;

        // Crystallize with Ĥ attenuation and φ-dwell timing
        // Pass mode_norm_sq and syntony for Ĥ operator
        let (new_lattice, final_precision, _actual_duration) =
            crystallize_with_dwell(
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
    pub fn crystallize_cpu(&mut self, values: &[f64], precision: i64) -> Result<f64, ResonantError> {
        if values.len() != self.lattice.len() {
            return Err(ResonantError::ShapeMismatch(
                format!("Values length {} doesn't match lattice length {}", values.len(), self.lattice.len())
            ));
        }

        // Apply Ĥ attenuation + snap to lattice
        self.lattice = harmonize_and_crystallize(values, &self.mode_norm_sq, self.syntony, precision);
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
                "wake_flux requires Crystallized phase".to_string()
            ));
        }

        self.phase = ResonantPhase::Transitioning;

        // Ensure kernels are loaded
        ensure_srt_kernels_loaded(&device)
            .map_err(|e| ResonantError::CudaError(e.to_string()))?;

        // Project exact lattice to f64
        let floats: Vec<f64> = self.lattice.iter().map(|g| g.to_f64()).collect();
        let n = floats.len();

        // Upload lattice and mode norms to GPU
        let gpu_lattice = device.htod_sync_copy(&floats)
            .map_err(|e| ResonantError::CudaError(e.to_string()))?;
        let gpu_mode_norms = device.htod_sync_copy(&self.mode_norm_sq)
            .map_err(|e| ResonantError::CudaError(e.to_string()))?;

        // Generate Gaussian noise on CPU and upload
        let mut rng = rand::thread_rng();
        let noise: Vec<f64> = (0..n).map(|_| rng.gen::<f64>() * 2.0 - 1.0).collect();
        let gpu_noise = device.htod_sync_copy(&noise)
            .map_err(|e| ResonantError::CudaError(e.to_string()))?;

        // Allocate output buffer
        let mut gpu_flux: CudaSlice<f64> = device.alloc_zeros(n)
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
        ).map_err(|e| ResonantError::CudaError(e.to_string()))?;
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
        let flux = self.flux.as_ref()
            .ok_or(ResonantError::NoFluxPresent)?;
        let device = self.device.as_ref()
            .ok_or(ResonantError::NoDevicePresent)?;

        // Upload mode norms if needed (they may already be there, but simpler to re-upload)
        let gpu_mode_norms = device.htod_sync_copy(&self.mode_norm_sq)
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
    pub fn to_floats(&self) -> Vec<f64> {
        self.lattice.iter().map(|g| g.to_f64()).collect()
    }

    /// Get the precision used for last crystallization.
    pub fn precision(&self) -> i64 {
        self.precision
    }

    /// Set the device index for CUDA operations.
    pub fn set_device_idx(&mut self, idx: usize) {
        self.device_idx = idx;
    }

    /// Recompute syntony from current lattice state.
    pub fn recompute_syntony(&mut self) {
        self.syntony = compute_lattice_syntony(&self.lattice, &self.mode_norm_sq);
    }

    /// Complete a full D→H cycle in CPU mode (Rust-accessible).
    ///
    /// This simulates what would happen with GPU differentiation
    /// by applying noise to the values and then crystallizing.
    ///
    /// Returns the new syntony value.
    pub fn run_cpu_cycle(&mut self, noise_scale: f64, precision: i64) -> Result<f64, ResonantError> {
        use rand::Rng;

        // Wake flux
        let mut values = self.to_floats();
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
        let mode_norms = mode_norm_sq.unwrap_or_else(|| {
            (0..data.len()).map(|i| (i as f64).powi(2)).collect()
        });
        Self::from_floats(&data, shape, mode_norms, precision)
            .map_err(|e| e.into())
    }

    /// Create from a list of GoldenExact values.
    #[staticmethod]
    #[pyo3(signature = (lattice, shape, mode_norm_sq=None))]
    fn from_golden_exact(
        lattice: Vec<GoldenExact>,
        shape: Vec<usize>,
        mode_norm_sq: Option<Vec<f64>>,
    ) -> PyResult<Self> {
        let mode_norms = mode_norm_sq.unwrap_or_else(|| {
            (0..lattice.len()).map(|i| (i as f64).powi(2)).collect()
        });
        Self::from_lattice(lattice, shape, mode_norms)
            .map_err(|e| e.into())
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

    /// String representation.
    fn __repr__(&self) -> String {
        format!(
            "ResonantTensor(shape={:?}, phase={}, syntony={:.4}, precision={})",
            self.shape, self.phase, self.syntony, self.precision
        )
    }

    /// Convert to list of floats.
    fn to_list(&self) -> Vec<f64> {
        self.to_floats()
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
                "wake_flux requires Crystallized phase".to_string()
            ).into());
        }

        let floats = self.to_floats();
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
}

impl Clone for ResonantTensor {
    fn clone(&self) -> Self {
        ResonantTensor {
            lattice: self.lattice.clone(),
            #[cfg(feature = "cuda")]
            flux: None, // Don't clone GPU memory
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

        let recovered = tensor.to_floats();
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
        let new_syntony = tensor.cpu_cycle(0.1, 100)
            .expect("Should complete cycle");

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
}
