//! Syntony-weighted softmax for theory-aligned classification.
//!
//! Standard softmax: p_i = exp(x_i) / Σ exp(x_j)
//! Syntonic softmax: p_i = w_i * exp(x_i) / Σ w_j * exp(x_j)
//!
//! where w(n) = exp(-|n|²/φ) is the golden measure weight.

use pyo3::prelude::*;
use crate::resonant::{ResonantTensor, ResonantPhase};
use super::PHI;


/// Syntonic softmax modes
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[pyclass]
pub enum SyntonicSoftmaxMode {
    /// Learn mode norms per feature (recommended)
    Learned,
    /// Accept pre-computed syntony values
    Provided,
    /// Standard softmax (ablation baseline)
    Identity,
}

#[pymethods]
impl SyntonicSoftmaxMode {
    #[new]
    fn new(mode_str: &str) -> PyResult<Self> {
        match mode_str {
            "learned" => Ok(SyntonicSoftmaxMode::Learned),
            "provided" => Ok(SyntonicSoftmaxMode::Provided),
            "identity" => Ok(SyntonicSoftmaxMode::Identity),
            _ => Err(pyo3::exceptions::PyValueError::new_err(
                format!("Unknown mode: {}. Use 'learned', 'provided', or 'identity'", mode_str)
            )),
        }
    }

    fn __repr__(&self) -> String {
        match self {
            SyntonicSoftmaxMode::Learned => "SyntonicSoftmaxMode::Learned".to_string(),
            SyntonicSoftmaxMode::Provided => "SyntonicSoftmaxMode::Provided".to_string(),
            SyntonicSoftmaxMode::Identity => "SyntonicSoftmaxMode::Identity".to_string(),
        }
    }
}

/// Syntonic softmax state (learnable mode norms)
#[pyclass]
#[derive(Clone, Debug)]
pub struct SyntonicSoftmaxState {
    /// Mode norms |n|² for each feature [num_features]
    pub mode_norms: Option<ResonantTensor>,

    /// Syntony weighting scale
    pub syntony_scale: f64,

    /// Softmax dimension (typically -1 for last dim)
    pub dim: isize,

    /// Mode
    pub mode: SyntonicSoftmaxMode,

    /// Number of features (for learned mode)
    pub num_features: Option<usize>,
}

#[pymethods]
impl SyntonicSoftmaxState {
    #[new]
    pub fn new(
        mode: SyntonicSoftmaxMode,
        dim: Option<isize>,
        num_features: Option<usize>,
        syntony_scale: Option<f64>,
    ) -> PyResult<Self> {
        // Validate
        if mode == SyntonicSoftmaxMode::Learned && num_features.is_none() {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "num_features required for learned mode"
            ));
        }

        // Initialize mode norms if learned
        let mode_norms = if mode == SyntonicSoftmaxMode::Learned {
            let n = num_features.unwrap();
            // Initialize to 1.0 (will be learned)
            Some(ResonantTensor::from_floats_default_modes(&vec![1.0; n], vec![n], 100).map_err(|e| PyErr::from(e))?)
        } else {
            None
        };

        Ok(Self {
            mode_norms,
            syntony_scale: syntony_scale.unwrap_or(1.0),
            dim: dim.unwrap_or(-1),
            mode,
            num_features,
        })
    }

    /// Forward pass: apply syntony-weighted softmax
    pub fn forward(
        &self,
        x: &ResonantTensor,
        syntony: Option<&ResonantTensor>,
    ) -> PyResult<ResonantTensor> {
        match self.mode {
            SyntonicSoftmaxMode::Identity => {
                // Standard softmax (no syntony weighting)
                let mut output = x.clone();
                output.softmax_core(32).map_err(|e| PyErr::from(e))?;
                Ok(output)
            },
            SyntonicSoftmaxMode::Learned => {
                // w(n) = exp(-|n|²/φ)
                let mode_norms = self.mode_norms.as_ref().unwrap();
                let weights = self.compute_golden_weights(mode_norms)?;

                // weighted_logits = logits + scale * log(weights)
                let log_weights = weights.log_core(32)?;
                let scaled_log_weights = log_weights.scalar_mul_core(self.syntony_scale).map_err(|e| PyErr::from(e))?;
                
                // Dispatch to CUDA or CPU
                #[cfg(feature = "cuda")]
                {
                    if x.device_idx().is_some() && x.phase() == ResonantPhase::Flux {
                        return self.forward_cuda(x, Some(&scaled_log_weights));
                    }
                }

                let weighted_logits = x.elementwise_add_core(&scaled_log_weights).map_err(|e| PyErr::from(e))?;
                let mut output = weighted_logits.clone();
                output.softmax_core(32).map_err(|e| PyErr::from(e))?;
                Ok(output)
            },
            SyntonicSoftmaxMode::Provided => {
                // Use provided syntony values
                let syntony = syntony.ok_or_else(|| {
                    pyo3::exceptions::PyValueError::new_err(
                        "syntony values required for 'provided' mode"
                    )
                })?;

                // Validate shape match
                if syntony.shape() != x.shape() {
                    return Err(pyo3::exceptions::PyValueError::new_err(
                        format!("Shape mismatch: x {:?} vs syntony {:?}",
                            x.shape(), syntony.shape())
                    ));
                }

                // Dispatch to CUDA or CPU
                #[cfg(feature = "cuda")]
                {
                    if x.device_idx().is_some() && x.phase() == ResonantPhase::Flux {
                        return self.forward_cuda(x, Some(syntony));
                    }
                }

                // weighted_logits = logits + scale * log(syntony)
                let log_syntony = syntony.log_core(32)?;
                let scaled_log_syntony = log_syntony.scalar_mul_core(self.syntony_scale).map_err(|e| PyErr::from(e))?;
                let weighted_logits = x.elementwise_add_core(&scaled_log_syntony).map_err(|e| PyErr::from(e))?;

                let mut output = weighted_logits.clone();
                output.softmax_core(32).map_err(|e| PyErr::from(e))?;
                Ok(output)
            },
        }
    }

    /// CUDA-accelerated forward pass
    #[cfg(feature = "cuda")]
    fn forward_cuda(
        &self,
        x: &ResonantTensor,
        weights: Option<&ResonantTensor>,
    ) -> PyResult<ResonantTensor> {
        use crate::tensor::srt_kernels::{cuda_syntonic_softmax_learned_f64, cuda_syntonic_softmax_provided_f64};

        let device_idx = x.device_idx().ok_or_else(|| {
            pyo3::exceptions::PyRuntimeError::new_err("Tensor must be on GPU for CUDA softmax")
        })?;
        
        let device = crate::tensor::cuda::device_manager::get_device(device_idx)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;

        let shape = x.shape();
        let batch_size = shape[0];
        let num_classes = if shape.len() > 1 { shape[1..].iter().product() } else { 1 };

        // Allocate output
        let mut out_flux = device.default_stream().alloc_zeros::<f64>(x.len())
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;

        let x_flux = x.flux_ref().ok_or_else(|| {
            pyo3::exceptions::PyRuntimeError::new_err("Flux missing for CUDA operation")
        })?;

        match self.mode {
            SyntonicSoftmaxMode::Learned => {
                let mode_norms = self.mode_norms.as_ref().unwrap();
                let norms_flux = mode_norms.flux_ref().ok_or_else(|| {
                    pyo3::exceptions::PyRuntimeError::new_err("mode_norms must be on GPU for CUDA softmax")
                })?;

                cuda_syntonic_softmax_learned_f64(
                    &device,
                    &mut out_flux,
                    x_flux,
                    norms_flux,
                    self.syntony_scale,
                    batch_size as i32,
                    num_classes as i32,
                ).map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e))?;
            },
            SyntonicSoftmaxMode::Provided => {
                let w = weights.unwrap();
                let w_flux = w.flux_ref().ok_or_else(|| {
                    pyo3::exceptions::PyRuntimeError::new_err("syntony weights must be on GPU for CUDA softmax")
                })?;

                cuda_syntonic_softmax_provided_f64(
                    &device,
                    &mut out_flux,
                    x_flux,
                    w_flux,
                    self.syntony_scale,
                    batch_size as i32,
                    num_classes as i32,
                ).map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e))?;
            },
            SyntonicSoftmaxMode::Identity => {
                let mut out = x.clone();
                out.softmax_core(32).map_err(|e| PyErr::from(e))?;
                return Ok(out);
            }
        }

        let mut output = x.clone();
        output.set_flux(out_flux);
        Ok(output)
    }

    /// Get current mode norms (for learned mode)
    fn get_mode_norms(&self) -> PyResult<Option<ResonantTensor>> {
        Ok(self.mode_norms.clone())
    }
}

impl SyntonicSoftmaxState {
    /// Compute golden measure weights: w(n) = exp(-|n|²/φ)
    fn compute_golden_weights(&self, mode_norms: &ResonantTensor) -> PyResult<ResonantTensor> {
        // w = exp(-mode_norms / φ)
        let neg_inv_phi = -1.0 / PHI;
        let scaled_norms = mode_norms.scalar_mul_core(neg_inv_phi).map_err(|e| PyErr::from(e))?;
        Ok(scaled_norms.exp_core(32).map_err(|e| PyErr::from(e))?)
    }
}

// Python-accessible function
#[pyfunction]
#[pyo3(name = "syntonic_softmax")]
pub fn syntonic_softmax_py(
    x: &ResonantTensor,
    dim: Option<isize>,
    mode_norms: Option<&ResonantTensor>,
    syntony_scale: Option<f64>,
) -> PyResult<ResonantTensor> {
    let state = if let Some(norms) = mode_norms {
        SyntonicSoftmaxState::new(
            SyntonicSoftmaxMode::Learned,
            dim,
            Some(norms.shape()[0]),
            syntony_scale,
        )?
    } else {
        SyntonicSoftmaxState::new(
            SyntonicSoftmaxMode::Identity,
            dim,
            None,
            syntony_scale,
        )?
    };

    state.forward(x, None)
}
