//! Phi-scaled residual operations for theory-aligned neural networks.
//!
//! Implements three residual modes:
//! - phi: output = identity + residual/φ (default, recommended)
//! - phi_symmetric: output = (identity + residual)/φ
//! - standard: output = identity + residual (for ablation)

use pyo3::prelude::*;
use crate::resonant::{ResonantTensor, ResonantPhase, ResonantError};
use super::PHI_INV;


/// Phi-residual modes
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[pyclass]
pub enum PhiResidualMode {
    /// output = identity + residual/φ (theory-aligned, default)
    Phi,
    /// output = (identity + residual)/φ (symmetric scaling)
    PhiSymmetric,
    /// output = identity + residual (standard ResNet)
    Standard,
}

#[pymethods]
impl PhiResidualMode {
    #[new]
    fn new(mode_str: &str) -> PyResult<Self> {
        match mode_str {
            "phi" => Ok(PhiResidualMode::Phi),
            "phi_symmetric" => Ok(PhiResidualMode::PhiSymmetric),
            "standard" => Ok(PhiResidualMode::Standard),
            _ => Err(pyo3::exceptions::PyValueError::new_err(
                format!("Unknown mode: {}. Use 'phi', 'phi_symmetric', or 'standard'", mode_str)
            )),
        }
    }

    fn __repr__(&self) -> String {
        match self {
            PhiResidualMode::Phi => "PhiResidualMode('phi')".to_string(),
            PhiResidualMode::PhiSymmetric => "PhiResidualMode('phi_symmetric')".to_string(),
            PhiResidualMode::Standard => "PhiResidualMode('standard')".to_string(),
        }
    }
}

impl ResonantTensor {
    /// Apply phi-residual connection: combines identity with residual using golden ratio
    pub fn phi_residual(
        identity: &ResonantTensor,
        residual: &ResonantTensor,
        mode: PhiResidualMode,
    ) -> Result<ResonantTensor, ResonantError> {
        // Validate shapes match
        if identity.shape() != residual.shape() {
            return Err(ResonantError::ShapeMismatch(
                format!("Identity shape {:?} does not match residual shape {:?}",
                    identity.shape(), residual.shape())
            ));
        }

        // Convert to floats for computation
        let identity_floats = identity.to_floats_core();
        let residual_floats = residual.to_floats_core();

        let output_floats: Vec<f64> = match mode {
            PhiResidualMode::Phi => {
                identity_floats.iter()
                    .zip(residual_floats.iter())
                    .map(|(i, r)| i + r * PHI_INV)
                    .collect()
            },
            PhiResidualMode::PhiSymmetric => {
                identity_floats.iter()
                    .zip(residual_floats.iter())
                    .map(|(i, r)| (i + r) * PHI_INV)
                    .collect()
            },
            PhiResidualMode::Standard => {
                identity_floats.iter()
                    .zip(residual_floats.iter())
                    .map(|(i, r)| i + r)
                    .collect()
            },
        };

        // Create output tensor
        ResonantTensor::from_floats(
            &output_floats,
            identity.shape().to_vec(),
            identity.mode_norm_sq().to_vec(),
            identity.precision(),
        )
    }

    /// Fused phi-residual + ReLU activation
    pub fn phi_residual_relu(
        identity: &ResonantTensor,
        residual: &ResonantTensor,
        mode: PhiResidualMode,
    ) -> Result<ResonantTensor, ResonantError> {
        let combined = Self::phi_residual(identity, residual, mode)?;

        let combined_floats = combined.to_floats_core();
        let relu_floats: Vec<f64> = combined_floats.iter()
            .map(|&x| if x > 0.0 { x } else { 0.0 })
            .collect();

        ResonantTensor::from_floats(
            &relu_floats,
            combined.shape().to_vec(),
            combined.mode_norm_sq().to_vec(),
            combined.precision(),
        )
    }
}

// Python-accessible wrapper functions
#[pyfunction]
pub fn phi_residual(
    identity: &ResonantTensor,
    residual: &ResonantTensor,
    mode: PhiResidualMode,
) -> PyResult<ResonantTensor> {
    ResonantTensor::phi_residual(identity, residual, mode)
        .map_err(|e| PyErr::from(e))
}

#[pyfunction]
pub fn phi_residual_relu(
    identity: &ResonantTensor,
    residual: &ResonantTensor,
    mode: PhiResidualMode,
) -> PyResult<ResonantTensor> {
    ResonantTensor::phi_residual_relu(identity, residual, mode)
        .map_err(|e| PyErr::from(e))
}

// =============================================================================
// CUDA Implementation (when feature enabled)
// =============================================================================

#[cfg(feature = "cuda")]
impl ResonantTensor {
    /// GPU-accelerated phi-residual operation
    pub fn phi_residual_cuda(
        identity: &ResonantTensor,
        residual: &ResonantTensor,
        mode: PhiResidualMode,
    ) -> Result<ResonantTensor, ResonantError> {
        use crate::tensor::srt_kernels::cuda_phi_residual_f64;

        let device_idx = identity.device_idx().or(residual.device_idx()).unwrap_or(0);
        let device = crate::tensor::cuda::device_manager::get_device(device_idx)
            .map_err(|e| ResonantError::CudaError(e.to_string()))?;

        // Ensure both tensors are in flux phase
        let mut identity = identity.clone();
        if identity.phase() != ResonantPhase::Flux {
            identity.wake_flux(device.clone())?;
        }

        let mut residual = residual.clone();
        if residual.phase() != ResonantPhase::Flux {
            residual.wake_flux(device.clone())?;
        }

        // Allocate output
        let n = identity.len();
        let mut out_flux = device.default_stream().alloc_zeros::<f64>(n)
            .map_err(|e| ResonantError::CudaError(e.to_string()))?;

        // Launch kernel
        let identity_flux = identity.flux_ref().ok_or(ResonantError::NoFluxPresent)?;
        let residual_flux = residual.flux_ref().ok_or(ResonantError::NoFluxPresent)?;

        cuda_phi_residual_f64(
            &device,
            &mut out_flux,
            identity_flux,
            residual_flux,
            mode,
        ).map_err(|e| ResonantError::CudaError(e))?;

        // Create output tensor (flux phase)
        let mut output = identity.clone();
        output.set_flux(out_flux);
        output.set_device_idx(device_idx);

        Ok(output)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_phi_residual_mode_phi() {
        let identity = ResonantTensor::from_floats(&vec![1.0; 4], vec![4], vec![0.0; 4], 100).unwrap();
        let residual = ResonantTensor::from_floats(
            &vec![1.0, 2.0, 3.0, 4.0],
            vec![4],
            vec![0.0; 4],
            100,
        ).unwrap();

        let result = ResonantTensor::phi_residual(
            &identity,
            &residual,
            PhiResidualMode::Phi,
        ).unwrap();

        let expected = vec![
            1.0 + 1.0 * PHI_INV,
            1.0 + 2.0 * PHI_INV,
            1.0 + 3.0 * PHI_INV,
            1.0 + 4.0 * PHI_INV,
        ];

        let result_floats = result.to_floats_core();
        for (a, b) in result_floats.iter().zip(expected.iter()) {
            assert!((a - b).abs() < 1e-10, "Expected {}, got {}", b, a);
        }
    }

    #[test]
    fn test_phi_residual_mode_symmetric() {
        let identity = ResonantTensor::from_floats(&vec![1.0; 4], vec![4], vec![0.0; 4], 100).unwrap();
        let residual = ResonantTensor::from_floats(
            &vec![2.0, 4.0, 6.0, 8.0],
            vec![4],
            vec![0.0; 4],
            100,
        ).unwrap();

        let result = ResonantTensor::phi_residual(
            &identity,
            &residual,
            PhiResidualMode::PhiSymmetric,
        ).unwrap();

        let expected = vec![
            (1.0 + 2.0) * PHI_INV,
            (1.0 + 4.0) * PHI_INV,
            (1.0 + 6.0) * PHI_INV,
            (1.0 + 8.0) * PHI_INV,
        ];

        let result_floats = result.to_floats_core();
        for (a, b) in result_floats.iter().zip(expected.iter()) {
            assert!((a - b).abs() < 1e-10, "Expected {}, got {}", b, a);
        }
    }

    #[test]
    fn test_phi_residual_mode_standard() {
        let identity = ResonantTensor::from_floats(&vec![1.0; 4], vec![4], vec![0.0; 4], 100).unwrap();
        let residual = ResonantTensor::from_floats(
            &vec![1.0, 2.0, 3.0, 4.0],
            vec![4],
            vec![0.0; 4],
            100,
        ).unwrap();

        let result = ResonantTensor::phi_residual(
            &identity,
            &residual,
            PhiResidualMode::Standard,
        ).unwrap();

        let expected = vec![2.0, 3.0, 4.0, 5.0];

        let result_floats = result.to_floats_core();
        for (a, b) in result_floats.iter().zip(expected.iter()) {
            assert!((a - b).abs() < 1e-10, "Expected {}, got {}", b, a);
        }
    }

    #[test]
    fn test_phi_residual_shape_mismatch() {
        let identity = ResonantTensor::from_floats(&vec![1.0; 4], vec![4], vec![0.0; 4], 100).unwrap();
        let residual = ResonantTensor::from_floats(&vec![1.0; 8], vec![8], vec![0.0; 8], 100).unwrap();

        let result = ResonantTensor::phi_residual(
            &identity,
            &residual,
            PhiResidualMode::Phi,
        );

        assert!(result.is_err());
    }
}
