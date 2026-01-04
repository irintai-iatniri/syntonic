//! Tensor storage with CPU and CUDA backends.
//!
//! This module provides NumPy-free tensor operations using the Python buffer protocol
//! for data transfer and optional CUDA acceleration via cudarc.

use pyo3::prelude::*;
use pyo3::types::PyList;
use ndarray::{ArrayD, IxDyn, Ix1, Ix2};
use ndarray_linalg::{Eig, Eigh, SVD, QR, Inverse, Solve, Determinant, Trace, Cholesky, UPLO};
use num_complex::Complex64;
use rand::Rng;

// Import exact types - all numerical constants derive from these
use crate::exact::golden::GoldenExact;
use crate::exact::constants::FundamentalConstant;
use crate::exact::symexpr::SymExpr;

#[cfg(feature = "cuda")]
use cudarc::driver::{CudaDevice, CudaSlice, DeviceSlice, LaunchConfig, LaunchAsync};
#[cfg(feature = "cuda")]
use std::sync::Arc;

/// Pre-compiled PTX kernels for different compute capabilities
/// These are compiled offline to ensure driver compatibility
#[cfg(feature = "cuda")]
const PTX_SM75: &str = include_str!("../../kernels/ptx/elementwise_sm75.ptx");
#[cfg(feature = "cuda")]
const PTX_SM80: &str = include_str!("../../kernels/ptx/elementwise_sm80.ptx");
#[cfg(feature = "cuda")]
const PTX_SM86: &str = include_str!("../../kernels/ptx/elementwise_sm86.ptx");
#[cfg(feature = "cuda")]
const PTX_SM90: &str = include_str!("../../kernels/ptx/elementwise_sm90.ptx");


/// Get the compute capability for the current device
#[cfg(feature = "cuda")]
fn get_device_compute_capability(device: &Arc<CudaDevice>) -> (i32, i32) {
    use cudarc::driver::sys::CUdevice_attribute_enum;
    use cudarc::driver::result;

    let ordinal = device.ordinal() as i32;

    let major = unsafe {
        result::device::get_attribute(ordinal, CUdevice_attribute_enum::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR)
            .unwrap_or(7)
    };
    let minor = unsafe {
        result::device::get_attribute(ordinal, CUdevice_attribute_enum::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR)
            .unwrap_or(0)
    };

    (major, minor)
}

/// Select appropriate pre-compiled PTX based on device compute capability
#[cfg(feature = "cuda")]
fn select_ptx(major: i32, minor: i32) -> &'static str {
    let cc = major * 10 + minor;
    if cc >= 90 { PTX_SM90 }
    else if cc >= 86 { PTX_SM86 }
    else if cc >= 80 { PTX_SM80 }
    else { PTX_SM75 }  // Minimum supported (Turing and above)
}

/// Ensure CUDA kernels are loaded for the given device
#[cfg(feature = "cuda")]
fn ensure_kernels_loaded(device: &Arc<CudaDevice>, _device_idx: usize) -> PyResult<()> {
    // Check if kernels already loaded by testing for a function
    if device.get_func("syntonic", "add_f64").is_some() {
        return Ok(());
    }

    // Select pre-compiled PTX based on device compute capability
    let (major, minor) = get_device_compute_capability(device);
    let ptx_source = select_ptx(major, minor);

    // Load pre-compiled PTX directly
    device.load_ptx(
        cudarc::nvrtc::Ptx::from_src(ptx_source),
        "syntonic",
        &[
            "add_f64", "add_f32", "sub_f64", "sub_f32",
            "mul_f64", "mul_f32", "div_f64", "div_f32",
            "neg_f64", "neg_f32", "abs_f64", "abs_f32",
            "scalar_add_f64", "scalar_mul_f64",
            "add_c128", "sub_c128", "mul_c128", "neg_c128",
        ]
    ).map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(
        format!("Failed to load CUDA kernels: {}", e)
    ))?;

    Ok(())
}

/// Get optimal launch configuration for n elements
#[cfg(feature = "cuda")]
fn launch_cfg(n: usize) -> LaunchConfig {
    let block_size = 256u32;
    let grid_size = ((n as u32) + block_size - 1) / block_size;
    LaunchConfig {
        block_dim: (block_size, 1, 1),
        grid_dim: (grid_size, 1, 1),
        shared_mem_bytes: 0,
    }
}

/// Device type for tensor storage
#[derive(Clone, Debug, PartialEq)]
pub enum DeviceType {
    Cpu,
    #[cfg(feature = "cuda")]
    Cuda(usize),  // Device index
}

impl DeviceType {
    pub fn from_str(s: &str) -> PyResult<Self> {
        if s == "cpu" {
            Ok(DeviceType::Cpu)
        } else if s.starts_with("cuda") {
            #[cfg(feature = "cuda")]
            {
                let idx = if s.contains(':') {
                    s.split(':').nth(1).unwrap_or("0").parse().unwrap_or(0)
                } else {
                    0
                };
                Ok(DeviceType::Cuda(idx))
            }
            #[cfg(not(feature = "cuda"))]
            {
                Err(PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(
                    "CUDA not available - compile with cuda feature"
                ))
            }
        } else {
            Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                format!("Unknown device: {}", s)
            ))
        }
    }

    pub fn to_string(&self) -> String {
        match self {
            DeviceType::Cpu => "cpu".to_string(),
            #[cfg(feature = "cuda")]
            DeviceType::Cuda(idx) => format!("cuda:{}", idx),
        }
    }
}

/// CPU tensor data storage
#[derive(Clone, Debug)]
pub enum CpuData {
    Float32(ArrayD<f32>),
    Float64(ArrayD<f64>),
    Complex128(ArrayD<Complex64>),
    Int64(ArrayD<i64>),
}

/// CUDA tensor data storage
#[cfg(feature = "cuda")]
pub enum CudaData {
    Float32(CudaSlice<f32>),
    Float64(CudaSlice<f64>),
    /// Complex128 stored as interleaved f64 pairs [re0, im0, re1, im1, ...]
    Complex128(CudaSlice<f64>),
}

/// Unified tensor data enum
#[derive(Clone)]
pub enum TensorData {
    Cpu(CpuData),
    #[cfg(feature = "cuda")]
    Cuda {
        data: Arc<CudaData>,
        device: Arc<CudaDevice>,
        shape: Vec<usize>,
        dtype: String,
    },
}

/// Core tensor storage
#[pyclass]
pub struct TensorStorage {
    pub(crate) data: TensorData,
    pub(crate) shape: Vec<usize>,
    pub(crate) device: DeviceType,
}

#[pymethods]
impl TensorStorage {
    /// Create tensor from a flat Python list with shape and dtype
    #[staticmethod]
    pub fn from_list(data: &Bound<'_, PyList>, shape: Vec<usize>, dtype: &str, device: &str) -> PyResult<Self> {
        let device_type = DeviceType::from_str(device)?;
        let total_size: usize = shape.iter().product();

        let cpu_data = match dtype {
            "float32" | "f32" => {
                let values: Vec<f32> = data.iter()
                    .map(|x| x.extract::<f32>())
                    .collect::<PyResult<Vec<_>>>()?;
                if values.len() != total_size {
                    return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                        format!("Data length {} doesn't match shape {:?}", values.len(), shape)
                    ));
                }
                CpuData::Float32(ArrayD::from_shape_vec(IxDyn(&shape), values)
                    .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?)
            },
            "float64" | "f64" => {
                let values: Vec<f64> = data.iter()
                    .map(|x| x.extract::<f64>())
                    .collect::<PyResult<Vec<_>>>()?;
                if values.len() != total_size {
                    return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                        format!("Data length {} doesn't match shape {:?}", values.len(), shape)
                    ));
                }
                CpuData::Float64(ArrayD::from_shape_vec(IxDyn(&shape), values)
                    .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?)
            },
            "complex128" | "c128" => {
                // Accept complex numbers directly from Python
                let values: Vec<Complex64> = data.iter()
                    .map(|x| {
                        // Try extracting as Python complex first
                        if let Ok(c) = x.extract::<num_complex::Complex<f64>>() {
                            Ok(Complex64::new(c.re, c.im))
                        } else if let Ok(f) = x.extract::<f64>() {
                            // Fall back to real number (imaginary = 0)
                            Ok(Complex64::new(f, 0.0))
                        } else {
                            Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>(
                                "Expected complex or float number"
                            ))
                        }
                    })
                    .collect::<PyResult<Vec<_>>>()?;
                if values.len() != total_size {
                    return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                        format!("Complex data length {} doesn't match shape {:?}",
                            values.len(), shape)
                    ));
                }
                CpuData::Complex128(ArrayD::from_shape_vec(IxDyn(&shape), values)
                    .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?)
            },
            "int64" | "i64" => {
                let values: Vec<i64> = data.iter()
                    .map(|x| x.extract::<i64>())
                    .collect::<PyResult<Vec<_>>>()?;
                if values.len() != total_size {
                    return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                        format!("Data length {} doesn't match shape {:?}", values.len(), shape)
                    ));
                }
                CpuData::Int64(ArrayD::from_shape_vec(IxDyn(&shape), values)
                    .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?)
            },
            _ => return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                format!("Unsupported dtype: {}", dtype)
            )),
        };

        // Transfer to CUDA if requested
        #[cfg(feature = "cuda")]
        if let DeviceType::Cuda(idx) = &device_type {
            return Self::cpu_to_cuda(cpu_data, shape, *idx);
        }

        Ok(TensorStorage {
            data: TensorData::Cpu(cpu_data),
            shape,
            device: device_type,
        })
    }

    #[staticmethod]
    pub fn zeros(shape: Vec<usize>, dtype: &str, device: &str) -> PyResult<Self> {
        let device_type = DeviceType::from_str(device)?;
        let dim = IxDyn(&shape);

        let cpu_data = match dtype {
            "float32" | "f32" => CpuData::Float32(ArrayD::zeros(dim)),
            "float64" | "f64" => CpuData::Float64(ArrayD::zeros(dim)),
            "complex128" | "c128" | "complex" => CpuData::Complex128(ArrayD::zeros(dim)),
            "int64" | "i64" | "int" => CpuData::Int64(ArrayD::zeros(dim)),
            _ => return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                format!("Unsupported dtype: {}", dtype)
            )),
        };

        #[cfg(feature = "cuda")]
        if let DeviceType::Cuda(idx) = &device_type {
            return Self::cpu_to_cuda(cpu_data, shape, *idx);
        }

        Ok(TensorStorage {
            data: TensorData::Cpu(cpu_data),
            shape,
            device: device_type,
        })
    }

    #[getter]
    pub fn shape(&self) -> Vec<usize> { self.shape.clone() }

    #[getter]
    pub fn size(&self) -> usize { self.shape.iter().product() }

    #[getter]
    pub fn dtype(&self) -> String {
        match &self.data {
            TensorData::Cpu(cpu) => match cpu {
                CpuData::Float32(_) => "float32".to_string(),
                CpuData::Float64(_) => "float64".to_string(),
                CpuData::Complex128(_) => "complex128".to_string(),
                CpuData::Int64(_) => "int64".to_string(),
            },
            #[cfg(feature = "cuda")]
            TensorData::Cuda { dtype, .. } => dtype.clone(),
        }
    }

    #[getter]
    pub fn device_name(&self) -> String {
        self.device.to_string()
    }

    /// Convert to flat Python list
    pub fn to_list(&self, py: Python<'_>) -> PyResult<Py<PyList>> {
        let cpu_data = self.ensure_cpu()?;

        match cpu_data {
            CpuData::Float32(arr) => {
                let list = PyList::new_bound(py, arr.iter().map(|x| *x));
                Ok(list.into())
            },
            CpuData::Float64(arr) => {
                let list = PyList::new_bound(py, arr.iter().map(|x| *x));
                Ok(list.into())
            },
            CpuData::Complex128(arr) => {
                // Return complex numbers directly to Python
                let list = PyList::new_bound(py, arr.iter().map(|c| {
                    pyo3::types::PyComplex::from_doubles_bound(py, c.re, c.im)
                }));
                Ok(list.into())
            },
            CpuData::Int64(arr) => {
                let list = PyList::new_bound(py, arr.iter().map(|x| *x));
                Ok(list.into())
            },
        }
    }

    /// Move tensor to specified device
    pub fn to_device(&self, device: &str) -> PyResult<TensorStorage> {
        let target_device = DeviceType::from_str(device)?;

        if self.device == target_device {
            return Ok(self.clone_storage());
        }

        #[cfg(feature = "cuda")]
        {
            match (&self.device, &target_device) {
                (DeviceType::Cpu, DeviceType::Cpu) => Ok(self.clone_storage()),

                (DeviceType::Cpu, DeviceType::Cuda(idx)) => {
                    let cpu_data = self.ensure_cpu()?;
                    Self::cpu_to_cuda(cpu_data, self.shape.clone(), *idx)
                },

                (DeviceType::Cuda(_), DeviceType::Cpu) => {
                    let cpu_data = self.ensure_cpu()?;
                    Ok(TensorStorage {
                        data: TensorData::Cpu(cpu_data),
                        shape: self.shape.clone(),
                        device: DeviceType::Cpu,
                    })
                },

                (DeviceType::Cuda(from), DeviceType::Cuda(to)) if from != to => {
                    // Cross-device transfer: CUDA -> CPU -> CUDA
                    let cpu_data = self.ensure_cpu()?;
                    Self::cpu_to_cuda(cpu_data, self.shape.clone(), *to)
                },

                (DeviceType::Cuda(_), DeviceType::Cuda(_)) => Ok(self.clone_storage()),
            }
        }

        #[cfg(not(feature = "cuda"))]
        {
            // Without CUDA feature, only CPU devices exist
            Ok(self.clone_storage())
        }
    }

    // ===== Arithmetic Operations =====

    pub fn add(&self, other: &TensorStorage) -> PyResult<TensorStorage> {
        // Try GPU-native operation if both tensors are on the same CUDA device index
        #[cfg(feature = "cuda")]
        if let (TensorData::Cuda { data: a, device: dev_a, .. },
                TensorData::Cuda { data: b, device: _, .. }) = (&self.data, &other.data) {
            // Compare device indices, not Arc pointers
            if let (DeviceType::Cuda(idx_a), DeviceType::Cuda(idx_b)) = (&self.device, &other.device) {
                if idx_a == idx_b {
                    ensure_kernels_loaded(dev_a, *idx_a)?;
                    return self.binary_cuda_op(a, b, dev_a, "add");
                }
            }
        }

        // CPU fallback
        let a = self.ensure_cpu()?;
        let b = other.ensure_cpu()?;

        let result = match (a, b) {
            (CpuData::Float64(a), CpuData::Float64(b)) => CpuData::Float64(&a + &b),
            (CpuData::Float32(a), CpuData::Float32(b)) => CpuData::Float32(&a + &b),
            (CpuData::Complex128(a), CpuData::Complex128(b)) => CpuData::Complex128(&a + &b),
            (CpuData::Int64(a), CpuData::Int64(b)) => CpuData::Int64(&a + &b),
            _ => return Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>("Dtype mismatch")),
        };

        Ok(Self::wrap_cpu(result, &self.device))
    }

    pub fn add_scalar(&self, scalar: f64) -> PyResult<TensorStorage> {
        let cpu = self.ensure_cpu()?;
        let result = match cpu {
            CpuData::Float64(arr) => CpuData::Float64(&arr + scalar),
            CpuData::Float32(arr) => CpuData::Float32(&arr + scalar as f32),
            CpuData::Complex128(arr) => CpuData::Complex128(&arr + Complex64::new(scalar, 0.0)),
            CpuData::Int64(arr) => CpuData::Int64(&arr + scalar as i64),
        };
        Ok(Self::wrap_cpu(result, &self.device))
    }

    pub fn sub(&self, other: &TensorStorage) -> PyResult<TensorStorage> {
        #[cfg(feature = "cuda")]
        if let (TensorData::Cuda { data: a, device: dev_a, .. },
                TensorData::Cuda { data: b, device: _, .. }) = (&self.data, &other.data) {
            if let (DeviceType::Cuda(idx_a), DeviceType::Cuda(idx_b)) = (&self.device, &other.device) {
                if idx_a == idx_b {
                    ensure_kernels_loaded(dev_a, *idx_a)?;
                    return self.binary_cuda_op(a, b, dev_a, "sub");
                }
            }
        }

        let a = self.ensure_cpu()?;
        let b = other.ensure_cpu()?;

        let result = match (a, b) {
            (CpuData::Float64(a), CpuData::Float64(b)) => CpuData::Float64(&a - &b),
            (CpuData::Float32(a), CpuData::Float32(b)) => CpuData::Float32(&a - &b),
            (CpuData::Complex128(a), CpuData::Complex128(b)) => CpuData::Complex128(&a - &b),
            (CpuData::Int64(a), CpuData::Int64(b)) => CpuData::Int64(&a - &b),
            _ => return Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>("Dtype mismatch")),
        };

        Ok(Self::wrap_cpu(result, &self.device))
    }

    pub fn sub_scalar(&self, scalar: f64) -> PyResult<TensorStorage> {
        let cpu = self.ensure_cpu()?;
        let result = match cpu {
            CpuData::Float64(arr) => CpuData::Float64(&arr - scalar),
            CpuData::Float32(arr) => CpuData::Float32(&arr - scalar as f32),
            CpuData::Complex128(arr) => CpuData::Complex128(&arr - Complex64::new(scalar, 0.0)),
            CpuData::Int64(arr) => CpuData::Int64(&arr - scalar as i64),
        };
        Ok(Self::wrap_cpu(result, &self.device))
    }

    pub fn mul(&self, other: &TensorStorage) -> PyResult<TensorStorage> {
        #[cfg(feature = "cuda")]
        if let (TensorData::Cuda { data: a, device: dev_a, .. },
                TensorData::Cuda { data: b, device: _, .. }) = (&self.data, &other.data) {
            if let (DeviceType::Cuda(idx_a), DeviceType::Cuda(idx_b)) = (&self.device, &other.device) {
                if idx_a == idx_b {
                    ensure_kernels_loaded(dev_a, *idx_a)?;
                    return self.binary_cuda_op(a, b, dev_a, "mul");
                }
            }
        }

        let a = self.ensure_cpu()?;
        let b = other.ensure_cpu()?;

        let result = match (a, b) {
            (CpuData::Float64(a), CpuData::Float64(b)) => CpuData::Float64(&a * &b),
            (CpuData::Float32(a), CpuData::Float32(b)) => CpuData::Float32(&a * &b),
            (CpuData::Complex128(a), CpuData::Complex128(b)) => CpuData::Complex128(&a * &b),
            (CpuData::Int64(a), CpuData::Int64(b)) => CpuData::Int64(&a * &b),
            _ => return Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>("Dtype mismatch")),
        };

        Ok(Self::wrap_cpu(result, &self.device))
    }

    pub fn mul_scalar(&self, scalar: f64) -> PyResult<TensorStorage> {
        let cpu = self.ensure_cpu()?;
        let result = match cpu {
            CpuData::Float64(arr) => CpuData::Float64(&arr * scalar),
            CpuData::Float32(arr) => CpuData::Float32(&arr * scalar as f32),
            CpuData::Complex128(arr) => CpuData::Complex128(&arr * Complex64::new(scalar, 0.0)),
            CpuData::Int64(arr) => CpuData::Int64(&arr * scalar as i64),
        };
        Ok(Self::wrap_cpu(result, &self.device))
    }

    pub fn div(&self, other: &TensorStorage) -> PyResult<TensorStorage> {
        #[cfg(feature = "cuda")]
        if let (TensorData::Cuda { data: a, device: dev_a, .. },
                TensorData::Cuda { data: b, device: _, .. }) = (&self.data, &other.data) {
            if let (DeviceType::Cuda(idx_a), DeviceType::Cuda(idx_b)) = (&self.device, &other.device) {
                if idx_a == idx_b {
                    ensure_kernels_loaded(dev_a, *idx_a)?;
                    return self.binary_cuda_op(a, b, dev_a, "div");
                }
            }
        }

        let a = self.ensure_cpu()?;
        let b = other.ensure_cpu()?;

        let result = match (a, b) {
            (CpuData::Float64(a), CpuData::Float64(b)) => CpuData::Float64(&a / &b),
            (CpuData::Float32(a), CpuData::Float32(b)) => CpuData::Float32(&a / &b),
            (CpuData::Complex128(a), CpuData::Complex128(b)) => CpuData::Complex128(&a / &b),
            (CpuData::Int64(a), CpuData::Int64(b)) => CpuData::Int64(&a / &b),
            _ => return Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>("Dtype mismatch")),
        };

        Ok(Self::wrap_cpu(result, &self.device))
    }

    pub fn div_scalar(&self, scalar: f64) -> PyResult<TensorStorage> {
        let cpu = self.ensure_cpu()?;
        let result = match cpu {
            CpuData::Float64(arr) => CpuData::Float64(&arr / scalar),
            CpuData::Float32(arr) => CpuData::Float32(&arr / scalar as f32),
            CpuData::Complex128(arr) => CpuData::Complex128(&arr / Complex64::new(scalar, 0.0)),
            CpuData::Int64(arr) => CpuData::Int64(&arr / scalar as i64),
        };
        Ok(Self::wrap_cpu(result, &self.device))
    }

    pub fn neg(&self) -> PyResult<TensorStorage> {
        #[cfg(feature = "cuda")]
        if let TensorData::Cuda { data, device, .. } = &self.data {
            let device_idx = match &self.device { DeviceType::Cuda(idx) => *idx, _ => 0 };
            ensure_kernels_loaded(device, device_idx)?;
            return self.unary_cuda_op(data, device, "neg");
        }

        let cpu = self.ensure_cpu()?;
        let result = match cpu {
            CpuData::Float64(arr) => CpuData::Float64(-&arr),
            CpuData::Float32(arr) => CpuData::Float32(-&arr),
            CpuData::Complex128(arr) => CpuData::Complex128(-&arr),
            CpuData::Int64(arr) => CpuData::Int64(-&arr),
        };
        Ok(Self::wrap_cpu(result, &self.device))
    }

    pub fn abs(&self) -> PyResult<TensorStorage> {
        #[cfg(feature = "cuda")]
        if let TensorData::Cuda { data, device, .. } = &self.data {
            let device_idx = match &self.device { DeviceType::Cuda(idx) => *idx, _ => 0 };
            ensure_kernels_loaded(device, device_idx)?;
            return self.unary_cuda_op(data, device, "abs");
        }

        let cpu = self.ensure_cpu()?;
        let result = match cpu {
            CpuData::Float64(arr) => CpuData::Float64(arr.mapv(|x| x.abs())),
            CpuData::Float32(arr) => CpuData::Float32(arr.mapv(|x| x.abs())),
            CpuData::Complex128(arr) => CpuData::Float64(arr.mapv(|x| x.norm())),
            CpuData::Int64(arr) => CpuData::Int64(arr.mapv(|x| x.abs())),
        };
        Ok(Self::wrap_cpu(result, &self.device))
    }

    pub fn norm(&self, _ord: Option<i32>) -> PyResult<f64> {
        let cpu = self.ensure_cpu()?;
        Ok(match cpu {
            CpuData::Float64(arr) => arr.iter().map(|x| x * x).sum::<f64>().sqrt(),
            CpuData::Float32(arr) => arr.iter().map(|x| (*x as f64).powi(2)).sum::<f64>().sqrt(),
            CpuData::Complex128(arr) => arr.iter().map(|x| x.norm_sqr()).sum::<f64>().sqrt(),
            CpuData::Int64(arr) => arr.iter().map(|x| (*x as f64).powi(2)).sum::<f64>().sqrt(),
        })
    }

    pub fn conj(&self) -> PyResult<TensorStorage> {
        let cpu = self.ensure_cpu()?;
        let result = match cpu {
            CpuData::Complex128(arr) => CpuData::Complex128(arr.mapv(|x| x.conj())),
            other => other.clone(),
        };
        Ok(Self::wrap_cpu(result, &self.device))
    }

    pub fn matmul(&self, other: &TensorStorage) -> PyResult<TensorStorage> {
        let a = self.ensure_cpu()?;
        let b = other.ensure_cpu()?;

        match (a, b) {
            (CpuData::Float64(a), CpuData::Float64(b)) => {
                let a_2d = a.clone().into_dimensionality::<Ix2>()
                    .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("ShapeError/IncompatibleShape: {}", e)))?;
                let b_2d = b.clone().into_dimensionality::<Ix2>()
                    .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("ShapeError/IncompatibleShape: {}", e)))?;
                Ok(Self::wrap_cpu(CpuData::Float64(a_2d.dot(&b_2d).into_dyn()), &self.device))
            },
            (CpuData::Complex128(a), CpuData::Complex128(b)) => {
                let a_2d = a.clone().into_dimensionality::<Ix2>()
                    .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("ShapeError/IncompatibleShape: {}", e)))?;
                let b_2d = b.clone().into_dimensionality::<Ix2>()
                    .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("ShapeError/IncompatibleShape: {}", e)))?;
                Ok(Self::wrap_cpu(CpuData::Complex128(a_2d.dot(&b_2d).into_dyn()), &self.device))
            },
            _ => Err(PyErr::new::<pyo3::exceptions::PyValueError, _>("ShapeError/IncompatibleShape: incompatible shapes")),
        }
    }

    pub fn transpose(&self) -> PyResult<TensorStorage> {
        let cpu = self.ensure_cpu()?;
        let result = match cpu {
            CpuData::Float64(arr) => CpuData::Float64(arr.t().to_owned()),
            CpuData::Float32(arr) => CpuData::Float32(arr.t().to_owned()),
            CpuData::Complex128(arr) => CpuData::Complex128(arr.t().to_owned()),
            CpuData::Int64(arr) => CpuData::Int64(arr.t().to_owned()),
        };
        Ok(Self::wrap_cpu(result, &self.device))
    }

    // ===== Linear Algebra =====

    pub fn eig(&self) -> PyResult<(TensorStorage, TensorStorage)> {
        let cpu = self.ensure_cpu()?;
        match cpu {
            CpuData::Float64(arr) => {
                let arr_2d = arr.clone().into_dimensionality::<Ix2>()
                    .map_err(|_| PyErr::new::<pyo3::exceptions::PyValueError, _>("Matrix must be 2D"))?;
                let (e, v) = arr_2d.eig()
                    .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
                Ok((
                    Self::wrap_cpu(CpuData::Complex128(e.into_dyn()), &self.device),
                    Self::wrap_cpu(CpuData::Complex128(v.into_dyn()), &self.device)
                ))
            },
            CpuData::Complex128(arr) => {
                let arr_2d = arr.clone().into_dimensionality::<Ix2>()
                    .map_err(|_| PyErr::new::<pyo3::exceptions::PyValueError, _>("Matrix must be 2D"))?;
                let (e, v) = arr_2d.eig()
                    .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
                Ok((
                    Self::wrap_cpu(CpuData::Complex128(e.into_dyn()), &self.device),
                    Self::wrap_cpu(CpuData::Complex128(v.into_dyn()), &self.device)
                ))
            },
            _ => Err(PyErr::new::<pyo3::exceptions::PyNotImplementedError, _>("Eig only for f64/c128")),
        }
    }

    pub fn eigh(&self) -> PyResult<(TensorStorage, TensorStorage)> {
        let cpu = self.ensure_cpu()?;
        match cpu {
            CpuData::Float64(arr) => {
                let arr_2d = arr.clone().into_dimensionality::<Ix2>()
                    .map_err(|_| PyErr::new::<pyo3::exceptions::PyValueError, _>("Matrix must be 2D"))?;
                let (e, v) = arr_2d.eigh(UPLO::Lower)
                    .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
                Ok((
                    Self::wrap_cpu(CpuData::Float64(e.into_dyn()), &self.device),
                    Self::wrap_cpu(CpuData::Float64(v.into_dyn()), &self.device)
                ))
            },
            CpuData::Complex128(arr) => {
                let arr_2d = arr.clone().into_dimensionality::<Ix2>()
                    .map_err(|_| PyErr::new::<pyo3::exceptions::PyValueError, _>("Matrix must be 2D"))?;
                let (e, v) = arr_2d.eigh(UPLO::Lower)
                    .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
                Ok((
                    Self::wrap_cpu(CpuData::Float64(e.into_dyn()), &self.device),
                    Self::wrap_cpu(CpuData::Complex128(v.into_dyn()), &self.device)
                ))
            },
            _ => Err(PyErr::new::<pyo3::exceptions::PyNotImplementedError, _>("Eigh only for f64/c128")),
        }
    }

    pub fn svd(&self, full_matrices: bool) -> PyResult<(TensorStorage, TensorStorage, TensorStorage)> {
        let cpu = self.ensure_cpu()?;
        match cpu {
            CpuData::Float64(arr) => {
                let arr_2d = arr.clone().into_dimensionality::<Ix2>()
                    .map_err(|_| PyErr::new::<pyo3::exceptions::PyValueError, _>("Matrix must be 2D"))?;
                let (u, s, vt) = arr_2d.svd(true, full_matrices)
                    .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
                Ok((
                    Self::wrap_cpu(CpuData::Float64(u.unwrap().into_dyn()), &self.device),
                    Self::wrap_cpu(CpuData::Float64(s.into_dyn()), &self.device),
                    Self::wrap_cpu(CpuData::Float64(vt.unwrap().into_dyn()), &self.device)
                ))
            },
            CpuData::Complex128(arr) => {
                let arr_2d = arr.clone().into_dimensionality::<Ix2>()
                    .map_err(|_| PyErr::new::<pyo3::exceptions::PyValueError, _>("Matrix must be 2D"))?;
                let (u, s, vt) = arr_2d.svd(true, full_matrices)
                    .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
                Ok((
                    Self::wrap_cpu(CpuData::Complex128(u.unwrap().into_dyn()), &self.device),
                    Self::wrap_cpu(CpuData::Float64(s.into_dyn()), &self.device),
                    Self::wrap_cpu(CpuData::Complex128(vt.unwrap().into_dyn()), &self.device)
                ))
            },
            _ => Err(PyErr::new::<pyo3::exceptions::PyNotImplementedError, _>("SVD only for f64/c128")),
        }
    }

    pub fn qr(&self) -> PyResult<(TensorStorage, TensorStorage)> {
        let cpu = self.ensure_cpu()?;
        match cpu {
            CpuData::Float64(arr) => {
                let arr_2d = arr.clone().into_dimensionality::<Ix2>()
                    .map_err(|_| PyErr::new::<pyo3::exceptions::PyValueError, _>("Matrix must be 2D"))?;
                let (q, r) = arr_2d.qr()
                    .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
                Ok((
                    Self::wrap_cpu(CpuData::Float64(q.into_dyn()), &self.device),
                    Self::wrap_cpu(CpuData::Float64(r.into_dyn()), &self.device)
                ))
            },
            CpuData::Complex128(arr) => {
                let arr_2d = arr.clone().into_dimensionality::<Ix2>()
                    .map_err(|_| PyErr::new::<pyo3::exceptions::PyValueError, _>("Matrix must be 2D"))?;
                let (q, r) = arr_2d.qr()
                    .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
                Ok((
                    Self::wrap_cpu(CpuData::Complex128(q.into_dyn()), &self.device),
                    Self::wrap_cpu(CpuData::Complex128(r.into_dyn()), &self.device)
                ))
            },
            _ => Err(PyErr::new::<pyo3::exceptions::PyNotImplementedError, _>("QR only for f64/c128")),
        }
    }

    pub fn inv(&self) -> PyResult<TensorStorage> {
        let cpu = self.ensure_cpu()?;
        match cpu {
            CpuData::Float64(arr) => {
                let arr_2d = arr.clone().into_dimensionality::<Ix2>()
                    .map_err(|_| PyErr::new::<pyo3::exceptions::PyValueError, _>("Matrix must be 2D"))?;
                let res = arr_2d.inv()
                    .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
                Ok(Self::wrap_cpu(CpuData::Float64(res.into_dyn()), &self.device))
            },
            CpuData::Complex128(arr) => {
                let arr_2d = arr.clone().into_dimensionality::<Ix2>()
                    .map_err(|_| PyErr::new::<pyo3::exceptions::PyValueError, _>("Matrix must be 2D"))?;
                let res = arr_2d.inv()
                    .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
                Ok(Self::wrap_cpu(CpuData::Complex128(res.into_dyn()), &self.device))
            },
            _ => Err(PyErr::new::<pyo3::exceptions::PyNotImplementedError, _>("Inv only for f64/c128")),
        }
    }

    pub fn solve(&self, b: &TensorStorage) -> PyResult<TensorStorage> {
        let a_cpu = self.ensure_cpu()?;
        let b_cpu = b.ensure_cpu()?;

        match (a_cpu, b_cpu) {
            (CpuData::Float64(a), CpuData::Float64(rhs)) => {
                let a_2d = a.clone().into_dimensionality::<Ix2>()
                    .map_err(|_| PyErr::new::<pyo3::exceptions::PyValueError, _>("A must be 2D"))?;
                if let Ok(rhs_1d) = rhs.clone().into_dimensionality::<Ix1>() {
                    let res = a_2d.solve(&rhs_1d)
                        .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
                    Ok(Self::wrap_cpu(CpuData::Float64(res.into_dyn()), &self.device))
                } else {
                    let rhs_2d = rhs.clone().into_dimensionality::<Ix2>()
                        .map_err(|_| PyErr::new::<pyo3::exceptions::PyValueError, _>("B must be 1D or 2D"))?;
                    let a_inv = a_2d.inv()
                        .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
                    Ok(Self::wrap_cpu(CpuData::Float64(a_inv.dot(&rhs_2d).into_dyn()), &self.device))
                }
            },
            (CpuData::Complex128(a), CpuData::Complex128(rhs)) => {
                let a_2d = a.clone().into_dimensionality::<Ix2>()
                    .map_err(|_| PyErr::new::<pyo3::exceptions::PyValueError, _>("A must be 2D"))?;
                if let Ok(rhs_1d) = rhs.clone().into_dimensionality::<Ix1>() {
                    let res = a_2d.solve(&rhs_1d)
                        .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
                    Ok(Self::wrap_cpu(CpuData::Complex128(res.into_dyn()), &self.device))
                } else {
                    let rhs_2d = rhs.clone().into_dimensionality::<Ix2>()
                        .map_err(|_| PyErr::new::<pyo3::exceptions::PyValueError, _>("B must be 1D or 2D"))?;
                    let a_inv = a_2d.inv()
                        .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
                    Ok(Self::wrap_cpu(CpuData::Complex128(a_inv.dot(&rhs_2d).into_dyn()), &self.device))
                }
            },
            _ => Err(PyErr::new::<pyo3::exceptions::PyNotImplementedError, _>("Solve only for f64/c128")),
        }
    }

    pub fn det(&self) -> PyResult<Complex64> {
        let cpu = self.ensure_cpu()?;
        match cpu {
            CpuData::Float64(arr) => {
                let arr_2d = arr.clone().into_dimensionality::<Ix2>()
                    .map_err(|_| PyErr::new::<pyo3::exceptions::PyValueError, _>("Matrix must be 2D"))?;
                let res = arr_2d.det()
                    .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
                Ok(Complex64::new(res, 0.0))
            },
            CpuData::Complex128(arr) => {
                let arr_2d = arr.clone().into_dimensionality::<Ix2>()
                    .map_err(|_| PyErr::new::<pyo3::exceptions::PyValueError, _>("Matrix must be 2D"))?;
                arr_2d.det()
                    .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))
            },
            _ => Err(PyErr::new::<pyo3::exceptions::PyNotImplementedError, _>("Det only for f64/c128")),
        }
    }

    pub fn trace(&self) -> PyResult<Complex64> {
        let cpu = self.ensure_cpu()?;
        match cpu {
            CpuData::Float64(arr) => {
                let arr_2d = arr.clone().into_dimensionality::<Ix2>()
                    .map_err(|_| PyErr::new::<pyo3::exceptions::PyValueError, _>("Matrix must be 2D"))?;
                let res = arr_2d.trace()
                    .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
                Ok(Complex64::new(res, 0.0))
            },
            CpuData::Complex128(arr) => {
                let arr_2d = arr.clone().into_dimensionality::<Ix2>()
                    .map_err(|_| PyErr::new::<pyo3::exceptions::PyValueError, _>("Matrix must be 2D"))?;
                arr_2d.trace()
                    .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))
            },
            _ => Err(PyErr::new::<pyo3::exceptions::PyNotImplementedError, _>("Trace only for f64/c128")),
        }
    }

    pub fn cholesky(&self) -> PyResult<TensorStorage> {
        let cpu = self.ensure_cpu()?;
        match cpu {
            CpuData::Float64(arr) => {
                let arr_2d = arr.clone().into_dimensionality::<Ix2>()
                    .map_err(|_| PyErr::new::<pyo3::exceptions::PyValueError, _>("Matrix must be 2D"))?;
                let res = arr_2d.cholesky(UPLO::Lower)
                    .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
                Ok(Self::wrap_cpu(CpuData::Float64(res.into_dyn()), &self.device))
            },
            CpuData::Complex128(arr) => {
                let arr_2d = arr.clone().into_dimensionality::<Ix2>()
                    .map_err(|_| PyErr::new::<pyo3::exceptions::PyValueError, _>("Matrix must be 2D"))?;
                let res = arr_2d.cholesky(UPLO::Lower)
                    .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
                Ok(Self::wrap_cpu(CpuData::Complex128(res.into_dyn()), &self.device))
            },
            _ => Err(PyErr::new::<pyo3::exceptions::PyNotImplementedError, _>("Cholesky only for f64/c128")),
        }
    }

    // ===== SRT φ-Algebra Operations =====

    /// Golden commutator: [A, B]_φ = AB - φ⁻¹BA
    /// This is the fundamental bracket for SRT Lie algebra representations.
    pub fn phi_bracket(&self, other: &TensorStorage) -> PyResult<TensorStorage> {
        // Compute AB
        let ab = self.matmul(other)?;
        // Compute BA
        let ba = other.matmul(self)?;
        // Compute φ⁻¹BA
        let phi_inv_ba = ba.mul_scalar(Self::phi_inv())?;
        // Result: AB - φ⁻¹BA
        ab.sub(&phi_inv_ba)
    }

    /// Golden anticommutator: {A, B}_φ = AB + φ⁻¹BA
    /// Symmetric counterpart to the φ-bracket.
    pub fn phi_antibracket(&self, other: &TensorStorage) -> PyResult<TensorStorage> {
        // Compute AB
        let ab = self.matmul(other)?;
        // Compute BA
        let ba = other.matmul(self)?;
        // Compute φ⁻¹BA
        let phi_inv_ba = ba.mul_scalar(Self::phi_inv())?;
        // Result: AB + φ⁻¹BA
        ab.add(&phi_inv_ba)
    }

    /// Golden-scaled matmul: φ^k × (A @ B)
    /// Used for hierarchical scale operations in SRT.
    pub fn mm_phi(&self, other: &TensorStorage, k: i32) -> PyResult<TensorStorage> {
        let result = self.matmul(other)?;
        let scale = Self::phi_power(k);
        result.mul_scalar(scale)
    }

    /// Corrected matmul: (1 + sign × q/N) × (A @ B)
    /// Applies SRT correction factor based on algebraic structure dimension.
    ///
    /// The correction factor is derived symbolically as (1 ± q/N) then evaluated.
    pub fn mm_corrected(&self, other: &TensorStorage, n: u32, sign: i8) -> PyResult<TensorStorage> {
        let result = self.matmul(other)?;
        let correction = Self::correction_factor(n, sign);
        result.mul_scalar(correction)
    }

    /// Matmul with additive term: α(A @ B) + βC
    /// Used for iterative refinement and accumulation.
    pub fn mm_add(&self, other: &TensorStorage, c: &TensorStorage, alpha: f64, beta: f64) -> PyResult<TensorStorage> {
        let ab = self.matmul(other)?;
        let scaled_ab = ab.mul_scalar(alpha)?;
        let scaled_c = c.mul_scalar(beta)?;
        scaled_ab.add(&scaled_c)
    }

    /// Transposed matmul: Aᵀ @ B
    pub fn mm_tn(&self, other: &TensorStorage) -> PyResult<TensorStorage> {
        let a_t = self.transpose()?;
        a_t.matmul(other)
    }

    /// Matmul with transposed right: A @ Bᵀ
    pub fn mm_nt(&self, other: &TensorStorage) -> PyResult<TensorStorage> {
        let b_t = other.transpose()?;
        self.matmul(&b_t)
    }

    /// Double transposed matmul: Aᵀ @ Bᵀ
    pub fn mm_tt(&self, other: &TensorStorage) -> PyResult<TensorStorage> {
        let a_t = self.transpose()?;
        let b_t = other.transpose()?;
        a_t.matmul(&b_t)
    }

    /// Hermitian-None matmul: A† @ B (conjugate transpose of A times B)
    /// Critical for quantum/complex operations in SRT.
    pub fn mm_hn(&self, other: &TensorStorage) -> PyResult<TensorStorage> {
        let a_h = self.transpose()?.conj()?;
        a_h.matmul(other)
    }

    /// None-Hermitian matmul: A @ B† (A times conjugate transpose of B)
    pub fn mm_nh(&self, other: &TensorStorage) -> PyResult<TensorStorage> {
        let b_h = other.transpose()?.conj()?;
        self.matmul(&b_h)
    }

    /// Batched matrix multiplication: C[i] = A[i] @ B[i]
    /// For 3D tensors, applies matmul along the first (batch) dimension.
    pub fn bmm(&self, other: &TensorStorage) -> PyResult<TensorStorage> {
        // Ensure both are 3D
        if self.shape.len() != 3 || other.shape.len() != 3 {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                format!("bmm requires 3D tensors, got shapes {:?} and {:?}", self.shape, other.shape)
            ));
        }
        if self.shape[0] != other.shape[0] {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                format!("Batch sizes must match: {} vs {}", self.shape[0], other.shape[0])
            ));
        }

        let batch_size = self.shape[0];
        let m = self.shape[1];
        let k = self.shape[2];
        let n = other.shape[2];

        if k != other.shape[1] {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                format!("Inner dimensions must match: {} vs {}", k, other.shape[1])
            ));
        }

        let a_cpu = self.ensure_cpu()?;
        let b_cpu = other.ensure_cpu()?;

        match (a_cpu, b_cpu) {
            (CpuData::Float64(a_arr), CpuData::Float64(b_arr)) => {
                let mut result = ndarray::Array3::<f64>::zeros((batch_size, m, n));
                for i in 0..batch_size {
                    let a_slice = a_arr.slice(ndarray::s![i, .., ..]).to_owned();
                    let b_slice = b_arr.slice(ndarray::s![i, .., ..]).to_owned();
                    let a_2d = a_slice.into_dimensionality::<Ix2>().unwrap();
                    let b_2d = b_slice.into_dimensionality::<Ix2>().unwrap();
                    let c = a_2d.dot(&b_2d);
                    result.slice_mut(ndarray::s![i, .., ..]).assign(&c);
                }
                Ok(Self::wrap_cpu_data(CpuData::Float64(result.into_dyn())))
            },
            (CpuData::Float32(a_arr), CpuData::Float32(b_arr)) => {
                let mut result = ndarray::Array3::<f32>::zeros((batch_size, m, n));
                for i in 0..batch_size {
                    let a_slice = a_arr.slice(ndarray::s![i, .., ..]).to_owned();
                    let b_slice = b_arr.slice(ndarray::s![i, .., ..]).to_owned();
                    let a_2d = a_slice.into_dimensionality::<Ix2>().unwrap();
                    let b_2d = b_slice.into_dimensionality::<Ix2>().unwrap();
                    let c = a_2d.dot(&b_2d);
                    result.slice_mut(ndarray::s![i, .., ..]).assign(&c);
                }
                Ok(Self::wrap_cpu_data(CpuData::Float32(result.into_dyn())))
            },
            (CpuData::Complex128(a_arr), CpuData::Complex128(b_arr)) => {
                let mut result = ndarray::Array3::<Complex64>::zeros((batch_size, m, n));
                for i in 0..batch_size {
                    let a_slice = a_arr.slice(ndarray::s![i, .., ..]).to_owned();
                    let b_slice = b_arr.slice(ndarray::s![i, .., ..]).to_owned();
                    let a_2d = a_slice.into_dimensionality::<Ix2>().unwrap();
                    let b_2d = b_slice.into_dimensionality::<Ix2>().unwrap();
                    let c = a_2d.dot(&b_2d);
                    result.slice_mut(ndarray::s![i, .., ..]).assign(&c);
                }
                Ok(Self::wrap_cpu_data(CpuData::Complex128(result.into_dyn())))
            },
            _ => Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>("Unsupported dtype combination for bmm")),
        }
    }

    /// Golden phase matmul: e^{iπn/φ} × (A @ B)
    /// Applies SRT phase rotation based on golden ratio.
    pub fn mm_golden_phase(&self, other: &TensorStorage, n: i32) -> PyResult<TensorStorage> {
        let result = self.matmul(other)?;
        // Compute phase: e^{iπn/φ} = cos(πn/φ) + i·sin(πn/φ)
        let phi = Self::phi();
        let angle = std::f64::consts::PI * (n as f64) / phi;
        let cos_a = angle.cos();
        let sin_a = angle.sin();

        // Apply phase rotation
        let cpu = result.ensure_cpu()?;
        let rotated = match cpu {
            CpuData::Float64(arr) => {
                // For real, just multiply by cos(angle) (imaginary part would be sin*arr)
                CpuData::Float64(arr.mapv(|x| x * cos_a))
            },
            CpuData::Float32(arr) => {
                CpuData::Float32(arr.mapv(|x| x * (cos_a as f32)))
            },
            CpuData::Complex128(arr) => {
                let phase = Complex64::new(cos_a, sin_a);
                CpuData::Complex128(arr.mapv(|x| x * phase))
            },
            CpuData::Int64(_) => {
                return Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>(
                    "Golden phase not supported for int64"
                ));
            },
        };
        Ok(Self::wrap_cpu_data(rotated))
    }

    /// Golden-weighted matmul: C[i,j] = Σₖ A[i,k] × B[k,j] × exp(-k²/φ)
    /// Applies Golden Gaussian weights during matrix multiplication.
    pub fn mm_golden_weighted(&self, other: &TensorStorage) -> PyResult<TensorStorage> {
        let a_cpu = self.ensure_cpu()?;
        let b_cpu = other.ensure_cpu()?;
        let phi = Self::phi();

        match (a_cpu, b_cpu) {
            (CpuData::Float64(a_arr), CpuData::Float64(b_arr)) => {
                let a_2d = a_arr.clone().into_dimensionality::<Ix2>()
                    .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("A must be 2D: {}", e)))?;
                let b_2d = b_arr.clone().into_dimensionality::<Ix2>()
                    .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("B must be 2D: {}", e)))?;

                let (m, k_a) = (a_2d.nrows(), a_2d.ncols());
                let (k_b, n) = (b_2d.nrows(), b_2d.ncols());

                if k_a != k_b {
                    return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                        format!("Dimension mismatch: {} vs {}", k_a, k_b)
                    ));
                }

                // Precompute golden weights: w[k] = exp(-k²/φ)
                let weights: Vec<f64> = (0..k_a).map(|k| (-(k as f64).powi(2) / phi).exp()).collect();

                // Weighted matmul
                let mut result = ndarray::Array2::<f64>::zeros((m, n));
                for i in 0..m {
                    for j in 0..n {
                        let mut sum = 0.0;
                        for k in 0..k_a {
                            sum += a_2d[[i, k]] * b_2d[[k, j]] * weights[k];
                        }
                        result[[i, j]] = sum;
                    }
                }
                Ok(Self::wrap_cpu_data(CpuData::Float64(result.into_dyn())))
            },
            _ => Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>(
                "mm_golden_weighted currently only supports float64"
            )),
        }
    }

    /// Weighted sum of tensors: Σ w_i × T_i
    /// Used for DHSR projection summation over lattice points.
    #[staticmethod]
    pub fn projection_sum(weights: &Bound<'_, PyList>, tensors: &Bound<'_, PyList>) -> PyResult<TensorStorage> {
        let n = weights.len();
        if n == 0 {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>("Empty weights list"));
        }
        if n != tensors.len() {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                format!("Weights ({}) and tensors ({}) must have same length", n, tensors.len())
            ));
        }

        // Extract first tensor to get shape and dtype
        let first_tensor: PyRef<TensorStorage> = tensors.get_item(0)?.extract()?;
        let first_weight: f64 = weights.get_item(0)?.extract()?;
        let mut result = first_tensor.mul_scalar(first_weight)?;

        // Accumulate remaining terms
        for i in 1..n {
            let w: f64 = weights.get_item(i)?.extract()?;
            let t: PyRef<TensorStorage> = tensors.get_item(i)?.extract()?;
            let scaled = t.mul_scalar(w)?;
            result = result.add(&scaled)?;
        }

        Ok(result)
    }

    /// Syntony-scaled matmul: σ(ψ) × (A @ B)
    /// Applies syntony measure as scaling factor.
    pub fn mm_syntony(&self, other: &TensorStorage) -> PyResult<TensorStorage> {
        let result = self.matmul(other)?;
        // Compute syntony from self
        let syntony = self.compute_syntony_basic();
        // Scale by (1 - q × syntony) for coherent states
        let scale = 1.0 - Self::q_deficit() * syntony;
        result.mul_scalar(scale)
    }

    // ===== DHSR Operations =====

    pub fn compute_syntony_basic(&self) -> f64 {
        const EPSILON: f64 = 1e-10;
        const ALPHA_0: f64 = 0.1;

        let d_psi = match self.differentiate(ALPHA_0) {
            Ok(s) => s,
            Err(_) => return 0.5,
        };

        let h_d_psi = match d_psi.harmonize(Self::phi_inv(), 0.0) {
            Ok(s) => s,
            Err(_) => return 0.5,
        };

        let self_cpu = match self.ensure_cpu() {
            Ok(c) => c,
            Err(_) => return 0.5,
        };
        let d_cpu = match d_psi.ensure_cpu() {
            Ok(c) => c,
            Err(_) => return 0.5,
        };
        let hd_cpu = match h_d_psi.ensure_cpu() {
            Ok(c) => c,
            Err(_) => return 0.5,
        };

        let (numerator, denominator) = match (self_cpu, d_cpu, hd_cpu) {
            (CpuData::Float64(psi), CpuData::Float64(d), CpuData::Float64(hd)) => {
                let num: f64 = hd.iter().zip(d.iter())
                    .map(|(a, b)| (a - b).powi(2)).sum::<f64>().sqrt();
                let den: f64 = d.iter().zip(psi.iter())
                    .map(|(a, b)| (a - b).powi(2)).sum::<f64>().sqrt();
                (num, den)
            },
            (CpuData::Float32(psi), CpuData::Float32(d), CpuData::Float32(hd)) => {
                let num: f64 = hd.iter().zip(d.iter())
                    .map(|(a, b)| ((*a - *b) as f64).powi(2)).sum::<f64>().sqrt();
                let den: f64 = d.iter().zip(psi.iter())
                    .map(|(a, b)| ((*a - *b) as f64).powi(2)).sum::<f64>().sqrt();
                (num, den)
            },
            (CpuData::Complex128(psi), CpuData::Complex128(d), CpuData::Complex128(hd)) => {
                let num: f64 = hd.iter().zip(d.iter())
                    .map(|(a, b)| (*a - *b).norm_sqr()).sum::<f64>().sqrt();
                let den: f64 = d.iter().zip(psi.iter())
                    .map(|(a, b)| (*a - *b).norm_sqr()).sum::<f64>().sqrt();
                (num, den)
            },
            _ => return 0.5,
        };

        (numerator / (denominator + EPSILON)).clamp(0.0, 1.0)
    }

    pub fn free_energy(&self) -> f64 {
        let cpu = match self.ensure_cpu() {
            Ok(c) => c,
            Err(_) => return 0.0,
        };

        match cpu {
            CpuData::Float64(arr) => Self::compute_free_energy(&arr.iter().cloned().collect::<Vec<_>>()),
            CpuData::Float32(arr) => Self::compute_free_energy(&arr.iter().map(|x| *x as f64).collect::<Vec<_>>()),
            CpuData::Complex128(arr) => Self::compute_free_energy(&arr.iter().map(|x| x.norm()).collect::<Vec<_>>()),
            CpuData::Int64(arr) => Self::compute_free_energy(&arr.iter().map(|x| *x as f64).collect::<Vec<_>>()),
        }
    }

    pub fn compute_tv_sum(&self) -> f64 {
        let cpu = match self.ensure_cpu() {
            Ok(c) => c,
            Err(_) => return 0.0,
        };

        match cpu {
            CpuData::Float64(arr) => {
                let flat: Vec<_> = arr.iter().cloned().collect();
                if flat.len() < 2 { return 0.0; }
                flat.windows(2).map(|w| (w[1] - w[0]).abs()).sum()
            },
            CpuData::Float32(arr) => {
                let flat: Vec<_> = arr.iter().cloned().collect();
                if flat.len() < 2 { return 0.0; }
                flat.windows(2).map(|w| (w[1] - w[0]).abs() as f64).sum()
            },
            CpuData::Complex128(arr) => {
                let flat: Vec<_> = arr.iter().cloned().collect();
                if flat.len() < 2 { return 0.0; }
                flat.windows(2).map(|w| (w[1] - w[0]).norm()).sum()
            },
            CpuData::Int64(arr) => {
                let flat: Vec<_> = arr.iter().cloned().collect();
                if flat.len() < 2 { return 0.0; }
                flat.windows(2).map(|w| (w[1] - w[0]).abs() as f64).sum()
            },
        }
    }

    pub fn differentiate(&self, alpha: f64) -> PyResult<TensorStorage> {
        let cpu = self.ensure_cpu()?;

        match cpu {
            CpuData::Float64(arr) => {
                let n = arr.len();
                if n == 0 { return Ok(self.clone_storage()); }

                let values: Vec<f64> = arr.iter().cloned().collect();
                let syntony = 1.0 - Self::compute_shannon_entropy(&values);
                let effective_alpha = alpha * (1.0 - syntony);

                let original_energy: f64 = arr.iter().map(|x| x * x).sum();
                if original_energy < 1e-15 { return Ok(self.clone_storage()); }

                let mut result = arr.clone();
                let mut rng = rand::thread_rng();
                let mean_amp = (original_energy / n as f64).sqrt();

                for (i, x) in result.iter_mut().enumerate() {
                    let mode_weight = (i as f64) / (n as f64);
                    let noise: f64 = rng.gen::<f64>() * 2.0 - 1.0;
                    *x = *x + effective_alpha * noise * mode_weight * mean_amp;
                }

                let result_energy: f64 = result.iter().map(|x| x * x).sum();
                if result_energy > 1e-15 {
                    let scale = (original_energy / result_energy).sqrt();
                    for x in result.iter_mut() { *x *= scale; }
                }

                Ok(Self::wrap_cpu(CpuData::Float64(result), &self.device))
            },
            CpuData::Float32(arr) => {
                let n = arr.len();
                if n == 0 { return Ok(self.clone_storage()); }

                let values: Vec<f64> = arr.iter().map(|x| *x as f64).collect();
                let syntony = 1.0 - Self::compute_shannon_entropy(&values);
                let effective_alpha = alpha * (1.0 - syntony);

                let original_energy: f64 = arr.iter().map(|x| (*x as f64).powi(2)).sum();
                if original_energy < 1e-15 { return Ok(self.clone_storage()); }

                let mut result = arr.clone();
                let mut rng = rand::thread_rng();
                let mean_amp = (original_energy / n as f64).sqrt();

                for (i, x) in result.iter_mut().enumerate() {
                    let mode_weight = (i as f64) / (n as f64);
                    let noise: f64 = rng.gen::<f64>() * 2.0 - 1.0;
                    *x = *x + (effective_alpha * noise * mode_weight * mean_amp) as f32;
                }

                let result_energy: f64 = result.iter().map(|x| (*x as f64).powi(2)).sum();
                if result_energy > 1e-15 {
                    let scale = ((original_energy / result_energy).sqrt()) as f32;
                    for x in result.iter_mut() { *x *= scale; }
                }

                Ok(Self::wrap_cpu(CpuData::Float32(result), &self.device))
            },
            CpuData::Complex128(arr) => {
                let n = arr.len();
                if n == 0 { return Ok(self.clone_storage()); }

                let values: Vec<f64> = arr.iter().map(|x| x.norm()).collect();
                let syntony = 1.0 - Self::compute_shannon_entropy(&values);
                let effective_alpha = alpha * (1.0 - syntony);

                let original_energy: f64 = arr.iter().map(|x| x.norm_sqr()).sum();
                if original_energy < 1e-15 { return Ok(self.clone_storage()); }

                let mean_amp = (original_energy / n as f64).sqrt();
                let mut result = arr.clone();
                let mut rng = rand::thread_rng();

                for (i, x) in result.iter_mut().enumerate() {
                    let mode_weight = (i as f64) / (n as f64);
                    let noise_real: f64 = rng.gen::<f64>() * 2.0 - 1.0;
                    let noise_imag: f64 = rng.gen::<f64>() * 2.0 - 1.0;
                    let noise = Complex64::new(noise_real, noise_imag) * mode_weight * mean_amp;
                    *x = *x + noise * effective_alpha;
                }

                let result_energy: f64 = result.iter().map(|x| x.norm_sqr()).sum();
                if result_energy > 1e-15 {
                    let scale = (original_energy / result_energy).sqrt();
                    for x in result.iter_mut() { *x *= scale; }
                }

                Ok(Self::wrap_cpu(CpuData::Complex128(result), &self.device))
            },
            CpuData::Int64(_) => Ok(self.clone_storage()),
        }
    }

    pub fn harmonize(&self, strength: f64, _gamma: f64) -> PyResult<TensorStorage> {
        let gamma = if (strength - Self::phi_inv()).abs() < 0.001 { Self::phi_inv() } else { strength };
        let cpu = self.ensure_cpu()?;

        match cpu {
            CpuData::Float64(arr) => {
                let n = arr.len();
                if n == 0 { return Ok(self.clone_storage()); }

                let total_energy: f64 = arr.iter().map(|x| x * x).sum();
                if total_energy < 1e-15 { return Ok(self.clone_storage()); }

                let golden_weights: Vec<f64> = (0..n)
                    .map(|i| (-((i as f64).powi(2)) / Self::phi()).exp())
                    .collect();
                let weight_sum: f64 = golden_weights.iter().sum();

                let target: Vec<f64> = golden_weights.iter()
                    .map(|w| (total_energy * w / weight_sum * (1.0 - Self::q_deficit())).sqrt())
                    .collect();

                let mut result = arr.clone();
                for (i, x) in result.iter_mut().enumerate() {
                    let sign = if *x >= 0.0 { 1.0 } else { -1.0 };
                    let target_val = sign * target[i];
                    *x = (1.0 - gamma) * (*x) + gamma * target_val;
                }

                Ok(Self::wrap_cpu(CpuData::Float64(result), &self.device))
            },
            CpuData::Float32(arr) => {
                let n = arr.len();
                if n == 0 { return Ok(self.clone_storage()); }

                let total_energy: f64 = arr.iter().map(|x| (*x as f64).powi(2)).sum();
                if total_energy < 1e-15 { return Ok(self.clone_storage()); }

                let golden_weights: Vec<f64> = (0..n)
                    .map(|i| (-((i as f64).powi(2)) / Self::phi()).exp())
                    .collect();
                let weight_sum: f64 = golden_weights.iter().sum();

                let target: Vec<f32> = golden_weights.iter()
                    .map(|w| (total_energy * w / weight_sum * (1.0 - Self::q_deficit())).sqrt() as f32)
                    .collect();

                let mut result = arr.clone();
                for (i, x) in result.iter_mut().enumerate() {
                    let sign = if *x >= 0.0 { 1.0f32 } else { -1.0f32 };
                    let target_val = sign * target[i];
                    *x = (1.0 - gamma as f32) * (*x) + (gamma as f32) * target_val;
                }

                Ok(Self::wrap_cpu(CpuData::Float32(result), &self.device))
            },
            CpuData::Complex128(arr) => {
                let n = arr.len();
                if n == 0 { return Ok(self.clone_storage()); }

                let total_energy: f64 = arr.iter().map(|x| x.norm_sqr()).sum();
                if total_energy < 1e-15 { return Ok(self.clone_storage()); }

                let golden_weights: Vec<f64> = (0..n)
                    .map(|i| (-((i as f64).powi(2)) / Self::phi()).exp())
                    .collect();
                let weight_sum: f64 = golden_weights.iter().sum();

                let target_amplitudes: Vec<f64> = golden_weights.iter()
                    .map(|w| (total_energy * w / weight_sum * (1.0 - Self::q_deficit())).sqrt())
                    .collect();

                let mut result = arr.clone();
                for (i, x) in result.iter_mut().enumerate() {
                    let phase = x.arg();
                    let target_val = Complex64::from_polar(target_amplitudes[i], phase);
                    *x = (*x) * (1.0 - gamma) + target_val * gamma;
                }

                Ok(Self::wrap_cpu(CpuData::Complex128(result), &self.device))
            },
            CpuData::Int64(_) => Ok(self.clone_storage()),
        }
    }
}

// Private implementation
impl TensorStorage {
    // All constants derived from exact symbolic infrastructure
    // No hardcoded floating-point values

    /// φ - derived from GoldenExact::phi()
    #[inline]
    fn phi() -> f64 {
        GoldenExact::phi().to_f64()
    }

    /// φ⁻¹ = φ - 1 - derived from GoldenExact::phi_hat()
    #[inline]
    fn phi_inv() -> f64 {
        GoldenExact::phi_hat().to_f64()
    }

    /// q - the universal syntony deficit - derived from FundamentalConstant::Q
    #[inline]
    fn q_deficit() -> f64 {
        FundamentalConstant::Q.approx_f64()
    }

    /// E* = e^π - π - derived from FundamentalConstant::EStar
    #[inline]
    fn e_star() -> f64 {
        FundamentalConstant::EStar.approx_f64()
    }

    /// φ^k - derived from exact Fibonacci formula via GoldenExact::phi_power(k)
    #[inline]
    fn phi_power(k: i32) -> f64 {
        GoldenExact::phi_power(k).to_f64()
    }

    /// Symbolic correction (1 + sign × q/N) evaluated to f64
    #[inline]
    fn correction_factor(n: u32, sign: i8) -> f64 {
        let one = SymExpr::from_int(1);
        let q = SymExpr::q();
        let n_expr = SymExpr::from_int(n as i128);
        let q_over_n = q.div(n_expr);

        if sign >= 0 {
            one.add(q_over_n).eval_f64()
        } else {
            one.sub(q_over_n).eval_f64()
        }
    }

    /// Generic binary CUDA operation (add, sub, mul, div)
    #[cfg(feature = "cuda")]
    fn binary_cuda_op(&self, a: &Arc<CudaData>, b: &Arc<CudaData>, device: &Arc<CudaDevice>, op: &str) -> PyResult<TensorStorage> {
        let n = self.shape.iter().product::<usize>();
        let cfg = launch_cfg(n);

        let (out_data, out_dtype) = match (a.as_ref(), b.as_ref()) {
            (CudaData::Float64(a_slice), CudaData::Float64(b_slice)) => {
                let mut out: CudaSlice<f64> = device.alloc_zeros(n)
                    .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
                let func = device.get_func("syntonic", &format!("{}_f64", op)).unwrap();
                unsafe { func.launch(cfg, (&mut out, a_slice, b_slice, n as i32)) }
                    .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
                (CudaData::Float64(out), "float64".to_string())
            },
            (CudaData::Float32(a_slice), CudaData::Float32(b_slice)) => {
                let mut out: CudaSlice<f32> = device.alloc_zeros(n)
                    .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
                let func = device.get_func("syntonic", &format!("{}_f32", op)).unwrap();
                unsafe { func.launch(cfg, (&mut out, a_slice, b_slice, n as i32)) }
                    .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
                (CudaData::Float32(out), "float32".to_string())
            },
            (CudaData::Complex128(a_slice), CudaData::Complex128(b_slice)) => {
                let mut out: CudaSlice<f64> = device.alloc_zeros(n * 2)
                    .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
                let func = device.get_func("syntonic", &format!("{}_c128", op)).unwrap();
                unsafe { func.launch(cfg, (&mut out, a_slice, b_slice, n as i32)) }
                    .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
                (CudaData::Complex128(out), "complex128".to_string())
            },
            _ => return Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>("Dtype mismatch on CUDA")),
        };

        Ok(TensorStorage {
            data: TensorData::Cuda {
                data: Arc::new(out_data),
                device: device.clone(),
                shape: self.shape.clone(),
                dtype: out_dtype,
            },
            shape: self.shape.clone(),
            device: self.device.clone(),
        })
    }

    /// Generic unary CUDA operation (neg, abs)
    #[cfg(feature = "cuda")]
    fn unary_cuda_op(&self, a: &Arc<CudaData>, device: &Arc<CudaDevice>, op: &str) -> PyResult<TensorStorage> {
        let n = self.shape.iter().product::<usize>();
        let cfg = launch_cfg(n);

        let (out_data, out_dtype) = match a.as_ref() {
            CudaData::Float64(a_slice) => {
                let mut out: CudaSlice<f64> = device.alloc_zeros(n)
                    .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
                let func = device.get_func("syntonic", &format!("{}_f64", op)).unwrap();
                unsafe { func.launch(cfg, (&mut out, a_slice, n as i32)) }
                    .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
                (CudaData::Float64(out), "float64".to_string())
            },
            CudaData::Float32(a_slice) => {
                let mut out: CudaSlice<f32> = device.alloc_zeros(n)
                    .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
                let func = device.get_func("syntonic", &format!("{}_f32", op)).unwrap();
                unsafe { func.launch(cfg, (&mut out, a_slice, n as i32)) }
                    .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
                (CudaData::Float32(out), "float32".to_string())
            },
            CudaData::Complex128(a_slice) => {
                // For neg_c128, output is complex. For abs, would need different handling
                if op == "neg" {
                    let mut out: CudaSlice<f64> = device.alloc_zeros(n * 2)
                        .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
                    let func = device.get_func("syntonic", "neg_c128").unwrap();
                    unsafe { func.launch(cfg, (&mut out, a_slice, n as i32)) }
                        .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
                    (CudaData::Complex128(out), "complex128".to_string())
                } else {
                    // abs on complex returns to CPU (complex abs not implemented in CUDA yet)
                    return Err(PyErr::new::<pyo3::exceptions::PyNotImplementedError, _>(
                        "abs() on complex CUDA tensors not yet implemented"
                    ));
                }
            },
        };

        Ok(TensorStorage {
            data: TensorData::Cuda {
                data: Arc::new(out_data),
                device: device.clone(),
                shape: self.shape.clone(),
                dtype: out_dtype,
            },
            shape: self.shape.clone(),
            device: self.device.clone(),
        })
    }

    fn wrap_cpu(data: CpuData, device: &DeviceType) -> Self {
        let shape = match &data {
            CpuData::Float32(a) => a.shape().to_vec(),
            CpuData::Float64(a) => a.shape().to_vec(),
            CpuData::Complex128(a) => a.shape().to_vec(),
            CpuData::Int64(a) => a.shape().to_vec(),
        };
        TensorStorage { data: TensorData::Cpu(data), shape, device: device.clone() }
    }

    /// Wrap CPU data with default CPU device
    fn wrap_cpu_data(data: CpuData) -> Self {
        Self::wrap_cpu(data, &DeviceType::Cpu)
    }

    fn clone_storage(&self) -> TensorStorage {
        TensorStorage {
            data: self.data.clone(),
            shape: self.shape.clone(),
            device: self.device.clone(),
        }
    }

    fn ensure_cpu(&self) -> PyResult<CpuData> {
        match &self.data {
            TensorData::Cpu(cpu) => Ok(cpu.clone()),
            #[cfg(feature = "cuda")]
            TensorData::Cuda { data, device, shape, dtype: _ } => {
                // Transfer from GPU to CPU
                Self::cuda_to_cpu(data, device, shape)
            },
        }
    }

    #[cfg(feature = "cuda")]
    fn cpu_to_cuda(cpu_data: CpuData, shape: Vec<usize>, device_idx: usize) -> PyResult<TensorStorage> {
        let device = CudaDevice::new(device_idx)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("CUDA init failed: {}", e)))?;

        let (cuda_data, dtype) = match cpu_data {
            CpuData::Float32(arr) => {
                let slice = device.htod_sync_copy(arr.as_slice().unwrap())
                    .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
                (CudaData::Float32(slice), "float32".to_string())
            },
            CpuData::Float64(arr) => {
                let slice = device.htod_sync_copy(arr.as_slice().unwrap())
                    .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
                (CudaData::Float64(slice), "float64".to_string())
            },
            CpuData::Complex128(arr) => {
                // Complex64 is #[repr(C)] so we can safely reinterpret as [f64; 2]
                // This avoids allocating a new Vec and is much faster
                let complex_slice = arr.as_slice().unwrap();
                let interleaved: &[f64] = unsafe {
                    std::slice::from_raw_parts(
                        complex_slice.as_ptr() as *const f64,
                        complex_slice.len() * 2
                    )
                };
                let slice = device.htod_sync_copy(interleaved)
                    .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
                (CudaData::Complex128(slice), "complex128".to_string())
            },
            CpuData::Int64(_) => return Err(PyErr::new::<pyo3::exceptions::PyNotImplementedError, _>(
                "Int64 not supported on CUDA"
            )),
        };

        Ok(TensorStorage {
            data: TensorData::Cuda {
                data: Arc::new(cuda_data),
                device,
                shape: shape.clone(),
                dtype,
            },
            shape,
            device: DeviceType::Cuda(device_idx),
        })
    }

    #[cfg(feature = "cuda")]
    fn cuda_to_cpu(data: &Arc<CudaData>, device: &Arc<CudaDevice>, shape: &[usize]) -> PyResult<CpuData> {
        let dim = IxDyn(shape);

        match data.as_ref() {
            CudaData::Float32(slice) => {
                let mut host_data = vec![0f32; slice.len()];
                device.dtoh_sync_copy_into(slice, &mut host_data)
                    .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
                Ok(CpuData::Float32(ArrayD::from_shape_vec(dim, host_data)
                    .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?))
            },
            CudaData::Float64(slice) => {
                let mut host_data = vec![0f64; slice.len()];
                device.dtoh_sync_copy_into(slice, &mut host_data)
                    .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
                Ok(CpuData::Float64(ArrayD::from_shape_vec(dim, host_data)
                    .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?))
            },
            CudaData::Complex128(slice) => {
                // slice contains interleaved [re0, im0, re1, im1, ...]
                // Complex64 is #[repr(C)] so we can zero-copy into it
                let num_complex = slice.len() / 2;

                // Allocate uninitialized buffer for f64 values
                let mut host_data: Vec<f64> = Vec::with_capacity(slice.len());
                unsafe { host_data.set_len(slice.len()); }

                device.dtoh_sync_copy_into(slice, &mut host_data)
                    .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;

                // Convert the Vec<f64> to Vec<Complex64> via reinterpret
                // This is safe because Complex64 is #[repr(C)] with layout [re, im]
                let complex_data: Vec<Complex64> = unsafe {
                    let ptr = host_data.as_mut_ptr() as *mut Complex64;
                    let cap = host_data.capacity() / 2;
                    std::mem::forget(host_data);  // Prevent double-free
                    Vec::from_raw_parts(ptr, num_complex, cap)
                };

                Ok(CpuData::Complex128(ArrayD::from_shape_vec(dim, complex_data)
                    .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?))
            },
        }
    }

    fn compute_shannon_entropy(values: &[f64]) -> f64 {
        if values.is_empty() { return 0.0; }
        let sum_sq: f64 = values.iter().map(|x| x * x).sum();
        if sum_sq < 1e-15 { return 0.0; }
        let entropy: f64 = values.iter()
            .map(|x| {
                let p = (x * x) / sum_sq;
                if p > 1e-15 { -p * p.ln() } else { 0.0 }
            })
            .sum();
        let max_entropy = (values.len() as f64).ln();
        if max_entropy > 0.0 { entropy / max_entropy } else { 0.0 }
    }

    fn compute_free_energy(values: &[f64]) -> f64 {
        if values.is_empty() { return 0.0; }
        let sum_sq: f64 = values.iter().map(|x| x * x).sum();
        if sum_sq < 1e-15 { return 0.0; }
        let n = values.len() as f64;
        let mut free_energy = 0.0;
        for (i, x) in values.iter().enumerate() {
            let rho = (x * x) / sum_sq;
            if rho > 1e-15 {
                let norm_i = (i as f64) / n;
                let entropy_term = rho * rho.ln();
                let potential_term = rho * norm_i * norm_i / Self::phi();
                free_energy += entropy_term + potential_term;
            }
        }
        free_energy / Self::e_star()
    }

    // ===== Internal Methods for linalg module =====

    /// Internal access to CPU data (for linalg module use)
    pub(crate) fn ensure_cpu_internal(&self) -> Result<CpuData, pyo3::PyErr> {
        self.ensure_cpu()
    }

    /// Internal access to device reference
    pub(crate) fn device_ref(&self) -> &DeviceType {
        &self.device
    }

    /// Internal clone
    pub(crate) fn clone_storage_internal(&self) -> TensorStorage {
        self.clone_storage()
    }

    /// Create TensorStorage from CPU data (internal constructor for linalg)
    pub(crate) fn new_from_cpu(data: CpuData, shape: Vec<usize>, device: DeviceType) -> TensorStorage {
        TensorStorage {
            data: TensorData::Cpu(data),
            shape,
            device,
        }
    }

    /// Create TensorStorage from CUDA data (internal constructor for cuda module)
    #[cfg(feature = "cuda")]
    pub(crate) fn new_from_cuda(
        cuda_data: CudaData,
        device: std::sync::Arc<CudaDevice>,
        shape: Vec<usize>,
        device_idx: usize,
    ) -> TensorStorage {
        let dtype = match &cuda_data {
            CudaData::Float32(_) => "float32".to_string(),
            CudaData::Float64(_) => "float64".to_string(),
            CudaData::Complex128(_) => "complex128".to_string(),
        };

        TensorStorage {
            data: TensorData::Cuda {
                data: std::sync::Arc::new(cuda_data),
                device,
                shape: shape.clone(),
                dtype,
            },
            shape,
            device: DeviceType::Cuda(device_idx),
        }
    }
}


/// Check if CUDA is available
#[pyfunction]
pub fn cuda_is_available() -> bool {
    #[cfg(feature = "cuda")]
    {
        CudaDevice::new(0).is_ok()
    }
    #[cfg(not(feature = "cuda"))]
    {
        false
    }
}

/// Get CUDA device count
#[pyfunction]
pub fn cuda_device_count() -> usize {
    #[cfg(feature = "cuda")]
    {
        CudaDevice::count().unwrap_or(0) as usize
    }
    #[cfg(not(feature = "cuda"))]
    {
        0
    }
}
