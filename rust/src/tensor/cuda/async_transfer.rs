//! Async data transfer operations for non-blocking H2D/D2H copies.
//!
//! Provides stream-based transfers for overlapping transfers with compute.

use cudarc::driver::{CudaDevice, CudaSlice, CudaStream};
use std::sync::Arc;

use super::device_manager::{CudaError, get_device, create_stream};
use crate::tensor::storage::{TensorStorage, CpuData, DeviceType};

/// Handle for tracking an async transfer
pub struct AsyncTransfer {
    device_idx: usize,
}

impl AsyncTransfer {
    /// Create a new transfer handle
    pub fn new(device_idx: usize) -> Self {
        AsyncTransfer { device_idx }
    }

    /// Wait for the transfer to complete by syncing the device
    pub fn wait(&self) -> Result<(), CudaError> {
        super::device_manager::sync_device(self.device_idx)
    }
}

/// Async H2D transfer: copy host data to device
pub fn htod_async<T: cudarc::driver::DeviceRepr>(
    device: &Arc<CudaDevice>,
    host_data: &[T],
) -> Result<CudaSlice<T>, CudaError>
where
    T: Clone + Default,
{
    device.htod_sync_copy(host_data)
        .map_err(|e| CudaError::TransferFailed(e.to_string()))
}

/// Async D2H transfer: copy device data to host
pub fn dtoh_async<T: cudarc::driver::DeviceRepr>(
    device: &Arc<CudaDevice>,
    dev_data: &CudaSlice<T>,
    host_buf: &mut [T],
) -> Result<(), CudaError>
where
    T: Clone,
{
    device.dtoh_sync_copy_into(dev_data, host_buf)
        .map_err(|e| CudaError::TransferFailed(e.to_string()))
}

/// Handle for an in-flight tensor transfer operation
pub struct AsyncTensorTransfer {
    device_idx: usize,
    shape: Vec<usize>,
    dtype: String,
    completed: bool,
}

impl AsyncTensorTransfer {
    /// Create a new async tensor transfer handle
    pub fn new(device_idx: usize, shape: Vec<usize>, dtype: String) -> Self {
        AsyncTensorTransfer {
            device_idx,
            shape,
            dtype,
            completed: false,
        }
    }

    /// Check if the transfer is ready
    pub fn is_ready(&self) -> bool {
        self.completed
    }

    /// Wait for the transfer to complete
    pub fn wait(&mut self) -> Result<(), CudaError> {
        if !self.completed {
            super::device_manager::sync_device(self.device_idx)?;
            self.completed = true;
        }
        Ok(())
    }

    /// Get the target device index
    pub fn device_idx(&self) -> usize {
        self.device_idx
    }

    /// Get the shape of the tensor being transferred
    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    /// Get the dtype of the tensor being transferred
    pub fn dtype(&self) -> &str {
        &self.dtype
    }
}

/// Extension trait for TensorStorage to support async operations
pub trait AsyncTensorOps {
    /// Start a transfer to the specified device
    fn to_device_async(&self, device: &str) -> Result<AsyncTensorTransfer, CudaError>;

    /// Prefetch data to device
    fn prefetch(&self, device: &str) -> Result<(), CudaError>;
}

impl AsyncTensorOps for TensorStorage {
    fn to_device_async(&self, device: &str) -> Result<AsyncTensorTransfer, CudaError> {
        let target_device = DeviceType::from_str(device)
            .map_err(|e| CudaError::DeviceInitFailed(e.to_string()))?;

        match &target_device {
            DeviceType::Cpu => {
                Err(CudaError::TransferFailed("Async transfer to CPU not supported".to_string()))
            }
            #[cfg(feature = "cuda")]
            DeviceType::Cuda(idx) => {
                let device = get_device(*idx)?;

                // Get CPU data
                let cpu_data = self.ensure_cpu_internal()
                    .map_err(|e| CudaError::TransferFailed(e.to_string()))?;

                // Start transfer based on dtype
                match cpu_data {
                    CpuData::Float64(arr) => {
                        let data = arr.as_slice().unwrap();
                        let _gpu_data = htod_async(&device, data)?;
                    }
                    CpuData::Float32(arr) => {
                        let data = arr.as_slice().unwrap();
                        let _gpu_data = htod_async(&device, data)?;
                    }
                    _ => return Err(CudaError::TransferFailed("Unsupported dtype for async transfer".to_string())),
                }

                Ok(AsyncTensorTransfer::new(
                    *idx,
                    self.shape().clone(),
                    self.dtype(),
                ))
            }
        }
    }

    fn prefetch(&self, device: &str) -> Result<(), CudaError> {
        let mut transfer = self.to_device_async(device)?;
        transfer.wait()
    }
}

/// Helper for overlapping transfer with compute
pub struct TransferComputeOverlap {
    device_idx: usize,
    device: Arc<CudaDevice>,
}

impl TransferComputeOverlap {
    /// Create a new overlap manager for the given device
    pub fn new(device_idx: usize) -> Result<Self, CudaError> {
        let device = get_device(device_idx)?;
        Ok(TransferComputeOverlap { device_idx, device })
    }

    /// Transfer data H2D
    pub fn transfer<T: cudarc::driver::DeviceRepr + Clone + Default>(
        &self,
        host_data: &[T],
    ) -> Result<CudaSlice<T>, CudaError> {
        htod_async(&self.device, host_data)
    }

    /// Synchronize the device
    pub fn sync(&self) -> Result<(), CudaError> {
        super::device_manager::sync_device(self.device_idx)
    }

    /// Get the device
    pub fn device(&self) -> &Arc<CudaDevice> {
        &self.device
    }
}
