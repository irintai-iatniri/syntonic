use ndarray::{Array2, ArrayBase, Data, Ix2};
use ndarray_linalg::{Eig, Eigh, SVD, QR, Inverse, Solve, Determinant, Trace, Cholesky};
use num_complex::Complex64;
use pyo3::prelude::*;
use crate::tensor::storage::TensorStorage;

// We need to implement these on TensorStorage or expose them.
// Since ndarray-linalg works on traits, we need to extract the array, compute, and wrap back.

// Helper trait to unify linalg ops across f64 and c64
pub trait LinAlgBackend {
    fn eig_calc(&self) -> PyResult<(TensorStorage, TensorStorage)>;
    fn eigh_calc(&self) -> PyResult<(TensorStorage, TensorStorage)>;
    fn svd_calc(&self, full_matrices: bool) -> PyResult<(TensorStorage, TensorStorage, TensorStorage)>;
    fn qr_calc(&self) -> PyResult<(TensorStorage, TensorStorage)>;
    fn inv_calc(&self) -> PyResult<TensorStorage>;
    fn det_calc(&self) -> PyResult<Complex64>;
    fn cholesky_calc(&self) -> PyResult<TensorStorage>;
}

// Implement for TensorStorage
#[pymethods]
impl TensorStorage {
    pub fn eig(&self) -> PyResult<(TensorStorage, TensorStorage)> {
         // Dispatch based on internal type
         // NOTE: ndarray-linalg requires 2D arrays (matrices).
         // Phase 1 assumes 2D for linalg ops.
         
         // Implementation detail: we need to access the internal `data` enum.
         // Since `data` is private in storage.rs, we likely need to move this impl block there 
         // OR make `data` pub(crate). 
         // Moving this logic to storage.rs is cleaner for private access.
         // I will output this file content but effectively merge it into storage.rs logic 
         // OR use a method on TensorStorage to get the data view? 
         // Better to implement `linalg` methods directly in `TensorStorage` impl block in `storage.rs`
         // or make `data` pub(crate).
         
         Err(PyErr::new::<pyo3::exceptions::PyNotImplementedError, _>("Moved to storage.rs"))
    }
}
