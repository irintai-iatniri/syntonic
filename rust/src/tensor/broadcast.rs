//! Broadcasting utilities for element-wise operations.
//!
//! Provides NumPy-style broadcasting for tensors with different shapes.
//!
//! Broadcasting rules:
//! 1. If tensors have different number of dimensions, prepend 1s to the smaller shape
//! 2. For each dimension, sizes must be equal or one of them must be 1
//! 3. The tensor with size 1 is "stretched" to match the other

use pyo3::prelude::*;

/// Compute the broadcast shape for two shapes.
///
/// Returns None if shapes are incompatible.
///
/// # Examples
/// ```
/// assert_eq!(broadcast_shape(&[3, 1], &[1, 4]), Some(vec![3, 4]));
/// assert_eq!(broadcast_shape(&[2, 3], &[3]), Some(vec![2, 3]));
/// assert_eq!(broadcast_shape(&[2, 3], &[4]), None);
/// ```
pub fn broadcast_shape(a: &[usize], b: &[usize]) -> Option<Vec<usize>> {
    let max_ndim = a.len().max(b.len());
    let mut result = Vec::with_capacity(max_ndim);
    
    // Iterate from the last dimension backwards
    for i in 0..max_ndim {
        let dim_a = if i < a.len() { a[a.len() - 1 - i] } else { 1 };
        let dim_b = if i < b.len() { b[b.len() - 1 - i] } else { 1 };
        
        if dim_a == dim_b {
            result.push(dim_a);
        } else if dim_a == 1 {
            result.push(dim_b);
        } else if dim_b == 1 {
            result.push(dim_a);
        } else {
            // Incompatible dimensions
            return None;
        }
    }
    
    result.reverse();
    Some(result)
}

/// Check if shapes are broadcastable.
pub fn are_broadcastable(a: &[usize], b: &[usize]) -> bool {
    broadcast_shape(a, b).is_some()
}

/// Compute strides for a tensor with given shape.
pub fn compute_strides(shape: &[usize]) -> Vec<usize> {
    let mut strides = vec![1; shape.len()];
    for i in (0..shape.len().saturating_sub(1)).rev() {
        strides[i] = strides[i + 1] * shape[i + 1];
    }
    strides
}

/// Calculate the linear index in a tensor given multi-dimensional indices.
pub fn linear_index(indices: &[usize], strides: &[usize]) -> usize {
    indices.iter().zip(strides.iter()).map(|(&i, &s)| i * s).sum()
}

/// Calculate multi-dimensional indices from a linear index.
pub fn unravel_index(linear_idx: usize, shape: &[usize]) -> Vec<usize> {
    let strides = compute_strides(shape);
    let mut indices = Vec::with_capacity(shape.len());
    let mut remaining = linear_idx;
    
    for &stride in &strides {
        indices.push(remaining / stride);
        remaining %= stride;
    }
    
    indices
}

/// Map a linear index from the output tensor to source tensor indices,
/// handling broadcasting (size-1 dimensions map to index 0).
pub fn broadcast_index(
    linear_idx: usize,
    output_shape: &[usize],
    source_shape: &[usize],
) -> usize {
    let output_indices = unravel_index(linear_idx, output_shape);
    let output_strides = compute_strides(output_shape);
    let source_strides = compute_strides(source_shape);
    
    let offset = output_shape.len().saturating_sub(source_shape.len());
    let mut source_idx = 0;
    
    for (i, &dim) in source_shape.iter().enumerate() {
        let out_idx = output_indices[offset + i];
        // If dimension is 1, broadcast (always use index 0)
        let src_idx = if dim == 1 { 0 } else { out_idx };
        source_idx += src_idx * source_strides[i];
    }
    
    source_idx
}

/// Apply a binary operation with broadcasting.
///
/// # Arguments
/// * `a` - First tensor data
/// * `a_shape` - First tensor shape
/// * `b` - Second tensor data
/// * `b_shape` - Second tensor shape
/// * `op` - Binary operation to apply
///
/// # Returns
/// Result tensor with broadcast shape
pub fn broadcast_binary_op<F>(
    a: &[f64],
    a_shape: &[usize],
    b: &[f64],
    b_shape: &[usize],
    op: F,
) -> Option<(Vec<f64>, Vec<usize>)>
where
    F: Fn(f64, f64) -> f64,
{
    let out_shape = broadcast_shape(a_shape, b_shape)?;
    let out_size: usize = out_shape.iter().product();
    
    let mut result = Vec::with_capacity(out_size);
    
    for i in 0..out_size {
        let a_idx = broadcast_index(i, &out_shape, a_shape);
        let b_idx = broadcast_index(i, &out_shape, b_shape);
        result.push(op(a[a_idx], b[b_idx]));
    }
    
    Some((result, out_shape))
}

/// Broadcast addition.
pub fn broadcast_add(
    a: &[f64],
    a_shape: &[usize],
    b: &[f64],
    b_shape: &[usize],
) -> Option<(Vec<f64>, Vec<usize>)> {
    broadcast_binary_op(a, a_shape, b, b_shape, |x, y| x + y)
}

/// Broadcast multiplication.
pub fn broadcast_mul(
    a: &[f64],
    a_shape: &[usize],
    b: &[f64],
    b_shape: &[usize],
) -> Option<(Vec<f64>, Vec<usize>)> {
    broadcast_binary_op(a, a_shape, b, b_shape, |x, y| x * y)
}

/// Broadcast subtraction.
pub fn broadcast_sub(
    a: &[f64],
    a_shape: &[usize],
    b: &[f64],
    b_shape: &[usize],
) -> Option<(Vec<f64>, Vec<usize>)> {
    broadcast_binary_op(a, a_shape, b, b_shape, |x, y| x - y)
}

/// Broadcast division.
pub fn broadcast_div(
    a: &[f64],
    a_shape: &[usize],
    b: &[f64],
    b_shape: &[usize],
) -> Option<(Vec<f64>, Vec<usize>)> {
    broadcast_binary_op(a, a_shape, b, b_shape, |x, y| x / y)
}

// === In-place Operations ===

/// In-place scalar addition.
pub fn inplace_add_scalar(data: &mut [f64], scalar: f64) {
    for x in data.iter_mut() {
        *x += scalar;
    }
}

/// In-place scalar multiplication.
pub fn inplace_mul_scalar(data: &mut [f64], scalar: f64) {
    for x in data.iter_mut() {
        *x *= scalar;
    }
}

/// In-place scalar subtraction.
pub fn inplace_sub_scalar(data: &mut [f64], scalar: f64) {
    for x in data.iter_mut() {
        *x -= scalar;
    }
}

/// In-place scalar division.
pub fn inplace_div_scalar(data: &mut [f64], scalar: f64) {
    let inv = 1.0 / scalar;
    for x in data.iter_mut() {
        *x *= inv;
    }
}

/// In-place negation.
pub fn inplace_negate(data: &mut [f64]) {
    for x in data.iter_mut() {
        *x = -*x;
    }
}

/// In-place absolute value.
pub fn inplace_abs(data: &mut [f64]) {
    for x in data.iter_mut() {
        *x = x.abs();
    }
}

/// In-place clamp.
pub fn inplace_clamp(data: &mut [f64], min: f64, max: f64) {
    for x in data.iter_mut() {
        *x = x.clamp(min, max);
    }
}

/// In-place golden weight: x = exp(-|n|²/φ)
pub fn inplace_golden_weight(data: &mut [f64], phi: f64) {
    for x in data.iter_mut() {
        *x = (-*x / phi).exp();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_broadcast_shape() {
        assert_eq!(broadcast_shape(&[3, 1], &[1, 4]), Some(vec![3, 4]));
        assert_eq!(broadcast_shape(&[2, 3], &[3]), Some(vec![2, 3]));
        assert_eq!(broadcast_shape(&[5, 4], &[1]), Some(vec![5, 4]));
        assert_eq!(broadcast_shape(&[1], &[5, 4]), Some(vec![5, 4]));
        assert_eq!(broadcast_shape(&[2, 3], &[4]), None);
    }

    #[test]
    fn test_broadcast_add() {
        // [2, 3] + [3] -> [2, 3]
        let a = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let b = vec![10.0, 20.0, 30.0];
        let (result, shape) = broadcast_add(&a, &[2, 3], &b, &[3]).unwrap();
        assert_eq!(shape, vec![2, 3]);
        assert_eq!(result, vec![11.0, 22.0, 33.0, 14.0, 25.0, 36.0]);
    }

    #[test]
    fn test_broadcast_mul_scalar_like() {
        // [3, 4] * [1] -> [3, 4]
        let a = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0];
        let b = vec![2.0];
        let (result, shape) = broadcast_mul(&a, &[3, 4], &b, &[1]).unwrap();
        assert_eq!(shape, vec![3, 4]);
        assert_eq!(result[0], 2.0);
        assert_eq!(result[11], 24.0);
    }

    #[test]
    fn test_inplace_ops() {
        let mut data = vec![1.0, 2.0, 3.0];
        inplace_add_scalar(&mut data, 10.0);
        assert_eq!(data, vec![11.0, 12.0, 13.0]);
        
        inplace_mul_scalar(&mut data, 2.0);
        assert_eq!(data, vec![22.0, 24.0, 26.0]);
        
        inplace_negate(&mut data);
        assert_eq!(data, vec![-22.0, -24.0, -26.0]);
        
        inplace_abs(&mut data);
        assert_eq!(data, vec![22.0, 24.0, 26.0]);
    }
}
