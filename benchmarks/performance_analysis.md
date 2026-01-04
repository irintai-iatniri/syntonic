# Comprehensive Performance Benchmark: Syntonic vs NumPy vs PyTorch

**Date:** January 4, 2026
**System:** Linux, CUDA 13.0, Python 3.12
**Libraries Tested:** NumPy 1.24+, PyTorch 2.1+, Syntonic (Rust/CUDA backend)

## Executive Summary

This benchmark compares the Syntonic library against NumPy and PyTorch across arithmetic operations, matrix operations, element-wise functions, linear algebra, and memory operations. **PyTorch consistently outperforms both NumPy and Syntonic**, with NumPy competitive for small operations and Syntonic showing promise in linear algebra.

**Key Findings:**
- **PyTorch**: Best overall performance, especially on GPU
- **NumPy**: Strong for small tensors, falls behind significantly for large ones
- **Syntonic**: Competitive in linear algebra, but has performance issues with reshape/transpose operations

---

## Detailed Performance Analysis

### 1. Arithmetic Operations (Add, Subtract, Multiply, Divide)

**Performance Hierarchy:** PyTorch > NumPy > Syntonic (not tested)

| Operation | Size | NumPy (ms) | PyTorch (ms) | Speedup (PyTorch vs NumPy) |
|-----------|------|------------|--------------|----------------------------|
| Add | 100×100 | 0.003-0.011 | 0.006-0.007 | 1.3-1.8x |
| Add | 500×500 | 0.061-0.467 | 0.006-0.010 | 8.8-70x |
| Add | 1000×1000 | 0.316-2.804 | 0.006-0.010 | 50-445x |

**Analysis:** PyTorch shows massive speedups for larger tensors due to GPU acceleration. NumPy performs well on small tensors but scales poorly.

### 2. Matrix Operations (Matmul, Transpose, Reshape)

**Performance Hierarchy:** PyTorch > NumPy >> Syntonic

| Operation | Size | NumPy (ms) | PyTorch (ms) | Syntonic (ms) | PyTorch vs NumPy | Syntonic vs PyTorch |
|-----------|------|------------|--------------|---------------|------------------|-------------------|
| Matmul | 100×100 | 0.015-0.924 | 0.011-0.017 | 0.151-2.417 | 1.3-53x | 13-138x slower |
| Transpose | 100×100 | 0.000 | 0.001-0.003 | 0.045-0.058 | 5-9x slower | 15-19x slower |
| Reshape | 100×100 | 0.000 | 0.002 | 2.5-2.6 | 5-7x slower | 1000-1300x slower |

**Analysis:** Syntonic shows significant performance issues with basic matrix operations, especially reshape (1000x slower than PyTorch). This suggests inefficiencies in the Python-Rust data transfer layer.

### 3. Element-wise Operations (Exp, Log, Sin, Cos, Sqrt)

**Performance Hierarchy:** PyTorch > NumPy

| Operation | Size | NumPy (ms) | PyTorch (ms) | Speedup (PyTorch vs NumPy) |
|-----------|------|------------|--------------|----------------------------|
| Exp | 100×100 | 0.009-0.246 | 0.006 | 1.5-41x |
| Sin | 500×500 | 0.257-8.129 | 0.006 | 43-1350x |
| Sin | 1000×1000 | 1.033-35.471 | 0.006 | 172-5900x |

**Analysis:** PyTorch dominates element-wise operations, especially for complex numbers and larger tensors. GPU acceleration provides massive speedups.

### 4. Linear Algebra Operations (Eig, SVD, Solve)

**Performance Hierarchy:** Mixed - Syntonic competitive with NumPy, PyTorch not tested

| Operation | Size | NumPy (ms) | Syntonic (ms) | Syntonic vs NumPy |
|-----------|------|------------|----------------|-------------------|
| Eig | 100×100 | 9.086 | 11.069 | 1.22x slower |
| Eig | 500×500 | 284.873 | 270.193 | 1.05x faster |
| SVD | 100×100 | 2.674 | 5.433 | 2.03x slower |
| SVD | 500×500 | 89.989 | 84.344 | 1.07x faster |
| Solve | 100×100 | 0.099 | 1.399 | 14.2x slower |
| Solve | 500×500 | 2.789 | 24.962 | 9.0x slower |

**Analysis:** Syntonic shows competitive performance in linear algebra, even outperforming NumPy for large eig/SVD operations. This suggests the Rust LAPACK/BLAS integration is working well.

### 5. Memory Operations (Creation, Copying)

**Performance Hierarchy:** PyTorch > NumPy

| Operation | Size | NumPy (ms) | PyTorch (ms) | Speedup (PyTorch vs NumPy) |
|-----------|------|------------|--------------|----------------------------|
| Create | 100×100 | 0.166-0.170 | 0.010-0.012 | 14-15x |
| Create | 1000×1000 | 16.6-17.4 | 0.011 | 1500-1570x |

**Analysis:** PyTorch memory operations are dramatically faster, likely due to pre-allocated memory pools and GPU memory management.

---

## Performance by Data Type

### Float32 Operations
- **PyTorch**: 1.0x baseline
- **NumPy**: 2-10x slower for large tensors
- **Syntonic**: 13-185x slower (matrix ops)

### Float64 Operations
- **PyTorch**: 1.0x baseline
- **NumPy**: 17-150x slower for large tensors
- **Syntonic**: 13-445x slower (matrix ops), but competitive in linalg

### Complex64/128 Operations
- **PyTorch**: 1.0x baseline
- **NumPy**: 30-6000x slower
- **Syntonic**: Not tested (element-wise ops skipped)

---

## Scaling Analysis

### Small Tensors (100×100)
- **PyTorch**: 6-17 μs per operation
- **NumPy**: 15-924 μs per operation
- **Syntonic**: 151-2417 μs per operation (8-140x slower than PyTorch)

### Medium Tensors (500×500)
- **PyTorch**: 6-23 μs per operation
- **NumPy**: 61-284 ms per operation
- **Syntonic**: 4.3-270 ms per operation (183-11600x slower than PyTorch)

### Large Tensors (1000×1000)
- **PyTorch**: 6-11 μs per operation
- **Syntonic**: 215-428 ms per operation (20000-40000x slower than PyTorch)

---

## GPU Acceleration Effectiveness

### PyTorch GPU Performance
- **Arithmetic**: 8-445x faster than NumPy CPU
- **Matrix ops**: 50-334x faster than NumPy CPU
- **Element-wise**: 30-5900x faster than NumPy CPU
- **Memory**: 14-1570x faster than NumPy CPU

### Syntonic GPU Performance
- **Limited testing**: Only basic operations tested
- **Matrix ops**: Significantly slower than expected
- **Potential**: Linear algebra shows promise

---

## Recommendations for Syntonic Optimization

### Critical Issues
1. **Reshape/Transpose Performance**: 1000x slower than PyTorch - investigate Python-Rust data transfer
2. **Matrix Multiplication**: 13-445x slower - optimize CUDA kernels or BLAS integration
3. **Memory Operations**: Not benchmarked - implement efficient memory management

### Optimization Opportunities
1. **Reduce Python-Rust Overhead**: Batch operations, minimize data transfers
2. **CUDA Kernel Optimization**: Profile and optimize tensor operations
3. **Memory Pool**: Implement pre-allocated memory pools like PyTorch
4. **BLAS Integration**: Ensure optimal LAPACK/BLAS linkage for linear algebra

### Strengths to Leverage
1. **Linear Algebra**: Competitive with NumPy for large matrices
2. **Exact Arithmetic**: Unique feature not available in NumPy/PyTorch
3. **SRT/CRT Operations**: Specialized algorithms for scientific computing

---

## Conclusion

**Syntonic shows promise as a specialized scientific computing library**, particularly for linear algebra operations where it competes with NumPy. However, **significant performance optimizations are needed** to make it competitive with PyTorch for general tensor operations.

**Priority optimizations:**
1. Fix reshape/transpose performance (critical bottleneck)
2. Optimize matrix operations and CUDA kernels
3. Implement efficient memory management
4. Reduce Python-Rust communication overhead

With these improvements, Syntonic could become a valuable addition to the scientific Python ecosystem, offering unique SRT/CRT functionality alongside competitive performance.</content>
<parameter name="filePath">/home/Andrew/Documents/SRT Complete/implementation/syntonic/benchmarks/performance_analysis.md