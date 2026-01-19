# ResonantTensor Upgrade Implementation Plan

## Executive Summary
This document outlines the roadmap for upgrading `syntonic.nn.ResonantTensor` to achieve feature parity with modern tensor libraries while maintaining SRT's exact arithmetic and golden resonance principles. The focus is on eliminating performance bottlenecks (Python fallbacks) and filling critical gaps in the operator set.

## 1. Direct GPU Initialization
**Current Status**: incomplete (`# TODO` in `__init__`).
**Issue**: Tensors are initialized on CPU and copied to GPU.
**Plan**:
1.  **Rust Backend**: Implement `new_cuda(data, shape)` or `from_host_cuda` in `lib.rs` / `tensor.rs` that allocates directly on device.
2.  **Java/Python Binding**: Expose this constructor.
3.  **Python Wrapper**: Update `ResonantTensor.__init__` to detect `device='cuda'` and call the direct allocator.

## 2. Structural Operations (Native Backend)
**Current Status**: Reliant on slow Python `itertools` fallbacks for `transpose`, `permute`, `view`, `index_select`.
**Issue**: backend methods are missing/unbound, causing massive slowdowns on reshaping.
**Plan**:
1.  **Rust Backend**: Ensure `permute`, `transpose`, `reshape` (view) are implemented in `src/tensor.rs` using stride manipulation (zero-copy where possible).
2.  **Validation**: Verify `index_select` handles bounds checking in Rust.
3.  **Python Wrapper**: Remove `try...except AttributeError` blocks; calls should succeed on backend or raise valid errors.

## 3. Mathematical Completeness
**Current Status**: Missing basic trig/pow ops.
**Issue**: Operations like `sin`, `cos`, `pow`, `abs`, `sqrt` are absent.
**Plan**:
1.  **Rust Backend**: Implement element-wise `sin`, `cos`, `pow`, `sqrt` leveraging `GoldenExact` approximations or lifting to float/flux for non-exact ops.
2.  **API**: Expose as methods `tensor.sin()`.
3.  **Python Wrapper**: Add corresponding methods to `ResonantTensor` class.

## 4. Reduction Operations
**Current Status**: Only `mean`, `var` exist.
**Issue**: Missing `sum`, `max`, `min`, `prod`, `norm` (standard p-norm).
**Plan**:
1.  **Rust Backend**: Implement reduction kernels (`sum`, `max`...) accepting `dim` argument.
2.  **Python Wrapper**: Implement `sum(dim=...)` wrappers.
3.  **Special Handling**: `sum` should support keeping explicit lattice precision vs float fallback.

## 5. Advanced Indexing
**Current Status**: Rudimentary.
**Issue**: No boolean masking (`x[x>0]`) or complex slicing.
**Plan**:
1.  **Backend**: Implement `masked_select` (boolean mask) and `scatter` / `gather`.
2.  **Python Wrapper**: Upgrade `__getitem__` to handle:
    -   Boolean tensors (masks).
    -   `None` (newaxis).
    -   `Ellipsis` (...).

## 6. Comparison Operators
**Current Status**: Missing.
**Issue**: Cannot compare tensors.
**Plan**:
1.  **Backend**: Implement `eq`, `ne`, `lt`, `le`, `gt`, `ge` returning a Boolean/Byte Tensor.
2.  **Python Wrapper**: Implement `__eq__`, `__lt__` dunder methods.

## 7. Immediate Fixes (Bug Report)
**Detected Issue**: `IndexError` in `syntonic_attention.py` when handling 1D tensors (e.g. during specific `gnostic_ouroboros` injections).
**Plan**:
- Update `syntonic_attention.py` to robustly handle 1D inputs (view as [1, D] or [L, D]).
