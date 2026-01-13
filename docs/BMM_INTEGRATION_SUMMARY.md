## **BMM Integration & ResonantTensor Updates**

This document summarizes the changes and additions made while integrating the new Batched Matrix Multiplication (BMM) path into the Resonant Engine. The edits span both Rust and Python layers and include a small verification test. Each file below lists what was changed or added and why.

- **File:** python/syntonic/nn/architectures/syntonic_attention_pure.py
  - **Change:** Refactored the attention code to use the Rust-backed BMM/matmul path instead of explicit Python loops.
  - **Details:** Replaced per-head and per-batch loops with calls to `ResonantTensor.matmul` so the heavy-lifting (Q @ K^T and Attn @ V) is dispatched to the Rust backend. This simplifies the multi-head code and enables GPU-accelerated BMM when available.
  - **Why:** Cleaner code and improved performance by using the optimized Rust/CUDA kernels for attention score computation and value aggregation.

- **File:** python/syntonic/nn/resonant_tensor.py
  - **Change:** Updated the Python wrapper to expose and use the new Rust `zeros` factory and to continue delegating `matmul` calls to the Rust core.
  - **Details:** `ResonantTensor.zeros` now calls into `_core.ResonantTensor.zeros(...)` (the Rust-backed constructor that creates exact `GoldenExact::zero()` lattice values). The `matmul` method continues to call the underlying Rust `matmul` implementation.
  - **Why:** Ensure zero tensors are constructed exactly (using GoldenExact) and to route matmul/BMM to the high-performance backend.

- **File:** rust/kernels/matmul.cu
  - **Change / Role:** The CUDA kernel implementing matrix-multiply and batched matmul is available and is used by the Rust bindings (specifically the `cuda_matmul_nt_f64` and `cuda_bmm_nt_f64` kernels).
  - **Details:** The Rust-side dispatch calls `cuda_bmm_nt_f64` for true BMM (Rank 3+ inputs) and `cuda_matmul_nt_f64` for standard 2D matmul/broadcasted matmul. The kernels provide the GPU-accelerated numerical path for the D-phase / flux computations.
  - **Why:** Provide high-throughput BLAS-style operations on GPU for attention and other batched linear algebra workloads.

- **File:** rust/src/resonant/tensor.rs
  - **Change:** Major updates to `matmul_core`, `zeros` factory exposure, and PyO3 wrappers.
  - **Details:**
    - Restored a robust `zeros(shape, precision)` factory which constructs a crystallized `ResonantTensor` filled with `GoldenExact::zero()`.
    - Reworked `matmul_core` dispatch logic to correctly detect batch dimensions and choose between:
      - strict BMM path for Rank>=3 (calls `cuda_bmm_nt_f64` on CUDA path or runs exact GoldenExact loops on CPU fallback), and
      - standard/broadcast matmul for Rank 2 or Rank3 x Rank2 broadcasted cases (calls `cuda_matmul_nt_f64` on CUDA path or exact Q(φ) computation on CPU fallback).
    - Added a PyO3 static wrapper named `py_zeros` (exposed as `zeros(...)` to Python) to avoid symbol conflicts while exposing the Rust factory to Python.
  - **Why:** Provide correct batched matmul semantics, ensure exact-zero construction for lattice-based tensors, and allow the Python layer to consume these features safely.

- **File:** rust/src/tensor/srt_kernels.rs
  - **Change / Role:** Kernel bindings and helpers used by `tensor.rs` and `lib.rs` were used to call `cuda_bmm_nt_f64` and related functions.
  - **Details:** The module exposes PHI constants and kernel entry points used by the ResonantTensor (e.g., `ensure_srt_kernels_loaded`, `cuda_matmul_nt_f64`, `cuda_bmm_nt_f64`, `cuda_resonant_d_phase_f64`, etc.). The Rust `matmul_core` relies on these bindings for CUDA-accelerated execution.
  - **Why:** Centralize kernel loading and provide safe Rust bindings to the CUDA implementations used for matmul and BMM.

- **File:** rust/src/lib.rs
  - **Change:** Confirmed/ensured that the new tensor functions/classes (including `ResonantTensor` and matmul variants) are registered and available to the Python module `syntonic._core`.
  - **Details:** The module exposes the `ResonantTensor` class and the various linear algebra functions (e.g., `py_mm`, `py_bmm`, `py_mm_nt`, etc.) so Python can call into the optimized Rust/CUDA code.
  - **Why:** Make the Rust-backed API available to Python code (including `ResonantTensor.zeros`, `matmul`, and kernel helpers).

- **File:** tests/test_bmm_verification.py
  - **Change:** Added/updated a small verification test that exercises batched matmul behavior.
  - **Details:** The test constructs small example tensors (batched and unbatched), runs `matmul` on them, and checks shape and basic numeric consistency between CPU GoldenExact fallback and the exposed matmul API. It serves as a guard ensuring the BMM path produces the expected shapes and that the Python wrapper consumes the Rust API correctly.
  - **Why:** Prevent regressions in the matmul/BMM dispatch and validate Python↔Rust integration.

---

If you want, I can also:

- Run the BMM verification test locally and share the output.
- Add a short code snippet to README showing example usage of `ResonantTensor.zeros(...)` and `matmul` for attention.

File created: `/docs/BMM_INTEGRATION_SUMMARY.md`
