# Syntonic Test Results Summary

**Test Run Date:** January 17, 2026  
**Total Tests:** 820 collected  
**Passed:** 744  
**Failed:** 75  
**Skipped:** 1  
**Warnings:** 5  
**Errors:** 1 (collection error)  

## Collection Errors

### 1. Import Error in `tests/test_benchmarks/test_convergence.py`
- **Error:** `ModuleNotFoundError: No module named 'syntonic.benchmarks.datasets'`
- **Impact:** Test collection failed, preventing this test from running
- **Root Cause:** Missing module `syntonic.benchmarks.datasets` (only `__pycache__` exists in benchmarks directory)

## Test Failures

### CUDA/GPU Related Failures (8 failures)

1. **State CUDA Transfer Failures** (4 failures)
   - `tests/test_core/test_state.py::TestStateDeviceOperations::test_cuda_transfer`
   - `tests/test_core/test_state.py::TestStateDeviceOperations::test_cuda_large_tensor`
   - `tests/test_core/test_state.py::TestStateDeviceOperations::test_cuda_complex`
   - `tests/test_core/test_state.py::TestStateDeviceOperations::test_cuda_2d`
   - **Error:** `ValueError: ShapeError/IncompatibleShape: incompatible shapes`
   - **Issue:** CUDA device transfers are failing with shape incompatibility errors

2. **Syntonic Softmax GPU Failures** (3 failures)
   - `tests/test_core/test_syntonic_softmax_phase2.py::TestF32Support::test_f32_identity_mode_gpu`
     - **Error:** Sum normalization failure (1.013 vs 1.0)
   - `tests/test_core/test_syntonic_softmax_phase2.py::TestGPUIdentityMode::test_gpu_identity_matches_cpu`
     - **Error:** GPU/CPU results don't match within tolerance
   - `tests/test_core/test_syntonic_softmax_phase2.py::TestPerformance::test_gpu_performance_comparison`
     - **Error:** `RuntimeError: mode_norms must be on GPU (F32) for CUDA softmax`

3. **Kernel Compilation Failure** (1 failure)
   - `tests/test_kernel_compilation.py::test_compilation_consistency`
   - **Error:** Missing PTX files for newer GPU architectures (sm75, sm80, sm86, sm90)
   - **Missing files:** `hierarchy_sm80.ptx`, `golden_gelu_sm86.ptx`, etc.

### Neural Network Failures (8 failures)

1. **Prime Syntony Gate Issues** (3 failures)
   - `tests/test_neural_networks.py::TestPrimeSyntonyGate::test_gate_creation`
     - **Error:** Missing `prime` attribute
   - `tests/test_neural_networks.py::TestPrimeSyntonyGate::test_gate_prime_constraint`
     - **Error:** Expected exception not raised for invalid prime constraint
   - `tests/test_srt/test_neural_networks.py::TestPrimeSyntonyGate::test_gate_forward_pass`
     - **Error:** Output norms don't match expected values (29.034 vs 1.0)

2. **Winding Attention Issues** (3 failures)
   - `tests/test_neural_networks.py::TestWindingAttention::test_attention_creation`
     - **Error:** Missing `dim` attribute
   - `tests/test_neural_networks.py::TestWindingAttention::test_attention_forward_pass`
   - `tests/test_neural_networks.py::TestWindingAttention::test_attention_gradient_flow`
     - **Error:** `RuntimeError: shape '[2, 10, 1, 7]' is invalid for input of size 210`

3. **SRT Transformer Issues** (2 failures)
   - `tests/test_neural_networks.py::TestSRTTransformerBlock::test_transformer_creation`
     - **Error:** Missing `gate` attribute
   - `tests/test_neural_networks.py::TestDimensionUtilities::test_non_prime_rejection`
     - **Error:** Expected exception not raised for non-prime dimensions

### Physics Engine Failures (25 failures)

1. **SRT Physics Engine Method Missing** (4 failures)
   - Multiple tests failing with `AttributeError: 'SRTPhysicsEngine' object has no attribute 'get_fundamental_forces'`
   - Missing methods: `get_fundamental_forces`, `get_stable_matter_generations`, `predict_dark_matter_mass`

2. **Simulator Method Missing** (17 failures)
   - ForceSimulator: missing `get_forces`, `is_valid_force`, `get_force_by_name`
   - MatterSimulator: missing `get_generations`, `is_generation_stable`, `is_mersenne_prime`, `get_generation_particles`
   - DarkSectorSimulator: missing `get_dark_matter_candidates`, `get_lucas_boost_factor`, `get_lucas_gaps`, `get_primary_dark_matter`
   - ConsciousnessSimulator: missing `get_transcendence_gates`, `compute_gamma_alignment`, `get_consciousness_threshold`, `compute_synchrony_measure`

3. **Physics Integration Failures** (4 failures)
   - Missing methods: `run_universe_simulation`, `validate_predictions`, `compute_theoretical_coherence`, `compare_to_standard_model`

### Prime Theory Failures (5 failures)

1. **Incorrect Prime Classifications** (3 failures)
   - `tests/test_prime_theory.py::TestPrimeTheory::test_mersenne_primes`
     - **Error:** 5 incorrectly classified as Mersenne prime
   - `tests/test_prime_theory.py::TestPrimeTheory::test_fermat_primes`
     - **Error:** 5 incorrectly classified as Fermat prime
   - `tests/test_prime_theory.py::TestPrimeTheory::test_stability_checks`
     - **Error:** Prime 13 incorrectly classified as stable

2. **Lucas Number Error** (1 failure)
   - `tests/test_prime_theory.py::TestPrimeTheory::test_lucas_numbers`
     - **Error:** L(1) = 1, expected 2

3. **Dark Matter Prediction Error** (1 failure)
   - `tests/test_prime_theory.py::TestPrimeTheory::test_dark_matter_prediction`
     - **Error:** Mass 857 GeV outside expected range 1100-1500 GeV

### Resonant Tensor Failures (7 failures)

1. **Length Method Issues** (5 failures)
   - `tests/test_resonant/test_tensor_wrapper.py::TestConstruction::*`
   - `tests/test_resonant/test_tensor_wrapper.py::TestOperatorOverloading::test_len`
   - **Error:** `len()` returns first dimension size instead of total elements

2. **Matmul Operator Failure** (1 failure)
   - `tests/test_resonant/test_tensor_wrapper.py::TestOperatorOverloading::test_matmul`
   - **Error:** `RuntimeError: Shape mismatch: Inner dimension mismatch: input ...10 vs weights ...20`

3. **Missing Method** (1 failure)
   - `tests/test_resonant/test_tensor_wrapper.py::TestPhaseCycles::test_wake_and_crystallize`
   - **Error:** Missing `crystallize` method

### Standard Model Physics Failures (3 failures)

1. **Mass Prediction Errors** (3 failures)
   - `tests/test_physics/test_standard_model.py::TestFermionMasses::test_quark_masses`
     - **Error:** u-quark mass prediction error > 0.01
   - `tests/test_physics/test_standard_model.py::TestHadronMasses::test_pion_mass`
     - **Error:** Pion mass prediction error > 0.01
   - `tests/test_physics/test_standard_model.py::TestHadronMasses::test_meson_pattern`
     - **Error:** Meson pattern prediction error > 0.01

### SRT Physics Validation Failure (1 failure)

- `tests/test_physics/test_srt_physics.py::TestSRTPhysicsEngine::test_prediction_validation`
  - **Error:** Prediction "gamma_synchrony" failed validation

### Memory Benchmark Failure (1 failure)

- `tests/test_srt_physics.py::TestPhysicsBenchmarks::test_memory_usage`
  - **Error:** `ModuleNotFoundError: No module named 'psutil'`

## Warnings

### UserWarnings (3 warnings)
- `tests/test_neural_networks.py::TestDimensionUtilities::test_prime_dimensions`
  - **Warning:** `embed_dim=5 is not a Mersenne prime. Consider using: 3, 7, 31, 127 for stability.`
  - **Warning:** `embed_dim=11 is not a Mersenne prime. Consider using: 3, 7, 31, 127 for stability.`
  - **Warning:** `embed_dim=13 is not a Mersenne prime. Consider using: 3, 7, 31, 127 for stability.`

### PytestReturnNotNoneWarning (2 warnings)
- `tests/test_resonant_matmul.py::test_resonant_matmul`
  - **Warning:** Test function returned `bool` instead of `None`
- `tests/test_versal_grip.py::test_versal_grip_strength`
  - **Warning:** Test function returned `bool` instead of `None`

## Summary

### Critical Issues
1. **CUDA/GPU functionality broken** - Shape errors in device transfers, GPU/CPU mismatches
2. **Missing physics engine methods** - Core SRT physics simulation methods not implemented
3. **Neural network layer issues** - Prime syntony gates and attention mechanisms failing
4. **Prime theory implementation errors** - Incorrect prime classifications and sequences

### Moderate Issues
1. **Resonant tensor API inconsistencies** - `len()` method behavior, missing methods
2. **Standard model predictions inaccurate** - Mass predictions outside acceptable error bounds
3. **Missing dependencies** - `psutil` not available for memory benchmarks

### Minor Issues
1. **Test style warnings** - Some tests return values instead of using assertions
2. **User warnings for non-optimal dimensions** - Expected warnings about prime dimensions

### Test Coverage
- **Coverage:** 42% overall
- **Core modules well-tested** (state, device, dtype, CRT, exact sequences)
- **NN modules poorly tested** (many 0% coverage in architectures, benchmarks)
- **Physics modules moderately tested** (9-68% coverage)

### Recommendations
1. Fix CUDA device transfer shape compatibility issues
2. Implement missing physics engine methods
3. Correct prime theory algorithms and classifications
4. Standardize ResonantTensor API (especially `len()` method)
5. Improve neural network layer implementations
6. Add missing dependencies or make them optional
7. Update test assertions to not return values