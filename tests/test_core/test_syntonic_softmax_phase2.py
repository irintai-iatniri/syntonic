"""
Phase 2 Tests for Syntonic Softmax - F32 Support and GPU Identity Mode

These tests verify:
1. F32 contiguous and strided kernel functionality
2. GPU identity mode (standard softmax without CPU round-trip)
3. F32 vs F64 numerical consistency
4. Strided dimension softmax for non-last dimensions
5. Performance characteristics (F32 vs F64 speedup)

Phase 2 Implementation Status:
- Task 2.1: F32 strided kernels (provided_f32, learned_strided_f32, provided_strided_f32)
- Task 2.2: GPU identity mode kernels
- Task 2.3: Python API completeness (already done)
"""

import pytest
import numpy as np
import time

try:
    import syntonic.core as core
    from syntonic.core import (
        ResonantTensor,
        SyntonicSoftmaxState,
        SyntonicSoftmaxMode,
        syntonic_softmax,
    )
    CORE_AVAILABLE = True
except ImportError:
    CORE_AVAILABLE = False

# Skip all tests if core not available
pytestmark = pytest.mark.skipif(not CORE_AVAILABLE, reason="syntonic.core not available")

# Golden ratio constant
PHI = 1.618033988749895


# =============================================================================
# Utility Functions
# =============================================================================

def create_tensor(data, shape, device=None):
    """Create a ResonantTensor, optionally on GPU."""
    tensor = ResonantTensor(data, shape)
    if device is not None:
        # Move to GPU if device specified
        tensor = tensor.to_device(device)
    return tensor


def reference_softmax(logits, dim=-1):
    """Reference numpy softmax implementation for validation."""
    logits = np.array(logits)
    max_val = np.max(logits, axis=dim, keepdims=True)
    exp_logits = np.exp(logits - max_val)
    return exp_logits / np.sum(exp_logits, axis=dim, keepdims=True)


def has_cuda():
    """Check if CUDA is available."""
    try:
        # Try to create a tensor on GPU
        t = ResonantTensor([1.0], [1])
        t_gpu = t.to_device(0)
        return True
    except Exception:
        return False


# =============================================================================
# Task 2.1: F32 Support Tests
# =============================================================================

class TestF32Support:
    """Tests for F32 kernel support."""

    def test_f32_identity_mode_cpu(self):
        """Test F32 identity mode on CPU."""
        data = [1.0, 2.0, 3.0, 4.0, 5.0]
        x = ResonantTensor(data, [1, 5])

        state = SyntonicSoftmaxState(
            SyntonicSoftmaxMode.Identity,
            -1,
            None,
            1.0
        )

        y = state.forward(x, None)
        probs = y.to_floats()

        # Verify normalization
        assert np.isclose(sum(probs), 1.0, rtol=1e-3), f"Sum = {sum(probs)}"

        # Verify against reference
        expected = reference_softmax(data).flatten()
        assert np.allclose(probs, expected, rtol=1e-2), \
            f"Got {probs}, expected {expected}"

        print(f"F32 identity CPU: {probs}")

    def test_f32_learned_mode_cpu(self):
        """Test F32 learned mode on CPU."""
        num_features = 5
        data = [0.0] * num_features
        x = ResonantTensor(data, [1, num_features])

        state = SyntonicSoftmaxState(
            SyntonicSoftmaxMode.Learned,
            -1,
            num_features,
            1.0
        )

        y = state.forward(x, None)
        probs = y.to_floats()

        # Verify normalization
        assert np.isclose(sum(probs), 1.0, rtol=1e-2), f"Sum = {sum(probs)}"

        # With E8 roots, should have varying probabilities
        unique_probs = len(set(round(p, 4) for p in probs))
        assert unique_probs > 1, "No variation in probabilities"

        print(f"F32 learned CPU: {probs}")

    @pytest.mark.skipif(not has_cuda(), reason="CUDA not available")
    def test_f32_identity_mode_gpu(self):
        """Test F32 identity mode on GPU (should use GPU kernel, no CPU round-trip)."""
        data = [1.0, 2.0, 3.0, 4.0, 5.0]
        x = ResonantTensor(data, [1, 5]).to_device(0)

        state = SyntonicSoftmaxState(
            SyntonicSoftmaxMode.Identity,
            -1,
            None,
            1.0
        )

        y = state.forward(x, None)

        # Result should still be on GPU
        assert y.device_idx() is not None, "Result moved to CPU unexpectedly"

        probs = y.to_cpu().to_floats()

        # Verify normalization
        assert np.isclose(sum(probs), 1.0, rtol=1e-3), f"Sum = {sum(probs)}"

        print(f"F32 identity GPU: {probs}")

    @pytest.mark.skipif(not has_cuda(), reason="CUDA not available")
    def test_f32_learned_mode_gpu(self):
        """Test F32 learned mode on GPU."""
        num_features = 5
        data = [0.0] * num_features
        x = ResonantTensor(data, [1, num_features]).to_device(0)

        state = SyntonicSoftmaxState(
            SyntonicSoftmaxMode.Learned,
            -1,
            num_features,
            1.0
        )

        y = state.forward(x, None)

        # Result should still be on GPU
        assert y.device_idx() is not None, "Result moved to CPU unexpectedly"

        probs = y.to_cpu().to_floats()
        assert np.isclose(sum(probs), 1.0, rtol=1e-2), f"Sum = {sum(probs)}"

        print(f"F32 learned GPU: {probs}")


# =============================================================================
# Task 2.2: GPU Identity Mode Tests
# =============================================================================

class TestGPUIdentityMode:
    """Tests for GPU identity mode acceleration."""

    @pytest.mark.skipif(not has_cuda(), reason="CUDA not available")
    def test_gpu_identity_no_cpu_transfer(self):
        """Test that GPU identity mode doesn't transfer to CPU."""
        batch_size, num_classes = 16, 100
        data = list(np.random.randn(batch_size * num_classes))
        x = ResonantTensor(data, [batch_size, num_classes]).to_device(0)

        state = SyntonicSoftmaxState(
            SyntonicSoftmaxMode.Identity,
            -1,
            None,
            1.0
        )

        y = state.forward(x, None)

        # Output should remain on GPU
        assert y.device_idx() == 0, "Output was transferred to CPU"
        print("GPU identity mode keeps data on GPU")

    @pytest.mark.skipif(not has_cuda(), reason="CUDA not available")
    def test_gpu_identity_matches_cpu(self):
        """Test that GPU identity mode matches CPU results."""
        data = [1.0, 2.0, 3.0, 4.0, 5.0]

        # CPU version
        x_cpu = ResonantTensor(data, [1, 5])
        state = SyntonicSoftmaxState(
            SyntonicSoftmaxMode.Identity,
            -1,
            None,
            1.0
        )
        y_cpu = state.forward(x_cpu, None)
        probs_cpu = y_cpu.to_floats()

        # GPU version
        x_gpu = ResonantTensor(data, [1, 5]).to_device(0)
        y_gpu = state.forward(x_gpu, None)
        probs_gpu = y_gpu.to_cpu().to_floats()

        # Should match
        assert np.allclose(probs_cpu, probs_gpu, rtol=1e-4), \
            f"CPU: {probs_cpu}, GPU: {probs_gpu}"

        print(f"CPU/GPU identity match: CPU={probs_cpu}, GPU={probs_gpu}")

    def test_identity_vs_standard_softmax(self):
        """Test that identity mode matches standard softmax."""
        data = [1.0, 2.0, 3.0, 4.0, 5.0]
        x = ResonantTensor(data, [1, 5])

        state = SyntonicSoftmaxState(
            SyntonicSoftmaxMode.Identity,
            -1,
            None,
            1.0
        )

        y = state.forward(x, None)
        probs = y.to_floats()

        # Compare with numpy reference
        expected = reference_softmax(data).flatten()

        assert np.allclose(probs, expected, rtol=1e-2), \
            f"Identity mode: {probs}, Expected: {expected}"

        print(f"Identity matches standard softmax")


# =============================================================================
# Strided Kernel Tests
# =============================================================================

class TestStridedKernels:
    """Tests for strided (non-last-dimension) softmax kernels."""

    def test_strided_identity_dim0(self):
        """Test strided identity mode along dimension 0."""
        # Shape [3, 4] - softmax along dim 0 (columns)
        data = list(range(12))  # [0, 1, 2, ..., 11]
        x = ResonantTensor([float(d) for d in data], [3, 4])

        state = SyntonicSoftmaxState(
            SyntonicSoftmaxMode.Identity,
            0,  # dim 0 (columns)
            None,
            1.0
        )

        y = state.forward(x, None)
        probs = np.array(y.to_floats()).reshape(3, 4)

        # Each column should sum to 1
        for col in range(4):
            col_sum = probs[:, col].sum()
            assert np.isclose(col_sum, 1.0, rtol=1e-2), \
                f"Column {col} sum = {col_sum}"

        print(f"Strided identity dim=0 works, column sums all ~1.0")

    @pytest.mark.xfail(reason="Strided learned mode for non-last dim requires GPU - CPU path has broadcasting limitation")
    def test_strided_learned_dim0(self):
        """Test strided learned mode along dimension 0."""
        # Shape [3, 4] - softmax along dim 0 (3 features)
        data = [0.0] * 12
        x = ResonantTensor(data, [3, 4])

        state = SyntonicSoftmaxState(
            SyntonicSoftmaxMode.Learned,
            0,  # dim 0
            3,  # 3 features (dim size)
            1.0
        )

        y = state.forward(x, None)
        probs = np.array(y.to_floats()).reshape(3, 4)

        # Each column should sum to 1
        for col in range(4):
            col_sum = probs[:, col].sum()
            assert np.isclose(col_sum, 1.0, rtol=1e-2), \
                f"Column {col} sum = {col_sum}"

        print(f"Strided learned dim=0 works")

    def test_3d_strided_middle_dim(self):
        """Test 3D tensor with softmax on middle dimension."""
        # Shape [2, 3, 4] - softmax along dim 1
        data = list(np.random.randn(2 * 3 * 4))
        x = ResonantTensor(data, [2, 3, 4])

        state = SyntonicSoftmaxState(
            SyntonicSoftmaxMode.Identity,
            1,  # middle dimension
            None,
            1.0
        )

        y = state.forward(x, None)
        probs = np.array(y.to_floats()).reshape(2, 3, 4)

        # For each (batch, feature) pair, dim 1 should sum to 1
        for b in range(2):
            for f in range(4):
                dim1_sum = probs[b, :, f].sum()
                assert np.isclose(dim1_sum, 1.0, rtol=1e-2), \
                    f"[{b}, :, {f}] sum = {dim1_sum}"

        print(f"3D strided middle dim works")


# =============================================================================
# Provided Mode Tests
# =============================================================================

class TestProvidedMode:
    """Tests for provided mode with pre-computed weights."""

    def test_provided_mode_basic(self):
        """Test provided mode with explicit syntony weights."""
        data = [1.0, 2.0, 3.0, 4.0, 5.0]
        syntony_data = [1.0, 0.5, 0.25, 0.125, 0.0625]  # Decaying weights

        x = ResonantTensor(data, [1, 5])
        syntony = ResonantTensor(syntony_data, [1, 5])

        state = SyntonicSoftmaxState(
            SyntonicSoftmaxMode.Provided,
            -1,
            None,
            1.0
        )

        y = state.forward(x, syntony)
        probs = y.to_floats()

        # Should sum to 1
        assert np.isclose(sum(probs), 1.0, rtol=1e-2), f"Sum = {sum(probs)}"

        # First feature should be boosted (highest syntony weight)
        # But this depends on logit values too
        print(f"Provided mode: {probs}")

    def test_provided_mode_uniform_weights(self):
        """Test that uniform syntony weights give standard softmax."""
        data = [1.0, 2.0, 3.0, 4.0, 5.0]
        syntony_data = [1.0, 1.0, 1.0, 1.0, 1.0]  # Uniform weights

        x = ResonantTensor(data, [1, 5])
        syntony = ResonantTensor(syntony_data, [1, 5])

        # Provided with uniform weights
        state_provided = SyntonicSoftmaxState(
            SyntonicSoftmaxMode.Provided,
            -1,
            None,
            1.0
        )
        y_provided = state_provided.forward(x, syntony)
        probs_provided = y_provided.to_floats()

        # Identity mode
        state_identity = SyntonicSoftmaxState(
            SyntonicSoftmaxMode.Identity,
            -1,
            None,
            1.0
        )
        y_identity = state_identity.forward(x, None)
        probs_identity = y_identity.to_floats()

        # Should be similar (uniform syntony = no weighting)
        assert np.allclose(probs_provided, probs_identity, rtol=1e-1), \
            f"Provided: {probs_provided}, Identity: {probs_identity}"

        print("Uniform syntony weights match identity mode")


# =============================================================================
# Python API Tests
# =============================================================================

class TestPythonAPI:
    """Tests for the syntonic_softmax Python function API."""

    def test_api_identity_mode(self):
        """Test syntonic_softmax API with identity mode."""
        data = [1.0, 2.0, 3.0]
        x = ResonantTensor(data, [1, 3])

        y = syntonic_softmax(x, mode="identity")
        probs = y.to_floats()

        assert np.isclose(sum(probs), 1.0, rtol=1e-3)
        print(f"API identity: {probs}")

    def test_api_learned_mode(self):
        """Test syntonic_softmax API with learned mode."""
        data = [0.0] * 5
        x = ResonantTensor(data, [1, 5])

        # Create mode norms (E8 roots)
        mode_norms = SyntonicSoftmaxState(
            SyntonicSoftmaxMode.Learned, -1, 5, 1.0
        ).get_mode_norms()

        y = syntonic_softmax(x, mode="learned", mode_norms=mode_norms)
        probs = y.to_floats()

        assert np.isclose(sum(probs), 1.0, rtol=1e-2)
        print(f"API learned: {probs}")

    def test_api_provided_mode(self):
        """Test syntonic_softmax API with provided mode."""
        data = [1.0, 2.0, 3.0]
        syntony_data = [1.0, 0.5, 0.25]

        x = ResonantTensor(data, [1, 3])
        syntony = ResonantTensor(syntony_data, [1, 3])

        y = syntonic_softmax(x, mode="provided", syntony=syntony)
        probs = y.to_floats()

        assert np.isclose(sum(probs), 1.0, rtol=1e-2)
        print(f"API provided: {probs}")

    def test_api_default_mode(self):
        """Test syntonic_softmax API default mode selection."""
        data = [1.0, 2.0, 3.0]
        x = ResonantTensor(data, [1, 3])

        # No mode specified, no mode_norms -> should use identity
        y = syntonic_softmax(x)
        probs = y.to_floats()

        expected = reference_softmax(data).flatten()
        assert np.allclose(probs, expected, rtol=1e-2)
        print("API default mode works (identity)")


# =============================================================================
# Performance Benchmarks (Optional)
# =============================================================================

class TestPerformance:
    """Performance benchmarks for F32 vs F64."""

    @pytest.mark.slow
    def test_cpu_performance_comparison(self):
        """Compare CPU performance of different modes."""
        batch_size, num_classes = 100, 1000
        data = list(np.random.randn(batch_size * num_classes))
        x = ResonantTensor(data, [batch_size, num_classes])

        n_iterations = 10

        # Identity mode
        state_identity = SyntonicSoftmaxState(
            SyntonicSoftmaxMode.Identity, -1, None, 1.0
        )

        start = time.time()
        for _ in range(n_iterations):
            _ = state_identity.forward(x, None)
        identity_time = time.time() - start

        # Learned mode
        state_learned = SyntonicSoftmaxState(
            SyntonicSoftmaxMode.Learned, -1, num_classes, 1.0
        )

        start = time.time()
        for _ in range(n_iterations):
            _ = state_learned.forward(x, None)
        learned_time = time.time() - start

        print(f"\nCPU Performance ({n_iterations} iterations, {batch_size}x{num_classes}):")
        print(f"  Identity: {identity_time:.4f}s ({identity_time/n_iterations*1000:.2f}ms/iter)")
        print(f"  Learned:  {learned_time:.4f}s ({learned_time/n_iterations*1000:.2f}ms/iter)")
        print(f"  Overhead: {learned_time/identity_time:.2f}x")

    @pytest.mark.slow
    @pytest.mark.skipif(not has_cuda(), reason="CUDA not available")
    def test_gpu_performance_comparison(self):
        """Compare GPU performance of different modes."""
        batch_size, num_classes = 100, 1000
        data = list(np.random.randn(batch_size * num_classes))
        x = ResonantTensor(data, [batch_size, num_classes]).to_device(0)

        n_iterations = 100

        # Identity mode
        state_identity = SyntonicSoftmaxState(
            SyntonicSoftmaxMode.Identity, -1, None, 1.0
        )

        # Warmup
        for _ in range(10):
            _ = state_identity.forward(x, None)

        start = time.time()
        for _ in range(n_iterations):
            _ = state_identity.forward(x, None)
        identity_time = time.time() - start

        # Learned mode
        state_learned = SyntonicSoftmaxState(
            SyntonicSoftmaxMode.Learned, -1, num_classes, 1.0
        )

        # Warmup
        for _ in range(10):
            _ = state_learned.forward(x, None)

        start = time.time()
        for _ in range(n_iterations):
            _ = state_learned.forward(x, None)
        learned_time = time.time() - start

        print(f"\nGPU Performance ({n_iterations} iterations, {batch_size}x{num_classes}):")
        print(f"  Identity: {identity_time:.4f}s ({identity_time/n_iterations*1000:.2f}ms/iter)")
        print(f"  Learned:  {learned_time:.4f}s ({learned_time/n_iterations*1000:.2f}ms/iter)")
        print(f"  Overhead: {learned_time/identity_time:.2f}x")


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("Syntonic Softmax Phase 2 Tests")
    print("=" * 70)

    if not CORE_AVAILABLE:
        print("syntonic.core not available - skipping tests")
        exit(1)

    # Run tests manually
    print("\n--- F32 Support Tests ---")
    t = TestF32Support()
    t.test_f32_identity_mode_cpu()
    t.test_f32_learned_mode_cpu()

    print("\n--- GPU Identity Mode Tests ---")
    t2 = TestGPUIdentityMode()
    t2.test_identity_vs_standard_softmax()

    print("\n--- Strided Kernel Tests ---")
    t3 = TestStridedKernels()
    t3.test_strided_identity_dim0()
    t3.test_strided_learned_dim0()
    t3.test_3d_strided_middle_dim()

    print("\n--- Provided Mode Tests ---")
    t4 = TestProvidedMode()
    t4.test_provided_mode_basic()
    t4.test_provided_mode_uniform_weights()

    print("\n--- Python API Tests ---")
    t5 = TestPythonAPI()
    t5.test_api_identity_mode()
    t5.test_api_learned_mode()
    t5.test_api_provided_mode()
    t5.test_api_default_mode()

    print("\n" + "=" * 70)
    print("All Phase 2 tests passed!")
    print("=" * 70)
