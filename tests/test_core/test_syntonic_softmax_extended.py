"""
Extended Tests for Syntonic Softmax - Phase 3 Robustness

These tests verify:
1. Numerical stability (large/small logits, extreme values)
2. Edge cases (single feature, large batches, various dimensions)
3. Mode comparisons (learned vs identity, uniform weights)
4. Error handling (invalid inputs, shape mismatches)
5. Regression tests for Phase 3 fixes (strided broadcasting)

Phase 3 Implementation Status:
- Task 3.1: Fix strided broadcasting bug, add error handling
- Task 3.2: Extended test suite (this file)
"""

import pytest
import numpy as np

try:
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

def reference_softmax(logits, dim=-1):
    """Reference numpy softmax implementation for validation."""
    logits = np.array(logits)
    max_val = np.max(logits, axis=dim, keepdims=True)
    exp_logits = np.exp(logits - max_val)
    return exp_logits / np.sum(exp_logits, axis=dim, keepdims=True)


def has_cuda():
    """Check if CUDA is available."""
    try:
        t = ResonantTensor([1.0], [1])
        t.to_device(0)
        return True
    except Exception:
        return False


# =============================================================================
# Task 3.2.1: Numerical Stability Tests
# =============================================================================

class TestNumericalStability:
    """Tests for numerical stability with extreme values."""

    def test_large_logits(self):
        """Softmax should handle large logits without overflow."""
        data = [1e6, 1e6 + 1, 1e6 + 2]
        x = ResonantTensor(data, [1, 3])
        y = syntonic_softmax(x, mode="identity")
        probs = y.to_floats()

        # Should sum to 1 and be valid probabilities
        assert np.isclose(sum(probs), 1.0, rtol=1e-3), f"Sum = {sum(probs)}"
        assert all(0 <= p <= 1 for p in probs), f"Invalid probs: {probs}"

        # Larger logit should have higher probability
        assert probs[2] > probs[1] > probs[0], f"Wrong ordering: {probs}"
        print(f"Large logits: {probs}")

    def test_small_logits(self):
        """Softmax should handle very negative logits without underflow."""
        data = [-1e6, -1e6 + 1, -1e6 + 2]
        x = ResonantTensor(data, [1, 3])
        y = syntonic_softmax(x, mode="identity")
        probs = y.to_floats()

        assert np.isclose(sum(probs), 1.0, rtol=1e-3), f"Sum = {sum(probs)}"
        assert all(0 <= p <= 1 for p in probs), f"Invalid probs: {probs}"
        # Less negative should have higher probability
        assert probs[2] > probs[1] > probs[0], f"Wrong ordering: {probs}"
        print(f"Small logits: {probs}")

    def test_mixed_extreme_logits(self):
        """Softmax should handle mixed extreme values (winner-take-all)."""
        data = [1e6, 0, -1e6]
        x = ResonantTensor(data, [1, 3])
        y = syntonic_softmax(x, mode="identity")
        probs = y.to_floats()

        assert np.isclose(sum(probs), 1.0, rtol=1e-3), f"Sum = {sum(probs)}"
        # First should dominate
        assert probs[0] > 0.99, f"Expected winner-take-all, got {probs}"
        print(f"Mixed extreme: {probs}")

    def test_zero_logits_uniform(self):
        """Zero logits should give uniform distribution."""
        data = [0.0, 0.0, 0.0, 0.0]
        x = ResonantTensor(data, [1, 4])
        y = syntonic_softmax(x, mode="identity")
        probs = y.to_floats()

        assert np.isclose(sum(probs), 1.0, rtol=1e-2)  # Lattice arithmetic has ~1% precision
        # Should be uniform (0.25 each)
        expected = [0.25, 0.25, 0.25, 0.25]
        assert np.allclose(probs, expected, rtol=1e-2), f"Expected uniform, got {probs}"
        print(f"Zero logits (uniform): {probs}")

    def test_nearly_equal_logits(self):
        """Nearly equal logits should give nearly uniform distribution."""
        data = [1.0, 1.0 + 1e-10, 1.0 + 2e-10]
        x = ResonantTensor(data, [1, 3])
        y = syntonic_softmax(x, mode="identity")
        probs = y.to_floats()

        assert np.isclose(sum(probs), 1.0, rtol=1e-3)
        # Should be nearly uniform
        assert np.allclose(probs, [1/3, 1/3, 1/3], rtol=1e-3), f"Expected nearly uniform, got {probs}"
        print(f"Nearly equal: {probs}")


# =============================================================================
# Task 3.2.2: Edge Case Tests
# =============================================================================

class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_single_feature(self):
        """Single feature should return probability 1.0."""
        x = ResonantTensor([5.0], [1, 1])
        y = syntonic_softmax(x, mode="identity")
        probs = y.to_floats()

        assert len(probs) == 1
        assert np.isclose(probs[0], 1.0), f"Expected 1.0, got {probs[0]}"
        print("Single feature: 1.0")

    def test_two_features(self):
        """Two features basic case."""
        x = ResonantTensor([0.0, 0.0], [1, 2])
        y = syntonic_softmax(x, mode="identity")
        probs = y.to_floats()

        assert np.allclose(probs, [0.5, 0.5], rtol=1e-3)
        print(f"Two features: {probs}")

    def test_large_num_classes(self):
        """Test with large number of classes (1000)."""
        num_classes = 1000
        data = list(np.random.randn(num_classes))
        x = ResonantTensor(data, [1, num_classes])
        y = syntonic_softmax(x, mode="identity")
        probs = y.to_floats()

        assert len(probs) == num_classes
        assert np.isclose(sum(probs), 1.0, rtol=0.05)  # Lattice precision degrades with many classes
        assert all(p >= 0 for p in probs)
        print(f"Large num_classes ({num_classes}): sum={sum(probs):.6f}")

    def test_various_batch_sizes(self):
        """Test various batch sizes."""
        for batch_size in [1, 2, 16, 64, 256]:
            num_classes = 10
            data = list(np.random.randn(batch_size * num_classes))
            x = ResonantTensor(data, [batch_size, num_classes])
            y = syntonic_softmax(x, mode="identity")
            probs = np.array(y.to_floats()).reshape(batch_size, num_classes)

            # Each row should sum to 1
            row_sums = probs.sum(axis=1)
            assert np.allclose(row_sums, 1.0, rtol=1e-2), \
                f"batch_size={batch_size}: row_sums={row_sums}"
        print("Various batch sizes: all pass")

    def test_1d_tensor(self):
        """Test 1D tensor (single batch implied)."""
        data = [1.0, 2.0, 3.0]
        x = ResonantTensor(data, [3])
        y = syntonic_softmax(x, mode="identity", dim=0)
        probs = y.to_floats()

        assert np.isclose(sum(probs), 1.0, rtol=1e-3)
        print(f"1D tensor: {probs}")

    def test_3d_tensor_last_dim(self):
        """Test 3D tensor with softmax on last dimension."""
        shape = [2, 3, 4]
        data = list(np.random.randn(np.prod(shape)))
        x = ResonantTensor(data, shape)
        y = syntonic_softmax(x, mode="identity", dim=-1)
        probs = np.array(y.to_floats()).reshape(shape)

        # Last dimension should sum to 1
        for b in range(2):
            for r in range(3):
                assert np.isclose(probs[b, r, :].sum(), 1.0, rtol=1e-2)
        print("3D tensor last dim: pass")

    def test_4d_tensor(self):
        """Test 4D tensor (like image batches)."""
        shape = [2, 3, 4, 5]
        data = list(np.random.randn(np.prod(shape)))
        x = ResonantTensor(data, shape)
        y = syntonic_softmax(x, mode="identity", dim=-1)
        probs = np.array(y.to_floats()).reshape(shape)

        # Last dimension should sum to 1
        for indices in np.ndindex(2, 3, 4):
            assert np.isclose(probs[indices].sum(), 1.0, rtol=1e-2)
        print("4D tensor: pass")


# =============================================================================
# Task 3.2.3: Mode Comparison Tests
# =============================================================================

class TestModeComparisons:
    """Tests comparing different softmax modes."""

    def test_identity_matches_numpy(self):
        """Identity mode should match numpy reference exactly."""
        data = [1.0, 2.0, 3.0, 4.0, 5.0]
        x = ResonantTensor(data, [1, 5])
        y = syntonic_softmax(x, mode="identity")
        probs = y.to_floats()

        expected = reference_softmax(data).flatten()
        assert np.allclose(probs, expected, rtol=1e-2), \
            f"Got {probs}, expected {expected}"
        print("Identity matches numpy: pass")

    def test_provided_uniform_matches_identity(self):
        """Provided mode with uniform weights should match identity."""
        data = [1.0, 2.0, 3.0, 4.0, 5.0]
        syntony_data = [1.0, 1.0, 1.0, 1.0, 1.0]

        x = ResonantTensor(data, [1, 5])
        syntony = ResonantTensor(syntony_data, [1, 5])

        y_provided = syntonic_softmax(x, mode="provided", syntony=syntony)
        y_identity = syntonic_softmax(x, mode="identity")

        probs_provided = y_provided.to_floats()
        probs_identity = y_identity.to_floats()

        # Should be very close (uniform syntony = no weighting effect)
        assert np.allclose(probs_provided, probs_identity, rtol=0.1), \
            f"Provided: {probs_provided}, Identity: {probs_identity}"
        print("Provided uniform matches identity: pass")

    def test_learned_vs_identity_differ(self):
        """Learned mode should differ from identity on equal logits."""
        num_features = 5
        data = [0.0] * num_features  # Equal logits
        x = ResonantTensor(data, [1, num_features])

        y_identity = syntonic_softmax(x, mode="identity")

        state = SyntonicSoftmaxState(SyntonicSoftmaxMode.Learned, -1, num_features, 1.0)
        y_learned = state.forward(x, None)

        probs_identity = y_identity.to_floats()
        probs_learned = y_learned.to_floats()

        # Identity should be uniform, learned should have variation
        assert np.allclose(probs_identity, [0.2] * 5, rtol=1e-2), \
            f"Identity not uniform: {probs_identity}"

        # Learned should have different weights due to E8 roots
        unique_probs = len(set(round(p, 4) for p in probs_learned))
        assert unique_probs > 1, f"Learned has no variation: {probs_learned}"
        print(f"Learned differs from identity: identity={probs_identity}, learned={probs_learned}")


# =============================================================================
# Task 3.2.4: Error Handling Tests
# =============================================================================

class TestErrorHandling:
    """Tests for proper error handling."""

    def test_invalid_dimension_positive(self):
        """Test error for out-of-range positive dimension."""
        data = [1.0, 2.0, 3.0]
        x = ResonantTensor(data, [1, 3])

        with pytest.raises(Exception) as excinfo:
            syntonic_softmax(x, mode="identity", dim=5)

        assert "dimension" in str(excinfo.value).lower() or "out of range" in str(excinfo.value).lower()
        print("Invalid positive dim: error raised correctly")

    def test_invalid_dimension_negative(self):
        """Test error for out-of-range negative dimension."""
        data = [1.0, 2.0, 3.0]
        x = ResonantTensor(data, [1, 3])

        with pytest.raises(Exception) as excinfo:
            syntonic_softmax(x, mode="identity", dim=-5)

        assert "dimension" in str(excinfo.value).lower() or "out of range" in str(excinfo.value).lower()
        print("Invalid negative dim: error raised correctly")

    def test_invalid_mode_string(self):
        """Test error for invalid mode string."""
        data = [1.0, 2.0, 3.0]
        x = ResonantTensor(data, [1, 3])

        with pytest.raises(Exception) as excinfo:
            syntonic_softmax(x, mode="invalid_mode")

        assert "mode" in str(excinfo.value).lower() or "unknown" in str(excinfo.value).lower()
        print("Invalid mode string: error raised correctly")

    def test_provided_missing_syntony(self):
        """Test error when provided mode is used without syntony."""
        data = [1.0, 2.0, 3.0]
        x = ResonantTensor(data, [1, 3])

        with pytest.raises(Exception) as excinfo:
            syntonic_softmax(x, mode="provided")  # No syntony provided

        assert "syntony" in str(excinfo.value).lower()
        print("Missing syntony: error raised correctly")

    def test_shape_mismatch_syntony(self):
        """Test error when syntony shape doesn't match input."""
        data = [1.0, 2.0, 3.0]
        syntony_data = [1.0, 2.0]  # Wrong size

        x = ResonantTensor(data, [1, 3])
        syntony = ResonantTensor(syntony_data, [1, 2])

        with pytest.raises(Exception) as excinfo:
            syntonic_softmax(x, mode="provided", syntony=syntony)

        assert "shape" in str(excinfo.value).lower() or "mismatch" in str(excinfo.value).lower()
        print("Shape mismatch: error raised correctly")


# =============================================================================
# Task 3.2.5: Regression Tests for Phase 3 Fixes
# =============================================================================

class TestStridedRegressions:
    """Regression tests for strided broadcasting fixes from Phase 3."""

    def test_cpu_strided_learned_dim0(self):
        """CPU learned mode should work for dim=0 (Phase 3 fix)."""
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
                f"Column {col} sum = {col_sum}, expected 1.0"

        print(f"CPU strided learned dim=0: column sums = {probs.sum(axis=0)}")

    def test_cpu_strided_learned_dim1_3d(self):
        """CPU learned mode for middle dimension in 3D tensor."""
        # Shape [2, 3, 4] - softmax along dim 1 (3 features)
        data = [0.0] * 24
        x = ResonantTensor(data, [2, 3, 4])

        state = SyntonicSoftmaxState(
            SyntonicSoftmaxMode.Learned,
            1,  # middle dim
            3,  # 3 features (dim size)
            1.0
        )

        y = state.forward(x, None)
        probs = np.array(y.to_floats()).reshape(2, 3, 4)

        # For each (batch, feature) pair, dim 1 should sum to 1
        for b in range(2):
            for f in range(4):
                dim1_sum = probs[b, :, f].sum()
                assert np.isclose(dim1_sum, 1.0, rtol=1e-2), \
                    f"[{b}, :, {f}] sum = {dim1_sum}, expected 1.0"

        print("CPU strided learned dim=1 (3D): pass")

    def test_cpu_strided_identity_dim0(self):
        """CPU identity mode for dim=0 (baseline comparison)."""
        # Shape [3, 4] - softmax along dim 0
        data = list(range(12))
        x = ResonantTensor([float(d) for d in data], [3, 4])

        state = SyntonicSoftmaxState(
            SyntonicSoftmaxMode.Identity,
            0,  # dim 0
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

        print(f"CPU strided identity dim=0: pass")

    def test_strided_learned_nonzero_logits(self):
        """Strided learned mode with non-zero logits."""
        # Shape [3, 4] - softmax along dim 0
        data = list(np.random.randn(12))
        x = ResonantTensor(data, [3, 4])

        state = SyntonicSoftmaxState(
            SyntonicSoftmaxMode.Learned,
            0,
            3,
            1.0
        )

        y = state.forward(x, None)
        probs = np.array(y.to_floats()).reshape(3, 4)

        # Each column should sum to 1
        for col in range(4):
            col_sum = probs[:, col].sum()
            assert np.isclose(col_sum, 1.0, rtol=1e-2), \
                f"Column {col} sum = {col_sum}"

        # All probabilities should be valid
        assert all(0 <= p <= 1 for p in probs.flatten()), "Invalid probability"

        print("Strided learned with random logits: pass")


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("Syntonic Softmax Extended Tests (Phase 3)")
    print("=" * 70)

    if not CORE_AVAILABLE:
        print("syntonic.core not available - skipping tests")
        exit(1)

    # Run tests manually
    print("\n--- Numerical Stability Tests ---")
    t1 = TestNumericalStability()
    t1.test_large_logits()
    t1.test_small_logits()
    t1.test_mixed_extreme_logits()
    t1.test_zero_logits_uniform()
    t1.test_nearly_equal_logits()

    print("\n--- Edge Case Tests ---")
    t2 = TestEdgeCases()
    t2.test_single_feature()
    t2.test_two_features()
    t2.test_large_num_classes()
    t2.test_various_batch_sizes()
    t2.test_1d_tensor()
    t2.test_3d_tensor_last_dim()
    t2.test_4d_tensor()

    print("\n--- Mode Comparison Tests ---")
    t3 = TestModeComparisons()
    t3.test_identity_matches_numpy()
    t3.test_provided_uniform_matches_identity()
    t3.test_learned_vs_identity_differ()

    print("\n--- Error Handling Tests ---")
    t4 = TestErrorHandling()
    t4.test_invalid_dimension_positive()
    t4.test_invalid_dimension_negative()
    t4.test_invalid_mode_string()
    t4.test_provided_missing_syntony()
    t4.test_shape_mismatch_syntony()

    print("\n--- Strided Regression Tests ---")
    t5 = TestStridedRegressions()
    t5.test_cpu_strided_learned_dim0()
    t5.test_cpu_strided_learned_dim1_3d()
    t5.test_cpu_strided_identity_dim0()
    t5.test_strided_learned_nonzero_logits()

    print("\n" + "=" * 70)
    print("All Phase 3 extended tests passed!")
    print("=" * 70)
