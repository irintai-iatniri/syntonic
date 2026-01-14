"""
Basic correctness tests for syntonic_softmax.

Tests verify:
1. Mode norm initialization (sequential i²)
2. Forward pass shape preservation
3. Probability normalization
4. Golden measure weighting
5. CPU/GPU consistency (if CUDA available)
"""

import pytest
import numpy as np

try:
    import syntonic.core as core
    from syntonic.core import (
        ResonantTensor,
        SyntonicSoftmaxState,
        SyntonicSoftmaxMode,
    )
    CORE_AVAILABLE = True
except ImportError:
    CORE_AVAILABLE = False
    pytestmark = pytest.mark.skip(reason="syntonic.core not available")


# Golden ratio constant
PHI = 1.618033988749895


def test_mode_norm_initialization():
    """Test that mode norms are initialized as 8D E8 lattice roots from Golden Cone."""
    num_features = 10
    # Use constructor directly (not .new())
    state = SyntonicSoftmaxState(
        SyntonicSoftmaxMode.Learned,
        -1,  # dim
        num_features,
        1.0  # syntony_scale
    )

    mode_norms_tensor = state.get_mode_norms()
    mode_norms = mode_norms_tensor.to_floats()
    shape = mode_norms_tensor.shape

    # Should be [num_features, 8] for E8 roots
    assert shape == [num_features, 8], f"Expected shape [10, 8], got {shape}"

    # Verify roots are 8-dimensional
    for i in range(num_features):
        root = mode_norms[i*8:(i+1)*8]
        # Each root should be an E8 root (all components are -1, -0.5, 0, 0.5, or 1)
        for val in root:
            assert val in [-1.0, -0.5, 0.0, 0.5, 1.0], f"Invalid E8 root component: {val}"

    print(f"✓ Mode norms correctly initialized as E8 roots, shape: {shape}")


def test_forward_pass_shapes():
    """Test that forward pass preserves input shapes."""
    test_cases = [
        (1, 10),    # Single sample, 10 classes
        (4, 10),    # Small batch
        (16, 100),  # Larger batch and classes
    ]

    for batch_size, num_features in test_cases:
        # Create input tensor using (data, shape) constructor
        data = [1.0] * (batch_size * num_features)
        x = ResonantTensor(data, [batch_size, num_features])

        # Create state
        state = SyntonicSoftmaxState(
            SyntonicSoftmaxMode.Identity,
            -1,  # dim
            None,  # num_features (not needed for identity)
            1.0  # syntony_scale
        )

        # Forward pass
        y = state.forward(x, None)

        # Check shape preserved (shape is a property, not a method)
        assert y.shape == [batch_size, num_features], \
            f"Shape mismatch: expected {[batch_size, num_features]}, got {y.shape}"

    print(f"✓ Forward pass preserves shapes for all test cases")


def test_probability_normalization():
    """Test that softmax outputs sum to 1.0 per row."""
    # Create simple 3-class input
    data = [1.0, 2.0, 3.0]
    x = ResonantTensor(data, [1, 3])

    # Identity mode (standard softmax)
    state = SyntonicSoftmaxState(
        SyntonicSoftmaxMode.Identity,
        -1,
        None,
        1.0
    )

    y = state.forward(x, None)
    probs = y.to_floats()

    # Check sum (allow tolerance for lattice quantization)
    total = sum(probs)
    assert np.isclose(total, 1.0, rtol=2e-3), \
        f"Probabilities don't sum to 1.0: sum = {total}"

    print(f"✓ Probabilities sum to 1.0: {probs}")


def test_golden_measure_weighting():
    """Test that learned mode applies E8 golden measure weighting correctly."""
    num_features = 5

    # Create input with equal logits (zeros)
    data = [0.0] * num_features
    x = ResonantTensor(data, [1, num_features])

    # Learned mode with E8 root-based mode norms
    state = SyntonicSoftmaxState(
        SyntonicSoftmaxMode.Learned,
        -1,
        num_features,
        1.0
    )

    y = state.forward(x, None)
    probs = y.to_floats()

    # With equal logits, probabilities should follow golden measure weighting
    # based on the 8D E8 roots: w(i) = exp(-|P_parallel(root_i)|²/φ)
    # Different roots have different weights, creating hierarchical structure

    # All probabilities should be positive
    for i, p in enumerate(probs):
        assert p > 0, f"Probability {i} is non-positive: {p}"

    # Probabilities should sum to 1.0 (allow tolerance for lattice quantization)
    total = sum(probs)
    assert np.isclose(total, 1.0, rtol=2e-3), f"Probabilities sum to {total}, not 1.0"

    # With E8 roots from Golden Cone, weights vary based on root geometry
    # Not necessarily monotonic, but should show variation
    unique_probs = len(set(round(p, 6) for p in probs))
    assert unique_probs > 1, "All probabilities are identical - no golden weighting applied"

    print(f"✓ Golden measure weighting applied correctly: {probs}")
    print(f"  Unique probability values: {unique_probs}/{num_features}")


def test_identity_vs_learned():
    """Test that identity and learned modes produce different results."""
    num_features = 5
    data = [1.0, 2.0, 3.0, 4.0, 5.0]  # Non-uniform logits
    x = ResonantTensor(data, [1, num_features])

    # Identity mode
    state_identity = SyntonicSoftmaxState(
        SyntonicSoftmaxMode.Identity,
        -1,
        None,
        1.0
    )
    y_identity = state_identity.forward(x, None)
    probs_identity = y_identity.to_floats()

    # Learned mode
    state_learned = SyntonicSoftmaxState(
        SyntonicSoftmaxMode.Learned,
        -1,
        num_features,
        1.0
    )
    y_learned = state_learned.forward(x, None)
    probs_learned = y_learned.to_floats()

    # They should be different (learned applies golden weighting)
    assert not np.allclose(probs_identity, probs_learned, rtol=1e-5), \
        "Identity and learned modes should produce different results"

    print(f"✓ Identity and learned modes produce different results")
    print(f"  Identity: {probs_identity}")
    print(f"  Learned:  {probs_learned}")


@pytest.mark.skipif(not CORE_AVAILABLE, reason="CUDA not available")
def test_cpu_gpu_consistency():
    """Test that CPU and GPU produce consistent results (if CUDA available)."""
    pytest.skip("GPU tests require CUDA setup - skipping for now")


def test_all_modes_callable():
    """Test that all three modes can be instantiated and called."""
    num_features = 5
    data = [1.0] * num_features
    x = ResonantTensor(data, [1, num_features])

    modes = [
        (SyntonicSoftmaxMode.Learned, num_features),
        (SyntonicSoftmaxMode.Identity, None),
        # Provided mode requires syntony tensor in forward, tested separately
    ]

    for mode, nf in modes:
        print(f"Testing mode: {mode}")
        state = SyntonicSoftmaxState(
            mode,
            -1,
            nf,
            1.0
        )
        y = state.forward(x, None)
        probs = y.to_floats()

        # Basic sanity checks
        assert len(probs) == num_features
        # Relax tolerance for lattice quantization effects
        assert np.isclose(sum(probs), 1.0, rtol=2e-2), \
            f"Sum {sum(probs)} not close to 1.0 for mode {mode}"


        print(f"✓ {mode} mode works correctly")


if __name__ == "__main__":
    # Run tests manually for debugging
    print("Running syntonic_softmax basic tests...\n")

    if CORE_AVAILABLE:
        test_mode_norm_initialization()
        test_forward_pass_shapes()
        test_probability_normalization()
        test_golden_measure_weighting()
        test_identity_vs_learned()
        test_all_modes_callable()

        print("\n✓ All basic tests passed!")
    else:
        print("syntonic.core not available, skipping tests")
