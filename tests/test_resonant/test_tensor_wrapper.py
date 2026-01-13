"""
Test suite for ResonantTensor Python wrapper.

Tests the Python wrapper layer over the Rust-backed ResonantTensor,
including type safety, operator overloading, and convenience methods.
"""

import pytest
from typing import List
from syntonic.nn.resonant_tensor import ResonantTensor


class TestConstruction:
    """Test construction and factory methods."""

    def test_basic_construction(self):
        """Test basic construction from floats."""
        data = [1.0, 2.0, 3.0, 4.0]
        tensor = ResonantTensor(data, shape=[2, 2])

        assert tensor.shape == [2, 2]
        assert len(tensor) == 4
        assert tensor.phase == "crystallized"
        assert 0.0 <= tensor.syntony <= 1.0

    def test_zeros_factory(self):
        """Test zeros factory method."""
        zeros = ResonantTensor.zeros([3, 3])

        assert zeros.shape == [3, 3]
        assert len(zeros) == 9

        floats = zeros.to_floats()
        assert all(abs(v) < 1e-6 for v in floats), "All values should be close to zero"

    def test_ones_factory(self):
        """Test ones factory method."""
        ones = ResonantTensor.ones([2, 2])

        assert ones.shape == [2, 2]
        assert len(ones) == 4

        floats = ones.to_floats()
        assert all(abs(v - 1.0) < 0.1 for v in floats), "All values should be close to one"

    def test_randn_factory(self):
        """Test random Gaussian factory method."""
        tensor = ResonantTensor.randn([5, 5])

        assert tensor.shape == [5, 5]
        assert len(tensor) == 25

        floats = tensor.to_floats()
        assert len(floats) == 25

        # Check that values are distributed (not all the same)
        unique_values = len(set(round(v, 2) for v in floats))
        assert unique_values > 10, "Should have diverse values"

    def test_randn_with_params(self):
        """Test randn with custom mean and std."""
        tensor = ResonantTensor.randn([100], mean=10.0, std=2.0, precision=100)

        floats = tensor.to_floats()
        mean = sum(floats) / len(floats)

        # Mean should be approximately 10 (with some tolerance for small sample)
        assert 8.0 < mean < 12.0, f"Mean {mean} should be near 10.0"


class TestProperties:
    """Test properties and accessors."""

    def test_properties(self):
        """Test property access."""
        tensor = ResonantTensor.randn([3, 4])

        # Shape
        assert tensor.shape == [3, 4]
        assert isinstance(tensor.shape, list)

        # Phase
        assert tensor.phase == "crystallized"
        assert isinstance(tensor.phase, str)

        # Syntony
        assert isinstance(tensor.syntony, float)
        assert 0.0 <= tensor.syntony <= 1.0

        # Precision
        assert isinstance(tensor.precision, int)
        assert tensor.precision == 100  # default

    def test_to_floats(self):
        """Test conversion to floats."""
        data = [1.0, 2.0, 3.0, 4.0]
        tensor = ResonantTensor(data, shape=[2, 2])

        floats = tensor.to_floats()
        assert isinstance(floats, list)
        assert len(floats) == 4

        # Values should be close to original (after lattice snapping)
        for orig, recovered in zip(data, floats):
            assert abs(orig - recovered) < 0.1

    def test_to_list(self):
        """Test to_list() alias."""
        tensor = ResonantTensor.ones([2])
        assert tensor.to_list() == tensor.to_floats()


class TestOperatorOverloading:
    """Test Python operator overloading."""

    def test_add_tensor(self):
        """Test tensor + tensor."""
        a = ResonantTensor([1.0, 2.0], [2])
        b = ResonantTensor([3.0, 4.0], [2])

        c = a + b

        assert isinstance(c, ResonantTensor)
        assert c.shape == [2]

        floats = c.to_floats()
        assert abs(floats[0] - 4.0) < 0.1
        assert abs(floats[1] - 6.0) < 0.1

    def test_add_scalar(self):
        """Test tensor + scalar."""
        a = ResonantTensor([1.0, 2.0], [2])
        c = a + 10.0

        assert isinstance(c, ResonantTensor)
        floats = c.to_floats()
        assert abs(floats[0] - 11.0) < 0.1
        assert abs(floats[1] - 12.0) < 0.1

    def test_mul_tensor(self):
        """Test tensor * tensor (Hadamard)."""
        a = ResonantTensor([2.0, 3.0], [2])
        b = ResonantTensor([4.0, 5.0], [2])

        c = a * b

        assert isinstance(c, ResonantTensor)
        floats = c.to_floats()
        assert abs(floats[0] - 8.0) < 0.1  # 2 * 4
        assert abs(floats[1] - 15.0) < 0.1  # 3 * 5

    def test_mul_scalar(self):
        """Test tensor * scalar."""
        a = ResonantTensor([2.0, 3.0], [2])
        c = a * 2.0

        floats = c.to_floats()
        assert abs(floats[0] - 4.0) < 0.1
        assert abs(floats[1] - 6.0) < 0.1

    def test_rmul_scalar(self):
        """Test scalar * tensor (right multiplication)."""
        a = ResonantTensor([2.0, 3.0], [2])
        c = 2.0 * a

        floats = c.to_floats()
        assert abs(floats[0] - 4.0) < 0.1
        assert abs(floats[1] - 6.0) < 0.1

    def test_neg(self):
        """Test -tensor."""
        a = ResonantTensor([1.0, -2.0, 3.0], [3])
        b = -a

        assert isinstance(b, ResonantTensor)
        floats = b.to_floats()
        assert abs(floats[0] - (-1.0)) < 0.1
        assert abs(floats[1] - 2.0) < 0.1
        assert abs(floats[2] - (-3.0)) < 0.1

    def test_matmul(self):
        """Test @ operator (matrix multiplication)."""
        x = ResonantTensor.randn([4, 10])  # Batch of 4, 10 features
        w = ResonantTensor.randn([20, 10])  # 10 → 20

        y = x @ w

        assert isinstance(y, ResonantTensor)
        assert y.shape == [4, 20]

    def test_len(self):
        """Test len() operator."""
        tensor = ResonantTensor.zeros([3, 4, 5])
        assert len(tensor) == 60


class TestActivations:
    """Test activation functions."""

    def test_relu(self):
        """Test ReLU activation."""
        x = ResonantTensor([-1.0, 2.0, -3.0, 4.0], [4])
        x.relu()

        floats = x.to_floats()
        assert abs(floats[0]) < 0.1  # -1 → 0
        assert floats[1] > 1.0  # 2 → 2
        assert abs(floats[2]) < 0.1  # -3 → 0
        assert floats[3] > 3.0  # 4 → 4

    def test_sigmoid(self):
        """Test sigmoid activation."""
        x = ResonantTensor([0.0, 1.0, -1.0], [3])
        x.sigmoid(precision=100)

        floats = x.to_floats()
        # sigmoid(0) ≈ 0.5
        assert 0.4 < floats[0] < 0.6
        # sigmoid(1) ≈ 0.73
        assert 0.6 < floats[1] < 0.8
        # sigmoid(-1) ≈ 0.27
        assert 0.2 < floats[2] < 0.4

    def test_softmax(self):
        """Test softmax activation."""
        logits = ResonantTensor([2.0, 1.0, 0.1], [3])
        logits.softmax(precision=32)

        probs = logits.to_floats()
        # Should sum to 1
        total = sum(probs)
        assert abs(total - 1.0) < 0.1

        # Highest logit should have highest probability
        assert probs[0] > probs[1] > probs[2]


class TestPhaseCycles:
    """Test DHSR phase transitions."""

    def test_cpu_cycle(self):
        """Test full CPU cycle."""
        tensor = ResonantTensor.randn([10], precision=100)
        initial_syntony = tensor.syntony

        new_syntony = tensor.cpu_cycle(noise_scale=0.1, precision=100)

        assert isinstance(new_syntony, float)
        assert 0.0 <= new_syntony <= 1.0
        assert tensor.phase == "crystallized"

    def test_wake_and_crystallize(self):
        """Test manual D→H cycle."""
        tensor = ResonantTensor.ones([5])

        # Wake flux
        flux = tensor.wake_flux()
        assert tensor.phase == "flux"
        assert isinstance(flux, list)
        assert len(flux) == 5

        # Crystallize
        new_syntony = tensor.crystallize(flux, precision=100)
        assert tensor.phase == "crystallized"
        assert isinstance(new_syntony, float)

    def test_batch_cpu_cycle(self):
        """Test batched CPU cycle."""
        batch_tensor = ResonantTensor.randn([8, 16])  # Batch of 8

        syntonies = batch_tensor.batch_cpu_cycle(noise_scale=0.05, precision=100)

        assert isinstance(syntonies, list)
        assert len(syntonies) == 8
        assert all(0.0 <= s <= 1.0 for s in syntonies)


class TestLinearAlgebra:
    """Test linear algebra operations."""

    def test_matmul(self):
        """Test matrix multiplication."""
        x = ResonantTensor.randn([4, 10])
        w = ResonantTensor.randn([20, 10])

        y = x.matmul(w)

        assert y.shape == [4, 20]
        assert isinstance(y, ResonantTensor)

    def test_add_bias(self):
        """Test bias addition."""
        x = ResonantTensor.randn([4, 10])
        bias = ResonantTensor.ones([10])

        x.add_bias(bias)  # In-place

        # Shape should be unchanged
        assert x.shape == [4, 10]


class TestAdvancedOperations:
    """Test advanced tensor operations."""

    def test_concat(self):
        """Test concatenation."""
        a = ResonantTensor([1.0, 2.0], [2])
        b = ResonantTensor([3.0, 4.0], [2])

        c = ResonantTensor.concat([a, b], dim=0)

        assert c.shape == [4]
        floats = c.to_floats()
        assert len(floats) == 4

    def test_index_select(self):
        """Test index selection."""
        x = ResonantTensor.randn([10, 5])
        selected = x.index_select([0, 2, 4], dim=0)

        assert selected.shape == [3, 5]

    def test_mean(self):
        """Test mean reduction."""
        x = ResonantTensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], [2, 3])

        # Global mean
        global_mean = x.mean()
        assert global_mean.shape == [1]

        # Mean along dimension
        mean_dim1 = x.mean(dim=1)
        assert mean_dim1.shape == [2]

    def test_layer_norm(self):
        """Test layer normalization."""
        x = ResonantTensor.randn([8, 16])
        normalized = x.layer_norm()

        assert normalized.shape == [8, 16]
        assert isinstance(normalized, ResonantTensor)


class TestGoldenOperations:
    """Test golden recursion operations."""

    def test_apply_recursion(self):
        """Test golden recursion map."""
        x = ResonantTensor.ones([3])
        initial = x.to_floats()[0]

        x.apply_recursion()

        after = x.to_floats()[0]
        # Should be scaled by φ ≈ 1.618
        assert abs(after / initial - 1.618) < 0.1

    def test_apply_inverse_recursion(self):
        """Test inverse golden recursion."""
        x = ResonantTensor.ones([3])
        initial = x.to_floats()[0]

        x.apply_inverse_recursion()

        after = x.to_floats()[0]
        # Should be scaled by 1/φ ≈ 0.618
        assert abs(after / initial - 0.618) < 0.1

    def test_prune_hierarchy(self):
        """Test hierarchical pruning."""
        x = ResonantTensor([0.001, 1.0, 0.002, 2.0], [4])
        x.prune_hierarchy(q=1.0, divisor=100.0)  # Threshold = 0.01

        floats = x.to_floats()
        # Small values should be pruned to zero
        assert abs(floats[0]) < 0.01
        assert abs(floats[2]) < 0.01
        # Large values should remain
        assert floats[1] > 0.5
        assert floats[3] > 1.0


class TestTypeAnnotations:
    """Test that type annotations work correctly."""

    def test_type_hints(self):
        """Verify type hints work (this is mainly for mypy/IDE)."""
        # This test mainly ensures the code typechecks with mypy
        tensor: ResonantTensor = ResonantTensor.randn([5, 5])
        floats: List[float] = tensor.to_floats()
        syntony: float = tensor.syntony
        shape: List[int] = tensor.shape

        assert isinstance(tensor, ResonantTensor)
        assert isinstance(floats, list)
        assert isinstance(syntony, float)
        assert isinstance(shape, list)


class TestRepr:
    """Test string representation."""

    def test_repr(self):
        """Test __repr__ method."""
        tensor = ResonantTensor.randn([3, 4])
        repr_str = repr(tensor)

        assert "ResonantTensor" in repr_str
        assert "shape" in repr_str
        assert "phase" in repr_str
        assert "syntony" in repr_str


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
