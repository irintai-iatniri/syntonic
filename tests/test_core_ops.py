"""Tests for core mathematical operations.

Tests exp, sum, mean, layer_norm, and dropout operations.
Verifies SRT-aligned golden target variance for normalization.
"""
import math
import pytest

# Golden ratio constants
PHI = 1.6180339887498949
PHI_INV = 0.6180339887498949

@pytest.fixture
def syn():
    """Import syntonic module."""
    import syntonic as syn
    return syn


class TestExp:
    """Test element-wise exponential."""
    
    def test_exp_basic(self, syn):
        """Test exp matches math.exp."""
        s = syn.state([0.0, 1.0, 2.0])
        result = s.exp()
        
        expected = [math.exp(0.0), math.exp(1.0), math.exp(2.0)]
        actual = result.to_list()
        
        for a, e in zip(actual, expected):
            assert abs(a - e) < 1e-10
    
    def test_exp_negative(self, syn):
        """Test exp on negative values."""
        s = syn.state([-1.0, -2.0])
        result = s.exp()
        
        expected = [math.exp(-1.0), math.exp(-2.0)]
        actual = result.to_list()
        
        for a, e in zip(actual, expected):
            assert abs(a - e) < 1e-10

    def test_exp_preserves_shape(self, syn):
        """Test exp preserves tensor shape."""
        s = syn.state([[1.0, 2.0], [3.0, 4.0]])
        result = s.exp()
        assert result.shape == (2, 2)


class TestExpGolden:
    """Test golden exponential: exp(-x/φ)."""
    
    def test_exp_golden_basic(self, syn):
        """Test exp_golden matches exp(-x/φ)."""
        s = syn.state([0.0, 1.0, PHI])
        result = s.exp_golden()
        
        expected = [
            math.exp(0.0),           # exp(0) = 1
            math.exp(-1.0/PHI),      # exp(-1/φ)
            math.exp(-PHI/PHI),      # exp(-1)
        ]
        actual = result.to_list()
        
        for a, e in zip(actual, expected):
            assert abs(a - e) < 1e-10
    
    def test_exp_golden_weight_measure(self, syn):
        """Test that exp_golden gives golden measure weights."""
        # w(n) = exp(-|n|²/φ) for mode norm squared input
        norm_sq = syn.state([0.0, 1.0, 4.0, 9.0])
        weights = norm_sq.exp_golden()
        
        expected = [
            math.exp(-0.0 * PHI_INV),
            math.exp(-1.0 * PHI_INV),
            math.exp(-4.0 * PHI_INV),
            math.exp(-9.0 * PHI_INV),
        ]
        actual = weights.to_list()
        
        for a, e in zip(actual, expected):
            assert abs(a - e) < 1e-10


class TestSum:
    """Test sum reduction."""
    
    def test_sum_total(self, syn):
        """Test total sum."""
        s = syn.state([1.0, 2.0, 3.0, 4.0])
        result = s.sum()
        assert abs(result - 10.0) < 1e-10
    
    def test_sum_axis(self, syn):
        """Test sum along axis."""
        s = syn.state([[1.0, 2.0], [3.0, 4.0]])
        result = s.sum(axis=0)  # Sum columns
        expected = [4.0, 6.0]
        actual = result.to_list()
        
        for a, e in zip(actual, expected):
            assert abs(a - e) < 1e-10


class TestMean:
    """Test mean reduction."""
    
    def test_mean_total(self, syn):
        """Test total mean."""
        s = syn.state([1.0, 2.0, 3.0, 4.0])
        result = s.mean()
        assert abs(result - 2.5) < 1e-10
    
    def test_mean_axis(self, syn):
        """Test mean along axis."""
        s = syn.state([[1.0, 2.0], [3.0, 4.0]])
        result = s.mean(axis=1)  # Mean of rows
        expected = [1.5, 3.5]
        actual = result.to_list()
        
        for a, e in zip(actual, expected):
            assert abs(a - e) < 1e-10


class TestLayerNorm:
    """Test layer normalization with golden target variance."""
    
    def test_layer_norm_standard(self, syn):
        """Test standard layer norm (golden_target=False)."""
        s = syn.state([1.0, 2.0, 3.0, 4.0, 5.0])
        result = s.layer_norm(golden_target=False)
        
        values = result.to_list()
        
        # Check mean ≈ 0
        mean = sum(values) / len(values)
        assert abs(mean) < 1e-10
        
        # Check variance ≈ 1
        var = sum((v - mean)**2 for v in values) / len(values)
        assert abs(var - 1.0) < 1e-5
    
    def test_layer_norm_golden_target(self, syn):
        """Test layer norm with golden target variance = 1/φ."""
        s = syn.state([1.0, 2.0, 3.0, 4.0, 5.0])
        result = s.layer_norm(golden_target=True)
        
        values = result.to_list()
        
        # Check mean ≈ 0
        mean = sum(values) / len(values)
        assert abs(mean) < 1e-10
        
        # Check variance ≈ 1/φ ≈ 0.618
        var = sum((v - mean)**2 for v in values) / len(values)
        assert abs(var - PHI_INV) < 1e-5, f"Variance {var} should be ≈ {PHI_INV}"


class TestDropout:
    """Test dropout regularization."""
    
    def test_dropout_inference_identity(self, syn):
        """Test dropout at inference is identity."""
        s = syn.state([1.0, 2.0, 3.0, 4.0])
        result = s.dropout(p=0.5, training=False)
        
        original = s.to_list()
        actual = result.to_list()
        
        for o, a in zip(original, actual):
            assert abs(o - a) < 1e-10
    
    def test_dropout_zero_probability(self, syn):
        """Test dropout with p=0 is identity."""
        s = syn.state([1.0, 2.0, 3.0, 4.0])
        result = s.dropout(p=0.0, training=True)
        
        original = s.to_list()
        actual = result.to_list()
        
        for o, a in zip(original, actual):
            assert abs(o - a) < 1e-10
    
    def test_dropout_deterministic(self, syn):
        """Test dropout with same seed produces same output."""
        s = syn.state([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
        
        result1 = s.dropout(p=0.5, training=True, seed=42)
        result2 = s.dropout(p=0.5, training=True, seed=42)
        
        actual1 = result1.to_list()
        actual2 = result2.to_list()
        
        for a1, a2 in zip(actual1, actual2):
            assert abs(a1 - a2) < 1e-10
    
    def test_dropout_scale(self, syn):
        """Test inverted dropout scaling."""
        s = syn.state([1.0] * 1000)  # Many elements for statistical test
        result = s.dropout(p=0.5, training=True, seed=123)
        
        values = result.to_list()
        non_zero = [v for v in values if v > 0]
        
        # Non-zero values should be scaled by 1/(1-p) = 2
        for v in non_zero:
            assert abs(v - 2.0) < 1e-10


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
