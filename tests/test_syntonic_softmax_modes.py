
import pytest
import math
from syntonic.core import ResonantTensor, syntonic_softmax

def test_learned_mode():
    """Test 'learned' mode with provided mode norms."""
    # Input: [batch=2, features=5]
    # Use simple values where we can predict behavior
    x_data = [1.0, 1.0, 1.0, 1.0, 1.0,  # Row 1
              2.0, 2.0, 2.0, 2.0, 2.0]  # Row 2 (shift shouldn't affect softmax much)
    x = ResonantTensor.from_floats(x_data, [2, 5], 100)
    
    # Mode norms: |n|² values
    # Let's say indices 0..4 have norms 0, 1, 4, 9, 16
    norms_data = [0.0, 1.0, 4.0, 9.0, 16.0]
    mode_norms = ResonantTensor.from_floats(norms_data, [5], 100)
    
    # Run softmax
    # w(n) = exp(-|n|²/φ)
    # n=0: exp(0) = 1
    # n=1: exp(-1/1.618) ≈ 0.539
    # Higher indices get much lower weights
    out = syntonic_softmax(x, mode='learned', mode_norms=mode_norms)
    
    # Check output shape
    assert out.shape == [2, 5]
    
    # Check probabilities sum to 1
    probs = out.to_floats()
    row1 = probs[0:5]
    row2 = probs[5:10]
    
    assert abs(sum(row1) - 1.0) < 1e-4
    assert abs(sum(row2) - 1.0) < 1e-4
    
    # Verify weighting effect: index 0 should be highest prob because it has lowest norm (highest weight)
    # even though inputs are equal
    assert row1[0] > row1[1]
    assert row1[1] > row1[2]
    
    print("\n✓ Learned mode passed")

def test_provided_mode():
    """Test 'provided' mode with pre-computed syntony weights."""
    batch = 2
    features = 3
    x = ResonantTensor.from_floats([1.0]*6, [batch, features], 100)
    
    # Precomputed weights (syntony)
    # Make feature 1 have high syntony -> high weight
    weights_data = [
        0.1, 0.9, 0.1,  # Row 1
        0.5, 0.5, 0.5   # Row 2
    ]
    syntony = ResonantTensor.from_floats(weights_data, [batch, features], 100)
    
    out = syntonic_softmax(x, mode='provided', syntony=syntony)
    
    probs = out.to_floats()
    row1 = probs[0:3]
    row2 = probs[3:6]
    
    # Row 1: middle element should be largest due to weight 0.9 vs 0.1
    assert row1[1] > row1[0]
    assert row1[1] > row1[2]
    
    # Row 2: all equal weights -> equal probs
    assert abs(row2[0] - row2[1]) < 1e-4
    assert abs(row2[1] - row2[2]) < 1e-4
    
    print("\n✓ Provided mode passed")

def test_identity_mode():
    """Test 'identity' mode (standard softmax behavior)."""
    x = ResonantTensor.from_floats([0.0, 0.0], [1, 2], 100)
    
    out = syntonic_softmax(x, mode='identity')
    
    probs = out.to_floats()
    # exp(0)/2*exp(0) = 0.5
    assert abs(probs[0] - 0.5) < 1e-4
    assert abs(probs[1] - 0.5) < 1e-4
    
    print("\n✓ Identity mode passed")

def test_error_handling():
    """Verify error messages for invalid inputs."""
    x = ResonantTensor.zeros([1, 5], 100)
    
    # 1. Invalid mode string
    with pytest.raises(ValueError, match="Unknown mode"):
        syntonic_softmax(x, mode='invalid_mode')
        
    # 2. Learned mode without mode_norms (Wait, Rust code might support default?)
    # Rust code: if mode_norms=None and mode='learned', it relies on state.mode_norms which initiates emptiness
    # Actually, syntonic_softmax_py creates state then forward.
    # State::new requires num_features for learned mode.
    # In syntonic_softmax_py:
    # let num_features = check mode_norms shape via Option
    # if mode_norms is None, num_features is None.
    # State::new(Learned, ..., num_features=None) -> Error "num_features required"
    # Wait, in syntonic_softmax_py line 656 calling State::new.
    # If mode_norms is None, num_features is None.
    # State::new checks: if mode == Learned && num_features.is_none() -> "num_features required"
    
    # Let's verify that error
    with pytest.raises(ValueError, match="num_features required"):
        syntonic_softmax(x, mode='learned')
        
    # 3. Provided mode without syntony
    with pytest.raises(ValueError, match="syntony tensor required"):
        syntonic_softmax(x, mode='provided')
        
    print("\n✓ Error handling passed")

if __name__ == "__main__":
    test_learned_mode()
    test_provided_mode()
    test_identity_mode()
    test_error_handling()