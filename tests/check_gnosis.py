"""
Test script for GnosisLayer
"""
from syntonic.nn.layers.gnosis import GnosisLayer
from syntonic._core import ResonantTensor

def test_gnosis():
    print("Testing Gnosis Layer...")
    
    # Initialize
    features = 16
    layer = GnosisLayer(features, retention_rate=0.5, threshold=0.3)
    print(f"Layer initialized: {layer}")
    
    # Create dummy input (Low Complexity, Low Syntony)
    # Uniform data = 0 variance = 0 complexity
    data_low = [0.5] * features * 2
    x_low = ResonantTensor(data_low, [2, features])
    
    out_low, g_low = layer.forward(x_low)
    print(f"Low Complexity Gnosis: {g_low:.4f} (Expected ~0.0)")
    
    # Create high complexity input
    # Alternating 0, 1 -> High Variance
    data_high = [1.0 if i % 2 == 0 else 0.0 for i in range(features * 2)]
    x_high = ResonantTensor(data_high, [2, features])
    # Assume syntony = 0.5 (default estimation)
    
    out_high, g_high = layer.forward(x_high)
    print(f"High Complexity Gnosis: {g_high:.4f} (Expected > 0)")
    
    if g_high > g_low:
        print("SUCCESS: High complexity yielded higher Gnosis.")
    else:
        print("FAILURE: Gnosis metric did not distinguish complexity.")
        
    print("Test Complete.")

if __name__ == "__main__":
    test_gnosis()
