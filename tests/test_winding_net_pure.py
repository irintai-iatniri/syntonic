
import sys
import os

# Ensure we can import syntonic
sys.path.append(os.path.join(os.getcwd(), 'python'))

from syntonic.nn.winding.winding_net_pure import PureWindingNet
from syntonic._core import WindingState, ResonantTensor

def test_pure_winding_net():
    print("Initializing PureWindingNet...")
    model = PureWindingNet(
        max_winding=3,
        base_dim=16,
        num_blocks=2,
        output_dim=2,
        precision=100
    )
    print("Model initialized.")

    # Create dummy windings
    windings = [
        WindingState(1, 0, 0, 0),
        WindingState(0, 1, 0, 0),
        WindingState(1, 1, 0, 0)
    ]
    print(f"Created {len(windings)} winding states.")

    # Forward pass
    print("Running forward pass...")
    try:
        output = model.forward(windings)
        print(f"Forward pass successful. Output shape: {output.shape}")
        
        # Check output content
        floats = output.to_floats()
        print(f"Output sample: {floats[:4]}...")
        
    except Exception as e:
        print(f"Forward pass failed: {e}")
        import traceback
        traceback.print_exc()
        return

    # Check stats
    stats = model.get_blockchain_stats()
    print("\nBlockchain Stats:")
    for k, v in stats.items():
        print(f"  {k}: {v}")
        
    # Check weights
    weights = model.get_weights()
    print(f"\nCollected {len(weights)} weight tensors.")
    
    print("\nSUCCESS: PureWindingNet is functional.")

if __name__ == "__main__":
    test_pure_winding_net()
