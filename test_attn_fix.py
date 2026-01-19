import sys
import os
# Ensure we can import syntonic
sys.path.append(os.getcwd() + '/python')
sys.path.append(os.getcwd())

import traceback
from syntonic.nn.architectures.syntonic_attention import SyntonicAttention
from syntonic.nn.resonant_tensor import ResonantTensor

def test_1d_attention():
    print("Testing 1D Attention logic...")
    # Initialize Attention (using values from gnostic_ouroboros)
    model = SyntonicAttention(d_model=248, n_heads=8, d_head=31)
    
    # Create 1D input [248]
    x = ResonantTensor.randn([248])
    print(f"Input shape: {x.shape}")

    try:
        # Forward pass
        out = model(x, x, x)
        print(f"Success! Output shape: {out.shape}")
        return True
    except Exception as e:
        print(f"FAILURE: {e}")
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_1d_attention()
    if success:
        sys.exit(0)
    else:
        sys.exit(1)
