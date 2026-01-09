"""
Verify PureWindingEncoder: Maps Standard Winding States to Resonant Embeddings.
"""

from syntonic.pure.winding_encoder import PureWindingEncoder
from syntonic._core import WindingState

def verify_encoder():
    print("Initializing PureWindingEncoder...")
    encoder = PureWindingEncoder(max_n=5, embed_dim=16)
    print(f"Encoder: {encoder}")
    
    # 1. Define sample winding states (Electron-like, Muon-like)
    # n7, n8, n9, n10
    electron = WindingState(1, 0, 0, 0)
    muon = WindingState(1, 1, 0, 0)
    vacuum = WindingState(0, 0, 0, 0)
    
    states = [electron, muon, vacuum]
    
    # 2. Encode
    print("\nEncoding states: [Electron, Muon, Vacuum]...")
    embedding_tensor = encoder.encode(states)
    
    print(f"Resulting Tensor: {embedding_tensor}")
    print(f"Shape: {embedding_tensor.shape}")
    print(f"Syntony: {embedding_tensor.syntony:.4f}")
    
    # 3. Basic checks
    if embedding_tensor.shape == [3, 16]:
        print("SUCCESS: Shape mismatch check passed.")
    else:
        print(f"FAILURE: Expected shape [3, 16], got {embedding_tensor.get_shape}")
        
    # Check if vacuum embedding is different from electron
    lattice = embedding_tensor.to_list()
    # electron data is at index 0-15
    # vacuum data is at index 32-47
    e_data = lattice[0:16]
    v_data = lattice[32:48]
    
    if e_data != v_data:
        print("SUCCESS: Embeddings are unique for different states.")
    else:
        # Note: with random init it's possible but unlikely they overlap 
        # unless both are zeroed by Golden Measure logic.
        print("WARNING: Embeddings for Electron and Vacuum are identical.")

if __name__ == "__main__":
    try:
        verify_encoder()
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
