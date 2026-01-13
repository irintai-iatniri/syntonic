
import pytest
import torch
import math
from syntonic.resonant.tensor import ResonantTensor, ResonantPhase

def test_bmm_cpu_exact():
    """Verify BMM on CPU using exact GoldenExact arithmetic."""
    print("\n--- Testing BMM CPU (Exact) ---")
    
    # Create [Batch=2, M=2, K=2]
    # Batch 1: Identity
    # Batch 2: Scaled Identity
    data_a = [
        1.0, 0.0,
        0.0, 1.0,
        2.0, 0.0,
        0.0, 2.0
    ]
    shape_a = [2, 2, 2]
    
    # Create [Batch=2, N=2, K=2]
    # We want to multiply A @ B^T.
    # If B is Identity, B^T is Identity.
    data_b = [
        1.0, 0.0,
        0.0, 1.0,
        0.5, 0.0,
        0.0, 0.5
    ]
    shape_b = [2, 2, 2] # N=2, K=2
    
    tensor_a = ResonantTensor(data_a, shape_a)
    tensor_b = ResonantTensor(data_b, shape_b)
    
    # Expected Result C = A @ B^T
    # Batch 1: I @ I = I
    # Batch 2: 2I @ (0.5I)^T = 2I @ 0.5I = I
    
    result = tensor_a.matmul(tensor_b)
    
    print(f"Result Shape: {result.shape}")
    assert result.shape == [2, 2, 2]
    
    floats = result.to_floats()
    print(f"Result Batch 1: {floats[0:4]}")
    print(f"Result Batch 2: {floats[4:8]}")
    
    # Verify Batch 1 is Identity
    assert math.isclose(floats[0], 1.0)
    assert math.isclose(floats[1], 0.0)
    assert math.isclose(floats[2], 0.0)
    assert math.isclose(floats[3], 1.0)
    
    # Verify Batch 2 is Identity (2.0 * 0.5 = 1.0)
    assert math.isclose(floats[4], 1.0)
    assert math.isclose(floats[5], 0.0)
    assert math.isclose(floats[6], 0.0)
    assert math.isclose(floats[7], 1.0)
    
    print("CPU BMM Passed!")

def test_bmm_gpu_flux():
    """Verify BMM on GPU using cuda_bmm_nt_f64."""
    if not torch.cuda.is_available():
        print("Skipping GPU test (no CUDA)")
        return

    print("\n--- Testing BMM GPU (Flux) ---")
    
    data_a = [
        1.0, 0.0,
        0.0, 1.0, 
        2.0, 0.0,
        0.0, 2.0
    ]
    shape_a = [2, 2, 2]
    
    data_b = [
        1.0, 0.0,
        0.0, 1.0,
        0.5, 0.0,
        0.0, 0.5
    ]
    shape_b = [2, 2, 2]
    
    tensor_a = ResonantTensor(data_a, shape_a)
    tensor_b = ResonantTensor(data_b, shape_b)
    
    # Move to Flux (GPU)
    tensor_a.wake_flux()
    tensor_b.wake_flux()
    
    assert tensor_a.phase == ResonantPhase.Flux
    assert tensor_b.phase == ResonantPhase.Flux
    
    # Matmul on GPU
    result = tensor_a.matmul(tensor_b)
    
    # Should perform BMM on GPU and return result (likely in Flux or Crystallized depending on implementation)
    # The current implementation returns Crystallized (downloads from GPU)
    # or if we are aggressive, maybe it stays in Flux?
    # Let's check.
    
    print(f"Result Phase: {result.phase}")
    print(f"Result Shape: {result.shape}")
    
    floats = result.to_floats()
    print(f"Result Batch 1: {floats[0:4]}")
    print(f"Result Batch 2: {floats[4:8]}")
    
    assert math.isclose(floats[0], 1.0)
    assert math.isclose(floats[3], 1.0)
    assert math.isclose(floats[4], 1.0)
    assert math.isclose(floats[7], 1.0)
    
    print("GPU BMM Passed!")

if __name__ == "__main__":
    try:
        test_bmm_cpu_exact()
        test_bmm_gpu_flux()
        print("\nAll BMM tests passed successfully.")
    except Exception as e:
        print(f"\nTEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        exit(1)
