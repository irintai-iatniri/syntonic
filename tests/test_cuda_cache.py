#!/usr/bin/env python3
"""
Test script to verify CUDA kernel caching performance improvement.
"""

import time
import numpy as np
import syntonic as syn

def test_cuda_performance():
    """Test CUDA performance with kernel caching."""
    print("Testing CUDA kernel caching performance...")

    # Create test tensors
    size = 1000000  # 1M elements
    a_data = np.random.randn(size).astype(np.float64)
    b_data = np.random.randn(size).astype(np.float64)

    try:
        # Create tensors on CUDA
        a = syn.state(a_data.tolist(), device=syn.cuda(0))
        b = syn.state(b_data.tolist(), device=syn.cuda(0))

        print(f"Created tensors of shape {a.shape} on {a.device}")

        # Warm up (first operation loads kernels)
        print("Warming up...")
        c = a + b
        result = c.to_list()
        print(f"Warm-up result shape: {len(result)}")

        # Benchmark multiple operations
        num_ops = 10
        print(f"Running {num_ops} add operations...")

        start_time = time.time()
        for i in range(num_ops):
            c = a + b
            # Force sync by accessing result
            _ = c.to_list()
        end_time = time.time()

        avg_time = (end_time - start_time) / num_ops
        print(".4f")
        print(".2f")

        # Test different operations
        print("Testing different operations...")
        operations = [
            ("add", lambda x, y: x + y),
            ("mul", lambda x, y: x * y),
            ("sub", lambda x, y: x - y),
        ]

        for op_name, op_func in operations:
            start_time = time.time()
            for i in range(5):  # Fewer ops for variety test
                c = op_func(a, b)
                _ = c.to_list()
            end_time = time.time()
            avg_time = (end_time - start_time) / 5
            print(".4f")

        print("CUDA kernel caching test completed successfully!")

    except Exception as e:
        print(f"CUDA test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_cuda_performance()