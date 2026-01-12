#!/usr/bin/env python3
"""
Compare CPU vs CUDA performance to verify caching improvement.
"""

import time
import numpy as np
import syntonic as syn

def benchmark_cpu_vs_cuda():
    """Compare CPU and CUDA performance."""
    print("Benchmarking CPU vs CUDA performance...")

    # Create test data
    size = 100000  # 100K elements for faster testing
    a_data = np.random.randn(size).astype(np.float64).tolist()
    b_data = np.random.randn(size).astype(np.float64).tolist()

    # CPU test
    print("Testing CPU performance...")
    a_cpu = syn.state(a_data, device=syn.cpu)
    b_cpu = syn.state(b_data, device=syn.cpu)

    num_ops = 5
    start_time = time.time()
    for i in range(num_ops):
        c = a_cpu + b_cpu
        _ = c.to_list()  # Force computation
    cpu_time = (time.time() - start_time) / num_ops
    print(".4f")

    # CUDA test (with caching)
    print("Testing CUDA performance (with caching)...")
    a_cuda = syn.state(a_data, device=syn.cuda(0))
    b_cuda = syn.state(b_data, device=syn.cuda(0))

    # Warm up (loads kernels)
    c_warmup = a_cuda + b_cuda
    _ = c_warmup.to_list()

    # Benchmark cached operations
    start_time = time.time()
    for i in range(num_ops):
        c = a_cuda + b_cuda
        _ = c.to_list()  # Force computation
    cuda_time = (time.time() - start_time) / num_ops
    print(".4f")

    speedup = cpu_time / cuda_time
    print(".1f")

    if speedup > 1:
        print("✅ CUDA with caching is faster than CPU!")
    else:
        print("❌ CUDA slower than CPU - caching may not be working")

    return speedup

if __name__ == "__main__":
    benchmark_cpu_vs_cuda()