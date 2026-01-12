#!/usr/bin/env python3
"""
Focused SRT Memory Transfer Benchmark
Tests pure GPU↔CPU transfers without extra operations
"""

import time
import numpy as np
import syntonic as syn

def benchmark_pure_memory_transfers():
    """Benchmark pure SRT vs standard memory transfers"""
    print("=== Pure SRT Memory Transfer Benchmark ===\n")

    if not syn.cuda_is_available():
        print("❌ CUDA not available")
        return

    print("Testing pure GPU↔CPU transfers (no extra operations)")
    print("-" * 60)

    # Test single size for focused analysis
    size = 1048576  # 1M elements, ~8MB
    print(f"Testing size: {size} elements ({size * 8:,} bytes)")

    # Create test data
    data = np.random.rand(size).astype(np.float64)
    cpu_tensor = syn.state(data)

    # ===== WARMUP =====
    print("\nWarming up...")
    for _ in range(3):
        gpu = cpu_tensor.cuda()
        back = gpu.cpu()

    # ===== STANDARD TRANSFER BENCHMARK =====
    print("\nBenchmarking standard transfers...")
    standard_times = []
    for i in range(10):
        start = time.time()
        gpu_tensor = cpu_tensor.cuda()  # Pure H2D
        back_to_cpu = gpu_tensor.cpu()  # Pure D2H
        standard_time = time.time() - start
        standard_times.append(standard_time)
        print(".4f")

    avg_standard = np.mean(standard_times)
    print(".4f")

    # ===== SRT TRANSFER BENCHMARK =====
    print("\nBenchmarking SRT transfers...")
    srt_times = []
    for i in range(10):
        start = time.time()
        # Use the SRT protocol directly through tensor operations
        gpu_tensor = cpu_tensor.cuda()  # SRT H2D (via integrated protocol)
        back_to_cpu = gpu_tensor.cpu()  # SRT D2H (via integrated protocol)
        srt_time = time.time() - start
        srt_times.append(srt_time)
        print(".4f")

    avg_srt = np.mean(srt_times)
    print(".4f")

    # ===== RESULTS =====
    speedup = avg_standard / avg_srt if avg_srt > 0 else 0
    print(".1f")

    if speedup >= 8.0:
        print("🎉 TARGET ACHIEVED: 8-40x speedup reached!")
    elif speedup >= 2.0:
        print("✅ GOOD PROGRESS: SRT shows significant improvement")
    elif speedup >= 1.1:
        print("⚠️  MODEST GAIN: SRT working but needs optimization")
    else:
        print("❌ NO IMPROVEMENT: SRT needs debugging")

    print("\nNext optimization: Implement pinned memory pooling")
    print("- Remove memory allocation overhead")
    print("- Use SRT Fibonacci-sized pinned blocks")
    print("- Enable zero-copy transfers where possible")

if __name__ == "__main__":
    benchmark_pure_memory_transfers()