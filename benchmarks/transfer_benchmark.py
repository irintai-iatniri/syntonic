#!/usr/bin/env python3
"""Compare synchronous vs SRT async GPU transfers."""

import time
import numpy as np
import syntonic

def mb(bytes_):
    return bytes_ / (1024 * 1024)

def run_h2d_case(n_elements: int = 8 * 1024 * 1024):
    print(f"\n=== H2D Transfer Benchmark: {n_elements:,} f64 elements ({mb(n_elements*8):.2f} MB) ===")
    data = np.arange(n_elements, dtype=np.float64)
    s = syntonic.state(data)

    # Sync transfer
    t0 = time.perf_counter()
    s_sync = s.cuda()
    t1 = time.perf_counter()
    sync_ms = (t1 - t0) * 1000
    print(f"Sync cuda(): {sync_ms:.2f} ms")

    # Async SRT transfer
    t0 = time.perf_counter()
    s_async = s.cuda_async()
    t1 = time.perf_counter()
    async_ms = (t1 - t0) * 1000
    print(f"SRT cuda_async(): {async_ms:.2f} ms")

    # Stats
    stats = syntonic.srt_transfer_stats(0)
    print(f"SRT H2D Stats: transfers={int(stats['total_transfers'])}, bytes={mb(stats['total_bytes']):.2f} MB, avg={stats['avg_transfer_time_us']:.1f} \u03bcs")

def run_d2h_case(n_elements: int = 8 * 1024 * 1024):
    print(f"\n=== D2H Transfer Benchmark: {n_elements:,} f64 elements ({mb(n_elements*8):.2f} MB) ===")
    
    # Create GPU tensor
    data = np.arange(n_elements, dtype=np.float64)
    s_gpu = syntonic.state(data).cuda()
    
    # Sync D2H transfer
    t0 = time.perf_counter()
    s_sync_cpu = s_gpu.cpu()
    t1 = time.perf_counter()
    sync_ms = (t1 - t0) * 1000
    print(f"Sync cpu(): {sync_ms:.2f} ms")
    
    # Async SRT D2H transfer
    t0 = time.perf_counter()
    s_async_cpu = s_gpu.cpu_async()
    t1 = time.perf_counter()
    async_ms = (t1 - t0) * 1000
    print(f"SRT cpu_async(): {async_ms:.2f} ms")
    
    # Stats
    stats = syntonic.srt_transfer_stats(0)
    print(f"SRT D2H Stats: transfers={int(stats['total_transfers'])}, bytes={mb(stats['total_bytes']):.2f} MB, avg={stats['avg_transfer_time_us']:.1f} \u03bcs")

def run_bidirectional_case(n_elements: int = 4 * 1024 * 1024):
    print(f"\n=== Bidirectional Transfer Benchmark: {n_elements:,} f64 elements ({mb(n_elements*8):.2f} MB) ===")
    
    data = np.arange(n_elements, dtype=np.float64)
    s = syntonic.state(data)
    
    # H2D + D2H round trip (Sync)
    t0 = time.perf_counter()
    s_round_sync = s.cuda().cpu()
    t1 = time.perf_counter()
    sync_round_ms = (t1 - t0) * 1000
    
    # H2D + D2H round trip (Async SRT)
    t0 = time.perf_counter()
    s_round_async = s.cuda_async().cpu_async()
    t1 = time.perf_counter()
    async_round_ms = (t1 - t0) * 1000
    
    print(f"Sync round-trip: {sync_round_ms:.2f} ms")
    print(f"SRT async round-trip: {async_round_ms:.2f} ms")
    print(f"Speedup: {sync_round_ms/async_round_ms:.2f}x")

if __name__ == "__main__":
    if not syntonic.cuda_is_available():
        print("CUDA not available; skipping benchmark.")
    else:
        # Warm-up
        run_h2d_case(1 * 1024 * 1024)
        run_d2h_case(1 * 1024 * 1024)
        
        # Main benchmarks
        run_h2d_case(8 * 1024 * 1024)
        run_d2h_case(8 * 1024 * 1024)
        run_bidirectional_case(4 * 1024 * 1024)
