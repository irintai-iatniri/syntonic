#!/usr/bin/env python3
"""Compare synchronous vs SRT async GPU transfers."""

import time
import numpy as np
import syntonic

def mb(bytes_):
    return bytes_ / (1024 * 1024)

def run_case(n_elements: int = 8 * 1024 * 1024):
    print(f"\n=== Transfer Benchmark: {n_elements:,} f64 elements ({mb(n_elements*8):.2f} MB) ===")
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
    print(f"SRT Stats: transfers={int(stats['total_transfers'])}, bytes={mb(stats['total_bytes']):.2f} MB, avg={stats['avg_transfer_time_us']:.1f} μs")

if __name__ == "__main__":
    if not syntonic.cuda_is_available():
        print("CUDA not available; skipping benchmark.")
    else:
        # Warm-up small
        run_case(1 * 1024 * 1024)
        # Larger
        run_case(8 * 1024 * 1024)
