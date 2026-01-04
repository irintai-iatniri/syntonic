import time
import numpy as np
import syntonic as syn

def benchmark_octonion_vs_state(n_iter=10000):
    print(f"Benchmarking Octonion vs State Multiplication ({n_iter} iterations)...")
    
    # Octonion setup
    o1 = syn.Octonion(1, 1, 1, 1, 1, 1, 1, 1)
    o2 = syn.Octonion(1, 0, 0, 0, 0, 0, 0, 0)
    
    start = time.perf_counter()
    for _ in range(n_iter):
        res = o1 * o2
    oct_time = time.perf_counter() - start
    print(f"Octonion multiplication: {oct_time:.6f}s ({oct_time/n_iter:.8f}s/op)")
    
    # State setup (requires element-wise ops or manual expansion for octonion logic)
    # Since State doesn't have Octonion logic built-in yet, we compare to
    # standard element-wise multiplication as a baseline for tensor overhead.
    s1 = syn.state([1]*8)
    s2 = syn.state([1]*8)
    
    start = time.perf_counter()
    for _ in range(n_iter):
        res = s1 * s2
    state_time = time.perf_counter() - start
    print(f"State element-wise (baseline): {state_time:.6f}s ({state_time/n_iter:.8f}s/op)")
    
    print(f"Overhead factor: {state_time / oct_time:.2f}x (Specific Octonion math is faster/slower than general Tensor)")

def benchmark_golden_number(n_iter=10000):
    print(f"\nBenchmarking GoldenNumber Evaluation ({n_iter} iterations)...")
    gn = syn.golden.phi()
    
    start = time.perf_counter()
    for _ in range(n_iter):
        val = gn.eval()
    t = time.perf_counter() - start
    print(f"GoldenNumber eval: {t:.6f}s ({t/n_iter:.8f}s/op)")

if __name__ == "__main__":
    benchmark_octonion_vs_state()
    benchmark_golden_number()
