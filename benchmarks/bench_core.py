
import time
import numpy as np
import syntonic as syn
from syntonic import linalg
import pytest

def benchmark_op(name, func, n_iter=100):
    start = time.perf_counter()
    for _ in range(n_iter):
        func()
    end = time.perf_counter()
    avg_time = (end - start) / n_iter
    return avg_time

def run_benchmarks():
    size = 1000
    print(f"Benchmarking with Size: {size}x{size}")
    
    # Setup Data
    arr_a = np.random.randn(size, size)
    arr_b = np.random.randn(size, size)
    
    state_a = syn.state.from_numpy(arr_a)
    state_b = syn.state.from_numpy(arr_b)
    
    # --- Addition ---
    t_np_add = benchmark_op("NP Add", lambda: arr_a + arr_b)
    t_sy_add = benchmark_op("SY Add", lambda: state_a + state_b)
    print(f"Add: NumPy={t_np_add*1e3:.3f}ms, Syntonic={t_sy_add*1e3:.3f}ms (Ratio: {t_sy_add/t_np_add:.2f}x)")
    
    # --- Matmul ---
    # Use smaller size for matmul to check CPU vs OpenBLAS linkage overhead
    size_mm = 200
    arr_am = np.random.randn(size_mm, size_mm)
    arr_bm = np.random.randn(size_mm, size_mm)
    state_am = syn.state.from_numpy(arr_am)
    state_bm = syn.state.from_numpy(arr_bm)
    
    t_np_mm = benchmark_op("NP Matmul", lambda: arr_am @ arr_bm, n_iter=20)
    t_sy_mm = benchmark_op("SY Matmul", lambda: state_am @ state_bm, n_iter=20)
    print(f"Matmul: NumPy={t_np_mm*1e3:.3f}ms, Syntonic={t_sy_mm*1e3:.3f}ms (Ratio: {t_sy_mm/t_np_mm:.2f}x)")
    
    # --- Eig ---
    size_eig = 100
    arr_e = np.random.randn(size_eig, size_eig)
    state_e = syn.state.from_numpy(arr_e)
    
    t_np_eig = benchmark_op("NP Eig", lambda: np.linalg.eig(arr_e), n_iter=10)
    t_sy_eig = benchmark_op("SY Eig", lambda: linalg.eig(state_e), n_iter=10)
    print(f"Eig: NumPy={t_np_eig*1e3:.3f}ms, Syntonic={t_sy_eig*1e3:.3f}ms (Ratio: {t_sy_eig/t_np_eig:.2f}x)")

if __name__ == "__main__":
    run_benchmarks()
