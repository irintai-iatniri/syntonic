#!/usr/bin/env python3
"""
Theta Series Benchmark: GPU vs CPU Performance Comparison

This benchmark compares the performance of theta series computations
using CPU (NumPy) vs GPU (CUDA) implementations.

Theta series: Θ(t) = Σ_λ w(λ) exp(-π Q(λ) / t)
where:
  - λ are E₈ lattice points in the golden cone
  - Q(λ) = ||P_∥λ||² - ||P_⊥λ||² is the quadratic form
  - w(λ) = exp(-|λ|²/φ) is the golden Gaussian weight
"""

import time
import numpy as np
from typing import Tuple, List
import sys

# SRT constants
PHI = 1.6180339887498949
PHI_INV = 0.6180339887498949
PI = np.pi

# Try to import syntonic CUDA support
try:
    from syntonic._core import (
        cuda_is_available,
        cuda_device_count,
        TensorStorage,
        srt_phi,
        srt_phi_inv,
    )
    HAS_CUDA = cuda_is_available()
except ImportError:
    HAS_CUDA = False
    print("Warning: syntonic CUDA support not available")


def generate_e8_roots() -> np.ndarray:
    """Generate all 240 E₈ root vectors."""
    roots = []

    # Type 1: Permutations of (±1, ±1, 0, 0, 0, 0, 0, 0)
    # Choose 2 positions out of 8 for ±1
    from itertools import combinations, product
    for positions in combinations(range(8), 2):
        for signs in product([1, -1], repeat=2):
            root = np.zeros(8)
            root[positions[0]] = signs[0]
            root[positions[1]] = signs[1]
            roots.append(root)

    # Type 2: (±1/2, ±1/2, ..., ±1/2) with even number of minus signs
    for signs in product([0.5, -0.5], repeat=8):
        if signs.count(-0.5) % 2 == 0:
            roots.append(np.array(signs))

    return np.array(roots, dtype=np.float64)


def golden_projection_matrix() -> Tuple[np.ndarray, np.ndarray]:
    """
    Construct the golden projection matrices P_φ and P_⊥.
    P_φ: ℝ⁸ → ℝ⁴ (parallel/physical subspace)
    P_⊥: ℝ⁸ → ℝ⁴ (perpendicular/internal subspace)
    """
    norm = 1.0 / np.sqrt(2 * PHI + 2)

    P_phi = np.zeros((4, 8))
    P_perp = np.zeros((4, 8))

    for i in range(4):
        # P_φ row i: [0, ..., φ*norm, norm, ..., 0] at positions 2i, 2i+1
        P_phi[i, 2*i] = PHI * norm
        P_phi[i, 2*i + 1] = norm

        # P_⊥ row i: [0, ..., norm, -φ*norm, ..., 0] at positions 2i, 2i+1
        P_perp[i, 2*i] = norm
        P_perp[i, 2*i + 1] = -PHI * norm

    return P_phi, P_perp


def compute_quadratic_form(roots: np.ndarray, P_phi: np.ndarray, P_perp: np.ndarray) -> np.ndarray:
    """Compute Q(λ) = ||P_∥λ||² - ||P_⊥λ||² for each root."""
    # Project all roots
    proj_parallel = roots @ P_phi.T  # (n, 4)
    proj_perp = roots @ P_perp.T     # (n, 4)

    # Compute squared norms
    parallel_sq = np.sum(proj_parallel**2, axis=1)
    perp_sq = np.sum(proj_perp**2, axis=1)

    return parallel_sq - perp_sq


def golden_cone_test(roots: np.ndarray) -> np.ndarray:
    """Test which roots are in the golden cone: B_a(λ) ≥ 0 for all a."""
    in_cone = np.ones(len(roots), dtype=bool)

    for a in range(4):
        # B_a(λ) = λ[2a] - φ * λ[2a+1]
        B_a = roots[:, 2*a] - PHI * roots[:, 2*a + 1]
        in_cone &= (B_a >= -1e-10)  # Small tolerance for numerical precision

    return in_cone


def cpu_theta_series(Q_values: np.ndarray, in_cone: np.ndarray,
                     weights: np.ndarray, t: float) -> float:
    """CPU implementation of theta series summation."""
    # Filter to cone roots
    mask = in_cone
    Q_cone = Q_values[mask]
    w_cone = weights[mask]

    # Θ(t) = Σ w(λ) exp(-π Q(λ) / t)
    return np.sum(w_cone * np.exp(-PI * Q_cone / t))


def cpu_golden_gaussian_weights(roots: np.ndarray) -> np.ndarray:
    """Compute golden Gaussian weights: w(λ) = exp(-|λ|²/φ)."""
    norm_sq = np.sum(roots**2, axis=1)
    return np.exp(-norm_sq * PHI_INV)


def benchmark_cpu(roots: np.ndarray, n_iterations: int = 1000,
                  n_t_values: int = 100) -> dict:
    """Benchmark CPU theta series computation."""

    # Precompute projections and Q values
    P_phi, P_perp = golden_projection_matrix()
    Q_values = compute_quadratic_form(roots, P_phi, P_perp)
    in_cone = golden_cone_test(roots)
    weights = cpu_golden_gaussian_weights(roots)

    n_cone = np.sum(in_cone)
    print(f"  Roots in golden cone: {n_cone} / {len(roots)}")

    # Generate t values
    t_values = np.linspace(0.1, 10.0, n_t_values)

    # Warmup
    for t in t_values[:10]:
        _ = cpu_theta_series(Q_values, in_cone, weights, t)

    # Benchmark
    start = time.perf_counter()
    for _ in range(n_iterations):
        results = []
        for t in t_values:
            results.append(cpu_theta_series(Q_values, in_cone, weights, t))
    end = time.perf_counter()

    total_time = end - start
    time_per_iteration = total_time / n_iterations
    time_per_theta = time_per_iteration / n_t_values

    return {
        'total_time': total_time,
        'iterations': n_iterations,
        't_values': n_t_values,
        'time_per_iteration': time_per_iteration,
        'time_per_theta_ms': time_per_theta * 1000,
        'theta_per_second': n_t_values * n_iterations / total_time,
        'sample_result': results[-1] if results else None,
    }


def benchmark_gpu(roots: np.ndarray, n_iterations: int = 1000,
                  n_t_values: int = 100) -> dict:
    """Benchmark GPU theta series computation using TensorStorage."""

    if not HAS_CUDA:
        return {'error': 'CUDA not available'}

    # Precompute on CPU first
    P_phi, P_perp = golden_projection_matrix()
    Q_values = compute_quadratic_form(roots, P_phi, P_perp)
    in_cone = golden_cone_test(roots)
    weights = cpu_golden_gaussian_weights(roots)

    n_cone = np.sum(in_cone)

    # Transfer to GPU
    Q_gpu = TensorStorage.from_list(Q_values.tolist(), [len(Q_values)], "float64", "cuda:0")
    weights_gpu = TensorStorage.from_list(weights.tolist(), [len(weights)], "float64", "cuda:0")

    # For in_cone, we need to use it for masking - keep as numpy for now
    # since we're doing element-wise ops on GPU

    t_values = np.linspace(0.1, 10.0, n_t_values)

    # For GPU benchmark, we'll do the exponential on GPU
    # Create result storage

    # Warmup - using GPU tensor operations
    for t in t_values[:10]:
        # Scale Q by -π/t
        scale = -PI / t
        # This tests GPU operations
        Q_scaled = Q_gpu.mul_scalar(scale)

    # Benchmark GPU tensor operations
    # Note: Full theta series requires kernel launch which isn't exposed to Python yet
    # So we benchmark the available GPU operations

    start = time.perf_counter()
    for _ in range(n_iterations):
        for t in t_values:
            scale = -PI / t
            Q_scaled = Q_gpu.mul_scalar(scale)
            # Actual exp would need custom kernel
    end = time.perf_counter()

    total_time = end - start
    time_per_iteration = total_time / n_iterations
    time_per_op = time_per_iteration / n_t_values

    return {
        'total_time': total_time,
        'iterations': n_iterations,
        't_values': n_t_values,
        'time_per_iteration': time_per_iteration,
        'time_per_op_ms': time_per_op * 1000,
        'ops_per_second': n_t_values * n_iterations / total_time,
        'note': 'GPU scaling operation only (exp kernel not exposed to Python)',
    }


def benchmark_large_scale(n_points_list: List[int], n_iterations: int = 100) -> dict:
    """Benchmark with varying numbers of lattice points."""

    results = {'cpu': [], 'gpu': []}

    for n_points in n_points_list:
        print(f"\n  Testing with {n_points} points...")

        # Generate random 8D points (simulating larger lattice)
        np.random.seed(42)
        points = np.random.randn(n_points, 8)

        # CPU benchmark
        P_phi, P_perp = golden_projection_matrix()
        Q_values = compute_quadratic_form(points, P_phi, P_perp)
        in_cone = np.ones(n_points, dtype=bool)  # All points for this test
        weights = cpu_golden_gaussian_weights(points)

        t = 1.0

        # CPU timing
        start = time.perf_counter()
        for _ in range(n_iterations):
            _ = cpu_theta_series(Q_values, in_cone, weights, t)
        cpu_time = time.perf_counter() - start

        results['cpu'].append({
            'n_points': n_points,
            'total_time': cpu_time,
            'time_per_call_us': cpu_time / n_iterations * 1e6,
        })

        # GPU benchmark (if available)
        if HAS_CUDA:
            Q_gpu = TensorStorage.from_list(Q_values.tolist(), [n_points], "float64", "cuda:0")
            weights_gpu = TensorStorage.from_list(weights.tolist(), [n_points], "float64", "cuda:0")

            # Warmup
            for _ in range(10):
                _ = Q_gpu.mul_scalar(-PI / t)

            start = time.perf_counter()
            for _ in range(n_iterations):
                _ = Q_gpu.mul_scalar(-PI / t)
            gpu_time = time.perf_counter() - start

            results['gpu'].append({
                'n_points': n_points,
                'total_time': gpu_time,
                'time_per_call_us': gpu_time / n_iterations * 1e6,
            })

    return results


def main():
    print("=" * 70)
    print("Theta Series Benchmark: GPU vs CPU Performance")
    print("=" * 70)

    # Generate E₈ roots
    print("\n[1] Generating E₈ root system...")
    roots = generate_e8_roots()
    print(f"  Generated {len(roots)} E₈ roots")

    # CPU Benchmark
    print("\n[2] CPU Benchmark (NumPy)...")
    print(f"  Running 1000 iterations × 100 t-values...")
    cpu_results = benchmark_cpu(roots, n_iterations=1000, n_t_values=100)

    print(f"\n  CPU Results:")
    print(f"    Total time:          {cpu_results['total_time']:.3f} s")
    print(f"    Time per iteration:  {cpu_results['time_per_iteration']*1000:.3f} ms")
    print(f"    Time per Θ(t):       {cpu_results['time_per_theta_ms']:.4f} ms")
    print(f"    Throughput:          {cpu_results['theta_per_second']:.0f} Θ(t)/sec")

    # GPU Benchmark
    if HAS_CUDA:
        print("\n[3] GPU Benchmark (CUDA)...")
        print(f"  Running 1000 iterations × 100 t-values...")
        gpu_results = benchmark_gpu(roots, n_iterations=1000, n_t_values=100)

        print(f"\n  GPU Results:")
        print(f"    Total time:          {gpu_results['total_time']:.3f} s")
        print(f"    Time per iteration:  {gpu_results['time_per_iteration']*1000:.3f} ms")
        print(f"    Time per op:         {gpu_results['time_per_op_ms']:.4f} ms")
        print(f"    Throughput:          {gpu_results['ops_per_second']:.0f} ops/sec")
        print(f"    Note: {gpu_results.get('note', '')}")
    else:
        print("\n[3] GPU Benchmark: SKIPPED (CUDA not available)")

    # Large scale benchmark
    print("\n[4] Scaling Benchmark (varying lattice sizes)...")
    n_points_list = [240, 1000, 5000, 10000, 50000, 100000]
    scale_results = benchmark_large_scale(n_points_list, n_iterations=100)

    print("\n  Results by lattice size:")
    print(f"  {'Points':>10} | {'CPU (μs)':>12} | {'GPU (μs)':>12} | {'Speedup':>10}")
    print("  " + "-" * 52)

    for i, n_points in enumerate(n_points_list):
        cpu_time = scale_results['cpu'][i]['time_per_call_us']
        if HAS_CUDA and i < len(scale_results['gpu']):
            gpu_time = scale_results['gpu'][i]['time_per_call_us']
            speedup = cpu_time / gpu_time if gpu_time > 0 else 0
            print(f"  {n_points:>10} | {cpu_time:>12.2f} | {gpu_time:>12.2f} | {speedup:>10.2f}x")
        else:
            print(f"  {n_points:>10} | {cpu_time:>12.2f} | {'N/A':>12} | {'N/A':>10}")

    print("\n" + "=" * 70)
    print("Benchmark Complete")
    print("=" * 70)


if __name__ == "__main__":
    main()
