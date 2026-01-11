#!/usr/bin/env python3
"""
CUDA Performance Benchmark for Syntonic

This benchmark tests CUDA performance of Syntonic tensor operations
against NumPy and PyTorch baselines.

Usage:
    python cuda_benchmark.py [--sizes SIZES] [--iterations ITERS]
"""

import time
import numpy as np
import argparse
import sys
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass

# Try to import PyTorch
try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    print("Warning: PyTorch not available")

# Try to import Syntonic
try:
    import syntonic as syn
    HAS_SYNTONIC = True
except ImportError:
    HAS_SYNTONIC = False
    print("Warning: Syntonic not available")

# Check CUDA availability
try:
    if HAS_SYNTONIC:
        CUDA_AVAILABLE = syn.cuda_is_available()
        CUDA_DEVICE_COUNT = syn.cuda_device_count() if CUDA_AVAILABLE else 0
    else:
        CUDA_AVAILABLE = False
        CUDA_DEVICE_COUNT = 0
except:
    CUDA_AVAILABLE = False
    CUDA_DEVICE_COUNT = 0

@dataclass
class BenchmarkResult:
    """Container for benchmark results."""
    operation: str
    library: str
    size: Tuple[int, ...]
    dtype: str
    device: str
    time_ms: float
    iterations: int
    throughput: float = 0.0  # operations per second
    speedup: float = 0.0     # speedup over CPU version
    error: Optional[str] = None

class CUDABenchmark:
    """CUDA benchmark suite for tensor libraries."""

    def __init__(self, sizes: List[int], iterations: int = 100):
        self.sizes = sizes
        self.iterations = iterations
        self.results: List[BenchmarkResult] = []

        # Data types to test
        self.dtypes = ['float32', 'float64']

        print(f"CUDA Benchmark Configuration:")
        print(f"  Sizes: {sizes}")
        print(f"  Iterations: {iterations}")
        print(f"  CUDA Available: {CUDA_AVAILABLE}")
        if CUDA_AVAILABLE:
            print(f"  CUDA Devices: {CUDA_DEVICE_COUNT}")
        print()

    def create_arrays(self, size: Tuple[int, ...], dtype: str, library: str, use_cuda: bool = False) -> Tuple[Any, Any]:
        """Create test arrays for given library and configuration."""
        np_dtype = getattr(np, dtype)

        if library == 'numpy':
            a = np.random.randn(*size).astype(np_dtype)
            b = np.random.randn(*size).astype(np_dtype)
            return a, b

        elif library == 'torch':
            if not HAS_TORCH:
                raise ImportError("PyTorch not available")
            torch_dtype = getattr(torch, dtype)
            device = 'cuda' if use_cuda else 'cpu'
            a = torch.randn(*size, dtype=torch_dtype, device=device)
            b = torch.randn(*size, dtype=torch_dtype, device=device)
            return a, b

        elif library == 'syntonic':
            if not HAS_SYNTONIC:
                raise ImportError("Syntonic not available")
            device_str = 'cuda' if use_cuda else 'cpu'
            device = syn.device(device_str)
            a_np = np.random.randn(*size).astype(np_dtype)
            b_np = np.random.randn(*size).astype(np_dtype)
            a = syn.state.from_numpy(a_np).to(device)
            b = syn.state.from_numpy(b_np).to(device)
            return a, b

        else:
            raise ValueError(f"Unknown library: {library}")

    def benchmark_operation(self, operation: str, func, size: Tuple[int, ...],
                          dtype: str, library: str, device: str) -> BenchmarkResult:
        """Benchmark a single operation."""
        try:
            # Warmup
            for _ in range(min(10, self.iterations // 10)):
                _ = func()

            # Benchmark
            start_time = time.perf_counter()
            for _ in range(self.iterations):
                result = func()
            end_time = time.perf_counter()

            total_time = end_time - start_time
            time_per_op = total_time / self.iterations
            time_ms = time_per_op * 1000

            # Calculate throughput (operations per second)
            throughput = self.iterations / total_time

            return BenchmarkResult(
                operation=operation,
                library=library,
                size=size,
                dtype=dtype,
                device=device,
                time_ms=time_ms,
                iterations=self.iterations,
                throughput=throughput
            )

        except Exception as e:
            return BenchmarkResult(
                operation=operation,
                library=library,
                size=size,
                dtype=dtype,
                device=device,
                time_ms=0.0,
                iterations=self.iterations,
                error=str(e)
            )

    def benchmark_arithmetic_cuda(self):
        """Benchmark basic arithmetic operations on CUDA."""
        print("Benchmarking CUDA Arithmetic Operations...")

        operations = [
            ('add', lambda a, b: a + b),
            ('subtract', lambda a, b: a - b),
            ('multiply', lambda a, b: a * b),
            ('divide', lambda a, b: a / b),
        ]

        libraries = ['torch', 'syntonic'] if HAS_SYNTONIC and HAS_TORCH else []

        for size in self.sizes:
            for dtype in self.dtypes:
                for lib in libraries:
                    try:
                        # CPU version for comparison
                        a_cpu, b_cpu = self.create_arrays((size, size), dtype, lib, use_cuda=False)
                        cpu_result = self.benchmark_operation(
                            f"{size}x{size}_cpu",
                            lambda: operations[0][1](a_cpu, b_cpu),  # Just add for CPU baseline
                            (size, size), dtype, lib, 'cpu'
                        )

                        # CUDA version
                        a_cuda, b_cuda = self.create_arrays((size, size), dtype, lib, use_cuda=True)
                        cuda_result = self.benchmark_operation(
                            f"{size}x{size}_cuda",
                            lambda: operations[0][1](a_cuda, b_cuda),  # Just add for CUDA
                            (size, size), dtype, lib, 'cuda'
                        )

                        # Calculate speedup
                        if cpu_result.time_ms > 0:
                            cuda_result.speedup = cpu_result.time_ms / cuda_result.time_ms

                        self.results.append(cpu_result)
                        self.results.append(cuda_result)

                    except Exception as e:
                        print(f"Error benchmarking {lib} {dtype} {size}x{size}: {e}")

    def benchmark_matmul_cuda(self):
        """Benchmark matrix multiplication on CUDA."""
        print("Benchmarking CUDA Matrix Multiplication...")

        libraries = ['torch', 'syntonic'] if HAS_SYNTONIC and HAS_TORCH else []

        for size in self.sizes:
            for dtype in self.dtypes:
                for lib in libraries:
                    try:
                        # Use smaller sizes for matmul to avoid memory issues
                        matmul_size = min(size, 500)

                        # CPU version
                        a_cpu, b_cpu = self.create_arrays((matmul_size, matmul_size), dtype, lib, use_cuda=False)
                        cpu_result = self.benchmark_operation(
                            f"matmul_{matmul_size}x{matmul_size}_cpu",
                            lambda: a_cpu @ b_cpu,
                            (matmul_size, matmul_size), dtype, lib, 'cpu'
                        )

                        # CUDA version
                        a_cuda, b_cuda = self.create_arrays((matmul_size, matmul_size), dtype, lib, use_cuda=True)
                        cuda_result = self.benchmark_operation(
                            f"matmul_{matmul_size}x{matmul_size}_cuda",
                            lambda: a_cuda @ b_cuda,
                            (matmul_size, matmul_size), dtype, lib, 'cuda'
                        )

                        # Calculate speedup
                        if cpu_result.time_ms > 0:
                            cuda_result.speedup = cpu_result.time_ms / cuda_result.time_ms

                        self.results.append(cpu_result)
                        self.results.append(cuda_result)

                    except Exception as e:
                        print(f"Error benchmarking {lib} matmul {size}x{size}: {e}")

    def benchmark_element_wise_cuda(self):
        """Benchmark element-wise operations on CUDA."""
        print("Benchmarking CUDA Element-wise Operations...")

        operations = [
            ('exp', lambda a: torch.exp(a) if hasattr(a, 'exp') else a.exp()),
            ('tanh', lambda a: torch.tanh(a) if hasattr(a, 'tanh') else a.exp_golden()),  # Use exp_golden for syntonic
        ]

        libraries = ['torch', 'syntonic'] if HAS_SYNTONIC and HAS_TORCH else []

        for size in self.sizes:
            for dtype in self.dtypes:
                for lib in libraries:
                    try:
                        # CPU version
                        a_cpu, _ = self.create_arrays((size, size), dtype, lib, use_cuda=False)
                        cpu_result = self.benchmark_operation(
                            f"exp_{size}x{size}_cpu",
                            lambda: operations[0][1](a_cpu),
                            (size, size), dtype, lib, 'cpu'
                        )

                        # CUDA version
                        a_cuda, _ = self.create_arrays((size, size), dtype, lib, use_cuda=True)
                        cuda_result = self.benchmark_operation(
                            f"exp_{size}x{size}_cuda",
                            lambda: operations[0][1](a_cuda),
                            (size, size), dtype, lib, 'cuda'
                        )

                        # Calculate speedup
                        if cpu_result.time_ms > 0:
                            cuda_result.speedup = cpu_result.time_ms / cuda_result.time_ms

                        self.results.append(cpu_result)
                        self.results.append(cuda_result)

                    except Exception as e:
                        print(f"Error benchmarking {lib} element-wise {size}x{size}: {e}")

    def run_all_benchmarks(self):
        """Run all CUDA benchmark categories."""
        print("=" * 80)
        print("STARTING CUDA PERFORMANCE BENCHMARK")
        print("=" * 80)

        if not CUDA_AVAILABLE:
            print("CUDA not available! Skipping CUDA benchmarks.")
            return

        start_time = time.time()

        self.benchmark_arithmetic_cuda()
        self.benchmark_matmul_cuda()
        self.benchmark_element_wise_cuda()

        end_time = time.time()
        total_time = end_time - start_time

        print(f"\nBenchmark completed in {total_time:.2f} seconds")
        print(f"Total operations benchmarked: {len(self.results)}")

    def print_summary(self):
        """Print a summary of CUDA benchmark results."""
        print("\n" + "=" * 80)
        print("CUDA BENCHMARK SUMMARY")
        print("=" * 80)

        # Group results by operation and library
        summary = {}
        for result in self.results:
            if result.error:
                continue

            key = f"{result.operation}_{result.library}"
            if key not in summary:
                summary[key] = {'cpu': None, 'cuda': None}

            if result.device == 'cpu':
                summary[key]['cpu'] = result
            elif result.device == 'cuda':
                summary[key]['cuda'] = result

        # Print performance comparison
        for op_key, devices in summary.items():
            cpu_result = devices['cpu']
            cuda_result = devices['cuda']

            if cpu_result and cuda_result:
                speedup = cuda_result.speedup
                print(f"{op_key}:")
                print(f"  CPU:  {cpu_result.time_ms:>8.1f} ms/op")
                print(f"  CUDA: {cuda_result.time_ms:>8.1f} ms/op")
                print(f"  Speedup: {speedup:>5.2f}x")
                print()


def main():
    parser = argparse.ArgumentParser(description="CUDA Performance Benchmark")
    parser.add_argument('--sizes', type=str, default='100,500,1000',
                       help='Comma-separated list of tensor sizes (default: 100,500,1000)')
    parser.add_argument('--iterations', type=int, default=100,
                       help='Number of iterations per benchmark (default: 100)')

    args = parser.parse_args()

    # Parse sizes
    try:
        sizes = [int(s.strip()) for s in args.sizes.split(',')]
    except ValueError:
        print("Error: Invalid sizes format. Use comma-separated integers.")
        sys.exit(1)

    # Check if required libraries are available
    if not HAS_SYNTONIC:
        print("Error: Syntonic library not available. Please install it first.")
        sys.exit(1)

    if not CUDA_AVAILABLE:
        print("Error: CUDA not available on this system.")
        sys.exit(1)

    # Create and run benchmark
    benchmark = CUDABenchmark(sizes, args.iterations)
    benchmark.run_all_benchmarks()
    benchmark.print_summary()


if __name__ == "__main__":
    main()