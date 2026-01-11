#!/usr/bin/env python3
"""
Comprehensive Performance Benchmark: Syntonic vs NumPy vs PyTorch

This benchmark suite compares the performance of Syntonic tensor operations
against NumPy and PyTorch across multiple dimensions:

1. Basic Arithmetic Operations (add, subtract, multiply, divide)
2. Matrix Operations (matmul, transpose, reshape)
3. Linear Algebra (eig, svd, solve, cholesky)
4. Element-wise Operations (exp, log, sin, cos, sqrt)
5. Memory Operations (creation, copying, conversion)
6. Different Data Types (float32, float64, complex64, complex128)
7. Scaling Performance (varying tensor sizes)
8. CPU vs GPU Performance (where applicable)

Usage:
    python comprehensive_benchmark.py [--sizes SIZES] [--iterations ITERS] [--gpu]

Example:
    python comprehensive_benchmark.py --sizes 100,500,1000 --iterations 100 --gpu
"""

import time
import numpy as np
import argparse
import sys
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass
import json
import os

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
    from syntonic import linalg
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
    memory_mb: float = 0.0   # memory usage in MB
    error: Optional[str] = None

class ComprehensiveBenchmark:
    """Comprehensive benchmark suite for tensor libraries."""

    def __init__(self, sizes: List[int], iterations: int = 100, use_gpu: bool = False):
        self.sizes = sizes
        self.iterations = iterations
        # Enable GPU by default if CUDA is available
        self.use_gpu = use_gpu or CUDA_AVAILABLE
        self.results: List[BenchmarkResult] = []

        # Data types to test
        self.dtypes = ['float32', 'float64']
        if HAS_TORCH:
            self.dtypes.extend(['complex64', 'complex128'])

        print(f"Benchmark Configuration:")
        print(f"  Sizes: {sizes}")
        print(f"  Iterations: {iterations}")
        print(f"  GPU: {'Enabled' if self.use_gpu else 'Disabled'}")
        print(f"  Libraries: NumPy, {'PyTorch, ' if HAS_TORCH else ''}{'Syntonic' if HAS_SYNTONIC else ''}")
        print(f"  Data Types: {self.dtypes}")
        print()

    def create_arrays(self, size: Tuple[int, ...], dtype: str, library: str) -> Tuple[Any, Any]:
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
            device = 'cuda' if self.use_gpu else 'cpu'
            a = torch.randn(*size, dtype=torch_dtype, device=device)
            b = torch.randn(*size, dtype=torch_dtype, device=device)
            return a, b

        elif library == 'syntonic':
            if not HAS_SYNTONIC:
                raise ImportError("Syntonic not available")
            device_str = 'cuda' if self.use_gpu else 'cpu'
            device = syn.device(device_str)
            a_np = np.random.randn(*size).astype(np_dtype)
            b_np = np.random.randn(*size).astype(np_dtype)
            a = syn.State.from_numpy(a_np).to(device)
            b = syn.State.from_numpy(b_np).to(device)
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

            # Estimate memory usage (rough approximation)
            if isinstance(size, tuple):
                elements = np.prod(size)
                bytes_per_element = {'float32': 4, 'float64': 8, 'complex64': 8, 'complex128': 16}[dtype]
                memory_mb = (elements * bytes_per_element * 2) / (1024 * 1024)  # *2 for a and b

            return BenchmarkResult(
                operation=operation,
                library=library,
                size=size,
                dtype=dtype,
                device=device,
                time_ms=time_ms,
                iterations=self.iterations,
                throughput=throughput,
                memory_mb=memory_mb
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

    def benchmark_arithmetic_ops(self):
        """Benchmark basic arithmetic operations."""
        print("Benchmarking Arithmetic Operations...")

        operations = [
            ('add', lambda a, b: a + b),
            ('subtract', lambda a, b: a - b),
            ('multiply', lambda a, b: a * b),
            ('divide', lambda a, b: a / b),
        ]

        libraries = ['numpy']
        if HAS_TORCH:
            libraries.append('torch')
        if HAS_SYNTONIC:
            libraries.append('syntonic')

        for size in self.sizes:
            for dtype in self.dtypes:
                for lib in libraries:
                    try:
                        a, b = self.create_arrays((size, size), dtype, lib)
                        device = 'cuda' if self.use_gpu and lib in ['torch', 'syntonic'] else 'cpu'

                        for op_name, op_func in operations:
                            result = self.benchmark_operation(
                                f"{op_name}_{size}x{size}",
                                lambda: op_func(a, b),
                                (size, size), dtype, lib, device
                            )
                            self.results.append(result)

                    except Exception as e:
                        print(f"Error benchmarking {lib} {dtype} {size}x{size}: {e}")

    def benchmark_matrix_ops(self):
        """Benchmark matrix operations."""
        print("Benchmarking Matrix Operations...")

        operations = [
            ('matmul', lambda a, b: a @ b),
            ('transpose', lambda a, b: a.T),
            ('reshape', lambda a, b: a.reshape(-1)),
        ]

        libraries = ['numpy']
        if HAS_TORCH:
            libraries.append('torch')
        if HAS_SYNTONIC:
            libraries.append('syntonic')

        for size in self.sizes:
            for dtype in self.dtypes:
                for lib in libraries:
                    try:
                        a, b = self.create_arrays((size, size), dtype, lib)
                        device = 'cuda' if self.use_gpu and lib in ['torch', 'syntonic'] else 'cpu'

                        for op_name, op_func in operations:
                            if op_name == 'matmul':
                                # For matmul, use smaller sizes to avoid memory issues
                                matmul_size = min(size, 500)
                                a_mat, b_mat = self.create_arrays((matmul_size, matmul_size), dtype, lib)
                                result = self.benchmark_operation(
                                    f"{op_name}_{matmul_size}x{matmul_size}",
                                    lambda: op_func(a_mat, b_mat),
                                    (matmul_size, matmul_size), dtype, lib, device
                                )
                            else:
                                result = self.benchmark_operation(
                                    f"{op_name}_{size}x{size}",
                                    lambda: op_func(a, None),
                                    (size, size), dtype, lib, device
                                )
                            self.results.append(result)

                    except Exception as e:
                        print(f"Error benchmarking {lib} {dtype} {size}x{size}: {e}")

    def benchmark_element_wise_ops(self):
        """Benchmark element-wise operations."""
        print("Benchmarking Element-wise Operations...")

        operations = [
            ('exp', lambda a: np.exp(a)),
            ('log', lambda a: np.log(np.abs(a) + 1e-8)),
            ('sin', lambda a: np.sin(a)),
            ('cos', lambda a: np.cos(a)),
            ('sqrt', lambda a: np.sqrt(np.abs(a))),
        ]

        libraries = ['numpy']
        if HAS_TORCH:
            libraries.append('torch')
            operations.extend([
                ('tanh', lambda a: torch.tanh(a)),
                ('relu', lambda a: torch.relu(a)),
            ])
        if HAS_SYNTONIC:
            libraries.append('syntonic')

        for size in self.sizes:
            for dtype in self.dtypes:
                for lib in libraries:
                    try:
                        a, _ = self.create_arrays((size, size), dtype, lib)
                        device = 'cuda' if self.use_gpu and lib in ['torch', 'syntonic'] else 'cpu'

                        for op_name, op_func in operations:
                            if lib == 'torch' and op_name in ['exp', 'log', 'sin', 'cos', 'sqrt']:
                                torch_op = getattr(torch, op_name)
                                result = self.benchmark_operation(
                                    f"{op_name}_{size}x{size}",
                                    lambda: torch_op(a),
                                    (size, size), dtype, lib, device
                                )
                            elif lib == 'syntonic':
                                # For syntonic, we need to check what operations are available
                                # For now, skip element-wise ops that might not be implemented
                                continue
                            else:
                                result = self.benchmark_operation(
                                    f"{op_name}_{size}x{size}",
                                    lambda: op_func(a),
                                    (size, size), dtype, lib, device
                                )
                            self.results.append(result)

                    except Exception as e:
                        print(f"Error benchmarking {lib} {dtype} {size}x{size}: {e}")

    def benchmark_linalg_ops(self):
        """Benchmark linear algebra operations."""
        print("Benchmarking Linear Algebra Operations...")

        operations = [
            ('eig', lambda a: np.linalg.eig(a)),
            ('svd', lambda a: np.linalg.svd(a)),
            ('solve', lambda a, b: np.linalg.solve(a, b)),
        ]

        libraries = ['numpy']
        if HAS_TORCH:
            libraries.append('torch')
        if HAS_SYNTONIC:
            libraries.append('syntonic')

        # Use smaller sizes for linalg operations
        linalg_sizes = [s for s in self.sizes if s <= 500]

        for size in linalg_sizes:
            for dtype in ['float64']:  # Most linalg ops work best with float64
                for lib in libraries:
                    try:
                        a, b = self.create_arrays((size, size), dtype, lib)
                        device = 'cuda' if self.use_gpu and lib in ['torch', 'syntonic'] else 'cpu'

                        for op_name, op_func in operations:
                            if lib == 'syntonic':
                                # Use syntonic's linalg operations
                                if op_name == 'eig':
                                    syn_op = lambda a: a.eig()
                                elif op_name == 'svd':
                                    syn_op = lambda a: a.svd()
                                elif op_name == 'solve':
                                    b_vec = self.create_arrays((size,), dtype, lib)[0]
                                    syn_op = lambda a: linalg.solve(a, b_vec)
                                result = self.benchmark_operation(
                                    f"{op_name}_{size}x{size}",
                                    lambda: syn_op(a),
                                    (size, size), dtype, lib, device
                                )
                            elif op_name == 'solve':
                                # For solve, b should be a vector or matrix with compatible dimensions
                                b_vec = self.create_arrays((size,), dtype, lib)[0]
                                result = self.benchmark_operation(
                                    f"{op_name}_{size}x{size}",
                                    lambda: op_func(a, b_vec),
                                    (size, size), dtype, lib, device
                                )
                            else:
                                result = self.benchmark_operation(
                                    f"{op_name}_{size}x{size}",
                                    lambda: op_func(a),
                                    (size, size), dtype, lib, device
                                )
                            self.results.append(result)

                    except Exception as e:
                        print(f"Error benchmarking {lib} {dtype} {size}x{size}: {e}")

    def benchmark_memory_ops(self):
        """Benchmark memory operations."""
        print("Benchmarking Memory Operations...")

        libraries = ['numpy']
        if HAS_TORCH:
            libraries.append('torch')
        if HAS_SYNTONIC:
            libraries.append('syntonic')

        for size in self.sizes:
            for dtype in self.dtypes:
                for lib in libraries:
                    try:
                        device = 'cuda' if self.use_gpu and lib in ['torch', 'syntonic'] else 'cpu'

                        # Creation benchmark
                        if lib == 'numpy':
                            create_func = lambda: np.random.randn(* (size, size)).astype(getattr(np, dtype))
                        elif lib == 'torch':
                            torch_dtype = getattr(torch, dtype)
                            create_func = lambda: torch.randn(*(size, size), dtype=torch_dtype, device=device)
                        elif lib == 'syntonic':
                            create_func = lambda: syn.State.random((size, size), dtype=dtype, device=device)

                        result = self.benchmark_operation(
                            f"create_{size}x{size}",
                            create_func,
                            (size, size), dtype, lib, device
                        )
                        self.results.append(result)

                        # Copy benchmark (if applicable)
                        if lib in ['torch', 'syntonic']:
                            a, _ = self.create_arrays((size, size), dtype, lib)
                            if lib == 'torch':
                                copy_func = lambda: a.clone()
                            else:
                                copy_func = lambda: a.copy()

                            result = self.benchmark_operation(
                                f"copy_{size}x{size}",
                                copy_func,
                                (size, size), dtype, lib, device
                            )
                            self.results.append(result)

                    except Exception as e:
                        print(f"Error benchmarking {lib} {dtype} {size}x{size}: {e}")

    def run_all_benchmarks(self):
        """Run all benchmark categories."""
        print("=" * 80)
        print("STARTING COMPREHENSIVE PERFORMANCE BENCHMARK")
        print("=" * 80)

        start_time = time.time()

        self.benchmark_arithmetic_ops()
        self.benchmark_matrix_ops()
        self.benchmark_element_wise_ops()
        self.benchmark_linalg_ops()
        self.benchmark_memory_ops()

        end_time = time.time()
        total_time = end_time - start_time

        print(f"\nBenchmark completed in {total_time:.2f} seconds")
        print(f"Total operations benchmarked: {len(self.results)}")

    def save_results(self, filename: str = "benchmark_results.json"):
        """Save benchmark results to JSON file."""
        results_dict = {
            'metadata': {
                'timestamp': time.time(),
                'sizes': self.sizes,
                'iterations': self.iterations,
                'gpu_enabled': self.use_gpu,
                'libraries': ['numpy'] + (['torch'] if HAS_TORCH else []) + (['syntonic'] if HAS_SYNTONIC else []),
                'cuda_available': CUDA_AVAILABLE,
                'cuda_devices': CUDA_DEVICE_COUNT
            },
            'results': [
                {
                    'operation': r.operation,
                    'library': r.library,
                    'size': r.size,
                    'dtype': r.dtype,
                    'device': r.device,
                    'time_ms': r.time_ms,
                    'iterations': r.iterations,
                    'throughput': r.throughput,
                    'memory_mb': r.memory_mb,
                    'error': r.error
                }
                for r in self.results
            ]
        }

        with open(filename, 'w') as f:
            json.dump(results_dict, f, indent=2)

        print(f"Results saved to {filename}")

    def print_summary(self):
        """Print a summary of benchmark results."""
        print("\n" + "=" * 80)
        print("BENCHMARK SUMMARY")
        print("=" * 80)

        # Group results by operation and size
        summary = {}
        for result in self.results:
            if result.error:
                continue

            key = f"{result.operation}_{result.dtype}"
            if key not in summary:
                summary[key] = {}
            if result.library not in summary[key]:
                summary[key][result.library] = []
            summary[key][result.library].append(result)

        # Print performance comparison for each operation
        for op_key, lib_results in summary.items():
            print(f"\n{op_key.upper()}:")
            print("-" * 40)

            # Find best performer for each operation
            best_time = float('inf')
            best_lib = None

            for lib, results in lib_results.items():
                if results:
                    avg_time = np.mean([r.time_ms for r in results])
                    if avg_time < best_time:
                        best_time = avg_time
                        best_lib = lib

            for lib, results in lib_results.items():
                if results:
                    avg_time = np.mean([r.time_ms for r in results])
                    speedup = avg_time / best_time if best_time > 0 else 1.0
                    print(f"  {lib:>10}: {avg_time:>8.3f} ms/op ({speedup:>5.2f}x)")


def main():
    parser = argparse.ArgumentParser(description="Comprehensive Tensor Library Benchmark")
    parser.add_argument('--sizes', type=str, default='100,500,1000',
                       help='Comma-separated list of tensor sizes (default: 100,500,1000)')
    parser.add_argument('--iterations', type=int, default=100,
                       help='Number of iterations per benchmark (default: 100)')
    parser.add_argument('--gpu', action='store_true',
                       help='Enable GPU benchmarking if available')
    parser.add_argument('--output', type=str, default='benchmark_results.json',
                       help='Output JSON file (default: benchmark_results.json)')

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

    # Create and run benchmark
    benchmark = ComprehensiveBenchmark(sizes, args.iterations, args.gpu)
    benchmark.run_all_benchmarks()
    benchmark.print_summary()
    benchmark.save_results(args.output)


if __name__ == "__main__":
    main()