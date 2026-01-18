#!/usr/bin/env python3
"""
SRT/CRT Comprehensive Test Suite

Runs all tests for the new SRT/CRT backend implementation and provides
a detailed validation report of the theoretical predictions.
"""

import sys
import os
import subprocess
import time
from pathlib import Path
from typing import Dict, List, Tuple

# Add the syntonic package to path
sys.path.insert(0, str(Path(__file__).parent.parent / "python"))

import syntonic._core as core
from syntonic.physics.srt_physics import SRTPhysicsEngine


class SRTCRTTestRunner:
    """Comprehensive test runner for SRT/CRT implementation."""

    def __init__(self):
        self.results = {}
        self.start_time = None

    def run_command(self, cmd: List[str], cwd: str = None) -> Tuple[int, str, str]:
        """Run a command and return exit code, stdout, stderr."""
        try:
            result = subprocess.run(
                cmd,
                cwd=cwd,
                capture_output=True,
                text=True,
                timeout=300
            )
            return result.returncode, result.stdout, result.stderr
        except subprocess.TimeoutExpired:
            return -1, "", "Command timed out"
        except Exception as e:
            return -1, "", str(e)

    def run_rust_tests(self) -> bool:
        """Run Rust unit tests."""
        print("🔧 Running Rust unit tests...")
        exit_code, stdout, stderr = self.run_command(
            ["cargo", "test"],
            cwd=str(Path(__file__).parent / "rust")
        )

        success = exit_code == 0
        self.results['rust_tests'] = {
            'success': success,
            'output': stdout,
            'errors': stderr
        }

        if success:
            print("✅ Rust tests passed")
        else:
            print("❌ Rust tests failed")
            print(stderr)

        return success

    def run_python_tests(self) -> bool:
        """Run Python unit tests."""
        print("🐍 Running Python unit tests...")

        test_files = [
            "tests/test_prime_theory.py",
            "tests/test_neural_networks.py",
            "tests/test_srt_physics.py"
        ]

        all_success = True
        for test_file in test_files:
            print(f"  Running {test_file}...")
            exit_code, stdout, stderr = self.run_command(
                ["python", "-m", "pytest", test_file, "-v"],
                cwd=str(Path(__file__).parent)
            )

            success = exit_code == 0
            self.results[f'python_{test_file.replace("/", "_").replace(".py", "")}'] = {
                'success': success,
                'output': stdout,
                'errors': stderr
            }

            if not success:
                all_success = False
                print(f"    ❌ {test_file} failed")
                print(stderr)
            else:
                print(f"    ✅ {test_file} passed")

        return all_success

    def validate_srt_predictions(self) -> Dict[str, bool]:
        """Validate SRT/CRT theoretical predictions."""
        print("🔬 Validating SRT/CRT Predictions...")

        validations = {}

        try:
            # Test force count (exactly 5 forces)
            fermat_primes = core.py_fermat_sequence(10)
            validations['force_count'] = len(fermat_primes) == 5
            print(f"  Forces: {len(fermat_primes)}/5 {'✅' if validations['force_count'] else '❌'}")

            # Test matter generations (exactly 4 stable)
            stable_windings = sum(1 for p in range(2, 12) if core.py_is_stable_winding(p))
            validations['matter_generations'] = stable_windings == 4
            print(f"  Matter generations: {stable_windings}/4 {'✅' if validations['matter_generations'] else '❌'}")

            # Test stability barrier
            barrier = core.py_get_stability_barrier()
            validations['stability_barrier'] = barrier == 11
            print(f"  Stability barrier: p={barrier} {'✅' if validations['stability_barrier'] else '❌'}")

            # Test dark matter mass prediction
            dm_mass = core.py_predict_dark_matter_mass(172.7)
            validations['dark_matter_mass'] = 1100 <= dm_mass <= 1500
            print(f"  Dark matter mass: {dm_mass:.0f} GeV {'✅' if validations['dark_matter_mass'] else '❌'}")

            # Test Fibonacci transcendence gates
            gates = sum(1 for n in range(20) if core.py_is_transcendence_gate(n))
            validations['fibonacci_gates'] = gates >= 7
            print(f"  Fibonacci gates: {gates} {'✅' if validations['fibonacci_gates'] else '❌'}")

            # Test Lucas dark boost
            boost = core.py_lucas_dark_boost()
            validations['lucas_boost'] = abs(boost - 6.854) < 0.01
            print(f"  Lucas boost: {boost:.3f} {'✅' if validations['lucas_boost'] else '❌'}")

        except Exception as e:
            print(f"  ❌ Validation error: {e}")
            validations['error'] = False

        self.results['srt_validations'] = validations
        return validations

    def benchmark_performance(self) -> Dict[str, float]:
        """Run performance benchmarks."""
        print("⚡ Running performance benchmarks...")

        benchmarks = {}

        try:
            # Benchmark prime computations
            import time

            # Mersenne prime check
            start = time.time()
            for p in range(2, 20):
                core.py_is_mersenne_prime(p)
            benchmarks['mersenne_primes'] = time.time() - start

            # Lucas number computation
            start = time.time()
            for n in range(1, 20):
                core.py_lucas_number(n)
            benchmarks['lucas_numbers'] = time.time() - start

            # Pisano period computation
            start = time.time()
            for p in [2, 3, 5, 7, 11, 13]:
                core.py_pisano_period(p)
            benchmarks['pisano_periods'] = time.time() - start

            print("  ✅ Benchmarks completed")
            for name, time_taken in benchmarks.items():
                print(".4f")

        except Exception as e:
            print(f"  ❌ Benchmark error: {e}")

        self.results['benchmarks'] = benchmarks
        return benchmarks

    def test_neural_networks(self) -> bool:
        """Test neural network components."""
        print("🧠 Testing neural networks...")

        try:
            from syntonic.nn.layers.prime_syntony_gate import (
                PrimeSyntonyGate,
                WindingAttention,
                SRTTransformerBlock
            )
            import torch

            # Test Prime Syntony Gate
            gate = PrimeSyntonyGate(7)
            x = torch.randn(2, 10, 7)
            output = gate(x)
            assert output.shape == x.shape

            # Test Winding Attention
            attention = WindingAttention(7, 1)
            q = torch.randn(2, 10, 7)
            output = attention(q, q, q)
            assert output.shape == q.shape

            # Test SRT Transformer
            transformer = SRTTransformerBlock(7, 1)
            output = transformer(x)
            assert output.shape == x.shape

            print("  ✅ Neural network tests passed")
            self.results['neural_networks'] = {'success': True}
            return True

        except Exception as e:
            print(f"  ❌ Neural network test failed: {e}")
            self.results['neural_networks'] = {'success': False, 'error': str(e)}
            return False

    def run_all_tests(self) -> bool:
        """Run the complete test suite."""
        self.start_time = time.time()

        print("🚀 SRT/CRT Comprehensive Test Suite")
        print("=" * 50)

        # Run all test components
        rust_success = self.run_rust_tests()
        python_success = self.run_python_tests()
        nn_success = self.test_neural_networks()
        validations = self.validate_srt_predictions()
        benchmarks = self.benchmark_performance()

        # Overall success
        all_validations_passed = all(validations.values())
        overall_success = (
            rust_success and
            python_success and
            nn_success and
            all_validations_passed
        )

        # Print summary
        self.print_summary(
            rust_success, python_success, nn_success,
            validations, benchmarks, overall_success
        )

        return overall_success

    def print_summary(self, rust_success, python_success, nn_success,
                     validations, benchmarks, overall_success):
        """Print comprehensive test summary."""
        duration = time.time() - self.start_time

        print("\n" + "=" * 50)
        print("📊 SRT/CRT TEST SUMMARY")
        print("=" * 50)

        print("\n🔧 Core Implementation:")
        print(f"  Rust Backend:     {'✅' if rust_success else '❌'}")
        print(f"  Python Bindings:  {'✅' if python_success else '❌'}")
        print(f"  Neural Networks:  {'✅' if nn_success else '❌'}")

        print("\n🔬 Theoretical Validation:")
        for prediction, valid in validations.items():
            status = '✅' if valid else '❌'
            print(f"  {prediction.replace('_', ' ').title()}: {status}")

        print("\n⚡ Performance Benchmarks:")
        for benchmark, time_taken in benchmarks.items():
            print(f"  {benchmark.replace('_', ' ').title()}: {time_taken:.4f}s")

        print("\n🎯 Overall Result:")
        if overall_success:
            print("  ✅ ALL TESTS PASSED - SRT/CRT Implementation Validated!")
        else:
            print("  ❌ Some tests failed - Check implementation")

        print(f"  Duration: {duration:.2f}s")
        # Theoretical coherence score
        validation_score = sum(validations.values()) / len(validations) * 100
        print(f"  Theoretical Coherence: {validation_score:.1f}%")
        if validation_score >= 95:
            print("  🌟 Excellent theoretical coherence!")
        elif validation_score >= 80:
            print("  ✨ Good theoretical coherence")
        else:
            print("  ⚠️  Theoretical coherence needs review")


def main():
    """Main test runner entry point."""
    runner = SRTCRTTestRunner()
    success = runner.run_all_tests()

    # Exit with appropriate code
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()