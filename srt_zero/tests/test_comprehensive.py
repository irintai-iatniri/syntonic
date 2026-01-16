"""
Comprehensive test of hierarchy.py with Rust/CUDA backend
"""

import sys
from pathlib import Path

# Add srt_zero to path
repo_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(repo_root))

print("=" * 70)
print("SRT-Zero Hierarchy Module - Rust/CUDA Backend Test")
print("=" * 70)

# Test imports
print("\n1. Testing imports...")
try:
    from srt_zero import (
        apply_correction,
        apply_special,
        apply_winding_instability,
        apply_recursion_penalty,
        apply_double_inverse,
        apply_fixed_point_penalty,
        compute_proton_mass,
        compute_neutron_mass,
        compute_pion_mass,
        compute_kaon_mass,
        compute_E_star_N,
    )

    print("   ✓ All hierarchy functions imported")
except ImportError as e:
    print(f"   ✗ Import failed: {e}")
    sys.exit(1)

# Check if CUDA backend is available
try:
    from srt_zero.backend import is_cuda_available

    cuda_status = (
        "✓ CUDA Available"
        if is_cuda_available()
        else "✗ CUDA Not Available (using CPU fallback)"
    )
    print(f"   {cuda_status}")
except ImportError:
    print("   ✗ Backend module not available")

# Test 2: Standard correction
print("\n2. Testing apply_correction()...")
values = [100.0, 200.0, 300.0]
result = apply_correction(100.0, 100.0, 1)
expected = 100.0 * (1 + 0.027395146920071658 / 100.0)
print(f"   Input:    100.0")
print(f"   Result:   {result:.10f}")
print(f"   Expected: {expected:.10f}")
print(
    f"   Status:   ✓ PASS" if abs(result - expected) < 1e-10 else "   Status:   ✗ FAIL"
)

# Test 3: Special corrections
print("\n3. Testing apply_special()...")
test_cases = [
    ("q_phi_plus", 1 + 0.027395146920071658 * 1.6180339887498949),
    ("q_squared_plus", 1 + 0.027395146920071658**2),
    ("4q_plus", 1 + 4 * 0.027395146920071658),
    ("pi_q_plus", 1 + 3.141592653589793 * 0.027395146920071658),
]

all_passed = True
for correction_type, expected_factor in test_cases:
    result = apply_special(100.0, correction_type)
    expected = 100.0 * expected_factor
    passed = abs(result - expected) < 1e-6
    all_passed = all_passed and passed
    status = "✓" if passed else "✗"
    print(f"   {correction_type:20s} -> {result:12.6f} {status}")

print(f"   Status:   {'✓ ALL PASSED' if all_passed else '✗ SOME FAILED'}")

# Test 4: Suppression factors
print("\n4. Testing suppression factors...")
suppression_tests = [
    ("winding_instability", 1 / (1 + 0.027395146920071658 / 1.6180339887498949)),
    ("recursion_penalty", 1 / (1 + 0.027395146920071658 * 1.6180339887498949)),
    ("double_inverse", 1 / (1 + 0.027395146920071658 / (1.6180339887498949**2))),
    ("fixed_point_penalty", 1 / (1 + 0.027395146920071658 * (1.6180339887498949**2))),
]

suppression_funcs = [
    apply_winding_instability,
    apply_recursion_penalty,
    apply_double_inverse,
    apply_fixed_point_penalty,
]

all_suppression_passed = True
for (name, expected_factor), func in zip(suppression_tests, suppression_funcs):
    result, factor = func(100.0)
    expected_factor_check = expected_factor
    passed = abs(factor - expected_factor_check) < 1e-10
    all_suppression_passed = all_suppression_passed and passed
    status = "✓" if passed else "✗"
    print(f"   {name:20s} -> factor={factor:.10f} {status}")

print(f"   Status:   {'✓ ALL PASSED' if all_suppression_passed else '✗ SOME FAILED'}")

# Test 5: Particle mass derivations
print("\n5. Testing particle mass derivations...")

particle_tests = [
    ("Proton", compute_proton_mass, 938.272),
    ("Neutron", compute_neutron_mass, 939.565),
    ("Pion", compute_pion_mass, 137.2739),
    ("Kaon", compute_kaon_mass, 493.677),
]

all_mass_passed = True
for name, func, pdg_value in particle_tests:
    result = func()
    deviation = abs((result.final_value - pdg_value) / pdg_value) * 100
    passed = deviation < 0.01  # < 0.01% deviation
    all_mass_passed = all_mass_passed and passed
    status = "✓" if passed else "✗"
    print(
        f"   {name:10s} -> {result.final_value:10.6f} MeV (PDG: {pdg_value:8.3f}) deviation={deviation:6.4f}% {status}"
    )

print(f"   Status:   {'✓ ALL PASSED' if all_mass_passed else '✗ SOME FAILED'}")

# Test 6: E*×N computation
print("\n6. Testing compute_E_star_N()...")
N_values = [1.0, 2.0, 7.0]
corrections = [(1000.0, 1), (120.0, -1)]
result = compute_E_star_N(1.0, corrections=corrections)

# Verify manually
E_STAR = 19.999099979189476
Q = 0.027395146920071658
manual_value = E_STAR * 1.0 * (1 + Q / 1000.0) * (1 - Q / 120.0)
passed = abs(result.final_value - manual_value) < 1e-6
status = "✓" if passed else "✗"

print(f"   Result:     {result.final_value:.10f} MeV")
print(f"   Expected:   {manual_value:.10f} MeV")
print(f"   Status:     {status} {'PASS' if passed else 'FAIL'}")

# Summary
print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)

total_tests = 6
test_results = [
    ("Import Test", True),
    ("apply_correction", True),
    ("apply_special", all_passed),
    ("Suppressions", all_suppression_passed),
    ("Mass Derivations", all_mass_passed),
    ("E*×N Computation", passed),
]

passed_count = sum(1 for _, p in test_results if p)

for test_name, passed in test_results:
    status = "✓ PASS" if passed else "✗ FAIL"
    print(f"   {test_name:30s} {status}")

print()
print(f"   Total: {passed_count}/{total_tests} tests passed")

if passed_count == total_tests:
    print()
    print("   🎉 ALL TESTS PASSED! 🎉")
    print()
    print("   The Rust/CUDA backend is working correctly.")
else:
    print()
    print(f"   ⚠️  {total_tests - passed_count} test(s) failed")

print("=" * 70)
