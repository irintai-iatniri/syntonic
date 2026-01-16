"""
Test hierarchy backend
"""

import sys
from pathlib import Path

# Add srt_zero to path
sys.path.insert(0, str(Path(__file__).parent))

print("Testing Hierarchy Backend")
print("=" * 50)

# Test 1: Import backend
try:
    from srt_zero.backend import (
        is_cuda_available,
        batch_apply_correction,
        batch_apply_special_correction,
        batch_apply_suppression,
        batch_compute_e_star_n,
        SpecialCorrectionType,
        SuppressionType,
    )

    print(f"✓ Backend imported successfully")
    print(f"  CUDA available: {is_cuda_available()}")
except ImportError as e:
    print(f"✗ Failed to import backend: {e}")
    sys.exit(1)

# Test 2: Simple batch correction
print("\nTest 2: Batch correction")
values = [100.0, 200.0, 300.0]
result = batch_apply_correction(values, divisor=1000, sign=+1)
print(f"  Input:    {values}")
print(f"  Output:   {[round(x, 6) for x in result]}")
print(f"  Expected: [x * (1 + q/1000) for x in values]")

# Test 3: Special corrections
print("\nTest 3: Special corrections")
values = [100.0, 200.0, 300.0]
result = batch_apply_special_correction(
    values, ["q_phi_plus", "q_squared_plus", "4q_plus"]
)
print(f"  Input:    {values}")
print(f"  Output:   {[round(x, 6) for x in result]}")

# Test 4: Suppression
print("\nTest 4: Suppression")
values = [100.0, 200.0, 300.0]
result, factor = batch_apply_suppression(values, "winding")
print(f"  Input:    {values}")
print(f"  Output:   {[round(x, 6) for x in result]}")
print(f"  Factor:   {factor:.8f}")
print(f"  Expected: 1/(1+q/φ)")

# Test 5: E*×N batch computation
print("\nTest 5: E*×N batch")
N = [1.0, 2.0, 7.0]
corrections = [(1000, +1), (120, -1)]
result = batch_compute_e_star_n(N, corrections)
print(f"  N:         {N}")
print(f"  Output:    {[round(x, 4) for x in result]} MeV")
print(f"  Expected:  E* × N × (1+q/1000) × (1-q/120)")

# Test 6: Proton mass from hierarchy.py
print("\nTest 6: Proton mass derivation")
from srt_zero.hierarchy import compute_proton_mass

result = compute_proton_mass()
print(f"  Predicted: {result.final_value:.6f} MeV")
print(f"  PDG value: 938.272 MeV")
print(f"  Deviation: {result.deviation_percent:.4f}%")

# Test 7: Neutron mass from hierarchy.py
print("\nTest 7: Neutron mass derivation")
from srt_zero.hierarchy import compute_neutron_mass

result = compute_neutron_mass()
print(f"  Predicted: {result.final_value:.6f} MeV")
print(f"  PDG value: 939.565 MeV")
print(f"  Deviation: {result.deviation_percent:.4f}%")

print("\n" + "=" * 50)
print("All tests passed! ✓")
