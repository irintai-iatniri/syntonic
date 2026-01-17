#!/usr/bin/env python3
"""
Test script for resonant_matmul function.
"""

import sys
import os
import numpy as np

# Add the syntonic package to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'python'))

try:
    from syntonic._core import ResonantTensor as _RT, py_versal_grip_strength
except ImportError as e:
    print(f"Failed to import syntonic core: {e}")
    print("Make sure the Rust extension is built and available.")
    sys.exit(1)

def test_resonant_matmul():
    """Test resonant_matmul function."""
    print("Testing resonant_matmul function...")

    # Create test tensors with different winding indices
    # Test case 1: Compatible geometries (same winding index)
    t1 = _RT([1.0, 2.0], [2], None, 5)  # winding_index = 5
    t2 = _RT([3.0, 4.0], [2], None, 5)  # winding_index = 5

    # Test case 2: Incompatible geometries (different winding indices)
    t3 = _RT([1.0, 2.0], [2], None, 7)  # winding_index = 7
    t4 = _RT([3.0, 4.0], [2], None, 11) # winding_index = 11

    # Test case 3: Compatible geometries with non-zero grip
    t5 = _RT([1.0, 2.0], [2], None, 5)  # winding_index = 5
    t6 = _RT([3.0, 4.0], [2], None, 5)  # winding_index = 5

    all_passed = True

    try:
        # Test 1: Compatible geometries
        result1 = _RT.resonant_matmul(t1, t2)
        grip1 = py_versal_grip_strength(5)
        expected1 = np.array([1.0*3.0 + 2.0*4.0]) * grip1  # dot product scaled by grip

        print(f"Test 1 - Compatible geometries (winding=5):")
        print(f"  Result: {result1.data}")
        print(f"  Expected: {expected1}")
        print(f"  Grip strength: {grip1}")

        if not np.allclose(result1.data, expected1, rtol=1e-10):
            print("  ERROR: Result doesn't match expected")
            all_passed = False
        else:
            print("  ✓ Passed")

    except Exception as e:
        print(f"  ERROR in test 1: {e}")
        all_passed = False

    try:
        # Test 2: Incompatible geometries
        result2 = _RT.resonant_matmul(t3, t4)
        grip2 = py_versal_grip_strength(7)  # Should be 0 since incompatible

        print(f"\nTest 2 - Incompatible geometries (winding=7,11):")
        print(f"  Result: {result2.data}")
        print(f"  Grip strength: {grip2}")

        if grip2 != 0.0:
            print("  ERROR: Grip strength should be 0 for incompatible geometries")
            all_passed = False

        # Result should be zero due to grip strength = 0
        if not np.allclose(result2.data, np.zeros_like(result2.data), rtol=1e-10):
            print("  ERROR: Result should be zero for incompatible geometries")
            all_passed = False
        else:
            print("  ✓ Passed")

    except Exception as e:
        print(f"  ERROR in test 2: {e}")
        all_passed = False

    try:
        # Test 3: Matrix multiplication
        # Create 2x2 matrices
        m1 = _RT([[1.0, 2.0], [3.0, 4.0]], [2, 2], None, 5)
        m2 = _RT([[5.0, 6.0], [7.0, 8.0]], [2, 2], None, 5)

        result3 = _RT.resonant_matmul(m1, m2)
        grip3 = py_versal_grip_strength(5)

        # Standard matrix multiplication result
        standard = np.array([[1*5+2*7, 1*6+2*8], [3*5+4*7, 3*6+4*8]])  # [[19, 22], [43, 50]]
        expected3 = standard * grip3

        print(f"\nTest 3 - Matrix multiplication (winding=5):")
        print(f"  Result:\n{result3.data}")
        print(f"  Expected:\n{expected3}")
        print(f"  Grip strength: {grip3}")

        if not np.allclose(result3.data, expected3, rtol=1e-10):
            print("  ERROR: Matrix multiplication result doesn't match expected")
            all_passed = False
        else:
            print("  ✓ Passed")

    except Exception as e:
        print(f"  ERROR in test 3: {e}")
        all_passed = False

    if all_passed:
        print("\n✓ All resonant_matmul tests passed!")
        return True
    else:
        print("\n✗ Some resonant_matmul tests failed!")
        return False

if __name__ == "__main__":
    success = test_resonant_matmul()
    sys.exit(0 if success else 1)