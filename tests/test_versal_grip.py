#!/usr/bin/env python3
"""
Test script for versal_grip_strength function.
"""

import sys
import os

# Add the syntonic package to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'python'))

try:
    from syntonic._core import py_versal_grip_strength, py_pisano_period
except ImportError as e:
    print(f"Failed to import syntonic core: {e}")
    print("Make sure the Rust extension is built and available.")
    sys.exit(1)

def test_versal_grip_strength():
    """Test versal_grip_strength for various primes."""
    print("Testing versal_grip_strength function...")
    
    # Test cases: (prime, expected_pi, expected_grip)
    test_cases = [
        (2, 3, 0.0),    # 3 % 2 != 0
        (3, 8, 0.0),    # 8 % 3 != 0  
        (5, 20, 4.0),   # 20 % 5 == 0, 20/5 = 4.0
        (7, 16, 0.0),   # 16 % 7 != 0
        (11, 10, 0.0),  # 10 % 11 != 0
        (13, 28, 0.0),  # 28 % 13 != 0
        (17, 36, 0.0),  # 36 % 17 != 0
        (19, 18, 0.0),  # 18 % 19 != 0
        (23, 48, 0.0),  # 48 % 23 != 0
    ]
    
    all_passed = True
    
    for p, expected_pi, expected_grip in test_cases:
        try:
            pi = py_pisano_period(p)
            grip = py_versal_grip_strength(p)
            
            print(f"p={p}: π(p)={pi}, grip={grip:.3f}")
            
            # Verify Pisano period
            if pi != expected_pi:
                print(f"  ERROR: Expected π({p})={expected_pi}, got {pi}")
                all_passed = False
            
            # Verify grip strength
            if abs(grip - expected_grip) > 1e-10:
                print(f"  ERROR: Expected grip={expected_grip}, got {grip}")
                all_passed = False
                
            # Additional check: if pi % p == 0, grip should be pi/p
            if pi % p == 0:
                expected_from_pi = pi / p
                if abs(grip - expected_from_pi) > 1e-10:
                    print(f"  ERROR: pi % p == 0, expected grip={expected_from_pi}, got {grip}")
                    all_passed = False
            else:
                if grip != 0.0:
                    print(f"  ERROR: pi % p != 0, expected grip=0.0, got {grip}")
                    all_passed = False
                    
        except Exception as e:
            print(f"  ERROR for p={p}: {e}")
            all_passed = False
    
    # Test non-prime (should still work, but grip=0 since not prime)
    try:
        grip_4 = py_versal_grip_strength(4)  # 4 is not prime
        if grip_4 != 0.0:
            print(f"ERROR: versal_grip_strength(4) should be 0.0 for non-prime, got {grip_4}")
            all_passed = False
        else:
            print("Non-prime test passed: versal_grip_strength(4) = 0.0")
    except Exception as e:
        print(f"ERROR testing non-prime: {e}")
        all_passed = False
    
    if all_passed:
        print("\n✓ All tests passed!")
        return True
    else:
        print("\n✗ Some tests failed!")
        return False

if __name__ == "__main__":
    success = test_versal_grip_strength()
    sys.exit(0 if success else 1)