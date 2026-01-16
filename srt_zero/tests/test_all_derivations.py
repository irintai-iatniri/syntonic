#!/usr/bin/env python3
"""
Test all particle derivations in SRT-Zero catalog.
"""

import sys
import os
from pathlib import Path

# Go to parent directory to ensure srt_zero is importable as a package
test_dir = Path(__file__).resolve()
repo_root = test_dir.parent
os.chdir(str(repo_root))

# Now try to import
try:
    from srt_zero.catalog import CATALOG
    from srt_zero.engine import DerivationEngine

    print("✓ Imports successful")
except ImportError as e:
    print(f"✗ Import failed: {e}")
    sys.exit(1)


def test_all_particles():
    """Test all particles in catalog."""
    engine = DerivationEngine()
    print(f"Testing {len(CATALOG)} particles...")
    print()

    failures = []
    successes = 0

    for key, config in CATALOG.items():
        try:
            if config.pdg_value == 0:
                print(f"WARNING: {config.name} has PDG value 0")
                continue

            result = engine.derive(config.name)

            if result.final_value is None or result.final_value == 0:
                if config.formula_type.name != "COSMOLOGY_SPECIAL":
                    failures.append((config.name, "Zero result"))
                else:
                    successes += 1
            else:
                if config.corrections and not result.steps:
                    failures.append((config.name, "Missing steps despite corrections"))
                else:
                    successes += 1

                    if successes % 50 == 0:
                        print(
                            f"Progress: {successes}/{len(CATALOG)} particles tested..."
                        )

        except Exception as e:
            failures.append((config.name, str(e)))
            if len(failures) <= 5:
                print(f"ERROR in {config.name}: {e}")

    print()
    print(f"Successes: {successes}")
    print(f"Failures: {len(failures)}")

    if failures:
        print("\nFailed Particles (first 10):")
        for name, error in failures[:10]:
            print(f"- {name}: {error}")
        if len(failures) > 10:
            print(f"... and {len(failures) - 10} more failures")

    print()
    if len(failures) == 0:
        print("ALL TESTS PASSED!")
        print(f"Successfully derived {successes} particle masses/observables")
    else:
        success_rate = 100 * successes / len(CATALOG)
        print(
            f"FAILURES: {len(failures)} tests failed ({success_rate:.1f}% success rate)"
        )


if __name__ == "__main__":
    test_all_particles()
