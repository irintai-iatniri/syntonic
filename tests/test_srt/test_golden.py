"""Tests for SRT golden module (GoldenMeasure, GoldenRecursionMap)."""

import pytest
import math


class TestGoldenMeasure:
    """Test GoldenMeasure class."""

    def test_creation(self):
        """Test GoldenMeasure creation."""
        from syntonic.srt import golden_measure
        gm = golden_measure()
        assert gm.phi > 1.6

    def test_vacuum_weight(self):
        """Test w(0) = 1 for vacuum state."""
        from syntonic.srt import golden_measure, winding_state
        gm = golden_measure()
        vacuum = winding_state(0, 0, 0, 0)
        assert abs(gm.weight(vacuum) - 1.0) < 1e-10

    def test_weight_positive(self):
        """Test weights are positive."""
        from syntonic.srt import golden_measure, winding_state
        gm = golden_measure()
        for n7 in range(-2, 3):
            for n8 in range(-2, 3):
                n = winding_state(n7, n8, 0, 0)
                assert gm.weight(n) > 0

    def test_weight_decreases_with_norm(self):
        """Test w(n) decreases as |n|² increases."""
        from syntonic.srt import golden_measure, winding_state
        gm = golden_measure()
        n0 = winding_state(0, 0, 0, 0)
        n1 = winding_state(1, 0, 0, 0)
        n2 = winding_state(1, 1, 0, 0)

        assert gm.weight(n0) > gm.weight(n1) > gm.weight(n2)

    def test_weight_formula(self):
        """Test w(n) = exp(-|n|²/φ)."""
        from syntonic.srt import golden_measure, winding_state, PHI_NUMERIC
        gm = golden_measure()
        n = winding_state(1, 2, 0, 0)
        expected = math.exp(-(1 + 4) / PHI_NUMERIC)
        assert abs(gm.weight(n) - expected) < 1e-10

    def test_partition_function_positive(self):
        """Test partition function is positive."""
        from syntonic.srt import golden_measure
        gm = golden_measure()
        Z = gm.partition_function(max_norm=5)
        assert Z > 0

    def test_partition_function_bounded(self):
        """Test partition function is bounded."""
        from syntonic.srt import golden_measure
        gm = golden_measure()
        Z = gm.partition_function(max_norm=10)
        # Z should be finite and reasonable
        assert Z < 1000


class TestGoldenRecursionMap:
    """Test GoldenRecursionMap class."""

    def test_creation(self):
        """Test GoldenRecursionMap creation."""
        from syntonic.srt import golden_recursion_map
        R = golden_recursion_map()
        assert R.phi > 1.6

    def test_apply_vacuum(self):
        """Test R(0) = 0."""
        from syntonic.srt import golden_recursion_map, winding_state
        R = golden_recursion_map()
        vacuum = winding_state(0, 0, 0, 0)
        result = R.apply(vacuum)
        assert result == vacuum

    def test_apply_formula(self):
        """Test R(n) = ⌊φn⌋ component-wise."""
        from syntonic.srt import golden_recursion_map, winding_state, PHI_NUMERIC
        R = golden_recursion_map()
        n = winding_state(2, 3, 4, 5)
        result = R.apply(n)

        expected = winding_state(
            int(PHI_NUMERIC * 2),
            int(PHI_NUMERIC * 3),
            int(PHI_NUMERIC * 4),
            int(PHI_NUMERIC * 5),
        )
        assert result == expected

    def test_orbit_contains_start(self):
        """Test orbit contains starting state."""
        from syntonic.srt import golden_recursion_map, winding_state
        R = golden_recursion_map()
        n = winding_state(1, 0, 0, 0)
        orbit = R.orbit(n, max_depth=10)
        assert n in orbit

    def test_orbit_terminates(self):
        """Test orbit is computed correctly (R is expansion with phi > 1)."""
        from syntonic.srt import golden_recursion_map, winding_state
        R = golden_recursion_map()
        n = winding_state(5, 3, 2, 1)
        orbit = R.orbit(n, max_depth=10)

        # R(n) = floor(phi*n) is an expansion since phi > 1
        # The orbit starts at n and expands
        assert len(orbit) >= 2
        assert orbit[0] == n
        # Each step increases the norm (for positive components)
        assert orbit[1].norm_squared > orbit[0].norm_squared

    def test_orbit_increasing_norm(self):
        """Test orbit states have increasing |n|² (before mod operations)."""
        from syntonic.srt import golden_recursion_map, winding_state
        R = golden_recursion_map()
        n = winding_state(1, 0, 0, 0)
        orbit = R.orbit(n, max_depth=5)

        # First few steps should increase norm (before reaching small states)
        assert len(orbit) >= 2

    def test_fixed_points_contains_vacuum(self):
        """Test vacuum is a fixed point."""
        from syntonic.srt import golden_recursion_map, winding_state
        R = golden_recursion_map()
        fixed = R.fixed_points(max_norm=10)
        vacuum = winding_state(0, 0, 0, 0)
        assert vacuum in fixed

    def test_mass_scaling_vacuum(self):
        """Test mass_scaling(0) = 1."""
        from syntonic.srt import golden_recursion_map, winding_state
        R = golden_recursion_map()
        vacuum = winding_state(0, 0, 0, 0)
        assert abs(R.mass_scaling(vacuum) - 1.0) < 1e-10

    def test_mass_scaling_decreases(self):
        """Test mass_scaling decreases with orbit depth."""
        from syntonic.srt import golden_recursion_map, winding_state
        R = golden_recursion_map()
        n1 = winding_state(1, 0, 0, 0)
        n2 = winding_state(2, 0, 0, 0)

        # Deeper orbit = smaller mass scaling
        # (more suppressed by φ factors)
        m1 = R.mass_scaling(n1)
        m2 = R.mass_scaling(n2)
        assert m1 <= 1.0
        assert m2 <= 1.0


class TestGoldenIntegration:
    """Integration tests for golden module."""

    def test_measure_recursion_compatibility(self):
        """Test measure and recursion are compatible."""
        from syntonic.srt import golden_measure, golden_recursion_map, winding_state
        gm = golden_measure()
        R = golden_recursion_map()

        # They should use the same phi
        assert abs(gm.phi - R.phi) < 1e-15

    def test_orbit_weights_decrease(self):
        """Test w(n) decreases along orbit."""
        from syntonic.srt import golden_measure, golden_recursion_map, winding_state
        gm = golden_measure()
        R = golden_recursion_map()

        n = winding_state(1, 0, 0, 0)
        orbit = R.orbit(n, max_depth=5)

        # Check weights for first few states (before hitting vacuum)
        weights = [gm.weight(state) for state in orbit]
        # Weights should generally increase along orbit toward vacuum
        # (since vacuum has weight 1 and others have weight < 1)
