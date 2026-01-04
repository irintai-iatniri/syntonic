"""Tests for SRT constants."""

import pytest
import math


class TestSRTConstants:
    """Test SRT-specific constants."""

    def test_torus_dimensions(self):
        """Test TORUS_DIMENSIONS = 4."""
        from syntonic.srt import TORUS_DIMENSIONS
        assert TORUS_DIMENSIONS == 4

    def test_e8_roots(self):
        """Test E8_ROOTS = 240."""
        from syntonic.srt import E8_ROOTS
        assert E8_ROOTS == 240

    def test_e8_positive_roots(self):
        """Test E8_POSITIVE_ROOTS = 120."""
        from syntonic.srt import E8_POSITIVE_ROOTS
        assert E8_POSITIVE_ROOTS == 120

    def test_e6_golden_cone(self):
        """Test E6_GOLDEN_CONE = 36."""
        from syntonic.srt import E6_GOLDEN_CONE
        assert E6_GOLDEN_CONE == 36

    def test_d4_kissing(self):
        """Test D4_KISSING = 24."""
        from syntonic.srt import D4_KISSING
        assert D4_KISSING == 24


class TestGoldenConstants:
    """Test golden ratio constants."""

    def test_phi_numeric(self):
        """Test PHI_NUMERIC = (1 + sqrt(5)) / 2."""
        from syntonic.srt import PHI_NUMERIC
        expected = (1 + math.sqrt(5)) / 2
        assert abs(PHI_NUMERIC - expected) < 1e-15

    def test_phi_squared(self):
        """Test PHI_SQUARED = PHI + 1."""
        from syntonic.srt import PHI_NUMERIC
        from syntonic.exact import PHI_SQUARED
        phi_squared_numeric = PHI_SQUARED.eval()
        assert abs(phi_squared_numeric - (PHI_NUMERIC + 1)) < 1e-15

    def test_phi_inverse(self):
        """Test PHI_INVERSE = PHI - 1."""
        from syntonic.srt import PHI_NUMERIC
        from syntonic.exact import PHI_INVERSE
        phi_inverse_numeric = PHI_INVERSE.eval()
        assert abs(phi_inverse_numeric - (PHI_NUMERIC - 1)) < 1e-15


class TestEStarConstant:
    """Test E* = e^π - π constant."""

    def test_e_star_numeric(self):
        """Test E_STAR_NUMERIC = e^π - π."""
        from syntonic.srt import E_STAR_NUMERIC
        expected = math.exp(math.pi) - math.pi
        assert abs(E_STAR_NUMERIC - expected) < 1e-10

    def test_e_star_approximate_value(self):
        """Test E* ≈ 19.999 (e^π - π)."""
        from syntonic.srt import E_STAR_NUMERIC
        # E* = e^π - π ≈ 19.999099979189474
        assert 19.99 < E_STAR_NUMERIC < 20.01
