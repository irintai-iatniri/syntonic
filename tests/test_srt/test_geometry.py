"""Tests for SRT geometry module (WindingState, T4Torus)."""

import pytest
import math
import cmath


class TestWindingState:
    """Test WindingState class."""

    def test_creation(self):
        """Test WindingState creation."""
        from syntonic.srt import winding_state
        n = winding_state(1, 2, 3, 4)
        assert n.n7 == 1
        assert n.n8 == 2
        assert n.n9 == 3
        assert n.n10 == 4

    def test_tuple_access(self):
        """Test tuple-like access."""
        from syntonic.srt import winding_state
        n = winding_state(1, 2, 3, 4)
        assert n.n == (1, 2, 3, 4)

    def test_norm_squared(self):
        """Test |n|² computation."""
        from syntonic.srt import winding_state
        n = winding_state(1, 2, 3, 4)
        assert n.norm_squared == 1 + 4 + 9 + 16  # 30

    def test_vacuum_state(self):
        """Test vacuum state |0,0,0,0⟩."""
        from syntonic.srt import winding_state
        vacuum = winding_state(0, 0, 0, 0)
        assert vacuum.norm_squared == 0
        assert vacuum.is_vacuum

    def test_inner_product(self):
        """Test inner product <n|m>."""
        from syntonic.srt import winding_state
        n = winding_state(1, 0, 0, 0)
        m = winding_state(2, 0, 0, 0)
        assert n.inner_product(m) == 2

    def test_negation(self):
        """Test negation -n."""
        from syntonic.srt import winding_state
        n = winding_state(1, 2, 3, 4)
        neg_n = -n
        assert neg_n.n == (-1, -2, -3, -4)

    def test_addition(self):
        """Test vector addition."""
        from syntonic.srt import winding_state
        n = winding_state(1, 0, 0, 0)
        m = winding_state(0, 1, 0, 0)
        result = n + m
        assert result.n == (1, 1, 0, 0)

    def test_golden_weight(self):
        """Test golden weight w(n) = exp(-|n|²/φ)."""
        from syntonic.srt import winding_state, PHI_NUMERIC
        n = winding_state(1, 0, 0, 0)
        expected = math.exp(-1 / PHI_NUMERIC)
        assert abs(n.golden_weight() - expected) < 1e-10

    def test_generation(self):
        """Test generation computation."""
        from syntonic.srt import winding_state
        # Generation k where mass ~ e^(-φk)
        vacuum = winding_state(0, 0, 0, 0)
        assert vacuum.generation == 0

    def test_equality(self):
        """Test equality comparison."""
        from syntonic.srt import winding_state
        n1 = winding_state(1, 2, 3, 4)
        n2 = winding_state(1, 2, 3, 4)
        n3 = winding_state(1, 2, 3, 5)
        assert n1 == n2
        assert n1 != n3

    def test_hashable(self):
        """Test that WindingState is hashable (can be used in sets/dicts)."""
        from syntonic.srt import winding_state
        n1 = winding_state(1, 2, 3, 4)
        n2 = winding_state(1, 2, 3, 4)
        s = {n1, n2}
        assert len(s) == 1

    def test_immutable(self):
        """Test that WindingState is immutable."""
        from syntonic.srt import winding_state
        n = winding_state(1, 2, 3, 4)
        with pytest.raises(AttributeError):
            n.n7 = 5


class TestT4Torus:
    """Test T4Torus class."""

    def test_creation(self):
        """Test T4Torus creation."""
        from syntonic.srt import t4_torus
        torus = t4_torus()
        assert torus.dimension == 4

    def test_fourier_mode(self):
        """Test Fourier mode exp(i·n·y/l)."""
        from syntonic.srt import t4_torus, winding_state
        import math
        torus = t4_torus()
        n = winding_state(1, 0, 0, 0)
        y = (math.pi, 0.0, 0.0, 0.0)

        # exp(i·1·π/1) = exp(iπ) = -1
        mode = torus.fourier_mode(n, y)
        assert abs(mode - (-1)) < 1e-10

    def test_fourier_mode_vacuum(self):
        """Test vacuum Fourier mode is 1."""
        from syntonic.srt import t4_torus, winding_state
        torus = t4_torus()
        vacuum = winding_state(0, 0, 0, 0)
        y = (0.1, 0.2, 0.3, 0.4)

        mode = torus.fourier_mode(vacuum, y)
        assert abs(mode - 1.0) < 1e-10

    def test_enumerate_windings(self):
        """Test winding state enumeration."""
        from syntonic.srt import t4_torus
        torus = t4_torus()
        windings = list(torus.enumerate_windings(max_norm=1))

        # |n|² = 0: (0,0,0,0) = 1 state
        # |n|² = 1: (+/-1,0,0,0) and perms = 8 states
        assert len(windings) == 9

    def test_enumerate_windings_norm_2(self):
        """Test winding enumeration with |n| ≤ 2 (so |n|² ≤ 4)."""
        from syntonic.srt import t4_torus
        torus = t4_torus()
        windings = list(torus.enumerate_windings(max_norm=2))

        # max_norm=2 means |n| ≤ 2, so |n|² ≤ 4
        # |n|² = 0: 1
        # |n|² = 1: 8
        # |n|² = 2: C(4,2) × 2² = 24
        # |n|² = 3: C(4,3) × 2³ = 32
        # |n|² = 4: 8 (±2,0,0,0 perms) + 16 (±1,±1,±1,±1) = 24
        assert len(windings) == 1 + 8 + 24 + 32 + 24  # = 89

    def test_coordinate_periodicity(self):
        """Test that coordinates are periodic with period 2π (for radius=1)."""
        import math
        from syntonic.srt import t4_torus, winding_state
        torus = t4_torus()
        n = winding_state(1, 1, 1, 1)

        y1 = (0.0, 0.0, 0.0, 0.0)
        y2 = (2*math.pi, 2*math.pi, 2*math.pi, 2*math.pi)  # Should be equivalent

        mode1 = torus.fourier_mode(n, y1)
        mode2 = torus.fourier_mode(n, y2)
        assert abs(mode1 - mode2) < 1e-10


class TestGeometryIntegration:
    """Integration tests for geometry module."""

    def test_winding_orthogonality(self):
        """Test orthogonality of different winding states."""
        from syntonic.srt import winding_state
        # Different winding states are orthogonal in the inner product
        n = winding_state(1, 0, 0, 0)
        m = winding_state(0, 1, 0, 0)
        assert n.inner_product(m) == 0

    def test_norm_squared_equals_inner_product(self):
        """Test |n|² = ⟨n|n⟩."""
        from syntonic.srt import winding_state
        n = winding_state(1, 2, 3, 4)
        assert n.norm_squared == n.inner_product(n)
