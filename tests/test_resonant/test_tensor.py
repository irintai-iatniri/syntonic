"""Tests for ResonantTensor - the dual-state tensor."""

import pytest
import math

from syntonic._core import ResonantTensor, GoldenExact

# Golden ratio constant
PHI = 1.6180339887498949


class TestResonantTensorConstruction:
    """Tests for ResonantTensor creation."""

    def test_from_floats(self):
        """Test creating ResonantTensor from float list."""
        data = [1.0, 2.0, 3.0, 4.0]
        tensor = ResonantTensor(data, [4])

        assert len(tensor) == 4
        assert tensor.shape == [4]
        assert tensor.phase == "crystallized"

    def test_from_floats_2d(self):
        """Test creating 2D ResonantTensor."""
        data = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
        tensor = ResonantTensor(data, [2, 3])

        assert len(tensor) == 6
        assert tensor.shape == [2, 3]

    def test_from_floats_with_mode_norms(self):
        """Test creating ResonantTensor with custom mode norms."""
        data = [1.0, 2.0, 3.0, 4.0]
        mode_norms = [0.0, 1.0, 4.0, 9.0]
        tensor = ResonantTensor(data, [4], mode_norms)

        assert len(tensor) == 4
        mode_norms_out = tensor.get_mode_norm_sq()
        assert mode_norms_out == mode_norms

    def test_from_floats_with_precision(self):
        """Test creating ResonantTensor with custom precision."""
        data = [1.0, PHI, 3.0, 4.0]
        tensor = ResonantTensor(data, [4], precision=1000)

        # Should snap accurately with high precision
        values = tensor.to_list()
        assert values[1] == pytest.approx(PHI, rel=1e-6)


class TestResonantTensorProperties:
    """Tests for ResonantTensor properties."""

    def test_syntony_range(self):
        """Syntony should be in [0, 1]."""
        data = [1.0, 2.0, 3.0, 4.0]
        tensor = ResonantTensor(data, [4])

        syntony = tensor.syntony
        assert 0.0 <= syntony <= 1.0

    def test_syntony_constant_mode(self):
        """Constant mode (DC) should have positive syntony."""
        # All same values - mostly DC component
        data = [1.0, 1.0, 1.0, 1.0]
        mode_norms = [0.0, 1.0, 4.0, 9.0]  # DC mode at index 0
        tensor = ResonantTensor(data, [4], mode_norms)

        # Syntony should be positive for constant signal
        assert tensor.syntony > 0.0

    def test_phase_initial(self):
        """Initial phase should be crystallized."""
        data = [1.0, 2.0, 3.0, 4.0]
        tensor = ResonantTensor(data, [4])

        assert tensor.phase == "crystallized"

    def test_repr(self):
        """Test string representation."""
        data = [1.0, 2.0, 3.0, 4.0]
        tensor = ResonantTensor(data, [4])

        repr_str = repr(tensor)
        assert "ResonantTensor" in repr_str
        assert "shape=[4]" in repr_str
        assert "crystallized" in repr_str


class TestResonantTensorLattice:
    """Tests for lattice operations."""

    def test_to_list(self):
        """Test converting lattice to float list."""
        data = [1.0, 2.0, 3.0, 4.0]
        tensor = ResonantTensor(data, [4])

        values = tensor.to_list()
        assert len(values) == 4

        # Values should be close to originals
        for orig, converted in zip(data, values):
            assert converted == pytest.approx(orig, abs=0.1)

    def test_get_lattice(self):
        """Test getting lattice as GoldenExact values."""
        data = [1.0, PHI, 3.0, 4.0]
        tensor = ResonantTensor(data, [4])

        lattice = tensor.get_lattice()
        assert len(lattice) == 4

        # Each element should be GoldenExact
        for g in lattice:
            assert hasattr(g, "rational_coefficient")
            assert hasattr(g, "phi_coefficient")

    def test_golden_values_preserved(self):
        """Golden ratio values should be preserved exactly."""
        data = [1.0, PHI, PHI * PHI, 3.0]
        tensor = ResonantTensor(data, [4], precision=100)

        lattice = tensor.get_lattice()

        # phi should snap to exactly phi
        assert lattice[1].eval() == pytest.approx(PHI, rel=1e-10)

        # phi^2 should snap to 1 + phi
        assert lattice[2].eval() == pytest.approx(PHI * PHI, rel=1e-10)


class TestResonantTensorPhaseTransitions:
    """Tests for CPU phase transitions (without GPU)."""

    def test_wake_flux_values(self):
        """Test CPU wake_flux that returns values."""
        data = [1.0, 2.0, 3.0, 4.0]
        tensor = ResonantTensor(data, [4])

        assert tensor.phase == "crystallized"

        values = tensor.wake_flux_values()
        assert tensor.phase == "flux"
        assert len(values) == 4

    def test_crystallize_from_values(self):
        """Test CPU crystallization from values."""
        data = [1.0, 2.0, 3.0, 4.0]
        tensor = ResonantTensor(data, [4])

        # Wake flux first
        tensor.wake_flux_values()
        assert tensor.phase == "flux"

        # Crystallize with new values
        new_values = [1.5, 2.5, 3.5, 4.5]
        syntony = tensor.crystallize_from_values(new_values, 100)

        assert tensor.phase == "crystallized"
        assert 0.0 <= syntony <= 1.0

    def test_cpu_cycle(self):
        """Test full CPU D→H cycle."""
        data = [1.0, 2.0, 3.0, 4.0]
        tensor = ResonantTensor(data, [4])

        initial_syntony = tensor.syntony

        # Run CPU cycle with noise
        syntony = tensor.cpu_cycle(0.1, 100)

        assert tensor.phase == "crystallized"
        assert 0.0 <= syntony <= 1.0

    def test_multiple_cpu_cycles(self):
        """Test running multiple CPU cycles."""
        data = [1.0, 2.0, 3.0, 4.0]
        tensor = ResonantTensor(data, [4])

        syntonies = []
        for _ in range(5):
            syntony = tensor.cpu_cycle(0.05, 100)
            syntonies.append(syntony)
            assert tensor.phase == "crystallized"

        # All syntonies should be valid
        assert all(0.0 <= s <= 1.0 for s in syntonies)


class TestResonantTensorFromGoldenExact:
    """Tests for creating ResonantTensor from GoldenExact values."""

    def test_from_golden_exact(self):
        """Test creating from GoldenExact lattice."""
        lattice = [
            GoldenExact.from_integers(1, 0),  # 1
            GoldenExact.golden_ratio(),        # φ
            GoldenExact.golden_squared(),      # φ²
            GoldenExact.from_integers(3, 0),  # 3
        ]

        tensor = ResonantTensor.from_golden_exact(lattice, [4])

        assert len(tensor) == 4
        assert tensor.phase == "crystallized"

        # Values should match
        values = tensor.to_list()
        assert values[0] == pytest.approx(1.0, rel=1e-10)
        assert values[1] == pytest.approx(PHI, rel=1e-10)
        assert values[2] == pytest.approx(PHI * PHI, rel=1e-10)
        assert values[3] == pytest.approx(3.0, rel=1e-10)


class TestResonantTensorSyntonyComputation:
    """Tests for syntony computation on crystallized lattice."""

    def test_syntony_dc_dominant(self):
        """DC-dominant signal should have high syntony."""
        # Large DC component (index 0), small oscillations
        data = [10.0, 0.1, 0.1, 0.1]
        mode_norms = [0.0, 1.0, 4.0, 9.0]
        tensor = ResonantTensor(data, [4], mode_norms)

        # High syntony expected
        assert tensor.syntony > 0.9

    def test_syntony_high_freq_dominant(self):
        """High-frequency dominant signal should have lower syntony."""
        # Small DC, large high-frequency
        data = [0.1, 0.1, 0.1, 10.0]
        mode_norms = [0.0, 1.0, 4.0, 9.0]
        tensor = ResonantTensor(data, [4], mode_norms)

        # Lower syntony expected (high freq is down-weighted)
        # But still positive
        assert 0.0 < tensor.syntony < 0.9

    def test_syntony_zero_signal(self):
        """Zero signal should have zero syntony."""
        data = [0.0, 0.0, 0.0, 0.0]
        mode_norms = [0.0, 1.0, 4.0, 9.0]
        tensor = ResonantTensor(data, [4], mode_norms)

        # Zero signal has zero syntony
        assert tensor.syntony == 0.0


class TestResonantTensorEdgeCases:
    """Tests for edge cases and error handling."""

    def test_single_element(self):
        """Test single-element tensor."""
        data = [PHI]
        tensor = ResonantTensor(data, [1])

        assert len(tensor) == 1
        assert tensor.to_list()[0] == pytest.approx(PHI, rel=1e-6)

    def test_large_values(self):
        """Test with moderately large values within precision bounds."""
        # Use values within the precision range (max_coeff=1000)
        data = [100.0, 200.0, 300.0]
        tensor = ResonantTensor(data, [3], precision=1000)

        values = tensor.to_list()
        for orig, converted in zip(data, values):
            # Should be reasonably close
            assert converted == pytest.approx(orig, rel=0.1)

    def test_negative_values(self):
        """Test with negative values."""
        data = [-1.0, -PHI, -3.0]
        tensor = ResonantTensor(data, [3])

        values = tensor.to_list()
        assert values[0] < 0
        assert values[1] == pytest.approx(-PHI, rel=1e-6)
        assert values[2] < 0

    def test_mixed_sign_values(self):
        """Test with mixed positive and negative values."""
        data = [-2.0, -1.0, 0.0, 1.0, 2.0]
        tensor = ResonantTensor(data, [5])

        values = tensor.to_list()
        assert len(values) == 5
        # Order should be preserved approximately
        for i in range(len(values) - 1):
            assert values[i] <= values[i + 1] + 0.5  # Allow some tolerance
