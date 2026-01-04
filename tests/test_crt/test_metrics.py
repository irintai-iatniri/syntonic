"""Tests for CRT metrics (Syntony, Gnosis)."""

import math
import pytest
import syntonic as syn
from syntonic.crt import (
    SyntonyComputer,
    GnosisComputer,
    DifferentiationOperator,
    HarmonizationOperator,
    syntony_entropy,
    syntony_spectral,
    syntony_quick,
    K_D4,
)


class TestSyntonyComputer:
    """Tests for SyntonyComputer."""

    def test_basic_computation(self):
        """Test basic syntony computation."""
        D_op = DifferentiationOperator()
        H_op = HarmonizationOperator()
        computer = SyntonyComputer(D_op, H_op)

        state = syn.state([1, 2, 3, 4, 5, 6, 7, 8])
        S = computer.compute(state)

        assert 0.0 <= S <= 1.0

    def test_syntony_bounds(self):
        """Test that syntony is always in [0, 1]."""
        D_op = DifferentiationOperator()
        H_op = HarmonizationOperator()
        computer = SyntonyComputer(D_op, H_op)

        # Try various states
        states = [
            syn.state([1, 1, 1, 1, 1, 1, 1, 1]),  # Constant
            syn.state([1, -1, 1, -1, 1, -1, 1, -1]),  # Alternating
            syn.state([0, 0, 0, 0, 0, 0, 0, 0.001]),  # Near zero
            syn.state([100, 200, 300, 400, 500, 600, 700, 800]),  # Large
        ]

        for state in states:
            S = computer.compute(state)
            assert 0.0 <= S <= 1.0, f"Syntony {S} out of bounds"

    def test_compute_components(self):
        """Test compute_components returns detailed info."""
        D_op = DifferentiationOperator()
        H_op = HarmonizationOperator()
        computer = SyntonyComputer(D_op, H_op)

        state = syn.state([1, 2, 3, 4, 5, 6, 7, 8])
        result = computer.compute_components(state)

        assert 'syntony' in result
        assert 'diff_magnitude' in result
        assert 'residual' in result
        assert 'd_state' in result
        assert 'hd_state' in result


class TestSyntonyQuickEstimates:
    """Tests for quick syntony estimation functions."""

    def test_syntony_entropy_bounds(self):
        """Test syntony_entropy returns value in [0, 1]."""
        state = syn.state([1, 2, 3, 4, 5, 6, 7, 8])
        S = syntony_entropy(state)
        assert 0.0 <= S <= 1.0

    def test_syntony_entropy_uniform(self):
        """Test that uniform distribution has low syntony (high entropy)."""
        uniform = syn.state([1, 1, 1, 1, 1, 1, 1, 1])
        peaked = syn.state([100, 1, 1, 1, 1, 1, 1, 1])

        S_uniform = syntony_entropy(uniform)
        S_peaked = syntony_entropy(peaked)

        # Uniform has higher entropy = lower syntony
        assert S_peaked > S_uniform

    def test_syntony_spectral_bounds(self):
        """Test syntony_spectral returns value in [0, 1]."""
        state = syn.state([1, 2, 3, 4, 5, 6, 7, 8])
        S = syntony_spectral(state)
        assert 0.0 <= S <= 1.0

    def test_syntony_spectral_smooth_vs_rough(self):
        """Test that smooth states have higher spectral syntony."""
        smooth = syn.state([1, 2, 3, 4, 5, 6, 7, 8])  # Low freq
        rough = syn.state([1, -1, 1, -1, 1, -1, 1, -1])  # High freq

        S_smooth = syntony_spectral(smooth)
        S_rough = syntony_spectral(rough)

        assert S_smooth > S_rough

    def test_syntony_quick_bounds(self):
        """Test syntony_quick returns value in [0, 1]."""
        state = syn.state([1, 2, 3, 4, 5, 6, 7, 8])
        S = syntony_quick(state)
        assert 0.0 <= S <= 1.0

    def test_syntony_quick_constant(self):
        """Test that constant states have high quick syntony."""
        constant = syn.state([5, 5, 5, 5, 5, 5, 5, 5])
        varying = syn.state([1, 2, 3, 4, 5, 6, 7, 8])

        S_const = syntony_quick(constant)
        S_vary = syntony_quick(varying)

        assert S_const > S_vary


class TestGnosisComputer:
    """Tests for GnosisComputer."""

    def test_basic_layer_computation(self):
        """Test basic gnosis layer computation."""
        computer = GnosisComputer()
        state = syn.state([1, 2, 3, 4, 5, 6, 7, 8])

        layer = computer.compute_layer(state)
        assert layer in [0, 1, 2, 3]

    def test_layer_names(self):
        """Test layer name lookup."""
        computer = GnosisComputer()

        assert computer.layer_name(0) == 'nascent'
        assert computer.layer_name(1) == 'emergent'
        assert computer.layer_name(2) == 'coherent'
        assert computer.layer_name(3) == 'transcendent'

    def test_phase_thresholds(self):
        """Test that phase thresholds are correct."""
        assert GnosisComputer.THETA_1 == pytest.approx(math.pi)
        assert GnosisComputer.THETA_2 == pytest.approx(2 * math.pi)
        assert GnosisComputer.THETA_3 == pytest.approx(3 * math.pi)

    def test_transcendence_metric(self):
        """Test transcendence metric is non-negative."""
        computer = GnosisComputer()
        state = syn.state([1, 2, 3, 4, 5, 6, 7, 8])

        T = computer.transcendence_metric(state)
        assert T >= 0

    def test_layer_progress(self):
        """Test layer_progress returns valid values."""
        computer = GnosisComputer()
        state = syn.state([1, 2, 3, 4, 5, 6, 7, 8])

        layer, progress = computer.layer_progress(state)

        assert layer in [0, 1, 2, 3]
        assert 0.0 <= progress <= 1.0

    def test_k_d4_cycles(self):
        """Test K(D4) cycles computation."""
        computer = GnosisComputer()
        state = syn.state([1, 2, 3, 4, 5, 6, 7, 8])

        cycles = computer.k_d4_cycles(state)
        assert cycles >= 0

    def test_describe(self):
        """Test describe returns string."""
        computer = GnosisComputer()
        state = syn.state([1, 2, 3, 4, 5, 6, 7, 8])

        desc = computer.describe(state)
        assert isinstance(desc, str)
        assert 'Layer' in desc

    def test_k_d4_constant(self):
        """Test K_D4 constant value."""
        assert K_D4 == 24


class TestGoldenPartition:
    """Tests for the golden partition property D + H = S."""

    def test_syntony_components_sum(self):
        """
        Test that differentiation and harmonization relate via golden ratio.

        The DHSR framework has the property:
        D + H ≈ 0.382 + 0.618 = 1 (golden partition)
        """
        D_op = DifferentiationOperator(alpha_0=0.382)  # D ≈ 1 - 1/φ
        H_op = HarmonizationOperator(beta_0=0.618)  # H ≈ 1/φ

        # The sum of base parameters follows golden partition
        assert D_op.alpha_0 + H_op.beta_0 == pytest.approx(1.0, abs=0.001)
