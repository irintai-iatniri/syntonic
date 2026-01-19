"""Tests for the Five Operators of Existence."""

import pytest
from srt_zero.operators import (
    winding_state,  # Import the factory function
    recursion_map,
    is_recursion_fixed_point,
    get_recursion_orbit,
    winding_norm,
    mersenne_number,
    is_stable_generation,
    get_generation,
    lucas_number,
    why_no_fourth_generation,
    dark_matter_mass_prediction,
    fermat_number,
    is_valid_force_index,
    apply_five_operators,
)
from srt_zero.hierarchy import PHI, M11_BARRIER, LUCAS_PRIMES_INDICES


class TestRecursionOperator:
    """Tests for Operator 1: Recursion (φ)"""

    def test_fixed_points(self):
        """Fixed points: |n_i| ≤ 3 for all i"""
        assert is_recursion_fixed_point(winding_state(0, 0, 0, 0))
        assert is_recursion_fixed_point(winding_state(1, 1, 1, 0))  # Proton
        assert is_recursion_fixed_point(winding_state(3, 2, 1, 0))
        assert not is_recursion_fixed_point(winding_state(4, 0, 0, 0))

    def test_recursion_map_growth(self):
        """Non-fixed points should grow under recursion"""
        n = winding_state(5, 3, 2, 0)
        r_n = recursion_map(n)
        assert winding_norm(r_n) > winding_norm(n)

    def test_orbit_convergence(self):
        """Fixed point orbits should be length 1"""
        fixed = winding_state(1, 1, 0, 0)
        orbit = get_recursion_orbit(fixed)
        assert len(orbit) == 1


class TestMersennePrimes:
    """Tests for Operator 4: Harmonization (Mersenne)"""

    def test_generation_count(self):
        """Exactly 3 stable generations"""
        assert is_stable_generation(2)  # Gen 1
        assert is_stable_generation(3)  # Gen 2
        assert is_stable_generation(5)  # Gen 3
        assert not is_stable_generation(11)  # 4th gen blocked

    def test_m11_barrier(self):
        """M₁₁ = 2047 = 23 × 89 is composite"""
        m11 = mersenne_number(11)
        assert m11 == M11_BARRIER
        assert m11 == 23 * 89

    def test_generation_mapping(self):
        """Correct generation assignments"""
        assert get_generation(2) == 1
        assert get_generation(3) == 2
        assert get_generation(5) == 3
        assert get_generation(11) is None


class TestFermatPrimes:
    """Tests for Operator 3: Differentiation (Fermat)"""

    def test_five_forces_only(self):
        """Only 5 Fermat primes exist → only 5 forces"""
        for n in range(5):
            assert is_valid_force_index(n)
        assert not is_valid_force_index(5)  # F₅ is composite

    def test_fermat_numbers(self):
        """Verify Fermat number computation"""
        assert fermat_number(0) == 3
        assert fermat_number(1) == 5
        assert fermat_number(2) == 17


class TestLucasPrimes:
    """Tests for Operator 5: Balance (Lucas)"""

    def test_lucas_sequence(self):
        """Verify Lucas number computation"""
        assert lucas_number(0) == 2
        assert lucas_number(1) == 1
        assert lucas_number(2) == 3
        assert lucas_number(3) == 4
        assert lucas_number(4) == 7

    def test_dark_matter_prediction(self):
        """Dark matter mass ≈ 1.18 TeV"""
        m_dm = dark_matter_mass_prediction(173000)  # m_top in MeV
        assert 1100000 < m_dm < 1200000  # 1.1-1.2 TeV range


class TestUnifiedOperators:
    """Tests for the unified five-operator application"""

    def test_proton_classification(self):
        """Proton winding (1,1,1,0) should be stable Gen 1"""
        proton_winding = winding_state(1, 1, 1, 0)
        result = apply_five_operators(proton_winding, 2)
        assert result.is_fixed_point
        assert result.generation == 1
        assert result.syntony > 0
