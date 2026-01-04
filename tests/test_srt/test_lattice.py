"""Tests for SRT lattice module (E8, D4, GoldenCone, QuadraticForm)."""

import pytest
import math
from fractions import Fraction


class TestE8Lattice:
    """Test E8Lattice class."""

    def test_creation(self):
        """Test E8Lattice creation."""
        from syntonic.srt import e8_lattice
        E8 = e8_lattice()
        assert E8 is not None

    def test_root_count(self):
        """Test E8 has exactly 240 roots."""
        from syntonic.srt import e8_lattice
        E8 = e8_lattice()
        assert len(E8.roots) == 240

    def test_positive_root_count(self):
        """Test E8 has exactly 120 positive roots."""
        from syntonic.srt import e8_lattice
        E8 = e8_lattice()
        assert len(E8.positive_roots) == 120

    def test_all_roots_norm_2(self):
        """Test all roots have |λ|² = 2."""
        from syntonic.srt import e8_lattice
        E8 = e8_lattice()
        for root in E8.roots:
            assert root.norm_squared == Fraction(2)

    def test_type_a_count(self):
        """Test 112 Type A roots (integer coords)."""
        from syntonic.srt import e8_lattice
        E8 = e8_lattice()
        assert len(E8.type_a_roots) == 112

    def test_type_b_count(self):
        """Test 128 Type B roots (half-integer coords)."""
        from syntonic.srt import e8_lattice
        E8 = e8_lattice()
        assert len(E8.type_b_roots) == 128

    def test_type_a_plus_b_equals_total(self):
        """Test Type A + Type B = 240."""
        from syntonic.srt import e8_lattice
        E8 = e8_lattice()
        assert len(E8.type_a_roots) + len(E8.type_b_roots) == 240

    def test_simple_roots_count(self):
        """Test 8 simple roots."""
        from syntonic.srt import e8_lattice
        E8 = e8_lattice()
        simple = E8.simple_roots()
        assert len(simple) == 8

    def test_simple_roots_norm(self):
        """Test simple roots have norm squared 2."""
        from syntonic.srt import e8_lattice
        from fractions import Fraction
        E8 = e8_lattice()
        simple = E8.simple_roots()
        for root in simple:
            # All simple roots have |α|² = 2
            assert root.norm_squared == Fraction(2)

    def test_cartan_matrix_shape(self):
        """Test Cartan matrix is 8x8."""
        from syntonic.srt import e8_lattice
        E8 = e8_lattice()
        cartan = E8.cartan_matrix()
        assert len(cartan) == 8
        assert all(len(row) == 8 for row in cartan)

    def test_cartan_matrix_diagonal(self):
        """Test Cartan matrix has 2s on diagonal."""
        from syntonic.srt import e8_lattice
        E8 = e8_lattice()
        cartan = E8.cartan_matrix()
        for i in range(8):
            assert cartan[i][i] == 2

    def test_highest_root(self):
        """Test highest root."""
        from syntonic.srt import e8_lattice
        E8 = e8_lattice()
        theta = E8.highest_root()
        assert theta.norm_squared == Fraction(2)

    def test_kissing_number(self):
        """Test K(E8) = 240."""
        from syntonic.srt import e8_lattice
        E8 = e8_lattice()
        assert E8.kissing_number == 240

    def test_coxeter_number(self):
        """Test Coxeter number h = 30."""
        from syntonic.srt import e8_lattice
        E8 = e8_lattice()
        assert E8.coxeter_number == 30


class TestD4Lattice:
    """Test D4Lattice class."""

    def test_creation(self):
        """Test D4Lattice creation."""
        from syntonic.srt import d4_lattice
        D4 = d4_lattice()
        assert D4 is not None

    def test_root_count(self):
        """Test D4 has exactly 24 roots."""
        from syntonic.srt import d4_lattice
        D4 = d4_lattice()
        assert len(D4.roots) == 24

    def test_positive_root_count(self):
        """Test D4 has exactly 12 positive roots."""
        from syntonic.srt import d4_lattice
        D4 = d4_lattice()
        assert len(D4.positive_roots) == 12

    def test_kissing_number(self):
        """Test K(D4) = 24."""
        from syntonic.srt import d4_lattice, K_D4
        D4 = d4_lattice()
        assert D4.kissing_number == 24
        assert K_D4 == 24

    def test_all_roots_norm_2(self):
        """Test all roots have |λ|² = 2."""
        from syntonic.srt import d4_lattice
        D4 = d4_lattice()
        for root in D4.roots:
            assert root.norm_squared == 2

    def test_simple_roots_count(self):
        """Test 4 simple roots."""
        from syntonic.srt import d4_lattice
        D4 = d4_lattice()
        simple = D4.simple_roots()
        assert len(simple) == 4

    def test_coxeter_number(self):
        """Test Coxeter number h = 6."""
        from syntonic.srt import d4_lattice
        D4 = d4_lattice()
        assert D4.coxeter_number == 6


class TestGoldenCone:
    """Test GoldenCone class."""

    def test_creation(self):
        """Test GoldenCone creation."""
        from syntonic.srt import golden_cone
        cone = golden_cone()
        assert cone is not None

    def test_root_count(self):
        """Test golden cone has exactly 36 roots."""
        from syntonic.srt import golden_cone
        cone = golden_cone()
        assert len(cone.roots) == 36

    def test_all_roots_in_cone(self):
        """Test all returned roots satisfy the 4 null vector conditions."""
        from syntonic.srt import golden_cone
        cone = golden_cone()
        for root in cone.roots:
            # Each root should pass the is_in_cone test
            assert cone.is_in_cone(root)

    def test_roots_are_e8_roots(self):
        """Test cone roots are E8 roots (|λ|² = 2)."""
        from syntonic.srt import golden_cone
        cone = golden_cone()
        for root in cone.roots:
            assert root.norm_squared == Fraction(2)

    def test_is_e6_positive_roots(self):
        """Test 36 = |Φ⁺(E6)|."""
        from syntonic.srt import golden_cone
        cone = golden_cone()
        # E6 has 72 roots, 36 positive
        assert cone.num_roots == 36


class TestQuadraticForm:
    """Test QuadraticForm class."""

    def test_creation(self):
        """Test QuadraticForm creation."""
        from syntonic.srt import quadratic_form
        Q = quadratic_form()
        assert Q is not None

    def test_phi_value(self):
        """Test phi is golden ratio."""
        from syntonic.srt import quadratic_form, PHI_NUMERIC
        Q = quadratic_form()
        assert abs(Q.phi - PHI_NUMERIC) < 1e-15

    def test_evaluate_returns_float(self):
        """Test evaluate returns float."""
        from syntonic.srt import quadratic_form
        Q = quadratic_form()
        v = (1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        result = Q.evaluate(v)
        assert isinstance(result, float)

    def test_classify_spacelike(self):
        """Test classification of spacelike vectors."""
        from syntonic.srt import quadratic_form
        Q = quadratic_form()
        # Find a vector with Q > 0
        v = (1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        result = Q.evaluate(v)
        if result > 1e-10:
            assert Q.classify(v) == 'spacelike'

    def test_projection_dimensions(self):
        """Test projection returns 4D vector."""
        from syntonic.srt import quadratic_form
        Q = quadratic_form()
        v = (1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0)
        proj_parallel = Q.project_parallel(v)
        proj_perp = Q.project_perpendicular(v)
        assert len(proj_parallel) == 4
        assert len(proj_perp) == 4

    def test_is_in_cone_consistent(self):
        """Test is_in_cone is consistent with evaluate."""
        from syntonic.srt import quadratic_form
        Q = quadratic_form()
        v = (1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        in_cone = Q.is_in_cone(v)
        Q_val = Q.evaluate(v)
        assert in_cone == (Q_val >= -1e-10)


class TestGoldenProjector:
    """Test GoldenProjector class."""

    def test_creation(self):
        """Test GoldenProjector creation."""
        from syntonic.srt import golden_projector
        P = golden_projector()
        assert P is not None

    def test_project_parallel(self):
        """Test parallel projection."""
        from syntonic.srt import golden_projector
        P = golden_projector()
        v = (1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        proj = P.project_parallel(v)
        assert len(proj) == 4

    def test_project_perpendicular(self):
        """Test perpendicular projection."""
        from syntonic.srt import golden_projector
        P = golden_projector()
        v = (1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        proj = P.project_perpendicular(v)
        assert len(proj) == 4


class TestLatticeIntegration:
    """Integration tests for lattice module."""

    def test_e8_contains_d4(self):
        """Test E8 structure relates to D4."""
        from syntonic.srt import e8_lattice, d4_lattice
        E8 = e8_lattice()
        D4 = d4_lattice()
        # D4 kissing number divides E8 kissing number
        assert E8.kissing_number % D4.kissing_number == 0

    def test_golden_cone_roots_are_e8_roots(self):
        """Test golden cone roots are E8 roots (from all 240)."""
        from syntonic.srt import e8_lattice, golden_cone
        E8 = e8_lattice()
        cone = golden_cone()

        all_roots = set(tuple(r.coords) for r in E8.roots)
        for root in cone.roots:
            assert tuple(root.coords) in all_roots
