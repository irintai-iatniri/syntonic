"""Integration tests for the complete SRT module."""

import pytest
import math


class TestSRTSystem:
    """Test SRTSystem class."""

    def test_creation(self):
        """Test SRTSystem creation."""
        from syntonic.srt import create_srt_system
        srt = create_srt_system()
        assert srt is not None

    def test_all_components_present(self):
        """Test all components are present."""
        from syntonic.srt import create_srt_system
        srt = create_srt_system()

        assert srt.torus is not None
        assert srt.golden_map is not None
        assert srt.golden_measure is not None
        assert srt.e8 is not None
        assert srt.d4 is not None
        assert srt.golden_cone is not None
        assert srt.theta is not None
        assert srt.heat is not None
        assert srt.laplacian is not None
        assert srt.syntony is not None

    def test_phi_consistent(self):
        """Test phi is consistent across all components."""
        from syntonic.srt import create_srt_system
        srt = create_srt_system()

        phi = srt.phi
        assert abs(srt.golden_map.phi - phi) < 1e-15
        assert abs(srt.golden_measure.phi - phi) < 1e-15
        assert abs(srt.theta.phi - phi) < 1e-15
        assert abs(srt.heat.phi - phi) < 1e-15
        assert abs(srt.laplacian.phi - phi) < 1e-15
        assert abs(srt.syntony.phi - phi) < 1e-15

    def test_vacuum_state(self):
        """Test vacuum state method."""
        from syntonic.srt import create_srt_system
        srt = create_srt_system()
        vacuum = srt.vacuum_state()
        assert vacuum.norm_squared == 0
        assert vacuum.is_vacuum

    def test_verify_e8_count(self):
        """Test E8 root count verification."""
        from syntonic.srt import create_srt_system
        srt = create_srt_system()
        assert srt.verify_e8_count()

    def test_verify_golden_cone_count(self):
        """Test golden cone count verification."""
        from syntonic.srt import create_srt_system
        srt = create_srt_system()
        assert srt.verify_golden_cone_count()

    def test_verify_d4_kissing(self):
        """Test D4 kissing number verification."""
        from syntonic.srt import create_srt_system
        srt = create_srt_system()
        assert srt.verify_d4_kissing()

    def test_verify_all(self):
        """Test all verifications pass."""
        from syntonic.srt import create_srt_system
        srt = create_srt_system()
        results = srt.verify_all()

        for name, (expected, actual, passed) in results.items():
            assert passed, f"{name}: expected {expected}, got {actual}"

    def test_summary(self):
        """Test summary generation."""
        from syntonic.srt import create_srt_system
        srt = create_srt_system()
        summary = srt.summary()
        assert "SRT System" in summary
        assert "E8" in summary
        assert "D4" in summary


class TestKeyIdentities:
    """Test key mathematical identities from the plan."""

    def test_e8_root_count_240(self):
        """Verify E8 has exactly 240 roots."""
        from syntonic.srt import e8_lattice
        E8 = e8_lattice()
        assert len(E8.roots) == 240

    def test_e8_positive_roots_120(self):
        """Verify E8 has exactly 120 positive roots."""
        from syntonic.srt import e8_lattice
        E8 = e8_lattice()
        assert len(E8.positive_roots) == 120

    def test_golden_cone_36(self):
        """Verify golden cone has exactly 36 roots."""
        from syntonic.srt import golden_cone
        cone = golden_cone()
        assert len(cone.roots) == 36

    def test_d4_kissing_24(self):
        """Verify D4 kissing number is 24."""
        from syntonic.srt import d4_lattice
        D4 = d4_lattice()
        assert D4.kissing_number == 24

    def test_e_star_value(self):
        """Verify E* = e^π - π ≈ 19.999."""
        from syntonic.srt import compute_e_star, E_STAR_NUMERIC
        E_star = compute_e_star()
        expected = math.exp(math.pi) - math.pi  # ≈ 19.999099979189474
        assert abs(E_star - expected) < 1e-6
        # E* = e^π - π ≈ 19.999
        assert abs(E_star - 19.999) < 0.01

    def test_all_roots_norm_2(self):
        """Verify all E8 roots have |λ|² = 2."""
        from syntonic.srt import e8_lattice
        from fractions import Fraction
        E8 = e8_lattice()
        for root in E8.roots:
            assert root.norm_squared == Fraction(2)

    def test_syntony_bound(self):
        """Verify S ≤ φ for all tested states."""
        from syntonic.srt import syntony_functional, winding_state, PHI_NUMERIC
        S = syntony_functional(max_norm=3)

        # Test vacuum
        vacuum = winding_state(0, 0, 0, 0)
        _, bounded = S.verify_bound({vacuum: complex(1.0)})
        assert bounded

        # Test some excited states
        for n7 in range(-2, 3):
            for n8 in range(-2, 3):
                n = winding_state(n7, n8, 0, 0)
                _, bounded = S.verify_bound({n: complex(1.0)})
                assert bounded


class TestModuleImports:
    """Test that all expected imports work."""

    def test_constants_import(self):
        """Test constants are importable."""
        from syntonic.srt import (
            PHI, PHI_SQUARED, PHI_INVERSE, PHI_NUMERIC,
            E_STAR_NUMERIC, Q_DEFICIT_NUMERIC, STRUCTURE_DIMENSIONS,
            TORUS_DIMENSIONS, E8_ROOTS, E8_POSITIVE_ROOTS,
            E6_GOLDEN_CONE, D4_KISSING,
        )

    def test_geometry_import(self):
        """Test geometry classes are importable."""
        from syntonic.srt import (
            WindingState, winding_state,
            T4Torus, t4_torus,
        )

    def test_golden_import(self):
        """Test golden classes are importable."""
        from syntonic.srt import (
            GoldenMeasure, golden_measure,
            GoldenRecursionMap, golden_recursion_map,
        )

    def test_lattice_import(self):
        """Test lattice classes are importable."""
        from syntonic.srt import (
            D4Root, D4Lattice, d4_lattice, K_D4,
            E8Root, E8Lattice, e8_lattice,
            GoldenProjector, golden_projector,
            GoldenCone, golden_cone,
            QuadraticForm, quadratic_form, compute_Q,
        )

    def test_spectral_import(self):
        """Test spectral classes are importable."""
        from syntonic.srt import (
            ThetaSeries, theta_series,
            HeatKernel, heat_kernel,
            KnotLaplacian, knot_laplacian,
            MobiusRegularizer, mobius_regularizer,
            compute_e_star,
        )

    def test_functional_import(self):
        """Test functional classes are importable."""
        from syntonic.srt import (
            SyntonyFunctional, syntony_functional,
            compute_syntony,
        )

    def test_system_import(self):
        """Test system classes are importable."""
        from syntonic.srt import (
            SRTSystem, create_srt_system,
        )


class TestCrossModuleConsistency:
    """Test consistency across modules."""

    def test_golden_cone_uses_e8(self):
        """Test golden cone properly filters E8 roots."""
        from syntonic.srt import e8_lattice, golden_cone
        E8 = e8_lattice()
        cone = golden_cone(e8=E8)

        # All cone roots should be in E8 roots (from all 240)
        e8_all = set(tuple(r.coords) for r in E8.roots)
        for root in cone.roots:
            assert tuple(root.coords) in e8_all

    def test_syntony_uses_laplacian(self):
        """Test syntony functional uses knot Laplacian."""
        from syntonic.srt import syntony_functional
        S = syntony_functional()
        L = S.laplacian
        assert L is not None

    def test_heat_kernel_uses_golden_measure(self):
        """Test heat kernel uses golden ratio."""
        from syntonic.srt import heat_kernel, PHI_NUMERIC
        K = heat_kernel()
        assert abs(K.phi - PHI_NUMERIC) < 1e-15
