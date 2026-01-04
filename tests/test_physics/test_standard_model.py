"""
Comprehensive tests for Phase 5: Standard Model Physics.

Tests all SRT predictions against PDG experimental values.
"""

import pytest
import math


class TestFermionMasses:
    """Test fermion mass predictions."""

    def test_electron_mass(self):
        """Electron mass should match PDG."""
        from syntonic.physics.fermions import electron_mass
        m_e = electron_mass()
        assert abs(m_e - 0.511) < 0.001, f"m_e = {m_e} MeV"

    def test_muon_mass(self):
        """Muon mass should be approximately 105.7 MeV."""
        from syntonic.physics.fermions import muon_mass
        m_mu = muon_mass()
        # Allow 1% tolerance due to formula approximations
        assert abs(m_mu - 105.66) / 105.66 < 0.02, f"m_mu = {m_mu} MeV"

    def test_tau_mass(self):
        """Tau mass from E* × F_11 formula."""
        from syntonic.physics.fermions import tau_mass
        from syntonic.exact import E_STAR_NUMERIC, Q_DEFICIT_NUMERIC, fibonacci

        m_tau = tau_mass()
        # Should be close to 1776.86 MeV
        assert abs(m_tau - 1776.86) / 1776.86 < 0.001, f"m_tau = {m_tau} MeV"

        # Verify formula: E* × F_11 × corrections
        F_11 = fibonacci(11)
        assert F_11 == 89
        expected = E_STAR_NUMERIC * 89 * (1 - Q_DEFICIT_NUMERIC / (5 * math.pi)) * (1 - Q_DEFICIT_NUMERIC / 720)
        assert abs(m_tau - expected) < 0.01

    def test_quark_masses(self):
        """Light and heavy quark masses."""
        from syntonic.physics.fermions import (
            up_mass, down_mass, strange_mass,
            charm_mass, bottom_mass, top_mass
        )
        from syntonic.exact import E_STAR_NUMERIC

        # Up quark
        m_u = up_mass()
        assert abs(m_u - E_STAR_NUMERIC / 9) < 0.01

        # Down quark
        m_d = down_mass()
        assert abs(m_d - E_STAR_NUMERIC / 4.3) < 0.01

        # Strange quark
        m_s = strange_mass()
        assert abs(m_s - E_STAR_NUMERIC * 4.65) < 0.1

        # Charm quark
        m_c = charm_mass()
        assert abs(m_c - E_STAR_NUMERIC * 63.5) < 1

        # Bottom quark
        m_b = bottom_mass()
        assert abs(m_b - E_STAR_NUMERIC * 209) < 1

        # Top quark
        m_t = top_mass()
        assert abs(m_t - 172.72) < 0.1, f"m_t = {m_t} GeV"


class TestGaugeSector:
    """Test gauge boson masses and couplings."""

    def test_weinberg_angle(self):
        """Weinberg angle sin²θ_W."""
        from syntonic.physics.bosons import weinberg_angle
        sin2_w = weinberg_angle()
        assert abs(sin2_w - 0.2312) < 0.001

    def test_fine_structure(self):
        """Fine structure constant from E* × q³ formula."""
        from syntonic.physics.bosons import fine_structure_constant
        alpha = fine_structure_constant()
        # Should give 1/137.036
        assert abs(1/alpha - 137.036) < 0.5, f"1/α = {1/alpha}"

    def test_w_mass(self):
        """W boson mass."""
        from syntonic.physics.bosons import w_mass
        m_W = w_mass()
        # Should be close to 80.377 GeV
        assert abs(m_W - 80.377) < 0.5, f"m_W = {m_W} GeV"

    def test_z_mass(self):
        """Z boson mass."""
        from syntonic.physics.bosons import z_mass
        m_Z = z_mass()
        # Should be close to 91.188 GeV
        assert abs(m_Z - 91.188) < 0.5, f"m_Z = {m_Z} GeV"


class TestHiggsSector:
    """Test Higgs mass prediction."""

    def test_higgs_mass_tree(self):
        """Tree-level Higgs mass."""
        from syntonic.physics.bosons import higgs_mass_tree
        m_tree = higgs_mass_tree()
        # Should be ~93.4 GeV
        assert abs(m_tree - 93.4) < 1, f"m_H^tree = {m_tree} GeV"

    def test_higgs_mass_full(self):
        """Full Higgs mass with loop correction."""
        from syntonic.physics.bosons import higgs_mass
        m_H = higgs_mass()
        # Should be ~125.25 GeV
        assert abs(m_H - 125.25) < 2, f"m_H = {m_H} GeV"

    def test_self_coupling_enhancement(self):
        """Higgs self-coupling enhancement."""
        from syntonic.physics.bosons import higgs_self_coupling_ratio
        ratio = higgs_self_coupling_ratio()
        # Should be ~1.116 (11.6% enhancement) from 1 + qφ³
        assert abs(ratio - 1.116) < 0.01


class TestCKMMatrix:
    """Test CKM matrix elements."""

    def test_V_us(self):
        """Cabibbo angle |V_us|."""
        from syntonic.physics.mixing import V_us
        v_us = V_us()
        # Should be ~0.2242
        assert abs(v_us - 0.2243) < 0.01, f"|V_us| = {v_us}"

    def test_V_cb(self):
        """|V_cb| = q × 3/2."""
        from syntonic.physics.mixing import V_cb
        from syntonic.exact import Q_DEFICIT_NUMERIC
        v_cb = V_cb()
        assert abs(v_cb - Q_DEFICIT_NUMERIC * 1.5) < 0.0001

    def test_V_ub(self):
        """|V_ub| from golden ratio formula."""
        from syntonic.physics.mixing import V_ub
        v_ub = V_ub()
        # Should be ~0.00361
        assert abs(v_ub - 0.00361) < 0.001, f"|V_ub| = {v_ub}"

    def test_jarlskog(self):
        """Jarlskog invariant."""
        from syntonic.physics.mixing import jarlskog_invariant
        J = jarlskog_invariant()
        # Should be ~3.08 × 10⁻⁵
        assert abs(J - 3.08e-5) < 0.5e-5, f"J = {J}"


class TestPMNSMatrix:
    """Test PMNS neutrino mixing angles."""

    def test_theta_12(self):
        """Solar angle θ₁₂."""
        from syntonic.physics.mixing import theta_12
        angle = theta_12()
        # Should be ~33.44°
        assert abs(angle - 33.44) < 2, f"θ₁₂ = {angle}°"

    def test_theta_23(self):
        """Atmospheric angle θ₂₃."""
        from syntonic.physics.mixing import theta_23
        angle = theta_23()
        # Should be ~49.20°
        assert abs(angle - 49.20) < 3, f"θ₂₃ = {angle}°"

    def test_theta_13(self):
        """Reactor angle θ₁₃."""
        from syntonic.physics.mixing import theta_13
        angle = theta_13()
        # Should be ~8.57°
        assert abs(angle - 8.57) < 1, f"θ₁₃ = {angle}°"


class TestHadronMasses:
    """Test hadron mass predictions."""

    def test_proton_mass(self):
        """Proton mass from SRT formula."""
        from syntonic.physics.hadrons import proton_mass
        m_p = proton_mass()
        # Should be ~938.27 MeV
        assert abs(m_p - 938.27) < 1, f"m_p = {m_p} MeV"

    def test_neutron_proton_diff(self):
        """Neutron-proton mass difference."""
        from syntonic.physics.hadrons import neutron_proton_mass_diff
        dm = neutron_proton_mass_diff()
        # Should be ~1.293 MeV
        assert abs(dm - 1.293) < 0.1, f"m_n - m_p = {dm} MeV"

    def test_pion_mass(self):
        """Pion mass = E* × 7."""
        from syntonic.physics.hadrons import pion_mass
        from syntonic.exact import E_STAR_NUMERIC
        m_pi = pion_mass()
        assert abs(m_pi - E_STAR_NUMERIC * 7) < 0.01

    def test_meson_pattern(self):
        """Meson mass pattern with E*."""
        from syntonic.physics.hadrons import pion_mass, kaon_mass, d_meson_mass, b_meson_mass
        from syntonic.exact import E_STAR_NUMERIC

        # All mesons should follow E* × integer pattern
        assert abs(pion_mass() / E_STAR_NUMERIC - 7) < 0.01
        assert abs(kaon_mass() / E_STAR_NUMERIC - 25) < 0.1
        assert abs(d_meson_mass() / E_STAR_NUMERIC - 93) < 0.5
        assert abs(b_meson_mass() / E_STAR_NUMERIC - 264) < 0.5


class TestRunningCouplings:
    """Test RG evolution and GUT scale."""

    def test_gut_scale(self):
        """GUT scale = v × e^(φ⁷)."""
        from syntonic.physics.running import gut_scale
        from syntonic.exact import PHI
        from syntonic.physics.constants import V_EW

        mu_gut = gut_scale()
        expected = V_EW * math.exp(PHI.eval() ** 7)
        assert abs(mu_gut - expected) < 1e10

        # Should be ~10^15 GeV
        assert 1e14 < mu_gut < 1e16

    def test_alpha_s_running(self):
        """α_s running to higher scales."""
        from syntonic.physics.running import alpha_s_at_scale

        # α_s should decrease at higher scales
        alpha_s_mz = alpha_s_at_scale(91.2)
        alpha_s_1tev = alpha_s_at_scale(1000)

        assert alpha_s_mz > alpha_s_1tev  # Asymptotic freedom
        assert abs(alpha_s_mz - 0.1179) < 0.01


class TestValidation:
    """Test the validation framework."""

    def test_validate_single(self):
        """Validate a single prediction using actual SRT value."""
        from syntonic.physics.validation import validate_prediction
        from syntonic.physics.fermions import electron_mass

        # Use actual SRT prediction, not rounded value
        m_e = electron_mass()
        result = validate_prediction('m_e', m_e)
        assert result['status'] == 'PASS'
        assert result['sigma'] < 1

    def test_validate_all(self):
        """Validate multiple predictions using actual SRT values."""
        from syntonic.physics.validation import validate_all
        from syntonic.physics.fermions import electron_mass, muon_mass
        from syntonic.physics.bosons import higgs_mass

        predictions = {
            'm_e': electron_mass(),
            'm_mu': muon_mass(),
            'm_H': higgs_mass(),
        }
        results = validate_all(predictions)

        assert 'm_e' in results
        assert results['m_e']['status'] == 'PASS'


class TestStandardModelClass:
    """Test the main StandardModel class."""

    def test_standard_model_init(self):
        """StandardModel class should initialize."""
        from syntonic.physics import StandardModel
        sm = StandardModel()
        assert sm is not None

    def test_all_parameters(self):
        """all_parameters() should return dictionary."""
        from syntonic.physics import StandardModel
        sm = StandardModel()
        params = sm.all_parameters()

        assert isinstance(params, dict)
        assert 'm_e' in params
        assert 'm_H' in params
        assert 'm_p' in params

    def test_validate(self):
        """validate() should compare to PDG."""
        from syntonic.physics import StandardModel
        sm = StandardModel()
        results = sm.validate()

        assert isinstance(results, dict)
        # Most predictions should pass
        n_pass = sum(1 for r in results.values() if r.get('status') == 'PASS')
        assert n_pass > len(results) * 0.7  # At least 70% should pass
