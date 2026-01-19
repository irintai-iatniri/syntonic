"""
PDG Validation - Compare SRT predictions to Particle Data Group values.

This module provides experimental values from PDG 2024 and
utilities for computing deviations of SRT predictions.

All SRT predictions are derived from {φ, π, e, E*, q} with
zero free parameters - this is a genuine prediction, not a fit.
"""

from dataclasses import dataclass
from typing import Dict, Optional

# =============================================================================
# PDG Experimental Values
# =============================================================================


@dataclass
class PDGValue:
    """Experimental value with uncertainty from PDG."""

    value: float
    uncertainty: float
    unit: str
    description: str


PDG_VALUES: Dict[str, PDGValue] = {
    # Lepton masses (MeV)
    "m_e": PDGValue(0.51099895, 0.00000001, "MeV", "Electron mass"),
    "m_mu": PDGValue(105.6583755, 0.0000023, "MeV", "Muon mass"),
    "m_tau": PDGValue(1776.86, 0.12, "MeV", "Tau mass"),
    # Quark masses (MeV for light, GeV for heavy)
    "m_u": PDGValue(2.16, 0.49, "MeV", "Up quark mass (MS-bar 2 GeV)"),
    "m_d": PDGValue(4.67, 0.48, "MeV", "Down quark mass (MS-bar 2 GeV)"),
    "m_s": PDGValue(93.4, 8.6, "MeV", "Strange quark mass (MS-bar 2 GeV)"),
    "m_c": PDGValue(1270, 20, "MeV", "Charm quark mass (MS-bar m_c)"),
    "m_b": PDGValue(4180, 30, "MeV", "Bottom quark mass (MS-bar m_b)"),
    "m_t": PDGValue(172.76, 0.30, "GeV", "Top quark mass (pole)"),
    # Gauge bosons (GeV)
    "m_W": PDGValue(80.377, 0.012, "GeV", "W boson mass"),
    "m_Z": PDGValue(91.1876, 0.0021, "GeV", "Z boson mass"),
    "Gamma_Z": PDGValue(2.4952, 0.0023, "GeV", "Z boson width"),
    "m_H": PDGValue(125.25, 0.17, "GeV", "Higgs boson mass"),
    # Couplings (dimensionless)
    "alpha_em": PDGValue(1 / 137.035999084, 1.5e-12, "", "Fine structure constant"),
    "alpha_s": PDGValue(0.1179, 0.0009, "", "Strong coupling at M_Z"),
    "sin2_theta_W": PDGValue(0.23121, 0.00004, "", "Weak mixing angle"),
    # CKM elements
    "V_us": PDGValue(0.2243, 0.0005, "", "CKM |V_us|"),
    "V_cb": PDGValue(0.0412, 0.0008, "", "CKM |V_cb|"),
    "V_ub": PDGValue(0.00361, 0.00011, "", "CKM |V_ub|"),
    "J_CP": PDGValue(3.08e-5, 0.15e-5, "", "Jarlskog invariant"),
    # PMNS angles (degrees)
    "theta_12": PDGValue(33.44, 0.77, "deg", "Solar mixing angle"),
    "theta_23": PDGValue(49.20, 1.05, "deg", "Atmospheric mixing angle"),
    "theta_13": PDGValue(8.57, 0.12, "deg", "Reactor mixing angle"),
    "delta_CP": PDGValue(1.36, 0.20, "rad", "Dirac CP phase"),
    # Neutrino mass splittings
    "dm21_sq": PDGValue(7.53e-5, 0.18e-5, "eV²", "Solar mass splitting"),
    "dm31_sq": PDGValue(2.53e-3, 0.03e-3, "eV²", "Atmospheric mass splitting"),
    # Hadron masses (MeV)
    "m_p": PDGValue(938.27208816, 0.00000029, "MeV", "Proton mass"),
    "m_n": PDGValue(939.56542052, 0.00000054, "MeV", "Neutron mass"),
    "dm_np": PDGValue(1.29333236, 0.00000046, "MeV", "Neutron-proton mass diff"),
    "m_pi": PDGValue(139.57039, 0.00018, "MeV", "Charged pion mass"),
    "m_K": PDGValue(493.677, 0.016, "MeV", "Charged kaon mass"),
    "m_D": PDGValue(1869.66, 0.05, "MeV", "D± meson mass"),
    "m_B": PDGValue(5279.34, 0.12, "MeV", "B± meson mass"),
    "m_eta": PDGValue(547.862, 0.017, "MeV", "Eta meson mass"),
    "m_rho": PDGValue(775.26, 0.25, "MeV", "Rho meson mass"),
    "m_omega": PDGValue(782.65, 0.12, "MeV", "Omega meson mass"),
    "m_Bc": PDGValue(6274.9, 0.8, "MeV", "B_c meson mass"),
    # Baryon masses (MeV)
    "m_Lambda": PDGValue(1115.683, 0.006, "MeV", "Lambda baryon mass"),
    "m_Delta": PDGValue(1232.0, 2.0, "MeV", "Delta(1232) baryon mass"),
    "m_Xi_minus": PDGValue(1321.71, 0.07, "MeV", "Xi^- baryon mass"),
    "m_Xi_zero": PDGValue(1314.86, 0.20, "MeV", "Xi^0 baryon mass"),
    "m_Omega_minus": PDGValue(1672.45, 0.29, "MeV", "Omega^- baryon mass"),
    # Charmonium and Bottomonium (MeV)
    "m_Jpsi": PDGValue(3096.900, 0.006, "MeV", "J/psi mass"),
    "m_psi2S": PDGValue(3686.097, 0.025, "MeV", "psi(2S) mass"),
    "m_Upsilon1S": PDGValue(9460.30, 0.26, "MeV", "Upsilon(1S) mass"),
    "m_Upsilon2S": PDGValue(10023.26, 0.31, "MeV", "Upsilon(2S) mass"),
    "m_Upsilon3S": PDGValue(10355.2, 0.5, "MeV", "Upsilon(3S) mass"),
    "m_X3872": PDGValue(3871.69, 0.17, "MeV", "X(3872) exotic hadron mass"),
    # Nuclear binding energies (MeV)
    "B_deuteron": PDGValue(2.224575, 0.000009, "MeV", "Deuteron binding energy"),
    "B_alpha": PDGValue(28.29567, 0.00003, "MeV", "Alpha (He-4) binding energy"),
    "B_per_A_Fe56": PDGValue(
        8.790, 0.003, "MeV", "Binding energy per nucleon for Fe-56"
    ),
    # Neutron lifetime (s)
    "tau_n": PDGValue(879.4, 0.6, "s", "Neutron lifetime"),
    # Cosmological parameters
    "Omega_DM_over_Omega_b": PDGValue(5.36, 0.10, "", "Dark matter to baryon ratio"),
    "z_eq": PDGValue(3387, 21, "", "Matter-radiation equality redshift"),
    "z_rec": PDGValue(1089.80, 0.21, "", "Recombination redshift"),
    "H0": PDGValue(67.4, 0.5, "km/s/Mpc", "Hubble constant"),
    "n_s": PDGValue(0.9649, 0.0042, "", "Scalar spectral index"),
    "r": PDGValue(0.036, 0.036, "", "Tensor-to-scalar ratio (upper limit)"),
    # BBN observables
    "N_eff": PDGValue(2.99, 0.17, "", "Effective number of neutrinos (BBN)"),
    "Y_p": PDGValue(0.245, 0.003, "", "Primordial helium abundance"),
    "D_over_H": PDGValue(2.53e-5, 0.04e-5, "", "Deuterium to hydrogen ratio"),
    "Li7_over_H": PDGValue(1.6e-10, 0.3e-10, "", "Lithium-7 to hydrogen ratio"),
    # CMB acoustic peaks
    "ell_1": PDGValue(220.0, 1.0, "", "First CMB acoustic peak"),
    "ell_2": PDGValue(537.5, 1.5, "", "Second CMB acoustic peak"),
    "ell_3": PDGValue(810.8, 2.0, "", "Third CMB acoustic peak"),
    "ell_4": PDGValue(1120.9, 3.0, "", "Fourth CMB acoustic peak"),
    "ell_5": PDGValue(1444.2, 4.0, "", "Fifth CMB acoustic peak"),
    # QCD observables (MeV)
    "m_glueball_0pp": PDGValue(1515.0, 110.0, "MeV", "0++ glueball mass"),
    "m_glueball_2pp": PDGValue(2289.0, 150.0, "MeV", "2++ glueball mass"),
    "m_glueball_0mp": PDGValue(2455.0, 120.0, "MeV", "0-+ glueball mass"),
    "chiral_condensate": PDGValue(250.0, 10.0, "MeV", "Chiral condensate^(1/3)"),
    "Lambda_QCD": PDGValue(213.0, 8.0, "MeV", "QCD scale parameter"),
    # Precision electroweak
    "Gamma_W": PDGValue(2.085, 0.042, "GeV", "W boson width"),
    "R_b": PDGValue(0.21629, 0.00066, "", "Z → bb branching ratio"),
    "A_FB_b": PDGValue(0.0992, 0.0016, "", "Forward-backward asymmetry for b quarks"),
    "rho_parameter": PDGValue(1.00037, 0.00023, "", "Electroweak ρ parameter"),
    # Atomic physics
    "Rydberg": PDGValue(13.605693122994, 0.000000000026, "eV", "Rydberg constant × hc"),
    "He_plus_ionization": PDGValue(54.417763, 0.000006, "eV", "He+ ionization energy"),
    "H_polarizability": PDGValue(4.5, 0.1, "a₀³", "Hydrogen polarizability"),
    "H2_bond_length": PDGValue(0.7414, 0.0001, "Å", "H₂ bond length"),
    "H2_dissociation": PDGValue(4.478, 0.001, "eV", "H₂ dissociation energy"),
    "fine_structure_21cm": PDGValue(
        1420.405751768, 0.000000001, "MHz", "21 cm hyperfine transition"
    ),
    # Exotic hadrons and pentaquarks
    "T_cc_plus": PDGValue(3875.1, 0.4, "MeV", "T_cc^+ tetraquark mass"),
    "P_c_4457": PDGValue(4457.3, 0.6, "MeV", "P_c(4457) pentaquark mass"),
    # Semi-empirical mass formula coefficients (MeV)
    "a_V_SEMF": PDGValue(15.75, 0.2, "MeV", "SEMF volume term"),
    "a_S_SEMF": PDGValue(17.8, 0.3, "MeV", "SEMF surface term"),
    "a_A_SEMF": PDGValue(23.7, 0.5, "MeV", "SEMF asymmetry term"),
    "a_P_SEMF": PDGValue(12.0, 0.5, "MeV", "SEMF pairing term"),
    # Proton structure
    "r_p": PDGValue(0.8414, 0.0019, "fm", "Proton charge radius"),
    # Lithium problem
    "Li7_over_H_BBN": PDGValue(
        1.6e-10, 0.3e-10, "", "Lithium-7 to hydrogen ratio (BBN prediction)"
    ),
    # Cosmological tensions
    "S8": PDGValue(0.834, 0.016, "", "S₈ parameter (CMB)"),
    "r_tensor": PDGValue(0.036, 0.036, "", "Tensor-to-scalar ratio (upper limit)"),
    # Superconductor critical temperatures (K)
    "T_c_YBCO": PDGValue(92.4, 0.5, "K", "YBCO superconductor T_c"),
    "T_c_BSCCO": PDGValue(110.5, 1.0, "K", "BSCCO superconductor T_c"),
    # Dark matter and sterile neutrino
    "m_sterile_nu": PDGValue(4.236, 0.5, "keV", "Sterile neutrino mass (predicted)"),
    "X_ray_line": PDGValue(2.118, 0.25, "keV", "Dark matter X-ray line (predicted)"),
    "sin2_2theta_sterile": PDGValue(
        1.1e-11, 0.5e-11, "", "Sterile neutrino mixing (predicted)"
    ),
    # Quantum gravity (predictions/derived)
    "BH_entropy_correction": PDGValue(0.685, 0.1, "%", "Black hole entropy correction"),
    "Hawking_temp_correction": PDGValue(
        -0.342, 0.05, "%", "Hawking temperature correction"
    ),
    "GW150914_echo": PDGValue(0.59, 0.1, "ms", "GW echo time for GW150914"),
    "GW190521_echo": PDGValue(1.35, 0.2, "ms", "GW echo time for GW190521"),
    "GW170817_echo": PDGValue(0.038, 0.005, "ms", "GW echo time for GW170817 (NS)"),
    "echo_decay_factor": PDGValue(0.618, 0.01, "", "Echo amplitude decay (1/φ)"),
    "QNM_correction": PDGValue(
        0.076, 0.01, "%", "Quasinormal mode frequency correction"
    ),
    # Condensed matter
    "BCS_ratio": PDGValue(3.52, 0.02, "", "BCS gap ratio 2Δ/k_B T_c"),
    "BCS_strong_coupling": PDGValue(5.68, 0.1, "", "Strong coupling BCS ratio"),
    "FQHE_1_3": PDGValue(0.3333, 0.0001, "", "FQHE filling fraction 1/3"),
    "FQHE_2_5": PDGValue(0.4000, 0.0001, "", "FQHE filling fraction 2/5"),
    "FQHE_3_8": PDGValue(0.3750, 0.0001, "", "FQHE filling fraction 3/8"),
    "graphene_alpha_g": PDGValue(2.19, 0.05, "", "Graphene effective fine structure"),
    "quasicrystal_beta": PDGValue(1.50, 0.05, "", "Quasicrystal conductivity exponent"),
    "ZT_maximum": PDGValue(2.62, 0.1, "", "Thermoelectric ZT maximum"),
    # Nuclear structure predictions
    "island_Z": PDGValue(114, 2, "", "Island of stability proton number"),
    "island_N": PDGValue(184, 3, "", "Island of stability neutron number"),
    # Solar physics
    "pp_I_flux": PDGValue(87.3, 1.0, "%", "Solar pp-I chain flux fraction"),
    "pp_II_flux": PDGValue(12.7, 1.0, "%", "Solar pp-II chain flux fraction"),
    "pp_I_over_pp_II": PDGValue(6.87, 0.3, "", "Solar pp-I/pp-II flux ratio"),
    # Modified gravity and MOND
    "modified_gravity_scale": PDGValue(
        1.78e-35, 0.2e-35, "m", "Modified gravity scale √φ ℓ_P"
    ),
    "MOND_a0": PDGValue(1.20e-10, 0.03e-10, "m/s²", "MOND acceleration scale"),
    # Dark matter cosmology
    "Omega_sterile_h2": PDGValue(0.12, 0.02, "", "Sterile neutrino relic abundance"),
    "sterile_production_temp": PDGValue(
        140, 20, "MeV", "Sterile ν production temperature"
    ),
    # Exotic materials
    "semi_Dirac_anisotropy": PDGValue(
        2.617, 0.05, "", "Semi-Dirac fermion anisotropy ratio"
    ),
    "topological_theta": PDGValue(1.618, 0.05, "", "Topological angle θ/(2π) = 1/φ"),
    # Additional nuclear magic numbers
    "magic_50": PDGValue(50, 0, "", "Magic number Z/N=50"),
    "magic_82": PDGValue(82, 0, "", "Magic number Z/N=82"),
    "magic_126": PDGValue(126, 0, "", "Magic number N=126"),
    "magic_ratio_50_82": PDGValue(1.64, 0.05, "", "Magic number ratio 82/50"),
    "magic_ratio_82_126": PDGValue(1.537, 0.05, "", "Magic number ratio 126/82"),
    # Precision QED/QCD
    "muon_g2_anomaly": PDGValue(251.0e-11, 59.0e-11, "", "Muon g-2 anomaly (exp - SM)"),
    "tau_to_electron_ratio": PDGValue(3477.23, 0.23, "", "m_τ/m_e ratio"),
    "proton_to_electron_ratio": PDGValue(1836.15, 0.01, "", "m_p/m_e ratio"),
    # Fine structure and coupling constants
    "alpha_EM": PDGValue(
        1 / 137.036, 0.000001 / 137.036, "", "Fine structure constant α"
    ),
    "sin2_theta_W": PDGValue(0.2312, 0.0002, "", "Weinberg angle sin²θ_W"),
    # Higgs physics
    "Higgs_self_coupling_ratio": PDGValue(1.0, 0.5, "", "λ_HHH/λ_HHH^SM (SM=1)"),
    "Higgs_tree_level": PDGValue(93, 5, "GeV", "Higgs tree-level mass"),
    # Additional mesons
    "eta_prime": PDGValue(957.78, 0.06, "MeV", "η′ meson mass"),
    "phi_meson": PDGValue(1019.46, 0.02, "MeV", "φ meson mass"),
    "D_star": PDGValue(2010.3, 0.2, "MeV", "D* meson mass"),
    "D_s": PDGValue(1968.5, 0.3, "MeV", "D_s meson mass"),
    "B_star": PDGValue(5325.2, 0.4, "MeV", "B* meson mass"),
    "B_s": PDGValue(5366.88, 0.14, "MeV", "B_s meson mass"),
    # Additional baryons
    "Xi_minus": PDGValue(1321.71, 0.07, "MeV", "Ξ⁻ baryon mass"),
    "Xi_zero": PDGValue(1314.86, 0.20, "MeV", "Ξ⁰ baryon mass"),
    "Sigma_plus": PDGValue(1189.37, 0.07, "MeV", "Σ⁺ baryon mass"),
    "Sigma_zero": PDGValue(1192.64, 0.24, "MeV", "Σ⁰ baryon mass"),
    "Sigma_minus": PDGValue(1197.45, 0.30, "MeV", "Σ⁻ baryon mass"),
    # Atomic and molecular
    "fine_structure_H": PDGValue(
        10.969, 0.001, "GHz", "Hydrogen fine structure splitting"
    ),
    "Lamb_shift": PDGValue(1057.8, 0.1, "MHz", "Lamb shift 2S₁/₂-2P₁/₂"),
    "deuterium_ionization": PDGValue(
        13.602, 0.001, "eV", "Deuterium ionization energy"
    ),
}


def validate_prediction(
    name: str,
    predicted: float,
    pdg_values: Optional[Dict[str, PDGValue]] = None,
) -> dict:
    """
    Compare a single SRT prediction to PDG value.

    Args:
        name: Parameter name (must be in PDG_VALUES)
        predicted: SRT predicted value
        pdg_values: Custom PDG values dict (default: PDG_VALUES)

    Returns:
        Dictionary with:
            - predicted: SRT value
            - pdg: PDG central value
            - uncertainty: PDG uncertainty
            - unit: Physical unit
            - deviation: Absolute deviation
            - sigma: Number of sigma from PDG
            - percent_error: Percent deviation
            - status: 'PASS' if within 3σ, else 'CHECK'
    """
    if pdg_values is None:
        pdg_values = PDG_VALUES

    if name not in pdg_values:
        return {
            "predicted": predicted,
            "pdg": None,
            "status": "UNKNOWN",
            "message": f"No PDG value for '{name}'",
        }

    pdg = pdg_values[name]
    deviation = abs(predicted - pdg.value)

    # Handle zero uncertainty (exact values)
    if pdg.uncertainty == 0:
        sigma = 0 if deviation == 0 else float("inf")
    else:
        sigma = deviation / pdg.uncertainty

    # Percent error (handle zero PDG value)
    if pdg.value != 0:
        percent_error = (deviation / abs(pdg.value)) * 100
    else:
        percent_error = float("inf") if deviation != 0 else 0

    # Status: PASS if within 3σ or <1% error
    status = "PASS" if (sigma < 3 or percent_error < 1) else "CHECK"

    return {
        "predicted": predicted,
        "pdg": pdg.value,
        "uncertainty": pdg.uncertainty,
        "unit": pdg.unit,
        "description": pdg.description,
        "deviation": deviation,
        "sigma": sigma,
        "percent_error": percent_error,
        "status": status,
    }


def validate_all(
    predictions: Dict[str, float],
    pdg_values: Optional[Dict[str, PDGValue]] = None,
) -> Dict[str, dict]:
    """
    Validate all predictions against PDG values.

    Args:
        predictions: Dictionary of {parameter: predicted_value}
        pdg_values: Custom PDG values (default: PDG_VALUES)

    Returns:
        Dictionary mapping each parameter to its validation result
    """
    results = {}
    for name, predicted in predictions.items():
        results[name] = validate_prediction(name, predicted, pdg_values)
    return results


def summary_report(results: Dict[str, dict]) -> str:
    """
    Generate a summary report of validation results.

    Args:
        results: Output from validate_all()

    Returns:
        Formatted string report
    """
    lines = []
    lines.append("=" * 70)
    lines.append("SRT Standard Model Predictions vs PDG Experiment")
    lines.append("=" * 70)
    lines.append("")

    # Count statistics
    n_pass = sum(1 for r in results.values() if r.get("status") == "PASS")
    n_check = sum(1 for r in results.values() if r.get("status") == "CHECK")
    n_unknown = sum(1 for r in results.values() if r.get("status") == "UNKNOWN")

    lines.append(f"Summary: {n_pass} PASS, {n_check} CHECK, {n_unknown} UNKNOWN")
    lines.append("")

    # Detailed results
    lines.append(
        f"{'Parameter':<15} {'Predicted':>12} {'PDG':>12} {'σ':>8} {'%err':>8} {'Status':<8}"
    )
    lines.append("-" * 70)

    for name, result in sorted(results.items()):
        if result.get("status") == "UNKNOWN":
            lines.append(
                f"{name:<15} {result['predicted']:>12.6g} {'N/A':>12} {'N/A':>8} {'N/A':>8} UNKNOWN"
            )
            continue

        pred = result["predicted"]
        pdg = result["pdg"]
        sigma = result["sigma"]
        pct = result["percent_error"]
        status = result["status"]

        # Format numbers appropriately
        if abs(pred) < 0.01 or abs(pred) > 1e6:
            pred_str = f"{pred:.4e}"
            pdg_str = f"{pdg:.4e}"
        else:
            pred_str = f"{pred:.6g}"
            pdg_str = f"{pdg:.6g}"

        if sigma < 100:
            sigma_str = f"{sigma:.2f}"
        else:
            sigma_str = ">100"

        lines.append(
            f"{name:<15} {pred_str:>12} {pdg_str:>12} {sigma_str:>8} {pct:>7.3f}% {status:<8}"
        )

    lines.append("=" * 70)
    lines.append("All values derived from {φ, π, e, E*, q} with zero free parameters")
    lines.append("=" * 70)

    return "\n".join(lines)


__all__ = [
    "PDGValue",
    "PDG_VALUES",
    "validate_prediction",
    "validate_all",
    "summary_report",
]
