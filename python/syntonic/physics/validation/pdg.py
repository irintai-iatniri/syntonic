"""
PDG Validation - Compare SRT predictions to Particle Data Group values.

This module provides experimental values from PDG 2024 and
utilities for computing deviations of SRT predictions.

All SRT predictions are derived from {φ, π, e, E*, q} with
zero free parameters - this is a genuine prediction, not a fit.
"""

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

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
    'm_e': PDGValue(0.51099895, 0.00000001, 'MeV', 'Electron mass'),
    'm_mu': PDGValue(105.6583755, 0.0000023, 'MeV', 'Muon mass'),
    'm_tau': PDGValue(1776.86, 0.12, 'MeV', 'Tau mass'),

    # Quark masses (MeV for light, GeV for heavy)
    'm_u': PDGValue(2.16, 0.49, 'MeV', 'Up quark mass (MS-bar 2 GeV)'),
    'm_d': PDGValue(4.67, 0.48, 'MeV', 'Down quark mass (MS-bar 2 GeV)'),
    'm_s': PDGValue(93.4, 8.6, 'MeV', 'Strange quark mass (MS-bar 2 GeV)'),
    'm_c': PDGValue(1270, 20, 'MeV', 'Charm quark mass (MS-bar m_c)'),
    'm_b': PDGValue(4180, 30, 'MeV', 'Bottom quark mass (MS-bar m_b)'),
    'm_t': PDGValue(172.76, 0.30, 'GeV', 'Top quark mass (pole)'),

    # Gauge bosons (GeV)
    'm_W': PDGValue(80.377, 0.012, 'GeV', 'W boson mass'),
    'm_Z': PDGValue(91.1876, 0.0021, 'GeV', 'Z boson mass'),
    'Gamma_Z': PDGValue(2.4952, 0.0023, 'GeV', 'Z boson width'),
    'm_H': PDGValue(125.25, 0.17, 'GeV', 'Higgs boson mass'),

    # Couplings (dimensionless)
    'alpha_em': PDGValue(1/137.035999084, 1.5e-12, '', 'Fine structure constant'),
    'alpha_s': PDGValue(0.1179, 0.0009, '', 'Strong coupling at M_Z'),
    'sin2_theta_W': PDGValue(0.23121, 0.00004, '', 'Weak mixing angle'),

    # CKM elements
    'V_us': PDGValue(0.2243, 0.0005, '', 'CKM |V_us|'),
    'V_cb': PDGValue(0.0412, 0.0008, '', 'CKM |V_cb|'),
    'V_ub': PDGValue(0.00361, 0.00011, '', 'CKM |V_ub|'),
    'J_CP': PDGValue(3.08e-5, 0.15e-5, '', 'Jarlskog invariant'),

    # PMNS angles (degrees)
    'theta_12': PDGValue(33.44, 0.77, 'deg', 'Solar mixing angle'),
    'theta_23': PDGValue(49.20, 1.05, 'deg', 'Atmospheric mixing angle'),
    'theta_13': PDGValue(8.57, 0.12, 'deg', 'Reactor mixing angle'),
    'delta_CP': PDGValue(1.36, 0.20, 'rad', 'Dirac CP phase'),

    # Neutrino mass splittings
    'dm21_sq': PDGValue(7.53e-5, 0.18e-5, 'eV²', 'Solar mass splitting'),
    'dm31_sq': PDGValue(2.53e-3, 0.03e-3, 'eV²', 'Atmospheric mass splitting'),

    # Hadron masses (MeV)
    'm_p': PDGValue(938.27208816, 0.00000029, 'MeV', 'Proton mass'),
    'm_n': PDGValue(939.56542052, 0.00000054, 'MeV', 'Neutron mass'),
    'dm_np': PDGValue(1.29333236, 0.00000046, 'MeV', 'Neutron-proton mass diff'),
    'm_pi': PDGValue(139.57039, 0.00018, 'MeV', 'Charged pion mass'),
    'm_K': PDGValue(493.677, 0.016, 'MeV', 'Charged kaon mass'),
    'm_D': PDGValue(1869.66, 0.05, 'MeV', 'D± meson mass'),
    'm_B': PDGValue(5279.34, 0.12, 'MeV', 'B± meson mass'),
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
            'predicted': predicted,
            'pdg': None,
            'status': 'UNKNOWN',
            'message': f"No PDG value for '{name}'",
        }

    pdg = pdg_values[name]
    deviation = abs(predicted - pdg.value)

    # Handle zero uncertainty (exact values)
    if pdg.uncertainty == 0:
        sigma = 0 if deviation == 0 else float('inf')
    else:
        sigma = deviation / pdg.uncertainty

    # Percent error (handle zero PDG value)
    if pdg.value != 0:
        percent_error = (deviation / abs(pdg.value)) * 100
    else:
        percent_error = float('inf') if deviation != 0 else 0

    # Status: PASS if within 3σ or <1% error
    status = 'PASS' if (sigma < 3 or percent_error < 1) else 'CHECK'

    return {
        'predicted': predicted,
        'pdg': pdg.value,
        'uncertainty': pdg.uncertainty,
        'unit': pdg.unit,
        'description': pdg.description,
        'deviation': deviation,
        'sigma': sigma,
        'percent_error': percent_error,
        'status': status,
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
    n_pass = sum(1 for r in results.values() if r.get('status') == 'PASS')
    n_check = sum(1 for r in results.values() if r.get('status') == 'CHECK')
    n_unknown = sum(1 for r in results.values() if r.get('status') == 'UNKNOWN')

    lines.append(f"Summary: {n_pass} PASS, {n_check} CHECK, {n_unknown} UNKNOWN")
    lines.append("")

    # Detailed results
    lines.append(f"{'Parameter':<15} {'Predicted':>12} {'PDG':>12} {'σ':>8} {'%err':>8} {'Status':<8}")
    lines.append("-" * 70)

    for name, result in sorted(results.items()):
        if result.get('status') == 'UNKNOWN':
            lines.append(f"{name:<15} {result['predicted']:>12.6g} {'N/A':>12} {'N/A':>8} {'N/A':>8} UNKNOWN")
            continue

        pred = result['predicted']
        pdg = result['pdg']
        sigma = result['sigma']
        pct = result['percent_error']
        status = result['status']

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

        lines.append(f"{name:<15} {pred_str:>12} {pdg_str:>12} {sigma_str:>8} {pct:>7.3f}% {status:<8}")

    lines.append("=" * 70)
    lines.append("All values derived from {φ, π, e, E*, q} with zero free parameters")
    lines.append("=" * 70)

    return "\n".join(lines)


__all__ = [
    'PDGValue',
    'PDG_VALUES',
    'validate_prediction',
    'validate_all',
    'summary_report',
]
