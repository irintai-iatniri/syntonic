"""
SRT-Zero: Auto-Miner
====================
Automated Geometric Discovery Engine for Unsolved Physical Constants.

Features:
- Progress bars with tqdm
- Structured logging
- JSON results export
- Full hierarchy integration
"""

import json
import logging
import time
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any
from tqdm import tqdm
from .config import get_config
from .engine import DerivationEngine, MassMiner
from syntonic.core.constants import UniverseSeeds
from .geometry import GeometricInvariants
from .catalog import PARTICLE_CATALOG, get_particle
from .hierarchy import Q, E_STAR, PHI

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger(__name__)


# ==============================================================================
# TARGET LIST: Complete Physics Predictions (188+ observables)
# ==============================================================================
TARGETS = [
    # ─── LEPTONS ───
    {"name": "Electron", "mass": 0.5109989461, "unit": "MeV", "type": "lepton"},
    {"name": "Muon", "mass": 105.6583755, "unit": "MeV", "type": "lepton"},
    {"name": "Tau", "mass": 1776.86, "unit": "MeV", "type": "lepton"},
    {"name": "Neutrino (ν₁)", "mass": 2.02, "unit": "meV", "type": "neutrino"},
    {"name": "Neutrino (ν₂)", "mass": 8.61, "unit": "meV", "type": "neutrino"},
    {"name": "Neutrino (ν₃)", "mass": 50.1, "unit": "meV", "type": "neutrino"},
    
    # ─── QUARKS ───
    {"name": "Up Quark", "mass": 2.16, "unit": "MeV", "type": "quark"},
    {"name": "Down Quark", "mass": 4.67, "unit": "MeV", "type": "quark"},
    {"name": "Strange Quark", "mass": 93.4, "unit": "MeV", "type": "quark"},
    {"name": "Charm Quark", "mass": 1270.0, "unit": "MeV", "type": "quark"},
    {"name": "Bottom Quark", "mass": 4180.0, "unit": "MeV", "type": "quark"},
    {"name": "Top Quark", "mass": 172760.0, "unit": "MeV", "type": "quark"},
    
    # ─── GAUGE BOSONS ───
    {"name": "W Boson", "mass": 80379.0, "unit": "MeV", "type": "gauge"},
    {"name": "Z Boson", "mass": 91187.6, "unit": "MeV", "type": "gauge"},
    {"name": "Higgs Boson", "mass": 125100.0, "unit": "MeV", "type": "higgs"},
    
    # ─── DECAY WIDTHS ───
    {"name": "Γ_Z", "mass": 2.4952, "unit": "GeV", "type": "decay_width"},
    {"name": "Γ_W", "mass": 2.085, "unit": "GeV", "type": "decay_width"},
    
    # ─── LIGHT MESONS ───
    {"name": "Pion (Charged)", "mass": 139.57039, "unit": "MeV", "type": "meson"},
    {"name": "Pion (Neutral)", "mass": 134.9768, "unit": "MeV", "type": "meson"},
    {"name": "Kaon (Charged)", "mass": 493.677, "unit": "MeV", "type": "meson"},
    {"name": "Kaon (Neutral)", "mass": 497.611, "unit": "MeV", "type": "meson"},
    {"name": "Eta Meson", "mass": 547.862, "unit": "MeV", "type": "meson"},
    {"name": "Rho Meson", "mass": 775.26, "unit": "MeV", "type": "meson"},
    {"name": "Omega Meson", "mass": 782.66, "unit": "MeV", "type": "meson"},
    
    # ─── HEAVY MESONS ───
    {"name": "D Meson", "mass": 1864.84, "unit": "MeV", "type": "meson"},
    {"name": "B Meson", "mass": 5279.66, "unit": "MeV", "type": "meson"},
    {"name": "Bc Meson", "mass": 6274.9, "unit": "MeV", "type": "meson"},
    
    # ─── QUARKONIUM ───
    {"name": "J/psi", "mass": 3096.9, "unit": "MeV", "type": "quarkonium"},
    {"name": "Psi(2S)", "mass": 3686.1, "unit": "MeV", "type": "quarkonium"},
    {"name": "Upsilon(1S)", "mass": 9460.3, "unit": "MeV", "type": "quarkonium"},
    {"name": "Upsilon(2S)", "mass": 10023.3, "unit": "MeV", "type": "quarkonium"},
    {"name": "Upsilon(3S)", "mass": 10355.2, "unit": "MeV", "type": "quarkonium"},
    
    # ─── BARYONS ───
    {"name": "Proton", "mass": 938.27208816, "unit": "MeV", "type": "baryon"},
    {"name": "Neutron", "mass": 939.56542052, "unit": "MeV", "type": "baryon"},
    {"name": "Lambda", "mass": 1115.683, "unit": "MeV", "type": "baryon"},
    {"name": "Sigma+", "mass": 1189.37, "unit": "MeV", "type": "baryon"},
    {"name": "Delta", "mass": 1232.0, "unit": "MeV", "type": "baryon"},
    {"name": "Xi-", "mass": 1321.71, "unit": "MeV", "type": "baryon"},
    {"name": "Omega-", "mass": 1672.45, "unit": "MeV", "type": "baryon"},
    
    # ─── MIXING ANGLES & CKM/PMNS ───
    {"name": "|V_us|", "mass": 0.2243, "unit": "dimensionless", "type": "ckm"},
    {"name": "|V_cb|", "mass": 0.0415, "unit": "dimensionless", "type": "ckm"},
    {"name": "|V_ub|", "mass": 0.00361, "unit": "dimensionless", "type": "ckm"},
    {"name": "θ₁₂", "mass": 33.44, "unit": "degrees", "type": "pmns"},
    {"name": "θ₂₃", "mass": 49.2, "unit": "degrees", "type": "pmns"},
    {"name": "θ₁₃", "mass": 8.57, "unit": "degrees", "type": "pmns"},
    {"name": "δ_CP (PMNS)", "mass": 195.0, "unit": "degrees", "type": "pmns"},
    {"name": "J_CP", "mass": 3.08e-05, "unit": "dimensionless", "type": "pmns"},
    
    # ─── NUCLEAR BINDING ENERGIES ───
    {"name": "Deuteron (B_d)", "mass": 2.22457, "unit": "MeV", "type": "binding"},
    {"name": "Alpha (B_α)", "mass": 28.2961, "unit": "MeV", "type": "binding"},
    {"name": "⁵⁶Fe Binding Energy", "mass": 8.79, "unit": "MeV", "type": "binding"},
    {"name": "Triton Binding Energy", "mass": 8.482, "unit": "MeV", "type": "binding"},
    
    # ─── GLUEBALLS ───
    {"name": "Glueball(0++)", "mass": 1517.0, "unit": "MeV", "type": "glueball"},
    {"name": "Glueball(2++)", "mass": 2220.0, "unit": "MeV", "type": "glueball"},
    {"name": "Glueball(0-+)", "mass": 2500.0, "unit": "MeV", "type": "glueball"},
    
    # ─── RARE HADRONS ───
    {"name": "T_cc++", "mass": 3875.1, "unit": "MeV", "type": "exotic"},
    {"name": "X(3872)", "mass": 3871.65, "unit": "MeV", "type": "exotic"},
    {"name": "P_c(4457)", "mass": 4457.0, "unit": "MeV", "type": "exotic"},
    
    # ─── NUCLEAR SEMI-EMPIRICAL MASS FORMULA ───
    {"name": "SEMF a_s", "mass": 17.8, "unit": "MeV", "type": "semf"},
    {"name": "SEMF a_v", "mass": 15.75, "unit": "MeV", "type": "semf"},
    {"name": "SEMF a_a", "mass": 23.7, "unit": "MeV", "type": "semf"},
    {"name": "SEMF a_p", "mass": 12.0, "unit": "MeV", "type": "semf"},
    {"name": "SEMF a_c", "mass": 0.711, "unit": "MeV", "type": "semf"},
    
    # ─── ATOMIC & FINE STRUCTURE ───
    {"name": "Fine Structure Constant (α⁻¹)", "mass": 137.036, "unit": "dimensionless", "type": "constant"},
    {"name": "Rydberg Constant", "mass": 13.606, "unit": "eV", "type": "constant"},
    {"name": "He+ Ionization Energy", "mass": 54.418, "unit": "eV", "type": "atomic"},
    {"name": "Hydrogen Polarizability", "mass": 4.5, "unit": "a0³", "type": "atomic"},
    {"name": "H₂ Bond Length", "mass": 0.741, "unit": "Angstrom", "type": "molecular"},
    {"name": "H₂ Dissociation Energy", "mass": 4.478, "unit": "eV", "type": "molecular"},
    {"name": "Fine Structure 2P", "mass": 10.97, "unit": "GHz", "type": "atomic"},
    {"name": "Hyperfine 21cm", "mass": 1420.406, "unit": "MHz", "type": "atomic"},
    {"name": "Proton Radius", "mass": 0.8414, "unit": "fm", "type": "nuclear"},
    
    # ─── FUNDAMENTAL PHYSICS RATIOS ───
    {"name": "R_b (Z→bb/Z→all)", "mass": 0.21629, "unit": "ratio", "type": "precision"},
    {"name": "sin²θ_W", "mass": 0.2312, "unit": "dimensionless", "type": "ewk"},
    
    # ─── COSMOLOGICAL PARAMETERS ───
    {"name": "Hubble Constant (H₀)", "mass": 67.4, "unit": "km/s/Mpc", "type": "cosmology"},
    {"name": "Redshift Equality (z_eq)", "mass": 3400.0, "unit": "redshift", "type": "cosmology"},
    {"name": "Redshift Recombination (z_rec)", "mass": 1100.0, "unit": "redshift", "type": "cosmology"},
    {"name": "Dark Energy Density (ρ_Λ)", "mass": 2.25, "unit": "meV", "type": "cosmology"},
    {"name": "Baryon Asymmetry (η_B)", "mass": 6.1e-10, "unit": "dimensionless", "type": "cosmology"},
    {"name": "Scalar Spectral Index (n_s)", "mass": 0.9649, "unit": "dimensionless", "type": "cosmology"},
    {"name": "Tensor-to-Scalar Ratio (r)", "mass": 0.00328, "unit": "dimensionless", "type": "cosmology"},
    {"name": "Equation of State (w)", "mass": -1.03, "unit": "dimensionless", "type": "cosmology"},
    {"name": "Effective # Neutrino Species (N_eff)", "mass": 2.99, "unit": "dimensionless", "type": "cosmology"},
    {"name": "Dark Matter/Baryon Ratio", "mass": 5.36, "unit": "ratio", "type": "cosmology"},
    
    # ─── PRIMORDIAL ABUNDANCES ───
    {"name": "Helium-4 Abundance (Y_p)", "mass": 0.245, "unit": "dimensionless", "type": "bbn"},
    {"name": "Deuterium Abundance (D/H)", "mass": 2.53e-05, "unit": "dimensionless", "type": "bbn"},
    {"name": "Lithium-7 Abundance (⁷Li/H)", "mass": 1.6e-10, "unit": "dimensionless", "type": "bbn"},
    
    # ─── CMB PEAKS ───
    {"name": "CMB Peak ℓ₁", "mass": 220.0, "unit": "multipole", "type": "cmb"},
    {"name": "CMB Peak ℓ₂", "mass": 537.5, "unit": "multipole", "type": "cmb"},
    {"name": "CMB Peak ℓ₃", "mass": 810.8, "unit": "multipole", "type": "cmb"},
    {"name": "CMB Peak ℓ₄", "mass": 1120.9, "unit": "multipole", "type": "cmb"},
    {"name": "CMB Peak ℓ₅", "mass": 1444.2, "unit": "multipole", "type": "cmb"},
    {"name": "CMB Peak Ratio H₂/H₁", "mass": 0.458, "unit": "ratio", "type": "cmb"},
    {"name": "CMB Peak Ratio H₃/H₁", "mass": 0.37, "unit": "ratio", "type": "cmb"},
    
    # ─── ANOMALOUS MAGNETIC MOMENTS ───
    {"name": "Muon g-2 (a_μ)", "mass": 2.51e-09, "unit": "dimensionless", "type": "precision"},
    {"name": "Tau g-2 (a_τ)", "mass": 0.00118, "unit": "dimensionless", "type": "precision"},
    
    # ─── PARTICLE LIFETIMES & CONSTANTS ───
    {"name": "Neutron Lifetime (τ_n)", "mass": 879.4, "unit": "s", "type": "fundamental"},
    
    # ─── QUANTUM CHROMODYNAMICS ───
    {"name": "Sterile Neutrino Mixing", "mass": 1.14e-11, "unit": "dimensionless", "type": "bsm"},
    
    # ─── CONDENSED MATTER ───
    {"name": "BCS Gap Ratio (2Δ₀/kTc)", "mass": 3.52, "unit": "dimensionless", "type": "cmatter"},
    {"name": "YBCO Critical Temperature", "mass": 92.4, "unit": "Kelvin", "type": "cmatter"},
    {"name": "BSCCO Critical Temperature", "mass": 110.0, "unit": "Kelvin", "type": "cmatter"},
    {"name": "Graphene Fermi Velocity", "mass": 1000000.0, "unit": "m/s", "type": "cmatter"},
    
    # ─── BLACK HOLE THERMODYNAMICS ───
    {"name": "BH Entropy Correction", "mass": 1.0, "unit": "ratio", "type": "gravity"},
    {"name": "Hawking Temperature Correction", "mass": 1.0, "unit": "ratio", "type": "gravity"},
    
    # ─── GRAVITATIONAL WAVES ───
    {"name": "GW150914 Echo Delay", "mass": 0.59, "unit": "ms", "type": "gw"},
    {"name": "GW190521 Echo Delay", "mass": 1.35, "unit": "ms", "type": "gw"},
    {"name": "GW170817 Echo Delay", "mass": 0.038, "unit": "ms", "type": "gw"},
    
    # ─── MODIFIED GRAVITY ───
    {"name": "MOND Acceleration Scale (a₀)", "mass": 1.2e-10, "unit": "m/s²", "type": "bsm"},
    
    # ─── UNIFIED GRAND THEORIES ───
    {"name": "GUT Unification Scale (μ_GUT)", "mass": 1000000000000000.0, "unit": "GeV", "type": "gut"},
    {"name": "Reheating Temperature (T_reh)", "mass": 10000000000.0, "unit": "GeV", "type": "cosmology"},
]


class AutoMiner:
    """Automated geometric discovery engine."""
    
    def __init__(self):
        """Initialize the mining engine."""
        logger.info("Initializing SRT-Zero Auto-Miner...")
        self.seeds = UniverseSeeds()
        self.geo = GeometricInvariants()
        self.engine = DerivationEngine(self.seeds, self.geo)
        self.miner = MassMiner(self.engine)
        self.config = get_config()
        self.results: list[dict[str, Any]] = []
        
        logger.info(f"q = {float(Q):.10f}")
        logger.info(f"E* = {float(E_STAR):.10f}")
        logger.info(f"φ = {float(PHI):.10f}")
    
    def mine_single(self, target: dict) -> dict | None:
        """
        Mine a single target for geometric formulas.
        Handles different units and scales (MeV, GeV, meV, degrees, dimensionless, etc.)
        
        Returns the best matching formula or None.
        """
        name = target["name"]
        value = target["mass"]
        unit = target.get("unit", "MeV")
        
        # Normalize to base unit for mining (MeV for energies)
        normalized_value = self._normalize_value(value, unit)
        
        # Scale tolerance based on magnitude
        tolerance = self._scale_tolerance(normalized_value)
        
        logger.debug(f"Mining {name} ({value} {unit}, normalized: {normalized_value:.4e})...")
        
        # Try all mining methods
        all_matches = []
        
        # Standard E* Template
        matches_e = self.miner.mine_E_star(normalized_value, tolerance_percent=tolerance)
        if matches_e:
            all_matches.extend(matches_e)
        
        # Proton Resonance
        matches_p = self.miner.mine_from_proton(normalized_value, tolerance_percent=tolerance)
        if matches_p:
            all_matches.extend(matches_p)
        
        # Power templates (for extreme scales)
        matches_pow = self.miner.mine_E_star_power(normalized_value, tolerance_percent=tolerance * 2)
        if matches_pow:
            all_matches.extend(matches_pow)
        
        # Special templates (PHI, compound formulas)
        matches_special = self.miner.mine_special_template(normalized_value, tolerance_percent=tolerance * 2)
        if matches_special:
            all_matches.extend(matches_special)
        
        # Select best match
        if all_matches:
            best_match = min(all_matches, key=lambda x: x["error_percent"])
            source_type = best_match.get("formula", "Unknown")
            
            return {
                "target": name,
                "value": value,
                "unit": unit,
                "normalized_value": normalized_value,
                "source": source_type,
                "formula": best_match,
                "error_percent": best_match["error_percent"],
            }
        return None
    
    def _normalize_value(self, value: float, unit: str) -> float:
        """
        Normalize value to MeV base unit for mining.
        Handles various physics units.
        """
        unit_lower = unit.lower().strip()
        
        # Energy conversions to MeV
        if unit_lower == "gev":
            return value * 1000.0
        elif unit_lower == "mev":
            return value
        elif unit_lower == "kev":
            return value / 1000.0
        elif unit_lower == "mev":
            return value / 1000000.0
        
        # For non-energy units, scale to a more mineable range
        # Angles, ratios, and cosmological values need special handling
        if unit_lower in ["degrees", "degree", "deg"]:
            # Keep angles as-is (0-180 range)
            return value
        elif unit_lower in ["dimensionless", "ratio", "ev", "kelvin", "k", "s", 
                            "m/s", "m/s/mpc", "km/s/mpc", "redshift", "gev", 
                            "mhz", "ghz", "angstrom", "fm", "a0", "a0³", "ms", "multipole"]:
            # For these, return as-is (they're in reasonable numerical ranges)
            return value
        else:
            # Default: return as-is
            return value
    
    def _scale_tolerance(self, normalized_value: float) -> float:
        """
        Scale mining tolerance based on value magnitude.
        Larger values get proportionally larger tolerances.
        """
        base_tolerance = self.config.mining_tolerance * 100
        
        if abs(normalized_value) < 0.01:
            # Very small values (< 0.01): higher relative tolerance
            return base_tolerance * 5
        elif abs(normalized_value) < 1:
            # Small values (0.01 to 1): higher tolerance
            return base_tolerance * 3
        elif abs(normalized_value) < 100:
            # Medium values (1 to 100): standard tolerance
            return base_tolerance
        elif abs(normalized_value) < 10000:
            # Large values (100 to 10k): standard tolerance
            return base_tolerance
        else:
            # Very large values (>10k): relaxed tolerance
            return base_tolerance * 2
    
    def mine_all(self, targets: list[dict] | None = None) -> list[dict]:
        """
        Mine all targets and return results.
        
        Args:
            targets: List of target dictionaries. Defaults to TARGETS.
        
        Returns:
            List of successful mining results.
        """
        if targets is None:
            targets = TARGETS
        
        logger.info(f"Starting auto-mine sequence ({len(targets)} targets)")
        self.results = []
        
        iterator = tqdm(targets, desc="Mining", unit="particle") 
        
        for target in iterator:
            result = self.mine_single(target)
            if result:
                self.results.append(result)
                logger.info(f"Match found for {target['name']}: {result['formula']['description']} "
                            f"({result['error_percent']:.4f}% error)")
        
        logger.info(f"Mining complete: {len(self.results)}/{len(targets)} matches found")
        return self.results
    
    def save_results(self, path: Path | str | None = None):
        """Save results to JSON file."""
        if path is None:
            path = self.config.results_file
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        output = {
            "timestamp": datetime.now().isoformat(),
            "config": self.config.to_dict(),
            "constants": {
                "q": float(Q),
                "E_star": float(E_STAR),
                "phi": float(PHI),
            },
            "results": self.results,
            "summary": {
                "total_targets": len(TARGETS),
                "matches_found": len(self.results),
                "success_rate": len(self.results) / len(TARGETS) * 100 if TARGETS else 0,
            }
        }
        
        with open(path, "w") as f:
            json.dump(output, f, indent=2, default=str)
        
        logger.info(f"Results saved to {path}")
    
    def print_summary(self):
        """Print a summary of mining results."""
        print("\n" + "=" * 70)
        print("MINING COMPLETE - SUMMARY")
        print("=" * 70)
        print(f"{'Target':<20} {'Source':<18} {'Error':>10}")
        print("-" * 70)
        
        for res in sorted(self.results, key=lambda x: x["error_percent"]):
            print(f"{res['target']:<20} {res['source']:<18} {res['error_percent']:>9.4f}%")
        
        print("-" * 70)
        print(f"Total: {len(self.results)} matches found")


def auto_mine():
    """Run the auto-mining process."""
    miner = AutoMiner()
    miner.mine_all()
    miner.print_summary()
    
    if miner.config.save_results:
        miner.save_results()


if __name__ == "__main__":
    try:
        auto_mine()
    except KeyboardInterrupt:
        print("\nMining interrupted by user.")