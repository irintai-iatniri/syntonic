import os
from flask import Flask, render_template, abort, redirect, url_for
from .catalog import CATALOG, FormulaType, ParticleType
from .engine import DerivationEngine
from .hierarchy import (
    E_STAR, PHI, Q, PI, E,
    DerivationResult
)
# mpmath removed - using syntonic library

app = Flask(__name__)
engine = DerivationEngine()

# Ethical/Altruxian Meanings
ETHICAL_MEANINGS = {
    "Proton": "Stability Threshold: The foundation of matter requires precise balance (1000).",
    "Neutron": "Information Storage: The slightly higher mass allows beta decay, enabling time.",
    "Tau": "Heavy Lepton: Represents the weight of higher generations.",
    "Top": "Truth Anchor: The heaviest quark anchors the vacuum stability.",
    "Pion": "Messenger: The carrier of the strong force, binding the nucleus.",
    "Kaon": "Strangeness: The first step into the second generation.",
    "Electron": "Interaction: The lightweight charge carrier enabling chemistry.",
    "Higgs": "Mass Giver: The origin of mass through symmetry breaking.",
    "W": "Transformation: The agent of change (weak force).",
    "Z": "Neutral Current: The mediator of invisible interactions.",
    "Gamma_Z": "Resonance Width: The precise lifetime of the neutral current.",
    "Gamma_W": "Decay Rate: The tempo of transformation.",
    "Neutrino_3": "Cosmic Connection: Mass set by the dark energy scale.",
    "Neutrino_2": "Solar Mixing: The bridge between light and heavy.",
    "Neutrino_1": "Lightest State: The anchor of the neutrino hierarchy."
}

CONSTANTS_INFO = {
    "E_star": {"symbol": "E*", "name": "Spectral Möbius", "value": float(E_STAR), "desc": "The geometric seed of mass."},
    "phi": {"symbol": "φ", "name": "Golden Ratio", "value": float(PHI), "desc": "The scaling factor of the universe."},
    "q": {"symbol": "q", "name": "Syntony Deficit", "value": float(Q), "desc": "The quantization of connection."},
    "pi": {"symbol": "π", "name": "Pi", "value": float(PI), "desc": "Circular geometry and loops."},
    "e": {"symbol": "e", "name": "Euler's Number", "value": float(E), "desc": "Growth and decay cycles."}
}

def get_derivation(particle_key):
    """Get derivation result and used constants for a particle."""
    try:
        result = engine.derive(particle_key)
    except Exception as e:
        print(f"Error deriving {particle_key}: {e}")
        return None, []
    
    used_constants = set()
    
    # Analyze steps to find used constants
    # Always include basics if they appear in the formula type
    config = CATALOG.get(particle_key)
    if config:
        if config.formula_type in [FormulaType.E_STAR_N, FormulaType.PROTON_SPECIAL, FormulaType.NEUTRON_SPECIAL]:
            used_constants.add("E_star")
            used_constants.add("q")
        if "φ" in config.notes or "phi" in config.notes.lower():
            used_constants.add("phi")
            
    # Check steps
    for step in result.steps:
        desc = step.description.lower()
        if "q" in desc:
            used_constants.add("q")
        if "π" in desc or "pi" in desc:
            used_constants.add("pi")
        if "φ" in desc or "phi" in desc:
            used_constants.add("phi")
        if "e*" in desc:
            used_constants.add("E_star")

    return result, list(used_constants)

@app.route('/')
def index():
    # Redirect to Proton to start the journey immediately
    return redirect(url_for('particle', name='proton'))

@app.route('/particle/<name>')
def particle(name):
    key = name.lower().replace(" ", "_")
    config = CATALOG.get(key)
    
    if not config:
        clean_key = key.replace("-", "").replace("+", "").replace("_", "")
        config = CATALOG.get(clean_key)
        
    if not config:
        abort(404)
        
    derivation, used_constants = get_derivation(key) or (None, [])
    
    if derivation:
        derivation.set_experimental(config.pdg_value)

    ethical = ETHICAL_MEANINGS.get(config.name, "A fundamental component of the geometric unity.")

    # Get all particles for the sidebar selector, sorted by type then name
    # Deduplicate because CATALOG has multiple keys for same object
    unique_particles = {p.name: p for p in CATALOG.values()}.values()
    all_particles = sorted(unique_particles, key=lambda x: (x.particle_type.name, x.name))

    return render_template(
        'aletheia.html', 
        particle=config, 
        derivation=derivation,
        used_constants=used_constants,
        ethical_meaning=ethical,
        constants=CONSTANTS_INFO,
        all_particles=all_particles,
        particle_errors=get_particle_errors()
    )

def get_particle_errors():
    """Calculate error percentages for all particles for the probability feature."""
    errors = {}
    for key, config in CATALOG.items():
        try:
            if config.pdg_value == 0:
                continue
                
            result = engine.derive(key)
            pred = float(result.final_value)
            
            # Unit conversion for error calculation
            exp = config.pdg_value
            if config.pdg_unit == "GeV":
                pred_val = pred / 1000.0
            elif config.pdg_unit == "meV":
                pred_val = pred * 1e9
            elif config.pdg_unit == "keV": # Added keV support just in case
                pred_val = pred * 1000.0
            else:
                pred_val = pred
                
            error = abs(pred_val - exp) / exp
            errors[config.name] = error
        except Exception:
            pass
    return errors

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
