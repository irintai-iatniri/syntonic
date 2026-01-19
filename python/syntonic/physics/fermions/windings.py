"""
Fermion Winding Assignments - T⁴ winding configurations for all fermions.

Each fermion is mapped to a winding state |n₇, n₈, n₉, n₁₀> on the
internal 4-torus. These quantum numbers determine mass generation,
charge quantization, and mixing properties.

Winding Matrices (from phase5-spec §5.2):

First Generation (k=1):
    Up:      (1,1,0,0), |n|² = 2
    Down:    (1,0,0,0), |n|² = 1
    Electron: (0,0,0,0), vacuum

Second Generation (k=2):
    Charm:   (1,1,1,0), |n|² = 3
    Strange: (1,1,0,0), |n|² = 2
    Muon:    (0,1,0,0), |n|² = 1

Third Generation (k=3):
    Top:     (2,1,1,0), |n|² = 6
    Bottom:  (1,1,1,0), |n|² = 3
    Tau:     (1,0,0,0), |n|² = 1
"""

from syntonic.srt.geometry.winding import winding_state

# =============================================================================
# Lepton Windings
# =============================================================================

# First generation: electron (vacuum state for leptons)
ELECTRON_WINDING = winding_state(0, 0, 0, 0)

# Second generation: muon
MUON_WINDING = winding_state(0, 1, 0, 0)

# Third generation: tau
TAU_WINDING = winding_state(1, 0, 0, 0)

# =============================================================================
# Up-Type Quark Windings
# =============================================================================

# First generation: up quark
UP_WINDING = winding_state(1, 1, 0, 0)

# Second generation: charm quark
CHARM_WINDING = winding_state(1, 1, 1, 0)

# Third generation: top quark
TOP_WINDING = winding_state(2, 1, 1, 0)

# =============================================================================
# Down-Type Quark Windings
# =============================================================================

# First generation: down quark
DOWN_WINDING = winding_state(1, 0, 0, 0)

# Second generation: strange quark
STRANGE_WINDING = winding_state(1, 1, 0, 0)

# Third generation: bottom quark
BOTTOM_WINDING = winding_state(1, 1, 1, 0)

# =============================================================================
# Grouped Collections
# =============================================================================

LEPTON_WINDINGS = {
    "electron": ELECTRON_WINDING,
    "muon": MUON_WINDING,
    "tau": TAU_WINDING,
}

UP_TYPE_WINDINGS = {
    "up": UP_WINDING,
    "charm": CHARM_WINDING,
    "top": TOP_WINDING,
}

DOWN_TYPE_WINDINGS = {
    "down": DOWN_WINDING,
    "strange": STRANGE_WINDING,
    "bottom": BOTTOM_WINDING,
}

ALL_FERMION_WINDINGS = {
    **LEPTON_WINDINGS,
    **UP_TYPE_WINDINGS,
    **DOWN_TYPE_WINDINGS,
}

# =============================================================================
# Generation Properties
# =============================================================================


def generation(fermion: str) -> int:
    """
    Return the generation (1, 2, or 3) for a fermion.

    Args:
        fermion: Fermion name (e.g., 'electron', 'charm', 'bottom')

    Returns:
        Generation number
    """
    gen1 = {"electron", "up", "down"}
    gen2 = {"muon", "charm", "strange"}
    gen3 = {"tau", "top", "bottom"}

    if fermion in gen1:
        return 1
    elif fermion in gen2:
        return 2
    elif fermion in gen3:
        return 3
    else:
        raise ValueError(f"Unknown fermion: {fermion}")


def winding_norm_squared(fermion: str) -> int:
    """
    Return |n|² for a fermion's winding state.

    Args:
        fermion: Fermion name

    Returns:
        Squared norm of winding state
    """
    winding = ALL_FERMION_WINDINGS.get(fermion)
    if winding is None:
        raise ValueError(f"Unknown fermion: {fermion}")
    return winding.norm_squared


__all__ = [
    # Individual windings
    "ELECTRON_WINDING",
    "MUON_WINDING",
    "TAU_WINDING",
    "UP_WINDING",
    "DOWN_WINDING",
    "STRANGE_WINDING",
    "CHARM_WINDING",
    "BOTTOM_WINDING",
    "TOP_WINDING",
    # Collections
    "LEPTON_WINDINGS",
    "UP_TYPE_WINDINGS",
    "DOWN_TYPE_WINDINGS",
    "ALL_FERMION_WINDINGS",
    # Functions
    "generation",
    "winding_norm_squared",
]
