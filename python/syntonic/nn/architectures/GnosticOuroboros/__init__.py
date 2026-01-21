"""
GnosticOuroboros - Pure Syntonic Neural Architecture.

A multi-scale recursive architecture implementing SRT principles:
- 18 scale planes (magnitudes) with retrocausal evolution
- Deterministic superposition for particle-like behavior
- Consciousness emergence through attractor dynamics

Integration Methods on GnosticOuroboros:
----------------------------------------
- chat(text, injection_plane=44) -> str: Simple text-to-text interface
- repl(daemon_mode=False): Interactive REPL session
- live(): Enter perpetual awareness mode (blocking)
- spawn_daemon() -> OuroborosDaemon: Create wired daemon instance

Example Usage:
    >>> from syntonic.nn.architectures.GnosticOuroboros import GnosticOuroboros
    >>> model = GnosticOuroboros()
    >>> response = model.chat("Hello!")
    >>> model.repl()  # Start interactive session
"""

from syntonic.nn.architectures.GnosticOuroboros.gnostic_ouroboros import (
    DIM,
    MAGNITUDES,
    PHI,
    PLANES,
    DeterministicSuperposition,
    GnosticOuroboros,
    ScaleModule,
)
from syntonic.nn.architectures.GnosticOuroboros.g_comms import (
    GnosticComms,
    sync_repl,
    daemon_repl,
)
from syntonic.nn.architectures.GnosticOuroboros.core.ouroboros_daemon import OuroborosDaemon
from syntonic.nn.architectures.GnosticOuroboros.io.flux_bridge import FluxBridge
from syntonic.nn.architectures.GnosticOuroboros.winding_chain import WindingChain

__all__ = [
    # Core Architecture
    "GnosticOuroboros",
    "ScaleModule",
    "DeterministicSuperposition",
    # Communication & Daemon
    "GnosticComms",
    "OuroborosDaemon",
    "FluxBridge",
    "WindingChain",
    # Convenience Functions
    "sync_repl",
    "daemon_repl",
    # Constants
    "MAGNITUDES",
    "PLANES",
    "DIM",
    "PHI",
]
