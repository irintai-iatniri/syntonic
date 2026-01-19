"""
Syntonic - Tensor library for Cosmological and Syntony Recursion Theory

Syntonic provides tensor operations and state evolution primitives for
the DHSR (Differentiation-Harmonization-Syntony-Recursion) framework
used in Cosmological Recursion Theory (CRT) and Syntony Recursion Theory (SRT).

Basic Usage:
    >>> import syntonic as syn
    >>> psi = syn.state([1, 2, 3, 4])
    >>> psi.shape
    (4,)
    >>> psi.syntony
    0.5

    >>> # DHSR cycle
    >>> evolved = psi.differentiate().harmonize()
"""

# Linear algebra submodule
# Hypercomplex numbers submodule
# Exact arithmetic submodule
# CRT Core submodule
# SRT submodule (spectral and geometry)
# Applications submodule
from syntonic import applications, crt, exact, hypercomplex, linalg, srt
from syntonic._version import __version__, __version_info__

# Prime Selection (SRT/CRT Physics) - temporarily disabled
# from syntonic.srt.prime_selection import (
#     FERMAT_PRIMES,
#     MERSENNE_EXPONENTS,
#     LUCAS_PRIMES,
#     fermat_number,
#     mersenne_number,
#     lucas_number,
# )
# Consciousness/Gnosis
from syntonic.consciousness.gnosis import (
    COLLAPSE_THRESHOLD,
    gnosis_score,
    is_conscious,
)
from syntonic.core import (
    # Devices
    Device,
    # Data types
    DType,
    # ResonantTensor,  # Use Python wrapper from syntonic.nn instead
    RESConfig,
    ResonantEvolver,
    RESResult,
    # State class
    State,
    _debug_stress_pool_take,
    complex64,
    complex128,
    cpu,
    cuda,
    cuda_device_count,
    cuda_is_available,
    device,
    float32,
    float64,
    get_dtype,
    int32,
    int64,
    promote_dtypes,
    srt_memory_resonance,
    srt_pool_stats,
    srt_reserve_memory,
    srt_transfer_stats,
    srt_wait_for_resonance,
    state,
    winding,
)

# Extended Hierarchy
from syntonic.crt.extended_hierarchy import (
    apply_collapse_threshold,
    apply_coxeter_kissing,
    apply_e7_correction,
)
from syntonic.exact import (
    E_STAR_NUMERIC,
    PHI,
    PHI_INVERSE,
    PHI_NUMERIC,
    PHI_SQUARED,
    Q_DEFICIT_NUMERIC,
    STRUCTURE_DIMENSIONS,
    GoldenExact,
    Rational,
    correction_factor,
    fibonacci,
    golden_number,
    lucas,
)

# Exceptions
from syntonic.exceptions import (
    DeviceError,
    DTypeError,
    LinAlgError,
    ShapeError,
    SyntonicError,
)
from syntonic.hypercomplex import (
    Octonion,
    Quaternion,
    octonion,
    quaternion,
)

# Core classes and functions
from syntonic.nn.resonant_tensor import ResonantTensor

# Neural Networks submodule (optional, may require PyTorch)
try:
    from syntonic import nn
except (ImportError, NameError):
    # This happens if PyTorch is missing or if there are broken dependencies
    # in the legacy Torch code. We ignore it for pure-resonant mode.
    nn = None

__all__ = [
    # Version
    "__version__",
    "__version_info__",
    # State
    "State",
    "state",
    "ResonantTensor",
    "RESConfig",
    "RESResult",
    "ResonantEvolver",
    # DTypes
    "DType",
    "float32",
    "float64",
    "complex64",
    "complex128",
    "int32",
    "int64",
    "winding",
    "get_dtype",
    "promote_dtypes",
    # Devices
    "Device",
    "cpu",
    "cuda",
    "cuda_is_available",
    "cuda_device_count",
    "srt_transfer_stats",
    "srt_reserve_memory",
    "srt_wait_for_resonance",
    "srt_pool_stats",
    "srt_memory_resonance",
    "device",
    # Exceptions
    "SyntonicError",
    "DeviceError",
    "DTypeError",
    "ShapeError",
    "LinAlgError",
    # Submodules
    "linalg",
    "hypercomplex",
    "crt",
    "srt",
    "applications",
    "nn",
    # Hypercomplex types
    "Quaternion",
    "Octonion",
    "quaternion",
    "octonion",
    # Exact arithmetic
    "exact",
    "GoldenExact",
    "Rational",
    "PHI",
    "PHI_SQUARED",
    "PHI_INVERSE",
    "PHI_NUMERIC",
    "E_STAR_NUMERIC",
    "Q_DEFICIT_NUMERIC",
    "STRUCTURE_DIMENSIONS",
    "fibonacci",
    "lucas",
    "correction_factor",
    "golden_number",
    # Prime Selection (SRT/CRT Physics)
    "FERMAT_PRIMES",
    "MERSENNE_EXPONENTS",
    "LUCAS_PRIMES",
    "fermat_number",
    "mersenne_number",
    "lucas_number",
    # Consciousness
    "COLLAPSE_THRESHOLD",
    "is_conscious",
    "gnosis_score",
    # Extended Hierarchy
    "apply_e7_correction",
    "apply_collapse_threshold",
    "apply_coxeter_kissing",
]
