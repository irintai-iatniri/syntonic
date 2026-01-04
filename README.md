# Syntonic

**The Syntony Recursion Theory (SRT) Computational Library.**

Syntonic is a hybrid Python/Rust library designed to model recursive information structures using the DHSR (Differentiation-Harmonization-Syntony-Recursion) framework. It provides high-performance tensor operations, native linear algebra, and specific abstractions for modeling consciousness and physical theory.

## Features (Phase 1)

- **State**: The fundamental object representing information configurations.
- **DType System**: Explicit support for `float64`, `complex128` (CRT/SRT standard).
- **Native Backend**: Tensor operations backed by Rust `ndarray` and system BLAS/LAPACK.
- **Linear Algebra**: Fast spectral decompositions (`eig`, `svd`, `eigh`) and solvers.
- **DHSR Pipeline**: Tools for simulating recursive state evolution.

## Installation

### Prerequisites
- Python 3.9+
- Rust 1.70+ (with `cargo`)
- System libraries: `libopenblas-dev`, `libssl-dev`

### From Source

```bash
git clone https://github.com/irintai-iatniri/syntonic.git
cd syntonic
maturin develop  # For development install
# OR
pip install .    # For release install
```

## Quickstart

```python
import syntonic as syn
from syntonic import linalg

# Create a state
psi = syn.state.random((10, 10))

# Apply recursion cycle (Differentiate -> Harmonize)
next_psi = psi.recurse()

# Analyze spectral properties
w, v = linalg.eig(next_psi)

print(f"Syntony: {next_psi.syntony:.4f}")
print(f"Dominant Eigenvalue: {w.numpy()[0]:.4f}")
```

## Testing

Run the test suite:

```bash
pytest tests/
```

## Performance

Phase 1 benchmarks indicate parity with NumPy for heavy linear algebra operations (e.g., Eigenvalue decomposition), with some overhead for lightweight arithmetic in debug builds.

---
*Syntonic is part of the SRT Research Project.*
