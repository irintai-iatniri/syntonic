# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Build Commands

```bash
# Development install (requires Rust toolchain)
maturin develop

# Release install
pip install .

# Install with optional dependencies
pip install ".[dev]"      # Development (pytest, ruff, mypy, black)
pip install ".[numpy]"    # NumPy interop
pip install ".[scipy]"    # SciPy (for expm, logm)
pip install ".[all]"      # All optional deps

# Run tests
pytest tests/
pytest tests/test_core/test_state.py           # Single test file
pytest tests/test_core/test_state.py::test_foo # Single test function
pytest -v --cov=syntonic                       # With coverage

# Linting and formatting
ruff check python/
black python/
mypy python/syntonic/
```

## Architecture

Syntonic is a hybrid Python/Rust library implementing Syntony Recursion Theory (SRT), a mathematical framework for deriving Standard Model physics from geometric structures.

### Language Split

- **Rust (`rust/src/`)**: Performance-critical tensor operations, hypercomplex types (Quaternion, Octonion), and exact arithmetic. Compiled via maturin/PyO3 to `syntonic._core`.
- **Python (`python/syntonic/`)**: High-level API, CRT/SRT operators, physics derivations.

### Core Modules

| Module | Purpose |
|--------|---------|
| `syntonic.core` | `State` (tensor wrapper), `DType` system (float64, complex128, winding), `Device` (cpu/cuda) |
| `syntonic.linalg` | Linear algebra (eig, svd, qr, cholesky, solve). Core ops are numpy-free; expm/logm require scipy |
| `syntonic.hypercomplex` | Quaternions and Octonions (implemented in Rust) |
| `syntonic.exact` | Exact arithmetic: `Rational`, `GoldenExact` (a + b*phi), Fibonacci/Lucas sequences |
| `syntonic.crt` | DHSR framework: Differentiation/Harmonization/Syntony/Recursion operators and metrics |
| `syntonic.srt` | SRT geometry: T4 torus, E8/D4 lattices, golden cone (36 roots), theta series, heat kernels |
| `syntonic.physics` | Standard Model derivation: fermion/boson masses, CKM/PMNS matrices, neutrino masses |

### Key Abstractions

- **State**: The fundamental object representing information configurations. Wraps Rust tensor storage.
- **DHSR cycle**: `psi.differentiate().harmonize()` or `RecursionOperator.apply(psi)` for state evolution.
- **SRTSystem**: `create_srt_system()` initializes E8 lattice (240 roots), golden cone (36 roots), spectral operators.
- **StandardModel**: `syn.physics.StandardModel()` derives all 25+ SM parameters from geometry (zero free parameters).

### Mathematical Constants

- `PHI` / `PHI_NUMERIC`: Golden ratio phi = (1+sqrt(5))/2
- `E_STAR_NUMERIC`: Spectral constant e^pi - pi ~ 19.999
- `Q_DEFICIT_NUMERIC`: Universal syntony deficit q ~ 0.027395

### Rust Backend Structure

- `rust/src/tensor/storage.rs`: Core tensor storage with BLAS/LAPACK integration
- `rust/src/tensor/linalg.rs`: Eigenvalue decomposition, SVD, etc.
- `rust/src/exact/golden.rs`: Exact golden ratio arithmetic
- `rust/src/hypercomplex/`: Quaternion and Octonion implementations

## System Requirements

- Python 3.10+
- Rust 1.70+ with cargo
- System libraries: `libopenblas-dev`, `libssl-dev`
- Optional: CUDA toolkit for GPU support (feature flag `cuda`)
