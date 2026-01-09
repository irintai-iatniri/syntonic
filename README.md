# Syntonic

**The Syntony Recursion Theory (SRT) Computational Library.**

Syntonic is a comprehensive hybrid Python/Rust library implementing the DHSR (Differentiation-Harmonization-Syntony-Recursion) framework for Cosmological Recursion Theory (CRT) and Syntony Recursion Theory (SRT). It provides high-performance tensor operations, exact arithmetic, hypercomplex numbers, spectral analysis, and physics applications for modeling consciousness, quantum field theory, and recursive information structures.

**Version:** 0.1.0 (Alpha)

## Features

- **Core Tensor Operations**: High-performance `State` class with native Rust backend and CUDA support
- **Extended Numerics**: Quaternions, octonions, golden ratio arithmetic, and exact rational computation
- **Spectral Analysis**: E₈ lattice operations, golden recursion, and winding number topology
- **Physics Applications**: Standard Model parameter derivation, neutrino mixing, and quantum field theory
- **Neural Networks**: Syntonic neural layers with DHSR activation functions and PyTorch-free implementations
- **CRT/SRT Framework**: Complete implementation of recursive state evolution and syntony metrics
- **Dual Precision**: Support for `float64`, `complex128`, and custom winding number types
- **Resonant Operations**: New activation functions (sigmoid, tanh) and layer normalization on the golden lattice

## Installation

### Prerequisites
- Python 3.10+
- Rust 1.70+ (with `cargo`)
- System libraries: `libopenblas-dev`, `libssl-dev`
- Optional: CUDA 12.0+ for GPU acceleration

### From Source

```bash
git clone https://github.com/irintai-iatniri/syntonic.git
cd syntonic

# Development install (recommended for development)
maturin develop

# OR release install
pip install .

# OR install with optional dependencies
pip install ".[all]"    # All optional dependencies
pip install ".[dev]"    # Development tools (pytest, ruff, mypy, black)
```

### Optional Dependencies

```bash
# Core functionality (no additional deps needed)
pip install .

# NumPy interop
pip install ".[numpy]"

# SciPy for advanced linear algebra (expm, logm)
pip install ".[scipy]"

# PyTorch for neural network components
pip install ".[torch]"

# Development and testing
pip install ".[dev]"

# Documentation
pip install ".[docs]"
```

## Licensing

Syntonic uses a dual-license model:

- **Research License**: Free for academic and non-commercial research (see `LICENSE-RESEARCH.md`)
- **Commercial License**: Required for commercial use (see `LICENSE-COMMERCIAL.md`)

## Quickstart

```python
import syntonic as syn
import numpy as np

# Create quantum-like states
psi = syn.state.random((10, 10), dtype=syn.complex128)

# Apply DHSR evolution cycle
evolved = psi.differentiate().harmonize().recurse()

# Analyze spectral properties
w, v = syn.linalg.eig(evolved)
syntony_measure = evolved.syntony

print(f"Syntony: {syntony_measure:.4f}")
print(f"Dominant eigenvalue: {w[0]:.4f}")

# Work with exact golden arithmetic
phi = syn.PHI
golden_state = syn.golden_number(1, 1)  # 1 + φ
result = golden_state * phi  # Exact computation

# Physics applications
from syntonic.physics import standard_model
sm_params = standard_model.parameters_from_q()
print(f"Fine structure constant: {sm_params.alpha}")

# Neural networks with resonant activations
from syntonic.nn import WindingNet
model = WindingNet(max_winding=3, base_dim=32, num_blocks=2, output_dim=2)
```

## Module Overview

### Core Modules
- **`syn.core`**: State class, dtypes, device management
- **`syn.linalg`**: Linear algebra operations (SVD, eigendecomposition, norms)
- **`syn.hypercomplex`**: Quaternion and octonion algebras
- **`syn.exact`**: Golden ratio and rational arithmetic

### Theory Modules
- **`syn.crt`**: Cosmological Recursion Theory operators
- **`syn.srt`**: Syntony Recursion Theory (spectral, geometry, lattice)

### Applications
- **`syn.physics`**: Standard Model, particle physics, quantum field theory
- **`syn.nn`**: Neural networks with syntonic activations (PyTorch-free implementations available)
- **`syn.applications`**: Cross-domain applications (chemistry, biology, consciousness)

## Recent Developments

**Version 0.1.0** includes new resonant tensor operations for PyTorch-free neural networks:
- Sigmoid and tanh activations on the golden lattice
- Element-wise operations (multiplication, addition)
- Layer normalization with golden target variance
- Tensor concatenation operations
- Winding network implementations for topological neural computing

## Testing

Run the comprehensive test suite:

```bash
# Install development dependencies
pip install ".[dev]"

# Run all tests
pytest tests/

# Run with coverage
pytest -v --cov=syntonic

# Run specific test modules
pytest tests/test_core/     # Core tensor operations
pytest tests/test_linalg/   # Linear algebra
pytest tests/test_physics/  # Physics derivations
```

Current test coverage includes core operations, hypercomplex arithmetic, spectral analysis, physics validations, and neural network components.

## Performance

Syntonic provides competitive performance for tensor operations with a focus on linear algebra and spectral computations. Recent benchmarks (January 2026) show:

- **Linear Algebra**: Competitive with NumPy for eigendecomposition and SVD, especially for large matrices (500×500+)
- **Matrix Operations**: Currently slower than PyTorch/NumPy for basic operations (add, multiply, transpose)
- **GPU Support**: CUDA acceleration available for GPU workloads
- **Memory Efficiency**: Optimized Rust backend with BLAS/LAPACK integration

For detailed performance analysis, see `benchmarks/performance_analysis.md` and `benchmark_results.json`.

## Documentation

- **API Reference**: Comprehensive API documentation for all modules
- **Theory Guide**: Mathematical foundations of CRT/SRT
- **Tutorials**: Step-by-step guides for different use cases
- **Examples**: Jupyter notebooks demonstrating applications

## Contributing

See `CONTRIBUTING.md` for development guidelines and contribution process.

## Citation

If you use Syntonic in research, please cite:

```
@software{syntonic2026,
  title={Syntonic: Tensor Library for Cosmological and Syntony Recursion Theory},
  author={Orth, Andrew},
  year={2026},
  url={https://github.com/irintai-iatniri/syntonic}
}
```

---
*Syntonic is part of the SRT Research Project.*
