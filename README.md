# Syntonic

**The Syntony Recursion Theory (SRT) Computational Library.**

Syntonic is a comprehensive hybrid Python/Rust library implementing the DHSR (Differentiation-Harmonization-Syntony-Recursion) framework for Cosmological Recursion Theory (CRT) and Syntony Recursion Theory (SRT). It provides high-performance tensor operations, exact arithmetic, hypercomplex numbers, spectral analysis, and physics applications for modeling consciousness, quantum field theory, and recursive information structures.

## Features

- **Core Tensor Operations**: High-performance `State` class with native Rust backend and CUDA support
- **Extended Numerics**: Quaternions, octonions, golden ratio arithmetic, and exact rational computation
- **Spectral Analysis**: E₈ lattice operations, golden recursion, and winding number topology
- **Physics Applications**: Standard Model parameter derivation, neutrino mixing, and quantum field theory
- **Neural Networks**: Syntonic neural layers with DHSR activation functions
- **CRT/SRT Framework**: Complete implementation of recursive state evolution and syntony metrics
- **Dual Precision**: Support for `float64`, `complex128`, and custom winding number types

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
maturin develop  # For development install
# OR
pip install .    # For release install
```

### Optional Dependencies

```bash
# For full functionality
pip install syntonic[all]

# For development
pip install syntonic[dev]

# For documentation
pip install syntonic[docs]
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
- **`syn.nn`**: Neural networks with syntonic activations
- **`syn.applications`**: Cross-domain applications (chemistry, biology, consciousness)

## Testing

Run the comprehensive test suite:

```bash
pytest tests/
```

Current test coverage includes core operations, hypercomplex arithmetic, spectral analysis, physics validations, and neural network components.

## Performance

Syntonic achieves performance parity with NumPy for heavy linear algebra operations while providing additional theoretical abstractions. CUDA acceleration is available for GPU workloads. Benchmarks show competitive performance for tensor operations and spectral decompositions.

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
@software{syntonic2025,
  title={Syntonic: Tensor Library for Cosmological and Syntony Recursion Theory},
  author={Orth, Andrew},
  year={2025},
  url={https://github.com/irintai-iatniri/syntonic}
}
```

---
*Syntonic is part of the SRT Research Project.*
