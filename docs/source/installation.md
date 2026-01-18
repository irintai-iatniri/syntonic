# Installation

## Requirements

- Python 3.10+
- Rust 1.70+ (for building from source)
- CUDA 12.0+ (optional, for GPU acceleration)

## Quick Install

```bash
pip install syntonic
```

## Development Install

Clone the repository and install in editable mode:

```bash
git clone https://github.com/irintai-iatniri/syntonic.git
cd syntonic
pip install maturin
maturin develop --release
```

## Optional Dependencies

Install with extras for full functionality:

```bash
# For development (testing, linting)
pip install syntonic[dev]

# For documentation building
pip install syntonic[docs]
```

## Verifying Installation

```python
import syntonic
print(f"Syntonic version: {syntonic.__version__}")
print(f"CUDA available: {syntonic.cuda_available()}")

# Quick test
from syntonic.core import PHI
print(f"Golden ratio φ: {PHI}")  # Should print 1.6180339887498948
```

## Building CUDA Kernels

If you have CUDA installed, compile the PTX kernels:

```bash
./compile_kernels.sh --all
```

This compiles for GPU architectures: sm_75, sm_80, sm_86, sm_90.
