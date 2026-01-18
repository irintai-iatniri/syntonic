# AGENTS.md - Syntonic Library Development Guide

This file provides comprehensive guidelines for agentic coding assistants working on the Syntonic library. The Syntonic library implements Syntony Recursion Theory (SRT) - a mathematical framework for deriving Standard Model physics from geometric structures.

## Build Commands

### Development Setup
```bash
# Install Rust toolchain (if not already installed)
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh

# Development install (compiles Rust extension)
maturin develop

# Install with development dependencies
pip install ".[dev]"

# Release build
pip install .
```

### Testing

#### Run All Tests
```bash
pytest tests/
pytest -v --cov=syntonic  # With verbose output and coverage
```

#### Run Single Test File
```bash
pytest tests/test_core/test_state.py
pytest tests/test_resonant_matmul.py
```

#### Run Single Test Function
```bash
pytest tests/test_core/test_state.py::test_state_creation
pytest tests/test_core/test_state.py::test_addition -v
```

#### Run Tests with Specific Markers
```bash
pytest -m "slow"  # Run slow tests only
pytest -m "not slow"  # Skip slow tests
```

### Linting and Formatting

#### Lint Python Code
```bash
ruff check python/
ruff check python/syntonic/core/state.py  # Single file
```

#### Format Python Code
```bash
black python/
black python/syntonic/core/state.py  # Single file
```

#### Type Check Python Code
```bash
mypy python/syntonic/
mypy python/syntonic/core/state.py  # Single file
```

#### Fix Import Issues
```bash
ruff check --fix python/  # Auto-fix issues
```

## Code Style Guidelines

### Python Code Style

#### Imports
```python
# Standard library imports first
from __future__ import annotations
from typing import TYPE_CHECKING, Any, Sequence, Union, Optional, Tuple, List
import math
import cmath
import random

# Third-party imports (minimal - syntonic is self-contained)

# Local imports (relative imports)
from syntonic.core.dtype import DType, float64, complex128
from syntonic.core.device import Device, cpu

# Conditional imports for type checking
if TYPE_CHECKING:
    from syntonic.crt.dhsr_evolution import SyntonyTrajectory
```

#### Type Hints
- Use full type hints with `typing` module
- Use `Union` for multiple types, `Optional` for nullable
- Use `from __future__ import annotations` for forward references
- Define type aliases at module level
```python
ArrayLike = Union[Sequence, 'State']
ShapeLike = Union[int, Tuple[int, ...]]
```

#### Naming Conventions
- **Classes**: `PascalCase` (e.g., `State`, `HarmonizationOperator`)
- **Functions/Methods**: `snake_case` (e.g., `compute_syntony`, `golden_distribution`)
- **Constants**: `UPPER_CASE` (e.g., `PHI_NUMERIC`, `Q_DEFICIT_NUMERIC`)
- **Private attributes/methods**: Leading underscore (e.g., `_storage`, `_compute_syntony`)
- **Modules**: `snake_case` (e.g., `core`, `crt`, `srt`)

#### Docstrings
Use Google-style docstrings with detailed parameter descriptions:
```python
def harmonize(
    psi: State,
    strength: float = PHI_INV,
    preserve_phase: bool = True
) -> State:
    """
    Apply Harmonization Operator Ĥ[Ψ].

    Projects toward Golden Measure equilibrium: ρ(n) ∝ exp(-n²/φ)

    ⚠️ CRITICAL: Weight assignment based on SPATIAL POSITION (mode index n),
    NOT on current magnitude.

    Args:
        psi: State vector (index = spatial position)
        strength: Projection strength γ ∈ [0, 1], default φ⁻¹
        preserve_phase: If True, keep phase from original

    Returns:
        Ĥ[Ψ] - harmonized state
    """
```

#### Error Handling
- Use specific exception types from `syntonic.exceptions`
- Provide clear error messages with context
- Validate inputs at function boundaries
```python
if n <= 0:
    raise ValueError(f"n must be positive, got {n}")

try:
    result = self._storage.operation()
except Exception as e:
    raise SyntonicError(f"Operation failed: {e}") from e
```

#### Constants and Magic Numbers
- Define mathematical constants at module level
- Use exact arithmetic from `syntonic.exact` when possible
- Avoid hardcoded floating-point numbers
```python
# Good
PHI = PHI_NUMERIC
PHI_INV = PHI_NUMERIC ** -1

# Avoid
PHI = 1.618033988749895  # Magic number
```

### Rust Code Style

#### Documentation
```rust
//! Module-level documentation with overview

/// Function documentation with parameter descriptions
/// 
/// # Arguments
/// * `n` - Description of n parameter
/// 
/// # Returns
/// Description of return value
/// 
/// # Example
/// ```
/// let result = function_name(42);
/// ```
```

#### Naming Conventions
- **Functions**: `snake_case`
- **Types/Structs**: `PascalCase`
- **Constants**: `UPPER_CASE`
- **Modules**: `snake_case`

#### Error Handling
- Use `Result<T, E>` for fallible operations
- Provide descriptive error messages
- Use `?` operator for error propagation

### Testing Guidelines

#### Test Structure
```python
import pytest
from syntonic.core import State, complex128

def test_state_creation():
    """Test basic State creation."""
    data = [1.0, 2.0, 3.0]
    state = State(data, dtype=complex128, shape=(3,))

    assert state.shape == (3,)
    assert state.dtype == complex128

def test_state_addition():
    """Test State addition."""
    a = State([1.0, 2.0], shape=(2,))
    b = State([3.0, 4.0], shape=(2,))

    result = a + b
    expected = [4.0, 6.0]

    assert result.to_list() == expected
```

#### Test Organization
- Place tests in `tests/` directory
- Name test files as `test_*.py`
- Use descriptive test function names: `test_feature_behavior`
- Use `pytest` fixtures for common setup

#### Property-Based Testing
```python
from hypothesis import given, strategies as st

@given(st.lists(st.floats(min_value=-10, max_value=10), min_size=1, max_size=100))
def test_state_norm_positive(data):
    """Test that State norm is always non-negative."""
    state = State(data, shape=(len(data),))
    norm = sum(abs(x)**2 for x in state.to_list())**0.5
    assert norm >= 0
```

### Architecture Principles

#### Syntonic Purity
- **NO external dependencies**: NumPy, PyTorch, SciPy are forbidden
- Use only syntonic's Rust backend for tensor operations
- All mathematics must trace back to SRT constants

#### SRT Constants Hierarchy
```python
# Fundamental constants (exact)
PHI = GoldenExact.golden_ratio()  # φ
PHI_INV = GoldenExact.coherence_parameter()  # φ⁻¹

# Numeric approximations (for floating-point operations)
PHI_NUMERIC = srt_phi()
Q_DEFICIT_NUMERIC = srt_q_deficit()
```

#### State vs ResonantTensor
- **State**: General-purpose tensor wrapper with DHSR operations
- **ResonantTensor**: Specialized for neural networks with mode norm tracking

#### DHSR Cycle Pattern
```python
# Preferred chaining syntax
result = psi.differentiate().harmonize().recurse()

# Alternative with explicit syntony
S = psi.syntony
result = psi.differentiate(alpha=0.1).harmonize(strength=S).recurse()
```

### Common Patterns

#### Tensor Operations
```python
# Create State from data
state = State(data, dtype=complex128, shape=(N, M))

# Element-wise operations
result = state.add(other_state)
result = state * scalar  # Scalar multiplication

# Conversion
flat_list = state.to_list()
nested_list = state.tolist()
```

#### Golden Measure Operations
```python
# Golden weights for N modes
weights = [math.exp(-n*n / PHI) for n in range(N)]
normalized_weights = [w / sum(weights) for w in weights]

# Golden initialization
variance = scale * math.exp(-mode_idx**2 / (2 * PHI))
```

#### Complex Number Handling
```python
# Complex operations are native in syntonic
complex_state = State([1+2j, 3+4j], dtype=complex128)
real_part = complex_state.real
imag_part = complex_state.imag
```

### Debugging and Development

#### Common Issues
- **Import Errors**: Check that `maturin develop` was run after Rust changes
- **CUDA Issues**: Ensure CUDA is properly configured for GPU operations
- **Type Errors**: Run `mypy` to catch type annotation issues
- **Test Failures**: Use `pytest -v -s` for detailed output

#### Performance Considerations
- Rust operations are preferred over Python loops
- Use `State` operations instead of manual list manipulation
- Batch operations when possible
- Profile with `pytest --durations=10` to identify slow tests

### Commit Message Conventions

Follow conventional commit format:
```
feat: add new DHSR operator implementation
fix: correct syntony computation in harmonization
docs: update API documentation for State class
test: add property tests for tensor operations
refactor: simplify golden measure calculations
```

### Code Review Checklist

- [ ] **Imports**: No external dependencies (NumPy/PyTorch/SciPy)
- [ ] **Types**: Full type hints with proper annotations
- [ ] **Tests**: Unit tests with good coverage
- [ ] **Documentation**: Complete docstrings with examples
- [ ] **Style**: Passes `ruff` and `black` formatting
- [ ] **Types**: Passes `mypy` type checking
- [ ] **Constants**: Uses SRT constants, not magic numbers
- [ ] **Performance**: Uses Rust backend for tensor operations

This guide ensures all contributions maintain the high standards of the Syntonic library while implementing SRT mathematics correctly.</content>
<parameter name="filePath">AGENTS.md