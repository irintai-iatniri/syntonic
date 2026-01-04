# SYNTONIC PHASE 1 - COMPLETE IMPLEMENTATION

**Timeline:** Weeks 1-6  
**Status:** Foundation Phase  
**Principle:** This phase must be 100% COMPLETE before Phase 2 begins.

---

## OVERVIEW

Phase 1 establishes the foundational infrastructure:

| Week | Focus | Deliverables |
|------|-------|--------------|
| 1 | Project Infrastructure | Repository, CI/CD, config files |
| 2 | Core State Class | `State` class, factory methods, arithmetic |
| 3 | Data Types & Devices | `DType`, `Device`, CUDA detection |
| 4 | Rust Core | `TensorStorage`, PyO3 bindings |
| 5 | Linear Algebra | `syn.linalg` module, decompositions |
| 6 | Testing & Polish | >90% coverage, documentation |

---

## WEEK 1: PROJECT INFRASTRUCTURE

### Repository Structure

```
syntonic/
├── .github/
│   └── workflows/
│       ├── ci.yml              # Continuous integration
│       ├── release.yml         # Release automation
│       └── docs.yml            # Documentation build
├── python/
│   └── syntonic/
│       ├── __init__.py
│       ├── _version.py
│       ├── core/
│       │   ├── __init__.py
│       │   ├── state.py        # State class
│       │   ├── dtype.py        # Data type definitions
│       │   └── device.py       # Device management
│       └── exceptions.py       # Custom exceptions
├── rust/
│   ├── Cargo.toml
│   └── src/
│       ├── lib.rs              # PyO3 module definition
│       └── tensor/
│           ├── mod.rs
│           ├── storage.rs      # Memory storage
│           └── ops.rs          # Basic operations
├── tests/
│   ├── conftest.py             # pytest fixtures
│   └── test_core/
│       └── test_state.py
├── pyproject.toml              # Project metadata + maturin
├── Cargo.toml                  # Workspace Cargo
└── README.md
```

### pyproject.toml

```toml
[build-system]
requires = ["maturin>=1.0,<2.0"]
build-backend = "maturin"

[project]
name = "syntonic"
version = "0.1.0"
description = "Tensor library for Cosmological and Syntony Recursion Theory"
requires-python = ">=3.10"
authors = [{name = "Irintai"}]
keywords = ["tensor", "physics", "CRT", "SRT", "syntony", "recursion"]

dependencies = [
    "numpy>=1.24.0",
]

[project.optional-dependencies]
dev = [
    "pytest>=7.0",
    "pytest-cov",
    "hypothesis>=6.0",
    "black",
    "ruff",
    "mypy",
]

[tool.maturin]
features = ["pyo3/extension-module"]
python-source = "python"
module-name = "syntonic._core"

[tool.pytest.ini_options]
testpaths = ["tests"]
addopts = "-v --cov=syntonic"
```

### rust/Cargo.toml

```toml
[package]
name = "syntonic-core"
version = "0.1.0"
edition = "2021"

[lib]
name = "syntonic_core"
crate-type = ["cdylib"]

[dependencies]
pyo3 = { version = "0.20", features = ["extension-module"] }
numpy = "0.20"
ndarray = "0.15"
num-complex = "0.4"
num-traits = "0.2"
thiserror = "1.0"

[profile.release]
lto = true
codegen-units = 1
```

### Week 1 Exit Criteria

- [ ] Git repository initialized with all config files
- [ ] CI/CD pipeline working (build + test)
- [ ] Basic Rust → Python binding compiles
- [ ] `import syntonic` works

---

## WEEK 2: CORE STATE CLASS

### State Class - COMPLETE IMPLEMENTATION

```python
# python/syntonic/core/state.py

"""
Core State class for Syntonic.

A State represents an evolving information configuration in the
DHSR (Differentiation-Harmonization-Syntony-Recursion) framework.
"""

from __future__ import annotations
from typing import TYPE_CHECKING, Any, Sequence, Union, Optional, Tuple
import numpy as np

from syntonic._core import TensorStorage  # Rust backend
from syntonic.core.dtype import DType, float64, complex128, get_dtype
from syntonic.core.device import Device, cpu, cuda
from syntonic.exceptions import SyntonicError, DeviceError

if TYPE_CHECKING:
    from syntonic.crt.evolution import SyntonyTrajectory

# Type aliases
ArrayLike = Union[Sequence, np.ndarray, 'State']
ShapeLike = Union[int, Tuple[int, ...]]


class State:
    """
    A State in the Syntonic framework.
    
    States are the fundamental objects in CRT/SRT, representing
    information configurations that evolve through DHSR cycles.
    
    Attributes:
        shape: Dimensions of the state
        dtype: Data type (float32, float64, complex64, complex128)
        device: Computation device (cpu, cuda)
        syntony: Current syntony value S(Ψ) ∈ [0, 1]
        gnosis: Current gnosis layer (0-3)
    
    Examples:
        >>> import syntonic as syn
        >>> psi = syn.state([1, 2, 3, 4])
        >>> psi.shape
        (4,)
        
        >>> # DHSR chaining (implemented in Phase 3)
        >>> result = psi.differentiate().harmonize()
    """
    
    __slots__ = ('_storage', '_dtype', '_device', '_syntony_cache', '_gnosis_cache')
    
    def __init__(
        self,
        data: Optional[ArrayLike] = None,
        *,
        dtype: Optional[DType] = None,
        device: Optional[Device] = None,
        shape: Optional[ShapeLike] = None,
    ):
        """
        Create a new State.
        
        Args:
            data: Initial data (list, numpy array, or another State)
            dtype: Data type (default: float64 for real, complex128 for complex)
            device: Device to store on (default: cpu)
            shape: Shape (required if data is None)
        """
        self._device = device or cpu
        self._syntony_cache: Optional[float] = None
        self._gnosis_cache: Optional[int] = None
        
        if data is not None:
            # Handle State input
            if isinstance(data, State):
                self._storage = data._storage.clone()
                self._dtype = data._dtype
                return
            
            # Convert to numpy array
            arr = np.asarray(data)
            
            # Infer dtype from data if not specified
            if dtype is None:
                if np.issubdtype(arr.dtype, np.complexfloating):
                    self._dtype = complex128
                else:
                    self._dtype = float64
            else:
                self._dtype = get_dtype(dtype)
            
            # Create storage via Rust backend
            self._storage = TensorStorage.from_numpy(
                arr.astype(self._dtype.numpy_dtype),
                device=self._device.name
            )
        elif shape is not None:
            self._dtype = dtype or float64
            shape_tuple = shape if isinstance(shape, tuple) else (shape,)
            self._storage = TensorStorage.zeros(
                shape_tuple,
                dtype=self._dtype.name,
                device=self._device.name
            )
        else:
            raise ValueError("Either data or shape must be provided")
    
    # ========== Properties ==========
    
    @property
    def shape(self) -> Tuple[int, ...]:
        """Shape of the state."""
        return tuple(self._storage.shape)
    
    @property
    def ndim(self) -> int:
        """Number of dimensions."""
        return len(self.shape)
    
    @property
    def size(self) -> int:
        """Total number of elements."""
        return self._storage.size
    
    @property
    def dtype(self) -> DType:
        """Data type."""
        return self._dtype
    
    @property
    def device(self) -> Device:
        """Device where state is stored."""
        return self._device
    
    @property
    def syntony(self) -> float:
        """
        Syntony index S(Ψ) ∈ [0, 1].
        
        Note: Returns 0.5 placeholder until Phase 3 implements
        the complete S(Ψ) = 1 - ||D̂Ψ - ĤD̂Ψ|| / ||D̂Ψ - Ψ|| formula.
        """
        if self._syntony_cache is None:
            # Phase 3 will replace this with actual computation
            self._syntony_cache = 0.5
        return self._syntony_cache
    
    @property
    def gnosis(self) -> int:
        """
        Gnosis layer (0-3).
        
        Note: Returns 0 placeholder until Phase 3 implements
        the complete gnosis computation based on Σ Tv thresholds.
        """
        if self._gnosis_cache is None:
            # Phase 3 will replace this with actual computation
            self._gnosis_cache = 0
        return self._gnosis_cache
    
    # ========== Conversion ==========
    
    def numpy(self) -> np.ndarray:
        """Convert to NumPy array."""
        return self._storage.to_numpy()
    
    def torch(self):
        """Convert to PyTorch tensor."""
        try:
            import torch
            return torch.from_numpy(self.numpy())
        except ImportError:
            raise ImportError("PyTorch not installed")
    
    def cuda(self, device_id: int = 0) -> 'State':
        """Move to CUDA device."""
        if self._device.is_cuda:
            return self
        new_storage = self._storage.to_cuda(device_id)
        return self._with_storage(new_storage, device=Device('cuda', device_id))
    
    def cpu(self) -> 'State':
        """Move to CPU."""
        if self._device.is_cpu:
            return self
        new_storage = self._storage.to_cpu()
        return self._with_storage(new_storage, device=cpu)
    
    def to(self, device: Device) -> 'State':
        """Move to specified device."""
        if device.is_cuda:
            return self.cuda(device.index or 0)
        return self.cpu()
    
    # ========== Arithmetic Operations ==========
    
    def __add__(self, other) -> 'State':
        if isinstance(other, State):
            new_storage = self._storage.add(other._storage)
        else:
            new_storage = self._storage.add_scalar(float(other))
        return self._with_storage(new_storage)
    
    def __radd__(self, other) -> 'State':
        return self.__add__(other)
    
    def __sub__(self, other) -> 'State':
        if isinstance(other, State):
            new_storage = self._storage.sub(other._storage)
        else:
            new_storage = self._storage.sub_scalar(float(other))
        return self._with_storage(new_storage)
    
    def __rsub__(self, other) -> 'State':
        return (-self).__add__(other)
    
    def __mul__(self, other) -> 'State':
        if isinstance(other, State):
            new_storage = self._storage.mul(other._storage)
        else:
            new_storage = self._storage.mul_scalar(float(other))
        return self._with_storage(new_storage)
    
    def __rmul__(self, other) -> 'State':
        return self.__mul__(other)
    
    def __truediv__(self, other) -> 'State':
        if isinstance(other, State):
            new_storage = self._storage.div(other._storage)
        else:
            new_storage = self._storage.div_scalar(float(other))
        return self._with_storage(new_storage)
    
    def __neg__(self) -> 'State':
        return self._with_storage(self._storage.neg())
    
    def __matmul__(self, other: 'State') -> 'State':
        """Matrix multiplication."""
        new_storage = self._storage.matmul(other._storage)
        return self._with_storage(new_storage)
    
    def __pow__(self, n: int) -> 'State':
        """Element-wise power."""
        new_storage = self._storage.pow(n)
        return self._with_storage(new_storage)
    
    # ========== Reduction Operations ==========
    
    def norm(self, ord: Optional[int] = 2) -> float:
        """
        Compute norm of state.
        
        Args:
            ord: Norm order (default: 2 for L2/Frobenius norm)
        
        Returns:
            Scalar norm value
        """
        return float(self._storage.norm(ord))
    
    def normalize(self) -> 'State':
        """Return normalized state (unit norm)."""
        n = self.norm()
        if n == 0:
            raise SyntonicError("Cannot normalize zero state")
        return self / n
    
    def sum(self, axis: Optional[int] = None) -> Union[float, 'State']:
        """Sum elements."""
        if axis is None:
            return float(self._storage.sum_all())
        return self._with_storage(self._storage.sum_axis(axis))
    
    def mean(self, axis: Optional[int] = None) -> Union[float, 'State']:
        """Mean of elements."""
        if axis is None:
            return float(self._storage.mean_all())
        return self._with_storage(self._storage.mean_axis(axis))
    
    def max(self, axis: Optional[int] = None) -> Union[float, 'State']:
        """Maximum element."""
        if axis is None:
            return float(self._storage.max_all())
        return self._with_storage(self._storage.max_axis(axis))
    
    def min(self, axis: Optional[int] = None) -> Union[float, 'State']:
        """Minimum element."""
        if axis is None:
            return float(self._storage.min_all())
        return self._with_storage(self._storage.min_axis(axis))
    
    def abs(self) -> 'State':
        """Element-wise absolute value."""
        return self._with_storage(self._storage.abs())
    
    # ========== Complex Operations ==========
    
    def conj(self) -> 'State':
        """Complex conjugate."""
        return self._with_storage(self._storage.conj())
    
    def real(self) -> 'State':
        """Real part."""
        return self._with_storage(self._storage.real())
    
    def imag(self) -> 'State':
        """Imaginary part."""
        return self._with_storage(self._storage.imag())
    
    @property
    def T(self) -> 'State':
        """Transpose."""
        return self._with_storage(self._storage.transpose())
    
    @property
    def H(self) -> 'State':
        """Conjugate transpose (Hermitian adjoint)."""
        return self.conj().T
    
    # ========== Shape Operations ==========
    
    def reshape(self, *shape) -> 'State':
        """Reshape state."""
        if len(shape) == 1 and isinstance(shape[0], (list, tuple)):
            shape = tuple(shape[0])
        return self._with_storage(self._storage.reshape(shape))
    
    def flatten(self) -> 'State':
        """Flatten to 1D."""
        return self.reshape(-1)
    
    def squeeze(self) -> 'State':
        """Remove dimensions of size 1."""
        return self._with_storage(self._storage.squeeze())
    
    def unsqueeze(self, dim: int) -> 'State':
        """Add dimension of size 1."""
        return self._with_storage(self._storage.unsqueeze(dim))
    
    # ========== DHSR Operations (Phase 3 implements these) ==========
    
    def differentiate(self, **kwargs) -> 'State':
        """
        Apply differentiation operator D̂.
        
        D̂[Ψ] = Ψ + Σᵢ αᵢ(S) P̂ᵢ[Ψ] + ζ(S) ∇²[Ψ]
        
        Note: Full implementation in Phase 3.
        Returns self for now to enable method chaining.
        """
        # Phase 3 will implement:
        # from syntonic.crt.operators import DifferentiationOperator
        # D = DifferentiationOperator(**kwargs)
        # return D(self)
        return self._with_storage(self._storage.clone())
    
    def harmonize(self, **kwargs) -> 'State':
        """
        Apply harmonization operator Ĥ.
        
        Ĥ[Ψ] = Ψ - Σᵢ βᵢ(S,Δ) Q̂ᵢ[Ψ] + γ(S) Ŝ_op[Ψ] + Δ_NL[Ψ]
        
        Note: Full implementation in Phase 3.
        Returns self for now to enable method chaining.
        """
        # Phase 3 will implement:
        # from syntonic.crt.operators import HarmonizationOperator
        # H = HarmonizationOperator(**kwargs)
        # return H(self)
        return self._with_storage(self._storage.clone())
    
    def recurse(self, **kwargs) -> 'State':
        """
        Apply recursion operator R̂ = Ĥ ∘ D̂.
        
        Note: Full implementation in Phase 3.
        """
        return self.differentiate(**kwargs).harmonize(**kwargs)
    
    # ========== Indexing ==========
    
    def __getitem__(self, key) -> 'State':
        new_storage = self._storage.getitem(key)
        return self._with_storage(new_storage)
    
    def __setitem__(self, key, value):
        if isinstance(value, State):
            self._storage.setitem(key, value._storage)
        else:
            self._storage.setitem_scalar(key, value)
        self._invalidate_caches()
    
    # ========== Representation ==========
    
    def __repr__(self) -> str:
        return f"State(shape={self.shape}, dtype={self._dtype.name}, device={self._device})"
    
    def __str__(self) -> str:
        arr_str = np.array2string(self.numpy(), precision=4, suppress_small=True)
        return f"State({arr_str})"
    
    def __len__(self) -> int:
        return self.shape[0] if self.shape else 0
    
    # ========== NumPy Protocol ==========
    
    def __array__(self, dtype=None) -> np.ndarray:
        arr = self.numpy()
        if dtype is not None:
            arr = arr.astype(dtype)
        return arr
    
    # ========== Private Methods ==========
    
    def _with_storage(
        self,
        storage: TensorStorage,
        device: Optional[Device] = None,
    ) -> 'State':
        """Create new State with given storage."""
        new_state = object.__new__(State)
        new_state._storage = storage
        new_state._dtype = self._dtype
        new_state._device = device or self._device
        new_state._syntony_cache = None
        new_state._gnosis_cache = None
        return new_state
    
    def _invalidate_caches(self):
        """Invalidate computed property caches."""
        self._syntony_cache = None
        self._gnosis_cache = None
    
    # ========== Class Methods ==========
    
    @classmethod
    def from_numpy(cls, arr: np.ndarray, **kwargs) -> 'State':
        """Create State from NumPy array."""
        return cls(arr, **kwargs)
    
    @classmethod
    def from_torch(cls, tensor, **kwargs) -> 'State':
        """Create State from PyTorch tensor."""
        return cls(tensor.detach().cpu().numpy(), **kwargs)


# ========== Factory Function ==========

def state(
    data: Optional[ArrayLike] = None,
    *,
    dtype: Optional[DType] = None,
    device: Optional[Device] = None,
    **kwargs,
) -> State:
    """
    Create a new State.
    
    This is the primary way to create States in Syntonic.
    
    Examples:
        >>> import syntonic as syn
        >>> psi = syn.state([1, 2, 3, 4])
        >>> psi = syn.state([[1, 2], [3, 4]], dtype=syn.complex128)
    """
    return State(data, dtype=dtype, device=device, **kwargs)


# ========== Namespace for Factory Methods ==========

class StateNamespace:
    """Namespace for state creation methods."""
    
    @staticmethod
    def zeros(shape: ShapeLike, *, dtype: DType = float64, device: Device = cpu) -> State:
        """Create zero-filled state."""
        return State(shape=shape, dtype=dtype, device=device)
    
    @staticmethod
    def ones(shape: ShapeLike, *, dtype: DType = float64, device: Device = cpu) -> State:
        """Create state filled with ones."""
        shape_tuple = shape if isinstance(shape, tuple) else (shape,)
        arr = np.ones(shape_tuple)
        return State(arr, dtype=dtype, device=device)
    
    @staticmethod
    def random(
        shape: ShapeLike,
        *,
        dtype: DType = float64,
        device: Device = cpu,
        seed: Optional[int] = None,
    ) -> State:
        """Create random state (uniform [0, 1])."""
        rng = np.random.default_rng(seed)
        shape_tuple = shape if isinstance(shape, tuple) else (shape,)
        if dtype.is_complex:
            arr = rng.random(shape_tuple) + 1j * rng.random(shape_tuple)
        else:
            arr = rng.random(shape_tuple)
        return State(arr, dtype=dtype, device=device)
    
    @staticmethod
    def randn(
        shape: ShapeLike,
        *,
        dtype: DType = float64,
        device: Device = cpu,
        seed: Optional[int] = None,
    ) -> State:
        """Create random state (standard normal)."""
        rng = np.random.default_rng(seed)
        shape_tuple = shape if isinstance(shape, tuple) else (shape,)
        if dtype.is_complex:
            arr = (rng.standard_normal(shape_tuple) + 
                   1j * rng.standard_normal(shape_tuple)) / np.sqrt(2)
        else:
            arr = rng.standard_normal(shape_tuple)
        return State(arr, dtype=dtype, device=device)
    
    @staticmethod
    def eye(n: int, *, dtype: DType = float64, device: Device = cpu) -> State:
        """Create identity matrix state."""
        return State(np.eye(n), dtype=dtype, device=device)
    
    @staticmethod
    def from_numpy(arr: np.ndarray, **kwargs) -> State:
        """Create from NumPy array."""
        return State.from_numpy(arr, **kwargs)
    
    @staticmethod
    def from_torch(tensor, **kwargs) -> State:
        """Create from PyTorch tensor."""
        return State.from_torch(tensor, **kwargs)


# Attach namespace to state function
state.zeros = StateNamespace.zeros
state.ones = StateNamespace.ones
state.random = StateNamespace.random
state.randn = StateNamespace.randn
state.eye = StateNamespace.eye
state.from_numpy = StateNamespace.from_numpy
state.from_torch = StateNamespace.from_torch
```

### Week 2 Exit Criteria

- [ ] `State` class with all properties functional
- [ ] All factory methods: `zeros`, `ones`, `random`, `randn`, `eye`
- [ ] All arithmetic: `+`, `-`, `*`, `/`, `@`, `**`
- [ ] All reductions: `norm`, `sum`, `mean`, `max`, `min`
- [ ] Shape operations: `reshape`, `flatten`, `squeeze`, `T`, `H`
- [ ] Conversion: `numpy()`, `torch()`, `cuda()`, `cpu()`
- [ ] DHSR stubs defined (for Phase 3)
- [ ] Unit tests passing

---

## WEEK 3: DATA TYPES AND DEVICE MANAGEMENT

### DType System

```python
# python/syntonic/core/dtype.py

"""Data type definitions for Syntonic."""

from dataclasses import dataclass
from typing import Optional
import numpy as np


@dataclass(frozen=True)
class DType:
    """Syntonic data type."""
    
    name: str
    numpy_dtype: np.dtype
    size: int  # bytes
    is_complex: bool = False
    is_floating: bool = True
    
    def __str__(self) -> str:
        return self.name
    
    def __repr__(self) -> str:
        return f"syn.{self.name}"


# Standard floating point
float32 = DType("float32", np.dtype(np.float32), 4)
float64 = DType("float64", np.dtype(np.float64), 8)  # DEFAULT

# Complex
complex64 = DType("complex64", np.dtype(np.complex64), 8, is_complex=True)
complex128 = DType("complex128", np.dtype(np.complex128), 16, is_complex=True)  # DEFAULT for complex

# Integer (for winding numbers)
int32 = DType("int32", np.dtype(np.int32), 4, is_floating=False)
int64 = DType("int64", np.dtype(np.int64), 8, is_floating=False)

# Winding type (alias for int64, semantically distinct for T⁴ indices)
winding = DType("winding", np.dtype(np.int64), 8, is_floating=False)

# Type mapping for conversions
_DTYPE_MAP = {
    'float32': float32, 'f32': float32,
    'float64': float64, 'f64': float64, 'float': float64,
    'complex64': complex64, 'c64': complex64,
    'complex128': complex128, 'c128': complex128, 'complex': complex128,
    'int32': int32, 'i32': int32,
    'int64': int64, 'i64': int64, 'int': int64,
    'winding': winding,
}


def get_dtype(dtype_spec) -> DType:
    """Get DType from various specifications."""
    if isinstance(dtype_spec, DType):
        return dtype_spec
    if isinstance(dtype_spec, str):
        if dtype_spec in _DTYPE_MAP:
            return _DTYPE_MAP[dtype_spec]
    if isinstance(dtype_spec, np.dtype):
        for dt in _DTYPE_MAP.values():
            if dt.numpy_dtype == dtype_spec:
                return dt
    raise ValueError(f"Unknown dtype: {dtype_spec}")


def promote_dtypes(dtype1: DType, dtype2: DType) -> DType:
    """Determine result dtype from two input dtypes."""
    # Complex takes precedence
    if dtype1.is_complex or dtype2.is_complex:
        if dtype1.size >= 16 or dtype2.size >= 16:
            return complex128
        return complex64
    # Larger precision wins
    if dtype1.size >= dtype2.size:
        return dtype1
    return dtype2
```

### Device Management

```python
# python/syntonic/core/device.py

"""Device management for Syntonic."""

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class Device:
    """Represents a computation device."""
    
    type: str  # 'cpu' or 'cuda'
    index: Optional[int] = None
    
    @property
    def name(self) -> str:
        if self.type == 'cpu':
            return 'cpu'
        return f'cuda:{self.index or 0}'
    
    @property
    def is_cpu(self) -> bool:
        return self.type == 'cpu'
    
    @property
    def is_cuda(self) -> bool:
        return self.type == 'cuda'
    
    def __str__(self) -> str:
        return self.name
    
    def __repr__(self) -> str:
        return f"syn.device('{self.name}')"


# Singleton CPU device
cpu = Device('cpu')


def cuda(device_id: int = 0) -> Device:
    """Get CUDA device."""
    if not cuda_is_available():
        raise RuntimeError("CUDA not available")
    return Device('cuda', device_id)


def cuda_is_available() -> bool:
    """Check if CUDA is available."""
    try:
        from syntonic._core import check_cuda_available
        return check_cuda_available()
    except ImportError:
        return False


def cuda_device_count() -> int:
    """Get number of CUDA devices."""
    if not cuda_is_available():
        return 0
    from syntonic._core import get_cuda_device_count
    return get_cuda_device_count()


def device(spec: str) -> Device:
    """Parse device from string specification."""
    if spec == 'cpu':
        return cpu
    if spec.startswith('cuda'):
        if ':' in spec:
            idx = int(spec.split(':')[1])
            return cuda(idx)
        return cuda(0)
    raise ValueError(f"Unknown device: {spec}")
```

### Week 3 Exit Criteria

- [ ] Complete `DType` system with all types
- [ ] `Device` class and utilities
- [ ] CUDA availability detection
- [ ] Device movement (`cuda()`, `cpu()`, `to()`)
- [ ] Type promotion rules
- [ ] Tests for type conversion and device movement

---

## WEEK 4: RUST CORE - TENSOR STORAGE

### TensorStorage Implementation

```rust
// rust/src/tensor/storage.rs

use ndarray::{Array, ArrayD, IxDyn};
use num_complex::Complex64;
use pyo3::prelude::*;
use numpy::{PyArrayDyn, PyReadonlyArrayDyn, IntoPyArray};

/// Core tensor storage backed by ndarray
#[pyclass]
pub struct TensorStorage {
    data: TensorData,
    shape: Vec<usize>,
    device: String,
}

enum TensorData {
    Float32(ArrayD<f32>),
    Float64(ArrayD<f64>),
    Complex64(ArrayD<Complex64>),
    Int64(ArrayD<i64>),
}

#[pymethods]
impl TensorStorage {
    #[getter]
    fn shape(&self) -> Vec<usize> {
        self.shape.clone()
    }
    
    #[getter]
    fn size(&self) -> usize {
        self.shape.iter().product()
    }
    
    #[staticmethod]
    fn from_numpy(py: Python, arr: PyReadonlyArrayDyn<'_, f64>, device: &str) -> PyResult<Self> {
        let array = arr.as_array().to_owned();
        let shape = array.shape().to_vec();
        
        Ok(TensorStorage {
            data: TensorData::Float64(array),
            shape,
            device: device.to_string(),
        })
    }
    
    #[staticmethod]
    fn zeros(shape: Vec<usize>, dtype: &str, device: &str) -> PyResult<Self> {
        let dim = IxDyn(&shape);
        
        let data = match dtype {
            "float32" => TensorData::Float32(ArrayD::zeros(dim)),
            "float64" => TensorData::Float64(ArrayD::zeros(dim)),
            "complex128" => TensorData::Complex64(ArrayD::zeros(dim)),
            "int64" => TensorData::Int64(ArrayD::zeros(dim)),
            _ => return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                format!("Unknown dtype: {}", dtype)
            )),
        };
        
        Ok(TensorStorage { data, shape, device: device.to_string() })
    }
    
    fn to_numpy<'py>(&self, py: Python<'py>) -> PyResult<&'py PyArrayDyn<f64>> {
        match &self.data {
            TensorData::Float64(arr) => Ok(arr.clone().into_pyarray(py)),
            _ => Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>(
                "to_numpy currently only supports float64"
            )),
        }
    }
    
    fn clone(&self) -> Self {
        TensorStorage {
            data: match &self.data {
                TensorData::Float32(a) => TensorData::Float32(a.clone()),
                TensorData::Float64(a) => TensorData::Float64(a.clone()),
                TensorData::Complex64(a) => TensorData::Complex64(a.clone()),
                TensorData::Int64(a) => TensorData::Int64(a.clone()),
            },
            shape: self.shape.clone(),
            device: self.device.clone(),
        }
    }
    
    // Arithmetic operations
    fn add(&self, other: &TensorStorage) -> PyResult<Self> {
        match (&self.data, &other.data) {
            (TensorData::Float64(a), TensorData::Float64(b)) => {
                Ok(TensorStorage {
                    data: TensorData::Float64(a + b),
                    shape: self.shape.clone(),
                    device: self.device.clone(),
                })
            }
            _ => Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>("Dtype mismatch")),
        }
    }
    
    fn sub(&self, other: &TensorStorage) -> PyResult<Self> {
        match (&self.data, &other.data) {
            (TensorData::Float64(a), TensorData::Float64(b)) => {
                Ok(TensorStorage {
                    data: TensorData::Float64(a - b),
                    shape: self.shape.clone(),
                    device: self.device.clone(),
                })
            }
            _ => Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>("Dtype mismatch")),
        }
    }
    
    fn mul(&self, other: &TensorStorage) -> PyResult<Self> {
        match (&self.data, &other.data) {
            (TensorData::Float64(a), TensorData::Float64(b)) => {
                Ok(TensorStorage {
                    data: TensorData::Float64(a * b),
                    shape: self.shape.clone(),
                    device: self.device.clone(),
                })
            }
            _ => Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>("Dtype mismatch")),
        }
    }
    
    fn add_scalar(&self, scalar: f64) -> Self {
        match &self.data {
            TensorData::Float64(a) => TensorStorage {
                data: TensorData::Float64(a + scalar),
                shape: self.shape.clone(),
                device: self.device.clone(),
            },
            _ => panic!("add_scalar only supports float64"),
        }
    }
    
    fn mul_scalar(&self, scalar: f64) -> Self {
        match &self.data {
            TensorData::Float64(a) => TensorStorage {
                data: TensorData::Float64(a * scalar),
                shape: self.shape.clone(),
                device: self.device.clone(),
            },
            _ => panic!("mul_scalar only supports float64"),
        }
    }
    
    fn neg(&self) -> Self {
        match &self.data {
            TensorData::Float64(a) => TensorStorage {
                data: TensorData::Float64(-a.clone()),
                shape: self.shape.clone(),
                device: self.device.clone(),
            },
            _ => panic!("neg only supports float64"),
        }
    }
    
    fn norm(&self, ord: Option<i32>) -> f64 {
        match &self.data {
            TensorData::Float64(a) => {
                let ord = ord.unwrap_or(2);
                if ord == 2 {
                    a.iter().map(|x| x * x).sum::<f64>().sqrt()
                } else if ord == 1 {
                    a.iter().map(|x| x.abs()).sum()
                } else {
                    a.iter().map(|x| x.abs().powi(ord)).sum::<f64>().powf(1.0 / ord as f64)
                }
            }
            _ => panic!("norm only supports float64"),
        }
    }
    
    fn sum_all(&self) -> f64 {
        match &self.data {
            TensorData::Float64(a) => a.iter().sum(),
            _ => panic!("sum_all only supports float64"),
        }
    }
    
    fn mean_all(&self) -> f64 {
        self.sum_all() / self.size() as f64
    }
}
```

### Week 4 Exit Criteria

- [ ] `TensorStorage` Rust struct with ndarray backend
- [ ] Float32, Float64, Complex128, Int64 support
- [ ] All arithmetic operations in Rust
- [ ] Norm computation
- [ ] Python ↔ Rust bindings working
- [ ] Performance tests vs NumPy

---

## WEEK 5: LINEAR ALGEBRA

### Linear Algebra Module

```python
# python/syntonic/linalg/__init__.py

"""Linear algebra operations for Syntonic."""

from syntonic.core.state import State
from syntonic._core import linalg as _linalg
from typing import Tuple, Optional


def matmul(a: State, b: State) -> State:
    """Matrix multiplication."""
    return a @ b


def dot(a: State, b: State) -> float:
    """Dot product."""
    return float(_linalg.dot(a._storage, b._storage))


def inner(a: State, b: State) -> float:
    """Inner product ⟨a|b⟩."""
    return dot(a.conj().flatten(), b.flatten())


def outer(a: State, b: State) -> State:
    """Outer product |a⟩⟨b|."""
    return State._from_storage(_linalg.outer(a._storage, b._storage))


def norm(x: State, ord: Optional[int] = None) -> float:
    """Vector or matrix norm."""
    return x.norm(ord)


def eig(a: State) -> Tuple[State, State]:
    """
    Eigenvalue decomposition.
    
    Returns:
        (eigenvalues, eigenvectors)
    """
    w, v = _linalg.eig(a._storage)
    return State._from_storage(w), State._from_storage(v)


def eigh(a: State) -> Tuple[State, State]:
    """
    Eigenvalue decomposition for Hermitian matrices.
    
    Returns:
        (eigenvalues, eigenvectors)
    """
    w, v = _linalg.eigh(a._storage)
    return State._from_storage(w), State._from_storage(v)


def svd(a: State, full_matrices: bool = True) -> Tuple[State, State, State]:
    """
    Singular value decomposition.
    
    Returns:
        (U, S, Vh)
    """
    u, s, vh = _linalg.svd(a._storage, full_matrices)
    return State._from_storage(u), State._from_storage(s), State._from_storage(vh)


def qr(a: State) -> Tuple[State, State]:
    """
    QR decomposition.
    
    Returns:
        (Q, R) where A = QR
    """
    q, r = _linalg.qr(a._storage)
    return State._from_storage(q), State._from_storage(r)


def cholesky(a: State) -> State:
    """
    Cholesky decomposition.
    
    Returns:
        L where A = LL*
    """
    return State._from_storage(_linalg.cholesky(a._storage))


def inv(a: State) -> State:
    """Matrix inverse."""
    return State._from_storage(_linalg.inv(a._storage))


def pinv(a: State) -> State:
    """Moore-Penrose pseudo-inverse."""
    return State._from_storage(_linalg.pinv(a._storage))


def det(a: State) -> float:
    """Matrix determinant."""
    return float(_linalg.det(a._storage))


def trace(a: State) -> float:
    """Matrix trace."""
    return float(_linalg.trace(a._storage))


def solve(a: State, b: State) -> State:
    """
    Solve linear system Ax = b.
    
    Returns:
        x such that Ax = b
    """
    return State._from_storage(_linalg.solve(a._storage, b._storage))


def expm(a: State) -> State:
    """
    Matrix exponential exp(A).
    
    Important for CRT: evolution operators.
    """
    return State._from_storage(_linalg.expm(a._storage))


def logm(a: State) -> State:
    """Matrix logarithm log(A)."""
    return State._from_storage(_linalg.logm(a._storage))
```

### Week 5 Exit Criteria

- [ ] All decompositions: `eig`, `eigh`, `svd`, `qr`, `cholesky`
- [ ] Solvers: `solve`, `inv`, `pinv`
- [ ] Matrix functions: `expm`, `logm`
- [ ] Inner/outer products
- [ ] Tests against NumPy/SciPy reference implementations
- [ ] Performance benchmarks

---

## WEEK 6: TESTING & POLISH

### Test Suite

```python
# tests/test_core/test_state.py

import syntonic as syn
import numpy as np
import pytest
from hypothesis import given, strategies as st


class TestStateCreation:
    """Tests for State creation."""
    
    def test_from_list(self):
        psi = syn.state([1, 2, 3, 4])
        assert psi.shape == (4,)
        assert psi.dtype == syn.float64
        assert np.allclose(psi.numpy(), [1, 2, 3, 4])
    
    def test_from_complex_list(self):
        psi = syn.state([1+2j, 3+4j])
        assert psi.shape == (2,)
        assert psi.dtype == syn.complex128
    
    def test_zeros(self):
        psi = syn.state.zeros((3, 3))
        assert psi.shape == (3, 3)
        assert np.allclose(psi.numpy(), 0)
    
    def test_ones(self):
        psi = syn.state.ones((5,))
        assert np.allclose(psi.numpy(), 1)
    
    def test_random_seeded(self):
        psi1 = syn.state.random((10,), seed=42)
        psi2 = syn.state.random((10,), seed=42)
        assert np.allclose(psi1.numpy(), psi2.numpy())
    
    def test_eye(self):
        I = syn.state.eye(3)
        assert np.allclose(I.numpy(), np.eye(3))
    
    @given(st.lists(st.floats(allow_nan=False, allow_infinity=False), 
                    min_size=1, max_size=100))
    def test_from_arbitrary_list(self, data):
        psi = syn.state(data)
        assert psi.shape == (len(data),)


class TestStateArithmetic:
    """Tests for State arithmetic operations."""
    
    def test_add_states(self):
        a = syn.state([1, 2, 3])
        b = syn.state([4, 5, 6])
        c = a + b
        assert np.allclose(c.numpy(), [5, 7, 9])
    
    def test_sub_states(self):
        a = syn.state([5, 5, 5])
        b = syn.state([1, 2, 3])
        c = a - b
        assert np.allclose(c.numpy(), [4, 3, 2])
    
    def test_mul_states(self):
        a = syn.state([1, 2, 3])
        b = syn.state([2, 2, 2])
        c = a * b
        assert np.allclose(c.numpy(), [2, 4, 6])
    
    def test_add_scalar(self):
        a = syn.state([1, 2, 3])
        b = a + 10
        assert np.allclose(b.numpy(), [11, 12, 13])
    
    def test_mul_scalar(self):
        a = syn.state([1, 2, 3])
        b = a * 2
        assert np.allclose(b.numpy(), [2, 4, 6])
    
    def test_neg(self):
        a = syn.state([1, -2, 3])
        b = -a
        assert np.allclose(b.numpy(), [-1, 2, -3])
    
    def test_matmul(self):
        A = syn.state([[1, 2], [3, 4]])
        B = syn.state([[5, 6], [7, 8]])
        C = A @ B
        expected = np.array([[19, 22], [43, 50]])
        assert np.allclose(C.numpy(), expected)


class TestStateReductions:
    """Tests for State reduction operations."""
    
    def test_norm_l2(self):
        psi = syn.state([3, 4])
        assert np.isclose(psi.norm(), 5.0)
    
    def test_norm_l1(self):
        psi = syn.state([3, -4])
        assert np.isclose(psi.norm(ord=1), 7.0)
    
    def test_normalize(self):
        psi = syn.state([3, 4]).normalize()
        assert np.isclose(psi.norm(), 1.0)
    
    def test_sum(self):
        psi = syn.state([1, 2, 3, 4])
        assert np.isclose(psi.sum(), 10.0)
    
    def test_mean(self):
        psi = syn.state([1, 2, 3, 4])
        assert np.isclose(psi.mean(), 2.5)


class TestStateDHSR:
    """Tests for DHSR operations (stubs in Phase 1)."""
    
    def test_differentiate_returns_state(self):
        psi = syn.state([1, 2, 3, 4])
        d_psi = psi.differentiate()
        assert isinstance(d_psi, syn.State)
        assert d_psi.shape == psi.shape
    
    def test_harmonize_returns_state(self):
        psi = syn.state([1, 2, 3, 4])
        h_psi = psi.harmonize()
        assert isinstance(h_psi, syn.State)
        assert h_psi.shape == psi.shape
    
    def test_recurse_chains(self):
        psi = syn.state([1, 2, 3, 4])
        r_psi = psi.recurse()
        assert isinstance(r_psi, syn.State)
    
    def test_dhsr_chaining(self):
        psi = syn.state.random((10,))
        result = psi.differentiate().harmonize().differentiate().harmonize()
        assert result.shape == psi.shape


class TestStateDevice:
    """Tests for device operations."""
    
    def test_default_cpu(self):
        psi = syn.state([1, 2, 3])
        assert psi.device == syn.cpu
        assert psi.device.is_cpu
    
    @pytest.mark.skipif(not syn.cuda_is_available(), reason="CUDA not available")
    def test_cuda_transfer(self):
        psi = syn.state([1, 2, 3]).cuda()
        assert psi.device.is_cuda
        psi_cpu = psi.cpu()
        assert psi_cpu.device.is_cpu
        assert np.allclose(psi_cpu.numpy(), [1, 2, 3])


class TestStateInterop:
    """Tests for NumPy/PyTorch interoperability."""
    
    def test_to_numpy(self):
        psi = syn.state([1, 2, 3])
        arr = psi.numpy()
        assert isinstance(arr, np.ndarray)
        assert np.allclose(arr, [1, 2, 3])
    
    def test_from_numpy(self):
        arr = np.array([1.0, 2.0, 3.0])
        psi = syn.state.from_numpy(arr)
        assert np.allclose(psi.numpy(), arr)
    
    def test_numpy_protocol(self):
        psi = syn.state([1, 2, 3])
        arr = np.asarray(psi)
        assert isinstance(arr, np.ndarray)
    
    def test_numpy_operations(self):
        psi = syn.state([1, 2, 3])
        result = np.sum(psi)  # Uses __array__ protocol
        assert np.isclose(result, 6.0)
```

### Week 6 Exit Criteria

- [ ] Test coverage >90%
- [ ] Property-based tests with Hypothesis
- [ ] All docstrings complete
- [ ] Package installable via `pip install -e .`
- [ ] CI/CD passing
- [ ] README with quickstart guide

---

## PHASE 1 EXIT CRITERIA

| Component | Requirement | Status |
|-----------|-------------|--------|
| `State` class | All methods functional | [ ] |
| Factory methods | `zeros`, `ones`, `random`, `eye`, etc. | [ ] |
| Arithmetic | `+`, `-`, `*`, `/`, `@`, `**` | [ ] |
| Reductions | `norm`, `sum`, `mean`, `max`, `min` | [ ] |
| Shape ops | `reshape`, `T`, `H`, `squeeze` | [ ] |
| Conversion | `numpy()`, `torch()`, `cuda()`, `cpu()` | [ ] |
| DType system | All types working | [ ] |
| Device system | CPU + CUDA detection | [ ] |
| Rust core | TensorStorage functional | [ ] |
| Linear algebra | All decompositions | [ ] |
| Test coverage | >90% | [ ] |
| Documentation | Complete | [ ] |

**Phase 1 is COMPLETE when all boxes are checked.**

---

*Document Version: 1.0*  
*This phase must be 100% complete before starting Phase 2.*