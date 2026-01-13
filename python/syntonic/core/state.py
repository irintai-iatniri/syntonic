"""
Core State class for Syntonic.

A State represents an evolving information configuration in the
DHSR (Differentiation-Harmonization-Syntony-Recursion) framework.

This module is completely NumPy-free - all tensor operations are handled
by the Rust backend via from_list/to_list data transfer.
"""

from __future__ import annotations
from typing import TYPE_CHECKING, Any, Sequence, Union, Optional, Tuple, List
import math
import cmath
import random

from syntonic._core import TensorStorage, cuda_is_available, cuda_device_count
from syntonic.core.dtype import DType, float64, float32, complex128, int64, get_dtype
from syntonic.core.device import Device, cpu

if TYPE_CHECKING:
    from syntonic.crt.evolution import SyntonyTrajectory

# Type aliases
ArrayLike = Union[Sequence, 'State']
ShapeLike = Union[int, Tuple[int, ...]]


def _flatten(data: Any, depth: int = 0) -> Tuple[List, List[int]]:
    """
    Recursively flatten nested lists and compute shape.

    Returns:
        (flat_list, shape)
    """
    # Handle numpy arrays
    try:
        import numpy as np
        if isinstance(data, np.ndarray):
            # Convert numpy array to nested list structure first
            if data.ndim == 0:
                return [data.item()], []
            elif data.ndim == 1:
                return data.tolist(), [data.shape[0]]
            else:
                # For multi-dimensional arrays, convert to nested list
                nested = data.tolist()
                return _flatten(nested, depth)
    except ImportError:
        pass  # numpy not available, continue with normal processing

    if not isinstance(data, (list, tuple)):
        return [data], []

    if len(data) == 0:
        return [], [0]

    # Check if all elements are the same type
    first = data[0]
    if isinstance(first, (list, tuple)):
        # Nested structure
        results = []
        shapes = []
        for item in data:
            flat, shape = _flatten(item, depth + 1)
            results.extend(flat)
            shapes.append(shape)

        # Verify all shapes match
        if not all(s == shapes[0] for s in shapes):
            raise ValueError("Inconsistent shapes in nested structure")

        return results, [len(data)] + shapes[0]
    else:
        # Leaf level
        return list(data), [len(data)]


def _unflatten(flat: List, shape: List[int]) -> Any:
    """Reshape flat list back into nested structure."""
    if len(shape) == 0:
        return flat[0] if flat else 0.0  # pragma: no cover
    if len(shape) == 1:
        return flat[:shape[0]]

    # Compute subshape size
    subshape = shape[1:]
    subsize = 1
    for s in subshape:
        subsize *= s

    result = []
    for i in range(shape[0]):
        start = i * subsize
        end = start + subsize
        result.append(_unflatten(flat[start:end], subshape))
    return result


def _is_complex(value: Any) -> bool:
    """Check if a value is complex."""
    return isinstance(value, complex)


def _has_complex(data: List) -> bool:
    """Check if any element in flat list is complex."""
    return any(_is_complex(x) for x in data)


class State:
    """
    A State in the Syntonic framework.

    States are the fundamental objects in CRT/SRT, representing
    information configurations that evolve through DHSR cycles.

    Attributes:
        shape: Dimensions of the state
        dtype: Data type (float32, float64, complex64, complex128)
        device: Computation device (cpu, cuda)
        syntony: Current syntony value S(Psi) in [0, 1]
        gnosis: Current gnosis layer (0-3)

    Examples:
        >>> import syntonic as syn
        >>> psi = syn.state([1, 2, 3, 4])
        >>> psi.shape
        (4,)

        >>> # DHSR chaining
        >>> result = psi.differentiate().harmonize()
    """

    __slots__ = ('_storage', '_dtype', '_device', '_shape', '_syntony_cache', '_gnosis_cache')

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
            data: Initial data (list or another State)
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
                flat = data.to_list()
                self._dtype = data._dtype
                self._shape = data._shape
                self._storage = TensorStorage.from_list(
                    flat,
                    list(self._shape),
                    self._dtype.name,
                    self._device.name
                )
                return

            # Flatten and compute shape
            flat_data, computed_shape = _flatten(data)

            # Use explicit shape if provided, otherwise use computed shape
            if shape is not None:
                self._shape = shape if isinstance(shape, tuple) else (shape,)
                # Verify total size matches
                expected_size = 1
                for s in self._shape:
                    expected_size *= s
                if len(flat_data) != expected_size:
                    raise ValueError(f"Data length {len(flat_data)} doesn't match shape {self._shape}")
            else:
                self._shape = tuple(computed_shape)

            # Infer dtype from data if not specified
            if dtype is None:
                if _has_complex(flat_data):
                    self._dtype = complex128
                else:
                    self._dtype = float64
            else:
                self._dtype = get_dtype(dtype)

            # Create storage via Rust backend - pass data directly
            # Rust handles complex numbers natively via PyO3
            self._storage = TensorStorage.from_list(
                flat_data,
                list(self._shape),
                self._dtype.name,
                self._device.name
            )
        elif shape is not None:
            self._dtype = dtype or float64
            self._shape = shape if isinstance(shape, tuple) else (shape,)
            self._storage = TensorStorage.zeros(
                list(self._shape),
                self._dtype.name,
                self._device.name
            )
        else:
            raise ValueError("Either data or shape must be provided")

    # ========== Properties ==========

    @property
    def shape(self) -> Tuple[int, ...]:
        """Shape of the state."""
        return self._shape

    @property
    def ndim(self) -> int:
        """Number of dimensions."""
        return len(self._shape)

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
        Syntony index S(Psi) in [0, 1].

        Uses the Rust backend's compute_syntony_basic method which implements:
        S(Psi) = ||H[D[Psi]] - D[Psi]|| / (||D[Psi] - Psi|| + epsilon)
        """
        if self._syntony_cache is None:
            self._syntony_cache = self._storage.compute_syntony_basic()
        return self._syntony_cache

    @property
    def gnosis(self) -> int:
        """
        Gnosis layer (0-3).

        Computed from total variation sum Sigma Tv thresholds.
        """
        if self._gnosis_cache is None:
            tv = self._storage.compute_tv_sum()
            if tv < 0.1:
                self._gnosis_cache = 0
            elif tv < 0.5:
                self._gnosis_cache = 1
            elif tv < 1.0:
                self._gnosis_cache = 2
            else:
                self._gnosis_cache = 3
        return self._gnosis_cache

    @property
    def free_energy(self) -> float:
        """Free energy F[rho] measuring deviation from Golden Measure."""
        return self._storage.free_energy()

    # ========== Conversion ==========

    def to_list(self) -> List:
        """Convert to Python list (flat)."""
        # Rust returns complex numbers directly via PyO3
        return list(self._storage.to_list())

    def tolist(self) -> Any:
        """Convert to nested Python list matching shape."""
        flat = self.to_list()
        return _unflatten(flat, list(self._shape))

    def numpy(self):
        """
        Convert to NumPy array.

        Note: This requires numpy to be installed. For NumPy-free operation,
        use to_list() or tolist() instead.
        """
        try:
            import numpy as np
            flat = self.to_list()
            if self._dtype.name == 'complex128':
                arr = np.array(flat, dtype=np.complex128)
            elif self._dtype.name == 'float32':
                arr = np.array(flat, dtype=np.float32)  # pragma: no cover
            elif self._dtype.name == 'int64':
                arr = np.array(flat, dtype=np.int64)  # pragma: no cover
            else:
                arr = np.array(flat, dtype=np.float64)
            return arr.reshape(self._shape)
        except ImportError:  # pragma: no cover
            raise ImportError("NumPy not installed. Use to_list() or tolist() for NumPy-free operation.")

    def torch(self):
        """Convert to PyTorch tensor."""
        try:
            import torch
            flat = self.to_list()
            if self._dtype.name == 'complex128':
                t = torch.tensor(flat, dtype=torch.complex128)  # pragma: no cover
            elif self._dtype.name == 'float32':
                t = torch.tensor(flat, dtype=torch.float32)  # pragma: no cover
            elif self._dtype.name == 'int64':
                t = torch.tensor(flat, dtype=torch.int64)  # pragma: no cover
            else:
                t = torch.tensor(flat, dtype=torch.float64)
            return t.reshape(self._shape)
        except ImportError:  # pragma: no cover
            raise ImportError("PyTorch not installed")

    def cuda(self, device_id: int = 0) -> 'State':
        """Move to CUDA device."""
        if not cuda_is_available():
            raise RuntimeError("CUDA not available")
        if self._device.is_cuda and self._device.index == device_id:
            return self
        new_storage = self._storage.to_device(f'cuda:{device_id}')
        new_state = object.__new__(State)
        new_state._storage = new_storage
        new_state._dtype = self._dtype
        new_state._shape = self._shape
        new_state._device = Device('cuda', device_id)
        new_state._syntony_cache = None
        new_state._gnosis_cache = None
        return new_state

    def cuda_async(self, device_id: int = 0, wait: bool = True) -> 'State':
        """Move to CUDA using SRT-optimized async transfer.

        When `wait` is True (default), waits for transfer completion.
        """
        if not cuda_is_available():
            raise RuntimeError("CUDA not available")
        if self._device.is_cuda and self._device.index == device_id:
            return self

        # Use SRT-optimized async transfer path
        new_storage = self._storage.to_cuda_async_srt(device_id)

        # Optionally ensure device sync (already synced in Rust path)
        if wait:
            from syntonic.core import TensorStorage
            TensorStorage.sync_cuda_device(device_id)

        new_state = object.__new__(State)
        new_state._storage = new_storage
        new_state._dtype = self._dtype
        new_state._shape = self._shape
        new_state._device = Device('cuda', device_id)
        new_state._syntony_cache = None
        new_state._gnosis_cache = None
        return new_state

    def cpu_async(self, wait: bool = True) -> 'State':
        """Move to CPU using SRT-optimized async transfer.

        When `wait` is True (default), waits for transfer completion.
        """
        if self._device.is_cpu:
            return self

        # Use SRT-optimized async D2H transfer path
        # Assume device_id is 0 if not specified on device
        device_id = self._device.index if self._device.index is not None else 0
        new_storage = self._storage.to_cpu_async_srt(device_id)

        # Optionally ensure device sync (already synced in Rust path)
        if wait:
            from syntonic.core import TensorStorage
            TensorStorage.sync_cuda_device(device_id)

        new_state = object.__new__(State)
        new_state._storage = new_storage
        new_state._dtype = self._dtype
        new_state._shape = self._shape
        new_state._device = cpu
        new_state._syntony_cache = None
        new_state._gnosis_cache = None
        return new_state

    def cpu(self) -> 'State':
        """Move to CPU."""
        if self._device.is_cpu:
            return self
        new_storage = self._storage.to_device('cpu')
        new_state = object.__new__(State)
        new_state._storage = new_storage
        new_state._dtype = self._dtype
        new_state._shape = self._shape
        new_state._device = cpu
        new_state._syntony_cache = None
        new_state._gnosis_cache = None
        return new_state

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
        # Compute result shape for matmul
        if len(self._shape) == 2 and len(other._shape) == 2:
            new_shape = (self._shape[0], other._shape[1])
        else:
            new_shape = tuple(new_storage.shape)  # pragma: no cover
        return self._with_storage(new_storage, shape=new_shape)

    def __pow__(self, n: int) -> 'State':
        """Element-wise power."""
        flat = self.to_list()
        if self._dtype.name == 'complex128':
            powered = [x ** n for x in flat]
        else:
            powered = [x ** n for x in flat]
        return State(powered, dtype=self._dtype, device=self._device, shape=self._shape)

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
        from syntonic.exceptions import SyntonicError
        n = self.norm()
        if n == 0:
            raise SyntonicError("Cannot normalize zero state")
        return self / n

    def sum(self, axis: Optional[int] = None, out=None, **kwargs) -> Union[float, 'State']:
        """Sum elements."""
        flat = self.to_list()
        if axis is None:
            if self._dtype.name == 'complex128':
                return sum(flat)
            return float(sum(flat))
        # Sum along axis - implemented in pure Python
        return self._reduce_axis(axis, lambda vals: sum(vals))

    def mean(self, axis: Optional[int] = None) -> Union[float, 'State']:
        """Mean of elements."""
        flat = self.to_list()
        if axis is None:
            if self._dtype.name == 'complex128':
                return sum(flat) / len(flat)
            return float(sum(flat)) / len(flat)
        return self._reduce_axis(axis, lambda vals: sum(vals) / len(vals))

    def max(self, axis: Optional[int] = None) -> Union[float, 'State']:
        """Maximum element."""
        flat = self.to_list()
        if axis is None:
            if self._dtype.name == 'complex128':
                # For complex, max by magnitude
                return max(flat, key=lambda x: abs(x))
            return float(max(flat))
        return self._reduce_axis(axis, lambda vals: max(vals))

    def min(self, axis: Optional[int] = None) -> Union[float, 'State']:
        """Minimum element."""
        flat = self.to_list()
        if axis is None:
            if self._dtype.name == 'complex128':
                # For complex, min by magnitude
                return min(flat, key=lambda x: abs(x))
            return float(min(flat))
        return self._reduce_axis(axis, lambda vals: min(vals))

    def _reduce_axis(self, axis: int, reducer) -> 'State':
        """Apply reduction along an axis."""
        nested = self.tolist()
        if axis < 0:
            axis = len(self._shape) + axis

        # Build new shape
        new_shape = list(self._shape)
        new_shape.pop(axis)

        # Perform reduction
        result = self._reduce_recursive(nested, axis, 0, reducer)
        flat, _ = _flatten(result)

        return State(flat, dtype=self._dtype, device=self._device, shape=tuple(new_shape) if new_shape else (1,))

    def _reduce_recursive(self, data, axis, current_depth, reducer):
        """Recursively reduce along axis."""
        if current_depth == axis:
            # This is the axis to reduce
            if isinstance(data[0], list):
                # Reduce nested lists element-wise
                num_elements = len(data[0])
                result = []
                for i in range(num_elements):
                    column = [row[i] if isinstance(row, list) else row for row in data]
                    if isinstance(column[0], list):
                        result.append(self._reduce_recursive(column, 0, 0, reducer))  # pragma: no cover
                    else:
                        result.append(reducer(column))
                return result
            else:
                return reducer(data)
        else:
            # Recurse deeper
            return [self._reduce_recursive(item, axis, current_depth + 1, reducer) for item in data]

    def abs(self) -> 'State':
        """Element-wise absolute value."""
        return self._with_storage(self._storage.abs())

    def exp(self) -> 'State':
        """Element-wise exponential."""
        return self._with_storage(self._storage.exp())

    def log(self) -> 'State':
        """Element-wise natural logarithm."""
        return self._with_storage(self._storage.log())

    def sin(self) -> 'State':
        """Element-wise sine."""
        return self._with_storage(self._storage.sin())

    def cos(self) -> 'State':
        """Element-wise cosine."""
        return self._with_storage(self._storage.cos())

    def sqrt(self) -> 'State':
        """Element-wise square root."""
        return self._with_storage(self._storage.sqrt())

    def tanh(self) -> 'State':
        """Element-wise hyperbolic tangent."""
        return self._with_storage(self._storage.tanh())

    def sigmoid(self) -> 'State':
        """Element-wise sigmoid: 1/(1 + exp(-x))."""
        return self._with_storage(self._storage.sigmoid())

    def relu(self) -> 'State':
        """Element-wise ReLU: max(0, x)."""
        return self._with_storage(self._storage.relu())

    def exp_golden(self) -> 'State':
        """Golden exponential: exp(-x/φ).
        
        Used for computing golden measure weights w(n) = exp(-|n|²/φ).
        This is the unique optimal measure from SRT Axiom 4.
        """
        return self._with_storage(self._storage.exp_golden())

    def layer_norm(
        self,
        weight: Optional['State'] = None,
        bias: Optional['State'] = None,
        eps: float = 1e-5,
        golden_target: bool = True,
    ) -> 'State':
        """Layer normalization with optional golden target variance.
        
        Args:
            weight: Optional scale parameter (gamma)
            bias: Optional shift parameter (beta)
            eps: Numerical stability epsilon
            golden_target: If True, normalize to variance = 1/φ ≈ 0.618
                          (SRT syntonic equilibrium)
        
        Returns:
            Normalized state
        """
        w_storage = weight._storage if weight is not None else None
        b_storage = bias._storage if bias is not None else None
        return self._with_storage(
            self._storage.layer_norm(w_storage, b_storage, eps, golden_target)
        )

    def dropout(
        self,
        p: float = 0.5,
        training: bool = True,
        seed: Optional[int] = None,
    ) -> 'State':
        """Apply dropout regularization.
        
        Uses inverted dropout: active units scaled by 1/(1-p).
        At inference (training=False), returns identity.
        
        Args:
            p: Dropout probability (0 ≤ p < 1)
            training: Whether in training mode
            seed: Optional RNG seed for reproducibility
        
        Returns:
            State with dropout applied
        """
        return self._with_storage(self._storage.dropout(p, training, seed))

    # ========== Complex Operations ==========

    def conj(self) -> 'State':
        """Complex conjugate."""
        return self._with_storage(self._storage.conj())

    def real(self) -> 'State':
        """Real part."""
        if self._dtype.name != 'complex128':
            return self
        flat = self.to_list()
        real_flat = [x.real for x in flat]
        return State(real_flat, dtype=float64, device=self._device, shape=self._shape)

    def imag(self) -> 'State':
        """Imaginary part."""
        if self._dtype.name != 'complex128':
            flat = [0.0] * self.size
            return State(flat, dtype=float64, device=self._device, shape=self._shape)
        flat = self.to_list()
        imag_flat = [x.imag for x in flat]
        return State(imag_flat, dtype=float64, device=self._device, shape=self._shape)

    @property
    def T(self) -> 'State':
        """Transpose."""
        new_storage = self._storage.transpose()
        new_shape = tuple(reversed(self._shape))
        return self._with_storage(new_storage, shape=new_shape)

    @property
    def H(self) -> 'State':
        """Conjugate transpose (Hermitian adjoint)."""
        return self.conj().T

    # ========== Shape Operations ==========

    def reshape(self, *shape) -> 'State':
        """Reshape state."""
        if len(shape) == 1 and isinstance(shape[0], (list, tuple)):
            shape = tuple(shape[0])

        # Handle -1 dimension
        new_shape = list(shape)
        neg_idx = None
        known_size = 1
        for i, s in enumerate(new_shape):
            if s == -1:
                if neg_idx is not None:
                    raise ValueError("Only one dimension can be -1")
                neg_idx = i
            else:
                known_size *= s

        if neg_idx is not None:
            new_shape[neg_idx] = self.size // known_size

        # Verify total size matches
        total = 1
        for s in new_shape:
            total *= s
        if total != self.size:
            raise ValueError(f"Cannot reshape size {self.size} to shape {tuple(new_shape)}")

        # Create new state with same flat data but different shape
        flat = self.to_list()
        return State(flat, dtype=self._dtype, device=self._device, shape=tuple(new_shape))

    def flatten(self) -> 'State':
        """Flatten to 1D."""
        return self.reshape(-1)

    def squeeze(self) -> 'State':
        """Remove dimensions of size 1."""
        new_shape = tuple(s for s in self._shape if s != 1)
        if not new_shape:
            new_shape = (1,)
        return self.reshape(new_shape)

    def unsqueeze(self, dim: int) -> 'State':
        """Add dimension of size 1."""
        new_shape = list(self._shape)
        if dim < 0:
            dim = len(new_shape) + dim + 1
        new_shape.insert(dim, 1)
        return self.reshape(tuple(new_shape))

    # ========== DHSR Operations ==========

    def differentiate(self, alpha: float = 0.1) -> 'State':
        """
        Apply differentiation operator D-hat.

        D-hat[Psi] increases complexity with syntony-dependent coupling.
        alpha(S) = alpha_0 * (1 - S)

        Args:
            alpha: Base differentiation strength (default: 0.1)

        Returns:
            Differentiated state
        """
        new_storage = self._storage.differentiate(alpha)
        return self._with_storage(new_storage)

    def harmonize(self, strength: float = 0.618, gamma: float = 0.0) -> 'State':
        """
        Apply harmonization operator H-hat.

        Projects toward Golden Measure equilibrium.
        H-hat[Psi] = (1 - gamma)*Psi + gamma*target

        Args:
            strength: Harmonization strength (default: 1/phi = 0.618)
            gamma: Additional gamma parameter (default: 0.0)

        Returns:
            Harmonized state
        """
        new_storage = self._storage.harmonize(strength, gamma)
        return self._with_storage(new_storage)

    def recurse(self, alpha: float = 0.1, strength: float = 0.618) -> 'State':
        """
        Apply recursion operator R-hat = H-hat compose D-hat.

        Performs one complete DHSR cycle.

        Args:
            alpha: Differentiation strength
            strength: Harmonization strength

        Returns:
            Recursed state
        """
        return self.differentiate(alpha).harmonize(strength)

    # ========== Indexing ==========

    def __getitem__(self, key) -> 'State':
        nested = self.tolist()
        result = nested[key]
        if isinstance(result, list):
            flat, shape = _flatten(result)
            return State(flat, dtype=self._dtype, device=self._device, shape=tuple(shape))
        else:
            # Single element
            return State([result], dtype=self._dtype, device=self._device, shape=(1,))

    def __setitem__(self, key, value):
        nested = self.tolist()
        if isinstance(value, State):
            nested[key] = value.tolist()
        else:
            nested[key] = value
        flat, _ = _flatten(nested)

        # Pass data directly to Rust - it handles all types
        self._storage = TensorStorage.from_list(
            flat,
            list(self._shape),
            self._dtype.name,
            self._device.name
        )
        self._invalidate_caches()

    # ========== Representation ==========

    def __repr__(self) -> str:
        return f"State(shape={self.shape}, dtype={self._dtype.name}, device={self._device})"

    def __str__(self) -> str:
        flat = self.to_list()
        if len(flat) <= 10:
            content = str(flat)
        else:
            content = f"[{flat[0]}, {flat[1]}, ..., {flat[-2]}, {flat[-1]}]"
        return f"State({content})"

    def __len__(self) -> int:
        return self.shape[0] if self.shape else 0

    # ========== Array Protocol ==========

    def __array__(self, dtype=None):
        """Support conversion to numpy array."""
        return self.numpy() if dtype is None else self.numpy().astype(dtype)

    # ========== Private Methods ==========

    def _with_storage(
        self,
        storage: TensorStorage,
        device: Optional[Device] = None,
        shape: Optional[Tuple[int, ...]] = None,
    ) -> 'State':
        """Create new State with given storage."""
        new_state = object.__new__(State)
        new_state._storage = storage
        new_state._dtype = self._dtype
        new_state._shape = shape if shape is not None else self._shape
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
    def from_list(cls, data: List, shape: Tuple[int, ...], **kwargs) -> 'State':
        """Create State from flat list with explicit shape."""
        state = cls.__new__(cls)
        state._shape = shape
        state._device = kwargs.get('device', cpu)
        state._dtype = kwargs.get('dtype', float64)
        state._syntony_cache = None
        state._gnosis_cache = None

        # Pass data directly to Rust - it handles all types
        state._storage = TensorStorage.from_list(
            data,
            list(shape),
            state._dtype.name,
            state._device.name
        )
        return state

    @classmethod
    def from_numpy(cls, arr, **kwargs) -> 'State':
        """Create State from NumPy array."""
        try:
            import numpy as np
            flat = arr.flatten().tolist()
            shape = arr.shape

            # Infer dtype
            if np.issubdtype(arr.dtype, np.complexfloating):
                dtype = kwargs.get('dtype', complex128)
            else:
                dtype = kwargs.get('dtype', float64)

            return cls(flat, dtype=dtype, shape=shape, **{k: v for k, v in kwargs.items() if k != 'dtype'})
        except ImportError:  # pragma: no cover
            raise ImportError("NumPy not installed")

    @classmethod
    def from_torch(cls, tensor, **kwargs) -> 'State':  # pragma: no cover
        """Create State from PyTorch tensor."""
        flat = tensor.detach().cpu().flatten().tolist()
        shape = tuple(tensor.shape)
        return cls(flat, shape=shape, **kwargs)


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
        size = 1
        for s in shape_tuple:
            size *= s
        flat = [1.0] * size
        return State(flat, dtype=dtype, device=device, shape=shape_tuple)

    @staticmethod
    def random(
        shape: ShapeLike,
        *,
        dtype: DType = float64,
        device: Device = cpu,
        seed: Optional[int] = None,
    ) -> State:
        """Create random state (uniform [0, 1])."""
        if seed is not None:
            random.seed(seed)
        shape_tuple = shape if isinstance(shape, tuple) else (shape,)
        size = 1
        for s in shape_tuple:
            size *= s

        if dtype.is_complex:
            flat = [complex(random.random(), random.random()) for _ in range(size)]
        else:
            flat = [random.random() for _ in range(size)]
        return State(flat, dtype=dtype, device=device, shape=shape_tuple)

    @staticmethod
    def randn(
        shape: ShapeLike,
        *,
        dtype: DType = float64,
        device: Device = cpu,
        seed: Optional[int] = None,
    ) -> State:
        """Create random state (standard normal)."""
        if seed is not None:
            random.seed(seed)
        shape_tuple = shape if isinstance(shape, tuple) else (shape,)
        size = 1
        for s in shape_tuple:
            size *= s

        def box_muller():
            """Generate standard normal using Box-Muller transform."""
            u1 = random.random()
            u2 = random.random()
            z = math.sqrt(-2.0 * math.log(u1)) * math.cos(2.0 * math.pi * u2)
            return z

        if dtype.is_complex:
            flat = [complex(box_muller(), box_muller()) / math.sqrt(2) for _ in range(size)]
        else:
            flat = [box_muller() for _ in range(size)]
        return State(flat, dtype=dtype, device=device, shape=shape_tuple)

    @staticmethod
    def eye(n: int, *, dtype: DType = float64, device: Device = cpu) -> State:
        """Create identity matrix state."""
        flat = [0.0] * (n * n)
        for i in range(n):
            flat[i * n + i] = 1.0
        return State(flat, dtype=dtype, device=device, shape=(n, n))

    @staticmethod
    def from_list(data: List, shape: Tuple[int, ...], **kwargs) -> State:
        """Create from flat list with explicit shape."""
        return State.from_list(data, shape, **kwargs)

    @staticmethod
    def from_numpy(arr, **kwargs) -> State:
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
state.from_list = StateNamespace.from_list
state.from_numpy = StateNamespace.from_numpy
state.from_torch = StateNamespace.from_torch
