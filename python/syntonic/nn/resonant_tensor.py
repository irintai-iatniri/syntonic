"""
ResonantTensor Python Wrapper

This module provides a clean Python interface around the Rust-backed ResonantTensor
with full type annotations, comprehensive docstrings, and Pythonic operators.

The ResonantTensor is the core data structure of the Resonant Engine, maintaining
dual representations:
- Exact Q(φ) lattice (mathematical purity)
- Ephemeral flux values (CUDA-accelerated differentiation)
"""

from __future__ import annotations
from typing import List, Optional, Union, Any
from syntonic._core import ResonantTensor as _RustResonantTensor, GoldenExact


class ResonantTensor:
    """
    A tensor that exists in dual representations: exact Q(φ) lattice and ephemeral flux.

    The ResonantTensor maintains mathematical purity through exact golden ratio arithmetic
    while supporting CUDA-accelerated differentiation and harmonization cycles.

    Attributes:
        syntony: Current syntony value S ∈ [0, 1]
        phase: Current phase ("crystallized" or "flux")
        shape: Tensor shape as list of dimensions
        shape: Tensor shape as list of dimensions
        precision: Lattice precision for crystallization
        device: Device location ('cpu' or 'cuda:N')

    Examples:
        >>> # Create from floats with default mode norms
        >>> data = [1.0, 2.0, 3.0, 4.0]
        >>> tensor = ResonantTensor(data, shape=[2, 2])
        >>> print(tensor)
        ResonantTensor(shape=[2, 2], phase=crystallized, syntony=0.8234, precision=100)

        >>> # Run a DHSR cycle
        >>> new_syntony = tensor.cpu_cycle(noise_scale=0.1, precision=100)
        >>> print(f"New syntony: {new_syntony:.4f}")

        >>> # Use Pythonic operators
        >>> a = ResonantTensor.randn([3, 3])
        >>> b = ResonantTensor.randn([3, 3])
        >>> c = a + b  # Element-wise addition
        >>> d = a * 2.0  # Scalar multiplication
        >>> e = a @ b  # Matrix multiplication
    """

    def __init__(
        self,
        data: List[float],
        shape: List[int],
        mode_norm_sq: Optional[List[float]] = None,
        precision: int = 100,
        device: str = 'cpu'
    ):
        """
        Create a ResonantTensor from floating-point data.

        Args:
            data: Flattened tensor values
            shape: Shape of the tensor (e.g., [batch, features])
            mode_norm_sq: Mode norms |n|² for each element (defaults to [i² for i in range(size)])
            precision: Maximum coefficient for golden lattice snapping (default: 100)

        Examples:
            >>> # Basic tensor
            >>> tensor = ResonantTensor([1.0, 2.0, 3.0, 4.0], shape=[2, 2])

            >>> # With custom mode norms
            >>> norms = [0.0, 1.0, 4.0, 9.0]  # Custom mode structure
            >>> tensor = ResonantTensor([1.0, 2.0, 3.0, 4.0], shape=[2, 2], mode_norm_sq=norms)
        """
        if device.startswith('cuda'):
            # Parse device index
            if ':' in device:
                idx = int(device.split(':')[1])
            else:
                idx = 0
                
            # Create on CPU first then move
            # TODO: Direct GPU creation if supported by Rust
            self._inner = _RustResonantTensor(data, shape, mode_norm_sq, precision)
            self._inner = self._inner.to_device(idx)
        else:
            self._inner = _RustResonantTensor(data, shape, mode_norm_sq, precision)
            
        self._device_str = device

    # =========================================================================
    # Properties
    # =========================================================================

    @property
    def syntony(self) -> float:
        """Get the current syntony value S ∈ [0, 1]."""
        return self._inner.syntony

    @property
    def phase(self) -> str:
        """Get the current phase ("crystallized" or "flux")."""
        return self._inner.phase

    @property
    def shape(self) -> List[int]:
        """Get the tensor shape."""
        return self._inner.shape

    @property
    def precision(self) -> int:
        """Get the precision used for last crystallization."""
        return self._inner.precision

    @property
    def device(self) -> str:
        """Get the device string (e.g. 'cpu', 'cuda:0')."""
        return self._device_str

    # =========================================================================
    # Factory Methods
    # =========================================================================

    @classmethod
    def from_floats_32(
        cls,
        data: List[float],
        shape: List[int],
        precision: int = 100
    ) -> "ResonantTensor":
        """
        Create from float32 values (alias for constructor).

        Args:
            data: Flattened tensor values
            shape: Tensor shape
            precision: Lattice precision

        Returns:
            New ResonantTensor
        """
        return cls(data, shape, precision=precision, device=device)

    @classmethod
    def from_golden_exact(
        cls,
        lattice: List[GoldenExact],
        shape: List[int],
        mode_norm_sq: Optional[List[float]] = None
    ) -> "ResonantTensor":
        """
        Create from exact Q(φ) lattice values.

        Args:
            lattice: List of GoldenExact values (a + b·φ)
            shape: Tensor shape
            mode_norm_sq: Optional mode norms

        Returns:
            New ResonantTensor in crystallized phase

        Examples:
            >>> from syntonic._core import GoldenExact
            >>> lattice = [GoldenExact.from_integers(1, 0), GoldenExact.from_integers(0, 1)]
            >>> tensor = ResonantTensor.from_golden_exact(lattice, shape=[2])
        """
        instance = cls.__new__(cls)
        instance._inner = _RustResonantTensor.from_golden_exact(lattice, shape, mode_norm_sq)
        instance._device_str = 'cpu' # Default to cpu for lattice creation
        return instance

    @classmethod
    def zeros(cls, shape: List[int], precision: int = 100, device: str = 'cpu') -> "ResonantTensor":
        """
        Create a zero-initialized tensor.

        Args:
            shape: Tensor shape
            precision: Lattice precision

        Returns:
            New tensor filled with zeros (using exact GoldenExact::zero)

        Examples:
            >>> zeros = ResonantTensor.zeros([3, 3])
            >>> assert all(v == 0.0 for v in zeros.to_floats())
        """
        instance = cls.__new__(cls)
        instance._inner = _RustResonantTensor.zeros(shape, precision)
        if device != 'cpu':
            instance._inner = instance._inner.to_device(device)
        instance._device_str = device
        return instance

    @classmethod
    def ones(cls, shape: List[int], precision: int = 100, device: str = 'cpu') -> "ResonantTensor":
        """
        Create a ones-initialized tensor.

        Args:
            shape: Tensor shape
            precision: Lattice precision

        Returns:
            New tensor filled with ones

        Examples:
            >>> ones = ResonantTensor.ones([2, 2])
            >>> assert all(abs(v - 1.0) < 0.01 for v in ones.to_floats())
        """
        size = 1
        for dim in shape:
            size *= dim
        return cls([1.0] * size, shape, precision=precision, device=device)

    @classmethod
    def randn(
        cls,
        shape: List[int],
        mean: float = 0.0,
        std: float = 1.0,
        precision: int = 100
    ) -> "ResonantTensor":
        """
        Create a tensor with random Gaussian values.

        Args:
            shape: Tensor shape
            mean: Mean of Gaussian distribution
            std: Standard deviation
            precision: Lattice precision

        Returns:
            New tensor with random values

        Examples:
            >>> # Standard normal
            >>> tensor = ResonantTensor.randn([100, 100])

            >>> # Custom distribution
            >>> tensor = ResonantTensor.randn([10, 10], mean=5.0, std=2.0)
        """
        import random
        size = 1
        for dim in shape:
            size *= dim
        data = [random.gauss(mean, std) for _ in range(size)]
        return cls(data, shape, precision=precision, device=device)

    # =========================================================================
    # Conversion Methods
    # =========================================================================

    def to_floats(self) -> List[float]:
        """
        Convert to list of floats (approximate representation).

        Returns:
            List of float values

        Examples:
            >>> tensor = ResonantTensor([1.0, 2.0, 3.0], [3])
            >>> floats = tensor.to_floats()
        """
        return self._inner.to_floats()

    def to_list(self) -> List[float]:
        """
        Alias for to_floats().

        Returns:
            List of float values
        """
        return self._inner.to_list()

    def to_lattice(self) -> List[GoldenExact]:
        """
        Get exact Q(φ) lattice values.

        Returns:
            List of GoldenExact values (a + b·φ)

        Examples:
            >>> tensor = ResonantTensor([1.618, 2.0], [2])
            >>> lattice = tensor.to_lattice()
            >>> for g in lattice:
            ...     print(f"{g}")  # Shows exact representation
        """
        return self._inner.get_lattice()

    def get_mode_norms(self) -> List[float]:
        """
        Get mode norm squared values.

        Returns:
            List of |n|² values for each element
        """
        return self._inner.get_mode_norm_sq()

    def to(self, device: str) -> "ResonantTensor":
        """
        Move tensor to device.
        
        Args:
            device: Target device ('cpu', 'cuda:0', etc.)
            
        Returns:
            New tensor on target device (or self if already there)
        """
        # Optimized: check if already on device
        if self.device == device:
            return self

        # Call Rust backend
        if device == 'cpu':
            new_inner = self._inner.to_cpu()
        elif device.startswith('cuda'):
            if ':' in device:
                idx = int(device.split(':')[1])
            else:
                idx = 0
            new_inner = self._inner.to_device(idx)
        else:
            raise ValueError(f"Unsupported device: {device}")
            
        new_tensor = ResonantTensor._wrap(new_inner)
        new_tensor._device_str = device
        return new_tensor

    def cuda(self, device_id: int = 0) -> "ResonantTensor":
        """Move to CUDA."""
        return self.to(f'cuda:{device_id}')

    def cpu(self) -> "ResonantTensor":
        """Move to CPU."""
        return self.to('cpu')

    # =========================================================================
    # Phase Transitions (DHSR Cycle)
    # =========================================================================

    def wake_flux(self) -> List[float]:
        """
        Enter D-phase: project lattice → flux values.

        Returns:
            Flux values as list of floats

        Examples:
            >>> tensor = ResonantTensor.ones([2, 2])
            >>> flux = tensor.wake_flux()
            >>> assert tensor.phase == "flux"
        """
        return self._inner.wake_flux_values()

    # Internal wrapper
    @classmethod
    def _wrap(cls, inner: _RustResonantTensor, device: str = 'cpu') -> "ResonantTensor":
        """Wrap a Rust tensor without triggering __init__."""
        instance = cls.__new__(cls)
        instance._inner = inner
        instance._device_str = device
        return instance
        """
        Enter H-phase: snap flux → lattice.

        Args:
            values: Flux values to crystallize
            precision: Lattice precision for snapping

        Returns:
            New syntony value after crystallization

        Examples:
            >>> tensor = ResonantTensor.zeros([3])
            >>> flux = tensor.wake_flux()
            >>> # Modify flux somehow
            >>> new_syntony = tensor.crystallize(flux, precision=100)
        """
        return self._inner.crystallize_from_values(values, precision)

    def cpu_cycle(self, noise_scale: float = 0.01, precision: int = 100) -> float:
        """
        Run full D→H cycle in CPU mode.

        This simulates the DHSR (Differentiation-Harmonization-Syntony-Recursion) cycle:
        1. D-phase: Add noise and scale by mode structure
        2. H-phase: Snap back to Q(φ) lattice with attenuation

        Args:
            noise_scale: Scale of stochastic noise in D-phase
            precision: Lattice precision for crystallization

        Returns:
            New syntony value after cycle

        Examples:
            >>> tensor = ResonantTensor.randn([10, 10])
            >>> for _ in range(100):
            ...     syntony = tensor.cpu_cycle(noise_scale=0.1)
            >>> print(f"Final syntony: {syntony:.4f}")
        """
        return self._inner.cpu_cycle(noise_scale, precision)

    def batch_cpu_cycle(self, noise_scale: float = 0.01, precision: int = 100) -> List[float]:
        """
        Run batched D→H cycle (for batch dimension).

        Assumes first dimension is batch, applies cycle to each sample independently.

        Args:
            noise_scale: Noise scale
            precision: Lattice precision

        Returns:
            List of syntony values, one per batch sample

        Examples:
            >>> batch_tensor = ResonantTensor.randn([8, 16])  # Batch of 8
            >>> syntonies = batch_tensor.batch_cpu_cycle(noise_scale=0.05)
            >>> assert len(syntonies) == 8
        """
        return self._inner.batch_cpu_cycle(noise_scale, precision)

    # =========================================================================
    # Linear Algebra
    # =========================================================================

    def matmul(self, weights: "ResonantTensor") -> "ResonantTensor":
        """
        Matrix multiplication: self @ weights.

        Performs Y = X @ W^T where self is X and weights is W.
        All arithmetic is exact in Q(φ).

        Args:
            weights: Weight tensor (out_features, in_features)

        Returns:
            New tensor with result

        Examples:
            >>> x = ResonantTensor.randn([4, 10])  # Batch of 4, 10 features
            >>> w = ResonantTensor.randn([20, 10])  # 10 → 20
            >>> y = x.matmul(w)  # [4, 20]
        """
        result = self._inner.matmul(weights._inner)
        return ResonantTensor._wrap(result, device=self.device)

    def add_bias(self, bias: "ResonantTensor") -> None:
        """
        Add bias in-place.

        Args:
            bias: Bias tensor (must match output dimension)

        Examples:
            >>> x = ResonantTensor.randn([4, 10])
            >>> bias = ResonantTensor.randn([10])
            >>> x.add_bias(bias)  # In-place
        """
        self._inner.add_bias(bias._inner)

    # =========================================================================
    # Activations
    # =========================================================================

    def relu(self) -> None:
        """
        Apply ReLU activation in-place.

        Snaps all negative lattice values to zero.

        Examples:
            >>> x = ResonantTensor([-1.0, 2.0, -3.0, 4.0], [4])
            >>> x.relu()
            >>> assert x.to_floats()[0] == 0.0
            >>> assert x.to_floats()[1] > 0.0
        """
        self._inner.relu()

    def sigmoid(self, precision: int = 100) -> None:
        """
        Apply sigmoid activation in-place: σ(x) = 1 / (1 + e^(-x)).

        Args:
            precision: Lattice precision for snapping result

        Examples:
            >>> x = ResonantTensor([0.0, 1.0, -1.0], [3])
            >>> x.sigmoid()
            >>> floats = x.to_floats()
            >>> assert 0.4 < floats[0] < 0.6  # sigmoid(0) ≈ 0.5
        """
        self._inner.sigmoid(precision)

    def tanh(self, precision: int = 100) -> None:
        """
        Apply tanh activation in-place.

        Args:
            precision: Lattice precision

        Examples:
            >>> x = ResonantTensor([0.0, 1.0, -1.0], [3])
            >>> x.tanh()
        """
        self._inner.tanh(precision)

    def gelu(self, precision: int = 100) -> None:
        """
        Apply GELU activation in-place.

        GELU(x) = 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x^3)))

        Args:
            precision: Lattice precision

        Examples:
            >>> x = ResonantTensor.randn([10, 10])
            >>> x.gelu()
        """
        self._inner.gelu(precision)

    def softmax(self, dim: Optional[int] = None, precision: int = 32) -> None:
        """
        Apply softmax along a dimension in-place.

        Uses numerically stable computation: softmax(x) = exp(x - max(x)) / sum(exp(x - max(x)))
        
        Args:
            dim: Dimension to apply softmax along. If None, defaults to -1 (last dimension).
            precision: Lattice precision

        Examples:
            >>> # Classification logits
            >>> logits = ResonantTensor([2.0, 1.0, 0.1], [3])
            >>> logits.softmax()
            >>> probs = logits.to_floats()
            >>> assert abs(sum(probs) - 1.0) < 0.01
        """
        self._inner.softmax(dim, precision)

    def dropout(self, p: float = 0.5) -> None:
        """
        Apply dropout in-place.
        
        Randomly zeroes out elements with probability p.
        Scaling is applied to preserve expected sum.
        
        Args:
            p: Probability of an element to be zeroed.
        """
        self._inner.dropout(p)

    # =========================================================================
    # Arithmetic Operations
    # =========================================================================

    def elementwise_add(self, other: "ResonantTensor") -> "ResonantTensor":
        """
        Element-wise addition: self + other.

        Args:
            other: Tensor with same shape

        Returns:
            New tensor with sum

        Examples:
            >>> a = ResonantTensor([1.0, 2.0], [2])
            >>> b = ResonantTensor([3.0, 4.0], [2])
            >>> c = a.elementwise_add(b)  # [4.0, 6.0]
        """
        result = self._inner.elementwise_add(other._inner)
        return ResonantTensor._wrap(result, device=self.device)

    def elementwise_mul(self, other: "ResonantTensor") -> "ResonantTensor":
        """
        Element-wise multiplication (Hadamard product): self * other.

        Args:
            other: Tensor with same shape

        Returns:
            New tensor with element-wise product

        Examples:
            >>> a = ResonantTensor([2.0, 3.0], [2])
            >>> b = ResonantTensor([4.0, 5.0], [2])
            >>> c = a.elementwise_mul(b)  # [8.0, 15.0]
        """
        result = self._inner.elementwise_mul(other._inner)
        return ResonantTensor._wrap(result, device=self.device)

    def scalar_mul(self, scalar: float) -> "ResonantTensor":
        """
        Multiply by scalar: self * scalar.

        Args:
            scalar: Scalar value

        Returns:
            New tensor with all elements multiplied

        Examples:
            >>> x = ResonantTensor([1.0, 2.0, 3.0], [3])
            >>> y = x.scalar_mul(2.0)  # [2.0, 4.0, 6.0]
        """
        result = self._inner.scalar_mul(scalar)
        return ResonantTensor._wrap(result, device=self.device)

    def scalar_add(self, scalar: float) -> "ResonantTensor":
        """
        Add scalar to all elements: self + scalar.

        Args:
            scalar: Scalar value

        Returns:
            New tensor with scalar added

        Examples:
            >>> x = ResonantTensor([1.0, 2.0, 3.0], [3])
            >>> y = x.scalar_add(10.0)  # [11.0, 12.0, 13.0]
        """
        result = self._inner.scalar_add(scalar)
        return ResonantTensor._wrap(result, device=self.device)

    def negate(self) -> "ResonantTensor":
        """
        Negate: -self.

        Returns:
            New tensor with all elements negated

        Examples:
            >>> x = ResonantTensor([1.0, -2.0, 3.0], [3])
            >>> y = x.negate()  # [-1.0, 2.0, -3.0]
        """
        result = self._inner.negate()
        return ResonantTensor._wrap(result, device=self.device)

    def one_minus(self) -> "ResonantTensor":
        """
        Compute 1 - self.

        Common pattern in gating mechanisms and attention.

        Returns:
            New tensor with 1 - x for each element

        Examples:
            >>> x = ResonantTensor([0.2, 0.5, 0.8], [3])
            >>> y = x.one_minus()  # [0.8, 0.5, 0.2]
        """
        result = self._inner.one_minus()
        return ResonantTensor._wrap(result, device=self.device)

    # =========================================================================
    # Math Functions
    # =========================================================================

    def log(self, precision: Optional[int] = None) -> "ResonantTensor":
        """
        Natural logarithm: ln(x).

        Args:
            precision: Lattice precision (defaults to tensor's precision)

        Returns:
            New tensor with logarithm applied

        Examples:
            >>> x = ResonantTensor([1.0, 2.718, 7.389], [3])
            >>> y = x.log()  # [0.0, 1.0, 2.0]
        """
        result = self._inner.log(precision)
        return ResonantTensor._wrap(result, device=self.device)

    def exp(self, precision: Optional[int] = None) -> "ResonantTensor":
        """
        Natural exponential: e^x.

        Args:
            precision: Lattice precision

        Returns:
            New tensor with exponential applied

        Examples:
            >>> x = ResonantTensor([0.0, 1.0, 2.0], [3])
            >>> y = x.exp()  # [1.0, 2.718, 7.389]
        """
        result = self._inner.exp(precision)
        return ResonantTensor._wrap(result, device=self.device)

    # =========================================================================
    # Advanced Operations
    # =========================================================================

    @staticmethod
    def concat(tensors: List["ResonantTensor"], dim: int = -1) -> "ResonantTensor":
        """
        Concatenate tensors along a dimension.

        Args:
            tensors: List of tensors to concatenate
            dim: Dimension to concatenate along (supports negative indexing)

        Returns:
            New concatenated tensor

        Examples:
            >>> a = ResonantTensor([1.0, 2.0], [2])
            >>> b = ResonantTensor([3.0, 4.0], [2])
            >>> c = ResonantTensor.concat([a, b], dim=0)  # Shape: [4]
        """
        if not tensors:
            raise ValueError("concat expects at least one tensor")
            
        ndim = len(tensors[0].shape)
        if dim < 0:
            dim += ndim

        # PyO3 requires Python context for static methods
        # This is a workaround to get the Python module
        import sys
        if 'syntonic._core' not in sys.modules:
            import syntonic._core

        # Get the module that contains ResonantTensor
        core_module = sys.modules['syntonic._core']

        # Create Py references for PyO3
        from syntonic._core import ResonantTensor as _RT
        inner_list = [t._inner for t in tensors]

        # Call the static method with Python context
        # Fixed: correct arguments (tensors, dim) without module context
        result = _RT.concat(inner_list, dim)
        return ResonantTensor._wrap(result, device=tensors[0].device if tensors else 'cpu')

    def index_select(self, indices: List[int], dim: int = 0) -> "ResonantTensor":
        """
        Select slices along a dimension.

        Args:
            indices: Indices to select
            dim: Dimension to select along

        Returns:
            New tensor with selected slices

        Examples:
            >>> x = ResonantTensor.randn([10, 5])
            >>> selected = x.index_select([0, 2, 4], dim=0)  # Shape: [3, 5]
        """
        ndim = len(self.shape)
        if dim < 0:
            dim += ndim

        try:
            result = self._inner.index_select(indices, dim)
            return ResonantTensor._wrap(result, device=self.device)
        except AttributeError:
            # Fallback for when backend doesn't implement index_select
            import itertools
            
            shape = self.shape
            
            # Calculate strides for source tensor
            strides = [1] * ndim
            for i in range(ndim - 2, -1, -1):
                strides[i] = strides[i+1] * shape[i+1]
                
            # Prepare new shape
            new_shape = list(shape)
            new_shape[dim] = len(indices)
            
            # Get source data
            lattice = self.to_lattice()
            new_lattice = []
            
            # Generate coordinates for new tensor
            # The trick: we want to iterate over the NEW tensor's layout
            # But capture values from the OLD tensor using 'indices' map
            
            # Ranges for iteration: same as shape, but for the selection dim
            # we iterate 0..len(indices). We will use this to look up the real index.
            iter_ranges = [range(s) for s in new_shape]
            
            for coord in itertools.product(*iter_ranges):
                # Map coordinate to source coordinate
                src_coord = list(coord)
                src_coord[dim] = indices[coord[dim]] # Look up real index
                
                # Compute flat index
                flat_idx = sum(c * s for c, s in zip(src_coord, strides))
                new_lattice.append(lattice[flat_idx])
                
            return ResonantTensor.from_golden_exact(new_lattice, new_shape, self.get_mode_norms())

    def view(self, *shape: int) -> "ResonantTensor":
        """
        Returns a new tensor with the same data but different shape.
        
        Args:
            *shape: New shape dimensions
            
        Returns:
            ResonantTensor with new shape
        """
        # Handle list vs varargs
        if len(shape) == 1 and isinstance(shape[0], (list, tuple)):
            new_shape = list(shape[0])
        else:
            new_shape = list(shape)
            
        # Handle -1 (infer dimension)
        if -1 in new_shape:
            total_elements = len(self)
            known_elements = 1
            minus_one_idx = -1
            for i, dim in enumerate(new_shape):
                if dim == -1:
                    if minus_one_idx != -1:
                        raise ValueError("Only one dimension can be -1")
                    minus_one_idx = i
                else:
                    known_elements *= dim
            
            if total_elements % known_elements != 0:
                raise ValueError(f"Shape mismatch: {total_elements} not divisible by {known_elements}")
            
            new_shape[minus_one_idx] = total_elements // known_elements
            
        try:
            # Try Rust implementation if available
            result = self._inner.view(new_shape)
            return ResonantTensor._wrap(result)
        except AttributeError:
            # Python fallback: re-create with new shape
            # Since data is contiguous in C-order, this is just a metadata change
            # for the lattice list
            
            # Validate size
            current_size = len(self)
            new_size = 1
            for dim in new_shape:
                new_size *= dim
                
            if current_size != new_size:
                raise ValueError(f"Shape mismatch: cannot reshape {self.shape} ({current_size}) to {new_shape} ({new_size})")
                
            # Create new tensor using existing lattice data
            # Note: We duplicate data here because we can't easily share the underlying Rust vector
            # passing it back through Python
            lattice = self.to_lattice()
            
            # Simple metadata change - mode norms must be resized or reused?
            # If shape changes, mode norms flat list is still valid for element-wise ops,
            # but might need re-indexing for spectral operations.
            # For now, reuse existing mode norms as they are flat.
            return ResonantTensor.from_golden_exact(lattice, new_shape, self.get_mode_norms())

    def reshape(self, *shape: int) -> "ResonantTensor":
        """Alias for view()."""
        return self.view(*shape)

    def transpose(self, dim0: int, dim1: int) -> "ResonantTensor":
        """
        Returns a tensor that is a transposed version of input.
        The given dimensions are swapped.
        
        Args:
            dim0: First dimension to swap
            dim1: Second dimension to swap
            
        Returns:
            Transposed ResonantTensor
        """
        ndim = len(self.shape)
        if dim0 < 0: dim0 += ndim
        if dim1 < 0: dim1 += ndim
        
        try:
            result = self._inner.transpose(dim0, dim1)
            return ResonantTensor._wrap(result)
        except AttributeError:
            # Python fallback
            import itertools
            
            shape = list(self.shape)
            
            # Swap dimensions in shape
            new_shape = list(shape)
            new_shape[dim0], new_shape[dim1] = new_shape[dim1], new_shape[dim0]
            
            # Calculate strides for source tensor
            strides = [1] * ndim
            for i in range(ndim - 2, -1, -1):
                strides[i] = strides[i+1] * shape[i+1]
                
            # Perform transpose
            lattice = self.to_lattice()
            new_lattice = []
            
            # Iterate over NEW shape
            # To simulate nested loops of variable depth:
            ranges = [range(s) for s in new_shape]
            
            for coord in itertools.product(*ranges):
                # Convert new coord to old coord (swap back)
                old_coord = list(coord)
                old_coord[dim0], old_coord[dim1] = old_coord[dim1], old_coord[dim0]
                
                # Calculate flat index in source
                flat_idx = sum(c * s for c, s in zip(old_coord, strides))
                new_lattice.append(lattice[flat_idx])
                
            return ResonantTensor.from_golden_exact(new_lattice, new_shape, self.get_mode_norms())

    def permute(self, *dims: int) -> "ResonantTensor":
        """
        Returns a view of the original tensor with its dimensions permuted.
        
        Args:
            *dims: The desired ordering of dimensions
            
        Returns:
            Permuted ResonantTensor
        """
        # Handle list vs varargs
        if len(dims) == 1 and isinstance(dims[0], (list, tuple)):
            perm = list(dims[0])
        else:
            perm = list(dims)
            
        if len(perm) != len(self.shape):
            raise ValueError(f"Permutation size {len(perm)} must match tensor dimension {len(self.shape)}")
        
        ndim = len(self.shape)
        # Normalize negative dimensions
        perm = [d + ndim if d < 0 else d for d in perm]
            
        try:
            result = self._inner.permute(perm)
            return ResonantTensor._wrap(result)
        except AttributeError:
            # Python fallback
            import itertools
            
            shape = self.shape
            
            # Validate permutation
            if set(perm) != set(range(ndim)):
                raise ValueError(f"Invalid permutation {perm} for {ndim} dimensions")
                
            new_shape = [shape[i] for i in perm]
            
            # Calculate strides for source tensor
            strides = [1] * ndim
            for i in range(ndim - 2, -1, -1):
                strides[i] = strides[i+1] * shape[i+1]
                
            # Perform permutation
            lattice = self.to_lattice()
            new_lattice = []
            
            ranges = [range(s) for s in new_shape]
            
            for coord in itertools.product(*ranges):
                # coord is in new order. We need to map back to source order.
                # If new[i] comes from old[perm[i]], then:
                # new_coord[i] corresponds to axis perm[i] in source.
                # So source_coord[perm[i]] = new_coord[i]
                
                old_coord = [0] * ndim
                for i, p in enumerate(perm):
                    old_coord[p] = coord[i]
                    
                flat_idx = sum(c * s for c, s in zip(old_coord, strides))
                new_lattice.append(lattice[flat_idx])
                
            return ResonantTensor.from_golden_exact(new_lattice, new_shape, self.get_mode_norms())


    def layer_norm(
        self,
        gamma: Optional["ResonantTensor"] = None,
        beta: Optional["ResonantTensor"] = None,
        eps: float = 1e-8,
        golden_target: bool = True
    ) -> "ResonantTensor":
        """
        Layer normalization across last dimension.

        Args:
            gamma: Optional scale parameter
            beta: Optional shift parameter
            eps: Small constant for numerical stability
            golden_target: If True, scale to target variance = 1/φ

        Returns:
            New normalized tensor

        Examples:
            >>> x = ResonantTensor.randn([8, 16])
            >>> normalized = x.layer_norm()
        """
        gamma_inner = gamma._inner if gamma else None
        beta_inner = beta._inner if beta else None
        result = self._inner.layer_norm(gamma_inner, beta_inner, eps, golden_target)
        return ResonantTensor._wrap(result)

    def mean(
        self,
        dim: Optional[int] = None,
        keepdim: bool = False,
        precision: Optional[int] = None
    ) -> "ResonantTensor":
        """
        Mean reduction along a dimension.

        Args:
            dim: Dimension to reduce (None = global mean)
            keepdim: Keep reduced dimension with size 1
            precision: Lattice precision

        Returns:
            New tensor with mean values

        Examples:
            >>> x = ResonantTensor.randn([4, 10])
            >>> mean_per_sample = x.mean(dim=1)  # Shape: [4]
            >>> global_mean = x.mean()  # Shape: [1]
        """
        if dim is not None:
             ndim = len(self.shape)
             if dim < 0:
                 dim += ndim
        
        result = self._inner.mean(dim, keepdim, precision)
        return ResonantTensor._wrap(result)

    def var(
        self,
        dim: Optional[int] = None,
        keepdim: bool = False,
        precision: Optional[int] = None
    ) -> "ResonantTensor":
        """
        Variance reduction along a dimension (population variance).

        Args:
            dim: Dimension to reduce
            keepdim: Keep reduced dimension
            precision: Lattice precision

        Returns:
            New tensor with variance values

        Examples:
            >>> x = ResonantTensor.randn([4, 10])
            >>> var_per_sample = x.var(dim=1)
        """
        if dim is not None:
             ndim = len(self.shape)
             if dim < 0:
                 dim += ndim
                 
        result = self._inner.var(dim, keepdim, precision)
        return ResonantTensor._wrap(result)

    # =========================================================================
    # Golden Recursion Operations
    # =========================================================================

    def apply_recursion(self) -> None:
        """
        Apply golden recursion map R(n) = floor(φ·n).

        Scales all lattice values by φ (exactly). This is the Fibonacci scaling
        property of the golden lattice: (a, b) → (b, a+b).

        Examples:
            >>> x = ResonantTensor.ones([3])
            >>> x.apply_recursion()  # Values scaled by φ ≈ 1.618
        """
        self._inner.apply_recursion()

    def apply_inverse_recursion(self) -> None:
        """
        Apply inverse golden recursion map R^{-1}(n) = floor(n/φ).

        Scales all lattice values by 1/φ.

        Examples:
            >>> x = ResonantTensor.ones([3])
            >>> x.apply_inverse_recursion()  # Values scaled by 1/φ ≈ 0.618
        """
        self._inner.apply_inverse_recursion()

    def prune_hierarchy(self, q: float, divisor: float = 248.0) -> None:
        """
        Snap values below threshold to zero (hierarchical pruning).

        Threshold is q/divisor. Values with |v| < threshold are set to zero.

        Args:
            q: Base threshold scale
            divisor: Divisor for threshold (default: 248, related to e^π - π ≈ 19.999)

        Examples:
            >>> x = ResonantTensor.randn([100])
            >>> x.prune_hierarchy(q=1.0, divisor=100.0)  # Prune values < 0.01
        """
        self._inner.prune_hierarchy(q, divisor)

    # =========================================================================
    # Operator Overloading
    # =========================================================================

    def __add__(self, other: Union["ResonantTensor", float]) -> "ResonantTensor":
        """
        Addition operator: self + other.

        Supports both tensor-tensor and tensor-scalar addition.

        Examples:
            >>> a = ResonantTensor([1.0, 2.0], [2])
            >>> b = ResonantTensor([3.0, 4.0], [2])
            >>> c = a + b  # Element-wise
            >>> d = a + 10.0  # Scalar
        """
        if isinstance(other, ResonantTensor):
            return self.elementwise_add(other)
        else:
            return self.scalar_add(float(other))

    def __mul__(self, other: Union["ResonantTensor", float]) -> "ResonantTensor":
        """
        Multiplication operator: self * other.

        Supports both tensor-tensor (Hadamard) and tensor-scalar multiplication.

        Examples:
            >>> a = ResonantTensor([2.0, 3.0], [2])
            >>> b = ResonantTensor([4.0, 5.0], [2])
            >>> c = a * b  # Element-wise
            >>> d = a * 2.0  # Scalar
        """
        if isinstance(other, ResonantTensor):
            return self.elementwise_mul(other)
        else:
            return self.scalar_mul(float(other))

    def __rmul__(self, other: float) -> "ResonantTensor":
        """Right multiplication: scalar * self."""
        return self.scalar_mul(float(other))

    def __neg__(self) -> "ResonantTensor":
        """
        Negation operator: -self.

        Examples:
            >>> x = ResonantTensor([1.0, -2.0], [2])
            >>> y = -x  # [-1.0, 2.0]
        """
        return self.negate()

    def __matmul__(self, other: "ResonantTensor") -> "ResonantTensor":
        """
        Matrix multiplication operator: self @ other.

        Examples:
            >>> x = ResonantTensor.randn([4, 10])
            >>> w = ResonantTensor.randn([20, 10])
            >>> y = x @ w  # [4, 20]
        """
        return self.matmul(other)

    def __getitem__(self, key: Union[int, slice, tuple, List[int]]) -> "ResonantTensor":
        """
        Support indexing and slicing.
        
        Examples:
            >>> x = ResonantTensor.randn([4, 4])
            >>> a = x[0]    # Indexing
            >>> b = x[0:2]  # Slicing
            >>> c = x[:, 1] # Complex slicing
        """
        # Normalize key to tuple
        if not isinstance(key, tuple):
            key = (key,)
            
        current_tensor = self
        
        # We need to process dimensions. 
        # Since indexing changes shapes and potentially ranks (if we squeeze),
        # we have to be careful with dimension management.
        # We use a strategy of applying index_select for each slice/index,
        # then squeezing dimensions that were scalar indexed.
        
        reduced_dims = []
        original_shape_len = len(self.shape)
        
        if len(key) > original_shape_len:
             raise IndexError(f"Too many indices for tensor of dimension {original_shape_len}")

        for i, k in enumerate(key):
            current_dim_size = current_tensor.shape[i]
            
            if isinstance(k, int):
                # Handle negative index
                if k < 0:
                    k += current_dim_size
                if k < 0 or k >= current_dim_size:
                    raise IndexError(f"Index {k} out of bounds for dimension {i} with size {current_dim_size}")
                
                # Select single index
                indices = [k]
                current_tensor = current_tensor.index_select(indices, dim=i)
                reduced_dims.append(i)
                
            elif isinstance(k, slice):
                start, stop, step = k.indices(current_dim_size)
                indices = list(range(start, stop, step))
                current_tensor = current_tensor.index_select(indices, dim=i)
            
            elif k is Ellipsis:
                 # Ellipsis support requires more complex logic to map remaining dimensions
                 # For now, simplistic implementation assuming it is at the end if present or 
                 # filling gaps.
                 # Given this is a simple wrapper, we might skip full numpy fancy indexing for now.
                 pass
                 
        # Squeeze out dimensions that were integer-indexed
        # Since index_select preserves rank (returns shape [1, ...]), we can view/reshape
        if reduced_dims:
            final_shape = [d for idx, d in enumerate(current_tensor.shape) if idx not in reduced_dims]
            if not final_shape:
                # Reduced to scalar? ResonantTensor usually wraps a list.
                # Shape []? or [1]?
                # Our Rust backend might expect at least [1].
                # Let's keep [1] if it becomes scalar, effectively a 1-element tensor.
                # Or we can return just the float value?
                # PyTorch x[0] returns 0-d tensor. float(x[0]) gives float.
                pass 
            
            # Use view to squeeze. 
            # Note: Rust backend view checks element count match.
            # Squeezing [1, 5, 1] -> [5] preserves element count.
            if final_shape:
                current_tensor = current_tensor.view(final_shape)
            # If final_shape is empty, it means we selected down to a single element.
            # In that case, we might want to leave it as [1] or support 0-d tensors if backend allows.
            # Currently backend shape is Vec<usize>, so [] is valid 0-d.
            # Let's try to view as [] if possible, or keep as [1] for safety if backed logic assumes dims > 0.
            
        return current_tensor
        
    def __len__(self) -> int:
        """
        Get size of the first dimension.

        Examples:
            >>> x = ResonantTensor.zeros([3, 4])
            >>> assert len(x) == 3
        """
        if not self.shape:
             return 0
        return self.shape[0]

    def __repr__(self) -> str:
        """String representation."""
        return self._inner.__repr__()

    # =========================================================================
    # Internal Helper
    # =========================================================================

    @staticmethod
    def _wrap(inner: _RustResonantTensor) -> "ResonantTensor":
        """
        Wrap a Rust ResonantTensor in Python wrapper.

        Internal method for wrapping Rust results.
        """
        instance = ResonantTensor.__new__(ResonantTensor)
        instance._inner = inner
        return instance