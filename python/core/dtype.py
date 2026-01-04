"""Data type definitions for Syntonic."""

from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Union
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

# Winding type (alias for int64, semantically distinct for T^4 indices)
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


def get_dtype(dtype_spec: Union[DType, str, np.dtype]) -> DType:
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
