"""
Data type definitions for Syntonic.

Pure Rust-based dtype system without numpy dependencies.
Matches the Rust backend's CpuData/CudaData enum types.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Union, Any


@dataclass(frozen=True)
class DType:
    """
    Syntonic data type - matches Rust backend types.

    This is a pure Python implementation that mirrors the Rust
    CpuData/CudaData enum types without numpy dependencies.
    """

    name: str
    size: int  # bytes
    is_complex: bool = False
    is_floating: bool = True
    rust_type: str = ""  # Corresponding Rust type name

    def __str__(self) -> str:
        return self.name

    def __repr__(self) -> str:
        return f"syn.{self.name}"

    def __eq__(self, other) -> bool:
        if isinstance(other, DType):
            return self.name == other.name
        elif isinstance(other, str):
            return self.name == other
        return False

    def __hash__(self) -> int:
        return hash(self.name)


# Standard floating point types (matching Rust f32/f64)
float32 = DType("float32", 4, rust_type="f32")
float64 = DType("float64", 8, rust_type="f64")  # DEFAULT

# Complex types (matching Rust Complex64)
complex64 = DType("complex64", 8, is_complex=True, rust_type="Complex64")
complex128 = DType(
    "complex128", 16, is_complex=True, rust_type="Complex64"
)  # DEFAULT for complex

# Integer types (matching Rust i64)
int32 = DType("int32", 4, is_floating=False, rust_type="i32")
int64 = DType("int64", 8, is_floating=False, rust_type="i64")

# Winding type (alias for int64, semantically distinct for T^4 indices)
winding = DType("winding", 8, is_floating=False, rust_type="i64")

# Golden exact type (for SRT arithmetic)
golden_exact = DType("golden_exact", 16, rust_type="GoldenExact")

# Type mapping for conversions
_DTYPE_MAP = {
    # Float aliases
    "float32": float32,
    "f32": float32,
    "float64": float64,
    "f64": float64,
    "float": float64,
    # Complex aliases
    "complex64": complex64,
    "c64": complex64,
    "complex128": complex128,
    "c128": complex128,
    "complex": complex128,
    # Integer aliases
    "int32": int32,
    "i32": int32,
    "int64": int64,
    "i64": int64,
    "int": int64,
    # Special types
    "winding": winding,
    "golden_exact": golden_exact,
    "golden": golden_exact,
}


def get_dtype(dtype_spec: Union[DType, str, Any]) -> DType:
    """
    Get DType from various specifications.

    Args:
        dtype_spec: DType instance, string name, or other type spec

    Returns:
        Corresponding DType instance

    Raises:
        ValueError: If dtype_spec is not recognized
    """
    if isinstance(dtype_spec, DType):
        return dtype_spec

    if isinstance(dtype_spec, str):
        dtype_lower = dtype_spec.lower()
        if dtype_lower in _DTYPE_MAP:
            return _DTYPE_MAP[dtype_lower]

        # Try to match partial names
        for key, dtype in _DTYPE_MAP.items():
            if key in dtype_lower or dtype_lower in key:
                return dtype

    # Try to infer from Python types
    if dtype_spec is float:
        return float64
    elif dtype_spec is int:
        return int64
    elif dtype_spec is complex:
        return complex128

    # Try to match by size/type inspection
    if hasattr(dtype_spec, "itemsize"):
        # Looks like a numpy-like dtype
        size = getattr(dtype_spec, "itemsize", 8)
        is_complex = getattr(dtype_spec, "kind", "") == "c"

        if is_complex:
            return complex128 if size >= 16 else complex64
        else:
            return float64 if size >= 8 else float32

    raise ValueError(f"Unknown dtype specification: {dtype_spec}")


def promote_dtypes(dtype1: DType, dtype2: DType) -> DType:
    """
    Determine result dtype from two input dtypes.

    Follows type promotion rules:
    1. Complex takes precedence over real
    2. Higher precision takes precedence over lower precision
    3. Larger size takes precedence over smaller size

    Args:
        dtype1: First dtype
        dtype2: Second dtype

    Returns:
        Promoted dtype
    """
    # Complex takes precedence
    if dtype1.is_complex and not dtype2.is_complex:
        return dtype1
    elif dtype2.is_complex and not dtype1.is_complex:
        return dtype2
    elif dtype1.is_complex and dtype2.is_complex:
        # Both complex - larger size wins
        return dtype1 if dtype1.size >= dtype2.size else dtype2

    # Both real - larger size wins
    if dtype1.size > dtype2.size:
        return dtype1
    elif dtype2.size > dtype1.size:
        return dtype2
    else:
        # Same size - prefer float64 over float32, int64 over int32
        if dtype1.name == "float64":
            return dtype1
        elif dtype2.name == "float64":
            return dtype2
        else:
            return dtype1  # Default to first


def is_compatible_dtype(dtype1: DType, dtype2: DType) -> bool:
    """
    Check if two dtypes are compatible for operations.

    Args:
        dtype1: First dtype
        dtype2: Second dtype

    Returns:
        True if compatible, False otherwise
    """
    # Same dtype is always compatible
    if dtype1 == dtype2:
        return True

    # Complex can operate with real (result will be complex)
    if dtype1.is_complex != dtype2.is_complex:
        return True

    # Different precisions are compatible (will promote)
    return True


def can_cast_dtype(from_dtype: DType, to_dtype: DType) -> bool:
    """
    Check if dtype can be cast to another dtype.

    Args:
        from_dtype: Source dtype
        to_dtype: Target dtype

    Returns:
        True if casting is possible
    """
    # Same dtype
    if from_dtype == to_dtype:
        return True

    # Complex to real (lossy)
    if from_dtype.is_complex and not to_dtype.is_complex:
        return True

    # Real to complex (safe)
    if not from_dtype.is_complex and to_dtype.is_complex:
        return True

    # Different precisions
    if from_dtype.is_floating == to_dtype.is_floating:
        return True

    # Integer to float (safe)
    if not from_dtype.is_floating and to_dtype.is_floating:
        return True

    # Float to integer (lossy)
    if from_dtype.is_floating and not to_dtype.is_floating:
        return True

    return False


def get_default_dtype(is_complex: bool = False) -> DType:
    """
    Get default dtype for given type category.

    Args:
        is_complex: Whether to return complex dtype

    Returns:
        Default dtype
    """
    return complex128 if is_complex else float64


def get_dtype_info(dtype: DType) -> dict:
    """
    Get detailed information about a dtype.

    Args:
        dtype: DType to inspect

    Returns:
        Dictionary with dtype information
    """
    return {
        "name": dtype.name,
        "size": dtype.size,
        "is_complex": dtype.is_complex,
        "is_floating": dtype.is_floating,
        "rust_type": dtype.rust_type,
        "bytes_per_element": dtype.size,
        "complex_elements": 2 if dtype.is_complex else 1,
    }


# Export commonly used dtypes at module level
__all__ = [
    "DType",
    "float32",
    "float64",
    "complex64",
    "complex128",
    "int32",
    "int64",
    "winding",
    "golden_exact",
    "get_dtype",
    "promote_dtypes",
    "is_compatible_dtype",
    "can_cast_dtype",
    "get_default_dtype",
    "get_dtype_info",
]
