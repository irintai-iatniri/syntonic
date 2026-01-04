"""Tests for DType system."""

import syntonic as syn
from syntonic.core.dtype import (
    DType, float32, float64, complex64, complex128,
    int32, int64, winding, get_dtype, promote_dtypes
)
import numpy as np
import pytest


class TestDTypeBasics:
    """Tests for DType class basics."""

    def test_str(self):
        assert str(float64) == "float64"
        assert str(complex128) == "complex128"

    def test_repr(self):
        assert repr(float64) == "syn.float64"
        assert repr(complex128) == "syn.complex128"

    def test_float32_properties(self):
        assert float32.name == "float32"
        assert float32.size == 4
        assert not float32.is_complex
        assert float32.is_floating

    def test_float64_properties(self):
        assert float64.name == "float64"
        assert float64.size == 8
        assert not float64.is_complex
        assert float64.is_floating

    def test_complex64_properties(self):
        assert complex64.name == "complex64"
        assert complex64.size == 8
        assert complex64.is_complex
        assert complex64.is_floating

    def test_complex128_properties(self):
        assert complex128.name == "complex128"
        assert complex128.size == 16
        assert complex128.is_complex
        assert complex128.is_floating

    def test_int32_properties(self):
        assert int32.name == "int32"
        assert int32.size == 4
        assert not int32.is_complex
        assert not int32.is_floating

    def test_int64_properties(self):
        assert int64.name == "int64"
        assert int64.size == 8
        assert not int64.is_complex
        assert not int64.is_floating

    def test_winding_properties(self):
        assert winding.name == "winding"
        assert winding.size == 8
        assert not winding.is_complex
        assert not winding.is_floating


class TestGetDType:
    """Tests for get_dtype function."""

    def test_get_dtype_from_dtype(self):
        assert get_dtype(float64) is float64
        assert get_dtype(complex128) is complex128

    def test_get_dtype_from_string(self):
        assert get_dtype('float32') is float32
        assert get_dtype('f32') is float32
        assert get_dtype('float64') is float64
        assert get_dtype('f64') is float64
        assert get_dtype('float') is float64
        assert get_dtype('complex64') is complex64
        assert get_dtype('c64') is complex64
        assert get_dtype('complex128') is complex128
        assert get_dtype('c128') is complex128
        assert get_dtype('complex') is complex128
        assert get_dtype('int32') is int32
        assert get_dtype('i32') is int32
        assert get_dtype('int64') is int64
        assert get_dtype('i64') is int64
        assert get_dtype('int') is int64
        assert get_dtype('winding') is winding

    def test_get_dtype_from_numpy_dtype(self):
        assert get_dtype(np.dtype(np.float32)) is float32
        assert get_dtype(np.dtype(np.float64)) is float64
        assert get_dtype(np.dtype(np.complex64)) is complex64
        assert get_dtype(np.dtype(np.complex128)) is complex128
        assert get_dtype(np.dtype(np.int32)) is int32
        assert get_dtype(np.dtype(np.int64)) is int64

    def test_get_dtype_unknown_raises(self):
        with pytest.raises(ValueError, match="Unknown dtype"):
            get_dtype('unknown')
        with pytest.raises(ValueError, match="Unknown dtype"):
            get_dtype(np.dtype(np.float16))


class TestPromoteDTypes:
    """Tests for promote_dtypes function."""

    def test_promote_same_dtype(self):
        assert promote_dtypes(float64, float64) is float64
        assert promote_dtypes(float32, float32) is float32

    def test_promote_float_sizes(self):
        # Larger precision wins
        assert promote_dtypes(float32, float64) is float64
        assert promote_dtypes(float64, float32) is float64

    def test_promote_complex_precedence(self):
        # Complex takes precedence
        assert promote_dtypes(float64, complex128) is complex128
        assert promote_dtypes(complex128, float64) is complex128

    def test_promote_complex64_with_float(self):
        # complex64 with float32 stays complex64
        assert promote_dtypes(complex64, float32) is complex64
        assert promote_dtypes(float32, complex64) is complex64

    def test_promote_complex128_with_float(self):
        # complex128 with any float -> complex128
        assert promote_dtypes(complex128, float32) is complex128
        assert promote_dtypes(float64, complex128) is complex128

    def test_promote_complex_sizes(self):
        # When both complex, larger wins
        assert promote_dtypes(complex64, complex128) is complex128
        assert promote_dtypes(complex128, complex64) is complex128
