"""Core module for Syntonic."""

from syntonic.core.state import State, state
from syntonic.core.dtype import (
    DType,
    float32,
    float64,
    complex64,
    complex128,
    int32,
    int64,
    winding,
    get_dtype,
    promote_dtypes,
)
from syntonic.core.device import (
    Device,
    cpu,
    cuda,
    cuda_is_available,
    cuda_device_count,
    device,
)

__all__ = [
    # State
    'State',
    'state',
    # DTypes
    'DType',
    'float32',
    'float64',
    'complex64',
    'complex128',
    'int32',
    'int64',
    'winding',
    'get_dtype',
    'promote_dtypes',
    # Devices
    'Device',
    'cpu',
    'cuda',
    'cuda_is_available',
    'cuda_device_count',
    'device',
]
