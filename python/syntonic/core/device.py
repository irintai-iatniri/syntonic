"""Device management for Syntonic."""

from __future__ import annotations
from dataclasses import dataclass
from typing import Optional

# Import CUDA functions from Rust backend
from syntonic._core import cuda_is_available as _rust_cuda_available
from syntonic._core import cuda_device_count as _rust_cuda_count


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
    return Device('cuda', device_id)  # pragma: no cover


def cuda_is_available() -> bool:
    """Check if CUDA is available via the Rust backend."""
    return _rust_cuda_available()


def cuda_device_count() -> int:
    """Get number of CUDA devices via the Rust backend."""
    return _rust_cuda_count()


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
