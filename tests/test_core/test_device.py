"""Tests for Device management."""

import syntonic as syn
from syntonic.core.device import (
    Device, cpu, cuda, cuda_is_available, cuda_device_count, device
)
import pytest


class TestDeviceBasics:
    """Tests for Device class basics."""

    def test_cpu_device(self):
        assert cpu.type == 'cpu'
        assert cpu.index is None
        assert cpu.name == 'cpu'
        assert cpu.is_cpu
        assert not cpu.is_cuda

    def test_cpu_str(self):
        assert str(cpu) == 'cpu'

    def test_cpu_repr(self):
        assert repr(cpu) == "syn.device('cpu')"

    def test_cuda_device_properties(self):
        cuda_dev = Device('cuda', 0)
        assert cuda_dev.type == 'cuda'
        assert cuda_dev.index == 0
        assert cuda_dev.name == 'cuda:0'
        assert not cuda_dev.is_cpu
        assert cuda_dev.is_cuda

    def test_cuda_device_with_index(self):
        cuda_dev = Device('cuda', 1)
        assert cuda_dev.name == 'cuda:1'

    def test_cuda_device_no_index(self):
        cuda_dev = Device('cuda')
        assert cuda_dev.name == 'cuda:0'  # Defaults to 0

    def test_cuda_str(self):
        cuda_dev = Device('cuda', 0)
        assert str(cuda_dev) == 'cuda:0'

    def test_cuda_repr(self):
        cuda_dev = Device('cuda', 0)
        assert repr(cuda_dev) == "syn.device('cuda:0')"


class TestCudaFunctions:
    """Tests for CUDA utility functions."""

    def test_cuda_is_available(self):
        # CUDA is available on this system
        assert cuda_is_available() == True

    def test_cuda_device_count(self):
        # At least one CUDA device available
        assert cuda_device_count() >= 1

    def test_cuda_function_returns_device(self):
        """Test cuda() returns a valid CUDA device."""
        cuda_dev = cuda(0)
        assert cuda_dev.is_cuda
        assert cuda_dev.index == 0

    def test_cuda_function_with_different_index(self):
        """Test cuda() with device index."""
        cuda_dev = cuda(0)
        assert cuda_dev.name == 'cuda:0'


class TestDeviceParser:
    """Tests for device() parsing function."""

    def test_parse_cpu(self):
        dev = device('cpu')
        assert dev is cpu

    def test_parse_cuda(self):
        """Test parsing cuda device string."""
        dev = device('cuda')
        assert dev.is_cuda
        assert dev.index == 0

    def test_parse_cuda_with_index(self):
        """Test parsing cuda:N device strings."""
        dev = device('cuda:0')
        assert dev.is_cuda
        assert dev.index == 0

    def test_parse_unknown_raises(self):
        with pytest.raises(ValueError, match="Unknown device"):
            device('tpu')

        with pytest.raises(ValueError, match="Unknown device"):
            device('mps')


class TestDeviceInSyntonic:
    """Tests for device usage in syntonic module."""

    def test_syn_cpu(self):
        assert syn.cpu is cpu
        assert syn.cpu.is_cpu

    def test_syn_cuda_is_available(self):
        assert syn.cuda_is_available() == True

    def test_syn_cuda_device_count(self):
        assert syn.cuda_device_count() >= 1

    def test_syn_device_parser(self):
        assert syn.device('cpu') is syn.cpu

    def test_syn_device_cuda(self):
        """Test syn.device with CUDA."""
        dev = syn.device('cuda:0')
        assert dev.is_cuda
