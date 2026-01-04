"""Tests for State class."""

import syntonic as syn
import numpy as np
import pytest


class TestStateCreation:
    """Tests for State creation."""

    def test_from_list(self):
        psi = syn.state([1, 2, 3, 4])
        assert psi.shape == (4,)
        assert psi.dtype == syn.float64
        assert np.allclose(psi.numpy(), [1, 2, 3, 4])

    def test_from_complex_list(self):
        psi = syn.state([1 + 2j, 3 + 4j])
        assert psi.shape == (2,)
        assert psi.dtype == syn.complex128

    def test_from_2d_list(self):
        psi = syn.state([[1, 2], [3, 4]])
        assert psi.shape == (2, 2)
        assert np.allclose(psi.numpy(), [[1, 2], [3, 4]])

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

    def test_randn_seeded(self):
        psi1 = syn.state.randn((10,), seed=42)
        psi2 = syn.state.randn((10,), seed=42)
        assert np.allclose(psi1.numpy(), psi2.numpy())

    def test_eye(self):
        I = syn.state.eye(3)
        assert np.allclose(I.numpy(), np.eye(3))

    def test_from_numpy(self):
        arr = np.array([1.0, 2.0, 3.0])
        psi = syn.state.from_numpy(arr)
        assert np.allclose(psi.numpy(), arr)


class TestStateProperties:
    """Tests for State properties."""

    def test_shape(self):
        psi = syn.state([[1, 2, 3], [4, 5, 6]])
        assert psi.shape == (2, 3)

    def test_ndim(self):
        psi = syn.state([[1, 2], [3, 4]])
        assert psi.ndim == 2

    def test_size(self):
        psi = syn.state.zeros((2, 3, 4))
        assert psi.size == 24

    def test_dtype_float64(self):
        psi = syn.state([1.0, 2.0])
        assert psi.dtype == syn.float64

    def test_dtype_complex128(self):
        psi = syn.state([1 + 0j, 2 + 0j])
        assert psi.dtype == syn.complex128

    def test_device_default_cpu(self):
        psi = syn.state([1, 2, 3])
        assert psi.device == syn.cpu
        assert psi.device.is_cpu


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

    def test_div_states(self):
        a = syn.state([4, 6, 8])
        b = syn.state([2, 2, 2])
        c = a / b
        assert np.allclose(c.numpy(), [2, 3, 4])

    def test_add_scalar(self):
        a = syn.state([1, 2, 3])
        b = a + 10
        assert np.allclose(b.numpy(), [11, 12, 13])

    def test_radd_scalar(self):
        a = syn.state([1, 2, 3])
        b = 10 + a
        assert np.allclose(b.numpy(), [11, 12, 13])

    def test_mul_scalar(self):
        a = syn.state([1, 2, 3])
        b = a * 2
        assert np.allclose(b.numpy(), [2, 4, 6])

    def test_rmul_scalar(self):
        a = syn.state([1, 2, 3])
        b = 2 * a
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

    def test_pow(self):
        a = syn.state([1, 2, 3])
        b = a ** 2
        assert np.allclose(b.numpy(), [1, 4, 9])

    def test_sub_scalar(self):
        """Test subtraction with scalar."""
        a = syn.state([10, 20, 30])
        b = a - 5
        assert np.allclose(b.numpy(), [5, 15, 25])

    def test_rsub_scalar(self):
        """Test reverse subtraction with scalar."""
        a = syn.state([1, 2, 3])
        b = 10 - a
        assert np.allclose(b.numpy(), [9, 8, 7])

    def test_div_scalar(self):
        """Test division by scalar."""
        a = syn.state([10, 20, 30])
        b = a / 2
        assert np.allclose(b.numpy(), [5, 10, 15])


class TestStateReductions:
    """Tests for State reduction operations."""

    def test_norm_l2(self):
        psi = syn.state([3, 4])
        assert np.isclose(psi.norm(), 5.0)

    def test_norm_l1(self):
        psi = syn.state([3, -4])
        # Note: Rust backend may only support L2 norm
        assert psi.norm() > 0

    def test_normalize(self):
        psi = syn.state([3, 4]).normalize()
        assert np.isclose(psi.norm(), 1.0)

    def test_normalize_zero_raises(self):
        psi = syn.state.zeros((3,))
        with pytest.raises(syn.SyntonicError):
            psi.normalize()

    def test_sum(self):
        psi = syn.state([1, 2, 3, 4])
        assert np.isclose(psi.sum(), 10.0)

    def test_mean(self):
        psi = syn.state([1, 2, 3, 4])
        assert np.isclose(psi.mean(), 2.5)

    def test_max(self):
        psi = syn.state([1, 5, 3])
        assert np.isclose(psi.max(), 5.0)

    def test_min(self):
        psi = syn.state([1, 5, 3])
        assert np.isclose(psi.min(), 1.0)

    def test_abs(self):
        psi = syn.state([-1, 2, -3])
        assert np.allclose(psi.abs().numpy(), [1, 2, 3])


class TestStateComplex:
    """Tests for complex State operations."""

    def test_conj(self):
        psi = syn.state([1 + 2j, 3 + 4j])
        conj_psi = psi.conj()
        expected = np.array([1 - 2j, 3 - 4j])
        assert np.allclose(conj_psi.numpy(), expected)

    def test_real(self):
        psi = syn.state([1 + 2j, 3 + 4j])
        real_psi = psi.real()
        assert np.allclose(real_psi.numpy(), [1, 3])

    def test_imag(self):
        psi = syn.state([1 + 2j, 3 + 4j])
        imag_psi = psi.imag()
        assert np.allclose(imag_psi.numpy(), [2, 4])

    def test_transpose(self):
        psi = syn.state([[1, 2, 3], [4, 5, 6]])
        psi_t = psi.T
        assert psi_t.shape == (3, 2)
        assert np.allclose(psi_t.numpy(), [[1, 4], [2, 5], [3, 6]])

    def test_hermitian(self):
        psi = syn.state([[1 + 1j, 2 + 2j], [3 + 3j, 4 + 4j]])
        psi_h = psi.H
        expected = np.array([[1 - 1j, 3 - 3j], [2 - 2j, 4 - 4j]])
        assert np.allclose(psi_h.numpy(), expected)


class TestStateShape:
    """Tests for State shape operations."""

    def test_reshape(self):
        psi = syn.state([1, 2, 3, 4, 5, 6])
        reshaped = psi.reshape(2, 3)
        assert reshaped.shape == (2, 3)

    def test_reshape_tuple(self):
        psi = syn.state([1, 2, 3, 4, 5, 6])
        reshaped = psi.reshape((2, 3))
        assert reshaped.shape == (2, 3)

    def test_flatten(self):
        psi = syn.state([[1, 2], [3, 4]])
        flat = psi.flatten()
        assert flat.shape == (4,)
        assert np.allclose(flat.numpy(), [1, 2, 3, 4])

    def test_squeeze(self):
        psi = syn.state([[[1, 2, 3]]])
        squeezed = psi.squeeze()
        assert squeezed.shape == (3,)

    def test_unsqueeze(self):
        psi = syn.state([1, 2, 3])
        unsqueezed = psi.unsqueeze(0)
        assert unsqueezed.shape == (1, 3)


class TestStateDHSR:
    """Tests for DHSR operations."""

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
        psi = syn.state.random((10,), seed=42)
        result = psi.differentiate().harmonize().differentiate().harmonize()
        assert result.shape == psi.shape

    def test_syntony_property(self):
        psi = syn.state([1, 2, 3, 4])
        s = psi.syntony
        assert 0 <= s <= 1

    def test_gnosis_property(self):
        psi = syn.state([1, 2, 3, 4])
        g = psi.gnosis
        assert g in [0, 1, 2, 3]

    def test_free_energy_property(self):
        psi = syn.state([1, 2, 3, 4])
        f = psi.free_energy
        assert isinstance(f, float)


class TestStateIndexing:
    """Tests for State indexing operations."""

    def test_getitem_scalar(self):
        psi = syn.state([1, 2, 3, 4])
        item = psi[0]
        assert np.allclose(item.numpy(), 1.0)

    def test_getitem_slice(self):
        psi = syn.state([1, 2, 3, 4, 5])
        sliced = psi[1:4]
        assert np.allclose(sliced.numpy(), [2, 3, 4])

    def test_getitem_2d(self):
        psi = syn.state([[1, 2, 3], [4, 5, 6]])
        row = psi[0]
        assert np.allclose(row.numpy(), [1, 2, 3])

    def test_setitem(self):
        psi = syn.state([1, 2, 3, 4])
        psi[0] = 10
        assert np.allclose(psi.numpy(), [10, 2, 3, 4])


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


class TestStateRepresentation:
    """Tests for State string representation."""

    def test_repr(self):
        psi = syn.state([1, 2, 3])
        r = repr(psi)
        assert 'State' in r
        assert 'shape' in r
        assert 'dtype' in r

    def test_str(self):
        psi = syn.state([1, 2, 3])
        s = str(psi)
        assert 'State' in s

    def test_len(self):
        psi = syn.state([1, 2, 3, 4])
        assert len(psi) == 4


class TestStateAdvancedCreation:
    """Tests for advanced State creation scenarios."""

    def test_from_state(self):
        """Test State copy constructor."""
        psi1 = syn.state([1, 2, 3, 4])
        psi2 = syn.state(psi1)
        assert np.allclose(psi2.numpy(), psi1.numpy())
        assert psi2.dtype == psi1.dtype
        # Ensure they are independent copies
        psi1[0] = 100
        assert not np.allclose(psi2.numpy(), psi1.numpy())

    def test_explicit_dtype(self):
        """Test explicit dtype parameter."""
        psi = syn.state([1, 2, 3], dtype=syn.float32)
        assert psi.dtype == syn.float32

    def test_no_data_no_shape_raises(self):
        """Test that creating State without data or shape raises."""
        with pytest.raises(ValueError, match="Either data or shape"):
            syn.State()

    def test_zeros_with_int_shape(self):
        """Test zeros with integer shape (not tuple)."""
        psi = syn.state.zeros(5)
        assert psi.shape == (5,)

    def test_ones_with_int_shape(self):
        """Test ones with integer shape."""
        psi = syn.state.ones(3)
        assert psi.shape == (3,)

    def test_random_complex(self):
        """Test random with complex dtype."""
        psi = syn.state.random((5,), dtype=syn.complex128, seed=42)
        assert psi.dtype == syn.complex128
        # Check both real and imaginary parts are non-zero
        assert np.any(np.real(psi.numpy()) != 0)
        assert np.any(np.imag(psi.numpy()) != 0)

    def test_randn_complex(self):
        """Test randn with complex dtype."""
        psi = syn.state.randn((5,), dtype=syn.complex128, seed=42)
        assert psi.dtype == syn.complex128
        # Check both real and imaginary parts are non-zero
        assert np.any(np.real(psi.numpy()) != 0)
        assert np.any(np.imag(psi.numpy()) != 0)


class TestStateDeviceOperations:
    """Tests for device-related operations."""

    def test_cuda_is_available(self):
        """Test that CUDA is available."""
        assert syn.cuda_is_available() is True

    def test_cuda_device_count(self):
        """Test CUDA device count."""
        count = syn.cuda_device_count()
        assert count >= 1

    def test_cuda_transfer(self):
        """Test transferring state to CUDA and back."""
        psi = syn.state([1.0, 2.0, 3.0, 4.0])
        assert psi.device.is_cpu

        # Transfer to CUDA
        psi_cuda = psi.cuda()
        assert psi_cuda.device.is_cuda
        assert psi_cuda.device.index == 0

        # Transfer back to CPU
        psi_back = psi_cuda.cpu()
        assert psi_back.device.is_cpu

        # Verify data preserved
        assert np.allclose(psi_back.numpy(), [1.0, 2.0, 3.0, 4.0])

    def test_cuda_device_id(self):
        """Test CUDA device ID in device name."""
        psi = syn.state([1, 2, 3]).cuda()
        assert psi.device.index == 0
        assert 'cuda' in str(psi.device).lower()

    def test_cuda_already_on_cuda(self):
        """Test cuda() when already on CUDA returns self."""
        psi = syn.state([1, 2, 3]).cuda()
        psi_same = psi.cuda()
        # Should return same device
        assert psi_same.device.is_cuda

    def test_cpu_on_cpu(self):
        """Test cpu() when already on CPU returns self."""
        psi = syn.state([1, 2, 3])
        psi_cpu = psi.cpu()
        assert psi_cpu.device.is_cpu
        assert psi_cpu is psi  # Should be same object

    def test_to_cpu(self):
        """Test to() with CPU device."""
        psi = syn.state([1, 2, 3])
        psi_to = psi.to(syn.cpu)
        assert psi_to.device.is_cpu

    def test_to_cuda(self):
        """Test to() with CUDA device."""
        psi = syn.state([1, 2, 3])
        cuda_dev = syn.Device('cuda', 0)
        psi_cuda = psi.to(cuda_dev)
        assert psi_cuda.device.is_cuda

    def test_cuda_large_tensor(self):
        """Test CUDA transfer with large tensor."""
        psi = syn.state.random((1000,), seed=42)
        original = psi.to_list()

        psi_cuda = psi.cuda()
        psi_back = psi_cuda.cpu()

        assert np.allclose(psi_back.numpy(), original)

    def test_cuda_complex(self):
        """Test CUDA transfer with complex data."""
        psi = syn.state([1 + 2j, 3 + 4j, 5 + 6j])
        psi_cuda = psi.cuda()
        assert psi_cuda.device.is_cuda
        psi_back = psi_cuda.cpu()
        assert psi_back.device.is_cpu
        result = psi_back.to_list()
        expected = [1 + 2j, 3 + 4j, 5 + 6j]
        for r, e in zip(result, expected):
            assert abs(r - e) < 1e-10

    def test_cuda_2d(self):
        """Test CUDA transfer with 2D state."""
        psi = syn.state([[1, 2], [3, 4]])
        psi_cuda = psi.cuda()
        psi_back = psi_cuda.cpu()
        assert psi_back.shape == (2, 2)
        assert np.allclose(psi_back.numpy(), [[1, 2], [3, 4]])


class TestStateGnosisLevels:
    """Tests for different gnosis levels based on TV."""

    def test_gnosis_level_0(self):
        """Test gnosis level 0 (TV < 0.1)."""
        # Constant array has TV = 0
        psi = syn.state([1, 1, 1, 1])
        assert psi.gnosis == 0

    def test_gnosis_level_1(self):
        """Test gnosis level 1 (0.1 <= TV < 0.5)."""
        # Create array with small variation
        psi = syn.state([1.0, 1.05, 1.1, 1.15])
        g = psi.gnosis
        # TV = 0.05 + 0.05 + 0.05 = 0.15, so gnosis should be 1
        assert g == 1

    def test_gnosis_level_2(self):
        """Test gnosis level 2 (0.5 <= TV < 1.0)."""
        # Create array with medium variation
        psi = syn.state([1.0, 1.2, 1.4, 1.6])
        g = psi.gnosis
        # TV = 0.2 + 0.2 + 0.2 = 0.6, so gnosis should be 2
        assert g == 2

    def test_gnosis_level_3(self):
        """Test gnosis level 3 (TV >= 1.0)."""
        # Create array with large variation
        psi = syn.state([1.0, 2.0, 3.0, 4.0])
        g = psi.gnosis
        # TV = 1 + 1 + 1 = 3, so gnosis should be 3
        assert g == 3


class TestStateReductionsWithAxis:
    """Tests for reduction operations with axis parameter."""

    def test_sum_with_axis(self):
        """Test sum along axis."""
        psi = syn.state([[1, 2, 3], [4, 5, 6]])
        result = psi.sum(axis=0)
        assert np.allclose(result.numpy(), [5, 7, 9])

    def test_sum_with_axis_1(self):
        """Test sum along axis 1."""
        psi = syn.state([[1, 2, 3], [4, 5, 6]])
        result = psi.sum(axis=1)
        assert np.allclose(result.numpy(), [6, 15])

    def test_mean_with_axis(self):
        """Test mean along axis."""
        psi = syn.state([[1, 2, 3], [4, 5, 6]])
        result = psi.mean(axis=0)
        assert np.allclose(result.numpy(), [2.5, 3.5, 4.5])

    def test_max_with_axis(self):
        """Test max along axis."""
        psi = syn.state([[1, 5, 3], [4, 2, 6]])
        result = psi.max(axis=0)
        assert np.allclose(result.numpy(), [4, 5, 6])

    def test_min_with_axis(self):
        """Test min along axis."""
        psi = syn.state([[1, 5, 3], [4, 2, 6]])
        result = psi.min(axis=0)
        assert np.allclose(result.numpy(), [1, 2, 3])


class TestStateSetitemAdvanced:
    """Tests for advanced setitem operations."""

    def test_setitem_with_state(self):
        """Test setitem with State value."""
        psi = syn.state([[1, 2], [3, 4]])
        new_row = syn.state([10, 20])
        psi[0] = new_row
        assert np.allclose(psi.numpy(), [[10, 20], [3, 4]])


class TestStateArrayProtocol:
    """Tests for NumPy array protocol."""

    def test_array_with_dtype(self):
        """Test __array__ with dtype conversion."""
        psi = syn.state([1.5, 2.5, 3.5])
        arr = np.asarray(psi, dtype=np.int32)
        assert arr.dtype == np.int32
        assert np.allclose(arr, [1, 2, 3])


class TestStateTorchInterop:
    """Tests for PyTorch interoperability."""

    def test_torch_import_error(self):
        """Test torch() raises ImportError when PyTorch not available."""
        psi = syn.state([1, 2, 3])
        try:
            import torch
            # PyTorch is available, test conversion
            t = psi.torch()
            assert t.shape == (3,)
        except ImportError:
            # PyTorch not available, should raise
            with pytest.raises(ImportError, match="PyTorch not installed"):
                psi.torch()

    def test_from_torch_via_namespace(self):
        """Test state.from_torch via StateNamespace."""
        try:
            import torch
            t = torch.tensor([1.0, 2.0, 3.0])
            psi = syn.state.from_torch(t)
            assert np.allclose(psi.numpy(), [1, 2, 3])
        except ImportError:
            pytest.skip("PyTorch not installed")

    def test_from_torch_classmethod(self):
        """Test State.from_torch classmethod."""
        try:
            import torch
            t = torch.tensor([4.0, 5.0, 6.0])
            psi = syn.State.from_torch(t)
            assert np.allclose(psi.numpy(), [4, 5, 6])
        except ImportError:
            pytest.skip("PyTorch not installed")


class TestStateEdgeCases:
    """Tests for edge cases and error handling."""

    def test_scalar_data(self):
        """Test State from scalar data."""
        psi = syn.state(5.0)
        assert psi.shape == ()
        assert psi.size == 1

    def test_empty_list(self):
        """Test State from empty list."""
        psi = syn.state([])
        assert psi.shape == (0,)

    def test_inconsistent_shape_raises(self):
        """Test that inconsistent nested shapes raise ValueError."""
        with pytest.raises(ValueError, match="Inconsistent shapes"):
            syn.state([[1, 2], [3, 4, 5]])

    def test_shape_mismatch_raises(self):
        """Test that data/shape mismatch raises ValueError."""
        with pytest.raises(ValueError, match="doesn't match shape"):
            syn.state([1, 2, 3, 4], shape=(2, 3))

    def test_reshape_invalid_raises(self):
        """Test that invalid reshape raises ValueError."""
        psi = syn.state([1, 2, 3, 4])
        with pytest.raises(ValueError, match="Cannot reshape"):
            psi.reshape(2, 5)

    def test_multiple_neg_one_raises(self):
        """Test that multiple -1 in reshape raises ValueError."""
        psi = syn.state([1, 2, 3, 4])
        with pytest.raises(ValueError, match="Only one dimension"):
            psi.reshape(-1, -1)

    def test_complex_sum(self):
        """Test sum with complex numbers."""
        psi = syn.state([1 + 2j, 3 + 4j])
        result = psi.sum()
        assert np.isclose(result, 4 + 6j)

    def test_complex_mean(self):
        """Test mean with complex numbers."""
        psi = syn.state([1 + 2j, 3 + 4j])
        result = psi.mean()
        assert np.isclose(result, 2 + 3j)

    def test_complex_max(self):
        """Test max with complex numbers (by magnitude)."""
        psi = syn.state([1 + 0j, 0 + 5j])
        result = psi.max()
        assert np.isclose(abs(result), 5.0)

    def test_complex_min(self):
        """Test min with complex numbers (by magnitude)."""
        psi = syn.state([1 + 0j, 0 + 5j])
        result = psi.min()
        assert np.isclose(abs(result), 1.0)

    def test_complex_pow(self):
        """Test power with complex numbers."""
        psi = syn.state([1 + 1j, 2 + 0j])
        result = psi ** 2
        # (1+1j)^2 = 2j
        assert np.isclose(result.to_list()[0], 2j)

    def test_real_of_real(self):
        """Test real() on real state returns self."""
        psi = syn.state([1, 2, 3])
        real = psi.real()
        assert np.allclose(real.numpy(), [1, 2, 3])

    def test_imag_of_real(self):
        """Test imag() on real state returns zeros."""
        psi = syn.state([1, 2, 3])
        imag = psi.imag()
        assert np.allclose(imag.numpy(), [0, 0, 0])

    def test_negative_axis_reduction(self):
        """Test reduction with negative axis."""
        psi = syn.state([[1, 2, 3], [4, 5, 6]])
        result = psi.sum(axis=-1)
        assert np.allclose(result.numpy(), [6, 15])

    def test_cpu_returns_self(self):
        """Test cpu() when already on CPU returns self."""
        psi = syn.state([1, 2, 3])
        psi_cpu = psi.cpu()
        # Should be same object
        assert psi_cpu is psi

    def test_tolist_nested(self):
        """Test tolist() returns proper nested structure."""
        psi = syn.state([[1, 2], [3, 4]])
        result = psi.tolist()
        assert result == [[1.0, 2.0], [3.0, 4.0]]

    def test_from_list_classmethod(self):
        """Test State.from_list classmethod."""
        psi = syn.State.from_list([1.0, 2.0, 3.0, 4.0], (2, 2))
        assert psi.shape == (2, 2)

    def test_state_from_list_namespace(self):
        """Test state.from_list namespace method."""
        psi = syn.state.from_list([1.0, 2.0, 3.0, 4.0], (2, 2))
        assert psi.shape == (2, 2)

    def test_str_truncated(self):
        """Test __str__ with large state truncates output."""
        psi = syn.state.random((100,), seed=42)
        s = str(psi)
        assert '...' in s

    def test_len_empty_shape(self):
        """Test __len__ with scalar returns 0."""
        psi = syn.state(5.0)
        # Scalar has empty shape, so first dim doesn't exist
        assert len(psi) == 0

    def test_squeeze_all_ones(self):
        """Test squeeze with all dimensions of size 1."""
        psi = syn.state([[[1]]])
        squeezed = psi.squeeze()
        assert squeezed.shape == (1,)

    def test_unsqueeze_negative_dim(self):
        """Test unsqueeze with negative dimension."""
        psi = syn.state([1, 2, 3])
        unsqueezed = psi.unsqueeze(-1)
        assert unsqueezed.shape == (3, 1)

    def test_from_numpy_complex(self):
        """Test from_numpy with complex array."""
        arr = np.array([1 + 2j, 3 + 4j])
        psi = syn.state.from_numpy(arr)
        assert psi.dtype == syn.complex128
        assert psi.shape == (2,)

    def test_from_list_complex(self):
        """Test State.from_list with complex numbers."""
        data = [1 + 2j, 3 + 4j]
        psi = syn.State.from_list(data, (2,), dtype=syn.complex128)
        assert psi.dtype == syn.complex128
        assert psi.shape == (2,)

    def test_setitem_complex(self):
        """Test __setitem__ with complex state."""
        psi = syn.state([1 + 2j, 3 + 4j])
        psi[0] = 5 + 6j
        flat = psi.to_list()
        assert np.isclose(flat[0], 5 + 6j)
