"""Tests for linear algebra module."""

import syntonic as syn
from syntonic import linalg
import numpy as np
import pytest


class TestBasicOperations:
    """Tests for basic linear algebra operations."""

    def test_dot(self):
        a = syn.state([1, 2, 3])
        b = syn.state([4, 5, 6])
        result = linalg.dot(a, b)
        expected = 1 * 4 + 2 * 5 + 3 * 6
        assert np.isclose(result, expected)

    def test_inner(self):
        a = syn.state([1, 2, 3])
        b = syn.state([4, 5, 6])
        result = linalg.inner(a, b)
        assert np.isclose(result, np.vdot(a.numpy(), b.numpy()))

    def test_inner_complex(self):
        a = syn.state([1 + 1j, 2 + 2j])
        b = syn.state([3 + 3j, 4 + 4j])
        result = linalg.inner(a, b)
        expected = np.vdot(a.numpy(), b.numpy())
        assert np.isclose(result, expected)

    def test_outer(self):
        a = syn.state([1, 2])
        b = syn.state([3, 4])
        result = linalg.outer(a, b)
        expected = np.array([[3, 4], [6, 8]])
        assert np.allclose(result.numpy(), expected)

    def test_norm(self):
        a = syn.state([3, 4])
        result = linalg.norm(a)
        assert np.isclose(result, 5.0)

    def test_matmul(self):
        A = syn.state([[1, 2], [3, 4]])
        B = syn.state([[5, 6], [7, 8]])
        C = linalg.matmul(A, B)
        expected = np.array([[19, 22], [43, 50]])
        assert np.allclose(C.numpy(), expected)


class TestDecompositions:
    """Tests for matrix decompositions."""

    def test_eig(self):
        A = syn.state([[1, 2], [2, 1]])
        w, v = linalg.eig(A)
        # Check eigenvalue equation A @ v = v @ diag(w)
        # Just verify shapes for now
        assert w.size == 2

    def test_eigh(self):
        # Symmetric matrix
        A = syn.state([[2, 1], [1, 2]])
        w, v = linalg.eigh(A)
        # Eigenvalues should be real for symmetric matrix
        assert w.size == 2

    def test_svd(self):
        A = syn.state([[1, 2], [3, 4], [5, 6]])
        U, S, Vh = linalg.svd(A)
        # Check shapes
        assert U.shape[0] == 3
        assert len(S.shape) == 1

    def test_qr(self):
        A = syn.state([[1, 2], [3, 4], [5, 6]])
        Q, R = linalg.qr(A)
        # Verify A = QR
        reconstructed = Q @ R
        assert np.allclose(reconstructed.numpy(), A.numpy(), atol=1e-10)

    def test_cholesky(self, symmetric_positive_definite):
        A = syn.state(symmetric_positive_definite)
        L = linalg.cholesky(A)
        # Verify A = L @ L.T
        reconstructed = L @ L.T
        assert np.allclose(reconstructed.numpy(), A.numpy(), atol=1e-10)


class TestSolvers:
    """Tests for linear system solvers."""

    def test_inv(self):
        A = syn.state([[1, 2], [3, 4]])
        A_inv = linalg.inv(A)
        # A @ A_inv should be identity
        identity = A @ A_inv
        expected = np.eye(2)
        assert np.allclose(identity.numpy(), expected, atol=1e-10)

    def test_pinv(self):
        A = syn.state([[1, 2], [3, 4], [5, 6]])
        A_pinv = linalg.pinv(A)
        # A_pinv @ A should be close to identity (for overdetermined)
        assert A_pinv.shape == (2, 3)

    def test_solve(self):
        A = syn.state([[3, 1], [1, 2]])
        b = syn.state([9, 8])
        x = linalg.solve(A, b)
        # Verify A @ x = b by using numpy for the matmul
        result = np.dot(A.numpy(), x.numpy())
        assert np.allclose(result, b.numpy(), atol=1e-10)

    def test_det(self):
        A = syn.state([[1, 2], [3, 4]])
        d = linalg.det(A)
        expected = 1 * 4 - 2 * 3
        assert np.isclose(d.real, expected, atol=1e-10)

    def test_trace(self):
        A = syn.state([[1, 2], [3, 4]])
        t = linalg.trace(A)
        expected = 1 + 4
        assert np.isclose(t.real, expected)


class TestMatrixFunctions:
    """Tests for matrix functions (require scipy)."""

    def test_expm(self):
        pytest.importorskip("scipy")
        from scipy.linalg import expm as scipy_expm
        A = syn.state([[0, 1], [-1, 0]])
        result = linalg.expm(A)
        expected = scipy_expm(A.numpy())
        assert np.allclose(result.numpy(), expected, atol=1e-10)

    def test_logm(self):
        pytest.importorskip("scipy")
        from scipy.linalg import logm as scipy_logm
        # Use a simple positive definite matrix
        A = syn.state([[2, 0], [0, 3]])
        result = linalg.logm(A)
        expected = scipy_logm(A.numpy())
        assert np.allclose(result.numpy(), expected, atol=1e-10)


class TestMatrixFunctionsImportError:
    """Tests for matrix functions when scipy is not available."""

    def test_expm_import_error(self, monkeypatch):
        """Test expm raises ImportError when scipy not available."""
        import builtins
        original_import = builtins.__import__

        def mock_import(name, *args, **kwargs):
            if name == 'scipy.linalg' or name.startswith('scipy'):
                raise ImportError("No module named 'scipy'")
            return original_import(name, *args, **kwargs)

        # We need to reload the module to trigger the import error
        import syntonic.linalg as linalg_mod
        A = syn.state([[1, 0], [0, 1]])

        monkeypatch.setattr(builtins, '__import__', mock_import)
        with pytest.raises(ImportError, match="scipy required for matrix exponential"):
            linalg_mod.expm(A)

    def test_logm_import_error(self, monkeypatch):
        """Test logm raises ImportError when scipy not available."""
        import builtins
        original_import = builtins.__import__

        def mock_import(name, *args, **kwargs):
            if name == 'scipy.linalg' or name.startswith('scipy'):
                raise ImportError("No module named 'scipy'")
            return original_import(name, *args, **kwargs)

        import syntonic.linalg as linalg_mod
        A = syn.state([[2, 0], [0, 3]])

        monkeypatch.setattr(builtins, '__import__', mock_import)
        with pytest.raises(ImportError, match="scipy required for matrix logarithm"):
            linalg_mod.logm(A)


class TestLinalgEdgeCases:
    """Tests for edge cases in linalg module."""

    def test_dot_complex(self):
        """Test dot product with complex numbers."""
        a = syn.state([1 + 1j, 2 + 2j])
        b = syn.state([1 - 1j, 2 - 2j])
        result = linalg.dot(a, b)
        # (1+1j)(1-1j) + (2+2j)(2-2j) = 2 + 8 = 10
        assert np.isclose(result.real, 10.0)

    def test_dot_length_mismatch(self):
        """Test dot product raises on length mismatch."""
        a = syn.state([1, 2, 3])
        b = syn.state([1, 2])
        with pytest.raises(syn.LinAlgError, match="same length"):
            linalg.dot(a, b)

    def test_inner_length_mismatch(self):
        """Test inner product raises on length mismatch."""
        a = syn.state([1, 2, 3])
        b = syn.state([1, 2])
        with pytest.raises(syn.LinAlgError, match="same length"):
            linalg.inner(a, b)

    def test_solve_2d_b(self):
        """Test solve with 2D b vector."""
        A = syn.state([[1, 0], [0, 2]])
        B = syn.state([[1, 2], [3, 4]])
        X = linalg.solve(A, B)
        # X should satisfy A @ X = B
        assert X.shape == (2, 2)

    def test_expm_complex(self):
        """Test expm with complex matrix."""
        pytest.importorskip("scipy")
        A = syn.state([[1j, 0], [0, -1j]], dtype=syn.complex128)
        result = linalg.expm(A)
        assert result.dtype == syn.complex128

    def test_logm_returns_complex(self):
        """Test logm returns complex result."""
        pytest.importorskip("scipy")
        A = syn.state([[2, 0], [0, 3]])
        result = linalg.logm(A)
        # log of real positive matrix can have complex entries
        assert result.dtype == syn.complex128

    def test_logm_complex_input(self):
        """Test logm with complex input."""
        pytest.importorskip("scipy")
        A = syn.state([[1 + 0j, 0], [0, 1 + 0j]], dtype=syn.complex128)
        result = linalg.logm(A)
        # log(I) = 0
        assert result.dtype == syn.complex128
