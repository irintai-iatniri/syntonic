"""
Linear algebra operations for Syntonic.

This module is designed to be NumPy-free for core operations.
Some advanced operations (expm, logm, pinv) require scipy/numpy
and will raise ImportError if not available.
"""

from __future__ import annotations
from typing import Tuple, Optional, Union

from syntonic.core.state import State
from syntonic.core.dtype import float64, complex128
from syntonic.exceptions import LinAlgError


def matmul(a: State, b: State) -> State:
    """Matrix multiplication."""
    return a @ b


def dot(a: State, b: State) -> float:
    """Dot product of flattened arrays."""
    a_flat = a.to_list()
    b_flat = b.to_list()
    if len(a_flat) != len(b_flat):
        raise LinAlgError("Vectors must have same length for dot product")

    # Handle complex numbers
    if a.dtype.name == 'complex128' or b.dtype.name == 'complex128':
        result = sum(x * y for x, y in zip(a_flat, b_flat))
        return result
    return float(sum(x * y for x, y in zip(a_flat, b_flat)))


def inner(a: State, b: State) -> Union[float, complex]:
    """Inner product <a|b> (conjugate first argument for complex)."""
    a_flat = a.to_list()
    b_flat = b.to_list()
    if len(a_flat) != len(b_flat):
        raise LinAlgError("Vectors must have same length for inner product")

    if a.dtype.name == 'complex128':
        # Conjugate first argument
        return sum(x.conjugate() * y for x, y in zip(a_flat, b_flat))
    return float(sum(x * y for x, y in zip(a_flat, b_flat)))


def outer(a: State, b: State) -> State:
    """Outer product |a><b|."""
    a_flat = a.to_list()
    b_flat = b.to_list()

    # Build outer product matrix
    result = []
    for x in a_flat:
        for y in b_flat:
            result.append(x * y)

    new_shape = (len(a_flat), len(b_flat))
    return State(result, dtype=a.dtype, device=a.device, shape=new_shape)


def norm(x: State, ord: Optional[int] = None) -> float:
    """Vector or matrix norm."""
    return x.norm(ord)


def _state_from_storage(storage, dtype, device, shape):
    """Create State from storage with explicit shape."""
    s = State.__new__(State)
    s._storage = storage
    s._dtype = dtype
    s._device = device
    s._shape = shape
    s._syntony_cache = None
    s._gnosis_cache = None
    return s


def eig(a: State) -> Tuple[State, State]:
    """
    Eigenvalue decomposition.

    Returns:
        (eigenvalues, eigenvectors)
    """
    w_storage, v_storage = a._storage.eig()
    n = a.shape[0]

    # Eigenvalues shape is (n,), eigenvectors shape is (n, n)
    w = _state_from_storage(w_storage, complex128, a.device, (n,))
    v = _state_from_storage(v_storage, complex128, a.device, (n, n))
    return w, v


def eigh(a: State) -> Tuple[State, State]:
    """
    Eigenvalue decomposition for Hermitian matrices.

    Returns:
        (eigenvalues, eigenvectors)
    """
    w_storage, v_storage = a._storage.eigh()
    n = a.shape[0]

    # For Hermitian, eigenvalues are real
    w = _state_from_storage(w_storage, float64, a.device, (n,))
    v = _state_from_storage(v_storage, a.dtype, a.device, (n, n))
    return w, v


def svd(a: State, full_matrices: bool = True) -> Tuple[State, State, State]:
    """
    Singular value decomposition.

    Returns:
        (U, S, Vh)
    """
    u_storage, s_storage, vh_storage = a._storage.svd(full_matrices)
    m, n = a.shape[0], a.shape[1]
    k = min(m, n)

    u = _state_from_storage(u_storage, a.dtype, a.device, (m, m) if full_matrices else (m, k))
    s = _state_from_storage(s_storage, float64, a.device, (k,))
    vh = _state_from_storage(vh_storage, a.dtype, a.device, (n, n) if full_matrices else (k, n))
    return u, s, vh


def qr(a: State) -> Tuple[State, State]:
    """
    QR decomposition.

    Returns:
        (Q, R) where A = QR
    """
    q_storage, r_storage = a._storage.qr()
    m, n = a.shape[0], a.shape[1]
    k = min(m, n)

    q = _state_from_storage(q_storage, a.dtype, a.device, (m, k))
    r = _state_from_storage(r_storage, a.dtype, a.device, (k, n))
    return q, r


def cholesky(a: State) -> State:
    """
    Cholesky decomposition.

    Returns:
        L where A = LL*
    """
    l_storage = a._storage.cholesky()
    return _state_from_storage(l_storage, a.dtype, a.device, a.shape)


def inv(a: State) -> State:
    """Matrix inverse."""
    inv_storage = a._storage.inv()
    return _state_from_storage(inv_storage, a.dtype, a.device, a.shape)


def pinv(a: State) -> State:
    """
    Moore-Penrose pseudo-inverse.

    Computes via SVD: A^+ = V @ S^+ @ U^H
    """
    # Use full_matrices=True to avoid None values from SVD
    u, s, vh = svd(a, full_matrices=True)

    m, n = a.shape
    k = min(m, n)

    # Invert non-zero singular values
    s_flat = s.to_list()
    tol = max(a.shape) * max(abs(x) for x in s_flat) * 1e-15
    s_inv = [1.0/x if abs(x) > tol else 0.0 for x in s_flat]

    # Build S^+ matrix (n x m) - transpose of diagonal matrix with inverted singular values
    s_inv_mat = [0.0] * (n * m)
    for i in range(k):
        s_inv_mat[i * m + i] = s_inv[i]

    s_inv_state = State(s_inv_mat, dtype=float64, device=a.device, shape=(n, m))

    # A^+ = Vh^H @ S^+ @ U^H = V @ S^+ @ U^H
    return vh.H @ s_inv_state @ u.H


def det(a: State) -> complex:
    """Matrix determinant."""
    return a._storage.det()


def trace(a: State) -> complex:
    """Matrix trace."""
    return a._storage.trace()


def solve(a: State, b: State) -> State:
    """
    Solve linear system Ax = b.

    Returns:
        x such that Ax = b
    """
    x_storage = a._storage.solve(b._storage)

    # Determine output shape
    if b.ndim == 1:
        out_shape = b.shape
    else:
        out_shape = (a.shape[1], b.shape[1])

    return _state_from_storage(x_storage, a.dtype, a.device, out_shape)


def expm(a: State) -> State:
    """
    Matrix exponential exp(A).

    Important for CRT: evolution operators.

    Note: Requires scipy for matrix exponential.
    """
    try:
        from scipy.linalg import expm as scipy_expm
        # Convert via to_list and back
        flat = a.to_list()

        # Need numpy for scipy
        import numpy as np
        arr = np.array(flat).reshape(a.shape)
        if a.dtype.is_complex:
            arr = arr.astype(np.complex128)
        result = scipy_expm(arr)
        return State(result.flatten().tolist(), dtype=a.dtype, device=a.device, shape=a.shape)
    except ImportError:
        raise ImportError("scipy required for matrix exponential")


def logm(a: State) -> State:
    """
    Matrix logarithm log(A).

    Note: Requires scipy for matrix logarithm.
    """
    try:
        from scipy.linalg import logm as scipy_logm
        import numpy as np
        flat = a.to_list()
        arr = np.array(flat).reshape(a.shape)
        if a.dtype.is_complex:
            arr = arr.astype(np.complex128)
        result = scipy_logm(arr)
        return State(result.flatten().tolist(), dtype=complex128, device=a.device, shape=a.shape)
    except ImportError:
        raise ImportError("scipy required for matrix logarithm")


__all__ = [
    'matmul',
    'dot',
    'inner',
    'outer',
    'norm',
    'eig',
    'eigh',
    'svd',
    'qr',
    'cholesky',
    'inv',
    'pinv',
    'det',
    'trace',
    'solve',
    'expm',
    'logm',
]
