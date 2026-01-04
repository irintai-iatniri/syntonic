"""pytest configuration and fixtures for Syntonic tests."""

import pytest


@pytest.fixture
def sample_vector():
    """Sample 1D vector for testing."""
    return [1.0, 2.0, 3.0, 4.0]


@pytest.fixture
def sample_matrix():
    """Sample 2D matrix for testing."""
    return [[1.0, 2.0], [3.0, 4.0]]


@pytest.fixture
def sample_complex_vector():
    """Sample complex vector for testing."""
    return [1 + 2j, 3 + 4j, 5 + 6j]


@pytest.fixture
def identity_3x3():
    """3x3 identity matrix."""
    return [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]


@pytest.fixture
def symmetric_positive_definite():
    """A symmetric positive definite matrix for Cholesky tests."""
    return [[4.0, 2.0], [2.0, 5.0]]
