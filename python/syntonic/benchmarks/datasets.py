"""
Synthetic dataset utilities for benchmarking.

Provides toy datasets for comparing Resonant Engine vs standard ML:
- XOR: Classic non-linear classification
- Moons: Two interleaving half-circles
- Circles: Concentric circles
- Spiral: Interleaved spirals (hardest)

All functions return (X, y) tuples compatible with both RES and PyTorch.
"""

import numpy as np
from typing import Tuple, Optional


def make_xor(
    n_samples: int = 1000,
    noise: float = 0.1,
    seed: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate XOR classification dataset.

    Four quadrants with alternating labels:
    - (+, +) -> 0, (-, -) -> 0
    - (+, -) -> 1, (-, +) -> 1

    Args:
        n_samples: Total number of samples
        noise: Standard deviation of Gaussian noise
        seed: Random seed for reproducibility

    Returns:
        X: Features of shape (n_samples, 2)
        y: Labels of shape (n_samples,) with values 0 or 1
    """
    rng = np.random.RandomState(seed)

    # Generate points in [-1, 1] x [-1, 1]
    X = rng.uniform(-1, 1, size=(n_samples, 2))

    # XOR labels: same sign -> 0, different sign -> 1
    y = ((X[:, 0] > 0) != (X[:, 1] > 0)).astype(np.int64)

    # Add noise
    if noise > 0:
        X += rng.normal(0, noise, size=X.shape)

    return X.astype(np.float64), y


def make_moons(
    n_samples: int = 1000,
    noise: float = 0.1,
    seed: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate two interleaving half-circles (moons) dataset.

    Classic non-linear separation problem where standard linear
    classifiers fail but non-linear methods succeed.

    Args:
        n_samples: Total number of samples (split evenly between classes)
        noise: Standard deviation of Gaussian noise
        seed: Random seed for reproducibility

    Returns:
        X: Features of shape (n_samples, 2)
        y: Labels of shape (n_samples,) with values 0 or 1
    """
    rng = np.random.RandomState(seed)

    n_samples_per_class = n_samples // 2

    # First moon (top)
    theta1 = np.linspace(0, np.pi, n_samples_per_class)
    x1 = np.cos(theta1)
    y1 = np.sin(theta1)

    # Second moon (bottom, shifted)
    theta2 = np.linspace(0, np.pi, n_samples - n_samples_per_class)
    x2 = 1 - np.cos(theta2)
    y2 = 0.5 - np.sin(theta2)

    # Combine
    X = np.vstack([
        np.column_stack([x1, y1]),
        np.column_stack([x2, y2])
    ])

    y = np.hstack([
        np.zeros(n_samples_per_class),
        np.ones(n_samples - n_samples_per_class)
    ]).astype(np.int64)

    # Shuffle
    indices = rng.permutation(len(y))
    X, y = X[indices], y[indices]

    # Add noise
    if noise > 0:
        X += rng.normal(0, noise, size=X.shape)

    return X.astype(np.float64), y


def make_circles(
    n_samples: int = 1000,
    noise: float = 0.1,
    factor: float = 0.5,
    seed: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate concentric circles dataset.

    Inner and outer circles with different labels.
    Requires radial separation rather than linear.

    Args:
        n_samples: Total number of samples
        noise: Standard deviation of Gaussian noise
        factor: Scale factor for inner circle (0 < factor < 1)
        seed: Random seed for reproducibility

    Returns:
        X: Features of shape (n_samples, 2)
        y: Labels of shape (n_samples,) with values 0 or 1
    """
    rng = np.random.RandomState(seed)

    n_outer = n_samples // 2
    n_inner = n_samples - n_outer

    # Outer circle
    theta_outer = rng.uniform(0, 2 * np.pi, n_outer)
    X_outer = np.column_stack([np.cos(theta_outer), np.sin(theta_outer)])

    # Inner circle
    theta_inner = rng.uniform(0, 2 * np.pi, n_inner)
    X_inner = factor * np.column_stack([np.cos(theta_inner), np.sin(theta_inner)])

    # Combine
    X = np.vstack([X_outer, X_inner])
    y = np.hstack([np.zeros(n_outer), np.ones(n_inner)]).astype(np.int64)

    # Shuffle
    indices = rng.permutation(len(y))
    X, y = X[indices], y[indices]

    # Add noise
    if noise > 0:
        X += rng.normal(0, noise, size=X.shape)

    return X.astype(np.float64), y


def make_spiral(
    n_samples: int = 1000,
    noise: float = 0.1,
    n_classes: int = 2,
    seed: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate interleaved spirals dataset.

    This is one of the hardest 2D classification problems.
    Spirals wind around each other, requiring complex decision boundaries.

    Args:
        n_samples: Total number of samples
        noise: Standard deviation of Gaussian noise
        n_classes: Number of spiral arms (2 or 3 typical)
        seed: Random seed for reproducibility

    Returns:
        X: Features of shape (n_samples, 2)
        y: Labels of shape (n_samples,) with values 0 to n_classes-1
    """
    rng = np.random.RandomState(seed)

    n_per_class = n_samples // n_classes

    X_list = []
    y_list = []

    for c in range(n_classes):
        # Angle offset for this class
        offset = c * 2 * np.pi / n_classes

        # Radius increases with angle
        n_points = n_per_class if c < n_classes - 1 else n_samples - c * n_per_class
        theta = np.linspace(0, 3 * np.pi, n_points) + offset
        r = theta / (3 * np.pi)  # Normalized radius [0, 1]

        x = r * np.cos(theta)
        y_coord = r * np.sin(theta)

        X_list.append(np.column_stack([x, y_coord]))
        y_list.append(np.full(n_points, c))

    X = np.vstack(X_list)
    y = np.hstack(y_list).astype(np.int64)

    # Shuffle
    indices = rng.permutation(len(y))
    X, y = X[indices], y[indices]

    # Add noise
    if noise > 0:
        X += rng.normal(0, noise, size=X.shape)

    return X.astype(np.float64), y


def make_golden_sequence(
    n_samples: int = 100,
    seed: Optional[int] = None,
) -> np.ndarray:
    """
    Generate a sequence with golden ratio structure.

    Values follow patterns related to phi for testing
    exact lattice preservation in Resonant Engine.

    Args:
        n_samples: Length of sequence
        seed: Random seed for reproducibility

    Returns:
        X: Sequence of shape (n_samples,) with golden structure
    """
    PHI = 1.6180339887498948482

    rng = np.random.RandomState(seed)

    # Generate Fibonacci-like structure with noise
    fib = np.zeros(n_samples)
    fib[0] = 1.0
    if n_samples > 1:
        fib[1] = PHI

    for i in range(2, n_samples):
        # Exact golden recurrence with small perturbation
        fib[i] = fib[i-1] + fib[i-2]

    # Normalize to [0, 1] range
    fib = fib / fib.max()

    # Add small noise while preserving phi structure
    noise = rng.normal(0, 0.01, size=n_samples)
    fib += noise

    return fib.astype(np.float64)


def train_test_split(
    X: np.ndarray,
    y: np.ndarray,
    test_size: float = 0.2,
    seed: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Split dataset into train and test sets.

    Args:
        X: Features
        y: Labels
        test_size: Fraction for test set
        seed: Random seed

    Returns:
        X_train, X_test, y_train, y_test
    """
    rng = np.random.RandomState(seed)

    n = len(y)
    n_test = int(n * test_size)

    indices = rng.permutation(n)
    test_idx = indices[:n_test]
    train_idx = indices[n_test:]

    return X[train_idx], X[test_idx], y[train_idx], y[test_idx]
