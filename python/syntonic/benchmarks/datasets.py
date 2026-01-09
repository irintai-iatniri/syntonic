"""
Synthetic dataset utilities for benchmarking.

Provides toy datasets for comparing Resonant Engine vs standard ML:
- XOR: Classic non-linear classification
- Moons: Two interleaving half-circles
- Circles: Concentric circles
- Spiral: Interleaved spirals (hardest)

All functions return (X, y) tuples as pure Python lists (no NumPy).
"""

import math
import random
from typing import Tuple, Optional, List


class PureRNG:
    """Pure Python random number generator using stdlib random."""

    def __init__(self, seed: Optional[int] = None):
        self.rng = random.Random(seed)

    def uniform(self, low: float, high: float, size: Tuple[int, ...]) -> List:
        """Generate uniform random numbers in [low, high)."""
        if len(size) == 1:
            return [self.rng.uniform(low, high) for _ in range(size[0])]
        elif len(size) == 2:
            return [[self.rng.uniform(low, high) for _ in range(size[1])]
                    for _ in range(size[0])]
        else:
            raise ValueError(f"Unsupported size dimension: {len(size)}")

    def normal(self, mean: float, std: float, size: Tuple[int, ...]) -> List:
        """Generate normal (Gaussian) random numbers using Box-Muller."""
        if len(size) == 1:
            return [self.rng.gauss(mean, std) for _ in range(size[0])]
        elif len(size) == 2:
            return [[self.rng.gauss(mean, std) for _ in range(size[1])]
                    for _ in range(size[0])]
        else:
            raise ValueError(f"Unsupported size dimension: {len(size)}")

    def permutation(self, n: int) -> List[int]:
        """Generate a random permutation of range(n) using Fisher-Yates."""
        indices = list(range(n))
        for i in range(n - 1, 0, -1):
            j = self.rng.randint(0, i)
            indices[i], indices[j] = indices[j], indices[i]
        return indices


def linspace(start: float, stop: float, num: int) -> List[float]:
    """Generate evenly spaced numbers over a specified interval."""
    if num == 1:
        return [start]
    step = (stop - start) / (num - 1)
    return [start + step * i for i in range(num)]


def column_stack(arr1: List[float], arr2: List[float]) -> List[List[float]]:
    """Stack two 1D lists as columns into a 2D list."""
    return [[arr1[i], arr2[i]] for i in range(len(arr1))]


def vstack(arrays: List[List[List[float]]]) -> List[List[float]]:
    """Vertically stack 2D lists."""
    result = []
    for arr in arrays:
        result.extend(arr)
    return result


def hstack(arrays: List[List]) -> List:
    """Horizontally stack 1D lists."""
    result = []
    for arr in arrays:
        result.extend(arr)
    return result


def zeros(n: int) -> List[float]:
    """Create a list of zeros."""
    return [0.0] * n


def ones(n: int) -> List[float]:
    """Create a list of ones."""
    return [1.0] * n


def full(n: int, value: float) -> List[float]:
    """Create a list filled with a constant value."""
    return [float(value)] * n


def add_noise_2d(X: List[List[float]], noise_matrix: List[List[float]]) -> List[List[float]]:
    """Add noise to a 2D list in-place."""
    for i in range(len(X)):
        for j in range(len(X[i])):
            X[i][j] += noise_matrix[i][j]
    return X


def make_xor(
    n_samples: int = 1000,
    noise: float = 0.1,
    seed: Optional[int] = None,
) -> Tuple[List[List[float]], List[int]]:
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
        X: Features as list of lists, shape (n_samples, 2)
        y: Labels as list of ints, shape (n_samples,) with values 0 or 1
    """
    rng = PureRNG(seed)

    # Generate points in [-1, 1] x [-1, 1]
    X = rng.uniform(-1, 1, size=(n_samples, 2))

    # XOR labels: same sign -> 0, different sign -> 1
    y = [int((X[i][0] > 0) != (X[i][1] > 0)) for i in range(n_samples)]

    # Add noise
    if noise > 0:
        noise_matrix = rng.normal(0, noise, size=(n_samples, 2))
        X = add_noise_2d(X, noise_matrix)

    return X, y


def make_moons(
    n_samples: int = 1000,
    noise: float = 0.1,
    seed: Optional[int] = None,
) -> Tuple[List[List[float]], List[int]]:
    """
    Generate two interleaving half-circles (moons) dataset.

    Classic non-linear separation problem where standard linear
    classifiers fail but non-linear methods succeed.

    Args:
        n_samples: Total number of samples (split evenly between classes)
        noise: Standard deviation of Gaussian noise
        seed: Random seed for reproducibility

    Returns:
        X: Features as list of lists, shape (n_samples, 2)
        y: Labels as list of ints, shape (n_samples,) with values 0 or 1
    """
    rng = PureRNG(seed)

    n_samples_per_class = n_samples // 2

    # First moon (top)
    theta1 = linspace(0, math.pi, n_samples_per_class)
    x1 = [math.cos(t) for t in theta1]
    y1 = [math.sin(t) for t in theta1]

    # Second moon (bottom, shifted)
    theta2 = linspace(0, math.pi, n_samples - n_samples_per_class)
    x2 = [1 - math.cos(t) for t in theta2]
    y2 = [0.5 - math.sin(t) for t in theta2]

    # Combine
    X = vstack([
        column_stack(x1, y1),
        column_stack(x2, y2)
    ])

    y = hstack([
        zeros(n_samples_per_class),
        ones(n_samples - n_samples_per_class)
    ])
    y = [int(label) for label in y]

    # Shuffle
    indices = rng.permutation(len(y))
    X = [X[i] for i in indices]
    y = [y[i] for i in indices]

    # Add noise
    if noise > 0:
        noise_matrix = rng.normal(0, noise, size=(len(X), 2))
        X = add_noise_2d(X, noise_matrix)

    return X, y


def make_circles(
    n_samples: int = 1000,
    noise: float = 0.1,
    factor: float = 0.5,
    seed: Optional[int] = None,
) -> Tuple[List[List[float]], List[int]]:
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
        X: Features as list of lists, shape (n_samples, 2)
        y: Labels as list of ints, shape (n_samples,) with values 0 or 1
    """
    rng = PureRNG(seed)

    n_outer = n_samples // 2
    n_inner = n_samples - n_outer

    # Outer circle
    theta_outer = rng.uniform(0, 2 * math.pi, (n_outer,))
    X_outer = column_stack(
        [math.cos(t) for t in theta_outer],
        [math.sin(t) for t in theta_outer]
    )

    # Inner circle
    theta_inner = rng.uniform(0, 2 * math.pi, (n_inner,))
    X_inner = column_stack(
        [factor * math.cos(t) for t in theta_inner],
        [factor * math.sin(t) for t in theta_inner]
    )

    # Combine
    X = vstack([X_outer, X_inner])
    y = hstack([zeros(n_outer), ones(n_inner)])
    y = [int(label) for label in y]

    # Shuffle
    indices = rng.permutation(len(y))
    X = [X[i] for i in indices]
    y = [y[i] for i in indices]

    # Add noise
    if noise > 0:
        noise_matrix = rng.normal(0, noise, size=(len(X), 2))
        X = add_noise_2d(X, noise_matrix)

    return X, y


def make_spiral(
    n_samples: int = 1000,
    noise: float = 0.1,
    n_classes: int = 2,
    seed: Optional[int] = None,
) -> Tuple[List[List[float]], List[int]]:
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
        X: Features as list of lists, shape (n_samples, 2)
        y: Labels as list of ints, shape (n_samples,) with values 0 to n_classes-1
    """
    rng = PureRNG(seed)

    n_per_class = n_samples // n_classes

    X_list = []
    y_list = []

    for c in range(n_classes):
        # Angle offset for this class
        offset = c * 2 * math.pi / n_classes

        # Radius increases with angle
        n_points = n_per_class if c < n_classes - 1 else n_samples - c * n_per_class
        theta = [t + offset for t in linspace(0, 3 * math.pi, n_points)]
        r = [t / (3 * math.pi) for t in linspace(0, 3 * math.pi, n_points)]  # Normalized radius [0, 1]

        x = [r[i] * math.cos(theta[i]) for i in range(n_points)]
        y_coord = [r[i] * math.sin(theta[i]) for i in range(n_points)]

        X_list.append(column_stack(x, y_coord))
        y_list.append(full(n_points, c))

    X = vstack(X_list)
    y = hstack(y_list)
    y = [int(label) for label in y]

    # Shuffle
    indices = rng.permutation(len(y))
    X = [X[i] for i in indices]
    y = [y[i] for i in indices]

    # Add noise
    if noise > 0:
        noise_matrix = rng.normal(0, noise, size=(len(X), 2))
        X = add_noise_2d(X, noise_matrix)

    return X, y


def make_golden_sequence(
    n_samples: int = 100,
    seed: Optional[int] = None,
) -> List[float]:
    """
    Generate a sequence with golden ratio structure.

    Values follow patterns related to phi for testing
    exact lattice preservation in Resonant Engine.

    Args:
        n_samples: Length of sequence
        seed: Random seed for reproducibility

    Returns:
        X: Sequence as list of floats, shape (n_samples,) with golden structure
    """
    PHI = 1.6180339887498948482

    rng = PureRNG(seed)

    # Generate Fibonacci-like structure with noise
    fib = [0.0] * n_samples
    fib[0] = 1.0
    if n_samples > 1:
        fib[1] = PHI

    for i in range(2, n_samples):
        # Exact golden recurrence with small perturbation
        fib[i] = fib[i-1] + fib[i-2]

    # Normalize to [0, 1] range
    max_val = max(fib)
    fib = [f / max_val for f in fib]

    # Add small noise while preserving phi structure
    noise = rng.normal(0, 0.01, (n_samples,))
    fib = [fib[i] + noise[i] for i in range(n_samples)]

    return fib


def train_test_split(
    X: List[List[float]],
    y: List[int],
    test_size: float = 0.2,
    seed: Optional[int] = None,
) -> Tuple[List[List[float]], List[List[float]], List[int], List[int]]:
    """
    Split dataset into train and test sets.

    Args:
        X: Features as list of lists
        y: Labels as list
        test_size: Fraction for test set
        seed: Random seed

    Returns:
        X_train, X_test, y_train, y_test
    """
    rng = PureRNG(seed)

    n = len(y)
    n_test = int(n * test_size)

    indices = rng.permutation(n)
    test_idx = indices[:n_test]
    train_idx = indices[n_test:]

    X_train = [X[i] for i in train_idx]
    X_test = [X[i] for i in test_idx]
    y_train = [y[i] for i in train_idx]
    y_test = [y[i] for i in test_idx]

    return X_train, X_test, y_train, y_test
