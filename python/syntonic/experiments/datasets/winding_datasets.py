"""
Dataset loaders for syntonic softmax validation experiments.

Provides three datasets:
1. XOR via winding states (200 samples)
2. Particle classification: leptons vs quarks (9 samples)
3. CSV winding dataset (n7, n8 → target)
"""

import random
import csv
from pathlib import Path
from typing import List, Tuple, Dict
from syntonic._core import WindingState

# =============================================================================
# 1. XOR Winding Dataset
# =============================================================================

def load_winding_xor_dataset(
    n_samples: int = 200,
    noise: float = 0.1,
    seed: int = 42
) -> Tuple[List[WindingState], List[int]]:
    """
    Generate XOR dataset mapped to winding states.

    Mapping:
    (0,0) -> Class 0 -> Winding(0,0,0,0)
    (0,1) -> Class 1 -> Winding(0,2,0,0)
    (1,0) -> Class 1 -> Winding(2,0,0,0)
    (1,1) -> Class 0 -> Winding(2,2,0,0)

    Args:
        n_samples: Total number of samples (divided equally among 4 classes)
        noise: Gaussian noise std (added to integer winding numbers)
        seed: Random seed for reproducibility

    Returns:
        (winding_states, labels) tuple
    """
    random.seed(seed)
    X = []
    y = []

    n_per_class = n_samples // 4

    # Class 0: (0,0)
    for _ in range(n_per_class):
        X.append(WindingState(
            0 + int(random.gauss(0, noise)),
            0 + int(random.gauss(0, noise)),
            0, 0
        ))
        y.append(0)

    # Class 0: (1,1) -> mapped to (2,2) for separation
    for _ in range(n_per_class):
        X.append(WindingState(
            2 + int(random.gauss(0, noise)),
            2 + int(random.gauss(0, noise)),
            0, 0
        ))
        y.append(0)

    # Class 1: (0,1) -> mapped to (0,2)
    for _ in range(n_per_class):
        X.append(WindingState(
            0 + int(random.gauss(0, noise)),
            2 + int(random.gauss(0, noise)),
            0, 0
        ))
        y.append(1)

    # Class 1: (1,0) -> mapped to (2,0)
    for _ in range(n_per_class):
        X.append(WindingState(
            2 + int(random.gauss(0, noise)),
            0 + int(random.gauss(0, noise)),
            0, 0
        ))
        y.append(1)

    # Shuffle
    combined = list(zip(X, y))
    random.shuffle(combined)
    X, y = zip(*combined)

    return list(X), list(y)


# =============================================================================
# 2. Particle Classification Dataset (Leptons vs Quarks)
# =============================================================================

def load_particle_dataset() -> Tuple[List[WindingState], List[int], List[str]]:
    """
    Load particle classification dataset: leptons vs quarks.

    Uses real fermion winding states from SRT:
    - Leptons (Class 0): electron, muon, tau
    - Quarks (Class 1): up, down, charm, strange, top, bottom

    Returns:
        (winding_states, labels, names) tuple
        - winding_states: List of 9 WindingState objects
        - labels: List of 9 class labels (0=lepton, 1=quark)
        - names: List of 9 particle names for debugging
    """
    # Lepton windings (Class 0)
    leptons = [
        (WindingState(0, 0, 0, 0), 0, 'electron'),  # e: vacuum state
        (WindingState(0, 1, 0, 0), 0, 'muon'),      # μ: |n|²=1
        (WindingState(1, 0, 0, 0), 0, 'tau'),       # τ: |n|²=1
    ]

    # Quark windings (Class 1)
    quarks = [
        (WindingState(1, 1, 0, 0), 1, 'up'),        # u: |n|²=2
        (WindingState(1, 0, 0, 0), 1, 'down'),      # d: |n|²=1
        (WindingState(1, 1, 1, 0), 1, 'charm'),     # c: |n|²=3
        (WindingState(1, 1, 0, 0), 1, 'strange'),   # s: |n|²=2
        (WindingState(2, 1, 1, 0), 1, 'top'),       # t: |n|²=6
        (WindingState(1, 1, 1, 0), 1, 'bottom'),    # b: |n|²=3
    ]

    all_particles = leptons + quarks
    winding_states = [w for w, _, _ in all_particles]
    labels = [label for _, label, _ in all_particles]
    names = [name for _, _, name in all_particles]

    return winding_states, labels, names


# =============================================================================
# 3. CSV Winding Dataset
# =============================================================================

def load_csv_dataset(
    csv_path: str = None,
    train_split: float = 0.8,
    seed: int = 42
) -> Tuple[
    List[WindingState], List[int],  # Train
    List[WindingState], List[int]   # Test
]:
    """
    Load winding dataset from CSV file.

    CSV format:
        n7,n8,target
        0,0,1
        0,1,0
        ...

    Args:
        csv_path: Path to CSV file (default: data/winding_dataset.csv)
        train_split: Fraction of data for training
        seed: Random seed for train/test split

    Returns:
        (X_train, y_train, X_test, y_test) tuple
    """
    if csv_path is None:
        # Default to data/winding_dataset.csv relative to project root
        project_root = Path(__file__).parent.parent.parent.parent.parent
        csv_path = project_root / "data" / "winding_dataset.csv"
    else:
        csv_path = Path(csv_path)

    if not csv_path.exists():
        raise FileNotFoundError(f"CSV dataset not found: {csv_path}")

    # Load CSV
    winding_states = []
    labels = []

    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Skip comment rows
            if row['n7'].startswith('#'):
                continue
            n7 = int(row['n7'])
            n8 = int(row['n8'])
            target = int(row['target'])

            # Create winding state (only use n7, n8 dimensions)
            winding_states.append(WindingState(n7, n8, 0, 0))
            labels.append(target)

    # Train/test split
    random.seed(seed)
    combined = list(zip(winding_states, labels))
    random.shuffle(combined)

    split_idx = int(len(combined) * train_split)
    train_data = combined[:split_idx]
    test_data = combined[split_idx:]

    X_train = [w for w, _ in train_data]
    y_train = [label for _, label in train_data]
    X_test = [w for w, _ in test_data]
    y_test = [label for _, label in test_data]

    return X_train, y_train, X_test, y_test


# =============================================================================
# Utility Functions
# =============================================================================

def train_test_split(
    X: List[WindingState],
    y: List[int],
    test_size: float = 0.2,
    seed: int = 42
) -> Tuple[
    List[WindingState], List[int],  # Train
    List[WindingState], List[int]   # Test
]:
    """
    Split dataset into train and test sets.

    Args:
        X: Winding states
        y: Labels
        test_size: Fraction of data for testing
        seed: Random seed

    Returns:
        (X_train, y_train, X_test, y_test) tuple
    """
    random.seed(seed)
    combined = list(zip(X, y))
    random.shuffle(combined)

    split_idx = int(len(combined) * (1 - test_size))
    train_data = combined[:split_idx]
    test_data = combined[split_idx:]

    X_train = [w for w, _ in train_data]
    y_train = [label for _, label in train_data]
    X_test = [w for w, _ in test_data]
    y_test = [label for _, label in test_data]

    return X_train, y_train, X_test, y_test


__all__ = [
    'load_winding_xor_dataset',
    'load_particle_dataset',
    'load_csv_dataset',
    'train_test_split',
]
