# Hierarchy Constants

Complete Lie group dimensional constants for SRT physics implementations.

## Exposed Constants

The following constants are now exposed from the Rust backend:

### E₈ Exceptional Group Family

| Constant | Value | Description | Python Access |
|----------|-------|-------------|---------------|
| `E8_DIM` | 248 | Adjoint representation dimension | `syntonic.crt.hierarchy_e8_dim()` |
| `E8_ROOTS` | 240 | Number of root vectors | `syntonic.crt.hierarchy_e8_roots()` |
| `E8_POSITIVE_ROOTS` | 120 | Positive roots (half total) | `syntonic.crt.hierarchy_e8_positive_roots()` |
| `E8_RANK` | 8 | Cartan subalgebra dimension | `syntonic.crt.hierarchy_e8_rank()` |
| `E8_COXETER` | 30 | Coxeter number | `syntonic.crt.hierarchy_e8_coxeter()` |

### E₇ Intermediate Unification Family

| Constant | Value | Description | Python Access |
|----------|-------|-------------|---------------|
| `E7_DIM` | 133 | Adjoint representation dimension | `syntonic.crt.hierarchy_e7_dim()` |
| `E7_ROOTS` | 126 | Number of root vectors | `syntonic.crt.hierarchy_e7_roots()` |
| `E7_POSITIVE_ROOTS` | 63 | Positive roots | `syntonic.crt.hierarchy_e7_positive_roots()` |
| `E7_FUNDAMENTAL` | 56 | Fundamental representation | `syntonic.crt.hierarchy_e7_fundamental()` |
| `E7_RANK` | 7 | Cartan subalgebra dimension | `syntonic.crt.hierarchy_e7_rank()` |
| `E7_COXETER` | 18 | Coxeter number | `syntonic.crt.hierarchy_e7_coxeter()` |

### E₆ Golden Cone Family

| Constant | Value | Description | Python Access |
|----------|-------|-------------|---------------|
| `E6_DIM` | 78 | Adjoint representation dimension | `syntonic.crt.hierarchy_e6_dim()` |
| `E6_ROOTS` | 72 | Number of root vectors | `syntonic.crt.hierarchy_e6_roots()` |
| `E6_POSITIVE_ROOTS` | 36 | **Golden Cone cardinality Φ⁺(E₆)** | `syntonic.crt.hierarchy_e6_positive_roots()` |
| `E6_FUNDAMENTAL` | 27 | Fundamental representation | `syntonic.crt.hierarchy_e6_fundamental()` |
| `E6_RANK` | 6 | Cartan subalgebra dimension | `syntonic.crt.hierarchy_e6_rank()` |
| `E6_COXETER` | 12 | Coxeter number | `syntonic.crt.hierarchy_e6_coxeter()` |

### D₄ Triality Family

| Constant | Value | Description | Python Access |
|----------|-------|-------------|---------------|
| `D4_DIM` | 28 | Adjoint representation dimension | `syntonic.crt.hierarchy_d4_dim()` |
| `D4_KISSING` | 24 | **Consciousness threshold K(D₄)** | `syntonic.crt.hierarchy_d4_kissing()` |
| `D4_RANK` | 4 | Cartan subalgebra dimension | `syntonic.crt.hierarchy_d4_rank()` |
| `D4_COXETER` | 6 | Coxeter number | `syntonic.crt.hierarchy_d4_coxeter()` |

### G₂ Octonion Family

| Constant | Value | Description | Python Access |
|----------|-------|-------------|---------------|
| `G2_DIM` | 14 | Adjoint representation dimension | `syntonic.crt.hierarchy_g2_dim()` |
| `G2_RANK` | 2 | Cartan subalgebra dimension | `syntonic.crt.hierarchy_g2_rank()` |

### F₄ Jordan Algebra Family

| Constant | Value | Description | Python Access |
|----------|-------|-------------|---------------|
| `F4_DIM` | 52 | Adjoint representation dimension | `syntonic.crt.hierarchy_f4_dim()` |
| `F4_RANK` | 4 | Cartan subalgebra dimension | `syntonic.crt.hierarchy_f4_rank()` |

## Derived Quantities

| Constant | Value | Description | Python Access |
|----------|-------|-------------|---------------|
| `COXETER_KISSING_720` | 720 | E₈_COXETER × D₄_KISSING | `syntonic.crt.hierarchy_coxeter_kissing_720()` |
| `HIERARCHY_EXPONENT` | 719 | Coxeter-Kissing minus 1 | `syntonic.crt.hierarchy_exponent()` |

## SRT Constants Module

Constants are also available through the SRT module:

```{eval-rst}
.. automodule:: syntonic.srt.constants
   :members:
   :undoc-members:
   :show-inheritance:
```

### Key SRT Hierarchy Constants

```python
from syntonic.srt.constants import (
    E8_ROOTS, E8_POSITIVE_ROOTS, E8_RANK,
    E7_ROOTS, E7_POSITIVE_ROOTS, E7_FUNDAMENTAL,
    E6_POSITIVE_ROOTS,  # Golden Cone = 36
    D4_KISSING,         # Consciousness threshold = 24
    FIBONACCI_PRIMES    # [2, 3, 5, 13, 89, ...]
)
```

## CRT Structure Dimensions

Complete dimensional dictionary:

```{eval-rst}
.. automodule:: syntonic.crt
   :members: STRUCTURE_DIMENSIONS
   :undoc-members:
```

## Physical Applications

### Neural Network Architecture

| Constant | Use Case | Example |
|----------|----------|---------|
| `E8_DIM = 248` | Optimal embedding dimension | `nn.Linear(248, hidden)` |
| `D4_KISSING = 24` | Stable attention heads | `MultiHeadAttention(24)` |
| `E6_POSITIVE_ROOTS = 36` | Golden Cone layers | `nn.Linear(hidden, 36)` |

### Physics Computations

| Constant | Use Case | Example |
|----------|----------|---------|
| `E8_ROOTS = 240` | Gauge symmetry breaking | Mass scale corrections |
| `E6_POSITIVE_ROOTS = 36` | Golden Cone cardinality | Unification predictions |
| `D4_KISSING = 24` | Consciousness threshold | Emergence calculations |

### Fibonacci Prime Gates

```python
from syntonic.srt.constants import FIBONACCI_PRIMES

# Access Fibonacci primes for transcendence gates
fib_primes = FIBONACCI_PRIMES  # [2, 3, 5, 13, 89, 233, ...]
```

## Correction Functions

```{eval-rst}
.. automodule:: syntonic.crt
   :members:
   :undoc-members:
   :show-inheritance:
```

## Example Usage

```python
# Access hierarchy constants
from syntonic.crt import STRUCTURE_DIMENSIONS
from syntonic.srt.constants import E8_ROOTS, D4_KISSING, FIBONACCI_PRIMES

# E8 exceptional group properties
e8_roots = STRUCTURE_DIMENSIONS['E8_ROOTS']  # 240
e8_rank = STRUCTURE_DIMENSIONS['E8_RANK']    # 8

# Consciousness threshold (D4 kissing number)
consciousness_threshold = D4_KISSING  # 24

# Fibonacci transcendence gates
transcendence_gates = FIBONACCI_PRIMES[:5]  # [2, 3, 5, 13, 89]

# Apply hierarchy corrections
from syntonic.crt import hierarchy_apply_correction

mass_corrected = hierarchy_apply_correction(
    electron_mass_mev,
    divisors=[E8_ROOTS, D4_KISSING],
    signs=[1, -1]
)
```
