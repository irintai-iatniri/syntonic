# Syntonic API Reference

**Version:** 0.1.0

Syntonic is a tensor library for Cosmological Recursion Theory (CRT) and Syntony Recursion Theory (SRT). It provides tensor operations and state evolution primitives for the DHSR (Differentiation-Harmonization-Syntony-Recursion) framework.

---

## Quick Start

```python
import syntonic as syn

# Create a state
psi = syn.state([1, 2, 3, 4])
print(psi.shape)    # (4,)
print(psi.syntony)  # Syntony index S(Psi) in [0, 1]

# DHSR cycle
evolved = psi.differentiate().harmonize()
```

---

## Table of Contents

- [State Class](#state-class)
- [State Factory Functions](#state-factory-functions)
- [Data Types](#data-types)
- [Device Management](#device-management)
- [Linear Algebra (syn.linalg)](#linear-algebra)
- [Hypercomplex Numbers (syn.hypercomplex)](#hypercomplex-numbers)
- [Exact Arithmetic (syn.exact)](#exact-arithmetic)
- [CRT Core (syn.crt)](#crt-core)
- [Exceptions](#exceptions)

---

## State Class

The `State` class is the fundamental object in Syntonic, representing information configurations that evolve through DHSR cycles.

### Constructor

```python
syn.State(data=None, *, dtype=None, device=None, shape=None)
syn.state(data=None, *, dtype=None, device=None, **kwargs)  # Preferred factory
```

**Parameters:**
- `data` - Initial data (list, nested list, or another State)
- `dtype` - Data type (default: `float64` for real, `complex128` for complex)
- `device` - Computation device (default: `cpu`)
- `shape` - Explicit shape (required if `data` is None)

**Examples:**
```python
# From list
psi = syn.state([1, 2, 3, 4])

# 2D state
matrix = syn.state([[1, 2], [3, 4]])

# Complex state
c = syn.state([1+2j, 3+4j])

# With explicit dtype
f32 = syn.state([1, 2, 3], dtype=syn.float32)
```

### Properties

| Property | Type | Description |
|----------|------|-------------|
| `shape` | `tuple[int, ...]` | Dimensions of the state |
| `ndim` | `int` | Number of dimensions |
| `size` | `int` | Total number of elements |
| `dtype` | `DType` | Data type |
| `device` | `Device` | Storage device |
| `syntony` | `float` | Syntony index S(Psi) in [0, 1] |
| `gnosis` | `int` | Gnosis layer (0-3) |
| `free_energy` | `float` | Free energy F[rho] |
| `T` | `State` | Transpose |
| `H` | `State` | Conjugate transpose (Hermitian adjoint) |

### DHSR Operations

These are the core operations for CRT/SRT state evolution.

#### `differentiate(alpha=0.1)`

Apply differentiation operator D-hat. Increases complexity with syntony-dependent coupling.

```python
psi_d = psi.differentiate(alpha=0.1)
```

**Parameters:**
- `alpha` - Base differentiation strength (default: 0.1)

#### `harmonize(strength=0.618, gamma=0.0)`

Apply harmonization operator H-hat. Projects toward Golden Measure equilibrium.

```python
psi_h = psi.harmonize(strength=0.618)
```

**Parameters:**
- `strength` - Harmonization strength (default: 1/phi = 0.618)
- `gamma` - Additional gamma parameter (default: 0.0)

#### `recurse(alpha=0.1, strength=0.618)`

Apply recursion operator R-hat = H-hat compose D-hat. Performs one complete DHSR cycle.

```python
psi_next = psi.recurse()
```

### Arithmetic Operations

States support standard arithmetic with other States or scalars:

```python
# Addition/Subtraction
c = a + b
c = a - b
c = a + 2.0

# Multiplication/Division
c = a * b      # Element-wise
c = a * 3.0    # Scalar
c = a / b

# Negation
c = -a

# Matrix multiplication
c = a @ b

# Power
c = a ** 2
```

### Reduction Operations

#### `norm(ord=2)`

Compute vector/matrix norm.

```python
n = psi.norm()      # L2 norm (default)
n = psi.norm(ord=1) # L1 norm
```

#### `normalize()`

Return unit-normalized state.

```python
psi_unit = psi.normalize()
```

#### `sum(axis=None)`

Sum elements, optionally along an axis.

```python
total = psi.sum()
row_sums = matrix.sum(axis=1)
```

#### `mean(axis=None)`

Mean of elements.

#### `max(axis=None)` / `min(axis=None)`

Maximum/minimum elements.

#### `abs()`

Element-wise absolute value.

```python
magnitudes = psi.abs()
```

### Complex Operations

#### `conj()`

Complex conjugate.

#### `real()` / `imag()`

Extract real or imaginary parts.

```python
re = c.real()
im = c.imag()
```

### Shape Operations

#### `reshape(*shape)`

Reshape state to new dimensions.

```python
matrix = psi.reshape(2, 2)
flat = matrix.reshape(-1)  # -1 infers dimension
```

#### `flatten()`

Flatten to 1D.

#### `squeeze()`

Remove dimensions of size 1.

#### `unsqueeze(dim)`

Add dimension of size 1 at position `dim`.

### Device Transfer

#### `cuda(device_id=0)`

Move state to CUDA device.

```python
if syn.cuda_is_available():
    psi_gpu = psi.cuda()
```

#### `cpu()`

Move state to CPU.

#### `to(device)`

Move to specified device.

```python
psi_gpu = psi.to(syn.cuda(0))
```

### Conversion Methods

#### `to_list()`

Convert to flat Python list.

#### `tolist()`

Convert to nested Python list matching shape.

#### `numpy()`

Convert to NumPy array (requires numpy).

```python
arr = psi.numpy()
```

#### `torch()`

Convert to PyTorch tensor (requires torch).

### Indexing

States support standard Python indexing:

```python
psi[0]       # First element (returns State)
psi[1:3]     # Slice
matrix[0]    # First row
matrix[0, 1] # Element at row 0, col 1
```

---

## State Factory Functions

Factory methods are available on the `state` function:

### `syn.state.zeros(shape, *, dtype=float64, device=cpu)`

Create zero-filled state.

```python
z = syn.state.zeros((3, 3))
```

### `syn.state.ones(shape, *, dtype=float64, device=cpu)`

Create state filled with ones.

```python
o = syn.state.ones((2, 4))
```

### `syn.state.random(shape, *, dtype=float64, device=cpu, seed=None)`

Create random state with uniform values in [0, 1].

```python
r = syn.state.random((100,), seed=42)
```

### `syn.state.randn(shape, *, dtype=float64, device=cpu, seed=None)`

Create random state with standard normal distribution.

```python
n = syn.state.randn((10, 10))
```

### `syn.state.eye(n, *, dtype=float64, device=cpu)`

Create identity matrix.

```python
I = syn.state.eye(4)
```

### `syn.state.from_numpy(arr, **kwargs)`

Create State from NumPy array.

```python
import numpy as np
arr = np.array([[1, 2], [3, 4]])
psi = syn.state.from_numpy(arr)
```

### `syn.state.from_torch(tensor, **kwargs)`

Create State from PyTorch tensor.

### `syn.state.from_list(data, shape, **kwargs)`

Create State from flat list with explicit shape.

```python
psi = syn.state.from_list([1, 2, 3, 4, 5, 6], shape=(2, 3))
```

---

## Data Types

Syntonic provides the following data types:

| Type | Description | Size |
|------|-------------|------|
| `syn.float32` | 32-bit float | 4 bytes |
| `syn.float64` | 64-bit float (default) | 8 bytes |
| `syn.complex64` | 64-bit complex | 8 bytes |
| `syn.complex128` | 128-bit complex (default for complex) | 16 bytes |
| `syn.int32` | 32-bit integer | 4 bytes |
| `syn.int64` | 64-bit integer | 8 bytes |
| `syn.winding` | Winding number type (alias for int64) | 8 bytes |

### DType Class

```python
@dataclass(frozen=True)
class DType:
    name: str
    numpy_dtype: np.dtype
    size: int
    is_complex: bool = False
    is_floating: bool = True
```

### Helper Functions

#### `syn.get_dtype(dtype_spec)`

Convert string or numpy dtype to Syntonic DType.

```python
dt = syn.get_dtype('float32')
dt = syn.get_dtype('complex')  # Returns complex128
```

#### `syn.promote_dtypes(dtype1, dtype2)`

Determine result dtype from two input dtypes.

```python
result_dt = syn.promote_dtypes(syn.float32, syn.complex64)  # complex64
```

---

## Device Management

### Device Class

```python
@dataclass(frozen=True)
class Device:
    type: str       # 'cpu' or 'cuda'
    index: int | None = None
```

**Properties:**
- `name` - String representation (e.g., 'cpu', 'cuda:0')
- `is_cpu` - True if CPU device
- `is_cuda` - True if CUDA device

### Device Constants and Functions

#### `syn.cpu`

Singleton CPU device.

#### `syn.cuda(device_id=0)`

Get CUDA device by index.

```python
gpu0 = syn.cuda(0)
gpu1 = syn.cuda(1)
```

#### `syn.cuda_is_available()`

Check if CUDA is available.

```python
if syn.cuda_is_available():
    psi = psi.cuda()
```

#### `syn.cuda_device_count()`

Get number of available CUDA devices.

#### `syn.device(spec)`

Parse device from string.

```python
d = syn.device('cpu')
d = syn.device('cuda:0')
```

---

## Linear Algebra

The `syn.linalg` module provides linear algebra operations.

```python
import syntonic as syn
from syntonic import linalg
# or
import syntonic.linalg as la
```

### Basic Operations

#### `linalg.matmul(a, b)`

Matrix multiplication (equivalent to `a @ b`).

#### `linalg.dot(a, b)`

Dot product of flattened arrays.

```python
d = linalg.dot(a, b)
```

#### `linalg.inner(a, b)`

Inner product <a|b> (conjugates first argument for complex).

```python
ip = linalg.inner(psi, phi)  # <psi|phi>
```

#### `linalg.outer(a, b)`

Outer product |a><b|.

```python
op = linalg.outer(psi, phi)  # |psi><phi|
```

#### `linalg.norm(x, ord=None)`

Vector or matrix norm.

### Decompositions

#### `linalg.eig(a)`

Eigenvalue decomposition.

```python
eigenvalues, eigenvectors = linalg.eig(A)
```

**Returns:** `(eigenvalues, eigenvectors)` tuple of States

#### `linalg.eigh(a)`

Eigenvalue decomposition for Hermitian matrices.

```python
eigenvalues, eigenvectors = linalg.eigh(H)
```

**Returns:** `(eigenvalues, eigenvectors)` - eigenvalues are real

#### `linalg.svd(a, full_matrices=True)`

Singular value decomposition.

```python
U, S, Vh = linalg.svd(A)
```

**Returns:** `(U, S, Vh)` tuple of States

#### `linalg.qr(a)`

QR decomposition.

```python
Q, R = linalg.qr(A)
```

**Returns:** `(Q, R)` where A = QR

#### `linalg.cholesky(a)`

Cholesky decomposition for positive definite matrices.

```python
L = linalg.cholesky(A)  # A = LL*
```

### Matrix Operations

#### `linalg.inv(a)`

Matrix inverse.

```python
A_inv = linalg.inv(A)
```

#### `linalg.pinv(a)`

Moore-Penrose pseudo-inverse.

```python
A_pinv = linalg.pinv(A)
```

#### `linalg.det(a)`

Matrix determinant.

```python
d = linalg.det(A)
```

#### `linalg.trace(a)`

Matrix trace.

```python
t = linalg.trace(A)
```

#### `linalg.solve(a, b)`

Solve linear system Ax = b.

```python
x = linalg.solve(A, b)
```

### Matrix Functions (require scipy)

#### `linalg.expm(a)`

Matrix exponential exp(A). Important for CRT evolution operators.

```python
exp_H = linalg.expm(-1j * H * t)  # Time evolution
```

#### `linalg.logm(a)`

Matrix logarithm log(A).

```python
log_U = linalg.logm(U)
```

---

## Hypercomplex Numbers

The `syn.hypercomplex` module provides quaternions and octonions.

```python
import syntonic as syn
# Direct access
q = syn.quaternion(1, 2, 3, 4)
o = syn.octonion(1, 2, 3, 4, 5, 6, 7, 8)

# Via submodule
from syntonic import hypercomplex
q = hypercomplex.quaternion(1, 0, 0, 0)
```

### Quaternion

Quaternions are 4D hypercomplex numbers: q = a + bi + cj + dk

- Non-commutative: q1 * q2 != q2 * q1
- Associative: (q1 * q2) * q3 == q1 * (q2 * q3)
- Hamilton product: i^2 = j^2 = k^2 = ijk = -1

#### Factory Function

```python
syn.quaternion(a, b=0.0, c=0.0, d=0.0)
```

**Parameters:**
- `a` - Real (scalar) part
- `b` - i component
- `c` - j component
- `d` - k component

**Example:**
```python
q = syn.quaternion(1, 2, 3, 4)  # 1 + 2i + 3j + 4k
```

#### Quaternion Properties

| Property | Description |
|----------|-------------|
| `real` | Scalar part (a) |
| `i`, `j`, `k` | Vector components |

#### Quaternion Methods

| Method | Description |
|--------|-------------|
| `norm()` | Euclidean norm |
| `normalize()` | Return unit quaternion |
| `conjugate()` | Quaternion conjugate (a - bi - cj - dk) |
| `inverse()` | Multiplicative inverse |
| `to_rotation_matrix()` | Convert to 3x3 rotation matrix |
| `rotate_vector(v)` | Rotate a 3D vector |

#### Quaternion Basis Elements

```python
from syntonic.hypercomplex import I, J, K

# Verify quaternion algebra
assert I * J == K
assert J * K == I
assert K * I == J
assert I * I == syn.quaternion(-1)
```

### Octonion

Octonions are 8D hypercomplex numbers: o = e0 + e1*e1 + ... + e7*e7

- Non-commutative AND non-associative
- Cayley-Dickson construction from quaternions
- Related to exceptional Lie groups (G2, E8)

#### Factory Function

```python
syn.octonion(e0, e1=0.0, e2=0.0, e3=0.0, e4=0.0, e5=0.0, e6=0.0, e7=0.0)
```

**Example:**
```python
o = syn.octonion(1, 2, 3, 4, 5, 6, 7, 8)
```

#### Octonion Methods

| Method | Description |
|--------|-------------|
| `norm()` | Euclidean norm |
| `normalize()` | Return unit octonion |
| `conjugate()` | Octonion conjugate |
| `inverse()` | Multiplicative inverse |
| `real()` | Real (scalar) part |
| `associator(a, b, c)` | Measure non-associativity: (ab)c - a(bc) |

#### Octonion Basis Elements

```python
from syntonic.hypercomplex import E0, E1, E2, E3, E4, E5, E6, E7
```

---

## Exact Arithmetic

The `syn.exact` module provides exact arithmetic over the golden field Q(φ), enabling symbolic computation with golden ratio expressions.

```python
import syntonic as syn
from syntonic import exact

# Exact golden ratio arithmetic
phi = syn.PHI
phi_squared = phi * phi  # Exact: 1 + φ
```

### GoldenExact Class

Represents numbers of the form a + b·φ where a, b are rationals.

#### Constructor

```python
exact.GoldenExact(a, b=0)
```

**Parameters:**
- `a` - Rational coefficient of 1
- `b` - Rational coefficient of φ

**Example:**
```python
from syntonic.exact import GoldenExact, Rational

# Create 3 + 2φ
g = GoldenExact(Rational(3), Rational(2))

# Arithmetic preserves exactness
h = g * g  # Exact result
```

#### GoldenExact Methods

| Method | Description |
|--------|-------------|
| `to_float()` | Convert to floating point approximation |
| `conjugate()` | Galois conjugate (a + b·φ̄ where φ̄ = -1/φ) |
| `norm()` | Field norm N(x) = x · x̄ |

### Rational Class

Arbitrary precision rational numbers.

```python
from syntonic.exact import Rational

r = Rational(22, 7)  # 22/7
```

### Constants

The module provides both exact and numeric constants:

#### Exact Constants

| Constant | Value | Description |
|----------|-------|-------------|
| `syn.PHI` | GoldenExact | Exact golden ratio φ = (1 + √5)/2 |
| `syn.PHI_SQUARED` | GoldenExact | Exact φ² = φ + 1 |
| `syn.PHI_INVERSE` | GoldenExact | Exact 1/φ = φ - 1 |

#### Numeric Constants

| Constant | Value | Description |
|----------|-------|-------------|
| `syn.PHI_NUMERIC` | 1.6180339... | Float approximation of φ |
| `syn.E_STAR_NUMERIC` | 19.999099... | e^π - π ≈ 20 (SRT fundamental) |
| `syn.Q_DEFICIT_NUMERIC` | 9.00901×10⁻⁴ | Syntony deficit q = 20 - E* |

#### Structure Dimensions

```python
syn.STRUCTURE_DIMENSIONS
# {'E8': 248, 'D4': 24, 'torus_S1': 1, 'torus_T2': 2, ...}
```

### Sequence Functions

#### `syn.fibonacci(n)`

Compute the n-th Fibonacci number.

```python
fib_10 = syn.fibonacci(10)  # 55
```

#### `syn.lucas(n)`

Compute the n-th Lucas number.

```python
luc_10 = syn.lucas(10)  # 123
```

### Correction Factors

#### `syn.correction_factor(structure, sign=1)`

Compute SRT correction factors for various structures.

```python
# E8 correction factor
c_e8 = syn.correction_factor('E8', sign=1)
```

**Parameters:**
- `structure` - One of: 'E8', 'D4', 'torus', 'full'
- `sign` - +1 or -1 for positive/negative correction

#### `syn.golden_number(a, b)`

Create an exact golden number a + b·φ from integers.

```python
g = syn.golden_number(3, 2)  # 3 + 2φ
```

---

## CRT Core

The `syn.crt` module provides the complete DHSR (Differentiation-Harmonization-Syntony-Recursion) operator framework for Cosmological Recursion Theory.

```python
import syntonic as syn
from syntonic import crt
```

### Quick Start

```python
from syntonic.crt import create_dhsr_system, DHSREvolver

# Create complete DHSR system
R_op, S_comp, G_comp = create_dhsr_system()

# Apply recursion operator
psi = syn.state([1, 2, 3, 4, 5, 6, 7, 8])
evolved = R_op.apply(psi)

# Compute metrics
syntony = S_comp.compute(psi)      # S(Ψ) in [0, 1]
gnosis = G_comp.compute_layer(psi)  # Layer 0-3

# Full evolution with trajectory tracking
evolver = DHSREvolver()
trajectory = evolver.evolve(psi, n_steps=100)
```

### Operators

#### DifferentiationOperator (D̂)

Applies differentiation: D̂[Ψ] = Ψ + Σᵢ αᵢ(S) P̂ᵢ[Ψ] + ζ(S) ∇²[Ψ]

```python
from syntonic.crt import DifferentiationOperator

D_op = DifferentiationOperator(
    alpha_0=0.1,     # Base coupling strength
    zeta_0=0.01,     # Laplacian coupling
    num_modes=8,     # Number of Fourier modes
)

d_state = D_op.apply(psi, syntony=0.5)
mag = D_op.differentiation_magnitude(psi)
```

#### HarmonizationOperator (Ĥ)

Applies harmonization: Ĥ[Ψ] = Ψ - Σᵢ βᵢ(S,Δ_D) Q̂ᵢ[Ψ] + γ(S) Ŝ_op[Ψ]

```python
from syntonic.crt import HarmonizationOperator

H_op = HarmonizationOperator(
    beta_0=0.618,    # Base damping (1/φ)
    gamma_0=0.1,     # Nonlinear coupling
    num_dampers=3,   # Damping cascade levels
)

h_state = H_op.apply(psi, syntony=0.5, delta_d=0.1)
mag = H_op.harmonization_magnitude(psi)
```

#### RecursionOperator (R̂)

Combined operator R̂ = Ĥ ∘ D̂

```python
from syntonic.crt import RecursionOperator

R_op = RecursionOperator(diff_op=D_op, harm_op=H_op)

# Single step
evolved = R_op.apply(psi)

# With detailed info
evolved, info = R_op.apply_with_info(psi)
# info contains: diff_magnitude, harm_magnitude, total_change, d_state

# Multiple iterations
trajectory = R_op.iterate(psi, n_steps=10)  # Returns list of states

# Find fixed point
fixed, n_iters, converged = R_op.find_fixed_point(psi, tol=1e-6, max_iter=100)
```

### Projectors

#### FourierProjector

Projects onto Fourier modes.

```python
from syntonic.crt import FourierProjector

proj = FourierProjector(mode_indices=[0, 1, 2], size=64)
projected = proj.project(psi)
```

#### DampingProjector

Low-pass damping filter.

```python
from syntonic.crt import DampingProjector

damper = DampingProjector(cutoff_fraction=0.5, order=4)
damped = damper.project(psi)
```

#### LaplacianOperator

Discrete Laplacian ∇².

```python
from syntonic.crt import LaplacianOperator

laplacian = LaplacianOperator(boundary='periodic')
lap_psi = laplacian.apply(psi)
```

### Metrics

#### SyntonyComputer

Full syntony computation: S(Ψ) = 1 - ||D̂[Ψ] - Ĥ[D̂[Ψ]]|| / (||D̂[Ψ] - Ψ|| + ε)

```python
from syntonic.crt import SyntonyComputer

S_comp = SyntonyComputer(D_op, H_op)

S = S_comp.compute(psi)  # Returns float in [0, 1]

# Detailed components
result = S_comp.compute_components(psi)
# result contains: syntony, diff_magnitude, residual, d_state, hd_state
```

#### Quick Syntony Estimates

For faster computation without full operator application:

```python
from syntonic.crt import syntony_entropy, syntony_spectral, syntony_quick

S_ent = syntony_entropy(psi)    # Entropy-based estimate
S_spec = syntony_spectral(psi)  # Spectral-based estimate
S_quick = syntony_quick(psi)    # Combined quick estimate
```

#### GnosisComputer

Gnosis layer classification based on accumulated phase.

```python
from syntonic.crt import GnosisComputer, K_D4

G_comp = GnosisComputer()

layer = G_comp.compute_layer(psi)  # Returns 0, 1, 2, or 3
name = G_comp.layer_name(layer)    # 'nascent', 'emergent', 'coherent', 'transcendent'

# Detailed info
layer, progress = G_comp.layer_progress(psi)  # progress in [0, 1]
T = G_comp.transcendence_metric(psi)
cycles = G_comp.k_d4_cycles(psi)  # K(D4) = 24 cycle count

desc = G_comp.describe(psi)  # Human-readable description
```

**Gnosis Layers:**

| Layer | Name | Phase Threshold |
|-------|------|-----------------|
| 0 | nascent | θ < π |
| 1 | emergent | π ≤ θ < 2π |
| 2 | coherent | 2π ≤ θ < 3π |
| 3 | transcendent | θ ≥ 3π |

### Evolution

#### SyntonyTrajectory

Dataclass for tracking evolution trajectories.

```python
from syntonic.crt import SyntonyTrajectory

# Properties
traj.n_steps           # Number of evolution steps
traj.states            # List of State objects
traj.syntony_values    # List of S(Ψ) values
traj.gnosis_values     # List of gnosis layers
traj.phase_values      # List of accumulated phases
traj.change_magnitudes # List of ||Ψ_{n+1} - Ψ_n||

traj.initial_state / traj.final_state
traj.initial_syntony / traj.final_syntony
traj.syntony_delta     # Final - Initial syntony
traj.syntony_trend     # 'increasing', 'decreasing', or 'stable'
traj.converged         # True if final change < 1e-6

traj.summary()         # Human-readable summary string
```

#### DHSREvolver

Full evolution engine with trajectory tracking.

```python
from syntonic.crt import DHSREvolver

evolver = DHSREvolver(
    recursion_op=R_op,       # Optional, creates default if None
    syntony_computer=S_comp,
    gnosis_computer=G_comp,
)

# Basic evolution
traj = evolver.evolve(psi, n_steps=100)

# With early stopping
traj = evolver.evolve(psi, n_steps=1000, early_stop=True, tol=1e-6)

# Find attractor
attractor, traj = evolver.find_attractor(psi, tol=1e-4, max_iter=500)

# Stability analysis
analysis = evolver.analyze_stability(
    psi,
    perturbation_scale=0.01,
    n_perturbations=10,
    n_steps=50,
)
# analysis contains: base_syntony, mean_final_syntony, syntony_variance,
#                    stable, convergence_rate

# Find all attractors from multiple initial states
attractors = evolver.find_all_attractors(
    initial_states,
    tol=1e-4,
    cluster_tol=0.1,
)
# Returns list of (attractor_state, count) tuples
```

### Factory Functions

#### `create_dhsr_system(**kwargs)`

Create a complete DHSR system with matched operators.

```python
from syntonic.crt import create_dhsr_system

R_op, S_comp, G_comp = create_dhsr_system(
    alpha_0=0.1,    # Differentiation strength
    beta_0=0.618,   # Harmonization strength (1/φ)
    zeta_0=0.01,    # Laplacian coupling
    gamma_0=0.1,    # Nonlinear coupling
    num_modes=8,    # Fourier modes
    num_dampers=3,  # Damping levels
)
```

#### `create_evolver(**kwargs)`

Create a configured evolver in one call.

```python
from syntonic.crt import create_evolver

evolver = create_evolver(alpha_0=0.1, beta_0=0.618)
traj = evolver.evolve(psi, n_steps=100)
```

#### `create_mode_projectors(size, num_modes)`

Create a list of Fourier mode projectors.

```python
from syntonic.crt import create_mode_projectors

projectors = create_mode_projectors(size=64, num_modes=8)
```

#### `create_damping_cascade(num_levels)`

Create a cascade of damping projectors with decreasing cutoffs.

```python
from syntonic.crt import create_damping_cascade

dampers = create_damping_cascade(num_levels=4)
# dampers[0].cutoff_fraction > dampers[1].cutoff_fraction > ...
```

---

## Exceptions

All Syntonic exceptions inherit from `SyntonicError`.

```python
from syntonic import (
    SyntonicError,
    DeviceError,
    DTypeError,
    ShapeError,
    LinAlgError,
)
```

| Exception | Description |
|-----------|-------------|
| `SyntonicError` | Base exception for all Syntonic errors |
| `DeviceError` | Device-related operations (CUDA unavailable, transfer failed) |
| `DTypeError` | Data type operations (incompatible types) |
| `ShapeError` | Tensor shape operations (broadcast failure, reshape error) |
| `LinAlgError` | Linear algebra operations (singular matrix, dimension mismatch) |

**Example:**
```python
from syntonic import LinAlgError

try:
    x = linalg.solve(singular_matrix, b)
except LinAlgError as e:
    print(f"Linear algebra error: {e}")
```

---

## Full Example: DHSR Evolution

### Basic Evolution (Quick Methods)

```python
import syntonic as syn
from syntonic import linalg

# Create initial state
psi = syn.state.randn((64,), dtype=syn.complex128, seed=42)
psi = psi.normalize()

# Evolve through DHSR cycles using quick methods
trajectory = [psi]
for _ in range(100):
    psi = psi.recurse(alpha=0.1, strength=0.618)
    trajectory.append(psi)

# Check convergence via syntony
print(f"Initial syntony: {trajectory[0].syntony:.4f}")
print(f"Final syntony: {trajectory[-1].syntony:.4f}")

# Analyze with linear algebra
final = trajectory[-1].reshape(8, 8)
eigenvalues, eigenvectors = linalg.eigh(final.H @ final)
print(f"Spectral structure: {eigenvalues.to_list()[:3]}")
```

### Full CRT Evolution (syn.crt Module)

```python
import syntonic as syn
from syntonic.crt import (
    create_dhsr_system,
    DHSREvolver,
    create_evolver,
)

# Create complete DHSR system with golden ratio parameters
R_op, S_comp, G_comp = create_dhsr_system(
    alpha_0=0.1,     # Differentiation strength
    beta_0=0.618,    # Harmonization strength (1/φ)
)

# Create initial state
psi = syn.state([1, 2, 3, 4, 5, 6, 7, 8])

# Compute initial metrics
print(f"Initial Syntony: {S_comp.compute(psi):.4f}")
print(f"Initial Gnosis Layer: {G_comp.layer_name(G_comp.compute_layer(psi))}")

# Create evolver and run evolution
evolver = DHSREvolver(
    recursion_op=R_op,
    syntony_computer=S_comp,
    gnosis_computer=G_comp,
)

# Evolve with trajectory tracking
trajectory = evolver.evolve(psi, n_steps=100, early_stop=True, tol=1e-6)

# Analyze trajectory
print(f"\nEvolution Summary:")
print(f"  Steps taken: {trajectory.n_steps}")
print(f"  Syntony: {trajectory.initial_syntony:.4f} -> {trajectory.final_syntony:.4f}")
print(f"  Trend: {trajectory.syntony_trend}")
print(f"  Converged: {trajectory.converged}")
print(trajectory.summary())

# Find attractor
attractor, traj = evolver.find_attractor(psi, tol=1e-4, max_iter=200)
print(f"\nAttractor found after {traj.n_steps} steps")
print(f"Attractor syntony: {S_comp.compute(attractor):.4f}")

# Stability analysis
stability = evolver.analyze_stability(
    psi,
    perturbation_scale=0.01,
    n_perturbations=5,
    n_steps=50,
)
print(f"\nStability Analysis:")
print(f"  Stable: {stability['stable']}")
print(f"  Convergence rate: {stability['convergence_rate']:.4f}")
```

---

## Backend Information

Syntonic uses a Rust backend (`syntonic._core`) for high-performance tensor operations. The Python layer is NumPy-free for core operations, with optional NumPy/SciPy integration for advanced features like `expm` and `logm`.

CUDA support is provided through the Rust backend when available. Check availability with `syn.cuda_is_available()`.
