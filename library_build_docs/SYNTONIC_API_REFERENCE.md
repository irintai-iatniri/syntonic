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
- [SRT Module (syn.srt)](#srt-module)
- [Applications (syn.applications)](#applications)
- [Neural Networks (syn.nn)](#neural-networks)
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

## SRT Module

The `syn.srt` module provides the complete Syntony Recursion Theory framework, implementing winding states on T⁴, spectral operations, golden measure weighting, and lattice structures.

```python
import syntonic as syn
from syntonic import srt

# Or import specific submodules
from syntonic.srt import geometry, spectral, golden, lattice, functional
```

### Winding States (syn.srt.geometry)

Winding states |n⟩ = |n₇, n₈, n₉, n₁₀⟩ are configurations of winding numbers on the internal 4-torus T⁴. These are the fundamental quantum numbers in SRT.

#### WindingState Class

The `WindingState` class is implemented in Rust for high performance (~50x faster enumeration).

```python
from syntonic.srt.geometry import WindingState, winding_state

# Create winding states
n = winding_state(1, 2, 0, -1)  # |1, 2, 0, -1⟩
vacuum = winding_state(0, 0, 0, 0)  # |0, 0, 0, 0⟩
```

**Properties:**

| Property | Type | Description |
|----------|------|-------------|
| `n7`, `n8`, `n9`, `n10` | `int` | Individual winding components |
| `n` | `tuple` | Components as (n7, n8, n9, n10) tuple |
| `norm_squared` | `int` | |n|² = n₇² + n₈² + n₉² + n₁₀² |
| `norm` | `float` | |n| = √(n₇² + n₈² + n₉² + n₁₀²) |
| `generation` | `int` | Generation number (1, 2, or 3) |

**Methods:**

| Method | Description |
|--------|-------------|
| `is_zero()` | Returns True if vacuum state |0,0,0,0⟩ |
| `golden_weight(phi=None)` | Compute w(n) = exp(-|n|²/φ) |
| `to_tuple()` | Convert to (n7, n8, n9, n10) tuple |

#### Enumeration Functions

```python
from syntonic.srt.geometry import (
    enumerate_windings,
    enumerate_windings_by_norm,
    enumerate_windings_exact_norm,
    count_windings,
)

# Enumerate all states with |n| ≤ 3
states = enumerate_windings(max_norm=3)

# Group by norm squared
by_norm = enumerate_windings_by_norm(max_norm_sq=10)
# by_norm[1] = [|1,0,0,0⟩, |0,1,0,0⟩, ...]  (8 states)
# by_norm[2] = [|1,1,0,0⟩, |1,0,1,0⟩, ...]  (24 states)

# Get only states with exact |n|² = 4
exact = enumerate_windings_exact_norm(norm_sq=4)

# Count without enumerating
count = count_windings(max_norm=5)
```

### Spectral Operations (syn.srt.spectral)

Spectral operators on the winding lattice, including theta series, heat kernels, and the knot Laplacian.

#### ThetaSeries

The theta series Θ₄(t) = Σₙ w(n) · exp(-π|n|²/t) on the 4-torus.

```python
from syntonic.srt.spectral import theta_series, ThetaSeries

# Create theta series
theta = theta_series(max_norm=20)

# Evaluate at t
value = theta.evaluate(1.0)  # Θ₄(1) at self-dual point

# Without golden weight (standard Jacobi theta)
standard = theta.evaluate_without_golden(1.0)

# Check functional equation Θ₄(1/t) = t² · Θ₄(t)
lhs, rhs, error = theta.functional_equation_check(2.0)

# Derivative
d_theta = theta.derivative(1.0, order=1)
```

**Properties:**

| Property | Description |
|----------|-------------|
| `phi` | Golden ratio value |
| `max_norm` | Maximum |n|² included |
| `num_terms` | Number of winding states |

#### HeatKernel

Heat kernel trace K(t) = Tr[exp(-t·L²)] = Σₙ w(n)·exp(-t·λₙ).

```python
from syntonic.srt.spectral import heat_kernel, HeatKernel

K = heat_kernel(max_norm=20)

# Evaluate heat kernel
trace = K.evaluate(0.1)  # Short time
trace = K.evaluate(10.0)  # Long time → approaches w(0) = 1

# Without golden weight
standard_trace = K.evaluate_without_golden(0.1)

# Spectral zeta function ζ_L(s)
zeta = K.spectral_zeta(2.0)  # Real s
zeta = K.spectral_zeta(complex(2.0, 0.5))  # Complex s

# Regularized determinant det'(L²)
det = K.spectral_determinant()

# Eigenvalue density histogram
density = K.eigenvalue_density(lambda_max=100.0, bins=50)

# Weyl law verification
count, expected, error = K.weyl_law_check(lambda_max=1000.0)
```

#### KnotLaplacian

The knot Laplacian L²_knot with eigenvalues λₙ = 4π²|n|² · (1 + φ^(-|n|²)).

```python
from syntonic.srt.spectral import knot_laplacian, KnotLaplacian
from syntonic.srt.geometry import winding_state

L = knot_laplacian(max_norm=20)

n = winding_state(1, 0, 0, 0)

# Eigenvalue components
base = L.base_eigenvalue(n)  # λ⁰ₙ = 4π²|n|²
potential = L.knot_potential(n)  # V_knot(n) = 4π²|n|² · φ^(-|n|²)
full = L.eigenvalue(n)  # λₙ = λ⁰ₙ + V_knot(n)

# Mass from eigenvalue
m_sq = L.mass_squared(n)  # m² = λₙ / 4π²

# Full spectrum sorted by eigenvalue
spectrum = L.spectrum(max_norm=10)
# Returns [(WindingState, eigenvalue), ...]

# Spectral properties
gap = L.spectral_gap()  # First positive eigenvalue
spacings = L.level_spacing(max_levels=10)
degeneracy = L.degeneracy(n_sq=1)  # Count states with |n|²=1

# Apply to wavefunction
coeffs = {winding_state(1,0,0,0): 1.0+0j, winding_state(0,1,0,0): 0.5+0j}
result = L.apply(coeffs)  # L²_knot|Ψ⟩

# Heat kernel trace using knot Laplacian
trace = L.heat_kernel_trace(t=0.1)

# Spectral zeta
zeta = L.zeta_function(complex(2.0, 0.0))
```

### Golden Operations (syn.srt.golden)

Golden measure weighting and the golden recursion map.

#### GoldenMeasure

The golden measure w(n) = exp(-|n|²/φ) on the winding lattice.

```python
from syntonic.srt.golden import golden_measure, GoldenMeasure, golden_weight
from syntonic.srt.geometry import winding_state

measure = golden_measure()

n = winding_state(1, 1, 0, 0)

# Weight computation
w = measure.weight(n)  # exp(-|n|²/φ)
log_w = measure.log_weight(n)  # -|n|²/φ

# Partition function Z = Σₙ w(n) (uses Rust backend)
Z = measure.partition_function(max_norm=10)

# Weighted sums
values = {winding_state(1,0,0,0): 2.0, winding_state(0,1,0,0): 3.0}
weighted_sum = measure.weighted_sum(values)

# Expectation values
def f(n):
    return n.norm_squared

mean = measure.expectation(f, max_norm=10)  # ⟨f⟩ = Σ w(n)f(n) / Z
var = measure.variance(f, max_norm=10)  # Var(f) = ⟨f²⟩ - ⟨f⟩²

# Theta sum (uses Rust backend)
theta = measure.theta_sum(t=1.0, max_norm=10)

# Convenience function
w = golden_weight(winding_state(1, 0, 0, 0))
```

#### GoldenRecursionMap

The golden recursion R: n → floor(φ·n) on Z⁴.

```python
from syntonic.srt.golden import golden_recursion, GoldenRecursionMap
from syntonic.srt.geometry import winding_state

R = golden_recursion()

n = winding_state(5, 3, 0, 0)

# Apply recursion
Rn = R.apply(n)  # floor(φ·n) = |8, 4, 0, 0⟩
Rn = R(n)  # Callable syntax

# Approximate inverse
n_approx = R.apply_inverse_approx(Rn)

# Orbit computation
orbit = R.orbit(n, max_depth=100)
# [|5,3,0,0⟩, |8,4,0,0⟩, |12,6,0,0⟩, ...]

# Orbit depth determines mass hierarchy
depth = R.orbit_depth(n)  # k where mass ~ exp(-φᵏ)
gen = R.generation(n)  # Generation number 1, 2, 3, ...

# Mass scaling factor
mass_scale = R.mass_scaling(n)  # exp(-φᵏ)

# Fixed and periodic points
is_fixed = R.is_fixed_point(n)
fixed = R.fixed_points(max_norm=10)
periodic = R.periodic_points(period=2, max_norm=10)

# Orbit classification
orbit_type = R.classify_orbit(n)
# Returns 'fixed', 'periodic-k', 'convergent', or 'divergent'
```

### Lattice Operations (syn.srt.lattice)

E8 and D4 lattice structures fundamental to SRT.

#### E8Lattice

The E8 lattice with 240 roots.

```python
from syntonic.srt.lattice import e8_lattice, E8Lattice, E8Root

E8 = e8_lattice()

# Get all 240 roots
roots = E8.roots

# Access a specific root
root = roots[0]
coords = root.coords  # 8-tuple of exact coordinates
norm_sq = root.norm_squared  # Always 2 for E8 roots

# Cartan matrix (8x8)
cartan = E8.cartan_matrix()

# Simple roots and positive roots
simple = E8.simple_roots()  # 8 simple roots
positive = E8.positive_roots()  # 120 positive roots

# Weyl group action
reflected = E8.weyl_reflection(root, simple_root_index=0)

# Inner product
ip = E8.inner_product(root, roots[1])

# Dynkin labels
labels = E8.dynkin_labels(root)
```

#### D4Lattice

The D4 lattice with kissing number K(D4) = 24.

```python
from syntonic.srt.lattice import d4_lattice, D4Lattice, D4Root, K_D4

D4 = d4_lattice()

# Kissing number constant
print(K_D4)  # 24

# Get all 24 roots
roots = D4.roots

# Cartan matrix (4x4)
cartan = D4.cartan_matrix()

# Simple and positive roots
simple = D4.simple_roots()  # 4 simple roots
positive = D4.positive_roots()  # 12 positive roots

# Triality automorphism (D4 has S3 outer automorphism)
triality = D4.triality_orbit(root)  # Returns 3 related roots
```

#### GoldenCone

The golden cone containing 36 roots (= Φ⁺(E6)).

```python
from syntonic.srt.lattice import golden_cone, GoldenCone, golden_projector

# Create golden projector P_φ: R⁸ → R⁴
P = golden_projector()

# Project E8 root to parallel/perpendicular components
parallel, perp = P.project(e8_root)

# Create golden cone
cone = golden_cone()

# Get 36 golden cone roots
roots = cone.roots

# Check if root is in cone
in_cone = cone.contains(e8_root)

# Quadratic form Q(λ) = |P_∥|² - |P_⊥|²
from syntonic.srt.lattice import quadratic_form, compute_Q

Q = quadratic_form()
q_value = Q.evaluate(e8_root)

# Quick computation
q = compute_Q(e8_root)
```

### Syntony Functional (syn.srt.functional)

The syntony functional S[Ψ] with fundamental bound S ≤ φ.

```python
from syntonic.srt.functional import (
    syntony_functional,
    SyntonyFunctional,
    compute_syntony,
)

# Create syntony functional
S = syntony_functional(max_norm=20)

# Evaluate on winding state coefficients
coefficients = {
    winding_state(0,0,0,0): 1.0,
    winding_state(1,0,0,0): 0.5,
}

# Compute S[Ψ]
syntony = S.evaluate(coefficients)

# Verify bound S ≤ φ
bound_satisfied = S.verify_bound(coefficients)

# Find ground state (minimizes energy subject to S ≤ φ)
ground = S.ground_state(max_norm=10)

# Quick computation
s = compute_syntony(coefficients)
```

### SRT Constants

```python
from syntonic.srt.lattice import K_D4  # Kissing number = 24

# Structure dimensions
# E8: 248 dimensions
# D4: 24 roots (kissing number)
# Golden cone: 36 roots
```

---

## Applications

The `syn.applications` module provides applied science frameworks that leverage syntony for physics, chemistry, biology, consciousness, and ecology.

```python
import syntonic as syn
from syntonic import applications

# Or import specific submodules
from syntonic.applications import thermodynamics, chemistry, biology, consciousness, ecology
```

### Thermodynamics (syn.applications.thermodynamics)

Syntony-based thermodynamics where free energy is related to syntony deficits.

#### DHSREngine

The DHSR thermodynamic engine models heat engines through differentiation-harmonization cycles.

```python
from syntonic.applications.thermodynamics import DHSREngine

engine = DHSREngine(
    hot_reservoir_temp=500.0,
    cold_reservoir_temp=300.0,
    working_medium_size=100,
)

# Run thermodynamic cycle
result = engine.run_cycle(n_steps=100)
print(f"Efficiency: {result['efficiency']:.4f}")
print(f"Work output: {result['work']:.4f}")
print(f"Carnot efficiency: {result['carnot_efficiency']:.4f}")
```

#### SyntonicEntropy

Entropy computation from syntony distributions.

```python
from syntonic.applications.thermodynamics import SyntonicEntropy

# Create entropy computer
entropy = SyntonicEntropy()

# Compute entropy of a state
S = entropy.compute(psi)  # S = -∫ ρ log ρ dμ_φ

# Verify second law
delta_S = entropy.entropy_production(initial, final)  # Should be ≥ 0
```

#### PhaseTransition

Model phase transitions via syntony bifurcations.

```python
from syntonic.applications.thermodynamics import PhaseTransition

transition = PhaseTransition(
    order_parameter='syntony',
    critical_temp=1.0,
)

# Find critical point
T_c = transition.critical_temperature()
order = transition.order_parameter(T=0.5)  # Below T_c
```

### Chemistry (syn.applications.chemistry)

Syntonic chemistry with golden-ratio based electronegativity and bonding.

#### SRTElectronegativity

Electronegativity derived from winding structure.

```python
from syntonic.applications.chemistry import SRTElectronegativity

EN = SRTElectronegativity()

# Get electronegativity for elements
chi_H = EN.get('H')   # Hydrogen
chi_O = EN.get('O')   # Oxygen
chi_C = EN.get('C')   # Carbon

# Electronegativity difference for bonding
delta_chi = EN.difference('H', 'O')
```

#### MolecularBonding

Molecular bonding analysis via syntony.

```python
from syntonic.applications.chemistry import MolecularBonding

bonding = MolecularBonding()

# Analyze bond
bond_strength = bonding.bond_strength('C', 'O', bond_order=2)
bond_syntony = bonding.bond_syntony('H', 'H')

# Molecular stability
stability = bonding.molecular_stability(['C', 'H', 'H', 'H', 'H'])  # Methane
```

#### PeriodicTable

Syntonic periodic table with golden shell structure.

```python
from syntonic.applications.chemistry import PeriodicTable

pt = PeriodicTable()

# Element properties
element = pt.get_element(6)  # Carbon
shell_config = pt.shell_configuration('Fe')
golden_period = pt.golden_period(element)
```

### Biology (syn.applications.biology)

Biological systems modeled through syntonic coherence.

#### LifeTopology

Topological characterization of life via syntony thresholds.

```python
from syntonic.applications.biology import LifeTopology

life = LifeTopology()

# Life threshold
S_life = life.life_threshold  # S ≥ φ - q for life

# Check if system is "alive"
is_alive = life.is_alive(system_syntony)

# Topological genus of metabolic network
genus = life.metabolic_genus(network)
```

#### Abiogenesis

Model the origin of life as syntony phase transition.

```python
from syntonic.applications.biology import Abiogenesis

abiogenesis = Abiogenesis()

# Probability of life emergence
P = abiogenesis.emergence_probability(
    temperature=300.0,
    concentration=1e-6,
    time=1e9,  # years
)

# Critical syntony for self-replication
S_critical = abiogenesis.self_replication_threshold()
```

#### Metabolism

Metabolic network analysis.

```python
from syntonic.applications.biology import Metabolism

metabolism = Metabolism()

# ATP synthesis efficiency
efficiency = metabolism.atp_efficiency(syntony=0.8)

# Metabolic syntony
S_metabolic = metabolism.network_syntony(reaction_network)
```

### Consciousness (syn.applications.consciousness)

Consciousness modeling via integrated information and gnosis layers.

#### ConsciousnessThreshold

The consciousness threshold S_c from CRT.

```python
from syntonic.applications.consciousness import ConsciousnessThreshold

threshold = ConsciousnessThreshold()

# Consciousness threshold
S_c = threshold.value  # ≈ φ - q for full consciousness

# Check consciousness level
level = threshold.consciousness_level(syntony)  # 0-3 scale
is_conscious = threshold.is_conscious(syntony)
```

#### GnosisLayers

The four gnosis layers of consciousness.

```python
from syntonic.applications.consciousness import GnosisLayers, GnosisLayer

gnosis = GnosisLayers()

# Get layer from syntony
layer = gnosis.get_layer(syntony)  # Returns GnosisLayer enum

# Layer names
print(GnosisLayer.NASCENT.name)      # Layer 0
print(GnosisLayer.EMERGENT.name)     # Layer 1
print(GnosisLayer.COHERENT.name)     # Layer 2
print(GnosisLayer.TRANSCENDENT.name) # Layer 3

# Transition thresholds
thresholds = gnosis.layer_thresholds()
```

#### ConsciousnessGnosis

Consciousness analysis through gnosis layer transitions.

```python
from syntonic.applications.consciousness import ConsciousnessGnosis

cg = ConsciousnessGnosis()

# Compute syntony for a neural state
syntony = cg.compute_syntony(neural_state)

# Get gnosis layer for given syntony
layer = cg.get_gnosis_layer(syntony)

# Check consciousness threshold
is_conscious = cg.is_above_threshold(syntony)
```

#### Qualia

Qualia modeling through spectral structure.

```python
from syntonic.applications.consciousness import Qualia

qualia = Qualia()

# Qualia space dimension
dim = qualia.space_dimension(phi=2.5)

# Distinguish qualia states
distance = qualia.distance(qualia_1, qualia_2)
```

### Ecology (syn.applications.ecology)

Ecosystem dynamics through syntony.

#### Ecosystem

General ecosystem modeling.

```python
from syntonic.applications.ecology import Ecosystem

eco = Ecosystem(n_species=10)

# Ecosystem syntony
S_eco = eco.syntony()

# Stability analysis
stable = eco.is_stable()
resilience = eco.resilience()
```

#### FoodWeb

Food web analysis.

```python
from syntonic.applications.ecology import FoodWeb

web = FoodWeb()

# Add species and interactions
web.add_species('grass', trophic_level=1)
web.add_species('rabbit', trophic_level=2)
web.add_predation('rabbit', 'grass')

# Syntonic metrics
connectance = web.connectance()
web_syntony = web.syntony()
```

#### GaiaHypothesis

Planetary-scale syntony (Gaia hypothesis).

```python
from syntonic.applications.ecology import GaiaHypothesis

gaia = GaiaHypothesis()

# Planetary syntony
S_gaia = gaia.planetary_syntony(biosphere_state)

# Homeostatic capacity
homeostasis = gaia.homeostatic_index()
```

---

## Neural Networks

The `syn.nn` module provides CRT-native neural network architectures that embed the DHSR cycle directly into deep learning. Networks optimize for both task performance AND syntony.

```python
import syntonic.nn as snn
import torch

# Quick example
model = snn.SyntonicMLP(784, [256, 128], 10)
optimizer = snn.SyntonicAdam(model.parameters(), lr=0.001)
criterion = snn.SyntonicLoss(torch.nn.CrossEntropyLoss())

for inputs, targets in dataloader:
    outputs = model(inputs)
    loss, metrics = criterion(outputs, targets, model)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step(syntony=model.syntony)

print(f"Model syntony: {model.syntony:.4f}")
```

### Core Concepts

**Syntonic Loss:**
```
L_total = L_task + λ(1 - S_model) + μC_{iπ}
```

Where:
- `L_task` = Standard task loss (CrossEntropy, MSE, etc.)
- `S_model` = Model syntony (measures representational coherence)
- `C_{iπ}` = Phase alignment cost (i ≃ π constraint)
- `λ, μ` = Regularization weights

**Block Syntony:**
```
S_block = 1 - ||D(x) - x|| / ||D(x) - H(D(x))||
```

**Syntony-Modulated Learning Rate:**
```
lr_eff = lr × (1 + α(S - S_target))
```

**Constants:**
- `PHI = (1 + √5) / 2 ≈ 1.618`
- `Q_DEFICIT ≈ 0.027395`
- `S_TARGET = PHI - Q_DEFICIT ≈ 1.591`

### Layers (syn.nn.layers)

DHSR-structured neural network layers.

#### DifferentiationLayer

Differentiation operator D̂[x] = x + ReLU(W·x + b).

```python
from syntonic.nn.layers import DifferentiationLayer

diff = DifferentiationLayer(
    in_features=256,
    out_features=256,
    alpha_scale=0.1,  # Differentiation strength
)

x_diff = diff(x)
```

#### HarmonizationLayer

Harmonization operator Ĥ[x] = x - σ(W_H·x) + tanh(W_S·x).

```python
from syntonic.nn.layers import HarmonizationLayer

harm = HarmonizationLayer(
    in_features=256,
    out_features=256,
    beta_scale=0.618,   # Damping strength (1/φ)
    gamma_scale=0.1,    # Syntony projection strength
)

x_harm = harm(x)
```

#### RecursionBlock

Complete DHSR cycle R̂ = Ĥ ∘ D̂ with syntony tracking.

```python
from syntonic.nn.layers import RecursionBlock

block = RecursionBlock(
    dim=256,
    alpha=0.1,
    beta=0.618,
    use_gate=True,  # Syntonic gate for adaptive mixing
)

x_out = block(x)
print(f"Block syntony: {block.syntony:.4f}")
```

#### SyntonicGate

Adaptive gate mixing based on local syntony.

```python
from syntonic.nn.layers import SyntonicGate, AdaptiveGate

gate = SyntonicGate(dim=256)
x_mixed = gate(x_diff, x_harm)  # Syntony-weighted combination
```

#### SyntonicNorm

Normalization layer targeting golden variance.

```python
from syntonic.nn.layers import SyntonicNorm, GoldenNorm

norm = SyntonicNorm(dim=256)  # Target variance = 1/φ
x_normed = norm(x)
```

### Loss Functions (syn.nn.loss)

#### SyntonicLoss

Main loss function with syntony regularization.

```python
from syntonic.nn.loss import SyntonicLoss

criterion = SyntonicLoss(
    task_loss=torch.nn.CrossEntropyLoss(),
    lambda_syntony=0.1,  # Syntony penalty weight
    mu_phase=0.01,       # Phase alignment weight
    syntony_target=1.591, # Target syntony (φ - q)
)

loss, metrics = criterion(pred, target, model, inputs)
# metrics = {'loss': ..., 'syntony': ..., 'loss_task': ..., 'loss_syntony': ...}
```

#### PhaseAlignmentLoss

i ≃ π phase alignment constraint.

```python
from syntonic.nn.loss import PhaseAlignmentLoss, IPiConstraint

phase_loss = PhaseAlignmentLoss(
    mu=0.01,
    target_phase=math.pi / 2,
)

loss = phase_loss(outputs)
```

#### GoldenDecay

Golden ratio-based weight decay.

```python
from syntonic.nn.loss import GoldenDecay

decay = GoldenDecay(
    model=model,
    lambda_decay=0.01,
    # Earlier layers decay faster (φ^{-l} scaling)
)

reg_loss = decay()  # Add to total loss
```

### Optimizers (syn.nn.optim)

#### SyntonicAdam

Adam with syntony-modulated learning rate.

```python
from syntonic.nn.optim import SyntonicAdam

optimizer = SyntonicAdam(
    model.parameters(),
    lr=0.001,
    alpha_syntony=0.1,  # LR modulation strength
)

# In training loop:
optimizer.step(syntony=model.syntony)  # Pass current syntony

print(f"Effective LR: {optimizer.effective_lr}")
```

#### SyntonicSGD / SyntonicMomentum

SGD variants with syntony modulation.

```python
from syntonic.nn.optim import SyntonicSGD, SyntonicMomentum

# Syntonic SGD with LR modulation
optimizer = SyntonicSGD(model.parameters(), lr=0.01)

# Syntonic Momentum (momentum scales with syntony)
optimizer = SyntonicMomentum(
    model.parameters(),
    lr=0.01,
    momentum=0.9,
    momentum_modulation=0.2,  # How much syntony affects momentum
)
```

#### GoldenScheduler

Golden ratio-based learning rate scheduler.

```python
from syntonic.nn.optim import GoldenScheduler, WarmupGoldenScheduler

# lr(t) = lr_0 × φ^{-t/T}
scheduler = GoldenScheduler(optimizer, T_max=100)

# With warmup
scheduler = WarmupGoldenScheduler(
    optimizer,
    warmup_steps=1000,
    T_max=10000,
)
```

### Architectures (syn.nn.architectures)

#### SyntonicMLP

Multi-layer perceptron with DHSR layers.

```python
from syntonic.nn.architectures import SyntonicMLP

model = SyntonicMLP(
    input_dim=784,
    hidden_dims=[512, 256, 128],
    output_dim=10,
    dropout=0.1,
    use_recursion=True,  # Use RecursionBlocks
)

output = model(x)
print(f"Model syntony: {model.syntony:.4f}")
print(f"Layer syntonies: {model.layer_syntonies}")
```

#### SyntonicCNN

Convolutional network with DHSR.

```python
from syntonic.nn.architectures import SyntonicCNN, RecursionConvBlock

model = SyntonicCNN(
    in_channels=3,
    num_classes=10,
    base_channels=64,
)

# Or build custom with RecursionConvBlock
block = RecursionConvBlock(64, 128, downsample=True)
```

#### CRTTransformer

Transformer with DH-structured layers.

```python
from syntonic.nn.architectures import CRTTransformer, DHTransformerLayer

model = CRTTransformer(
    src_vocab=10000,
    tgt_vocab=10000,
    d_model=512,
    n_heads=8,
    n_encoder_layers=6,
    n_decoder_layers=6,
)

# Encode
memory = model.encode(src)

# Decode
logits = model.decode(tgt, memory)

print(f"Transformer syntony: {model.syntony:.4f}")
```

#### SyntonicAttention

Attention with syntony tracking.

```python
from syntonic.nn.architectures import SyntonicAttention, GnosisAttention

# Standard syntonic attention
attn = SyntonicAttention(d_model=512)
output, weights = attn(query, key, value, return_attention=True)

# Gnosis-weighted attention (high syntony = more attention)
attn = GnosisAttention(d_model=512)
output = attn(query, key, value)
```

### Training (syn.nn.training)

#### SyntonicTrainer

Complete training loop with syntony tracking.

```python
from syntonic.nn.training import SyntonicTrainer, TrainingConfig

config = TrainingConfig(
    epochs=100,
    lr=0.001,
    lambda_syntony=0.1,
    use_syntonic_optimizer=True,
    use_archonic_detection=True,
)

trainer = SyntonicTrainer(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    config=config,
)

history = trainer.train()
print(f"Final syntony: {trainer.current_syntony:.4f}")
```

#### Callbacks

```python
from syntonic.nn.training import (
    SyntonyCallback,
    ArchonicEarlyStop,
    SyntonyCheckpoint,
)

callbacks = [
    SyntonyCallback(log_interval=10),
    ArchonicEarlyStop(patience=20),  # Stop if stuck
    SyntonyCheckpoint('checkpoints/', save_best=True),
]

trainer = SyntonicTrainer(model, loader, callbacks=callbacks)
```

### Analysis (syn.nn.analysis)

#### ArchonicDetector

Detect "stuck" archonic patterns.

```python
from syntonic.nn.analysis import ArchonicDetector, detect_archonic_pattern

detector = ArchonicDetector(
    window_size=50,
    variance_threshold=0.01,
)

# In training loop:
report = detector.update(model.syntony)
if report.is_archonic:
    print(f"Archonic pattern detected! Severity: {report.severity:.3f}")
    print(report.recommendation)
```

#### EscapeMechanism

Escape archonic patterns.

```python
from syntonic.nn.analysis import NoiseInjection, LearningRateShock

# Noise injection
escape = NoiseInjection(scale=0.01, target='weights')
if detector.is_archonic:
    escape.apply(model)

# LR shock
shock = LearningRateShock(multiplier=3.0, duration=10)
shock.apply(model, optimizer)
```

#### NetworkHealth

Monitor network health.

```python
from syntonic.nn.analysis import NetworkHealth, HealthReport

health = NetworkHealth(model)
report = health.check()

if not report.is_healthy:
    print(report)  # Shows warnings and recommendations
```

### Benchmarks (syn.nn.benchmarks)

#### BenchmarkSuite

Standard benchmark comparisons.

```python
from syntonic.nn.benchmarks import BenchmarkSuite, run_mnist_benchmark

# Run MNIST benchmark
results = run_mnist_benchmark(
    syntonic_model=syntonic_mlp,
    baseline_model=standard_mlp,
    epochs=20,
)

BenchmarkSuite.print_comparison(results)
# Shows: accuracy, loss, convergence speed, syntony
```

#### ConvergenceAnalyzer

Analyze convergence rates.

```python
from syntonic.nn.benchmarks import ConvergenceAnalyzer, compare_convergence

results = compare_convergence(
    syntonic_model, baseline_model,
    train_loader, test_loader,
    epochs=100,
)

# Expected: ~35% faster convergence for syntonic networks
```

#### AblationStudy

Component ablation analysis.

```python
from syntonic.nn.benchmarks import AblationStudy

study = AblationStudy(model, epochs=20)
results = study.run(train_loader, test_loader)
AblationStudy.print_report(results)
# Shows contribution of each DHSR component
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

Syntonic uses a Rust backend (`syntonic._core`) compiled via PyO3/maturin for high-performance operations. The Python layer is NumPy-free for core operations, with optional NumPy/SciPy integration for advanced features like `expm` and `logm`.

### Rust Backend Components

| Component | Description | Speedup |
|-----------|-------------|---------|
| `WindingState` | Winding state with i64 components | ~50x enumeration |
| `enumerate_windings` | Lattice point enumeration | ~50x vs Python |
| `theta_series_evaluate` | Theta series computation | ~50-100x |
| `heat_kernel_trace` | Heat kernel trace | ~50-100x |
| `heat_kernel_weighted` | Golden-weighted heat kernel | ~50-100x |
| `spectral_zeta` | Spectral zeta function | ~50-100x |
| `partition_function` | Golden measure partition function | ~50x |

### CUDA Support

CUDA support is provided through cudarc 0.12 when the `cuda` feature is enabled at build time.

```python
import syntonic as syn

# Check CUDA availability
if syn.cuda_is_available():
    print(f"CUDA devices: {syn.cuda_device_count()}")

    # Move state to GPU
    psi_gpu = psi.cuda(device_id=0)

    # Operations run on GPU
    result = psi_gpu @ other_gpu

    # Move back to CPU
    result_cpu = result.cpu()
```

**Supported CUDA Architectures:**
- SM 7.5 (Turing: RTX 20xx, GTX 16xx)
- SM 8.0 (Ampere: A100)
- SM 8.6 (Ampere: RTX 30xx)
- SM 9.0 (Hopper: H100)

### Building with CUDA

```bash
# Development build with CUDA
maturin develop --features cuda

# Release build with CUDA
maturin build --release --features cuda
```

The CUDA backend requires:
- NVIDIA GPU with compute capability ≥ 7.5
- CUDA toolkit 12.0+ (driver supports backward compatibility)
