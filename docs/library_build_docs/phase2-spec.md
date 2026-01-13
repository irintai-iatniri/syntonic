# SYNTONIC PHASE 2 - COMPLETE IMPLEMENTATION

**Timeline:** Weeks 7-10  
**Status:** Extended Numerics Phase  
**Prerequisites:** Phase 1 100% COMPLETE  
**Principle:** This phase must be 100% COMPLETE before Phase 3 begins.

---

## OVERVIEW

Phase 2 extends Syntonic's numeric capabilities:

| Week | Focus | Deliverables |
|------|-------|--------------|
| 7 | Quaternion System | `Quaternion` class, Hamilton product, rotations |
| 8 | Octonion System | `Octonion` class, Cayley product, G₂ automorphisms |
| 9 | Symbolic Core | `GoldenNumber`, `Expr` tree, SRT constants |
| 10 | Integration & Testing | State integration, mode switching, >90% coverage |

---

## PHASE 1 APIS (Must Be Complete)

Before starting Phase 2, verify these Phase 1 APIs work:

```python
import syntonic as syn

# State creation - MUST WORK
psi = syn.state([1, 2, 3, 4])
psi = syn.state.zeros((4, 4))
psi = syn.state.ones((4, 4))
psi = syn.state.random((4, 4), seed=42)

# Properties - MUST WORK
psi.shape       # Tuple[int, ...]
psi.dtype       # DType
psi.device      # Device
psi.norm()      # float
psi.normalize() # State

# Arithmetic - MUST WORK
psi + other
psi - other
psi * other
psi @ other

# Conversion - MUST WORK
psi.numpy()     # np.ndarray
psi.cuda()      # State on GPU
psi.cpu()       # State on CPU

# DTypes - MUST WORK
syn.float32
syn.float64     # DEFAULT
syn.complex64
syn.complex128  # DEFAULT for complex
syn.int64
syn.winding

# Devices - MUST WORK
syn.cpu
syn.cuda(0)
syn.cuda_is_available()
```

---

## WEEK 7: QUATERNION SYSTEM

### Mathematical Background

Quaternions ℍ = {a + bi + cj + dk} where:
- i² = j² = k² = ijk = -1
- ij = k, jk = i, ki = j (cyclic)
- ji = -k, kj = -i, ik = -j (anti-cyclic)

**Hamilton Product (NON-COMMUTATIVE):**
```
(a₁ + b₁i + c₁j + d₁k)(a₂ + b₂i + c₂j + d₂k) =
  (a₁a₂ - b₁b₂ - c₁c₂ - d₁d₂) +
  (a₁b₂ + b₁a₂ + c₁d₂ - d₁c₂)i +
  (a₁c₂ - b₁d₂ + c₁a₂ + d₁b₂)j +
  (a₁d₂ + b₁c₂ - c₁b₂ + d₁a₂)k
```

### Module Structure

```
syntonic/
├── python/syntonic/
│   └── hypercomplex/
│       ├── __init__.py
│       ├── quaternion.py      # Python API
│       └── octonion.py        # Python API
└── rust/src/
    └── hypercomplex/
        ├── mod.rs
        ├── quaternion.rs      # Rust implementation
        └── octonion.rs        # Rust implementation
```

### Quaternion Implementation

```python
# syntonic/hypercomplex/quaternion.py

"""
Quaternion algebra ℍ for Syntonic.

Used in CRT/SRT for:
- SU(2) spinor representations
- 3D rotations (SO(3) via unit quaternions)
- Winding state phase operations

CRITICAL: Hamilton product is NON-COMMUTATIVE: q₁*q₂ ≠ q₂*q₁
"""

import numpy as np
from dataclasses import dataclass
from typing import Tuple, Union, Optional


@dataclass
class Quaternion:
    """
    Quaternion q = a + bi + cj + dk.
    
    Properties:
        a: Real (scalar) part
        b, c, d: Imaginary (vector) parts for i, j, k
    
    Examples:
        >>> q = syn.quaternion(1, 2, 3, 4)  # 1 + 2i + 3j + 4k
        >>> q.norm()
        5.477...
        >>> q.conjugate()
        Quaternion(1, -2, -3, -4)
    """
    a: float  # Real part
    b: float  # i coefficient
    c: float  # j coefficient
    d: float  # k coefficient
    
    @property
    def real(self) -> float:
        """Scalar part."""
        return self.a
    
    @property
    def imag(self) -> Tuple[float, float, float]:
        """Vector part (b, c, d)."""
        return (self.b, self.c, self.d)
    
    @property
    def components(self) -> Tuple[float, float, float, float]:
        """All components (a, b, c, d)."""
        return (self.a, self.b, self.c, self.d)
    
    @property
    def i(self) -> float:
        """i component."""
        return self.b
    
    @property
    def j(self) -> float:
        """j component."""
        return self.c
    
    @property
    def k(self) -> float:
        """k component."""
        return self.d
    
    def conjugate(self) -> 'Quaternion':
        """
        Quaternion conjugate q* = a - bi - cj - dk.
        
        Property: q * q* = |q|²
        """
        return Quaternion(self.a, -self.b, -self.c, -self.d)
    
    def norm(self) -> float:
        """
        Quaternion norm |q| = √(a² + b² + c² + d²).
        """
        return np.sqrt(self.a**2 + self.b**2 + self.c**2 + self.d**2)
    
    def normalize(self) -> 'Quaternion':
        """
        Return unit quaternion q/|q|.
        
        Unit quaternions represent rotations in 3D.
        """
        n = self.norm()
        if n == 0:
            raise ValueError("Cannot normalize zero quaternion")
        return Quaternion(self.a/n, self.b/n, self.c/n, self.d/n)
    
    def inverse(self) -> 'Quaternion':
        """
        Multiplicative inverse q⁻¹ = q*/|q|².
        
        Property: q * q⁻¹ = 1
        """
        n2 = self.a**2 + self.b**2 + self.c**2 + self.d**2
        if n2 == 0:
            raise ValueError("Cannot invert zero quaternion")
        conj = self.conjugate()
        return Quaternion(conj.a/n2, conj.b/n2, conj.c/n2, conj.d/n2)
    
    def __add__(self, other: 'Quaternion') -> 'Quaternion':
        """Component-wise addition."""
        if isinstance(other, (int, float)):
            return Quaternion(self.a + other, self.b, self.c, self.d)
        return Quaternion(
            self.a + other.a,
            self.b + other.b,
            self.c + other.c,
            self.d + other.d
        )
    
    def __sub__(self, other: 'Quaternion') -> 'Quaternion':
        """Component-wise subtraction."""
        if isinstance(other, (int, float)):
            return Quaternion(self.a - other, self.b, self.c, self.d)
        return Quaternion(
            self.a - other.a,
            self.b - other.b,
            self.c - other.c,
            self.d - other.d
        )
    
    def __mul__(self, other: Union['Quaternion', float]) -> 'Quaternion':
        """
        Hamilton product (NON-COMMUTATIVE!).
        
        (a₁ + b₁i + c₁j + d₁k)(a₂ + b₂i + c₂j + d₂k) =
          (a₁a₂ - b₁b₂ - c₁c₂ - d₁d₂) +
          (a₁b₂ + b₁a₂ + c₁d₂ - d₁c₂)i +
          (a₁c₂ - b₁d₂ + c₁a₂ + d₁b₂)j +
          (a₁d₂ + b₁c₂ - c₁b₂ + d₁a₂)k
        
        WARNING: q1 * q2 ≠ q2 * q1 in general!
        """
        if isinstance(other, (int, float)):
            return Quaternion(self.a*other, self.b*other, self.c*other, self.d*other)
        
        a1, b1, c1, d1 = self.a, self.b, self.c, self.d
        a2, b2, c2, d2 = other.a, other.b, other.c, other.d
        
        return Quaternion(
            a1*a2 - b1*b2 - c1*c2 - d1*d2,  # Real part
            a1*b2 + b1*a2 + c1*d2 - d1*c2,  # i part
            a1*c2 - b1*d2 + c1*a2 + d1*b2,  # j part
            a1*d2 + b1*c2 - c1*b2 + d1*a2   # k part
        )
    
    def __rmul__(self, other: float) -> 'Quaternion':
        """Scalar multiplication from left."""
        return Quaternion(self.a*other, self.b*other, self.c*other, self.d*other)
    
    def __truediv__(self, other: Union['Quaternion', float]) -> 'Quaternion':
        """Division: q₁/q₂ = q₁ * q₂⁻¹."""
        if isinstance(other, (int, float)):
            return Quaternion(self.a/other, self.b/other, self.c/other, self.d/other)
        return self * other.inverse()
    
    def __neg__(self) -> 'Quaternion':
        """Negation."""
        return Quaternion(-self.a, -self.b, -self.c, -self.d)
    
    def __eq__(self, other: 'Quaternion') -> bool:
        """Equality check."""
        if not isinstance(other, Quaternion):
            return False
        return (np.isclose(self.a, other.a) and 
                np.isclose(self.b, other.b) and
                np.isclose(self.c, other.c) and 
                np.isclose(self.d, other.d))
    
    def __repr__(self) -> str:
        return f"Quaternion({self.a}, {self.b}, {self.c}, {self.d})"
    
    def __str__(self) -> str:
        parts = []
        if self.a != 0 or (self.b == 0 and self.c == 0 and self.d == 0):
            parts.append(f"{self.a}")
        if self.b != 0:
            parts.append(f"{self.b:+}i" if parts else f"{self.b}i")
        if self.c != 0:
            parts.append(f"{self.c:+}j" if parts else f"{self.c}j")
        if self.d != 0:
            parts.append(f"{self.d:+}k" if parts else f"{self.d}k")
        return "".join(parts) if parts else "0"
    
    # ========== Rotation Methods ==========
    
    def to_rotation_matrix(self) -> np.ndarray:
        """
        Convert unit quaternion to 3×3 rotation matrix.
        
        R = [[1-2(c²+d²),  2(bc-da),    2(bd+ca)  ],
             [2(bc+da),    1-2(b²+d²),  2(cd-ba)  ],
             [2(bd-ca),    2(cd+ba),    1-2(b²+c²)]]
        
        Returns:
            3×3 rotation matrix in SO(3)
        """
        q = self.normalize()
        a, b, c, d = q.a, q.b, q.c, q.d
        
        return np.array([
            [1 - 2*(c*c + d*d), 2*(b*c - d*a),     2*(b*d + c*a)    ],
            [2*(b*c + d*a),     1 - 2*(b*b + d*d), 2*(c*d - b*a)    ],
            [2*(b*d - c*a),     2*(c*d + b*a),     1 - 2*(b*b + c*c)]
        ])
    
    def rotate_vector(self, v: np.ndarray) -> np.ndarray:
        """
        Rotate 3D vector using quaternion: v' = q v q*
        
        Args:
            v: 3D vector as numpy array
        
        Returns:
            Rotated 3D vector
        """
        q = self.normalize()
        v_quat = Quaternion(0, v[0], v[1], v[2])
        rotated = q * v_quat * q.conjugate()
        return np.array([rotated.b, rotated.c, rotated.d])
    
    # ========== Class Methods ==========
    
    @classmethod
    def from_scalar(cls, s: float) -> 'Quaternion':
        """Create quaternion from scalar (real quaternion)."""
        return cls(s, 0, 0, 0)
    
    @classmethod
    def from_vector(cls, v: np.ndarray) -> 'Quaternion':
        """Create pure quaternion from 3D vector (zero real part)."""
        return cls(0, v[0], v[1], v[2])
    
    @classmethod
    def from_axis_angle(cls, axis: np.ndarray, theta: float) -> 'Quaternion':
        """
        Create unit quaternion from axis-angle representation.
        
        q = cos(θ/2) + sin(θ/2)(nₓi + nᵧj + n_zk)
        
        Args:
            axis: Unit rotation axis [nₓ, nᵧ, n_z]
            theta: Rotation angle in radians
        
        Returns:
            Unit quaternion representing the rotation
        """
        axis = np.asarray(axis, dtype=float)
        axis = axis / np.linalg.norm(axis)  # Normalize
        
        half = theta / 2
        s = np.sin(half)
        
        return cls(
            np.cos(half),
            s * axis[0],
            s * axis[1],
            s * axis[2]
        )
    
    @classmethod
    def from_euler(cls, roll: float, pitch: float, yaw: float) -> 'Quaternion':
        """
        Create quaternion from Euler angles (ZYX convention).
        
        Args:
            roll: Rotation about x-axis (radians)
            pitch: Rotation about y-axis (radians)
            yaw: Rotation about z-axis (radians)
        
        Returns:
            Unit quaternion representing the rotation
        """
        cr, sr = np.cos(roll/2), np.sin(roll/2)
        cp, sp = np.cos(pitch/2), np.sin(pitch/2)
        cy, sy = np.cos(yaw/2), np.sin(yaw/2)
        
        return cls(
            cr*cp*cy + sr*sp*sy,
            sr*cp*cy - cr*sp*sy,
            cr*sp*cy + sr*cp*sy,
            cr*cp*sy - sr*sp*cy
        )
    
    @classmethod
    def slerp(cls, q1: 'Quaternion', q2: 'Quaternion', t: float) -> 'Quaternion':
        """
        Spherical linear interpolation between quaternions.
        
        Args:
            q1: Start quaternion
            q2: End quaternion
            t: Interpolation parameter ∈ [0, 1]
        
        Returns:
            Interpolated quaternion
        """
        # Normalize inputs
        q1 = q1.normalize()
        q2 = q2.normalize()
        
        # Compute dot product
        dot = q1.a*q2.a + q1.b*q2.b + q1.c*q2.c + q1.d*q2.d
        
        # Handle negative dot (opposite hemispheres)
        if dot < 0:
            q2 = -q2
            dot = -dot
        
        # Clamp for numerical stability
        dot = min(1.0, dot)
        
        theta = np.arccos(dot)
        sin_theta = np.sin(theta)
        
        if abs(sin_theta) < 1e-10:
            # Linear interpolation for small angles
            return cls(
                q1.a + t*(q2.a - q1.a),
                q1.b + t*(q2.b - q1.b),
                q1.c + t*(q2.c - q1.c),
                q1.d + t*(q2.d - q1.d)
            ).normalize()
        
        s1 = np.sin((1-t)*theta) / sin_theta
        s2 = np.sin(t*theta) / sin_theta
        
        return cls(
            s1*q1.a + s2*q2.a,
            s1*q1.b + s2*q2.b,
            s1*q1.c + s2*q2.c,
            s1*q1.d + s2*q2.d
        )


# ========== Unit Quaternions ==========

I = Quaternion(0, 1, 0, 0)  # i
J = Quaternion(0, 0, 1, 0)  # j
K = Quaternion(0, 0, 0, 1)  # k
ONE = Quaternion(1, 0, 0, 0)  # 1


# ========== Factory Function ==========

def quaternion(a: float, b: float = 0, c: float = 0, d: float = 0) -> Quaternion:
    """Create quaternion a + bi + cj + dk."""
    return Quaternion(a, b, c, d)

quaternion.from_scalar = Quaternion.from_scalar
quaternion.from_vector = Quaternion.from_vector
quaternion.from_axis_angle = Quaternion.from_axis_angle
quaternion.from_euler = Quaternion.from_euler
quaternion.slerp = Quaternion.slerp
quaternion.I = I
quaternion.J = J
quaternion.K = K
```

### Week 7 Exit Criteria

- [ ] `Quaternion` class with all operations
- [ ] Hamilton product (verified non-commutative)
- [ ] `ij = k`, `jk = i`, `ki = j` (verified)
- [ ] Rotation matrix conversion
- [ ] SLERP interpolation
- [ ] Axis-angle and Euler conversions
- [ ] Unit tests passing

---

## WEEK 8: OCTONION SYSTEM

### Mathematical Background

Octonions 𝕆 = {a₀ + a₁e₁ + ... + a₇e₇} where eᵢ² = -1.

**CRITICAL: Octonions are NON-ASSOCIATIVE**
```
(o₁ * o₂) * o₃ ≠ o₁ * (o₂ * o₃)  in general
```

**Associator:**
```
A(X, Y, Z) = (XY)Z - X(YZ)
```

**CRT Connection:** G₂ = Aut(𝕆) is the automorphism group of octonions (dimension 14).

### Cayley Multiplication Table

The Fano plane structure determines multiplication:

```
     e₁  e₂  e₃  e₄  e₅  e₆  e₇
e₁   -1  e₄  e₇ -e₂  e₆ -e₅ -e₃
e₂  -e₄  -1  e₅  e₁ -e₃  e₇ -e₆
e₃  -e₇ -e₅  -1  e₆  e₂ -e₄  e₁
e₄   e₂ -e₁ -e₆  -1  e₇  e₃ -e₅
e₅  -e₆  e₃ -e₂ -e₇  -1  e₁  e₄
e₆   e₅ -e₇  e₄ -e₃ -e₁  -1  e₂
e₇   e₃  e₆ -e₁  e₅ -e₄ -e₂  -1
```

### Octonion Implementation

```python
# syntonic/hypercomplex/octonion.py

"""
Octonion algebra 𝕆 for Syntonic.

Used in CRT/SRT for:
- G₂ = Aut(𝕆) symmetry analysis
- Exceptional Lie group structures
- Non-associative stability analysis

CRITICAL: Octonions are NON-ASSOCIATIVE: (o₁*o₂)*o₃ ≠ o₁*(o₂*o₃)

The associator A(X,Y,Z) = (XY)Z - X(YZ) measures non-associativity.
CRT predicts syntony S_𝕆(Ψ) is maximized when associators vanish.
"""

import numpy as np
from dataclasses import dataclass
from typing import Tuple, Union, List


# Cayley multiplication table indices (Fano plane structure)
# MU[i][j] gives the index k such that e_i * e_j = ±e_k
_MU = [
    [0, 3, 6, 1, 5, 4, 2],  # e₁ * e_j
    [3, 0, 4, 0, 2, 6, 5],  # e₂ * e_j
    [6, 4, 0, 5, 1, 3, 0],  # e₃ * e_j
    [1, 0, 5, 0, 6, 2, 4],  # e₄ * e_j
    [5, 2, 1, 6, 0, 0, 3],  # e₅ * e_j
    [4, 6, 3, 2, 0, 0, 1],  # e₆ * e_j
    [2, 5, 0, 4, 3, 1, 0],  # e₇ * e_j
]

# Signs for Cayley multiplication
_SIGNS = [
    [-1,  1,  1, -1,  1, -1, -1],
    [-1, -1,  1,  1, -1,  1, -1],
    [-1, -1, -1,  1,  1, -1,  1],
    [ 1, -1, -1, -1,  1,  1, -1],
    [-1,  1, -1, -1, -1,  1,  1],
    [ 1, -1,  1, -1, -1, -1,  1],
    [ 1,  1, -1,  1, -1, -1, -1],
]


@dataclass
class Octonion:
    """
    Octonion o = a₀ + a₁e₁ + a₂e₂ + ... + a₇e₇.
    
    CRITICAL: Non-associative algebra!
    (o₁ * o₂) * o₃ ≠ o₁ * (o₂ * o₃) in general.
    
    Properties:
        components: Tuple of 8 real numbers (a₀, a₁, ..., a₇)
    
    Examples:
        >>> o = syn.octonion(1, 2, 3, 4, 5, 6, 7, 8)
        >>> o.norm()
        14.28...
        >>> o.conjugate()
        Octonion(1, -2, -3, -4, -5, -6, -7, -8)
    """
    a0: float
    a1: float
    a2: float
    a3: float
    a4: float
    a5: float
    a6: float
    a7: float
    
    @property
    def real(self) -> float:
        """Scalar part a₀."""
        return self.a0
    
    @property
    def imag(self) -> Tuple[float, float, float, float, float, float, float]:
        """Imaginary parts (a₁, ..., a₇)."""
        return (self.a1, self.a2, self.a3, self.a4, self.a5, self.a6, self.a7)
    
    @property
    def components(self) -> Tuple[float, ...]:
        """All 8 components."""
        return (self.a0, self.a1, self.a2, self.a3, 
                self.a4, self.a5, self.a6, self.a7)
    
    def __getitem__(self, i: int) -> float:
        """Get component aᵢ."""
        return self.components[i]
    
    def conjugate(self) -> 'Octonion':
        """
        Octonion conjugate o* = a₀ - a₁e₁ - ... - a₇e₇.
        
        Property: o * o* = |o|²
        """
        return Octonion(
            self.a0, -self.a1, -self.a2, -self.a3,
            -self.a4, -self.a5, -self.a6, -self.a7
        )
    
    def norm(self) -> float:
        """
        Octonion norm |o| = √(Σaᵢ²).
        """
        return np.sqrt(sum(x*x for x in self.components))
    
    def normalize(self) -> 'Octonion':
        """Return unit octonion o/|o|."""
        n = self.norm()
        if n == 0:
            raise ValueError("Cannot normalize zero octonion")
        c = self.components
        return Octonion(*(x/n for x in c))
    
    def inverse(self) -> 'Octonion':
        """
        Multiplicative inverse o⁻¹ = o*/|o|².
        """
        n2 = sum(x*x for x in self.components)
        if n2 == 0:
            raise ValueError("Cannot invert zero octonion")
        conj = self.conjugate()
        return Octonion(*(x/n2 for x in conj.components))
    
    def __add__(self, other: 'Octonion') -> 'Octonion':
        """Component-wise addition."""
        if isinstance(other, (int, float)):
            return Octonion(self.a0 + other, self.a1, self.a2, self.a3,
                           self.a4, self.a5, self.a6, self.a7)
        return Octonion(*(a + b for a, b in zip(self.components, other.components)))
    
    def __sub__(self, other: 'Octonion') -> 'Octonion':
        """Component-wise subtraction."""
        if isinstance(other, (int, float)):
            return Octonion(self.a0 - other, self.a1, self.a2, self.a3,
                           self.a4, self.a5, self.a6, self.a7)
        return Octonion(*(a - b for a, b in zip(self.components, other.components)))
    
    def __mul__(self, other: Union['Octonion', float]) -> 'Octonion':
        """
        Cayley product (NON-ASSOCIATIVE!).
        
        WARNING: (o₁*o₂)*o₃ ≠ o₁*(o₂*o₃) in general!
        
        Uses Fano plane structure for imaginary unit multiplication.
        """
        if isinstance(other, (int, float)):
            return Octonion(*(x * other for x in self.components))
        
        a = list(self.components)
        b = list(other.components)
        
        # Result components
        result = [0.0] * 8
        
        # Real part: a₀b₀ - Σᵢ aᵢbᵢ
        result[0] = a[0]*b[0] - sum(a[i]*b[i] for i in range(1, 8))
        
        # Imaginary parts
        for i in range(1, 8):
            # eᵢ contribution from a₀bᵢ + aᵢb₀
            result[i] = a[0]*b[i] + a[i]*b[0]
            
            # Cross terms from eⱼeₖ = ±eᵢ
            for j in range(1, 8):
                for k in range(1, 8):
                    if j != k:
                        # Check if eⱼ * eₖ contributes to eᵢ
                        if _MU[j-1][k-1] == i:
                            result[i] += _SIGNS[j-1][k-1] * a[j] * b[k]
        
        return Octonion(*result)
    
    def __rmul__(self, other: float) -> 'Octonion':
        """Scalar multiplication from left."""
        return Octonion(*(x * other for x in self.components))
    
    def __truediv__(self, other: Union['Octonion', float]) -> 'Octonion':
        """Division: o₁/o₂ = o₁ * o₂⁻¹."""
        if isinstance(other, (int, float)):
            return Octonion(*(x / other for x in self.components))
        return self * other.inverse()
    
    def __neg__(self) -> 'Octonion':
        """Negation."""
        return Octonion(*(-x for x in self.components))
    
    def __eq__(self, other: 'Octonion') -> bool:
        """Equality check."""
        if not isinstance(other, Octonion):
            return False
        return all(np.isclose(a, b) for a, b in zip(self.components, other.components))
    
    def __repr__(self) -> str:
        return f"Octonion{self.components}"
    
    # ========== Associativity Analysis ==========
    
    @staticmethod
    def associator(x: 'Octonion', y: 'Octonion', z: 'Octonion') -> 'Octonion':
        """
        Compute associator A(x, y, z) = (xy)z - x(yz).
        
        The associator measures non-associativity.
        A = 0 if and only if the multiplication is associative for those elements.
        
        CRT Application: Syntony S_𝕆 is maximized when associators vanish
        for operators D̂, Ĥ, R̂.
        """
        left = (x * y) * z
        right = x * (y * z)
        return left - right
    
    @staticmethod
    def is_associative(x: 'Octonion', y: 'Octonion', z: 'Octonion', 
                       eps: float = 1e-10) -> bool:
        """Check if multiplication is associative for given elements."""
        A = Octonion.associator(x, y, z)
        return A.norm() < eps
    
    @staticmethod
    def alternativity_check(x: 'Octonion', y: 'Octonion', 
                           eps: float = 1e-10) -> bool:
        """
        Check alternativity: (xx)y = x(xy) and (xy)y = x(yy).
        
        Octonions are alternative (satisfy weaker condition than associativity).
        """
        left1 = (x * x) * y
        right1 = x * (x * y)
        alt1 = (left1 - right1).norm() < eps
        
        left2 = (x * y) * y
        right2 = x * (y * y)
        alt2 = (left2 - right2).norm() < eps
        
        return alt1 and alt2
    
    # ========== Subalgebra Detection ==========
    
    def quaternion_subalgebra(self) -> 'Quaternion':
        """
        Find quaternion subalgebra ℍ ⊂ 𝕆 containing this octonion.
        
        Projects to the quaternion part (first 4 components).
        """
        from syntonic.hypercomplex.quaternion import Quaternion
        return Quaternion(self.a0, self.a1, self.a2, self.a3)
    
    # ========== Class Methods ==========
    
    @classmethod
    def from_real(cls, r: float) -> 'Octonion':
        """Create real octonion."""
        return cls(r, 0, 0, 0, 0, 0, 0, 0)
    
    @classmethod
    def from_quaternion(cls, q) -> 'Octonion':
        """Embed quaternion ℍ → 𝕆."""
        return cls(q.a, q.b, q.c, q.d, 0, 0, 0, 0)
    
    @classmethod
    def unit(cls, i: int) -> 'Octonion':
        """Get unit octonion eᵢ (i=0 gives 1, i=1-7 gives e₁-e₇)."""
        components = [0.0] * 8
        components[i] = 1.0
        return cls(*components)


# ========== CRT Stability Analysis ==========

def syntonic_stability(D_op: Octonion, H_op: Octonion, R_op: Octonion) -> float:
    """
    Compute syntonic stability S_𝕆 for octonion-valued operators.
    
    S_𝕆(Ψ) is maximized when:
    1. Associator A(D̂, Ĥ, R̂) vanishes, OR
    2. Malcev identity holds
    
    Returns:
        float: Syntonic stability measure ∈ [0, 1]
    """
    A = Octonion.associator(D_op, H_op, R_op)
    denom = D_op.norm() * H_op.norm() * R_op.norm()
    if denom == 0:
        return 1.0
    return 1.0 - A.norm() / denom


def malcev_check(X: Octonion, Y: Octonion, Z: Octonion, W: Octonion, 
                 eps: float = 1e-10) -> bool:
    """
    Verify Malcev identity for Lie algebra structure.
    
    A(X,Y,Z)W - A(X,Y,W)Z = A(X,Y,ZW) - A(XY,Z,W)
    
    Required for exotic recursion in high-energy CRT.
    """
    lhs = Octonion.associator(X, Y, Z) * W - Octonion.associator(X, Y, W) * Z
    rhs = Octonion.associator(X, Y, Z*W) - Octonion.associator(X*Y, Z, W)
    return (lhs - rhs).norm() < eps


# ========== Factory Function ==========

def octonion(a0: float, a1: float = 0, a2: float = 0, a3: float = 0,
             a4: float = 0, a5: float = 0, a6: float = 0, a7: float = 0) -> Octonion:
    """Create octonion a₀ + a₁e₁ + ... + a₇e₇."""
    return Octonion(a0, a1, a2, a3, a4, a5, a6, a7)

octonion.from_real = Octonion.from_real
octonion.from_quaternion = Octonion.from_quaternion
octonion.unit = Octonion.unit
octonion.associator = Octonion.associator
octonion.is_associative = Octonion.is_associative
```

### Week 8 Exit Criteria

- [ ] `Octonion` class with all operations
- [ ] Cayley product (verified non-associative)
- [ ] Associator computation
- [ ] Alternativity verification
- [ ] `syntonic_stability()` function
- [ ] `malcev_check()` function
- [ ] Unit tests passing

---

## WEEK 9: SYMBOLIC COMPUTATION SYSTEM

### Overview

The symbolic subsystem provides **exact computation** for:

| Constant | Symbol | Value | SRT Meaning |
|----------|--------|-------|-------------|
| Golden Ratio | φ | (1+√5)/2 | Recursion scaling |
| Spectral Constant | E* | e^π - π ≈ 19.999 | Moebius heat kernel |
| Syntony Deficit | q | (2φ + e/2φ²)/(φ⁴E*) | Vacuum geometry |

### GoldenNumber Implementation

```python
# syntonic/symbolic/golden.py

"""
Exact arithmetic in the golden ring ℤ[φ] = {a + bφ : a, b ∈ ℚ}.

The golden ring is closed under +, -, *, / due to the key identity:
    φ² = φ + 1

This allows exact computation without floating-point errors.

Source: Foundations.md §1.2
"""

from fractions import Fraction
from dataclasses import dataclass
from typing import Union
import math


@dataclass(frozen=True)
class GoldenNumber:
    """
    Exact representation of a + bφ where a, b ∈ ℚ.
    
    Key Identity: φ² = φ + 1
    
    This identity is used to reduce all expressions to the form a + bφ.
    
    Examples:
        >>> phi = GoldenNumber(0, 1)  # φ
        >>> phi ** 2
        GoldenNumber(1, 1)  # φ² = 1 + φ
        >>> phi ** 10
        GoldenNumber(34, 55)  # Fibonacci coefficients!
    """
    a: Fraction  # Rational part (coefficient of 1)
    b: Fraction  # Coefficient of φ
    
    def __init__(self, a: Union[int, float, Fraction], 
                 b: Union[int, float, Fraction] = 0):
        # Convert to Fraction for exact arithmetic
        object.__setattr__(self, 'a', Fraction(a))
        object.__setattr__(self, 'b', Fraction(b))
    
    @property
    def conjugate(self) -> 'GoldenNumber':
        """
        Galois conjugate: φ → φ' = 1 - φ = -1/φ.
        
        If x = a + bφ, then x' = a + b - bφ = (a+b) - bφ
        
        Property: x * x' = N(x) = field norm
        """
        return GoldenNumber(self.a + self.b, -self.b)
    
    @property
    def norm(self) -> Fraction:
        """
        Field norm N(x) = x · x' = a² + ab - b².
        
        Always rational (in ℚ).
        """
        return self.a * self.a + self.a * self.b - self.b * self.b
    
    def __add__(self, other: Union['GoldenNumber', int, float]) -> 'GoldenNumber':
        if isinstance(other, GoldenNumber):
            return GoldenNumber(self.a + other.a, self.b + other.b)
        return GoldenNumber(self.a + other, self.b)
    
    def __radd__(self, other: Union[int, float]) -> 'GoldenNumber':
        return self.__add__(other)
    
    def __sub__(self, other: Union['GoldenNumber', int, float]) -> 'GoldenNumber':
        if isinstance(other, GoldenNumber):
            return GoldenNumber(self.a - other.a, self.b - other.b)
        return GoldenNumber(self.a - other, self.b)
    
    def __rsub__(self, other: Union[int, float]) -> 'GoldenNumber':
        return GoldenNumber(other - self.a, -self.b)
    
    def __mul__(self, other: Union['GoldenNumber', int, float]) -> 'GoldenNumber':
        """
        Multiplication using φ² = φ + 1.
        
        (a₁ + b₁φ)(a₂ + b₂φ) = a₁a₂ + (a₁b₂ + a₂b₁)φ + b₁b₂φ²
                             = a₁a₂ + (a₁b₂ + a₂b₁)φ + b₁b₂(φ + 1)
                             = (a₁a₂ + b₁b₂) + (a₁b₂ + a₂b₁ + b₁b₂)φ
        """
        if isinstance(other, GoldenNumber):
            new_a = self.a * other.a + self.b * other.b  # Using φ² = φ + 1
            new_b = self.a * other.b + self.b * other.a + self.b * other.b
            return GoldenNumber(new_a, new_b)
        return GoldenNumber(self.a * other, self.b * other)
    
    def __rmul__(self, other: Union[int, float]) -> 'GoldenNumber':
        return self.__mul__(other)
    
    def __truediv__(self, other: Union['GoldenNumber', int, float]) -> 'GoldenNumber':
        """
        Division: x/y = x · y' / N(y).
        
        Uses conjugate to rationalize denominator.
        """
        if isinstance(other, GoldenNumber):
            # x/y = x · y' / N(y)
            conj = other.conjugate
            num = self * conj
            den = other.norm
            if den == 0:
                raise ZeroDivisionError("Cannot divide by zero GoldenNumber")
            return GoldenNumber(num.a / den, num.b / den)
        if other == 0:
            raise ZeroDivisionError("Cannot divide by zero")
        return GoldenNumber(self.a / other, self.b / other)
    
    def __rtruediv__(self, other: Union[int, float]) -> 'GoldenNumber':
        return GoldenNumber(other, 0) / self
    
    def __pow__(self, n: int) -> 'GoldenNumber':
        """
        Integer power using fast exponentiation.
        
        Fibonacci connection: φⁿ = F_{n-1} + F_n · φ
        """
        if n == 0:
            return GoldenNumber(1, 0)
        if n < 0:
            return GoldenNumber(1, 0) / (self ** (-n))
        if n == 1:
            return self
        
        # Fast exponentiation
        if n % 2 == 0:
            half = self ** (n // 2)
            return half * half
        return self * (self ** (n - 1))
    
    def __neg__(self) -> 'GoldenNumber':
        return GoldenNumber(-self.a, -self.b)
    
    def __abs__(self) -> float:
        return abs(float(self))
    
    def __float__(self) -> float:
        """Convert to floating point."""
        PHI_NUMERIC = (1 + math.sqrt(5)) / 2
        return float(self.a) + float(self.b) * PHI_NUMERIC
    
    def __eq__(self, other) -> bool:
        if isinstance(other, GoldenNumber):
            return self.a == other.a and self.b == other.b
        if isinstance(other, (int, float)):
            return self.a == other and self.b == 0
        return False
    
    def __hash__(self) -> int:
        return hash((self.a, self.b))
    
    def __repr__(self) -> str:
        return f"GoldenNumber({self.a}, {self.b})"
    
    def __str__(self) -> str:
        if self.b == 0:
            return str(self.a)
        if self.a == 0:
            if self.b == 1:
                return "φ"
            return f"{self.b}φ"
        if self.b == 1:
            return f"{self.a} + φ"
        if self.b > 0:
            return f"{self.a} + {self.b}φ"
        return f"{self.a} - {-self.b}φ"


# ========== Canonical Instances ==========

PHI = GoldenNumber(0, 1)           # φ
PHI_SQUARED = GoldenNumber(1, 1)   # φ² = 1 + φ
PHI_INVERSE = GoldenNumber(-1, 1)  # 1/φ = φ - 1
ONE = GoldenNumber(1, 0)           # 1
ZERO = GoldenNumber(0, 0)          # 0


# ========== Fibonacci/Lucas Connection ==========

def fibonacci(n: int) -> int:
    """
    Compute Fibonacci number F_n.
    
    Connection: φⁿ = F_{n-1} + F_n · φ
    """
    if n <= 0:
        return 0
    if n == 1:
        return 1
    
    phi_n = PHI ** n
    return int(phi_n.b)


def lucas(n: int) -> int:
    """
    Compute Lucas number L_n.
    
    L_n = φⁿ + (φ')ⁿ = φⁿ + (-1/φ)ⁿ
    """
    if n == 0:
        return 2
    if n == 1:
        return 1
    
    phi_n = PHI ** n
    # L_n = 2*a + b for φⁿ = a + bφ (when n > 0)
    return int(2 * phi_n.a + phi_n.b)


def zeckendorf(n: int) -> list:
    """
    Zeckendorf representation: express n as sum of non-consecutive Fibonacci numbers.
    
    Every positive integer has a unique Zeckendorf representation.
    """
    if n <= 0:
        return []
    
    # Find largest Fibonacci ≤ n
    fibs = [1, 2]
    while fibs[-1] < n:
        fibs.append(fibs[-1] + fibs[-2])
    
    result = []
    remaining = n
    for f in reversed(fibs):
        if f <= remaining:
            result.append(f)
            remaining -= f
        if remaining == 0:
            break
    
    return result
```

### SRT Constants

```python
# syntonic/symbolic/constants.py

"""
Fundamental constants of Syntony Recursion Theory.

All constants are derived from pure geometry—zero free parameters.

Source: Foundations.md §1.1-1.2
"""

import math
import numpy as np
from syntonic.symbolic.golden import GoldenNumber, PHI


# ========== Numeric Constants ==========

PHI_NUMERIC = (1 + math.sqrt(5)) / 2  # φ ≈ 1.618033988749895

E_STAR_NUMERIC = math.exp(math.pi) - math.pi  # E* ≈ 19.999099979189474

# Syntony deficit: q = (2φ + e/2φ²) / (φ⁴ × E*)
Q_DEFICIT_NUMERIC = (
    (2 * PHI_NUMERIC + math.e / (2 * PHI_NUMERIC**2)) /
    (PHI_NUMERIC**4 * E_STAR_NUMERIC)
)  # q ≈ 0.027395146920...


# ========== Heat Kernel Correction Factors ==========

# Structure dimensions for correction factors (1 ± q/N)
STRUCTURE_DIMENSIONS = {
    'E8_dim': 248,           # dim(E₈) - full adjoint
    'E8_roots': 240,         # |Φ(E₈)| - all roots
    'E8_positive': 120,      # |Φ⁺(E₈)| - positive roots (chiral)
    'E6_dim': 78,            # dim(E₆)
    'E6_positive': 36,       # |Φ⁺(E₆)| = Golden Cone roots
    'E6_fundamental': 27,    # dim(27_E₆) - fundamental rep
    'D4_kissing': 24,        # K(D₄) - consciousness threshold
    'G2_dim': 14,            # dim(G₂) = Aut(𝕆)
}


def correction_factor(structure: str, sign: int = 1) -> float:
    """
    Compute correction factor (1 ± q/N) for given structure.
    
    Args:
        structure: Key from STRUCTURE_DIMENSIONS
        sign: +1 or -1
    
    Returns:
        Correction factor (1 + sign * q/N)
    
    Source: Equations.md Part III
    """
    N = STRUCTURE_DIMENSIONS[structure]
    return 1.0 + sign * Q_DEFICIT_NUMERIC / N


# ========== E* Three-Term Decomposition ==========

def e_star_decomposition() -> dict:
    """
    Three-term decomposition of E*.
    
    E* = e^π - π = Γ(1/4)² + π(π-1) + (35/12)e^{-π} + Δ
    
    - Bulk: Γ(1/4)² ≈ 13.145
    - Torsion: π(π-1) ≈ 6.728
    - Cone: (35/12)e^{-π} ≈ 0.126
    - Residual: Δ ≈ 4.30×10⁻⁷
    
    Source: Equations.md Part I
    """
    from scipy.special import gamma
    
    bulk = gamma(0.25) ** 2
    torsion = math.pi * (math.pi - 1)
    cone = (35/12) * math.exp(-math.pi)
    total = E_STAR_NUMERIC
    residual = total - bulk - torsion - cone
    
    return {
        'total': total,
        'bulk': bulk,
        'torsion': torsion,
        'cone': cone,
        'residual': residual,
        'agreement': 1 - abs(residual) / total
    }


# ========== Golden Partition ==========

# D + H = S → 0.382 + 0.618 = 1
GOLDEN_D = 1 / PHI_NUMERIC**2  # ≈ 0.381966 (Differentiation contribution)
GOLDEN_H = 1 / PHI_NUMERIC     # ≈ 0.618034 (Harmonization contribution)

# Verify: GOLDEN_D + GOLDEN_H ≈ 1.0
assert abs(GOLDEN_D + GOLDEN_H - 1.0) < 1e-10, "Golden partition must sum to 1"
```

### Week 9 Exit Criteria

- [ ] `GoldenNumber` class with exact arithmetic
- [ ] φ² = φ + 1 identity verified
- [ ] `PHI`, `PHI_SQUARED`, `PHI_INVERSE` constants
- [ ] `fibonacci(n)` and `lucas(n)` functions
- [ ] `E_STAR_NUMERIC` ≈ 19.999099979
- [ ] `Q_DEFICIT_NUMERIC` ≈ 0.027395
- [ ] Correction factor functions
- [ ] Unit tests passing

---

## WEEK 10: INTEGRATION & TESTING

### Module Integration

```python
# syntonic/__init__.py additions for Phase 2

# Hypercomplex types
from syntonic.hypercomplex.quaternion import (
    Quaternion, quaternion, I, J, K
)
from syntonic.hypercomplex.octonion import (
    Octonion, octonion, syntonic_stability
)

# Symbolic computation
from syntonic.symbolic.golden import (
    GoldenNumber, PHI, PHI_SQUARED, PHI_INVERSE,
    fibonacci, lucas, zeckendorf
)
from syntonic.symbolic.constants import (
    PHI_NUMERIC, E_STAR_NUMERIC, Q_DEFICIT_NUMERIC,
    GOLDEN_D, GOLDEN_H, STRUCTURE_DIMENSIONS,
    correction_factor, e_star_decomposition
)

# Convenient aliases
phi = PHI
E_star = E_STAR_NUMERIC
q = Q_DEFICIT_NUMERIC

# DTypes for hypercomplex (add to dtype.py)
quaternion64 = DType("quaternion64", None, 16)    # 4×float32
quaternion128 = DType("quaternion128", None, 32)  # 4×float64 (DEFAULT)
octonion128 = DType("octonion128", None, 32)      # 8×float32
octonion256 = DType("octonion256", None, 64)      # 8×float64 (DEFAULT)
```

### Test Suite

```python
# tests/phase2/test_quaternion.py

import syntonic as syn
import numpy as np
import pytest


class TestQuaternionBasic:
    def test_construction(self):
        q = syn.quaternion(1, 2, 3, 4)
        assert q.real == 1
        assert q.imag == (2, 3, 4)
    
    def test_conjugate(self):
        q = syn.quaternion(1, 2, 3, 4)
        qc = q.conjugate()
        assert qc.real == 1
        assert qc.imag == (-2, -3, -4)
    
    def test_norm(self):
        q = syn.quaternion(1, 2, 3, 4)
        assert np.isclose(q.norm(), np.sqrt(30))
    
    def test_hamilton_product_non_commutative(self):
        q1 = syn.quaternion(1, 2, 0, 0)
        q2 = syn.quaternion(1, 0, 3, 0)
        assert q1 * q2 != q2 * q1  # NON-COMMUTATIVE!
    
    def test_ij_equals_k(self):
        i = syn.quaternion.I
        j = syn.quaternion.J
        k = syn.quaternion.K
        assert i * j == k
        assert j * k == i
        assert k * i == j
    
    def test_ji_equals_neg_k(self):
        i = syn.quaternion.I
        j = syn.quaternion.J
        k = syn.quaternion.K
        assert j * i == -k
    
    def test_rotation_matrix(self):
        # 90° rotation about z-axis
        q = syn.quaternion.from_axis_angle([0, 0, 1], np.pi/2)
        R = q.to_rotation_matrix()
        expected = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
        assert np.allclose(R, expected, atol=1e-10)
    
    def test_slerp(self):
        q1 = syn.quaternion(1, 0, 0, 0)
        q2 = syn.quaternion(0, 1, 0, 0)
        mid = syn.quaternion.slerp(q1, q2, 0.5)
        assert np.isclose(mid.norm(), 1.0)


class TestOctonionBasic:
    def test_non_associative(self):
        e1 = syn.octonion.unit(1)
        e2 = syn.octonion.unit(2)
        e4 = syn.octonion.unit(4)
        
        left = (e1 * e2) * e4
        right = e1 * (e2 * e4)
        assert left != right  # NON-ASSOCIATIVE!
    
    def test_associator_nonzero(self):
        o1 = syn.octonion(1, 1, 0, 0, 0, 0, 0, 0)
        o2 = syn.octonion(0, 0, 1, 1, 0, 0, 0, 0)
        o3 = syn.octonion(0, 0, 0, 0, 1, 1, 0, 0)
        
        A = syn.octonion.associator(o1, o2, o3)
        assert A.norm() > 0  # Non-zero associator
    
    def test_alternativity(self):
        o1 = syn.octonion(1, 2, 3, 4, 5, 6, 7, 8)
        o2 = syn.octonion(8, 7, 6, 5, 4, 3, 2, 1)
        assert syn.octonion.is_associative(o1, o1, o2)  # Alternativity


class TestGoldenNumber:
    def test_phi_squared(self):
        phi = syn.GoldenNumber(0, 1)
        phi_sq = phi ** 2
        assert phi_sq == syn.GoldenNumber(1, 1)  # φ² = 1 + φ
    
    def test_phi_powers_fibonacci(self):
        phi = syn.GoldenNumber(0, 1)
        # φⁿ = F_{n-1} + F_n·φ
        phi_10 = phi ** 10
        assert phi_10 == syn.GoldenNumber(34, 55)
    
    def test_division(self):
        phi = syn.GoldenNumber(0, 1)
        one = phi / phi
        assert one == syn.GoldenNumber(1, 0)
    
    def test_numeric_conversion(self):
        phi = syn.GoldenNumber(0, 1)
        assert np.isclose(float(phi), (1 + np.sqrt(5)) / 2)
    
    def test_fibonacci_connection(self):
        for n in range(1, 15):
            phi_n = syn.PHI ** n
            assert int(phi_n.b) == syn.fibonacci(n)


class TestSRTConstants:
    def test_E_star_near_20(self):
        assert np.isclose(syn.E_star, 19.999099979, rtol=1e-9)
    
    def test_q_value(self):
        assert np.isclose(syn.q, 0.027395, rtol=1e-4)
    
    def test_golden_partition(self):
        assert np.isclose(syn.GOLDEN_D + syn.GOLDEN_H, 1.0)
        assert np.isclose(syn.GOLDEN_D, 1/syn.PHI_NUMERIC**2)
        assert np.isclose(syn.GOLDEN_H, 1/syn.PHI_NUMERIC)
    
    def test_correction_factors(self):
        # (1 + q/120) for chiral corrections
        factor = syn.correction_factor('E8_positive', +1)
        assert np.isclose(factor, 1 + syn.q/120)
```

### Week 10 Exit Criteria

- [ ] All Phase 2 modules integrated
- [ ] `syn.quaternion()`, `syn.octonion()` work
- [ ] `syn.GoldenNumber()` works
- [ ] `syn.phi`, `syn.E_star`, `syn.q` accessible
- [ ] Mode switching (future: `syn.set_mode('symbolic')`)
- [ ] Test coverage >90%
- [ ] Documentation complete

---

## PHASE 2 EXIT CRITERIA

| Component | Requirement | Status |
|-----------|-------------|--------|
| Quaternion class | All operations, rotation | [ ] |
| Hamilton product | Non-commutative verified | [ ] |
| Octonion class | All operations | [ ] |
| Cayley product | Non-associative verified | [ ] |
| Associator | Correct computation | [ ] |
| GoldenNumber | Exact arithmetic | [ ] |
| φ² = φ + 1 | Identity verified | [ ] |
| E* constant | ≈ 19.999099979 | [ ] |
| q constant | ≈ 0.027395 | [ ] |
| Golden partition | D + H = 1 | [ ] |
| Test coverage | >90% | [ ] |
| Documentation | Complete | [ ] |

**Phase 2 is COMPLETE when all boxes are checked.**

---

## DEPENDENCIES FOR PHASE 3

Phase 3 (CRT Core) requires these Phase 2 APIs:

```python
# From Phase 2, Phase 3 will use:

# Exact golden ratio arithmetic
from syntonic.symbolic.golden import PHI, PHI_INVERSE, GoldenNumber

# SRT constants
from syntonic.symbolic.constants import (
    PHI_NUMERIC, Q_DEFICIT_NUMERIC, E_STAR_NUMERIC,
    GOLDEN_D, GOLDEN_H, correction_factor
)

# These must work:
phi = syn.PHI
float(phi)          # → 1.618...
phi ** 2            # → GoldenNumber(1, 1) = 1 + φ
syn.GOLDEN_D        # → 0.381966...
syn.GOLDEN_H        # → 0.618033...
syn.q               # → 0.027395...
```

---

*Document Version: 1.0*  
*This phase must be 100% complete before starting Phase 3.*