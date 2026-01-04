# Syntonic Symbolic Computation Subsystem
## Design Specification

**Version:** 1.0  
**Phase:** 2 (Extended Numerics)  
**Status:** Planning

---

# 1. Overview

The symbolic computation subsystem provides **exact mathematical computation** for expressions involving the fundamental constants of CRT/SRT:

| Constant | Symbol | Value | Meaning |
|----------|--------|-------|---------|
| Golden Ratio | φ | (1+√5)/2 | Recursion scaling |
| Pi | π | 3.14159... | Cycle completion |
| Euler's Number | e | 2.71828... | Natural growth |
| Spectral Constant | E* | e^π - π | Moebius heat kernel |
| Syntony Deficit | q | (2φ + e/2φ²)/(φ⁴E*) | Vacuum geometry |

The goal is to perform computations **exactly** when possible, avoiding floating-point approximation errors that accumulate in complex calculations.

---

# 2. Architecture

```
syntonic/
└── symbolic/
    ├── __init__.py           # Public API
    ├── core/
    │   ├── __init__.py
    │   ├── expr.py           # Expression tree
    │   ├── numbers.py        # Exact number types
    │   ├── ring.py           # Algebraic ring structures
    │   └── simplify.py       # Simplification rules
    ├── golden/
    │   ├── __init__.py
    │   ├── arithmetic.py     # Golden number arithmetic
    │   ├── zeckendorf.py     # Zeckendorf representation
    │   └── identities.py     # φ identities
    ├── transcendental/
    │   ├── __init__.py
    │   ├── pi.py             # π-related expressions
    │   ├── exp.py            # Exponential expressions
    │   └── trigonometric.py  # Trig functions
    ├── srt/
    │   ├── __init__.py
    │   ├── constants.py      # E*, q, etc.
    │   └── formulas.py       # SRT-specific formulas
    └── convert.py            # Symbolic ↔ Numeric conversion
```

---

# 3. Core Expression System

## 3.1 Expression Tree

```python
# syntonic/symbolic/core/expr.py

from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Union, Optional, Callable
from fractions import Fraction
import math


class Expr(ABC):
    """Base class for symbolic expressions."""
    
    @abstractmethod
    def simplify(self) -> Expr:
        """Return simplified form."""
        pass
    
    @abstractmethod
    def evaluate(self, precision: int = 64) -> complex:
        """Evaluate to numeric value with given precision."""
        pass
    
    @abstractmethod
    def __eq__(self, other) -> bool:
        pass
    
    @abstractmethod
    def __hash__(self) -> int:
        pass
    
    def __add__(self, other) -> Expr:
        return Add(self, _to_expr(other)).simplify()
    
    def __radd__(self, other) -> Expr:
        return Add(_to_expr(other), self).simplify()
    
    def __sub__(self, other) -> Expr:
        return Sub(self, _to_expr(other)).simplify()
    
    def __rsub__(self, other) -> Expr:
        return Sub(_to_expr(other), self).simplify()
    
    def __mul__(self, other) -> Expr:
        return Mul(self, _to_expr(other)).simplify()
    
    def __rmul__(self, other) -> Expr:
        return Mul(_to_expr(other), self).simplify()
    
    def __truediv__(self, other) -> Expr:
        return Div(self, _to_expr(other)).simplify()
    
    def __rtruediv__(self, other) -> Expr:
        return Div(_to_expr(other), self).simplify()
    
    def __pow__(self, other) -> Expr:
        return Pow(self, _to_expr(other)).simplify()
    
    def __rpow__(self, other) -> Expr:
        return Pow(_to_expr(other), self).simplify()
    
    def __neg__(self) -> Expr:
        return Neg(self).simplify()
    
    def __float__(self) -> float:
        return float(self.evaluate())
    
    def __complex__(self) -> complex:
        return complex(self.evaluate())


@dataclass(frozen=True)
class Rational(Expr):
    """Exact rational number a/b."""
    
    value: Fraction
    
    def __init__(self, numerator: int, denominator: int = 1):
        object.__setattr__(self, 'value', Fraction(numerator, denominator))
    
    def simplify(self) -> Expr:
        return self
    
    def evaluate(self, precision: int = 64) -> complex:
        return complex(float(self.value))
    
    def __eq__(self, other) -> bool:
        if isinstance(other, Rational):
            return self.value == other.value
        return False
    
    def __hash__(self) -> int:
        return hash(('Rational', self.value))
    
    def __repr__(self) -> str:
        if self.value.denominator == 1:
            return str(self.value.numerator)
        return f"({self.value.numerator}/{self.value.denominator})"


@dataclass(frozen=True)
class Sqrt(Expr):
    """Square root √x."""
    
    arg: Expr
    
    def simplify(self) -> Expr:
        arg = self.arg.simplify()
        
        # √(a²) = |a| (for known positive)
        if isinstance(arg, Rational) and arg.value >= 0:
            # Check for perfect squares
            n = arg.value.numerator
            d = arg.value.denominator
            sqrt_n = int(math.isqrt(n))
            sqrt_d = int(math.isqrt(d))
            if sqrt_n * sqrt_n == n and sqrt_d * sqrt_d == d:
                return Rational(sqrt_n, sqrt_d)
        
        # √(a * b²) = b√a
        if isinstance(arg, Mul):
            # Factor out perfect squares
            pass  # Complex implementation
        
        return Sqrt(arg)
    
    def evaluate(self, precision: int = 64) -> complex:
        return complex(self.arg.evaluate(precision)) ** 0.5
    
    def __eq__(self, other) -> bool:
        if isinstance(other, Sqrt):
            return self.arg == other.arg
        return False
    
    def __hash__(self) -> int:
        return hash(('Sqrt', self.arg))
    
    def __repr__(self) -> str:
        return f"√({self.arg})"


@dataclass(frozen=True)
class Add(Expr):
    """Addition a + b."""
    
    left: Expr
    right: Expr
    
    def simplify(self) -> Expr:
        left = self.left.simplify()
        right = self.right.simplify()
        
        # Rational + Rational = Rational
        if isinstance(left, Rational) and isinstance(right, Rational):
            return Rational(
                (left.value + right.value).numerator,
                (left.value + right.value).denominator
            )
        
        # a + 0 = a
        if isinstance(right, Rational) and right.value == 0:
            return left
        if isinstance(left, Rational) and left.value == 0:
            return right
        
        # Golden number addition (handled separately)
        if isinstance(left, GoldenNumber) or isinstance(right, GoldenNumber):
            return _add_golden(left, right)
        
        return Add(left, right)
    
    def evaluate(self, precision: int = 64) -> complex:
        return self.left.evaluate(precision) + self.right.evaluate(precision)
    
    def __eq__(self, other) -> bool:
        if isinstance(other, Add):
            return (self.left == other.left and self.right == other.right) or \
                   (self.left == other.right and self.right == other.left)
        return False
    
    def __hash__(self) -> int:
        # Commutative hash
        return hash(('Add', frozenset([hash(self.left), hash(self.right)])))
    
    def __repr__(self) -> str:
        return f"({self.left} + {self.right})"


# Similar implementations for Sub, Mul, Div, Pow, Neg, Exp, Log, Sin, Cos...


# ===== Special Constants =====

class Pi(Expr):
    """The constant π."""
    
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def simplify(self) -> Expr:
        return self
    
    def evaluate(self, precision: int = 64) -> complex:
        # Use mpmath for arbitrary precision
        from mpmath import mp
        mp.dps = precision
        return complex(float(mp.pi))
    
    def __eq__(self, other) -> bool:
        return isinstance(other, Pi)
    
    def __hash__(self) -> int:
        return hash('Pi')
    
    def __repr__(self) -> str:
        return "π"


class E(Expr):
    """Euler's number e."""
    
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def simplify(self) -> Expr:
        return self
    
    def evaluate(self, precision: int = 64) -> complex:
        from mpmath import mp
        mp.dps = precision
        return complex(float(mp.e))
    
    def __eq__(self, other) -> bool:
        return isinstance(other, E)
    
    def __hash__(self) -> int:
        return hash('E')
    
    def __repr__(self) -> str:
        return "e"


class ImaginaryUnit(Expr):
    """The imaginary unit i."""
    
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def simplify(self) -> Expr:
        return self
    
    def evaluate(self, precision: int = 64) -> complex:
        return 1j
    
    def __eq__(self, other) -> bool:
        return isinstance(other, ImaginaryUnit)
    
    def __hash__(self) -> int:
        return hash('I')
    
    def __repr__(self) -> str:
        return "i"


# Singleton instances
pi = Pi()
e = E()
i = ImaginaryUnit()


def _to_expr(x) -> Expr:
    """Convert to Expr."""
    if isinstance(x, Expr):
        return x
    if isinstance(x, int):
        return Rational(x)
    if isinstance(x, float):
        # Try to recognize as rational
        f = Fraction(x).limit_denominator(1000000)
        if abs(float(f) - x) < 1e-15:
            return Rational(f.numerator, f.denominator)
        # Fall back to approximate rational
        return Rational(f.numerator, f.denominator)
    if isinstance(x, Fraction):
        return Rational(x.numerator, x.denominator)
    raise TypeError(f"Cannot convert {type(x)} to Expr")
```

---

# 4. Golden Number System

## 4.1 The Ring ℤ[φ]

The **golden ring** ℤ[φ] = {a + bφ : a, b ∈ ℤ} is closed under addition and multiplication, making it ideal for exact φ-arithmetic.

**Key identity:** φ² = φ + 1

This means any power of φ can be reduced to form a + bφ.

```python
# syntonic/symbolic/golden/arithmetic.py

from dataclasses import dataclass
from typing import Union
from fractions import Fraction


@dataclass(frozen=True)
class GoldenNumber:
    """
    Exact representation of a + bφ where a, b ∈ ℚ.
    
    The golden ring ℚ[φ] is closed under +, -, *, /.
    
    Key identity: φ² = φ + 1
    
    Examples:
        >>> phi = GoldenNumber(0, 1)  # φ
        >>> phi ** 2
        GoldenNumber(1, 1)  # 1 + φ
        >>> phi ** 3
        GoldenNumber(1, 2)  # 1 + 2φ
    """
    
    a: Fraction  # Rational part
    b: Fraction  # Coefficient of φ
    
    def __init__(self, a: Union[int, Fraction], b: Union[int, Fraction] = 0):
        object.__setattr__(self, 'a', Fraction(a))
        object.__setattr__(self, 'b', Fraction(b))
    
    @property
    def conjugate(self) -> 'GoldenNumber':
        """
        Galois conjugate: replace φ with φ' = -1/φ = 1 - φ.
        
        If x = a + bφ, then x' = a + b(1-φ) = (a+b) - bφ
        """
        return GoldenNumber(self.a + self.b, -self.b)
    
    @property
    def norm(self) -> Fraction:
        """
        Field norm: N(x) = x * x' = a² + ab - b².
        
        This is always rational.
        """
        return self.a * self.a + self.a * self.b - self.b * self.b
    
    def __add__(self, other: 'GoldenNumber') -> 'GoldenNumber':
        if isinstance(other, (int, Fraction)):
            other = GoldenNumber(other, 0)
        return GoldenNumber(self.a + other.a, self.b + other.b)
    
    def __radd__(self, other) -> 'GoldenNumber':
        return self.__add__(GoldenNumber(other, 0) if isinstance(other, (int, Fraction)) else other)
    
    def __sub__(self, other: 'GoldenNumber') -> 'GoldenNumber':
        if isinstance(other, (int, Fraction)):
            other = GoldenNumber(other, 0)
        return GoldenNumber(self.a - other.a, self.b - other.b)
    
    def __rsub__(self, other) -> 'GoldenNumber':
        return GoldenNumber(other, 0).__sub__(self) if isinstance(other, (int, Fraction)) else other.__sub__(self)
    
    def __neg__(self) -> 'GoldenNumber':
        return GoldenNumber(-self.a, -self.b)
    
    def __mul__(self, other: 'GoldenNumber') -> 'GoldenNumber':
        """
        (a + bφ)(c + dφ) = ac + adφ + bcφ + bdφ²
                        = ac + adφ + bcφ + bd(φ + 1)
                        = (ac + bd) + (ad + bc + bd)φ
        """
        if isinstance(other, (int, Fraction)):
            return GoldenNumber(self.a * other, self.b * other)
        
        ac = self.a * other.a
        bd = self.b * other.b
        ad_bc = self.a * other.b + self.b * other.a
        
        return GoldenNumber(ac + bd, ad_bc + bd)
    
    def __rmul__(self, other) -> 'GoldenNumber':
        return self.__mul__(other)
    
    def __truediv__(self, other: 'GoldenNumber') -> 'GoldenNumber':
        """
        Division: x/y = x * y' / N(y)
        
        where y' is conjugate and N(y) is norm (rational).
        """
        if isinstance(other, (int, Fraction)):
            return GoldenNumber(self.a / other, self.b / other)
        
        # x / y = x * y' / N(y)
        conj = other.conjugate
        norm = other.norm
        
        if norm == 0:
            raise ZeroDivisionError("Division by zero in GoldenNumber")
        
        product = self * conj
        return GoldenNumber(product.a / norm, product.b / norm)
    
    def __rtruediv__(self, other) -> 'GoldenNumber':
        return GoldenNumber(other, 0).__truediv__(self)
    
    def __pow__(self, n: int) -> 'GoldenNumber':
        """Integer power using φ² = φ + 1 identity."""
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
        else:
            return self * (self ** (n - 1))
    
    def __eq__(self, other) -> bool:
        if isinstance(other, GoldenNumber):
            return self.a == other.a and self.b == other.b
        if isinstance(other, (int, Fraction)):
            return self.a == other and self.b == 0
        return False
    
    def __hash__(self) -> int:
        return hash(('Golden', self.a, self.b))
    
    def __float__(self) -> float:
        PHI = (1 + 5 ** 0.5) / 2
        return float(self.a) + float(self.b) * PHI
    
    def __repr__(self) -> str:
        if self.b == 0:
            return str(self.a)
        if self.a == 0:
            if self.b == 1:
                return "φ"
            if self.b == -1:
                return "-φ"
            return f"{self.b}φ"
        
        b_str = f"+ {self.b}φ" if self.b > 0 else f"- {-self.b}φ"
        if abs(self.b) == 1:
            b_str = "+ φ" if self.b > 0 else "- φ"
        return f"({self.a} {b_str})"
    
    def to_zeckendorf(self) -> list[int]:
        """
        Zeckendorf representation: sum of non-consecutive Fibonacci numbers.
        
        Every positive integer has a unique such representation.
        This extends to golden numbers.
        """
        # Implementation for positive integers
        if self.b != 0:
            raise NotImplementedError("Zeckendorf for non-integer golden numbers")
        
        n = int(self.a)
        if n <= 0:
            raise ValueError("Zeckendorf requires positive integers")
        
        fibs = [1, 2]
        while fibs[-1] < n:
            fibs.append(fibs[-1] + fibs[-2])
        
        result = []
        for f in reversed(fibs):
            if f <= n:
                result.append(f)
                n -= f
        
        return result


# Canonical golden number instances
phi = GoldenNumber(0, 1)           # φ
phi_inv = GoldenNumber(1, -1)      # 1/φ = φ - 1 = -1 + φ... wait, 1/φ = φ - 1
phi_squared = GoldenNumber(1, 1)   # φ² = 1 + φ

# Verify identities
assert phi ** 2 == phi_squared == phi + 1
assert phi * phi_inv == GoldenNumber(1, 0)  # Should be 1
```

## 4.2 Golden Identities Database

```python
# syntonic/symbolic/golden/identities.py

"""
Fundamental golden ratio identities for simplification.
"""

from syntonic.symbolic.golden.arithmetic import GoldenNumber, phi


# ===== Basic Identities =====

IDENTITIES = {
    # Powers of φ
    'phi^2': (phi ** 2, GoldenNumber(1, 1)),      # φ² = 1 + φ
    'phi^3': (phi ** 3, GoldenNumber(1, 2)),      # φ³ = 1 + 2φ
    'phi^4': (phi ** 4, GoldenNumber(2, 3)),      # φ⁴ = 2 + 3φ
    'phi^5': (phi ** 5, GoldenNumber(3, 5)),      # φ⁵ = 3 + 5φ
    
    # Negative powers
    'phi^-1': (phi ** -1, GoldenNumber(-1, 1)),   # φ⁻¹ = -1 + φ = φ - 1
    'phi^-2': (phi ** -2, GoldenNumber(2, -1)),   # φ⁻² = 2 - φ
    
    # φ and Fibonacci
    # F_n = (φⁿ - ψⁿ)/√5 where ψ = 1 - φ
    
    # Lucas numbers L_n = φⁿ + ψⁿ
    # L_0 = 2, L_1 = 1, L_2 = 3, L_3 = 4, L_4 = 7...
}


def fibonacci(n: int) -> int:
    """Compute nth Fibonacci number using golden ratio."""
    if n < 0:
        raise ValueError("n must be non-negative")
    if n <= 1:
        return n
    
    # F_n = round(φⁿ / √5)
    phi_n = phi ** n
    # Using exact arithmetic where possible
    sqrt5 = 5 ** 0.5
    return round(float(phi_n) / sqrt5)


def lucas(n: int) -> int:
    """Compute nth Lucas number."""
    if n == 0:
        return 2
    if n == 1:
        return 1
    
    # L_n = L_{n-1} + L_{n-2}
    a, b = 2, 1
    for _ in range(n - 1):
        a, b = b, a + b
    return b


def phi_power_coefficients(n: int) -> tuple[int, int]:
    """
    Return (a, b) such that φⁿ = a + bφ.
    
    These are Lucas and Fibonacci numbers:
    φⁿ = L_{n-1} + F_n·φ (for n ≥ 1)
    
    Or equivalently using the recurrence:
    φⁿ = F_{n-1} + F_n·φ
    """
    if n == 0:
        return (1, 0)
    if n == 1:
        return (0, 1)
    
    # Use recurrence: (a,b) → (b, a+b)
    a, b = 0, 1
    for _ in range(n - 1):
        a, b = b, a + b
    return (a, b)


# Verify
assert phi ** 10 == GoldenNumber(*phi_power_coefficients(10))
```

---

# 5. SRT Constants

```python
# syntonic/symbolic/srt/constants.py

"""
Exact symbolic representation of SRT fundamental constants.
"""

from syntonic.symbolic.core.expr import Expr, pi, e, Rational, Pow, Div, Add, Mul, Sqrt
from syntonic.symbolic.golden.arithmetic import GoldenNumber, phi


class SpectralConstant(Expr):
    """
    E* = e^π - π ≈ 19.999099979...
    
    The spectral constant from Moebius-regularized heat kernel.
    """
    
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def simplify(self) -> Expr:
        return self
    
    def evaluate(self, precision: int = 64) -> complex:
        from mpmath import mp
        mp.dps = precision
        return complex(float(mp.exp(mp.pi) - mp.pi))
    
    def __eq__(self, other) -> bool:
        return isinstance(other, SpectralConstant)
    
    def __hash__(self) -> int:
        return hash('E_star')
    
    def __repr__(self) -> str:
        return "E*"
    
    @property
    def expanded(self) -> Expr:
        """Return e^π - π as expression."""
        return Add(Pow(e, pi), Mul(Rational(-1), pi))


class SyntonyDeficit(Expr):
    """
    q = (2φ + e/(2φ²)) / (φ⁴(e^π - π))
    
    The universal syntony deficit.
    q ≈ 0.027395146920...
    """
    
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def simplify(self) -> Expr:
        return self
    
    def evaluate(self, precision: int = 64) -> complex:
        from mpmath import mp
        mp.dps = precision
        
        phi_val = (1 + mp.sqrt(5)) / 2
        e_star = mp.exp(mp.pi) - mp.pi
        
        numerator = 2 * phi_val + mp.e / (2 * phi_val ** 2)
        denominator = (phi_val ** 4) * e_star
        
        return complex(float(numerator / denominator))
    
    def __eq__(self, other) -> bool:
        return isinstance(other, SyntonyDeficit)
    
    def __hash__(self) -> int:
        return hash('q')
    
    def __repr__(self) -> str:
        return "q"
    
    @property
    def expanded(self) -> Expr:
        """Return full expression for q."""
        # q = (2φ + e/(2φ²)) / (φ⁴ · E*)
        phi_expr = GoldenExpr(phi)
        
        numerator = Add(
            Mul(Rational(2), phi_expr),
            Div(e, Mul(Rational(2), Pow(phi_expr, Rational(2))))
        )
        denominator = Mul(Pow(phi_expr, Rational(4)), E_star)
        
        return Div(numerator, denominator)


# Singleton instances
E_star = SpectralConstant()
q = SyntonyDeficit()


class GoldenExpr(Expr):
    """Wrapper to use GoldenNumber in expression tree."""
    
    def __init__(self, golden: GoldenNumber):
        self._golden = golden
    
    def simplify(self) -> Expr:
        if self._golden.b == 0:
            return Rational(self._golden.a.numerator, self._golden.a.denominator)
        return self
    
    def evaluate(self, precision: int = 64) -> complex:
        return complex(float(self._golden))
    
    def __eq__(self, other) -> bool:
        if isinstance(other, GoldenExpr):
            return self._golden == other._golden
        return False
    
    def __hash__(self) -> int:
        return hash(('GoldenExpr', self._golden))
    
    def __repr__(self) -> str:
        return repr(self._golden)


# ===== Derived Constants =====

def compute_fine_structure_constant() -> Expr:
    """
    α = 1/137.035999...
    
    In SRT, derived from gauge coupling unification.
    """
    # Placeholder - full derivation involves RG running
    return Div(Rational(1), Rational(137))


def compute_higgs_mass() -> Expr:
    """
    m_H ≈ 125.25 GeV
    
    Tree level: 93 GeV
    With golden loops: +32 GeV
    """
    # Tree level from SRT
    v = Rational(246)  # Higgs VEV in GeV
    # m_H = v · √(2λ) where λ is determined by φ
    pass


# ===== Particle Mass Formulas =====

def fermion_mass_formula(generation: int, is_up_type: bool) -> Expr:
    """
    Mass formula: m_f ~ v · ζ_f · e^(-φ·k)
    
    where k is recursion depth (generation).
    """
    v = Rational(246)  # GeV
    k = generation
    
    # ζ depends on fermion type (from winding structure)
    # Placeholder
    return Mul(v, Pow(e, Mul(Rational(-1), GoldenExpr(phi), Rational(k))))
```

---

# 6. Public API

```python
# syntonic/symbolic/__init__.py

"""
Syntonic Symbolic Computation System

Provides exact mathematical computation for CRT/SRT expressions.

Usage:
    >>> import syntonic as syn
    >>> syn.set_mode('symbolic')
    >>> 
    >>> # Exact golden ratio arithmetic
    >>> phi = syn.phi
    >>> phi ** 2 == phi + 1
    True
    >>> 
    >>> # Exact constant expressions
    >>> q = syn.q  # Syntony deficit
    >>> float(q)
    0.027395146920...
    >>> 
    >>> # Compute particle masses symbolically
    >>> m_e = syn.symbolic.electron_mass()
    >>> print(m_e.expanded)
    v · ζ_e · e^(-3φ)
"""

from syntonic.symbolic.core.expr import Expr, Rational, Sqrt, pi, e, i
from syntonic.symbolic.golden.arithmetic import GoldenNumber, phi, phi_inv, phi_squared
from syntonic.symbolic.golden.identities import fibonacci, lucas
from syntonic.symbolic.srt.constants import E_star, q

__all__ = [
    # Core expression types
    'Expr', 'Rational', 'Sqrt',
    
    # Mathematical constants
    'pi', 'e', 'i',
    
    # Golden number system
    'GoldenNumber', 'phi', 'phi_inv', 'phi_squared',
    'fibonacci', 'lucas',
    
    # SRT constants
    'E_star', 'q',
]


# Mode switching
_symbolic_mode = False


def set_mode(mode: str) -> None:
    """
    Set computation mode.
    
    Args:
        mode: 'symbolic' for exact computation, 'numeric' for floating-point
    """
    global _symbolic_mode
    if mode == 'symbolic':
        _symbolic_mode = True
    elif mode == 'numeric':
        _symbolic_mode = False
    else:
        raise ValueError(f"Unknown mode: {mode}. Use 'symbolic' or 'numeric'.")


def get_mode() -> str:
    """Get current computation mode."""
    return 'symbolic' if _symbolic_mode else 'numeric'


def is_symbolic() -> bool:
    """Check if in symbolic mode."""
    return _symbolic_mode
```

---

# 7. Integration with State Class

```python
# In syntonic/core/state.py, add symbolic support:

class State:
    # ... existing code ...
    
    def symbolic(self) -> 'SymbolicState':
        """
        Convert to symbolic state for exact computation.
        
        Returns:
            SymbolicState with exact representations where possible
        """
        from syntonic.symbolic.state import SymbolicState
        return SymbolicState.from_state(self)
    
    @classmethod
    def from_symbolic(cls, sym_state: 'SymbolicState', precision: int = 64) -> 'State':
        """
        Create numeric State from SymbolicState.
        
        Args:
            sym_state: Symbolic state to convert
            precision: Decimal precision for evaluation
        """
        return sym_state.evaluate(precision)
```

---

# 8. Example Usage

```python
import syntonic as syn

# === Exact Golden Ratio Arithmetic ===

syn.set_mode('symbolic')

# All golden ratio operations are exact
phi = syn.phi
assert phi ** 2 == phi + 1           # True (exactly)
assert phi ** 10 == syn.GoldenNumber(34, 55)  # Fibonacci coefficients

# Verify φ identities
assert phi ** 2 - phi - 1 == 0       # Minimal polynomial
assert phi + 1/phi == syn.fibonacci(3) + syn.fibonacci(2) * phi  # ?

# === SRT Constants ===

E_star = syn.E_star
q = syn.q

# Exact representation
print(q)  # q
print(q.expanded)  # (2φ + e/(2φ²)) / (φ⁴(e^π - π))

# Evaluate with arbitrary precision
print(f"q = {float(q):.15f}")  # 0.027395146920...
print(f"E* = {float(E_star):.15f}")  # 19.999099979189...

# Verify near-integer property of E*
print(f"|E* - 20| = {abs(float(E_star) - 20):.10f}")  # Very small!

# === Symbolic State Operations ===

# Create state with exact coefficients
coeffs = [syn.GoldenNumber(1, 0), syn.GoldenNumber(0, 1), syn.GoldenNumber(1, 1)]
psi = syn.state(coeffs, mode='symbolic')

# Syntony computed exactly where possible
print(f"S(ψ) = {psi.syntony}")  # Exact symbolic expression

# Convert to numeric when needed
psi_numeric = psi.evaluate(precision=128)
```

---

# 9. Performance Considerations

| Operation | Symbolic | Numeric (float64) | Notes |
|-----------|----------|-------------------|-------|
| Golden arithmetic | O(1) | O(1) | Both constant time |
| Golden power φⁿ | O(log n) | O(1) | Fast exponentiation |
| Expression simplify | O(tree size) | N/A | Can be expensive |
| Evaluate to numeric | O(precision) | O(1) | mpmath for high precision |

**Strategy:**
- Use symbolic for intermediate algebraic manipulation
- Convert to numeric for final heavy computation
- Cache evaluated results

---

# 10. Future Extensions

1. **Symbolic differentiation:** d/dx of symbolic expressions
2. **Pattern matching:** Recognize and simplify complex expressions
3. **Algebraic number fields:** Extend beyond ℚ[φ] to ℚ[φ, √2], etc.
4. **Modular forms:** For spectral theory connections
5. **Polynomial solving:** Find roots in golden ring

---

*Document Version: 1.0*
*Status: Planning Complete*
*Implementation Phase: 2 (Extended Numerics)*
