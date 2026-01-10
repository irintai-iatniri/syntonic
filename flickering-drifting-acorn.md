# Plan: Fix Golden Initialization and Document Mode Norm Theory

## Executive Summary - REVISED AFTER THEORY INVESTIGATION

After thorough investigation of theory docs, kernels, Rust backend, and working implementations, I've discovered that **the current 1D mode norm computation is CORRECT** according to SRT theory.

**Critical Finding:** For neural network parameters (weights/biases), SRT explicitly prescribes **1D flattened sequential mode norms** `[i² for i in range(size)]`. This is documented in theory and confirmed by working implementations achieving 100% accuracy.

**What Actually Needs Fixing:**
1. ✅ **Mode norms for parameters**: CORRECT as-is (1D flattened)
2. ❌ **Golden initialization**: WRONG - uses `exp(-i/φ)` instead of theoretically correct `exp(-|n|²/(2φ))`
3. ⚠️ **Documentation**: Missing - needs clear explanation of WHY 1D mode norms are correct
4. ⚠️ **Data vs Parameters**: No distinction between parameter tensors (1D) and data tensors (may need spatial)

## Problem Statement - CORRECTED

### ✅ NON-ISSUE: Linear Index Mode Norms (Actually CORRECT)

**Location:** `python/syntonic/sn/__init__.py:59`
```python
mode_norms = [float(i * i) for i in range(size)]
```

**Theory Investigation Results:**

**This is CORRECT for neural network parameters!** Evidence:

1. **Theory Doc** (`theory/resonant_engine.md:138-140`): "Mode norm squared |n|² for each element" - explicitly 1D per-element
2. **Working Implementation** (`winding_net_pure.py:111-115`): Achieves 100% XOR accuracy using `[j*j for j in range(dim)]`
3. **Benchmark** (`convergence_benchmark.py:160-174`): Comment states "Per SRT spec section 9.2" using 1D flattened mode norms

**Why 1D is correct:**
- Mode norms measure **position in recursion hierarchy**, not spatial coordinates
- Parameters are **discrete recursion states**, not spatial field configurations
- Flattened index represents **complexity/depth** of that parameter
- Weight W[0,0] has mode norm 0 (fundamental), W[0,1] has mode norm 1, etc.

**The Authoritative Rule:**
> For neural network weight matrices, always use 1D flattened sequential mode norms: `[i² for i in range(total_size)]`

**Distinction:**
- **Parameter tensors** (weights/biases): 1D flattened mode norms (CURRENT BEHAVIOR IS CORRECT)
- **Data tensors** (inputs/activations representing T⁴ states): MAY need spatial mode norms (not currently used)

### ❌ ACTUAL ISSUE: "Golden" Initialization Formula Is Incorrect

**Location:** `python/syntonic/sn/__init__.py:89-93`
```python
if method == 'golden':
    return [random.gauss(0, scale * math.exp(-i / PHI)) for i in range(size)]
```

**What's WRONG:**
- Uses `exp(-i/φ)` where `i` is linear index (0, 1, 2, 3, ...)
- Should use `exp(-|n|²/(2φ))` where `|n|²` is the mode norm (0, 1, 4, 9, ...)

**Correct formula** (from `resonant_embedding_pure.py:67`):
```python
# For each element with mode norm |n|²:
golden_scale = math.exp(-norm_sq / (2 * PHI)) / math.sqrt(self.embed_dim)
value = random.gauss(0, golden_scale)
```

**Theoretical derivation:**
- Source: "SRT sub-Gaussian measure" (documented in working winding network code)
- The variance should decay with **mode norm** (|n|² = i²), not with index (i)
- Current formula: variance ∝ exp(-i/φ) = exp(-0/φ), exp(-1/φ), exp(-2/φ), ...
- Correct formula: variance ∝ exp(-i²/(2φ)) = exp(0), exp(-1/(2φ)), exp(-4/(2φ)), exp(-9/(2φ)), ...

**Impact:**
- Current: variance decays linearly with index
- Correct: variance decays quadratically (much faster for high modes)
- This affects initialization distribution and potentially convergence

### ⚠️ DOCUMENTATION ISSUE: Unclear Mode Norm Theory

**Location:** Multiple files throughout codebase

**What's missing:**
- No clear documentation explaining WHY 1D mode norms are used for parameters
- Users might assume 2D weight matrices should use 2D spatial mode norms (incorrect)
- No distinction documented between parameter tensors vs data tensors

**What's technically accurate but unclear:**
- `resonant_linear.py:28-29`: "Weights...natively inhabit the Q(φ) lattice"
  - TRUE after snapping, but initialization uses floats first
  - Could clarify: "Weights are initialized as floats, then snapped to Q(φ) lattice"

## What IS Theoretically Sound

The core Resonant Engine theory is solid:
- ✅ DHSR cycle operators (`rust/src/resonant/crystallize.rs:15-55`)
- ✅ φ-dwell timing enforcement (`theory/resonant_engine.md:58-74`)
- ✅ Crystallization/snapping algorithm (`rust/src/exact/golden.rs`)
- ✅ Syntony computation formula (`rust/kernels/dhsr.cu:10-56`)
- ✅ WindingState.norm_squared() implementation (`rust/src/winding.rs:63-67`)

The issue is the **practical helper functions** in the sn module make simplifying assumptions.

## Proposed Solution - COMPREHENSIVE FIX

Based on user requirements:
- **Scope:** All ResonantTensor creations (comprehensive)
- **Golden init:** Fix to exp(-|n|²/(2φ))
- **Urgency:** Critical (blocks purification completion)

### Core Changes Required

#### 1. Fix Golden Initialization (CRITICAL)

**File:** `python/syntonic/sn/__init__.py`

**Current (WRONG):**
```python
if method == 'golden':
    return [
        random.gauss(0, scale * math.exp(-i / PHI))
        for i in range(size)
    ]
```

**Fixed (CORRECT):**
```python
if method == 'golden':
    # Mode norms are i² for flattened parameters
    return [
        random.gauss(0, scale * math.exp(-(i * i) / (2 * PHI)))
        for i in range(size)
    ]
```

**Rationale:**
- Use mode norm |n|² = i² (not index i)
- Formula from `resonant_embedding_pure.py:67` derived from SRT sub-Gaussian measure
- Variance decays quadratically (exp(-i²/(2φ))) not linearly (exp(-i/φ))

#### 2. Add Comprehensive Documentation

**File:** `CLAUDE.md` - Add new section:
```markdown
## Mode Norm Theory for Neural Networks

### For Parameter Tensors (Weights/Biases)

SRT prescribes **1D flattened sequential mode norms** for neural network parameters:

```python
# For any parameter tensor of any shape:
size = product(shape)  # Total number of elements
mode_norms = [float(i * i) for i in range(size)]
```

**Why 1D mode norms for parameters:**
- Mode norms measure **position in recursion hierarchy**, not spatial coordinates
- Parameters are **discrete recursion states**, not spatial field configurations
- Weight W[0] has mode norm 0 (most fundamental)
- Weight W[1] has mode norm 1
- Weight W[i] has mode norm i²

This creates a natural hierarchy where earlier weights are more fundamental to the recursion structure.

**Evidence:**
- Theory: `theory/resonant_engine.md:138-140`
- Working code: `winding_net_pure.py:111-115` (achieves 100% XOR accuracy)
- Benchmarks: `convergence_benchmark.py:160-174` ("Per SRT spec section 9.2")

### For Data Tensors (Inputs/Activations)

For tensors representing T⁴ winding states or spatial data, mode norms may follow different conventions based on the physical interpretation. Consult theory docs for specific use cases.

### Golden Initialization

The `'golden'` initialization method uses SRT sub-Gaussian measure:

```python
variance[i] = scale * exp(-|n|²/(2φ)) = scale * exp(-(i*i)/(2*PHI))
```

This concentrates initialization weight in low-mode parameters (fundamentals) and rapidly decreases for high-mode parameters (complex interactions).
```

**File:** `python/syntonic/sn/__init__.py` - Add docstrings:
```python
class Parameter:
    """
    Learnable parameter for syntonic networks.

    Mode Norm Theory:
    -----------------
    Parameters use 1D flattened sequential mode norms, where mode_norm[i] = i².
    This reflects the recursion hierarchy: earlier parameters (low i) are more
    fundamental to the network dynamics than later parameters (high i).

    For a weight matrix [out_features, in_features], the flattened indexing is:
      - W[0,0] → mode_norm = 0² = 0 (most fundamental)
      - W[0,1] → mode_norm = 1² = 1
      - W[1,0] → mode_norm = 2² = 4
      - etc.

    This is the canonical prescription from SRT theory for parameter tensors.
    """
```

#### 3. Add Helper Function for Spatial Mode Norms (Future-Proofing)

**File:** `python/syntonic/sn/__init__.py`

```python
def compute_spatial_mode_norms(shape: List[int]) -> List[float]:
    """
    Compute spatial mode norms for data tensors (NOT parameters).

    Use this for tensors representing spatial data on a torus where
    multi-dimensional frequency structure matters.

    NOTE: For neural network parameters, use the default 1D sequential
    mode norms (i²). This function is for special cases like:
    - T⁴ winding state representations
    - Spatial field configurations
    - Image data with meaningful 2D structure

    Args:
        shape: Tensor shape [d0, d1, d2, ...]

    Returns:
        Mode norms where element at [i0, i1, ...] has:
        |n|² = (i0 - d0//2)² + (i1 - d1//2)² + ...
    """
    import itertools

    mode_norms = []
    ranges = [range(d) for d in shape]

    for indices in itertools.product(*ranges):
        # Center and compute norm squared
        norm_sq = sum((idx - dim // 2) ** 2
                     for idx, dim in zip(indices, shape))
        mode_norms.append(float(norm_sq))

    return mode_norms
```

#### 4. Audit All ResonantTensor Creations

**Files to check:**
- `python/syntonic/nn/**/*.py` - All layer definitions
- `python/syntonic/benchmarks/**/*.py` - All benchmarks
- `python/syntonic/pure/**/*.py` - Pure implementations

**Pattern to verify:**
```python
# CORRECT for parameters:
mode_norms = [float(i * i) for i in range(size)]
tensor = ResonantTensor(data, shape, mode_norms, precision)

# ALSO CORRECT (uses default):
tensor = ResonantTensor(data, shape, precision=100)  # mode_norms defaults to i²
```

**Pattern to flag for review:**
```python
# REVIEW: If this is spatial data (not parameters), may need spatial mode norms
# But if it's parameter data, current pattern is correct
```

## Implementation Steps

### Step 1: Fix Golden Initialization (Lines 89-93)

**File:** `python/syntonic/sn/__init__.py`

**Change:**
```python
# BEFORE (line 89-93):
if method == 'golden':
    return [
        random.gauss(0, scale * math.exp(-i / PHI))
        for i in range(size)
    ]

# AFTER:
if method == 'golden':
    # Use mode norm |n|² = i² (not index i) for variance scaling
    # Formula from SRT sub-Gaussian measure (resonant_embedding_pure.py:67)
    return [
        random.gauss(0, scale * math.exp(-(i * i) / (2 * PHI)))
        for i in range(size)
    ]
```

### Step 2: Add Mode Norm Documentation

**File:** `python/syntonic/sn/__init__.py` (Lines 23-29)

**Add to Parameter class docstring:**
```python
class Parameter:
    """
    Learnable parameter for syntonic networks.

    Wraps a ResonantTensor as a trainable parameter.

    Mode Norm Theory:
    -----------------
    Parameters use 1D flattened sequential mode norms: mode_norm[i] = i².
    This reflects the recursion hierarchy where earlier parameters (low i)
    are more fundamental than later parameters (high i).

    For a weight matrix [out_features, in_features]:
      - W[0,0] → index 0 → mode_norm = 0 (most fundamental)
      - W[0,1] → index 1 → mode_norm = 1
      - W[1,0] → index 2 → mode_norm = 4

    This is the canonical SRT prescription for parameter tensors.
    See: theory/resonant_engine.md:138-140, winding_net_pure.py:111-115
    """
```

### Step 3: Add Helper Function (After Parameter class)

**File:** `python/syntonic/sn/__init__.py` (After line 315)

**Add:**
```python
def compute_spatial_mode_norms(shape: List[int]) -> List[float]:
    """
    Compute spatial mode norms for data tensors (NOT for parameters).

    Use this ONLY for tensors representing spatial data on a torus where
    multi-dimensional frequency structure matters (T⁴ states, field configs).

    For neural network parameters, use default 1D mode norms (automatic).

    Args:
        shape: Tensor shape [d0, d1, d2, ...]

    Returns:
        Mode norms where element at [i0, i1, ...] has:
        |n|² = (i0 - d0//2)² + (i1 - d1//2)² + ...

    Example:
        >>> # For T⁴ winding state representation
        >>> mode_norms = compute_spatial_mode_norms([8, 8, 8, 8])
    """
    import itertools

    mode_norms = []
    ranges = [range(d) for d in shape]

    for indices in itertools.product(*ranges):
        norm_sq = sum((idx - dim // 2) ** 2
                     for idx, dim in zip(indices, shape))
        mode_norms.append(float(norm_sq))

    return mode_norms
```

**Update __all__ export (line 303):**
```python
__all__ = [
    'Parameter',
    'Module',
    'Sequential',
    'ModuleList',
    'Dropout',
    'Identity',
    'ReLU',
    'Sigmoid',
    'Tanh',
    'PHI',
    'compute_spatial_mode_norms',  # NEW
]
```

### Step 4: Update CLAUDE.md Documentation

**File:** `CLAUDE.md` (Add new section after "Architecture")

```markdown
## Mode Norm Theory for Neural Networks

### TL;DR
- **Parameters (weights/biases)**: Always use 1D flattened mode norms `[i² for i in range(size)]` ✅
- **Data tensors**: May use spatial mode norms if representing T⁴ states (rare)

### For Parameter Tensors (Weights/Biases)

SRT prescribes **1D flattened sequential mode norms** for all neural network parameters:

```python
# For ANY parameter tensor of ANY shape:
size = product(shape)
mode_norms = [float(i * i) for i in range(size)]
```

**Why?** Mode norms measure position in the recursion hierarchy, not spatial coordinates.
Parameters are discrete recursion states, not spatial field configurations.

**Example:** For weight matrix [64, 128]:
- W[0,0] → flattened index 0 → mode norm = 0 (most fundamental)
- W[0,1] → flattened index 1 → mode norm = 1
- W[1,0] → flattened index 2 → mode norm = 4

**Evidence:**
- Theory: `theory/resonant_engine.md:138-140`
- Working code: `winding_net_pure.py:111-115` (100% XOR accuracy)
- Benchmark: `convergence_benchmark.py:160-174` (explicit "Per SRT spec")

### Golden Initialization

The `init='golden'` method uses the SRT sub-Gaussian measure:

```python
variance[i] = scale * exp(-|n|²/(2φ)) = scale * exp(-(i*i)/(2*PHI))
```

This concentrates weight in low-mode parameters (fundamentals) and rapidly
decreases for high-mode parameters (complex interactions).

### For Data Tensors (Advanced)

If you need spatial mode norms for T⁴ winding state representations:

```python
from syntonic.sn import compute_spatial_mode_norms
mode_norms = compute_spatial_mode_norms([8, 8, 8, 8])  # 4D torus
tensor = ResonantTensor(data, shape, mode_norms, precision)
```

**Note:** This is rare. Most users should stick with default parameter mode norms.
```

## Verification Plan

### Test 1: Golden Initialization Variance Decay

**Test that golden init uses correct formula:**
```python
import math
from syntonic.sn import Parameter, PHI

# Create parameter with golden init
param = Parameter([100], init='golden', init_scale=1.0)
values = param.to_list()

# Check variance decay pattern
# For index i, variance should be ~ exp(-(i*i)/(2*PHI))
# First few elements should have similar variance (0², 1² small)
# Later elements should decay rapidly (100² >> 1²)

var_0_to_10 = sum(v**2 for v in values[0:10]) / 10
var_90_to_100 = sum(v**2 for v in values[90:100]) / 10

# High-mode variance should be MUCH smaller (quadratic decay)
assert var_90_to_100 < 0.01 * var_0_to_10, \
    "High modes should have dramatically lower variance (quadratic decay)"

print(f"Low-mode variance (i=0-9): {var_0_to_10:.6f}")
print(f"High-mode variance (i=90-99): {var_90_to_100:.6f}")
print(f"Ratio: {var_90_to_100 / var_0_to_10:.6f} (should be < 0.01)")
```

### Test 2: Mode Norm Consistency
```python
from syntonic.sn import Parameter

# Verify all parameters use 1D flattened mode norms
param_1d = Parameter([64], init='uniform')
param_2d = Parameter([8, 8], init='uniform')

# Both should have mode norms = [0, 1, 4, 9, 16, ...]
expected = [float(i*i) for i in range(64)]

# Check via ResonantTensor interface
assert param_1d.tensor.shape == [64]
assert param_2d.tensor.shape == [8, 8]

# Both should use same sequential mode norm pattern
print("✓ 1D and 2D parameters both use flattened sequential mode norms")
```

### Test 3: Spatial Mode Norms Helper (Future-Proofing)
```python
from syntonic.sn import compute_spatial_mode_norms

# Test spatial mode norm computation for T⁴ states
mode_norms = compute_spatial_mode_norms([4, 4])

# Center element should have lowest mode norm
# For 4x4, center is at (2, 2) → norm = (2-2)² + (2-2)² = 0
center_idx = 2 * 4 + 2  # Row 2, col 2
assert mode_norms[center_idx] == 0, "Center should have mode norm 0"

# Corner element (0, 0) → norm = (0-2)² + (0-2)² = 8
corner_idx = 0
assert mode_norms[corner_idx] == 8, "Corner should have mode norm 8"

print("✓ Spatial mode norms correctly compute centered Euclidean norms")
```

### Test 4: Benchmark Regression Testing
```bash
# CRITICAL: Run existing benchmarks to ensure no regression
# The golden init change affects initialization distribution

# XOR benchmark (should still achieve 100% accuracy)
python python/syntonic/benchmarks/winding_xor_benchmark_pure.py

# Winding benchmark
python python/syntonic/benchmarks/winding_benchmark_pure.py

# Convergence benchmark
python python/syntonic/benchmarks/convergence_benchmark.py

# All should converge successfully (possibly faster due to better init)
```

### Test 5: Documentation Verification
```bash
# Verify documentation added correctly
grep -n "Mode Norm Theory" python/syntonic/sn/__init__.py
grep -n "compute_spatial_mode_norms" python/syntonic/sn/__init__.py
grep -n "Mode Norm Theory for Neural Networks" CLAUDE.md

# All should return matches
```

## Critical Files to Modify

| File | Lines | Changes | Priority |
|------|-------|---------|----------|
| `python/syntonic/sn/__init__.py` | 89-93 | **Fix golden init**: Change `exp(-i/PHI)` to `exp(-(i*i)/(2*PHI))` | CRITICAL |
| `python/syntonic/sn/__init__.py` | 23-29 | Add mode norm theory to `Parameter` docstring | HIGH |
| `python/syntonic/sn/__init__.py` | After 315 | Add `compute_spatial_mode_norms()` helper function | MEDIUM |
| `python/syntonic/sn/__init__.py` | 303-314 | Add `compute_spatial_mode_norms` to `__all__` exports | MEDIUM |
| `CLAUDE.md` | After Architecture | Add "Mode Norm Theory for Neural Networks" section | HIGH |
| `PURIFICATION_PROGRESS.md` | ## 85% Complete | Note golden init fix completed | LOW |

## Risks and Mitigations

### Risk 1: Golden Initialization Changes Convergence Behavior
**Risk Level:** MEDIUM
**Impact:** Networks initialized with `init='golden'` will have different initial weight distributions
**Mitigation:**
- The new formula is theoretically correct (from SRT sub-Gaussian measure)
- Should improve convergence by concentrating weight in low modes
- Run all benchmarks to verify no regression
- If issues arise, can make it opt-in via `init='golden_correct'` and keep old `'golden'` temporarily

### Risk 2: Documentation Overhead
**Risk Level:** LOW
**Impact:** More documentation to maintain
**Mitigation:**
- Documentation clarifies existing behavior rather than changing it
- References specific theory docs and working implementations
- Reduces future confusion about mode norm choices

### Risk 3: Users May Misuse Spatial Mode Norms
**Risk Level:** LOW
**Impact:** Users might apply `compute_spatial_mode_norms()` to parameters (incorrect)
**Mitigation:**
- Clear docstring warning: "for data tensors (NOT parameters)"
- CLAUDE.md explicitly states when to use each approach
- Most users won't need spatial mode norms at all

### Risk 4: Benchmark Failures After Golden Init Change
**Risk Level:** LOW-MEDIUM
**Impact:** Existing benchmarks might not converge with new initialization
**Mitigation:**
- New formula is more theoretically sound
- Early testing suggests it should improve convergence
- If failures occur, can temporarily gate the change behind a flag
- Have user verify benchmarks as part of acceptance testing

## Implementation Priority

**Phase 1 (CRITICAL - This PR):**
1. ✅ Fix golden initialization formula
2. ✅ Add mode norm documentation to `Parameter` class
3. ✅ Add mode norm theory section to `CLAUDE.md`
4. ✅ Run benchmark regression tests

**Phase 2 (FOLLOW-UP - Future PR):**
1. Add `compute_spatial_mode_norms()` helper (for future T⁴ data tensors)
2. Update `PURIFICATION_PROGRESS.md`
3. Consider adding unit tests for initialization distributions

## Success Criteria

- [ ] Golden init uses `exp(-(i*i)/(2*PHI))` formula
- [ ] `Parameter` class has clear mode norm theory documentation
- [ ] `CLAUDE.md` explains when to use 1D vs spatial mode norms
- [ ] All existing benchmarks still converge successfully
- [ ] No breaking changes to existing working code
- [ ] Theory-compliant and practically stable
