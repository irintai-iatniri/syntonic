# Syntonic Unused API Integration Plan

## Overview
The codebase has ~46 warnings (14 unused imports + 32 unused variables) mostly from APIs awaiting integration. This document outlines how to use and expose these functions properly.

## Unused Import Issues (F401) - 14 instances

### 1. **Extended Hierarchy Corrections** (6 imports)
**Location**: `python/syntonic/crt/__init__.py`

```python
# Currently imported but unused:
- hierarchy_apply_collapse_threshold_correction
- hierarchy_apply_coxeter_kissing_correction  
- hierarchy_apply_e7_correction
- apply_collapse_threshold
- apply_coxeter_kissing
- apply_e7_correction
```

**Solution**: 
- Add to `__all__` in `crt/__init__.py` (already in main `__all__`, but not in crt module)
- Create a public API section in `crt/__init__.py` for hierarchy operations
- Add example usage in docstring

**Action**:
```python
# In crt/__init__.py, add to __all__:
__all__ = [
    # ... existing ...
    "hierarchy_apply_collapse_threshold_correction",
    "hierarchy_apply_coxeter_kissing_correction",
    "hierarchy_apply_e7_correction",
    "apply_collapse_threshold",
    "apply_coxeter_kissing",
    "apply_e7_correction",
    # ... rest ...
]
```

---

### 2. **Debug/Internal API** (1 import)
**Location**: `python/syntonic/__init__.py:55`

```python
_debug_stress_pool_take  # Internal debugging function
```

**Solution**: 
- Either mark as internal (keep `_` prefix, don't export) OR
- If it's useful for debugging, expose as `debug_stress_pool_take` in public API
- Add documentation for when/why to use it

**Action**: Remove from main `__all__` in `__init__.py`, or rename and document if useful

---

### 3. **SRT Core Functions** (2 imports)
**Location**: `python/syntonic/srt/__init__.py` (inferred from error)

```python
srt_phi_inv              # Inverse golden ratio
srt_structure_dimension  # Structure dimension getter
```

**Solution**: 
- Add to `srt/__init__.py` `__all__`
- These are useful utilities, expose them

---

### 4. **Neural Network Activation** (1 import)
**Location**: `python/syntonic/nn/__init__.py`

```python
GoldenGELU  # Custom GELU activation function
```

**Solution**:
- Add to `nn/__init__.py` `__all__`
- Document as alternative activation function using golden measure

---

### 5. **Visualization Dependencies** (1 import)
**Location**: Various test/example files

```python
matplotlib.pyplot  # Used in visualization examples
```

**Solution**:
- Make matplotlib an optional dependency
- Wrap imports in try/except
- Add conditional visualization functions

---

## Unused Variable Issues (F841) - 32 instances

These are mostly calculation steps that compute values but don't use them. Solutions:

### Category A: Intermediate Calculations (Should Be Used)
- `n`, `theta_values`, `S_before`, `S_after` in DHSR reference
- `phi` in constants computation
- `k_B`, `hbar`, `E_gap` in physical constants

**Solution**: 
- Review the algorithms - likely these values should be used
- Either use them in calculations, or remove dead code
- Add documentation explaining why they're computed

### Category B: Statistical Results (Should Be Exposed)
- `activation_stats`, `mean`, `std` - These are useful metrics
- `bars`, `seq_len`, `batch_size` - Visualization/debug info

**Solution**:
- Store in result objects instead of local variables
- Return as part of structured results
- Add to public API

---

## Implementation Strategy

### Phase 1: Quick Wins - Add to `__all__` (15 mins)
1. Add all 6 hierarchy correction functions to `crt/__init__.py` `__all__`
2. Add `srt_phi_inv` and `srt_structure_dimension` to `srt/__init__.py` `__all__`
3. Add `GoldenGELU` to `nn/__init__.py` `__all__`

### Phase 2: Fix Variable Issues (1-2 hours)
1. **Periodic Table** (`periodic_table.py:148`): Remove unused `n` assignment
2. **Constants** (`constants.py:416`): Review `phi` computation - likely unused
3. **Band Theory** (`band_theory.py:178`): Check if `k_B` needs to be used
4. **Superconductivity** (`superconductivity.py:135`): Review `hbar` usage
5. **Neural** (`neural.py:303`): Check if `E_gap` should be used
6. **DHSR Reference** (`dhsr_reference.py:229`): Review `theta_values` - looks incomplete
7. **Statistics** (various): Convert to returned result objects

### Phase 3: Documentation (30 mins)
1. Add usage examples to docstrings
2. Document when/why to use hierarchy corrections
3. Create API reference page

---

## Code Quality Improvements

### Option A: Maintain but Hide
```python
# In __init__.py
__all__ = [
    "public_api_item",
    # ... other items ...
]
# Private APIs are still accessible as module.item but not advertised
```

### Option B: Expose and Document
```python
__all__ = [
    "public_api_item",
    "previously_private_item",  # New! See docs for usage
]

# Add docstring example:
"""
Example:
    >>> from syntonic.crt import hierarchy_apply_e7_correction
    >>> corrected = hierarchy_apply_e7_correction(state, param)
"""
```

### Option C: Remove Dead Code
```python
# If not needed, delete the calculation entirely
# phi = PHI.eval()  # REMOVED: unused intermediate value
```

---

## File Changes Summary

| File | Changes | Impact |
|------|---------|--------|
| `python/syntonic/__init__.py` | Fix `_debug_stress_pool_take` | Main API surface |
| `python/syntonic/crt/__init__.py` | Add 6 hierarchy funcs to `__all__` | CRT API |
| `python/syntonic/srt/__init__.py` | Add 2 SRT funcs to `__all__` | SRT API |
| `python/syntonic/nn/__init__.py` | Add GoldenGELU to `__all__` | NN API |
| `various/` | Remove 32 unused local variables | Code quality |

---

## Testing Strategy

After changes:
```bash
# Run linter
ruff check python/

# Run type checker  
mypy python/syntonic

# Run unit tests
pytest tests/ -v

# Check that exports work
python3 -c "from syntonic.crt import hierarchy_apply_e7_correction; print('OK')"
```

---

## Expected Outcome

✅ 14 unused imports → Added to `__all__` (properly exported)
✅ 32 unused variables → Removed or integrated into results
✅ 0 warnings from F401/F841
✅ Public APIs properly documented
✅ Dead code removed
