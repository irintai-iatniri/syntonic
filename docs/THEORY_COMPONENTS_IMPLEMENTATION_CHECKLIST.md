# Theory-Aligned Components - Implementation Checklist

**Status:** Ready to Implement  
**Date:** January 8, 2026  
**Estimated Total Time:** 8-12 hours (1-2 days)  
**Dependencies:** None (uses existing PyTorch)

---

## Phase 1: PhiResidual (2-3 hours)

### Step 1.1: Create Core Module
**File:** `python/syntonic/nn/layers/phi_residual.py`  
**Lines:** ~100  
**Time:** 1 hour

**Implementation:**
```python
"""
PhiResidual: Golden ratio-weighted residual connections.

Implements: output = identity + residual/φ
"""
import math
import torch
import torch.nn as nn

PHI = (1 + math.sqrt(5)) / 2

class PhiResidual(nn.Module):
    """
    Golden ratio-weighted residual connection.
    
    Modes:
    - 'phi': output = identity + residual/φ (default)
    - 'phi_symmetric': output = (identity + residual)/φ
    - 'standard': output = identity + residual (for ablation)
    """
    
    def __init__(self, module: nn.Module, mode: str = 'phi'):
        super().__init__()
        self.module = module
        self.mode = mode
        if mode not in ['phi', 'phi_symmetric', 'standard']:
            raise ValueError(f"Unknown mode: {mode}")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        residual = self.module(x)
        
        if self.mode == 'phi':
            return identity + residual / PHI
        elif self.mode == 'phi_symmetric':
            return (identity + residual) / PHI
        else:  # standard
            return identity + residual

def make_phi_residual_block(module: nn.Module, mode: str = 'phi') -> PhiResidual:
    """Convenience function to wrap a module in PhiResidual."""
    return PhiResidual(module, mode=mode)
```

**Checklist:**
- [ ] Create file `python/syntonic/nn/layers/phi_residual.py`
- [ ] Implement `PhiResidual` class with three modes
- [ ] Implement `make_phi_residual_block()` helper
- [ ] Add docstrings with usage examples

---

### Step 1.2: Create Test Suite
**File:** `tests/test_nn/test_phi_residual.py`  
**Lines:** ~150  
**Time:** 1 hour

**Tests to Implement:**
```python
import pytest
import torch
import torch.nn as nn
import math
from syntonic.nn.layers.phi_residual import PhiResidual

PHI = (1 + math.sqrt(5)) / 2

def test_phi_residual_mode_phi():
    """Verify phi mode: output = identity + residual/φ"""
    module = nn.Linear(10, 10)
    block = PhiResidual(module, mode='phi')
    x = torch.randn(32, 10)
    y = block(x)
    
    expected = x + module(x) / PHI
    assert torch.allclose(y, expected, atol=1e-6)

def test_phi_residual_mode_symmetric():
    """Verify symmetric mode: output = (identity + residual)/φ"""
    # ... implementation

def test_phi_residual_mode_standard():
    """Verify standard mode matches regular residual"""
    # ... implementation

def test_phi_residual_gradient():
    """Verify gradients flow correctly"""
    # ... implementation

def test_phi_residual_preserves_magnitude():
    """Verify phi residual dampens growth compared to standard"""
    # ... implementation

def test_phi_residual_in_mlp():
    """Integration test with multi-layer MLP"""
    # ... implementation
```

**Checklist:**
- [ ] Create file `tests/test_nn/test_phi_residual.py`
- [ ] Test mode='phi' correctness
- [ ] Test mode='phi_symmetric' correctness
- [ ] Test mode='standard' correctness
- [ ] Test gradient flow
- [ ] Test magnitude dampening property
- [ ] Integration test with MLP

---

### Step 1.3: Update Exports and Documentation
**Time:** 30 minutes

**Files to Update:**

1. **`python/syntonic/nn/layers/__init__.py`**
   ```python
   from .phi_residual import PhiResidual, make_phi_residual_block
   
   __all__ = [
       # ... existing exports
       'PhiResidual',
       'make_phi_residual_block',
   ]
   ```

2. **`python/syntonic/nn/__init__.py`**
   ```python
   from syntonic.nn.layers import (
       # ... existing imports
       PhiResidual,
       make_phi_residual_block,
   )
   
   __all__ = [
       # ... existing exports
       'PhiResidual',
       'make_phi_residual_block',
   ]
   ```

3. **`library_build_docs/SYNTONIC_API_REFERENCE.md`**
   Add section:
   ```markdown
   ### PhiResidual
   
   Golden ratio-weighted residual connections.
   
   ```python
   from syntonic.nn.layers import PhiResidual
   
   # Wrap any module
   block = PhiResidual(nn.Sequential(
       nn.Linear(128, 128),
       nn.ReLU(),
   ), mode='phi')
   
   x = torch.randn(32, 128)
   y = block(x)  # output = x + module(x)/φ
   ```
   
   **Modes:**
   - `phi`: output = identity + residual/φ (default, recommended)
   - `phi_symmetric`: output = (identity + residual)/φ
   - `standard`: standard residual (for ablation studies)
   ```

**Checklist:**
- [ ] Update `python/syntonic/nn/layers/__init__.py`
- [ ] Update `python/syntonic/nn/__init__.py`
- [ ] Add API documentation to `SYNTONIC_API_REFERENCE.md`
- [ ] Run tests: `pytest tests/test_nn/test_phi_residual.py -v`

---

## Phase 2: GoldenBatchNorm2d (2-3 hours)

### Step 2.1: Create Core Module
**File:** `python/syntonic/nn/layers/golden_batch_norm.py`  
**Lines:** ~180  
**Time:** 1.5 hours

**Implementation:**
```python
"""
GoldenBatchNorm2d: Batch normalization targeting variance = 1/φ.
"""
import math
import torch
import torch.nn as nn

PHI = (1 + math.sqrt(5)) / 2
PHI_INV = 1 / PHI

class GoldenBatchNorm2d(nn.Module):
    """
    Batch normalization with golden target variance.
    
    Normalizes to mean=0, variance=1/φ ≈ 0.618
    """
    
    def __init__(
        self,
        num_features: int,
        eps: float = 1e-5,
        momentum: float = 0.1,
        affine: bool = True,
        track_running_stats: bool = True,
    ):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.affine = affine
        self.track_running_stats = track_running_stats
        
        if affine:
            self.weight = nn.Parameter(torch.ones(num_features))
            self.bias = nn.Parameter(torch.zeros(num_features))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)
        
        if track_running_stats:
            self.register_buffer('running_mean', torch.zeros(num_features))
            self.register_buffer('running_var', torch.ones(num_features) / PHI)
            self.register_buffer('num_batches_tracked', torch.tensor(0, dtype=torch.long))
        else:
            self.register_buffer('running_mean', None)
            self.register_buffer('running_var', None)
            self.register_buffer('num_batches_tracked', None)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply golden batch normalization."""
        if self.training or not self.track_running_stats:
            # Compute batch statistics
            mean = x.mean(dim=(0, 2, 3), keepdim=True)
            var = x.var(dim=(0, 2, 3), unbiased=False, keepdim=True)
            
            if self.track_running_stats:
                self.running_mean = (1 - self.momentum) * self.running_mean + \
                                   self.momentum * mean.squeeze()
                self.running_var = (1 - self.momentum) * self.running_var + \
                                  self.momentum * var.squeeze()
                self.num_batches_tracked += 1
        else:
            mean = self.running_mean.view(1, -1, 1, 1)
            var = self.running_var.view(1, -1, 1, 1)
        
        # Normalize to N(0, 1)
        x_norm = (x - mean) / torch.sqrt(var + self.eps)
        
        # Scale to golden target variance: N(0, 1/φ)
        x_norm = x_norm * math.sqrt(PHI_INV)
        
        # Affine transform
        if self.affine:
            gamma = self.weight.view(1, -1, 1, 1)
            beta = self.bias.view(1, -1, 1, 1)
            x_norm = gamma * x_norm + beta
        
        return x_norm
```

**Checklist:**
- [ ] Create file `python/syntonic/nn/layers/golden_batch_norm.py`
- [ ] Implement `GoldenBatchNorm2d` class
- [ ] Add proper buffer registration for running stats
- [ ] Implement training/eval mode switching
- [ ] Add docstrings

---

### Step 2.2: Create Test Suite
**File:** `tests/test_nn/test_golden_batch_norm.py`  
**Lines:** ~200  
**Time:** 1 hour

**Tests to Implement:**
```python
import pytest
import torch
import torch.nn as nn
from syntonic.nn.layers.golden_batch_norm import GoldenBatchNorm2d

PHI = (1 + 5**0.5) / 2

def test_golden_batch_norm_variance():
    """Verify output variance ≈ 1/φ"""
    bn = GoldenBatchNorm2d(64)
    bn.eval()
    
    x = torch.randn(32, 64, 28, 28)
    y = bn(x)
    
    var = y.var(dim=(0, 2, 3))
    target_var = 1 / PHI
    assert torch.allclose(var.mean(), torch.tensor(target_var), atol=0.1)

def test_golden_batch_norm_training():
    """Verify running stats update correctly"""
    # ... implementation

def test_golden_batch_norm_gradient():
    """Verify gradients flow"""
    # ... implementation

def test_golden_bn_in_cnn():
    """Integration test in CNN"""
    # ... implementation
```

**Checklist:**
- [ ] Create file `tests/test_nn/test_golden_batch_norm.py`
- [ ] Test variance target
- [ ] Test running stats update
- [ ] Test gradient flow
- [ ] Test eval mode
- [ ] Integration test with CNN

---

### Step 2.3: Update Exports and Documentation
**Time:** 30 minutes

**Checklist:**
- [ ] Update `python/syntonic/nn/layers/__init__.py`
- [ ] Update `python/syntonic/nn/__init__.py`
- [ ] Add API documentation to `SYNTONIC_API_REFERENCE.md`
- [ ] Run tests: `pytest tests/test_nn/test_golden_batch_norm.py -v`

---

## Phase 3: SyntonicSoftmax (4-6 hours)

### Step 3.1: Create Core Module
**File:** `python/syntonic/nn/layers/syntonic_softmax.py`  
**Lines:** ~250  
**Time:** 2.5 hours

**Implementation:**
```python
"""
SyntonicSoftmax: Syntony-weighted softmax.
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

PHI = (1 + math.sqrt(5)) / 2

class SyntonicSoftmax(nn.Module):
    """
    Syntony-weighted softmax.
    
    Three modes:
    - 'learned': Learn mode norms per feature
    - 'provided': Accept pre-computed syntony values
    - 'identity': Standard softmax (ablation)
    """
    
    def __init__(
        self,
        dim: int = -1,
        mode: str = 'learned',
        num_features: Optional[int] = None,
        syntony_scale: float = 1.0,
    ):
        super().__init__()
        self.dim = dim
        self.mode = mode
        self.syntony_scale = syntony_scale
        
        if mode == 'learned':
            assert num_features is not None
            self.mode_norms = nn.Parameter(torch.ones(num_features))
        else:
            self.mode_norms = None
    
    def forward(
        self,
        x: torch.Tensor,
        syntony: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Apply syntony-weighted softmax."""
        if self.mode == 'identity':
            return F.softmax(x, dim=self.dim)
        
        elif self.mode == 'learned':
            # w(n) = exp(-|n|²/φ)
            weights = torch.exp(-self.mode_norms / PHI)
            
            # Broadcast to match input shape
            shape = [1] * x.ndim
            shape[self.dim] = -1
            weights = weights.view(shape)
            
            weighted_x = x + self.syntony_scale * torch.log(weights + 1e-8)
            return F.softmax(weighted_x, dim=self.dim)
        
        elif self.mode == 'provided':
            assert syntony is not None
            assert syntony.shape == x.shape
            
            weights = torch.pow(syntony.clamp(min=1e-6), self.syntony_scale)
            weighted_x = x + torch.log(weights + 1e-8)
            return F.softmax(weighted_x, dim=self.dim)
        
        else:
            raise ValueError(f"Unknown mode: {self.mode}")

class SyntonicClassifier(nn.Module):
    """
    Classifier with syntony-weighted softmax.
    """
    def __init__(self, d_model: int, num_classes: int):
        super().__init__()
        self.linear = nn.Linear(d_model, num_classes)
        self.softmax = SyntonicSoftmax(
            dim=-1,
            mode='learned',
            num_features=num_classes,
            syntony_scale=1.0,
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.linear(x)
        return self.softmax(logits)
```

**Checklist:**
- [ ] Create file `python/syntonic/nn/layers/syntonic_softmax.py`
- [ ] Implement `SyntonicSoftmax` with three modes
- [ ] Implement `SyntonicClassifier`
- [ ] Add docstrings

---

### Step 3.2: Create Test Suite
**File:** `tests/test_nn/test_syntonic_softmax.py`  
**Lines:** ~250  
**Time:** 2 hours

**Tests to Implement:**
```python
def test_syntonic_softmax_learned():
    """Test learned mode"""
    # ... implementation

def test_syntonic_softmax_weights_high_syntony():
    """Verify high-syntony features weighted more"""
    # ... implementation

def test_syntonic_softmax_provided():
    """Test provided mode"""
    # ... implementation

def test_syntonic_softmax_identity():
    """Test identity mode matches standard softmax"""
    # ... implementation

def test_syntonic_softmax_gradient():
    """Test gradient flow"""
    # ... implementation

def test_syntonic_classifier():
    """Test full classifier"""
    # ... implementation
```

**Checklist:**
- [ ] Create file `tests/test_nn/test_syntonic_softmax.py`
- [ ] Test all three modes
- [ ] Test syntony weighting property
- [ ] Test gradient flow
- [ ] Test classifier integration
- [ ] Test probability sum=1 invariant

---

### Step 3.3: Update Exports and Documentation
**Time:** 1 hour

**Checklist:**
- [ ] Update `python/syntonic/nn/layers/__init__.py`
- [ ] Update `python/syntonic/nn/__init__.py`
- [ ] Add API documentation to `SYNTONIC_API_REFERENCE.md`
- [ ] Run tests: `pytest tests/test_nn/test_syntonic_softmax.py -v`

---

## Phase 4: Integration & Validation (2-3 hours)

### Step 4.1: Update Existing Architectures
**Time:** 1.5 hours

**Files to Update:**

1. **`python/syntonic/nn/architectures/syntonic_mlp.py`**
   - Replace manual `self._residual_scale = 1.0 / PHI` with `PhiResidual`
   
2. **`python/syntonic/nn/architectures/syntonic_cnn.py`**
   - Replace `nn.BatchNorm2d` with `GoldenBatchNorm2d` (optional flag)
   - Replace manual `+ identity / PHI` with consistent pattern

3. **`python/syntonic/nn/architectures/syntonic_transformer.py`**
   - Update residual connections to use `PhiResidual`

**Checklist:**
- [ ] Update SyntonicMLP to use PhiResidual
- [ ] Add use_golden_norm flag to SyntonicCNN
- [ ] Update transformer residuals
- [ ] Maintain backward compatibility

---

### Step 4.2: Run Full Test Suite
**Time:** 30 minutes

```bash
# Run all new tests
pytest tests/test_nn/test_phi_residual.py -v
pytest tests/test_nn/test_golden_batch_norm.py -v
pytest tests/test_nn/test_syntonic_softmax.py -v

# Run integration tests
pytest tests/test_nn/ -v

# Check coverage
pytest tests/test_nn/ --cov=syntonic.nn.layers --cov-report=term-missing
```

**Checklist:**
- [ ] All PhiResidual tests pass
- [ ] All GoldenBatchNorm2d tests pass
- [ ] All SyntonicSoftmax tests pass
- [ ] Integration tests pass
- [ ] No regressions in existing tests

---

### Step 4.3: Simple Benchmark
**Time:** 1 hour

Create quick benchmark to verify improvements:

```python
# benchmarks/theory_components_demo.py
"""
Quick demo of theory-aligned components vs standard PyTorch.
"""
import torch
import torch.nn as nn
from syntonic.nn.layers import PhiResidual, GoldenBatchNorm2d, SyntonicClassifier

def standard_cnn():
    return nn.Sequential(
        nn.Conv2d(3, 64, 3, padding=1),
        nn.BatchNorm2d(64),
        nn.ReLU(),
        nn.Conv2d(64, 128, 3, padding=1),
        nn.BatchNorm2d(128),
        nn.ReLU(),
        nn.AdaptiveAvgPool2d(1),
        nn.Flatten(),
        nn.Linear(128, 10),
    )

def syntonic_cnn():
    return nn.Sequential(
        nn.Conv2d(3, 64, 3, padding=1),
        GoldenBatchNorm2d(64),  # Golden variance
        nn.ReLU(),
        PhiResidual(nn.Sequential(  # Phi-weighted residual
            nn.Conv2d(64, 64, 3, padding=1),
            GoldenBatchNorm2d(64),
            nn.ReLU(),
        )),
        nn.Conv2d(64, 128, 3, padding=1),
        GoldenBatchNorm2d(128),
        nn.ReLU(),
        nn.AdaptiveAvgPool2d(1),
        nn.Flatten(),
        SyntonicClassifier(128, 10),  # Syntonic softmax
    )

# Quick MNIST test
# ... training loop comparing both
```

**Checklist:**
- [ ] Create benchmark script
- [ ] Compare standard vs syntonic on MNIST
- [ ] Verify syntonic converges
- [ ] Document results

---

## Phase 5: Documentation & Cleanup (1-2 hours)

### Step 5.1: Update Documentation

**Files to Update:**
1. `library_build_docs/SYNTONIC_API_REFERENCE.md` - Add all three components
2. `README.md` - Add quick example
3. `CHANGELOG.md` - Add release notes

**Checklist:**
- [ ] Complete API reference entries
- [ ] Add usage examples
- [ ] Add migration guide (standard → syntonic)
- [ ] Update README with new components

---

### Step 5.2: Final Verification

**Checklist:**
- [ ] All tests pass: `pytest tests/test_nn/ -v`
- [ ] Code formatted: `black python/syntonic/nn/`
- [ ] Type hints correct: `mypy python/syntonic/nn/layers/`
- [ ] No lint errors: `ruff check python/syntonic/nn/`
- [ ] Documentation builds correctly
- [ ] Examples run without errors

---

## Summary Checklist

### Phase 1: PhiResidual (2-3 hours)
- [ ] Implement core module
- [ ] Write tests
- [ ] Update exports
- [ ] Update documentation

### Phase 2: GoldenBatchNorm2d (2-3 hours)
- [ ] Implement core module
- [ ] Write tests
- [ ] Update exports
- [ ] Update documentation

### Phase 3: SyntonicSoftmax (4-6 hours)
- [ ] Implement core module
- [ ] Write tests
- [ ] Update exports
- [ ] Update documentation

### Phase 4: Integration (2-3 hours)
- [ ] Update existing architectures
- [ ] Run full test suite
- [ ] Create benchmark

### Phase 5: Documentation (1-2 hours)
- [ ] Update API docs
- [ ] Update README
- [ ] Final verification

---

## Total Estimated Time: 11-17 hours (1.5-2 days)

**Recommended Schedule:**
- **Day 1 Morning:** PhiResidual (complete)
- **Day 1 Afternoon:** GoldenBatchNorm2d (complete)
- **Day 1 Evening:** SyntonicSoftmax (start)
- **Day 2 Morning:** SyntonicSoftmax (finish)
- **Day 2 Afternoon:** Integration & testing
- **Day 2 Evening:** Documentation & cleanup

---

## Success Criteria

✅ **All three components implemented**  
✅ **100% test coverage**  
✅ **All tests passing**  
✅ **Documentation complete**  
✅ **Benchmark shows syntonic benefits**  
✅ **Backward compatible with existing code**  
✅ **Ready for production use**

---

**Next Action:** Start with Phase 1, Step 1.1 (Create PhiResidual module)
