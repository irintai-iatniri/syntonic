# Syntonic Library Purification Plan

## Overview
Refactor 37 files to remove PyTorch/NumPy dependencies and use the pure Rust backend (`syntonic._core`). This involves replacing gradient-based optimization with RES (Resonant Evolution Strategy) and migrating all tensor operations to `ResonantTensor`.

## Status Update
✅ **Phase 0 Complete**: Rust APIs Added
- Added `sigmoid()`, `tanh()`, `elementwise_mul()`, `elementwise_add()`, `layer_norm()` to ResonantTensor
- All APIs tested and working
- Documentation: `NEW_API_REFERENCE.md`

## Key Architecture Changes

### Replacement Patterns
- `torch.Tensor` → `ResonantTensor` (dual-state: Q(φ) lattice + GPU flux)
- `nn.Module` → Plain Python classes
- `nn.Linear` → `ResonantLinear` (already exists)
- `nn.Parameter` → `ResonantParameter` (already exists)
- Gradient descent → RES (evolutionary selection on golden lattice)
- `loss.backward()` → `tensor.cpu_cycle()` (DHSR cycle)
- PyTorch optimizers → DELETE (RES handles all optimization)

### Available APIs (Updated)
- **ResonantTensor**: `.syntony`, `.matmul()`, `.relu()`, `.sigmoid()`, `.tanh()`, `.cpu_cycle()`, `.apply_recursion()`, `.wake_flux()`, `.crystallize()`
- **New APIs**: `.elementwise_mul()`, `.elementwise_add()`, `.layer_norm(gamma, beta, eps, golden_target)`
- **CUDA Kernels**: `layer_norm_f64`, `sigmoid_f64`, `tanh_f64`, `exp_golden_f64`, `compute_syntony_f32`, `dhsr_cycle_f32`
- **Completed**: `ResonantLinear`, `ResonantParameter`, `WindingStateEmbedding`, `WindingDHSRBlock`

### Critical Missing APIs
⚠️ **Must implement before Phase 1.2**:
1. **`concat()`** - Concatenation along dimension (CRITICAL for gates and multi-head)
2. **`gelu()`** - GELU activation (for advanced gates)
3. **`norm()`** - L2 norm computation (for syntony diagnostics)
4. Statistics utilities (for running stats in normalization)

## Implementation Phases

### Phase 1: Foundation Infrastructure (Week 1)

**1.1 Pure Python Utilities (Simple - 2-3 days)**

#### `python/syntonic/benchmarks/datasets.py`
**NumPy Operations to Replace** (~40 operations):
- Random generation: `RandomState`, `uniform`, `normal`, `permutation`
- Array ops: `vstack`, `hstack`, `column_stack`, `zeros`, `ones`
- Math: `linspace`, `cos`, `sin`, `pi`
- Type conversions: `astype(float64)`, `astype(int64)`

**Implementation Pattern**:
```python
class PureRNG:
    """Pure Python RNG using stdlib random."""
    def __init__(self, seed=None):
        import random
        self.rng = random.Random(seed)

    def uniform(self, low, high, size):
        # Box-Muller for normal, Fisher-Yates for permutation
        # Return nested lists instead of arrays
```

**Key Functions**:
- `make_xor()`, `make_moons()`, `make_circles()`, `make_spiral()` - Use PureRNG and list operations
- `linspace()` - Pure Python generator
- Array stacking - List comprehensions

**Estimated Effort**: 1-2 days (well-understood replacements)

#### `python/syntonic/benchmarks/fitness.py`
**NumPy Operations to Replace** (~15 operations):
- `softmax` - `exp`, `sum` with numerical stability
- `cross_entropy` - `log`, indexing
- `accuracy` - `argmax`, comparison
- Matrix ops - Use `ResonantTensor.to_floats()` then pure Python

**Implementation Pattern**:
```python
def softmax_pure(logits):
    """Numerically stable softmax."""
    import math
    maxes = [max(row) for row in logits]
    exp_vals = [[math.exp(logits[i][j] - maxes[i])
                 for j in range(len(logits[i]))]
                for i in range(len(logits))]
    sums = [sum(row) for row in exp_vals]
    return [[exp_vals[i][j] / sums[i]
             for j in range(len(exp_vals[i]))]
            for i in range(len(exp_vals))]
```

**Estimated Effort**: 1 day

---

**1.2 Core Layer Primitives (Medium - 4-5 days)**

⚠️ **BLOCKED**: Requires `concat()` API for multi-head modules and gates

#### `python/syntonic/nn/layers/normalization.py`
**Current**: 260 lines, uses `nn.Module`, `nn.Parameter`, torch stats
**Target**: Pure Python class with ResonantTensor

**Classes to Refactor**:
1. **SyntonicNorm** (lines 22-93)
   - Use `layer_norm(golden_target=True)` with gamma/beta as ResonantParameter
   - Xavier initialization → Initialize ResonantParameter with golden measure

2. **GoldenNorm** (lines 99-186)
   - Layer norm with running statistics tracking (CPU-side Python state)
   - Use `layer_norm(golden_target=True)`
   - Momentum updates: Pure Python logic

3. **RecursionLayerNorm** (lines 192-256)
   - Depth-indexed parameters (keep as Python dict)
   - Golden decay: `φ^(-i)` for depth `i`
   - Use `layer_norm()` + `elementwise_mul()` for scaling

**Key Changes**:
- Remove `nn.Module` → Plain Python class
- Remove `nn.Parameter` → `ResonantParameter`
- Keep initialization patterns (golden measure)

**Estimated Effort**: 1 day

#### `python/syntonic/nn/layers/differentiation.py`
**Current**: 162 lines, `nn.Module` with multi-head support
**Target**: Pure composition of ResonantLinear + activation

**Classes to Refactor**:
1. **DifferentiationLayer** (lines 23-91)
   ```python
   # Current: d_x = alpha_scale * F.relu(self.linear(x))
   # New:
   def forward(self, x: ResonantTensor) -> ResonantTensor:
       d_x = self.linear.forward(x)  # ResonantLinear
       d_x.relu()  # In-place activation
       # Scale by alpha
       alpha_tensor = ResonantTensor([self.alpha_scale] * d_x.len(), d_x.shape)
       d_x = d_x.elementwise_mul(alpha_tensor)
       return d_x
   ```

2. **DifferentiationModule** (lines 97-158) - **BLOCKED by concat()**
   - Multi-head projection (one linear per head)
   - Concatenate heads → **NEEDS concat() API**
   - Output projection

**Estimated Effort**: 1 day (+ waiting for concat API)

#### `python/syntonic/nn/layers/harmonization.py`
**Current**: 169 lines, dual pathway (damping + syntony)
**Target**: Pure ResonantTensor operations

**Formula**: `x - β·σ(W_H·x) + γ·tanh(W_S·x)`

**Implementation**:
```python
def forward(self, x: ResonantTensor) -> ResonantTensor:
    # Damping pathway
    damp = self.damp_linear.forward(x)
    damp.sigmoid(precision=100)
    damp = damp.elementwise_mul(beta_scalar)

    # Syntony pathway
    syntony = self.syntony_linear.forward(x)
    syntony.tanh(precision=100)
    syntony = syntony.elementwise_mul(gamma_scalar)

    # Combine: x - damp + syntony
    result = x.elementwise_add(damp.negate())
    result = result.elementwise_add(syntony)
    return result
```

**Estimated Effort**: 1 day

#### `python/syntonic/nn/layers/recursion.py`
**Current**: 236 lines, orchestrates D+H+Gate
**Target**: Pure composition (no direct tensor ops)

**RecursionBlock flow**:
```
Input → DifferentiationLayer → HarmonizationLayer → SyntonicGate → Output
```

**Implementation**: Just call refactored component layers
- No direct tensor operations
- Pure composition logic
- Syntony tracking (diagnostic)

**Estimated Effort**: 0.5 days (after components done)

#### `python/syntonic/nn/layers/syntonic_gate.py` - **BLOCKED**
**Current**: 185 lines, adaptive gating
**Target**: Pure gating logic

**CRITICAL BLOCKER**: Requires `concat()` for `[x, x_processed]`

**SyntonicGate formula**:
```python
combined = concat([x, x_processed])  # NEEDS concat() API
gate = sigmoid(linear(relu(linear(combined))))
output = gate * x_processed + (1-gate) * x
```

**Estimated Effort**: 1 day (after concat API)

### Phase 2: Winding Components (Week 2)

- `python/syntonic/nn/winding/syntony.py` - Use `ResonantTensor.syntony` property
- `python/syntonic/nn/winding/prime_selection.py` - Pure Python bitwise ops + masking
- `python/syntonic/nn/winding/resonant_embedding.py` - Already refactored, verify imports
- `python/syntonic/nn/winding/winding_net.py` - Compose pure layers (depends on Phase 1)

### Phase 3: Network Architectures (Week 3)

- `python/syntonic/nn/architectures/embeddings.py` - Use `WindingStateEmbedding` pattern for token embeddings
- `python/syntonic/nn/architectures/syntonic_mlp.py` - Chain `ResonantLinear` layers manually
- `python/syntonic/nn/architectures/syntonic_attention.py` - Use `GoldenConeAttention` from `python/syntonic/pure/resonant_transformer.py`
- `python/syntonic/nn/architectures/syntonic_transformer.py` - Adapt `PureResonantTransformer` from `pure/` directory
- `python/syntonic/nn/architectures/syntonic_cnn.py` - **REQUIRES API**: Add `.conv2d()` method OR use im2col workaround

**Optional API for CNN (can defer):**
```rust
// In rust/src/resonant/tensor.rs
pub fn conv2d(&self, kernel: &ResonantTensor, bias: Option<&ResonantTensor>, stride: usize, padding: usize) -> ResonantTensor
```

### Phase 4: Loss and Optimization (Week 4)

**4.1 Loss Functions**
- `python/syntonic/nn/loss/syntony_metrics.py` - Query `.syntony` property + pure Python aggregation
- `python/syntonic/nn/loss/phase_alignment.py` - Use `.phase` property + pure Python math
- `python/syntonic/nn/loss/regularization.py` - Use `.to_floats()` + pure Python norm calculation
- `python/syntonic/nn/loss/syntonic_loss.py` - Pure Python function combining task loss + syntony penalty

**4.2 DELETE Optimizers (4 files)**
- `python/syntonic/nn/optim/gradient_mod.py` - DELETE
- `python/syntonic/nn/optim/schedulers.py` - DELETE
- `python/syntonic/nn/optim/syntonic_adam.py` - DELETE
- `python/syntonic/nn/optim/syntonic_sgd.py` - DELETE
- Update `python/syntonic/nn/optim/__init__.py` to remove references

**Replacement**: All training uses RES (already in `rust/src/resonant/evolver.rs`)

### Phase 5: Training Infrastructure (Week 5)

- `python/syntonic/nn/training/metrics.py` - Pure Python metrics on `ResonantTensor.to_floats()`
- `python/syntonic/nn/training/callbacks.py` - Pure Python callback system for RES generations
- `python/syntonic/nn/training/trainer.py` - **CRITICAL**: Replace gradient descent with RES-based training loop

**RESTrainer Pattern:**
```python
class RESTrainer:
    def train(self, data, epochs=100):
        params = self._collect_parameters(self.model)

        for epoch in range(epochs):
            for param in params:
                evolver = RESEvolver(param.tensor, self.config)

                def fitness(candidate):
                    # Evaluate model with candidate parameter
                    return -loss + 0.1 * candidate.syntony

                result = evolver.run_with_fitness(fitness, generations=10)
                param._tensor = result.best_tensor
```

### Phase 6: Analysis and Benchmarks (Week 6)

**6.1 Analysis Tools**
- `python/syntonic/nn/analysis/health.py` - Pure Python stats on `.to_floats()`
- `python/syntonic/nn/analysis/escape.py` - Pure Python syntony history analysis
- `python/syntonic/nn/analysis/visualization.py` - Keep matplotlib, change data source to `.to_floats()`

**6.2 Internal Benchmarks**
- `python/syntonic/nn/benchmarks/standard.py` - Use pure components
- `python/syntonic/nn/benchmarks/ablation.py` - Use pure components
- `python/syntonic/nn/benchmarks/convergence.py` - Compare RES methodology

**6.3 Root Benchmarks**
- `python/syntonic/benchmarks/winding_xor_benchmark.py` - Pure `WindingNet` + RES
- `python/syntonic/benchmarks/winding_benchmark.py` - Pure implementation
- `python/syntonic/benchmarks/convergence_benchmark.py` - RES-based training
- `python/syntonic/benchmarks/comparative_resonant_benchmark.py` - Pure Python data gen
- Verify `pure_resonant_benchmark.py`, `thorough_pure_benchmark.py`, `transformer_benchmark.py` are already pure

## Dependency Order

Files must be refactored in this order:

```
Phase 1: datasets.py, fitness.py → normalization.py → differentiation.py, harmonization.py → recursion.py, syntonic_gate.py
Phase 2: syntony.py, prime_selection.py → winding_net.py
Phase 3: embeddings.py, syntonic_mlp.py, syntonic_attention.py → syntonic_transformer.py, syntonic_cnn.py
Phase 4: syntony_metrics.py, phase_alignment.py, regularization.py → syntonic_loss.py → DELETE optim/*
Phase 5: metrics.py, callbacks.py → trainer.py
Phase 6: All analysis and benchmarks (mostly independent)
```

## Critical Files

Top 5 files requiring most attention:
1. `rust/src/resonant/tensor.rs` - Add 4-5 new methods (layer_norm, sigmoid, elementwise ops, conv2d)
2. `python/syntonic/nn/training/trainer.py` - Complete paradigm shift to RES
3. `python/syntonic/nn/layers/normalization.py` - Foundation for all layers
4. `python/syntonic/nn/loss/syntonic_loss.py` - Central loss function
5. `python/syntonic/nn/architectures/syntonic_cnn.py` - Most complex architecture

## Validation Strategy

After each phase:
1. Create unit tests for refactored files
2. Verify syntony preservation (syntony > 0)
3. Run `winding_xor_benchmark.py` to compare performance
4. Test full pipeline: data → model → loss → RES → output

## Risk Mitigation

**High-Risk Items:**
- Conv2D kernel development → **Mitigation**: Use im2col workaround initially
- RES trainer paradigm shift → **Mitigation**: Keep PyTorch version alongside for comparison
- Elementwise operations → **Mitigation**: CPU fallbacks first, optimize later

**Medium-Risk Items:**
- Loss function semantics → Ensure pure Python matches PyTorch behavior
- Callback system → Iterate on RES callback design

## Success Criteria

- ✅ Zero PyTorch/NumPy imports in all 37 files
- ✅ All benchmarks pass with accuracy ≥ PyTorch baseline
- ✅ RES training converges (syntony increases over generations)
- ✅ All 4 optim files deleted
- ✅ Full test suite passes

## Execution Strategy

### Option A: Sequential (Recommended)
1. **Phase 1.1 First** (datasets.py, fitness.py) - No blockers, 2-3 days
2. **Add concat() API** - 1 day Rust implementation
3. **Phase 1.2** (core layers) - 4-5 days with all APIs available
4. Continue with Phase 2+

**Advantages**: Early wins, test pure Python patterns, unblock Phase 1.2

### Option B: API-First
1. **Add concat() API first** - 1 day
2. **Phase 1.1 + 1.2 in parallel** - Can work on both simultaneously
3. Continue with Phase 2+

**Advantages**: Complete all Phase 1 together

**Recommendation**: **Option A** - Get Phase 1.1 working first to validate pure Python patterns and build confidence

---

## Next Immediate Steps

1. ✅ Rust APIs (sigmoid, tanh, elementwise_mul/add, layer_norm) - DONE
2. **START HERE**: Refactor `datasets.py` (1-2 days)
   - Implement PureRNG class
   - Replace all NumPy array operations with list operations
   - Test with existing benchmarks
3. Refactor `fitness.py` (1 day)
   - Implement pure softmax, cross_entropy, accuracy
   - Use ResonantTensor.to_floats() for data extraction
4. Add `concat()` API to ResonantTensor (1 day)
5. Proceed with Phase 1.2 (normalization, diff, harm, recursion, gate)

---

## Summary

**Total Files**: 37
- 4 DELETE (optim/)
- 33 REFACTOR (replace PyTorch → Pure Rust backend)
- **Phase 1.1**: 2 files (datasets.py, fitness.py) - Ready to start
- **Phase 1.2**: 5 files (layers) - Blocked on concat() API
- Estimated effort: 3-4 weeks full-time, 6-8 weeks part-time

**Critical Path**:
```
Phase 1.1 (3 days) → concat() API (1 day) → Phase 1.2 (5 days) → Phase 2 (4 days) → ...
```
