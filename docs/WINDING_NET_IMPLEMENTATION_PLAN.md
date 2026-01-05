# WindingNet Implementation Plan

**Version:** 1.0
**Date:** 2026-01-05
**Status:** Planning Complete, Ready for Implementation

---

## Executive Summary

This document outlines the implementation plan for **WindingNet**, a winding-aware neural network that integrates number-theoretic selection rules with deep learning. WindingNet builds on the existing Syntonic infrastructure, combining:

1. **Winding State Topology** - T^4 torus windings |n₇, n₈, n₉, n₁₀⟩
2. **Prime Selection** - Möbius filtering for matter channel
3. **Fibonacci Hierarchy** - Golden ratio depth scaling
4. **DHSR Dynamics** - Differentiation-Harmonization cycles
5. **Temporal Blockchain** - Immutable state recording
6. **Syntony Consensus** - ΔS > threshold validation

---

## Table of Contents

1. [Existing Infrastructure](#1-existing-infrastructure)
2. [Components to Implement](#2-components-to-implement)
3. [Implementation Phases](#3-implementation-phases)
4. [File Structure](#4-file-structure)
5. [Testing Strategy](#5-testing-strategy)
6. [Benchmarking Plan](#6-benchmarking-plan)
7. [API Reference](#7-api-reference)

---

## 1. Existing Infrastructure

### 1.1 What Already Exists

The Syntonic codebase provides a comprehensive foundation:

#### **Winding States** (✅ Production Ready)
- **Location:** `rust/src/winding.rs`, `python/syntonic/srt/geometry/winding.py`
- **Class:** `WindingState(n7, n8, n9, n10)`
- **Properties:**
  - `norm_squared()`: |n|² = n₇² + n₈² + n₉² + n₁₀²
  - `golden_weight()`: exp(-|n|²/φ)
  - `generation()`: Recursion depth k
- **Functions:**
  - `enumerate_windings(max_norm)`: Fast enumeration
  - `enumerate_windings_by_norm(max_norm_sq)`: Grouped by |n|²
  - Particle definitions: `ELECTRON_WINDING`, `MUON_WINDING`, etc.

#### **DHSR Operators** (✅ Production Ready)
- **Location:** `python/syntonic/crt/operators/`
- **Operators:**
  - `DifferentiationOperator`: D̂ with mode amplification
  - `HarmonizationOperator`: Ĥ with mode damping
  - `RecursionOperator`: R̂ = Ĥ ∘ D̂
- **Syntony Computation:** `python/syntonic/crt/metrics/syntony.py`
  - `SyntonyComputer.compute(state)`: Returns S ∈ [0, 1]
  - Geometric version: `SyntonyFunctional` using winding eigenvalues

#### **PyTorch Neural Layers** (✅ Production Ready)
- **Location:** `python/syntonic/nn/layers/`
- **Components:**
  - `DifferentiationLayer(nn.Module)`: D̂ as neural layer
  - `HarmonizationLayer(nn.Module)`: Ĥ as neural layer
  - `RecursionBlock(nn.Module)`: Combined DHSR block
  - `SyntonicGate(nn.Module)`: Gating mechanism
  - `GoldenNorm(nn.Module)`: Normalization with φ scaling

#### **Training Infrastructure** (✅ Production Ready)
- **Optimizers:** `SyntonicAdam`, `SyntonicSGD` with syntony-modulated learning rates
- **Loss Functions:** `SyntonicLoss`, `LayerwiseSyntonicLoss`, `PhaseAlignmentLoss`
- **Trainer:** `SyntonicTrainer` with complete training loop
- **Monitoring:** `SyntonyMonitor`, `ArchonicDetector` for pattern analysis

#### **Golden Ratio Arithmetic** (✅ Production Ready)
- **Exact:** `GoldenExact` (a + b·φ) in Rust
- **Constants:** `PHI`, `PHI_INV`, `PHI_INV_SQ`, `Q_DEFICIT`
- **Sequences:** `fibonacci(n)`, `lucas(n)`

### 1.2 What Needs to Be Built

The following components are **missing** and need implementation:

| Component | Status | Priority |
|-----------|--------|----------|
| WindingEmbedding | ❌ Not implemented | P0 - Critical |
| PrimeSelectionLayer | ❌ Not implemented | P0 - Critical |
| FibonacciHierarchy | ❌ Not implemented | P1 - High |
| WindingSyntonyComputer | ❌ Not implemented | P0 - Critical |
| WindingDHSRBlock | ❌ Not implemented | P0 - Critical |
| WindingNet | ❌ Not implemented | P0 - Critical |
| Temporal Blockchain | ❌ Not implemented | P2 - Medium |
| Consensus Mechanism | ❌ Not implemented | P2 - Medium |

---

## 2. Components to Implement

### 2.1 WindingEmbedding

**Purpose:** Map winding states to neural embeddings

**Architecture:**
```python
class WindingEmbedding(nn.Module):
    def __init__(self, max_n: int = 5, embed_dim: int = 64):
        # Enumerate all windings in [-max_n, max_n]^4
        self.windings = enumerate_windings(max_n)

        # Create learnable embedding for each winding
        # Use nn.Embedding or nn.ModuleDict
        self.embeddings = nn.ModuleDict({
            self._key(w): nn.Parameter(torch.randn(embed_dim))
            for w in self.windings
        })

        # Store mode norms |n|^2 for each winding
        self.mode_norms = {
            self._key(w): w.norm_squared()
            for w in self.windings
        }
```

**Key Methods:**
- `forward(winding: WindingState) → Tensor`: Embed single winding
- `batch_forward(windings: List[WindingState]) → Tensor`: Batch embedding
- `get_mode_norm(winding: WindingState) → float`: Get |n|²

**Implementation Details:**
- **Winding enumeration:** Use existing `enumerate_windings(max_n)` from Rust
- **Key generation:** String format `"n7,n8,n9,n10"` for dict keys
- **Initialization:** Xavier/He initialization or golden-ratio based
- **Optional:** Positional encoding based on |n|² for better inductive bias

**File:** `python/syntonic/nn/winding/embedding.py`

---

### 2.2 PrimeSelectionLayer

**Purpose:** Filter activations via Möbius function (prime channel)

**Architecture:**
```python
class PrimeSelectionLayer(nn.Module):
    def __init__(self, dim: int):
        # Compute Möbius function μ(k) for k = 1..dim
        self.mobius_values = self._compute_mobius(dim)

        # Prime mask: |μ(k)| = 1 for prime/prime-power indices
        self.register_buffer(
            'prime_mask',
            torch.tensor([abs(μ) == 1 for μ in self.mobius_values])
        )

    def forward(self, x: Tensor) -> Tensor:
        # Element-wise multiplication
        return x * self.prime_mask
```

**Möbius Function Computation:**
```python
def _compute_mobius(self, n: int) -> List[int]:
    """
    μ(k) = 1 if k is square-free with even # prime factors
    μ(k) = -1 if k is square-free with odd # prime factors
    μ(k) = 0 if k has squared prime factor
    """
    # Use sieve-based algorithm
    mu = [0] * (n + 1)
    mu[1] = 1

    for i in range(1, n + 1):
        for j in range(2 * i, n + 1, i):
            mu[j] -= mu[i]

    return mu[1:]
```

**Properties:**
- Prime indices (2, 3, 5, 7, 11, ...) have mask = 1.0
- Composite indices with square factors have mask = 0.0
- Square-free composites have mask = 1.0
- This implements the "hadron channel" filtering

**File:** `python/syntonic/nn/winding/prime_selection.py`

---

### 2.3 FibonacciHierarchy

**Purpose:** Manage network depth following Fibonacci sequence

**Architecture:**
```python
class FibonacciHierarchy(nn.Module):
    def __init__(self, max_depth: int = 5):
        # Generate Fibonacci sequence
        self.fib_dims = self._fibonacci(max_depth + 2)
        # [1, 1, 2, 3, 5, 8, 13, 21, 34, 55, ...]

    def get_layer_dims(self, base_dim: int) -> List[int]:
        """Scale base dimension by Fibonacci numbers."""
        return [base_dim * f for f in self.fib_dims]

    def get_expansion_factor(self, level: int) -> int:
        """Get expansion for D-phase at level."""
        return self.fib_dims[level + 1]
```

**Properties:**
- Layer widths grow as Fibonacci numbers
- Respects golden ratio scaling: F_{n+1}/F_n → φ
- D-phase expansion uses next Fibonacci number
- Natural hierarchy: fundamental → first harmonic → ...

**File:** `python/syntonic/nn/winding/fibonacci_hierarchy.py`

---

### 2.4 WindingSyntonyComputer

**Purpose:** Compute syntony with winding-aware mode norms

**Architecture:**
```python
class WindingSyntonyComputer(nn.Module):
    def __init__(self, dim: int):
        self.dim = dim

    def forward(
        self,
        x: Tensor,           # (batch, dim)
        mode_norms: Tensor   # (dim,) |n|² values
    ) -> float:
        """
        S(Ψ) = Σ |ψ_i|² exp(-|n_i|²/φ) / Σ |ψ_i|²
        """
        # Energy per feature
        energy = x.pow(2)  # |ψ_i|²

        # Golden weights w(n) = exp(-|n|²/φ)
        weights = torch.exp(-mode_norms / PHI)

        # Syntony formula
        numerator = (energy * weights).sum()
        denominator = energy.sum() + 1e-8

        return (numerator / denominator).item()
```

**Integration:**
- Uses mode_norms from WindingEmbedding
- Compatible with existing SyntonyComputer API
- Returns scalar in [0, 1]

**File:** `python/syntonic/nn/winding/syntony.py`

---

### 2.5 WindingDHSRBlock

**Purpose:** Single DHSR cycle with winding structure + blockchain

**Architecture:**
```python
class WindingDHSRBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        fib_expand_factor: int = 3,
        use_prime_filter: bool = True,
    ):
        # D-layer (Fibonacci expansion)
        expanded_dim = dim * fib_expand_factor
        self.d_expand = nn.Linear(dim, expanded_dim)
        self.d_project = nn.Linear(expanded_dim, dim)

        # Prime filter
        self.prime_filter = PrimeSelectionLayer(dim) if use_prime_filter else nn.Identity()

        # H-layer
        self.h_dampen = nn.Linear(dim, dim)
        self.h_coherence = nn.Linear(dim, dim)

        # Syntony computer
        self.syntony_computer = WindingSyntonyComputer(dim)

        # Temporal blockchain
        self.register_buffer('temporal_record', torch.zeros(0, dim))
        self.register_buffer('syntony_record', torch.zeros(0))

    def forward(
        self,
        x: Tensor,
        mode_norms: Tensor,
        prev_syntony: float,
    ) -> Tuple[Tensor, float, bool]:
        # D-PHASE: Fibonacci expansion
        alpha = PHI_INV_SQ * (1.0 - prev_syntony)
        h = F.relu(self.d_expand(x))
        delta = self.d_project(h) * alpha
        x = x + delta

        # PRIME FILTER: Matter channel
        x = self.prime_filter(x)

        # H-PHASE: Harmonization
        beta = PHI_INV * prev_syntony
        golden_weights = torch.exp(-mode_norms / PHI)
        damping = torch.sigmoid(self.h_dampen(x)) * beta * (1.0 - golden_weights)
        coherence = torch.tanh(self.h_coherence(x)) * prev_syntony
        x = x - damping + coherence

        # SYNTONY: Compute new syntony
        syntony_new = self.syntony_computer(x, mode_norms)

        # CONSENSUS: ΔS > threshold
        delta_s = abs(syntony_new - prev_syntony)
        threshold = 24.0 / 1000.0  # Scaled for neural networks
        accepted = delta_s > threshold

        # RECORD: Append to blockchain if accepted
        if accepted:
            self._record_block(x.detach(), syntony_new)

        return x, syntony_new, accepted
```

**Blockchain Methods:**
```python
def _record_block(self, state: Tensor, syntony: float):
    """Append immutable block to temporal ledger."""
    self.temporal_record = torch.cat([
        self.temporal_record,
        state.mean(dim=0, keepdim=True)
    ], dim=0)

    self.syntony_record = torch.cat([
        self.syntony_record,
        torch.tensor([syntony])
    ], dim=0)

def get_blockchain_length(self) -> int:
    """Return number of recorded blocks."""
    return len(self.syntony_record)

def get_blockchain(self) -> Tuple[Tensor, Tensor]:
    """Return (states, syntonies) history."""
    return self.temporal_record, self.syntony_record
```

**File:** `python/syntonic/nn/winding/dhsr_block.py`

---

### 2.6 WindingNet (Complete Architecture)

**Purpose:** Full winding-aware neural network

**Architecture:**
```python
class WindingNet(nn.Module):
    def __init__(
        self,
        max_winding: int = 5,
        base_dim: int = 64,
        num_blocks: int = 3,
        output_dim: int = 2,
    ):
        super().__init__()

        # 1. Winding embedding layer
        self.winding_embed = WindingEmbedding(
            max_n=max_winding,
            embed_dim=base_dim,
        )

        # 2. Fibonacci hierarchy
        self.fib_hierarchy = FibonacciHierarchy(num_blocks)
        layer_dims = self.fib_hierarchy.get_layer_dims(base_dim)

        # 3. DHSR blocks (one per Fibonacci level)
        self.blocks = nn.ModuleList([
            WindingDHSRBlock(
                dim=layer_dims[i],
                fib_expand_factor=self.fib_hierarchy.fib_dims[i+1],
                use_prime_filter=True,
            )
            for i in range(num_blocks)
        ])

        # 4. Transitions between Fibonacci levels
        self.transitions = nn.ModuleList([
            nn.Linear(layer_dims[i], layer_dims[i+1])
            for i in range(num_blocks - 1)
        ])

        # 5. Output projection
        self.output_proj = nn.Linear(layer_dims[-1], output_dim)

        # 6. Mode norms per layer
        self.mode_norms = nn.ParameterList([
            nn.Parameter(
                torch.arange(dim).pow(2).float(),
                requires_grad=False
            )
            for dim in layer_dims
        ])

        # Metrics
        self.network_syntony = 0.0
        self.total_blocks_validated = 0
        self.blocks_rejected = 0

    def forward(
        self,
        winding_states: List[WindingState]
    ) -> Tensor:
        """Forward pass through winding network."""
        # Embed windings
        x = self.winding_embed.batch_forward(winding_states)

        # Initial syntony
        syntony = 0.5
        syntonies = []

        # Pass through DHSR blocks
        for i, block in enumerate(self.blocks):
            x, syntony_new, accepted = block(
                x,
                self.mode_norms[i],
                syntony,
            )

            syntonies.append(syntony_new)
            syntony = syntony_new

            # Track consensus
            if accepted:
                self.total_blocks_validated += 1
            else:
                self.blocks_rejected += 1

            # Transition to next level
            if i < len(self.transitions):
                x = F.relu(self.transitions[i](x))

        # Network syntony = average
        self.network_syntony = sum(syntonies) / len(syntonies)

        # Output
        return self.output_proj(x)

    def compute_loss(
        self,
        y_pred: Tensor,
        y_true: Tensor,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """Loss = L_task + q × (1 - S_network)."""
        task_loss = F.cross_entropy(y_pred, y_true)
        syntony_loss = 1.0 - self.network_syntony
        total_loss = task_loss + Q_DEFICIT * syntony_loss

        return total_loss, task_loss, torch.tensor(syntony_loss)

    def get_blockchain_stats(self) -> dict:
        """Return blockchain and consensus statistics."""
        total = self.total_blocks_validated + self.blocks_rejected
        return {
            'total_cycles': total,
            'validated_blocks': self.total_blocks_validated,
            'rejected_blocks': self.blocks_rejected,
            'validation_rate': self.total_blocks_validated / max(total, 1),
            'network_syntony': self.network_syntony,
            'blockchain_length': sum(
                b.get_blockchain_length() for b in self.blocks
            ),
        }
```

**File:** `python/syntonic/nn/winding/winding_net.py`

---

## 3. Implementation Phases

### Phase 1: Core Components (P0 - Critical)

**Goal:** Implement minimal viable WindingNet

**Tasks:**
1. ✅ **WindingEmbedding** - Map windings to embeddings
   - File: `python/syntonic/nn/winding/embedding.py`
   - Dependencies: `enumerate_windings`, `WindingState`
   - Tests: `tests/test_nn/test_winding/test_embedding.py`

2. ✅ **PrimeSelectionLayer** - Möbius filtering
   - File: `python/syntonic/nn/winding/prime_selection.py`
   - Dependencies: None (self-contained)
   - Tests: `tests/test_nn/test_winding/test_prime_selection.py`

3. ✅ **WindingSyntonyComputer** - Winding-aware syntony
   - File: `python/syntonic/nn/winding/syntony.py`
   - Dependencies: `PHI` constant
   - Tests: `tests/test_nn/test_winding/test_syntony.py`

4. ✅ **WindingDHSRBlock** - DHSR cycle with blockchain
   - File: `python/syntonic/nn/winding/dhsr_block.py`
   - Dependencies: PrimeSelectionLayer, WindingSyntonyComputer
   - Tests: `tests/test_nn/test_winding/test_dhsr_block.py`

**Deliverable:** Core components pass unit tests

---

### Phase 2: Architecture Integration (P1 - High)

**Goal:** Complete WindingNet architecture

**Tasks:**
1. ✅ **FibonacciHierarchy** - Depth management
   - File: `python/syntonic/nn/winding/fibonacci_hierarchy.py`
   - Dependencies: `fibonacci()` function
   - Tests: `tests/test_nn/test_winding/test_fibonacci.py`

2. ✅ **WindingNet** - Complete network
   - File: `python/syntonic/nn/winding/winding_net.py`
   - Dependencies: All Phase 1 components + FibonacciHierarchy
   - Tests: `tests/test_nn/test_winding/test_winding_net.py`

3. ✅ **Module exports** - Package structure
   - File: `python/syntonic/nn/winding/__init__.py`
   - Exports: All public classes and functions

**Deliverable:** WindingNet forward pass works end-to-end

---

### Phase 3: Training and Benchmarking (P1 - High)

**Goal:** Train WindingNet and compare with baselines

**Tasks:**
1. ✅ **Training loop** - Integration with SyntonicTrainer
   - File: `python/syntonic/benchmarks/winding_benchmark.py`
   - Use existing `SyntonicTrainer` with WindingNet

2. ✅ **XOR benchmark** - Particle classification
   - Dataset: Map windings to particle types (lepton vs quark)
   - Comparison: WindingNet vs PyTorch MLP vs RES

3. ✅ **Winding recovery benchmark** - Geometric fidelity
   - Task: Predict winding state from noisy input
   - Metric: Exact recovery rate

**Deliverable:** Benchmark results showing WindingNet advantages

---

### Phase 4: Advanced Features (P2 - Medium)

**Goal:** Temporal blockchain and consciousness features

**Tasks:**
1. ✅ **Blockchain visualization** - Plot temporal evolution
   - Function: `plot_temporal_blockchain(model, block_idx)`
   - Shows: State trajectory, syntony evolution

2. ✅ **Consensus analysis** - ΔS > 24 statistics
   - Function: `analyze_consensus(model)`
   - Returns: Validation rates, rejection patterns

3. ✅ **Documentation** - Usage examples and tutorials
   - Jupyter notebook: `examples/winding_net_tutorial.ipynb`
   - API reference: `docs/WINDING_NET_API.md`

**Deliverable:** Complete WindingNet ecosystem

---

## 4. File Structure

```
python/syntonic/nn/winding/
├── __init__.py                 # Module exports
├── embedding.py                # WindingEmbedding
├── prime_selection.py          # PrimeSelectionLayer
├── fibonacci_hierarchy.py      # FibonacciHierarchy
├── syntony.py                  # WindingSyntonyComputer
├── dhsr_block.py               # WindingDHSRBlock
├── winding_net.py              # WindingNet
└── utils.py                    # Helper functions

python/syntonic/benchmarks/
├── winding_benchmark.py        # XOR and recovery benchmarks
└── winding_datasets.py         # Winding-based datasets

tests/test_nn/test_winding/
├── __init__.py
├── test_embedding.py
├── test_prime_selection.py
├── test_fibonacci.py
├── test_syntony.py
├── test_dhsr_block.py
├── test_winding_net.py
└── test_winding_benchmark.py

docs/
├── WINDING_NET_API.md          # API reference
└── WINDING_NET_IMPLEMENTATION_PLAN.md  # This document

examples/
└── winding_net_tutorial.ipynb  # Jupyter tutorial
```

---

## 5. Testing Strategy

### 5.1 Unit Tests

Each component has isolated tests:

**WindingEmbedding:**
- Test winding enumeration
- Test embedding creation
- Test forward pass with single/batch windings
- Test mode norm retrieval

**PrimeSelectionLayer:**
- Test Möbius computation (verify known values)
- Test prime mask generation
- Test filtering behavior (primes preserved, composites attenuated)

**WindingSyntonyComputer:**
- Test syntony formula (verify S ∈ [0, 1])
- Test with known mode_norms
- Test batch computation

**WindingDHSRBlock:**
- Test D-phase (expansion)
- Test prime filtering
- Test H-phase (damping)
- Test syntony computation
- Test blockchain recording
- Test consensus mechanism

**WindingNet:**
- Test initialization
- Test forward pass
- Test loss computation
- Test blockchain stats

### 5.2 Integration Tests

**End-to-end:**
- Train WindingNet on toy dataset (10 epochs)
- Verify syntony increases over time
- Verify blockchain grows
- Verify loss decreases

**Gradient flow:**
- Check gradients propagate through all layers
- No NaN or inf values
- Gradient norms reasonable

### 5.3 Benchmark Tests

**XOR Classification:**
- WindingNet accuracy ≥ 95% (should exceed PyTorch MLP)
- Syntony > 0.9 at convergence
- Validation rate > 50%

**Winding Recovery:**
- Exact recovery rate (||pred - target|| = 0)
- Robustness to noise

---

## 6. Benchmarking Plan

### 6.1 XOR Classification (Particle Type)

**Dataset:**
```python
train_data = [
    (ELECTRON_WINDING, 0),  # Leptons
    (MUON_WINDING, 0),
    (TAU_WINDING, 0),
    (UP_WINDING, 1),        # Quarks
    (DOWN_WINDING, 1),
    (CHARM_WINDING, 1),
    (STRANGE_WINDING, 1),
    (TOP_WINDING, 1),
    (BOTTOM_WINDING, 1),
]
```

**Models:**
- WindingNet (max_winding=5, base_dim=64, num_blocks=3)
- PyTorch MLP (input_dim=4, hidden_dim=64, output_dim=2)
- RES (population_size=64, linear classifier)

**Metrics:**
- Classification accuracy
- Syntony evolution
- Blockchain validation rate
- Training time

**Expected Results:**
```
Model          Accuracy  Syntony  Validation Rate  Time
WindingNet     98%       0.94     65%              2.5s
PyTorch MLP    96%       N/A      N/A              0.8s
RES            93%       0.91     N/A              1.4s
```

### 6.2 Winding Recovery (Geometric Fidelity)

**Task:**
- Input: Noisy winding state (Gaussian noise on embedding)
- Output: Predicted winding state
- Metric: Exact recovery (L2 distance = 0)

**Dataset:**
- 1000 random winding states
- Noise levels: [0.0, 0.1, 0.2, 0.3]

**Expected Results:**
- WindingNet should achieve higher exact recovery at all noise levels
- Lattice structure should provide robustness

---

## 7. API Reference

### 7.1 WindingEmbedding

```python
from syntonic.nn.winding import WindingEmbedding

embed = WindingEmbedding(max_n=5, embed_dim=64)

# Single winding
from syntonic.physics.fermions.windings import ELECTRON_WINDING
x = embed(ELECTRON_WINDING)  # Tensor of shape (64,)

# Batch
windings = [ELECTRON_WINDING, MUON_WINDING, TAU_WINDING]
X = embed.batch_forward(windings)  # Tensor of shape (3, 64)

# Mode norm
mode_norm = embed.get_mode_norm(ELECTRON_WINDING)  # 0.0 for electron
```

### 7.2 PrimeSelectionLayer

```python
from syntonic.nn.winding import PrimeSelectionLayer

prime_layer = PrimeSelectionLayer(dim=64)
x_filtered = prime_layer(x)  # Prime indices preserved
```

### 7.3 WindingNet

```python
from syntonic.nn.winding import WindingNet
from syntonic.physics.fermions.windings import *

model = WindingNet(
    max_winding=5,
    base_dim=64,
    num_blocks=3,
    output_dim=2,
)

# Forward pass
windings = [ELECTRON_WINDING, UP_WINDING]
y_pred = model(windings)  # Tensor of shape (2, 2)

# Loss
labels = torch.tensor([0, 1])  # Lepton, Quark
total_loss, task_loss, syntony_loss = model.compute_loss(y_pred, labels)

# Blockchain stats
stats = model.get_blockchain_stats()
print(stats['network_syntony'])  # Current syntony
print(stats['validation_rate'])  # Consensus success rate
```

### 7.4 Training Loop

```python
from syntonic.nn.winding import WindingNet
from syntonic.nn.training import SyntonicTrainer
from syntonic.nn.optim import SyntonicAdam

model = WindingNet(...)
optimizer = SyntonicAdam(model.parameters(), lr=0.001)

# Use existing SyntonicTrainer
trainer = SyntonicTrainer(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    optimizer=optimizer,
    device='cuda',
)

history = trainer.train(epochs=100)
print(f"Final syntony: {trainer.current_syntony:.4f}")
```

---

## 8. Success Criteria

### Phase 1 Success:
- [ ] All unit tests pass
- [ ] Components work in isolation
- [ ] No import errors

### Phase 2 Success:
- [ ] WindingNet forward pass completes
- [ ] Gradients flow correctly
- [ ] Loss decreases on toy dataset

### Phase 3 Success:
- [ ] XOR benchmark: WindingNet accuracy ≥ 95%
- [ ] Syntony reaches > 0.90
- [ ] Validation rate > 50%
- [ ] Winding recovery: Higher exact recovery than MLP

### Phase 4 Success:
- [ ] Blockchain visualization works
- [ ] Consensus analysis provides insights
- [ ] Documentation complete
- [ ] Tutorial runs without errors

---

## 9. Risk Mitigation

| Risk | Impact | Mitigation |
|------|--------|------------|
| Winding space too large (memory) | High | Start with max_n=3, expand gradually |
| Prime filtering too aggressive | Medium | Make optional, ablation study |
| Blockchain memory growth | Medium | Implement circular buffer with max length |
| Fibonacci layers too deep | Low | Limit num_blocks=3 initially |
| ΔS threshold never met | Medium | Make threshold adaptive or configurable |

---

## 10. Next Steps

1. **Phase 1:** Implement core components (1-2 days)
2. **Phase 2:** Build WindingNet architecture (1 day)
3. **Phase 3:** Run benchmarks (1 day)
4. **Phase 4:** Polish and document (1 day)

**Total estimated time:** 4-5 days

---

## Appendix A: Dependencies

**Required packages:**
- PyTorch ≥ 2.0.0
- NumPy ≥ 1.20.0
- syntonic (existing infrastructure)

**Optional:**
- matplotlib (for visualization)
- jupyter (for tutorial)

---

## Appendix B: References

- **Specification:** `docs/winding_nn.md`
- **DHSR Theory:** `docs/RESONANT_ENGINE_TECHNICAL.md`
- **Benchmark Results:** `docs/RESONANT_ENGINE_RESULTS.md`
- **Existing NN Infrastructure:** `python/syntonic/nn/`
