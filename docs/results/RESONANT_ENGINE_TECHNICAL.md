# Resonant Engine Technical Specification
## Version 2.0 — Corrected

---

## Overview

The Resonant Engine instantiates Syntony Recursion Theory (SRT) as hardware-native computation. It operates on a **dual-state architecture**:

- **Lattice (CPU):** Exact values in the golden field Q(φ) — eternal, governing
- **Flux (GPU):** Approximate f64 values — ephemeral, exploratory

The system enforces **φ-resonance**: the ratio of H-phase to D-phase duration approaches the golden ratio.

**This is not an optimizer that uses golden lattice. It is the theory instantiated in silicon.**

---

# Part I: Mathematical Foundation

## 1.1 The Golden Field Q(φ)

All exact values live in the **golden field**:

```
Q(φ) = { a + b·φ : a, b ∈ ℚ }
```

Where:
- **φ = (1 + √5) / 2 = 1.6180339887498949...** (golden ratio)
- **a, b** are rational numbers (stored as integer pairs)

This is a degree-2 algebraic extension of the rationals, closed under all arithmetic operations.

## 1.2 Fundamental Identities

| Identity | Formula | Use |
|----------|---------|-----|
| Self-similarity | φ² = φ + 1 | Multiplication rule |
| Inverse | 1/φ = φ - 1 = 0.618... | H-phase coefficient |
| Inverse squared | 1/φ² = 2 - φ = 0.382... | D-phase coefficient |
| Conjugate | φ̂ = (1 - √5)/2 = -0.618... | Norm computation |
| Norm | N(a + bφ) = a² + ab - b² | Algebraic norm |

## 1.3 Exact Representation

```rust
struct GoldenExact {
    a: Rational,  // coefficient of 1
    b: Rational,  // coefficient of φ
}

// Example representations:
// 3.0       → GoldenExact { a: 3/1, b: 0/1 }
// φ         → GoldenExact { a: 0/1, b: 1/1 }
// φ² = 1+φ  → GoldenExact { a: 1/1, b: 1/1 }
// 1/φ = φ-1 → GoldenExact { a: -1/1, b: 1/1 }
```

## 1.4 Arithmetic in Q(φ)

**Addition:**
```
(a₁ + b₁φ) + (a₂ + b₂φ) = (a₁ + a₂) + (b₁ + b₂)φ
```

**Multiplication** (using φ² = φ + 1):
```
(a₁ + b₁φ)(a₂ + b₂φ) = (a₁a₂ + b₁b₂) + (a₁b₂ + a₂b₁ + b₁b₂)φ
```

**Division:**
```
1/(a + bφ) = (a + b - bφ) / N(a + bφ)
where N(a + bφ) = a² + ab - b²
```

---

# Part II: Universal Constants

## 2.1 Golden Ratio Family

| Symbol | Value | Derivation |
|--------|-------|------------|
| φ | 1.6180339887498949 | (1 + √5) / 2 |
| φ⁻¹ | 0.6180339887498949 | φ - 1 |
| φ⁻² | 0.3819660112501051 | 2 - φ |
| φ̂ | -0.6180339887498949 | (1 - √5) / 2 |

## 2.2 SRT Constants

| Symbol | Value | Meaning | Source |
|--------|-------|---------|--------|
| q | 0.027395146920 | Universal syntony deficit | Möbius-regularized geometry |
| E* | 19.999099979... | e^π - π | Spectral constant |
| K(D₄) | 24 | Consciousness threshold | D₄ kissing number |

**q IS NOT A HYPERPARAMETER.** It is derived from:

```
q = (2φ + e/(2φ²)) / (φ⁴ · (e^π - π))
```

This unifies {φ, π, e, 1, E*} in a single geometric expression.

---

# Part III: The DHSR Operators

## 3.1 Differentiation Operator D̂

**D̂ spreads energy to higher modes, generating novelty.**

**Formula:**
```
D̂[ψ]ₙ = ψₙ × (1 + α(S) × √|n|²)
```

**Coefficient:**
```
α(S) = φ⁻² × (1 - S) ≈ 0.382 × (1 - S)
```

**Properties:**
- High syntony (S → 1): α → 0, minimal differentiation
- Low syntony (S → 0): α → 0.382, maximum differentiation
- Always amplifies high-|n|² modes relative to low

**CUDA Implementation** (`rust/kernels/dhsr.cu`):
```cuda
extern "C" __global__ void differentiation_c128(
    double *out,
    const double *in,
    const double *mode_norm_sq,
    double syntony,
    int n
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    int idx = i * 2;  // Interleaved complex: [re₀, im₀, re₁, im₁, ...]
    double norm_sq = mode_norm_sq[i];

    double alpha = PHI_INV_SQ_F64 * (1.0 - syntony);
    double scale = 1.0 + alpha * sqrt(norm_sq);

    out[idx]     = in[idx]     * scale;  // Real part
    out[idx + 1] = in[idx + 1] * scale;  // Imaginary part
}
```

## 3.2 Harmonization Operator Ĥ

**Ĥ concentrates energy toward golden measure, enforcing coherence.**

**Formula:**
```
Ĥ[ψ]ₙ = ψₙ × (1 - β(S) × (1 - w(n)))
```

**Coefficient:**
```
β(S) = φ⁻¹ × S ≈ 0.618 × S
```

**Golden weight:**
```
w(n) = exp(-|n|²/φ)
```

**Properties:**
- High syntony (S → 1): β → 0.618, strong harmonization
- Low syntony (S → 0): β → 0, minimal harmonization
- Always attenuates modes with low golden weight (high |n|²)

**Implementation Note:** In the Resonant Engine, Ĥ is implemented via **crystallization** — the snap to Q(φ) lattice *plus* the attenuation of non-golden modes.

```rust
fn harmonize_and_crystallize(
    flux: &[f64],
    mode_norm_sq: &[f64],
    syntony: f64,
    precision: i64,
) -> Vec<GoldenExact> {
    let beta = PHI_INV * syntony;

    flux.iter()
        .zip(mode_norm_sq.iter())
        .map(|(&val, &norm_sq)| {
            // Apply Ĥ attenuation
            let golden_weight = (-norm_sq / PHI).exp();
            let h_scale = 1.0 - beta * (1.0 - golden_weight);
            let harmonized = val * h_scale;

            // Snap to lattice
            GoldenExact::find_nearest(harmonized, precision)
        })
        .collect()
}
```

## 3.3 Recursion Operator R̂

**R̂ = Ĥ ∘ D̂ — the complete cycle.**

In the Resonant Engine, this becomes:

```
R̂ = crystallize ∘ differentiate ∘ wake_flux
```

Each cycle:
1. Projects lattice to flux (wake)
2. Applies D̂ on GPU (differentiate)
3. Applies Ĥ and snaps back to lattice (crystallize)
4. Destroys flux (enforce ephemerality)

## 3.4 Syntony S(Ψ)

**Syntony measures coherence with the golden structure.**

**Formula:**
```
S(Ψ) = Σₙ |ψₙ|² × w(n) / Σₙ |ψₙ|²
     = Σₙ |ψₙ|² × exp(-|n|²/φ) / Σₙ |ψₙ|²
```

**Range:** S ∈ [0, 1]
- S = 1: All weight on |n|² = 0 (vacuum)
- S = 0: All weight on |n|² → ∞

**Interpretation:** S measures how much the state aligns with the golden measure. High-syntony states are stable; low-syntony states are chaotic.

---

# Part IV: Mode Norm Structure

## 4.1 The Problem

For a tensor of n values, what is |n|² for each index?

This depends on the **physical interpretation** of the tensor.

## 4.2 Option A: Winding Space (Correct for SRT)

If the tensor represents a WindingField on T⁴:

```
|n|² = n₇² + n₈² + n₉² + n₁₀²
```

For a linearized winding lattice with max_n = N:
```rust
fn winding_mode_norms(max_n: i64) -> Vec<f64> {
    let mut norms = Vec::new();
    for n7 in -max_n..=max_n {
        for n8 in -max_n..=max_n {
            for n9 in -max_n..=max_n {
                for n10 in -max_n..=max_n {
                    let norm_sq = (n7*n7 + n8*n8 + n9*n9 + n10*n10) as f64;
                    norms.push(norm_sq);
                }
            }
        }
    }
    norms
}
```

## 4.3 Option B: Fourier Modes (Approximation)

For a 1D signal interpreted as Fourier coefficients:

```
mode_norm_sq[i] = i²
```

This treats index i as frequency mode i. **This is an approximation** for systems not directly on T⁴.

## 4.4 Option C: Custom Structure

For neural network weights, define problem-specific structure:

```rust
// Example: 2-layer MLP with weight matrices W1 (input × hidden), W2 (hidden × output)
fn mlp_mode_norms(input_dim: usize, hidden_dim: usize, output_dim: usize) -> Vec<f64> {
    let mut norms = Vec::new();

    // W1: input connections have low mode (more fundamental)
    for i in 0..input_dim {
        for h in 0..hidden_dim {
            norms.push((i + h) as f64);  // Sum of indices
        }
    }

    // W2: output connections have higher mode
    for h in 0..hidden_dim {
        for o in 0..output_dim {
            norms.push((input_dim + h + o) as f64);
        }
    }

    norms
}
```

## 4.5 Requirement

Every ResonantTensor **must** have an explicit mode_norm_sq interpretation. Using `i²` without justification is a placeholder, not a solution.

---

# Part V: ResonantTensor Structure

## 5.1 Definition

```rust
#[pyclass]
pub struct ResonantTensor {
    // === THE ESSENCE (eternal) ===
    /// Exact values in Q(φ)
    lattice: Vec<GoldenExact>,

    // === THE SHADOW (ephemeral) ===
    /// GPU flux — None when crystallized
    #[cfg(feature = "cuda")]
    flux: Option<Arc<CudaSlice<f64>>>,

    // === METADATA ===
    /// Tensor shape
    shape: Vec<usize>,

    /// Mode norm squared for each element
    mode_norm_sq: Vec<f64>,

    /// Cached syntony (recomputed after crystallize)
    syntony: f64,

    /// Current phase
    phase: ResonantPhase,

    /// Lattice precision (max coefficient magnitude)
    precision: i64,

    // === TIMING ===
    /// Last D-phase duration (nanoseconds)
    last_d_duration_ns: u64,

    /// CUDA device reference
    #[cfg(feature = "cuda")]
    device: Option<Arc<CudaDevice>>,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum ResonantPhase {
    /// Lattice only, no flux allocated
    Crystalline,
    /// Flux allocated, D-phase active
    Fluxed,
}
```

## 5.2 Complex Support

For quantum states, use interleaved complex representation:

```rust
pub struct ResonantTensorComplex {
    /// Real parts in Q(φ)
    lattice_re: Vec<GoldenExact>,
    /// Imaginary parts in Q(φ)
    lattice_im: Vec<GoldenExact>,

    /// GPU flux: interleaved [re₀, im₀, re₁, im₁, ...]
    #[cfg(feature = "cuda")]
    flux: Option<Arc<CudaSlice<f64>>>,

    // ... same metadata
}
```

**Note:** The CUDA kernels (`differentiation_c128`, `harmonization_c128`, `dhsr_cycle_c128`) already support interleaved complex format.

## 5.3 Invariants

1. **Lattice always valid:** Even during Fluxed phase, lattice contains last crystallized state
2. **Flux only during D-phase:** `flux = Some(...)` only between wake and crystallize
3. **Syntony always current:** Recomputed after every crystallize
4. **Mode norms immutable:** Set at construction, never changed

---

# Part VI: The Resonant Cycle

## 6.1 Cycle Structure

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         RESONANT CYCLE                                  │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│   CRYSTALLINE                                 FLUXED                    │
│   ┌─────────────────┐                         ┌─────────────────┐       │
│   │ lattice: Q(φ)   │────── wake_flux() ─────▶│ flux: f64 (GPU) │       │
│   │ flux: None      │                         │ lattice: Q(φ)   │       │
│   └────────▲────────┘                         └────────┬────────┘       │
│            │                                           │                │
│            │                                  differentiate()           │
│            │                                  D̂ kernel on GPU           │
│            │                                  Duration: t_D             │
│            │                                           │                │
│            │                                           ▼                │
│   crystallize() ◀──────────────────────────────────────┤                │
│   • Download flux from GPU                             │                │
│   • Apply Ĥ attenuation                                │                │
│   • Snap to Q(φ) lattice                               │                │
│   • Enforce φ-dwell: t_H ≥ φ × t_D                     │                │
│   • Recompute syntony S(Ψ)                             │                │
│            │                                           │                │
│            ▼                                           │                │
│   destroy_shadow()                                     │                │
│   flux = None ─────────────────────────────────────────┘                │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

## 6.2 Implementation

```rust
impl ResonantTensor {
    /// Execute one complete resonant cycle
    pub fn cycle(&mut self, noise_scale: Option<f64>) -> CycleResult {
        // === D-PHASE (GPU) ===
        let d_start = Instant::now();

        self.wake_flux()?;
        self.differentiate(noise_scale)?;

        let d_duration = d_start.elapsed();
        self.last_d_duration_ns = d_duration.as_nanos() as u64;

        // === H-PHASE (CPU) with φ-dwell ===
        let target_h_ns = (d_duration.as_nanos() as f64 * PHI) as u64;
        let target_h_duration = Duration::from_nanos(target_h_ns);

        let h_start = Instant::now();

        let syntony = self.crystallize_with_dwell(target_h_duration)?;

        let h_duration = h_start.elapsed();

        // === DESTROY SHADOW ===
        self.destroy_shadow();

        CycleResult {
            syntony,
            d_duration,
            h_duration,
            target_ratio: PHI,
            actual_ratio: h_duration.as_nanos() as f64 / d_duration.as_nanos() as f64,
        }
    }
}
```

## 6.3 Wake Flux

```rust
impl ResonantTensor {
    /// Project lattice to GPU flux
    #[cfg(feature = "cuda")]
    pub fn wake_flux(&mut self) -> Result<(), ResonantError> {
        if self.phase != ResonantPhase::Crystalline {
            return Err(ResonantError::InvalidPhase("wake requires Crystalline"));
        }

        let device = self.device.as_ref()
            .ok_or(ResonantError::NoDevice)?;

        // Project exact → f64
        let floats: Vec<f64> = self.lattice
            .iter()
            .map(|g| g.to_f64())
            .collect();

        // Upload to GPU
        let gpu_slice = device.htod_sync_copy(&floats)?;
        self.flux = Some(Arc::new(gpu_slice));
        self.phase = ResonantPhase::Fluxed;

        Ok(())
    }
}
```

## 6.4 Differentiate

```rust
impl ResonantTensor {
    /// Apply D̂ on GPU with optional stochastic noise
    #[cfg(feature = "cuda")]
    pub fn differentiate(&mut self, noise_scale: Option<f64>) -> Result<(), ResonantError> {
        if self.phase != ResonantPhase::Fluxed {
            return Err(ResonantError::InvalidPhase("differentiate requires Fluxed"));
        }

        let flux = self.flux.as_ref().ok_or(ResonantError::NoFlux)?;
        let device = self.device.as_ref().ok_or(ResonantError::NoDevice)?;
        let n = self.lattice.len();

        // Upload mode norms
        let mode_norms = device.htod_sync_copy(&self.mode_norm_sq)?;

        // Allocate output
        let mut output: CudaSlice<f64> = device.alloc_zeros(n)?;

        // Launch D̂ kernel
        let cfg = launch_cfg_256(n);
        let func = device.get_func("dhsr", "differentiation_f64")?;

        unsafe {
            func.launch(cfg, (
                &mut output,
                flux.as_ref(),
                &mode_norms,
                self.syntony,
                n as i32,
            ))?;
        }

        // Optional: apply stochastic noise for exploration
        if let Some(scale) = noise_scale {
            self.apply_noise(&mut output, scale)?;
        }

        self.flux = Some(Arc::new(output));
        Ok(())
    }

    /// Apply stochastic perturbation (separate from D̂)
    #[cfg(feature = "cuda")]
    fn apply_noise(&self, flux: &mut CudaSlice<f64>, scale: f64) -> Result<(), ResonantError> {
        let n = flux.len();
        let device = self.device.as_ref().ok_or(ResonantError::NoDevice)?;

        let noise_scale = scale * (1.0 - self.syntony);  // Less noise when syntony high
        let seed = rand::random::<u64>();

        let func = device.get_func("resonant", "apply_noise_f64")?;
        let cfg = launch_cfg_256(n);

        unsafe {
            func.launch(cfg, (flux, noise_scale, seed, n as i32))?;
        }

        Ok(())
    }
}
```

## 6.5 Crystallize with φ-Dwell

```rust
impl ResonantTensor {
    /// Apply Ĥ, snap to lattice, enforce φ-dwell timing
    #[cfg(feature = "cuda")]
    pub fn crystallize_with_dwell(
        &mut self,
        target_duration: Duration,
    ) -> Result<f64, ResonantError> {
        if self.phase != ResonantPhase::Fluxed {
            return Err(ResonantError::InvalidPhase("crystallize requires Fluxed"));
        }

        let flux = self.flux.as_ref().ok_or(ResonantError::NoFlux)?;
        let device = self.device.as_ref().ok_or(ResonantError::NoDevice)?;
        let start = Instant::now();

        // Download from GPU
        let mut host_data = vec![0.0f64; flux.len()];
        device.dtoh_sync_copy_into(flux, &mut host_data)?;

        // Apply Ĥ attenuation + snap to lattice
        let beta = PHI_INV * self.syntony;

        self.lattice = host_data
            .par_iter()  // Parallel via rayon
            .zip(self.mode_norm_sq.par_iter())
            .map(|(&val, &norm_sq)| {
                // Ĥ attenuation
                let golden_weight = (-norm_sq / PHI).exp();
                let h_scale = 1.0 - beta * (1.0 - golden_weight);
                let harmonized = val * h_scale;

                // Snap to Q(φ)
                GoldenExact::find_nearest(harmonized, self.precision)
            })
            .collect();

        // φ-DWELL ENFORCEMENT
        // If we finished early, deepen the lattice precision
        let mut current_precision = self.precision;
        while start.elapsed() < target_duration && current_precision < 1000 {
            current_precision += 10;

            // Re-snap with higher precision (uses remaining time productively)
            let floats: Vec<f64> = self.lattice.iter().map(|g| g.to_f64()).collect();
            self.lattice = floats
                .par_iter()
                .map(|&x| GoldenExact::find_nearest(x, current_precision))
                .collect();
        }

        // Recompute syntony
        self.syntony = self.compute_lattice_syntony();
        self.phase = ResonantPhase::Crystalline;

        Ok(self.syntony)
    }

    /// Compute syntony from lattice (no GPU)
    fn compute_lattice_syntony(&self) -> f64 {
        let mut numerator = 0.0;
        let mut denominator = 0.0;

        for (g, &norm_sq) in self.lattice.iter().zip(self.mode_norm_sq.iter()) {
            let val = g.to_f64();
            let amp_sq = val * val;
            let weight = (-norm_sq / PHI).exp();

            numerator += amp_sq * weight;
            denominator += amp_sq;
        }

        if denominator < 1e-15 { 0.0 } else { numerator / denominator }
    }
}
```

## 6.6 Destroy Shadow

```rust
impl ResonantTensor {
    /// Explicitly destroy flux — enforces ephemerality
    pub fn destroy_shadow(&mut self) {
        self.flux = None;
        // Phase should already be Crystalline after crystallize()
        // This is defensive
        self.phase = ResonantPhase::Crystalline;
    }
}
```

---

# Part VII: Lattice Snapping Algorithm

## 7.1 The Approximation Problem

Given a float x, find (a, b) ∈ ℤ² minimizing:

```
| a + b·φ - x |
```

Subject to |a|, |b| ≤ max_coeff.

## 7.2 Grid Search (max_coeff ≤ 100)

```rust
impl GoldenExact {
    pub fn find_nearest(x: f64, max_coeff: i64) -> Self {
        if max_coeff <= 100 {
            Self::find_nearest_grid(x, max_coeff)
        } else {
            Self::find_nearest_lll(x, max_coeff)
        }
    }

    fn find_nearest_grid(x: f64, max_coeff: i64) -> Self {
        let mut best_a = 0i64;
        let mut best_b = 0i64;
        let mut best_error = f64::INFINITY;

        for b in -max_coeff..=max_coeff {
            // Optimal a for this b
            let a_float = x - (b as f64) * PHI;
            let a = a_float.round() as i64;

            if a.abs() <= max_coeff {
                let error = (a as f64 + (b as f64) * PHI - x).abs();
                if error < best_error {
                    best_error = error;
                    best_a = a;
                    best_b = b;

                    // Early exit at machine precision
                    if error < 1e-14 {
                        break;
                    }
                }
            }
        }

        GoldenExact::new(
            Rational::from(best_a),
            Rational::from(best_b),
        )
    }
}
```

## 7.3 LLL Algorithm (max_coeff > 100)

For high precision, use lattice reduction:

```rust
fn find_nearest_lll(x: f64, max_coeff: i64) -> GoldenExact {
    const K: f64 = 1e12;  // Scaling factor

    // Basis: [1, 0, K], [0, 1, K·φ]
    // Finding short vectors finds good approximations

    let mut b1 = [1.0, 0.0, K];
    let mut b2 = [0.0, 1.0, K * PHI];

    // LLL reduction with δ = 0.75
    lll_reduce(&mut b1, &mut b2, 0.75);

    // Extract approximation from reduced basis
    // ... (full LLL implementation)

    GoldenExact::new(
        Rational::from(a),
        Rational::from(b),
    )
}
```

---

# Part VIII: Resonant Evolution Strategy (RES)

## 8.1 Key Insight

**Syntony is a universal fitness proxy.** High-syntony configurations are geometrically beautiful; geometric beauty correlates with functional fitness.

This allows **80% of candidates to die cheap** (CPU syntony check) before expensive GPU/fitness evaluation.

## 8.2 Configuration

```rust
pub struct RESConfig {
    /// Population size per generation
    pub population_size: usize,       // Default: 64

    /// Fraction surviving syntony filter
    pub survivor_fraction: f64,       // Default: 0.25 (16 survivors)

    /// Mutation magnitude in Q(φ)
    pub mutation_scale: f64,          // Default: 0.3

    /// Syntony weight in selection
    /// THIS IS q — NOT A HYPERPARAMETER
    pub lambda: f64,                  // 0.027395146920

    /// Lattice precision
    pub precision: i64,               // Default: 100

    /// D-phase noise scale
    pub noise_scale: f64,             // Default: 0.01
}
```

## 8.3 Algorithm

```
╔═══════════════════════════════════════════════════════════════════════════╗
║                        RES: ONE GENERATION                                ║
╠═══════════════════════════════════════════════════════════════════════════╣
║                                                                           ║
║  STEP 1: MUTATION (CPU, parallel)                                         ║
║  ─────────────────────────────────                                        ║
║  for i in 0..population_size:                                             ║
║      noise = randn(n) * mutation_scale                                    ║
║      mutant[i] = snap_to_lattice(best + noise, precision)                 ║
║      mutant[i].syntony = compute_lattice_syntony(mutant[i])               ║
║                                                                           ║
║  Cost: O(population × n × precision)                                      ║
║                                                                           ║
╠═══════════════════════════════════════════════════════════════════════════╣
║                                                                           ║
║  STEP 2: SYNTONY FILTER (CPU, cheap)                                      ║
║  ────────────────────────────────────                                     ║
║  sort mutants by syntony descending                                       ║
║  survivors = mutants[0 : survivor_count]                                  ║
║                                                                           ║
║  Reject: 75% of candidates                                                ║
║  Cost: O(population × log(population))                                    ║
║                                                                           ║
╠═══════════════════════════════════════════════════════════════════════════╣
║                                                                           ║
║  STEP 3: DHSR CYCLE (GPU + CPU)  ◀── THIS WAS MISSING                     ║
║  ──────────────────────────────────                                       ║
║  for s in survivors:                                                      ║
║      s.wake_flux()                // Project to GPU                       ║
║      s.differentiate(noise)       // D̂ on GPU                             ║
║      s.crystallize_with_dwell()   // Ĥ + snap + φ-dwell                   ║
║      s.destroy_shadow()           // Enforce ephemerality                 ║
║      s.flux_syntony = s.syntony   // Post-cycle syntony                   ║
║                                                                           ║
║  Cost: O(survivors × (GPU_kernel + crystallize))                          ║
║                                                                           ║
╠═══════════════════════════════════════════════════════════════════════════╣
║                                                                           ║
║  STEP 4: FITNESS EVALUATION (task-specific)                               ║
║  ──────────────────────────────────────────                               ║
║  for s in survivors:                                                      ║
║      s.fitness = fitness_fn(s)    // e.g., -cross_entropy                 ║
║                                                                           ║
║  Cost: Depends on task                                                    ║
║                                                                           ║
╠═══════════════════════════════════════════════════════════════════════════╣
║                                                                           ║
║  STEP 5: SELECTION                                                        ║
║  ─────────────────                                                        ║
║  for s in survivors:                                                      ║
║      s.score = s.fitness + λ × s.flux_syntony                             ║
║                where λ = q = 0.027395...                                  ║
║                                                                           ║
║  winner = argmax(score)                                                   ║
║                                                                           ║
╚═══════════════════════════════════════════════════════════════════════════╝
```

## 8.4 Implementation

```rust
impl ResonantEvolver {
    pub fn evolve_generation<F>(
        &mut self,
        best: &ResonantTensor,
        fitness_fn: F,
    ) -> Result<ResonantTensor, ResonantError>
    where
        F: Fn(&ResonantTensor) -> f64 + Sync,
    {
        // STEP 1: Mutation (parallel)
        let mutants: Vec<ResonantTensor> = (0..self.config.population_size)
            .into_par_iter()
            .map(|_| self.mutate(best))
            .collect();

        // STEP 2: Syntony filter
        let mut with_syntony: Vec<_> = mutants
            .into_iter()
            .map(|m| {
                let s = m.syntony;
                (m, s)
            })
            .collect();

        with_syntony.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

        let survivor_count = (self.config.population_size as f64
            * self.config.survivor_fraction) as usize;

        let mut survivors: Vec<_> = with_syntony
            .into_iter()
            .take(survivor_count)
            .map(|(m, _)| m)
            .collect();

        // STEP 3: DHSR cycle for each survivor
        for s in &mut survivors {
            s.cycle(Some(self.config.noise_scale))?;
        }

        // STEP 4: Fitness evaluation
        let fitness_scores: Vec<f64> = survivors
            .par_iter()
            .map(|s| fitness_fn(s))
            .collect();

        // STEP 5: Selection
        let winner_idx = survivors
            .iter()
            .zip(fitness_scores.iter())
            .enumerate()
            .map(|(i, (s, &f))| {
                let score = f + self.config.lambda * s.syntony;
                (i, score)
            })
            .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
            .map(|(i, _)| i)
            .unwrap();

        self.generation += 1;

        Ok(survivors.remove(winner_idx))
    }

    fn mutate(&self, parent: &ResonantTensor) -> ResonantTensor {
        let mut rng = rand::thread_rng();

        let mutated: Vec<GoldenExact> = parent.lattice
            .iter()
            .map(|g| {
                let noise = rng.gen_range(-0.5..0.5) * self.config.mutation_scale;
                let perturbed = g.to_f64() + noise;
                GoldenExact::find_nearest(perturbed, self.config.precision)
            })
            .collect();

        ResonantTensor::from_lattice(
            mutated,
            parent.shape.clone(),
            parent.mode_norm_sq.clone(),
            self.config.precision,
        )
    }
}
```

---

# Part IX: Benchmark Configuration

## 9.1 XOR Classification

**Dataset:**
```python
X, y = make_xor(n_samples=500, noise=0.1, seed=42)
# X: shape (500, 2), features in [-1, 1]
# y: shape (500,), labels {0, 1}
```

**Polynomial Features:**
```
[x₁, x₂] → [x₁, x₂, x₁·x₂, x₁², x₂²]
```

The x₁·x₂ term makes XOR linearly separable.

**Linear Classifier:**
```
logits = X_poly @ W    # (500, 5) @ (5, 2) → (500, 2)
loss = cross_entropy(logits, y)
```

**Weight matrix: 5 × 2 = 10 parameters**

## 9.2 Mode Norm Assignment

For 10 weights, assign mode norms based on feature structure:

```python
# Features: [x₁, x₂, x₁·x₂, x₁², x₂²]
# Interpretation:
#   - Linear terms (x₁, x₂): fundamental, low mode
#   - Interaction (x₁·x₂): medium mode
#   - Quadratic (x₁², x₂²): high mode

mode_norm_sq = [
    # Class 0 weights      Class 1 weights
    0, 0,                  # x₁ → class 0, 1
    0, 0,                  # x₂ → class 0, 1
    1, 1,                  # x₁·x₂ → class 0, 1
    4, 4,                  # x₁² → class 0, 1
    4, 4,                  # x₂² → class 0, 1
]
```

This assigns higher mode norm to higher-order polynomial features.

## 9.3 Full Configuration

```python
# Create tensor
initial_weights = np.random.randn(10) * 0.1
tensor = ResonantTensor.from_array(
    initial_weights,
    shape=[5, 2],
    mode_norm_sq=mode_norm_sq,
    precision=100,
)

# Create evolver
evolver = ResonantEvolver(
    config=RESConfig(
        population_size=64,
        survivor_fraction=0.25,
        mutation_scale=0.3,
        lambda_syntony=0.027395146920,  # q
        precision=100,
        noise_scale=0.01,
    ),
    device="cuda:0",
)

# Fitness function
def fitness(t: ResonantTensor) -> float:
    W = t.to_numpy().reshape(5, 2)
    logits = X_poly @ W
    return -cross_entropy(logits, y)

# Evolve
for gen in range(100):
    tensor = evolver.evolve_generation(tensor, fitness)
    print(f"Gen {gen}: S={tensor.syntony:.4f}, acc={accuracy(tensor):.2%}")
```

---

# Part X: File Locations

| Component | Path |
|-----------|------|
| GoldenExact | `rust/src/exact/golden.rs` |
| ResonantTensor | `rust/src/resonant/tensor.rs` |
| Crystallization | `rust/src/resonant/crystallize.rs` |
| φ-Dwell Enforcer | `rust/src/resonant/resonance.rs` |
| ResonantEvolver | `rust/src/resonant/evolver.rs` |
| D̂ CUDA Kernel | `rust/kernels/dhsr.cu` |
| Noise CUDA Kernel | `rust/kernels/resonant_d.cu` |
| SRT Constants | `rust/kernels/srt_constants.cuh` |
| Python Bindings | `python/syntonic/resonant/` |
| XOR Benchmark | `python/syntonic/benchmarks/convergence_benchmark.py` |

---

# Part XI: Timing Characteristics

| Operation | Location | Time (n=10⁶) | Notes |
|-----------|----------|--------------|-------|
| wake_flux | PCIe H→D | ~100ms | Memory bandwidth |
| differentiate | GPU | ~1ms | 3000+ CUDA cores |
| crystallize | CPU | ~600ms | Parallel via rayon |
| φ-dwell deepening | CPU | Variable | Fills to φ×t_D |
| syntony computation | CPU | ~10ms | Single pass |
| lattice snap | CPU | O(n × precision) | Grid search |

**Target ratio:** t_H / t_D → φ ≈ 1.618

---

# Appendix A: Fibonacci Connection

Powers of φ are Fibonacci combinations:

```
φⁿ = Fₙ·φ + Fₙ₋₁
```

| n | φⁿ | GoldenExact |
|---|-----|-------------|
| 0 | 1 | (1, 0) |
| 1 | φ | (0, 1) |
| 2 | φ+1 | (1, 1) |
| 3 | 2φ+1 | (1, 2) |
| 4 | 3φ+2 | (2, 3) |
| 5 | 5φ+3 | (3, 5) |
| 6 | 8φ+5 | (5, 8) |

The coefficients follow the Fibonacci sequence.

---

# Appendix B: Why Golden Ratio?

φ is the unique positive solution to x² = x + 1. This self-similarity makes it the natural eigenvalue for recursive systems.

In SRT, φ governs:

1. **Operator coefficients:**
   - D̂: α = φ⁻² × (1 - S)
   - Ĥ: β = φ⁻¹ × S

2. **Syntony measure:**
   - w(n) = exp(-|n|²/φ)

3. **Timing:**
   - H-phase = φ × D-phase

4. **Lattice structure:**
   - All values in Q(φ) = {a + bφ}

The hypothesis: systems with high syntony (alignment with φ-structure) exhibit better stability, generalization, and coherence.

---

# Appendix C: Connection to Winding Simulator

The ResonantTensor can represent a WindingField:

```rust
impl From<WindingField> for ResonantTensor {
    fn from(field: WindingField) -> Self {
        let (states, amplitudes) = field.to_dense();

        // Lattice: snap complex amplitudes to Q(φ)
        let lattice_re: Vec<GoldenExact> = amplitudes.iter()
            .map(|c| GoldenExact::find_nearest(c.re, 100))
            .collect();
        let lattice_im: Vec<GoldenExact> = amplitudes.iter()
            .map(|c| GoldenExact::find_nearest(c.im, 100))
            .collect();

        // Mode norms from winding: |n|² = n₇² + n₈² + n₉² + n₁₀²
        let mode_norm_sq: Vec<f64> = states.iter()
            .map(|s| s.norm_squared() as f64)
            .collect();

        ResonantTensor::from_complex_lattice(
            lattice_re,
            lattice_im,
            mode_norm_sq,
        )
    }
}
```

This connects the discrete winding dynamics to the continuous resonant evolution.

---

*End of Corrected Technical Specification v2.0*