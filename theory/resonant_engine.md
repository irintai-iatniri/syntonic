# Resonant Engine Implementation Plan
## Version 1.0 — January 2026

---

## Executive Summary

The Resonant Engine instantiates SRT/CRT as a hardware-native architecture where:
- **GPU = D̂ (Differentiation)**: Chaotic flux generator, approximate, parallel
- **CPU = Ĥ (Harmonization)**: Exact lattice enforcer, serial, governing
- **PCIe bus = Phase boundary**: Enforces φ-resonance dwell time

This document specifies the complete implementation for handoff to coding AI.

---

# Part I: Architecture

## 1.1 Core Principle

The "No Float Paradox" is resolved by treating floats as **ephemeral shadows** cast by the **eternal exact lattice**. Floats exist only during D-phase; they are destroyed each cycle by crystallization.

```
┌─────────────────────────────────────────────────────────────────┐
│                    RESONANT ENGINE CYCLE                        │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌─────────────┐         ┌─────────────┐                       │
│  │   LATTICE   │         │    FLUX     │                       │
│  │  (CPU/Exact)│         │ (GPU/Float) │                       │
│  │ GoldenExact │         │  CudaSlice  │                       │
│  └──────┬──────┘         └──────┬──────┘                       │
│         │                       │                               │
│         │   wake_flux()         │                               │
│         │   (project exact→f64) │                               │
│         ├──────────────────────►│                               │
│         │                       │                               │
│         │                       │  differentiate()              │
│         │                       │  (D̂ kernel, 0.382 dwell)     │
│         │                       │                               │
│         │   crystallize()       │                               │
│         │   (snap f64→exact)    │                               │
│         │◄──────────────────────┤                               │
│         │                       │                               │
│         │   destroy_shadow()    │                               │
│         │   (flux = None)       │                               │
│         │              ─────────┘                               │
│         │                                                       │
│  ┌──────▼──────┐                                               │
│  │  SYNTONY    │  S(Ψ) computed on crystallized lattice        │
│  │  CHECK      │  If S < threshold: flag for RES filtering     │
│  └─────────────┘                                               │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

## 1.2 φ-Resonance Dwell Enforcement

The golden ratio split is **not metaphorical**—it derives from the operator coefficients:
- D̂ coefficient: α = φ⁻² × (1 - S) ≈ 0.382 × (1 - S)
- Ĥ coefficient: β = φ⁻¹ × S ≈ 0.618 × S

**Wall-clock enforcement:**
```
target_h_duration = d_duration × φ   // φ ≈ 1.618
actual_h_duration = measure(crystallize())

if actual_h_duration < target_h_duration:
    deepen_lattice_search(target_h_duration - actual_h_duration)
```

The "spare time" is used productively to increase lattice precision, not wasted.

## 1.3 File Structure

```
syntonic/
├── rust/
│   └── src/
│       ├── resonant/                    # NEW MODULE
│       │   ├── mod.rs
│       │   ├── tensor.rs                # ResonantTensor struct
│       │   ├── crystallize.rs           # Golden snap algorithms
│       │   ├── resonance.rs             # φ-dwell enforcer
│       │   └── evolver.rs               # RES discrete learning
│       ├── exact/
│       │   └── golden.rs                # EXISTING (GoldenExact)
│       └── tensor/
│           └── storage.rs               # EXISTING (TensorStorage)
├── rust/kernels/
│   ├── dhsr.cu                          # EXISTING (modify for D-only)
│   ├── resonant_d.cu                    # NEW: D-phase only kernels
│   └── srt_constants.cuh                # EXISTING
└── python/
    └── syntonic/
        └── resonant/                    # NEW MODULE
            ├── __init__.py
            ├── tensor.py                # Python bindings
            ├── engine.py                # ResonantEngine class
            └── evolver.py               # RES Python interface
```

---

# Part II: ResonantTensor

## 2.1 Rust Struct Definition

**File:** `rust/src/resonant/tensor.rs`

```rust
use crate::exact::golden::GoldenExact;
use crate::tensor::storage::{TensorStorage, CudaData, DeviceType};
use std::sync::Arc;

#[cfg(feature = "cuda")]
use cudarc::driver::{CudaDevice, CudaSlice};

/// Dual-state tensor: exact lattice (CPU) + ephemeral flux (GPU)
#[pyclass]
pub struct ResonantTensor {
    /// THE ESSENCE: Exact truth in Q(φ)
    /// Shape: flattened 1D for now, reshape metadata separate
    lattice: Vec<GoldenExact>,
    
    /// THE SHADOW: Approximate flux (None when not in D-phase)
    #[cfg(feature = "cuda")]
    flux: Option<Arc<CudaSlice<f64>>>,
    
    /// Shape metadata
    shape: Vec<usize>,
    
    /// Current syntony (cached, recomputed after crystallize)
    syntony: f64,
    
    /// Mode norm squared |n|² for each element (precomputed)
    /// Used by D̂/Ĥ kernels
    mode_norm_sq: Vec<f64>,
    
    /// CUDA device reference
    #[cfg(feature = "cuda")]
    device: Option<Arc<CudaDevice>>,
    
    /// Phase tracking
    phase: ResonantPhase,
    
    /// Timing for resonance enforcement
    last_d_duration_ns: u64,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum ResonantPhase {
    /// Lattice state, no flux allocated
    Crystalline,
    /// Flux allocated, D-phase active or complete
    Fluxed,
    /// Error state (should never persist)
    Invalid,
}
```

## 2.2 Core Methods

### 2.2.1 Construction

```rust
impl ResonantTensor {
    /// Create from existing GoldenExact lattice
    pub fn from_lattice(
        lattice: Vec<GoldenExact>,
        shape: Vec<usize>,
        mode_norm_sq: Vec<f64>,
    ) -> Self {
        assert_eq!(lattice.len(), shape.iter().product::<usize>());
        assert_eq!(lattice.len(), mode_norm_sq.len());
        
        // Compute initial syntony from lattice
        let syntony = Self::compute_lattice_syntony(&lattice, &mode_norm_sq);
        
        ResonantTensor {
            lattice,
            flux: None,
            shape,
            syntony,
            mode_norm_sq,
            device: None,
            phase: ResonantPhase::Crystalline,
            last_d_duration_ns: 0,
        }
    }
    
    /// Create from f64 data by snapping to nearest golden lattice points
    pub fn from_floats(
        data: &[f64],
        shape: Vec<usize>,
        mode_norm_sq: Vec<f64>,
        precision: u32,  // Fibonacci index for snap precision
    ) -> Self {
        let lattice: Vec<GoldenExact> = data
            .iter()
            .map(|&x| GoldenExact::find_nearest(x, precision))
            .collect();
        
        Self::from_lattice(lattice, shape, mode_norm_sq)
    }
    
    /// Compute syntony directly from lattice (no GPU)
    fn compute_lattice_syntony(
        lattice: &[GoldenExact],
        mode_norm_sq: &[f64],
    ) -> f64 {
        let phi_inv = GoldenExact::phi_hat().to_f64();  // ≈ 0.618
        
        let mut numerator = 0.0;
        let mut denominator = 0.0;
        
        for (g, &norm_sq) in lattice.iter().zip(mode_norm_sq.iter()) {
            let val = g.to_f64();
            let amp_sq = val * val;  // Real-valued lattice
            let weight = (-norm_sq * phi_inv).exp();
            
            numerator += amp_sq * weight;
            denominator += amp_sq;
        }
        
        if denominator < 1e-15 {
            0.0
        } else {
            numerator / denominator
        }
    }
}
```

### 2.2.2 Phase Transitions

```rust
impl ResonantTensor {
    /// PROJECT: Lattice → Flux (exact → approximate)
    /// This is the "wake_flux" operation
    #[cfg(feature = "cuda")]
    pub fn wake_flux(&mut self, device: Arc<CudaDevice>) -> Result<(), ResonantError> {
        if self.phase != ResonantPhase::Crystalline {
            return Err(ResonantError::InvalidPhaseTransition(
                "wake_flux requires Crystalline phase"
            ));
        }
        
        // Project exact lattice to f64
        let floats: Vec<f64> = self.lattice
            .iter()
            .map(|g| g.to_f64())
            .collect();
        
        // Upload to GPU
        let gpu_slice = device.htod_sync_copy(&floats)
            .map_err(ResonantError::CudaError)?;
        
        self.flux = Some(Arc::new(gpu_slice));
        self.device = Some(device);
        self.phase = ResonantPhase::Fluxed;
        
        Ok(())
    }
    
    /// CRYSTALLIZE: Flux → Lattice (approximate → exact)
    /// This is the core H-phase operation on CPU
    #[cfg(feature = "cuda")]
    pub fn crystallize(&mut self, precision: u32) -> Result<f64, ResonantError> {
        if self.phase != ResonantPhase::Fluxed {
            return Err(ResonantError::InvalidPhaseTransition(
                "crystallize requires Fluxed phase"
            ));
        }
        
        let flux = self.flux.as_ref()
            .ok_or(ResonantError::NoFluxPresent)?;
        let device = self.device.as_ref()
            .ok_or(ResonantError::NoDevicePresent)?;
        
        // Download from GPU
        let mut host_data = vec![0.0f64; flux.len()];
        device.dtoh_sync_copy_into(flux, &mut host_data)
            .map_err(ResonantError::CudaError)?;
        
        // GOLDEN SNAP: Find nearest exact lattice point for each element
        // This is computationally heavy — satisfies 0.618 dwell naturally
        self.lattice = host_data
            .par_iter()  // rayon parallel
            .map(|&x| GoldenExact::find_nearest(x, precision))
            .collect();
        
        // Recompute syntony on crystallized lattice
        self.syntony = Self::compute_lattice_syntony(&self.lattice, &self.mode_norm_sq);
        
        // Destroy the shadow
        self.flux = None;
        self.phase = ResonantPhase::Crystalline;
        
        Ok(self.syntony)
    }
    
    /// Explicitly destroy flux without crystallizing (discard D-phase work)
    pub fn destroy_shadow(&mut self) {
        self.flux = None;
        if self.phase == ResonantPhase::Fluxed {
            self.phase = ResonantPhase::Crystalline;
        }
    }
}
```

### 2.2.3 Golden Snap Algorithm

**File:** `rust/src/resonant/crystallize.rs`

The `find_nearest` function must be implemented on `GoldenExact`. This is a Diophantine approximation problem.

```rust
impl GoldenExact {
    /// Find the nearest element of Q(φ) to a given float
    /// Uses continued fraction expansion for optimal approximation
    /// 
    /// precision: Fibonacci index controlling denominator bound
    ///            F_precision is the maximum denominator allowed
    pub fn find_nearest(x: f64, precision: u32) -> Self {
        // The golden ratio has the simplest continued fraction: [1; 1, 1, 1, ...]
        // Elements of Q(φ) = {a + b·φ : a, b ∈ Q} are dense in R
        // We find the best approximation with bounded denominator
        
        let phi = Self::phi().to_f64();
        let max_denom = fibonacci(precision) as i128;
        
        // Strategy: decompose x = a_approx + b_approx·φ
        // where a_approx, b_approx are rationals with bounded denominator
        
        // First, find b by: b ≈ (x - round(x)) / (φ - 1)
        // Then: a ≈ x - b·φ
        
        let mut best = GoldenExact::zero();
        let mut best_error = f64::INFINITY;
        
        // Search over rational b with denominator up to max_denom
        for b_denom in 1..=max_denom.min(1000) {
            for b_num in -(b_denom * 3)..=(b_denom * 3) {
                let b = Rational::new(b_num, b_denom);
                let b_f64 = b.to_f64();
                
                // Given b, optimal a is (x - b·φ) rounded to nearest rational
                let a_target = x - b_f64 * phi;
                let a = Rational::from_f64_approx(a_target, max_denom);
                
                let candidate = GoldenExact::new(a, b);
                let error = (candidate.to_f64() - x).abs();
                
                if error < best_error {
                    best_error = error;
                    best = candidate;
                    
                    // Early exit if we're within machine epsilon
                    if error < 1e-14 {
                        return best;
                    }
                }
            }
        }
        
        best
    }
}
```

**Note:** The above is a naive implementation. Production version should use:
1. Simultaneous Diophantine approximation (LLL algorithm)
2. Cached Fibonacci bounds
3. Early termination heuristics

---

# Part III: CUDA Kernels

## 3.1 D-Phase Only Kernel

**File:** `rust/kernels/resonant_d.cu`

The existing `dhsr.cu` fuses D̂ and Ĥ. For the Resonant Engine, we need D̂ alone since Ĥ is performed on CPU via crystallization.

```cuda
#include "srt_constants.cuh"

// =============================================================================
// RESONANT D-PHASE: Differentiation Only
// =============================================================================

/// Stochastic Differentiation for Resonant Engine
/// D̂(ψ)[n] = ψ[n] × (1 + α(S) × √|n|² + noise)
/// 
/// The noise term enables exploration in the flux domain.
/// Noise scale decreases with syntony (high S = less exploration needed)
extern "C" __global__ void resonant_differentiate_f64(
    double *out,                    // Output flux
    const double *in,               // Input flux (projected from lattice)
    const double *mode_norm_sq,     // |n|² for each mode
    double syntony,                 // Current S(Ψ)
    double noise_scale,             // Base noise magnitude
    unsigned long long seed,        // RNG seed
    int n
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    
    double norm_sq = mode_norm_sq[i];
    double S = syntony;
    
    // Diffusion coefficient α(S) = φ⁻² × (1 - S)
    double alpha = PHI_INV_SQ_F64 * (1.0 - S);
    
    // Base scale from D̂ operator
    double scale = 1.0 + alpha * sqrt(norm_sq);
    
    // Stochastic perturbation (decreases with syntony)
    // Uses simple xorshift for speed
    unsigned long long state = seed ^ (i * 2654435761ULL);
    state ^= state >> 12;
    state ^= state << 25;
    state ^= state >> 27;
    double u = (double)(state * 0x2545F4914F6CDD1DULL) / (double)ULLONG_MAX;
    double noise = (u - 0.5) * noise_scale * (1.0 - S);
    
    out[i] = in[i] * (scale + noise);
}

/// Complex version (interleaved re/im)
extern "C" __global__ void resonant_differentiate_c128(
    double *out,
    const double *in,
    const double *mode_norm_sq,
    double syntony,
    double noise_scale,
    unsigned long long seed,
    int n
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    
    int idx = i * 2;
    double norm_sq = mode_norm_sq[i];
    double S = syntony;
    
    double alpha = PHI_INV_SQ_F64 * (1.0 - S);
    double scale = 1.0 + alpha * sqrt(norm_sq);
    
    // Independent noise for real and imaginary
    unsigned long long state_re = seed ^ (i * 2654435761ULL);
    unsigned long long state_im = seed ^ ((i + n) * 2654435761ULL);
    
    state_re ^= state_re >> 12; state_re ^= state_re << 25; state_re ^= state_re >> 27;
    state_im ^= state_im >> 12; state_im ^= state_im << 25; state_im ^= state_im >> 27;
    
    double u_re = (double)(state_re * 0x2545F4914F6CDD1DULL) / (double)ULLONG_MAX;
    double u_im = (double)(state_im * 0x2545F4914F6CDD1DULL) / (double)ULLONG_MAX;
    
    double noise_re = (u_re - 0.5) * noise_scale * (1.0 - S);
    double noise_im = (u_im - 0.5) * noise_scale * (1.0 - S);
    
    out[idx]     = in[idx]     * (scale + noise_re);
    out[idx + 1] = in[idx + 1] * (scale + noise_im);
}
```

## 3.2 Syntony Computation Kernel (Unchanged)

The existing `compute_syntony_f32/c128` kernels in `dhsr.cu` are correct and should be reused for evaluating flux syntony during RES filtering.

## 3.3 Kernel Registration

**File:** `rust/src/resonant/mod.rs`

```rust
#[cfg(feature = "cuda")]
const RESONANT_FUNCS: &[&str] = &[
    "resonant_differentiate_f64",
    "resonant_differentiate_c128",
];

#[cfg(feature = "cuda")]
pub fn ensure_resonant_kernels_loaded(device: &Arc<CudaDevice>) -> PyResult<()> {
    // Load PTX similar to srt_kernels.rs
    // ...
}
```

---

# Part IV: Resonance Enforcer

## 4.1 φ-Dwell Controller

**File:** `rust/src/resonant/resonance.rs`

```rust
use std::time::{Instant, Duration};

/// Golden ratio for timing calculations
const PHI: f64 = 1.6180339887498948482;

/// Controls φ-resonance dwell timing
pub struct ResonanceEnforcer {
    /// Minimum D-phase duration (nanoseconds)
    min_d_duration_ns: u64,
    
    /// Whether to enforce strict φ ratio
    strict_resonance: bool,
    
    /// Statistics
    total_d_time_ns: u64,
    total_h_time_ns: u64,
    cycle_count: u64,
}

impl ResonanceEnforcer {
    pub fn new(strict: bool) -> Self {
        ResonanceEnforcer {
            min_d_duration_ns: 1_000_000,  // 1ms minimum
            strict_resonance: strict,
            total_d_time_ns: 0,
            total_h_time_ns: 0,
            cycle_count: 0,
        }
    }
    
    /// Execute a complete resonant cycle with φ-dwell enforcement
    pub fn execute_cycle<F, G>(
        &mut self,
        tensor: &mut ResonantTensor,
        d_phase: F,
        h_phase: G,
        precision: u32,
    ) -> Result<CycleResult, ResonantError>
    where
        F: FnOnce(&mut ResonantTensor) -> Result<(), ResonantError>,
        G: FnOnce(&mut ResonantTensor, u32, Duration) -> Result<f64, ResonantError>,
    {
        // === D-PHASE (GPU) ===
        let d_start = Instant::now();
        d_phase(tensor)?;
        let d_duration = d_start.elapsed();
        
        // === H-PHASE (CPU) with φ-enforcement ===
        let target_h_duration = Duration::from_nanos(
            (d_duration.as_nanos() as f64 * PHI) as u64
        );
        
        let h_start = Instant::now();
        let new_syntony = h_phase(tensor, precision, target_h_duration)?;
        let h_duration = h_start.elapsed();
        
        // Update statistics
        self.total_d_time_ns += d_duration.as_nanos() as u64;
        self.total_h_time_ns += h_duration.as_nanos() as u64;
        self.cycle_count += 1;
        
        // Compute actual ratio
        let actual_ratio = h_duration.as_nanos() as f64 / d_duration.as_nanos() as f64;
        
        Ok(CycleResult {
            syntony: new_syntony,
            d_duration,
            h_duration,
            target_ratio: PHI,
            actual_ratio,
            ratio_error: (actual_ratio - PHI).abs() / PHI,
        })
    }
    
    /// Get cumulative resonance statistics
    pub fn resonance_ratio(&self) -> f64 {
        if self.total_d_time_ns == 0 {
            0.0
        } else {
            self.total_h_time_ns as f64 / self.total_d_time_ns as f64
        }
    }
}

#[derive(Debug, Clone)]
pub struct CycleResult {
    pub syntony: f64,
    pub d_duration: Duration,
    pub h_duration: Duration,
    pub target_ratio: f64,
    pub actual_ratio: f64,
    pub ratio_error: f64,
}
```

## 4.2 Adaptive Precision Deepening

When H-phase completes faster than target, we increase crystallization precision:

```rust
impl ResonantTensor {
    /// Crystallize with adaptive precision to meet dwell target
    pub fn crystallize_with_dwell(
        &mut self,
        base_precision: u32,
        target_duration: Duration,
    ) -> Result<f64, ResonantError> {
        let start = Instant::now();
        
        // First pass at base precision
        let mut precision = base_precision;
        let mut syntony = self.crystallize(precision)?;
        
        // If we have time remaining, deepen the search
        while start.elapsed() < target_duration && precision < 30 {
            precision += 1;
            
            // Re-snap with higher precision
            // This is "meditation" — using spare cycles productively
            let floats: Vec<f64> = self.lattice.iter()
                .map(|g| g.to_f64())
                .collect();
            
            self.lattice = floats
                .par_iter()
                .map(|&x| GoldenExact::find_nearest(x, precision))
                .collect();
        }
        
        // Final syntony computation
        self.syntony = Self::compute_lattice_syntony(&self.lattice, &self.mode_norm_sq);
        
        Ok(self.syntony)
    }
}
```

---

# Part V: Resonant Evolution Strategy (RES)

## 5.1 Design Philosophy

The key insight: **lattice syntony correlates with task fitness** (geometric beauty predicts utility).

This allows 80% of candidates to be filtered cheaply on CPU before GPU evaluation.

## 5.2 Population Structure

**File:** `rust/src/resonant/evolver.rs`

```rust
/// A candidate in the RES population
#[derive(Clone)]
pub struct Candidate {
    /// The lattice configuration
    pub tensor: ResonantTensor,
    
    /// Lattice syntony (cheap to compute, CPU only)
    pub lattice_syntony: f64,
    
    /// Flux syntony after D-phase (requires GPU)
    pub flux_syntony: Option<f64>,
    
    /// Task fitness (requires forward pass)
    pub fitness: Option<f64>,
    
    /// Combined score: fitness + λ × flux_syntony
    pub score: Option<f64>,
}

/// Configuration for RES
pub struct RESConfig {
    /// Population size (50-100 recommended)
    pub population_size: usize,
    
    /// Number of survivors after lattice filtering (10-20 recommended)
    pub survivor_count: usize,
    
    /// λ weight for syntony in final score
    /// CRITICAL: This is q (syntony deficit), not a hyperparameter
    pub lambda: f64,
    
    /// Mutation scale (how much to perturb lattice)
    pub mutation_scale: f64,
    
    /// Precision for crystallization
    pub precision: u32,
    
    /// Noise scale for D-phase
    pub noise_scale: f64,
}

impl Default for RESConfig {
    fn default() -> Self {
        RESConfig {
            population_size: 64,
            survivor_count: 16,
            lambda: 0.027395146920,  // q — THE syntony deficit
            mutation_scale: 0.1,
            precision: 12,  // F_12 = 144 max denominator
            noise_scale: 0.01,
        }
    }
}
```

## 5.3 RES Algorithm

```rust
pub struct ResonantEvolver {
    config: RESConfig,
    enforcer: ResonanceEnforcer,
    generation: u64,
    
    #[cfg(feature = "cuda")]
    device: Arc<CudaDevice>,
}

impl ResonantEvolver {
    /// Execute one generation of RES
    pub fn evolve_generation<F>(
        &mut self,
        parent: &ResonantTensor,
        fitness_fn: F,
    ) -> Result<ResonantTensor, ResonantError>
    where
        F: Fn(&ResonantTensor) -> f64,
    {
        // === STEP 1: SPAWN MUTANTS (CPU, fast) ===
        let mutants: Vec<Candidate> = (0..self.config.population_size)
            .into_par_iter()  // rayon parallel
            .map(|_| {
                let tensor = self.mutate_lattice(parent);
                let lattice_syntony = tensor.syntony;
                Candidate {
                    tensor,
                    lattice_syntony,
                    flux_syntony: None,
                    fitness: None,
                    score: None,
                }
            })
            .collect();
        
        // === STEP 2: FILTER BY LATTICE SYNTONY (CPU, cheap) ===
        // Sort by lattice syntony, keep top survivor_count
        let mut sorted = mutants;
        sorted.sort_by(|a, b| {
            b.lattice_syntony.partial_cmp(&a.lattice_syntony).unwrap()
        });
        let survivors: Vec<Candidate> = sorted
            .into_iter()
            .take(self.config.survivor_count)
            .collect();
        
        // === STEP 3: MANIFEST IN FLUX (GPU) ===
        // Only survivors get GPU evaluation
        let mut evaluated: Vec<Candidate> = survivors
            .into_iter()
            .map(|mut c| {
                // Wake flux
                c.tensor.wake_flux(self.device.clone())?;
                
                // Apply D-phase
                self.apply_differentiation(&mut c.tensor)?;
                
                // Compute flux syntony (GPU reduction)
                c.flux_syntony = Some(self.compute_flux_syntony(&c.tensor)?);
                
                // Crystallize
                c.tensor.crystallize(self.config.precision)?;
                
                // Compute task fitness
                c.fitness = Some(fitness_fn(&c.tensor));
                
                // Combined score: fitness + λ × flux_syntony
                c.score = Some(
                    c.fitness.unwrap() + 
                    self.config.lambda * c.flux_syntony.unwrap()
                );
                
                Ok(c)
            })
            .collect::<Result<Vec<_>, ResonantError>>()?;
        
        // === STEP 4: SELECT WINNER ===
        evaluated.sort_by(|a, b| {
            b.score.unwrap().partial_cmp(&a.score.unwrap()).unwrap()
        });
        
        let winner = evaluated.into_iter().next().unwrap();
        
        self.generation += 1;
        
        Ok(winner.tensor)
    }
    
    /// Mutate a lattice configuration
    fn mutate_lattice(&self, parent: &ResonantTensor) -> ResonantTensor {
        let mut rng = rand::thread_rng();
        
        let mutated_lattice: Vec<GoldenExact> = parent.lattice
            .iter()
            .map(|g| {
                // Perturb in Q(φ) space
                let delta_a = (rng.gen::<f64>() - 0.5) * self.config.mutation_scale;
                let delta_b = (rng.gen::<f64>() - 0.5) * self.config.mutation_scale;
                
                // Convert perturbation to GoldenExact
                let perturbation = GoldenExact::from_f64_approx(
                    delta_a + delta_b * GoldenExact::phi().to_f64(),
                    self.config.precision
                );
                
                *g + perturbation
            })
            .collect();
        
        ResonantTensor::from_lattice(
            mutated_lattice,
            parent.shape.clone(),
            parent.mode_norm_sq.clone(),
        )
    }
    
    /// Apply D-phase differentiation on GPU
    #[cfg(feature = "cuda")]
    fn apply_differentiation(&self, tensor: &mut ResonantTensor) -> Result<(), ResonantError> {
        let flux = tensor.flux.as_ref().ok_or(ResonantError::NoFluxPresent)?;
        let n = flux.len();
        
        // Upload mode_norm_sq
        let mode_norms = self.device.htod_sync_copy(&tensor.mode_norm_sq)?;
        
        // Allocate output
        let mut output: CudaSlice<f64> = self.device.alloc_zeros(n)?;
        
        // Launch kernel
        let cfg = launch_cfg_256(n);
        let seed = rand::random::<u64>();
        
        let func = self.device.get_func("resonant", "resonant_differentiate_f64")?;
        unsafe {
            func.launch(cfg, (
                &mut output,
                flux.as_ref(),
                &mode_norms,
                tensor.syntony,
                self.config.noise_scale,
                seed,
                n as i32,
            ))?;
        }
        
        tensor.flux = Some(Arc::new(output));
        Ok(())
    }
}
```

## 5.4 Gradient Hint from Snap Distance

The difference between pre-snap and post-snap values provides a "synthetic gradient":

```rust
impl ResonantEvolver {
    /// Compute gradient hint from crystallization snap distance
    /// This biases future mutations toward directions that reduce snap error
    pub fn compute_snap_gradient(
        pre_snap: &[f64],
        post_snap: &[GoldenExact],
    ) -> Vec<f64> {
        pre_snap.iter()
            .zip(post_snap.iter())
            .map(|(&pre, post)| {
                let post_f64 = post.to_f64();
                // Gradient points from pre toward post
                post_f64 - pre
            })
            .collect()
    }
    
    /// Apply directed mutation using snap gradient
    fn mutate_directed(
        &self,
        parent: &ResonantTensor,
        gradient: &[f64],
        gradient_weight: f64,
    ) -> ResonantTensor {
        let mut rng = rand::thread_rng();
        
        let mutated_lattice: Vec<GoldenExact> = parent.lattice
            .iter()
            .zip(gradient.iter())
            .map(|(g, &grad)| {
                // Random component
                let random_delta = (rng.gen::<f64>() - 0.5) * self.config.mutation_scale;
                
                // Directed component (follows gradient)
                let directed_delta = grad * gradient_weight * self.config.mutation_scale;
                
                // Combined perturbation
                let total_delta = random_delta * (1.0 - gradient_weight) + directed_delta;
                
                let perturbation = GoldenExact::from_f64_approx(
                    total_delta,
                    self.config.precision
                );
                
                *g + perturbation
            })
            .collect();
        
        ResonantTensor::from_lattice(
            mutated_lattice,
            parent.shape.clone(),
            parent.mode_norm_sq.clone(),
        )
    }
}
```

---

# Part VI: Python Interface

## 6.1 ResonantTensor Bindings

**File:** `python/syntonic/resonant/tensor.py`

```python
from syntonic.core import ResonantTensor as _ResonantTensor
from syntonic.exact import GoldenExact, PHI_NUMERIC
import numpy as np
from typing import Optional, List, Tuple

class ResonantTensor:
    """
    Dual-state tensor for the Resonant Engine.
    
    Contains:
        - lattice: Exact values in Q(φ)
        - flux: Ephemeral GPU floats (when in D-phase)
    
    Example:
        >>> tensor = ResonantTensor.from_array(np.random.randn(64))
        >>> tensor.wake_flux("cuda:0")
        >>> tensor.differentiate()
        >>> syntony = tensor.crystallize()
        >>> print(f"Post-crystallization syntony: {syntony:.4f}")
    """
    
    def __init__(self, _inner: _ResonantTensor):
        self._inner = _inner
    
    @classmethod
    def from_array(
        cls,
        data: np.ndarray,
        mode_norm_sq: Optional[np.ndarray] = None,
        precision: int = 12,
    ) -> 'ResonantTensor':
        """Create from numpy array by snapping to golden lattice."""
        flat = data.flatten().astype(np.float64)
        
        if mode_norm_sq is None:
            # Default: use index as mode (|n|² = n²)
            mode_norm_sq = np.arange(len(flat), dtype=np.float64) ** 2
        
        inner = _ResonantTensor.from_floats(
            flat.tolist(),
            list(data.shape),
            mode_norm_sq.tolist(),
            precision,
        )
        return cls(inner)
    
    @property
    def syntony(self) -> float:
        """Current syntony S(Ψ)."""
        return self._inner.syntony
    
    @property
    def phase(self) -> str:
        """Current phase: 'crystalline' or 'fluxed'."""
        return self._inner.phase
    
    @property
    def shape(self) -> Tuple[int, ...]:
        """Tensor shape."""
        return tuple(self._inner.shape)
    
    def wake_flux(self, device: str = "cuda:0") -> None:
        """Project lattice to GPU flux."""
        self._inner.wake_flux(device)
    
    def differentiate(self, noise_scale: float = 0.01) -> None:
        """Apply D̂ operator on GPU."""
        self._inner.differentiate(noise_scale)
    
    def crystallize(self, precision: int = 12) -> float:
        """Snap flux back to lattice, return new syntony."""
        return self._inner.crystallize(precision)
    
    def to_numpy(self) -> np.ndarray:
        """Convert lattice to numpy array (evaluates exact → float)."""
        return np.array(self._inner.to_list()).reshape(self.shape)
```

## 6.2 ResonantEngine Class

**File:** `python/syntonic/resonant/engine.py`

```python
from syntonic.resonant.tensor import ResonantTensor
from syntonic.core import ResonanceEnforcer as _Enforcer
from syntonic.exact import Q_DEFICIT_NUMERIC
from typing import Callable, Optional
import time

class ResonantEngine:
    """
    The Resonant Engine: hardware-native SRT/CRT.
    
    GPU = D̂ (Differentiation)
    CPU = Ĥ (Harmonization via crystallization)
    PCIe = Phase boundary (φ-dwell enforcer)
    
    Example:
        >>> engine = ResonantEngine(device="cuda:0")
        >>> tensor = ResonantTensor.from_array(initial_state)
        >>> 
        >>> for _ in range(100):
        ...     result = engine.cycle(tensor)
        ...     print(f"S={result.syntony:.4f}, ratio={result.actual_ratio:.3f}")
    """
    
    PHI = 1.6180339887498948482
    
    def __init__(
        self,
        device: str = "cuda:0",
        strict_resonance: bool = True,
        base_precision: int = 12,
        noise_scale: float = 0.01,
    ):
        self.device = device
        self.strict_resonance = strict_resonance
        self.base_precision = base_precision
        self.noise_scale = noise_scale
        
        self._enforcer = _Enforcer(strict_resonance)
        
        # Statistics
        self.total_cycles = 0
        self.cumulative_d_time = 0.0
        self.cumulative_h_time = 0.0
    
    def cycle(
        self,
        tensor: ResonantTensor,
        precision: Optional[int] = None,
    ) -> 'CycleResult':
        """
        Execute one complete DHSR cycle with φ-resonance.
        
        Args:
            tensor: ResonantTensor to evolve
            precision: Crystallization precision (default: base_precision)
        
        Returns:
            CycleResult with timing and syntony information
        """
        precision = precision or self.base_precision
        
        # === D-PHASE (GPU) ===
        d_start = time.perf_counter_ns()
        
        tensor.wake_flux(self.device)
        tensor.differentiate(self.noise_scale)
        
        d_end = time.perf_counter_ns()
        d_duration_ns = d_end - d_start
        
        # === H-PHASE (CPU) with φ-dwell ===
        target_h_ns = int(d_duration_ns * self.PHI)
        
        h_start = time.perf_counter_ns()
        
        if self.strict_resonance:
            syntony = tensor._inner.crystallize_with_dwell(
                precision,
                target_h_ns,
            )
        else:
            syntony = tensor.crystallize(precision)
        
        h_end = time.perf_counter_ns()
        h_duration_ns = h_end - h_start
        
        # Update statistics
        self.total_cycles += 1
        self.cumulative_d_time += d_duration_ns / 1e9
        self.cumulative_h_time += h_duration_ns / 1e9
        
        actual_ratio = h_duration_ns / d_duration_ns if d_duration_ns > 0 else 0
        
        return CycleResult(
            syntony=syntony,
            d_duration_ms=d_duration_ns / 1e6,
            h_duration_ms=h_duration_ns / 1e6,
            target_ratio=self.PHI,
            actual_ratio=actual_ratio,
        )
    
    @property
    def resonance_ratio(self) -> float:
        """Cumulative H/D time ratio (should approach φ)."""
        if self.cumulative_d_time == 0:
            return 0.0
        return self.cumulative_h_time / self.cumulative_d_time


class CycleResult:
    """Result of one resonant cycle."""
    
    def __init__(
        self,
        syntony: float,
        d_duration_ms: float,
        h_duration_ms: float,
        target_ratio: float,
        actual_ratio: float,
    ):
        self.syntony = syntony
        self.d_duration_ms = d_duration_ms
        self.h_duration_ms = h_duration_ms
        self.target_ratio = target_ratio
        self.actual_ratio = actual_ratio
        self.ratio_error = abs(actual_ratio - target_ratio) / target_ratio
    
    def __repr__(self) -> str:
        return (
            f"CycleResult(S={self.syntony:.4f}, "
            f"D={self.d_duration_ms:.2f}ms, H={self.h_duration_ms:.2f}ms, "
            f"ratio={self.actual_ratio:.3f})"
        )
```

## 6.3 RES Python Interface

**File:** `python/syntonic/resonant/evolver.py`

```python
from syntonic.resonant.tensor import ResonantTensor
from syntonic.resonant.engine import ResonantEngine
from syntonic.exact import Q_DEFICIT_NUMERIC
from typing import Callable, List, Optional
import numpy as np

class ResonantEvolver:
    """
    Resonant Evolution Strategy: discrete learning without backprop.
    
    Key insight: syntony is a universal fitness proxy.
    80% of candidates die cheap on CPU (lattice syntony check).
    Only promising geometries get GPU evaluation.
    
    Example:
        >>> evolver = ResonantEvolver(population_size=64, survivors=16)
        >>> 
        >>> def fitness(tensor):
        ...     return -loss_function(tensor.to_numpy())
        >>> 
        >>> for gen in range(100):
        ...     state = evolver.evolve(state, fitness)
        ...     print(f"Gen {gen}: S={state.syntony:.4f}")
    """
    
    def __init__(
        self,
        population_size: int = 64,
        survivors: int = 16,
        mutation_scale: float = 0.1,
        precision: int = 12,
        device: str = "cuda:0",
        lambda_weight: Optional[float] = None,  # Defaults to q
    ):
        self.population_size = population_size
        self.survivors = survivors
        self.mutation_scale = mutation_scale
        self.precision = precision
        self.device = device
        
        # λ = q (syntony deficit) — NOT A HYPERPARAMETER
        self.lambda_weight = lambda_weight or Q_DEFICIT_NUMERIC
        
        self.engine = ResonantEngine(device=device, base_precision=precision)
        self.generation = 0
    
    def evolve(
        self,
        parent: ResonantTensor,
        fitness_fn: Callable[[ResonantTensor], float],
    ) -> ResonantTensor:
        """
        Evolve for one generation.
        
        Args:
            parent: Current best tensor
            fitness_fn: Fitness function (higher = better)
        
        Returns:
            Winner of this generation
        """
        # STEP 1: Spawn mutants (CPU, parallel via numpy)
        mutants = [self._mutate(parent) for _ in range(self.population_size)]
        
        # STEP 2: Filter by lattice syntony (cheap)
        mutants.sort(key=lambda t: t.syntony, reverse=True)
        survivors = mutants[:self.survivors]
        
        # STEP 3: GPU evaluation of survivors
        scores = []
        for tensor in survivors:
            # D-phase
            result = self.engine.cycle(tensor)
            flux_syntony = result.syntony
            
            # Task fitness
            fitness = fitness_fn(tensor)
            
            # Combined score: fitness + λ × syntony
            score = fitness + self.lambda_weight * flux_syntony
            scores.append((tensor, score, fitness, flux_syntony))
        
        # STEP 4: Select winner
        scores.sort(key=lambda x: x[1], reverse=True)
        winner, best_score, best_fitness, best_syntony = scores[0]
        
        self.generation += 1
        
        return winner
    
    def _mutate(self, parent: ResonantTensor) -> ResonantTensor:
        """Mutate lattice configuration."""
        data = parent.to_numpy()
        
        # Perturbation in value space
        noise = np.random.randn(*data.shape) * self.mutation_scale
        mutated = data + noise
        
        return ResonantTensor.from_array(
            mutated,
            mode_norm_sq=parent._inner.mode_norm_sq,
            precision=self.precision,
        )
```

---

# Part VII: Integration Points

## 7.1 With Existing syntonic.crt

The Resonant Engine can wrap existing DHSR operators:

```python
from syntonic.crt import create_dhsr_system, DHSREvolver
from syntonic.resonant import ResonantEngine, ResonantTensor

# Existing API
R_op, S_comp, G_comp = create_dhsr_system()

# New Resonant API
engine = ResonantEngine()
tensor = ResonantTensor.from_array(initial_state)

# They can interoperate:
# - ResonantTensor.to_state() → syntonic.core.State
# - State.to_resonant() → ResonantTensor
```

## 7.2 With Existing CUDA Kernels

The `dhsr.cu` kernels are retained for:
- Full fused DHSR when GPU-only mode is desired
- Syntony computation (`compute_syntony_c128`)
- Gnosis computation (`compute_gnosis_f32`)

New `resonant_d.cu` provides D-only with stochastic perturbation.

## 7.3 With Existing Exact Arithmetic

`GoldenExact` needs one new method:
- `find_nearest(f64, precision) -> GoldenExact`

This is the only addition to `rust/src/exact/golden.rs`.

---

# Part VIII: Testing Strategy

## 8.1 Unit Tests

```
tests/
└── test_resonant/
    ├── test_tensor.py           # ResonantTensor construction, phases
    ├── test_crystallize.py      # Golden snap accuracy
    ├── test_resonance.py        # φ-dwell enforcement
    ├── test_evolver.py          # RES algorithm
    └── test_integration.py      # Full cycle, interop with crt
```

## 8.2 Key Assertions

```python
def test_phi_resonance():
    """Verify cumulative H/D ratio approaches φ."""
    engine = ResonantEngine(strict_resonance=True)
    tensor = ResonantTensor.from_array(np.random.randn(1024))
    
    for _ in range(100):
        engine.cycle(tensor)
    
    ratio = engine.resonance_ratio
    assert abs(ratio - 1.618) < 0.1, f"Ratio {ratio} not near φ"

def test_syntony_increases():
    """Verify syntony generally increases over cycles."""
    engine = ResonantEngine()
    tensor = ResonantTensor.from_array(np.random.randn(256))
    
    initial_s = tensor.syntony
    for _ in range(50):
        result = engine.cycle(tensor)
    
    assert result.syntony > initial_s * 0.9  # Allow some variance

def test_lattice_exactness():
    """Verify lattice values are truly in Q(φ)."""
    tensor = ResonantTensor.from_array(np.array([1.0, 1.618, 2.618]))
    
    for g in tensor._inner.lattice:
        # GoldenExact should have rational a, b
        a_num, a_den = g.rational_coefficient
        b_num, b_den = g.phi_coefficient
        assert isinstance(a_num, int) and isinstance(a_den, int)
        assert isinstance(b_num, int) and isinstance(b_den, int)
```

---

# Part IX: Performance Considerations

## 9.1 Expected Bottlenecks

| Operation | Location | Expected Time | Notes |
|-----------|----------|---------------|-------|
| `wake_flux` | PCIe H→D | ~1ms for 1M elements | Memory bandwidth limited |
| `differentiate` | GPU | ~0.1ms for 1M elements | Compute bound |
| `crystallize` | CPU | ~10ms for 1M elements | Intentionally slow for φ-dwell |
| Golden snap | CPU | O(n × precision²) | Dominant cost in H-phase |

## 9.2 Optimizations

1. **Batch crystallization**: Process elements in cache-friendly blocks
2. **Precomputed Fibonacci bounds**: Cache F_k for k ≤ 30
3. **SIMD for snap search**: AVX2 parallel search over candidate rationals
4. **Async PCIe**: Overlap transfer with computation where possible

## 9.3 Memory Layout

```
ResonantTensor memory:
├── lattice: Vec<GoldenExact>     # 2×16 bytes per element (two Rationals)
│   └── ~32 bytes/element         # Each Rational: num(i128) + den(i128)
├── flux: CudaSlice<f64>          # 8 bytes/element (GPU)
├── mode_norm_sq: Vec<f64>        # 8 bytes/element (CPU, precomputed)
└── metadata: ~64 bytes           # shape, phase, syntony, etc.

Total CPU: ~40 bytes/element
Total GPU: ~8 bytes/element (when fluxed)
```

---

# Part X: Deliverables Checklist

## 10.1 Rust Components

- [ ] `rust/src/resonant/mod.rs` — Module declaration
- [ ] `rust/src/resonant/tensor.rs` — ResonantTensor struct
- [ ] `rust/src/resonant/crystallize.rs` — Golden snap algorithm
- [ ] `rust/src/resonant/resonance.rs` — φ-dwell enforcer
- [ ] `rust/src/resonant/evolver.rs` — RES implementation
- [ ] `rust/src/exact/golden.rs` — Add `find_nearest` method

## 10.2 CUDA Components

- [ ] `rust/kernels/resonant_d.cu` — D-phase only kernel with noise
- [ ] `rust/kernels/ptx/resonant_d_sm*.ptx` — Compiled PTX

## 10.3 Python Components

- [ ] `python/syntonic/resonant/__init__.py` — Module exports
- [ ] `python/syntonic/resonant/tensor.py` — ResonantTensor bindings
- [ ] `python/syntonic/resonant/engine.py` — ResonantEngine class
- [ ] `python/syntonic/resonant/evolver.py` — RES Python interface

## 10.4 Tests

- [ ] `tests/test_resonant/test_tensor.py`
- [ ] `tests/test_resonant/test_crystallize.py`
- [ ] `tests/test_resonant/test_resonance.py`
- [ ] `tests/test_resonant/test_evolver.py`
- [ ] `tests/test_resonant/test_integration.py`

## 10.5 Documentation

- [ ] Docstrings for all public APIs
- [ ] `docs/resonant_engine.md` — User guide
- [ ] `CHANGELOG.md` — Update with new module

---

# Appendix A: Mathematical Reference

## A.1 Operator Definitions

**Differentiation:**
$$\hat{D}[\psi]_n = \psi_n \times \left(1 + \alpha(S) \sqrt{|n|^2}\right)$$
$$\alpha(S) = \phi^{-2}(1 - S) \approx 0.382(1 - S)$$

**Harmonization (via crystallization):**
$$\hat{H}[\psi]_n = \text{snap}(\psi_n) \in \mathbb{Q}(\phi)$$

**Syntony:**
$$S(\Psi) = \frac{\sum_n |\psi_n|^2 e^{-|n|^2/\phi}}{\sum_n |\psi_n|^2}$$

## A.2 Golden Field Q(φ)

Elements: $a + b\phi$ where $a, b \in \mathbb{Q}$

Key identity: $\phi^2 = \phi + 1$

Inverse: $\frac{1}{\phi} = \phi - 1$

Norm: $N(a + b\phi) = a^2 + ab - b^2$

## A.3 Resonance Ratio

Target: $\frac{t_H}{t_D} = \phi \approx 1.618$

Equivalently: $\frac{t_D}{t_D + t_H} = \phi^{-2} \approx 0.382$

---

*End of Implementation Plan v1.0*