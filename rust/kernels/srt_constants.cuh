// =============================================================================
// SRT Theory Constants - Exact Algebraic and Transcendental Constants
// =============================================================================
//
// IMPORTANT: SRT is an EXACT theory built on the algebraic number field Q(φ).
//
// This header distinguishes between:
//   1. ALGEBRAIC constants - Exactly representable in Q(φ) as (a, b) meaning a + b·φ
//   2. TRANSCENDENTAL constants - Involve π, e and can only be approximated
//
// The Lattice/Flux Duality:
//   - LATTICE PHASE (CPU): Uses exact GoldenExact arithmetic (a + b·φ, a,b ∈ Q)
//   - FLUX PHASE (GPU): Uses floating-point approximations for speed
//   - CRYSTALLIZATION: Snaps GPU results back to exact Q(φ) lattice
//
// GPU kernels operate in Flux phase. All floating-point values here are
// APPROXIMATIONS of exact quantities. The authoritative exact values live
// in the Rust GoldenExact infrastructure.
//
// =============================================================================

#ifndef SRT_CONSTANTS_CUH
#define SRT_CONSTANTS_CUH

#include <cuda_runtime.h>

// =============================================================================
// Warp-Level Reduction Primitives (CUDA Helper Functions)
// =============================================================================

__device__ __forceinline__ float warp_reduce_sum(float val) {
    for (int offset = 16; offset > 0; offset >>= 1) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

__device__ __forceinline__ double warp_reduce_sum_f64(double val) {
    for (int offset = 16; offset > 0; offset >>= 1) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

__device__ __forceinline__ float warp_reduce_max(float val) {
    for (int offset = 16; offset > 0; offset >>= 1) {
        val = fmaxf(val, __shfl_down_sync(0xffffffff, val, offset));
    }
    return val;
}

__device__ __forceinline__ double warp_reduce_max_f64(double val) {
    for (int offset = 16; offset > 0; offset >>= 1) {
        val = fmax(val, __shfl_down_sync(0xffffffff, val, offset));
    }
    return val;
}

// =============================================================================
// ALGEBRAIC CONSTANTS - Exactly representable in Q(φ)
// =============================================================================
//
// In Q(φ), elements are represented as a + b·φ where a, b ∈ Q (rationals).
// For each algebraic constant, we provide:
//   - The exact representation (a, b) as integer pairs
//   - The floating-point approximation for GPU computation
//
// Key algebraic identities:
//   φ² = φ + 1        (fundamental recurrence)
//   φ · φ̂ = -1        (where φ̂ = 1 - φ is the Galois conjugate)
//   1/φ = φ - 1       (inverse)
//   √5 = 2φ - 1       (sqrt(5) is in Q(φ))
//   φⁿ = Fₙ₋₁ + Fₙ·φ  (Fibonacci representation)
//
// =============================================================================

// -----------------------------------------------------------------------------
// Golden Ratio φ = (1 + √5)/2
// EXACT: (0, 1) meaning 0 + 1·φ
// -----------------------------------------------------------------------------
#define PHI_EXACT_A             0
#define PHI_EXACT_B             1
#define PHI_F32                 1.6180339887498948482f
#define PHI_F64                 1.6180339887498948482

// -----------------------------------------------------------------------------
// Golden Ratio Squared φ² = φ + 1
// EXACT: (1, 1) meaning 1 + 1·φ
// -----------------------------------------------------------------------------
#define PHI_SQ_EXACT_A          1
#define PHI_SQ_EXACT_B          1
#define PHI_SQ_F32              2.6180339887498948482f
#define PHI_SQ_F64              2.6180339887498948482

// -----------------------------------------------------------------------------
// Golden Ratio Inverse 1/φ = φ - 1 (also written φ̂ or ψ)
// EXACT: (-1, 1) meaning -1 + 1·φ
// -----------------------------------------------------------------------------
#define PHI_INV_EXACT_A         (-1)
#define PHI_INV_EXACT_B         1
#define PHI_INV_F32             0.6180339887498948482f
#define PHI_INV_F64             0.6180339887498948482

// -----------------------------------------------------------------------------
// Golden Ratio Inverse Squared 1/φ² = 2 - φ
// EXACT: (2, -1) meaning 2 - 1·φ
// -----------------------------------------------------------------------------
#define PHI_INV_SQ_EXACT_A      2
#define PHI_INV_SQ_EXACT_B      (-1)
#define PHI_INV_SQ_F32          0.3819660112501051518f
#define PHI_INV_SQ_F64          0.3819660112501051518

// -----------------------------------------------------------------------------
// Square Root of 5: √5 = 2φ - 1
// EXACT: (-1, 2) meaning -1 + 2·φ
// -----------------------------------------------------------------------------
#define SQRT5_EXACT_A           (-1)
#define SQRT5_EXACT_B           2
#define SQRT5_F32               2.2360679774997896964f
#define SQRT5_F64               2.2360679774997896964

// -----------------------------------------------------------------------------
// Galois Conjugate φ̂ = (1 - √5)/2 = 1 - φ
// EXACT: (1, -1) meaning 1 - 1·φ
// Note: φ̂ ≈ -0.618... (negative!)
// -----------------------------------------------------------------------------
#define PHI_CONJ_EXACT_A        1
#define PHI_CONJ_EXACT_B        (-1)
#define PHI_CONJ_F32            (-0.6180339887498948482f)
#define PHI_CONJ_F64            (-0.6180339887498948482)

// -----------------------------------------------------------------------------
// Unity (for completeness)
// EXACT: (1, 0)
// -----------------------------------------------------------------------------
#define ONE_EXACT_A             1
#define ONE_EXACT_B             0

// =============================================================================
// EXACT φⁿ via Fibonacci Numbers
// =============================================================================
//
// The EXACT representation of φⁿ is: φⁿ = F_{n-1} + F_n · φ
// where F_n is the n-th Fibonacci number (F₀=0, F₁=1, F₂=1, F₃=2, ...)
//
// This is EXACT in Q(φ) - no floating point approximation needed!
// GPU code can use these Fibonacci pairs for exact scaling operations.
//
// =============================================================================

// Fibonacci numbers F_n (exact integers)
#define FIB_0   0
#define FIB_1   1
#define FIB_2   1
#define FIB_3   2
#define FIB_4   3
#define FIB_5   5
#define FIB_6   8
#define FIB_7   13
#define FIB_8   21
#define FIB_9   34
#define FIB_10  55
#define FIB_11  89
#define FIB_12  144
#define FIB_13  233
#define FIB_14  377
#define FIB_15  610
#define FIB_16  987
#define FIB_17  1597
#define FIB_18  2584
#define FIB_19  4181
#define FIB_20  6765
#define FIB_21  10946
#define FIB_22  17711
#define FIB_23  28657
#define FIB_24  46368
#define FIB_25  75025
#define FIB_26  121393
#define FIB_27  196418
#define FIB_28  317811
#define FIB_29  514229
#define FIB_30  832040
#define FIB_31  1346269

// φⁿ EXACT representations as (F_{n-1}, F_n) pairs
// φ⁰ = 1 = (1, 0)   [F_{-1}=1 by convention]
// φ¹ = φ = (0, 1)
// φ² = 1 + φ = (1, 1)
// φ³ = 1 + 2φ = (1, 2)
// φ⁴ = 2 + 3φ = (2, 3)
// ...
#define PHI_POW_0_EXACT_A   1
#define PHI_POW_0_EXACT_B   0
#define PHI_POW_1_EXACT_A   0
#define PHI_POW_1_EXACT_B   1
#define PHI_POW_2_EXACT_A   1
#define PHI_POW_2_EXACT_B   1
#define PHI_POW_3_EXACT_A   1
#define PHI_POW_3_EXACT_B   2
#define PHI_POW_4_EXACT_A   2
#define PHI_POW_4_EXACT_B   3
#define PHI_POW_5_EXACT_A   3
#define PHI_POW_5_EXACT_B   5
#define PHI_POW_6_EXACT_A   5
#define PHI_POW_6_EXACT_B   8
#define PHI_POW_7_EXACT_A   8
#define PHI_POW_7_EXACT_B   13
#define PHI_POW_8_EXACT_A   13
#define PHI_POW_8_EXACT_B   21
#define PHI_POW_9_EXACT_A   21
#define PHI_POW_9_EXACT_B   34
#define PHI_POW_10_EXACT_A  34
#define PHI_POW_10_EXACT_B  55

// Floating-point approximations for φⁿ (GPU Flux phase only)
#define PHI_POW_0_F64   1.0
#define PHI_POW_1_F64   1.6180339887498948482
#define PHI_POW_2_F64   2.6180339887498948482
#define PHI_POW_3_F64   4.2360679774997896964
#define PHI_POW_4_F64   6.8541019662496845446
#define PHI_POW_5_F64   11.0901699437494742410
#define PHI_POW_6_F64   17.9442719099991587856
#define PHI_POW_7_F64   29.0344418537486330266
#define PHI_POW_8_F64   46.9787137637477918122
#define PHI_POW_9_F64   76.0131556174964248388
#define PHI_POW_10_F64  122.9918693812442166510

// =============================================================================
// TRANSCENDENTAL CONSTANTS - Cannot be exactly represented in Q(φ)
// =============================================================================
//
// These constants involve π and e, which are transcendental numbers
// not in Q(φ). They can only be approximated.
//
// The values here are derived from exact formulas but stored as approximations.
// =============================================================================

// -----------------------------------------------------------------------------
// π (Archimedes' constant) - TRANSCENDENTAL
// Defines toroidal boundary of vacuum (modular volume = π/3)
// -----------------------------------------------------------------------------
#define PI_F32                  3.14159265358979323846f
#define PI_F64                  3.14159265358979323846

// -----------------------------------------------------------------------------
// e (Euler's number) - TRANSCENDENTAL
// Base of natural logarithm, appears in heat kernel decay
// -----------------------------------------------------------------------------
#define EULER_E_F32             2.71828182845904523536f
#define EULER_E_F64             2.71828182845904523536

// -----------------------------------------------------------------------------
// γ (Euler-Mascheroni constant) - TRANSCENDENTAL
// Limit of (1 + 1/2 + ... + 1/n - ln(n)) as n → ∞
// -----------------------------------------------------------------------------
#define EULER_GAMMA_F32         0.57721566490153286061f
#define EULER_GAMMA_F64         0.57721566490153286061

// -----------------------------------------------------------------------------
// E* = e^π - π (Spectral Möbius constant) - TRANSCENDENTAL
// The finite part of the Möbius-regularized heat kernel trace on Golden Lattice
// EXACT FORMULA: E* = e^π - π
// Decomposition: E* = Γ(1/4)² + π(π-1) + (35/12)e^(-π) + Δ
//   where Δ ≈ 4.30 × 10⁻⁷ (the residual driving cosmic evolution)
// -----------------------------------------------------------------------------
#define E_STAR_F32              19.999099979189475767f
#define E_STAR_F64              19.999099979189475767

// -----------------------------------------------------------------------------
// q (Universal Syntony Deficit) - TRANSCENDENTAL (derived)
// EXACT FORMULA: q = (2φ + e/(2φ²)) / (φ⁴ · E*)
// This is THE fundamental constant scaling all physical observables.
// Not a free parameter - derived from geometry!
// -----------------------------------------------------------------------------
#define Q_DEFICIT_F32           0.027395146920f
#define Q_DEFICIT_F64           0.027395146920

// =============================================================================
// EXACT ARITHMETIC HELPER FUNCTIONS
// =============================================================================
//
// These functions perform operations that ARE exact in Q(φ).
// They work with the (a, b) representation where value = a + b·φ.
//
// For GPU efficiency, we use floating-point but the OPERATIONS are exact.
// =============================================================================

// Multiply by φ: (a + b·φ) × φ = b + (a + b)·φ
// Uses identity: φ² = φ + 1
__device__ __forceinline__ void mul_phi_exact(double* a, double* b) {
    double new_a = *b;
    double new_b = *a + *b;
    *a = new_a;
    *b = new_b;
}

// Divide by φ: (a + b·φ) / φ = (b - a) + a·φ
// Uses identity: 1/φ = φ - 1
__device__ __forceinline__ void div_phi_exact(double* a, double* b) {
    double new_a = *b - *a;
    double new_b = *a;
    *a = new_a;
    *b = new_b;
}

// Galois conjugate: (a + b·φ)* = (a + b) - b·φ
// Maps φ → 1-φ (the other root of x² - x - 1 = 0)
__device__ __forceinline__ void conjugate_exact(double* a, double* b) {
    double new_a = *a + *b;
    double new_b = -(*b);
    *a = new_a;
    *b = new_b;
}

// Field norm: N(a + b·φ) = a² + ab - b² (always rational/real)
__device__ __forceinline__ double norm_exact(double a, double b) {
    return a * a + a * b - b * b;
}

// Field trace: Tr(a + b·φ) = 2a + b (always rational/real)
__device__ __forceinline__ double trace_exact(double a, double b) {
    return 2.0 * a + b;
}

// Evaluate (a, b) to floating-point: a + b·φ
__device__ __forceinline__ double eval_golden(double a, double b) {
    return a + b * PHI_F64;
}

__device__ __forceinline__ float eval_golden_f32(float a, float b) {
    return a + b * PHI_F32;
}

// Golden weight: w(n) = exp(-|n|²/φ) - the fundamental measure on T⁴
// Note: The exponent |n|²/φ can be computed exactly if |n|² is integer
__device__ __forceinline__ double golden_weight(double norm_sq) {
    return exp(-norm_sq * PHI_INV_F64);
}

__device__ __forceinline__ float golden_weight_f32(float norm_sq) {
    return __expf(-norm_sq * PHI_INV_F32);
}

// =============================================================================
// LIE GROUP STRUCTURE CONSTANTS (Exact Integers)
// =============================================================================

// E₈ - Exceptional unification
#define E8_ROOTS            240     // |Φ(E₈)| - total roots
#define E8_POSITIVE_ROOTS   120     // |Φ⁺(E₈)| - positive roots
#define E8_RANK             8       // rank(E₈) - Cartan dimension
#define E8_COXETER          30      // h(E₈) - Coxeter number
#define E8_DIMENSION        248     // dim(E₈) - adjoint representation

// E₇ - Intermediate unification
#define E7_ROOTS            126     // |Φ(E₇)|
#define E7_POSITIVE_ROOTS   63      // |Φ⁺(E₇)|
#define E7_RANK             7
#define E7_COXETER          18
#define E7_DIMENSION        133
#define E7_FUNDAMENTAL      56      // dim of fundamental rep

// E₆ - GUT unification / Golden Cone
#define E6_ROOTS            72      // |Φ(E₆)|
#define E6_POSITIVE_ROOTS   36      // |Φ⁺(E₆)| = GOLDEN CONE
#define E6_RANK             6
#define E6_COXETER          12
#define E6_DIMENSION        78
#define E6_FUNDAMENTAL      27

// D₄ - Spacetime/Consciousness
#define D4_ROOTS            24      // |Φ(D₄)|
#define D4_RANK             4
#define D4_COXETER          6
#define D4_DIMENSION        28
#define D4_KISSING          24      // Consciousness emergence threshold

// =============================================================================
// PRIME NUMBER THEORY CONSTANTS
// =============================================================================

// Mersenne Prime Exponents: p where M_p = 2^p - 1 is prime
// These govern matter stability (M₁₁ barrier prevents 4th generation)
#define MERSENNE_2     true
#define MERSENNE_3     true
#define MERSENNE_5     true
#define MERSENNE_7     true
#define MERSENNE_13    true
#define MERSENNE_17    true     // 2^17-1 = 131071 is prime
#define MERSENNE_19    true     // 2^19-1 = 524287 is prime
#define MERSENNE_31    true     // 2^31-1 = 2147483647 is prime
#define MERSENNE_61    true
#define MERSENNE_89    true
#define MERSENNE_107   true
#define MERSENNE_127   true

// Fermat Primes: F_n = 2^(2^n) + 1
// These govern force existence (F₅ composite means no 6th force)
#define FERMAT_0       3        // 2^1 + 1 = 3 (prime) - Strong force
#define FERMAT_1       5        // 2^2 + 1 = 5 (prime) - Electroweak
#define FERMAT_2       17       // 2^4 + 1 = 17 (prime) - EM boundary
#define FERMAT_3       257      // 2^8 + 1 = 257 (prime) - Gravity
#define FERMAT_4       65537    // 2^16 + 1 = 65537 (prime) - Versal
// FERMAT_5 = 4294967297 = 641 × 6700417 (COMPOSITE - no 6th force!)

// Pisano Periods π(m): period of Fibonacci sequence mod m
#define PISANO_1     1
#define PISANO_2     3
#define PISANO_3     8
#define PISANO_4     6
#define PISANO_5     20
#define PISANO_6     24
#define PISANO_7     16
#define PISANO_8     12
#define PISANO_9     24
#define PISANO_10    60

// =============================================================================
// BLOCK SIZE CONFIGURATION (SRT-optimized)
// =============================================================================

#define BLOCK_E8_ROOTS      240     // One thread per E₈ root
#define BLOCK_GOLDEN_CONE   36      // E₆ positive roots (Golden Cone)
#define BLOCK_WINDING_4D    256     // 4⁴ for T⁴ winding space
#define BLOCK_GENERATIONS   192     // 64 × 3 generations
#define BLOCK_D4_KISSING    96      // 4 × 24 consciousness threshold
#define BLOCK_DEFAULT       256     // Default for general ops

// Projection normalization: 1/√2 (for E₈ → 4D projections)
#define PROJ_NORM_F32       0.70710678118654752440f
#define PROJ_NORM_F64       0.70710678118654752440

#endif // SRT_CONSTANTS_CUH
