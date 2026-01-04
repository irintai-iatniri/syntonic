// Syntonic CUDA Kernels - SRT Constants Header
// Fundamental constants from Syntonic Resonance Theory

#ifndef SRT_CONSTANTS_CUH
#define SRT_CONSTANTS_CUH

// =============================================================================
// Golden Ratio Constants (φ = (1 + √5) / 2)
// =============================================================================

// Double precision
#define PHI_F64         1.6180339887498948482
#define PHI_INV_F64     0.6180339887498948482   // 1/φ = φ - 1
#define PHI_SQ_F64      2.6180339887498948482   // φ² = φ + 1
#define PHI_INV_SQ_F64  0.3819660112501051518   // 1/φ² = 1 - 1/φ

// Single precision (fast intrinsics)
#define PHI_F32         1.6180339887f
#define PHI_INV_F32     0.6180339887f
#define PHI_SQ_F32      2.6180339887f
#define PHI_INV_SQ_F32  0.3819660112f

// =============================================================================
// Mathematical Constants
// =============================================================================

#define PI_F64          3.14159265358979323846
#define PI_F32          3.14159265f
#define TWO_PI_F64      6.28318530717958647693
#define TWO_PI_F32      6.28318530f
#define SQRT2_F64       1.41421356237309504880
#define SQRT2_F32       1.41421356f
#define SQRT5_F64       2.23606797749978969641
#define SQRT5_F32       2.23606797f

// =============================================================================
// SRT Structure Constants
// =============================================================================

// E₈ lattice dimensions
#define E8_DIM              8       // Dimension of E₈ root system
#define E8_ROOTS            240     // Number of E₈ roots
#define E8_ROOTS_POSITIVE   120     // Positive roots (chiral)

// E₆ lattice (Golden Cone)
#define E6_DIM              6       // Dimension of E₆
#define E6_POSITIVE_ROOTS   36      // |Φ⁺(E₆)| - Golden Cone count

// Winding space
#define WINDING_DIM         4       // (n₇, n₈, n₉, n₁₀)

// D₄ kissing number (consciousness threshold)
#define D4_KISSING          24

// =============================================================================
// SRT Correction Factor
// =============================================================================

// q = W(∞) - 1 ≈ 0.027395146920
#define Q_DEFICIT_F64       0.027395146920
#define Q_DEFICIT_F32       0.02739515f

// Structure dimensions for corrections N(structure)
#define N_E8_DIM            248     // dim(E₈)
#define N_E8_ROOTS          240     // |Φ(E₈)|
#define N_E8_ROOTS_POS      120     // |Φ⁺(E₈)|
#define N_E6_DIM            78      // dim(E₆)
#define N_E6_GOLDEN_CONE    36      // |Φ⁺(E₆)|
#define N_E6_27             27      // dim(27_E₆)
#define N_D4_KISSING        24      // K(D₄)
#define N_G2_DIM            14      // dim(G₂)

// =============================================================================
// Block Size Constants (SRT-optimized)
// =============================================================================

#define BLOCK_E8_ROOTS      240     // One per E₈ root
#define BLOCK_GOLDEN_CONE   36      // E₆ positive roots
#define BLOCK_WINDING_4D    256     // 4⁴ for winding space
#define BLOCK_GENERATIONS   192     // 64 × 3 generations
#define BLOCK_D4_KISSING    96      // 4 × 24 consciousness threshold
#define BLOCK_DEFAULT       256     // Default for general ops

// =============================================================================
// Projection Matrix Constants
// =============================================================================

// Normalization factor: 1/√(2φ+2) for golden projection P_φ
#define PROJ_NORM_F64       0.3717480344601846
#define PROJ_NORM_F32       0.37174803f

// φ × norm factor
#define PHI_NORM_F64        (PHI_F64 * PROJ_NORM_F64)
#define PHI_NORM_F32        (PHI_F32 * PROJ_NORM_F32)

// =============================================================================
// Device Intrinsic Helpers (inline functions)
// =============================================================================

// Fast reciprocal square root (single precision)
__device__ __forceinline__ float rsqrt_fast(float x) {
    return __frsqrt_rn(x);
}

// Fast exponential with φ⁻¹ scaling: exp(-x/φ)
__device__ __forceinline__ float exp_golden_inv(float x) {
    return __expf(-x * PHI_INV_F32);
}

// Fused multiply-add with φ: a * φ + b
__device__ __forceinline__ float fma_phi(float a, float b) {
    return __fmaf_rn(a, PHI_F32, b);
}

// Fused multiply-add with φ⁻¹: a * φ⁻¹ + b
__device__ __forceinline__ float fma_phi_inv(float a, float b) {
    return __fmaf_rn(a, PHI_INV_F32, b);
}

// Double precision fused phi operations
__device__ __forceinline__ double fma_phi_f64(double a, double b) {
    return fma(a, PHI_F64, b);
}

// =============================================================================
// Warp-Level Reduction Primitives
// =============================================================================

// Warp-level sum reduction (assumes warp size 32)
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

// Warp-level max reduction
__device__ __forceinline__ float warp_reduce_max(float val) {
    for (int offset = 16; offset > 0; offset >>= 1) {
        val = fmaxf(val, __shfl_down_sync(0xffffffff, val, offset));
    }
    return val;
}

#endif // SRT_CONSTANTS_CUH
