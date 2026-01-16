//! Hierarchy correction kernels for SRT-Zero
//!
//! This module provides GPU-accelerated operations for:
//! - Batched correction application: value * (1 ± q/divisor)
//! - Special corrections (q²/φ, q·φ, 4q, etc.)
//! - Suppression factors (1/(1+q/φ), 1/(1+q·φ), etc.)
//! - Nested correction chains
//!
//! Usage:
//!   Apply 1000+ particle mass corrections in parallel on GPU

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cmath>

// SRT constants (must match Python hierarchy.py)
#define PHI            1.6180339887498948482
#define PHI_INV        0.6180339887498948482
#define PHI_SQUARED    2.6180339887498948482
#define PHI_CUBED      4.2360679774997896964
#define PHI_FOURTH     6.8541019662496845446
#define PHI_FIFTH     11.0901699437494742410
#define PI             3.14159265358979323846
#define E              2.71828182845904523536
#define E_STAR         19.999099979189476
#define Q              0.027395146920071658

// =============================================================================
// STANDARD CORRECTION: value * (1 ± q/divisor)
// =============================================================================

__global__ void apply_correction_f64(
    const double* __restrict__ values,
    const double* __restrict__ divisors,
    const int* __restrict__ signs,
    double* __restrict__ outputs,
    int n
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    double value = values[idx];
    double divisor = divisors[idx];
    int sign = signs[idx];

    if (divisor == 0.0) {
        outputs[idx] = value;
        return;
    }

    double factor = 1.0 + sign * Q / divisor;
    outputs[idx] = value * factor;
}

__global__ void apply_correction_f32(
    const float* __restrict__ values,
    const float* __restrict__ divisors,
    const int* __restrict__ signs,
    float* __restrict__ outputs,
    int n
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    float value = values[idx];
    float divisor = divisors[idx];
    int sign = signs[idx];

    if (divisor == 0.0f) {
        outputs[idx] = value;
        return;
    }

    float factor = 1.0f + sign * ((float)Q) / divisor;
    outputs[idx] = value * factor;
}

// Single-divisor version for batch processing
__global__ void apply_correction_uniform_f64(
    const double* __restrict__ values,
    double divisor,
    int sign,
    double* __restrict__ outputs,
    int n
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    double value = values[idx];
    double factor = 1.0 + sign * Q / divisor;
    outputs[idx] = value * factor;
}

// =============================================================================
// SPECIAL CORRECTIONS: q²/φ, q·φ, 4q, etc.
// =============================================================================

__global__ void apply_special_correction_f64(
    const double* __restrict__ values,
    const int* __restrict__ types,
    double* __restrict__ outputs,
    int n
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    double value = values[idx];
    int type = types[idx];
    double factor = 1.0;

    // Standard q*phi^n
    if (type == 0) {          // q_phi_plus
        factor = 1.0 + Q * PHI;
    } else if (type == 1) {   // q_phi_minus
        factor = 1.0 - Q * PHI;
    } else if (type == 2) {   // q_phi_squared_plus
        factor = 1.0 + Q * PHI_SQUARED;
    } else if (type == 3) {   // q_phi_squared_minus
        factor = 1.0 - Q * PHI_SQUARED;
    } else if (type == 4) {   // q_phi_cubed_plus
        factor = 1.0 + Q * PHI_CUBED;
    } else if (type == 5) {   // q_phi_cubed_minus
        factor = 1.0 - Q * PHI_CUBED;
    } else if (type == 6) {   // q_phi_fourth_plus
        factor = 1.0 + Q * PHI_FOURTH;
    } else if (type == 7) {   // q_phi_fourth_minus
        factor = 1.0 - Q * PHI_FOURTH;
    } else if (type == 8) {   // q_phi_fifth_plus
        factor = 1.0 + Q * PHI_FIFTH;
    } else if (type == 9) {   // q_phi_fifth_minus
        factor = 1.0 - Q * PHI_FIFTH;

    // q^2 terms
    } else if (type == 10) {  // q_squared_plus
        factor = 1.0 + Q * Q;
    } else if (type == 11) {  // q_squared_minus
        factor = 1.0 - Q * Q;
    } else if (type == 12) {  // q_squared_phi_plus
        factor = 1.0 + Q * Q / PHI;
    } else if (type == 13) {  // q_squared_phi_minus
        factor = 1.0 - Q * Q / PHI;
    } else if (type == 14) {  // q_sq_phi_sq_plus
        factor = 1.0 + Q * Q / PHI_SQUARED;
    } else if (type == 15) {  // q_sq_phi_sq_minus
        factor = 1.0 - Q * Q / PHI_SQUARED;
    } else if (type == 16) {  // q_sq_phi_plus
        factor = 1.0 + Q * Q * PHI;

    // Multiples of q
    } else if (type == 17) {  // 4q_plus
        factor = 1.0 + 4.0 * Q;
    } else if (type == 18) {  // 4q_minus
        factor = 1.0 - 4.0 * Q;
    } else if (type == 19) {  // 3q_plus
        factor = 1.0 + 3.0 * Q;
    } else if (type == 20) {  // 3q_minus
        factor = 1.0 - 3.0 * Q;
    } else if (type == 21) {  // 6q_plus
        factor = 1.0 + 6.0 * Q;
    } else if (type == 22) {  // 8q_plus
        factor = 1.0 + 8.0 * Q;
    } else if (type == 23) {  // pi_q_plus
        factor = 1.0 + PI * Q;

    // Special cases
    } else if (type == 24) {  // q_cubed
        factor = 1.0 + Q * Q * Q;
    } else if (type == 25) {  // q_phi_div_4pi_plus
        factor = 1.0 + Q * PHI / (4.0 * PI);
    } else if (type == 26) {  // 8q_inv_plus (q/8)
        factor = 1.0 + Q / 8.0;
    } else if (type == 27) {  // q_squared_half_plus
        factor = 1.0 + Q * Q / 2.0;
    } else if (type == 28) {  // q_6pi_plus (q/(6π))
        factor = 1.0 + Q / (6.0 * PI);
    } else if (type == 29) {  // q_phi_inv_plus (q/φ)
        factor = 1.0 + Q / PHI;
    }

    outputs[idx] = value * factor;
}

// =============================================================================
// SUPPRESSION FACTORS
// =============================================================================

__global__ void apply_winding_instability_f64(
    const double* __restrict__ values,
    double* __restrict__ outputs,
    int n
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    double value = values[idx];
    double factor = 1.0 / (1.0 + Q / PHI);
    outputs[idx] = value * factor;
}

__global__ void apply_recursion_penalty_f64(
    const double* __restrict__ values,
    double* __restrict__ outputs,
    int n
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    double value = values[idx];
    double factor = 1.0 / (1.0 + Q * PHI);
    outputs[idx] = value * factor;
}

__global__ void apply_double_inverse_f64(
    const double* __restrict__ values,
    double* __restrict__ outputs,
    int n
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    double value = values[idx];
    double factor = 1.0 / (1.0 + Q / PHI_SQUARED);
    outputs[idx] = value * factor;
}

__global__ void apply_fixed_point_penalty_f64(
    const double* __restrict__ values,
    double* __restrict__ outputs,
    int n
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    double value = values[idx];
    double factor = 1.0 / (1.0 + Q * PHI_SQUARED);
    outputs[idx] = value * factor;
}

// =============================================================================
// NESTED CORRECTION CHAINS
// =============================================================================

__global__ void apply_correction_chain_f64(
    double* __restrict__ values,
    const double* __restrict__ divisors,
    const int* __restrict__ signs,
    const int* __restrict__ chain_lengths,
    const int* __restrict__ chain_starts,
    double* __restrict__ outputs,
    int n_values,
    int n_corrections
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_values) return;

    double value = values[idx];
    int chain_len = chain_lengths[idx];
    int chain_start = chain_starts[idx];

    // Apply corrections in sequence
    for (int i = 0; i < chain_len; i++) {
        int corr_idx = chain_start + i;
        if (corr_idx >= n_corrections) break;

        double divisor = divisors[corr_idx];
        int sign = signs[corr_idx];

        if (divisor != 0.0) {
            double factor = 1.0 + sign * Q / divisor;
            value = value * factor;
        }
    }

    outputs[idx] = value;
}

// =============================================================================
// BATCH E_STAR_N COMPUTATION
// =============================================================================

__global__ void compute_e_star_n_f64(
    const double* __restrict__ N,
    const double* __restrict__ divisors,
    const int* __restrict__ signs,
    int n_corrections_per_value,
    double* __restrict__ outputs,
    int n
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    double N_val = N[idx];
    double value = E_STAR * N_val;

    // Apply corrections
    for (int i = 0; i < n_corrections_per_value; i++) {
        int corr_idx = idx * n_corrections_per_value + i;
        double divisor = divisors[corr_idx];
        int sign = signs[corr_idx];

        if (divisor != 0.0) {
            double factor = 1.0 + sign * Q / divisor;
            value = value * factor;
        }
    }

    outputs[idx] = value;
}

// =============================================================================
// GEOMETRIC DIVISORS LOOKUP
// =============================================================================

// Pre-computed divisors from hierarchy.py
__constant__ double GEOMETRIC_DIVISORS[84];

// Initialize divisors constant memory (called from host)
extern "C" cudaError_t init_geometric_divisors(
    const double* host_divisors,
    int count
) {
    return cudaMemcpyToSymbol(GEOMETRIC_DIVISORS, host_divisors,
                           sizeof(double) * count, 0,
                           cudaMemcpyHostToDevice);
}

__global__ void apply_correction_by_name_f64(
    const double* __restrict__ values,
    const int* __restrict__ divisor_indices,
    const int* __restrict__ signs,
    double* __restrict__ outputs,
    int n
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    double value = values[idx];
    int div_idx = divisor_indices[idx];
    int sign = signs[idx];

    if (div_idx < 0 || div_idx >= 84) {
        outputs[idx] = value;
        return;
    }

    double divisor = GEOMETRIC_DIVISORS[div_idx];
    double factor = 1.0 + sign * Q / divisor;
    outputs[idx] = value * factor;
}

// =============================================================================
// HELPER: COMPUTE PHI POWERS
// =============================================================================

__global__ void compute_phi_powers_f64(
    double* __restrict__ outputs,
    int max_power,
    int n
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    int power = idx % (max_power + 1);
    double phi_power = 1.0;

    for (int i = 0; i < power; i++) {
        phi_power *= PHI;
    }

    outputs[idx] = phi_power;
}

// =============================================================================
// HOST INTERFACE
// =============================================================================

extern "C" {

// Apply correction with single divisor for batch
cudaError_t apply_correction_batch(
    const double* d_values,
    double divisor,
    int sign,
    double* d_outputs,
    int n,
    cudaStream_t stream = 0
) {
    int threads = 256;
    int blocks = (n + threads - 1) / threads;

    apply_correction_uniform_f64<<<blocks, threads, 0, stream>>>(
        d_values, divisor, sign, d_outputs, n
    );

    return cudaGetLastError();
}

// Apply suppression factors
cudaError_t apply_suppression(
    const double* d_values,
    int suppression_type,
    double* d_outputs,
    int n,
    cudaStream_t stream = 0
) {
    int threads = 256;
    int blocks = (n + threads - 1) / threads;

    switch (suppression_type) {
        case 0: // winding_instability
            apply_winding_instability_f64<<<blocks, threads, 0, stream>>>(
                d_values, d_outputs, n
            );
            break;
        case 1: // recursion_penalty
            apply_recursion_penalty_f64<<<blocks, threads, 0, stream>>>(
                d_values, d_outputs, n
            );
            break;
        case 2: // double_inverse
            apply_double_inverse_f64<<<blocks, threads, 0, stream>>>(
                d_values, d_outputs, n
            );
            break;
        case 3: // fixed_point_penalty
            apply_fixed_point_penalty_f64<<<blocks, threads, 0, stream>>>(
                d_values, d_outputs, n
            );
            break;
    }

    return cudaGetLastError();
}

// Batch E*×N computation with corrections
cudaError_t compute_e_star_n_batch(
    const double* d_N,
    const double* d_divisors,
    const int* d_signs,
    int n_corrections_per_value,
    double* d_outputs,
    int n,
    cudaStream_t stream = 0
) {
    int threads = 256;
    int blocks = (n + threads - 1) / threads;

    compute_e_star_n_f64<<<blocks, threads, 0, stream>>>(
        d_N, d_divisors, d_signs, n_corrections_per_value, d_outputs, n
    );

    return cudaGetLastError();
}

} // extern "C"
