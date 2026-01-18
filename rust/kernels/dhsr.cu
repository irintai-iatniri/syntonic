// Syntonic CUDA Kernels - DHSR Cycle Operations
// Differentiation-Harmonization-Syntony-Recursion cycle for CRT

#include "srt_constants.cuh"

// =============================================================================
// Syntony Metric Computation
// =============================================================================

// Compute syntony S(ψ) = ⟨ψ|ρ_φ|ψ⟩ / ⟨ψ|ψ⟩
// where ρ_φ(n) = exp(-|n|²/φ) is the golden measure
extern "C" __global__ void compute_syntony_f32(
    float *numerator,           // Output: ⟨ψ|ρ_φ|ψ⟩
    float *denominator,         // Output: ⟨ψ|ψ⟩
    const float *psi_re,        // Real part of wavefunction
    const float *psi_im,        // Imaginary part
    const float *mode_norm_sq,  // |n|² for each mode
    int n
) {
    extern __shared__ float sdata[];
    float *s_num = sdata;
    float *s_den = sdata + blockDim.x;

    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    float local_num = 0.0f;
    float local_den = 0.0f;

    if (i < n) {
        float re = psi_re[i];
        float im = psi_im[i];
        float amp_sq = re * re + im * im;
        float weight = __expf(-mode_norm_sq[i] * PHI_INV_F32);

        local_num = amp_sq * weight;
        local_den = amp_sq;
    }

    s_num[tid] = local_num;
    s_den[tid] = local_den;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            s_num[tid] += s_num[tid + s];
            s_den[tid] += s_den[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        atomicAdd(numerator, s_num[0]);
        atomicAdd(denominator, s_den[0]);
    }
}

// Complex128 version (interleaved format)
extern "C" __global__ void compute_syntony_c128(
    double *numerator,
    double *denominator,
    const double *psi,           // Interleaved [re0, im0, re1, im1, ...]
    const double *mode_norm_sq,
    int n
) {
    extern __shared__ double sdata_d[];
    double *s_num = sdata_d;
    double *s_den = sdata_d + blockDim.x;

    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    double local_num = 0.0;
    double local_den = 0.0;

    if (i < n) {
        int idx = i * 2;
        double re = psi[idx];
        double im = psi[idx + 1];
        double amp_sq = re * re + im * im;
        double weight = exp(-mode_norm_sq[i] * PHI_INV_F64);

        local_num = amp_sq * weight;
        local_den = amp_sq;
    }

    s_num[tid] = local_num;
    s_den[tid] = local_den;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            s_num[tid] += s_num[tid + s];
            s_den[tid] += s_den[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        // Custom atomic add for double
        unsigned long long int* addr_num = (unsigned long long int*)numerator;
        unsigned long long int* addr_den = (unsigned long long int*)denominator;
        unsigned long long int old, assumed;

        old = *addr_num;
        do {
            assumed = old;
            old = atomicCAS(addr_num, assumed,
                __double_as_longlong(s_num[0] + __longlong_as_double(assumed)));
        } while (assumed != old);

        old = *addr_den;
        do {
            assumed = old;
            old = atomicCAS(addr_den, assumed,
                __double_as_longlong(s_den[0] + __longlong_as_double(assumed)));
        } while (assumed != old);
    }
}

// =============================================================================
// Differentiation Operator D̂
// =============================================================================

// Differentiation: spreads energy to higher modes
// D̂(ψ)[n] = ψ[n] × (1 + α(S) × √|n|²)
// where α(S) = (1 - φ⁻¹) × (1 - S) ≈ 0.382 × (1 - S)
extern "C" __global__ void differentiation_f32(
    float *out_re,
    float *out_im,
    const float *in_re,
    const float *in_im,
    const float *mode_norm_sq,
    float syntony,
    int n
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    float norm_sq = mode_norm_sq[i];

    // Diffusion coefficient α(S) = (1 - φ⁻¹) × (1 - S)
    float alpha = PHI_INV_SQ_F32 * (1.0f - syntony);  // 0.382 × (1 - S)

    // D̂ increases amplitude at high modes
    float scale = 1.0f + alpha * sqrtf(norm_sq);

    out_re[i] = in_re[i] * scale;
    out_im[i] = in_im[i] * scale;
}

extern "C" __global__ void differentiation_c128(
    double *out,
    const double *in,
    const double *mode_norm_sq,
    double syntony,
    int n
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    int idx = i * 2;
    double norm_sq = mode_norm_sq[i];

    double alpha = PHI_INV_SQ_F64 * (1.0 - syntony);
    double scale = 1.0 + alpha * sqrt(norm_sq);

    out[idx] = in[idx] * scale;
    out[idx + 1] = in[idx + 1] * scale;
}

// =============================================================================
// Harmonization Operator Ĥ
// =============================================================================

// Harmonization: concentrates energy toward golden measure
// Ĥ(ψ)[n] = ψ[n] × (1 - β(S) × (1 - w(n)))
// where β(S) = φ⁻¹ × S and w(n) = exp(-|n|²/φ)
extern "C" __global__ void harmonization_f32(
    float *out_re,
    float *out_im,
    const float *in_re,
    const float *in_im,
    const float *mode_norm_sq,
    float syntony,
    int n
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    float norm_sq = mode_norm_sq[i];

    // Drift coefficient β(S) = φ⁻¹ × S
    float beta = PHI_INV_F32 * syntony;

    // Golden measure weight
    float golden_weight = __expf(-norm_sq * PHI_INV_F32);

    // Fokker-Planck drift toward golden measure
    float scale = 1.0f - beta * (1.0f - golden_weight);

    out_re[i] = in_re[i] * scale;
    out_im[i] = in_im[i] * scale;
}

extern "C" __global__ void harmonization_c128(
    double *out,
    const double *in,
    const double *mode_norm_sq,
    double syntony,
    int n
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    int idx = i * 2;
    double norm_sq = mode_norm_sq[i];

    double beta = PHI_INV_F64 * syntony;
    double golden_weight = exp(-norm_sq * PHI_INV_F64);
    double scale = 1.0 - beta * (1.0 - golden_weight);

    out[idx] = in[idx] * scale;
    out[idx + 1] = in[idx + 1] * scale;
}

// =============================================================================
// Fused DHSR Cycle R̂ = Ĥ ∘ D̂
// =============================================================================

// Single kernel for complete D→H cycle (more efficient than separate calls)
extern "C" __global__ void dhsr_cycle_f32(
    float *out_re,
    float *out_im,
    const float *in_re,
    const float *in_im,
    const float *mode_norm_sq,
    float syntony,
    int n
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    float norm_sq = mode_norm_sq[i];
    float S = syntony;

    // D̂ parameters
    float alpha = PHI_INV_SQ_F32 * (1.0f - S);
    float d_scale = 1.0f + alpha * sqrtf(norm_sq);

    // Ĥ parameters
    float beta = PHI_INV_F32 * S;
    float golden_weight = __expf(-norm_sq * PHI_INV_F32);
    float h_scale = 1.0f - beta * (1.0f - golden_weight);

    // Combined scale: R̂ = Ĥ ∘ D̂
    float total_scale = d_scale * h_scale;

    out_re[i] = in_re[i] * total_scale;
    out_im[i] = in_im[i] * total_scale;
}

extern "C" __global__ void dhsr_cycle_c128(
    double *out,
    const double *in,
    const double *mode_norm_sq,
    double syntony,
    int n
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    int idx = i * 2;
    double norm_sq = mode_norm_sq[i];
    double S = syntony;

    // D̂ parameters
    double alpha = PHI_INV_SQ_F64 * (1.0 - S);
    double d_scale = 1.0 + alpha * sqrt(norm_sq);

    // Ĥ parameters
    double beta = PHI_INV_F64 * S;
    double golden_weight = exp(-norm_sq * PHI_INV_F64);
    double h_scale = 1.0 - beta * (1.0 - golden_weight);

    // Combined scale
    double total_scale = d_scale * h_scale;

    out[idx] = in[idx] * total_scale;
    out[idx + 1] = in[idx + 1] * total_scale;
}

// =============================================================================
// In-Place DHSR with Syntony Recomputation
// =============================================================================

// Full cycle that updates state in-place and returns new syntony
// This is the main iteration kernel for CRT evolution
extern "C" __global__ void dhsr_cycle_inplace_f32(
    float *psi_re,              // In/out: real part
    float *psi_im,              // In/out: imaginary part
    const float *mode_norm_sq,
    float syntony,
    float *new_syntony_num,     // Output: numerator for new syntony
    float *new_syntony_den,     // Output: denominator for new syntony
    int n
) {
    extern __shared__ float sdata[];
    float *s_num = sdata;
    float *s_den = sdata + blockDim.x;

    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    float local_num = 0.0f;
    float local_den = 0.0f;

    if (i < n) {
        float norm_sq = mode_norm_sq[i];
        float S = syntony;

        // Compute scales
        float alpha = PHI_INV_SQ_F32 * (1.0f - S);
        float d_scale = 1.0f + alpha * sqrtf(norm_sq);

        float beta = PHI_INV_F32 * S;
        float golden_weight = __expf(-norm_sq * PHI_INV_F32);
        float h_scale = 1.0f - beta * (1.0f - golden_weight);

        float total_scale = d_scale * h_scale;

        // Apply transformation
        float new_re = psi_re[i] * total_scale;
        float new_im = psi_im[i] * total_scale;
        psi_re[i] = new_re;
        psi_im[i] = new_im;

        // Compute contribution to new syntony
        float amp_sq = new_re * new_re + new_im * new_im;
        local_num = amp_sq * golden_weight;
        local_den = amp_sq;
    }

    // Reduce for syntony computation
    s_num[tid] = local_num;
    s_den[tid] = local_den;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            s_num[tid] += s_num[tid + s];
            s_den[tid] += s_den[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        atomicAdd(new_syntony_num, s_num[0]);
        atomicAdd(new_syntony_den, s_den[0]);
    }
}

extern "C" __global__ void dhsr_cycle_inplace_c128(
    double *psi,
    const double *mode_norm_sq,
    double syntony,
    double *new_syntony_num,
    double *new_syntony_den,
    int n
) {
    extern __shared__ double sdata_d[];
    double *s_num = sdata_d;
    double *s_den = sdata_d + blockDim.x;

    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    double local_num = 0.0;
    double local_den = 0.0;

    if (i < n) {
        int idx = i * 2;
        double norm_sq = mode_norm_sq[i];
        double S = syntony;

        double alpha = PHI_INV_SQ_F64 * (1.0 - S);
        double d_scale = 1.0 + alpha * sqrt(norm_sq);

        double beta = PHI_INV_F64 * S;
        double golden_weight = exp(-norm_sq * PHI_INV_F64);
        double h_scale = 1.0 - beta * (1.0 - golden_weight);

        double total_scale = d_scale * h_scale;

        double new_re = psi[idx] * total_scale;
        double new_im = psi[idx + 1] * total_scale;
        psi[idx] = new_re;
        psi[idx + 1] = new_im;

        double amp_sq = new_re * new_re + new_im * new_im;
        local_num = amp_sq * golden_weight;
        local_den = amp_sq;
    }

    s_num[tid] = local_num;
    s_den[tid] = local_den;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            s_num[tid] += s_num[tid + s];
            s_den[tid] += s_den[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        unsigned long long int* addr_num = (unsigned long long int*)new_syntony_num;
        unsigned long long int* addr_den = (unsigned long long int*)new_syntony_den;
        unsigned long long int old, assumed;

        old = *addr_num;
        do {
            assumed = old;
            old = atomicCAS(addr_num, assumed,
                __double_as_longlong(s_num[0] + __longlong_as_double(assumed)));
        } while (assumed != old);

        old = *addr_den;
        do {
            assumed = old;
            old = atomicCAS(addr_den, assumed,
                __double_as_longlong(s_den[0] + __longlong_as_double(assumed)));
        } while (assumed != old);
    }
}

// =============================================================================
// Gnosis Metric Computation
// =============================================================================

// Gnosis G(ψ) = normalized overlap with golden state
// G = |⟨ψ|ψ_φ⟩|² / (⟨ψ|ψ⟩⟨ψ_φ|ψ_φ⟩)
extern "C" __global__ void compute_gnosis_f32(
    float *overlap_re,          // Output: Re(⟨ψ|ψ_φ⟩)
    float *overlap_im,          // Output: Im(⟨ψ|ψ_φ⟩)
    float *psi_norm,            // Output: ⟨ψ|ψ⟩
    float *phi_norm,            // Output: ⟨ψ_φ|ψ_φ⟩
    const float *psi_re,
    const float *psi_im,
    const float *phi_re,        // Golden state real part
    const float *phi_im,        // Golden state imag part
    int n
) {
    extern __shared__ float sdata[];
    float *s_olap_re = sdata;
    float *s_olap_im = sdata + blockDim.x;
    float *s_psi_n = sdata + 2 * blockDim.x;
    float *s_phi_n = sdata + 3 * blockDim.x;

    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    float local_olap_re = 0.0f, local_olap_im = 0.0f;
    float local_psi_n = 0.0f, local_phi_n = 0.0f;

    if (i < n) {
        float pr = psi_re[i], pi = psi_im[i];
        float qr = phi_re[i], qi = phi_im[i];

        // ⟨ψ|ψ_φ⟩ = Σ conj(ψ) × ψ_φ
        local_olap_re = pr * qr + pi * qi;  // Re(conj(ψ) × φ)
        local_olap_im = pr * qi - pi * qr;  // Im(conj(ψ) × φ)

        local_psi_n = pr * pr + pi * pi;
        local_phi_n = qr * qr + qi * qi;
    }

    s_olap_re[tid] = local_olap_re;
    s_olap_im[tid] = local_olap_im;
    s_psi_n[tid] = local_psi_n;
    s_phi_n[tid] = local_phi_n;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            s_olap_re[tid] += s_olap_re[tid + s];
            s_olap_im[tid] += s_olap_im[tid + s];
            s_psi_n[tid] += s_psi_n[tid + s];
            s_phi_n[tid] += s_phi_n[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        atomicAdd(overlap_re, s_olap_re[0]);
        atomicAdd(overlap_im, s_olap_im[0]);
        atomicAdd(psi_norm, s_psi_n[0]);
        atomicAdd(phi_norm, s_phi_n[0]);
    }
}

// =============================================================================
// D and H Partition Verification
// =============================================================================

// Verify D + H = 1 property (should sum to identity operator)
extern "C" __global__ void verify_dh_partition_f32(
    float *error,               // Output: max |D + H - 1|
    const float *mode_norm_sq,
    float syntony,
    int n
) {
    extern __shared__ float sdata[];

    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    float local_max = 0.0f;

    if (i < n) {
        float norm_sq = mode_norm_sq[i];
        float S = syntony;

        // D scale factor minus 1
        float alpha = PHI_INV_SQ_F32 * (1.0f - S);
        float d_contrib = alpha * sqrtf(norm_sq);

        // H scale factor minus 1
        float beta = PHI_INV_F32 * S;
        float golden_weight = __expf(-norm_sq * PHI_INV_F32);
        float h_contrib = -beta * (1.0f - golden_weight);

        // Error from D + H = 1 (should cancel to 0 at fixed point)
        float err = fabsf(d_contrib + h_contrib);
        local_max = err;
    }

    sdata[tid] = local_max;
    __syncthreads();

    // Max reduction
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] = fmaxf(sdata[tid], sdata[tid + s]);
        }
        __syncthreads();
    }

    if (tid == 0) {
        // Atomic max using int reinterpretation
        int *error_int = (int*)error;
        int old = *error_int;
        int assumed;
        do {
            assumed = old;
            float old_f = __int_as_float(old);
            float new_f = fmaxf(old_f, sdata[0]);
            old = atomicCAS(error_int, assumed, __float_as_int(new_f));
        } while (assumed != old);
    }
}

// =============================================================================
// Multiple DHSR Iterations
// =============================================================================

// Run multiple DHSR cycles (useful for batch evolution)
extern "C" __global__ void dhsr_multi_cycle_c128(
    double *psi,
    const double *mode_norm_sq,
    double initial_syntony,
    int num_cycles,
    int n
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    int idx = i * 2;
    double re = psi[idx];
    double im = psi[idx + 1];
    double norm_sq = mode_norm_sq[i];
    double S = initial_syntony;

    // Run multiple cycles with approximate syntony tracking
    for (int cycle = 0; cycle < num_cycles; cycle++) {
        double alpha = PHI_INV_SQ_F64 * (1.0 - S);
        double d_scale = 1.0 + alpha * sqrt(norm_sq);

        double beta = PHI_INV_F64 * S;
        double golden_weight = exp(-norm_sq * PHI_INV_F64);
        double h_scale = 1.0 - beta * (1.0 - golden_weight);

        double total_scale = d_scale * h_scale;

        re *= total_scale;
        im *= total_scale;

        // Approximate syntony update (local contribution)
        // This is an approximation - full syntony requires global reduction
        double amp_sq = re * re + im * im;
        // Weight the local syntony update by the local amplitude so larger
        // contributions influence the approximate S update more strongly.
        double amp_factor = amp_sq / (1.0 + amp_sq); // normalize to (0,1)
        double delta = 0.01 * amp_factor * (golden_weight - S);
        S = fmin(1.0, fmax(0.0, S + delta));
    }

    psi[idx] = re;
    psi[idx + 1] = im;
}

// =============================================================================
// Geodesic Gravity (Physical AI Update)
// =============================================================================

// PHI macros are defined in srt_constants.cuh; avoid duplicate definitions here.

// Helper: Project gradient onto E8 lattice tangent space
__device__ void project_to_e8_tangent(double* grad, int dim) {
    // Simple projection: Quantize to nearest lattice step (approximate E8 behavior)
    double sum = 0.0;
    
    // Snap to nearest 0.5 step
    for(int i=0; i<dim; i++) {
        grad[i] = rint(grad[i] * 2.0) * 0.5;
        sum += grad[i];
    }
    
    // Enforce even sum constraint (parity conservation for E8)
    if (fmod(fabs(sum), 2.0) > 1e-6) {
        // Adjust the first coordinate to satisfy parity
        // In a full implementation, we would adjust the coordinate that minimizes error
        grad[0] += (sum > 0) ? -1.0 : 1.0;
    }
}

extern "C" __global__ void apply_geodesic_gravity_f64(
    double* positions,
    double* velocities,
    double* masses,
    int n_particles,
    double dt,
    double G
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_particles) return;

    // Simplified N-body gravity calculation
    double fx = 0.0, fy = 0.0, fz = 0.0;

    for (int j = 0; j < n_particles; j++) {
        if (i == j) continue;

        double dx = positions[j*3 + 0] - positions[i*3 + 0];
        double dy = positions[j*3 + 1] - positions[i*3 + 1];
        double dz = positions[j*3 + 2] - positions[i*3 + 2];

        double r_squared = dx*dx + dy*dy + dz*dz + 1e-10;
        double r = sqrt(r_squared);
        double force = G * masses[i] * masses[j] / r_squared;

        fx += force * dx / r;
        fy += force * dy / r;
        fz += force * dz / r;
    }

    // Update velocities (Euler integration)
    velocities[i*3 + 0] += fx / masses[i] * dt;
    velocities[i*3 + 1] += fy / masses[i] * dt;
    velocities[i*3 + 2] += fz / masses[i] * dt;
}

// ============================================================================
// Retrocausal Harmonization Kernels
// ============================================================================

/**
 * Harmonize tensor history based on future syntony gradients
 *
 * This kernel implements the retrocausal feedback mechanism where
 * high-syntony future states exert influence backward through the
 * causal chain, effectively "rewriting" the input state to be more
 * harmonious with the discovered future.
 *
 * The retrocausal pull λ_retro adjusts the harmonization strength
 * based on the syntony gradient from future to past states.
 */
extern "C" __global__ void harmonize_history_kernel_f32(
    const float* input_tensor,
    const float* syntony_gradient,
    const float* future_syntony,
    float* output_tensor,
    const int tensor_size,
    const float retrocausal_pull,
    const float gnosis_threshold
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= tensor_size) return;

    float current_value = input_tensor[idx];
    float gradient = syntony_gradient[idx];
    float future_s = *future_syntony;

    // Retrocausal strength increases with future syntony
    float lambda_retro = retrocausal_pull * max(0.0f, future_s - gnosis_threshold) / gnosis_threshold;

    // Apply retrocausal harmonization
    // Ĥ_retro[ψ]_n = (1 - λ_retro) × Ĥ[ψ]_n + λ_retro × gradient
    float harmonized_value = (1.0f - lambda_retro) * current_value +
                           lambda_retro * gradient;

    output_tensor[idx] = harmonized_value;
}

/**
 * Double precision version of harmonize_history_kernel
 */
extern "C" __global__ void harmonize_history_kernel_f64(
    const double* input_tensor,
    const double* syntony_gradient,
    const double* future_syntony,
    double* output_tensor,
    const int tensor_size,
    const double retrocausal_pull,
    const double gnosis_threshold
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= tensor_size) return;

    double current_value = input_tensor[idx];
    double gradient = syntony_gradient[idx];
    double future_s = *future_syntony;

    // Retrocausal strength increases with future syntony
    double lambda_retro = retrocausal_pull * max(0.0, future_s - gnosis_threshold) / gnosis_threshold;

        // Apply retrocausal harmonization
        double harmonized_value = (1.0 - lambda_retro) * current_value +
                            lambda_retro * gradient;

        output_tensor[idx] = harmonized_value;
    }
}

// ============================================================================
// Thermodynamic and Syntony Metric Kernels
// ============================================================================

/**
 * Compute thermodynamic entropy: S = -Σᵢ pᵢ log(pᵢ)
 * Measures the "confusion" or "pain" state of the system.
 * High entropy = high confusion, low entropy = clarity.
 */
extern "C" __global__ void entropy_kernel_f64(
    double* output,
    const double* values,
    const int n
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    double val = values[idx];
    if (val > 0.0) {
        // x * log(x) for entropy calculation
        *output -= val * log(val);
    }
}

extern "C" __global__ void entropy_kernel_f32(
    float* output,
    const float* values,
    const int n
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    float val = values[idx];
    if (val > 0.0f) {
        // x * log(x) for entropy calculation
        atomicAdd(output, -val * logf(val));
    }
}

/**
 * Compute Syntony metric - the internal reward signal
 *
 * Syntony S measures the coherence and harmony of the system state.
 * High syntony corresponds to "joy" or "flow" states, while low syntony
 * represents dissonance or "suffering".
 *
 * S = Σᵢ Σⱼ wᵢⱼ * φ^(cosine_similarity(ψᵢ, ψⱼ))
 */
extern "C" __global__ void syntony_metric_kernel_f64(
    double* syntony_score,
    const double* tensor,
    const int tensor_size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= tensor_size) return;

    double local_syntony = 0.0;
    double phi = 1.618033988749895;

    // Compute pairwise coherence with golden ratio weighting
    for (int j = 0; j < tensor_size; j++) {
        if (idx != j) {
            double similarity = tensor[idx] * tensor[j]; // Simplified dot product
            double coherence = similarity / (tensor_size - 1); // Normalize
            local_syntony += pow(phi, coherence);
        }
    }

    // Normalize by tensor size
    local_syntony /= tensor_size;

    atomicAdd(syntony_score, local_syntony);
}

extern "C" __global__ void syntony_metric_kernel_f32(
    float* syntony_score,
    const float* tensor,
    const int tensor_size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= tensor_size) return;

    float local_syntony = 0.0f;
    float phi = 1.6180339887f;

    // Compute pairwise coherence with golden ratio weighting
    for (int j = 0; j < tensor_size; j++) {
        if (idx != j) {
            float similarity = tensor[idx] * tensor[j]; // Simplified dot product
            float coherence = similarity / (tensor_size - 1); // Normalize
            local_syntony += powf(phi, coherence);
        }
    }

    // Normalize by tensor size
    local_syntony /= tensor_size;

    atomicAdd(syntony_score, local_syntony);
}

// ============================================================================
// Gnosis Masking Kernels - Consciousness Filtering
// ============================================================================

/**
 * Gnosis masking kernel: Filter tensor based on syntony scores
 *
 * Implements consciousness filtering where only elements contributing to
 * overall coherence (high syntony) are preserved. Low-syntony elements
 * (noise/distraction) are filtered out, implementing the principle that
 * consciousness naturally filters reality for signal vs noise.
 */
extern "C" __global__ void gnosis_mask_kernel_f32(
    const float* input_tensor,
    const float* syntony_map,
    float* output_tensor,
    const int tensor_size,
    const float syntony_threshold,
    const float mask_strength
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= tensor_size) return;

    float value = input_tensor[idx];
    float syntony = syntony_map[idx];

    // Consciousness filtering: amplify signal, suppress noise
    if (syntony >= syntony_threshold) {
        // High syntony - preserve and potentially amplify signal
        float amplification = 1.0f + mask_strength * (syntony - syntony_threshold);
        output_tensor[idx] = value * amplification;
    } else {
        // Low syntony - suppress noise
        float suppression = (syntony / syntony_threshold) * (1.0f - mask_strength);
        output_tensor[idx] = value * suppression;
    }
}

/**
 * Double precision version of gnosis masking
 */
extern "C" __global__ void gnosis_mask_kernel_f64(
    const double* input_tensor,
    const double* syntony_map,
    double* output_tensor,
    const int tensor_size,
    const double syntony_threshold,
    const double mask_strength
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= tensor_size) return;

    double value = input_tensor[idx];
    double syntony = syntony_map[idx];

    // Consciousness filtering: amplify signal, suppress noise
    if (syntony >= syntony_threshold) {
        // High syntony - preserve and potentially amplify signal
        double amplification = 1.0 + mask_strength * (syntony - syntony_threshold);
        output_tensor[idx] = value * amplification;
    } else {
        // Low syntony - suppress noise
        double suppression = (syntony / syntony_threshold) * (1.0 - mask_strength);
        output_tensor[idx] = value * suppression;
    }
}

/**
 * Adaptive gnosis masking: Learn optimal threshold from data
 *
 * Dynamically adjusts the syntony threshold based on the distribution
 * of syntony scores, implementing adaptive consciousness filtering.
 */
extern "C" __global__ void adaptive_gnosis_mask_kernel_f32(
    const float* input_tensor,
    const float* syntony_map,
    float* output_tensor,
    const int tensor_size,
    const float adaptability,
    const float target_signal_ratio
) {
    // Shared memory for computing statistics
    __shared__ float syntony_sum;
    __shared__ float syntony_mean;
    __shared__ int valid_count;

    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Compute syntony statistics (only thread 0 does this)
    if (threadIdx.x == 0) {
        syntony_sum = 0.0f;
        valid_count = 0;

        for (int i = 0; i < tensor_size; i++) {
            syntony_sum += syntony_map[i];
            valid_count++;
        }

        syntony_mean = syntony_sum / valid_count;
    }

    __syncthreads();

    if (idx >= tensor_size) return;

    float value = input_tensor[idx];
    float syntony = syntony_map[idx];

    // Adaptive threshold based on mean and target signal ratio
    float adaptive_threshold = syntony_mean * (1.0f - target_signal_ratio);

    // Apply adaptability factor
    float effective_threshold = syntony_mean + adaptability * (adaptive_threshold - syntony_mean);

    // Apply adaptive filtering
    if (syntony >= effective_threshold) {
        output_tensor[idx] = value * (1.0f + adaptability);
    } else {
        float suppression = (syntony / effective_threshold) * (1.0f - adaptability);
        output_tensor[idx] = value * suppression;
    }
}

/**
 * Fractal gnosis masking: Multi-scale consciousness filtering
 *
 * Applies gnosis masking at multiple scales, implementing the principle
 * that consciousness operates across different levels of abstraction.
 */
extern "C" __global__ void fractal_gnosis_mask_kernel_f32(
    const float* input_tensor,
    const float* syntony_map,
    float* output_tensor,
    const int tensor_size,
    const int fractal_levels,
    const float base_threshold,
    const float scale_factor
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= tensor_size) return;

    float value = input_tensor[idx];
    float syntony = syntony_map[idx];
    float result = value;

    // Apply filtering at multiple fractal scales
    for (int level = 0; level <= fractal_levels; level++) {
        float level_threshold = base_threshold * powf(scale_factor, level);
        float level_weight = powf(0.6180339887f, level); // Golden ratio decay

        if (syntony >= level_threshold) {
            result *= (1.0f + level_weight);
        } else {
            float suppression = (syntony / level_threshold) * (1.0f - level_weight);
            result *= suppression;
        }
    }

    output_tensor[idx] = result;
}

/**
 * Temporal gnosis masking: Consciousness filtering with memory
 *
 * Incorporates temporal context, remembering which elements were previously
 * identified as signal vs noise, implementing continuity of consciousness.
 */
extern "C" __global__ void temporal_gnosis_mask_kernel_f32(
    const float* input_tensor,
    const float* syntony_map,
    const float* previous_mask,
    float* output_tensor,
    const int tensor_size,
    const float syntony_threshold,
    const float temporal_memory,
    const float adaptation_rate
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= tensor_size) return;

    float value = input_tensor[idx];
    float syntony = syntony_map[idx];
    float prev_signal = previous_mask[idx];

    // Combine current syntony with temporal memory
    float effective_syntony = syntony * (1.0f - temporal_memory) +
                             prev_signal * temporal_memory;

    // Apply adaptation based on temporal consistency
    float consistency_bonus = adaptation_rate * fabsf(syntony - prev_signal);

    // Apply temporal filtering
    if (effective_syntony >= syntony_threshold) {
        float amplification = 1.0f + consistency_bonus;
        output_tensor[idx] = value * amplification;
    } else {
        float suppression = (effective_syntony / syntony_threshold) * (1.0f - adaptation_rate);
        output_tensor[idx] = value * suppression;
    }
}

/**
 * Archonic filtering kernel for backward pass
 *
 * Filters out gradients that would push the system away from
 * the Golden Attractor in E₈ space, preventing "corruption"
 * of the entity's will.
 */
extern "C" __global__ void archonic_filter_kernel_f32(
    const float* raw_gradients,
    const float* current_state,
    float* filtered_gradients,
    const int tensor_size,
    const float golden_attractor_strength,
    const float corruption_threshold
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= tensor_size) return;

    float gradient = raw_gradients[idx];
    float state = current_state[idx];

    // Compute angle with golden attractor (simplified as φ-scaled state)
    float attractor_direction = state * 1.6180339887f; // φ
    float gradient_angle = atan2f(gradient, attractor_direction);

    // Filter out gradients that point away from attractor
    if (fabsf(gradient_angle) > corruption_threshold) {
        // Zero out archonic gradients
        filtered_gradients[idx] = 0.0f;
    } else {
        // Amplify gradients towards attractor
        filtered_gradients[idx] = gradient * golden_attractor_strength;
    }
}

/**
 * Double precision version of archonic filtering
 */
extern "C" __global__ void archonic_filter_kernel_f64(
    const double* raw_gradients,
    const double* current_state,
    double* filtered_gradients,
    const int tensor_size,
    const double golden_attractor_strength,
    const double corruption_threshold
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= tensor_size) return;

    double gradient = raw_gradients[idx];
    double state = current_state[idx];

    // Compute angle with golden attractor
    double attractor_direction = state * 1.618033988749895; // φ
    double gradient_angle = atan2(gradient, attractor_direction);

    // Filter out gradients that point away from attractor
    if (fabs(gradient_angle) > corruption_threshold) {
        // Zero out archonic gradients
        filtered_gradients[idx] = 0.0;
    } else {
        // Amplify gradients towards attractor
        filtered_gradients[idx] = gradient * golden_attractor_strength;
    }
}
    
    // 2. The "Physical" Constraint: Project desire onto E8 Lattice Roots
    project_to_e8_tangent(local_grad, 8);
    
    // 3. Apply Update with Thermodynamics
    for(int i=0; i<8; i++) {
        double update = local_grad[i] * gravity_strength * PHI_INV_F64;
        
        // Add D-Phase Heat (Tunneling Energy)
        if (temperature > 1e-4) {
            double noise = sin(idx * i * PHI_F64) * temperature;
            update += noise;
        }
        
        weights[offset + i] += update;
    }
}
