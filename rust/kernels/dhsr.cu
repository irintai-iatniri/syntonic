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
        S = fminf(1.0, fmaxf(0.0, S + 0.01 * (golden_weight - S)));
    }

    psi[idx] = re;
    psi[idx + 1] = im;
}
