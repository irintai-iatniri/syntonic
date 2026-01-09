// Syntonic CUDA Kernels - Syntony-Weighted Softmax
// Softmax with golden measure weighting w(n) = exp(-|n|²/φ)

#include "srt_constants.cuh"
#include <float.h>

// =============================================================================
// Syntonic Softmax: learned mode (exp(-mode_norms/φ) weighting)
// =============================================================================

extern "C" __global__ void syntonic_softmax_learned_f64(
    double *out,                  // Output probabilities [batch, num_classes]
    const double *logits,         // Input logits [batch, num_classes]
    const double *mode_norms,     // Mode norms |n|² [num_classes]
    double syntony_scale,
    int batch_size,
    int num_classes
) {
    extern __shared__ double shared[];
    double *s_max = shared;
    double *s_sum = shared + blockDim.x;

    int batch_idx = blockIdx.x;
    int tid = threadIdx.x;

    if (batch_idx >= batch_size) return;

    const double *x = logits + batch_idx * num_classes;
    double *y = out + batch_idx * num_classes;

    // Step 1: Find max for numerical stability
    double local_max = -DBL_MAX;
    for (int i = tid; i < num_classes; i += blockDim.x) {
        double log_weight = -mode_norms[i] * PHI_INV_F64 * syntony_scale;
        double weighted_logit = x[i] + log_weight;
        local_max = fmax(local_max, weighted_logit);
    }

    s_max[tid] = local_max;
    __syncthreads();

    // Block-level max reduction
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            s_max[tid] = fmax(s_max[tid], s_max[tid + s]);
        }
        __syncthreads();
    }

    double max_val = s_max[0];
    __syncthreads();

    // Step 2: Compute exp(x - max) and sum
    double local_sum = 0.0;
    for (int i = tid; i < num_classes; i += blockDim.x) {
        double log_weight = -mode_norms[i] * PHI_INV_F64 * syntony_scale;
        double weighted_logit = x[i] + log_weight;
        local_sum += exp(weighted_logit - max_val);
    }

    s_sum[tid] = local_sum;
    __syncthreads();

    // Block-level sum reduction
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            s_sum[tid] += s_sum[tid + s];
        }
        __syncthreads();
    }

    double sum = s_sum[0];
    __syncthreads();

    // Step 3: Normalize
    for (int i = tid; i < num_classes; i += blockDim.x) {
        double log_weight = -mode_norms[i] * PHI_INV_F64 * syntony_scale;
        double weighted_logit = x[i] + log_weight;
        y[i] = exp(weighted_logit - max_val) / sum;
    }
}

extern "C" __global__ void syntonic_softmax_learned_f32(
    float *out,
    const float *logits,
    const float *mode_norms,
    float syntony_scale,
    int batch_size,
    int num_classes
) {
    extern __shared__ float shared_f[];
    float *s_max = shared_f;
    float *s_sum = shared_f + blockDim.x;

    int batch_idx = blockIdx.x;
    int tid = threadIdx.x;

    if (batch_idx >= batch_size) return;

    const float *x = logits + batch_idx * num_classes;
    float *y = out + batch_idx * num_classes;

    // Step 1: Find max
    float local_max = -FLT_MAX;
    for (int i = tid; i < num_classes; i += blockDim.x) {
        float log_weight = -mode_norms[i] * PHI_INV_F32 * syntony_scale;
        float weighted_logit = x[i] + log_weight;
        local_max = fmaxf(local_max, weighted_logit);
    }

    s_max[tid] = local_max;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            s_max[tid] = fmaxf(s_max[tid], s_max[tid + s]);
        }
        __syncthreads();
    }

    float max_val = s_max[0];
    __syncthreads();

    // Step 2: Sum
    float local_sum = 0.0f;
    for (int i = tid; i < num_classes; i += blockDim.x) {
        float log_weight = -mode_norms[i] * PHI_INV_F32 * syntony_scale;
        float weighted_logit = x[i] + log_weight;
        local_sum += __expf(weighted_logit - max_val);
    }

    s_sum[tid] = local_sum;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            s_sum[tid] += s_sum[tid + s];
        }
        __syncthreads();
    }

    float sum = s_sum[0];
    __syncthreads();

    // Step 3: Normalize
    for (int i = tid; i < num_classes; i += blockDim.x) {
        float log_weight = -mode_norms[i] * PHI_INV_F32 * syntony_scale;
        float weighted_logit = x[i] + log_weight;
        y[i] = __expf(weighted_logit - max_val) / sum;
    }
}

// =============================================================================
// Syntonic Softmax: provided mode (direct syntony weights)
// =============================================================================

extern "C" __global__ void syntonic_softmax_provided_f64(
    double *out,
    const double *logits,
    const double *syntony,       // Pre-computed syntony weights (same shape as logits)
    double syntony_scale,
    int batch_size,
    int num_classes
) {
    extern __shared__ double shared_p[];
    double *s_max = shared_p;
    double *s_sum = shared_p + blockDim.x;

    int batch_idx = blockIdx.x;
    int tid = threadIdx.x;

    if (batch_idx >= batch_size) return;

    const double *x = logits + batch_idx * num_classes;
    const double *w = syntony + batch_idx * num_classes;
    double *y = out + batch_idx * num_classes;

    // Find max(x + scale * log(w))
    double local_max = -DBL_MAX;
    for (int i = tid; i < num_classes; i += blockDim.x) {
        double log_weight = log(fmax(w[i], 1e-10)) * syntony_scale;
        local_max = fmax(local_max, x[i] + log_weight);
    }

    s_max[tid] = local_max;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) s_max[tid] = fmax(s_max[tid], s_max[tid + s]);
        __syncthreads();
    }

    double max_val = s_max[0];
    __syncthreads();

    // Sum
    double local_sum = 0.0;
    for (int i = tid; i < num_classes; i += blockDim.x) {
        double log_weight = log(fmax(w[i], 1e-10)) * syntony_scale;
        local_sum += exp(x[i] + log_weight - max_val);
    }

    s_sum[tid] = local_sum;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) s_sum[tid] += s_sum[tid + s];
        __syncthreads();
    }

    double sum = s_sum[0];
    __syncthreads();

    // Normalize
    for (int i = tid; i < num_classes; i += blockDim.x) {
        double log_weight = log(fmax(w[i], 1e-10)) * syntony_scale;
        y[i] = exp(x[i] + log_weight - max_val) / sum;
    }
}
