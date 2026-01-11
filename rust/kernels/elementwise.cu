// Syntonic CUDA Kernels - Element-wise Operations
// Compiled offline for multi-version driver compatibility

extern "C" __global__ void add_f64(double *out, const double *a, const double *b, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = a[i] + b[i];
}

extern "C" __global__ void add_f32(float *out, const float *a, const float *b, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = a[i] + b[i];
}

extern "C" __global__ void sub_f64(double *out, const double *a, const double *b, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = a[i] - b[i];
}

extern "C" __global__ void sub_f32(float *out, const float *a, const float *b, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = a[i] - b[i];
}

extern "C" __global__ void mul_f64(double *out, const double *a, const double *b, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = a[i] * b[i];
}

extern "C" __global__ void mul_f32(float *out, const float *a, const float *b, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = a[i] * b[i];
}

extern "C" __global__ void div_f64(double *out, const double *a, const double *b, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = a[i] / b[i];
}

extern "C" __global__ void div_f32(float *out, const float *a, const float *b, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = a[i] / b[i];
}

extern "C" __global__ void neg_f64(double *out, const double *a, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = -a[i];
}

extern "C" __global__ void neg_f32(float *out, const float *a, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = -a[i];
}

extern "C" __global__ void abs_f64(double *out, const double *a, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = fabs(a[i]);
}

extern "C" __global__ void abs_f32(float *out, const float *a, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = fabsf(a[i]);
}

extern "C" __global__ void scalar_add_f64(double *out, const double *a, double scalar, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = a[i] + scalar;
}

extern "C" __global__ void scalar_mul_f64(double *out, const double *a, double scalar, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = a[i] * scalar;
}

// Mathematical functions
extern "C" __global__ void exp_f64(double *out, const double *a, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = exp(a[i]);
}

extern "C" __global__ void exp_f32(float *out, const float *a, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = expf(a[i]);
}

extern "C" __global__ void log_f64(double *out, const double *a, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = log(a[i]);
}

extern "C" __global__ void log_f32(float *out, const float *a, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = logf(a[i]);
}

extern "C" __global__ void sin_f64(double *out, const double *a, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = sin(a[i]);
}

extern "C" __global__ void sin_f32(float *out, const float *a, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = sinf(a[i]);
}

extern "C" __global__ void cos_f64(double *out, const double *a, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = cos(a[i]);
}

extern "C" __global__ void cos_f32(float *out, const float *a, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = cosf(a[i]);
}

extern "C" __global__ void sqrt_f64(double *out, const double *a, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = sqrt(a[i]);
}

extern "C" __global__ void sqrt_f32(float *out, const float *a, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = sqrtf(a[i]);
}

extern "C" __global__ void tanh_f64(double *out, const double *a, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = tanh(a[i]);
}

extern "C" __global__ void tanh_f32(float *out, const float *a, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = tanhf(a[i]);
}

extern "C" __global__ void sigmoid_f64(double *out, const double *a, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = 1.0 / (1.0 + exp(-a[i]));
}

extern "C" __global__ void sigmoid_f32(float *out, const float *a, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = 1.0f / (1.0f + expf(-a[i]));
}

extern "C" __global__ void relu_f64(double *out, const double *a, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = fmax(a[i], 0.0);
}

extern "C" __global__ void relu_f32(float *out, const float *a, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = fmaxf(a[i], 0.0f);
}

// Golden exponential: exp(-x/φ)
extern "C" __global__ void exp_golden_f64(double *out, const double *a, int n) {
    const double PHI_INV = 0.6180339887498949;  // 1/φ
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = exp(-a[i] * PHI_INV);
}

extern "C" __global__ void exp_golden_f32(float *out, const float *a, int n) {
    const float PHI_INV = 0.6180339887498949f;  // 1/φ
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = expf(-a[i] * PHI_INV);
}

// Complex operations (interleaved format: [re0, im0, re1, im1, ...])
extern "C" __global__ void add_c128(double *out, const double *a, const double *b, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int idx = i * 2;
    if (i < n) {
        out[idx] = a[idx] + b[idx];
        out[idx+1] = a[idx+1] + b[idx+1];
    }
}

extern "C" __global__ void sub_c128(double *out, const double *a, const double *b, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int idx = i * 2;
    if (i < n) {
        out[idx] = a[idx] - b[idx];
        out[idx+1] = a[idx+1] - b[idx+1];
    }
}

extern "C" __global__ void mul_c128(double *out, const double *a, const double *b, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int idx = i * 2;
    if (i < n) {
        double ar = a[idx], ai = a[idx+1];
        double br = b[idx], bi = b[idx+1];
        out[idx] = ar*br - ai*bi;
        out[idx+1] = ar*bi + ai*br;
    }
}

extern "C" __global__ void neg_c128(double *out, const double *a, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int idx = i * 2;
    if (i < n) {
        out[idx] = -a[idx];
        out[idx+1] = -a[idx+1];
    }
}
