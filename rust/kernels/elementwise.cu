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
