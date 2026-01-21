// Syntonic CUDA Kernels - Element-wise Operations
// Compiled offline for multi-version driver compatibility

extern "C" __global__ void add_f64(double *out, const double *a,
                                   const double *b, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n)
    out[i] = a[i] + b[i];
}

extern "C" __global__ void add_f32(float *out, const float *a, const float *b,
                                   int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n)
    out[i] = a[i] + b[i];
}

extern "C" __global__ void sub_f64(double *out, const double *a,
                                   const double *b, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n)
    out[i] = a[i] - b[i];
}

extern "C" __global__ void sub_f32(float *out, const float *a, const float *b,
                                   int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n)
    out[i] = a[i] - b[i];
}

extern "C" __global__ void mul_f64(double *out, const double *a,
                                   const double *b, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n)
    out[i] = a[i] * b[i];
}

extern "C" __global__ void mul_f32(float *out, const float *a, const float *b,
                                   int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n)
    out[i] = a[i] * b[i];
}

extern "C" __global__ void div_f64(double *out, const double *a,
                                   const double *b, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n)
    out[i] = a[i] / b[i];
}

extern "C" __global__ void div_f32(float *out, const float *a, const float *b,
                                   int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n)
    out[i] = a[i] / b[i];
}

extern "C" __global__ void neg_f64(double *out, const double *a, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n)
    out[i] = -a[i];
}

extern "C" __global__ void neg_f32(float *out, const float *a, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n)
    out[i] = -a[i];
}

extern "C" __global__ void abs_f64(double *out, const double *a, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n)
    out[i] = fabs(a[i]);
}

extern "C" __global__ void abs_f32(float *out, const float *a, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n)
    out[i] = fabsf(a[i]);
}

extern "C" __global__ void scalar_add_f64(double *out, const double *a,
                                          double scalar, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n)
    out[i] = a[i] + scalar;
}

extern "C" __global__ void scalar_mul_f64(double *out, const double *a,
                                          double scalar, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n)
    out[i] = a[i] * scalar;
}

// Mathematical functions
extern "C" __global__ void exp_f64(double *out, const double *a, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n)
    out[i] = exp(a[i]);
}

extern "C" __global__ void exp_f32(float *out, const float *a, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n)
    out[i] = expf(a[i]);
}

extern "C" __global__ void log_f64(double *out, const double *a, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n)
    out[i] = log(a[i]);
}

extern "C" __global__ void log_f32(float *out, const float *a, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n)
    out[i] = logf(a[i]);
}

extern "C" __global__ void sin_f64(double *out, const double *a, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n)
    out[i] = sin(a[i]);
}

extern "C" __global__ void sin_f32(float *out, const float *a, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n)
    out[i] = sinf(a[i]);
}

extern "C" __global__ void cos_f64(double *out, const double *a, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n)
    out[i] = cos(a[i]);
}

extern "C" __global__ void cos_f32(float *out, const float *a, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n)
    out[i] = cosf(a[i]);
}

extern "C" __global__ void sqrt_f64(double *out, const double *a, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n)
    out[i] = sqrt(a[i]);
}

extern "C" __global__ void sqrt_f32(float *out, const float *a, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n)
    out[i] = sqrtf(a[i]);
}

extern "C" __global__ void tanh_f64(double *out, const double *a, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n)
    out[i] = tanh(a[i]);
}

extern "C" __global__ void tanh_f32(float *out, const float *a, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n)
    out[i] = tanhf(a[i]);
}

extern "C" __global__ void sigmoid_f64(double *out, const double *a, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n)
    out[i] = 1.0 / (1.0 + exp(-a[i]));
}

extern "C" __global__ void sigmoid_f32(float *out, const float *a, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n)
    out[i] = 1.0f / (1.0f + expf(-a[i]));
}

extern "C" __global__ void relu_f64(double *out, const double *a, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n)
    out[i] = fmax(a[i], 0.0);
}

extern "C" __global__ void relu_f32(float *out, const float *a, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n)
    out[i] = fmaxf(a[i], 0.0f);
}

// Golden exponential: exp(-x/φ)
extern "C" __global__ void exp_golden_f64(double *out, const double *a, int n) {
  const double PHI_INV = 0.6180339887498949; // 1/φ
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n)
    out[i] = exp(-a[i] * PHI_INV);
}

extern "C" __global__ void exp_golden_f32(float *out, const float *a, int n) {
  const float PHI_INV = 0.6180339887498949f; // 1/φ
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n)
    out[i] = expf(-a[i] * PHI_INV);
}

// Complex operations (interleaved format: [re0, im0, re1, im1, ...])
extern "C" __global__ void add_c128(double *out, const double *a,
                                    const double *b, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int idx = i * 2;
  if (i < n) {
    out[idx] = a[idx] + b[idx];
    out[idx + 1] = a[idx + 1] + b[idx + 1];
  }
}

extern "C" __global__ void sub_c128(double *out, const double *a,
                                    const double *b, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int idx = i * 2;
  if (i < n) {
    out[idx] = a[idx] - b[idx];
    out[idx + 1] = a[idx + 1] - b[idx + 1];
  }
}

extern "C" __global__ void mul_c128(double *out, const double *a,
                                    const double *b, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int idx = i * 2;
  if (i < n) {
    double ar = a[idx], ai = a[idx + 1];
    double br = b[idx], bi = b[idx + 1];
    out[idx] = ar * br - ai * bi;
    out[idx + 1] = ar * bi + ai * br;
  }
}

extern "C" __global__ void neg_c128(double *out, const double *a, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int idx = i * 2;
  if (i < n) {
    out[idx] = -a[idx];
    out[idx + 1] = -a[idx + 1];
  }
}

extern "C" __global__ void div_c128(double *out, const double *a,
                                    const double *b, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int idx = i * 2;
  if (i < n) {
    double ar = a[idx], ai = a[idx + 1];
    double br = b[idx], bi = b[idx + 1];
    double denom = br * br + bi * bi;
    out[idx] = (ar * br + ai * bi) / denom;
    out[idx + 1] = (ai * br - ar * bi) / denom;
  }
}

extern "C" __global__ void abs_c128(double *out, const double *a, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int idx = i * 2;
  if (i < n) {
    double re = a[idx];
    double im = a[idx + 1];
    out[i] = sqrt(re * re + im * im);
  }
}

extern "C" __global__ void exp_c128(double *out, const double *a, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int idx = i * 2;
  if (i < n) {
    double re = a[idx];
    double im = a[idx + 1];
    double exp_re = exp(re);
    double cos_im, sin_im;
    sincos(im, &sin_im, &cos_im);
    out[idx] = exp_re * cos_im;
    out[idx + 1] = exp_re * sin_im;
  }
}

// ============================================================================
// Broadcast Operations (Tensor op Scalar_Tensor)
// These kernels read the 'b' operand from a single memory address and apply
// it to the entire 'a' array. This avoids CPU roundtrips.
// ============================================================================

extern "C" __global__ void add_broadcast_scalar_f64(double *out,
                                                    const double *a,
                                                    const double *b_scalar,
                                                    int n) {
  // Read the scalar once from global memory (L2/Constant cache will optimize
  // this)
  double scalar = *b_scalar;

  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n)
    out[i] = a[i] + scalar;
}

extern "C" __global__ void add_broadcast_scalar_f32(float *out, const float *a,
                                                    const float *b_scalar,
                                                    int n) {
  float scalar = *b_scalar;

  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n)
    out[i] = a[i] + scalar;
}

extern "C" __global__ void sub_broadcast_scalar_f64(double *out,
                                                    const double *a,
                                                    const double *b_scalar,
                                                    int n) {
  double scalar = *b_scalar;

  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n)
    out[i] = a[i] - scalar;
}

extern "C" __global__ void sub_broadcast_scalar_f32(float *out, const float *a,
                                                    const float *b_scalar,
                                                    int n) {
  float scalar = *b_scalar;

  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n)
    out[i] = a[i] - scalar;
}

extern "C" __global__ void mul_broadcast_scalar_f64(double *out,
                                                    const double *a,
                                                    const double *b_scalar,
                                                    int n) {
  double scalar = *b_scalar;

  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n)
    out[i] = a[i] * scalar;
}

extern "C" __global__ void mul_broadcast_scalar_f32(float *out, const float *a,
                                                    const float *b_scalar,
                                                    int n) {
  float scalar = *b_scalar;

  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n)
    out[i] = a[i] * scalar;
}

extern "C" __global__ void div_broadcast_scalar_f64(double *out,
                                                    const double *a,
                                                    const double *b_scalar,
                                                    int n) {
  double scalar = *b_scalar;

  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n)
    out[i] = a[i] / scalar;
}

extern "C" __global__ void div_broadcast_scalar_f32(float *out, const float *a,
                                                    const float *b_scalar,
                                                    int n) {
  float scalar = *b_scalar;

  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n)
    out[i] = a[i] / scalar;
}

// ============================================================================
// Toroidal Math Functions (T⁴ Geometry)
// ============================================================================

/**
 * Toroidal sine function for winding phase calculations
 * sin(θ) where θ represents position on T⁴ torus
 */
extern "C" __global__ void sin_toroidal_f64(double *out, const double *a,
                                            int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    // Normalize to [0, 2π] for toroidal geometry
    double theta = fmod(a[i], 2.0 * M_PI);
    if (theta < 0)
      theta += 2.0 * M_PI;
    out[i] = sin(theta);
  }
}

extern "C" __global__ void sin_toroidal_f32(float *out, const float *a, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    float theta = fmodf(a[i], 2.0f * M_PI);
    if (theta < 0)
      theta += 2.0f * M_PI;
    out[i] = sinf(theta);
  }
}

/**
 * Toroidal cosine function for winding phase calculations
 */
extern "C" __global__ void cos_toroidal_f64(double *out, const double *a,
                                            int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    double theta = fmod(a[i], 2.0 * M_PI);
    if (theta < 0)
      theta += 2.0 * M_PI;
    out[i] = cos(theta);
  }
}

extern "C" __global__ void cos_toroidal_f32(float *out, const float *a, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    float theta = fmodf(a[i], 2.0f * M_PI);
    if (theta < 0)
      theta += 2.0f * M_PI;
    out[i] = cosf(theta);
  }
}

/**
 * Toroidal atan2 function for phase angle calculations on T⁴
 * Returns angle in [0, 2π] range for toroidal topology
 */
extern "C" __global__ void atan2_toroidal_f64(double *out, const double *y,
                                              const double *x, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    double angle = atan2(y[i], x[i]);
    // Normalize to [0, 2π] for toroidal geometry
    if (angle < 0)
      angle += 2.0 * M_PI;
    out[i] = angle;
  }
}

extern "C" __global__ void atan2_toroidal_f32(float *out, const float *y,
                                              const float *x, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    float angle = atan2f(y[i], x[i]);
    if (angle < 0)
      angle += 2.0f * M_PI;
    out[i] = angle;
  }
}

// ============================================================================
// Golden Exponentials (Consciousness Growth Functions)
// ============================================================================

/**
 * Golden exponential: φ^x - Natural growth function of consciousness
 * This represents the exponential growth pattern observed in biological
 * and conscious systems, following the golden ratio scaling.
 */
extern "C" __global__ void phi_exp_f64(double *out, const double *a, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    // φ^x = φ * (φ^x) but computed efficiently
    double phi = 1.618033988749895;
    out[i] = pow(phi, a[i]);
  }
}

extern "C" __global__ void phi_exp_f32(float *out, const float *a, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    float phi = 1.6180339887f;
    out[i] = powf(phi, a[i]);
  }
}

/**
 * Inverse golden exponential: φ^(-x)
 */
extern "C" __global__ void phi_exp_inv_f64(double *out, const double *a,
                                           int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    double phi = 1.618033988749895;
    out[i] = pow(phi, -a[i]);
  }
}

extern "C" __global__ void phi_exp_inv_f32(float *out, const float *a, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    float phi = 1.6180339887f;
    out[i] = powf(phi, -a[i]);
  }
}
