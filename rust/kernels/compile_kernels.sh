#!/bin/bash
# Compile CUDA kernels to PTX for the current toolkit version
# Run this script to generate PTX files for all kernel modules

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Get CUDA version
if [ -z "$CUDA_PATH" ]; then
    CUDA_PATH="/usr/local/cuda"
fi

NVCC="$CUDA_PATH/bin/nvcc"
if [ ! -x "$NVCC" ]; then
    echo "Error: nvcc not found at $NVCC"
    exit 1
fi

# Get CUDA version number
CUDA_VERSION=$($NVCC --version | grep -oP 'release \K[0-9]+\.[0-9]+' | head -1)
CUDA_MAJOR=$(echo $CUDA_VERSION | cut -d. -f1)
CUDA_MINOR=$(echo $CUDA_VERSION | cut -d. -f2)

echo "Compiling kernels with CUDA $CUDA_VERSION"
echo "=========================================="

# Create output directory
mkdir -p ptx

# List of kernel source files
KERNEL_FILES=(
    "phi_residual"
    "golden_batch_norm"
    "syntonic_softmax"
    "golden_ops"
    "e8_projection"
    "heat_kernel"
    "dhsr"
    "corrections"
    "resonant_d"
    "core_ops"
    "elementwise"
    "matmul"
    "conv_ops"
    "winding_ops"
)

# Compute capabilities to compile for
COMPUTE_CAPS=("75" "80" "86" "90")

# Compile each kernel for each compute capability
for kernel in "${KERNEL_FILES[@]}"; do
    echo ""
    echo "Compiling: ${kernel}.cu"

    if [ ! -f "${kernel}.cu" ]; then
        echo "  WARNING: ${kernel}.cu not found, skipping"
        continue
    fi

    for sm in "${COMPUTE_CAPS[@]}"; do
        output="ptx/${kernel}_sm${sm}.ptx"
        echo "  -> ${output}"
        "$NVCC" -ptx -arch=compute_${sm} "${kernel}.cu" -o "${output}"
    done
done

echo ""
echo "=========================================="
echo "Compilation complete!"
echo ""
echo "Generated PTX files:"
ls -la ptx/

echo ""
echo "PTX files are in: $SCRIPT_DIR/ptx/"
echo ""
echo "Kernel summary:"
echo "  - elementwise.cu    : Basic element-wise operations (add, sub, mul, div, neg, abs)"
echo "  - golden_ops.cu     : Golden ratio operations (φ scaling, gaussian weights, recursion)"
echo "  - e8_projection.cu  : E₈ lattice projections (P_φ, P_⊥, quadratic form, cone test)"
echo "  - heat_kernel.cu    : Heat kernel / theta series summation"
echo "  - dhsr.cu           : DHSR cycle operations (differentiation, harmonization, syntony)"
echo "  - corrections.cu    : SRT correction factors (1 ± q/N)"
echo "  - resonant_d.cu     : Resonant D-phase operations (flux generation, noise, syntony)"
