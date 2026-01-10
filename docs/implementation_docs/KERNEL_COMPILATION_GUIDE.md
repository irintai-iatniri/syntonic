# CUDA Kernel Compilation Guide

**Last Updated:** 2026-01-08
**Status:** Ready for Use

---

## Quick Start

### Compile All Kernels (Recommended)

```bash
# Using bash script (Linux/macOS)
./compile_kernels.sh

# Using Python script (Cross-platform)
python compile_kernels.py
```

### Compile Specific Kernel

```bash
# Compile only phi_residual
./compile_kernels.sh --kernel=phi_residual
python compile_kernels.py --kernel=phi_residual
```

### Compile for Specific GPU Architecture

```bash
# Compile all kernels for RTX 30 series (SM80)
./compile_kernels.sh --arch=sm_80
python compile_kernels.py --arch=sm_80
```

---

## Prerequisites

### Required

1. **CUDA Toolkit** (version 11.0 or later)
   - Download: https://developer.nvidia.com/cuda-downloads
   - Must include `nvcc` compiler

2. **C++ Compiler** (gcc/g++ or MSVC)
   - Linux: `sudo apt install build-essential`
   - macOS: Install Xcode Command Line Tools
   - Windows: Visual Studio with C++ support

### Verify Installation

```bash
# Check nvcc
nvcc --version

# Expected output:
# nvcc: NVIDIA (R) Cuda compiler driver
# ...
# Build cuda_XX.X.r11.X
```

---

## Compilation Scripts

### Bash Script (`compile_kernels.sh`)

**Platform:** Linux, macOS, WSL
**Features:**
- Fast execution
- Colored output
- Comprehensive error handling
- Parallel compilation support (future)

**Usage:**
```bash
# Make executable (first time only)
chmod +x compile_kernels.sh

# Run
./compile_kernels.sh [OPTIONS]
```

### Python Script (`compile_kernels.py`)

**Platform:** Cross-platform (Linux, macOS, Windows)
**Features:**
- Better Windows support
- Python-friendly error messages
- Easy to extend
- No need to make executable

**Usage:**
```bash
python compile_kernels.py [OPTIONS]
```

---

## Command Line Options

Both scripts support the same options:

| Option | Description | Example |
|--------|-------------|---------|
| `--all` | Compile all kernels (default) | `./compile_kernels.sh --all` |
| `--kernel=NAME` | Compile specific kernel | `--kernel=phi_residual` |
| `--arch=ARCH` | Compile for specific GPU arch | `--arch=sm_80` |
| `--clean` | Remove all PTX files | `--clean` |
| `--list` | List available kernels | `--list` |
| `--help` | Show help message | `--help` |

---

## GPU Architectures

The scripts compile for 4 GPU architectures by default:

| Code | Architecture | GPUs | Released |
|------|-------------|------|----------|
| `sm_75` | Turing | RTX 20 series, T4, Quadro RTX | 2018 |
| `sm_80` | Ampere | A100, RTX 30 series | 2020 |
| `sm_86` | Ampere | RTX 30 series (refined) | 2020 |
| `sm_90` | Hopper | H100, GH100 | 2022 |

**Note:** Compiling for older architectures (SM75) ensures compatibility with more GPUs.

---

## Available Kernels

The following kernels will be compiled:

| Kernel | Status | Description |
|--------|--------|-------------|
| `phi_residual` | ✅ Ready | Phi-scaled residual connections |
| `golden_batch_norm` | 🚧 Pending | Golden batch normalization (Phase 2) |
| `syntonic_softmax` | 🚧 Pending | Syntony-weighted softmax (Phase 3) |
| `golden_ops` | ✅ Existing | Golden ratio operations |
| `e8_projection` | ✅ Existing | E₈ lattice projections |
| `heat_kernel` | ✅ Existing | Heat kernel / theta series |
| `dhsr` | ✅ Existing | DHSR cycle operations |
| `corrections` | ✅ Existing | SRT correction factors |
| `resonant_d` | ✅ Existing | Resonant D-phase |
| `core_ops` | ✅ Existing | Core mathematical operations |
| `elementwise` | ✅ Existing | Element-wise operations |

**Total:** 11 kernels × 4 architectures = **44 PTX files**

---

## Output Files

Compiled PTX files are stored in:
```
rust/kernels/ptx/
├── phi_residual_sm75.ptx
├── phi_residual_sm80.ptx
├── phi_residual_sm86.ptx
├── phi_residual_sm90.ptx
├── golden_batch_norm_sm75.ptx
├── ... (44 files total)
```

**File Sizes:** Each PTX file is typically 50-200 KB.

---

## Compilation Process

### Step-by-Step

1. **Check `nvcc` availability**
   - Script verifies CUDA Toolkit is installed
   - Displays version information

2. **Create output directory**
   - Creates `rust/kernels/ptx/` if it doesn't exist

3. **Compile each kernel**
   - For each `.cu` file in `rust/kernels/`
   - Generate PTX for each architecture
   - Display progress and errors

4. **Summary**
   - Reports successful/failed compilations
   - Lists all generated PTX files

### Example Output

```
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  Syntonic CUDA Kernel Compiler
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

✓ Found nvcc version: 12.0
✓ PTX directory ready: rust/kernels/ptx

Compiling phi_residual.cu...
  Compiling phi_residual.cu for sm_75...
    ✓ Generated phi_residual_sm75.ptx (156K)
  Compiling phi_residual.cu for sm_80...
    ✓ Generated phi_residual_sm80.ptx (158K)
  Compiling phi_residual.cu for sm_86...
    ✓ Generated phi_residual_sm86.ptx (159K)
  Compiling phi_residual.cu for sm_90...
    ✓ Generated phi_residual_sm90.ptx (162K)
✓ phi_residual: All 4 architectures compiled successfully

... (more kernels) ...

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  Compilation Summary
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Total kernels: 11
Successful: 11

Generated PTX files:
  phi_residual_sm75.ptx (156K)
  phi_residual_sm80.ptx (158K)
  ... (44 files total)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  ✓ All kernels compiled successfully!
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

---

## Troubleshooting

### Error: `nvcc not found`

**Problem:** CUDA Toolkit not installed or not in PATH

**Solution:**
```bash
# Check if CUDA is installed
ls /usr/local/cuda*/bin/nvcc

# Add to PATH (Linux/macOS)
export PATH=/usr/local/cuda/bin:$PATH

# Add to PATH permanently (add to ~/.bashrc or ~/.zshrc)
echo 'export PATH=/usr/local/cuda/bin:$PATH' >> ~/.bashrc
source ~/.bashrc
```

### Error: `fatal error: srt_constants.cuh: No such file or directory`

**Problem:** Include path not correct

**Solution:**
```bash
# Verify srt_constants.cuh exists
ls rust/kernels/srt_constants.cuh

# If missing, check the CUDA kernel files include it
grep "srt_constants.cuh" rust/kernels/*.cu
```

### Error: Compilation failed for specific architecture

**Problem:** CUDA version doesn't support newer architectures

**Solution:**
```bash
# Compile only for supported architectures
./compile_kernels.sh --arch=sm_75  # Oldest, most compatible

# Check CUDA Toolkit version
nvcc --version  # Need 11.0+ for sm_80, 11.8+ for sm_90
```

### Warning: Some kernels failed to compile

**Problem:** Kernel source files missing (Phase 2/3 not yet implemented)

**Solution:**
```bash
# This is expected if Phase 2 and 3 aren't complete yet
# Compile only existing kernels
./compile_kernels.sh --kernel=phi_residual
./compile_kernels.sh --kernel=golden_ops
# etc.
```

---

## Integration with Rust Build

After compiling PTX files, integrate with Rust:

### 1. Verify PTX Files Exist

```bash
ls -lh rust/kernels/ptx/*.ptx
```

### 2. Build Rust Package

```bash
cd rust
cargo build --release --features cuda
```

### 3. Install Python Package

```bash
maturin develop --release --features cuda
```

### 4. Test Import

```python
from syntonic._core import PhiResidualMode, phi_residual
print("✓ Phi-residual operations available")
```

---

## Continuous Integration (CI)

For automated builds, add to CI pipeline:

```yaml
# Example GitHub Actions workflow
- name: Install CUDA Toolkit
  uses: Jimver/cuda-toolkit@v0.2.11
  with:
    cuda: '12.0.0'

- name: Compile CUDA Kernels
  run: python compile_kernels.py

- name: Build Rust Package
  run: |
    cd rust
    cargo build --release --features cuda
```

---

## Performance Notes

### Compilation Time

- **Single kernel, single arch:** ~5-10 seconds
- **Single kernel, all archs:** ~20-40 seconds
- **All kernels, all archs:** ~3-5 minutes

### PTX File Sizes

- **Per kernel:** ~150-200 KB per architecture
- **Total:** ~6-8 MB for all kernels and architectures

---

## Development Workflow

### When Adding New Kernels

1. **Create `.cu` file**
   ```bash
   touch rust/kernels/my_new_kernel.cu
   ```

2. **Add to kernel list**
   - Edit `compile_kernels.sh` and add `"my_new_kernel"` to `KERNELS` array
   - Edit `compile_kernels.py` and add `"my_new_kernel"` to `KERNELS` list

3. **Compile**
   ```bash
   ./compile_kernels.sh --kernel=my_new_kernel
   ```

4. **Integrate into Rust**
   - Add PTX includes to `rust/src/tensor/srt_kernels.rs`
   - Add kernel function names
   - Add loader function

---

## Advanced Usage

### Custom Compilation Flags

Edit the scripts to add custom nvcc flags:

```bash
# In compile_kernels.sh
NVCC_FLAGS="-ptx --use_fast_math -O3 -lineinfo -I${KERNEL_DIR}"
```

### Compile for Specific CUDA Version

```bash
# Target specific CUDA version
nvcc --cuda-version=11.8 -arch=sm_90 ...
```

### Generate Assembly (SASS)

```bash
# Generate SASS instead of PTX for debugging
nvcc -cubin -arch=sm_80 rust/kernels/phi_residual.cu
cuobjdump -sass phi_residual.cubin
```

---

## FAQs

**Q: Do I need to recompile for every GPU?**
A: No. PTX is portable across compatible architectures. We compile for 4 archs to optimize performance.

**Q: Can I skip older architectures?**
A: Yes. Use `--arch=sm_90` to compile only for Hopper GPUs. But this limits compatibility.

**Q: What if compilation fails?**
A: Check error messages. Common issues: missing CUDA, wrong paths, syntax errors in .cu files.

**Q: How do I verify PTX files are correct?**
A: Use `cuobjdump -ptx phi_residual_sm80.ptx` to inspect PTX assembly.

**Q: Can I compile on a machine without GPU?**
A: Yes! You only need CUDA Toolkit (nvcc), not an actual GPU. The GPU is needed at runtime.

---

## Next Steps

After successful compilation:

1. ✅ Verify all PTX files generated
2. ✅ Build Rust package with `cargo build --release --features cuda`
3. ✅ Install Python package with `maturin develop`
4. ✅ Run unit tests: `cargo test phi_residual`
5. ✅ Test Python API: `python -c "from syntonic._core import phi_residual"`
6. ➡️ **Move to Phase 2:** Golden Batch Normalization implementation

---

## Support

For issues or questions:
- Check `rust/kernels/*.cu` for source code
- Review `rust/src/tensor/srt_kernels.rs` for integration
- See `PHASE1_PHI_RESIDUAL_COMPLETE.md` for implementation details

---

**Ready to compile?**

```bash
./compile_kernels.sh
```
