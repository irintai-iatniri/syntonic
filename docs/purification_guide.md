# Syntonic Library Purification: Refactoring Guide

All files must be refactored to use the pure Rust backend (`syntonic._core`) instead of PyTorch/NumPy.

## ✅ Completed (4 files)

| File | Status |
| :--- | :--- |
| [resonant_linear.py](file:///home/Andrew/Documents/SRT%20Complete/implementation/syntonic/python/syntonic/nn/layers/resonant_linear.py) | ✅ Done |
| [resonant_parameter.py](file:///home/Andrew/Documents/SRT%20Complete/implementation/syntonic/python/syntonic/nn/layers/resonant_parameter.py) | ✅ Done |
| [embedding.py](file:///home/Andrew/Documents/SRT%20Complete/implementation/syntonic/python/syntonic/nn/winding/embedding.py) | ✅ Done |
| [dhsr_block.py](file:///home/Andrew/Documents/SRT%20Complete/implementation/syntonic/python/syntonic/nn/winding/dhsr_block.py) | ✅ Done |

---

## 🔴 Requires Refactoring (37 files)

### `nn/layers/` (5 files)

| File | PyTorch Usage | Rust Replacement |
| :--- | :--- | :--- |
| [differentiation.py](file:///home/Andrew/Documents/SRT%20Complete/implementation/syntonic/python/syntonic/nn/layers/differentiation.py) | `nn.Module`, tensor ops | `ResonantTensor.cpu_cycle()` |
| [harmonization.py](file:///home/Andrew/Documents/SRT%20Complete/implementation/syntonic/python/syntonic/nn/layers/harmonization.py) | `nn.Module`, golden ops | `ResonantTensor.apply_recursion()` |
| [normalization.py](file:///home/Andrew/Documents/SRT%20Complete/implementation/syntonic/python/syntonic/nn/layers/normalization.py) | `nn.LayerNorm` | `layer_norm_f64` CUDA kernel |
| [recursion.py](file:///home/Andrew/Documents/SRT%20Complete/implementation/syntonic/python/syntonic/nn/layers/recursion.py) | `nn.Module`, recursion | `ResonantTensor.apply_recursion()` |
| [syntonic_gate.py](file:///home/Andrew/Documents/SRT%20Complete/implementation/syntonic/python/syntonic/nn/layers/syntonic_gate.py) | `nn.Linear`, sigmoid | Custom gate with `ResonantLinear` |

---

### `nn/winding/` (4 files)

| File | PyTorch Usage | Rust Replacement |
| :--- | :--- | :--- |
| [prime_selection.py](file:///home/Andrew/Documents/SRT%20Complete/implementation/syntonic/python/syntonic/nn/winding/prime_selection.py) | `nn.Module`, masking | Pure Python bitwise ops |
| [resonant_embedding.py](file:///home/Andrew/Documents/SRT%20Complete/implementation/syntonic/python/syntonic/nn/winding/resonant_embedding.py) | `ResonantParameter` (old) | Already refactored `ResonantParameter` |
| [syntony.py](file:///home/Andrew/Documents/SRT%20Complete/implementation/syntonic/python/syntonic/nn/winding/syntony.py) | `torch.Tensor` ops | `ResonantTensor.syntony` property |
| [winding_net.py](file:///home/Andrew/Documents/SRT%20Complete/implementation/syntonic/python/syntonic/nn/winding/winding_net.py) | Full network | Compose pure layers |

---

### `nn/architectures/` (5 files)

| File | PyTorch Usage | Rust Replacement |
| :--- | :--- | :--- |
| [embeddings.py](file:///home/Andrew/Documents/SRT%20Complete/implementation/syntonic/python/syntonic/nn/architectures/embeddings.py) | `nn.Embedding` | `WindingStateEmbedding` |
| [syntonic_attention.py](file:///home/Andrew/Documents/SRT%20Complete/implementation/syntonic/python/syntonic/nn/architectures/syntonic_attention.py) | `nn.MultiheadAttention` | `GoldenConeAttention` from `pure/` |
| [syntonic_cnn.py](file:///home/Andrew/Documents/SRT%20Complete/implementation/syntonic/python/syntonic/nn/architectures/syntonic_cnn.py) | `nn.Conv2d` | **Needs CUDA kernel** |
| [syntonic_mlp.py](file:///home/Andrew/Documents/SRT%20Complete/implementation/syntonic/python/syntonic/nn/architectures/syntonic_mlp.py) | `nn.Sequential` | Compose `ResonantLinear` |
| [syntonic_transformer.py](file:///home/Andrew/Documents/SRT%20Complete/implementation/syntonic/python/syntonic/nn/architectures/syntonic_transformer.py) | Full transformer | `PureResonantTransformer` |

---

### `nn/loss/` (4 files)

| File | PyTorch Usage | Rust Replacement |
| :--- | :--- | :--- |
| [phase_alignment.py](file:///home/Andrew/Documents/SRT%20Complete/implementation/syntonic/python/syntonic/nn/loss/phase_alignment.py) | `torch.Tensor` ops | Pure Python + syntony |
| [regularization.py](file:///home/Andrew/Documents/SRT%20Complete/implementation/syntonic/python/syntonic/nn/loss/regularization.py) | `torch.norm` | `ResonantTensor.to_floats()` + Python |
| [syntonic_loss.py](file:///home/Andrew/Documents/SRT%20Complete/implementation/syntonic/python/syntonic/nn/loss/syntonic_loss.py) | `nn.Module` loss | Pure Python loss function |
| [syntony_metrics.py](file:///home/Andrew/Documents/SRT%20Complete/implementation/syntonic/python/syntonic/nn/loss/syntony_metrics.py) | `torch.Tensor` ops | `ResonantTensor.syntony` |

---

### `nn/optim/` (4 files) — **DELETE or REPLACE**

| File | Action |
| :--- | :--- |
| [gradient_mod.py](file:///home/Andrew/Documents/SRT%20Complete/implementation/syntonic/python/syntonic/nn/optim/gradient_mod.py) | **DELETE** (no gradients in RES) |
| [schedulers.py](file:///home/Andrew/Documents/SRT%20Complete/implementation/syntonic/python/syntonic/nn/optim/schedulers.py) | **DELETE** (no LR scheduling needed) |
| [syntonic_adam.py](file:///home/Andrew/Documents/SRT%20Complete/implementation/syntonic/python/syntonic/nn/optim/syntonic_adam.py) | **DELETE** → Use RES |
| [syntonic_sgd.py](file:///home/Andrew/Documents/SRT%20Complete/implementation/syntonic/python/syntonic/nn/optim/syntonic_sgd.py) | **DELETE** → Use RES |

---

### `nn/training/` (3 files)

| File | PyTorch Usage | Rust Replacement |
| :--- | :--- | :--- |
| [callbacks.py](file:///home/Andrew/Documents/SRT%20Complete/implementation/syntonic/python/syntonic/nn/training/callbacks.py) | Training hooks | Pure Python callbacks |
| [metrics.py](file:///home/Andrew/Documents/SRT%20Complete/implementation/syntonic/python/syntonic/nn/training/metrics.py) | `torch.Tensor` | Pure Python metrics |
| [trainer.py](file:///home/Andrew/Documents/SRT%20Complete/implementation/syntonic/python/syntonic/nn/training/trainer.py) | Full trainer | RES-based trainer |

---

### `nn/analysis/` (3 files)

| File | PyTorch Usage | Rust Replacement |
| :--- | :--- | :--- |
| [escape.py](file:///home/Andrew/Documents/SRT%20Complete/implementation/syntonic/python/syntonic/nn/analysis/escape.py) | Tensor analysis | Pure Python |
| [health.py](file:///home/Andrew/Documents/SRT%20Complete/implementation/syntonic/python/syntonic/nn/analysis/health.py) | Tensor stats | Pure Python |
| [visualization.py](file:///home/Andrew/Documents/SRT%20Complete/implementation/syntonic/python/syntonic/nn/analysis/visualization.py) | Plotting | Keep matplotlib only |

---

### `nn/benchmarks/` (3 files)

| File | PyTorch Usage | Rust Replacement |
| :--- | :--- | :--- |
| [ablation.py](file:///home/Andrew/Documents/SRT%20Complete/implementation/syntonic/python/syntonic/nn/benchmarks/ablation.py) | Full benchmark | Refactor to use pure |
| [convergence.py](file:///home/Andrew/Documents/SRT%20Complete/implementation/syntonic/python/syntonic/nn/benchmarks/convergence.py) | Comparison | Refactor |
| [standard.py](file:///home/Andrew/Documents/SRT%20Complete/implementation/syntonic/python/syntonic/nn/benchmarks/standard.py) | Standard tests | Refactor |

---

### `benchmarks/` (6 files)

| File | PyTorch Usage | Rust Replacement |
| :--- | :--- | :--- |
| [comparative_resonant_benchmark.py](file:///home/Andrew/Documents/SRT%20Complete/implementation/syntonic/python/syntonic/benchmarks/comparative_resonant_benchmark.py) | NumPy data gen | Pure Python |
| [convergence_benchmark.py](file:///home/Andrew/Documents/SRT%20Complete/implementation/syntonic/python/syntonic/benchmarks/convergence_benchmark.py) | Full PyTorch | Replace with pure |
| [datasets.py](file:///home/Andrew/Documents/SRT%20Complete/implementation/syntonic/python/syntonic/benchmarks/datasets.py) | NumPy arrays | Pure Python lists |
| [fitness.py](file:///home/Andrew/Documents/SRT%20Complete/implementation/syntonic/python/syntonic/benchmarks/fitness.py) | NumPy | Pure Python |
| [winding_benchmark.py](file:///home/Andrew/Documents/SRT%20Complete/implementation/syntonic/python/syntonic/benchmarks/winding_benchmark.py) | PyTorch | Pure |
| [winding_xor_benchmark.py](file:///home/Andrew/Documents/SRT%20Complete/implementation/syntonic/python/syntonic/benchmarks/winding_xor_benchmark.py) | PyTorch | Pure |

---

## Summary

| Category | Files | Priority |
| :--- | :--- | :--- |
| Layers | 5 | P1 |
| Winding | 4 | P1 |
| Architectures | 5 | P1 |
| Loss | 4 | P2 |
| Optim | 4 | **DELETE** |
| Training | 3 | P2 |
| Analysis | 3 | P3 |
| NN Benchmarks | 3 | P3 |
| Root Benchmarks | 6 | P3 |
| **Total** | **37** | |
