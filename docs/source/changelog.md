# Changelog

All notable changes to Syntonic will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Sphinx documentation with MyST-Parser support
- Full CUDA kernel implementations for hierarchy and golden_gelu
- Retrocausal harmonization with attractor memory

### Fixed
- Variable naming in hierarchy.rs (N → n)
- Unused offset variables in matmul.cu
- Rational import moved to test module in retrocausal.rs

## [0.1.0] - 2026-01-17

### Added
- Initial release
- Core DHSR operators (differentiate, harmonize, syntony, recurse)
- Q(φ) exact arithmetic via GoldenExact
- ResonantTensor with dual lattice/ephemeral representation
- Neural network layers: GoldenGELU, PhiResidual, GoldenBatchNorm, SyntonicSoftmax
- CUDA kernels for all SRT operations
- E₈ lattice projections and golden cone tests
- Hierarchy correction system for particle physics
- Python bindings via PyO3
- Comprehensive test suite

### Technical
- Rust backend with cudarc 0.18.2
- CUDA 12.0 support
- GPU architectures: sm_75, sm_80, sm_86, sm_90
