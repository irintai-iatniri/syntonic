## Plan: API completion and snap hardening

Complete missing ResonantTensor APIs (concat semantics, GELU, layer_norm golden_target, stats), tighten golden snap/φ-dwell precision, add targeted tests, and update docs while pruning only when a better replacement exists.

### Steps
1. Define concat contract (dim checks, negative dims, mode_norm propagation) and document/update implementations in [rust/src/resonant/tensor.rs](rust/src/resonant/tensor.rs) and bindings in [rust/src/lib.rs](rust/src/lib.rs).
2. Add GELU and mean/var helpers to ResonantTensor/storage and export via [python/syntonic/__init__.py](python/syntonic/__init__.py); ensure golden snap consistency and docstrings aligned with theory.
3. Refine layer_norm golden_target behavior (gamma/beta, snap) and golden snap precision (find_nearest) plus φ-dwell enforcement defaults in [rust/src/resonant/tensor.rs](rust/src/resonant/tensor.rs).
4. Extend tests for concat edge cases, GELU, layer_norm (golden_target on/off), stats, snap correctness, φ-dwell ratio, and optional CUDA D→H cycle in [tests](tests) (add new files as needed).
5. Prune legacy/duplicate paths only when superseded (e.g., tensor_storage legacy variants) and sync exports in [rust/src/lib.rs](rust/src/lib.rs) and [python/syntonic/__init__.py](python/syntonic/__init__.py).
6. Refresh API docs and resonant engine notes in [docs/SYNTONIC_API_REFERENCE.md](docs/SYNTONIC_API_REFERENCE.md) and [docs/resonant_engine.md](docs/resonant_engine.md) to reflect new APIs, snap precision, and φ-dwell constraints.

### Further Considerations
1. Confirm desired concat semantics: strict dimension match except axis? allow mixed mode_norm? Option A strict / Option B permissive.
2. Decide GELU flavor (approx vs erf) and whether to snap outputs by default.
3. Set target snap precision threshold and φ-dwell tolerance for tests to avoid flakiness.
