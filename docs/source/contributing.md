# Contributing to Syntonic

Thank you for your interest in contributing to Syntonic!

## Development Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/irintai-iatniri/syntonic.git
   cd syntonic
   ```

2. Install development dependencies:
   ```bash
   pip install maturin
   pip install -e ".[dev]"
   ```

3. Build the Rust extension:
   ```bash
   maturin develop --release
   ```

4. Run tests:
   ```bash
   pytest
   ```

## Code Style

- Python: Follow PEP 8, use `black` for formatting
- Rust: Follow standard Rust style, use `cargo fmt`
- Documentation: Use Google-style docstrings

## Pull Request Process

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Ensure all tests pass
6. Submit a pull request

## Theory Alignment

All contributions must maintain alignment with Syntony Recursion Theory:

- Golden ratio φ is fundamental to all operations
- Target syntony is S* = 1/φ ≈ 0.618
- DHSR cycle is the core operation
- Q(φ) arithmetic preserves exactness

## Questions?

Open an issue on GitHub or contact the maintainers.
