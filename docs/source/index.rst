.. Syntonic documentation master file

=====================================
Syntonic Documentation
=====================================

**Syntonic** is a tensor library for Cosmological and Syntony Recursion Theory (CRT/SRT), providing
GPU-accelerated primitives for golden ratio algebra, E₈ lattice operations, and the DHSR cycle
(Differentiation, Harmonization, Syntony, Recursion).

.. note::
   
   Syntonic implements theory-correct mathematics where the golden ratio φ = (1 + √5)/2 is
   fundamental to all operations, targeting syntony S* = 1/φ ≈ 0.618.

Key Features
============

- **Q(φ) Exact Arithmetic**: Lossless golden ratio lattice computations
- **Resonant Tensors**: Dual representation with exact lattice + ephemeral flux
- **DHSR Cycle**: Theory-aligned differentiation, harmonization, syntony, recursion
- **GPU Acceleration**: CUDA kernels for high-performance computation
- **Neural Network Layers**: GoldenGELU, PhiResidual, GoldenBatchNorm, SyntonicSoftmax

Quick Start
===========

.. code-block:: python

   import syntonic
   from syntonic.core import ResonantState
   from syntonic.nn import GoldenGELU, PhiResidualConnection
   
   # Create a resonant state
   state = ResonantState.from_floats([1.0, 2.0, 3.0, 4.0])
   
   # Compute syntony (target: 1/φ ≈ 0.618)
   print(f"Syntony: {state.syntony():.6f}")
   
   # Apply DHSR cycle
   state.differentiate()
   state.harmonize()
   syntony = state.compute_syntony()
   state.recurse()

Documentation Contents
======================

.. toctree::
   :maxdepth: 2
   :caption: Getting Started
   
   installation
   quickstart
   concepts

.. toctree::
   :maxdepth: 2
   :caption: Theory
   
   theory/dhsr
   theory/golden_ratio
   theory/e8_lattice
   theory/retrocausal

.. toctree::
   :maxdepth: 2
   :caption: API Reference
   
   api/core
   api/nn
   api/crt
   api/resonant
   api/hierarchy

.. toctree::
   :maxdepth: 2
   :caption: Development
   
   contributing
   changelog

Indices and Tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

Mathematical Foundation
=======================

The fundamental equation of Syntony Recursion Theory:

.. math::

   \mathcal{S}[\psi] = 1 - H[\psi] = 1 - \left( -\sum_i p_i \log_\varphi p_i \right)

Where:
- :math:`\mathcal{S}` is the syntony measure
- :math:`H` is the golden-base entropy
- :math:`p_i` are probability amplitudes
- The target equilibrium is :math:`\mathcal{S}^* = 1/\varphi \approx 0.618`

The DHSR cycle drives evolution toward this fixed point:

.. math::

   \psi_{n+1} = \mathcal{R} \circ \mathcal{S} \circ \mathcal{H} \circ \mathcal{D}[\psi_n]
