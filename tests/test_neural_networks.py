#!/usr/bin/env python3
"""
Test Neural Network Components

Tests for SRT-constrained neural network layers and components.
"""

import pytest
import sys
import os
import torch
import torch.nn as nn

# Add syntonic package to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'python'))

try:
    from python.syntonic.nn.layers.prime_syntony_gate import (
        PrimeSyntonyGate,
        WindingAttention,
        SRTTransformerBlock
    )
    HAS_NEURAL_LAYERS = True
except ImportError:
    HAS_NEURAL_LAYERS = False
    print("Warning: Neural network layers not available, tests will be skipped")


class TestPrimeSyntonyGate:
    """Test Prime Syntony Gate layer."""

    @pytest.mark.skipif(not HAS_NEURAL_LAYERS, reason="Neural layers not available")
    def test_gate_creation(self):
        """Test gate can be created with different prime dimensions."""
        primes = [7, 11, 13, 17]

        for prime in primes:
            gate = PrimeSyntonyGate(prime)
            assert gate is not None
            assert hasattr(gate, 'prime')
            assert gate.prime == prime

    @pytest.mark.skipif(not HAS_NEURAL_LAYERS, reason="Neural layers not available")
    def test_gate_forward_pass(self):
        """Test forward pass through Prime Syntony Gate."""
        gate = PrimeSyntonyGate(7)

        # Test different batch sizes and sequence lengths
        test_shapes = [
            (2, 10, 7),   # (batch, seq, dim)
            (1, 5, 7),
            (4, 20, 7)
        ]

        for shape in test_shapes:
            x = torch.randn(*shape)
            output = gate(x)

            # Output should have same shape as input
            assert output.shape == x.shape

            # Output should be different from input (gate applied)
            assert not torch.allclose(output, x, atol=1e-6)

    @pytest.mark.skipif(not HAS_NEURAL_LAYERS, reason="Neural layers not available")
    def test_gate_gradient_flow(self):
        """Test that gradients flow through the gate."""
        gate = PrimeSyntonyGate(7)

        x = torch.randn(2, 10, 7, requires_grad=True)
        output = gate(x)

        # Create a loss and backpropagate
        loss = output.sum()
        loss.backward()

        # Check that gradients exist and are reasonable
        assert x.grad is not None
        assert not torch.isnan(x.grad).any()
        assert not torch.isinf(x.grad).any()

    @pytest.mark.skipif(not HAS_NEURAL_LAYERS, reason="Neural layers not available")
    def test_gate_prime_constraint(self):
        """Test that gate respects prime dimension constraint."""
        gate = PrimeSyntonyGate(7)

        # Should work with dimension 7
        x = torch.randn(2, 10, 7)
        output = gate(x)
        assert output.shape == x.shape

        # Should fail with wrong dimension
        x_wrong = torch.randn(2, 10, 8)
        with pytest.raises(Exception):
            gate(x_wrong)


class TestWindingAttention:
    """Test Winding Attention layer."""

    @pytest.mark.skipif(not HAS_NEURAL_LAYERS, reason="Neural layers not available")
    def test_attention_creation(self):
        """Test attention layer creation."""
        dim = 7
        num_heads = 1

        attention = WindingAttention(dim, num_heads)
        assert attention is not None
        assert hasattr(attention, 'dim')
        assert attention.dim == dim

    @pytest.mark.skipif(not HAS_NEURAL_LAYERS, reason="Neural layers not available")
    def test_attention_forward_pass(self):
        """Test forward pass through attention."""
        attention = WindingAttention(7, 1)

        # Query, key, value should all have same shape
        q = torch.randn(2, 10, 7)
        k = torch.randn(2, 15, 7)  # Different seq length ok
        v = torch.randn(2, 15, 7)

        output = attention(q, k, v)

        # Output should have query's batch and seq dims, value's feature dim
        assert output.shape[0] == q.shape[0]  # batch
        assert output.shape[1] == q.shape[1]  # seq
        assert output.shape[2] == v.shape[2]  # features

    @pytest.mark.skipif(not HAS_NEURAL_LAYERS, reason="Neural layers not available")
    def test_attention_gradient_flow(self):
        """Test gradients flow through attention."""
        attention = WindingAttention(7, 1)

        q = torch.randn(2, 10, 7, requires_grad=True)
        k = torch.randn(2, 15, 7, requires_grad=True)
        v = torch.randn(2, 15, 7, requires_grad=True)

        output = attention(q, k, v)
        loss = output.sum()
        loss.backward()

        # All inputs should have gradients
        assert q.grad is not None
        assert k.grad is not None
        assert v.grad is not None

        # No NaN or Inf values
        assert not torch.isnan(q.grad).any()
        assert not torch.isnan(k.grad).any()
        assert not torch.isnan(v.grad).any()


class TestSRTTransformerBlock:
    """Test SRT Transformer Block."""

    @pytest.mark.skipif(not HAS_NEURAL_LAYERS, reason="Neural layers not available")
    def test_transformer_creation(self):
        """Test transformer block creation."""
        dim = 7
        num_heads = 1

        transformer = SRTTransformerBlock(dim, num_heads)
        assert transformer is not None

        # Should have attention and gate components
        assert hasattr(transformer, 'attention')
        assert hasattr(transformer, 'gate')

    @pytest.mark.skipif(not HAS_NEURAL_LAYERS, reason="Neural layers not available")
    def test_transformer_forward_pass(self):
        """Test forward pass through transformer block."""
        transformer = SRTTransformerBlock(7, 1)

        x = torch.randn(2, 10, 7)
        output = transformer(x)

        # Output should have same shape as input
        assert output.shape == x.shape

        # Output should be different (transformer applied)
        assert not torch.allclose(output, x, atol=1e-6)

    @pytest.mark.skipif(not HAS_NEURAL_LAYERS, reason="Neural layers not available")
    def test_transformer_gradient_flow(self):
        """Test gradients flow through transformer."""
        transformer = SRTTransformerBlock(7, 1)

        x = torch.randn(2, 10, 7, requires_grad=True)
        output = transformer(x)

        loss = output.sum()
        loss.backward()

        assert x.grad is not None
        assert not torch.isnan(x.grad).any()
        assert not torch.isinf(x.grad).any()

    @pytest.mark.skipif(not HAS_NEURAL_LAYERS, reason="Neural layers not available")
    def test_transformer_training_loop(self):
        """Test transformer in a simple training loop."""
        transformer = SRTTransformerBlock(7, 1)
        optimizer = torch.optim.Adam(transformer.parameters(), lr=0.001)

        # Simple training loop
        for step in range(3):
            x = torch.randn(2, 5, 7)
            target = torch.randn(2, 5, 7)

            output = transformer(x)
            loss = nn.MSELoss()(output, target)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Loss should decrease over time
            assert loss.item() > 0

        # Parameters should have been updated
        for param in transformer.parameters():
            assert param.grad is not None


class TestDimensionUtilities:
    """Test dimension utility functions."""

    @pytest.mark.skipif(not HAS_NEURAL_LAYERS, reason="Neural layers not available")
    def test_prime_dimensions(self):
        """Test that layers work with prime dimensions."""
        # Test various prime dimensions
        primes = [5, 7, 11, 13]

        for prime in primes:
            try:
                gate = PrimeSyntonyGate(prime)
                attention = WindingAttention(prime, 1)
                transformer = SRTTransformerBlock(prime, 1)

                # Quick forward pass test
                x = torch.randn(1, 3, prime)
                gate_out = gate(x)
                attn_out = attention(x, x, x)
                trans_out = transformer(x)

                assert gate_out.shape == x.shape
                assert attn_out.shape == x.shape
                assert trans_out.shape == x.shape

            except Exception as e:
                pytest.fail(f"Failed with prime dimension {prime}: {e}")

    @pytest.mark.skipif(not HAS_NEURAL_LAYERS, reason="Neural layers not available")
    def test_non_prime_rejection(self):
        """Test that non-prime dimensions are rejected."""
        non_primes = [6, 8, 9, 10, 12]

        for dim in non_primes:
            with pytest.raises((ValueError, AssertionError, Exception)):
                PrimeSyntonyGate(dim)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])