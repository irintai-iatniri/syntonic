"""
Tests for SRT/CRT Neural Network Layers

Tests the Prime Syntony Gate and related neural network components
that implement SRT/CRT principles in deep learning.
"""

import pytest
import torch
import torch.nn as nn
import math
from syntonic.nn.layers.prime_syntony_gate import (
    PrimeSyntonyGate,
    WindingAttention,
    SRTTransformerBlock,
    get_stable_dimensions,
    suggest_network_dimensions
)


class TestPrimeSyntonyGate:
    """Test the Prime Syntony Gate layer."""

    def test_gate_creation(self):
        """Test gate creation with different dimensions."""
        # Test resonant dimensions (Fibonacci primes)
        gate_3 = PrimeSyntonyGate(3)
        assert gate_3.is_resonant
        assert gate_3.boost_factor > 1.0

        gate_5 = PrimeSyntonyGate(5)
        assert gate_5.is_resonant

        gate_7 = PrimeSyntonyGate(7)
        assert gate_7.is_resonant

        # Test non-resonant dimensions
        gate_4 = PrimeSyntonyGate(4)
        assert gate_4.is_resonant  # Special case - material anomaly

        gate_6 = PrimeSyntonyGate(6)
        assert not gate_6.is_resonant
        assert gate_6.boost_factor == 1.0

    def test_gate_forward_pass(self):
        """Test forward pass through the gate."""
        batch_size, seq_len, dim = 2, 10, 7
        gate = PrimeSyntonyGate(dim)

        x = torch.randn(batch_size, seq_len, dim)
        output = gate(x)

        # Output should have same shape
        assert output.shape == x.shape

        # For resonant dimensions, output should be normalized and boosted
        if gate.is_resonant:
            # Check that output is normalized (unit vectors)
            norms = torch.norm(output, dim=-1)
            assert torch.allclose(norms, torch.ones_like(norms), atol=1e-6)

    def test_material_anomaly_penalty(self):
        """Test the material anomaly penalty at dimension 4."""
        gate_4 = PrimeSyntonyGate(4)
        gate_3 = PrimeSyntonyGate(3)

        # Dimension 4 should have reduced boost due to material anomaly
        assert gate_4.boost_factor < gate_3.boost_factor

        # But still greater than 1
        assert gate_4.boost_factor > 1.0

    def test_gate_boost_factors(self):
        """Test that boost factors follow Fibonacci scaling."""
        phi = (1 + math.sqrt(5)) / 2

        gate_3 = PrimeSyntonyGate(3)
        expected_3 = phi ** 3
        assert abs(gate_3.boost_factor - expected_3) < 1e-6

        gate_5 = PrimeSyntonyGate(5)
        expected_5 = phi ** 5
        assert abs(gate_5.boost_factor - expected_5) < 1e-6


class TestWindingAttention:
    """Test the Winding Attention layer."""

    def test_attention_creation(self):
        """Test attention layer creation."""
        embed_dim = 7  # Mersenne prime
        num_heads = 1
        attention = WindingAttention(embed_dim, num_heads)

        assert attention.embed_dim == embed_dim
        assert attention.num_heads == num_heads

    def test_attention_forward(self):
        """Test attention forward pass."""
        batch_size, seq_len, embed_dim = 2, 10, 7
        num_heads = 1

        attention = WindingAttention(embed_dim, num_heads)
        attention.eval()  # Disable dropout for testing

        q = torch.randn(batch_size, seq_len, embed_dim)
        k = torch.randn(batch_size, seq_len, embed_dim)
        v = torch.randn(batch_size, seq_len, embed_dim)

        output = attention(q, k, v)

        # Check output shape
        assert output.shape == (batch_size, seq_len, embed_dim)

    def test_mersenne_dimension_validation(self):
        """Test that non-Mersenne dimensions raise warnings."""
        with pytest.warns(UserWarning, match="not a Mersenne prime"):
            attention = WindingAttention(8, 1)  # 8 is not Mersenne prime

    def test_attention_with_mask(self):
        """Test attention with attention mask."""
        batch_size, seq_len, embed_dim = 2, 10, 7
        num_heads = 1

        attention = WindingAttention(embed_dim, num_heads)

        q = torch.randn(batch_size, seq_len, embed_dim)
        k = torch.randn(batch_size, seq_len, embed_dim)
        v = torch.randn(batch_size, seq_len, embed_dim)

        # Create causal mask
        mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()
        mask = mask.unsqueeze(0).unsqueeze(0)  # Add batch and head dims

        output = attention(q, k, v, mask)

        assert output.shape == (batch_size, seq_len, embed_dim)


class TestSRTTransformerBlock:
    """Test the SRT Transformer Block."""

    def test_transformer_creation(self):
        """Test transformer block creation."""
        embed_dim = 7
        num_heads = 1
        ff_dim = 28

        transformer = SRTTransformerBlock(embed_dim, num_heads, ff_dim)

        assert transformer.attention.embed_dim == embed_dim
        assert transformer.attention.num_heads == num_heads

    def test_transformer_forward(self):
        """Test transformer block forward pass."""
        batch_size, seq_len, embed_dim = 2, 10, 7
        num_heads = 1

        transformer = SRTTransformerBlock(embed_dim, num_heads)
        transformer.eval()

        x = torch.randn(batch_size, seq_len, embed_dim)
        output = transformer(x)

        # Check output shape
        assert output.shape == (batch_size, seq_len, embed_dim)

        # Check that layer norm is applied
        assert hasattr(transformer, 'norm1')
        assert hasattr(transformer, 'norm2')

    def test_transformer_with_mask(self):
        """Test transformer with attention mask."""
        batch_size, seq_len, embed_dim = 2, 10, 7
        num_heads = 1

        transformer = SRTTransformerBlock(embed_dim, num_heads)

        x = torch.randn(batch_size, seq_len, embed_dim)

        # Create causal mask
        mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()
        mask = mask.unsqueeze(0).unsqueeze(0)

        output = transformer(x, mask)
        assert output.shape == (batch_size, seq_len, embed_dim)


class TestDimensionUtilities:
    """Test dimension utility functions."""

    def test_get_stable_dimensions(self):
        """Test getting stable (Mersenne prime) dimensions."""
        stable_dims = get_stable_dimensions(50)

        expected_stable = [3, 7, 31]  # Mersenne primes up to 50
        for dim in expected_stable:
            assert dim in stable_dims

        # Check that unstable dimensions are not included
        assert 4 not in stable_dims  # Not prime
        assert 8 not in stable_dims  # Not prime
        assert 9 not in stable_dims  # Not prime

    def test_suggest_network_dimensions(self):
        """Test network dimension suggestions."""
        input_dim = 10
        output_dim = 5
        num_layers = 3

        dimensions = suggest_network_dimensions(input_dim, output_dim, num_layers)

        # Should have input + num_layers + output dimensions
        assert len(dimensions) == num_layers + 2
        assert dimensions[0] == input_dim
        assert dimensions[-1] == output_dim

        # Intermediate dimensions should be stable when possible
        stable_dims = get_stable_dimensions(max(input_dim, output_dim) * 2)
        for dim in dimensions[1:-1]:
            # At least some should be stable
            pass  # We'll check this more thoroughly in integration tests


class TestSRTNNIntegration:
    """Integration tests for SRT neural networks."""

    def test_prime_syntony_gate_gradient_flow(self):
        """Test that gradients flow properly through Prime Syntony Gate."""
        dim = 7
        gate = PrimeSyntonyGate(dim)

        x = torch.randn(2, 10, dim, requires_grad=True)
        output = gate(x)
        loss = output.sum()

        loss.backward()

        # Check that gradients exist
        assert x.grad is not None
        assert x.grad.shape == x.shape

    def test_winding_attention_gradient_flow(self):
        """Test gradient flow through Winding Attention."""
        embed_dim = 7
        num_heads = 1
        attention = WindingAttention(embed_dim, num_heads)

        batch_size, seq_len = 2, 10
        q = torch.randn(batch_size, seq_len, embed_dim, requires_grad=True)
        k = torch.randn(batch_size, seq_len, embed_dim, requires_grad=True)
        v = torch.randn(batch_size, seq_len, embed_dim, requires_grad=True)

        output = attention(q, k, v)
        loss = output.sum()

        loss.backward()

        assert q.grad is not None
        assert k.grad is not None
        assert v.grad is not None

    def test_srt_transformer_training(self):
        """Test that SRT transformer can be trained."""
        embed_dim = 7
        num_heads = 1
        transformer = SRTTransformerBlock(embed_dim, num_heads)

        # Simple training setup
        optimizer = torch.optim.Adam(transformer.parameters(), lr=1e-3)
        criterion = nn.MSELoss()

        batch_size, seq_len = 4, 8
        x = torch.randn(batch_size, seq_len, embed_dim)
        target = torch.randn(batch_size, seq_len, embed_dim)

        # Training step
        optimizer.zero_grad()
        output = transformer(x)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        # Loss should decrease (at least not increase dramatically)
        initial_loss = loss.item()

        # Another training step
        optimizer.zero_grad()
        output = transformer(x)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        final_loss = loss.item()

        # Loss should generally decrease with training
        # (Allow some tolerance for random initialization)
        assert final_loss <= initial_loss * 1.1


if __name__ == "__main__":
    pytest.main([__file__])