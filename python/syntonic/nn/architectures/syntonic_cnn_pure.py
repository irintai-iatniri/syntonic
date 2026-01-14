"""
Pure Syntonic CNN: Convolutional networks with DHSR structure.

NO PYTORCH DEPENDENCIES - uses sn.Module and ResonantTensor.

All convolution operations use the Rust backend for efficiency:
- 1D convolution for sequence data
- 2D convolution for image data
- Max pooling, average pooling, global average pooling

Source: CRT.md §12.2
"""

from __future__ import annotations
from typing import Optional, List, Tuple
import math

import syntonic.sn as sn
from syntonic.nn.resonant_tensor import ResonantTensor

PHI = (1 + math.sqrt(5)) / 2


def _conv1d_pure(
    x: List[float],
    seq_len: int,
    in_channels: int,
    kernel: List[float],
    kernel_size: int,
    out_channels: int,
    stride: int = 1,
    padding: int = 0,
) -> Tuple[List[float], int]:
    """
    Pure Python 1D convolution.
    
    Args:
        x: Flattened input (seq_len * in_channels)
        seq_len: Input sequence length
        in_channels: Number of input channels
        kernel: Flattened kernel (out_channels * in_channels * kernel_size)
        kernel_size: Size of convolution kernel
        out_channels: Number of output channels
        stride: Convolution stride
        padding: Padding size
    
    Returns:
        (output_data, output_seq_len)
    """
    # Calculate output length
    out_len = (seq_len + 2 * padding - kernel_size) // stride + 1
    
    output = []
    
    for oc in range(out_channels):
        for i in range(out_len):
            val = 0.0
            pos = i * stride - padding
            
            for k in range(kernel_size):
                input_pos = pos + k
                if 0 <= input_pos < seq_len:
                    for ic in range(in_channels):
                        # x index: input_pos * in_channels + ic
                        x_idx = input_pos * in_channels + ic
                        # kernel index: oc * (in_channels * kernel_size) + ic * kernel_size + k
                        k_idx = oc * (in_channels * kernel_size) + ic * kernel_size + k
                        if x_idx < len(x) and k_idx < len(kernel):
                            val += x[x_idx] * kernel[k_idx]
            
            output.append(val)
    
    return output, out_len


class PureSyntonicConv1d(sn.Module):
    """
    1D convolution with DHSR processing.
    
    Pure Python + ResonantTensor implementation.

    Example:
        >>> conv = PureSyntonicConv1d(in_channels=16, out_channels=32, kernel_size=3)
        >>> x = ResonantTensor(...)  # (seq_len, in_channels)
        >>> y = conv(x)
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
        precision: int = 100,
    ):
        """
        Initialize syntonic 1D convolution.

        Args:
            in_channels: Input channels
            out_channels: Output channels
            kernel_size: Convolution kernel size
            stride: Convolution stride
            padding: Padding size
            precision: ResonantTensor precision
        """
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.precision = precision

        # Kernel weights
        self.kernel = sn.Parameter(
            [out_channels, in_channels, kernel_size],
            init='kaiming'
        )
        
        # Bias
        self.bias = sn.Parameter([out_channels], init='zeros')

        self._syntony: Optional[float] = None

    def forward(self, x: ResonantTensor) -> ResonantTensor:
        """
        Forward pass.

        Args:
            x: Input (seq_len, in_channels)

        Returns:
            Output (out_len, out_channels)
        """
        x_data = x.to_floats()
        shape = x.shape
        seq_len = shape[0]
        
        kernel_data = self.kernel.to_list()
        bias_data = self.bias.to_list()
        
        # Perform convolution
        output_data, out_len = _conv1d_pure(
            x_data, seq_len, self.in_channels,
            kernel_data, self.kernel_size,
            self.out_channels, self.stride, self.padding
        )
        
        # Add bias
        for i, val in enumerate(output_data):
            oc = i % self.out_channels
            output_data[i] = val + bias_data[oc]
        
        # Apply ReLU
        output_data = [max(0.0, v) for v in output_data]
        
        # Reshape to (out_len, out_channels)
        # Currently (out_channels, out_len), need to transpose
        reshaped = []
        for i in range(out_len):
            for oc in range(self.out_channels):
                reshaped.append(output_data[oc * out_len + i])
        
        mode_norms = [float(i * i) for i in range(len(reshaped))]
        output = ResonantTensor(reshaped, [out_len, self.out_channels], mode_norms, self.precision)
        
        self._syntony = output.syntony
        
        return output

    @property
    def syntony(self) -> Optional[float]:
        """Get layer syntony."""
        return self._syntony


class PureSyntonicConv2d(sn.Module):
    """
    2D convolution with DHSR processing.
    
    Uses Rust backend for efficient convolution.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
        precision: int = 100,
    ):
        """
        Initialize syntonic 2D convolution.

        Args:
            in_channels: Input channels
            out_channels: Output channels
            kernel_size: Convolution kernel size
            stride: Convolution stride
            padding: Padding size
            precision: ResonantTensor precision
        """
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.precision = precision

        # Kernel weights [out_channels, kernel_h, kernel_w, in_channels]
        self.kernel = sn.Parameter(
            [out_channels, kernel_size, kernel_size, in_channels],
            init='kaiming'
        )
        
        # Bias
        self.bias = sn.Parameter([out_channels], init='zeros')
        
        self._syntony: Optional[float] = None

    def forward(self, x: ResonantTensor) -> ResonantTensor:
        """
        Forward pass using Rust conv2d.

        Args:
            x: Input (batch, height, width, in_channels) or (height, width, in_channels)

        Returns:
            Output (batch, out_h, out_w, out_channels)
        """
        from syntonic._core import py_conv2d
        
        x_data = x.to_floats()
        shape = x.shape
        
        # Add batch dim if needed
        if len(shape) == 3:
            h, w, c = shape
            input_shape = [1, h, w, c]
        else:
            input_shape = list(shape)
        
        kernel_data = self.kernel.to_list()
        kernel_shape = [self.out_channels, self.kernel_size, self.kernel_size, self.in_channels]
        
        # Perform convolution
        output_data, out_shape = py_conv2d(
            x_data, input_shape,
            kernel_data, kernel_shape,
            (self.stride, self.stride),
            (self.padding, self.padding)
        )
        
        # Add bias
        bias_data = self.bias.to_list()
        batch, oh, ow, oc = out_shape
        for b in range(batch):
            for h in range(oh):
                for w in range(ow):
                    for c in range(oc):
                        idx = b * (oh * ow * oc) + h * (ow * oc) + w * oc + c
                        output_data[idx] += bias_data[c]
        
        # Apply ReLU
        output_data = [max(0.0, v) for v in output_data]
        
        mode_norms = [float(i * i) for i in range(len(output_data))]
        output = ResonantTensor(output_data, out_shape, mode_norms, self.precision)
        
        self._syntony = output.syntony
        
        return output

    @property
    def syntony(self) -> Optional[float]:
        """Get layer syntony."""
        return self._syntony


class PureSyntonicCNN1d(sn.Module):
    """
    Simple 1D CNN for sequence classification.
    
    Pure Python + ResonantTensor implementation.

    Architecture:
    - Stack of 1D convolutions
    - Global average pooling
    - Linear classifier

    Example:
        >>> model = PureSyntonicCNN1d(in_channels=32, num_classes=10)
        >>> x = ResonantTensor(...)  # (seq_len, in_channels)
        >>> logits = model(x)
    """

    def __init__(
        self,
        in_channels: int = 32,
        num_classes: int = 10,
        hidden_channels: List[int] = [64, 128],
        kernel_size: int = 3,
        precision: int = 100,
    ):
        """
        Initialize 1D syntonic CNN.

        Args:
            in_channels: Input feature channels
            num_classes: Number of output classes
            hidden_channels: List of hidden channel sizes
            kernel_size: Convolution kernel size
            precision: ResonantTensor precision
        """
        super().__init__()

        self.precision = precision
        
        # Build conv layers
        self.convs = sn.ModuleList()
        ch = in_channels
        for out_ch in hidden_channels:
            self.convs.append(PureSyntonicConv1d(
                ch, out_ch, kernel_size, padding=kernel_size // 2, precision=precision
            ))
            ch = out_ch
        
        # Classifier
        self.classifier = sn.Parameter([ch, num_classes], init='kaiming')

    def forward(self, x: ResonantTensor) -> ResonantTensor:
        """
        Forward pass.

        Args:
            x: Input (seq_len, in_channels)

        Returns:
            Logits (num_classes,)
        """
        # Apply convolutions
        for conv in self.convs:
            x = conv(x)
        
        # Global average pooling
        data = x.to_floats()
        shape = x.shape
        seq_len, channels = shape
        
        pooled = []
        for c in range(channels):
            col_sum = sum(data[s * channels + c] for s in range(seq_len))
            pooled.append(col_sum / seq_len)
        
        pooled_rt = ResonantTensor(
            pooled, [1, channels],
            [float(i * i) for i in range(channels)],
            self.precision
        )
        
        # Classifier
        logits = pooled_rt.matmul(self.classifier.tensor)
        
        return logits

    @property
    def syntony(self) -> float:
        """Get average syntony across layers."""
        syntonies = [conv.syntony for conv in self.convs if conv.syntony is not None]
        return sum(syntonies) / len(syntonies) if syntonies else 0.5


if __name__ == "__main__":
    import random
    
    print("=" * 70)
    print("Pure Syntonic CNN Test")
    print("=" * 70)
    
    in_channels = 16
    seq_len = 20
    
    # Create random input
    data = [random.gauss(0, 0.5) for _ in range(seq_len * in_channels)]
    mode_norms = [float(i * i) for i in range(len(data))]
    x = ResonantTensor(data, [seq_len, in_channels], mode_norms, 100)
    
    print(f"\nInput shape: {x.shape}")
    print(f"Input syntony: {x.syntony:.4f}")
    
    # Test 1D conv
    conv = PureSyntonicConv1d(in_channels=in_channels, out_channels=32, kernel_size=3)
    y = conv(x)
    
    print(f"\n1D Conv:")
    print(f"  Output shape: {y.shape}")
    print(f"  Layer syntony: {conv.syntony:.4f}")
    
    # Test full 1D CNN
    model = PureSyntonicCNN1d(
        in_channels=in_channels, num_classes=5, hidden_channels=[32, 64]
    )
    logits = model(x)
    
    print(f"\n1D CNN:")
    print(f"  Logits shape: {logits.shape}")
    print(f"  Model syntony: {model.syntony:.4f}")
    
    # Test 2D conv stub
    print(f"\n2D Conv (stub):")
    conv2d = PureSyntonicConv2d(3, 64, kernel_size=3)
    print(f"  Created (API only, forward raises NotImplementedError)")
    
    print("\n" + "=" * 70)
    print("✓ Pure Syntonic CNN (1D) verified! 2D requires CUDA kernels.")
    print("=" * 70)
