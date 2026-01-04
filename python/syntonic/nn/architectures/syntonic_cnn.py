"""
Syntonic CNN: Convolutional networks with DHSR structure.

Convolutional layers followed by DHSR processing for
syntonic feature extraction.

Source: CRT.md §12.2
"""

from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List, Tuple, Union
import math

from syntonic.nn.layers import (
    DifferentiationLayer,
    HarmonizationLayer,
    RecursionBlock,
    SyntonicNorm,
)

PHI = (1 + math.sqrt(5)) / 2


class SyntonicConv2d(nn.Module):
    """
    2D convolution with DHSR processing.

    Applies convolution followed by differentiation
    and harmonization in channel dimension.

    Example:
        >>> conv = SyntonicConv2d(3, 64, kernel_size=3)
        >>> x = torch.randn(32, 3, 28, 28)
        >>> y = conv(x)
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, Tuple[int, int]] = 3,
        stride: Union[int, Tuple[int, int]] = 1,
        padding: Union[int, Tuple[int, int]] = 1,
        use_recursion: bool = True,
    ):
        """
        Initialize syntonic convolution.

        Args:
            in_channels: Input channels
            out_channels: Output channels
            kernel_size: Convolution kernel size
            stride: Convolution stride
            padding: Padding size
            use_recursion: Use RecursionBlock
        """
        super().__init__()

        self.conv = nn.Conv2d(
            in_channels, out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
        )

        # DHSR in channel dimension
        if use_recursion:
            self.recursion = RecursionBlock(out_channels)
        else:
            self.diff = DifferentiationLayer(out_channels, out_channels)
            self.harm = HarmonizationLayer(out_channels, out_channels)

        self.use_recursion = use_recursion
        self.norm = nn.BatchNorm2d(out_channels)

        self._syntony: Optional[float] = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input (batch, channels, height, width)

        Returns:
            Output (batch, out_channels, h', w')
        """
        # Convolution
        x = self.conv(x)
        x = self.norm(x)

        # DHSR processing (applied to channel dimension)
        batch, channels, height, width = x.shape

        # Reshape for DHSR: (batch*height*width, channels)
        x_flat = x.permute(0, 2, 3, 1).reshape(-1, channels)

        if self.use_recursion:
            x_flat = self.recursion(x_flat)
            self._syntony = self.recursion.syntony
        else:
            x_diff = self.diff(x_flat)
            x_harm = self.harm(x_diff)
            x_flat = x_harm
            self._syntony = self._compute_syntony(x_flat, x_diff, x_harm)

        # Reshape back
        x = x_flat.reshape(batch, height, width, channels).permute(0, 3, 1, 2)

        return x

    def _compute_syntony(self, x, x_diff, x_harm) -> float:
        """Compute block syntony."""
        with torch.no_grad():
            diff_norm = torch.norm(x_diff - x).item()
            harm_diff_norm = torch.norm(x_diff - x_harm).item()
            S = 1.0 - diff_norm / (harm_diff_norm + 1e-8)
            return max(0.0, min(1.0, S))

    @property
    def syntony(self) -> Optional[float]:
        """Get layer syntony."""
        return self._syntony


class RecursionConvBlock(nn.Module):
    """
    Convolutional block with full DHSR cycle.

    Conv → BN → D → H → Activation

    Example:
        >>> block = RecursionConvBlock(64, 128)
        >>> x = torch.randn(32, 64, 14, 14)
        >>> y = block(x)
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
        downsample: bool = False,
    ):
        """
        Initialize recursion conv block.

        Args:
            in_channels: Input channels
            out_channels: Output channels
            kernel_size: Kernel size
            stride: Convolution stride
            padding: Padding
            downsample: Apply 2x spatial downsampling
        """
        super().__init__()

        if downsample:
            stride = 2
            padding = kernel_size // 2

        self.conv1 = nn.Conv2d(
            in_channels, out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
        )
        self.bn1 = nn.BatchNorm2d(out_channels)

        self.conv2 = nn.Conv2d(
            out_channels, out_channels,
            kernel_size=kernel_size,
            stride=1,
            padding=padding,
        )
        self.bn2 = nn.BatchNorm2d(out_channels)

        # DHSR layers
        self.diff = DifferentiationLayer(out_channels, out_channels)
        self.harm = HarmonizationLayer(out_channels, out_channels)

        # Residual connection
        self.residual = nn.Sequential()
        if in_channels != out_channels or downsample:
            self.residual = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride=stride),
                nn.BatchNorm2d(out_channels),
            )

        self._syntony: Optional[float] = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with DHSR."""
        identity = self.residual(x)

        # First conv
        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out)

        # Second conv
        out = self.conv2(out)
        out = self.bn2(out)

        # DHSR processing
        batch, channels, height, width = out.shape
        out_flat = out.permute(0, 2, 3, 1).reshape(-1, channels)

        x_orig = out_flat.clone()
        out_diff = self.diff(out_flat)
        out_harm = self.harm(out_diff)

        self._syntony = self._compute_syntony(x_orig, out_diff, out_harm)

        out = out_harm.reshape(batch, height, width, channels).permute(0, 3, 1, 2)

        # Residual connection (golden scaled)
        out = out + identity / PHI

        return F.relu(out)

    def _compute_syntony(self, x, x_diff, x_harm) -> float:
        """Compute block syntony."""
        with torch.no_grad():
            diff_norm = torch.norm(x_diff - x).item()
            harm_diff_norm = torch.norm(x_diff - x_harm).item()
            S = 1.0 - diff_norm / (harm_diff_norm + 1e-8)
            return max(0.0, min(1.0, S))

    @property
    def syntony(self) -> Optional[float]:
        """Get block syntony."""
        return self._syntony


class SyntonicCNN(nn.Module):
    """
    Complete syntonic CNN for image classification.

    Architecture:
    - Initial convolution
    - Stack of RecursionConvBlocks
    - Global average pooling
    - Syntonic MLP head

    Example:
        >>> model = SyntonicCNN(num_classes=10)
        >>> x = torch.randn(32, 3, 32, 32)
        >>> y = model(x)
        >>> print(f"Syntony: {model.syntony:.4f}")
    """

    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = 10,
        base_channels: int = 64,
        num_blocks: List[int] = [2, 2, 2, 2],
    ):
        """
        Initialize syntonic CNN.

        Args:
            in_channels: Input image channels
            num_classes: Number of output classes
            base_channels: Base channel count
            num_blocks: Blocks per stage [stage1, stage2, ...]
        """
        super().__init__()

        self.in_channels = in_channels
        self.num_classes = num_classes

        # Initial convolution
        self.conv1 = nn.Conv2d(in_channels, base_channels, 7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm2d(base_channels)
        self.pool1 = nn.MaxPool2d(3, stride=2, padding=1)

        # Build stages
        self.stages = nn.ModuleList()
        in_ch = base_channels

        for stage_idx, n_blocks in enumerate(num_blocks):
            out_ch = base_channels * (2 ** stage_idx)
            downsample = stage_idx > 0

            blocks = []
            for block_idx in range(n_blocks):
                blocks.append(RecursionConvBlock(
                    in_ch if block_idx == 0 else out_ch,
                    out_ch,
                    downsample=downsample and block_idx == 0,
                ))
            self.stages.append(nn.Sequential(*blocks))
            in_ch = out_ch

        # Global pooling and classifier
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(in_ch, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        # Initial layers
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.pool1(x)

        # Stages
        for stage in self.stages:
            x = stage(x)

        # Classifier
        x = self.pool(x)
        x = x.flatten(1)
        x = self.fc(x)

        return x

    @property
    def syntony(self) -> float:
        """Get average syntony across all blocks."""
        syntonies = []
        for stage in self.stages:
            for block in stage:
                if hasattr(block, 'syntony') and block.syntony is not None:
                    syntonies.append(block.syntony)
        return sum(syntonies) / len(syntonies) if syntonies else 0.5

    def get_feature_maps(self, x: torch.Tensor) -> List[torch.Tensor]:
        """Get intermediate feature maps."""
        features = []

        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.pool1(x)
        features.append(x)

        for stage in self.stages:
            x = stage(x)
            features.append(x)

        return features
