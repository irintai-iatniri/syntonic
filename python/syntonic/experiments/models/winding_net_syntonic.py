"""
WindingNetSyntonic - PureWindingNet with configurable syntonic softmax.

Extends PureWindingNet to support:
1. Multiple softmax modes (identity, learned, provided, random)
2. E6 root interpolation for num_classes > 36
3. Different mode norm sources (e6, random_uniform, random_golden, etc.)
4. Dual baseline support (PyTorch-style vs syntonic identity)
"""

from typing import List, Optional
import math
import random

from syntonic._core import ResonantTensor, WindingState, SyntonicSoftmaxState, SyntonicSoftmaxMode
from syntonic.nn.winding.winding_net_pure import PureWindingNet

PHI = 1.618033988749895


class WindingNetSyntonic(PureWindingNet):
    """
    PureWindingNet with configurable syntonic softmax.

    Adds optional syntonic softmax layer after output projection.
    """

    def __init__(
        self,
        max_winding: int = 5,
        base_dim: int = 64,
        num_blocks: int = 3,
        output_dim: int = 2,
        softmax_mode: str = "identity",  # "identity", "learned", "provided", "pytorch"
        mode_norm_source: str = "e6",    # "e6", "random_uniform", "random_golden", etc.
        syntony_scale: float = 1.0,
        use_prime_filter: bool = True,
        consensus_threshold: float = 0.024,
        dropout: float = 0.1,
        precision: int = 100,
    ):
        """
        Initialize WindingNetSyntonic.

        Args:
            max_winding: Maximum winding number for enumeration
            base_dim: Base embedding dimension
            num_blocks: Number of DHSR blocks
            output_dim: Number of output classes
            softmax_mode: Softmax variant to use:
                - "identity": SyntonicSoftmaxMode.Identity (standard softmax)
                - "learned": SyntonicSoftmaxMode.Learned (E8 mode norms)
                - "provided": SyntonicSoftmaxMode.Provided (requires syntony tensor)
                - "pytorch": Pure PyTorch-style softmax (ResonantTensor.softmax)
            mode_norm_source: Source for mode norms (only used if softmax_mode="learned"):
                - "e6": 36 E6 roots from Golden Cone
                - "random_uniform": Random 36 vectors, uniform distribution
                - "random_golden": Random 36 vectors, golden distribution
            syntony_scale: Scale factor for syntonic softmax
            use_prime_filter: Whether to use prime selection in blocks
            consensus_threshold: ΔS threshold for block validation
            dropout: Dropout rate
            precision: ResonantTensor precision
        """
        super().__init__(
            max_winding=max_winding,
            base_dim=base_dim,
            num_blocks=num_blocks,
            output_dim=output_dim,
            use_prime_filter=use_prime_filter,
            consensus_threshold=consensus_threshold,
            dropout=dropout,
            precision=precision,
        )

        self.softmax_mode = softmax_mode
        self.mode_norm_source = mode_norm_source
        self.syntony_scale = syntony_scale
        self.softmax_state = None
        self.softmax_syntony = 0.0  # Track softmax-specific syntony

        # Create syntonic softmax state if needed
        if softmax_mode == "learned":
            self.softmax_state = self._create_softmax_state(mode_norm_source, output_dim)
        elif softmax_mode == "identity":
            self.softmax_state = SyntonicSoftmaxState(
                SyntonicSoftmaxMode.Identity,
                -1,  # dim (last dimension)
                None,  # num_features (not needed for identity)
                syntony_scale
            )
        # For "provided" and "pytorch", we don't create state here

    def _create_softmax_state(
        self,
        source: str,
        num_features: int
    ) -> SyntonicSoftmaxState:
        """
        Create SyntonicSoftmaxState with specified mode norm source.

        When num_features > 36, we interpolate between the 36 E6 roots
        using linear interpolation in 8D space.

        Args:
            source: Mode norm source ("e6", "random_uniform", "random_golden")
            num_features: Number of output classes

        Returns:
            SyntonicSoftmaxState configured with appropriate mode norms
        """
        if source == "e6":
            # Use 36 E6 roots from Golden Cone (already in Rust backend)
            # The SyntonicSoftmaxState constructor will generate them
            pass
        elif source == "random_uniform":
            # TODO: Generate random 36 vectors with uniform distribution
            # For now, fall back to e6
            pass
        elif source == "random_golden":
            # TODO: Generate random 36 vectors with golden distribution
            # For now, fall back to e6
            pass

        # Create state with learned mode
        # Note: E6 root interpolation happens automatically in Rust backend
        # when num_features > 36 (cycling through roots)
        return SyntonicSoftmaxState(
            SyntonicSoftmaxMode.Learned,
            -1,  # dim (last dimension)
            num_features,
            self.syntony_scale
        )

    def forward(
        self,
        winding_states: List[WindingState],
        syntony_tensor: Optional[ResonantTensor] = None
    ) -> ResonantTensor:
        """
        Forward pass with optional syntonic softmax.

        Args:
            winding_states: List of WindingState objects
            syntony_tensor: Optional syntony values for "provided" mode

        Returns:
            Output probabilities (after softmax)
        """
        # 1. Get logits from parent forward (PureWindingNet)
        logits = super().forward(winding_states)

        # 2. Apply softmax based on mode
        if self.softmax_mode == "pytorch":
            # Pure ResonantTensor softmax (baseline) - in-place operation
            logits.softmax(precision=self.precision)
            return logits

        elif self.softmax_mode in ["identity", "learned"]:
            # Use syntonic softmax state
            output = self.softmax_state.forward(logits, None)
            # TODO: Extract softmax-specific syntony when Rust backend supports it
            self.softmax_syntony = 0.0
            return output

        elif self.softmax_mode == "provided":
            # Use provided syntony tensor
            if syntony_tensor is None:
                raise ValueError("softmax_mode='provided' requires syntony_tensor")

            state = SyntonicSoftmaxState(
                SyntonicSoftmaxMode.Provided,
                -1,
                self.output_dim,
                self.syntony_scale
            )
            output = state.forward(logits, syntony_tensor)
            self.softmax_syntony = 0.0
            return output

        else:
            raise ValueError(f"Unknown softmax_mode: {self.softmax_mode}")

    def get_all_syntonies(self) -> dict:
        """
        Get syntony values at all three levels.

        Returns:
            Dictionary with:
            - "network": Overall network syntony
            - "layers": List of per-layer syntonies
            - "softmax": Softmax-specific syntony
        """
        return {
            "network": self.network_syntony,
            "layers": self.layer_syntonies,
            "softmax": self.softmax_syntony,
        }


def create_baseline_model(
    output_dim: int,
    max_winding: int = 5,
    base_dim: int = 64,
    num_blocks: int = 3,
    precision: int = 100,
    baseline_type: str = "pytorch"
) -> WindingNetSyntonic:
    """
    Create a baseline model for comparison.

    Args:
        output_dim: Number of output classes
        max_winding: Maximum winding number
        base_dim: Base embedding dimension
        num_blocks: Number of DHSR blocks
        precision: ResonantTensor precision
        baseline_type: Type of baseline:
            - "pytorch": Standard softmax via ResonantTensor.softmax()
            - "identity": SyntonicSoftmaxMode.Identity

    Returns:
        WindingNetSyntonic configured as baseline
    """
    return WindingNetSyntonic(
        max_winding=max_winding,
        base_dim=base_dim,
        num_blocks=num_blocks,
        output_dim=output_dim,
        softmax_mode=baseline_type,
        mode_norm_source="e6",  # Doesn't matter for baselines
        precision=precision,
    )


def create_syntonic_model(
    output_dim: int,
    max_winding: int = 5,
    base_dim: int = 64,
    num_blocks: int = 3,
    precision: int = 100,
    mode_norm_source: str = "e6",
    syntony_scale: float = 1.0
) -> WindingNetSyntonic:
    """
    Create a syntonic model with E8-based softmax.

    Args:
        output_dim: Number of output classes
        max_winding: Maximum winding number
        base_dim: Base embedding dimension
        num_blocks: Number of DHSR blocks
        precision: ResonantTensor precision
        mode_norm_source: Source for E8 roots ("e6", "random_uniform", etc.)
        syntony_scale: Syntony scaling factor

    Returns:
        WindingNetSyntonic configured with E8 softmax
    """
    return WindingNetSyntonic(
        max_winding=max_winding,
        base_dim=base_dim,
        num_blocks=num_blocks,
        output_dim=output_dim,
        softmax_mode="learned",
        mode_norm_source=mode_norm_source,
        syntony_scale=syntony_scale,
        precision=precision,
    )


__all__ = [
    'WindingNetSyntonic',
    'create_baseline_model',
    'create_syntonic_model',
]
