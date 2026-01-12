"""
Pure Syntonic Trainer: Training loop with Retrocausal RES.

PURE IMPLEMENTATION: Uses ResonantTensor and Retrocausal RES,
no PyTorch dependencies.

Provides a gradient-free training loop that:
1. Uses Retrocausal RES for weight optimization
2. Tracks syntony throughout training
3. Detects archonic patterns
4. Uses attractor memory for guidance

Source: CRT.md §12.2
"""

from __future__ import annotations
from typing import Optional, List, Dict, Any, Callable, Tuple, Protocol
from dataclasses import dataclass, field
import math
import time

from syntonic._core import (
    ResonantTensor,
    ResonantEvolver,
    RESConfig,
    RESResult,
    py_apply_geodesic_slide,
)
from syntonic.resonant.retrocausal import (
    RetrocausalConfig,
    create_retrocausal_evolver,
)
from syntonic.nn.loss.syntony_metrics import SyntonyTracker

PHI = (1 + math.sqrt(5)) / 2
Q_DEFICIT = 0.027395146920
S_TARGET = PHI - Q_DEFICIT


class PureModel(Protocol):
    """Protocol for pure syntonic models."""
    def forward(self, x: ResonantTensor) -> ResonantTensor: ...
    def get_weights(self) -> List[ResonantTensor]: ...
    def set_weights(self, weights: List[ResonantTensor]) -> None: ...
    @property
    def syntony(self) -> float: ...


@dataclass
class RESTrainingConfig:
    """Configuration for Retrocausal RES training."""
    
    # Evolution parameters
    max_generations: int = 500
    population_size: int = 32
    syntony_threshold: float = 0.95
    
    # Retrocausal parameters
    attractor_capacity: int = 32
    pull_strength: float = 0.3
    attractor_min_syntony: float = 0.7
    attractor_decay_rate: float = 0.98
    noise_scale: float = 0.1
    
    # Syntony targets
    syntony_target: float = S_TARGET
    
    # Training options
    use_archonic_detection: bool = True
    log_interval: int = 10
    eval_interval: int = 1
    
    # Loss parameters
    lambda_syntony: float = 0.1
    mu_phase: float = 0.01


class RetrocausalTrainer:
    """
    Pure training loop using Retrocausal RES.

    Handles:
    - Weight evolution via attractor-guided RES
    - Syntony computation and tracking
    - Archonic pattern detection
    - Retrocausal guidance

    Example:
        >>> from syntonic.nn.training import RetrocausalTrainer, RESTrainingConfig
        >>> config = RESTrainingConfig(max_generations=200)
        >>> trainer = RetrocausalTrainer(model, train_data, config=config)
        >>> result = trainer.train()
        >>> print(f"Final syntony: {result['final_syntony']:.4f}")
    """

    def __init__(
        self,
        model: PureModel,
        train_data: List[Tuple[ResonantTensor, ResonantTensor]],
        val_data: Optional[List[Tuple[ResonantTensor, ResonantTensor]]] = None,
        config: Optional[RESTrainingConfig] = None,
        loss_fn: Optional[Callable[[ResonantTensor, ResonantTensor], float]] = None,
        callbacks: Optional[List[Any]] = None,
    ):
        """
        Initialize trainer.

        Args:
            model: Pure syntonic model to train
            train_data: Training data as list of (input, target) tuples
            val_data: Validation data
            config: Training configuration
            loss_fn: Loss function (default: MSE + syntony)
            callbacks: List of callbacks
        """
        self.config = config or RESTrainingConfig()
        self.model = model
        self.train_data = train_data
        self.val_data = val_data
        self.callbacks = callbacks or []
        
        # Loss function
        if loss_fn is None:
            from syntonic.nn.loss.syntonic_loss import mse_loss
            self.loss_fn = mse_loss
        else:
            self.loss_fn = loss_fn
        
        # Tracking
        self._syntony_tracker = SyntonyTracker(window_size=100)
        self._loss_history: List[float] = []
        self._current_syntony = 0.5
        self._current_generation = 0
        
        # Create evolver template from model weights
        self._weight_templates = model.get_weights()

    def train(self) -> Dict[str, Any]:
        """
        Run Retrocausal RES training.

        Returns:
            Training result dictionary
        """
        start_time = time.time()
        
        # Create retrocausal config
        res_config = RetrocausalConfig(
            population_size=self.config.population_size,
            max_generations=self.config.max_generations,
            # Note: syntony_threshold is used for evaluation, not RES evolution
            attractor_capacity=self.config.attractor_capacity,
            attractor_pull_strength=self.config.pull_strength,
            attractor_min_syntony=self.config.attractor_min_syntony,
            attractor_decay_rate=self.config.attractor_decay_rate,
        ).to_res_config()
        
        # Evolve each weight tensor
        all_results = []
        evolved_weights = []
        
        for i, template in enumerate(self._weight_templates):
            # Create evolver for this weight
            evolver = create_retrocausal_evolver(
                template=template,
                population_size=self.config.population_size,
                attractor_capacity=self.config.attractor_capacity,
                pull_strength=self.config.pull_strength,
                min_syntony=self.config.attractor_min_syntony,
                decay_rate=self.config.attractor_decay_rate,
            )
            
            # Run evolution with fitness function
            result = self._evolve_weight(evolver, i)
            all_results.append(result)
            evolved_weights.append(result['best_tensor'])
            
            if i % self.config.log_interval == 0:
                self._log_progress(i, len(self._weight_templates), result)
        
        # Apply evolved weights to model
        self.model.set_weights(evolved_weights)
        
        # Final validation
        final_loss, final_syntony = self._evaluate()
        
        elapsed_time = time.time() - start_time
        
        return {
            'final_syntony': final_syntony,
            'final_loss': final_loss,
            'generations': sum(r['generations'] for r in all_results) / len(all_results),
            'elapsed_time': elapsed_time,
            'weight_results': all_results,
            'syntony_history': self._syntony_tracker.history,
            'loss_history': self._loss_history,
        }

    def _evolve_weight(
        self,
        evolver: ResonantEvolver,
        weight_idx: int,
    ) -> Dict[str, Any]:
        """
        Evolve a single weight tensor using Geodesic Gravity.
        """
        # 1. Run Standard RES Evolution
        result: RESResult = evolver.run()
        best_tensor = result.winner
        
        # 2. Apply Geodesic Gravity Slide (The "Physical Lock")
        # Only apply if we have a valid attractor (template)
        if self.config.pull_strength > 0:
             # Calculate Physics Parameters
             # High Syntony = Low Temp (Freeze). Low Syntony = High Temp (Melt).
             temp = (1.0 - result.final_syntony) * self.config.noise_scale
             gravity = self.config.pull_strength * PHI # Scale by Phi
             
             # Get the attractor (using the template/history as the guide)
             attractor = self._weight_templates[weight_idx]
             
             # Create mode norms if they don't exist (needed for metric)
             if not hasattr(self, '_mode_norms_cache'):
                 self._mode_norms_cache = {}
             
             w_shape = best_tensor.shape
             if w_shape not in self._mode_norms_cache:
                 # Helper to get norms on correct device
                 from syntonic._core import py_standard_mode_norms
                 self._mode_norms_cache[w_shape] = py_standard_mode_norms(w_shape, best_tensor.device)
             
             norms = self._mode_norms_cache[w_shape]

             # Apply the Kernel via Rust Bridge
             try:
                 py_apply_geodesic_slide(
                     best_tensor._storage,
                     attractor._storage,
                     norms._storage,
                     gravity,
                     temp
                 )
             except Exception as e:
                 print(f"Warning: Geodesic slide skipped: {e}")
        
        self._syntony_tracker.update(result.final_syntony)
        
        return {
            'weight_idx': weight_idx,
            'final_syntony': result.final_syntony,
            'generations': result.generations,
            'converged': result.converged,
            'best_tensor': result.winner,
            'is_archonic': getattr(result, 'is_archonic', False)
        }

    def _evaluate(self) -> Tuple[float, float]:
        """
        Evaluate model on training data.
        
        Returns:
            (loss, syntony)
        """
        total_loss = 0.0
        n_samples = 0
        
        for inputs, targets in self.train_data:
            outputs = self.model.forward(inputs)
            loss = self.loss_fn(outputs, targets)
            # Convert loss to float if it's a ResonantTensor
            if hasattr(loss, 'to_floats'):
                loss_val = loss.to_floats()[0]
            else:
                loss_val = float(loss)
            total_loss += loss_val
            n_samples += 1
        
        avg_loss = total_loss / max(1, n_samples)
        model_syntony = self.model.syntony
        
        self._loss_history.append(avg_loss)
        return avg_loss, model_syntony

    def _log_progress(
        self,
        current: int,
        total: int,
        result: Dict[str, Any],
    ):
        """Log training progress."""
        for callback in self.callbacks:
            if hasattr(callback, 'on_weight_evolved'):
                callback.on_weight_evolved(
                    self,
                    current,
                    total,
                    result,
                )

    @property
    def current_syntony(self) -> float:
        """Get current model syntony."""
        return self._current_syntony

    @property
    def syntony_history(self) -> List[float]:
        """Get syntony history."""
        return self._syntony_tracker.history

    @property
    def is_archonic(self) -> bool:
        """Check if training is in archonic pattern."""
        return self._syntony_tracker.is_archonic()


# Convenience function
def train_with_retrocausal_res(
    model: PureModel,
    train_data: List[Tuple[ResonantTensor, ResonantTensor]],
    max_generations: int = 500,
    population_size: int = 32,
    pull_strength: float = 0.3,
) -> Dict[str, Any]:
    """
    Quick training function with sensible defaults.
    
    Args:
        model: Pure syntonic model
        train_data: Training data
        max_generations: Maximum generations per weight
        population_size: Population size for RES
        pull_strength: Retrocausal pull strength
    
    Returns:
        Training results
    """
    config = RESTrainingConfig(
        max_generations=max_generations,
        population_size=population_size,
        pull_strength=pull_strength,
    )
    trainer = RetrocausalTrainer(model, train_data, config=config)
    return trainer.train()


# Aliases
PureTrainer = RetrocausalTrainer
PureTrainingConfig = RESTrainingConfig


if __name__ == "__main__":
    """Test pure trainer."""
    print("Testing Pure Retrocausal Trainer...")
    print("Note: Full test requires a PureModel implementation.")
    
    # Verify imports work
    config = RESTrainingConfig()
    print(f"Config created: population_size={config.population_size}")
    print(f"  max_generations={config.max_generations}")
    print(f"  pull_strength={config.pull_strength}")
    print(f"  syntony_target={config.syntony_target:.4f}")
    
    print("✅ Pure Retrocausal Trainer imports verified!")
