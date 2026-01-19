"""
Fitness function wrappers for Resonant Evolution Strategy.

These classes wrap task-specific losses (classification, regression)
as fitness functions for use with ResonantEvolver. The pattern is:
- Fitness is computed externally in Python
- RES uses syntony for pre-filtering (cheap CPU check)
- Final score = task_fitness + lambda * syntony

Key insight: Higher fitness = better (RES maximizes).
So we negate losses: fitness = -loss
"""

import math
import random
from typing import Callable, List, Optional, Tuple

# Import ResonantTensor from the core module
from syntonic._core import ResonantTensor

# Universal syntony deficit - NOT a hyperparameter!
Q_DEFICIT = 0.027395146920


def matrix_multiply(A: List[List[float]], B: List[List[float]]) -> List[List[float]]:
    """
    Pure Python matrix multiplication: C = A @ B.

    Args:
        A: Matrix of shape (m, n)
        B: Matrix of shape (n, p)

    Returns:
        C: Matrix of shape (m, p)
    """
    m = len(A)
    n = len(A[0])
    p = len(B[0])

    result = [[0.0] * p for _ in range(m)]
    for i in range(m):
        for j in range(p):
            for k in range(n):
                result[i][j] += A[i][k] * B[k][j]

    return result


def softmax(x: List[List[float]]) -> List[List[float]]:
    """Numerically stable softmax for 2D list."""
    result = []
    for row in x:
        # Find max for numerical stability
        max_val = max(row)
        # Compute exp(x - max)
        exp_vals = [math.exp(val - max_val) for val in row]
        # Normalize
        sum_exp = sum(exp_vals)
        result.append([e / sum_exp for e in exp_vals])
    return result


def cross_entropy(logits: List[List[float]], labels: List[int]) -> float:
    """Cross-entropy loss for classification."""
    probs = softmax(logits)
    n = len(labels)

    # Compute log probabilities with clipping for numerical stability
    log_sum = 0.0
    for i in range(n):
        prob = max(probs[i][labels[i]], 1e-10)  # Clip
        log_sum += math.log(prob)

    return -log_sum / n


def accuracy(logits: List[List[float]], labels: List[int]) -> float:
    """Classification accuracy."""
    correct = 0
    for i in range(len(labels)):
        pred_class = max(range(len(logits[i])), key=lambda j: logits[i][j])
        if pred_class == labels[i]:
            correct += 1
    return correct / len(labels)


class ClassificationFitness:
    """
    Fitness wrapper for classification tasks.

    Uses a simple linear model: logits = X @ W
    where W is extracted from the ResonantTensor.

    The fitness is negative cross-entropy (higher = better).
    Combined score: fitness + Q_DEFICIT * syntony

    Example:
        >>> fitness = ClassificationFitness(X_train, y_train, n_features=2, n_classes=2)
        >>> score = fitness(tensor)  # Returns combined score
        >>> acc = fitness.accuracy(tensor)  # Returns accuracy
    """

    def __init__(
        self,
        X: List[List[float]],
        y: List[int],
        n_features: Optional[int] = None,
        n_classes: Optional[int] = None,
        lambda_syntony: float = Q_DEFICIT,
    ):
        """
        Initialize classification fitness.

        Args:
            X: Training features as list of lists, shape (n_samples, n_features)
            y: Training labels as list of ints, shape (n_samples,)
            n_features: Number of input features (inferred if None)
            n_classes: Number of output classes (inferred if None)
            lambda_syntony: Weight for syntony in combined score
        """
        self.X = X
        self.y = y
        self.n_features = n_features or len(X[0])
        self.n_classes = n_classes or (max(y) + 1)
        self.lambda_syntony = lambda_syntony

        # Expected weight shape
        self.weight_size = self.n_features * self.n_classes

    def _extract_weights(self, tensor: ResonantTensor) -> List[List[float]]:
        """Extract weight matrix from tensor as list of lists."""
        values = tensor.to_list()

        # Pad with zeros if needed
        if len(values) < self.weight_size:
            values = values + [0.0] * (self.weight_size - len(values))

        # Take first weight_size values and reshape to (n_features, n_classes)
        values = values[: self.weight_size]
        W = []
        for i in range(self.n_features):
            row = values[i * self.n_classes : (i + 1) * self.n_classes]
            W.append(row)

        return W

    def loss(self, tensor: ResonantTensor) -> float:
        """Compute cross-entropy loss."""
        W = self._extract_weights(tensor)
        logits = matrix_multiply(self.X, W)
        return cross_entropy(logits, self.y)

    def accuracy(self, tensor: ResonantTensor) -> float:
        """Compute classification accuracy."""
        W = self._extract_weights(tensor)
        logits = matrix_multiply(self.X, W)
        return accuracy(logits, self.y)

    def task_fitness(self, tensor: ResonantTensor) -> float:
        """Task fitness (negative loss, higher = better)."""
        return -self.loss(tensor)

    def __call__(self, tensor: ResonantTensor) -> float:
        """
        Combined fitness: task_fitness + lambda * syntony.

        This is the score used by RES for selection.
        """
        task_fit = self.task_fitness(tensor)
        syntony = tensor.syntony
        return task_fit + self.lambda_syntony * syntony

    def evaluate_population(
        self,
        tensors: List[ResonantTensor],
    ) -> List[Tuple[float, float, float]]:
        """
        Evaluate fitness for a population of tensors.

        Returns list of (combined_score, task_fitness, syntony) tuples.
        """
        results = []
        for t in tensors:
            task_fit = self.task_fitness(t)
            syntony = t.syntony
            combined = task_fit + self.lambda_syntony * syntony
            results.append((combined, task_fit, syntony))
        return results


class RegressionFitness:
    """
    Fitness wrapper for regression tasks.

    Uses a simple linear model: y_pred = X @ W
    where W is extracted from the ResonantTensor.

    The fitness is negative MSE (higher = better).
    Combined score: fitness + Q_DEFICIT * syntony
    """

    def __init__(
        self,
        X: List[List[float]],
        y: List[float],
        n_features: Optional[int] = None,
        n_outputs: int = 1,
        lambda_syntony: float = Q_DEFICIT,
    ):
        """
        Initialize regression fitness.

        Args:
            X: Training features as list of lists, shape (n_samples, n_features)
            y: Training targets as list, shape (n_samples,) or (n_samples, n_outputs)
            n_features: Number of input features (inferred if None)
            n_outputs: Number of output dimensions
            lambda_syntony: Weight for syntony in combined score
        """
        self.X = X
        # Ensure y is 2D
        if isinstance(y[0], (int, float)):
            self.y = [[val] for val in y]
        else:
            self.y = y
        self.n_features = n_features or len(X[0])
        self.n_outputs = n_outputs
        self.lambda_syntony = lambda_syntony

        self.weight_size = self.n_features * self.n_outputs

    def _extract_weights(self, tensor: ResonantTensor) -> List[List[float]]:
        """Extract weight matrix from tensor."""
        values = tensor.to_list()

        # Pad with zeros if needed
        if len(values) < self.weight_size:
            values = values + [0.0] * (self.weight_size - len(values))

        # Reshape to (n_features, n_outputs)
        values = values[: self.weight_size]
        W = []
        for i in range(self.n_features):
            row = values[i * self.n_outputs : (i + 1) * self.n_outputs]
            W.append(row)

        return W

    def loss(self, tensor: ResonantTensor) -> float:
        """Compute MSE loss."""
        W = self._extract_weights(tensor)
        y_pred = matrix_multiply(self.X, W)

        # Compute MSE
        total_error = 0.0
        n_samples = len(y_pred)
        n_outputs = len(y_pred[0])

        for i in range(n_samples):
            for j in range(n_outputs):
                error = y_pred[i][j] - self.y[i][j]
                total_error += error * error

        return total_error / n_samples

    def task_fitness(self, tensor: ResonantTensor) -> float:
        """Task fitness (negative MSE, higher = better)."""
        return -self.loss(tensor)

    def __call__(self, tensor: ResonantTensor) -> float:
        """Combined fitness: task_fitness + lambda * syntony."""
        task_fit = self.task_fitness(tensor)
        syntony = tensor.syntony
        return task_fit + self.lambda_syntony * syntony


class WavefunctionFitness:
    """
    Fitness for winding state recovery in T^4.

    Measures distance to target winding numbers,
    demonstrating exact lattice preservation.
    """

    def __init__(
        self,
        target_windings: Tuple[int, int, int, int],
        lambda_syntony: float = Q_DEFICIT,
    ):
        """
        Initialize wavefunction fitness.

        Args:
            target_windings: Target (n7, n8, n9, n10) winding numbers
            lambda_syntony: Weight for syntony in combined score
        """
        self.target = list(target_windings)
        self.lambda_syntony = lambda_syntony

    def winding_distance(self, tensor: ResonantTensor) -> float:
        """Compute L2 distance to target windings."""
        values = tensor.to_list()[:4]

        # Pad with zeros if needed
        if len(values) < 4:
            values = values + [0.0] * (4 - len(values))

        # Compute L2 distance
        sum_sq = sum((values[i] - self.target[i]) ** 2 for i in range(4))
        return math.sqrt(sum_sq)

    def task_fitness(self, tensor: ResonantTensor) -> float:
        """Task fitness (negative distance, higher = better)."""
        return -self.winding_distance(tensor)

    def __call__(self, tensor: ResonantTensor) -> float:
        """Combined fitness: task_fitness + lambda * syntony."""
        task_fit = self.task_fitness(tensor)
        syntony = tensor.syntony
        return task_fit + self.lambda_syntony * syntony


def evolve_with_fitness(
    evolver,
    fitness_fn: Callable[[ResonantTensor], float],
    generations: int = 100,
    verbose: bool = False,
) -> dict:
    """
    Run RES evolution with external fitness function.

    This wraps the evolver's step() with external fitness evaluation.
    Pattern: step() -> evaluate best with fitness -> track progress

    Args:
        evolver: ResonantEvolver instance
        fitness_fn: Fitness function (tensor -> float)
        generations: Number of generations to run
        verbose: Print progress

    Returns:
        Dict with 'best_tensor', 'best_fitness', 'history'
    """
    history = {
        "syntony": [],
        "fitness": [],
        "generation": [],
    }

    best_fitness = float("-inf")
    best_tensor = None

    for gen in range(generations):
        # RES step uses syntony internally
        syntony = evolver.step()

        # Evaluate fitness externally
        current = evolver.best
        if current is not None:
            fitness = fitness_fn(current)

            if fitness > best_fitness:
                best_fitness = fitness
                best_tensor = current

            history["syntony"].append(syntony)
            history["fitness"].append(fitness)
            history["generation"].append(gen)

            if verbose and gen % 10 == 0:
                print(f"Gen {gen}: syntony={syntony:.4f}, fitness={fitness:.4f}")

    return {
        "best_tensor": best_tensor,
        "best_fitness": best_fitness,
        "final_syntony": evolver.best_syntony,
        "generations": generations,
        "history": history,
    }


class FitnessGuidedEvolver:
    """
    Resonant Evolution Strategy (RES) with external fitness function.

    This implements the full RES algorithm per the SRT specification:

    STEP 1: MUTATION (CPU)
        Generate mutants by perturbing in Q(φ), snap to lattice

    STEP 2: SYNTONY FILTER (CPU, cheap)
        Keep top ~25% by lattice syntony (rejects 75% before expensive eval)

    STEP 3: DHSR CYCLE (CPU or GPU)  ◀── THIS WAS MISSING IN ORIGINAL
        For each survivor:
        - Apply D̂ (differentiation with noise)
        - Apply Ĥ (harmonization attenuation)
        - Crystallize back to Q(φ) lattice
        - Record flux_syntony (post-cycle)

    STEP 4: FITNESS EVALUATION
        Evaluate task-specific fitness for each survivor

    STEP 5: SELECTION
        score = fitness + λ × flux_syntony
        where λ = q = 0.027395... (universal constant, NOT hyperparameter)
    """

    def __init__(
        self,
        template: ResonantTensor,
        fitness_fn: Callable[[ResonantTensor], float],
        population_size: int = 64,
        survivor_fraction: float = 0.25,
        mutation_scale: float = 0.3,
        noise_scale: float = 0.01,
        precision: int = 100,
        lambda_syntony: float = Q_DEFICIT,
        seed: Optional[int] = None,
    ):
        """
        Initialize RES evolver.

        Args:
            template: Initial tensor template
            fitness_fn: Fitness function (tensor -> float), higher = better
            population_size: Mutants per generation (default: 64)
            survivor_fraction: Fraction to keep after syntony filter (default: 0.25)
            mutation_scale: Mutation magnitude in Q(φ) (default: 0.3)
            noise_scale: D-phase noise scale (default: 0.01)
            precision: Lattice precision (max coefficient, default: 100)
            lambda_syntony: Syntony weight = q = 0.027395... (NOT a hyperparameter!)
            seed: Random seed
        """
        self.template = template
        self.fitness_fn = fitness_fn
        self.population_size = population_size
        self.survivor_count = max(1, int(population_size * survivor_fraction))
        self.mutation_scale = mutation_scale
        self.noise_scale = noise_scale
        self.precision = precision
        self.lambda_syntony = lambda_syntony
        self.rng = random.Random(seed)

        # State
        self.best_tensor = template
        self.best_score = float("-inf")
        self.generation = 0
        self.history = {"score": [], "fitness": [], "syntony": [], "flux_syntony": []}

    def _mutate(self, tensor: ResonantTensor) -> ResonantTensor:
        """Create a mutant by perturbing the tensor values in Q(φ)."""
        values = tensor.to_list()
        shape = tensor.shape
        mode_norms = tensor.get_mode_norm_sq()

        # Perturbation scaled by mutation_scale
        perturbation = [
            self.rng.gauss(0, self.mutation_scale) for _ in range(len(values))
        ]
        new_values = [values[i] + perturbation[i] for i in range(len(values))]

        # Snap to Q(φ) lattice
        return ResonantTensor(new_values, shape, mode_norms, self.precision)

    def _run_dhsr_cycle(self, tensor: ResonantTensor) -> float:
        """
        Run one DHSR cycle: D̂ (differentiate) then Ĥ (harmonize + crystallize).

        This is the core resonant operation that was MISSING from the original.

        Returns the post-cycle (flux) syntony.
        """
        # cpu_cycle applies:
        # 1. D̂ operator: flux[i] = lattice[i] * (1 + α(S) * √|n|²) + noise
        # 2. Ĥ operator: harmonized[i] = flux[i] * (1 - β(S) * (1 - w(n)))
        # 3. Snap to Q(φ) lattice
        # Returns new syntony
        return tensor.cpu_cycle(self.noise_scale, self.precision)

    def step(self) -> float:
        """
        Run one generation of RES.

        Returns combined score of best individual.
        """
        # ═══════════════════════════════════════════════════════════════════
        # STEP 1: MUTATION (CPU)
        # Generate mutants by perturbing best tensor, snap to Q(φ) lattice
        # ═══════════════════════════════════════════════════════════════════
        mutants = [self._mutate(self.best_tensor) for _ in range(self.population_size)]

        # ═══════════════════════════════════════════════════════════════════
        # STEP 2: SYNTONY FILTER (CPU, cheap)
        # Keep top survivors by lattice syntony
        # This rejects ~75% of candidates BEFORE expensive evaluation
        # ═══════════════════════════════════════════════════════════════════
        mutants_with_syntony = [(m, m.syntony) for m in mutants]
        mutants_with_syntony.sort(key=lambda x: x[1], reverse=True)
        survivors = [m for m, s in mutants_with_syntony[: self.survivor_count]]

        # ═══════════════════════════════════════════════════════════════════
        # STEP 3: DHSR CYCLE (CPU)  ◀── THIS WAS MISSING
        # For each survivor, run D→H cycle:
        #   - Apply D̂ (differentiation with noise)
        #   - Apply Ĥ (harmonization attenuation)
        #   - Crystallize back to Q(φ)
        #   - Record flux_syntony (post-cycle)
        # ═══════════════════════════════════════════════════════════════════
        flux_syntonies = []
        for s in survivors:
            flux_syntony = self._run_dhsr_cycle(s)
            flux_syntonies.append(flux_syntony)

        # ═══════════════════════════════════════════════════════════════════
        # STEP 4: FITNESS EVALUATION
        # Evaluate task-specific fitness for each survivor
        # ═══════════════════════════════════════════════════════════════════
        scored = []
        for i, tensor in enumerate(survivors):
            fitness = self.fitness_fn(tensor)
            flux_syntony = flux_syntonies[i]
            # Selection uses FLUX syntony (post-cycle), not pre-cycle
            combined = fitness + self.lambda_syntony * flux_syntony
            scored.append((tensor, combined, fitness, flux_syntony))

        # ═══════════════════════════════════════════════════════════════════
        # STEP 5: SELECTION
        # score = fitness + λ × flux_syntony
        # where λ = q = 0.027395... (universal constant)
        # ═══════════════════════════════════════════════════════════════════
        scored.sort(key=lambda x: x[1], reverse=True)
        best, best_score, best_fitness, best_flux_syntony = scored[0]

        # Update state if improved
        if best_score > self.best_score:
            self.best_tensor = best
            self.best_score = best_score

        # Track history
        self.generation += 1
        self.history["score"].append(best_score)
        self.history["fitness"].append(best_fitness)
        self.history["syntony"].append(self.best_tensor.syntony)
        self.history["flux_syntony"].append(best_flux_syntony)

        return best_score

    def run(self, max_generations: int = 100, verbose: bool = False) -> dict:
        """
        Run evolution for multiple generations.

        Returns dict with best_tensor, final_score, history.
        """
        for gen in range(max_generations):
            score = self.step()
            if verbose and gen % 20 == 0:
                print(f"Gen {gen}: score={score:.4f}")

        return {
            "best_tensor": self.best_tensor,
            "final_score": self.best_score,
            "generations": max_generations,
            "history": self.history,
        }
