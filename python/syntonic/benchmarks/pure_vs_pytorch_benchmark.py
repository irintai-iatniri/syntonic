"""
Pure Syntonic vs PyTorch Benchmark

Compares:
1. Forward pass speed
2. Training speed
3. Memory usage
4. Final accuracy

Task: XOR classification (simple but non-trivial)
"""

import time
import sys
import random
from typing import List, Tuple, Dict, Any

# Pure Syntonic imports
from syntonic._core import ResonantTensor, WindingState, GoldenExact
import syntonic.sn as sn
from syntonic.nn.winding.winding_net_pure import PureWindingNet

# Try PyTorch import
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    PYTORCH_AVAILABLE = True
except ImportError:
    PYTORCH_AVAILABLE = False
    print("⚠️  PyTorch not available - will only benchmark Pure Syntonic")


# =============================================================================
# Data Generation
# =============================================================================

def generate_xor_data(n_samples: int = 1000) -> Tuple[List[WindingState], List[int]]:
    """Generate XOR dataset as winding states."""
    X = []
    y = []

    n_per_class = n_samples // 4

    # Class 0: (0,0)
    for _ in range(n_per_class):
        X.append(WindingState(0, 0, 0, 0))
        y.append(0)

    # Class 0: (1,1) -> mapped to (2,2)
    for _ in range(n_per_class):
        X.append(WindingState(2, 2, 0, 0))
        y.append(0)

    # Class 1: (0,1) -> mapped to (0,2)
    for _ in range(n_per_class):
        X.append(WindingState(0, 2, 0, 0))
        y.append(1)

    # Class 1: (1,0) -> mapped to (2,0)
    for _ in range(n_per_class):
        X.append(WindingState(2, 0, 0, 0))
        y.append(1)

    # Shuffle
    combined = list(zip(X, y))
    random.shuffle(combined)
    X, y = zip(*combined)

    return list(X), list(y)


def generate_xor_data_pytorch(n_samples: int = 1000):
    """Generate XOR dataset as PyTorch tensors."""
    import torch

    X = []
    y = []

    n_per_class = n_samples // 4

    # Class 0: (0,0)
    for _ in range(n_per_class):
        X.append([0.0, 0.0])
        y.append(0)

    # Class 0: (1,1)
    for _ in range(n_per_class):
        X.append([1.0, 1.0])
        y.append(0)

    # Class 1: (0,1)
    for _ in range(n_per_class):
        X.append([0.0, 1.0])
        y.append(1)

    # Class 1: (1,0)
    for _ in range(n_per_class):
        X.append([1.0, 0.0])
        y.append(1)

    # Shuffle
    combined = list(zip(X, y))
    random.shuffle(combined)
    X, y = zip(*combined)

    return torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.long)


# =============================================================================
# PyTorch Model
# =============================================================================

if PYTORCH_AVAILABLE:
    class PyTorchMLP(nn.Module):
        """Simple PyTorch MLP for XOR."""
        def __init__(self, hidden_dim: int = 32):
            super().__init__()
            self.fc1 = nn.Linear(2, hidden_dim)
            self.fc2 = nn.Linear(hidden_dim, hidden_dim)
            self.fc3 = nn.Linear(hidden_dim, 2)
            self.relu = nn.ReLU()

        def forward(self, x):
            x = self.relu(self.fc1(x))
            x = self.relu(self.fc2(x))
            x = self.fc3(x)
            return x


# =============================================================================
# Evolutionary Optimizer for Pure Syntonic
# =============================================================================

class PureSyntonicOptimizer:
    """Evolutionary optimizer for gradient-free syntonic models."""
    
    def __init__(self, model, pop_size=20, elitism=4, mutation_scale=0.1):
        self.model = model
        self.pop_size = pop_size
        self.elitism = elitism
        self.mutation_scale = mutation_scale
        self.population = []
        
    def evaluate(self, weights, X, y):
        """Evaluate fitness of a weight set."""
        self.model.set_weights(weights)
        logits = self.model(X)
        preds = argmax_batch(logits, dim=1)
        acc = sum(1 for p, t in zip(preds, y) if p == t) / len(y)
        syntony = self.model.syntony
        return acc, syntony
        
    def mutate_weights(self, weights):
        """Mutate a set of weights."""
        mutated = []
        for w in weights:
            # Add small random perturbation to lattice values
            lattice = w.get_lattice()
            perturbed_lattice = []
            for g in lattice:
                # Small random change in golden coefficients
                da = random.gauss(0, self.mutation_scale)
                db = random.gauss(0, self.mutation_scale)
                perturbed_g = g + GoldenExact.nearest(da, 10) + GoldenExact.nearest(db, 10) * GoldenExact.golden_ratio()
                perturbed_lattice.append(perturbed_g)
            mutated_w = ResonantTensor.from_golden_exact(perturbed_lattice, w.shape)
            mutated.append(mutated_w)
        return mutated
        
    def train(self, X, y, generations=50):
        """Evolutionary training."""
        start_time = time.perf_counter()
        
        # Initialize with current weights
        best_weights = self.model.get_weights()
        best_acc, best_syntony = self.evaluate(best_weights, X, y)
        self.population = [(best_weights, best_acc, best_syntony)]
        
        for gen in range(generations):
            # Sort by accuracy
            self.population.sort(key=lambda x: x[1], reverse=True)
            elites = self.population[:self.elitism]
            
            # Generate new population
            new_pop = elites.copy()
            while len(new_pop) < self.pop_size:
                parent = random.choice(elites)
                mutated = self.mutate_weights(parent[0])
                acc, syntony = self.evaluate(mutated, X, y)
                new_pop.append((mutated, acc, syntony))
                
            self.population = new_pop
            best_acc, best_syntony = self.population[0][1], self.population[0][2]
            
            if best_acc > 0.95:  # Early stop if good enough
                break
                
        duration = time.perf_counter() - start_time
        
        # Set best weights
        self.model.set_weights(self.population[0][0])
        return self.population[0][1], self.population[0][2], duration

def argmax_batch(tensor: ResonantTensor, dim: int = 1) -> List[int]:
    """Compute argmax for ResonantTensor."""
    floats = tensor.to_floats()
    shape = tensor.shape

    if len(shape) != 2:
        raise ValueError(f"Expected 2D tensor, got shape {shape}")

    batch_size, num_features = shape
    result = []

    for b in range(batch_size):
        start_idx = b * num_features
        row = floats[start_idx:start_idx + num_features]
        max_idx = max(range(len(row)), key=lambda i: row[i])
        result.append(max_idx)

    return result


def measure_memory():
    """Measure current memory usage."""
    try:
        import psutil
        process = psutil.Process()
        return process.memory_info().rss / 1024 / 1024  # MB
    except ImportError:
        return 0.0


# =============================================================================
# Benchmark: Pure Syntonic
# =============================================================================

def benchmark_pure_syntonic(n_samples: int = 1000, epochs: int = 50):
    """Benchmark Pure Syntonic implementation."""
    print("\n" + "=" * 70)
    print("BENCHMARK: Pure Syntonic (ResonantTensor + sn)")
    print("=" * 70)

    # Generate data
    X_train, y_train = generate_xor_data(n_samples)

    # Create model
    model = PureWindingNet(
        max_winding=5,
        base_dim=32,
        num_blocks=2,
        output_dim=2,
        precision=64,
    )

    mem_start = measure_memory()

    # Warm-up
    _ = model(X_train[:10])

    # Forward pass benchmark
    print(f"\n📊 Forward Pass Benchmark ({min(len(X_train), 100)} samples)")
    forward_times = []
    for _ in range(10):
        start = time.perf_counter()
        logits = model(X_train[:min(len(X_train), 100)])
        forward_times.append(time.perf_counter() - start)

    avg_forward = sum(forward_times) / len(forward_times)
    n_forward = min(len(X_train), 100)
    print(f"  Average time: {avg_forward*1000:.2f}ms")
    print(f"  Throughput: {n_forward/avg_forward:.0f} samples/sec")

    # Training benchmark
    print(f"\n🏋️  Training Benchmark ({epochs} generations)")
    optimizer = PureSyntonicOptimizer(model, pop_size=20, elitism=4)
    final_acc, final_syntony, train_time = optimizer.train(X_train, y_train, generations=epochs)

    mem_end = measure_memory()

    print(f"\n📈 Results:")
    print(f"  Final Accuracy: {final_acc:.2%}")
    print(f"  Final Syntony: {final_syntony:.4f}")
    print(f"  Training Time: {train_time:.2f}s")
    print(f"  Time per Generation: {train_time/epochs*1000:.1f}ms")
    print(f"  Memory Used: {mem_end - mem_start:.1f} MB")

    return {
        'framework': 'Pure Syntonic',
        'forward_time': avg_forward,
        'throughput': n_forward/avg_forward,
        'train_time': train_time,
        'time_per_epoch': train_time/epochs,
        'final_accuracy': final_acc,
        'memory_mb': mem_end - mem_start,
    }


# =============================================================================
# Benchmark: PyTorch
# =============================================================================

def benchmark_pytorch(n_samples: int = 1000, epochs: int = 50):
    """Benchmark PyTorch implementation."""
    if not PYTORCH_AVAILABLE:
        return None

    print("\n" + "=" * 70)
    print("BENCHMARK: PyTorch")
    print("=" * 70)

    # Generate data
    X_train, y_train = generate_xor_data_pytorch(n_samples)

    # Create model
    model = PyTorchMLP(hidden_dim=32)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    mem_start = measure_memory()

    # Warm-up
    with torch.no_grad():
        _ = model(X_train[:10])

    # Forward pass benchmark
    n_forward = min(len(X_train), 100)
    print(f"\n📊 Forward Pass Benchmark ({n_forward} samples)")
    forward_times = []
    model.eval()
    with torch.no_grad():
        for _ in range(10):
            start = time.perf_counter()
            logits = model(X_train[:n_forward])
            forward_times.append(time.perf_counter() - start)

    avg_forward = sum(forward_times) / len(forward_times)
    print(f"  Average time: {avg_forward*1000:.2f}ms")
    print(f"  Throughput: {n_forward/avg_forward:.0f} samples/sec")

    # Training benchmark
    print(f"\n🏋️  Training Benchmark ({epochs} epochs)")
    train_start = time.perf_counter()

    for epoch in range(epochs):
        model.train()

        # Forward pass
        logits = model(X_train)
        loss = criterion(logits, y_train)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Track accuracy
        with torch.no_grad():
            preds = logits.argmax(dim=1)
            acc = (preds == y_train).float().mean().item()

        if (epoch + 1) % 10 == 0:
            print(f"  Epoch {epoch+1:3d}/{epochs}: Acc={acc:.2%} Loss={loss.item():.4f}")

    train_time = time.perf_counter() - train_start

    # Final evaluation
    model.eval()
    with torch.no_grad():
        logits = model(X_train)
        preds = logits.argmax(dim=1)
        final_acc = (preds == y_train).float().mean().item()

    mem_end = measure_memory()
    mem_used = mem_end - mem_start

    print(f"\n📈 Results:")
    print(f"  Final Accuracy: {final_acc:.2%}")
    print(f"  Training Time: {train_time:.2f}s")
    print(f"  Time per Epoch: {train_time/epochs*1000:.1f}ms")
    print(f"  Memory Used: {mem_used:.1f} MB")

    return {
        'framework': 'PyTorch',
        'forward_time': avg_forward,
        'throughput': n_forward/avg_forward,
        'train_time': train_time,
        'time_per_epoch': train_time/epochs,
        'final_accuracy': final_acc,
        'memory_mb': mem_used,
    }


# =============================================================================
# Comparison Table
# =============================================================================

def print_comparison(results: List[Dict[str, Any]]):
    """Print comparison table."""
    if not results:
        return

    print("\n" + "=" * 70)
    print("📊 BENCHMARK COMPARISON")
    print("=" * 70)

    print(f"\n{'Metric':<25} {'Pure Syntonic':<20} {'PyTorch':<20} {'Ratio':<10}")
    print("-" * 70)

    syntonic = results[0]
    pytorch = results[1] if len(results) > 1 and results[1] else None

    if pytorch:
        # Forward pass
        ratio = syntonic['forward_time'] / pytorch['forward_time']
        print(f"{'Forward Pass (ms)':<25} {syntonic['forward_time']*1000:>18.2f}  {pytorch['forward_time']*1000:>18.2f}  {ratio:>8.2f}x")

        # Throughput
        ratio = syntonic['throughput'] / pytorch['throughput']
        print(f"{'Throughput (samp/s)':<25} {syntonic['throughput']:>18.0f}  {pytorch['throughput']:>18.0f}  {ratio:>8.2f}x")

        # Training time
        ratio = syntonic['train_time'] / pytorch['train_time']
        print(f"{'Training Time (s)':<25} {syntonic['train_time']:>18.2f}  {pytorch['train_time']:>18.2f}  {ratio:>8.2f}x")

        # Time per epoch
        ratio = syntonic['time_per_epoch'] / pytorch['time_per_epoch']
        print(f"{'Time per Epoch (ms)':<25} {syntonic['time_per_epoch']*1000:>18.1f}  {pytorch['time_per_epoch']*1000:>18.1f}  {ratio:>8.2f}x")

        # Accuracy
        print(f"{'Final Accuracy':<25} {syntonic['final_accuracy']:>17.1%}  {pytorch['final_accuracy']:>17.1%}  {'':>10}")

        # Memory
        if syntonic['memory_mb'] > 0 and pytorch['memory_mb'] > 0:
            ratio = syntonic['memory_mb'] / pytorch['memory_mb']
            print(f"{'Memory (MB)':<25} {syntonic['memory_mb']:>18.1f}  {pytorch['memory_mb']:>18.1f}  {ratio:>8.2f}x")
    else:
        print(f"{'Forward Pass (ms)':<25} {syntonic['forward_time']*1000:>18.2f}  {'N/A':<20}")
        print(f"{'Throughput (samp/s)':<25} {syntonic['throughput']:>18.0f}  {'N/A':<20}")
        print(f"{'Training Time (s)':<25} {syntonic['train_time']:>18.2f}  {'N/A':<20}")
        print(f"{'Time per Epoch (ms)':<25} {syntonic['time_per_epoch']*1000:>18.1f}  {'N/A':<20}")
        print(f"{'Final Accuracy':<25} {syntonic['final_accuracy']:>17.1%}  {'N/A':<20}")

    print("\n" + "=" * 70)
    print("Notes:")
    print("  • Pure Syntonic uses gradient-free RES evolution")
    print("  • PyTorch uses Adam optimizer with backpropagation")
    print("  • Ratio > 1.0 means Syntonic is slower, < 1.0 means faster")
    print("=" * 70)


# =============================================================================
# Main
# =============================================================================

def main():
    print("=" * 70)
    print("PURE SYNTONIC vs PYTORCH BENCHMARK")
    print("=" * 70)
    print(f"Task: XOR Classification")
    print(f"Dataset: 200 samples")
    print(f"Epochs: 20")
    print(f"Architecture: 2 layers, 32 hidden dim")

    results = []

    # Benchmark Pure Syntonic
    syntonic_results = benchmark_pure_syntonic(n_samples=200, epochs=20)
    results.append(syntonic_results)

    # Benchmark PyTorch
    if PYTORCH_AVAILABLE:
        pytorch_results = benchmark_pytorch(n_samples=200, epochs=20)
        if pytorch_results:
            results.append(pytorch_results)

    # Print comparison
    print_comparison(results)


if __name__ == "__main__":
    main()
