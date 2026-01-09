"""
Pure Resonant XOR Benchmark.
Strictly zero dependencies on PyTorch or NumPy.
"""

from syntonic.pure.resonant_engine_net import WindingEngine
from syntonic._core import ResonantTensor, GoldenExact
import random
import time

def make_xor_dataset(n_samples=200, noise=0.1):
    """Generate XOR dataset using only standard Python random."""
    data = []
    labels = []
    for _ in range(n_samples):
        # Choose one of four quadrants
        quad = random.randint(0, 3)
        if quad == 0: # (0,0) -> 0
            x, y = 0, 0
            label = 0
        elif quad == 1: # (1,1) -> 0
            x, y = 1, 1
            label = 0
        elif quad == 2: # (0,1) -> 1
            x, y = 0, 1
            label = 1
        else: # (1,0) -> 1
            x, y = 1, 0
            label = 1
            
        # Add noise
        x += (random.random() - 0.5) * 2 * noise
        y += (random.random() - 0.5) * 2 * noise
        data.append([x, y])
        labels.append(label)
    return data, labels

def evaluate_accuracy(engine, data, labels):
    """Compute accuracy without any ML libraries."""
    correct = 0
    # Process sample by sample (or batch if we want)
    # Convert data to ResonantTensor
    # Flatten data for tensor creation
    flat_data = [val for sublist in data for val in sublist]
    x_tensor = ResonantTensor(flat_data, [len(data), 2])
    
    # Forward pass
    out_tensor = engine.forward(x_tensor)
    
    # Get floats back for final check
    out_floats = out_tensor.to_floats()
    
    for i in range(len(labels)):
        # Binary classification: index 1 > index 0 means class 1
        p0 = out_floats[i*2]
        p1 = out_floats[i*2 + 1]
        pred = 1 if p1 > p0 else 0
        if pred == labels[i]:
            correct += 1
            
    return correct / len(labels)

def mutate_param(param, scale=0.1, precision=100):
    """Mutate a ResonantTensor on the Golden Lattice."""
    lattice = param.to_lattice_list()
    new_lattice = []
    for g in lattice:
        # Random perturbation in Q(phi) space
        da = random.randint(-1, 1) if random.random() < scale else 0
        db = random.randint(-1, 1) if random.random() < scale else 0
        if da == 0 and db == 0:
            new_lattice.append(g)
        else:
            p = GoldenExact.from_integers(da, db)
            new_lattice.append(g + p)
            
    return ResonantTensor.from_golden_exact(
        new_lattice, param.shape
    )

def train_res(engine, data, labels, generations=100, pop_size=10):
    """Simple Resonant Evolution Strategy (RES) in Pure Python."""
    best_acc = evaluate_accuracy(engine, data, labels)
    print(f"Initial Accuracy: {best_acc:.2%}")
    
    for gen in range(generations):
        improved = False
        current_params = engine.get_parameters()
        
        # Spawn population
        for _ in range(pop_size):
            # Mutate all layers
            mutated_params = [mutate_param(p) for p in current_params]
            
            # Temporary apply to engine
            engine.set_parameters(mutated_params)
            acc = evaluate_accuracy(engine, data, labels)
            
            if acc > best_acc:
                best_acc = acc
                current_params = mutated_params
                improved = True
                print(f"Gen {gen}: Accuracy improved to {best_acc:.2%}")
            
        # Restore best or keep improved
        engine.set_parameters(current_params)
        
        if not improved and gen % 10 == 0:
            print(f"Gen {gen}: Stagnant at {best_acc:.2%}")

    return best_acc

if __name__ == "__main__":
    print("============================================================")
    print("Pure Resonant XOR Benchmark (No Torch/No NumPy)")
    print("============================================================")
    
    # 1. Dataset
    train_data, train_labels = make_xor_dataset(160)
    test_data, test_labels = make_xor_dataset(40)
    
    # 2. Model: [2, 8, 2]
    engine = WindingEngine([2, 8, 2])
    
    # 3. Train via RES
    start_time = time.time()
    final_acc = train_res(engine, train_data, train_labels, generations=50, pop_size=20)
    duration = time.time() - start_time
    
    # 4. Evaluate
    test_acc = evaluate_accuracy(engine, test_data, test_labels)
    
    print("============================================================")
    print(f"Final Test Accuracy: {test_acc:.2%}")
    print(f"Training Time: {duration:.2f}s")
    print("============================================================")
