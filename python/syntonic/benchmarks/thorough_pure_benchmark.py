"""
Thorough Pure Resonant Benchmark Suite.
Strictly zero dependencies on PyTorch or NumPy.
"""

from syntonic.pure.resonant_engine_net import WindingEngine
from syntonic._core import ResonantTensor, GoldenExact
import random
import time
import math

# =============================================================================
# 1. NATIVE DATASET GENERATORS (No NumPy)
# =============================================================================

def make_circles_dataset(n_samples=200, noise=0.05, factor=0.8):
    """Generate concentric circles dataset."""
    data = []
    labels = []
    
    n_samples_out = n_samples // 2
    n_samples_in = n_samples - n_samples_out
    
    # Outer circle
    for _ in range(n_samples_out):
        angle = random.uniform(0, 2 * math.pi)
        x = math.cos(angle) + random.gauss(0, noise)
        y = math.sin(angle) + random.gauss(0, noise)
        data.append([x, y])
        labels.append(0)
        
    # Inner circle
    for _ in range(n_samples_in):
        angle = random.uniform(0, 2 * math.pi)
        x = factor * math.cos(angle) + random.gauss(0, noise)
        y = factor * math.sin(angle) + random.gauss(0, noise)
        data.append([x, y])
        labels.append(1)
        
    return data, labels

def make_moons_dataset(n_samples=200, noise=0.05):
    """Generate two interleaving half circles."""
    data = []
    labels = []
    
    n_samples_out = n_samples // 2
    n_samples_in = n_samples - n_samples_out
    
    # Upper moon
    for i in range(n_samples_out):
        angle = math.pi * (i / n_samples_out)
        x = math.cos(angle) + random.gauss(0, noise)
        y = math.sin(angle) + random.gauss(0, noise)
        data.append([x, y])
        labels.append(0)
        
    # Lower moon
    for i in range(n_samples_in):
        angle = math.pi * (i / n_samples_in)
        x = 1 - math.cos(angle) + random.gauss(0, noise)
        y = 0.5 - math.sin(angle) + random.gauss(0, noise)
        data.append([x, y])
        labels.append(1)
        
    return data, labels

def make_spirals_dataset(n_samples=200, noise=0.1):
    """Generate interleaved spirals."""
    data = []
    labels = []
    
    def get_spiral(offset, label):
        points = []
        for i in range(n_samples // 2):
            r = i / (n_samples // 2) * 2
            t = 1.75 * i / (n_samples // 2) * 2 * math.pi + offset
            x = r * math.cos(t) + random.gauss(0, noise)
            y = r * math.sin(t) + random.gauss(0, noise)
            points.append([x, y])
            labels.append(label)
        return points

    data.extend(get_spiral(0, 0))
    data.extend(get_spiral(math.pi, 1))
    
    return data, labels

# =============================================================================
# 2. ENHANCED RESONANT EVOLUTION STRATEGY (RES)
# =============================================================================

class EnhancedRES:
    """Robust Resonant Evolution Strategy with elitism and syntony pruning."""
    def __init__(self, engine, pop_size=32, elitism_size=4, mutation_scale=0.1):
        self.engine = engine
        self.pop_size = pop_size
        self.elitism_size = elitism_size
        self.mutation_scale = mutation_scale
        self.population = [] # List of (params, accuracy, syntony)
        
    def evaluate(self, params, data, labels):
        self.engine.set_parameters(params)
        
        flat_data = [val for sublist in data for val in sublist]
        x_tensor = ResonantTensor(flat_data, [len(data), 2])
        
        out_tensor = self.engine.forward(x_tensor)
        out_floats = out_tensor.to_floats()
        
        # Calculate accuracy
        correct = 0
        for i in range(len(labels)):
            p0, p1 = out_floats[i*2], out_floats[i*2+1]
            pred = 1 if p1 > p0 else 0
            if pred == labels[i]:
                correct += 1
        acc = correct / len(labels)
        
        # Calculate average syntony of parameter tensors
        total_s = sum(p.syntony for p in params)
        avg_s = total_s / len(params)
        
        return acc, avg_s

    def mutate_params(self, params, scale):
        mutated = []
        for p in params:
            lattice = p.to_lattice_list()
            new_lattice = []
            for g in lattice:
                if random.random() < scale:
                    da, db = random.randint(-1, 1), random.randint(-1, 1)
                    p_exact = GoldenExact.from_integers(da, db)
                    new_lattice.append(g + p_exact)
                else:
                    new_lattice.append(g)
            mutated.append(ResonantTensor.from_golden_exact(new_lattice, p.shape))
        return mutated

    def step(self, data, labels, generation):
        # 1. Initialize population if needed
        if not self.population:
            initial_params = self.engine.get_parameters()
            acc, s = self.evaluate(initial_params, data, labels)
            self.population = [(initial_params, acc, s)]

        # 2. Elitism: Keep top survivors
        self.population.sort(key=lambda x: x[1], reverse=True)
        elites = self.population[:self.elitism_size]
        
        # 3. Spawn mutants from elites
        new_candidates = elites.copy()
        while len(new_candidates) < self.pop_size:
            parent = random.choice(elites)
            # Adaptive mutation scale: more syntony = smaller mutations
            scale = self.mutation_scale * (2.0 - parent[2]) 
            mut_params = self.mutate_params(parent[0], scale)
            acc, s = self.evaluate(mut_params, data, labels)
            new_candidates.append((mut_params, acc, s))
            
        self.population = new_candidates
        best = max(self.population, key=lambda x: x[1])
        return best[1], best[2]

# =============================================================================
# 3. TEST RUNNER
# =============================================================================

def run_dataset_benchmark(name, generator, dims=[2, 16, 2], generations=50):
    print(f"\n--- Benchmarking: {name} ---")
    data, labels = generator(200)
    test_data, test_labels = generator(50)
    
    engine = WindingEngine(dims)
    res = EnhancedRES(engine, pop_size=40, elitism_size=6)
    
    start_time = time.time()
    for gen in range(generations):
        best_acc, best_s = res.step(data, labels, gen)
        if gen % 10 == 0 or best_acc > 0.95:
            print(f"Gen {gen:02d} | Acc: {best_acc:.2%} | Syn: {best_s:.4f}")
            if best_acc > 0.98: break
            
    duration = time.time() - start_time
    # Final eval
    best_params = res.population[0][0]
    final_acc, final_s = res.evaluate(best_params, test_data, test_labels)
    
    print(f"RESULT: {name} | Test Acc: {final_acc:.2%} | Syn: {final_s:.4f} | Time: {duration:.2f}s")
    return final_acc, final_s, duration

if __name__ == "__main__":
    print("============================================================")
    print("THOROUGH PURE RESONANT ENGINE BENCHMARK")
    print("============================================================")
    
    results = {}
    
    # 1. Circle Search
    results['Circles'] = run_dataset_benchmark("Circles", make_circles_dataset)
    
    # 2. Moon Search
    results['Moons'] = run_dataset_benchmark("Moons", make_moons_dataset)
    
    # 3. Spiral Search (The Hard Test)
    results['Spirals'] = run_dataset_benchmark("Spirals", make_spirals_dataset, dims=[2, 32, 2], generations=100)
    
    print("\n" + "="*60)
    print(f"{'Dataset':<12} | {'Acc':<8} | {'Syntony':<10} | {'Time':<8}")
    print("-" * 60)
    for name, (acc, s, t) in results.items():
        print(f"{name:<12} | {acc:<8.2%} | {s:<10.4f} | {t:<8.2f}s")
    print("="*60)
