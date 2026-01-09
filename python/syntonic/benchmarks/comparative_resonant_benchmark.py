"""
Comparative Resonant Benchmark: Pure Winding Engine vs. PyTorch MLP.
Evaluates accuracy, training time, and syntony across multiple datasets.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import time
import random

from syntonic.pure.resonant_engine_net import WindingEngine
from syntonic._core import ResonantTensor, GoldenExact
from syntonic.pure.resonant_engine_net import ResonantLayer # For internal access if needed

# =============================================================================
# 1. DATASET GENERATORS (NumPy for consistency)
# =============================================================================

def make_xor(n=500, noise=0.1):
    X = np.random.randn(n, 2) * noise
    y = np.zeros(n)
    # XOR quadrants
    X[:n//4] += [0, 0]; y[:n//4] = 0
    X[n//4:2*n//4] += [1, 1]; y[n//4:2*n//4] = 0
    X[2*n//4:3*n//4] += [0, 1]; y[2*n//4:3*n//4] = 1
    X[3*n//4:] += [1, 0]; y[3*n//4:] = 1
    return X.astype(np.float32), y.astype(np.int64)

def make_moons(n=500, noise=0.1):
    from sklearn.datasets import make_moons as sk_moons
    X, y = sk_moons(n_samples=n, noise=noise, random_state=42)
    return X.astype(np.float32), y.astype(np.int64)

def make_circles(n=500, noise=0.1, factor=0.5):
    from sklearn.datasets import make_circles as sk_circles
    X, y = sk_circles(n_samples=n, noise=noise, factor=factor, random_state=42)
    return X.astype(np.float32), y.astype(np.int64)

# =============================================================================
# 2. PYTORCH BASELINE
# =============================================================================

class TorchBaseline(nn.Module):
    def __init__(self, dims):
        super().__init__()
        layers = []
        for i in range(len(dims)-1):
            layers.append(nn.Linear(dims[i], dims[i+1]))
            if i < len(dims)-2:
                layers.append(nn.ReLU())
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

def train_torch(X, y, dims, epochs=100, lr=0.01):
    model = TorchBaseline(dims)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    
    X_t = torch.from_numpy(X)
    y_t = torch.from_numpy(y)
    
    start_time = time.time()
    for _ in range(epochs):
        optimizer.zero_grad()
        out = model(X_t)
        loss = criterion(out, y_t)
        loss.backward()
        optimizer.step()
    duration = time.time() - start_time
    
    return model, duration

# =============================================================================
# 3. PURE RESONANT ENGINE OPTIMIZER (Enhanced RES)
# =============================================================================

class RESOptimizer:
    def __init__(self, engine, pop_size=32, elitism=4, scales=(0.1, 0.05)):
        self.engine = engine
        self.pop_size = pop_size
        self.elitism = elitism
        self.scales = scales
        self.population = []

    def evaluate(self, params, X, y):
        self.engine.set_parameters(params)
        x_tensor = ResonantTensor(X.flatten().tolist(), [X.shape[0], X.shape[1]])
        out = self.engine.forward(x_tensor)
        out_f = out.to_floats()
        
        preds = []
        for i in range(len(y)):
            preds.append(1 if out_f[i*2+1] > out_f[i*2] else 0)
        
        acc = sum(p == t for p, t in zip(preds, y)) / len(y)
        avg_s = sum(p.syntony for p in params) / len(params)
        return acc, avg_s

    def mutate(self, params, scale):
        mutated = []
        for p in params:
            lattice = p.to_lattice_list()
            new_lattice = []
            for g in lattice:
                if random.random() < scale:
                    da, db = random.randint(-1, 1), random.randint(-1, 1)
                    new_lattice.append(g + GoldenExact.from_integers(da, db))
                else:
                    new_lattice.append(g)
            mutated.append(ResonantTensor.from_golden_exact(new_lattice, p.shape))
        return mutated

    def train(self, X, y, generations=100):
        start_time = time.time()
        best_params = self.engine.get_parameters()
        best_acc, best_s = self.evaluate(best_params, X, y)
        self.population = [(best_params, best_acc, best_s)]
        
        for gen in range(generations):
            self.population.sort(key=lambda x: x[1], reverse=True)
            elites = self.population[:self.elitism]
            
            new_pop = elites.copy()
            while len(new_pop) < self.pop_size:
                p = random.choice(elites)
                scale = self.scales[0] if p[1] < 0.8 else self.scales[1]
                mut = self.mutate(p[0], scale)
                acc, s = self.evaluate(mut, X, y)
                new_pop.append((mut, acc, s))
                
            self.population = new_pop
            best_acc, best_s = self.population[0][1], self.population[0][2]
            
            if best_acc > 0.98: break
            
        duration = time.time() - start_time
        return self.population[0][0], best_acc, best_s, duration

# =============================================================================
# 4. COMPARATIVE EXPERIMENT
# =============================================================================

def run_comparison(name, generator, dims=[2, 16, 2], n_train=400, n_test=100):
    print(f"\nComparing on: {name}")
    X, y = generator(n_train + n_test)
    X_train, X_test = X[:n_train], X[n_train:]
    y_train, y_test = y[:n_train], y[n_train:]

    # --- Torch ---
    torch_model, torch_time = train_torch(X_train, y_train, dims, epochs=200)
    with torch.no_grad():
        torch_out = torch_model(torch.from_numpy(X_test))
        torch_preds = torch_out.argmax(dim=1).numpy()
        torch_acc = (torch_preds == y_test).mean()

    # --- Pure Resonant ---
    engine = WindingEngine(dims)
    optimizer = RESOptimizer(engine, pop_size=40, elitism=8)
    res_params, res_train_acc, res_train_s, res_time = optimizer.train(X_train, y_train, generations=100)
    
    # Eval test
    engine.set_parameters(res_params)
    x_test_tensor = ResonantTensor(X_test.flatten().tolist(), [X_test.shape[0], X_test.shape[1]])
    test_out = engine.forward(x_test_tensor)
    test_out_f = test_out.to_floats()
    res_preds = [1 if test_out_f[i*2+1] > test_out_f[i*2] else 0 for i in range(len(y_test))]
    res_test_acc = sum(p == t for p, t in zip(res_preds, y_test)) / len(y_test)

    return {
        'torch_acc': torch_acc, 'torch_time': torch_time,
        'res_acc': res_test_acc, 'res_time': res_time, 'res_syntony': res_train_s
    }

if __name__ == "__main__":
    print("-" * 60)
    print("COMPARATIVE RESONANT BENCHMARK (Pure Engine vs. Torch Adam)")
    print("-" * 60)
    
    tasks = [
        ("XOR", make_xor),
        ("Moons", make_moons),
        ("Circles", make_circles)
    ]
    
    results = {}
    for name, gen in tasks:
        results[name] = run_comparison(name, gen)

    print("\n" + "="*85)
    print(f"{'Task':<10} | {'Torch Acc':<12} | {'RES Acc':<12} | {'Torch Time':<12} | {'RES Time':<12} | {'Syntony'}")
    print("-" * 85)
    for name, res in results.items():
        print(f"{name:<10} | {res['torch_acc']:<12.2%} | {res['res_acc']:<12.2%} | "
              f"{res['torch_time']:<12.2f}s | {res['res_time']:<12.2f}s | {res['res_syntony']:.4f}")
    print("="*85)
