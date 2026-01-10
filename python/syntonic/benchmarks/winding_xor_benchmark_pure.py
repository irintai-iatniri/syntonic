"""
WindingNet XOR Benchmark (Pure Python)

This benchmark trains a WindingNet on the XOR problem using the Resonant Engine.
It demonstrates:
1. Pure Python implementation (no PyTorch)
2. Retrocausal Training (gradient-free, syntony-guided)
3. Winding state embedding stability
4. Blockchain consensus mechanism

Target: >95% accuracy on XOR task without backpropagation.
"""

import time
import random
import math
from typing import List, Tuple, Dict

# Pure Syntonic imports
import syntonic.sn as sn
from syntonic._core import ResonantTensor, WindingState
from syntonic.nn.winding.winding_net_pure import PureWindingNet
from syntonic.nn.training.trainer import RetrocausalTrainer, RESTrainingConfig


def argmax_batch(tensor: ResonantTensor, dim: int = 1) -> List[int]:
    """
    Compute argmax along a dimension for a batch tensor.

    Args:
        tensor: ResonantTensor of shape (batch, features)
        dim: Dimension to reduce (1 = along features)

    Returns:
        List of argmax indices for each batch element
    """
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


def make_xor_dataset(n_samples: int = 200, noise: float = 0.1) -> Tuple[List[WindingState], List[int]]:
    """
    Generate XOR dataset mapped to windings.
    
    Mapping:
    (0,0) -> Class 0 -> Winding(0,0)
    (0,1) -> Class 1 -> Winding(0,2)
    (1,0) -> Class 1 -> Winding(2,0)
    (1,1) -> Class 0 -> Winding(2,2)
    
    Noise added to fractional part (which WindingState ignores, but we allow 
    some variation if we were using continuous embeddings. Here we map 
    discrete regions to integers).
    """
    X = []
    y = []
    
    n_per_class = n_samples // 4
    
    # Class 0: (0,0)
    for _ in range(n_per_class):
        X.append(WindingState(0 + int(random.gauss(0, noise)), 0 + int(random.gauss(0, noise)), 0, 0))
        y.append(0)
        
    # Class 0: (1,1) -> mapped to (2,2) for separation
    for _ in range(n_per_class):
        X.append(WindingState(2 + int(random.gauss(0, noise)), 2 + int(random.gauss(0, noise)), 0, 0))
        y.append(0)
        
    # Class 1: (0,1) -> mapped to (0,2)
    for _ in range(n_per_class):
        X.append(WindingState(0 + int(random.gauss(0, noise)), 2 + int(random.gauss(0, noise)), 0, 0))
        y.append(1)
        
    # Class 1: (1,0) -> mapped to (2,0)
    for _ in range(n_per_class):
        X.append(WindingState(2 + int(random.gauss(0, noise)), 0 + int(random.gauss(0, noise)), 0, 0))
        y.append(1)
        
    # Shuffle
    combined = list(zip(X, y))
    random.shuffle(combined)
    X, y = zip(*combined)
    return list(X), list(y)

def run_benchmark():
    print("=" * 70)
    print("Pure Syntonic XOR Benchmark (Retrocausal Training)")
    print("=" * 70)

    # 1. Prepare Data
    n_samples = 200
    X, y = make_xor_dataset(n_samples=n_samples, noise=0.1)
    
    split = int(0.8 * n_samples)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]
    
    print(f"Dataset: {n_samples} samples")
    print(f"Train: {len(X_train)}, Test: {len(X_test)}")
    
    # 2. Initialize Model
    model = PureWindingNet(
        max_winding=5,
        base_dim=32,
        num_blocks=2,
        output_dim=2,
        precision=64,
        consensus_threshold=0.02
    )
    
    print("\nModel Initialized:")
    print(f"  Architecture: PureWindingNet")
    print(f"  Base Dim: {model.base_dim}")
    print(f"  Blocks: {model.num_blocks}")
    print(f"  Precision: {model.precision}-bit")

    # 3. Configure Trainer
    config = RESTrainingConfig(
        max_generations=20,     # It converges very fast
        population_size=16,
        learning_rate=0.05, # Not used in retrocausal but kept for compatibility
        syntony_threshold=0.8,
        retrocausal_enabled=True,
        log_interval=5
    )
    
    # Manual Training Loop
    # We use manual loop compatible with WindingState inputs
    

def create_batches(X: List[WindingState], y: List[int], batch_size: int) -> List[Tuple[List[WindingState], ResonantTensor]]:
    """Create batches for training."""
    batches = []
    num_samples = len(X)
    for i in range(0, num_samples, batch_size):
        batch_X = X[i:i+batch_size]
        batch_y = y[i:i+batch_size]
        
        # Convert targets to ResonantTensor (one-hot or indices)
        # For our simple loss, indices are fine if custom loss handles it
        # But let's use floats for MSE compatibility if needed
        # Mapping 0 -> [1, 0], 1 -> [0, 1] for checking
        batch_y_onehot = []
        for label in batch_y:
            if label == 0:
                batch_y_onehot.extend([1.0, 0.0])
            else:
                batch_y_onehot.extend([0.0, 1.0])
        
        target_tensor = ResonantTensor(batch_y_onehot, [len(batch_y), 2], [1.0]*(len(batch_y)*2), 64)
        batches.append((batch_X, target_tensor))
    return batches

def run_benchmark():
    print("=" * 70)
    print("Pure Syntonic XOR Benchmark (Retrocausal Training)")
    print("=" * 70)

    # 1. Prepare Data
    n_samples = 200
    X, y = make_xor_dataset(n_samples=n_samples, noise=0.1)
    
    split = int(0.8 * n_samples)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]
    
    print(f"Dataset: {n_samples} samples")
    print(f"Train: {len(X_train)}, Test: {len(X_test)}")
    
    # Batch the data
    batch_size = 16
    train_batches = create_batches(X_train, y_train, batch_size)
    # val_batches = create_batches(X_test, y_test, batch_size) # Trainer evaluates on train_data by default
    
    # 2. Initialize Model
    model = PureWindingNet(
        max_winding=5,
        base_dim=32,
        num_blocks=2,
        output_dim=2,
        precision=64,
        consensus_threshold=0.02
    )
    
    print("\nModel Initialized:")
    print(f"  Architecture: PureWindingNet")
    print(f"  Base Dim: {model.base_dim}")
    print(f"  Blocks: {model.num_blocks}")
    print(f"  Precision: {model.precision}-bit")

    # 3. Configure Trainer
    config = RESTrainingConfig(
        max_generations=20,     # It converges very fast
        population_size=16,
        syntony_threshold=0.8,
        log_interval=5
    )
    
    print(f"\n{'='*70}")
    print("Starting Retrocausal Training...")
    print(f"{'='*70}")
    
    trainer = RetrocausalTrainer(
        model=model,
        config=config,
        train_data=train_batches,
        val_data=None 
    )
    
    start_time = time.time()
    
    # Run training
    results = trainer.train()
    
    train_time = time.time() - start_time
    
    print(f"\nTraining Complete in {train_time:.2f}s")
    print(f"Final Syntony: {results['final_syntony']:.4f}")
    if 'final_loss' in results:
        print(f"Final Loss:    {results['final_loss']:.4f}")

    
    # 4. Evaluation
    print(f"\n{'='*70}")
    print("Evaluation")
    print(f"{'='*70}")
    
    model.eval()
    logits = model(X_test)
    preds = argmax_batch(logits, dim=1)
    test_acc = sum(1 for p, t in zip(preds, y_test) if p == t) / len(y_test)
    stats = model.get_blockchain_stats()
    
    print(f"Test Accuracy:     {test_acc:.2%}")
    print(f"Final Syntony:     {stats['network_syntony']:.4f}")
    print(f"Blockchain Length: {stats['blockchain_length']}")
    print(f"Training Time:     {train_time:.2f}s")
    
    if test_acc > 0.9:
        print("\nSUCCESS: WindingNet solved XOR > 90% accuracy!")
    else:
        print("\nWARNING: Accuracy below 90%. Adjust hyperparameters.")

if __name__ == "__main__":
    run_benchmark()
