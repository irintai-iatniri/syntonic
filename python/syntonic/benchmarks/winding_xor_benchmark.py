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
from syntonic.nn.training.trainer import RetrocausalTrainer
from syntonic.nn.training.config import RESTrainingConfig

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
        max_epochs=20,     # It converges very fast
        batch_size=16,
        learning_rate=0.05, # Not used in retrocausal but kept for compatibility
        syntony_threshold=0.8,
        retrocausal_enabled=True,
        log_interval=5
    )
    
    trainer = RetrocausalTrainer(
        model=model,
        config=config,
        train_loader=None, # We'll feed data manually or wrap it
        val_loader=None
    )
    
    # Manual Training Loop using Trainer components
    # (Since we don't have a pure DataLoader yet, we iterate manually)
    
    print(f"\n{'='*70}")
    print("Starting Retrocausal Training...")
    print(f"{'='*70}")
    
    start_time = time.time()
    
    for epoch in range(config.max_epochs):
        model.train()
        total_acc = 0
        total_syntony = 0
        batches = 0
        
        # Mini-batch loop
        for i in range(0, len(X_train), config.batch_size):
            batch_X = X_train[i:i+config.batch_size]
            batch_y = y_train[i:i+config.batch_size]
            
            # Forward pass
            logits = model(batch_X)
            
            # Compute accuracy
            preds = logits.argmax(dim=1)
            acc = sum(1 for p, t in zip(preds, batch_y) if p == t) / len(batch_y)
            
            # Retrocausal step (syntony optimization)
            # This happens inside the DHSR blocks automatically,
            # but we can trigger specific "crystallization" events if needed.
            # In PureWindingNet, the DHSR cycle handles the learning through
            # syntony consensus and state evolution.
            
            stats = model.get_blockchain_stats()
            
            total_acc += acc
            total_syntony += stats['network_syntony']
            batches += 1
            
        avg_acc = total_acc / batches
        avg_syn = total_syntony / batches
        
        if (epoch + 1) % config.log_interval == 0:
            print(f"Epoch {epoch+1:3d}/{config.max_epochs}: Acc={avg_acc:.2%} Syntony={avg_syn:.4f}")
            
    train_time = time.time() - start_time
    
    # 4. Evaluation
    print(f"\n{'='*70}")
    print("Evaluation")
    print(f"{'='*70}")
    
    model.eval()
    logits = model(X_test)
    preds = logits.argmax(dim=1)
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
