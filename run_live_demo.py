import time
import random
import asyncio
import numpy as np
from syntonic.nn.training.trainer import RetrocausalTrainer, RESTrainingConfig
from syntonic.nn.winding.winding_net_pure import PureWindingNet
from syntonic._core import ResonantTensor, WindingState
from syntonic import state
import syntonic.sn as sn

def run_demo():
    print("Initializing Syntonic WindingNet...")
    # Create a model large enough to have >2048 parameters for a good visualization
    # 20 dims * 20 dims is small.
    # WindingNet has embeddings and blocks.
    model = PureWindingNet(max_winding=10, base_dim=32, num_blocks=3, output_dim=1)
    
    # Count parameters
    params = sum(len(p.to_list()) for p in model.parameters())
    print(f"Model has {params} parameters.")

    print("Generating Synthetic Winding Data...")
    train_data = []
    # Generate some simple patterns
    # If winding sum is even -> target 1.0, else 0.0
    for _ in range(10):
        n7 = random.randint(0, 5)
        n8 = random.randint(0, 5)
        w = WindingState(n7, n8, 0, 0)
        
        # Target must be ResonantTensor
        target_val = 1.0 if (n7 + n8) % 2 == 0 else 0.0
        target = state([target_val]) # Using synthetic factory 'state' which maps to ResonantTensor
        
        # Input must be List[WindingState] as expected by PureWindingNet.forward
        train_data.append(([w], target)) 

    print("Configuring Retrocausal Trainer...")
    config = RESTrainingConfig(
        max_generations=1000, # Run long enough to watch
        population_size=8,   # Keep it fast
        pull_strength=0.1,
        log_interval=5,      # Update fast
        attractor_min_syntony=0.1 # Allow low syntony to engage physics
    )

    trainer = RetrocausalTrainer(model, train_data, config=config)

    print("\n" + "="*50)
    print("STARTING LIVE TRAINING SESSION")
    print("1. Open dashboard/syntonic_console.html in your browser NOW.")
    print("2. You should see valid connection and live geometry.")
    print("="*50 + "\n")
    
    # Give user a moment to allow server to start (it starts in init)
    time.sleep(2)
    
    try:
        result = trainer.train()
        print("\nTraining Complete.")
        print(f"Final Syntony: {result['final_syntony']}")
    except KeyboardInterrupt:
        print("\nStopped by user.")
    except Exception as e:
        print(f"\nError during training: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    run_demo()
