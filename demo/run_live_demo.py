import time
import random
import asyncio
from python.syntonic.nn.training.trainer import RetrocausalTrainer, RESTrainingConfig
from python.syntonic.nn.winding.winding_net_pure import PureWindingNet
from python.syntonic._core import ResonantTensor, WindingState
from python.syntonic import state
import python.syntonic.sn as sn

def run_demo():
    print("Initializing Syntonic WindingNet...")
    # Create a model large enough to have >2048 parameters for a good visualization
    # 20 dims * 20 dims is small.
    # WindingNet has embeddings and blocks.
    model = PureWindingNet(max_winding=10, base_dim=32, num_blocks=6, output_dim=1)
    
    # Count parameters
    params = sum(len(p.to_list()) for p in model.parameters())
    print(f"Model has {params} parameters.")

    # Try to load a CSV dataset if present, otherwise fall back to synthetic generation
    train_data = []
    dataset_path = "data/winding_dataset.csv"
    try:
        import csv
        with open(dataset_path, newline='') as csvfile:
            reader = csv.DictReader(filter(lambda row: row[0] != '#', csvfile))
            for row in reader:
                try:
                    n7 = int(row.get('n7', 0))
                    n8 = int(row.get('n8', 0))
                    target_val = float(row.get('target', (n7 + n8) % 2 == 0 and 1.0 or 0.0))
                    w = WindingState(n7, n8, 0, 0)
                    target = ResonantTensor([target_val], [1, 1], [0.0], 100)
                    train_data.append(([w], target))
                except Exception:
                    continue
        if not train_data:
            raise FileNotFoundError
        print(f"Loaded dataset from {dataset_path} with {len(train_data)} samples.")
    except Exception:
        print("Generating Synthetic Winding Data...")
        # Generate some simple patterns
        # If winding sum is even -> target 1.0, else 0.0
        for _ in range(200):
            n7 = random.randint(0, 5)
            n8 = random.randint(0, 5)
            w = WindingState(n7, n8, 0, 0)
            target_val = 1.0 if (n7 + n8) % 2 == 0 else 0.0
            target = ResonantTensor([target_val], [1, 1], [0.0], 100)  # Shape [1, 1] to match output
            train_data.append(([w], target))

    print("Testing forward pass...")
    test_input = [WindingState(1, 2, 0, 0)]
    try:
        test_output = model(test_input)
        print(f"Test output shape: {test_output.shape}")
        print(f"Test output data: {test_output.to_floats()}")
    except Exception as e:
        print(f"Test forward failed: {e}")
        import traceback
        traceback.print_exc()
        return 

    print("Configuring Retrocausal Trainer...")
    config = RESTrainingConfig(
        max_generations=100000, # Run long enough to watch
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