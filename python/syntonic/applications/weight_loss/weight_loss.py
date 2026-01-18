import time
import math
import requests
import json
from typing import List, Tuple

# --- SYNTONIC IMPORTS ---
import syntonic as syn
from python.syntonic.nn.architectures.syntonic_mlp import PureSyntonicMLP
from syntonic.nn.training.trainer import RetrocausalTrainer, RESTrainingConfig
from syntonic.nn.resonant_tensor import ResonantTensor

# --- CONFIGURATION ---
USDA_API_KEY = "Dov8H2sSCjEG5brmfiAOJpKeyAiYDHhNTbDiolVL"  # Your Key
USER_WEIGHT = 267
GALLBLADDER_FAT_LIMIT = 10.0  # grams per meal

# --- STEP 1: DATA ACQUISITION & TEACHER SIGNAL ---

class USDADataFetcher:
    def __init__(self, api_key):
        self.api_key = api_key
        self.base_url = "https://api.nal.usda.gov/fdc/v1"
    
    def fetch_training_batch(self, queries: List[str]) -> List[dict]:
        """Fetches real food data to build the training set."""
        dataset = []
        print(f"📡 Connecting to USDA API for {len(queries)} food items...")
        
        for q in queries:
            try:
                # Fetch Foundation foods (raw ingredients)
                url = f"{self.base_url}/foods/search"
                params = {
                    "query": q,
                    "api_key": self.api_key,
                    "pageSize": 1,
                    "dataType": ["Foundation", "SR Legacy"]
                }
                resp = requests.get(url, params=params)
                data = resp.json()
                
                if data.get('foods'):
                    food = data['foods'][0]
                    macros = self._extract_macros(food)
                    score = self._calculate_teacher_score(macros)
                    
                    dataset.append({
                        "name": food['description'],
                        "inputs": [macros['p'], macros['f'], macros['c'], macros['kcal']],
                        "target": [score]
                    })
                    print(f"   ✓ Found: {food['description'][:30]}... (Score: {score:.2f})")
                time.sleep(0.5) # Be nice to the API
            except Exception as e:
                print(f"   ✗ Error fetching {q}: {e}")
                
        return dataset

    def _extract_macros(self, food_item):
        """Extracts P/F/C/Kcal per 100g."""
        nutrients = {'p': 0.0, 'f': 0.0, 'c': 0.0, 'kcal': 0.0}
        mapping = {"203": 'p', "204": 'f', "205": 'c', "208": 'kcal'}
        
        for n in food_item['foodNutrients']:
            nid = str(n['nutrientNumber'])
            if nid in mapping:
                nutrients[mapping[nid]] = n['value']
        return nutrients

    def _calculate_teacher_score(self, m):
        """
        THE LOGIC KERNEL:
        Calculates the 'Ideal' score based on your biology.
        Returns 0.0 (Toxic) to 1.0 (Perfect Fuel).
        """
        # 1. Gallbladder Safety Check (Hard Cutoff logic)
        if m['f'] > GALLBLADDER_FAT_LIMIT:
            # Penalize heavily if fat > 10g
            return 0.1
            
        # 2. Thermic Efficiency (Protein focus)
        # Ideal: High Protein, Low/Moderate Calorie
        protein_ratio = (m['p'] * 4) / max(1, m['kcal'])
        
        # 3. Insulin Control (Carb penalty)
        # Penalize high carb density
        carb_ratio = (m['c'] * 4) / max(1, m['kcal'])
        
        score = 0.5 + (protein_ratio * 0.5) - (carb_ratio * 0.3)
        
        # 4. Global constraints
        return max(0.0, min(1.0, score))

# --- STEP 2: MODEL DEFINITION ---

class NutriNet(PureSyntonicMLP):
    """
    Wrapper to ensure PureSyntonicMLP adheres to the RetrocausalTrainer protocol.
    """
    def get_weights(self) -> List[ResonantTensor]:
        """Collect all evolving tensors (weights/biases) from the network."""
        weights = []
        for layer in self.hidden_layers:
            # ResonantLinear wraps weights in .linear
            if hasattr(layer, 'linear'):
                weights.append(layer.linear.weight.tensor)
                if layer.linear.bias is not None:
                    weights.append(layer.linear.bias.tensor)
        # Output layer
        weights.append(self.output_layer.weight.tensor)
        if self.output_layer.bias is not None:
            weights.append(self.output_layer.bias.tensor)
        return weights

    def set_weights(self, weights: List[ResonantTensor]) -> None:
        """Apply evolved weights back to the network."""
        idx = 0
        for layer in self.hidden_layers:
            if hasattr(layer, 'linear'):
                layer.linear.weight.tensor = weights[idx]
                idx += 1
                if layer.linear.bias is not None:
                    layer.linear.bias.tensor = weights[idx]
                    idx += 1
        
        self.output_layer.weight.tensor = weights[idx]
        idx += 1
        if self.output_layer.bias is not None:
            self.output_layer.bias.tensor = weights[idx]

# --- STEP 3: MAIN EXECUTION ---

def main():
    print("🔮 Initializing Resonant Metabolic Engine...")
    
    # 1. Prepare Data
    fetcher = USDADataFetcher(USDA_API_KEY)
    
    # Training Data: Mix of "Good" (Tier 1/2) and "Bad" (High Fat/Carb) foods
    training_foods = [
        # GOOD (High Protein / Low Fat)
        "raw chicken breast", "egg whites", "cod raw", "spinach raw", 
        "tuna in water", "turkey breast", "broccoli", "shrimp raw",
        # BAD (Gallbladder Risk / Low Thermic)
        "cheddar cheese", "bacon raw", "sausage", "avocado", 
        "almonds", "butter", "heavy cream", "croissant"
    ]
    
    raw_data = fetcher.fetch_training_batch(training_foods)
    
    # Convert to ResonantTensors (The format Syntonic expects)
    train_data = []
    for item in raw_data:
        # Normalize inputs roughly to 0-1 range (assuming max 100g/1000kcal)
        inputs = [x / 100.0 for x in item['inputs']] 
        # Create Tensors
        t_in = ResonantTensor(inputs, shape=[1, 4])
        t_target = ResonantTensor(item['target'], shape=[1, 1])
        train_data.append((t_in, t_target))

    # 2. Initialize Network
    # Input: 4 (Prot, Fat, Carb, Kcal) -> Hidden: [16, 8] -> Output: 1 (Score)
    model = NutriNet(
        input_dim=4,
        hidden_dims=[16, 8],
        output_dim=1,
        use_recursion=True # Enable DHSR cycles for better convergence
    )
    
    # 3. Configure Retrocausal Training
    # We use "Attractor Dynamics" instead of Gradient Descent
    config = RESTrainingConfig(
        max_generations=50,       # Evolve for 50 cycles
        population_size=20,       # 20 parallel timelines per weight
        pull_strength=0.3,        # Strength of the "Teacher" attractor
        syntony_target=syn.PHI_NUMERIC - syn.Q_DEFICIT_NUMERIC, # 1.59...
        enable_viz=False
    )
    
    print("\n🧬 Beginning Retrocausal Evolution...")
    trainer = RetrocausalTrainer(model, train_data, config=config)
    
    # 4. Train
    results = trainer.train()
    
    print("\n" + "="*40)
    print("TRAINING COMPLETE")
    print("="*40)
    print(f"Final Network Syntony: {results['final_syntony']:.4f}")
    print(f"Convergence Loss:      {results['final_loss']:.4f}")
    
    # 5. Inference Test (Verify "Intuition")
    print("\n🧪 TESTING MODEL INTUITION:")
    
    test_cases = [
        ("Ideal Meal (Chicken)", [31.0, 3.0, 0.0, 165.0]), # P, F, C, Kcal
        ("Danger Meal (Cheese)", [25.0, 33.0, 1.0, 400.0]), # High Fat!
    ]
    
    for name, macros in test_cases:
        # Prepare input
        norm_macros = [x / 100.0 for x in macros]
        t_in = ResonantTensor(norm_macros, shape=[1, 4])
        
        # Forward pass
        prediction = model.forward(t_in)
        score = prediction.to_floats()[0]
        
        print(f"\nFood: {name}")
        print(f"Macros: P:{macros[0]}g F:{macros[1]}g")
        print(f" AI Resonance Score: {score:.4f} / 1.0")
        
        if score > 0.8:
            print(" >> ✅ APPROVED: High Thermic / Gallbladder Safe")
        elif score < 0.4:
            print(" >> ❌ REJECTED: Low Thermic / High Fat Risk")
        else:
            print(" >> ⚠️ CAUTION: Moderate suitability")

if __name__ == "__main__":
    main()
