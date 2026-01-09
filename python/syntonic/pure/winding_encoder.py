"""
PureWindingEncoder: Maps winding states to resonant embeddings.
Strictly zero dependencies on PyTorch or NumPy.
"""

from syntonic._core import ResonantTensor, GoldenExact, enumerate_windings, WindingState
import random
import math

class PureWindingEncoder:
    """
    Pure Python Winding Encoder.
    Uses ResonantTensor as an embedding table.
    """
    def __init__(self, max_n=5, embed_dim=64, precision=100):
        self.max_n = max_n
        self.embed_dim = embed_dim
        self.precision = precision
        
        # 1. Enumerate all possible windings within max_n
        self.windings = enumerate_windings(max_n)
        self.num_windings = len(self.windings)
        
        # 2. Map winding state to index
        self.winding_to_idx = {
            self._key(w): i for i, w in enumerate(self.windings)
        }
        
        # 3. Create embedding table as a ResonantTensor
        # Initialize with Golden Measure: weight ~ exp(-|n|²/2φ)
        lattice = []
        mode_norms = []
        PHI = (1 + math.sqrt(5)) / 2
        
        for w in self.windings:
            norm_sq = w.norm_squared
            # Scaled initialization: small random integers, attenuated by norm
            scale = math.exp(-norm_sq / (2 * PHI))
            
            for _ in range(embed_dim):
                # Random integer components (-1, 0, 1) scaled by golden measure logic
                # For pure lattice, we just use small integers
                if random.random() < scale:
                    a = random.randint(-1, 1)
                    b = 0
                else:
                    a, b = 0, 0
                lattice.append(GoldenExact.from_integers(a, b))
                mode_norms.append(float(norm_sq))
                
        self.weight = ResonantTensor.from_golden_exact(
            lattice, [self.num_windings, embed_dim], mode_norms
        )

    def _key(self, w):
        return (w.n7, w.n8, w.n9, w.n10)

    def encode(self, winding_states):
        """
        Encode a list of WindingState objects into a ResonantTensor.
        
        Args:
            winding_states: List of WindingState
            
        Returns:
            ResonantTensor of shape [batch, embed_dim]
        """
        batch_size = len(winding_states)
        result_lattice = []
        result_norms = []
        
        # Access original lattice for "lookup"
        # Since we don't have native slice/index yet, we do it in Python
        # (This is okay for the encoder as it's usually at the boundary)
        source_lattice = self.weight.to_lattice_list()
        
        for w in winding_states:
            idx = self.winding_to_idx.get(self._key(w))
            if idx is None:
                raise ValueError(f"Winding {w} out of range (max_n={self.max_n})")
                
            start = idx * self.embed_dim
            end = (idx + 1) * self.embed_dim
            result_lattice.extend(source_lattice[start:end])
            
            # Winding norm is the same for the entire embedding vector
            norm = float(w.norm_squared)
            result_norms.extend([norm] * self.embed_dim)
            
        return ResonantTensor.from_golden_exact(
            result_lattice, [batch_size, self.embed_dim], result_norms
        )

    def __repr__(self):
        return f"PureWindingEncoder(max_n={self.max_n}, num_windings={self.num_windings}, embed_dim={self.embed_dim})"
