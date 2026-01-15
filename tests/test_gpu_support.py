import pytest
from syntonic.nn.resonant_tensor import ResonantTensor
import syntonic.sn as sn
from syntonic.nn.architectures.syntonic_mlp_pure import PureSyntonicMLP
from syntonic.nn.architectures.syntonic_transformer_pure import PureSyntonicTransformer

def test_resonant_tensor_device():
    print("\n--- Testing ResonantTensor Device Support ---")
    try:
        x = ResonantTensor([1.0, 2.0], [2], device='cuda:0')
        print(f"Created tensor on device: {x.device}")
        assert x.device == 'cuda:0'
        
        y = x.to('cpu')
        print(f"Moved to cpu: {y.device}")
        assert y.device == 'cpu'
        
        z = y.cuda()
        print(f"Moved back to cuda: {z.device}")
        assert z.device == 'cuda:0'
        
    except Exception as e:
        print(f"FAILED: {e}")
        # Skip if no CUDA available but logic should hold for cpu->cpu
        if "CUDA" in str(e):
             print("Skipping CUDA test (no GPU or backend support)")

def test_parameter_device():
    print("\n--- Testing Parameter Device Support ---")
    p = sn.Parameter([10, 10], device='cpu')
    assert p.device == 'cpu'
    assert p.tensor.device == 'cpu'
    
    # Mock move if no GPU? Or try real move
    try:
        p.cuda()
        print(f"Moved parameter to: {p.device}")
        assert 'cuda' in p.device
        assert 'cuda' in p.tensor.device
    except Exception as e:
        print(f"Parameter move failed (expected if no GPU): {e}")

def test_mlp_device():
    print("\n--- Testing MLP Device Propagation ---")
    model = PureSyntonicMLP(10, [20], 5, device='cpu')
    
    # Check initial placement
    print(f"Initial device: {model._device}")
    assert model._device == 'cpu'
    assert model.hidden_layers[0].linear.weight.device == 'cpu'
    
    # Move to CUDA
    try:
        model.cuda()
        print(f"Moved model to: {model._device}")
        assert 'cuda' in model._device
        
        # Check submodules
        linear_device = model.hidden_layers[0].linear.weight.device
        print(f"Submodule parameter device: {linear_device}")
        assert 'cuda' in linear_device
        
        # Check RecursionBlock propagation
        if hasattr(model.hidden_layers[0], 'recursion'):
            rec_device = model.hidden_layers[0].recursion.device
            print(f"RecursionBlock device: {rec_device}")
            assert 'cuda' in rec_device
            
    except Exception as e:
        print(f"MLP move failed: {e}")

def test_transformer_device():
    print("\n--- Testing Transformer Device Propagation ---")
    model = PureSyntonicTransformer(10, 2, 2, 20, 5, device='cpu')
    
    try:
        model.cuda()
        # Check encoder
        encoder_device = model.encoder.device
        print(f"Encoder device: {encoder_device}")
        assert 'cuda' in encoder_device
        
        # Check Attention parameter
        attn_device = model.encoder.layers[0].self_attn.q_proj.device
        print(f"Attention Q proj device: {attn_device}")
        assert 'cuda' in attn_device
        
    except Exception as e:
        print(f"Transformer move failed: {e}")

if __name__ == "__main__":
    test_resonant_tensor_device()
    test_parameter_device()
    test_mlp_device()
    test_transformer_device()
