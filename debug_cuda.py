import sys
try:
    from syntonic.core import ResonantTensor
    print("Import successful")
    t = ResonantTensor([1.0], [1])
    print("Tensor created on CPU")
    t_gpu = t.to_device(0)
    print("Tensor moved to GPU")
    print(f"Device index: {t_gpu.device_idx()}")
except Exception as e:
    print(f"FAILED: {e}")
    import traceback
    traceback.print_exc()
