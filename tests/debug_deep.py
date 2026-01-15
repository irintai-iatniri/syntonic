from syntonic._core import ResonantTensor as RT

def debug():
    t = RT.zeros([2], 100)
    print(f"Attributes: {dir(t)}")
    
    print(f"Initial (CPU?): device_idx={t.device_idx()}")
    
    # Check if 'is_cuda' or similar exists
    if hasattr(t, 'is_cuda'):
        print(f"is_cuda: {t.is_cuda}")
    if hasattr(t, 'is_cpu'):
        print(f"is_cpu: {t.is_cpu}")
        
    # Try explicit to_cpu
    t_cpu = t.to_cpu()
    print(f"After to_cpu: device_idx={t_cpu.device_idx()}")
    
    # Try to_device(0)
    try:
        t_gpu = t.to_device(0)
        print(f"After to_device(0): device_idx={t_gpu.device_idx()}")
    except Exception as e:
        print(f"to_device(0) failed: {e}")

if __name__ == "__main__":
    debug()
