#!/usr/bin/env python3
"""Test basic CUDA operations without SRT"""

import syntonic

print('=== Basic CUDA Test (No SRT) ===\n')

# Check CUDA availability
print('CUDA available:', syntonic.cuda_is_available())
print('CUDA devices:', syntonic.cuda_device_count())

# Try direct TensorStorage creation on GPU
print('\nTest 1: Direct GPU tensor creation...')
try:
    from syntonic._core import TensorStorage
    import numpy as np
    
    data = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float64)
    print(f'Created numpy array: {data}')
    
    # Try creating directly on CPU first
    print('Creating CPU TensorStorage...')
    t_cpu = TensorStorage(data, list(data.shape), 'float64', 'cpu')
    print(f'✓ CPU tensor shape: {t_cpu.shape()}')
    
    # Try to move to GPU
    print('Moving to CUDA...')
    t_gpu = t_cpu.to_cuda(0)
    print(f'✓ GPU tensor shape: {t_gpu.shape()}')
    
    print('\n✅ Basic CUDA works!')
    
except Exception as e:
    print(f'\n❌ Error: {e}')
    import traceback
    traceback.print_exc()
