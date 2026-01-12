#!/usr/bin/env python3
"""Test SRT Memory Transfer Protocol Integration"""

import syntonic
import numpy as np

print('=== SRT Memory Transfer Protocol Integration Test ===\n')

# Create test data - smaller size for initial test
size = 100  # Small test
data = np.arange(size, dtype=np.float64)
print(f'Created {size:,} element array ({data.nbytes} bytes)\n')

# Test 1: Basic transfer without stats call
print('Test 1: Basic GPU transfer')
t = syntonic.state(data)
print(f'CPU state shape: {t.shape}')
print('Calling cuda()...')
t_gpu = t.cuda()
print(f'✓ GPU state shape: {t_gpu.shape}')
print('Transfer successful!\n')

# Test 2: Now try stats
print('Test 2: Checking SRT stats...')
try:
    stats_h2d = syntonic.srt_transfer_stats(0)
    print(f'Stats retrieved successfully')
    print(f'  - Transfers: {stats_h2d["total_transfers"]:.0f}')
    print(f'  - Bytes: {stats_h2d["total_bytes"]:.0f}')
except Exception as e:
    print(f'Stats failed: {e}')

# Check stats after H2D
stats_h2d = syntonic.srt_transfer_stats(0)
print(f'\nAfter Host→Device transfer:')
print(f'  - Transfers: {stats_h2d["total_transfers"]:.0f}')
print(f'  - Bytes: {stats_h2d["total_bytes"]:.0f} ({stats_h2d["total_bytes"] / 1e6:.2f} MB)')
print(f'  - Avg transfer time: {stats_h2d["avg_transfer_time_us"]:.2f} μs')
print(f'  - Resonance efficiency: {stats_h2d["resonance_efficiency"]:.6f}')
print(f'  - q-correction factor: {stats_h2d["q_correction_applied"]:.6f}')

print('\nNow exercising SRT async transfer path with larger data...')
big = np.arange(1024 * 1024, dtype=np.float64)
tb = syntonic.state(big).cuda_async()
stats_h2d = syntonic.srt_transfer_stats(0)
print('After SRT H2D:')
print(f'  - Transfers: {stats_h2d["total_transfers"]:.0f}')
print(f'  - Bytes: {stats_h2d["total_bytes"]:.0f} ({stats_h2d["total_bytes"] / 1e6:.2f} MB)')
print(f'  - Avg transfer time: {stats_h2d["avg_transfer_time_us"]:.2f} μs')
print(f'  - Resonance efficiency: {stats_h2d["resonance_efficiency"]:.6f}')
print(f'  - q-correction factor: {stats_h2d["q_correction_applied"]:.6f}')

# Transfer back to CPU
print('\nTransferring back to CPU (sync path)...')
t_cpu = t_gpu.cpu()
print(f'✓ CPU state shape: {t_cpu.shape}')

# Final stats
stats_final = syntonic.srt_transfer_stats(0)
print(f'\nFinal stats (after roundtrip):')
print(f'  - Total transfers: {stats_final["total_transfers"]:.0f}')
print(f'  - Total bytes: {stats_final["total_bytes"]:.0f} ({stats_final["total_bytes"] / 1e6:.2f} MB)')
print(f'  - Avg transfer time: {stats_final["avg_transfer_time_us"]:.2f} μs')
print(f'  - Resonance efficiency: {stats_final["resonance_efficiency"]:.6f}')
print(f'  - q-correction factor: {stats_final["q_correction_applied"]:.6f}')

# Verify data integrity
result = np.array(t_cpu.tolist())
if np.allclose(result, data):
    print('\n✅ Data integrity verified - transfer path works!')
    print('ℹ️ Note: SRT stats may be zero for synchronous transfers')
else:
    print('\n❌ Data corruption detected!')
    
print('\n=== Test Complete ===')
