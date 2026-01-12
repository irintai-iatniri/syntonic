#!/usr/bin/env python3
"""
CUDA Kernel Profiling Script for Syntonic Operations

This script profiles key Syntonic operations to identify performance bottlenecks.
"""

import time
import numpy as np
import subprocess
import os

def create_profile_script(operation_name, setup_code, operation_code, num_runs=100):
    """Create a profiling script for nsys."""
    script_content = f'''
import sys
sys.path.insert(0, '/home/Andrew/Documents/SRT Complete/implementation/syntonic')
import syntonic as syn
import numpy as np
import time

# Setup
{setup_code}

# Warmup
for _ in range(5):
    {operation_code}

# Profile runs
start_time = time.time()
for i in range({num_runs}):
    {operation_code}
end_time = time.time()

print(f"Total time for {num_runs} operations: {{end_time - start_time:.4f}} seconds")
print(f"Average time per operation: {{(end_time - start_time) / {num_runs} * 1000:.4f}} ms")
'''

    script_path = f'/tmp/profile_{operation_name}.py'
    with open(script_path, 'w') as f:
        f.write(script_content)

    return script_path

def profile_with_nsys(operation_name, script_path):
    """Profile using nsys."""
    print(f"\n=== Profiling {operation_name} ===")

    output_file = f'/tmp/{operation_name}_profile'

    cmd = [
        'nsys', 'profile',
        '--output', output_file,
        '--force-overwrite', 'true',
        '--trace', 'cuda,nvtx,cublas,cudnn',
        '--cuda-memory-usage', 'true',
        'python3', script_path
    ]

    print("Running nsys profiling...")
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode == 0:
        print(f"✓ Profile saved to {output_file}.nsys-rep")
        print("Timing output:")
        print(result.stdout)
        return True
    else:
        print(f"✗ Profiling failed: {result.stderr}")
        return False

def main():
    print("Starting CUDA Kernel Profiling for Syntonic Operations")
    print("=" * 60)

    operations = [
        ('addition_1000x1000',
         '''
a_np = np.random.randn(1000, 1000).astype(np.float32)
b_np = np.random.randn(1000, 1000).astype(np.float32)
a = syn.State.from_numpy(a_np).cuda()
b = syn.State.from_numpy(b_np).cuda()
         ''',
         'c = a + b'),

        ('matmul_500x500',
         '''
a_np = np.random.randn(500, 500).astype(np.float32)
b_np = np.random.randn(500, 500).astype(np.float32)
a = syn.State.from_numpy(a_np).cuda()
b = syn.State.from_numpy(b_np).cuda()
         ''',
         'c = a @ b'),

        ('exp_1000x1000',
         '''
a_np = np.random.randn(1000, 1000).astype(np.float32)
a = syn.State.from_numpy(a_np).cuda()
         ''',
         'b = a.exp()'),

        ('memory_transfer_h2d',
         '''
a_np = np.random.randn(1000, 1000).astype(np.float32)
         ''',
         'a = syn.State.from_numpy(a_np).cuda()'),

        ('memory_transfer_d2h',
         '''
a_np = np.random.randn(1000, 1000).astype(np.float32)
a = syn.State.from_numpy(a_np).cuda()
         ''',
         'b = a.numpy()'),
    ]

    successful_profiles = []

    for op_name, setup_code, op_code in operations:
        script_path = create_profile_script(op_name, setup_code, op_code)
        success = profile_with_nsys(op_name, script_path)
        if success:
            successful_profiles.append(op_name)

    print("\n" + "=" * 60)
    print("Profiling complete!")
    print(f"Successfully profiled {len(successful_profiles)} operations")

    if successful_profiles:
        print("\nTo analyze results, use:")
        print("  nsys-ui /tmp/*_profile.nsys-rep")
        print("\nOr get summary stats:")
        for op in successful_profiles:
            print(f"  nsys stats /tmp/{op}_profile.nsys-rep")

if __name__ == "__main__":
    main()