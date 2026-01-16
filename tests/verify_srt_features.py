import syntonic
import time

def test_memory_pool_api():
    print("Testing SRT Memory Pool API...")
    
    # Test 1: Reserve Memory (device_idx, size)
    try:
        print("1. Calling srt_reserve_memory(0, 10MB)...")
        syntonic.srt_reserve_memory(0, 1024 * 1024 * 10) 
        print("   ✓ Success")
    except Exception as e:
        print(f"   ✗ Failed: {e}")

    # Test 2: Pool Stats (device_idx)
    try:
        print("2. Calling srt_pool_stats(0)...")
        stats = syntonic.srt_pool_stats(0)
        print(f"   ✓ Success: {stats}")
    except AttributeError:
        print("   ✗ Failed: Function not found")
    except Exception as e:
        print(f"   ✗ Failed: {e}")

    # Test 3: Resonance
    try:
        print("3. Calling srt_memory_resonance(0, 0)...")
        # Assuming block_id 0 exists or is valid to query? Or just passes through.
        res = syntonic.srt_memory_resonance(0, 0)
        print(f"   ✓ Success: {res}")
    except AttributeError:
        print("   ✗ Failed: Function not found")
    except Exception as e:
        print(f"   ✗ Failed: {e}")

    # Test 4: Wait for Resonance (device_idx)
    try:
        print("4. Calling srt_wait_for_resonance(0)...")
        start = time.time()
        syntonic.srt_wait_for_resonance(0) 
        print(f"   ✓ Success (waited {time.time() - start:.4f}s)")
    except AttributeError:
        print("   ✗ Failed: Function not found")
    except Exception as e:
        print(f"   ✗ Failed: {e}")

    # Test 5: Debug Stress (device_idx)
    try:
        print("5. Calling _debug_stress_pool_take(0)...")
        val = syntonic._debug_stress_pool_take(0)
        print(f"   ✓ Success: Returned {val}")
    except AttributeError:
        print("   ✗ Failed: Function not found")
    except Exception as e:
        print(f"   ✗ Failed: {e}")
        if "CUDA" in str(e) or "Driver" in str(e) or "PTX" in str(e):
             print("     (Expected failure if CUDA driver mismatch, but function exists)")

def test_cpu_fallback_explicit():
    print("\nTesting CPU Fallback Explicitly...")
    try:
        # Create two tensors using state factory
        t1 = syntonic.state([1.0, 2.0, 3.0])
        t2 = syntonic.state([4.0, 5.0, 6.0])
        
        # Try to move them to CUDA
        try:
            print("Attempting to move tensors to CUDA...")
            # Use syntonic.device object
            dev = syntonic.device("cuda")
            t1_cuda = t1.to(dev)
            t2_cuda = t2.to(dev)
            print(f"Moved to CUDA?. t1 device: {t1_cuda.device}")
            
            print("Performing addition (should trigger kernel load + fallback)...")
            t3 = t1_cuda + t2_cuda
            print(f"Result: {t3}")
            # If fallback works, it should compute correctly. 
            # Note: The fallback might return a CPU tensor if it failed on GPU.
            print(f"Result device: {t3.device}")
            
        except Exception as e:
            print(f"Caught exception during CUDA ops: {e}")
            if "FALLBACK" in str(e).upper() or "CPU" in str(e).upper():
                print("Fallback detected in exception message!")
            elif "PTX" in str(e):
                 print("PTX Error verify: " + str(e))
                
    except Exception as e:
        print(f"General failure: {e}")

if __name__ == "__main__":
    test_memory_pool_api()
    test_cpu_fallback_explicit()
