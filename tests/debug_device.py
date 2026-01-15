from syntonic._core import ResonantTensor as RT

def debug():
    try:
        t = RT.zeros([2, 2], 100)
        print(f"Created CPU tensor.")
        print(f"device_idx(): {t.device_idx()}")
        
        try:
            t.to_device(0)
            print("to_device(0) (int) worked.")
        except Exception as e:
            print(f"to_device(0) failed: {e}")
            
        try:
            t.to_device("cuda:0")
            print("to_device('cuda:0') (str) worked.")
        except Exception as e:
            print(f"to_device('cuda:0') failed: {e}")
            
    except Exception as e:
        print(f"General failure: {e}")

if __name__ == "__main__":
    debug()
