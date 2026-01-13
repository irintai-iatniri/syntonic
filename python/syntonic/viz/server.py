import asyncio
import websockets
import json
import threading
import numpy as np

# Global state buffer (Thread-safe)
LATEST_STATE = {
    "weights": [],
    "syntony": 0.0,
    "temperature": 0.0,
    "phase": "D"
}

async def handler(websocket):
    """Push state to the browser at 60fps"""
    while True:
        try:
            # Serialize the latest state
            # Note: We downsample weights to avoid choking the websocket
            data = json.dumps(LATEST_STATE)
            await websocket.send(data)
            await asyncio.sleep(0.016) # ~60 FPS
        except websockets.exceptions.ConnectionClosed:
            break

async def start_server():
    print(">>> SYNTONIC CONSOLE: Listening on localhost:8765")
    async with websockets.serve(handler, "localhost", 8765):
        # Create a Future that never completes to keep the server running
        await asyncio.Future()

def launch_background_thread():
    """Call this from trainer.py to start the server non-blocking"""
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    t = threading.Thread(target=loop.run_until_complete, args=(start_server(),), daemon=True)
    t.start()

def update_monitor(model, syntony, temp, phase):
    """Called by the Trainer loop to update global state"""
    global LATEST_STATE
    
    # Extract raw weights from the model (CPU copy)
    # Support both PyTorch (detach().cpu()) and Pure Syntonic (to_list())
    flat_weights = []
    
    # Check if model has standard parameters() method
    if not hasattr(model, 'parameters'):
         return

    for p in model.parameters():
        w_data = None
        
        # Handle Pure Syntonic Parameter
        if hasattr(p, 'to_list'): 
            w_data = np.array(p.to_list())
            
        # Handle Pure Syntonic Parameter (alternative access)
        elif hasattr(p, 'tensor') and hasattr(p.tensor, 'to_floats'):
             w_data = np.array(p.tensor.to_floats())
             
        # Handle PyTorch Tensor
        elif hasattr(p, 'detach'):
            w_data = p.detach().cpu().numpy().flatten()
            
        if w_data is not None:
            flat_weights.append(w_data)
        
        # Limit the size to avoid heavy serialization
        current_len = sum(len(w) for w in flat_weights)
        if current_len > 6144: # 2048 * 3
            break
    
    if not flat_weights:
        return

    full_vector = np.concatenate(flat_weights)
    if len(full_vector) > 6144:
        full_vector = full_vector[:6144] # Limit to 2048 * 3
    
    # Normalize for visualization (-50 to 50 range)
    if full_vector.std() > 0:
        full_vector = (full_vector - full_vector.mean()) / (full_vector.std() + 1e-6) * 25.0
    
    LATEST_STATE = {
        "weights": full_vector.tolist(),
        "syntony": float(syntony),
        "temperature": float(temp),
        "phase": "D" if phase > 0.5 else "H"
    }
