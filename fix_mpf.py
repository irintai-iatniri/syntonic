#!/usr/bin/env python3
"""Fix mpf() calls in srt_zero files - replace with native float."""

import re
from pathlib import Path

def fix_mpf_in_file(filepath):
    """Replace mpf() calls with native float values."""
    print(f"Processing {filepath}...")
    
    with open(filepath, 'r') as f:
        content = f.read()
    
    original = content
    
    # Replace mpf() calls with native float values
    # Handle mpf(variable) -> float(variable)
    content = re.sub(r'\bmpf\(([a-zA-Z_][a-zA-Z0-9_.]*)\)', r'float(\1)', content)
    
    # Handle mpf("string") -> float("string")
    content = re.sub(r'\bmpf\((["\'][^"\']+["\'])\)', r'float(\1)', content)
    
    # Handle mpf(number) -> number (for floats and ints)
    content = re.sub(r'\bmpf\((\d+\.\d+)\)', r'\1', content)
    content = re.sub(r'\bmpf\((\d+)\)', r'float(\1)', content)
    
    # Replace mp.sqrt with math.sqrt
    content = re.sub(r'mp\.sqrt', 'math.sqrt', content)
    
    # Replace mp.exp with math.exp  
    content = re.sub(r'mp\.exp', 'math.exp', content)
    
    if content != original:
        with open(filepath, 'w') as f:
            f.write(content)
        print(f"  ✓ Updated {filepath}")
        return True
    else:
        print(f"  - No changes needed for {filepath}")
        return False

if __name__ == '__main__':
    # Fix engine.py
    fix_mpf_in_file('srt_zero/engine.py')
    
    # Fix hierarchy.py
    fix_mpf_in_file('srt_zero/hierarchy.py')
    
    print("\nDone!")
