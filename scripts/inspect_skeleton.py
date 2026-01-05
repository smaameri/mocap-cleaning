import sys
import os

def inspect_bvh(path):
    print(f"INSPECTING: {os.path.basename(path)}")
    print("="*40)
    
    with open(path, 'r') as f:
        lines = f.readlines()
        
    # We only care about the HEADER (before "MOTION")
    # We want to see the hierarchy and OFFSETS to understand bone directions.
    
    indent_level = 0
    for line in lines:
        line = line.strip()
        
        if line == "MOTION":
            break
            
        if line.startswith("ROOT") or line.startswith("JOINT"):
            name = line.split()[1]
            print(f"{'  ' * indent_level}➢ NODE: {name}")
            indent_level += 1
            
        elif line.startswith("End"):
            print(f"{'  ' * indent_level}➢ End Site")
            indent_level += 1
            
        elif line.startswith("OFFSET"):
            # This is the crucial part. It tells us the bone axis.
            parts = line.split()
            x, y, z = parts[1], parts[2], parts[3]
            print(f"{'  ' * indent_level}  [OFFSET]: {x}, {y}, {z}")
            
        elif line.startswith("}"):
            indent_level -= 1

if __name__ == "__main__":
    # Hardcoded input path for ease
    input_path = r"D:\REAL WORLD DATA\ProxiData-20251205T112014Z-3-001\ProxiData\ProxiData_raw\BVH\KO_vignette_06_004_CN_AM0806.bvh"
    
    if len(sys.argv) > 1:
        input_path = sys.argv[1]
        
    inspect_bvh(input_path)