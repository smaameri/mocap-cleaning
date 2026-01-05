import sys
import os

def debug_frame0(path):
    print(f"DEBUGGING FRAME 0: {os.path.basename(path)}")
    print("="*40)
    
    with open(path, 'r') as f:
        content = f.read().split()
    
    iterator = iter(content)
    nodes = []
    channels_map = {}
    
    # Parse Hierarchy to get channel order
    token = next(iterator, None)
    while token:
        if token == "MOTION": break
        if token in ["ROOT", "JOINT"]:
            name = next(iterator)
            nodes.append(name)
            while True:
                t = next(iterator)
                if t == "CHANNELS":
                    c = int(next(iterator))
                    channels = [next(iterator) for _ in range(c)]
                    channels_map[name] = channels
                    break
        token = next(iterator, None)
        
    # Skip to Motion Data
    while token != "Frame": token = next(iterator)
    next(iterator) # Time
    next(iterator) # Time val
    
    # Read Frame 0 Values
    print(f"{'JOINT NAME':<20} | {'CHANNELS':<25} | {'VALUES (Frame 0)'}")
    print("-" * 70)
    
    for name in nodes:
        chs = channels_map.get(name, [])
        vals = []
        try:
            for _ in range(len(chs)):
                vals.append(float(next(iterator)))
        except StopIteration:
            break
            
        # Format values
        val_str = ", ".join([f"{v:.2f}" for v in vals])
        ch_str = "".join([c[0] for c in chs]) # Xpos -> X
        
        # Only print relevant bones to reduce noise
        if name in ["Hips", "LeftArm", "RightArm", "LeftShoulder", "RightShoulder", "LeftUpLeg", "Spine"]:
            print(f"{name:<20} | {ch_str:<25} | {val_str}")

if __name__ == "__main__":
    # Hardcoded path based on your previous messages
    default_path = r"D:\REAL WORLD DATA\ProxiData-20251205T112014Z-3-001\ProxiData\ProxiData_raw\BVH\KO_vignette_06_004_CN_AM0806.bvh"
    
    input_path = default_path
    if len(sys.argv) > 1:
        input_path = sys.argv[1]
        
    debug_frame0(input_path)