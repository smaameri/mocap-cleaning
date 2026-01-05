import numpy as np
import argparse
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.spatial.transform import Rotation as R
import os

# ==========================================
# 1. ZXY PARSER (Robust)
# ==========================================
def load_raw_zxy_bvh(path):
    print(f"Loading: {path}")
    with open(path, 'r') as f: 
        content = f.read().split()
    
    iterator = iter(content)
    
    class Node:
        def __init__(self, name, parent=None):
            self.name = name
            self.parent = parent
            self.offset = np.zeros(3)
            self.channels = []
            self.children = []
            
    root = None
    node_stack = []
    
    # 1. Parse Hierarchy
    token = next(iterator, None)
    while token:
        if token == "MOTION":
            break
            
        if token in ["ROOT", "JOINT"]:
            name = next(iterator)
            node = Node(name, node_stack[-1] if node_stack else None)
            if node_stack: node_stack[-1].children.append(node)
            else: root = node
            node_stack.append(node)
            
        elif token == "OFFSET":
            node_stack[-1].offset = np.array([float(next(iterator)) for _ in range(3)])
            
        elif token == "CHANNELS":
            c_count = int(next(iterator))
            node_stack[-1].channels = [next(iterator) for _ in range(c_count)]
            
        elif token == "End":
            next(iterator) # Site
            node = Node("EndSite", node_stack[-1])
            node_stack[-1].children.append(node)
            node_stack.append(node)
            
        elif token == "}":
            node_stack.pop()
            
        token = next(iterator, None)
        
    # 2. Parse Motion
    num_frames = 0
    frame_time = 0.00833
    
    while token:
        if token == "Frames:": 
            num_frames = int(next(iterator))
        elif token == "Frame": 
            if next(iterator) == "Time:": 
                frame_time = float(next(iterator))
            break
        token = next(iterator, None)
    
    # 3. Read All Values
    print(f"Reading {num_frames} frames...")
    vals = []
    try:
        while True:
            t = next(iterator, None)
            if t is None: break
            vals.append(float(t))
    except ValueError:
        pass
        
    # 4. Create Numpy Array (FIXED LOGIC)
    if num_frames > 0 and len(vals) > 0:
        # Calculate channels per frame
        channels_per_frame = len(vals) // num_frames
        frames = np.array(vals[:num_frames*channels_per_frame]).reshape(num_frames, channels_per_frame)
    else:
        print("Error: No frame data found.")
        frames = np.zeros((1, 1))

    return root, frames

# ==========================================
# 2. FK & RENDER
# ==========================================
def compute_fk(node, frame_data, channel_ptr, parent_pos, parent_rot):
    local_pos = node.offset.copy()
    local_rot = np.eye(3)
    
    # Consume channels for this node
    if hasattr(node, 'channels') and node.channels:
        vals = frame_data[channel_ptr : channel_ptr + len(node.channels)]
        
        # Apply Position
        if "Xposition" in node.channels: local_pos[0] += vals[node.channels.index("Xposition")]
        if "Yposition" in node.channels: local_pos[1] += vals[node.channels.index("Yposition")]
        if "Zposition" in node.channels: local_pos[2] += vals[node.channels.index("Zposition")]
            
        # Apply Rotation (Strict ZXY Order Check)
        rot_vals = []
        rot_order = ""
        for ch in node.channels:
            if "rotation" in ch:
                rot_vals.append(vals[node.channels.index(ch)])
                rot_order += ch[0].lower() # z, x, y
        
        if rot_vals:
            # BVH Euler is usually Intrinsic.
            r = R.from_euler(rot_order, rot_vals, degrees=True)
            local_rot = r.as_matrix()
            
        new_ptr = channel_ptr + len(node.channels)
    else:
        new_ptr = channel_ptr

    # Global Transform
    global_rot = parent_rot @ local_rot
    global_pos = parent_pos + (parent_rot @ local_pos)
    
    positions = [global_pos]
    connections = []
    
    for child in node.children:
        c_pos, c_conn, new_ptr = compute_fk(child, frame_data, new_ptr, global_pos, global_rot)
        positions.extend(c_pos)
        connections.append([global_pos, c_pos[0]])
        connections.extend(c_conn)
        
    return positions, connections, new_ptr

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    args = parser.parse_args()
    
    root, frames = load_raw_zxy_bvh(args.input)
    
    # Setup Plot
    fig = plt.figure(figsize=(10,10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Scale guess (if hips > 50, it's CM, scale down for view)
    scale_view = 1.0
    if np.max(np.abs(frames[0, :3])) > 50:
        scale_view = 0.01
        print("Detected CM units, scaling visualization by 0.01")

    def update(frame_idx):
        ax.cla()
        ax.set_title(f"Raw BVH Frame {frame_idx}")
        ax.set_axis_off()
        
        # Run FK
        positions, connections, _ = compute_fk(root, frames[frame_idx], 0, np.zeros(3), np.eye(3))
        
        # Convert to numpy for plotting
        positions = np.array(positions) * scale_view
        
        # PLOT: Swapping Y and Z so Y is Up in the plot
        # Matplotlib default: Z is up. BVH default: Y is up.
        # We plot BVH X->X, BVH Z->Y, BVH Y->Z
        
        # Scatter Joints
        ax.scatter(positions[:,0], positions[:,2], positions[:,1], s=10, c='red', depthshade=False)
        
        # Draw Bones
        for start, end in connections:
            s = np.array(start) * scale_view
            e = np.array(end) * scale_view
            ax.plot([s[0], e[0]], [s[2], e[2]], [s[1], e[1]], c='blue', lw=1)
            
        # Camera Center on Root
        root_p = positions[0]
        r = 1.5 # meters view radius
        ax.set_xlim(root_p[0]-r, root_p[0]+r)
        ax.set_ylim(root_p[2]-r, root_p[2]+r)
        ax.set_zlim(root_p[1]-r, root_p[1]+r)

    out_name = args.input.replace(".bvh", "_RAW_DEBUG.mp4")
    print(f"Rendering debug video to {out_name}...")
    
    # Render shorter segment (first 300 frames) to be fast
    max_frames = min(300, len(frames))
    ani = FuncAnimation(fig, update, frames=range(0, max_frames, 2), interval=33)
    ani.save(out_name, fps=30)
    print("Done.")

if __name__ == "__main__":
    main()