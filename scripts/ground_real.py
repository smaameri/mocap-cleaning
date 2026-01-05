import numpy as np
import argparse
import sys
import os
from tqdm import tqdm
from scipy.spatial.transform import Rotation as R

# ==========================================
# 1. SETTINGS (MUST MATCH YOUR RENDERER)
# ==========================================
SCALE_OFFSETS = 0.01   # Matches render_production_slow.py

# ==========================================
# 2. HELPER CLASSES (From Renderer)
# ==========================================
class BVHNode:
    def __init__(self, name, offset, parent=None):
        self.name = name
        self.offset = np.array(offset)
        self.parent = parent
        self.children = []

def load_skeleton_structure(path):
    with open(path, 'r') as f: content = f.read().split()
    iterator = iter(content)
    root = None
    node_stack = []
    token = next(iterator, None)
    while token:
        if token in ["ROOT", "JOINT"]:
            name = next(iterator)
            new_node = BVHNode(name, [0,0,0], node_stack[-1] if node_stack else None)
            if node_stack: node_stack[-1].children.append(new_node)
            else: root = new_node
            node_stack.append(new_node)
        elif token == "End":
            next(iterator)
            new_node = BVHNode("EndSite", [0,0,0], node_stack[-1])
            node_stack[-1].children.append(new_node)
            node_stack.append(new_node)
        elif token == "OFFSET":
            node_stack[-1].offset = np.array([float(next(iterator)) for _ in range(3)])
        elif token == "}" and node_stack:
            node_stack.pop()
        elif token == "MOTION":
            break
        token = next(iterator, None)
    return root

# ==========================================
# 3. KINEMATICS (From Renderer)
# ==========================================
def compute_fk(node, poses_map, parent_pos, parent_rot, trans_offset=None):
    local_rot = np.eye(3)
    if node.name in poses_map:
        local_rot = R.from_rotvec(poses_map[node.name]).as_matrix()
    
    global_rot = parent_rot @ local_rot
    scaled_offset = node.offset * SCALE_OFFSETS
    
    # ROOT HANDLING
    if trans_offset is not None:
        global_pos = trans_offset
    else:
        global_pos = parent_pos + (parent_rot @ scaled_offset)

    positions = [global_pos]
    
    for child in node.children:
        c_pos = compute_fk(child, poses_map, global_pos, global_rot, None)
        positions.extend(c_pos)
        
    return positions

# ==========================================
# 4. MAIN LOGIC
# ==========================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--npy", required=True, help="Path to the flying .npz file")
    parser.add_argument("--bvh", required=True, help="Path to the original .bvh file")
    parser.add_argument("--out", required=True, help="Output path for fixed .npz")
    args = parser.parse_args()

    print(f"--- GROUNDING WITH BVH SKELETON ---")
    
    # 1. Load Data
    data = np.load(args.npy)
    poses = data['poses']
    names = data['joint_names']
    
    # Handle key naming
    if 'transl' in data:
        trans = data['transl'].copy()
    else:
        trans = data['trans'].copy()
        
    T = poses.shape[0]
    
    # 2. Load Skeleton
    print(f">> Loading BVH Skeleton: {os.path.basename(args.bvh)}")
    root = load_skeleton_structure(args.bvh)
    
    # 3. Find Global Lowest Point (Z-axis)
    print(">> Scanning all frames for lowest Z point...")
    
    min_z = 9999.0
    
    # We don't need to check every single frame to find the floor, 
    # but checking stride of 5 ensures accuracy without waiting forever.
    check_frames = range(0, T, 5) 
    
    for f in tqdm(check_frames):
        pose_map = {name: poses[f, i] for i, name in enumerate(names)}
        
        # Run FK exactly like the renderer
        all_positions = compute_fk(root, pose_map, np.zeros(3), np.eye(3), trans[f])
        all_positions = np.array(all_positions)
        
        # Find lowest Z in this frame
        frame_min = np.min(all_positions[:, 2]) # Index 2 is Z (Up)
        
        if frame_min < min_z:
            min_z = frame_min

    print(f">> Lowest Point Detected (Z): {min_z:.4f} m")
    
    # 4. Apply Fix
    shift = -min_z
    print(f">> Applying Shift: {shift:.4f} m")
    
    trans[:, 2] += shift
    
    # 5. Save
    print(f">> Saving to {args.out}")
    np.savez(args.out,
             poses=poses,
             trans=trans,       
             transl=trans,      
             global_orient=data['global_orient'],
             body_pose=data['body_pose'],
             joint_names=names,
             fps=data['fps']
    )
    print("✅ DONE. Character is grounded.")

if __name__ == "__main__":
    main()