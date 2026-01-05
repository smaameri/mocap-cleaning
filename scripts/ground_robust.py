import numpy as np
import argparse
import sys
import os
from tqdm import tqdm
from scipy.spatial.transform import Rotation as R

# ==========================================
# 1. SETTINGS
# ==========================================
SCALE_OFFSETS = 0.01
# Uses the 1st percentile instead of 0 (min) to ignore outlier glitches
FLOOR_PERCENTILE = 1.0 

# ==========================================
# 2. HELPER CLASSES
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
# 3. KINEMATICS
# ==========================================
def compute_fk(node, poses_map, parent_pos, parent_rot, trans_offset=None):
    local_rot = np.eye(3)
    if node.name in poses_map:
        local_rot = R.from_rotvec(poses_map[node.name]).as_matrix()
    
    global_rot = parent_rot @ local_rot
    scaled_offset = node.offset * SCALE_OFFSETS
    
    if trans_offset is not None:
        global_pos = trans_offset
    else:
        global_pos = parent_pos + (parent_rot @ scaled_offset)

    # Return only this node's position + dict of children
    positions = {node.name: global_pos}
    
    for child in node.children:
        child_pos = compute_fk(child, poses_map, global_pos, global_rot, None)
        positions.update(child_pos)
        
    return positions

# ==========================================
# 4. MAIN LOGIC
# ==========================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--npy", required=True)
    parser.add_argument("--bvh", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--manual_shift", type=float, default=0.0, help="Manually move up (+) or down (-) in meters")
    args = parser.parse_args()

    print(f"--- ROBUST GROUNDING (Percentile Method) ---")
    
    # 1. Load Data
    data = np.load(args.npy)
    poses = data['poses']
    names = data['joint_names']
    
    if 'transl' in data: trans = data['transl'].copy()
    else: trans = data['trans'].copy()
    
    T = poses.shape[0]
    
    # 2. Load Skeleton
    root = load_skeleton_structure(args.bvh)
    
    # 3. Collect Lowest Z points
    print(">> Scanning for floor (ignoring outliers)...")
    
    lowest_z_per_frame = []
    
    # Check every 5th frame for speed
    check_frames = range(0, T, 5) 
    
    for f in tqdm(check_frames):
        pose_map = {name: poses[f, i] for i, name in enumerate(names)}
        
        # Get all joint positions
        all_pos_dict = compute_fk(root, pose_map, np.zeros(3), np.eye(3), trans[f])
        
        # Filter for Feet only
        feet_z = []
        for jname, pos in all_pos_dict.items():
            ln = jname.lower()
            if "foot" in ln or "toe" in ln or "ankle" in ln or "endsite" in ln:
                feet_z.append(pos[2]) # Z is Up
        
        if feet_z:
            lowest_z_per_frame.append(min(feet_z))
        else:
            # Fallback if naming fails
            all_z = [p[2] for p in all_pos_dict.values()]
            lowest_z_per_frame.append(min(all_z))

    # 4. Find Robust Floor Level
    # Use 1st percentile to find the "real" floor and ignore glitchy deep values
    robust_floor_z = np.percentile(lowest_z_per_frame, FLOOR_PERCENTILE)
    
    print(f">> Robust Floor Detected at Z: {robust_floor_z:.4f} m")
    
    # 5. Apply Fix
    shift = -robust_floor_z
    
    # Apply Manual Override if user requested
    if args.manual_shift != 0.0:
        print(f">> Adding Manual Offset: {args.manual_shift} m")
        shift += args.manual_shift
        
    print(f">> Total Shift Applied: {shift:.4f} m")
    trans[:, 2] += shift
    
    # 6. Save
    np.savez(args.out,
             poses=poses,
             trans=trans,       
             transl=trans,      
             global_orient=data['global_orient'],
             body_pose=data['body_pose'],
             joint_names=names,
             fps=data['fps']
    )
    print(f"✅ DONE. Saved to {args.out}")

if __name__ == "__main__":
    main()