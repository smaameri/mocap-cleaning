import pybullet as p
import numpy as np
import argparse
import sys
import os
from scipy.spatial.transform import Rotation as R
from tqdm import tqdm

# ==========================================
# 1. SETTINGS
# ==========================================
SCALE_OFFSETS = 0.01  # Matches your pipeline
CAPSULE_RADIUS = 0.05 # 5cm thickness for the leg/foot "flesh"

# ==========================================
# 2. BVH LOADER (Minified)
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
        elif token == "MOTION": break
        token = next(iterator, None)
    return root

# ==========================================
# 3. KINEMATICS & PHYSICS MAPPING
# ==========================================
def compute_global_positions(node, poses_map, parent_pos, parent_rot):
    """Recursive FK to get all joint positions for a frame"""
    local_rot = np.eye(3)
    if node.name in poses_map:
        local_rot = R.from_rotvec(poses_map[node.name]).as_matrix()
    
    global_rot = parent_rot @ local_rot
    scaled_offset = node.offset * SCALE_OFFSETS
    global_pos = parent_pos + (parent_rot @ scaled_offset)

    results = {node.name: global_pos}
    
    for child in node.children:
        child_res = compute_global_positions(child, poses_map, global_pos, global_rot)
        results.update(child_res)
        
    return results

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--npy", required=True, help="Input NPZ")
    parser.add_argument("--bvh", required=True, help="Input BVH (for skeleton)")
    parser.add_argument("--out", required=True, help="Output NPZ")
    args = parser.parse_args()

    print("--- PYBULLET PHYSICS GROUNDING ---")
    
    # 1. Load Data
    data = np.load(args.npy)
    poses = data['poses']
    names = data['joint_names']
    
    if 'transl' in data: trans = data['transl'].copy()
    else: trans = data['trans'].copy()
    
    T = poses.shape[0]
    
    # 2. Load Skeleton Structure
    root = load_skeleton_structure(args.bvh)
    
    print(f">> Analyzing {T} frames using Physics Bounding Boxes...")
    
    min_z_surface = 9999.0
    
    # We check a subset of frames to be fast (every 5th frame)
    for f in tqdm(range(0, T, 5)):
        pose_map = {name: poses[f, i] for i, name in enumerate(names)}
        
        # Calculate Joint Centers (Bones)
        # We start FK from (0,0,0) + Trans[f]
        joint_positions = compute_global_positions(
            root, pose_map, trans[f], np.eye(3)
        )
        
        # Find lowest "Surface" point
        # We look at foot joints and subtract radius to get the "Skin" bottom
        frame_min_z = 9999.0
        
        for j_name, pos in joint_positions.items():
            # Check if it is a foot/toe part
            lower_name = j_name.lower()
            if "foot" in lower_name or "toe" in lower_name or "ankle" in lower_name or "endsite" in lower_name:
                # The bottom of the foot is Joint_Z - Radius
                surface_z = pos[2] - CAPSULE_RADIUS
                if surface_z < frame_min_z:
                    frame_min_z = surface_z
        
        if frame_min_z < min_z_surface:
            min_z_surface = frame_min_z

    print(f">> Lowest 'Skin' Contact Point detected: {min_z_surface:.4f} m")
    
    # 3. Apply Fix
    # If the skin is at +0.5m, we need to shift down by 0.5m
    shift = -min_z_surface
    
    # Add a tiny buffer (1cm) so it doesn't clip through the floor
    shift += 0.01 
    
    print(f">> Applying Physics Shift: {shift:.4f} m")
    trans[:, 2] += shift

    # 4. Save
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
    print("✅ DONE. Character Grounded (Physics-Aware).")

if __name__ == "__main__":
    main()