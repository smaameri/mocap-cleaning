import numpy as np
import argparse
import os
import sys
from tqdm import tqdm
from scipy.spatial.transform import Rotation as R

# ==========================================
# 1. SETTINGS
# ==========================================
# Standard SMPL-H Joint Names
SMPL_H_NAMES = [
    "Hips", "LeftUpLeg", "RightUpLeg", "Spine", "LeftLeg", "RightLeg",
    "Spine1", "LeftFoot", "RightFoot", "Spine2", "LeftToeBase", "RightToeBase",
    "Neck", "LeftShoulder", "RightShoulder", "Head", "LeftArm", "RightArm",
    "LeftForeArm", "RightForeArm", "LeftHand", "RightHand",
    "LeftHandIndex1", "LeftHandIndex2", "LeftHandIndex3", "LeftHandMiddle1", "LeftHandMiddle2", "LeftHandMiddle3",
    "LeftHandPinky1", "LeftHandPinky2", "LeftHandPinky3", "LeftHandRing1", "LeftHandRing2", "LeftHandRing3",
    "LeftHandThumb1", "LeftHandThumb2", "LeftHandThumb3",
    "RightHandIndex1", "RightHandIndex2", "RightHandIndex3", "RightHandMiddle1", "RightHandMiddle2", "RightHandMiddle3",
    "RightHandPinky1", "RightHandPinky2", "RightHandPinky3", "RightHandRing1", "RightHandRing2", "RightHandRing3",
    "RightHandThumb1", "RightHandThumb2", "RightHandThumb3"
]

# ==========================================
# 2. CUSTOM PARSER (HANDLES ZXY & CM UNITS)
# ==========================================
class BVHNode:
    def __init__(self, name, offset, parent=None):
        self.name = name
        self.offset = np.array(offset, dtype=np.float32)
        self.parent = parent
        self.children = []
        self.channels = []

def load_bvh_zxy(path):
    """
    Parses BVH forcing ZXY rotation order interpretation.
    Fixes 'Wrong Motion' issues caused by XYZ assumptions.
    """
    with open(path, 'r') as f:
        content = f.read().split()
    
    iterator = iter(content)
    nodes = {}
    root = None
    node_stack = []
    frames = []
    frame_time = 0.00833
    
    # Hierarchy
    token = next(iterator, None)
    while token:
        if token == "HIERARCHY": pass
        elif token in ["ROOT", "JOINT"]:
            name = next(iterator)
            new_node = BVHNode(name, [0,0,0], node_stack[-1] if node_stack else None)
            nodes[name] = new_node
            if node_stack: node_stack[-1].children.append(new_node)
            else: root = new_node
            node_stack.append(new_node)
        elif token == "End":
            next(iterator) # Site
            new_node = BVHNode("EndSite", [0,0,0], node_stack[-1])
            node_stack[-1].children.append(new_node)
            node_stack.append(new_node)
        elif token == "OFFSET":
            node_stack[-1].offset = np.array([float(next(iterator)) for _ in range(3)])
        elif token == "CHANNELS":
            count = int(next(iterator))
            chs = [next(iterator) for _ in range(count)]
            node_stack[-1].channels = chs
        elif token == "}" and node_stack:
            node_stack.pop()
        elif token == "MOTION":
            break
        token = next(iterator, None)
        
    # Motion Data
    while token:
        if token == "Frames:": 
            num_frames = int(next(iterator))
        elif token == "Frame": 
            next(iterator) # Time:
            frame_time = float(next(iterator))
            data = []
            try:
                while True:
                    val = next(iterator)
                    data.append(float(val))
            except StopIteration:
                pass
            frames = np.array(data).reshape(num_frames, -1)
            break
        token = next(iterator, None)

    return root, nodes, frames, frame_time

# ==========================================
# 3. FK ENGINE (FOR GROUNDING)
# ==========================================
def compute_fk(node, poses_map, parent_pos, parent_rot, root_trans=None):
    """
    Computes global joint positions to find the feet.
    Supports injecting a global root translation (Z-up).
    """
    # Get local rotation
    if node.name in poses_map:
        local_rot = R.from_rotvec(poses_map[node.name]).as_matrix()
    else:
        local_rot = np.eye(3)
        
    global_rot = parent_rot @ local_rot
    
    # Scale offset (0.01 for CM -> M)
    scaled_offset = node.offset * 0.01
    
    if root_trans is not None:
        global_pos = root_trans
    else:
        global_pos = parent_pos + (parent_rot @ scaled_offset)
        
    positions = [global_pos]
    
    for child in node.children:
        child_pos = compute_fk(child, poses_map, global_pos, global_rot)
        positions.extend(child_pos)
        
    return positions

# ==========================================
# 4. MAIN PROCESS
# ==========================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    print(f"--- PROCESSING: {os.path.basename(args.input)} ---")
    
    # 1. LOAD (ZXY Aware)
    print(">> Parsing BVH (ZXY Mode)...")
    root, nodes, frames, ft = load_bvh_zxy(args.input)
    T = frames.shape[0]
    
    # 2. CONVERT & ROTATE
    print(">> Converting Coordinates (Y-Up -> Z-Up)...")
    poses = np.zeros((T, len(SMPL_H_NAMES), 3))
    trans = np.zeros((T, 3))
    
    # Rotation Matrix: 90 degrees on X-axis (Y -> Z, Z -> -Y)
    Rx90 = R.from_euler('x', 90, degrees=True).as_matrix()
    
    # Traverse channels linearly
    channel_ptr = 0
    node_channel_map = []
    
    def map_channels(node):
        nonlocal channel_ptr
        if hasattr(node, 'channels'):
            indices = list(range(channel_ptr, channel_ptr + len(node.channels)))
            node_channel_map.append((node, indices))
            channel_ptr += len(node.channels)
        for child in node.children:
            map_channels(child)
            
    map_channels(root)
    
    for t in tqdm(range(T)):
        for node, indices in node_channel_map:
            if node.name not in SMPL_H_NAMES: continue
            
            raw = frames[t, indices]
            
            # Extract Position (Hips only)
            pos = np.zeros(3)
            if len(indices) == 6: # Root usually has 6
                pos = raw[:3]
                rot_deg = raw[3:] # Z, X, Y
            else:
                rot_deg = raw # Z, X, Y
            
            # --- FIX 1: Rotation Order ZXY ---
            # BVH channels are: Zrotation, Xrotation, Yrotation
            # We must create matrix from this exact order
            local_mat = R.from_euler('zxy', rot_deg, degrees=True).as_matrix()
            
            idx = SMPL_H_NAMES.index(node.name)
            
            if node.name == "Hips":
                # --- FIX 2: Root Coordinate Rotation ---
                # Position: Scale (CM->M) AND Rotate (Y-Up -> Z-Up)
                pos_m = pos * 0.01
                trans[t] = Rx90 @ pos_m
                
                # Rotation: Apply Rx90 to the GLOBAL root rotation
                # Global_New = Rx90 * Global_Old
                poses[t, 0] = R.from_matrix(Rx90 @ local_mat).as_rotvec()
            else:
                # Body Joints:
                # Local rotations are relative to parent.
                # Since we are keeping the original BVH offset directions (just viewing them rotated),
                # we technically don't need to change the local rotation values relative to the parent bone!
                # The parent bone rotated, the offset rotated, so the local frame rotated.
                poses[t, idx] = R.from_matrix(local_mat).as_rotvec()

    # 3. GROUNDING (Physics Based)
    print(">> Grounding (Finding Feet)...")
    
    min_z_list = []
    # Check every 5th frame
    for t in range(0, T, 5):
        pose_map = {name: poses[t, i] for i, name in enumerate(SMPL_H_NAMES)}
        
        # Root Rot for FK: The poses[t,0] is already Global Z-up
        root_rot = R.from_rotvec(poses[t, 0]).as_matrix()
        
        # We need to run FK using the BVH offsets.
        # BUT the BVH offsets are Y-up. The Motion is Z-up.
        # WE MUST ROTATE THE ROOT ROTATION BACK locally? 
        # No, easier: We rotate the FK root matrix by Rx90 so it aligns with Y-up offsets?
        
        # Actually:
        # Poses[0] (Z-up) = Rx90 * Original_Root_Rot
        # Trans (Z-up) = Rx90 * Original_Trans
        
        # If we use Original Offsets (Y-up) in FK:
        # Joint_Pos = Parent_Pos + (Parent_Rot * Offset)
        # For Root: Parent_Rot is Poses[0]. Offset is (0,0,0).
        # For Spine: Parent_Rot is Poses[0]. Offset is (0, 10, 0).
        # Poses[0] includes the Rx90. 
        # Rx90 * (0, 10, 0) = (0, 0, 10).
        # THIS IS CORRECT! The Spine will point UP in Z.
        
        # So we can just use the standard FK logic.
        
        # Note: We must pass 'root_rot' matrix for the root node rotation
        # BUT 'compute_fk' applies local_rot. For Root, local_rot IS the global rot.
        # So we pass Identity as parent_rot.
        
        all_pos = compute_fk(root, pose_map, np.zeros(3), np.eye(3), root_trans=trans[t])
        
        # Get Z values (Height)
        zs = [p[2] for p in all_pos]
        min_z_list.append(min(zs))
        
    floor_z = np.percentile(min_z_list, 1) # 1st percentile to ignore glitches
    shift = -floor_z
    
    print(f">> Floor detected at Z={floor_z:.4f}m. Shifting by {shift:.4f}m.")
    trans[:, 2] += shift

    # 4. SAVE
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    np.savez(args.output,
             poses=poses,
             trans=trans,
             transl=trans,
             global_orient=poses[:, 0:1],
             body_pose=poses[:, 1:22].reshape(T, -1),
             joint_names=SMPL_H_NAMES,
             fps=1.0/ft
    )
    print(f"✅ CLEANED & SAVED: {args.output}")

if __name__ == "__main__":
    main()