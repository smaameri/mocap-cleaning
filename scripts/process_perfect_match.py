import numpy as np
import argparse
import os
import sys
from scipy.spatial.transform import Rotation as R
from tqdm import tqdm

# ==========================================
# 1. SETTINGS & CONSTANTS
# ==========================================
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

# Coordinate Rotation: Y-Up (BVH) -> Z-Up (SMPL)
RX_90 = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]])

# SMPL T-Pose Vectors (Normalized directions for Z-Up System)
# Based on standard SMPL topology
SMPL_VECTORS = {
    # Torso
    "Spine":        np.array([0, 0, 1]),     
    "Spine1":       np.array([0, 0, 1]),
    "Spine2":       np.array([0, 0, 1]),
    "Neck":         np.array([0, 0, 1]),
    "Head":         np.array([0, 0, 1]),
    # Legs
    "LeftUpLeg":    np.array([0, 0, -1]),    
    "LeftLeg":      np.array([0, 0, -1]),
    "RightUpLeg":   np.array([0, 0, -1]),
    "RightLeg":     np.array([0, 0, -1]),
    # Arms (Crucial: These are X-aligned)
    "LeftArm":      np.array([1, 0, 0]),     
    "LeftForeArm":  np.array([1, 0, 0]),
    "RightArm":     np.array([-1, 0, 0]),
    "RightForeArm": np.array([-1, 0, 0]),
}

# ==========================================
# 2. MATH HELPERS
# ==========================================
def normalize(v):
    norm = np.linalg.norm(v)
    return v if norm < 1e-6 else v / norm

def rotation_between_vectors(u, v):
    """Calculates minimal rotation to align vector u to vector v."""
    u = normalize(u)
    v = normalize(v)
    axis = np.cross(u, v)
    sin_val = np.linalg.norm(axis)
    cos_val = np.dot(u, v)
    
    # Check for parallel/anti-parallel
    if sin_val < 1e-6:
        if cos_val > 0: return np.eye(3) # Same direction
        else: # Opposite direction: 180 flip around X
            return R.from_euler('x', 180, degrees=True).as_matrix()
            
    axis = axis / sin_val
    angle = np.arctan2(sin_val, cos_val)
    return R.from_rotvec(axis * angle).as_matrix()

# ==========================================
# 3. BVH PARSER & FK (Hardcoded for this file)
# ==========================================
def load_bvh_positions(path):
    print(f"Processing: {os.path.basename(path)}")
    with open(path, 'r') as f: content = f.read().split()
    iterator = iter(content)
    
    # 1. READ HIERARCHY
    # We use a simple stack to build the tree so we can compute Global Positions
    class Node:
        def __init__(self, name, parent=None):
            self.name = name
            self.parent = parent
            self.offset = np.zeros(3)
            self.channels = []
            self.children = []
            
    nodes = {}
    root = None
    stack = []
    
    token = next(iterator, None)
    while token:
        if token == "MOTION": break
        
        if token in ["ROOT", "JOINT"]:
            name = next(iterator)
            node = Node(name, stack[-1] if stack else None)
            nodes[name] = node
            if stack: stack[-1].children.append(node)
            else: root = node
            stack.append(node)
            
        elif token == "End":
            next(iterator) 
            # End Sites don't have names in BVH, usually just "End Site"
            # We map them to parent name + _End
            name = f"{stack[-1].name}_End"
            node = Node(name, stack[-1])
            nodes[name] = node
            stack[-1].children.append(node)
            stack.append(node)
            
        elif token == "OFFSET":
            stack[-1].offset = np.array([float(next(iterator)) for _ in range(3)])
            
        elif token == "CHANNELS":
            c = int(next(iterator))
            stack[-1].channels = [next(iterator) for _ in range(c)]
            
        elif token == "}":
            stack.pop()
            
        token = next(iterator, None)
        
    # 2. READ MOTION
    while token != "Frames:": token = next(iterator)
    num_frames = int(next(iterator))
    while token != "Frame": token = next(iterator)
    next(iterator) # Time:
    ft = float(next(iterator))
    
    vals = []
    try:
        while True: vals.append(float(next(iterator)))
    except: pass
    frames = np.array(vals).reshape(num_frames, -1)
    
    # 3. COMPUTE GLOBAL POSITIONS (FK)
    T = num_frames
    global_pos = {name: np.zeros((T, 3)) for name in nodes}
    
    print(">> Calculating Global Positions...")
    for t in tqdm(range(T)):
        def recurse_fk(node, ptr, p_pos, p_rot):
            l_pos = node.offset.copy()
            l_rot = np.eye(3)
            
            if node.channels:
                d = frames[t, ptr : ptr+len(node.channels)]
                # Apply Pos
                if "Xposition" in node.channels: l_pos[0] += d[node.channels.index("Xposition")]
                if "Yposition" in node.channels: l_pos[1] += d[node.channels.index("Yposition")]
                if "Zposition" in node.channels: l_pos[2] += d[node.channels.index("Zposition")]
                
                # Apply Rot (ZXY)
                rot_cols = [i for i, c in enumerate(node.channels) if 'rotation' in c]
                if len(rot_cols) == 3:
                    order = "".join([node.channels[i][0].lower() for i in rot_cols])
                    # Note: We don't need to unwrap here because we are computing Position.
                    # Position is continuous even if Euler angles flip 360.
                    l_rot = R.from_euler(order, d[rot_cols], degrees=True).as_matrix()
                
                ptr += len(node.channels)
            
            g_rot = p_rot @ l_rot
            g_pos = p_pos + (p_rot @ l_pos)
            
            global_pos[node.name][t] = g_pos
            
            for child in node.children:
                ptr = recurse_fk(child, ptr, g_pos, g_rot)
            return ptr

        recurse_fk(root, 0, np.zeros(3), np.eye(3))
        
    return global_pos, ft, T

# ==========================================
# 4. VECTOR RETARGETING ENGINE
# ==========================================
def process_perfect_match(global_pos, T):
    poses = np.zeros((T, len(SMPL_H_NAMES), 3))
    trans = np.zeros((T, 3))
    
    # Scale Fix: Hips usually ~95cm in BVH. SMPL ~0.95m.
    # Check Y height of Hips at frame 0
    hips_h = global_pos["Hips"][0, 1] 
    scale = 0.01 if abs(hips_h) > 50 else 1.0
    print(f">> Applying Scale: {scale} (CM to Meters)")

    # Coordinate Transform: Y-Up (BVH) -> Z-Up (SMPL)
    # We apply this to the VECTORS, not the Rotations.
    def transform(v):
        return np.dot(v, RX_90.T)

    print(">> Retargeting Motion Vectors...")
    for t in tqdm(range(T)):
        
        # 1. ROOT POSITION
        raw_hips = global_pos["Hips"][t] * scale
        trans[t] = transform(raw_hips)
        
        # 2. ALIGN BONES (Vector Mapping)
        smpl_global_rots = {} # Store global orientations to compute local later
        
        def align(smpl_name, bvh_start, bvh_end, smpl_parent=None):
            if bvh_start not in global_pos or bvh_end not in global_pos:
                # Fallback: Copy parent or identity
                R_glob = smpl_global_rots.get(smpl_parent, np.eye(3)) if smpl_parent else np.eye(3)
                smpl_global_rots[smpl_name] = R_glob
                return

            # A. Get BVH Vector (Current Frame)
            p_start = global_pos[bvh_start][t] * scale
            p_end = global_pos[bvh_end][t] * scale
            bvh_vec_global = transform(p_end - p_start)
            
            # B. Get SMPL Rest Vector (Target)
            smpl_vec_rest = SMPL_VECTORS.get(smpl_name, np.array([0,0,1])) # Default Up
            
            # C. Calculate Rotation
            # Find R such that R * Rest = Current
            R_glob = rotation_between_vectors(smpl_vec_rest, bvh_vec_global)
            
            # D. Twist Correction (Heuristic)
            # Simple vector alignment ignores twist (roll).
            # We assume "No Twist" relative to parent if possible, or inherit twist.
            # Ideally: Match the 'Up' vector of the limb.
            # For simplicity: Vector alignment is usually stable for martial arts.
            
            smpl_global_rots[smpl_name] = R_glob
            
            # E. Compute Local Rotation
            # Local = Parent_Inv * Global
            if smpl_parent and smpl_parent in smpl_global_rots:
                R_parent = smpl_global_rots[smpl_parent]
                R_local = R_parent.T @ R_glob
            else:
                R_local = R_glob
            
            # Store
            idx = SMPL_H_NAMES.index(smpl_name)
            poses[t, idx] = R.from_matrix(R_local).as_rotvec()

        # --- MAPPING CHAIN ---
        # We Map: SMPL_Bone -> (BVH_Start, BVH_End)
        
        # Spine Chain
        align("Spine", "Hips", "Spine", None) # Root orientation base
        align("Spine1", "Spine", "Spine1", "Spine")
        align("Spine2", "Spine1", "Neck", "Spine1")
        align("Neck", "Neck", "Head", "Spine2")
        align("Head", "Head", "Head_End", "Neck")
        
        # Legs
        align("LeftUpLeg", "LeftUpLeg", "LeftLeg", "Hips")
        align("LeftLeg", "LeftLeg", "LeftFoot", "LeftUpLeg")
        align("RightUpLeg", "RightUpLeg", "RightLeg", "Hips")
        align("RightLeg", "RightLeg", "RightFoot", "RightUpLeg")
        
        # Arms (Crucial: Uses LeftArm->LeftForeArm vector)
        align("LeftArm", "LeftArm", "LeftForeArm", "Spine2") 
        align("LeftForeArm", "LeftForeArm", "LeftHand", "LeftArm")
        align("RightArm", "RightArm", "RightForeArm", "Spine2")
        align("RightForeArm", "RightForeArm", "RightHand", "RightArm")
        
        # Hands (Just align to fingers/end)
        align("LeftHand", "LeftHand", "LeftHandMiddle1", "LeftForeArm")
        align("RightHand", "RightHand", "RightHandMiddle1", "RightForeArm")
        
        # Root Orientation Fix
        # We want the SMPL root to face the direction of the hips.
        # Vector: RightHip - LeftHip
        r_hip = transform(global_pos["RightUpLeg"][t] - global_pos["LeftUpLeg"][t])
        up = transform(global_pos["Spine"][t] - global_pos["Hips"][t])
        
        # Gram-Schmidt Orthogonalization
        fwd = np.cross(up, r_hip) # Forward
        r_hip = np.cross(fwd, up) # Right
        
        # Normalize
        r_hip = normalize(r_hip)
        fwd = normalize(fwd)
        up = normalize(up)
        
        # Matrix [Right, Forward, Up] corresponds to Z-Up Basis
        # In SMPL: X=Right, Y=Forward, Z=Up ?
        # Actually SMPL Body is Z-Up. Rest pose faces Y+.
        # So X=Right, Y=Forward, Z=Up.
        root_mat = np.stack([r_hip, fwd, up], axis=1)
        poses[t, 0] = R.from_matrix(root_mat).as_rotvec()

    return poses, trans

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    
    # 1. Load BVH & Run FK
    global_pos, ft, T = load_bvh_positions(args.input)
    
    # 2. Retarget Vectors
    poses, trans = process_perfect_match(global_pos, T)
    
    # 3. Grounding (Snap Lowest Z to 0)
    min_z = np.min(trans[:, 2])
    print(f">> Grounding: Lowest Point {min_z:.4f}m -> 0.0m")
    trans[:, 2] -= min_z
    
    # 4. Save
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
    print(f"✅ DONE: {args.output}")

if __name__ == "__main__":
    main()