import numpy as np
import argparse
import os
import sys
from scipy.spatial.transform import Rotation as R
from tqdm import tqdm

# ==========================================
# 1. SETTINGS & SMPL CONSTANTS
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

# SMPL T-Pose Vectors (Normalized Directions in Z-Up)
SMPL_REST_VECTORS = {
    "Spine":        np.array([0, 0, 1]),      # Up
    "LeftUpLeg":    np.array([0, 0, -1]),     # Down
    "RightUpLeg":   np.array([0, 0, -1]),     # Down
    "LeftLeg":      np.array([0, 0, -1]),     # Down
    "RightLeg":     np.array([0, 0, -1]),     # Down
    "LeftArm":      np.array([1, 0, 0]),      # Left
    "RightArm":     np.array([-1, 0, 0]),     # Right
    "LeftForeArm":  np.array([1, 0, 0]),      # Left
    "RightForeArm": np.array([-1, 0, 0]),     # Right
    "LeftFoot":     np.array([0, 1, -0.2]),   # Forward/Down
    "RightFoot":    np.array([0, 1, -0.2]),   # Forward/Down
}

# Coordinate Rotation: 90 degrees around X (Y-Up -> Z-Up)
RX_90 = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]])

# ==========================================
# 2. MATH HELPERS
# ==========================================
def normalize(v):
    norm = np.linalg.norm(v)
    return v if norm < 1e-6 else v / norm

def rotation_between_vectors(u, v):
    """Calculates rotation matrix that rotates vector u to vector v."""
    u = normalize(u)
    v = normalize(v)
    axis = np.cross(u, v)
    sin_angle = np.linalg.norm(axis)
    cos_angle = np.dot(u, v)
    
    if sin_angle < 1e-6:
        if cos_angle > 0: return np.eye(3) 
        else: return R.from_euler('x', 180, degrees=True).as_matrix()
            
    axis = axis / sin_angle
    angle = np.arctan2(sin_angle, cos_angle)
    return R.from_rotvec(axis * angle).as_matrix()

# ==========================================
# 3. ROBUST BVH LOADER & FK
# ==========================================
def load_bvh_and_compute_positions(path):
    print(f"Reading & Computing FK: {os.path.basename(path)}")
    with open(path, 'r') as f: content = f.read().split()
    iterator = iter(content)
    
    class Node:
        def __init__(self, name, parent=None):
            self.name = name
            self.parent = parent
            self.offset = np.zeros(3)
            self.channels = []
            self.children = []
    
    nodes_dict = {}
    root = None
    node_stack = []
    
    # 1. PARSE HIERARCHY
    token = next(iterator, None)
    while token:
        if token == "MOTION": break
        
        if token in ["ROOT", "JOINT"]:
            name = next(iterator)
            node = Node(name, node_stack[-1] if node_stack else None)
            nodes_dict[name] = node
            if node_stack: node_stack[-1].children.append(node)
            else: root = node
            node_stack.append(node)
            
        elif token == "End":
            next(iterator) # Site
            node = Node(f"EndSite_{node_stack[-1].name}", node_stack[-1])
            nodes_dict[node.name] = node
            node_stack[-1].children.append(node)
            node_stack.append(node)
            
        elif token == "OFFSET":
            node_stack[-1].offset = np.array([float(next(iterator)) for _ in range(3)])
            
        elif token == "CHANNELS":
            c = int(next(iterator))
            node_stack[-1].channels = [next(iterator) for _ in range(c)]
            
        elif token == "}":
            if node_stack: node_stack.pop()
            
        token = next(iterator, None)
        
    # 2. PARSE MOTION
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
    
    # 3. RUN FK (Compute Global Positions)
    T = num_frames
    global_positions = {name: np.zeros((T, 3)) for name in nodes_dict}
    
    print(">> Calculating Global Joint Positions...")
    for t in tqdm(range(T)):
        def recurse(node, ptr, p_pos, p_rot):
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
                    l_rot = R.from_euler(order, d[rot_cols], degrees=True).as_matrix()
                
                ptr += len(node.channels)
            
            g_rot = p_rot @ l_rot
            g_pos = p_pos + (p_rot @ l_pos)
            
            global_positions[node.name][t] = g_pos
            
            for child in node.children:
                ptr = recurse(child, ptr, g_pos, g_rot)
            return ptr

        recurse(root, 0, np.zeros(3), np.eye(3))
        
    return global_positions, ft, T

# ==========================================
# 4. VECTOR RETARGETING ENGINE
# ==========================================
def process_vector_copy(global_pos, T):
    poses = np.zeros((T, len(SMPL_H_NAMES), 3))
    trans = np.zeros((T, 3))
    
    # Scale detection
    hips_h = global_pos["Hips"][0, 1] 
    scale = 0.01 if abs(hips_h) > 50 else 1.0
    print(f">> Detected Scale: {scale}")

    # Coordinate transform (Y-Up -> Z-Up)
    def to_smpl_space(v):
        return np.dot(v, RX_90.T)

    print(">> Retargeting Vectors...")
    for t in tqdm(range(T)):
        
        # 1. ROOT POSITION
        b_hips = global_pos["Hips"][t] * scale
        trans[t] = to_smpl_space(b_hips)
        
        # 2. SOLVE BONES
        smpl_global_rots = {} 

        def solve_bone(smpl_name, bvh_start, bvh_end, parent_name=None):
            # Safe check for keys
            start_key = None
            end_key = None
            
            # Fuzzy match BVH names
            for k in global_pos.keys():
                if bvh_start.lower() in k.lower(): start_key = k
                if bvh_end.lower() in k.lower(): end_key = k
            
            if not start_key or not end_key:
                smpl_global_rots[smpl_name] = np.eye(3)
                if parent_name and parent_name in smpl_global_rots:
                    smpl_global_rots[smpl_name] = smpl_global_rots[parent_name]
                return

            # BVH Vector
            p_start = global_pos[start_key][t] * scale
            p_end = global_pos[end_key][t] * scale
            bvh_vec = to_smpl_space(p_end - p_start)
            
            # SMPL Rest Vector
            smpl_vec = SMPL_REST_VECTORS.get(smpl_name, np.array([0,1,0]))
            
            # Align
            R_align = rotation_between_vectors(smpl_vec, bvh_vec)
            smpl_global_rots[smpl_name] = R_align
            
            # Local
            if parent_name and parent_name in smpl_global_rots:
                R_parent = smpl_global_rots[parent_name]
                R_loc = R_parent.T @ R_align
            else:
                R_loc = R_align
            
            idx = SMPL_H_NAMES.index(smpl_name)
            poses[t, idx] = R.from_matrix(R_loc).as_rotvec()

        # Solve Chain
        solve_bone("Spine", "Hips", "Neck")
        
        solve_bone("LeftUpLeg", "LeftUpLeg", "LeftLeg", "Hips")
        solve_bone("LeftLeg", "LeftLeg", "LeftFoot", "LeftUpLeg")
        solve_bone("RightUpLeg", "RightUpLeg", "RightLeg", "Hips")
        solve_bone("RightLeg", "RightLeg", "RightFoot", "RightUpLeg")
        
        solve_bone("LeftArm", "LeftArm", "LeftForeArm", "Spine")
        solve_bone("LeftForeArm", "LeftForeArm", "LeftHand", "LeftArm")
        solve_bone("RightArm", "RightArm", "RightForeArm", "Spine")
        solve_bone("RightForeArm", "RightForeArm", "RightHand", "RightArm")
        
        # Root Rotation (Approximate)
        # Use Hips->Neck as Up, Left->Right as Right
        u = normalize(to_smpl_space(global_pos["Neck"][t] - global_pos["Hips"][t]))
        # Need robust Right Hip finding
        r_hip_key = [k for k in global_pos.keys() if "RightUpLeg" in k or "RightThigh" in k or "RightHip" in k][0]
        l_hip_key = [k for k in global_pos.keys() if "LeftUpLeg" in k or "LeftThigh" in k or "LeftHip" in k][0]
        
        r = normalize(to_smpl_space(global_pos[r_hip_key][t] - global_pos[l_hip_key][t]))
        f = np.cross(u, r)
        r = np.cross(f, u)
        
        # SMPL Root Basis: X=Right, Y=Up, Z=Forward? 
        # Standard SMPL: Root is Identity = Facing Z.
        # Matrix columns: [Right, Forward, Up] for Z-Up system?
        # Actually SMPL Z-up: X=Right, Y=Forward, Z=Up
        root_mat = np.stack([r, f, u], axis=1)
        poses[t, 0] = R.from_matrix(root_mat).as_rotvec()
        
    return poses, trans

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    
    # 1. Load & FK
    global_pos, ft, T = load_bvh_and_compute_positions(args.input)
    
    # 2. Vector Retarget
    poses, trans = process_vector_copy(global_pos, T)
    
    # 3. Ground
    min_z = np.min(trans[:, 2])
    if min_z < 0.05:
        trans[:, 2] -= min_z
        
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
    print(f"✅ Saved VECTOR MAP: {args.output}")

if __name__ == "__main__":
    main()