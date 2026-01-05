import numpy as np
import argparse
import os
import sys
from scipy.spatial.transform import Rotation as R

# ==========================================
# 1. SETTINGS
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

# Mapping: SMPL Bone -> BVH Bone Name (Fuzzy Match)
# We will look for these strings in the BVH file
BONE_MAP = {
    "Hips": ["Hips", "Root"],
    "LeftUpLeg": ["LeftUpLeg", "LHip", "LeftHip"],
    "RightUpLeg": ["RightUpLeg", "RHip", "RightHip"],
    "Spine": ["Spine"],
    "Spine1": ["Spine1"],
    "Spine2": ["Spine2", "Chest"],
    "Neck": ["Neck"],
    "Head": ["Head"],
    "LeftShoulder": ["LeftShoulder", "LCollar"],
    "RightShoulder": ["RightShoulder", "RCollar"],
    "LeftArm": ["LeftArm", "LShoulder"],  # SMPL Arm = BVH Arm
    "RightArm": ["RightArm", "RShoulder"],
    "LeftForeArm": ["LeftForeArm", "LElbow"],
    "RightForeArm": ["RightForeArm", "RElbow"],
    "LeftHand": ["LeftHand", "LWrist"],
    "RightHand": ["RightHand", "RWrist"],
    "LeftLeg": ["LeftLeg", "LKnee"],
    "RightLeg": ["RightLeg", "RKnee"],
    "LeftFoot": ["LeftFoot", "LAnkle"],
    "RightFoot": ["RightFoot", "RAnkle"],
}

# ==========================================
# 2. HELPER: QUATERNION CONTINUITY
# ==========================================
def make_continuous(quats):
    """Fixes spinning glitches (Frame 800) by ensuring shortest path."""
    for i in range(1, len(quats)):
        if np.dot(quats[i], quats[i-1]) < 0:
            quats[i] = -quats[i]
    return quats

# ==========================================
# 3. BVH LOADER & FK ENGINE
# ==========================================
def load_bvh_global_rotations(path):
    print(f"Reading: {os.path.basename(path)}")
    with open(path, 'r') as f: content = f.read().split()
    iterator = iter(content)
    
    # 1. Parse Hierarchy
    parent_map = {} # name -> parent_name
    nodes = {}      # name -> channels
    node_stack = []
    
    token = next(iterator, None)
    while token:
        if token == "MOTION": break
        if token in ["ROOT", "JOINT"]:
            name = next(iterator)
            parent = node_stack[-1] if node_stack else None
            parent_map[name] = parent
            
            while True:
                t = next(iterator)
                if t == "CHANNELS":
                    c = int(next(iterator))
                    channels = [next(iterator) for _ in range(c)]
                    nodes[name] = channels
                    break
            node_stack.append(name)
        elif token == "End":
            next(iterator) # Site
            node_stack.append("EndSite")
        elif token == "}":
            node_stack.pop()
        token = next(iterator, None)
        
    # 2. Parse Motion
    while token != "Frames:": token = next(iterator)
    num_frames = int(next(iterator))
    while token != "Frame": token = next(iterator)
    next(iterator)
    ft = float(next(iterator))
    
    vals = []
    try:
        while True: vals.append(float(next(iterator)))
    except: pass
    frames = np.array(vals).reshape(num_frames, -1)
    
    # 3. Compute Global Rotations (FK)
    T = frames.shape[0]
    global_rots = {} # name -> (T, 3, 3) matrix
    
    # Iterate roughly in hierarchy order (dict keys are insertion ordered in Py3.7+)
    ptr = 0
    for name, channels in nodes.items():
        n_ch = len(channels)
        data = frames[:, ptr : ptr + n_ch]
        ptr += n_ch
        
        # Local Rotation
        rot_cols = [i for i, c in enumerate(channels) if 'rotation' in c]
        if len(rot_cols) == 3:
            order = "".join([channels[i][0].lower() for i in rot_cols])
            # Unwrap is vital for spins
            euler = np.unwrap(np.deg2rad(data[:, rot_cols]), axis=0)
            local_mat = R.from_euler(order, euler, degrees=False).as_matrix()
        else:
            local_mat = R.identity(T).as_matrix() # Or stacked identity
            # Fix identity shape
            local_mat = np.tile(np.eye(3), (T, 1, 1))

        # Global Rotation = Parent_Global * Local
        parent = parent_map.get(name)
        if parent and parent in global_rots:
            parent_mat = global_rots[parent]
            global_mat = np.matmul(parent_mat, local_mat)
        else:
            global_mat = local_mat
            
        global_rots[name] = global_mat
        
    # Also grab Root Position
    root_name = list(nodes.keys())[0]
    root_ch = nodes[root_name]
    pos_cols = [i for i, c in enumerate(root_ch) if 'position' in c]
    root_pos = np.zeros((T, 3))
    if len(pos_cols) == 3:
        root_pos = frames[:, :3] # Assuming root is first
        
    return global_rots, root_pos, ft, T

# ==========================================
# 4. DECOUPLED RETARGETING
# ==========================================
def process_decoupled(global_rots_bvh, root_pos, T):
    poses = np.zeros((T, len(SMPL_H_NAMES), 3))
    trans = np.zeros((T, 3))
    
    # Scale Check
    scale = 0.01 if np.max(np.abs(root_pos)) > 50 else 1.0
    print(f">> Scale: {scale}")
    
    # 1. Root Position (Y-Up -> Z-Up)
    # P_new = Rx90 * P_old
    trans = np.dot(root_pos * scale, RX_90.T)
    
    # 2. Build SMPL Globals
    smpl_global_mats = {} # name -> (T, 3, 3)
    
    # Helper: Find BVH Bone
    def get_bvh_mat(smpl_name):
        candidates = BONE_MAP.get(smpl_name, [])
        for c in candidates:
            # Fuzzy match
            for k in global_rots_bvh.keys():
                if c.lower() == k.lower(): return global_rots_bvh[k]
                if c.lower() in k.lower(): return global_rots_bvh[k]
        return None

    # Helper: To Z-Up (Rx90 * R_bvh)
    def to_z_up(mat):
        # We rotate the WORLD FRAME of the rotation matrix
        return np.matmul(RX_90, mat)

    # PROCESS HIERARCHY
    for i, name in enumerate(SMPL_H_NAMES):
        # A. Determine Target Global Orientation
        bvh_mat = get_bvh_mat(name)
        
        target_global = None
        if bvh_mat is not None:
            # Apply World Rotation (Y->Z)
            target_global = to_z_up(bvh_mat)
        else:
            # Fallback: Use Identity or Parent
            target_global = np.tile(np.eye(3), (T, 1, 1))

        # SPECIAL HANDLING: SHOULDERS (COLLARS)
        if name in ["LeftShoulder", "RightShoulder"]:
             # Dampen Collars to 30% to fix "Stiff" look but avoid "Hands behind back" twist
             # Strategy: Use parent (Spine) rotation + 30% of Collar Local
             # Actually, simpler: Just let them be target_global?
             # NO. BVH Collar is messed up (Y-axis).
             # Force SMPL Collar to be Neutral relative to Spine?
             # Let's try: Copy Global but DAMPEN the difference from Spine.
             pass # For now, let's trust the Global Copy for everything else.
             
        smpl_global_mats[name] = target_global
        
        # B. Compute Local Rotation
        # R_local = Inv(R_parent_global) * R_target_global
        
        # Find Parent Name
        parent_name = None
        # Hardcoded hierarchy check
        if i > 0: # Hips is root
            if name == "LeftUpLeg": parent_name = "Hips"
            elif name == "RightUpLeg": parent_name = "Hips"
            elif name == "Spine": parent_name = "Hips"
            elif name == "LeftLeg": parent_name = "LeftUpLeg"
            elif name == "RightLeg": parent_name = "RightUpLeg"
            elif name == "LeftFoot": parent_name = "LeftLeg"
            elif name == "RightFoot": parent_name = "RightLeg"
            elif name == "Spine1": parent_name = "Spine"
            elif name == "Spine2": parent_name = "Spine1"
            elif name == "Neck": parent_name = "Spine2"
            elif name == "Head": parent_name = "Neck"
            elif name == "LeftShoulder": parent_name = "Spine2"
            elif name == "RightShoulder": parent_name = "Spine2"
            elif name == "LeftArm": parent_name = "LeftShoulder"
            elif name == "RightArm": parent_name = "RightShoulder"
            elif name == "LeftForeArm": parent_name = "LeftArm"
            elif name == "RightForeArm": parent_name = "RightArm"
            elif name == "LeftHand": parent_name = "LeftForeArm"
            elif name == "RightHand": parent_name = "RightForeArm"
        
        if parent_name and parent_name in smpl_global_mats:
            parent_mat = smpl_global_mats[parent_name]
            # Inv(P) * G
            parent_inv = np.transpose(parent_mat, (0, 2, 1))
            local_mat = np.matmul(parent_inv, target_global)
        else:
            local_mat = target_global
            
        # C. Convert to Vector & Save
        # Fix continuity to prevent flipping
        quats = R.from_matrix(local_mat).as_quat()
        quats = make_continuous(quats)
        poses[:, i] = R.from_quat(quats).as_rotvec()
        
    return poses, trans

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    
    # 1. Load BVH & Compute Globals
    global_rots, root_pos, ft, T = load_bvh_global_rotations(args.input)
    
    # 2. Retarget (Decoupled Global Mapping)
    poses, trans = process_decoupled(global_rots, root_pos, T)
    
    # 3. Ground
    min_z = np.min(trans[:, 2])
    if min_z < 0.05:
        print(f"Grounding: {min_z:.4f}m")
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
    print(f"✅ Saved DECOUPLED FIX: {args.output}")

if __name__ == "__main__":
    main()