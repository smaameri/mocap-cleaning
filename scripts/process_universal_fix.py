import numpy as np
import argparse
import os
from scipy.spatial.transform import Rotation as R
from tqdm import tqdm

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
RX_90 = R.from_euler('x', 90, degrees=True).as_matrix()

# ==========================================
# 2. HELPER FUNCTIONS
# ==========================================
def fix_quaternion_continuity(quats):
    """
    Prevents the 'flipping' artifact during 360-degree spins.
    If the dot product between consecutive frames is negative, flip the sign.
    """
    for i in range(1, len(quats)):
        if np.dot(quats[i], quats[i-1]) < 0:
            quats[i] = -quats[i]
    return quats

def load_bvh_raw(path):
    print(f"Reading: {os.path.basename(path)}")
    with open(path, 'r') as f: content = f.read().split()
    iterator = iter(content)
    
    nodes = {}
    hierarchy = {}
    node_stack = []
    
    token = next(iterator, None)
    while token:
        if token == "MOTION": break
        if token in ["ROOT", "JOINT"]:
            name = next(iterator)
            parent = node_stack[-1] if node_stack else None
            hierarchy[name] = parent
            
            channels = []
            while True:
                t = next(iterator)
                if t == "CHANNELS":
                    count = int(next(iterator))
                    channels = [next(iterator) for _ in range(count)]
                    break
            
            nodes[name] = channels
            node_stack.append(name)
        
        elif token == "End":
            next(iterator) # Site
            node_stack.append("EndSite")
        elif token == "}":
            node_stack.pop()
        
        token = next(iterator, None)
    
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
    
    return frames, nodes, hierarchy, ft

# ==========================================
# 3. UNIVERSAL CONVERTER
# ==========================================
def process_universal(frames, nodes, hierarchy):
    T = frames.shape[0]
    
    # --- STEP 1: COMPUTE GLOBAL ROTATIONS (BVH SPACE) ---
    print(">> Step 1: Calculating Global BVH Motion...")
    global_quats_bvh = {} 
    
    # We must traverse linearly (parents first)
    # Since dicts preserve insertion order in Py3.7+, and BVH is parsed top-down,
    # iterating 'nodes' keys is safe for hierarchy order.
    
    curr_ptr = 0
    for name, channels in nodes.items():
        n_ch = len(channels)
        data = frames[:, curr_ptr : curr_ptr + n_ch]
        curr_ptr += n_ch
        
        # 1. Get Local Rotation (ZXY)
        rot_cols = [i for i, c in enumerate(channels) if 'rotation' in c]
        if len(rot_cols) == 3:
            order = "".join([channels[i][0].lower() for i in rot_cols]) # "zxy"
            # UNWRAP EULERS (Fix Frame 800 Flip)
            euler = np.unwrap(np.deg2rad(data[:, rot_cols]), axis=0)
            local_q = R.from_euler(order, euler, degrees=False)
        else:
            local_q = R.identity(T)
            
        # 2. Apply to Parent (FK)
        parent = hierarchy.get(name)
        if parent and parent in global_quats_bvh:
            parent_q = global_quats_bvh[parent] # (T, 4) Scipy Rotation
            # Global = Parent * Local
            global_q = parent_q * local_q
        else:
            global_q = local_q
            
        global_quats_bvh[name] = global_q
        
    # --- STEP 2: CONVERT TO SMPL (RETARGETING) ---
    print(">> Step 2: Retargeting to SMPL...")
    final_poses = np.zeros((T, len(SMPL_H_NAMES), 3))
    final_trans = np.zeros((T, 3))
    
    # Store converted globals to recalculate locals
    smpl_globals = {}
    
    # Process Root Position
    hips_chs = nodes.get("Hips", [])
    pos_cols = [i for i, c in enumerate(hips_chs) if 'position' in c]
    if len(pos_cols) == 3:
        # Scale & Rotate Position
        raw_pos = frames[:, :3] # Assuming hips at start
        scale = 0.01 if np.max(np.abs(raw_pos[0])) > 50 else 1.0
        # Rotate Pos: (x, y, z) -> (x, -z, y)
        final_trans = np.dot(raw_pos * scale, RX_90.T)

    for i, name in enumerate(SMPL_H_NAMES):
        # Find corresponding BVH node (Fuzzy Match)
        bvh_name = None
        if name in global_quats_bvh: bvh_name = name
        else:
            # Try mapping
            if name == "LeftUpLeg": bvh_name = "LeftUpLeg" 
            if name == "RightUpLeg": bvh_name = "RightUpLeg"
            # Add more if names differ significantly
        
        if not bvh_name: 
            # If no matching bone, use Identity or Parent
            smpl_globals[name] = R.identity(T)
            continue
            
        # Get BVH Global Rotation
        bvh_q = global_quats_bvh[bvh_name]
        
        # --- THE FIX: GLOBAL ROTATION MAPPING ---
        # Rotate the entire Orientation Frame from Y-Up to Z-Up
        # New_Global = RX_90 * Old_Global
        smpl_global_q = R.from_matrix(RX_90) * bvh_q
        
        smpl_globals[name] = smpl_global_q
        
        # --- COMPUTE LOCAL SMPL ---
        # Local = Parent_Global_Inv * Current_Global
        # Find SMPL Parent
        # (Hardcoded standard SMPL hierarchy for robustness)
        smpl_parent = None
        if i > 0: # Hips is 0
            # Simple lookup based on your SMPL_H_NAMES list structure
            # This is risky if list changes. 
            # Better: explicit map.
            if name == "LeftUpLeg": smpl_parent = "Hips"
            elif name == "RightUpLeg": smpl_parent = "Hips"
            elif name == "Spine": smpl_parent = "Hips"
            elif name == "LeftLeg": smpl_parent = "LeftUpLeg"
            elif name == "RightLeg": smpl_parent = "RightUpLeg"
            elif name == "LeftFoot": smpl_parent = "LeftLeg"
            elif name == "RightFoot": smpl_parent = "RightLeg"
            elif name == "Spine1": smpl_parent = "Spine"
            elif name == "Spine2": smpl_parent = "Spine1"
            elif name == "Neck": smpl_parent = "Spine2"
            elif name == "Head": smpl_parent = "Neck"
            elif name == "LeftShoulder": smpl_parent = "Spine2"
            elif name == "RightShoulder": smpl_parent = "Spine2"
            elif name == "LeftArm": smpl_parent = "LeftShoulder"
            elif name == "RightArm": smpl_parent = "RightShoulder"
            elif name == "LeftForeArm": smpl_parent = "LeftArm"
            elif name == "RightForeArm": smpl_parent = "RightArm"
            elif name == "LeftHand": smpl_parent = "LeftForeArm"
            elif name == "RightHand": smpl_parent = "RightForeArm"
        
        if smpl_parent and smpl_parent in smpl_globals:
            parent_q = smpl_globals[smpl_parent]
            local_q = parent_q.inv() * smpl_global_q
        else:
            # Root or Independent
            local_q = smpl_global_q
            
        # Store as Axis Angle
        # FIX: Stabilize Quats before converting to vector to avoid flips
        q_data = local_q.as_quat()
        q_data = fix_quaternion_continuity(q_data)
        final_poses[:, i] = R.from_quat(q_data).as_rotvec()

    return final_poses, final_trans

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    
    # 1. Load
    frames, nodes, hierarchy, ft = load_bvh_raw(args.input)
    
    # 2. Process
    poses, trans = process_universal(frames, nodes, hierarchy)
    
    # 3. Grounding
    min_z = np.min(trans[:, 2])
    if min_z < 0.05:
        print(f"Grounding: {min_z:.4f} -> 0.0")
        trans[:, 2] -= min_z
        
    # 4. Save
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    np.savez(args.output,
             poses=poses,
             trans=trans,
             transl=trans,
             global_orient=poses[:, 0:1],
             body_pose=poses[:, 1:22].reshape(frames.shape[0], -1),
             joint_names=SMPL_H_NAMES,
             fps=1.0/ft
    )
    print(f"✅ Saved UNIVERSAL FIX: {args.output}")

if __name__ == "__main__":
    main()