import numpy as np
import argparse
import os
import sys
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

# Coordinate Rotation: Y-Up -> Z-Up
RX_90 = R.from_euler('x', 90, degrees=True).as_matrix()

# ==========================================
# 2. LOADER
# ==========================================
def load_bvh_data(path):
    print(f"Reading: {os.path.basename(path)}")
    with open(path, 'r') as f:
        content = f.read().split()
    
    iterator = iter(content)
    nodes = {}
    node_order = []
    parent_map = {}
    
    # Hierarchy
    node_stack = []
    token = next(iterator, None)
    while token:
        if token == "MOTION": break
        if token in ["ROOT", "JOINT"]:
            name = next(iterator)
            node_order.append(name)
            parent = node_stack[-1] if node_stack else None
            parent_map[name] = parent
            
            # Read channels
            while True:
                t = next(iterator)
                if t == "CHANNELS":
                    count = int(next(iterator))
                    channels = [next(iterator) for _ in range(count)]
                    nodes[name] = channels
                    node_stack.append(name)
                    break
        elif token == "End":
            next(iterator) # Site
            node_stack.append("EndSite")
        elif token == "}":
            node_stack.pop()
        token = next(iterator, None)
    
    # Motion
    num_frames = 0
    frame_time = 0.008333
    while token:
        if token == "Frames:": num_frames = int(next(iterator))
        elif token == "Frame": 
            next(iterator)
            frame_time = float(next(iterator))
            break
        token = next(iterator, None)
        
    vals = []
    try:
        while True:
            t = next(iterator, None)
            if t is None: break
            vals.append(float(t))
    except: pass
    
    frames = np.array(vals).reshape(num_frames, -1)
    return frames, nodes, node_order, parent_map, frame_time

# ==========================================
# 3. GLOBAL REBAKE ENGINE
# ==========================================
def rebake_motion(frames, nodes, node_order, parent_map):
    T = frames.shape[0]
    
    # Store Global Quaternions for every joint
    global_quats = {name: np.zeros((T, 4)) for name in SMPL_H_NAMES}
    # Initialize with identity quaternions [x, y, z, w]
    for name in SMPL_H_NAMES:
        global_quats[name][:, 3] = 1.0
        
    root_pos = np.zeros((T, 3))
    
    # 1. FORWARD KINEMATICS (Calculate Globals)
    print(">> Computing Global Rotations...")
    ptr = 0
    
    # We must process frame by frame or vectorized by channel
    # Vectorized is faster.
    
    curr_ptr = 0
    for name in node_order:
        chs = nodes[name]
        n_ch = len(chs)
        data = frames[:, curr_ptr : curr_ptr + n_ch]
        curr_ptr += n_ch
        
        if name not in SMPL_H_NAMES: continue
        
        # Get Local Rotation
        rot_cols = [i for i, c in enumerate(chs) if 'rotation' in c]
        if len(rot_cols) == 3:
            order = "".join([chs[i][0].lower() for i in rot_cols]) # "zxy"
            # Unwrap Eulers first to be safe
            euler = np.unwrap(np.deg2rad(data[:, rot_cols]), axis=0)
            local_q = R.from_euler(order, euler, degrees=False)
        else:
            local_q = R.identity(T)
            
        # Get Position (Root only usually)
        if name == "Hips":
            pos_cols = [i for i, c in enumerate(chs) if 'position' in c]
            if len(pos_cols) == 3:
                # CM to Meters
                sample = data[0, pos_cols]
                scale = 0.01 if np.max(np.abs(sample)) > 50 else 1.0
                root_pos = data[:, pos_cols] * scale

        # Compute Global
        parent = parent_map.get(name)
        if parent and parent in global_quats:
            # Global = Parent_Global * Local
            p_q = R.from_quat(global_quats[parent])
            g_q = p_q * local_q
        else:
            # Root
            g_q = local_q
            
        global_quats[name] = g_q.as_quat()

    # 2. STABILIZE GLOBALS (The Magic Fix)
    print(">> Stabilizing Quaternions...")
    for name in SMPL_H_NAMES:
        # Enforce shortest path continuity on the GLOBAL rotation
        # This prevents the "mix" flipping issue
        q = global_quats[name]
        for i in range(1, T):
            if np.dot(q[i], q[i-1]) < 0:
                q[i] = -q[i]
        global_quats[name] = q

    # 3. RE-COMPUTE LOCALS (Inverse Kinematics)
    print(">> Rebaking Local SMPL Rotations...")
    final_poses = np.zeros((T, len(SMPL_H_NAMES), 3))
    final_trans = np.zeros((T, 3))
    
    # Fix Root Orientation (Y-Up -> Z-Up)
    # New_Root_Global = RX_90 * Old_Root_Global
    r_root_orig = R.from_quat(global_quats["Hips"])
    r_root_new = R.from_matrix(RX_90) * r_root_orig
    final_poses[:, 0] = r_root_new.as_rotvec()
    
    # Fix Root Position (Rotate Vector)
    final_trans = np.dot(root_pos, RX_90.T)
    
    # Fix Body Joints
    for i, name in enumerate(SMPL_H_NAMES):
        if i == 0: continue # Root done
        
        parent = parent_map.get(name)
        if parent not in SMPL_H_NAMES: continue # Should not happen if lists match
        
        # We need Local = Parent_Global_Inv * Global
        # BUT: We must respect that the whole world rotated by RX_90?
        # Actually, for internal joints, relative rotation is invariant to global rotation.
        # R_local = R_parent_global^-1 * R_child_global
        
        q_p = R.from_quat(global_quats[parent])
        q_c = R.from_quat(global_quats[name])
        
        # Calculate Local: Inv(Parent) * Child
        q_local = q_p.inv() * q_c
        
        final_poses[:, i] = q_local.as_rotvec()
        
    return final_poses, final_trans

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    
    # 1. Load
    frames, nodes, order, parents, ft = load_bvh_data(args.input)
    
    # 2. Rebake
    poses, trans = rebake_motion(frames, nodes, order, parents)
    
    # 3. Ground
    min_z = np.min(trans[:, 2])
    print(f"Grounding shift: {-min_z:.4f}m")
    trans[:, 2] -= min_z
    
    # 4. Save
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    np.savez(args.output,
             poses=poses,
             trans=trans,
             transl=trans,
             global_orient=poses[:, 0:1],
             body_pose=poses[:, 1:22].reshape(poses.shape[0], -1),
             joint_names=SMPL_H_NAMES,
             fps=1.0/ft
    )
    print(f"✅ DONE. Re-baked file: {args.output}")

if __name__ == "__main__":
    main()