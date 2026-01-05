import numpy as np
import argparse
import os
import sys
from scipy.spatial.transform import Rotation as R

# ==========================================
# 1. SETTINGS & MANUAL ADJUSTMENTS
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

# --- FIX 1: World Rotation (Stand Up) ---
RX_90 = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]])

# --- FIX 2: Arm Correction (Bring arms forward) ---
# Agar haath piche ja rahe hain, to unhe Z-axis par wapas rotate karo
ARM_ADJUST = 40.0 # Degrees
L_SHOULDER_FIX = R.from_euler('z', -ARM_ADJUST, degrees=True).as_matrix() # Left Aage
R_SHOULDER_FIX = R.from_euler('z', ARM_ADJUST, degrees=True).as_matrix()  # Right Aage

# ==========================================
# 2. HELPER: QUATERNION STABILIZATION
# ==========================================
def robust_smooth_quaternions(quats):
    """Frame 800+ ke spins ko fix karne ke liye smoothing"""
    for i in range(1, len(quats)):
        dot = np.dot(quats[i], quats[i-1])
        if dot < 0:
            quats[i] = -quats[i]
    return quats

# ==========================================
# 3. LOADER
# ==========================================
def load_bvh_data(path):
    print(f"Reading: {os.path.basename(path)}")
    with open(path, 'r') as f: content = f.read().split()
    iterator = iter(content)
    node_channel_map = []
    
    token = next(iterator, None)
    while token:
        if token == "MOTION": break
        if token in ["ROOT", "JOINT"]:
            name = next(iterator)
            while True:
                t = next(iterator)
                if t == "CHANNELS":
                    count = int(next(iterator))
                    channels = [next(iterator) for _ in range(count)]
                    node_channel_map.append({'name': name, 'channels': channels})
                    break
        token = next(iterator, None)
    
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
    return frames, node_channel_map, frame_time

# ==========================================
# 4. PROCESSING (MANUAL FIX)
# ==========================================
def process_data(frames, structure):
    T = frames.shape[0]
    poses = np.zeros((T, len(SMPL_H_NAMES), 3))
    trans = np.zeros((T, 3))
    
    ptr = 0
    for node in structure:
        name = node['name']
        n_ch = len(node['channels'])
        data = frames[:, ptr : ptr + n_ch]
        ptr += n_ch
        
        if name not in SMPL_H_NAMES: continue
        idx = SMPL_H_NAMES.index(name)
        
        # Extract Rotation
        rot_cols = [i for i, c in enumerate(node['channels']) if 'rotation' in c]
        if len(rot_cols) == 3:
            rot_order = "".join([node['channels'][i][0].lower() for i in rot_cols]) 
            
            # 1. Unwrap Angles (Fix Frame 800 Glitch)
            unwrapped = np.unwrap(np.deg2rad(data[:, rot_cols]), axis=0)
            
            # 2. Smooth Quaternions (Prevent Flipping)
            quats = R.from_euler(rot_order, unwrapped, degrees=False).as_quat()
            quats = robust_smooth_quaternions(quats)
            rot_mats = R.from_quat(quats).as_matrix()
            
            # --- APPLY CORRECTIONS ---
            
            if name == "Hips":
                # Root Fix: Y-Up -> Z-Up
                new_roots = np.matmul(RX_90, rot_mats)
                poses[:, idx] = R.from_matrix(new_roots).as_rotvec()
                
                # Root Pos Fix
                pos_cols = [i for i, c in enumerate(node['channels']) if 'position' in c]
                if len(pos_cols) == 3:
                    sample = data[0, pos_cols]
                    scale = 0.01 if np.max(np.abs(sample)) > 50 else 1.0
                    pos_vecs = data[:, pos_cols] * scale
                    trans = np.dot(pos_vecs, RX_90.T)
            
            elif name == "LeftArm" or name == "LeftShoulder":
                # Arm Fix: Rotate Forward
                # New = Correction * Old
                new_arms = np.matmul(L_SHOULDER_FIX, rot_mats)
                poses[:, idx] = R.from_matrix(new_arms).as_rotvec()
                
            elif name == "RightArm" or name == "RightShoulder":
                # Arm Fix: Rotate Forward
                new_arms = np.matmul(R_SHOULDER_FIX, rot_mats)
                poses[:, idx] = R.from_matrix(new_arms).as_rotvec()
                
            else:
                # Legs & Body: Keep original (Direct Map)
                poses[:, idx] = R.from_matrix(rot_mats).as_rotvec()

    return poses, trans, T

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    
    raw_frames, structure, ft = load_bvh_data(args.input)
    poses, trans, T = process_data(raw_frames, structure)
    
    # Grounding Check
    min_z = np.min(trans[:, 2])
    print(f"Lowest Z detected: {min_z:.4f}")
    if min_z < 0.05: 
        shift = -min_z
        print(f"Shifting up by {shift:.4f}m")
        trans[:, 2] += shift
    
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
    print(f"✅ Saved MANUAL FIXED File: {args.output}")

if __name__ == "__main__":
    main()