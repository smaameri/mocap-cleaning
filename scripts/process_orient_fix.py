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

# FIX 1: Stand Up (Y -> Z)
RX_90 = R.from_euler('x', 90, degrees=True).as_matrix()

# FIX 2: Turn Around (Front -> Back)
# Try 180 degrees. If still wrong, change to 0 or 90.
RZ_180 = R.from_euler('z', 180, degrees=True).as_matrix() 

# Combined Global Correction
GLOBAL_FIX = RZ_180 @ RX_90

# ==========================================
# 2. LOADER (Robust)
# ==========================================
def load_bvh_data(path):
    print(f"Reading: {os.path.basename(path)}")
    with open(path, 'r') as f:
        content = f.read().split()
    
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
# 3. PROCESSING
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
        
        # Identify Rotation
        rot_cols = [i for i, c in enumerate(node['channels']) if 'rotation' in c]
        if len(rot_cols) == 3:
            # Parse ZXY Euler -> Matrix directly (No Unwrap to avoid artifacts)
            # We assume continuous data stream from BVH
            euler_vals = data[:, rot_cols]
            rot_order = "".join([node['channels'][i][0].lower() for i in rot_cols]) 
            rot_mats = R.from_euler(rot_order, euler_vals, degrees=True).as_matrix()
            
            if name == "Hips":
                # --- ROOT: Apply Global Fix ---
                # Rotate the orientation of the Hips to face the correct way
                new_roots = np.matmul(GLOBAL_FIX, rot_mats)
                poses[:, idx] = R.from_matrix(new_roots).as_rotvec()
                
                # --- ROOT POS: Scale & Rotate ---
                pos_cols = [i for i, c in enumerate(node['channels']) if 'position' in c]
                if len(pos_cols) == 3:
                    sample = data[0, pos_cols]
                    scale = 0.01 if np.max(np.abs(sample)) > 50 else 1.0
                    pos_vecs = data[:, pos_cols] * scale
                    
                    # Apply the same Global Fix to the Translation vector
                    trans = np.dot(pos_vecs, GLOBAL_FIX.T)
            else:
                # --- BODY: Keep Local ---
                poses[:, idx] = R.from_matrix(rot_mats).as_rotvec()

    return poses, trans

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    
    raw_frames, structure, ft = load_bvh_data(args.input)
    poses, trans = process_data(raw_frames, structure)
    
    # Grounding
    min_z = np.min(trans[:, 2])
    shift = -min_z
    print(f"Grounding: Shifting up by {shift:.4f}m")
    trans[:, 2] += shift
    
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
    print(f"✅ Saved Fixed File: {args.output}")

if __name__ == "__main__":
    main()