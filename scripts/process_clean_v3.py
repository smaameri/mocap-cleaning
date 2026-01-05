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

# Coordinate Rotation: 90 degrees around X (Y-Up -> Z-Up)
RX_90 = np.array([
    [1, 0, 0],
    [0, 0, -1],
    [0, 1, 0]
])

# ==========================================
# 2. HELPER: QUATERNION STABILIZATION
# ==========================================
def fix_quaternion_continuity(quats):
    """
    Ensures that quaternions don't flip sign (q vs -q).
    This prevents the 'long way around' rotation glitch during turns.
    """
    for i in range(1, len(quats)):
        # Dot product checks if q[i] is 'close' to q[i-1]
        dot = np.dot(quats[i], quats[i-1])
        if dot < 0:
            # If negative, we are on the opposite side of the hypersphere.
            # Flip the sign to force the shortest path.
            quats[i] = -quats[i]
    return quats

# ==========================================
# 3. LOADER
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
# 4. PROCESSING
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
        
        # 1. Rotation Channels
        rot_cols = [i for i, c in enumerate(node['channels']) if 'rotation' in c]
        if len(rot_cols) == 3:
            euler_vals = data[:, rot_cols]
            rot_order = "".join([node['channels'][i][0].lower() for i in rot_cols]) # "zxy"
            
            # --- FIX: Convert to Quaternion & Stabilize ---
            # 1. Euler -> Quat
            quats = R.from_euler(rot_order, euler_vals, degrees=True).as_quat()
            
            # 2. Fix Flipping (The Magic Step)
            quats = fix_quaternion_continuity(quats)
            
            # 3. Quat -> Matrix
            rot_mats = R.from_quat(quats).as_matrix()
            
            if name == "Hips":
                # Apply Global Orientation Correction
                new_roots = np.matmul(RX_90, rot_mats)
                poses[:, idx] = R.from_matrix(new_roots).as_rotvec()
                
                # Apply Global Position Correction
                pos_cols = [i for i, c in enumerate(node['channels']) if 'position' in c]
                if len(pos_cols) == 3:
                    sample = data[0, pos_cols]
                    scale = 0.01 if np.max(np.abs(sample)) > 50 else 1.0
                    pos_vecs = data[:, pos_cols] * scale
                    trans = np.dot(pos_vecs, RX_90.T)
            else:
                poses[:, idx] = R.from_matrix(rot_mats).as_rotvec()

    return poses, trans, T

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    
    raw_frames, structure, ft = load_bvh_data(args.input)
    poses, trans, T = process_data(raw_frames, structure)
    
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
             body_pose=poses[:, 1:22].reshape(T, -1),
             joint_names=SMPL_H_NAMES,
             fps=1.0/ft
    )
    print(f"Saved STABILIZED (V3) file: {args.output}")

if __name__ == "__main__":
    main()