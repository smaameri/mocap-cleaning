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
# 2. ROBUST ZXY LOADER
# ==========================================
def load_bvh_data(path):
    print(f"Reading: {os.path.basename(path)}")
    with open(path, 'r') as f:
        content = f.read().split()
    
    iterator = iter(content)
    nodes = []
    
    # Simple hierarchy scan to map channels
    node_channel_map = []
    
    token = next(iterator, None)
    while token:
        if token == "MOTION": break
        if token in ["ROOT", "JOINT"]:
            name = next(iterator)
            # Find channels
            while True:
                t = next(iterator)
                if t == "CHANNELS":
                    count = int(next(iterator))
                    channels = [next(iterator) for _ in range(count)]
                    node_channel_map.append({'name': name, 'channels': channels})
                    break
        token = next(iterator, None)
    
    # Parse Motion
    num_frames = 0
    frame_time = 0.008333
    
    while token:
        if token == "Frames:": num_frames = int(next(iterator))
        elif token == "Frame": 
            next(iterator) # Time:
            frame_time = float(next(iterator))
            break
        token = next(iterator, None)
        
    # Read floats
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
# 3. PROCESSING (The Fix)
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
        
        # 1. Extract Rotation (ZXY)
        # Identify indices of Z, X, Y rotation
        rot_cols = [i for i, c in enumerate(node['channels']) if 'rotation' in c]
        rot_order = "".join([node['channels'][i][0].lower() for i in rot_cols]) # e.g., "zxy"
        
        if len(rot_cols) == 3:
            euler_vals = data[:, rot_cols]
            # Convert to Matrix
            rot_mats = R.from_euler(rot_order, euler_vals, degrees=True).as_matrix()
            
            if name == "Hips":
                # --- ROOT LOGIC ---
                # Apply Global Rotation (Rx90) to align Y-up world to Z-up world
                # Global_New = Rx90 * Global_Old
                new_roots = np.matmul(RX_90, rot_mats)
                poses[:, idx] = R.from_matrix(new_roots).as_rotvec()
                
                # Handle Position (Scale + Rotate)
                pos_cols = [i for i, c in enumerate(node['channels']) if 'position' in c]
                if len(pos_cols) == 3:
                    pos_vecs = data[:, pos_cols] * 0.01 # CM to Meters
                    # Apply Rx90 to position
                    trans = np.dot(pos_vecs, RX_90.T)
            else:
                # --- BODY LOGIC ---
                # DO NOT rotate these! They are local to the parent.
                # Since the parent (and the whole chain) was rotated at the root,
                # these local bend angles remain exactly the same.
                poses[:, idx] = R.from_matrix(rot_mats).as_rotvec()

    return poses, trans

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    
    # 1. Load
    raw_frames, structure, ft = load_bvh_data(args.input)
    
    # 2. Process (Correct Logic)
    poses, trans = process_data(raw_frames, structure)
    
    # 3. Simple Grounding
    min_z = np.min(trans[:, 2])
    if min_z < 0.1:
        shift = 0.3 - min_z # Force lowest point to be 30cm off floor
        print(f"Grounding shift: {shift:.3f}m")
        trans[:, 2] += shift
    
    # 4. Save
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    # Corrected variable name: len(raw_frames) instead of len(frames)
    np.savez(args.output,
             poses=poses,
             trans=trans,
             transl=trans,
             global_orient=poses[:, 0:1],
             body_pose=poses[:, 1:22].reshape(len(raw_frames), -1),
             joint_names=SMPL_H_NAMES,
             fps=1.0/ft
    )
    print(f"Saved CLEAN file: {args.output}")

if __name__ == "__main__":
    main()