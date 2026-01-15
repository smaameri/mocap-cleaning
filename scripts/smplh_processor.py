import numpy as np
import argparse
import os
import sys
from scipy.spatial.transform import Rotation as R
from scipy.signal import savgol_filter

# SMPL-H Full Joint List (52 Joints including Fingers)
SMPL_H_FULL = [
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

def load_bvh_raw(path):
    """Robust BVH Loader for ProxiData structure."""
    with open(path, 'r') as f: content = f.read().split()
    iterator = iter(content)
    structure = []
    token = next(iterator, None)
    while token != "MOTION":
        if token in ["ROOT", "JOINT"]:
            name = next(iterator)
            while True:
                t = next(iterator)
                if t == "CHANNELS":
                    count = int(next(iterator))
                    chs = [next(iterator) for _ in range(count)]
                    structure.append({'name': name, 'channels': chs})
                    break
        token = next(iterator, None)
    
    # Parse frames
    num_frames = 0
    m_fps = 120.0
    while token:
        if token == "Frames:": 
            num_frames = int(next(iterator))
        elif "Time" in token: 
            m_fps = 1.0 / float(next(iterator))
            break
        token = next(iterator, None)
    
    vals = [float(x) for x in list(iterator)]
    expected_size = num_frames * sum(len(n['channels']) for n in structure)
    
    if len(vals) > expected_size:
        vals = vals[:expected_size]
    elif len(vals) < expected_size:
        raise ValueError(f"Incomplete data: Read {len(vals)}, Expected {expected_size}")

    frames = np.array(vals).reshape(num_frames, -1)
    return frames, structure, m_fps

def process_ultimate_v3(frames, structure):
    T = frames.shape[0]
    poses = np.zeros((T, len(SMPL_H_FULL), 3))
    trans = np.zeros((T, 3))
    M_world = R.from_euler('x', 90, degrees=True).as_matrix()

    ptr = 0
    for node in structure:
        name = node['name']
        n_ch = len(node['channels'])
        data = frames[:, ptr : ptr + n_ch]
        ptr += n_ch
        
        if name not in SMPL_H_FULL: continue
        idx = SMPL_H_FULL.index(name)
        
        rot_indices = [i for i, c in enumerate(node['channels']) if 'rotation' in c]
        if len(rot_indices) == 3:
            order = "".join([node['channels'][i][0] for i in rot_indices]) 
            angles = np.deg2rad(data[:, rot_indices])
            angles = np.unwrap(angles, axis=0)
            
            if T > 11:
                angles = savgol_filter(angles, window_length=11, polyorder=3, axis=0)
            
            rot_mats = R.from_euler(order, angles, degrees=False).as_matrix()
            
            if name == "Hips":
                fixed_roots = np.matmul(M_world, rot_mats)
                poses[:, idx] = R.from_matrix(fixed_roots).as_rotvec()
                pos_indices = [i for i, c in enumerate(node['channels']) if 'position' in c]
                t_raw = data[:, pos_indices] * 0.01 
                trans = np.dot(t_raw, M_world.T)
            else:
                r_start_inv = R.from_matrix(rot_mats[0]).inv()
                for t in range(T):
                    corrected_mat = rot_mats[t] @ r_start_inv.as_matrix()
                    poses[t, idx] = R.from_matrix(corrected_mat).as_rotvec()
    return poses, trans

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    try:
        print(f"Processing: {os.path.basename(args.input)}")
        frames, structure, m_fps = load_bvh_raw(args.input)
        poses, trans = process_ultimate_v3(frames, structure)

        trans[:, 2] -= np.min(trans[:, 2])

        os.makedirs(os.path.dirname(args.output), exist_ok=True)
        np.savez(args.output,
                 poses=poses.reshape(poses.shape[0], -1),
                 trans=trans, transl=trans,
                 joint_names=SMPL_H_FULL,
                 mocap_framerate=int(m_fps))
        
        # FIXED: Removed Emoji for terminal compatibility
        print(f"SUCCESS: Corrected Fix Saved: {args.output}")

    except Exception as e:
        import traceback
        # FIXED: Removed Emoji for terminal compatibility
        print(f"FAILED: Error processing file: {str(e)}")
        print(traceback.format_exc())
        sys.exit(1)

if __name__ == "__main__":
    main()