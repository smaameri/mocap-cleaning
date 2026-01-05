import sys
import os
import argparse
import numpy as np
from scipy.spatial.transform import Rotation as R

# --- CONFIGURATION ---
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

ROOT_FIX = R.from_euler("xyz", [90, 0, 180], degrees=True).as_matrix()
ROOT_FIX_INV = ROOT_FIX.T

def parse_bvh_header(path):
    with open(path, 'r') as f:
        lines = f.readlines()

    channel_map = {}
    header_end_idx = 0  # Where HIERARCHY ends
    channel_count = 0
    curr_joint = None
    
    for i, line in enumerate(lines):
        parts = line.strip().split()
        if not parts: continue
        
        # STOP parsing when we hit the MOTION section
        if parts[0] == "MOTION":
            header_end_idx = i  # Stop BEFORE this line
            break
            
        if parts[0] in ["ROOT", "JOINT"]:
            curr_joint = parts[1]
        elif parts[0] == "CHANNELS":
            count = int(parts[1])
            ch_names = parts[2:]
            
            rot_order = ""
            pos_idx = {}
            rot_idx = {}
            
            for c_i, ch in enumerate(ch_names):
                g_i = channel_count + c_i
                if "position" in ch.lower():
                    if "X" in ch: pos_idx["x"] = g_i
                    if "Y" in ch: pos_idx["y"] = g_i
                    if "Z" in ch: pos_idx["z"] = g_i
                elif "rotation" in ch.lower():
                    if "X" in ch: rot_order += "x"; rot_idx["x"] = g_i
                    if "Y" in ch: rot_order += "y"; rot_idx["y"] = g_i
                    if "Z" in ch: rot_order += "z"; rot_idx["z"] = g_i
            
            channel_map[curr_joint] = {
                "rot_order": rot_order,
                "rot_indices": rot_idx,
                "pos_indices": pos_idx
            }
            channel_count += count
            
    return channel_map, lines[:header_end_idx], channel_count

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--npz", required=True)
    parser.add_argument("--template", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    print(f"--- EXPORTING BVH: {os.path.basename(args.output)} ---")

    # 1. LOAD
    data = np.load(args.npz)
    poses = data['poses']
    if 'transl' in data: trans = data['transl']
    elif 'trans' in data: trans = data['trans']
    else: trans = np.zeros((poses.shape[0], 3))

    T = poses.shape[0]

    # 2. TEMPLATE
    j_map, header_lines, n_channels = parse_bvh_header(args.template)
    
    # 3. SCALE (Meters -> CM)
    SCALE = 100.0
    
    # 4. BUILD
    bvh_frames = np.zeros((T, n_channels))
    
    for t in range(T):
        # ROOT
        if "Hips" in j_map:
            m = j_map["Hips"]
            root_p = ROOT_FIX_INV @ trans[t]
            root_p *= SCALE
            
            if "x" in m["pos_indices"]: bvh_frames[t, m["pos_indices"]["x"]] = root_p[0]
            if "y" in m["pos_indices"]: bvh_frames[t, m["pos_indices"]["y"]] = root_p[1]
            if "z" in m["pos_indices"]: bvh_frames[t, m["pos_indices"]["z"]] = root_p[2]
            
            r_mat = R.from_rotvec(poses[t, 0]).as_matrix()
            r_bvh = ROOT_FIX_INV @ r_mat
            
            order = m["rot_order"]
            if len(order) == 3:
                eul = R.from_matrix(r_bvh).as_euler(order, degrees=True)
                bvh_frames[t, m["rot_indices"][order[0]]] = eul[0]
                bvh_frames[t, m["rot_indices"][order[1]]] = eul[1]
                bvh_frames[t, m["rot_indices"][order[2]]] = eul[2]

        # JOINTS
        for i, name in enumerate(SMPL_H_NAMES):
            if i == 0: continue
            if name in j_map:
                m = j_map[name]
                order = m["rot_order"]
                if len(order) == 3:
                    r_loc = R.from_rotvec(poses[t, i])
                    eul = r_loc.as_euler(order, degrees=True)
                    bvh_frames[t, m["rot_indices"][order[0]]] = eul[0]
                    bvh_frames[t, m["rot_indices"][order[1]]] = eul[1]
                    bvh_frames[t, m["rot_indices"][order[2]]] = eul[2]

    # 5. WRITE
    with open(args.output, 'w') as f:
        f.writelines(header_lines) # Write Hierarchy
        if not header_lines[-1].endswith('\n'): f.write('\n')
        
        f.write("MOTION\n")
        f.write(f"Frames: {T}\n")
        f.write("Frame Time: 0.00833333\n")
        np.savetxt(f, bvh_frames, fmt="%.6f")
        
    print(f"✅ DONE. Saved {args.output}")

if __name__ == "__main__":
    main()