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

RX_90 = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]])

BONE_MAP = {
    "LeftUpLeg": ["LeftUpLeg", "LHip"], "RightUpLeg": ["RightUpLeg", "RHip"],
    "LeftArm": ["LeftArm", "LShoulder"], "RightArm": ["RightArm", "RShoulder"],
    "Spine": ["Spine"], "Spine1": ["Spine1"], "Neck": ["Neck"],
    "LeftShoulder": ["LeftShoulder"], "RightShoulder": ["RightShoulder"],
    "LeftForeArm": ["LeftForeArm"], "RightForeArm": ["RightForeArm"]
}

# ==========================================
# 2. HELPER: ROBUST CONTINUITY
# ==========================================
def make_continuous(quats):
    for i in range(1, len(quats)):
        if np.dot(quats[i], quats[i-1]) < 0: quats[i] = -quats[i]
    return quats

# ==========================================
# 3. LOADER & GLOBAL COMPUTATION
# ==========================================
def load_and_compute_globals(path):
    print(f"Reading: {os.path.basename(path)}")
    with open(path, 'r') as f: content = f.read().split()
    iterator = iter(content)
    
    parent_map = {}
    nodes = {}
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
        elif token == "End": next(iterator); node_stack.append("EndSite")
        elif token == "}": node_stack.pop()
        token = next(iterator, None)
        
    while token != "Frames:": token = next(iterator)
    num_frames = int(next(iterator))
    while token != "Frame": token = next(iterator)
    next(iterator); ft = float(next(iterator))
    
    vals = []
    try:
        while True: vals.append(float(next(iterator)))
    except: pass
    frames = np.array(vals).reshape(num_frames, -1)
    
    # Compute Global Rotations
    T = frames.shape[0]
    global_rots = {} 
    
    ptr = 0
    for name, channels in nodes.items():
        n_ch = len(channels)
        data = frames[:, ptr : ptr + n_ch]
        ptr += n_ch
        
        rot_cols = [i for i, c in enumerate(channels) if 'rotation' in c]
        if len(rot_cols) == 3:
            order = "".join([channels[i][0].lower() for i in rot_cols])
            euler = np.unwrap(np.deg2rad(data[:, rot_cols]), axis=0)
            local_mat = R.from_euler(order, euler, degrees=False).as_matrix()
        else:
            local_mat = np.tile(np.eye(3), (T, 1, 1))

        parent = parent_map.get(name)
        if parent and parent in global_rots:
            global_mat = np.matmul(global_rots[parent], local_mat)
        else:
            global_mat = local_mat
        global_rots[name] = global_mat
        
    root_pos = frames[:, :3] 
    return global_rots, root_pos, ft, T

# ==========================================
# 4. CUSTOM RANGE SCANNING (800-900)
# ==========================================
def find_best_frame_in_range(global_rots, T):
    # --- CUSTOM RANGE SETTING ---
    start_f = 800
    end_f = 900
    
    # Safety Check
    if start_f >= T:
        print(f"!! Warning: File has only {T} frames. Scanning 0-30 instead.")
        start_f = 0
        end_f = min(30, T)
    else:
        end_f = min(end_f, T)
        
    best_frame = start_f
    best_score = -999.0
    
    print(f">> Scanning Frames {start_f} to {end_f} for Optimal Pose...")
    
    # Ideal T-Pose Vectors (Global X)
    ideal_L = np.array([1, 0, 0])
    ideal_R = np.array([-1, 0, 0])
    
    l_arm_key = next((k for k in global_rots if "LeftArm" in k or "LShoulder" in k), None)
    r_arm_key = next((k for k in global_rots if "RightArm" in k or "RShoulder" in k), None)
    
    if not l_arm_key or not r_arm_key: return 0

    for t in range(start_f, end_f):
        R_L = global_rots[l_arm_key][t]
        R_R = global_rots[r_arm_key][t]
        
        # Assuming bone axis is X (based on your log)
        bone_axis = np.array([1, 0, 0])
        
        curr_L = R_L @ bone_axis
        curr_R = R_R @ bone_axis
        
        score = np.dot(curr_L, ideal_L) + np.dot(curr_R, ideal_R)
        
        if score > best_score:
            best_score = score
            best_frame = t
            
    print(f">> BEST REST POSE FOUND: Frame {best_frame} (Score: {best_score:.4f})")
    return best_frame

# ==========================================
# 5. RETARGETING ENGINE
# ==========================================
def process_stat_fix(global_rots_bvh, root_pos, T):
    poses = np.zeros((T, len(SMPL_H_NAMES), 3))
    
    scale = 0.01 if np.max(np.abs(root_pos)) > 50 else 1.0
    print(f">> Scale: {scale}")
    trans = np.dot(root_pos * scale, RX_90.T)
    
    # 1. FIND BEST FRAME (800-900)
    rest_frame_idx = find_best_frame_in_range(global_rots_bvh, T)
    
    # 2. CALIBRATE
    calibration_mats = {}
    for name in global_rots_bvh:
        rest_rot = global_rots_bvh[name][rest_frame_idx]
        calibration_mats[name] = np.linalg.inv(rest_rot)

    # 3. APPLY
    smpl_global_mats = {}
    
    for i, name in enumerate(SMPL_H_NAMES):
        bvh_key = None
        for cand in BONE_MAP.get(name, []):
            for k in global_rots_bvh.keys():
                if cand.lower() in k.lower(): bvh_key = k; break
            if bvh_key: break
            
        if bvh_key:
            # Calibrated Global Rotation
            bvh_curr = global_rots_bvh[bvh_key]
            calib = calibration_mats[bvh_key]
            delta = np.matmul(bvh_curr, calib)
            target_global = np.matmul(RX_90, delta)
        else:
            target_global = np.tile(np.eye(3), (T, 1, 1))
            
        smpl_global_mats[name] = target_global
        
        # Compute Local
        parent_name = None
        if i > 0:
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
            parent_inv = np.transpose(parent_mat, (0, 2, 1))
            local_mat = np.matmul(parent_inv, target_global)
        else:
            local_mat = target_global
            
        quats = R.from_matrix(local_mat).as_quat()
        quats = make_continuous(quats)
        poses[:, i] = R.from_quat(quats).as_rotvec()
        
    return poses, trans

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    
    global_rots, root_pos, ft, T = load_and_compute_globals(args.input)
    poses, trans = process_stat_fix(global_rots, root_pos, T)
    
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
    print(f"✅ SAVED 800-900 SCAN FIX: {args.output}")

if __name__ == "__main__":
    main()