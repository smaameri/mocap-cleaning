import numpy as np
import os
import sys
from scipy.spatial.transform import Rotation as R
from scipy.interpolate import interp1d

# ==========================================
# SETTINGS
# ==========================================
INPUT_BVH = r"D:\REAL WORLD DATA\ProxiData-20251205T112014Z-3-001\ProxiData\ProxiData_raw\BVH\KO_vignette_09_002_BJ_AM_0806.bvh"
OUTPUT_NPZ = "output/clean_input_npz/FINAL_SMPLH_PRODUCTION.npz"
TARGET_FPS = 24

# 52 Joints (SMPL-H Standard)
SMPL_H_NAMES = [
    "Hips", "LeftUpLeg", "RightUpLeg", "Spine", "LeftLeg", "RightLeg",
    "Spine1", "LeftFoot", "RightFoot", "Spine2", "LeftToeBase", "RightToeBase",
    "Neck", "LeftShoulder", "RightShoulder", "Head", "LeftArm", "RightArm",
    "LeftForeArm", "RightForeArm", "LeftHand", "RightHand",
    "LeftHandIndex1", "LeftHandIndex2", "LeftHandIndex3",
    "LeftHandMiddle1", "LeftHandMiddle2", "LeftHandMiddle3",
    "LeftHandPinky1", "LeftHandPinky2", "LeftHandPinky3",
    "LeftHandRing1", "LeftHandRing2", "LeftHandRing3",
    "LeftHandThumb1", "LeftHandThumb2", "LeftHandThumb3",
    "RightHandIndex1", "RightHandIndex2", "RightHandIndex3",
    "RightHandMiddle1", "RightHandMiddle2", "RightHandMiddle3",
    "RightHandPinky1", "RightHandPinky2", "RightHandPinky3",
    "RightHandRing1", "RightHandRing2", "RightHandRing3",
    "RightHandThumb1", "RightHandThumb2", "RightHandThumb3"
]

# Root Fix (Upright)
ROOT_FIX = R.from_euler("xyz", [90, 0, 180], degrees=True).as_matrix()

def parse_bvh_file(path):
    print(f"Reading BVH: {os.path.basename(path)}...", flush=True)
    with open(path, 'r') as f: content = f.read()

    motion_start = content.find("MOTION")
    hierarchy_str = content[:motion_start]
    motion_str = content[motion_start:]

    joint_map = {}
    lines = hierarchy_str.splitlines()
    iterator = iter(lines)
    channel_count = 0
    
    for line in iterator:
        parts = line.strip().split()
        if not parts: continue
        if parts[0] in ["ROOT", "JOINT"]:
            name = parts[1]
            while True:
                ln = next(iterator).strip().split()
                if ln and ln[0] == "CHANNELS":
                    count = int(ln[1])
                    joint_map[name] = {"start": channel_count, "count": count}
                    channel_count += count
                    break
    
    data_lines = []
    frame_time = 0.016
    for line in motion_str.splitlines():
        if "Frame Time:" in line:
            frame_time = float(line.split()[2])
        elif "MOTION" in line or "Frames:" in line:
            continue
        else:
            nums = list(map(float, line.strip().split()))
            if len(nums) > 0: data_lines.append(nums)
            
    return joint_map, np.array(data_lines), frame_time

def resample_motion(poses, trans, source_fps, target_fps):
    if abs(source_fps - target_fps) < 1.0: return poses, trans
    print(f"Resampling {source_fps:.2f} -> {target_fps} FPS...")
    n_frames = len(poses)
    duration = n_frames / source_fps
    new_n = int(duration * target_fps)
    old_t = np.linspace(0, duration, n_frames)
    new_t = np.linspace(0, duration, new_n)
    
    f_trans = interp1d(old_t, trans, axis=0, kind='linear', fill_value="extrapolate")
    new_trans = f_trans(new_t)
    
    J = poses.shape[1]
    flat_poses = poses.reshape(n_frames, -1)
    f_poses = interp1d(old_t, flat_poses, axis=0, kind='linear', fill_value="extrapolate")
    new_poses = f_poses(new_t).reshape(new_n, J, 3)
    
    return new_poses, new_trans

def main():
    print("--- PROCESS FINAL SMPL-H (PRODUCTION) ---")
    if not os.path.exists(INPUT_BVH):
        print(f"❌ Error: {INPUT_BVH} not found")
        return
    joint_map, frames, ft = parse_bvh_file(INPUT_BVH)
    source_fps = 1.0 / ft
    T_raw = frames.shape[0]
    J_smpl = len(SMPL_H_NAMES)
    
    print("Mapping to SMPL Skeleton...")
    poses = np.zeros((T_raw, J_smpl, 3))
    trans = np.zeros((T_raw, 3))
    
    # Hips
    if "Hips" in joint_map:
        info = joint_map["Hips"]
        idx = info['start']
        raw_trans = frames[:, idx:idx+3] * 0.01 
        trans = (ROOT_FIX @ raw_trans.T).T
        
        raw_rot = frames[:, idx+3:idx+6]
        r_mat = R.from_euler('zxy', raw_rot, degrees=True).as_matrix()
        poses[:, 0] = R.from_matrix(ROOT_FIX @ r_mat).as_rotvec()
    
    # Body
    for i, name in enumerate(SMPL_H_NAMES):
        if i == 0: continue
        if name in joint_map:
            info = joint_map[name]
            offset = 3 if info['count'] == 6 else 0
            idx = info['start'] + offset
            raw_euler = frames[:, idx:idx+3]
            poses[:, i] = R.from_euler('zxy', raw_euler, degrees=True).as_rotvec()

    poses, trans = resample_motion(poses, trans, source_fps, TARGET_FPS)

    # Prepare for Renderer
    global_orient = poses[:, 0]
    body_pose = poses[:, 1:22].reshape(poses.shape[0], -1)
    
    print(f"Saving {OUTPUT_NPZ}...")
    np.savez(OUTPUT_NPZ, 
             global_orient=global_orient,
             body_pose=body_pose,
             transl=trans,
             trans=trans,
             poses=poses,
             fps=TARGET_FPS)
    print(f"✅ DONE. Production Ready.")

if __name__ == "__main__":
    main()