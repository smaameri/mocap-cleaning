import numpy as np
import argparse
import sys
import os
from scipy.spatial.transform import Rotation as R
import motion.bvh_loader as bvh_loader

def normalize(v):
    n = np.linalg.norm(v)
    return v / (n + 1e-8)

def get_bvh_orientation_correction(bvh_offsets):
    """
    Automatically detects the BVH 'Up' and 'Forward' axes by analyzing 
    the skeleton structure (Hips -> Head, Hips -> Legs).
    """
    def get_vec(name):
        return bvh_offsets.get(name, np.zeros(3))

    # 1. Detect UP Vector (Hips -> Head/Neck)
    spine_candidates = ["Head", "Neck", "Spine2", "Spine1", "Spine"]
    spine_vec = np.array([0., 1., 0.]) # Default Y-up
    
    # Check hierarchy direction
    hips_offset = get_vec("Hips")
    for name in spine_candidates:
        if name in bvh_offsets:
            vec = get_vec(name)
            if np.linalg.norm(vec) > 0.01: 
                spine_vec = vec
                break
                
    up_axis = normalize(spine_vec)
    
    # 2. Detect RIGHT Vector (LeftLeg -> RightLeg)
    l_leg = get_vec("LeftUpLeg")
    r_leg = get_vec("RightUpLeg")
    
    if np.linalg.norm(l_leg) > 0 and np.linalg.norm(r_leg) > 0:
        right_axis = normalize(r_leg - l_leg)
    else:
        right_axis = np.array([1., 0., 0.])
        
    # 3. Calculate FORWARD (Cross Product)
    forward_axis = normalize(np.cross(up_axis, right_axis))
    # Re-calculate Right to ensure orthogonality
    right_axis = normalize(np.cross(forward_axis, up_axis))
    
    # 4. Construct Rotation Matrix (BVH -> SMPL Z-Up)
    # BVH Basis (Columns = [Right, Forward, Up])
    bvh_rotation = np.stack([right_axis, forward_axis, up_axis], axis=1)
    
    # Target: X=Right, Y=Forward, Z=Up (Standard 3D)
    target_basis = np.eye(3)
    
    # R_fix @ bvh_basis = target_basis  =>  R_fix = target_basis @ bvh_basis.T
    R_fix = target_basis @ bvh_rotation.T
    
    return R_fix

# SMPL Joint Names
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

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    print(f"--- AUTO-ORIENT & GROUND: {os.path.basename(args.input)} ---")
    
    # 1. Load BVH
    bvh = bvh_loader.load_bvh(args.input)
    frames = bvh["frames"]
    channels = bvh["channels"]
    c_idx = bvh["channel_index"]
    offsets = bvh["offsets"]
    ft = bvh["frame_time"]
    T = frames.shape[0]
    
    # 2. Detect Correct Rotation
    print(">> Detecting Skeleton Orientation...")
    R_fix = get_bvh_orientation_correction(offsets)
    
    # 3. Detect Scale
    start = c_idx["Hips"]
    sample_pos = frames[0, start:start+3]
    scale = 0.01 if np.max(np.abs(sample_pos)) > 10.0 else 1.0
    print(f">> Detected Scale: {scale}")

    # 4. Convert & Fix Orientation
    poses = np.zeros((T, len(SMPL_H_NAMES), 3))
    trans = np.zeros((T, 3))
    
    # Matrix helpers
    C = R_fix
    C_inv = R_fix.T
    
    print(">> Processing Frames...")
    for t in range(T):
        # Root Position
        h_chs = channels["Hips"]
        px = frames[t, start + h_chs.index("Xposition")]
        py = frames[t, start + h_chs.index("Yposition")]
        pz = frames[t, start + h_chs.index("Zposition")]
        
        raw_pos = np.array([px, py, pz]) * scale
        
        # Apply Orientation Fix to Position
        trans[t] = C @ raw_pos
        
        # Root Rotation
        r_chs = [c for c in h_chs if "rotation" in c]
        order = "".join([c[0].lower() for c in r_chs])
        r_vals = [frames[t, start + h_chs.index(c)] for c in r_chs]
        r_mat = R.from_euler(order, r_vals, degrees=True).as_matrix()
        
        # Apply Orientation Fix to Rotation: R_new = C @ R_old
        poses[t, 0] = R.from_matrix(C @ r_mat).as_rotvec()
        
        # Joint Rotations
        for i, name in enumerate(SMPL_H_NAMES):
            if i == 0 or name not in c_idx: continue
            
            idx = c_idx[name]
            j_chs = channels[name]
            r_chs = [c for c in j_chs if "rotation" in c]
            if not r_chs: continue
            
            order = "".join([c[0].lower() for c in r_chs])
            r_vals = [frames[t, idx + j_chs.index(c)] for c in r_chs]
            r_mat = R.from_euler(order, r_vals, degrees=True).as_matrix()
            
            # Local Rotation Fix: C @ R @ C.T
            poses[t, i] = R.from_matrix(C @ r_mat @ C_inv).as_rotvec()

    # 5. AUTO-GROUNDING
    print(">> Calculating Floor Level...")
    
    # Find lowest Z (Up) of the Hips
    min_hip_z = np.min(trans[:, 2])
    
    # Average human leg length is ~0.90m. 
    # Force the lowest point of the HIPS to be 0.90m high.
    target_hip_height = 0.90 
    
    offset_z = target_hip_height - min_hip_z
    print(f">> Shifting Z by {offset_z:.4f}m to stand character up.")
    trans[:, 2] += offset_z

    # 6. Save
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
    print(f"✅ DONE. Saved to {args.output}")

if __name__ == "__main__":
    main()