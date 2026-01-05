import numpy as np
import argparse
import sys
import os
from scipy.spatial.transform import Rotation as R
from tqdm import tqdm
import motion.bvh_loader as bvh_loader

# ==========================================
# 1. MATH: AUTOMATIC BASIS ESTIMATION
# ==========================================
def norm(v):
    n = np.linalg.norm(v)
    return v / (n if n > 1e-12 else 1.0)

def estimate_basis(offsets):
    """
    Analyzes skeleton offsets to find the True Up/Forward vectors.
    """
    # 1. Find Key Joints (Robust Lookup)
    def get_vec(name):
        return np.array(offsets[name]) if name in offsets else None
    
    hips = get_vec("Hips")
    if hips is None: return np.eye(3) # Fallback
    
    # UP VECTOR: Hips -> Spine/Neck
    spine = get_vec("Spine1") if "Spine1" in offsets else get_vec("Spine")
    if spine is None: spine = get_vec("Neck")
    
    if spine is not None:
        up_vec = spine - hips
    else:
        # Fallback: Check negative gravity axis? Assume Y is up
        up_vec = np.array([0, 1, 0])
        
    # RIGHT VECTOR: LeftUpLeg -> RightUpLeg
    l_leg = get_vec("LeftUpLeg")
    r_leg = get_vec("RightUpLeg")
    
    if l_leg is not None and r_leg is not None:
        right_vec = r_leg - l_leg
    else:
        right_vec = np.array([1, 0, 0]) # Fallback

    # ORTHONORMALIZE
    up_n = norm(up_vec)
    # Project right_vec to be perpendicular to Up
    right_n = norm(right_vec - np.dot(right_vec, up_n) * up_n)
    # Forward is Cross Product
    forward_n = np.cross(up_n, right_n)
    
    # Re-cross to ensure Right-Handed system
    right_n = np.cross(forward_n, up_n)
    
    # This matrix columns are [Right, Forward, Up] in BVH space
    BVH_basis = np.stack([right_n, forward_n, up_n], axis=1)
    
    # We want to map this to SMPL Basis: X=Right, Y=Up, Z=Forward (or standard Z-up)
    # Let's align to: X=Right, Y=Forward, Z=Up (Standard 3D)
    SMPL_basis = np.eye(3) 
    
    # Rotation R that aligns BVH to SMPL: R @ BVH = SMPL -> R = SMPL @ BVH.T
    R_align = SMPL_basis @ BVH_basis.T
    
    return R_align

# ==========================================
# 2. CONVERSION LOGIC
# ==========================================
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

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    
    print(f"--- SMART CONVERSION: {os.path.basename(args.input)} ---")
    
    # 1. LOAD
    bvh = bvh_loader.load_bvh(args.input)
    frames = bvh["frames"]
    channels = bvh["channels"]
    c_idx = bvh["channel_index"]
    offsets = bvh["offsets"]
    
    T = frames.shape[0]
    
    # 2. CALCULATE CORRECTION MATRIX (The "Magic" Step)
    print(">> Analyzing Skeleton Alignment...")
    R_fix = estimate_basis(offsets)
    print(f">> Correction Matrix:\n{np.round(R_fix, 3)}")
    
    # 3. DETECT SCALE
    # If hips move > 10 units, it's CM. If < 5, it's Meters.
    hips_pos_idx = c_idx["Hips"]
    sample = frames[0, hips_pos_idx:hips_pos_idx+3]
    scale = 0.01 if np.max(np.abs(sample)) > 10.0 else 1.0
    print(f">> Scale Factor: {scale}")

    # 4. CONVERT FRAMES
    print(">> Converting Motion...")
    
    poses = np.zeros((T, len(SMPL_H_NAMES), 3))
    trans = np.zeros((T, 3))
    
    # Precompute Rotation Transforms: C @ R @ C.T
    # This maps local BVH rotation into the corrected Space
    C = R_fix
    C_inv = R_fix.T
    
    for t in tqdm(range(T)):
        # --- ROOT POSITION ---
        start = c_idx["Hips"]
        # Assumes X Y Z order in channel map (standard)
        # We should parse strictly but standard BVH is usually Pos X,Y,Z
        # Let's rely on string parsing to be safe
        h_chs = channels["Hips"]
        px = frames[t, start + h_chs.index("Xposition")]
        py = frames[t, start + h_chs.index("Yposition")]
        pz = frames[t, start + h_chs.index("Zposition")]
        
        raw_pos = np.array([px, py, pz]) * scale
        trans[t] = C @ raw_pos  # Apply global alignment to position
        
        # --- ROOT ROTATION ---
        # Collect Euler angles
        r_chs = [c for c in h_chs if "rotation" in c]
        order = "".join([c[0].lower() for c in r_chs]) # e.g. "zxy"
        r_vals = [frames[t, start + h_chs.index(c)] for c in r_chs]
        
        r_mat = R.from_euler(order, r_vals, degrees=True).as_matrix()
        # Apply Alignment: New_Rot = C @ Old_Rot
        # Note: Root rotation is Global, so we just pre-multiply
        poses[t, 0] = R.from_matrix(C @ r_mat).as_rotvec()
        
        # --- JOINT ROTATIONS ---
        for i, name in enumerate(SMPL_H_NAMES):
            if i == 0 or name not in c_idx: continue
            
            start = c_idx[name]
            j_chs = channels[name]
            r_chs = [c for c in j_chs if "rotation" in c]
            if not r_chs: continue
            
            order = "".join([c[0].lower() for c in r_chs])
            r_vals = [frames[t, start + j_chs.index(c)] for c in r_chs]
            
            r_mat = R.from_euler(order, r_vals, degrees=True).as_matrix()
            
            # For child joints, the rotation is Local.
            # Transform Local Basis: R_new = C @ R_old @ C.T
            # This ensures the axis of rotation (like "Bend Knee") aligns with the new Up/Forward
            poses[t, i] = R.from_matrix(C @ r_mat @ C_inv).as_rotvec()

    # 5. ROBUST GROUNDING (Percentile)
    # Find the visual floor (lowest 1% of feet points)
    # We use the Transformed Hips to estimate floor relative to 0
    # Note: Accurately we should run FK, but simply checking the lowest point of 
    # the Hips usually correlates to "Leg Length".
    # Let's use the Hips Y Minimum.
    
    print(">> Grounding...")
    # Since we rotated everything, Z is likely UP now (Standard SMPL).
    # Let's check which axis has gravity.
    # Usually Z is up in SMPL.
    
    # We use a heuristic: The foot is roughly 85cm-95cm below the hips.
    # We find the lowest Hips Z.
    min_hip_z = np.percentile(trans[:, 2], 1)
    
    # Shift so that lowest hip is at +0.9m
    # This prevents the "flying" or "underground" issue
    floor_offset = 0.95 - min_hip_z
    trans[:, 2] += floor_offset
    
    print(f">> Auto-ground shift: {floor_offset:.3f}m")

    # 6. SAVE
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    np.savez(args.output,
             poses=poses,
             trans=trans,
             transl=trans,
             global_orient=poses[:, 0:1],
             body_pose=poses[:, 1:22].reshape(T, -1),
             joint_names=SMPL_H_NAMES,
             fps=1.0/bvh["frame_time"]
    )
    print(f"✅ DONE. Saved to {args.output}")

if __name__ == "__main__":
    main()