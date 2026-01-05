import numpy as np
import argparse
import sys
import os
from scipy.spatial.transform import Rotation as R

# =========================================================
# 1. LOCAL FK SOLVER (To measure True Height)
# =========================================================
# Standard SMPL topology parent indices
PARENTS = np.array([-1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 12, 12, 12, 13, 14, 16, 17, 18, 19], dtype=np.int32)

# Standard SMPL Rest Offsets
REST_OFFSETS = np.array([
    [ 0.000,  0.000,  0.000], [-0.090,  0.000,  0.000], [ 0.090,  0.000,  0.000],
    [ 0.000,  0.100,  0.000], [ 0.000, -0.420,  0.000], [ 0.000, -0.420,  0.000],
    [ 0.000,  0.100,  0.000], [ 0.000, -0.420,  0.000], [ 0.000, -0.420,  0.000],
    [ 0.000,  0.100,  0.000], [ 0.000, -0.050,  0.080], [ 0.000, -0.050,  0.080],
    [ 0.000,  0.120,  0.000], [-0.080,  0.000,  0.000], [ 0.080,  0.000,  0.000],
    [ 0.000,  0.100,  0.000], [-0.150,  0.000,  0.000], [ 0.150,  0.000,  0.000],
    [-0.300,  0.000,  0.000], [ 0.300,  0.000,  0.000], [-0.250,  0.000,  0.000],
    [ 0.250,  0.000,  0.000],
], dtype=np.float32)

def simple_fk(global_orient, body_pose, transl):
    """Computes joint positions without applying extra rotations."""
    T = global_orient.shape[0]
    body_pose = body_pose.reshape(T, 21, 3)
    joints = np.zeros((T, 22, 3), dtype=np.float32)
    rotations = np.zeros((T, 22, 3, 3), dtype=np.float32)

    for t in range(T):
        # Root
        rotations[t, 0] = R.from_rotvec(global_orient[t]).as_matrix()
        joints[t, 0] = transl[t]

        # Body
        for j in range(1, 22):
            parent = PARENTS[j]
            local_rot = R.from_rotvec(body_pose[t, j - 1]).as_matrix()
            rotations[t, j] = rotations[t, parent] @ local_rot
            joints[t, j] = joints[t, parent] + rotations[t, parent] @ REST_OFFSETS[j]
    return joints

# =========================================================
# 2. MAIN SCRIPT
# =========================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Path to the flying .npz file")
    parser.add_argument("--output", required=True, help="Path to save the grounded .npz file")
    args = parser.parse_args()

    print(f"--- FIXING Z-AXIS FLYING: {os.path.basename(args.input)} ---")
    
    # 1. Load Data
    data = np.load(args.input)
    # Handle both 'trans' (old) and 'transl' (new) keys
    if 'transl' in data:
        trans = data['transl'].copy()
    else:
        trans = data['trans'].copy()
        
    poses = data['poses']
    global_orient = poses[:, 0]
    body_pose = poses[:, 1:22].reshape(poses.shape[0], -1)

    # 2. Compute 3D Skeleton to find the FEET
    print(">> Calculating true 3D positions (FK)...")
    joints = simple_fk(global_orient, body_pose, trans)

    # 3. Find Lowest Z (Height)
    # In the renderer, Z is Up. We check joints 10 & 11 (Left/Right Toes/Ankles)
    # Indices 7 & 8 are Ankles, 10 & 11 are Toes in this topology
    foot_joints_z = joints[:, [7, 8, 10, 11], 2] 
    
    min_z = np.min(foot_joints_z)
    print(f">> Lowest foot height detected: {min_z:.4f} m")

    # 4. Apply Fix
    shift = -min_z
    print(f">> Applying Shift: {shift:.4f} m to Root Z-axis")
    
    trans[:, 2] += shift  # Fix Z axis

    # 5. Save
    print(f">> Saving to {args.output}")
    np.savez(args.output,
             poses=poses,
             trans=trans,       
             transl=trans,      
             global_orient=global_orient,
             body_pose=body_pose,
             joint_names=data['joint_names'],
             fps=data['fps']
    )
    print("✅ DONE. Character grounded on Z-axis.")

if __name__ == "__main__":
    main()