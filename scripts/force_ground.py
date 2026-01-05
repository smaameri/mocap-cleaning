import numpy as np
import argparse
import sys
import os

# Import your existing FK tools to calculate true foot positions
from motion.smpl_fk import smpl_fk

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Path to the flying .npz file")
    parser.add_argument("--output", required=True, help="Path to save the grounded .npz file")
    args = parser.parse_args()

    print(f"--- FORCING GROUND CONTACT: {os.path.basename(args.input)} ---")
    
    # 1. Load the Flying Data
    data = np.load(args.input)
    poses = data['poses']         # (T, 52, 3)
    trans = data['trans']         # (T, 3) - This is what we need to fix
    
    # Handle key naming differences (transl vs trans)
    if 'transl' in data:
        trans = data['transl']
    
    # 2. Compute the True 3D Skeleton (Forward Kinematics)
    # We need to know where the feet ARE right now to know how much to shift them.
    print(">> Computing 3D Skeleton to find feet...")
    
    # Flatten poses for FK (T, 156)
    T = poses.shape[0]
    body_pose_flat = poses[:, 1:22].reshape(T, -1)
    global_orient = poses[:, 0]
    
    # Get all joint positions
    joints = smpl_fk(global_orient, body_pose_flat, trans) # Shape: (T, 22, 3)
    
    # 3. Find the Floor
    # SMPL Joint 10 = Left Foot, 11 = Right Foot
    # We check the Y-height (index 1) of these joints
    left_foot_y = joints[:, 10, 1]
    right_foot_y = joints[:, 11, 1]
    
    # Find the lowest point EITHER foot ever reaches in the entire animation
    min_L = np.min(left_foot_y)
    min_R = np.min(right_foot_y)
    global_min_y = min(min_L, min_R)
    
    print(f">> Current Lowest Foot Height: {global_min_y:.4f} m")
    
    if global_min_y > 0.05:
        print(">> Detected FLYING character.")
    elif global_min_y < -0.05:
        print(">> Detected UNDERGROUND character.")
    else:
        print(">> Character is already near ground.")

    # 4. Apply the Shift
    shift_amount = -global_min_y
    print(f">> Applying Shift: {shift_amount:.4f} m")
    
    # We apply this shift directly to the Root Translation Y (index 1)
    trans[:, 1] += shift_amount
    
    # 5. Save Fixed File
    print(f">> Saving to {args.output}")
    np.savez(args.output,
             poses=poses,
             trans=trans,       # Updated
             transl=trans,      # Updated
             global_orient=global_orient,
             body_pose=body_pose_flat,
             joint_names=data['joint_names'],
             fps=data['fps']
    )
    print("✅ DONE. Character is now grounded.")

if __name__ == "__main__":
    main()