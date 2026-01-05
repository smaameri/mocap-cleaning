import numpy as np
from scipy.spatial.transform import Rotation as R

# =====================================================
# SMPL-H BODY FORWARD KINEMATICS (22 joints, no hands)
# =====================================================

PARENTS = np.array([
    -1,
     0, 0,
     0,
     1, 2,
     3,
     4, 5,
     6,
     7, 8,
     9,
    12,12,
    12,
    13,14,
    16,17,
    18,19
], dtype=np.int32)

REST_OFFSETS = np.array([
    [ 0.000,  0.000,  0.000],
    [-0.090,  0.000,  0.000],
    [ 0.090,  0.000,  0.000],
    [ 0.000,  0.100,  0.000],
    [ 0.000, -0.420,  0.000],
    [ 0.000, -0.420,  0.000],
    [ 0.000,  0.100,  0.000],
    [ 0.000, -0.420,  0.000],
    [ 0.000, -0.420,  0.000],
    [ 0.000,  0.100,  0.000],
    [ 0.000, -0.050,  0.080],
    [ 0.000, -0.050,  0.080],
    [ 0.000,  0.120,  0.000],
    [-0.080,  0.000,  0.000],
    [ 0.080,  0.000,  0.000],
    [ 0.000,  0.100,  0.000],
    [-0.150,  0.000,  0.000],
    [ 0.150,  0.000,  0.000],
    [-0.300,  0.000,  0.000],
    [ 0.300,  0.000,  0.000],
    [-0.250,  0.000,  0.000],
    [ 0.250,  0.000,  0.000],
], dtype=np.float32)

# ---- BVH → SMPL ROOT FIX (CRITICAL) ----
ROOT_FIX = R.from_euler("xyz", [90, 0, 180], degrees=True).as_matrix()

def smpl_fk(global_orient, body_pose, transl):
    assert global_orient.shape[1] == 3
    assert body_pose.shape[1] == 63
    assert transl.shape[1] == 3

    T = global_orient.shape[0]
    body_pose = body_pose.reshape(T, 21, 3)

    joints = np.zeros((T, 22, 3), dtype=np.float32)
    rotations = np.zeros((T, 22, 3, 3), dtype=np.float32)

    for t in range(T):
        R_root = R.from_rotvec(global_orient[t]).as_matrix()
        R_root = ROOT_FIX @ R_root   # <<< THIS WAS MISSING

        rotations[t, 0] = R_root
        joints[t, 0] = transl[t]

        for j in range(1, 22):
            parent = PARENTS[j]
            R_local = R.from_rotvec(body_pose[t, j - 1]).as_matrix()
            rotations[t, j] = rotations[t, parent] @ R_local
            joints[t, j] = joints[t, parent] + rotations[t, parent] @ REST_OFFSETS[j]

    return joints
