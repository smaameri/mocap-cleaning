import numpy as np
from scipy.spatial.transform import Rotation as R

# =====================================================
# SMPL BODY FORWARD KINEMATICS (22 joints)
# FINAL – LOCKED & CONSISTENT
# =====================================================

# Joint order (index reference)
# 0  pelvis
# 1  left_hip
# 2  right_hip
# 3  spine1
# 4  left_knee
# 5  right_knee
# 6  spine2
# 7  left_ankle
# 8  right_ankle
# 9  spine3
# 10 left_foot
# 11 right_foot
# 12 neck
# 13 left_collar
# 14 right_collar
# 15 head
# 16 left_shoulder
# 17 right_shoulder
# 18 left_elbow
# 19 right_elbow
# 20 left_wrist
# 21 right_wrist

PARENTS = np.array([
    -1,     # pelvis
     0, 0,  # hips
     0,     # spine1
     1, 2,  # knees
     3,     # spine2
     4, 5,  # ankles
     6,     # spine3
     7, 8,  # feet
     9,     # neck
    12,12,  # collars
    12,     # head
    13,14,  # shoulders
    16,17,  # elbows
    18,19   # wrists
], dtype=np.int32)

# Rest pose offsets (meters, parent-local space)
REST_OFFSETS = np.array([
    [ 0.000,  0.000,  0.000],  # pelvis

    [-0.090,  0.000,  0.000],  # left_hip
    [ 0.090,  0.000,  0.000],  # right_hip

    [ 0.000,  0.100,  0.000],  # spine1

    [ 0.000, -0.420,  0.000],  # left_knee
    [ 0.000, -0.420,  0.000],  # right_knee

    [ 0.000,  0.100,  0.000],  # spine2

    [ 0.000, -0.420,  0.000],  # left_ankle
    [ 0.000, -0.420,  0.000],  # right_ankle

    [ 0.000,  0.100,  0.000],  # spine3

    [ 0.000, -0.050,  0.080],  # left_foot
    [ 0.000, -0.050,  0.080],  # right_foot

    [ 0.000,  0.120,  0.000],  # neck

    [-0.080,  0.000,  0.000],  # left_collar
    [ 0.080,  0.000,  0.000],  # right_collar

    [ 0.000,  0.100,  0.000],  # head

    [-0.150,  0.000,  0.000],  # left_shoulder
    [ 0.150,  0.000,  0.000],  # right_shoulder

    [-0.300,  0.000,  0.000],  # left_elbow
    [ 0.300,  0.000,  0.000],  # right_elbow

    [-0.250,  0.000,  0.000],  # left_wrist
    [ 0.250,  0.000,  0.000],  # right_wrist
], dtype=np.float32)


def smpl_fk(global_orient, body_pose, transl):
    """
    Forward kinematics for SMPL body joints.

    global_orient : (T, 3) axis-angle (pelvis)
    body_pose     : (T, 63) axis-angle (21 joints)
    transl        : (T, 3)
    returns       : joints (T, 22, 3)
    """

    # ------------------------
    # Sanity checks
    # ------------------------
    assert global_orient.ndim == 2 and global_orient.shape[1] == 3
    assert body_pose.ndim == 2 and body_pose.shape[1] == 63
    assert transl.ndim == 2 and transl.shape[1] == 3

    T = global_orient.shape[0]
    J = 22

    body_pose = body_pose.reshape(T, 21, 3)

    joints = np.zeros((T, J, 3), dtype=np.float32)
    rotations = np.zeros((T, J, 3, 3), dtype=np.float32)

    for t in range(T):
        # Root
        rotations[t, 0] = R.from_rotvec(global_orient[t]).as_matrix()
        joints[t, 0] = transl[t]

        # Children
        for j in range(1, J):
            parent = PARENTS[j]
            R_local = R.from_rotvec(body_pose[t, j - 1]).as_matrix()
            rotations[t, j] = rotations[t, parent] @ R_local
            joints[t, j] = joints[t, parent] + rotations[t, parent] @ REST_OFFSETS[j]

    return joints
