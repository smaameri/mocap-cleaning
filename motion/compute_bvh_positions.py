import numpy as np
from scipy.spatial.transform import Rotation as R


def compute_positions_from_bvh(bvh):
    parents = bvh["parents"]
    offsets = np.array(bvh["offsets"])
    frames = np.array(bvh["frames"])
    T = bvh["n_frames"]
    J = len(parents)

    positions = np.zeros((T, J, 3))
    rotations = np.zeros((T, J, 3, 3))

    # Root indices
    root_trans_idx = bvh["channel_index"]["Hips"]["POSITION"]
    root_rot_idx   = bvh["channel_index"]["Hips"]["ROTATION"]

    for t in range(T):
        frame = frames[t]

        # Root translation
        root_trans = frame[root_trans_idx]

        # Root rotation (dataset uses ZXY order)
        root_euler = frame[root_rot_idx]
        R_root = R.from_euler("ZXY", root_euler, degrees=True).as_matrix()

        positions[t, 0] = root_trans
        rotations[t, 0] = R_root

        # Other joints
        for j in range(1, J):
            p = parents[j]

            j_rot_idx = bvh["channel_index"][bvh["joint_names"][j]]["ROTATION"]
            euler = frame[j_rot_idx]
            R_j = R.from_euler("ZXY", euler, degrees=True).as_matrix()

            rotations[t, j] = R_j

            # forward kinematics
            positions[t, j] = positions[t, p] + rotations[t, p] @ offsets[j]

    return positions
