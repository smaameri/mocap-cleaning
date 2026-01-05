import numpy as np
from scipy.spatial.transform import Rotation as R

# ============================================================
# CONFIG
# ============================================================
AUTO_COMPUTE_ROOT_FIX = True
TRANSL_CM_TO_M = True

# ============================================================
# SMPL-H BODY JOINT ORDER (21 joints, no hands)
# ============================================================
SMPL_BODY_21 = [
    "left_hip", "right_hip", "spine1",
    "left_knee", "right_knee", "spine2",
    "left_ankle", "right_ankle", "spine3",
    "left_foot", "right_foot", "neck",
    "left_collar", "right_collar", "head",
    "left_shoulder", "right_shoulder",
    "left_elbow", "right_elbow",
    "left_wrist", "right_wrist"
]

# ============================================================
# BVH → SMPL joint mapping (ProxiData)
# ============================================================
BVH_TO_SMPL = {
    "LeftUpLeg": "left_hip",
    "RightUpLeg": "right_hip",

    "Spine": "spine1",
    "Spine1": "spine2",
    "Spine2": "spine3",

    "LeftLeg": "left_knee",
    "RightLeg": "right_knee",

    "LeftFoot": "left_ankle",
    "RightFoot": "right_ankle",

    "LeftToeBase": "left_foot",
    "RightToeBase": "right_foot",

    "Neck": "neck",
    "Head": "head",

    "LeftShoulder": "left_collar",
    "RightShoulder": "right_collar",

    "LeftArm": "left_shoulder",
    "RightArm": "right_shoulder",

    "LeftForeArm": "left_elbow",
    "RightForeArm": "right_elbow",

    "LeftHand": "left_wrist",
    "RightHand": "right_wrist",
}

# ============================================================
# BVH → SMPL basis (VERIFIED)
# ============================================================
C_bvh2smpl = np.array([
    [-1.0,  0.0,  0.0],
    [ 0.0,  0.0,  1.0],
    [ 0.0,  1.0,  0.0],
], dtype=np.float32)

C_smpl2bvh = C_bvh2smpl.T.copy()

# ============================================================
# Helpers
# ============================================================
def _safe_channels(channels, name):
    return list(channels.get(name, []))


def _rotation_index_map(ch_list):
    if not ch_list:
        return None
    idx = {}
    for i, ch in enumerate(ch_list):
        ch = ch.lower()
        if "rotation" in ch:
            if ch.startswith("x"):
                idx["X"] = i
            elif ch.startswith("y"):
                idx["Y"] = i
            elif ch.startswith("z"):
                idx["Z"] = i
    return idx


def _read_zxy(frame, base, idx):
    if idx is None:
        return 0.0, 0.0, 0.0
    z = frame[base + idx.get("Z", 0)]
    x = frame[base + idx.get("X", 1)]
    y = frame[base + idx.get("Y", 2)]
    return float(z), float(x), float(y)


def _zxy_to_rotmat(z, x, y):
    R_bvh = R.from_euler("ZXY", [z, x, y], degrees=True).as_matrix()
    return C_bvh2smpl @ R_bvh @ C_smpl2bvh


# ============================================================
# MAIN
# ============================================================
def bvh_to_smplh(bvh):

    joint_names = bvh["joint_names"]
    channel_index = bvh["channel_index"]
    channels = bvh["channels"]
    frames = np.asarray(bvh["frames"], dtype=np.float32)

    T = frames.shape[0]

    global_orient = np.zeros((T, 3), dtype=np.float32)
    body_pose = np.zeros((T, 21 * 3), dtype=np.float32)
    transl = np.zeros((T, 3), dtype=np.float32)
    betas = np.zeros((16,), dtype=np.float32)

    body_index = {n: i for i, n in enumerate(SMPL_BODY_21)}
    rot_map = {j: _rotation_index_map(_safe_channels(channels, j)) for j in joint_names}

    if "Hips" not in channel_index:
        raise RuntimeError("BVH missing Hips joint")

    hips_ci = channel_index["Hips"]
    hips_channels = _safe_channels(channels, "Hips")

    # --------------------------------------------------------
    # AUTO ROOT FIX (frame 0, SMPL space)
    # --------------------------------------------------------
    root_fix = None
    if AUTO_COMPUTE_ROOT_FIX:
        rz, rx, ry = _read_zxy(frames[0], hips_ci, rot_map["Hips"])
        R0 = _zxy_to_rotmat(rz, rx, ry)
        root_fix = R.from_matrix(R0).inv().as_matrix().astype(np.float32)

    # --------------------------------------------------------
    # Per frame
    # --------------------------------------------------------
    for t in range(T):

        # ---------- Translation
        xi = hips_channels.index("Xposition")
        yi = hips_channels.index("Yposition")
        zi = hips_channels.index("Zposition")

        pos = frames[t, hips_ci + xi: hips_ci + zi + 1]
        pos = C_bvh2smpl @ pos
        if TRANSL_CM_TO_M:
            pos *= 0.01
        transl[t] = pos

        # ---------- Root rotation
        rz, rx, ry = _read_zxy(frames[t], hips_ci, rot_map["Hips"])
        R_root = _zxy_to_rotmat(rz, rx, ry)
        if root_fix is not None:
            R_root = root_fix @ R_root
        global_orient[t] = R.from_matrix(R_root).as_rotvec()

        # ---------- Body joints
        for bvh_joint, smpl_joint in BVH_TO_SMPL.items():
            if bvh_joint not in channel_index:
                continue

            smpl_idx = body_index[smpl_joint]
            ci = channel_index[bvh_joint]

            rz, rx, ry = _read_zxy(frames[t], ci, rot_map[bvh_joint])
            Rj = _zxy_to_rotmat(rz, rx, ry)

            if root_fix is not None:
                Rj = root_fix @ Rj

            body_pose[t, smpl_idx * 3:(smpl_idx + 1) * 3] = \
                R.from_matrix(Rj).as_rotvec()

    # --------------------------------------------------------
    # Floor alignment (Y-up)
    # --------------------------------------------------------
    min_y = np.min(transl[:, 1])
    transl[:, 1] -= min_y

    return {
        "global_orient": global_orient,
        "body_pose": body_pose,
        "transl": transl,
        "betas": betas
    }
