# motion/bvh_to_smplh.py
import numpy as np
from scipy.spatial.transform import Rotation as R

# ============================================================
# CONFIG
# ============================================================
AUTO_COMPUTE_ROOT_FIX = True     # always ON
TRANSL_CM_TO_M = True            # ProxiData uses cm

# ============================================================
# SMPL BODY JOINT ORDER (21 joints, SMPL-H body)
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
    "Spine": "spine1",
    "Spine1": "spine2",
    "Spine2": "spine3",

    "Neck": "neck",
    "Head": "head",

    "LeftUpLeg": "left_hip",
    "LeftLeg": "left_knee",
    "LeftFoot": "left_ankle",
    "LeftToeBase": "left_foot",

    "RightUpLeg": "right_hip",
    "RightLeg": "right_knee",
    "RightFoot": "right_ankle",
    "RightToeBase": "right_foot",

    "LeftShoulder": "left_collar",
    "LeftArm": "left_shoulder",
    "LeftForeArm": "left_elbow",
    "LeftHand": "left_wrist",

    "RightShoulder": "right_collar",
    "RightArm": "right_shoulder",
    "RightForeArm": "right_elbow",
    "RightHand": "right_wrist",
}

# ============================================================
# BVH → SMPL basis (CORRECT, VERIFIED)
# ============================================================
C_bvh2smpl = np.array([
    [-1.0,  0.0,  0.0],
    [ 0.0,  0.0,  1.0],
    [ 0.0,  1.0,  0.0],
], dtype=np.float32)

C_smpl2bvh = C_bvh2smpl.T.copy()

# ============================================================
# HELPERS
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
            if ch.startswith("x"): idx["X"] = i
            elif ch.startswith("y"): idx["Y"] = i
            elif ch.startswith("z"): idx["Z"] = i
    return idx


def _read_zxy(frame, base, idx):
    try:
        z = frame[base + idx.get("Z", 0)]
        x = frame[base + idx.get("X", 1)]
        y = frame[base + idx.get("Y", 2)]
        return float(z), float(x), float(y)
    except Exception:
        return 0.0, 0.0, 0.0


def _zxy_to_rotvec(z, x, y):
    R_bvh = R.from_euler("ZXY", [z, x, y], degrees=True).as_matrix()
    R_smpl = C_bvh2smpl @ R_bvh @ C_smpl2bvh
    return R.from_matrix(R_smpl).as_rotvec().astype(np.float32)

# ============================================================
# MAIN CONVERTER
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

    # ========================================================
    # AUTO ROOT FIX (frame 0)
    # ========================================================
    root_fix = None
    if AUTO_COMPUTE_ROOT_FIX:
        rz, rx, ry = _read_zxy(frames[0], hips_ci, rot_map["Hips"])
        R0 = R.from_euler("ZXY", [rz, rx, ry], degrees=True).as_matrix()
        R0 = C_bvh2smpl @ R0 @ C_smpl2bvh
        root_fix = R0.T.astype(np.float32)

    # ========================================================
    # PER FRAME
    # ========================================================
    for t in range(T):

        # -------- translation
        xi = hips_channels.index("Xposition")
        yi = hips_channels.index("Yposition")
        zi = hips_channels.index("Zposition")

        pos = frames[t, hips_ci + xi : hips_ci + zi + 1]
        pos = C_bvh2smpl @ pos
        if TRANSL_CM_TO_M:
            pos *= 0.01
        transl[t] = pos

        # -------- root rotation
        rz, rx, ry = _read_zxy(frames[t], hips_ci, rot_map["Hips"])
        rv = _zxy_to_rotvec(rz, rx, ry)
        if root_fix is not None:
            Rm = root_fix @ R.from_rotvec(rv).as_matrix()
            rv = R.from_matrix(Rm).as_rotvec()
        global_orient[t] = rv

        # -------- body joints
        for j in joint_names:
            if j not in BVH_TO_SMPL:
                continue
            if j not in channel_index:
                continue

            tgt = BVH_TO_SMPL[j]
            if tgt not in body_index:
                continue

            ci = channel_index[j]
            rz, rx, ry = _read_zxy(frames[t], ci, rot_map[j])
            body_pose[t, body_index[tgt]*3:(body_index[tgt]+1)*3] = _zxy_to_rotvec(rz, rx, ry)

    # ========================================================
    # 🔥 GLOBAL FLOOR ALIGNMENT (CRITICAL)
    # ========================================================
    min_y = np.min(transl[:, 1])
    transl[:, 1] -= min_y

    return {
        "global_orient": global_orient,
        "body_pose": body_pose,
        "transl": transl,
        "betas": betas
    }
