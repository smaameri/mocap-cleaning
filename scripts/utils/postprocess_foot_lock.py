import numpy as np
from motion.smpl_fk import smpl_fk

# -------------------------
# Foot joint indices (FK)
# -------------------------
LEFT_FOOT  = 7   # left_ankle
RIGHT_FOOT = 8   # right_ankle

# Thresholds
FOOT_VEL_EPS = 1e-3   # movement threshold to consider foot planted

def apply_foot_lock(global_orient, body_pose, transl):
    """
    Apply deterministic foot locking by adjusting translation only.

    global_orient: (T, 3)
    body_pose:     (T, 63)
    transl:        (T, 3)

    returns:
        locked_transl: (T, 3)
    """

    T = transl.shape[0]

    # Compute FK joints
    joints = smpl_fk(global_orient, body_pose, transl)  # (T, 22, 3)

    lf = joints[:, LEFT_FOOT]
    rf = joints[:, RIGHT_FOOT]

    locked_transl = transl.copy()

    # Foot velocities
    lf_vel = np.linalg.norm(np.diff(lf, axis=0), axis=1)
    rf_vel = np.linalg.norm(np.diff(rf, axis=0), axis=1)

    # Pad to length T
    lf_vel = np.concatenate([[lf_vel[0]], lf_vel])
    rf_vel = np.concatenate([[rf_vel[0]], rf_vel])

    # Ground reference heights
    lf_ground_y = lf[0, 1]
    rf_ground_y = rf[0, 1]

    for t in range(T):
        lf_planted = lf_vel[t] < FOOT_VEL_EPS
        rf_planted = rf_vel[t] < FOOT_VEL_EPS

        if lf_planted and not rf_planted:
            delta = lf[t, 1] - lf_ground_y
            locked_transl[t, 1] -= delta

        elif rf_planted and not lf_planted:
            delta = rf[t, 1] - rf_ground_y
            locked_transl[t, 1] -= delta

        elif lf_planted and rf_planted:
            delta = min(
                lf[t, 1] - lf_ground_y,
                rf[t, 1] - rf_ground_y
            )
            locked_transl[t, 1] -= delta

        # update reference when foot lifts
        if not lf_planted:
            lf_ground_y = lf[t, 1]
        if not rf_planted:
            rf_ground_y = rf[t, 1]

    return locked_transl
