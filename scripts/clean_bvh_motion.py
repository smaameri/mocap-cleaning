# scripts/clean_bvh_motion.py
"""
Simple "B" cleaning pass for SMPL-H npz files.
- orientation normalization
- root height stabilization (floor)
- planar drift removal
- approximate foot-lock by freezing root XZ during low-motion frames
- lightweight smoothing
Saves output as <stem>_cleaned.npz in same folder (keeps other arrays).
"""
import sys
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation as R

# CONFIG - tweak these
VERTICAL_LOW_PASS_ALPHA = 0.08   # lower -> stronger smoothing of height
POSE_LOW_PASS_ALPHA = 0.20       # smoothing for global_orient rotvecs
DRIFT_REMOVE = True
CONTACT_VEL_THRESH = 0.01        # m/frame (planar) => if planar vel < this consider foot contact
CONTACT_VERT_VEL_THRESH = 0.008  # m/frame vertical movement threshold
MIN_CONTACT_DURATION = 3         # frames: require this many consecutive contact frames for lock
FLOOR_PERCENTILE = 2.0           # percentile below which is considered floor baseline (robust)

def load_npz(p):
    d = np.load(p, allow_pickle=True)
    return {k: d[k] for k in d.files}

def save_npz(p, data):
    np.savez_compressed(p, **data)

def lowpass_1d(x, alpha):
    out = np.empty_like(x)
    out[0] = x[0]
    for i in range(1, len(x)):
        out[i] = alpha * x[i] + (1.0 - alpha) * out[i-1]
    return out

def lowpass_rotvecs(rv, alpha):
    # rv: (T,3) or (T,N,3) -> apply smoothing on rotvec components independently (simple)
    a = np.array(rv, dtype=np.float64)
    if a.ndim == 2:
        out = np.empty_like(a)
        out[0] = a[0]
        for t in range(1, a.shape[0]):
            out[t] = alpha * a[t] + (1 - alpha) * out[t-1]
        return out
    else:
        out = np.empty_like(a)
        out[0] = a[0]
        for t in range(1, a.shape[0]):
            out[t] = alpha * a[t] + (1 - alpha) * out[t-1]
        return out

def remove_planar_linear_trend(xz):
    # xz: (T,2) - subtract linear fit (trend) from each column
    T = xz.shape[0]
    t = np.arange(T).astype(np.float64)
    out = xz.copy().astype(np.float64)
    for i in range(2):
        p = np.polyfit(t, xz[:, i], 1)  # degree 1
        trend = np.polyval(p, t)
        out[:, i] = xz[:, i] - trend
    return out

def detect_contacts(transl):
    # approximate contact when planar speed small and vertical speed small
    # transl: (T,3) (x,y,z) where y is vertical in this dataset
    T = transl.shape[0]
    vel = np.vstack([np.zeros((1,3)), transl[1:] - transl[:-1]])  # per-frame displacement
    planar_speed = np.linalg.norm(vel[:, [0,2]], axis=1)
    vert_speed = np.abs(vel[:, 1])
    cand = (planar_speed < CONTACT_VEL_THRESH) & (vert_speed < CONTACT_VERT_VEL_THRESH)

    # require minimum consecutive frames to avoid spurious single-frame locks
    mask = np.zeros_like(cand, dtype=bool)
    i = 0
    while i < T:
        if cand[i]:
            j = i
            while j < T and cand[j]:
                j += 1
            length = j - i
            if length >= MIN_CONTACT_DURATION:
                mask[i:j] = True
            i = j
        else:
            i += 1
    return mask

def process_single(npz_path: Path, out_path: Path):
    d = load_npz(str(npz_path))
    # required: transl, global_orient
    transl = d.get("transl", None)
    if transl is None:
        # fallback: nothing to do
        print(f"[WARN] {npz_path.name} missing 'transl' -> skipping (saving copy).")
        save_npz(str(out_path), d)
        return

    T = int(transl.shape[0])

    # --- 1) Orientation normalization ---
    global_orient = d.get("global_orient", None)  # axis-angle (rotvec) (T,3)
    if global_orient is None:
        # nothing to do for orientation
        global_orient = np.zeros((T,3), dtype=np.float32)

    # compute root-fix matrix to align first-frame root to identity (upright)
    try:
        R0 = R.from_rotvec(global_orient[0]).as_matrix()  # 3x3
        root_fix = R0.T  # inverse of first-frame root
    except Exception:
        root_fix = np.eye(3)

    # apply root_fix to every frame
    rots = []
    for t in range(T):
        R_t = R.from_rotvec(global_orient[t]).as_matrix()
        R_fixed = root_fix @ R_t
        rots.append(R_fixed)
    rots = np.stack(rots, axis=0)
    # convert back to rotvecs
    global_orient_fixed = R.from_matrix(rots).as_rotvec()

    # light smoothing on rotvecs
    global_orient_fixed = lowpass_rotvecs(global_orient_fixed, POSE_LOW_PASS_ALPHA)

    # --- 2) Root height stabilization (vertical) ---
    # assume transl is (T,3) with Y vertical (dataset uses Y as up)
    transl = transl.astype(np.float64)
    vert = transl[:, 1].copy()

    # robust baseline: low percentile as floor
    baseline = np.percentile(vert, FLOOR_PERCENTILE)
    # shift so baseline -> 0 (floor)
    vert_shifted = vert - baseline

    # low-pass filter to remove jitter
    vert_smooth = lowpass_1d(vert_shifted, VERTICAL_LOW_PASS_ALPHA)

    # ensure min >= 0 (no penetration)
    vert_smooth = np.maximum(vert_smooth, 0.0)

    # replace vertical channel
    transl[:, 1] = vert_smooth

    # --- 3) Drift correction (X,Z) ---
    xz = transl[:, [0,2]]
    if DRIFT_REMOVE:
        xz_corr = remove_planar_linear_trend(xz)
        transl[:, 0] = xz_corr[:, 0]
        transl[:, 2] = xz_corr[:, 1]

    # --- 4) Approximate foot locking by freezing root XZ on contact frames ---
    contacts = detect_contacts(transl)
    # We'll freeze XZ at previous non-contact position during contact windows
    out_x = transl[:, 0].copy()
    out_z = transl[:, 2].copy()

    t = 0
    while t < T:
        if contacts[t]:
            # find contact segment
            j = t
            while j < T and contacts[j]:
                j += 1
            # freeze XZ to the start-of-contact pose (or last free pose if exists)
            ref_idx = t - 1 if (t - 1) >= 0 else t
            ref_x = out_x[ref_idx]
            ref_z = out_z[ref_idx]
            out_x[t:j] = ref_x
            out_z[t:j] = ref_z
            t = j
        else:
            t += 1
    transl[:, 0] = out_x
    transl[:, 2] = out_z

    # light smoothing for transl as well
    transl[:, 0] = lowpass_1d(transl[:, 0], 0.12)
    transl[:, 1] = lowpass_1d(transl[:, 1], 0.08)
    transl[:, 2] = lowpass_1d(transl[:, 2], 0.12)

    # --- 5) Rebuild output dict and save ---
    out = dict(d)  # copy all arrays
    out["transl"] = transl.astype(np.float32)
    out["global_orient"] = global_orient_fixed.astype(np.float32)

    # keep other arrays unchanged (body_pose, hands, betas)
    stem = npz_path.stem
    save_npz(str(out_path), out)
    print(f"Saved cleaned -> {out_path.name}")

def main(argv):
    if len(argv) < 2:
        print("Usage: python -m scripts.clean_bvh_motion path/to/file.npz")
        print("Or: python -m scripts.clean_bvh_motion --dir path/to/folder")
        return 1

    if argv[1] == "--dir":
        folder = Path(argv[2])
        files = sorted(folder.glob("*.npz"))
        for f in files:
            out = f.with_name(f.stem + "_cleaned.npz")
            process_single(f, out)
    else:
        inp = Path(argv[1])
        out = inp.with_name(inp.stem + "_cleaned.npz")
        process_single(inp, out)
    return 0

if __name__ == "__main__":
    sys.exit(main(sys.argv))
