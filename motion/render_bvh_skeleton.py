import argparse
import numpy as np
import cv2
from pathlib import Path
from tqdm import tqdm
import sys
from scipy.spatial.transform import Rotation as R

FILE_DIR = Path(__file__).resolve().parent
ROOT_DIR = FILE_DIR.parent
sys.path.insert(0, str(ROOT_DIR))

from motion.bvh_loader import load_bvh


def numeric_parents_array(bvh):
    """Return a numpy int array of parents aligned to joint_names."""
    names = bvh["joint_names"]
    parents_dict = bvh["parents"]
    parents = []
    for name in names:
        p = parents_dict[name]
        parents.append(-1 if p is None else names.index(p))
    return np.array(parents, dtype=np.int32)


def compute_positions_from_bvh(bvh):
    """
    Robust FK for your BVH loader output.
    - Ensures channel_index values are Python ints.
    - Builds offsets array from the offsets OrderedDict in joint order.
    - Validates shapes and types with helpful error messages.
    """
    names = bvh["joint_names"]
    parents = numeric_parents_array(bvh).astype(int).tolist()

    # Offsets: loader gives OrderedDict keyed by joint name — convert to ordered array
    offsets_dict = bvh["offsets"]
    try:
        offsets = np.array([offsets_dict[n] for n in names], dtype=float)
    except Exception as e:
        raise RuntimeError(f"Failed to build offsets array from loader. offsets type: {type(offsets_dict)}") from e

    # Frames and channel index
    frames = np.array(bvh["frames"], dtype=float)
    # Ensure channel_index values are python ints (avoid numpy scalar issues)
    try:
        channel_index = {k: int(v) for k, v in bvh["channel_index"].items()}
    except Exception as e:
        raise RuntimeError("channel_index values cannot be converted to int. Dump channel_index for debugging.") from e

    T = int(bvh["n_frames"])
    J = len(names)

    # Basic sanity checks
    if offsets.shape != (J, 3):
        raise RuntimeError(f"Offsets shape mismatch: expected ({J},3), got {offsets.shape}")

    if frames.ndim != 2:
        raise RuntimeError(f"Frames must be 2D array (T, channels). Got ndim={frames.ndim}, shape={frames.shape}")

    positions = np.zeros((T, J, 3), dtype=float)
    rotations = np.zeros((T, J, 3, 3), dtype=float)

    for t in range(T):
        f = frames[t]

        # ROOT
        root_name = names[0]
        start = channel_index[root_name]
        # guard against out-of-range channel indices
        if start + 6 > f.shape[0]:
            raise IndexError(f"Root channel index out of range for frame {t}: start={start}, frame_len={f.shape[0]}")

        root_pos = f[start:start + 3]
        root_eul = f[start + 3:start + 6]
        positions[t, 0] = root_pos
        rotations[t, 0] = R.from_euler("ZXY", root_eul, degrees=True).as_matrix()

        # CHILDREN
        for j in range(1, J):
            name = names[j]
            parent = parents[j]
            start = channel_index.get(name, None)
            if start is None:
                raise KeyError(f"Channel index missing for joint '{name}'")

            if start + 3 > f.shape[0]:
                raise IndexError(f"Channel index out of range for joint '{name}' frame {t}: start={start}, frame_len={f.shape[0]}")

            eul = f[start:start + 3]
            rotations[t, j] = R.from_euler("ZXY", eul, degrees=True).as_matrix()

            # Safety checks before the matrix multiply / indexing
            if not (isinstance(parent, int) and parent >= 0 and parent < J):
                # If parent == -1 (should only be for root) we can't compute relative; this is unexpected here
                raise RuntimeError(f"Invalid parent index for joint '{name}' (idx {j}): {parent}")

            if offsets[j].ndim != 1 or offsets[j].shape[0] != 3:
                raise RuntimeError(f"Offset for joint '{name}' has invalid shape: {np.shape(offsets[j])}")

            if rotations[t, parent].shape != (3, 3):
                raise RuntimeError(f"Rotation for parent index {parent} has invalid shape: {rotations[t, parent].shape}")

            positions[t, j] = positions[t, parent] + rotations[t, parent] @ offsets[j]

    return positions, np.array(parents, dtype=int)


def render_bvh(bvh_path, out_path, fps=30):
    print("Loading BVH:", bvh_path)
    bvh = load_bvh(bvh_path)

    print("Computing FK positions...")
    joints, parents = compute_positions_from_bvh(bvh)

    T, J, _ = joints.shape
    H = W = 512
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)

    writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H))

    # Project world X,Z -> 2D
    xy = joints[:, :, [0, 2]]
    mn = xy.reshape(-1, 2).min(0)
    mx = xy.reshape(-1, 2).max(0)
    span = max(mx - mn)
    scale = 0.8 * W / span
    center = (mn + mx) / 2

    print("Rendering...")
    for t in tqdm(range(T)):
        j = joints[t]
        x = (j[:, 0] - center[0]) * scale + W / 2
        y = (-(j[:, 2] - center[1])) * scale + H / 2
        pts = np.stack([x, y], axis=1).astype(int)

        frame = 255 * np.ones((H, W, 3), np.uint8)
        for i in range(J):
            p = int(parents[i])
            if p >= 0:
                cv2.line(frame, tuple(pts[i]), tuple(pts[p]), (0, 0, 0), 2)
        for p in pts:
            cv2.circle(frame, tuple(p), 3, (0, 0, 0), -1)

        writer.write(frame)

    writer.release()
    print("Saved:", out_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("bvh_path")
    parser.add_argument("--out", default="output/raw_bvh.mp4")
    parser.add_argument("--fps", type=int, default=30)
    args = parser.parse_args()

    render_bvh(args.bvh_path, args.out, fps=args.fps)
