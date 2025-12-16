# motion/render_bvh_skeleton.py
import argparse
import numpy as np
import cv2
from pathlib import Path
from tqdm import tqdm
import sys
from scipy.spatial.transform import Rotation as R

# Make project root importable
FILE_DIR = Path(__file__).resolve().parent
ROOT_DIR = FILE_DIR.parent
sys.path.insert(0, str(ROOT_DIR))

from motion.bvh_loader import load_bvh


def build_parent_index_array(joint_names, parents_dict):
    """Convert joint parent dict → numeric parent array."""
    name_to_idx = {name: i for i, name in enumerate(joint_names)}
    parents = []
    for name in joint_names:
        parent = parents_dict.get(name, None)
        if parent is None:
            parents.append(-1)
        else:
            if parent not in name_to_idx:
                raise KeyError(f"Parent '{parent}' for joint '{name}' not found")
            parents.append(name_to_idx[parent])
    return np.array(parents, dtype=int)


def fk_compute_positions(bvh):
    """Compute FK world positions for all joints in all frames."""
    joint_names = bvh["joint_names"]
    parents_dict = bvh["parents"]
    offsets = np.array(bvh["offsets"], float)
    frames = np.array(bvh["frames"], float)
    channel_index = bvh["channel_index"]

    T = int(bvh["n_frames"])
    J = len(joint_names)

    parents = build_parent_index_array(joint_names, parents_dict)
    positions = np.zeros((T, J, 3), float)
    rotations = np.zeros((T, J, 3, 3), float)

    # Root channel indices
    root = joint_names[0]
    root_start = channel_index[root]
    root_pos_idx = np.arange(root_start, root_start + 3)
    root_rot_idx = np.arange(root_start + 3, root_start + 6)

    C = frames.shape[1]
    if root_rot_idx[-1] >= C:
        raise IndexError("BVH channels inconsistent with root rotation count")

    # Precompute rotation indices for each joint
    rot_indices = []
    for name in joint_names:
        start = channel_index[name]
        idx = np.array([start, start + 1, start + 2])
        idx = np.clip(idx, 0, C - 1)
        rot_indices.append(idx)

    # FK per frame
    for t in range(T):
        f = frames[t]

        # root transform
        root_pos = f[root_pos_idx]
        root_euler = f[root_rot_idx]
        R_root = R.from_euler("ZXY", root_euler, degrees=True).as_matrix()

        positions[t, 0] = root_pos
        rotations[t, 0] = R_root

        # children
        for j in range(1, J):
            parent = parents[j]

            parent_rot = rotations[t, parent] if parent >= 0 else np.eye(3)
            parent_pos = positions[t, parent] if parent >= 0 else np.zeros(3)

            local_euler = f[rot_indices[j]]
            R_local = R.from_euler("ZXY", local_euler, degrees=True).as_matrix()
            R_glob = parent_rot @ R_local

            rotations[t, j] = R_glob
            positions[t, j] = parent_pos + parent_rot @ offsets[j]

    return positions, parents


def render_bvh(bvh_path, out_path, fps=30):
    bvh_path = Path(bvh_path)
    if not bvh_path.exists():
        raise FileNotFoundError(bvh_path)

    print("Loading BVH:", bvh_path)
    bvh = load_bvh(str(bvh_path))

    print("Computing FK...")
    joints, parents = fk_compute_positions(bvh)
    T, J, _ = joints.shape

    H = W = 720
    out_path = Path(out_path)
    out_path.parent.mkdir(exist_ok=True, parents=True)

    writer = cv2.VideoWriter(
        str(out_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (W, H),
    )

    # Normalize projection
    xy = joints[:, :, [0, 2]]
    mn = xy.reshape(-1, 2).min(axis=0)
    mx = xy.reshape(-1, 2).max(axis=0)
    span = max(mx - mn, 1.0)
    scale = 0.85 * W / span
    center = (mn + mx) / 2

    print("Rendering video...")
    for t in tqdm(range(T)):
        j = joints[t]
        x = (j[:, 0] - center[0]) * scale + W / 2
        y = (-(j[:, 2] - center[1])) * scale + H / 2
        pts = np.column_stack((x, y))

        frame = np.full((H, W, 3), 255, np.uint8)

        for i in range(J):
            p = parents[i]
            if p >= 0:
                cv2.line(
                    frame,
                    tuple(pts[i].astype(int)),
                    tuple(pts[p].astype(int)),
                    (0, 0, 0),
                    3,
                    cv2.LINE_AA,
                )

        for pt in pts:
            cv2.circle(frame, tuple(pt.astype(int)), 4, (0, 0, 0), -1)

        writer.write(frame)

    writer.release()
    print("Saved:", out_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("bvh_path")
    parser.add_argument("--out", default="output/raw_bvh.mp4")
    parser.add_argument("--fps", type=int, default=30)
    args = parser.parse_args()

    render_bvh(args.bvh_path, args.out, args.fps)
