# scripts/postprocess_smplh_stabilize.py

import sys
from pathlib import Path
import numpy as np

from scripts.postprocess_foot_lock import apply_foot_lock

"""
Post-process SMPL-H parameter NPZ

✔ Foot-locked grounding (translation only)
✔ NO rotation smoothing
✔ NO joint clamping
✔ Deterministic and reversible
✔ FK-consistent

Assumes:
- BVH → SMPL-H rotations are already correct
- FK offsets are correct
"""

# --------------------------------------------------
# IO helpers
# --------------------------------------------------

def load_npz(path: Path):
    data = np.load(path, allow_pickle=True)
    return {k: data[k] for k in data.files}

def save_npz(path: Path, data: dict):
    np.savez_compressed(path, **data)

# --------------------------------------------------
# Core processing
# --------------------------------------------------

def process_file(inp: Path, out: Path):
    print(f"Stabilizing (foot-lock): {inp.name}")

    data = load_npz(inp)

    # Required keys
    for k in ("global_orient", "body_pose", "transl"):
        if k not in data:
            raise RuntimeError(f"Missing key '{k}' in {inp}")

    # Force stable dtypes
    global_orient = np.asarray(data["global_orient"], dtype=np.float32)
    body_pose    = np.asarray(data["body_pose"], dtype=np.float32)
    transl       = np.asarray(data["transl"], dtype=np.float32)

    # Shape sanity
    if global_orient.ndim != 2 or global_orient.shape[1] != 3:
        raise ValueError("global_orient must be (T, 3)")
    if body_pose.ndim != 2 or body_pose.shape[1] % 3 != 0:
        raise ValueError("body_pose must be (T, N*3)")
    if transl.ndim != 2 or transl.shape[1] != 3:
        raise ValueError("transl must be (T, 3)")

    # --------------------------------------------------
    # APPLY FOOT LOCK (TRANSL ONLY)
    # --------------------------------------------------
    locked_transl = apply_foot_lock(
        global_orient=global_orient,
        body_pose=body_pose,
        transl=transl
    )

    # Write back (do NOT touch rotations)
    data["global_orient"] = global_orient
    data["body_pose"]    = body_pose
    data["transl"]       = locked_transl.astype(np.float32)

    save_npz(out, data)
    print(f"Saved → {out}")

# --------------------------------------------------
# CLI
# --------------------------------------------------

def main(argv):
    if len(argv) < 2:
        print("Usage:")
        print("  python -m scripts.postprocess_smplh_stabilize file.npz")
        print("  python -m scripts.postprocess_smplh_stabilize --dir folder/")
        return 1

    if argv[1] == "--dir":
        folder = Path(argv[2])
        for f in sorted(folder.glob("*.npz")):
            if f.name.endswith("_stable.npz"):
                continue
            process_file(f, f.with_name(f.stem + "_stable.npz"))
    else:
        inp = Path(argv[1])
        out = inp.with_name(inp.stem + "_stable.npz")
        process_file(inp, out)

    return 0

if __name__ == "__main__":
    sys.exit(main(sys.argv))
