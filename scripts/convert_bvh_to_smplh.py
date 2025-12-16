# scripts/convert_bvh_to_smplh.py

import sys
import os
import numpy as np

from motion import bvh_loader
from motion.bvh_to_smplh import bvh_to_smplh


def main(argv):
    if len(argv) < 3:
        print("Usage:")
        print("  python -m scripts.convert_bvh_to_smplh input.bvh output.npz")
        return 1

    bvh_path = argv[1]
    out_npz = argv[2]

    print("Loading BVH:", bvh_path)
    bvh = bvh_loader.load_bvh(bvh_path)

    print("Converting BVH → SMPL-H parameters (no mesh, no hands)...")
    smplh = bvh_to_smplh(bvh)   # 🔥 NO EXTRA ARGUMENTS

    os.makedirs(os.path.dirname(out_npz), exist_ok=True)

    print("Saving NPZ...")
    np.savez_compressed(
        out_npz,
        global_orient=smplh["global_orient"],
        body_pose=smplh["body_pose"],
        transl=smplh["transl"],
        betas=smplh["betas"],
    )

    print("Saved →", out_npz)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
