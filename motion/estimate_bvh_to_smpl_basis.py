# motion/estimate_bvh_to_smpl_basis.py
import numpy as np
import sys
from motion import bvh_loader
from scipy.spatial.transform import Rotation as R

def norm(v):
    v = np.asarray(v, dtype=np.float64)
    n = np.linalg.norm(v)
    return v / (n if n>1e-12 else 1.0)

def build_basis_from_offsets(offsets, hips_name="Hips"):
    # Try to find good named joints
    # Candidate joints: Spine, Spine1, LeftUpLeg, RightUpLeg
    def get(name):
        return np.asarray(offsets.get(name, None), dtype=np.float64) if name in offsets else None

    # prefer Spine1 if available, otherwise Spine
    spine = get("Spine1") if "Spine1" in offsets else get("Spine")
    left_up = get("LeftUpLeg")
    right_up = get("RightUpLeg")
    hips = get(hips_name)

    if hips is None:
        raise RuntimeError("Hips offset not found in BVH offsets.")

    # compute vectors Hips -> Spine1 and Hips -> LeftUpLeg / RightUpLeg
    up_vec = None
    if spine is not None:
        up_vec = spine - hips
    else:
        # fallback: average of neck/head direction
        for c in ("Neck","Head"):
            cand = get(c)
            if cand is not None:
                up_vec = cand - hips
                break
    if up_vec is None:
        raise RuntimeError("Could not determine BVH up vector (missing Spine/Neck offsets).")

    if left_up is not None and right_up is not None:
        # lateral axis from left to right (right - left)
        right_vec = right_up - left_up
    elif left_up is not None:
        right_vec = left_up - hips
    elif right_up is not None:
        right_vec = right_up - hips
    else:
        raise RuntimeError("Could not determine BVH lateral vector (missing LeftUpLeg/RightUpLeg).")

    # orthonormalize: right, forward, up (BVH local)
    up_n = norm(up_vec)
    right_n = norm(right_vec - np.dot(right_vec, up_n) * up_n)  # make perpendicular to up
    forward_n = np.cross(up_n, right_n)
    forward_n = norm(forward_n)

    # ensure right-handed basis: recompute right = cross(forward, up)
    right_n = np.cross(forward_n, up_n)
    right_n = norm(right_n)

    BVH_basis = np.stack([right_n, forward_n, up_n], axis=1)  # columns are basis vectors

    return BVH_basis

def main(path):
    print("Loading BVH:", path)
    bvh = bvh_loader.load_bvh(path)
    offsets_od = bvh.get("offsets", None)
    if offsets_od is None:
        raise RuntimeError("BVH has no 'offsets' key.")
    # offsets is an OrderedDict mapping joint->(x,y,z)
    offsets = {k: np.array(v, dtype=np.float64) for k,v in offsets_od.items()}

    BVH_basis = build_basis_from_offsets(offsets)
    print("\nBVH_basis (columns = right, forward, up):\n", np.round(BVH_basis,6))
    # SMPL canonical basis: X-right, Y-forward, Z-up => identity columns
    SMPL_basis = np.eye(3)

    # We want R such that: SMPL_basis = R @ BVH_basis  => R = SMPL_basis @ BVH_basis.T @ inv(BVH_basis @ BVH_basis.T)
    # but BVH_basis is orthonormal, so R = SMPL_basis @ BVH_basis.T = BVH_basis.T
    R_basis = SMPL_basis @ BVH_basis.T
    # ensure orthonormal by polar decomposition
    U, s, Vt = np.linalg.svd(R_basis)
    R_orth = U @ Vt

    print("\nEstimated rotation matrix (map BVH_coords -> SMPL_coords):\n", np.round(R_orth,6))
    r = R.from_matrix(R_orth)
    euler = r.as_euler('xyz', degrees=True)
    print("\nEuler XYZ (degrees) of rotation (apply as R.from_euler('xyz', euler, degrees=True)):\n", np.round(euler,4))

    # provide numpy-ready code line
    print("\nPaste this into your bvh_to_smplh.py (replace existing C_bvh2smpl):")
    print("C_bvh2smpl = np.array(")
    print(np.array2string(R_orth, separator=", "))
    print(")")
    print("\nAnd ensure C_smpl2bvh = C_bvh2smpl.T\n")

if __name__ == "__main__":
    main(sys.argv[1])
