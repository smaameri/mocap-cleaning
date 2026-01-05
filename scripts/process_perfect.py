import numpy as np
import argparse
import sys
import os
from tqdm import tqdm
from scipy.spatial.transform import Rotation as R
import motion.bvh_loader as bvh_loader

# ==========================================
# 1. SETTINGS
# ==========================================
SMPL_H_NAMES = [
    "Hips", "LeftUpLeg", "RightUpLeg", "Spine", "LeftLeg", "RightLeg",
    "Spine1", "LeftFoot", "RightFoot", "Spine2", "LeftToeBase", "RightToeBase",
    "Neck", "LeftShoulder", "RightShoulder", "Head", "LeftArm", "RightArm",
    "LeftForeArm", "RightForeArm", "LeftHand", "RightHand",
    "LeftHandIndex1", "LeftHandIndex2", "LeftHandIndex3", "LeftHandMiddle1", "LeftHandMiddle2", "LeftHandMiddle3",
    "LeftHandPinky1", "LeftHandPinky2", "LeftHandPinky3", "LeftHandRing1", "LeftHandRing2", "LeftHandRing3",
    "LeftHandThumb1", "LeftHandThumb2", "LeftHandThumb3",
    "RightHandIndex1", "RightHandIndex2", "RightHandIndex3", "RightHandMiddle1", "RightHandMiddle2", "RightHandMiddle3",
    "RightHandPinky1", "RightHandPinky2", "RightHandPinky3", "RightHandRing1", "RightHandRing2", "RightHandRing3",
    "RightHandThumb1", "RightHandThumb2", "RightHandThumb3"
]

# ==========================================
# 2. MATH HELPERS
# ==========================================
def normalize(v):
    n = np.linalg.norm(v)
    return v / (n + 1e-8)

def get_orientation_correction(offsets):
    """Detects 'Up' axis and creates a rotation matrix to fix it."""
    def get_vec(name):
        val = offsets.get(name, None)
        return np.array(val) if val is not None else np.zeros(3)

    # 1. Detect UP (Hips -> Head/Neck)
    spine_vec = np.array([0., 1., 0.])
    for name in ["Head", "Neck", "Spine2", "Spine1", "Spine"]:
        vec = get_vec(name)
        if np.linalg.norm(vec) > 0.01: 
            spine_vec = vec
            break
            
    # 2. Detect RIGHT (Legs)
    l_leg = get_vec("LeftUpLeg")
    r_leg = get_vec("RightUpLeg")
    if np.linalg.norm(l_leg) > 0.01 and np.linalg.norm(r_leg) > 0.01:
        right_vec = r_leg - l_leg
    else:
        right_vec = np.array([1., 0., 0.])

    # 3. Orthogonalize
    up_n = normalize(spine_vec)
    right_n = normalize(right_vec - np.dot(right_vec, up_n) * up_n)
    forward_n = normalize(np.cross(up_n, right_n))
    right_n = np.cross(forward_n, up_n)
    
    # 4. Create Fix Matrix (Maps BVH -> SMPL Z-Up)
    bvh_basis = np.stack([right_n, forward_n, up_n], axis=1)
    target_basis = np.eye(3)
    return target_basis @ bvh_basis.T

# ==========================================
# 3. FK ENGINE (For Grounding Calculation)
# ==========================================
class FKNode:
    def __init__(self, name, offset, children=None):
        self.name = name
        self.offset = np.array(offset)
        self.children = children if children else []

def build_fk_tree(bvh_offsets, bvh_hierarchy):
    nodes = {}
    for name in bvh_offsets:
        nodes[name] = FKNode(name, bvh_offsets[name])
    tree_root = None
    for name, parent_name in bvh_hierarchy.items():
        if parent_name is None: tree_root = nodes[name]
        elif parent_name in nodes: nodes[parent_name].children.append(nodes[name])
    return tree_root

def run_fk_grounding(node, poses, parent_pos, parent_rot, scale):
    local_pos = node.offset * scale
    local_q = R.from_rotvec(poses[node.name]).as_matrix() if node.name in poses else np.eye(3)
    global_rot = parent_rot @ local_q
    global_pos = parent_pos + (parent_rot @ local_pos)
    positions = [global_pos[2]] # Store Z (Height)
    for child in node.children:
        positions.extend(run_fk_grounding(child, poses, global_pos, global_rot, scale))
    return positions

# ==========================================
# 4. MAIN PROCESS
# ==========================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    print(f"\n🚀 STARTING MATH PROCESS: {os.path.basename(args.input)}")
    
    # 1. LOAD
    bvh = bvh_loader.load_bvh(args.input)
    frames = bvh["frames"]
    channels = bvh["channels"]
    c_idx = bvh["channel_index"]
    offsets = bvh["offsets"]
    parents = bvh["parents"]
    T = frames.shape[0]
    
    # 2. AUTO-SCALE
    start = c_idx["Hips"]
    sample_pos = frames[0, start:start+3]
    scale = 0.01 if np.max(np.abs(sample_pos)) > 10.0 else 1.0
    print(f">> 📏 Detected Scale: {scale} (CM -> Meters)")

    # 3. AUTO-ORIENT
    print(">> 🧭 Detecting Orientation...", end=" ")
    R_fix = get_orientation_correction(offsets)
    C = R_fix
    C_inv = R_fix.T
    print("Fixed.")

    # 4. CONVERT
    poses = np.zeros((T, len(SMPL_H_NAMES), 3))
    trans = np.zeros((T, 3))
    
    for t in tqdm(range(T), colour='green', desc="Converting Data"):
        # Root Position
        h_chs = channels["Hips"]
        px = frames[t, start + h_chs.index("Xposition")]
        py = frames[t, start + h_chs.index("Yposition")]
        pz = frames[t, start + h_chs.index("Zposition")]
        trans[t] = C @ (np.array([px, py, pz]) * scale)
        
        # Root Rotation
        r_chs = [c for c in h_chs if "rotation" in c]
        order = "".join([c[0].lower() for c in r_chs])
        r_vals = [frames[t, start + h_chs.index(c)] for c in r_chs]
        r_mat = R.from_euler(order, r_vals, degrees=True).as_matrix()
        poses[t, 0] = R.from_matrix(C @ r_mat).as_rotvec()
        
        # Joints
        for i, name in enumerate(SMPL_H_NAMES):
            if i == 0 or name not in c_idx: continue
            idx = c_idx[name]
            j_chs = channels[name]
            r_chs = [c for c in j_chs if "rotation" in c]
            if not r_chs: continue
            order = "".join([c[0].lower() for c in r_chs])
            r_vals = [frames[t, idx + j_chs.index(c)] for c in r_chs]
            r_mat = R.from_euler(order, r_vals, degrees=True).as_matrix()
            poses[t, i] = R.from_matrix(C @ r_mat @ C_inv).as_rotvec()

    # 5. GROUNDING (Physics Scan)
    print("\n>> 🦶 Grounding Character...")
    fk_root = build_fk_tree(offsets, parents)
    min_z_values = []
    
    # Scan every 10th frame
    for t in tqdm(range(0, T, 10), colour='cyan', desc="Scanning Floor"):
        pose_map = {name: poses[t, i] for i, name in enumerate(SMPL_H_NAMES)}
        root_rot_mat = R.from_rotvec(poses[t, 0]).as_matrix()
        all_z = run_fk_grounding(fk_root, pose_map, trans[t], root_rot_mat, scale)
        min_z_values.append(min(all_z))

    floor_level = np.percentile(min_z_values, 1) # 1st Percentile
    shift = -floor_level
    trans[:, 2] += shift
    print(f">> ✅ Floor found at {floor_level:.4f}m. Shifted up by {shift:.4f}m.")

    # 6. SAVE
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    np.savez(args.output,
             poses=poses,
             trans=trans,
             transl=trans,
             global_orient=poses[:, 0:1],
             body_pose=poses[:, 1:22].reshape(T, -1),
             joint_names=SMPL_H_NAMES,
             fps=1.0/bvh["frame_time"]
    )
    print(f"✅ NPZ Saved: {args.output}")

if __name__ == "__main__":
    main()