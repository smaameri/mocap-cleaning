import os
import sys
import numpy as np
import argparse
from scipy.spatial.transform import Rotation as R
import motion.bvh_loader as bvh_loader

# --- PATH FIX ---
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

# =========================================================
# HELPER CLASSES
# =========================================================
class BVHNode:
    def __init__(self, name, offset, parent=None):
        self.name = name
        self.offset = np.array(offset)
        self.parent = parent
        self.children = []

def load_skeleton_structure(path):
    """Parses the hierarchy from the BVH file text."""
    with open(path, 'r') as f: content = f.read().split()
    iterator = iter(content)
    root = None
    node_stack = []
    token = next(iterator, None)
    while token:
        if token in ["ROOT", "JOINT"]:
            name = next(iterator)
            new_node = BVHNode(name, [0,0,0], node_stack[-1] if node_stack else None)
            if node_stack: node_stack[-1].children.append(new_node)
            else: root = new_node
            node_stack.append(new_node)
        elif token == "End":
            next(iterator)
            new_node = BVHNode("EndSite", [0,0,0], node_stack[-1])
            node_stack[-1].children.append(new_node)
            node_stack.append(new_node)
        elif token == "OFFSET":
            node_stack[-1].offset = np.array([float(next(iterator)) for _ in range(3)])
        elif token == "}" and node_stack:
            node_stack.pop()
        elif token == "MOTION":
            break
        token = next(iterator, None)
    return root

def find_lowest_foot_y(node, frames, channel_map, start_idx, scale, parent_pos, parent_rot):
    """Recursively calculates Global Y positions to find the TRUE floor."""
    
    # 1. Get Local Rotation
    local_rot = np.eye(3)
    if node.name in channel_map:
        chs = channel_map[node.name]
        rot_cols = []
        rot_order = ""
        for c in chs:
            if "rotation" in c:
                if "X" in c: rot_order += "x"
                if "Y" in c: rot_order += "y"
                if "Z" in c: rot_order += "z"
                idx_in_row = start_idx[node.name] + chs.index(c)
                rot_cols.append(idx_in_row)
        
        if len(rot_order) == 3:
            raw_eulers = frames[:, rot_cols]
            r = R.from_euler(rot_order, raw_eulers, degrees=True)
            local_rot = r.as_matrix()

    # 2. Compute Global Rotation & Position
    if parent_rot.ndim == 3 and local_rot.ndim == 3:
        global_rot = np.matmul(parent_rot, local_rot)
    elif parent_rot.ndim == 3: 
        global_rot = parent_rot
    else: 
        global_rot = local_rot

    # Offset logic
    scaled_offset = node.offset * scale
    offset_broadcast = np.tile(scaled_offset, (len(frames), 1))
    rotated_offset = np.einsum('tij,tj->ti', parent_rot, offset_broadcast)
    global_pos = parent_pos + rotated_offset

    # 3. Check for Feet
    min_y = 9999.0
    # Enhanced keywords for martial arts data
    keywords = ["foot", "toe", "ankle", "endsite"]
    if any(x in node.name.lower() for x in keywords):
        min_y = np.min(global_pos[:, 1]) 
    
    # 4. Recurse
    for child in node.children:
        child_min = find_lowest_foot_y(child, frames, channel_map, start_idx, scale, global_pos, global_rot)
        if child_min < min_y:
            min_y = child_min
            
    return min_y

# =========================================================
# MAIN CONVERSION LOGIC
# =========================================================

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

ROOT_FIX = R.from_euler("xyz", [90, 0, 180], degrees=True).as_matrix()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    print(f"--- PROCESSING MARTIAL ARTS DATA: {os.path.basename(args.input)} ---")
    
    # 1. LOAD DATA
    bvh_data = bvh_loader.load_bvh(args.input)
    frames = bvh_data["frames"]
    channel_index = bvh_data["channel_index"]
    channels_map = bvh_data["channels"]
    ft = bvh_data["frame_time"]
    T = frames.shape[0]

    # 2. DETECT SCALE
    scale_factor = 1.0
    if "Hips" in channel_index:
        start = channel_index["Hips"]
        sample_pos = frames[0, start:start+3]
        if np.max(np.abs(sample_pos)) > 10.0:
            print(">> Detected CM units. Scaling by 0.01.")
            scale_factor = 0.01

    # 3. CALCULATE TRUE FLOOR (The Martial Arts Fix)
    print(">> Scanning full motion for lowest foot contact point...")
    skel_root = load_skeleton_structure(args.input)
    
    root_pos_dummy = np.zeros((T, 3))
    root_rot_dummy = np.repeat(np.eye(3)[np.newaxis, :, :], T, axis=0)
    
    min_foot_y_global = find_lowest_foot_y(
        skel_root, frames, channels_map, channel_index, scale_factor, 
        root_pos_dummy, root_rot_dummy
    )
    
    print(f">> Lowest foot point detected at Y = {min_foot_y_global:.4f}")

    # 4. APPLY FIX TO HIPS
    start = channel_index["Hips"]
    x_i = channels_map["Hips"].index("Xposition")
    y_i = channels_map["Hips"].index("Yposition")
    z_i = channels_map["Hips"].index("Zposition")
    
    raw_pos = np.stack([
        frames[:, start + x_i],
        frames[:, start + y_i],
        frames[:, start + z_i]
    ], axis=1) * scale_factor

    print(f">> Shifting Hips by {-min_foot_y_global:.4f} to ground the motion.")
    raw_pos[:, 1] -= min_foot_y_global

    trans = (ROOT_FIX @ raw_pos.T).T

    # 5. ROTATIONS
    J = len(SMPL_H_NAMES)
    poses = np.zeros((T, J, 3)) 
    
    rot_indices = [channels_map["Hips"].index(c) for c in channels_map["Hips"] if "rotation" in c]
    if len(rot_indices) == 3:
        order = "".join([c[0].lower() for c in channels_map["Hips"] if "rotation" in c])
        raw_rot = frames[:, start+rot_indices[0] : start+rot_indices[0]+3]
        r_hips = R.from_euler(order, raw_rot, degrees=True)
        poses[:, 0] = R.from_matrix(ROOT_FIX @ r_hips.as_matrix()).as_rotvec()

    for i, name in enumerate(SMPL_H_NAMES):
        if i == 0 or name not in channel_index: continue
        start = channel_index[name]
        ch_list = channels_map[name]
        indices = [ch_list.index(c) for c in ch_list if "rotation" in c]
        if len(indices) == 3:
            order = "".join([c[0].lower() for c in ch_list if "rotation" in c])
            raw_euler = frames[:, start+indices[0] : start+indices[0]+3]
            poses[:, i] = R.from_euler(order, raw_euler, degrees=True).as_rotvec()

    # 6. SAVE (CORRECTED SHAPES)
    output_dir = os.path.dirname(args.output)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    np.savez(args.output,
             poses=poses,
             trans=trans,
             transl=trans,           # Added for stabilizer compatibility
             global_orient=poses[:, 0], # FIXED: shape is now (T, 3)
             body_pose=poses[:, 1:22].reshape(T, -1),
             joint_names=SMPL_H_NAMES,
             fps=1.0/ft
    )
    print(f"✅ DONE. Saved {args.output}")

if __name__ == "__main__":
    main()