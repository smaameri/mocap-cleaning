import numpy as np
import os
import argparse
import glob
from tqdm import tqdm
from scipy.spatial.transform import Rotation as R

# --- 1. SKELETON DEFINITION (MATCHED TO RAW BVH) ---
def get_raw_skeleton():
    # Extracted from KO_vignette_06_004_CN_AM0806.bvh
    # Total Leg Length: 42.09 + 41.06 + 6.15 = ~89.3 cm
    
    # Leg Chain
    l_foot = ("LeftToeBase", [0.0, -6.150110, 14.192602], [("End Site", [0.0, 0.0, 3.78], None)])
    l_ankle = ("LeftFoot", [0.0, -41.059803, 0.0], [l_foot])
    l_knee = ("LeftLeg", [0.0, -42.089901, 0.0], [l_ankle])
    l_hip = ("LeftUpLeg", [9.461699, 0.0, 0.0], [l_knee])

    r_foot = ("RightToeBase", [0.0, -6.150110, 14.192600], [("End Site", [0.0, 0.0, 3.78], None)])
    r_ankle = ("RightFoot", [0.0, -41.059803, 0.0], [r_foot])
    r_knee = ("RightLeg", [0.0, -42.089901, 0.0], [r_ankle])
    r_hip = ("RightUpLeg", [-9.461700, 0.0, 0.0], [r_knee])

    # Spine Chain
    head = ("Neck", [0.0, 21.913322, 0.0], [("Head", [0.0, 14.608523, 1.89], [("End Site", [0.0, 18.92, 0.0], None)])])
    spine1 = ("Spine1", [0.0, 19.804367, 0.0], [head]) 
    
    # Arms
    l_index = ("LeftHandIndex1", [4.0, 1.0, 1.0], [("LeftHandIndex2", [3.0, 0.0, 0.0], [("LeftHandIndex3", [2.0, 0.0, 0.0], [("End Site", [1.0, 0.0, 0.0], None)])])])
    l_middle = ("LeftHandMiddle1", [4.0, 0.0, 1.0], [("LeftHandMiddle2", [3.0, 0.0, 0.0], [("LeftHandMiddle3", [2.0, 0.0, 0.0], [("End Site", [1.0, 0.0, 0.0], None)])])])
    l_pinky = ("LeftHandPinky1", [3.5, -2.0, 1.0], [("LeftHandPinky2", [2.5, 0.0, 0.0], [("LeftHandPinky3", [2.0, 0.0, 0.0], [("End Site", [1.0, 0.0, 0.0], None)])])])
    l_ring = ("LeftHandRing1", [3.8, -1.0, 1.0], [("LeftHandRing2", [3.0, 0.0, 0.0], [("LeftHandRing3", [2.0, 0.0, 0.0], [("End Site", [1.0, 0.0, 0.0], None)])])])
    l_thumb = ("LeftHandThumb1", [2.0, 2.0, 2.0], [("LeftHandThumb2", [3.0, 0.0, 0.0], [("LeftHandThumb3", [2.0, 0.0, 0.0], [("End Site", [1.0, 0.0, 0.0], None)])])])

    r_index = ("RightHandIndex1", [-4.0, 1.0, 1.0], [("RightHandIndex2", [-3.0, 0.0, 0.0], [("RightHandIndex3", [-2.0, 0.0, 0.0], [("End Site", [-1.0, 0.0, 0.0], None)])])])
    r_middle = ("RightHandMiddle1", [-4.0, 0.0, 1.0], [("RightHandMiddle2", [-3.0, 0.0, 0.0], [("RightHandMiddle3", [-2.0, 0.0, 0.0], [("End Site", [-1.0, 0.0, 0.0], None)])])])
    r_pinky = ("RightHandPinky1", [-3.5, -2.0, 1.0], [("RightHandPinky2", [-2.5, 0.0, 0.0], [("RightHandPinky3", [-2.0, 0.0, 0.0], [("End Site", [-1.0, 0.0, 0.0], None)])])])
    r_ring = ("RightHandRing1", [-3.8, -1.0, 1.0], [("RightHandRing2", [-3.0, 0.0, 0.0], [("RightHandRing3", [-2.0, 0.0, 0.0], [("End Site", [-1.0, 0.0, 0.0], None)])])])
    r_thumb = ("RightHandThumb1", [-2.0, 2.0, 2.0], [("RightHandThumb2", [-3.0, 0.0, 0.0], [("RightHandThumb3", [-2.0, 0.0, 0.0], [("End Site", [-1.0, 0.0, 0.0], None)])])])

    l_arm = ("LeftShoulder", [3.783731, 15.046199, -0.660873], [("LeftArm", [12.489, 0.0, 0.0], [("LeftForeArm", [28.30, 0.0, 0.0], [("LeftHand", [24.56, 0.0, 0.0], [l_index, l_middle, l_pinky, l_ring, l_thumb])])])])
    r_arm = ("RightShoulder", [-3.783730, 15.046199, -0.660884], [("RightArm", [-12.489, 0.0, 0.0], [("RightForeArm", [-28.30, 0.0, 0.0], [("RightHand", [-24.56, 0.0, 0.0], [r_index, r_middle, r_pinky, r_ring, r_thumb])])])])

    # Structure
    spine = ("Spine", [0.0, 7.484861, 0.0], [("Spine1", [0.0, 19.804367, 0.0], [head, l_arm, r_arm])])
    root = ("Hips", [0.0, 0.0, 0.0], [l_hip, r_hip, spine])
    return root

# --- 2. HIERARCHY WRITER ---
def write_hierarchy(node, file, indent_level=0):
    name, offset, children = node
    indent = "\t" * indent_level
    if name == "End Site":
        file.write(f"{indent}End Site\n{indent}{{\n{indent}\tOFFSET {offset[0]:.6f} {offset[1]:.6f} {offset[2]:.6f}\n{indent}}}\n")
        return
    
    if indent_level == 0: file.write(f"ROOT {name}\n")
    else: file.write(f"{indent}JOINT {name}\n")
    
    file.write(f"{indent}{{\n{indent}\tOFFSET {offset[0]:.6f} {offset[1]:.6f} {offset[2]:.6f}\n")
    
    if indent_level == 0: file.write(f"{indent}\tCHANNELS 6 Xposition Yposition Zposition Zrotation Xrotation Yrotation\n")
    else: file.write(f"{indent}\tCHANNELS 3 Zrotation Xrotation Yrotation\n")
    
    if children:
        for child in children: write_hierarchy(child, file, indent_level + 1)
    file.write(f"{indent}}}\n")

# --- 3. MAPPING ---
NAME_MAPPING = {
    "Hips": "Hips",
    "LeftUpLeg": "LeftUpLeg", "LeftLeg": "LeftLeg", "LeftFoot": "LeftFoot", "LeftToeBase": "LeftToeBase",
    "RightUpLeg": "RightUpLeg", "RightLeg": "RightLeg", "RightFoot": "RightFoot", "RightToeBase": "RightToeBase",
    "Spine": "Spine", "Spine1": "Spine1", "Neck": "Neck", "Head": "Head",
    "LeftShoulder": "LeftShoulder", "LeftArm": "LeftArm", "LeftForeArm": "LeftForeArm", "LeftHand": "LeftHand",
    "RightShoulder": "RightShoulder", "RightArm": "RightArm", "RightForeArm": "RightForeArm", "RightHand": "RightHand",
    # Fingers
    "LeftHandIndex1": "LeftHandIndex1", "LeftHandIndex2": "LeftHandIndex2", "LeftHandIndex3": "LeftHandIndex3",
    "LeftHandMiddle1": "LeftHandMiddle1", "LeftHandMiddle2": "LeftHandMiddle2", "LeftHandMiddle3": "LeftHandMiddle3",
    "LeftHandPinky1": "LeftHandPinky1", "LeftHandPinky2": "LeftHandPinky2", "LeftHandPinky3": "LeftHandPinky3",
    "LeftHandRing1": "LeftHandRing1", "LeftHandRing2": "LeftHandRing2", "LeftHandRing3": "LeftHandRing3",
    "LeftHandThumb1": "LeftHandThumb1", "LeftHandThumb2": "LeftHandThumb2", "LeftHandThumb3": "LeftHandThumb3",
    "RightHandIndex1": "RightHandIndex1", "RightHandIndex2": "RightHandIndex2", "RightHandIndex3": "RightHandIndex3",
    "RightHandMiddle1": "RightHandMiddle1", "RightHandMiddle2": "RightHandMiddle2", "RightHandMiddle3": "RightHandMiddle3",
    "RightHandPinky1": "RightHandPinky1", "RightHandPinky2": "RightHandPinky2", "RightHandPinky3": "RightHandPinky3",
    "RightHandRing1": "RightHandRing1", "RightHandRing2": "RightHandRing2", "RightHandRing3": "RightHandRing3",
    "RightHandThumb1": "RightHandThumb1", "RightHandThumb2": "RightHandThumb2", "RightHandThumb3": "RightHandThumb3",
}

FLAT_ORDER = []
def collect_order(node):
    name, _, children = node
    if name != "End Site": FLAT_ORDER.append(name)
    if children:
        for child in children: collect_order(child)
collect_order(get_raw_skeleton())

def convert_single_file(npz_path, output_dir):
    try:
        data = np.load(npz_path, allow_pickle=True)
        poses = data['poses']
        trans = data['trans'] if 'trans' in data else data['translation']
        
        # --- 1. CM SCALE ---
        scale = 100.0 if np.mean(np.abs(trans)) < 10.0 else 1.0
        trans_scaled = trans * scale

        # --- 2. ROTATION (Fix Lying Down) ---
        r_fix = R.from_euler('x', -90, degrees=True)
        trans_rotated = r_fix.apply(trans_scaled)
        
        # --- 3. GROUNDING (FRAME 0 ANCHOR) ---
        # Raw Skeleton Leg Height = 89.3 cm.
        # We assume Frame 0 is roughly standing or idle.
        # We calculate height of Frame 0.
        start_height = trans_rotated[0, 1]
        
        # We force Frame 0 to be at Leg Height (89.3cm).
        # This ignores the -1.85m outliers that were causing "Flying".
        ground_offset = 89.3 - start_height
        
        trans_rotated[:, 1] += ground_offset

        num_frames = poses.shape[0]
        poses_reshaped = poses.reshape(num_frames, -1, 3)
        
        src_names = [n.decode('utf-8') if isinstance(n, bytes) else str(n) for n in data.get('joint_names', [])]
        index_map = []
        for bvh_target in FLAT_ORDER:
            target = NAME_MAPPING.get(bvh_target, bvh_target)
            if target == "Hips" and "Pelvis" in src_names: target = "Pelvis"
            
            if target in src_names:
                index_map.append(src_names.index(target))
            else:
                index_map.append(None)

        output_path = os.path.join(output_dir, os.path.basename(npz_path).replace('.npz', '.bvh'))
        with open(output_path, 'w') as f:
            f.write("HIERARCHY\n")
            write_hierarchy(get_raw_skeleton(), f)
            f.write("MOTION\nFrames: {0}\nFrame Time: 0.008333\n".format(num_frames))
            
            for i in tqdm(range(num_frames), leave=False):
                # POSITION: X, Y, Z
                line = [trans_rotated[i, 0], trans_rotated[i, 1], trans_rotated[i, 2]]
                
                # ROTATION
                for idx, src_idx in enumerate(index_map):
                    if src_idx is not None and src_idx < poses_reshaped.shape[1]:
                        try:
                            r_curr = R.from_rotvec(poses_reshaped[i, src_idx])
                            # Apply -90 X to Root Joint
                            if idx == 0: r_curr = r_fix * r_curr
                            line.extend(r_curr.as_euler('zxy', degrees=True))
                        except: line.extend([0.0, 0.0, 0.0])
                    else: line.extend([0.0, 0.0, 0.0])
                f.write(" ".join("{0:.6f}".format(val) for val in line) + "\n")
        return True
    except Exception as e:
        print(f"Error {npz_path}: {e}")
        return False

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', type=str, required=True)
    parser.add_argument('--output_dir', type=str, required=True)
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    files = glob.glob(os.path.join(args.input_dir, "*.npz"))
    for f in tqdm(files): convert_single_file(f, args.output_dir)