import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from tqdm import tqdm
import os
import sys
import argparse
from scipy.spatial.transform import Rotation as R

# SETTINGS
VIDEO_FPS = 30         
FRAME_SKIP = 1         
SCALE_OFFSETS = 0.01   

# COLORS
BG_COLOR = "#000000"
BONE_COLOR = "#FFFFFF"
JOINT_COLOR = "#FF0000"
FLOOR_COLOR = "#444444"

# --- MINI BVH LOADER ---
class BVHNode:
    def __init__(self, name, offset, parent=None):
        self.name = name
        self.offset = np.array(offset)
        self.parent = parent
        self.children = []

def load_skeleton_structure(path):
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

# --- KINEMATICS ---
def compute_fk(node, poses_map, parent_pos, parent_rot, trans_offset=None):
    local_rot = np.eye(3)
    if node.name in poses_map:
        local_rot = R.from_rotvec(poses_map[node.name]).as_matrix()
    
    global_rot = parent_rot @ local_rot
    scaled_offset = node.offset * SCALE_OFFSETS
    
    # ROOT HANDLING: If trans_offset is provided, use it as the absolute global position
    if trans_offset is not None:
        global_pos = trans_offset
    else:
        global_pos = parent_pos + (parent_rot @ scaled_offset)

    positions = [global_pos]
    lines = []
    
    for child in node.children:
        c_pos, c_lines = compute_fk(child, poses_map, global_pos, global_rot, None)
        positions.extend(c_pos)
        lines.append([global_pos, c_pos[0]])
        lines.extend(c_lines)
    return positions, lines

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--npy", required=True)
    parser.add_argument("--bvh", required=True)
    parser.add_argument("--out", help="Output path")
    args = parser.parse_args()

    # Load Data
    data = np.load(args.npy)
    poses = data['poses']
    trans = data['trans']
    names = data['joint_names']
    T = poses.shape[0]

    # DEBUG: Check if we actually have movement
    move_range = np.max(trans, axis=0) - np.min(trans, axis=0)
    print(f"Motion Range (Meters): X={move_range[0]:.2f}, Y={move_range[1]:.2f}, Z={move_range[2]:.2f}")
    if np.linalg.norm(move_range) < 0.1:
        print("⚠️ WARNING: Very little root movement detected. Is the source data in-place?")

    root = load_skeleton_structure(args.bvh)
    
    print("Computing FK...")
    all_pos, all_lines = [], []
    for f in tqdm(range(0, T, FRAME_SKIP)):
        pose_map = {name: poses[f, i] for i, name in enumerate(names)}
        p, l = compute_fk(root, pose_map, np.zeros(3), np.eye(3), trans[f])
        all_pos.append(np.array(p))
        all_lines.append(l)

    # Setup Plot
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection="3d")
    plt.subplots_adjust(0,0,1,1)
    ax.set_facecolor(BG_COLOR)
    fig.patch.set_facecolor(BG_COLOR)
    ax.set_axis_off()

    num_frames = len(all_pos)
    
    # Static Grid Settings
    GRID_SIZE = 1.0  # Spacing between lines
    GRID_RADIUS = 3  # How many lines to draw around the center

    def update(i):
        ax.cla()
        ax.set_axis_off()
        Jt = all_pos[i]
        root_p = Jt[0]
        cx, cy, cz = root_p

        # --- STATIC WORLD GRID ---
        center_x = np.floor(cx / GRID_SIZE) * GRID_SIZE
        center_y = np.floor(cy / GRID_SIZE) * GRID_SIZE
        
        gx = np.arange(center_x - GRID_RADIUS, center_x + GRID_RADIUS + GRID_SIZE, GRID_SIZE)
        gy = np.arange(center_y - GRID_RADIUS, center_y + GRID_RADIUS + GRID_SIZE, GRID_SIZE)
        xx, yy = np.meshgrid(gx, gy)
        zz = np.zeros_like(xx)
        
        ax.plot_wireframe(xx, yy, zz, color=FLOOR_COLOR, alpha=0.5, linewidth=1.0)
        
        # Draw Skeleton
        for s, e in all_lines[i]:
            ax.plot([s[0], e[0]], [s[1], e[1]], [s[2], e[2]], color=BONE_COLOR, linewidth=1.5)
        ax.scatter(Jt[:, 0], Jt[:, 1], Jt[:, 2], c=JOINT_COLOR, s=5, depthshade=False)
        
        # Camera follows character
        ax.set_xlim(cx - 1.0, cx + 1.0)
        ax.set_ylim(cy - 1.0, cy + 1.0)
        ax.set_zlim(0, 2.0)
        ax.set_box_aspect([1,1,1])
        ax.view_init(elev=15, azim=-90)
        
        ax.text2D(0.05, 0.95, f"Frame: {i*FRAME_SKIP}/{T}", transform=ax.transAxes, color="white")

    out_path = args.out if args.out else args.npy.replace(".npz", "_DARK.mp4")
    print(f"Rendering to {out_path}...")
    ani = FuncAnimation(fig, update, frames=num_frames, interval=1000/VIDEO_FPS)
    ani.save(out_path, fps=VIDEO_FPS, dpi=100, savefig_kwargs={'facecolor':BG_COLOR})
    print("Done.")

if __name__ == "__main__":
    main()