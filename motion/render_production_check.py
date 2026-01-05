print("--- RENDERER INITIALIZING... ---", flush=True)
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from tqdm import tqdm
import os
import sys
from scipy.spatial.transform import Rotation as R

# ==========================================
# 1. SETTINGS
# ==========================================
# The NEW Stabilized File
NPZ_PATH = "output/clean_input_npz/FINAL_SMPLH_STABILIZED.npz"
BVH_PATH = r"D:\REAL WORLD DATA\ProxiData-20251205T112014Z-3-001\ProxiData\ProxiData_raw\BVH\KO_vignette_09_002_BJ_AM_0806.bvh"

# VISUAL SETTINGS
VIDEO_FPS = 60         # Smooth playback
FRAME_SKIP = 1         # Render every frame (High Detail)
SCALE_OFFSETS = 0.01   # Skeleton Scaling (Matches the 0.01 in the generator)

# COLORS
BG_COLOR = "#000000"
BONE_COLOR = "#FFFFFF"
JOINT_COLOR = "#FF0000"
FLOOR_COLOR = "#444444"

# ==========================================
# 2. LOADER
# ==========================================
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
        if token == "ROOT" or token == "JOINT":
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
        elif token == "}" and node_stack: node_stack.pop()
        elif token == "MOTION": break
        token = next(iterator, None)
    return root

# ==========================================
# 3. KINEMATICS
# ==========================================
def compute_fk(node, poses_map, parent_pos, parent_rot, trans_offset=None):
    local_rot = np.eye(3)
    if node.name in poses_map:
        rotvec = poses_map[node.name]
        local_rot = R.from_rotvec(rotvec).as_matrix()
    
    global_rot = parent_rot @ local_rot
    
    # Scale bone length for visualization
    scaled_offset = node.offset * SCALE_OFFSETS
    
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

# ==========================================
# 4. RENDER LOOP
# ==========================================
def main():
    print(f"--- RENDERING STABILIZED CHECK ---")
    
    if not os.path.exists(NPZ_PATH):
        print(f"❌ Error: {NPZ_PATH} not found. Run the generator first.")
        return
        
    data = np.load(NPZ_PATH)
    poses = data['poses']
    trans = data['trans']
    names = data['joint_names']
    
    T = poses.shape[0]
    print(f"Loaded {T} frames from stabilized data.")
    
    root = load_skeleton_structure(BVH_PATH)
    
    print("Computing 3D Positions...", flush=True)
    all_frames_pos = []
    all_frames_lines = []
    
    for f in tqdm(range(0, T, FRAME_SKIP)):
        pose_map = {name: poses[f, i] for i, name in enumerate(names)}
        root_trans = trans[f]
        
        pos, lines = compute_fk(root, pose_map, np.zeros(3), np.eye(3), root_trans)
        all_frames_pos.append(np.array(pos))
        all_frames_lines.append(lines)

    # Plot
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection="3d")
    plt.subplots_adjust(0,0,1,1)
    ax.set_facecolor(BG_COLOR)
    fig.patch.set_facecolor(BG_COLOR)
    ax.set_axis_off()
    
    num_render = len(all_frames_pos)

    def update(i):
        ax.cla()
        ax.set_axis_off()
        Jt = all_frames_pos[i]
        Lns = all_frames_lines[i]
        
        # Camera Center
        root_p = Jt[0]
        cx, cy, cz = root_p
        
        # Floor
        size = 2.0
        grid_x = np.arange(cx - size, cx + size, 0.5)
        grid_y = np.arange(cy - size, cy + size, 0.5)
        xx, yy = np.meshgrid(grid_x, grid_y)
        zz = np.zeros_like(xx)
        ax.plot_wireframe(xx, yy, zz, color=FLOOR_COLOR, alpha=0.5, linewidth=1.0)
        
        # Orientation Arrow (Blue = UP)
        ax.quiver(cx, cy, 0, 0, 0, 0.5, color='b') 

        # Skeleton
        for s, e in Lns:
            ax.plot([s[0], e[0]], [s[1], e[1]], [s[2], e[2]], color=BONE_COLOR, linewidth=1.5)
        ax.scatter(Jt[:, 0], Jt[:, 1], Jt[:, 2], c=JOINT_COLOR, s=10, depthshade=False)
        
        # Camera
        ax.set_xlim(cx - 1.2, cx + 1.2)
        ax.set_ylim(cy - 1.2, cy + 1.2)
        ax.set_zlim(0, 2.4)
        ax.set_box_aspect([1,1,1])
        ax.view_init(elev=15, azim=-90)
        
        ax.text2D(0.05, 0.95, f"STABILIZED: {i*FRAME_SKIP}/{T}", transform=ax.transAxes, color="white")

    out_path = "output/clean_input_npz/FINAL_STABILIZED_RENDER.mp4"
    print(f"Rendering to {out_path}...", flush=True)
    
    ani = FuncAnimation(fig, update, frames=num_render, interval=1000/VIDEO_FPS)
    progress = tqdm(total=num_render, unit="frames")
    ani.save(out_path, fps=VIDEO_FPS, dpi=100, savefig_kwargs={'facecolor':BG_COLOR},
             progress_callback=lambda i, n: progress.update(1))
    progress.close()
    print("Done.")

if __name__ == "__main__":
    main()