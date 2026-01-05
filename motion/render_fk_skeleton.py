import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from pathlib import Path
from tqdm import tqdm
from motion.smpl_fk import smpl_fk, PARENTS

# ==========================================
# 1. FINAL SETTINGS
# ==========================================
TARGET_FPS = 24        # Cinematic Framerate
SOURCE_FPS = 60        # Data Speed (Change to 120 if slow)
AUTO_GROUND = True     # Snap feet to floor

# VISUAL AESTHETICS (Black/Red/White)
BG_COLOR = "#000000"   # Pitch Black Background
FLOOR_COLOR = "#444444"# Visible Grey Grid
BONE_COLOR = "#FFFFFF" # White Bones
JOINT_COLOR = "#FF0000"# Red Joints

# CAMERA ANGLES (Crucial for "Standing" look)
# Elev 10 = Low angle (Heroic)
# Azim -90 = Front View
CAM_ELEV = 10
CAM_AZIM = -90 

def main(npz_path):
    print(f"--- RENDERING: FINAL ORIGINAL (NO ROTATION) ---", flush=True)
    data = np.load(npz_path)

    # 2. Compute FK
    print("Computing Joints...", flush=True)
    joints = smpl_fk(data["global_orient"], data["body_pose"], data["transl"])
    
    # 3. FIX SPEED (Real-Time)
    skip = int(round(SOURCE_FPS / TARGET_FPS))
    if skip < 1: skip = 1
    joints = joints[::skip]

    # 4. FIX GROUND (Snap Z-lowest to 0)
    # Since "Original" worked, Z is definitely the Up axis.
    if AUTO_GROUND:
        lowest_z = joints[:, :, 2].min(axis=1)
        joints[:, :, 2] -= lowest_z[:, None]

    T, J, _ = joints.shape

    # 5. Setup Scene
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection="3d")
    plt.subplots_adjust(0,0,1,1)
    
    # Set Colors
    ax.set_facecolor(BG_COLOR)
    fig.patch.set_facecolor(BG_COLOR)
    ax.set_axis_off()

    def update(f):
        ax.cla()
        ax.set_axis_off()
        
        # Get Frame Data
        Jt = joints[f]
        root = Jt[0]
        
        cx, cy, cz = root[0], root[1], root[2]
        
        # --- DRAW FLOOR (XY Plane at Z=0) ---
        size = 2.5
        grid_x = np.arange(cx - size, cx + size, 0.5)
        grid_y = np.arange(cy - size, cy + size, 0.5)
        xx, yy = np.meshgrid(grid_x, grid_y)
        zz = np.zeros_like(xx) 

        # Wireframe Grid
        ax.plot_wireframe(xx, yy, zz, color=FLOOR_COLOR, alpha=0.5, linewidth=1.0)

        # --- DRAW CHARACTER ---
        for j in range(1, J):
            p = PARENTS[j]
            ax.plot(
                [Jt[p, 0], Jt[j, 0]], 
                [Jt[p, 1], Jt[j, 1]], 
                [Jt[p, 2], Jt[j, 2]], 
                color=BONE_COLOR, linewidth=3.0, zorder=10
            )
            
        ax.scatter(Jt[:, 0], Jt[:, 1], Jt[:, 2], c=JOINT_COLOR, s=30, depthshade=False, zorder=20)

        # --- CAMERA ---
        ax.set_xlim(cx - 1.2, cx + 1.2)
        ax.set_ylim(cy - 1.2, cy + 1.2)
        ax.set_zlim(0, 2.4) 
        ax.set_box_aspect([1, 1, 1])
        
        # Set the "Standing" View
        ax.view_init(elev=CAM_ELEV, azim=CAM_AZIM) 
        
        ax.text2D(0.05, 0.95, f"Frame: {f}/{T}", transform=ax.transAxes, color="white", fontsize=10)

    # 6. Render
    out_path = Path(npz_path).with_name("FINAL_PERFECT_ORIGINAL.mp4")
    print(f"Rendering to {out_path}...", flush=True)
    
    ani = FuncAnimation(fig, update, frames=T, interval=1000/TARGET_FPS)
    
    progress = tqdm(total=T, unit="frames")
    ani.save(out_path, fps=TARGET_FPS, dpi=100, savefig_kwargs={'facecolor':BG_COLOR}, 
             progress_callback=lambda i, n: progress.update(1))
    progress.close()
    print("Done.", flush=True)

if __name__ == "__main__":
    import sys
    default_path = "output/clean_input_npz/test_fix.npz"
    path = sys.argv[1] if len(sys.argv) > 1 else default_path
    main(path)