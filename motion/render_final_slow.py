import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from pathlib import Path
from tqdm import tqdm
from motion.smpl_fk import smpl_fk, PARENTS

# ==========================================
# FINAL ADJUSTMENTS
# ==========================================
TARGET_FPS = 24        
DATA_FPS = 60          

# *** SPEED CONTROL ***
# 1.0 = Normal Speed (Previous setting)
# 2.0 = Two times Slower (Try this first!)
# 0.5 = Two times Faster
SLOW_DOWN = 2.0        

# *** THE FIXES ***
MIRROR_X = True        
AUTO_GROUND = True     

# *** VISUALS (WHITE THEME) ***
BG_COLOR = "#FFFFFF"   
FLOOR_COLOR = "#000000"
BONE_COLOR = "#000000" 
JOINT_COLOR = "#FF0000"

def main(npz_path):
    print(f"--- RENDERING: FINAL SLOW FIX (x{SLOW_DOWN}) ---", flush=True)
    data = np.load(npz_path)

    # 1. Compute Joints
    joints = smpl_fk(data["global_orient"], data["body_pose"], data["transl"])
    
    # 2. APPLY MIRROR FIX 
    if MIRROR_X:
        joints[:, :, 0] *= -1

    # 3. FIX GROUND 
    if AUTO_GROUND:
        lowest_z = joints[:, :, 2].min(axis=1)
        joints[:, :, 2] -= lowest_z[:, None]

    # 4. HANDLING SPEED (Applied Slow Down)
    T_orig = joints.shape[0]
    
    # Calculate duration based on the Slow Down factor
    # If SLOW_DOWN is 2.0, we pretend the video is twice as long.
    duration_seconds = (T_orig / DATA_FPS) * SLOW_DOWN
    
    target_frames = int(duration_seconds * TARGET_FPS)
    
    # Resample to the new (longer) length
    indices = np.linspace(0, T_orig - 1, target_frames).astype(int)
    joints = joints[indices]
    
    T, J, _ = joints.shape
    print(f"Resampled: {T_orig} frames -> {T} frames (Slower)")

    # 5. Setup Scene
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection="3d")
    plt.subplots_adjust(0,0,1,1)
    
    # Colors
    ax.set_facecolor(BG_COLOR)
    fig.patch.set_facecolor(BG_COLOR)
    
    # Clean up axes
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.grid(False) 
    ax.set_axis_off()

    def update(f):
        ax.cla()
        ax.set_axis_off()
        
        # Frame Data
        Jt = joints[f]
        root = Jt[0]
        cx, cy, cz = root
        
        # --- DRAW FLOOR ---
        size = 3.0
        gx = np.arange(cx - size, cx + size, 0.5)
        gy = np.arange(cy - size, cy + size, 0.5)
        xx, yy = np.meshgrid(gx, gy)
        zz = np.zeros_like(xx) 

        # Black lines on White
        ax.plot_wireframe(xx, yy, zz, color=FLOOR_COLOR, alpha=0.4, linewidth=0.8)

        # --- DRAW CHARACTER ---
        for j in range(1, J):
            p = PARENTS[j]
            ax.plot(
                [Jt[p, 0], Jt[j, 0]], 
                [Jt[p, 1], Jt[j, 1]], 
                [Jt[p, 2], Jt[j, 2]], 
                color=BONE_COLOR, linewidth=2.5, zorder=10
            )
            
        ax.scatter(Jt[:, 0], Jt[:, 1], Jt[:, 2], c=JOINT_COLOR, s=25, depthshade=False, zorder=20)

        # --- CAMERA ---
        ax.set_xlim(cx - 1.2, cx + 1.2)
        ax.set_ylim(cy - 1.2, cy + 1.2)
        ax.set_zlim(0, 2.4) 
        ax.set_box_aspect([1, 1, 1])
        
        ax.view_init(elev=10, azim=-90) 
        
        ax.text2D(0.05, 0.95, f"Frame: {f}/{T}", transform=ax.transAxes, color="black", fontsize=10)

    # 6. Render
    out_path = Path(npz_path).with_name("FINAL_WHITE_SLOW.mp4")
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