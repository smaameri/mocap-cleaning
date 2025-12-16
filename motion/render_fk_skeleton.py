import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from pathlib import Path
from motion.smpl_fk import smpl_fk, PARENTS


def main(npz_path):
    data = np.load(npz_path)
    joints = smpl_fk(
        data["global_orient"],
        data["body_pose"],
        data["transl"]
    )

    T, J, _ = joints.shape

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    def update(f):
        ax.cla()
        Jt = joints[f]

        ax.scatter(Jt[:,0], Jt[:,1], Jt[:,2], c="r")

        for j in range(1, J):
            p = PARENTS[j]
            ax.plot(
                [Jt[p,0], Jt[j,0]],
                [Jt[p,1], Jt[j,1]],
                [Jt[p,2], Jt[j,2]],
                c="k"
            )

        ax.set_xlim(-1,1)
        ax.set_ylim(0,2)
        ax.set_zlim(-1,1)
        ax.set_title(f"Frame {f}")

    ani = FuncAnimation(fig, update, frames=T, interval=30)
    out = Path(npz_path).with_name(Path(npz_path).stem + "_fk.mp4")
    ani.save(out, fps=30)
    print("Saved:", out)

if __name__ == "__main__":
    import sys
    main(sys.argv[1])
