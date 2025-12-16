# motion/print_frame0.py
import sys
from motion import bvh_loader

def main(path):
    bvh = bvh_loader.load_bvh(path)
    frames = bvh["frames"]
    channels = bvh["channels"]
    channel_index = bvh["channel_index"]
    joint_names = bvh["joint_names"]

    print("\n=== FRAME 0 ROTATIONS ===\n")
    row = frames[0]

    for j in joint_names:
        if j not in channel_index:
            continue

        ch = channels.get(j, [])
        if not ch:
            continue

        ci = channel_index[j]
        print(f"{j:20s}  ->  ", end="")

        # print first 6 channels if available (pos + rot)
        vals = row[ci:ci+6]
        print(" ".join([f"{v:8.3f}" for v in vals]))

    print("\nDone.\n")

if __name__ == "__main__":
    main(sys.argv[1])
