import sys
from motion import bvh_loader
import numpy as np

def main(path):
    bvh = bvh_loader.load_bvh(path)

    names = bvh["joint_names"]
    parents = bvh["parents"]        # OrderedDict
    offsets = bvh["offsets"]        # OrderedDict (your loader)
    
    # Use the hierarchy order
    print("=== JOINT AXES (X,Y,Z from offsets) ===\n")

    for i, j in enumerate(names):
        off = offsets.get(j, None)
        if off is None:
            print(f"{i:02d} {j:<20}  OFFSET = None")
            continue

        ox, oy, oz = off
        print(f"{i:02d} {j:<20}  X:{ox:+.3f}   Y:{oy:+.3f}   Z:{oz:+.3f}")

    print("\nDone.")

if __name__ == "__main__":
    main(sys.argv[1])
