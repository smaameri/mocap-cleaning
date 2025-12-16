import sys
import pprint
from motion import bvh_loader

def main(path):
    bvh = bvh_loader.load_bvh(path)
    print("\n=== BVH KEYS ===")
    print(list(bvh.keys()))

    print("\n=== joint_names (len) ===")
    print(len(bvh.get("joint_names", [])))
    print(bvh.get("joint_names", [])[:20])

    print("\n=== channel_index ===")
    print(bvh.get("channel_index", {}))

    # offsets if present
    print("\n=== joint_offsets (type + len) ===")
    offs = bvh.get("joint_offsets", None)
    if offs is None:
        print("joint_offsets = None")
    else:
        print(type(offs), len(offs))
        print("first 5:", offs[:5])

    print("\n=== channels keys ===")
    print(list(bvh.get("channels", {}).keys())[:20])

    print("\n=== First frame values (first 20 numbers) ===")
    frames = bvh.get("frames", None)
    if frames is None:
        print("frames = None")
    else:
        print(frames.shape)
        print(frames[0][:20])

if __name__ == "__main__":
    main(sys.argv[1])
