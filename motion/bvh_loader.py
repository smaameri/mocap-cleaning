# StableMotion/motion/bvh_loader.py
# Fully fixed BVH loader that handles:
# - nested joints
# - full hands
# - all CHANNELS blocks
# - End Site nodes
# - correct channel indexing
#
# Compatible with your 156-channel BVH files.

import os
import numpy as np
from collections import OrderedDict

# -------------------------------
# Parse End Site block
# -------------------------------
def _skip_end_site(lines, i):
    # after "End Site", next line should be "{"
    while i < len(lines) and "{" not in lines[i]:
        i += 1
    i += 1  # skip "{"
    # skip until "}"
    while i < len(lines) and "}" not in lines[i]:
        i += 1
    return i + 1  # skip "}"

# -------------------------------
# Parse a JOINT or ROOT block
# -------------------------------
def _parse_joint(lines, i):
    line = lines[i].strip()
    parts = line.split()
    name = parts[1]
    i += 1  # move to "{"

    # consume "{"
    while "{" not in lines[i]:
        i += 1
    i += 1

    node = {
        "name": name,
        "offset": np.zeros(3),
        "channels": [],
        "children": []
    }

    while i < len(lines):
        l = lines[i].strip()
        i += 1

        if l.startswith("OFFSET"):
            p = l.split()
            node["offset"] = np.array([float(p[1]), float(p[2]), float(p[3])], dtype=float)

        elif l.startswith("CHANNELS"):
            # e.g. CHANNELS 3 Zrotation Xrotation Yrotation
            parts = l.split()
            num = int(parts[1])
            node["channels"] = parts[2:2+num]

        elif l.startswith("JOINT"):
            child, i = _parse_joint(lines, i - 1)
            node["children"].append(child)

        elif l.startswith("End Site"):
            i = _skip_end_site(lines, i)

        elif l.startswith("}"):
            return node, i

    return node, i


# -------------------------------
# Flatten hierarchy
# -------------------------------
def _flatten(node, parent, joint_names, parents, offsets, channels):
    joint_names.append(node["name"])
    parents[node["name"]] = parent
    offsets[node["name"]] = node["offset"]
    channels[node["name"]] = node["channels"]

    for c in node["children"]:
        _flatten(c, node["name"], joint_names, parents, offsets, channels)


# -------------------------------
# MAIN LOAD FUNCTION
# -------------------------------
def load_bvh(path):
    if not os.path.exists(path):
        raise FileNotFoundError(path)

    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        lines = f.readlines()

    # find HIERARCHY
    i = 0
    while i < len(lines) and "HIERARCHY" not in lines[i]:
        i += 1
    i += 1

    # expect ROOT
    while i < len(lines) and not lines[i].strip().startswith("ROOT"):
        i += 1

    root, i = _parse_joint(lines, i)

    # flatten hierarchy
    joint_names = []
    parents = OrderedDict()
    offsets = OrderedDict()
    channels = OrderedDict()

    _flatten(root, None, joint_names, parents, offsets, channels)

    # find MOTION
    while i < len(lines) and "MOTION" not in lines[i]:
        i += 1
    i += 1

    # Frames: N
    parts = lines[i].strip().split()
    n_frames = int(parts[1])
    i += 1

    # Frame Time:
    parts = lines[i].strip().split()
    frame_time = float(parts[2])
    i += 1

    # count total channels
    total_channels = 0
    channel_index = {}
    raw_channel_order = []

    for j in joint_names:
        channel_index[j] = total_channels
        ch = channels[j]
        for name in ch:
            raw_channel_order.append(f"{j}_{name}")
        total_channels += len(ch)

    # read frame data
    frames = np.zeros((n_frames, total_channels), float)

    frame_id = 0
    while frame_id < n_frames and i < len(lines):
        line = lines[i].strip()
        i += 1
        if not line:
            continue
        vals = line.split()

        # Some BVH files wrap lines → gather multiple lines until enough values
        while len(vals) < total_channels and i < len(lines):
            extra = lines[i].strip().split()
            vals.extend(extra)
            i += 1

        if len(vals) != total_channels:
            raise ValueError(
                f"Frame {frame_id} has {len(vals)} values but expected {total_channels}"
            )

        frames[frame_id] = np.array([float(v) for v in vals], float)
        frame_id += 1

    return {
        "joint_names": joint_names,
        "parents": parents,
        "offsets": offsets,
        "channels": channels,
        "channel_index": channel_index,
        "frame_time": frame_time,
        "n_frames": n_frames,
        "frames": frames,
        "raw_channel_order": raw_channel_order,
    }


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python bvh_loader.py file.bvh")
        exit(0)
    out = load_bvh(sys.argv[1])
    print("Loaded BVH with:")
    print(len(out["joint_names"]), "joints")
    print(out["frames"].shape, "frames array")
