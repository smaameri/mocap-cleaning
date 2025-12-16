```markdown
# Mocap Cleaning: BVH → SMPL-H

This repository provides a minimal, research-focused pipeline for converting
BVH motion capture files into stable, correctly oriented SMPL-H motion.

The pipeline is designed for dataset cleaning and validation, not real-time use.

---

## Pipeline Overview

1. Load BVH motion (ZXY Euler rotations)
2. Convert BVH → SMPL-H using a verified basis transformation
3. Apply FK-consistent foot locking (translation only)
4. Export clean SMPL-H motion as `.npz`
5. Visualize motion using a forward-kinematics skeleton

---

## Folder Structure

```

.
├── motion/      # Core motion logic, FK, visualization
├── scripts/     # CLI tools for conversion and post-processing
└── README.md

````

- `motion/`  
  Contains SMPL-H forward kinematics and visualization utilities.

- `scripts/`  
  Command-line tools for BVH conversion and motion stabilization.

---

## Usage

### Convert BVH to SMPL-H

```bash
python -m scripts.convert_bvh_to_smplh input.bvh output.npz
````

This step parses BVH files with ZXY Euler rotations, applies the correct
BVH → SMPL-H basis transformation, and exports SMPL-H motion parameters.

---

### Stabilize Motion (Foot Lock)

```bash
python -m scripts.postprocess_smplh_stabilize output.npz
```

This step applies FK-consistent foot locking by adjusting translation only.
All joint rotations remain unchanged.

Output file:

```
output_stable.npz
```

---

### Visualize with FK Skeleton

```bash
python -m motion.render_fk_skeleton output_stable.npz
```

This visualization is intended for quick validation of orientation,
foot stability, and skeleton consistency.

---

## Notes

* Rotations are never altered during stabilization
* Foot locking operates on translation only
* Assumes a clean BVH joint hierarchy and known rotation order
* Intended for offline processing and dataset cleaning
* Not optimized for real-time or interactive use

---

## Intended Use

* Motion capture dataset cleanup
* BVH → SMPL-H research pipelines
* Debugging coordinate system and orientation issues
* Preprocessing motion for learning-based models

