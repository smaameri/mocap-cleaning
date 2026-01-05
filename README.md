```markdown
# Mocap Cleaning: BVH → SMPL-H (R&D Pipeline)

This repository hosts a research-focused pipeline for converting raw BVH motion capture data into stable, correctly oriented SMPL-H motion. 

**Current Status:** Active R&D. The pipeline includes a custom BVH parser, a forward kinematics (FK) engine, and multiple experimental solvers designed to handle coordinate system mismatches (Y-Up vs Z-Up) and skeleton hierarchy differences.

---

## Key Features

* **Custom BVH Parser:** Lightweight, NumPy-based parser (no heavy dependencies) that handles various Euler orders (ZXY, XYZ, etc.).
* **SMPL Forward Kinematics:** Custom FK engine to visualize and validate joint rotations without relying on heavy 3D software.
* **Coordinate Space Retargeting:** Solvers to map raw Motion Capture space to SMPL world space.
* **Statistical Calibration:** (WIP) Auto-detection of "T-Pose" frames to calculate rest-pose offset matrices dynamically.

---

## Folder Structure

```text
.
├── motion/          # Core FK engine, Rendering logic, Skeleton visualization
├── scripts/         # CLI tools and Experimental Retargeting Solvers
├── output/          # Generated .npz files and debug renders
└── README.md

```

---

## Experimental Solvers (Scripts)

We are currently testing multiple approaches to solve axis alignment and limb twisting artifacts.

| Script Name | Approach | Status |
| --- | --- | --- |
| `process_smart_fix.py` | **Hybrid Approach.** Uses parent-relative rotation mapping with a global root correction. | *Active Debugging* |
| `process_stat_fix.py` | **Statistical Calibration.** Scans specific frame ranges (e.g., 0-30 or 800-900) to find the optimal T-Pose for calibration. | *Testing* |
| `process_vector_copy.py` | **Vector Alignment.** Ignores joint rotation values and calculates angles based on global limb vectors. Good for physics, bad for "swing" velocity. | *Reference* |
| `process_global_decoupled.py` | **Decoupled Mapping.** Separates arm rotation from shoulder rotation to prevent "twisted spine" artifacts. | *Reference* |

---

## Usage

### 1. Running the Retargeting (Smart Fix)

This script applies the current best-performing logic (Smart Fix) to a raw BVH file.

```bash
python -m scripts.process_smart_fix --input "path/to/input.bvh" --output "output/result.npz"

```

### 2. Statistical Calibration (Frame Scan)

To fix issues where the skeleton starts "lying down" or twisted, use the statistical scanner to find a better Rest Pose.

```bash
python -m scripts.process_stat_fix_800 --input "path/to/input.bvh" --output "output/calibrated.npz"

```

### 3. Visualizing the Output

Render the `.npz` file to a video using the custom FK renderer.

```bash
python -m motion.render_production_slow --npy "output/result.npz" --bvh "path/to/reference.bvh"

```

---

## Known Issues (WIP)

* **Coordinate Flipping:** During complex rotations (e.g., spins), the root orientation may flip axes, causing the character to momentarily lie flat. This is currently being addressed via the `process_stat_fix` calibration logic.
* **Shoulder Twisting:** Direct mapping of BVH collars to SMPL shoulders can cause mesh clipping. Vector-based decoupling is being tested as a fix.

---

## Intended Use

* Motion capture dataset cleanup
* Developing robust BVH → SMPL-H pipelines
* Debugging mathematical inconsistencies in raw mocap data

```

```
