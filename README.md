```markdown
# StableMotion: Professional SMPL-H Processing Pipeline

**StableMotion** provides a complete, robust system to clean, fix, and convert Motion Capture data. It handles the full data lifecycle: from raw BVH cleaning to SMPL-H processing (`.npz`), and converting predictions back to visualizable BVH formats for Blender/Unity.

---

## 🌟 Why this project is needed?

Raw motion capture data and AI outputs often suffer from technical artifacts that make them unusable for production pipelines. This repository solves those critical issues:

* **⚡ Jitter Removal:** Eliminates sensor noise using advanced Savitzky-Golay filtering while preserving sharp martial arts movements.
* **🔄 Coordinate Mismatch:** Automatically handles the complex conversion between Research formats (Z-Up) and Animation standards (Y-Up).
* **🦶 Floating Characters:** Implements **Smart Grounding** logic to ensure characters stay firmly on the floor (fixing the "flying" glitch).
* **🦴 Skeleton Mapping:** Automatically retargets 52-joint SMPL-H data to standard BVH hierarchies.

---

## 📂 File Structure (The Engine Room)

The codebase is organized into a modular architecture separating logic from automation:

### `scripts/` (Automation & Processing)
* `batch_processor.py`: **The Forward Pipeline.** Cleans raw BVH files, applies smoothing, and exports to `.npz` for AI training.
* `batch_smart_bvh.py` (🆕 **New Feature**): **The Reverse Pipeline.** Converts processed `.npz` files back to `.bvh` for visualization.
    * **Auto-Scaling:** Detects Meters vs Centimeters.
    * **Rotation Fix:** Corrects the -90 degree orientation issue.
    * **Floor Detection:** Calculates the lowest hip position to fix vertical offsets.

### `motion/` (Core Logic)
* `visualizer.py`: Generates high-quality MP4 previews with a 3D grid.
* `smpl_fk.py`: Handles Forward Kinematics for exact joint position calculation.
* `bvh_loader.py`: Custom parser for reading complex hierarchies and End Sites.

---

## 🛠️ Key Technical Features

### 1. Advanced Jitter Removal (Savitzky-Golay)
Unlike basic linear smoothing which blurs motion, we utilize a **Savitzky-Golay filter** (window=11, poly=3). This preserves the high-frequency details of impacts (punches/kicks) while effectively removing high-frequency sensor noise.

### 2. Smart Grounding Logic
The pipeline performs a full-sequence scan to detect the **lowest Z-height of the pelvis**. It dynamically shifts the entire animation to align with a standard leg height (approx. 95cm), ensuring the character neither floats in the air nor clips through the floor.

### 3. Coordinate Transformation (Z-Up ↔ Y-Up)
The system automatically detects the input coordinate system (common in research datasets) and applies the necessary **Matrix Transformations** to ensure immediate compatibility with Blender, Unity, and Unreal Engine.

---

## 🚀 How to Use

### Installation
Ensure you have Python installed, then install the required dependencies:
```bash
pip install numpy scipy tqdm

```

### 1. Forward Pipeline (BVH → NPZ)

To clean raw motion files and prepare them for ML training:

```bash
python -m scripts.batch_processor

```

### 2. Reverse Pipeline (NPZ → BVH)

To convert AI/ML outputs back to animation files for Blender:

```bash
python -m scripts.batch_smart_bvh --input_dir ./data/input_npz --output_dir ./data/output_bvh

```

---

## 📝 Note on Retargeting & Limitations

While the pipeline is robust, motion capture data varies significantly between sources.

* **Motion Data:** The conversion logic correctly maps the skeleton and orientation. However, the final visual output may vary slightly depending on the specific input data quality.
* **Fine-tuning:** Due to retargeting limitations between SMPL and standard skeletons, some specific motions might require minor offset adjustments in Blender for perfect foot planting.

---

```

```