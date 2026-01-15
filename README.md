
---

# StableMotion: Professional BVH to SMPL-H Processing Pipeline

This repository provides a complete system to clean, fix, and convert raw Motion Capture (BVH) data into production-ready SMPL-H files. It is specifically designed to handle "ProxiData" martial arts sequences.

## 🌟 Why this project is needed?

Raw motion capture data often has technical errors that make it unusable for AI training or 3D animation. This pipeline automatically fixes:

* **Jitter (Shaking):** Removes sensor noise while keeping the sharp movement of martial arts.
* **Incorrect Scale:** Converts data from Centimeters to Meters (0.01 scale) for realistic body sizes.
* **Coordinate Mismatch:** Changes "Y-Up" raw data to "Z-Up" standard format.
* **Floating Characters:** Ensures the character is always touching the floor ().

---

## 📂 New Folder & File Structure

I have organized the code into a clean, professional structure so any developer can understand it immediately:

### 1. **`scripts/` (The Engine Room)**

* **`smplh_processor.py`**: The core logic file. It cleans the data using the Savitzky-Golay filter and fixes the orientation.
* **`batch_processor.py`**: A powerful automation tool. It allows you to process 100s of BVH files in one click instead of doing them one by one.

### 2. **`motion/` (The Skeleton Logic)**

* **`visualizer.py`**: Generates high-quality MP4 videos of the motion with a 3D floor grid for review.
* **`smpl_fk.py`**: Calculates the exact 3D position of every joint using Forward Kinematics.
* **`bvh_loader.py`**: A custom parser that reads BVH hierarchies and handles "End Sites" correctly.
* **`bvh_to_smplh.py`**: Maps raw joint names to the standard SMPL-H body model.

---

## 🛠️ Key Technical Features

### **1. Advanced Jitter Removal**

Instead of using basic smoothing, we use the **Savitzky-Golay filter** (window=11, poly=3). This allows the character to move smoothly while ensuring that fast actions (like a punch impact) remain sharp and don't look "blurry."

### **2. Automatic Floor Grounding**

We scan the entire animation to find the lowest point of the feet. The code then shifts the whole sequence so the character never "floats" or "sinks" into the floor.

### **3. Optimized for Production**

* **NPZ Exports:** Clean data is saved in `.npz` format for easy integration into Machine Learning models.
* **Clean Filenames:** No more messy timestamps or version numbers; files are saved with clear, professional names.

---

## 🚀 How to Use

To process all your raw files at once and generate both data and videos:

```bash
python -m scripts.batch_processor

```

All cleaned data will be available in the `output/clean_input_npz/` directory.

---
