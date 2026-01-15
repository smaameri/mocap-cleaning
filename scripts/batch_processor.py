import os
import glob
import subprocess
from tqdm import tqdm

SOURCE_DIR = r"D:\REAL WORLD DATA\ProxiData-20251205T112014Z-3-001\ProxiData\ProxiData_raw\BVH"
OUTPUT_DIR = r"D:\MLProjects\StableMotion\StableMotion\output\clean_input_npz"

def run_batch():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    bvh_files = glob.glob(os.path.join(SOURCE_DIR, "*.bvh"))
    print(f"Found {len(bvh_files)} files to process.")

    for bvh_path in tqdm(bvh_files, desc="Batch Processing"):
        file_name = os.path.basename(bvh_path).replace(".bvh", "")
        npz_output = os.path.join(OUTPUT_DIR, f"{file_name}.npz")

        convert_cmd = [
            "python", "-m", "scripts.smplh_processor",
            "--input", bvh_path,
            "--output", npz_output
        ]
        
        result = subprocess.run(convert_cmd, capture_output=True, text=True)
        
        if result.returncode != 0:
            print(f"\nERROR in {file_name}:")
            print(result.stderr)
            continue

        render_cmd = [
            "python", "-m", "motion.visualizer",
            "--npy", npz_output,
            "--bvh", bvh_path
        ]
        subprocess.run(render_cmd, capture_output=True)

    # FIXED: Removed Emoji
    print(f"\nBatch process finished successfully. Results in: {OUTPUT_DIR}")

if __name__ == "__main__":
    run_batch()