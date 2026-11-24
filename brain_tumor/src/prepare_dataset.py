"""Convert the Figshare .mat files to JPEG and split each class 70/15/15 (seed 42).

Extracted from the executed notebook `notebooks/01_data_conversion_split_and_MobileNetV2_10_epochs.ipynb` (cell 6,7); the model definition and the
training/evaluation code are unchanged. Only hard-coded Windows paths were replaced by the
environment variables below, and figures are written to disk instead of being displayed.

    BT_DATA_DIR    folder with train/, val/ and test/ sub-folders, each containing glioma/,
                   meningioma/ and pituitary/ (default: data/converted_images_split)
    BT_OUTPUT_DIR  where models, figures and CSV files are written (default: outputs/dataset)

Run from the `brain_tumor/` folder:  python src/prepare_dataset.py
"""
import os
DATA_DIR = os.path.abspath(os.environ.get("BT_DATA_DIR", "data/converted_images_split"))
OUT_DIR = os.path.abspath(os.environ.get("BT_OUTPUT_DIR", "outputs/dataset"))
os.makedirs(OUT_DIR, exist_ok=True)
MAT_DIR = os.path.abspath(os.environ.get("BT_MAT_DIR", "data/raw/BrainTumorDataset"))  # folder with the 3,064 .mat files
CONVERTED_DIR = os.path.abspath(os.environ.get("BT_CONVERTED_DIR", "data/converted_images"))
os.chdir(OUT_DIR)  # relative file names used below (models, figures) are written here

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
_fig_no = [0]
def _save_instead_of_show(*args, **kwargs):
    _fig_no[0] += 1
    plt.savefig(f"figure_{_fig_no[0]:02d}.png", dpi=200, bbox_inches="tight")
    plt.close("all")
plt.show = _save_instead_of_show

import os
import h5py
import numpy as np
import imageio

input_root = MAT_DIR  # Adjust this if needed
output_folder = CONVERTED_DIR

os.makedirs(output_folder, exist_ok=True)

saved_count = 0

# Walk through all subfolders to find .mat files
for root, dirs, files in os.walk(input_root):
    for file in files:
        if file.endswith('.mat'):
            file_path = os.path.join(root, file)
            try:
                with h5py.File(file_path, 'r') as f:
                    cjdata = f['cjdata']

                    # Load and transpose image matrix
                    image = np.array(cjdata['image']).T
                    label = int(np.array(cjdata['label'])[0][0])  # Should be 1, 2, or 3

                    # Normalize to 0–255
                    image = image.astype(np.float32)
                    image = (255 * (image - image.min()) / (image.max() - image.min())).astype(np.uint8)

                    # Save in corresponding label folder
                    label_folder = os.path.join(output_folder, str(label))
                    os.makedirs(label_folder, exist_ok=True)

                    image_name = os.path.splitext(file)[0] + '.jpg'
                    save_path = os.path.join(label_folder, image_name)
                    imageio.imwrite(save_path, image)

                    saved_count += 1
                    print(f"✅ Saved: {save_path}")

            except Exception as e:
                print(f"❌ Error in {file_path}: {e}")

print(f"\n✅ All images converted and saved: {saved_count}")
print(f"📁 Location: {output_folder}")


import os
import shutil
import random

# Set seed for reproducibility
random.seed(42)

# Paths
source_dir = CONVERTED_DIR
output_dir = DATA_DIR

# Split ratios
train_ratio = 0.7
val_ratio = 0.15
test_ratio = 0.15

# Create output directories
for split in ['train', 'val', 'test']:
    for class_name in os.listdir(source_dir):
        class_path = os.path.join(source_dir, class_name)
        if os.path.isdir(class_path):
            os.makedirs(os.path.join(output_dir, split, class_name), exist_ok=True)

# Split each class
for class_name in os.listdir(source_dir):
    class_path = os.path.join(source_dir, class_name)
    if not os.path.isdir(class_path):
        continue
    
    # List and shuffle images
    images = os.listdir(class_path)
    random.shuffle(images)

    total = len(images)
    train_end = int(train_ratio * total)
    val_end = train_end + int(val_ratio * total)

    splits = {
        'train': images[:train_end],
        'val': images[train_end:val_end],
        'test': images[val_end:]
    }

    # Move files into corresponding folders
    for split, files in splits.items():
        for file in files:
            src = os.path.join(class_path, file)
            dst = os.path.join(output_dir, split, class_name, file)
            shutil.copyfile(src, dst)

print("✅ Done! Images are split into train, val, and test.")
print(f"Path: {output_dir}")
