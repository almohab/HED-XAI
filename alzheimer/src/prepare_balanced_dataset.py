"""Create the balanced 2,000 + 2,000 image subset from the Kaggle OASIS-1 slices.

Extracted from the executed notebook `01_baselines_CNN_VGG16_MobileNetV2_InceptionV3_DenseNet121.ipynb` (cell 0); the model definition and the
training/evaluation code are unchanged. Only hard-coded Windows paths were replaced by the
environment variables below, and figures are written to disk instead of being displayed.

    AD_DATA_DIR    folder containing "Non Demented/" and "Very mild Dementia/" (default: data/balanced_data)
    AD_OUTPUT_DIR  where models, figures and CSV files are written (default: outputs/dataset)

Run from the `alzheimer/` folder:  python src/prepare_balanced_dataset.py
"""
import os
DATA_DIR = os.path.abspath(os.environ.get("AD_DATA_DIR", "data/balanced_data"))
OUT_DIR = os.path.abspath(os.environ.get("AD_OUTPUT_DIR", "outputs/dataset"))
os.makedirs(OUT_DIR, exist_ok=True)
SOURCE_DIR = os.path.abspath(os.environ.get("AD_SOURCE_DIR", "data/raw/Data"))  # original Kaggle "Data" folder
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
import shutil
import random
random.seed(42)  # added for reproducibility; the original run was unseeded

# Original dataset
source_dir = SOURCE_DIR

# New smaller dataset
target_dir = DATA_DIR

# Samples per class
samples_per_class = 2000

# Create target folder
if not os.path.exists(target_dir):
    os.makedirs(target_dir)

for class_name in os.listdir(source_dir):
    src_class = os.path.join(source_dir, class_name)
    dst_class = os.path.join(target_dir, class_name)

    os.makedirs(dst_class, exist_ok=True)

    images = os.listdir(src_class)
    selected = random.sample(images, min(samples_per_class, len(images)))

    for img in selected:
        shutil.copy(
            os.path.join(src_class, img),
            os.path.join(dst_class, img)
        )

print("Balanced dataset created!")
